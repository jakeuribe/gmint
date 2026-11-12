//! File format and the online coding loop.
//!
//! The input is cut into `batch` equal streams. Each step predicts one byte of every stream from
//! that stream's previous `ctx` bytes, codes the batch, then trains on it. The decoder runs the
//! same model on the bytes it has already recovered, so nothing but the coded bytes is stored.
//! The remainder that does not fill a step is stored raw.

use std::collections::VecDeque;
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::coder::{Decoder, Encoder};
use crate::fixed::{Arch, Config, Head, Model, Predictor, VOCAB};
use crate::spn::Spn;
use crate::optim::Rule;

const MAGIC: &[u8; 6] = b"GMINT1";
const HEADER_LEN: usize = 6 + 8 + 4 + 2 + 2 + 2 + 1 + 2 + 8 + 8 + 1 + 8 + 1 + 1 + 1;
const PRIOR_MAGIC: &[u8; 8] = b"GMPRIOR1";
/// Steps of history the update controller averages over.
const WINDOW: usize = 32;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ *b as u64).wrapping_mul(0x100_0000_01b3))
}

/// log2(x) in Q8 from integer ops only: exponent plus a linear mantissa. Within 0.09 bits of the
/// true value, which is enough for the controller and identical on every machine.
fn log2_q8(x: u64) -> u64 {
    let n = 63 - x.leading_zeros() as u64;
    let m = if n >= 8 { (x >> (n - 8)) & 0xFF } else { (x << (8 - n)) & 0xFF };
    (n << 8) | m
}

/// Bytes of context a match must share before it is proposed.
const MATCH_MIN: usize = 6;
/// Matches at least this long are reported at the higher trust level.
const MATCH_LONG: u32 = 16;
const MATCH_TABLE_LOG2: u32 = 22;
const NO_MATCH: usize = usize::MAX;

/// Long-range match finder: the model itself only sees the last few bytes, so this proposes the
/// byte that followed the most recent earlier occurrence of the current [`MATCH_MIN`]-byte
/// context, anywhere in what has been coded so far. It reads only bytes from completed steps,
/// which the decoder also has, and uses no floats, so both sides propose the same bytes.
pub struct Matcher {
    on: bool,
    /// Hash of a context to the position after it, plus one; 0 is empty.
    table: Vec<u64>,
    /// Per stream: where the byte now being proposed is read from, and how long the match has run.
    from: Vec<usize>,
    run: Vec<u32>,
}

impl Matcher {
    pub fn new(on: bool, batch: usize) -> Self {
        let table = if on { vec![0; 1 << MATCH_TABLE_LOG2] } else { Vec::new() };
        Self { on, table, from: vec![NO_MATCH; batch], run: vec![0; batch] }
    }

    fn slot(context: &[u8]) -> usize {
        let key = context.iter().fold(0u64, |k, b| (k << 8) | *b as u64);
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - MATCH_TABLE_LOG2)) as usize
    }

    /// Proposals for step `t`, `[batch, 2]`: (byte, trust level). Call before the step is coded.
    pub fn guesses(&mut self, data: &[u8], len: usize, t: usize, device: &Device) -> Tensor<2, Int> {
        let batch = self.from.len();
        let mut out = vec![0i32; batch * 2];
        if self.on {
            for s in 0..batch {
                let c = self.from[s];
                // the source byte must come from a step that is already complete
                if c != NO_MATCH && c < batch * len && c % len < t {
                    out[s * 2] = data[c] as i32;
                    out[s * 2 + 1] = if self.run[s] >= MATCH_LONG { 2 } else { 1 };
                } else {
                    self.from[s] = NO_MATCH;
                }
            }
        }
        Tensor::from_data(TensorData::new(out, [batch, 2]), device)
    }

    /// Records step `t` once its bytes are known: extends or drops each stream's match and
    /// indexes the contexts that now end at this step.
    pub fn advance(&mut self, data: &[u8], len: usize, t: usize) {
        if !self.on {
            return;
        }
        for s in 0..self.from.len() {
            let pos = s * len + t;
            if self.from[s] != NO_MATCH {
                if data[self.from[s]] == data[pos] {
                    self.from[s] += 1;
                    self.run[s] += 1;
                } else {
                    self.from[s] = NO_MATCH;
                }
            }
            if t + 1 >= MATCH_MIN {
                // Look the context up before indexing it, or it would only ever find itself.
                let slot = Self::slot(&data[pos + 1 - MATCH_MIN..=pos]);
                if self.from[s] == NO_MATCH && self.table[slot] != 0 {
                    (self.from[s], self.run[s]) = (self.table[slot] as usize - 1, 0);
                }
                self.table[slot] = pos as u64 + 2;
            }
        }
    }
}

/// Decides, per step, whether the model trains on the batch it just coded.
///
/// This is TRACE's back-prop controller with a weight on it. A step's code length is compared
/// with the mean of the last [`WINDOW`] steps; the update is skipped when the step was cheaper
/// than `weight/256` of that mean. Weight 0 never skips (ratio only); 256 is TRACE's rule (skip
/// whenever the step beat the mean). Skipping buys latency and costs ratio, so the weight is how
/// much latency counts in the trade the paper measures as LCR.
///
/// The decision uses only integer code lengths of already-coded bytes, never wall-clock time, so
/// the decoder makes the same one.
pub struct Controller {
    weight: u64,
    window: VecDeque<u64>,
    sum: u64,
}

impl Controller {
    pub fn new(weight: u16) -> Self {
        Self { weight: weight as u64, window: VecDeque::with_capacity(WINDOW), sum: 0 }
    }

    pub fn should_update(&mut self, cost_q8: u64) -> bool {
        let update = self.window.len() < WINDOW || cost_q8 * 256 * WINDOW as u64 >= self.sum * self.weight;
        if self.window.len() == WINDOW {
            self.sum -= self.window.pop_front().unwrap();
        }
        self.window.push_back(cost_q8);
        self.sum += cost_q8;
        update
    }
}

/// Pre-trained starting weights. They are not stored in compressed files: the decoder must be
/// given the same prior, and the file header carries its hash so a mismatch is caught up front.
pub struct Prior {
    pub ctx: usize,
    pub emb: usize,
    pub hid: usize,
    pub head: Head,
    pub matcher: bool,
    pub values: Vec<i32>,
    pub hash: u64,
}

impl Prior {
    fn encode(c: &Config, values: &[i32]) -> Vec<u8> {
        let mut out = PRIOR_MAGIC.to_vec();
        for d in [c.ctx, c.emb, c.hid, c.head.id() as usize, c.matcher as usize] {
            out.extend((d as u16).to_le_bytes());
        }
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        out
    }

    pub fn save(path: &str, model: &Model) -> std::io::Result<()> {
        std::fs::write(path, Self::encode(&model.cfg, &model.export()))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if b.len() < 18 || &b[..8] != PRIOR_MAGIC || (b.len() - 18) % 4 != 0 {
            return Err(format!("{path}: not a gmint prior"));
        }
        let dim = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
        let values = b[18..].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let head = Head::from_id(dim(14) as u8).ok_or_else(|| format!("{path}: unknown head"))?;
        Ok(Self { ctx: dim(8), emb: dim(10), hid: dim(12), head, matcher: dim(16) != 0, values, hash: fnv1a(&b) })
    }
}

/// Builds the predictor a file is coded with: fresh, or started from a prior.
fn build(m: Config, prior: Option<&Prior>, device: &Device) -> Result<Box<dyn Predictor>, String> {
    if m.arch == Arch::Spn {
        return match prior {
            None => Ok(Box::new(Spn::new(m, device))),
            Some(_) => Err("priors are not supported for the sum-product network yet".into()),
        };
    }
    let mut model = Model::new(m, device);
    if let Some(p) = prior {
        if (p.ctx, p.emb, p.hid, p.head, p.matcher) != (m.ctx, m.emb, m.hid, m.head, m.matcher) {
            return Err(format!(
                "prior is for ctx {} emb {} hid {} {:?} match {}, not the model's shape",
                p.ctx, p.emb, p.hid, p.head, p.matcher
            ));
        }
        model.import(&p.values)?;
    }
    Ok(Box::new(model))
}

/// One pass of online training over `data` with no coding, for building a prior. Returns the
/// code length the model would have produced, in Q8 bits.
pub fn train_pass(model: &mut Model, data: &[u8], device: &Device) -> u64 {
    let m = model.cfg;
    let len = data.len() / m.batch;
    let mut syms = vec![0i32; m.batch];
    let mut total_cost = 0;
    let mut matcher = Matcher::new(m.matcher, m.batch);
    for t in 0..len {
        let guess = matcher.guesses(data, len, t, device);
        let freqs = model.predict(contexts(data, len, t, &m, device), guess);
        let mut cost = 0;
        for (s, row) in freqs.chunks_exact(VOCAB).enumerate() {
            let sym = data[s * len + t] as usize;
            syms[s] = sym as i32;
            let total: u64 = row.iter().map(|f| *f as u64).sum();
            cost += log2_q8(total) - log2_q8(row[sym] as u64);
        }
        matcher.advance(data, len, t);
        model.observe(cost);
        model.learn(targets(&syms, device));
        total_cost += cost;
    }
    total_cost
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub model: Config,
    /// Latency weight for the update controller, 0..=256.
    pub lcr_weight: u16,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub steps: usize,
    pub updates: usize,
    pub seconds: f64,
    /// Compressed size after each tenth of the input, for the convergence curve.
    pub curve: Vec<(usize, usize)>,
}

fn header(n: u64, o: &Options, checksum: u64, prior_hash: u64) -> Vec<u8> {
    let m = &o.model;
    let mut h = Vec::with_capacity(HEADER_LEN);
    h.extend(MAGIC);
    h.extend(n.to_le_bytes());
    h.extend((m.batch as u32).to_le_bytes());
    h.extend((m.ctx as u16).to_le_bytes());
    h.extend((m.emb as u16).to_le_bytes());
    h.extend((m.hid as u16).to_le_bytes());
    h.push(m.lr as u8);
    h.extend(o.lcr_weight.to_le_bytes());
    h.extend(m.seed.to_le_bytes());
    h.extend(checksum.to_le_bytes());
    h.push(m.rule.id());
    h.extend(prior_hash.to_le_bytes());
    h.push(m.head.id());
    h.push(m.matcher as u8);
    h.push(m.arch.id());
    h
}

fn parse_header(b: &[u8]) -> Result<(u64, Options, u64, u64), String> {
    if b.len() < HEADER_LEN || &b[..6] != MAGIC {
        return Err("not a gmint file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let model = Config {
        batch: u32::from_le_bytes(b[14..18].try_into().unwrap()) as usize,
        ctx: u16_at(18) as usize,
        emb: u16_at(20) as usize,
        hid: u16_at(22) as usize,
        lr: b[24] as u32,
        rule: Rule::from_id(b[43]).ok_or("unknown update rule")?,
        head: Head::from_id(b[52]).ok_or("unknown head")?,
        matcher: b[53] != 0,
        arch: Arch::from_id(b[54]).ok_or("unknown model")?,
        seed: u64_at(27),
    };
    model.validate()?;
    Ok((u64_at(6), Options { model, lcr_weight: u16_at(25) }, u64_at(35), u64_at(44)))
}

/// Context tensor for step `t`: the `ctx` bytes before position `t` of every stream, zero before
/// the stream's start.
fn contexts(data: &[u8], len: usize, t: usize, m: &Config, device: &Device) -> Tensor<2, Int> {
    let mut v = Vec::with_capacity(m.batch * m.ctx);
    for s in 0..m.batch {
        let base = s * len;
        for j in 0..m.ctx {
            v.push(if t + j >= m.ctx { data[base + t + j - m.ctx] as i32 } else { 0 });
        }
    }
    Tensor::from_data(TensorData::new(v, [m.batch, m.ctx]), device)
}

fn targets(syms: &[i32], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(TensorData::new(syms.to_vec(), [syms.len()]), device)
}

pub fn compress(data: &[u8], o: &Options, prior: Option<&Prior>, device: &Device, progress: bool) -> Result<(Vec<u8>, Stats), String> {
    let m = o.model;
    let len = data.len() / m.batch;
    let mut model = build(m, prior, device)?;
    let mut ctl = Controller::new(o.lcr_weight);
    let mut enc = Encoder::new();
    let mut matcher = Matcher::new(m.matcher, m.batch);
    let mut stats = Stats::default();
    let (t0, mut last_print) = (Instant::now(), Instant::now());
    let mut syms = vec![0i32; m.batch];

    for t in 0..len {
        let guess = matcher.guesses(data, len, t, device);
        let freqs = model.predict(contexts(data, len, t, &m, device), guess);
        let mut cost = 0;
        for (s, row) in freqs.chunks_exact(VOCAB).enumerate() {
            let sym = data[s * len + t] as usize;
            syms[s] = sym as i32;
            let cum: u64 = row[..sym].iter().map(|f| *f as u64).sum();
            let total: u64 = cum + row[sym..].iter().map(|f| *f as u64).sum::<u64>();
            enc.encode(cum, row[sym] as u64, total);
            cost += log2_q8(total) - log2_q8(row[sym] as u64);
        }
        matcher.advance(data, len, t);
        model.observe(cost);
        if ctl.should_update(cost) {
            model.learn(targets(&syms, device));
            stats.updates += 1;
        }
        stats.steps += 1;
        if len >= 10 && (t + 1) % (len / 10) == 0 {
            stats.curve.push(((t + 1) * m.batch, enc.len()));
        }
        if progress && last_print.elapsed().as_secs() >= 5 {
            last_print = Instant::now();
            let done = (t + 1) * m.batch;
            eprintln!(
                "  {:5.1}%  ratio {:.3}  {:.0} KB/s",
                100.0 * done as f64 / data.len() as f64,
                done as f64 / enc.len().max(1) as f64,
                done as f64 / 1024.0 / t0.elapsed().as_secs_f64()
            );
        }
    }

    let mut out = header(data.len() as u64, o, fnv1a(data), prior.map_or(0, |p| p.hash));
    out.extend(enc.finish());
    out.extend(&data[len * m.batch..]);
    stats.seconds = t0.elapsed().as_secs_f64();
    Ok((out, stats))
}

pub fn decompress(file: &[u8], prior: Option<&Prior>, device: &Device, progress: bool) -> Result<(Vec<u8>, Stats), String> {
    let (n, o, checksum, prior_hash) = parse_header(file)?;
    match (prior_hash, prior) {
        (0, None) => {}
        (0, Some(_)) => return Err("this file was compressed without a prior".into()),
        (_, None) => return Err("this file needs the prior it was compressed with (--prior)".into()),
        (h, Some(p)) if h != p.hash => return Err("this is not the prior the file was compressed with".into()),
        _ => {}
    }
    let (m, n) = (o.model, n as usize);
    let len = n / m.batch;
    let tail = n - len * m.batch;
    if file.len() < HEADER_LEN + tail {
        return Err("file is truncated".into());
    }
    let coded = &file[HEADER_LEN..file.len() - tail];

    let mut data = vec![0u8; n];
    let mut model = build(m, prior, device)?;
    let mut ctl = Controller::new(o.lcr_weight);
    let mut dec = Decoder::new(coded);
    let mut matcher = Matcher::new(m.matcher, m.batch);
    let mut stats = Stats::default();
    let (t0, mut last_print) = (Instant::now(), Instant::now());
    let mut syms = vec![0i32; m.batch];

    for t in 0..len {
        let guess = matcher.guesses(&data, len, t, device);
        let freqs = model.predict(contexts(&data, len, t, &m, device), guess);
        let mut cost = 0;
        for (s, row) in freqs.chunks_exact(VOCAB).enumerate() {
            let total: u64 = row.iter().map(|f| *f as u64).sum();
            let target = dec.target(total);
            let (mut cum, mut sym) = (0u64, 0usize);
            while cum + row[sym] as u64 <= target {
                cum += row[sym] as u64;
                sym += 1;
            }
            dec.consume(cum, row[sym] as u64);
            data[s * len + t] = sym as u8;
            syms[s] = sym as i32;
            cost += log2_q8(total) - log2_q8(row[sym] as u64);
        }
        matcher.advance(&data, len, t);
        model.observe(cost);
        if ctl.should_update(cost) {
            model.learn(targets(&syms, device));
            stats.updates += 1;
        }
        stats.steps += 1;
        if progress && last_print.elapsed().as_secs() >= 5 {
            last_print = Instant::now();
            let done = (t + 1) * m.batch;
            eprintln!(
                "  {:5.1}%  {:.0} KB/s",
                100.0 * done as f64 / n as f64,
                done as f64 / 1024.0 / t0.elapsed().as_secs_f64()
            );
        }
    }
    data[len * m.batch..].copy_from_slice(&file[file.len() - tail..]);
    stats.seconds = t0.elapsed().as_secs_f64();
    if fnv1a(&data) != checksum {
        return Err("checksum mismatch: the decoder's model diverged from the encoder's, or the file is damaged".into());
    }
    Ok((data, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Rng;

    /// Repetitive but not trivial: words drawn from a small vocabulary, plus a tail that does not
    /// fill a step.
    fn sample(n: usize) -> Vec<u8> {
        let words: [&[u8]; 6] = [b"gradient ", b"coder ", b"range ", b"integer ", b"shift ", b"\n"];
        let mut rng = Rng(9);
        let mut out = Vec::new();
        while out.len() < n {
            out.extend(words[(rng.next() >> 40) as usize % words.len()]);
        }
        out.truncate(n);
        out
    }

    fn options(rule: Rule, lcr_weight: u16) -> Options {
        let model = Config { arch: Arch::Attn, ctx: 8, emb: 16, hid: 64, batch: 128, lr: 13, rule, head: Head::Softmax, matcher: false, seed: 7 };
        Options { model, lcr_weight }
    }

    #[test]
    fn round_trips_under_every_rule_and_weight() {
        let device = Device::flex();
        let data = sample(40_000 + 77);
        for (rule, weight) in [(Rule::Norm, 0), (Rule::Auto, 0), (Rule::Auto, 256), (Rule::Sgd, 0), (Rule::Sgd, 256)] {
            let (file, stats) = compress(&data, &options(rule, weight), None, &device, false).unwrap();
            assert!(file.len() < data.len() / 2, "{rule:?}/{weight}: only reached {} bytes", file.len());
            assert_eq!(weight == 0, stats.updates == stats.steps, "{rule:?}/{weight}: skipping is tied to the weight");
            let (back, _) = decompress(&file, None, &device, false).unwrap();
            assert!(back == data, "{rule:?}/{weight}: round trip differs");
        }
    }

    #[test]
    fn tree_heads_round_trip() {
        let device = Device::flex();
        let data = sample(40_000 + 77);
        for head in [Head::Tree, Head::RandomTree] {
            let mut o = options(Rule::Auto, 0);
            o.model.head = head;
            let (file, _) = compress(&data, &o, None, &device, false).unwrap();
            assert!(file.len() < data.len() / 2, "{head:?}: only reached {} bytes", file.len());
            assert!(decompress(&file, None, &device, false).unwrap().0 == data, "{head:?}: round trip differs");
        }
    }

    #[test]
    fn sum_product_network_round_trips() {
        let device = Device::flex();
        let data = sample(128 * 500 + 9);
        let mut o = options(Rule::Auto, 0);
        o.model = Config { arch: Arch::Spn, emb: 4, hid: 4, ..o.model };
        let (file, _) = compress(&data, &o, None, &device, false).unwrap();
        assert!(file.len() < data.len() * 3 / 4, "only reached {} of {}", file.len(), data.len());
        assert!(decompress(&file, None, &device, false).unwrap().0 == data);
    }

    /// Long repeats that an 8-byte context cannot see: random blocks, each written twice.
    fn repeats(n: usize) -> Vec<u8> {
        let mut rng = Rng(4);
        let mut out = Vec::new();
        while out.len() < n {
            let block: Vec<u8> = (0..300).map(|_| (rng.next() >> 40) as u8).collect();
            out.extend(&block);
            out.extend(&block);
        }
        out.truncate(n);
        out
    }

    #[test]
    fn matcher_finds_long_repeats_and_round_trips() {
        let device = Device::flex();
        let data = repeats(128 * 1200 + 5);
        let mut o = options(Rule::Auto, 0);
        let (plain, _) = compress(&data, &o, None, &device, false).unwrap();
        o.model.matcher = true;
        let (matched, _) = compress(&data, &o, None, &device, false).unwrap();
        // half the bytes are copies; without the matcher random bytes do not compress at all
        assert!(plain.len() > data.len() * 9 / 10, "plain model reached {} of {}", plain.len(), data.len());
        assert!(matched.len() < data.len() * 3 / 4, "matcher only reached {} of {}", matched.len(), data.len());
        assert!(decompress(&matched, None, &device, false).unwrap().0 == data);
    }

    #[test]
    fn input_shorter_than_a_step_is_stored() {
        let device = Device::flex();
        let data = sample(100);
        let (file, _) = compress(&data, &options(Rule::Norm, 16), None, &device, false).unwrap();
        assert_eq!(decompress(&file, None, &device, false).unwrap().0, data);
    }

    #[test]
    fn prior_helps_and_must_match() {
        let device = Device::flex();
        let o = options(Rule::Auto, 0);
        let mut model = Model::new(o.model, &device);
        train_pass(&mut model, &sample(60_000), &device);
        let bytes = Prior::encode(&o.model, &model.export());
        let dir = std::env::temp_dir().join(format!("gmint-prior-test-{}", std::process::id()));
        std::fs::write(&dir, &bytes).unwrap();
        let prior = Prior::load(dir.to_str().unwrap()).unwrap();
        std::fs::remove_file(&dir).unwrap();

        let data: Vec<u8> = sample(90_000).split_off(60_000);
        let (cold, _) = compress(&data, &o, None, &device, false).unwrap();
        let (warm, _) = compress(&data, &o, Some(&prior), &device, false).unwrap();
        assert!(warm.len() < cold.len(), "prior did not help: {} vs {}", warm.len(), cold.len());
        assert!(decompress(&warm, Some(&prior), &device, false).unwrap().0 == data);
        assert!(decompress(&warm, None, &device, false).is_err());
        assert!(decompress(&cold, Some(&prior), &device, false).is_err());
    }

    #[test]
    fn damage_is_reported() {
        let device = Device::flex();
        let data = sample(20_000);
        let (mut file, _) = compress(&data, &options(Rule::Norm, 16), None, &device, false).unwrap();
        let mid = file.len() / 2;
        file[mid] ^= 0x40;
        assert!(decompress(&file, None, &device, false).is_err());
        assert!(decompress(&file[..20], None, &device, false).is_err());
    }
}
