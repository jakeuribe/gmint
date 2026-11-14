//! File format and the online coding loop.
//!
//! The input is cut into `batch` equal streams, each taking it round-robin in blocks (see
//! [`to_streams`]). Each step predicts one byte of every stream from that stream's previous
//! `ctx` bytes, codes the batch, then trains on it. The decoder runs the same model on the bytes
//! it has already recovered, so nothing but the coded bytes is stored. The remainder that does
//! not fill a step is stored raw.
//!
//! Every stream has its own rANS coder state ([`crate::rans`]), so a step's bytes are decoded
//! all at once, on the GPU: the decoder no longer reads 256 frequencies per stream back to the
//! host and walks the streams one by one. Files from before (`GMINT1`, one range coder over every
//! stream in turn) still decode.

use std::collections::VecDeque;
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::coder::Decoder as RangeDecoder;
use crate::rans::{Decoder, Encoder};
use crate::fixed::{Arch, Config, Head, Model, Predictor, VOCAB};
use crate::mix::Mix;
use crate::optim::Rule;

/// Files coded with the interleaved rANS coder ([`crate::rans`]).
const MAGIC: &[u8; 6] = b"GMINT2";
/// Files from before it: one range coder ([`crate::coder`]) over every stream in turn. They
/// still decode; nothing writes them any more.
const MAGIC_RANGE: &[u8; 6] = b"GMINT1";
/// The range coder's header; the rANS format's adds the block size (see [`Options::block`]).
const HEADER_LEN_RANGE: usize = 6 + 8 + 4 + 2 + 2 + 2 + 1 + 2 + 8 + 8 + 1 + 8 + 1 + 1 + 1 + 2 + 1 + 1 + 2 + 1;
const HEADER_LEN: usize = HEADER_LEN_RANGE + 4;
const PRIOR_MAGIC: &[u8; 8] = b"GMPRIOR1";
/// Steps of history the update controller averages over.
const WINDOW: usize = 32;
/// Steps the encoder runs on the device between reads back to the host. It equals the auto
/// rule's window, so every window's costs are delivered before the first step that could use
/// the rate they set.
const FLUSH: usize = 32;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ *b as u64).wrapping_mul(0x100_0000_01b3))
}

/// log2(x) in Q8 from integer ops only: exponent plus a linear mantissa. Within 0.09 bits of the
/// true value, which is enough for the controller and identical on every machine.
pub(crate) fn log2_q8(x: u64) -> u64 {
    let n = 63 - x.leading_zeros() as u64;
    let m = if n >= 8 { (x >> (n - 8)) & 0xFF } else { (x << (8 - n)) & 0xFF };
    (n << 8) | m
}

/// Bytes of context a short match must share before it is proposed; a long one needs
/// [`MATCH_LONG_CONTEXT`] and is preferred.
const MATCH_MIN: usize = 5;
const MATCH_LONG_CONTEXT: usize = 8;
/// Matches at least this long are reported at the higher trust level.
const MATCH_LONG: u32 = 16;
const MATCH_TABLE_LOG2: u32 = 22;
const NO_MATCH: usize = usize::MAX;

/// How the match finder looks for and follows matches; the default is how files are coded.
///
/// Measured 2026-11-13 against the earlier finder (one 6-byte context, unverified, indexed as it
/// looked up; [`MatchStyle::earlier`]): smaller on all 14 file and config pairs tried, by 3.2% on
/// average without a prior (kc32 6.0%) and 1.9% with the best prior. One minimum length cannot
/// serve both kinds of file: very repetitive ones want only long, reliable matches, binaries gain
/// from short ones; preferring a verified long match and falling back to a short one does both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchStyle {
    /// Bytes of context a match must share before it is proposed.
    pub min: usize,
    /// Look every stream's context up before indexing any of the step's: otherwise a stream can
    /// find a context another stream indexed this very step, whose next byte is not coded yet,
    /// and lose the step.
    pub lookup_first: bool,
    /// Check a found context really is the stream's own, so a hash collision is not proposed.
    pub verify: bool,
    /// With a block layout ([`to_streams`]), follow a match from a block's last byte to the
    /// first byte of the next block of the input, not of the stream.
    pub follow_blocks: bool,
    /// A second, longer context (0 for none): looked up first, always verified, so a long
    /// reliable match is preferred and the short one is only the fallback.
    pub long: usize,
    /// Count a match's run from the length of the context it was found by, not from 0.
    pub run_from_context: bool,
}

impl Default for MatchStyle {
    fn default() -> Self {
        Self { min: MATCH_MIN, lookup_first: true, verify: true, follow_blocks: true, long: MATCH_LONG_CONTEXT, run_from_context: true }
    }
}

impl MatchStyle {
    /// The finder before 2026-11-13, for comparisons ([`Matcher::census`]).
    pub fn earlier() -> Self {
        Self { min: 6, lookup_first: false, verify: false, follow_blocks: false, long: 0, run_from_context: false }
    }
}

/// Long-range match finder: the model itself only sees the last few bytes, so this proposes the
/// byte that followed the most recent earlier occurrence of the current [`MatchStyle::min`]-byte
/// context, anywhere in what has been coded so far. It reads only bytes from completed steps,
/// which the decoder also has, and uses no floats, so both sides propose the same bytes.
pub struct Matcher {
    on: bool,
    style: MatchStyle,
    /// The block size of the stream order, 0 for none (see [`MatchStyle::follow_blocks`]).
    block: usize,
    /// Hash of a context to the position after it, plus one; 0 is empty.
    table: Vec<u64>,
    /// Per stream: where the byte now being proposed is read from, and how long the match has run.
    from: Vec<usize>,
    run: Vec<u32>,
    /// With [`MatchStyle::lookup_first`]: each stream's slot for the step being recorded.
    slots: Vec<usize>,
    /// With [`MatchStyle::long`]: the long contexts' table, and each stream's slot in it.
    long_table: Vec<u64>,
    long_slots: Vec<usize>,
}

impl Matcher {
    pub fn new(on: bool, batch: usize, block: usize) -> Self {
        Self::styled(on, batch, MatchStyle::default(), block)
    }

    pub fn styled(on: bool, batch: usize, style: MatchStyle, block: usize) -> Self {
        let table = if on { vec![0; 1 << MATCH_TABLE_LOG2] } else { Vec::new() };
        let long_table = if on && style.long > 0 { vec![0; 1 << MATCH_TABLE_LOG2] } else { Vec::new() };
        Self { on, style, block, table, from: vec![NO_MATCH; batch], run: vec![0; batch], slots: vec![0; batch], long_table, long_slots: vec![0; batch] }
    }

    fn slot(context: &[u8]) -> usize {
        let key = if context.len() <= 8 {
            context.iter().fold(0u64, |k, b| (k << 8) | *b as u64)
        } else {
            // longer than a word: every byte through the hash, not just the last eight
            context.iter().fold(0xcbf2_9ce4_8422_2325u64, |k, b| (k ^ *b as u64).wrapping_mul(0x100_0000_01b3))
        };
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - MATCH_TABLE_LOG2)) as usize
    }

    /// Proposals for step `t`, `[batch, 2]`: (byte, trust level). Call before the step is coded.
    pub fn guesses(&mut self, data: &[u8], len: usize, t: usize, device: &Device) -> Tensor<2, Int> {
        let batch = self.from.len();
        Tensor::from_data(TensorData::new(self.guess_values(data, len, t), [batch, 2]), device)
    }

    /// [`Self::guesses`] as host values, row-major.
    pub fn guess_values(&mut self, data: &[u8], len: usize, t: usize) -> Vec<i32> {
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
        out
    }

    /// The position after `c` in input order: `c + 1`, except across a block's end when
    /// following blocks.
    fn after(&self, c: usize, len: usize) -> usize {
        let b = self.block;
        if !self.style.follow_blocks || b == 0 || b >= len || len % b != 0 || (c % len) % b != b - 1 {
            return c + 1;
        }
        let (s, row) = (c / len, (c % len) / b);
        let batch = self.from.len();
        if s + 1 < batch { (s + 1) * len + row * b } else { (row + 1) * b }
    }

    /// Whether the `min` bytes ending at `pos` are a context of the input, not a join of blocks.
    fn whole_context(&self, pos: usize, len: usize, min: usize) -> bool {
        let b = self.block;
        !self.style.follow_blocks || b == 0 || b >= len || len % b != 0 || (pos % len) % b + 1 >= min
    }

    /// Records step `t` once its bytes are known: extends or drops each stream's match and
    /// indexes the contexts that now end at this step.
    pub fn advance(&mut self, data: &[u8], len: usize, t: usize) {
        if !self.on {
            return;
        }
        let min = self.style.min;
        let batch = self.from.len();
        for s in 0..batch {
            let pos = s * len + t;
            if self.from[s] != NO_MATCH {
                if data[self.from[s]] == data[pos] {
                    self.from[s] = self.after(self.from[s], len);
                    self.run[s] += 1;
                } else {
                    self.from[s] = NO_MATCH;
                }
            }
            // the long context first: a verified long match beats any short one
            let long = self.style.long;
            self.long_slots[s] = usize::MAX;
            if long > 0 && t + 1 >= long && self.whole_context(pos, len, long) {
                let slot = Self::slot(&data[pos + 1 - long..=pos]);
                self.long_slots[s] = slot;
                if self.from[s] == NO_MATCH && self.long_table[slot] != 0 {
                    let c = self.long_table[slot] as usize - 1;
                    if c >= long && data[c - long..c] == data[pos + 1 - long..=pos] {
                        (self.from[s], self.run[s]) = (c, if self.style.run_from_context { long as u32 } else { 0 });
                    }
                }
                if !self.style.lookup_first {
                    self.long_table[slot] = pos as u64 + 2;
                }
            }
            if t + 1 < min || !self.whole_context(pos, len, min) {
                self.slots[s] = usize::MAX;
                continue;
            }
            // Look the context up before indexing it, or it would only ever find itself.
            let slot = Self::slot(&data[pos + 1 - min..=pos]);
            self.slots[s] = slot;
            if self.from[s] == NO_MATCH && self.table[slot] != 0 {
                let c = self.table[slot] as usize - 1;
                // the context before the candidate must be this one (it is coded: c - 1 was)
                let same = !self.style.verify || (c >= min && data[c - min..c] == data[pos + 1 - min..=pos]);
                if same {
                    (self.from[s], self.run[s]) = (c, if self.style.run_from_context { min as u32 } else { 0 });
                }
            }
            if !self.style.lookup_first {
                self.table[slot] = pos as u64 + 2;
            }
        }
        if self.style.lookup_first {
            for s in 0..batch {
                let pos = (s * len + t) as u64 + 2;
                if self.slots[s] != usize::MAX {
                    self.table[self.slots[s]] = pos;
                }
                if self.long_slots[s] != usize::MAX {
                    self.long_table[self.long_slots[s]] = pos;
                }
            }
        }
    }

    /// Proposals made and how many were right, per trust level (1, 2), over a stream order: what
    /// the match finder alone would do on `data`, for comparing styles without a model.
    pub fn census(data: &[u8], batch: usize, style: MatchStyle, block: usize) -> [(u64, u64); 2] {
        let len = data.len() / batch;
        let mut m = Self::styled(true, batch, style, block);
        let mut out = [(0u64, 0u64); 2];
        for t in 0..len {
            let g = m.guess_values(data, len, t);
            for s in 0..batch {
                let level = g[s * 2 + 1] as usize;
                if level > 0 {
                    out[level - 1].0 += 1;
                    out[level - 1].1 += (g[s * 2] == data[s * len + t] as i32) as u64;
                }
            }
            m.advance(data, len, t);
        }
        out
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
    pub norm: bool,
    /// Trained with [`Config::mask`]; a file coded from the prior is coded the same way.
    pub mask: bool,
    pub values: Vec<i32>,
    pub hash: u64,
}

impl Prior {
    fn encode(c: &Config, values: &[i32]) -> Vec<u8> {
        let mut out = PRIOR_MAGIC.to_vec();
        // flags: bit 0 the match finder, bit 1 the normalised layer, bit 2 the mask (priors made
        // before it existed have the bit clear, and were trained without it)
        for d in [c.ctx, c.emb, c.hid, c.head.id() as usize, c.matcher as usize | (c.norm as usize) << 1 | (c.mask as usize) << 2] {
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
        let flags = dim(16);
        if flags > 7 {
            return Err(format!("{path}: unknown prior flags {flags:#x}"));
        }
        Ok(Self { ctx: dim(8), emb: dim(10), hid: dim(12), head, matcher: flags & 1 != 0, norm: flags & 2 != 0, mask: flags & 4 != 0, values, hash: fnv1a(&b) })
    }
}

/// Builds the predictor a file is coded with: fresh, or started from a prior.
fn build(m: Config, prior: Option<&Prior>, device: &Device) -> Result<Box<dyn Predictor>, String> {
    if m.arch != Arch::Attn || m.short > 0 {
        return match prior {
            None => Ok(Box::new(Mix::new(m, device))),
            Some(_) => Err("priors are only supported for the attention model alone so far".into()),
        };
    }
    let mut model = Model::new(m, device);
    if let Some(p) = prior {
        if (p.ctx, p.emb, p.hid, p.head, p.matcher, p.norm, p.mask) != (m.ctx, m.emb, m.hid, m.head, m.matcher, m.norm, m.mask) {
            return Err(format!(
                "prior is for ctx {} emb {} hid {} {:?} match {} norm {} mask {}, not the model's shape",
                p.ctx, p.emb, p.hid, p.head, p.matcher, p.norm, p.mask
            ));
        }
        model.import(&p.values)?;
    }
    Ok(Box::new(model))
}

/// Under profiling, waits for the device and charges the time since `since` to `phase`.
fn lap(profile: bool, device: &Device, since: Instant, phase: &mut f64) -> Instant {
    if !profile {
        return since;
    }
    let _ = device.sync();
    let now = Instant::now();
    *phase += (now - since).as_secs_f64();
    now
}

/// The warm-up pass: trains once more on steps `0..upto`, which both sides have by then. The
/// match finder is left out of the replay.
fn replay(model: &mut dyn Predictor, data: &[u8], len: usize, upto: usize, m: &Config, device: &Device) {
    let mut syms = vec![0i32; m.batch];
    for t in 0..upto {
        for (s, sym) in syms.iter_mut().enumerate() {
            *sym = data[s * len + t] as i32;
        }
        let target = targets(&syms, device);
        let _ = model.predict_coded(contexts(data, len, t, m, device), Tensor::zeros([m.batch, 2], device), target.clone());
        model.learn(target);
    }
}

/// Values per stream and step in the rows the encoder reads back: the cost frequencies'
/// (cum, freq, total), then the coder's (cum, freq); see [`crate::fixed::coded`].
const ROW: usize = 5;

/// Costs of a run of steps from their rows, read back in one go: calls `each(row)` for every
/// stream of every step, in coding order, then `after(cost)` per step.
fn drain(pending: &mut Vec<Tensor<2, Int>>, batch: usize, mut each: impl FnMut(&[i32]), mut after: impl FnMut(u64)) {
    if pending.is_empty() {
        return;
    }
    let rows = read_rows(pending);
    for step in rows.chunks_exact(batch * ROW) {
        step.chunks_exact(ROW).for_each(&mut each);
        after(step_cost(step));
    }
}

/// The rows of a run of steps (see [`ROW`]), read back in one go; empties `pending`.
fn read_rows(pending: &mut Vec<Tensor<2, Int>>) -> Vec<i32> {
    let mut parts = std::mem::take(pending);
    let joined = if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(parts, 0) };
    let data = joined.into_data();
    match data.as_slice::<i32>() {
        Ok(rows) => rows.to_vec(),
        Err(_) => data.convert::<i32>().try_to_vec().unwrap(),
    }
}

/// One step's code length in Q8 bits, from its rows, in the cost frequencies: what the update
/// controller and the auto rule steer by.
fn step_cost(step: &[i32]) -> u64 {
    step.chunks_exact(ROW).map(|row| log2_q8(row[2] as u64) - log2_q8(row[1] as u64)).sum()
}

/// A window whose rows were read back, and whose costs the model has seen, but which is not yet
/// coded: the coding waits until the next window is on the device. Its first step, and rows.
struct Deferred(usize, Vec<i32>);

impl Deferred {
    /// Codes the window, and records the convergence points that fall in it as the plain loop
    /// would have: the window's bytes are coded at its last step.
    fn finish(self, enc: &mut Encoder, stats: &mut Stats, batch: usize, len: usize) {
        let Deferred(start, rows) = self;
        let before = enc.len();
        for row in rows.chunks_exact(ROW) {
            enc.push(row[3] as u32, row[4] as u32);
        }
        let steps = rows.len() / (batch * ROW);
        for step in start..start + steps {
            if len >= 10 && (step + 1) % (len / 10) == 0 {
                stats.curve.push(((step + 1) * batch, if step + 1 == start + steps { enc.len() } else { before }));
            }
        }
    }
}

/// What a [`train_pass`] measured, both on each step's pass before its update.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassCost {
    /// The code length the coder would have produced, in Q8 bits.
    pub coded_q8: u64,
    /// The head's own loss, `-sum ln p` in Q8 natural-log units (see [`Model::pending_loss`]).
    pub loss_q8: u64,
}

impl PassCost {
    pub fn add(&mut self, o: PassCost) {
        self.coded_q8 += o.coded_q8;
        self.loss_q8 += o.loss_q8;
    }

    /// (coded, loss) in bits per byte over `bytes`.
    pub fn bits_per_byte(&self, bytes: usize) -> (f64, f64) {
        let n = bytes.max(1) as f64;
        (self.coded_q8 as f64 / 256.0 / n, self.loss_q8 as f64 / 256.0 / std::f64::consts::LN_2 / n)
    }
}

/// One pass of online training over `data` with no coding, for building a prior. Returns what
/// coding would have cost and the head's own loss. Nothing is read back from the device except
/// three numbers per stream and one per step, once every [`FLUSH`] steps; then `progress` is
/// called with the bytes trained on so far, their cost, and the model's rate.
/// `data` is in stream order already, with blocks of `block` bytes (see [`to_streams`]).
pub fn train_pass(model: &mut Model, data: &[u8], block: usize, device: &Device, mut progress: impl FnMut(usize, PassCost, i32)) -> PassCost {
    let m = model.cfg;
    let len = data.len() / m.batch;
    let mut syms = vec![0i32; m.batch];
    let mut total = PassCost::default();
    let mut matcher = Matcher::new(m.matcher, m.batch, block);
    let mut pending = Vec::with_capacity(FLUSH);
    let mut losses = Vec::with_capacity(FLUSH);
    for t in 0..len {
        for (s, sym) in syms.iter_mut().enumerate() {
            *sym = data[s * len + t] as i32;
        }
        let guess = matcher.guesses(data, len, t, device);
        let target = targets(&syms, device);
        pending.push(model.predict_coded(contexts(data, len, t, &m, device), guess, target.clone()));
        losses.push(model.pending_loss(&target));
        matcher.advance(data, len, t);
        model.learn(target);
        if pending.len() == FLUSH || t + 1 == len {
            drain(&mut pending, m.batch, |_| {}, |cost| {
                model.observe(cost);
                total.coded_q8 += cost;
            });
            let loss: Vec<i32> = Tensor::cat(std::mem::take(&mut losses), 0).into_data().convert::<i32>().try_to_vec().unwrap();
            total.loss_q8 += loss.iter().map(|l| *l as u64).sum::<u64>();
            progress((t + 1) * m.batch, total, model.rate());
        }
    }
    total
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub model: Config,
    /// Latency weight for the update controller, 0..=256.
    pub lcr_weight: u16,
    /// How the streams take the input: 0 for one contiguous slice each, else round-robin in
    /// blocks of this many bytes (see [`to_streams`]).
    pub block: u32,
}

/// The input in the order the streams code it. Every stream codes a contiguous run of
/// `len = n / batch` bytes of the result, and after step `t` the first `t` of every run are known
/// to both sides; what the match finder can copy from is limited to those.
///
/// With `block = 0` each stream's run is a contiguous slice of the input, so at step `t` the
/// known bytes are `t` from every slice, scattered over the whole input. With blocks, stream `s`
/// takes input blocks `s`, `s + batch`, `s + 2 batch`, ...: then everything before the current
/// row of blocks is already known, almost all of what precedes a position in the input. Blocks
/// too small cost more than they win (a block's first bytes see the previous block's end as
/// their context, and matches break at block ends); 2-4 KiB measured best.
///
/// When `block` does not divide `len`, each stream takes its whole blocks, then its own share of
/// what is left; the tail past `batch * len` stays where it is.
pub(crate) fn to_streams(data: &[u8], batch: usize, block: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    for_blocks(data.len(), batch, block, |from, to, n| out[to..to + n].copy_from_slice(&data[from..from + n]));
    out
}

/// The inverse of [`to_streams`].
fn from_streams(coded: &[u8], batch: usize, block: usize) -> Vec<u8> {
    let mut out = coded.to_vec();
    for_blocks(coded.len(), batch, block, |from, to, n| out[from..from + n].copy_from_slice(&coded[to..to + n]));
    out
}

/// Calls `each(input offset, stream-order offset, length)` for every piece [`to_streams`] moves.
fn for_blocks(n: usize, batch: usize, block: usize, mut each: impl FnMut(usize, usize, usize)) {
    let len = n / batch;
    if block == 0 || block >= len {
        return;
    }
    let (rows, rest) = (len / block, len % block);
    for s in 0..batch {
        for j in 0..rows {
            each((j * batch + s) * block, s * len + j * block, block);
        }
        if rest > 0 {
            each(batch * rows * block + s * rest, s * len + rows * block, rest);
        }
    }
}

#[derive(Debug, Default)]
pub struct Stats {
    pub steps: usize,
    pub updates: usize,
    pub seconds: f64,
    /// Compressed size after each tenth of the input, for the convergence curve.
    pub curve: Vec<(usize, usize)>,
    /// The predictor's health line at each point of the curve, when asked for.
    pub health: Vec<String>,
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
    h.extend([m.comps as u8, m.replicas as u8, m.short as u8, m.adapt]);
    h.extend((m.warm as u16).to_le_bytes());
    // bit 0 the normalised layer, bit 1 the mask; files from before the mask have it clear
    h.push(m.norm as u8 | (m.mask as u8) << 1);
    h.extend(o.block.to_le_bytes());
    h
}

/// The header's fields, and whether the file is in the rANS format (else the old range coder's).
fn parse_header(b: &[u8]) -> Result<(u64, Options, u64, u64, bool), String> {
    if b.len() < HEADER_LEN_RANGE || (&b[..6] != MAGIC && &b[..6] != MAGIC_RANGE) {
        return Err("not a gmint file".into());
    }
    let rans = &b[..6] == MAGIC;
    if rans && b.len() < HEADER_LEN {
        return Err("not a gmint file".into());
    }
    let block = if rans { u32::from_le_bytes(b[HEADER_LEN_RANGE..HEADER_LEN].try_into().unwrap()) } else { 0 };
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
        comps: b[55] as usize,
        replicas: b[56] as usize,
        short: b[57] as usize,
        adapt: b[58],
        warm: u16_at(59) as usize,
        norm: b[61] & 1 != 0,
        mask: b[61] & 2 != 0,
        seed: u64_at(27),
    };
    if b[61] > 3 {
        return Err("unknown model flags".into());
    }
    model.validate()?;
    Ok((u64_at(6), Options { model, lcr_weight: u16_at(25), block }, u64_at(35), u64_at(44), rans))
}

/// Context tensor for step `t`: the `ctx` bytes before position `t` of every stream, zero before
/// the stream's start.
fn contexts(data: &[u8], len: usize, t: usize, m: &Config, device: &Device) -> Tensor<2, Int> {
    let mut v = Vec::with_capacity(m.batch * m.ctx);
    context_values(&mut v, data, len, t, m);
    Tensor::from_data(TensorData::new(v, [m.batch, m.ctx]), device)
}

/// Appends [`contexts`] for step `t` to `v` as host values.
fn context_values(v: &mut Vec<i32>, data: &[u8], len: usize, t: usize, m: &Config) {
    for s in 0..m.batch {
        let base = s * len;
        for j in 0..m.ctx {
            v.push(if t + j >= m.ctx { data[base + t + j - m.ctx] as i32 } else { 0 });
        }
    }
}

/// One step's inputs as the host prepares them: contexts `[batch, ctx]`, the match finder's
/// proposals `[batch, 2]`, the true bytes `[batch]`, row-major.
struct StepInputs {
    ctx: Vec<i32>,
    guess: Vec<i32>,
    target: Vec<i32>,
}

/// Where the encoder is with its CUDA graph of one [`FLUSH`]-step window.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Graphing {
    /// The model cannot be replayed (see [`Predictor::graphable`]), or capture failed.
    Off,
    /// Not started: the next eligible window runs as the capture's warm-up.
    Idle,
    /// Warmed up: the next eligible window is recorded, then replayed.
    Warm,
    /// Captured: eligible windows are replayed.
    Ready,
}

/// The fixed device buffers a captured window reads its inputs from, one slot per step:
/// contexts `[FLUSH, batch, ctx]`, proposals `[FLUSH, batch, 2]`, true bytes `[FLUSH, batch]`,
/// each held flat (a flat buffer is never given padded rows) and rewritten before each replay.
struct Slots {
    ctx: Tensor<1, Int>,
    guess: Tensor<1, Int>,
    target: Tensor<1, Int>,
    batch: usize,
    width: usize,
}

impl Slots {
    /// Slots for `steps` steps.
    fn new(m: &Config, steps: usize, device: &Device) -> Self {
        let flat = |n: usize| Tensor::<1, Int>::from_data(TensorData::new(vec![0i32; n], [n]), device);
        Self { ctx: flat(steps * m.batch * m.ctx), guess: flat(steps * m.batch * 2), target: flat(steps * m.batch), batch: m.batch, width: m.ctx }
    }

    /// Step `i`'s inputs, as views of the slots.
    fn step(&self, i: usize) -> (Tensor<2, Int>, Tensor<2, Int>, Tensor<1, Int>) {
        let (b, c) = (self.batch, self.width);
        (
            self.ctx.clone().narrow(0, i * b * c, b * c).reshape([b, c]),
            self.guess.clone().narrow(0, i * b * 2, b * 2).reshape([b, 2]),
            self.target.clone().narrow(0, i * b, b),
        )
    }
}

fn targets(syms: &[i32], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(TensorData::new(syms.to_vec(), [syms.len()]), device)
}

pub fn compress(data: &[u8], o: &Options, prior: Option<&Prior>, device: &Device, progress: bool) -> Result<(Vec<u8>, Stats), String> {
    compress_watched(data, o, prior, device, progress, false)
}

/// [`compress`], optionally recording the predictor's health at every tenth of the input.
pub fn compress_watched(input: &[u8], o: &Options, prior: Option<&Prior>, device: &Device, progress: bool, health: bool) -> Result<(Vec<u8>, Stats), String> {
    let m = o.model;
    let ordered = to_streams(input, m.batch, o.block as usize);
    let data = &ordered[..];
    let len = data.len() / m.batch;
    let mut model = build(m, prior, device)?;
    let mut ctl = Controller::new(o.lcr_weight);
    let mut enc = Encoder::new(m.batch);
    let mut stats = Stats::default();
    let (t0, mut last_print) = (Instant::now(), Instant::now());
    // Every step's inputs depend on the data alone, never on the model, so a second thread
    // prepares them (contexts, the match finder's proposals, the true bytes) ahead of the device.
    std::thread::scope(|scope| -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<StepInputs>(4 * FLUSH);
    scope.spawn(move || {
        let mut matcher = Matcher::new(m.matcher, m.batch, o.block as usize);
        for t in 0..len {
            let guess = matcher.guess_values(data, len, t);
            let mut ctx = Vec::with_capacity(m.batch * m.ctx);
            context_values(&mut ctx, data, len, t, &m);
            let target = (0..m.batch).map(|s| data[s * len + t] as i32).collect();
            matcher.advance(data, len, t);
            if tx.send(StepInputs { ctx, guess, target }).is_err() {
                break;
            }
        }
    });
    let next = || rx.recv().map_err(|_| "the input thread stopped".to_string());

    // With no update skipping, nothing the host computes is needed until the auto rule's window
    // ends, so the device runs FLUSH steps before anything is read back. Skipping decides per
    // step from that step's cost, so then every step is read back.
    let flush = if o.lcr_weight == 0 { FLUSH } else { 1 };
    // GMINT_PROFILE=1 waits for the device after each phase of a step and reports where the time
    // went. The waits slow the run down, so the split is what to read, not the total.
    let profile = std::env::var_os("GMINT_PROFILE").is_some();
    let mut phases = [0f64; 4];
    let mut pending = Vec::with_capacity(flush);
    // Whole windows of FLUSH steps can run as one replayed CUDA graph: what a step depends on
    // from the host (its inputs, the optimizers' rates) is written to fixed device buffers first.
    // GMINT_GRAPH=0 turns this off; the output is the same either way.
    let mut graphing = if flush == FLUSH && !health && !profile && model.graphable() && std::env::var("GMINT_GRAPH").map_or(true, |v| v != "0") {
        Graphing::Idle
    } else {
        Graphing::Off
    };
    let slots = (graphing != Graphing::Off).then(|| Slots::new(&m, FLUSH, device));
    let mut captured: Vec<Tensor<2, Int>> = Vec::new();
    let mut deferred: Option<Deferred> = None;
    let mut t = 0;
    while t < len {
        // a window the graph can take: whole, starting with nothing pending, and with the
        // warm-up pass (if any) only after its last step; the first few run plainly, so every
        // kernel is compiled and the memory pool has settled before the capture's warm-up
        let window = graphing != Graphing::Off && pending.is_empty() && t >= 2 * FLUSH && t + FLUSH <= len && !(t < m.warm && m.warm < t + FLUSH);
        if window {
            let slots = slots.as_ref().unwrap();
            let (mut ctx_v, mut guess_v, mut target_v) =
                (Vec::with_capacity(FLUSH * m.batch * m.ctx), Vec::with_capacity(FLUSH * m.batch * 2), Vec::with_capacity(FLUSH * m.batch));
            for _ in 0..FLUSH {
                let step = next()?;
                ctx_v.extend(step.ctx);
                guess_v.extend(step.guess);
                target_v.extend(step.target);
            }
            crate::fused::write_ints(&slots.ctx, ctx_v);
            crate::fused::write_ints(&slots.guess, guess_v);
            crate::fused::write_ints(&slots.target, target_v);
            let probe = &slots.target;
            // the steps of the window, run (or recorded) plainly from the slots; their coder rows
            // are joined on the device, so the window reads back in one copy
            let run = |model: &mut Box<dyn Predictor>| -> Vec<Tensor<2, Int>> {
                let rows = (0..FLUSH)
                    .map(|i| {
                        let (ctx, guess, target) = slots.step(i);
                        let coded = model.predict_coded(ctx, guess, target.clone());
                        model.learn(target);
                        coded
                    })
                    .collect();
                vec![Tensor::cat(rows, 0)]
            };
            let outputs = match graphing {
                Graphing::Ready => {
                    model.refresh_steps();
                    crate::fused::graph(probe, crate::fused::Graph::Replay).map(|_| captured.clone())
                }
                Graphing::Idle => crate::fused::graph(probe, crate::fused::Graph::Prepare).map(|_| {
                    graphing = Graphing::Warm;
                    run(&mut model)
                }),
                _ => {
                    model.refresh_steps();
                    let recorded = crate::fused::graph(probe, crate::fused::Graph::Start).map(|_| run(&mut model));
                    match recorded.and_then(|out| crate::fused::graph(probe, crate::fused::Graph::Stop).map(|_| out)) {
                        Ok(out) => {
                            graphing = Graphing::Ready;
                            captured = out;
                            crate::fused::graph(probe, crate::fused::Graph::Replay).map(|_| captured.clone())
                        }
                        Err(e) => Err(e),
                    }
                }
            };
            let mut outputs = match outputs {
                Ok(out) => out,
                Err(e) => {
                    // a window that was only recorded has not run: run it plainly, and stay plain
                    if progress {
                        eprintln!("  CUDA graph unavailable ({e}); continuing without");
                    }
                    let _ = crate::fused::graph(probe, crate::fused::Graph::Drop);
                    graphing = Graphing::Off;
                    captured.clear();
                    run(&mut model)
                }
            };
            // the previous window is coded while the device runs this one
            if let Some(d) = deferred.take() {
                d.finish(&mut enc, &mut stats, m.batch, len);
            }
            let rows = read_rows(&mut outputs);
            for step in rows.chunks_exact(m.batch * ROW) {
                model.observe(step_cost(step));
            }
            deferred = Some(Deferred(t, rows));
            stats.updates += FLUSH;
            stats.steps += FLUSH;
            t += FLUSH;
            if t == m.warm {
                replay(model.as_mut(), data, len, m.warm, &m, device);
            }
            if progress && last_print.elapsed().as_secs() >= 5 {
                last_print = Instant::now();
                let done = t * m.batch;
                eprintln!(
                    "  {:5.1}%  ratio {:.3}  {:.0} KB/s",
                    100.0 * done as f64 / data.len() as f64,
                    done as f64 / enc.len().max(1) as f64,
                    done as f64 / 1024.0 / t0.elapsed().as_secs_f64()
                );
            }
            continue;
        }
        if let Some(d) = deferred.take() {
            d.finish(&mut enc, &mut stats, m.batch, len);
        }
        let mark = Instant::now();
        let step = next()?;
        let guess = Tensor::from_data(TensorData::new(step.guess, [m.batch, 2]), device);
        let target = targets(&step.target, device);
        let ctx = Tensor::from_data(TensorData::new(step.ctx, [m.batch, m.ctx]), device);
        let host_done = lap(profile, device, mark, &mut phases[0]);
        crate::ops::prof::lap(device, "host prep, uploads");
        pending.push(model.predict_coded(ctx, guess, target.clone()));
        crate::ops::prof::lap(device, "f: coded freqs");
        let forward_done = lap(profile, device, host_done, &mut phases[1]);
        let mut step_cost = 0;
        if flush == 1 {
            drain(&mut pending, m.batch, |row| enc.push(row[3] as u32, row[4] as u32), |cost| step_cost = cost);
        }
        if flush > 1 || ctl.should_update(step_cost) {
            model.learn(target);
            stats.updates += 1;
        }
        let learn_done = lap(profile, device, forward_done, &mut phases[2]);
        if flush == 1 {
            model.observe(step_cost);
        } else if pending.len() == flush || t + 1 == len {
            drain(&mut pending, m.batch, |row| enc.push(row[3] as u32, row[4] as u32), |cost| model.observe(cost));
        }
        crate::ops::prof::lap(device, "b: norm stats, rest");
        lap(profile, device, learn_done, &mut phases[3]);
        crate::ops::prof::lap(device, "readback, coder");
        if t == 511 {
            crate::ops::prof::reset();
        }
        // after the flush above, so the rate the replay trains at is the one the decoder has
        if t + 1 == m.warm {
            if !pending.is_empty() {
                drain(&mut pending, m.batch, |row| enc.push(row[3] as u32, row[4] as u32), |cost| model.observe(cost));
            }
            replay(model.as_mut(), data, len, m.warm, &m, device);
        }
        stats.steps += 1;
        if len >= 10 && (t + 1) % (len / 10) == 0 {
            stats.curve.push(((t + 1) * m.batch, enc.len()));
            if health {
                stats.health.push(model.health());
            }
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
        t += 1;
    }
    if let Some(d) = deferred.take() {
        d.finish(&mut enc, &mut stats, m.batch, len);
    }
    if graphing == Graphing::Ready {
        let _ = crate::fused::graph(&slots.as_ref().unwrap().target, crate::fused::Graph::Drop);
    }

    crate::ops::prof::report();
    if profile {
        let names = ["host prep + uploads", "forward", "backward + update", "readback + coder"];
        for (name, secs) in names.iter().zip(phases) {
            eprintln!("  {name:<20} {secs:>7.2}s  {:>7.1} us/step", secs * 1e6 / len.max(1) as f64);
        }
    }
    Ok(())
    })?;

    let mut out = header(data.len() as u64, o, fnv1a(input), prior.map_or(0, |p| p.hash));
    out.extend(enc.finish());
    out.extend(&data[len * m.batch..]);
    stats.seconds = t0.elapsed().as_secs_f64();
    Ok((out, stats))
}

/// The rANS decoder's state on the GPU: the lane states, the coded words, the read position.
struct GpuLanes {
    x: Tensor<1, Int>,
    words: Tensor<1, Int>,
    at: Tensor<1, Int>,
}

impl GpuLanes {
    fn new(coded: &[u8], lanes: usize, device: &Device) -> Self {
        let dec = Decoder::new(coded, lanes);
        let x: Vec<i32> = dec.x.iter().flat_map(|s| [*s as u32 as i32, (*s >> 32) as u32 as i32]).collect();
        let tail = coded.get(8 * lanes..).unwrap_or(&[]);
        let mut words: Vec<i32> = tail.chunks(4).map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            i32::from_le_bytes(b)
        }).collect();
        // an empty buffer is not a tensor; a zero word reads like the end of the stream
        if words.is_empty() {
            words.push(0);
        }
        let n = words.len();
        Self {
            x: Tensor::from_data(TensorData::new(x, [2 * lanes]), device),
            words: Tensor::from_data(TensorData::new(words, [n]), device),
            at: Tensor::from_data(TensorData::new(vec![0i32], [1]), device),
        }
    }

    /// One step: every stream's row `(byte, freq, total, cum, coder freq)`, read back.
    fn decode(&self, e: Tensor<2, Int>) -> Vec<i32> {
        let rows = crate::fused::rans_decode(e, &self.x, &self.words, &self.at);
        rows.into_data().convert::<i32>().try_to_vec().unwrap()
    }
}

/// One decoding step on the device from one step's [`Slots`]: the forward pass, the rANS step,
/// and training on the bytes it decoded, which never leave the device. Returns the rows (see
/// [`GpuOps::rans_decode`](crate::fused::GpuOps::rans_decode)). Only for graphable predictors.
fn decode_step(model: &mut Box<dyn Predictor>, lanes: &GpuLanes, slots: &Slots) -> Tensor<1, Int> {
    let (ctx, guess, _) = slots.step(0);
    let e = model.device_weights(ctx, guess).expect("a graphable predictor has its weights on the device");
    let rows = crate::fused::rans_decode(e, &lanes.x, &lanes.words, &lanes.at);
    let b = slots.batch;
    // the decoded bytes, column 0 of the rows; the add makes them a fresh contiguous [b] rather
    // than a strided view, whatever a kernel downstream assumes
    model.learn(rows.clone().reshape([b, ROW]).narrow(1, 0, 1).reshape([b]) + 0);
    rows
}

pub fn decompress(file: &[u8], prior: Option<&Prior>, device: &Device, progress: bool) -> Result<(Vec<u8>, Stats), String> {
    let (n, o, checksum, prior_hash, rans) = parse_header(file)?;
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
    let head = if rans { HEADER_LEN } else { HEADER_LEN_RANGE };
    if file.len() < head + tail {
        return Err("file is truncated".into());
    }
    let coded = &file[head..file.len() - tail];

    let mut data = vec![0u8; n];
    let mut model = build(m, prior, device)?;
    let mut ctl = Controller::new(o.lcr_weight);
    let (mut dec, mut old) = if rans { (Some(Decoder::new(coded, m.batch)), None) } else { (None, Some(RangeDecoder::new(coded))) };
    // files from before the rANS format were coded with the earlier match finder
    let mut matcher = if rans { Matcher::new(m.matcher, m.batch, o.block as usize) } else { Matcher::styled(m.matcher, m.batch, MatchStyle::earlier(), 0) };
    let mut stats = Stats::default();
    let (t0, mut last_print) = (Instant::now(), Instant::now());
    let mut syms = vec![0i32; m.batch];
    let mut q = vec![0u32; VOCAB];

    // On the GPU the rANS lanes decode there too: the byte weights stay on the device and only
    // each stream's row comes back, not all 256 frequencies, and no stream waits on another.
    let lanes = (rans && crate::fused::on_gpu(&Tensor::<1, Int>::zeros([1], device)) && m.batch <= 1024 && m.batch % 32 == 0)
        .then(|| GpuLanes::new(coded, m.batch, device));
    // As in the encoder, a step can run as a replayed CUDA graph when nothing in it waits on the
    // host: then the host only writes the step's inputs, replays, and reads the rows back. That
    // needs every step trained (the controller decides per step from its cost otherwise).
    let mut graphing = if lanes.is_some() && o.lcr_weight == 0 && model.graphable() && std::env::var("GMINT_GRAPH").map_or(true, |v| v != "0") {
        Graphing::Idle
    } else {
        Graphing::Off
    };
    let slots = (graphing != Graphing::Off).then(|| Slots::new(&m, 1, device));
    let mut captured: Option<Tensor<1, Int>> = None;

    for t in 0..len {
        if progress && last_print.elapsed().as_secs() >= 5 {
            last_print = Instant::now();
            let done = t * m.batch;
            eprintln!("  {:5.1}%  {:.0} KB/s", 100.0 * done as f64 / n as f64, done as f64 / 1024.0 / t0.elapsed().as_secs_f64());
        }
        // the first steps run plainly, so every kernel is compiled before the capture; the
        // warm-up pass is not part of a step
        if graphing != Graphing::Off && t >= 2 * FLUSH && t + 1 != m.warm {
            let (lanes, slots) = (lanes.as_ref().unwrap(), slots.as_ref().unwrap());
            let mut ctx_v = Vec::with_capacity(m.batch * m.ctx);
            context_values(&mut ctx_v, &data, len, t, &m);
            crate::fused::write_ints(&slots.ctx, ctx_v);
            crate::fused::write_ints(&slots.guess, matcher.guess_values(&data, len, t));
            let probe = &slots.target;
            let out = match graphing {
                Graphing::Ready => {
                    model.refresh_steps();
                    crate::fused::graph(probe, crate::fused::Graph::Replay).map(|_| captured.clone().unwrap())
                }
                Graphing::Idle => crate::fused::graph(probe, crate::fused::Graph::Prepare).map(|_| {
                    graphing = Graphing::Warm;
                    decode_step(&mut model, lanes, slots)
                }),
                _ => {
                    model.refresh_steps();
                    let recorded = crate::fused::graph(probe, crate::fused::Graph::Start).map(|_| decode_step(&mut model, lanes, slots));
                    match recorded.and_then(|out| crate::fused::graph(probe, crate::fused::Graph::Stop).map(|_| out)) {
                        Ok(out) => {
                            graphing = Graphing::Ready;
                            captured = Some(out.clone());
                            crate::fused::graph(probe, crate::fused::Graph::Replay).map(|_| out)
                        }
                        Err(e) => Err(e),
                    }
                }
            };
            let rows = match out {
                Ok(rows) => rows,
                Err(e) => {
                    // a step that was only recorded has not run: run it plainly, and stay plain
                    if progress {
                        eprintln!("  CUDA graph unavailable ({e}); continuing without");
                    }
                    let _ = crate::fused::graph(probe, crate::fused::Graph::Drop);
                    graphing = Graphing::Off;
                    captured = None;
                    decode_step(&mut model, lanes, slots)
                }
            };
            let rows: Vec<i32> = rows.into_data().convert::<i32>().try_to_vec().unwrap();
            for (s, row) in rows.chunks_exact(ROW).enumerate() {
                data[s * len + t] = row[0] as u8;
            }
            let cost = step_cost(&rows);
            matcher.advance(&data, len, t);
            // weight 0 always trains, as the step just did
            ctl.should_update(cost);
            stats.updates += 1;
            model.observe(cost);
            stats.steps += 1;
            continue;
        }
        let guess = matcher.guesses(&data, len, t, device);
        let ctx = contexts(&data, len, t, &m, device);
        if let Some(rows) = lanes.as_ref().and_then(|l| model.device_weights(ctx.clone(), guess.clone()).map(|e| l.decode(e))) {
            for (s, row) in rows.chunks_exact(ROW).enumerate() {
                data[s * len + t] = row[0] as u8;
                syms[s] = row[0];
            }
            let cost = step_cost(&rows);
            matcher.advance(&data, len, t);
            if ctl.should_update(cost) {
                model.learn(targets(&syms, device));
                stats.updates += 1;
            }
            model.observe(cost);
            if t + 1 == m.warm {
                replay(model.as_mut(), &data, len, m.warm, &m, device);
            }
            stats.steps += 1;
            continue;
        }
        let freqs = model.predict(ctx, guess);
        let mut cost = 0;
        for (s, row) in freqs.chunks_exact(VOCAB).enumerate() {
            let total: u64 = row.iter().map(|f| *f as u64).sum();
            let sym = match (&mut dec, &mut old) {
                (Some(dec), _) => {
                    // the coder's frequencies, from the cost frequencies
                    crate::rans::quantize(row, &mut q);
                    let slot = dec.slot(s);
                    let (mut cum, mut sym) = (0u32, 0usize);
                    while cum + q[sym] <= slot {
                        cum += q[sym];
                        sym += 1;
                    }
                    dec.consume(s, cum, q[sym]);
                    sym
                }
                (None, Some(old)) => {
                    let target = old.target(total);
                    let (mut cum, mut sym) = (0u64, 0usize);
                    while cum + row[sym] as u64 <= target {
                        cum += row[sym] as u64;
                        sym += 1;
                    }
                    old.consume(cum, row[sym] as u64);
                    sym
                }
                (None, None) => unreachable!(),
            };
            data[s * len + t] = sym as u8;
            syms[s] = sym as i32;
            cost += log2_q8(total) - log2_q8(row[sym] as u64);
        }
        matcher.advance(&data, len, t);
        if ctl.should_update(cost) {
            model.learn(targets(&syms, device));
            stats.updates += 1;
        }
        model.observe(cost);
        if t + 1 == m.warm {
            replay(model.as_mut(), &data, len, m.warm, &m, device);
        }
        stats.steps += 1;
    }
    if graphing == Graphing::Ready {
        let _ = crate::fused::graph(&slots.as_ref().unwrap().target, crate::fused::Graph::Drop);
    }
    data[len * m.batch..].copy_from_slice(&file[file.len() - tail..]);
    let data = from_streams(&data, m.batch, o.block as usize);
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
        let model = Config { arch: Arch::Attn, ctx: 8, emb: 16, hid: 64, batch: 128, lr: 13, rule, head: Head::Softmax, matcher: false, comps: 4, replicas: 4, short: 0, adapt: 0, warm: 0, norm: false, mask: false, seed: 7 };
        Options { model, lcr_weight, block: 0 }
    }

    #[test]
    fn round_trips_under_every_rule_and_weight() {
        let device = crate::fixed::test_device();
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
        let device = crate::fixed::test_device();
        let data = sample(40_000 + 77);
        for (head, matcher) in [(Head::Tree, false), (Head::RandomTree, false), (Head::Tree, true), (Head::RandomTree, true)] {
            let mut o = options(Rule::Auto, 0);
            o.model.head = head;
            o.model.matcher = matcher;
            let (file, _) = compress(&data, &o, None, &device, false).unwrap();
            assert!(file.len() < data.len() / 2, "{head:?}: only reached {} bytes", file.len());
            assert!(decompress(&file, None, &device, false).unwrap().0 == data, "{head:?}: round trip differs");
        }
    }

    #[test]
    fn every_mix_round_trips() {
        let device = crate::fixed::test_device();
        let data = sample(128 * 400 + 9);
        for (arch, matcher) in [(Arch::Spn, false), (Arch::Spn, true), (Arch::Both, false), (Arch::Both, true)] {
            let mut o = options(Rule::Auto, 0);
            o.model = Config { arch, matcher, ..o.model };
            let (file, _) = compress(&data, &o, None, &device, false).unwrap();
            // how well each mix compresses is measured on real files; here it only has to learn
            assert!(file.len() < data.len() * 9 / 10, "{arch:?}/{matcher}: only reached {} of {}", file.len(), data.len());
            assert!(decompress(&file, None, &device, false).unwrap().0 == data, "{arch:?}/{matcher}: round trip differs");
        }
    }

    #[test]
    fn extras_round_trip() {
        let device = crate::fixed::test_device();
        let data = sample(128 * 400 + 9);
        let base = options(Rule::Auto, 0).model;
        let variants = [
            ("tree head in a mix", Config { arch: Arch::Both, head: Head::Tree, matcher: true, ..base }),
            ("short branch", Config { short: 4, ..base }),
            ("short branch + network", Config { arch: Arch::Both, short: 2, ..base }),
            ("prune", Config { arch: Arch::Spn, adapt: 1, ..base }),
            ("prune and regrow", Config { arch: Arch::Both, adapt: 2, ..base }),
            ("warm-up", Config { warm: 96, matcher: true, ..base }),
            ("warm-up, off the window", Config { warm: 50, ..base }),
            ("normalised hidden layer", Config { norm: true, ..base }),
            ("masked clamps", Config { mask: true, ..base }),
            ("normalised, tree head, short branch", Config { norm: true, head: Head::Tree, short: 2, matcher: true, ..base }),
        ];
        for (name, model) in variants {
            let o = Options { model, lcr_weight: 0, block: 0 };
            let (file, _) = compress(&data, &o, None, &device, false).unwrap();
            assert!(file.len() < data.len() * 9 / 10, "{name}: only reached {} of {}", file.len(), data.len());
            assert!(decompress(&file, None, &device, false).unwrap().0 == data, "{name}: round trip differs");
        }
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
        let device = crate::fixed::test_device();
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

    /// A file in the range coder's format, written by commit 45c0779 with the match finder on
    /// (testdata/), still decodes: it needs the earlier match finder, not today's.
    #[test]
    fn range_coded_file_still_decodes() {
        let device = crate::fixed::test_device();
        let file = include_bytes!("../testdata/gmint1-match.gm");
        let want = include_bytes!("../testdata/gmint1-source.txt");
        assert_eq!(&file[..6], MAGIC_RANGE);
        assert_eq!(decompress(file, None, &device, false).unwrap().0, want.to_vec());
    }

    #[test]
    fn stream_order_inverts() {
        let data: Vec<u8> = (0..128 * 1000 + 37).map(|i| (i * 7 % 251) as u8).collect();
        for block in [0usize, 1, 3, 100, 512, 999, 1000, 5000] {
            let ordered = to_streams(&data, 128, block);
            assert_eq!(from_streams(&ordered, 128, block), data, "block {block}");
            if block == 0 || block >= 1000 {
                assert_eq!(ordered, data, "block {block} should leave the input alone");
            }
        }
        // stream 1 starts with input block 1, then block 1 + batch
        let ordered = to_streams(&data, 128, 100);
        assert_eq!(&ordered[1000..1100], &data[100..200]);
        assert_eq!(&ordered[1100..1200], &data[(128 + 1) * 100..(128 + 2) * 100]);
    }

    #[test]
    fn input_shorter_than_a_step_is_stored() {
        let device = crate::fixed::test_device();
        let data = sample(100);
        let (file, _) = compress(&data, &options(Rule::Norm, 16), None, &device, false).unwrap();
        assert_eq!(decompress(&file, None, &device, false).unwrap().0, data);
    }

    #[test]
    fn prior_helps_and_must_match() {
        let device = crate::fixed::test_device();
        for (norm, mask) in [(false, false), (true, false), (false, true)] {
            prior_round_trip(norm, mask, &device);
        }
    }

    fn prior_round_trip(norm: bool, mask: bool, device: &Device) {
        let mut o = options(Rule::Auto, 0);
        (o.model.norm, o.model.mask) = (norm, mask);
        let mut model = Model::new(o.model, &device);
        train_pass(&mut model, &sample(60_000), 0, &device, |_, _, _| {});
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
        let device = crate::fixed::test_device();
        let data = sample(20_000);
        let (mut file, _) = compress(&data, &options(Rule::Norm, 16), None, &device, false).unwrap();
        let mid = file.len() / 2;
        file[mid] ^= 0x40;
        assert!(decompress(&file, None, &device, false).is_err());
        assert!(decompress(&file[..20], None, &device, false).is_err());
    }
}
