//! Integer-only byte predictor: one attention layer over the context bytes, then a two-layer
//! head, with forward, backward and SGD all on `Tensor<_, Int>` (i32 fixed point).
//!
//! No float op touches model state, so any backend with exact i32 add/mul/div gives the same
//! bits; that is what lets a decoder on other hardware replay the encoder's training. Magnitudes
//! are clamped so no intermediate leaves i32, and the bound for each product is noted where it
//! is formed.
//!
//! Weights are kept as Q16 "shadow" values and used as Q8 (`shadow >> 8`), so updates smaller
//! than one forward step still accumulate. Activations are Q8, logits Q8, probabilities Q12.
//! Clamps on q/k/v, the residual and the error signals are passed through as identity in the
//! backward pass.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fused::{self, Mat, UNBOUNDED};
use crate::optim::{Optimizer, Rule};

pub const VOCAB: usize = 256;
/// Gradient sums are formed over this many rows at a time so they stay inside i32.
const CHUNK: usize = 128;

const ACT_MAX: i32 = 1023;
const ERR_MAX: i32 = 4095;
/// Shadow bound whose forward value stays within +-1023.
const SHADOW_MAX: i32 = (1 << 18) - 1;
/// Shadow bound for embeddings, whose forward value stays within +-128.
const SHADOW_EMB_MAX: i32 = (1 << 15) - 1;
pub const LOGIT_SPAN: i32 = 4095;
/// Normalised hidden layer: pre-activations are cut to this before the gain is applied, and the
/// gain (Q8) stays in `NORM_GAIN_MIN..=NORM_GAIN_MAX`, so their product stays under 2^29.
const NORM_IN: i32 = 1 << 17;
const NORM_GAIN_MIN: i32 = 16;
const NORM_GAIN_MAX: i32 = 4096;
/// Spread (mean absolute deviation, Q8) each unit's pre-activation is scaled to. 2.0 measured
/// best on KiCad against 0.5, 1.0, 3.0, 4.0 and 8.0.
const NORM_SPREAD: i32 = 512;
/// log2 of the number of steps the running statistics average over.
const NORM_EMA: u32 = 4;
const LUT_ONE: i64 = 1 << 20;
/// log2 of how much smaller each weight group's gradient scale is than the head's, measured
/// against autodiff in the tests; the copy gain's input is a bare 1 where the head's inputs are
/// Q8 activations.
const SGD_EMB: i32 = 11;
const SGD_VALUE: i32 = 3;
const SGD_QUERY_KEY: i32 = 1;
const SGD_COPY: i32 = 6;
/// The offsets in the optimizer's group order: emb, pos, wq, wk, wv, w1, w2, copy.
const GROUP_OFFSETS: [i32; 8] = [SGD_EMB, SGD_EMB, SGD_QUERY_KEY, SGD_QUERY_KEY, SGD_VALUE, 0, 0, SGD_COPY];
/// round(2^32 * exp(-1/256)): one Q8 logit step, as a Q32 multiplier.
const EXP_STEP_Q32: i64 = 4_278_222_805;

/// xorshift64*, so initial weights never depend on a backend RNG.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn ints(&mut self, n: usize, span: i32) -> Vec<i32> {
        (0..n).map(|_| (self.next() >> 33) as i32 % (2 * span + 1) - span).collect()
    }
}

/// exp(-d/256) in Q20 for d in 0..=LOGIT_SPAN, by integer recurrence (no libm, so identical everywhere).
pub fn exp_lut() -> Vec<i32> {
    let mut v = Vec::with_capacity(LOGIT_SPAN as usize + 1);
    let mut cur = LUT_ONE;
    for _ in 0..=LOGIT_SPAN {
        v.push(cur as i32);
        cur = (cur * EXP_STEP_Q32) >> 32;
    }
    v
}

/// Backend the tests run on: the CPU one unless `GMINT_TEST_DEVICE=cuda`. On CUDA the test also
/// takes the GPU for itself until its thread ends: the encoder records CUDA graphs, and a graph
/// recording is broken by any CUDA call from another thread of the process meanwhile.
#[cfg(test)]
pub fn test_device() -> Device {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    thread_local! {
        static HELD: std::cell::RefCell<Option<std::sync::MutexGuard<'static, ()>>> = const { std::cell::RefCell::new(None) };
    }
    let mut d = match std::env::var("GMINT_TEST_DEVICE").as_deref() {
        Ok("cuda") => {
            HELD.with(|h| {
                let mut h = h.borrow_mut();
                if h.is_none() {
                    *h = Some(GPU.lock().unwrap_or_else(|e| e.into_inner()));
                }
            });
            Device::cuda(0)
        }
        _ => Device::flex(),
    };
    let _ = d.configure(burn::tensor::IntDType::I32);
    d
}

/// `[VOCAB, VOCAB]`: row `v` is 1 at the byte values below `v`. Selecting rows by the true bytes
/// masks the frequencies that precede them.
pub fn below_table(device: &Device) -> Tensor<2, Int> {
    let v: Vec<i32> = (0..VOCAB * VOCAB).map(|i| ((i % VOCAB) < (i / VOCAB)) as i32).collect();
    Tensor::from_data(TensorData::new(v, [VOCAB, VOCAB]), device)
}

/// One-hot rows by table lookup: `one_hot` is about 70 times slower than a row select on CUDA
/// (burn 0.22.1), and a step needs several.
pub fn one_hot_rows(n: usize, device: &Device) -> Tensor<2, Int> {
    let v: Vec<i32> = (0..n * n).map(|i| (i % n == i / n) as i32).collect();
    Tensor::from_data(TensorData::new(v, [n, n]), device)
}

/// From Q20 byte weights `[B, VOCAB]` and the true bytes: (cumulative frequency below the byte,
/// its frequency, row total) as `[B, 3]`, in the coder's frequencies (`weight >> 4`, plus 1).
pub fn coded(e: Tensor<2, Int>, target: Tensor<1, Int>, below: &Tensor<2, Int>) -> Tensor<2, Int> {
    let [b, _] = e.dims();
    if fused::on_gpu(&e) {
        return fused::coded(e, target);
    }
    // each frequency is at most 2^16 + 1, so a row sums to under 2^25
    let f = e.bitwise_right_shift_scalar(4) + 1;
    let cum = (f.clone() * below.clone().select(0, target.clone())).sum_dim(1);
    let own = f.clone().gather(1, target.reshape([b, 1]));
    Tensor::cat(vec![cum, own, f.sum_dim(1)], 1)
}

/// Temporary A/B switch: train a tree head through the match boost (see `Model::head_error`).
fn through() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GMINT_THROUGH").is_some())
}

/// Temporary A/B switch while the leaky activation is being evaluated.
fn leaky() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GMINT_LEAKY").is_some())
}

/// Integer softmax over the last dim of Q8 logits, given the [`exp_lut`] table as a tensor:
/// returns (Q20 weights, Q12 probabilities).
pub fn softmax_q12<const D: usize>(lut: &Tensor<1, Int>, logits: Tensor<D, Int>) -> (Tensor<D, Int>, Tensor<D, Int>) {
    let dims = logits.dims();
    let n: usize = dims.iter().product();
    let d = (logits.clone().max_dim(D - 1) - logits).clamp(0, LOGIT_SPAN);
    let e = lut.clone().select(0, d.reshape([n])).reshape(dims);
    // sum(e) is in [2^20, 2^28]; it is non-negative, so the raw shift is safe
    let p = e.clone() / e.clone().sum_dim(D - 1).bitwise_right_shift_scalar(12);
    (e, p)
}

/// Floor division by 2^s for |t| < 2^30.
///
/// burn's `bitwise_right_shift_scalar` on a negative i32 is arithmetic on the CPU backend and
/// logical on CUDA (burn 0.22.1), so the value is biased non-negative first: the shift then means
/// the same thing everywhere.
pub fn shr_floor<const D: usize>(t: Tensor<D, Int>, s: u32) -> Tensor<D, Int> {
    const BIAS_LOG2: u32 = 30;
    if s == 0 {
        return t;
    }
    (t + (1i32 << BIAS_LOG2)).bitwise_right_shift_scalar(s as i32) - (1i32 << (BIAS_LOG2 - s))
}

/// Division by 2^s rounded to nearest, so negative updates are not biased toward -1.
pub fn shr_round<const D: usize>(t: Tensor<D, Int>, s: u32) -> Tensor<D, Int> {
    // [`shr_floor`] of `t + half`, with the two additions folded into one: every tensor op costs
    // about the same few microseconds whatever it does, so the count is what matters
    const BIAS_LOG2: u32 = 30;
    (t + ((1i32 << BIAS_LOG2) + (1i32 << (s - 1)))).bitwise_right_shift_scalar(s as i32) - (1i32 << (BIAS_LOG2 - s))
}

/// `x^T @ y` in chunks of [`CHUNK`] rows; see [`fused::chunked_xty`].
fn xty(x: Tensor<2, Int>, y: Tensor<2, Int>) -> Tensor<2, Int> {
    fused::chunked_xty(x, y, CHUNK)
}

/// What the codec needs from a model. Implement this to try a different predictor.
pub trait Predictor {
    /// Coder frequencies for the next byte of every stream, `[batch * VOCAB]` row-major, each at
    /// least 1. `ctx` is `[batch, ctx]` byte values. `guess` is `[batch, 2]`: the byte a long-range
    /// match proposes and how far to trust it (0 none, 1 short match, 2 long); all zeros when the
    /// match finder is off. Must not depend on the bytes being predicted.
    fn predict(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Vec<i32>;
    /// Train on the batch just predicted, given its true bytes.
    fn learn(&mut self, target: Tensor<1, Int>);
    /// What the encoder needs and nothing more: for every stream the cumulative frequency below
    /// the true byte, that byte's frequency, and the row total, `[batch, 3]`, left on the device.
    /// The encoder knows the bytes in advance, so it can run many steps and read these back in
    /// one go. The default goes through [`Self::predict`], which reads all 256 frequencies back.
    fn predict_coded(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>, target: Tensor<1, Int>) -> Tensor<2, Int> {
        let device = target.device();
        let syms: Vec<i32> = target.into_data().convert::<i32>().try_to_vec().unwrap();
        let freqs = self.predict(ctx, guess);
        let mut out = Vec::with_capacity(syms.len() * 3);
        for (row, sym) in freqs.chunks_exact(VOCAB).zip(&syms) {
            let cum: i32 = row[..*sym as usize].iter().sum();
            out.extend([cum, row[*sym as usize], cum + row[*sym as usize..].iter().sum::<i32>()]);
        }
        Tensor::from_data(TensorData::new(out, [syms.len(), 3]), &device)
    }
    /// Told the Q8 code length of every step, trained on or not, so a model can steer itself.
    /// Called after [`Self::learn`] for the same step, so a change takes effect from the next one.
    fn observe(&mut self, _cost_q8: u64) {}
    /// One line on how much of the model can still learn (see [`Model::health`]); empty when the
    /// predictor has nothing to report. Reads tensors back, so it is not for every step.
    fn health(&self) -> String {
        String::new()
    }
    /// Whether a run of steps can be captured as a CUDA graph and replayed: every step's work is
    /// on the GPU, and every piece of state a step changes is changed in place, so a replay
    /// carries it forward. See [`crate::codec`].
    fn graphable(&self) -> bool {
        false
    }
    /// Brings the device copies of the optimizers' steps up to date; a replayed step does not
    /// run the code that would.
    fn refresh_steps(&mut self) {}
}

/// Which predictor codes the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arch {
    /// One attention layer and a two-layer head ([`Model`]).
    Attn,
    /// A random sum-product network ([`crate::spn::Spn`]).
    Spn,
    /// Both, with their scores for each candidate byte added before the softmax.
    Both,
}

impl Arch {
    pub fn id(self) -> u8 {
        self as u8
    }
    pub fn from_id(id: u8) -> Option<Self> {
        [Self::Attn, Self::Spn, Self::Both].into_iter().find(|a| a.id() == id)
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "attn" => Some(Self::Attn),
            "spn" => Some(Self::Spn),
            "both" => Some(Self::Both),
            _ => None,
        }
    }
}

/// How the head turns its outputs into a distribution over the next byte.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Head {
    /// One logit per byte value and a 256-way softmax.
    Softmax,
    /// A depth-8 binary tree: 255 node logits, each a sigmoid deciding one bit, most significant
    /// first. A byte's probability is the product down its path, so it can get far closer to 1
    /// than the softmax's fixed-point floor allows, and only 8 outputs learn per byte.
    Tree,
    /// [`Head::Tree`] with the byte values dealt onto the leaves in a seeded random order, so the
    /// tree's splits do not follow the bytes' bit patterns.
    RandomTree,
}

impl Head {
    pub fn id(self) -> u8 {
        self as u8
    }
    pub fn from_id(id: u8) -> Option<Self> {
        [Self::Softmax, Self::Tree, Self::RandomTree].into_iter().find(|h| h.id() == id)
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "softmax" => Some(Self::Softmax),
            "tree" => Some(Self::Tree),
            "rtree" => Some(Self::RandomTree),
            _ => None,
        }
    }
    fn outputs(self) -> usize {
        if self == Self::Softmax { VOCAB } else { VOCAB - 1 }
    }
}

/// 256 ln(p / 4096) for a Q12 probability p in 0..=4096 (p = 0 is cut to `-LOGIT_SPAN`), by
/// inverting the exp table: the smallest d with exp(-d/256) <= p / 4096.
fn log_table() -> Vec<i32> {
    let exp = exp_lut();
    let mut logp = vec![-LOGIT_SPAN; 4097];
    let mut d = LOGIT_SPAN as usize;
    for p in 1..=4096usize {
        while d > 0 && (exp[d - 1] as i64) <= (p as i64) << 8 {
            d -= 1;
        }
        logp[p] = -(d as i32);
    }
    logp
}

/// Constant tables of a tree head.
struct Tree {
    /// Per level: the node each byte value passes through, `[VOCAB]`.
    node: Vec<Tensor<1, Int>>,
    /// Per level, `[1, VOCAB]`: +1 where the byte takes the 1-branch, -1 otherwise; and 0 or 4096,
    /// so `p * sign + offset` is the Q12 probability of the branch the byte takes.
    sign: Vec<Tensor<2, Int>>,
    offset: Vec<Tensor<2, Int>>,
    /// `[VOCAB, 255]`: 1 where a node is on the byte's path; and 4096 where the byte's bit there is 1.
    on_path: Tensor<2, Int>,
    bit_q12: Tensor<2, Int>,
    /// sigma(-d/256) in Q12 for d in 0..=LOGIT_SPAN.
    sigmoid: Tensor<1, Int>,
    /// `[VOCAB, 255]`: 1 where the byte lies in a node's 1-branch subtree.
    right: Tensor<2, Int>,
    /// `[255, VOCAB]`: 1 where the byte lies in the node's 1-branch (0-branch) subtree.
    right_t: Tensor<2, Int>,
    left_t: Tensor<2, Int>,
    /// 256 * ln(p / 4096) for a Q12 probability p in 0..=4096.
    logp: Tensor<1, Int>,
    /// Leaf position of every byte value, `[VOCAB]`, from which a kernel derives the path.
    leaf: Tensor<1, Int>,
    /// The byte value at every leaf position, `[VOCAB]`: the inverse of `leaf`.
    byte_at: Tensor<1, Int>,
}

impl Tree {
    fn new(random: bool, seed: u64, device: &Device) -> Self {
        // leaf position of every byte value
        let mut leaf: Vec<usize> = (0..VOCAB).collect();
        if random {
            let mut rng = Rng(seed ^ 0x7265_6574);
            for i in (1..VOCAB).rev() {
                leaf.swap(i, (rng.next() >> 33) as usize % (i + 1));
            }
        }
        let (mut node, mut sign, mut offset) = (Vec::new(), Vec::new(), Vec::new());
        let (mut on_path, mut bit_q12) = (vec![0i32; VOCAB * (VOCAB - 1)], vec![0i32; VOCAB * (VOCAB - 1)]);
        for level in 0..8 {
            // heap numbering: level l holds nodes (2^l - 1)..(2^(l+1) - 1)
            let ids: Vec<i32> = leaf.iter().map(|p| ((1 << level) - 1 + (p >> (8 - level))) as i32).collect();
            let bits: Vec<i32> = leaf.iter().map(|p| ((p >> (7 - level)) & 1) as i32).collect();
            for v in 0..VOCAB {
                on_path[v * (VOCAB - 1) + ids[v] as usize] = 1;
                bit_q12[v * (VOCAB - 1) + ids[v] as usize] = bits[v] * 4096;
            }
            let row = |v: Vec<i32>| Tensor::<2, Int>::from_data(TensorData::new(v, [1, VOCAB]), device);
            sign.push(row(bits.iter().map(|b| 2 * b - 1).collect()));
            offset.push(row(bits.iter().map(|b| (1 - b) * 4096).collect()));
            node.push(Tensor::from_data(TensorData::new(ids, [VOCAB]), device));
        }
        let table = |v: Vec<i32>| Tensor::<2, Int>::from_data(TensorData::new(v, [VOCAB, VOCAB - 1]), device);
        let right: Vec<i32> = bit_q12.iter().map(|b| b / 4096).collect();
        let left: Vec<i32> = on_path.iter().zip(&right).map(|(on, r)| on - r).collect();
        let (right_t, left_t) = (table(right.clone()).transpose(), table(left).transpose());
        let logp = log_table();
        // sigma(-x) = e^-x / (1 + e^-x), from the same integer exp table as the softmax
        let sigmoid: Vec<i32> = exp_lut().iter().map(|e| ((*e as i64 * 4096) / (LUT_ONE + *e as i64)) as i32).collect();
        Self {
            node,
            sign,
            offset,
            on_path: table(on_path),
            bit_q12: table(bit_q12),
            sigmoid: Tensor::from_data(TensorData::new(sigmoid, [LOGIT_SPAN as usize + 1]), device),
            right: table(right),
            right_t,
            left_t,
            logp: Tensor::from_data(TensorData::new(logp, [4097]), device),
            leaf: Tensor::from_data(TensorData::new(leaf.iter().map(|p| *p as i32).collect::<Vec<i32>>(), [VOCAB]), device),
            byte_at: {
                let mut at = vec![0i32; VOCAB];
                for (v, p) in leaf.iter().enumerate() {
                    at[*p] = v as i32;
                }
                Tensor::from_data(TensorData::new(at, [VOCAB]), device)
            },
        }
    }

    /// Q8 node logits `[B, 255]` to Q12 probabilities of each node's 1-branch, kept off 0 and 1.
    fn probs(&self, z: Tensor<2, Int>) -> Tensor<2, Int> {
        let [b, n] = z.dims();
        let low = self.sigmoid.clone().select(0, z.clone().abs().clamp(0, LOGIT_SPAN).reshape([b * n])).reshape([b, n]);
        // low is sigma(-|z|); for z >= 0 the answer is its complement
        // (z + 1).clamp(0, 1) is 1 exactly where z >= 0; a compare costs several times as much
        (low.clone() + (z + 1).clamp(0, 1) * (low * -2 + 4096)).clamp(1, 4095)
    }

    /// Node probabilities to each byte's Q8 log-probability, `[B, VOCAB]`: the sum of the logs of
    /// the branches on its path. This is what lets a tree head join a mix of per-byte scores.
    fn log_scores(&self, p: &Tensor<2, Int>) -> Tensor<2, Int> {
        if fused::on_gpu(p) {
            return fused::tree_log_scores(p, &self.leaf, &self.logp);
        }
        let [b, n] = p.dims();
        let log = |q: Tensor<2, Int>| self.logp.clone().select(0, q.reshape([b * n])).reshape([b, n]);
        // every node's log-probability of each branch, then summed along each byte's path by the
        // path tables: 8 terms of magnitude at most 4095 per byte
        log(p.clone()).matmul(self.right_t.clone()) + log(p.clone() * -1 + 4096).matmul(self.left_t.clone())
    }

    /// Node probabilities to byte weights `[B, VOCAB]` on the softmax's Q20 scale.
    fn leaves(&self, p: &Tensor<2, Int>) -> Tensor<2, Int> {
        let mut acc: Option<Tensor<2, Int>> = None;
        for level in 0..8 {
            let branch = p.clone().select(1, self.node[level].clone()) * self.sign[level].clone() + self.offset[level].clone();
            // acc <= 2^16 and branch <= 2^12; both non-negative, so the raw shift is safe
            acc = Some(match acc {
                None => branch * 16,
                Some(a) => (a * branch).bitwise_right_shift_scalar(12),
            });
        }
        acc.unwrap() * 16
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub arch: Arch,
    /// Context length in bytes; a power of two.
    pub ctx: usize,
    pub emb: usize,
    pub hid: usize,
    /// Streams coded per step; a power of two, at least 128.
    pub batch: usize,
    /// Learning rate; each +1 halves the step. See [`Rule`] for what it scales.
    pub lr: u32,
    pub rule: Rule,
    pub head: Head,
    /// Feed the head a long-range match proposal.
    pub matcher: bool,
    /// Sum-product network: distributions per leaf region (and sum nodes per layer), and the
    /// number of random replicas.
    pub comps: usize,
    pub replicas: usize,
    /// Extra attention branch over only the last `short` bytes (0 for none, else 2 or 4).
    pub short: usize,
    /// Sum-product network structure adaptation: 0 off, 1 prune weak children, 2 prune and regrow.
    pub adapt: u8,
    /// Steps after which the bytes coded so far are trained on once more (0 for never).
    pub warm: usize,
    /// Normalise the hidden layer's pre-activations per unit with running statistics.
    pub norm: bool,
    /// Pass no error back through a forward clamp where it held (on q, k, v and the residual;
    /// the hidden layer's ReLU always does this). Off, the clamps pass errors straight through.
    pub mask: bool,
    pub seed: u64,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.arch != Arch::Attn {
            // the batch bound keeps the network's batch-summed gradients inside i32
            let ok = self.ctx == crate::spn::CTX
                // comps^2 is a sum node's fan-in, which the log table covers up to 256
                && (2..=16).contains(&self.comps)
                && (1..=crate::spn::MAX_FAN_IN).contains(&self.replicas)
                && self.batch.is_power_of_two()
                && (CHUNK..=512).contains(&self.batch)
                && (6..=22).contains(&self.lr);
            if !ok {
                return Err(format!("unsupported sum-product network shape: {self:?}"));
            }
            if self.arch == Arch::Spn {
                return Ok(());
            }
        }
        if ![0, 2, 4].contains(&self.short) || self.adapt > 2 || self.warm > u16::MAX as usize {
            return Err(format!("unsupported options: {self:?}"));
        }
        let ok = self.ctx.is_power_of_two()
            && self.ctx <= 64
            && self.batch.is_power_of_two()
            && (CHUNK..=1 << 15).contains(&self.batch)
            // head products are bounded by ctx*emb and hid inputs of magnitude 2^10 each
            && self.ctx * self.emb <= 256
            && self.emb <= 64
            && (1..=256).contains(&self.hid)
            && (6..=22).contains(&self.lr);
        if ok { Ok(()) } else { Err(format!("unsupported model shape: {self:?}")) }
    }
}

/// The trainable state, as Q16 shadows.
pub struct Params {
    pub emb: Tensor<2, Int>,
    pub pos: Tensor<2, Int>,
    pub wq: Tensor<2, Int>,
    pub wk: Tensor<2, Int>,
    pub wv: Tensor<2, Int>,
    pub w1: Tensor<2, Int>,
    pub w2: Tensor<2, Int>,
    /// `[3, 1]`: logit added to the proposed byte, per trust level. Used as `shadow >> 6`, so it
    /// can reach 16 nats.
    pub copy: Tensor<2, Int>,
}

#[cfg(test)]
impl Params {
    pub fn each(&self) -> [&Tensor<2, Int>; 8] {
        [&self.emb, &self.pos, &self.wq, &self.wk, &self.wv, &self.w1, &self.w2, &self.copy]
    }
}

pub struct Model {
    pub cfg: Config,
    pub p: Params,
    lut: Tensor<1, Int>,
    below: Tensor<2, Int>,
    /// Identity tables, selected by row in place of `one_hot`.
    eye: Tensor<2, Int>,
    eye3: Tensor<2, Int>,
    tree: Option<Tree>,
    opt: Optimizer,
    /// The forward pass awaiting its targets.
    pending: Option<Cache>,
    /// Hidden activations of the last trained batch, kept for [`Self::health`].
    last_hidden: Option<Tensor<2, Int>>,
    /// Normalised hidden layer (`cfg.norm`): per unit, the gain that brings the spread of its
    /// pre-activation to [`NORM_SPREAD`].
    norm: Norm,
    /// Constant zeros: the match proposal of a model without one, `[B, 2]`, and the copy gain's
    /// gradient when nothing trains it, `[3, 1]`.
    no_guess: Tensor<2, Int>,
    zero3: Tensor<2, Int>,
    /// [`log_table`], for [`Model::pending_loss`] with a softmax head.
    logp: Tensor<1, Int>,
    device: Device,
}

/// Running spread of each hidden unit's pre-activation, and the gain that normalises it. The
/// forward pass reads the gain as a constant, so the backward pass through the layer is a plain
/// multiply by it, and both sides of the codec hold the same value at every step.
///
/// Only the spread is normalised. Subtracting the running mean as well was tried and cost over a
/// third of the ratio: the lower layers' weights ran to their clamps.
struct Norm {
    /// The gain scaled by 2^[`NORM_EMA`], so the exponential average loses nothing to rounding.
    gain_acc: Tensor<2, Int>,
    /// Q8, `[1, hid]`.
    gain: Tensor<2, Int>,
}

impl Norm {
    fn new(hid: usize, device: &Device) -> Self {
        // gain 1.0: the layer starts as the identity
        let gain = Tensor::<2, Int>::zeros([1, hid], device) + 256;
        Self { gain_acc: gain.clone() * (1 << NORM_EMA), gain }
    }

    /// Folds one batch of pre-activations `[B, hid]` into the running gain.
    fn observe(&mut self, z1: Tensor<2, Int>) {
        let [b, hid] = z1.dims();
        let rows = b.trailing_zeros();
        // cut so that a column of deviations sums to under 2^29
        let lim = ((1i32 << 28) / b as i32).min(NORM_IN);
        let z = z1.clamp(-lim, lim);
        let mean = shr_floor(z.clone().sum_dim(0), rows);
        // mean absolute deviation: a spread that needs no square root
        let mad = shr_floor((z - mean).abs().sum_dim(0), rows);
        // both sides non-negative, so the division means the same on every backend
        let gain = ((Tensor::<2, Int>::zeros([1, hid], &mad.device()) + 256 * NORM_SPREAD) / (mad + 1)).clamp(NORM_GAIN_MIN, NORM_GAIN_MAX);
        self.gain_acc = self.gain_acc.clone() + gain - self.gain.clone();
        self.gain = shr_floor(self.gain_acc.clone(), NORM_EMA);
    }
}

/// What the attention block hands on: the context bytes, `[B t]`; the embedded context and its
/// queries, keys and values, `[B, t, e]`; the attention weights, `[B, t, t]`; and the block's
/// output, `[B, t e]`.
struct Attended {
    idx: Tensor<1, Int>,
    x: Tensor<3, Int>,
    q: Tensor<3, Int>,
    k: Tensor<3, Int>,
    v: Tensor<3, Int>,
    a: Tensor<3, Int>,
    zf: Tensor<2, Int>,
}

/// Everything one forward pass leaves behind for the coder and for the update.
pub struct Cache {
    idx: Tensor<1, Int>,
    x: Tensor<3, Int>,
    q: Tensor<3, Int>,
    k: Tensor<3, Int>,
    v: Tensor<3, Int>,
    a: Tensor<3, Int>,
    zf: Tensor<2, Int>,
    h: Tensor<2, Int>,
    /// With the normalised hidden layer: the raw pre-activations and the gain they were scaled by.
    norm: Option<(Tensor<2, Int>, Tensor<2, Int>)>,
    /// Output scores before the match boost and the head.
    out: Tensor<2, Int>,
    e: Tensor<2, Int>,
    /// Q12: byte probabilities for the softmax head, node probabilities for a tree head.
    prob: Tensor<2, Int>,
    /// The weights the pass was made with, as Q16 shadows.
    wq: Tensor<2, Int>,
    wk: Tensor<2, Int>,
    wv: Tensor<2, Int>,
    w1: Tensor<2, Int>,
    w2: Tensor<2, Int>,
    /// The match proposal as it came, `[B, 2]`; see [`split_guess`].
    guess_raw: Tensor<2, Int>,
}

impl Cache {
    /// Coder frequencies, `[B * VOCAB]` row-major: every symbol at least 1, at most 2^16 + 1.
    pub fn freqs(&self) -> Vec<i32> {
        let f = self.e.clone().bitwise_right_shift_scalar(4) + 1;
        f.into_data().convert::<i32>().try_to_vec().unwrap()
    }
}

impl Model {
    pub fn new(cfg: Config, device: &Device) -> Self {
        cfg.validate().expect("config");
        let mut rng = Rng(cfg.seed);
        let (t, e, h) = (cfg.ctx, cfg.emb, cfg.hid);
        let mut mk = |r: usize, c: usize, span: i32| -> Tensor<2, Int> {
            Tensor::from_data(TensorData::new(rng.ints(r * c, span << 8), [r, c]), device)
        };
        let p = Params {
            emb: mk(VOCAB, e, 96),
            pos: mk(t, e, 24),
            wq: mk(e, e, 64),
            wk: mk(e, e, 64),
            wv: mk(e, e, 64),
            w1: mk(t * e, h, 24),
            w2: mk(h, cfg.head.outputs(), 16),
            copy: Tensor::zeros([3, 1], device),
        };
        let tree = (cfg.head != Head::Softmax).then(|| Tree::new(cfg.head == Head::RandomTree, cfg.seed, device));
        let lut = Tensor::from_data(TensorData::new(exp_lut(), [LOGIT_SPAN as usize + 1]), device);
        Self { cfg, p, lut, below: below_table(device), eye: one_hot_rows(VOCAB, device), eye3: one_hot_rows(3, device), tree, opt: Optimizer::new(cfg.rule, cfg.lr, 8).with_offsets(&GROUP_OFFSETS), pending: None, last_hidden: None, norm: Norm::new(cfg.hid, device), no_guess: Tensor::zeros([cfg.batch, 2], device), zero3: Tensor::zeros([3, 1], device), logp: Tensor::from_data(TensorData::new(log_table(), [4097]), device), device: device.clone() }
    }

    fn softmax<const D: usize>(&self, logits: Tensor<D, Int>) -> (Tensor<D, Int>, Tensor<D, Int>) {
        softmax_q12(&self.lut, logits)
    }

    /// The network up to its output scores, `[B, outputs]` in Q8, for a caller that applies its
    /// own head (see [`crate::mix::Mix`]). Train with [`Self::learn_from`].
    pub fn scores(&self, ctx: Tensor<2, Int>) -> (Tensor<2, Int>, Cache) {
        let cache = self.pass(ctx, self.no_guess.clone(), false);
        match &self.tree {
            None => (cache.out.clone(), cache),
            // a tree head's per-byte score is the byte's log-probability under the tree
            Some(tree) => (tree.log_scores(&cache.prob), cache),
        }
    }

    /// Trains on the error `g = dL/dscores` (Q12) for a pass made by [`Self::scores`].
    pub fn learn_from(&mut self, c: &Cache, g: Tensor<2, Int>) {
        let g = match &self.tree {
            None => g,
            // Back from per-byte errors to node logits: a node's logit raises the log-probability
            // of the bytes in its 1-branch by (1 - sigma) and lowers the rest of its subtree by
            // sigma. |g @ table| <= sum |g| <= 2^13, and times a Q12 probability stays under 2^26.
            Some(tree) if fused::on_gpu(&g) => fused::tree_back(g, &c.prob, &tree.byte_at),
            Some(tree) => {
                let right = g.clone().matmul(tree.right.clone());
                let under = g.matmul(tree.on_path.clone());
                (right - shr_round(c.prob.clone() * under, 12)).clamp(-4096, 4096)
            }
        };
        self.train(c, g, self.zero3.clone());
    }

    /// One training step from the error at the output scores and the copy gain's gradient. On
    /// the GPU under a plain SGD rule every weight group's gradient and step are one kernel.
    fn train(&mut self, c: &Cache, g: Tensor<2, Int>, dcopy: Tensor<2, Int>) {
        let steps = if fused::on_gpu(&g) { self.opt.device_steps(&self.device) } else { None };
        if let Some(steps) = steps {
            self.opt.tick();
            let mut p = self.grads_from(c, g, dcopy, Some(&steps));
            // without the match finder the gain's gradient is zero and so is the gain
            p.copy = if self.cfg.matcher { self.opt.step(7, &self.p.copy, p.copy, SGD_COPY, SHADOW_MAX) } else { self.p.copy.clone() };
            self.p = p;
            crate::ops::prof::lap(&self.device, "b: optimiser");
        } else {
            let grads = self.grads_from(c, g, dcopy, None);
            self.apply(grads);
        }
        self.trained(c);
    }

    /// Bookkeeping after a training step on the pass `c`.
    fn trained(&mut self, c: &Cache) {
        self.last_hidden = Some(c.h.clone());
        if let Some((z1, _)) = &c.norm {
            self.norm.observe(z1.clone());
        }
    }

    /// `ctx` is `[B, ctx]` byte values. Nothing here depends on the bytes being predicted.
    pub fn forward(&self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Cache {
        self.pass(ctx, guess, true)
    }

    /// [`Self::forward`]; without `weights` the byte weights are left out (`e` is then a
    /// placeholder, and so is a softmax head's `prob`), for a mix that applies its own head.
    fn pass(&self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>, weights: bool) -> Cache {
        let Config { ctx: t, emb: e, batch: b, .. } = self.cfg;
        // The weights go to the fused matmuls as shadows, which shift them to Q8 as they read.
        let (wq, wk, wv, w1, w2) = (self.p.wq.clone(), self.p.wk.clone(), self.p.wv.clone(), self.p.w1.clone(), self.p.w2.clone());
        let layer = |bound: i32| Mat { wt: false, wshift: 8, shift: 8, bound };

        // a model over fewer bytes than the context reads only the last of them
        let width = ctx.dims()[1];
        if fused::on_gpu(&ctx) && fused::attention_fits(t, e) {
            let (idx, x, q, k, v, a, zf) = fused::attn_forward(&ctx, &self.p.emb, &self.p.pos, [&wq, &wk, &wv], &self.lut, t, LOGIT_SPAN);
            let three = |m: Tensor<2, Int>| m.reshape([b, t, e]);
            let parts = Attended { idx, x: three(x), q: three(q), k: three(k), v: three(v), a: a.reshape([b, t, t]), zf };
            crate::ops::prof::lap(&self.device, "f: attention");
            return self.head(parts, guess, weights, (wq, wk, wv, w1, w2));
        }
        let ctx = if width > t { ctx.narrow(1, width - t, t) } else { ctx };
        let idx = ctx.reshape([b * t]);
        // |x| <= 128 + 128 = 2^8
        let x = if fused::on_gpu(&idx) {
            fused::embed(&self.p.emb, &self.p.pos, &idx).reshape([b, t, e])
        } else {
            shr_floor(self.p.emb.clone(), 8).select(0, idx.clone()).reshape([b, t, e]) + shr_floor(self.p.pos.clone(), 8).unsqueeze::<3>()
        };
        let xf = x.clone().reshape([b * t, e]);
        // |x @ w| <= 64 * 2^8 * 2^10 = 2^24
        let proj = |w: &Tensor<2, Int>| fused::matmul(xf.clone(), w.clone(), layer(ACT_MAX)).reshape([b, t, e]);
        crate::ops::prof::lap(&self.device, "f: weights, embed");
        let (q, k, v) = (proj(&wq), proj(&wk), proj(&wv));
        crate::ops::prof::lap(&self.device, "f: q k v");
        // |q @ k^T| <= 64 * 2^10 * 2^10 = 2^26
        let a = fused::attn_probs(q.clone(), k.clone(), &self.lut);
        // |a @ v| <= 64 * 2^12 * 2^10 = 2^28
        let y = fused::small_matmul(a.clone(), v.clone(), false, 12, UNBOUNDED);
        let zf = (x.clone() + y).clamp(-ACT_MAX, ACT_MAX).reshape([b, t * e]);
        crate::ops::prof::lap(&self.device, "f: attention");
        self.head(Attended { idx, x, q, k, v, a, zf }, guess, weights, (wq, wk, wv, w1, w2))
    }

    /// The pass from the attention block's output on: the hidden layer and the head.
    #[allow(clippy::type_complexity)]
    fn head(
        &self,
        Attended { idx, x, q, k, v, a, zf }: Attended,
        guess: Tensor<2, Int>,
        weights: bool,
        (wq, wk, wv, w1, w2): (Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>),
    ) -> Cache {
        let b = self.cfg.batch;
        let layer = |bound: i32| Mat { wt: false, wshift: 8, shift: 8, bound };
        // |zf @ w1| <= 256 * 1023 * 1023 < 2^28
        let z1 = fused::matmul(zf.clone(), w1.clone(), layer(UNBOUNDED));
        let (z1, norm) = if self.cfg.norm {
            // |cut| <= 2^17 and gain <= 2^12
            let cut = z1.clone().clamp(-NORM_IN, NORM_IN);
            (shr_floor(cut * self.norm.gain.clone(), 8), Some((z1, self.norm.gain.clone())))
        } else {
            (z1, None)
        };
        let h = if leaky() {
            // leaky ReLU, slope 1/8 below zero, so a unit that is off still gets a gradient
            z1.clone().clamp(0, ACT_MAX) + shr_floor(z1.clamp(-8 * ACT_MAX, 0), 3)
        } else {
            z1.clamp(0, ACT_MAX)
        };
        // |h @ w2| <= 256 * 1023 * 1023 < 2^28
        let scores = fused::matmul(h.clone(), w2.clone(), layer(UNBOUNDED));
        let mut out = scores.clone();
        crate::ops::prof::lap(&self.device, "f: hidden, scores");
        let guess_raw = guess;
        if self.cfg.matcher && self.tree.is_none() {
            let guess = split_guess(&guess_raw);
            // raise the proposed byte's logit by the gain learned for this trust level
            let gain = shr_floor(self.p.copy.clone(), 6).select(0, guess.1.clone());
            out = out + self.eye.clone().select(0, guess.0.clone().reshape([b])) * gain;
        }
        let (e_out, prob) = match &self.tree {
            None if !weights => (scores.clone(), scores.clone()),
            None => self.softmax(out),
            Some(tree) if !weights && fused::on_gpu(&out) => {
                let p = fused::tree_probs(out, &tree.sigmoid, LOGIT_SPAN);
                (p.clone(), p)
            }
            Some(tree) if fused::on_gpu(&out) => {
                let p = fused::tree_probs(out, &tree.sigmoid, LOGIT_SPAN);
                let e = fused::tree_leaves(&p, &tree.leaf, &guess_raw, &self.p.copy, &self.lut, self.cfg.matcher, LOGIT_SPAN);
                (e, p)
            }
            Some(tree) => {
                let p = tree.probs(out);
                let mut e = tree.leaves(&p);
                if self.cfg.matcher {
                    let guess = split_guess(&guess_raw);
                    // The tree has no per-byte logit to raise, so the boost scales every other
                    // byte's weight down by exp(-gain) instead; the coder normalises by the row
                    // total either way. Both factors are cut to 10 bits so the product fits.
                    let gain = shr_floor(self.p.copy.clone(), 6).select(0, guess.1.clone()).clamp(0, LOGIT_SPAN);
                    let keep = self.lut.clone().select(0, gain.reshape([b])).reshape([b, 1]).bitwise_right_shift_scalar(10);
                    let proposed = self.eye.clone().select(0, guess.0.clone().reshape([b]));
                    e = e.bitwise_right_shift_scalar(10) * (proposed.clone() * 1024 + (proposed * -1 + 1) * keep);
                }
                (e, p)
            }
        };
        crate::ops::prof::lap(&self.device, "f: head");
        Cache { idx, x, q, k, v, a, zf, h, norm, out: scores, e: e_out, prob, wq, wk, wv, w1, w2, guess_raw }
    }

    /// Gradients of the summed code length, on a per-128-rows scale, in `Params` layout.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn grads(&self, c: &Cache, target: Tensor<1, Int>) -> Params {
        let (g, dcopy) = self.head_error(c, target);
        self.grads_from(c, g, dcopy, None)
    }

    /// The error at the output scores and the copy gain's gradient.
    fn head_error(&self, c: &Cache, target: Tensor<1, Int>) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let b = self.cfg.batch;

        // head: dL/dlogits in Q12, |g| <= 2^12. Softmax: p - onehot. Tree: sigma - bit at the 8
        // nodes on the byte's path, zero elsewhere.
        let gpu = fused::on_gpu(&target);
        // Experiment (GMINT_THROUGH=1): train a tree head with the match boost on through the
        // boost, as a mix does: the error is taken at the coded distribution q (the boosted byte
        // weights), q - onehot per byte, and carried back onto the tree's nodes; the boost's gain
        // trains on the same error at the proposed byte.
        if let (Some(tree), true, true) = (&self.tree, self.cfg.matcher && gpu, through()) {
            let bytes = fused::mix_error(&c.e, target);
            let g = fused::tree_back(bytes.clone(), &c.prob, &tree.byte_at);
            let dcopy = fused::guess_sums(&bytes, &c.guess_raw, (b / CHUNK).trailing_zeros());
            return (g, dcopy);
        }
        let g = match &self.tree {
            None if gpu => fused::onehot_error(c.prob.clone(), target.clone(), 4096),
            Some(tree) if gpu => fused::tree_error(&c.prob, &target, &tree.leaf),
            None => c.prob.clone() - self.eye.clone().select(0, target.clone()) * 4096,
            Some(tree) => {
                tree.on_path.clone().select(0, target.clone()) * (c.prob.clone() - tree.bit_q12.clone().select(0, target.clone()))
            }
        };
        // d/dgain = the proposed byte's error, summed over the rows at each trust level:
        // at most B * 2^12 <= 2^27, then averaged down to the per-128-rows scale
        let dcopy = if self.cfg.matcher && gpu {
            // softmax: the proposed byte's probability; tree: its share of the boosted weights
            let (weights, normalise) = if self.tree.is_none() { (&c.prob, false) } else { (&c.e, true) };
            fused::copy_error(weights, &target, &c.guess_raw, normalise, (b / CHUNK).trailing_zeros())
        } else if self.cfg.matcher {
            let guess = split_guess(&c.guess_raw);
            // For a tree head the boost acts on the byte weights after the tree, so its error is
            // taken there: the boosted distribution's probability of the proposed byte, minus
            // whether that byte was right. The tree itself still trains on its own distribution.
            let at_guess = match &self.tree {
                None => g.clone().gather(1, guess.0.clone()),
                Some(_) => {
                    let p = c.e.clone() / c.e.clone().sum_dim(1).bitwise_right_shift_scalar(12).clamp(1, i32::MAX);
                    let hit = self.eye.clone().select(0, target.clone()).gather(1, guess.0.clone());
                    p.gather(1, guess.0.clone()) - hit * 4096
                }
            };
            shr_round0(self.eye3.clone().select(0, guess.1.clone()).transpose().matmul(at_guess), (b / CHUNK).trailing_zeros())
        } else {
            self.zero3.clone()
        };
        crate::ops::prof::lap(&self.device, "b: head error, copy");
        (g, dcopy)
    }

    /// The backward pass from the error at the output scores, `|g| <= 2^12`. With `step`, which
    /// needs the GPU and an [`Optimizer::sgd`] rule, every group but the copy gain comes back
    /// already stepped: the new weights in place of the gradients.
    fn grads_from(&self, c: &Cache, g: Tensor<2, Int>, dcopy: Tensor<2, Int>, step: Option<&Tensor<1, Int>>) -> Params {
        let Config { ctx: t, emb: e, batch: b, .. } = self.cfg;
        // `slot` is the group's place in the optimizer, as in [`Self::apply`]
        let group = |x: Tensor<2, Int>, y: Tensor<2, Int>, w: &Tensor<2, Int>, slot: usize| match step {
            Some(steps) => fused::xty_sgd(x, y, w, CHUNK, (steps, slot), SHADOW_MAX),
            None => xty(x, y),
        };
        // On the GPU a step writes its weights in place, so every group's weights are read for
        // the error they pass back before that group is stepped.
        // |g @ w2^T| <= 256 * 4096 * 1023 < 2^30
        let back = |bound: i32| Mat { wt: true, wshift: 8, shift: 8, bound };
        let dh = fused::matmul(g.clone(), c.w2.clone(), back(UNBOUNDED));
        let dh = if leaky() && self.cfg.mask {
            // slope 1 inside (0, ACT_MAX), 1/8 at or below zero, 0 where either clamp held
            let up = c.h.clone().clamp(0, 1);
            let inside = up.clone() * (c.h.clone() * -1 + ACT_MAX).clamp(0, 1);
            let below = (up * -1 + 1) * (c.h.clone() + ACT_MAX).clamp(0, 1);
            shr_floor(dh.clamp(-ERR_MAX, ERR_MAX) * (inside * 8 + below), 3)
        } else if leaky() {
            // slope 1 above zero (saturation passes straight through), 1/8 at or below it
            shr_floor(dh.clamp(-ERR_MAX, ERR_MAX) * (c.h.clone().clamp(0, 1) * 7 + 1), 3)
        } else if fused::on_gpu(&dh) {
            fused::relu_error(dh, &c.h, ACT_MAX, ERR_MAX)
        } else {
            // 1 where 0 < h < ACT_MAX, by clamps: a compare costs several times as much
            let live = c.h.clone().clamp(0, 1) * (c.h.clone() * -1 + ACT_MAX).clamp(0, 1);
            (dh * live).clamp(-ERR_MAX, ERR_MAX)
        };
        // through the normalisation, whose gain was a constant of the pass:
        // |dh * gain| <= 2^12 * 2^12
        let dh = match &c.norm {
            Some((_, gain)) => shr_floor(dh * gain.clone(), 8).clamp(-ERR_MAX, ERR_MAX),
            None => dh,
        };
        crate::ops::prof::lap(&self.device, "b: dh");
        // chunk sums: 128 * 2^10 * 2^12 = 2^29
        let dw2 = group(c.h.clone(), g, &self.p.w2, 6);
        crate::ops::prof::lap(&self.device, "b: dw2");
        // With the mask, each error below stops where its value's forward clamp to +-ACT_MAX
        // held: the clamp's own derivative, so no weight is pushed on past where it changes
        // anything. Without it, the clamps pass errors straight through.
        let mask = self.cfg.mask;
        // |dh @ w1^T| <= 256 * 4095 * 1023 < 2^30; the residual zf was clamped
        let dz = fused::matmul_gated(dh.clone(), c.w1.clone(), back(ERR_MAX), mask.then_some((&c.zf, ACT_MAX))).reshape([b, t, e]);
        // chunk sums: 128 * 2^10 * 2^12 = 2^29
        let dw1 = group(c.zf.clone(), dh, &self.p.w1, 5);
        crate::ops::prof::lap(&self.device, "b: dw1");

        crate::ops::prof::lap(&self.device, "b: dz");
        // attention: y = a @ v
        // |a^T @ dz| <= ctx * 4096 * 4095 < 2^30 for ctx <= 64
        let dv = fused::small_matmul_gated(c.a.clone(), dz.clone(), true, 12, ERR_MAX, mask.then_some((&c.v, ACT_MAX)));
        let ds = fused::attn_ds(dz.clone(), c.v.clone(), c.a.clone(), ERR_MAX);
        // |ds @ k| and |ds^T @ q| <= 64 * 2^12 * 2^10 = 2^28
        let dq = fused::small_matmul_gated(ds.clone(), c.k.clone(), false, 8, ERR_MAX, mask.then_some((&c.q, ACT_MAX)));
        let dk = fused::small_matmul_gated(ds, c.q.clone(), true, 8, ERR_MAX, mask.then_some((&c.k, ACT_MAX)));
        crate::ops::prof::lap(&self.device, "b: attention");
        let flat = |x: &Tensor<3, Int>| x.clone().reshape([b * t, e]);
        let xf = flat(&c.x);
        let (dqf, dkf, dvf) = (flat(&dq), flat(&dk), flat(&dv));
        // each term <= 64 * 2^12 * 2^10 = 2^28; the three are shifted before they are added
        let through = |d: Tensor<2, Int>, w: &Tensor<2, Int>| fused::matmul(d, w.clone(), back(UNBOUNDED));
        let dx = (flat(&dz) + (through(dqf.clone(), &c.wq) + through(dkf.clone(), &c.wk) + through(dvf.clone(), &c.wv)).clamp(-ERR_MAX, ERR_MAX))
            .clamp(-ERR_MAX, ERR_MAX);
        // chunk sums: 128 * 2^8 * 2^12 = 2^27
        let (dwq, dwk, dwv) = match step {
            Some(steps) if fused::qkv_fits(e) => {
                let [q, k, v] = fused::qkv_sgd(&xf, [&dqf, &dkf, &dvf], [&self.p.wq, &self.p.wk, &self.p.wv], CHUNK, steps, 2, SHADOW_MAX);
                (q, k, v)
            }
            _ => (group(xf.clone(), dqf, &self.p.wq, 2), group(xf.clone(), dkf, &self.p.wk, 3), group(xf, dvf, &self.p.wv, 4)),
        };
        crate::ops::prof::lap(&self.device, "b: dwq dwk dwv");

        // Row sums: B * ctx rows of magnitude 2^12. Up to 2^18 rows fit in 2^30 as they are; past
        // that the rows are pre-shifted by the excess. The rest of the averaging xty does (down to
        // a per-128-rows scale) happens after the sum, where it costs no precision.
        crate::ops::prof::lap(&self.device, "b: dx");
        let rows_log2 = (b * t / CHUNK).trailing_zeros();
        let pre = (b * t).trailing_zeros().saturating_sub(18);
        let dxs = shr_round0(dx, pre);
        let (demb, dpos) = match step {
            Some(steps) if fused::embed_pos_fits(t, e) => fused::embed_pos_sgd(&c.idx, &dxs, &self.p.emb, &self.p.pos, rows_log2 - pre, steps, [0, 1], SHADOW_EMB_MAX),
            Some(steps) if 256 % e == 0 => (
                fused::embed_sgd(&c.idx, &dxs, &self.p.emb, t, false, rows_log2 - pre, (steps, 0), SHADOW_EMB_MAX),
                fused::embed_sgd(&c.idx, &dxs, &self.p.pos, t, true, rows_log2 - pre, (steps, 1), SHADOW_EMB_MAX),
            ),
            step => {
                let demb = fused::rows_by_value(c.idx.clone(), dxs.clone(), VOCAB, rows_log2 - pre);
                let dpos = if fused::on_gpu(&dxs) {
                    fused::place_sums(&dxs, t, rows_log2 - pre)
                } else {
                    shr_round0(dxs.reshape([b, t, e]).sum_dim(0).reshape([t, e]), rows_log2 - pre)
                };
                match step {
                    Some(steps) => (fused::sgd_step(&self.p.emb, demb, (steps, 0), SHADOW_EMB_MAX), fused::sgd_step(&self.p.pos, dpos, (steps, 1), SHADOW_EMB_MAX)),
                    None => (demb, dpos),
                }
            }
        };

        crate::ops::prof::lap(&self.device, "b: demb dpos");
        Params { emb: demb, pos: dpos, wq: dwq, wk: dwk, wv: dwv, w1: dw1, w2: dw2, copy: dcopy }
    }

    pub fn apply(&mut self, g: Params) {
        let o = &mut self.opt;
        o.tick();
        self.p = Params {
            emb: o.step(0, &self.p.emb, g.emb, SGD_EMB, SHADOW_EMB_MAX),
            pos: o.step(1, &self.p.pos, g.pos, SGD_EMB, SHADOW_EMB_MAX),
            wq: o.step(2, &self.p.wq, g.wq, SGD_QUERY_KEY, SHADOW_MAX),
            wk: o.step(3, &self.p.wk, g.wk, SGD_QUERY_KEY, SHADOW_MAX),
            wv: o.step(4, &self.p.wv, g.wv, SGD_VALUE, SHADOW_MAX),
            w1: o.step(5, &self.p.w1, g.w1, 0, SHADOW_MAX),
            w2: o.step(6, &self.p.w2, g.w2, 0, SHADOW_MAX),
            // without the match finder the gain's gradient is zero and so is the gain
            copy: if self.cfg.matcher { o.step(7, &self.p.copy, g.copy, SGD_COPY, SHADOW_MAX) } else { self.p.copy.clone() },
        };
        crate::ops::prof::lap(&self.device, "b: optimiser");
    }

    pub fn update(&mut self, c: &Cache, target: Tensor<1, Int>) {
        let (g, dcopy) = self.head_error(c, target);
        self.train(c, g, dcopy);
    }
}

impl Model {
    /// How much of the model can still learn: per weight group the share of weights pinned at
    /// their clamp, and on the last batch the share of hidden units that were off or saturated
    /// (both pass no gradient). Reads tensors back, so it is for diagnostics, not the coding loop.
    pub fn health(&self) -> String {
        let pinned = |t: &Tensor<2, Int>, max: i32| -> f64 {
            let v: Vec<i32> = t.clone().into_data().convert::<i32>().try_to_vec().unwrap();
            v.iter().filter(|x| x.abs() >= max).count() as f64 * 100.0 / v.len() as f64
        };
        let p = &self.p;
        let (floor, blowups) = self.opt.floor();
        let mut out = format!(
            "rate {} (floor {floor}, {blowups} windows over raw); pinned: emb {:.1}% pos {:.1}% wq {:.1}% wk {:.1}% wv {:.1}% w1 {:.1}% w2 {:.1}%",
            self.opt.rate(),
            pinned(&p.emb, SHADOW_EMB_MAX),
            pinned(&p.pos, SHADOW_EMB_MAX),
            pinned(&p.wq, SHADOW_MAX),
            pinned(&p.wk, SHADOW_MAX),
            pinned(&p.wv, SHADOW_MAX),
            pinned(&p.w1, SHADOW_MAX),
            pinned(&p.w2, SHADOW_MAX)
        );
        if let Some(h) = &self.last_hidden {
            let v: Vec<i32> = h.clone().into_data().convert::<i32>().try_to_vec().unwrap();
            let share = |f: &dyn Fn(i32) -> bool| v.iter().filter(|x| f(**x)).count() as f64 * 100.0 / v.len() as f64;
            let width = self.cfg.hid;
            // a unit is dead when it is off for every row of the batch
            let dead = (0..width).filter(|u| v.iter().skip(*u).step_by(width).all(|x| *x == 0)).count();
            out += &format!(
                "; hidden: {:.1}% off, {:.1}% saturated, {dead} of {width} units off for the whole batch",
                share(&|x| x == 0),
                share(&|x| x >= ACT_MAX)
            );
        }
        if self.cfg.norm {
            let mut g: Vec<i32> = self.norm.gain.clone().into_data().convert::<i32>().try_to_vec().unwrap();
            g.sort_unstable();
            let x = |v: i32| v as f64 / 256.0;
            out += &format!("; gain min {:.2} median {:.2} max {:.2}", x(g[0]), x(g[g.len() / 2]), x(g[g.len() - 1]));
        }
        out
    }

    /// The learning rate currently in use (it moves under [`Rule::Auto`]).
    pub fn rate(&self) -> i32 {
        self.opt.rate()
    }

    fn shapes(cfg: &Config) -> [[usize; 2]; 8] {
        let (t, e, h) = (cfg.ctx, cfg.emb, cfg.hid);
        [[VOCAB, e], [t, e], [e, e], [e, e], [e, e], [t * e, h], [h, cfg.head.outputs()], [3, 1]]
    }

    /// All trainable weights (Q16 shadows) in a fixed order, then the optimizer's state, for
    /// saving as a prior. Without the optimizer state a model started from a prior would take
    /// first-step-sized updates and overwrite what it was given.
    pub fn export(&self) -> Vec<i32> {
        let p = &self.p;
        let mut out: Vec<i32> = [&p.emb, &p.pos, &p.wq, &p.wk, &p.wv, &p.w1, &p.w2, &p.copy]
            .into_iter()
            .flat_map(|t| t.clone().into_data().convert::<i32>().try_to_vec::<i32>().unwrap())
            .collect();
        if self.cfg.norm {
            out.extend(self.norm.gain_acc.clone().into_data().convert::<i32>().try_to_vec::<i32>().unwrap());
        }
        out.extend(self.opt.state());
        out
    }

    /// Replaces the weights with ones from [`Self::export`] on a model of the same shape.
    pub fn import(&mut self, values: &[i32]) -> Result<(), String> {
        let shapes = Self::shapes(&self.cfg);
        let weights = shapes.iter().map(|s| s[0] * s[1]).sum::<usize>();
        let hid = self.cfg.hid;
        let stats = if self.cfg.norm { hid } else { 0 };
        if values.len() != weights + stats + Optimizer::state_len(8) {
            return Err("prior does not match the model shape".into());
        }
        let mut at = 0;
        let mut next = |shape: [usize; 2]| -> Tensor<2, Int> {
            let n = shape[0] * shape[1];
            at += n;
            Tensor::from_data(TensorData::new(values[at - n..at].to_vec(), shape), &self.device)
        };
        let [emb, pos, wq, wk, wv, w1, w2, copy] = shapes.map(&mut next);
        self.p = Params { emb, pos, wq, wk, wv, w1, w2, copy };
        if self.cfg.norm {
            let gain_acc = next([1, hid]);
            self.norm = Norm { gain: shr_floor(gain_acc.clone(), NORM_EMA), gain_acc };
        }
        self.opt.restore(&values[weights + stats..], &self.device);
        Ok(())
    }
}

impl Model {
    /// The head's own loss on the pass just made, before the update: `-sum ln p` of the true
    /// bytes under the head's distribution, in Q8 (256 ln) units, `[1]`. It is what the gradient
    /// descends. The coder's code length differs from it by the match boost (applied after the
    /// head; its gain trains on its own error) and by the coder's quantised frequencies (each
    /// at least 1). Needs a preceding [`Predictor::predict_coded`]; read it before `learn`.
    pub fn pending_loss(&self, target: &Tensor<1, Int>) -> Tensor<1, Int> {
        let c = self.pending.as_ref().expect("pending_loss needs a forward pass");
        let b = self.cfg.batch;
        let picked = |t: Tensor<2, Int>| t.gather(1, target.clone().reshape([b, 1])).reshape([b]);
        // per row at least -8 * 4095, so a batch of up to 2^15 rows sums inside i32
        let log = match &self.tree {
            // the sum of the branch logs down the byte's path
            Some(tree) => picked(tree.log_scores(&c.prob)),
            None => self.logp.clone().select(0, picked(c.prob.clone()).clamp(0, 4096)),
        };
        -log.sum()
    }

    /// See [`Predictor::graphable`]: GPU kernels for the whole step under a plain SGD rule, and
    /// no running normalisation statistics (they are replaced, not updated in place).
    pub fn graphable(&self) -> bool {
        fused::on_gpu(&self.p.emb) && self.opt.sgd(0).is_some() && !self.cfg.norm && fused::attention_fits(self.cfg.ctx, self.cfg.emb)
    }

    pub fn refresh_steps(&mut self) {
        let _ = self.opt.device_steps(&self.device);
    }
}

impl Predictor for Model {
    fn predict(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Vec<i32> {
        let cache = self.forward(ctx, guess);
        let freqs = cache.freqs();
        self.pending = Some(cache);
        freqs
    }

    fn predict_coded(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>, target: Tensor<1, Int>) -> Tensor<2, Int> {
        let cache = self.forward(ctx, guess);
        let out = coded(cache.e.clone(), target, &self.below);
        self.pending = Some(cache);
        out
    }

    fn learn(&mut self, target: Tensor<1, Int>) {
        let cache = self.pending.take().expect("learn without a preceding predict");
        self.update(&cache, target);
    }

    fn observe(&mut self, cost_q8: u64) {
        self.opt.observe(cost_q8, 8 * 256 * self.cfg.batch as u64);
    }

    fn health(&self) -> String {
        Model::health(self)
    }

    fn graphable(&self) -> bool {
        Model::graphable(self)
    }

    fn refresh_steps(&mut self) {
        Model::refresh_steps(self)
    }
}

/// A match proposal `[B, 2]` as its byte `[B, 1]` and trust level `[B]`, for the tensor code
/// paths; the kernels read the pair as it comes.
pub fn split_guess(guess: &Tensor<2, Int>) -> (Tensor<2, Int>, Tensor<1, Int>) {
    let b = guess.dims()[0];
    (guess.clone().narrow(1, 0, 1), guess.clone().narrow(1, 1, 1).reshape([b]))
}

/// [`shr_round`] that tolerates a zero shift.
pub fn shr_round0<const D: usize>(t: Tensor<D, Int>, s: u32) -> Tensor<D, Int> {
    if s == 0 { t } else { shr_round(t, s) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::activation::log_softmax;

    const CFG: Config =
        Config { arch: Arch::Attn, ctx: 8, emb: 32, hid: 256, batch: 128, lr: 12, rule: Rule::Sgd, head: Head::Softmax, matcher: false, comps: 4, replicas: 4, short: 0, adapt: 0, warm: 0, norm: false, mask: true, seed: 7 };

    fn ints(t: &Tensor<2, Int>) -> Vec<i32> {
        t.clone().into_data().convert::<i32>().try_to_vec().unwrap()
    }

    fn no_guess(device: &Device) -> Tensor<2, Int> {
        Tensor::zeros([CFG.batch, 2], device)
    }

    fn batch(rng: &mut Rng, device: &Device) -> (Tensor<2, Int>, Tensor<1, Int>) {
        // skewed bytes, so the batch has structure to learn and rows repeat in the embedding
        let byte = |rng: &mut Rng| ((rng.next() >> 33) % 23 + 97) as i32;
        let ctx: Vec<i32> = (0..CFG.batch * CFG.ctx).map(|_| byte(rng)).collect();
        let tgt: Vec<i32> = ctx.chunks(CFG.ctx).map(|c| c[CFG.ctx - 1]).collect();
        (
            Tensor::from_data(TensorData::new(ctx, [CFG.batch, CFG.ctx]), device),
            Tensor::from_data(TensorData::new(tgt, [CFG.batch]), device),
        )
    }

    /// The handwritten integer backward pass against burn's float autodiff on the same network.
    /// Rounding and the pass-through clamps keep them from matching exactly, so the test asks
    /// that every parameter's gradient points the same way.
    #[test]
    fn integer_gradients_follow_autodiff() {
        gradients_follow_autodiff(Head::Softmax);
    }

    #[test]
    fn tree_gradients_follow_autodiff() {
        gradients_follow_autodiff(Head::Tree);
        gradients_follow_autodiff(Head::RandomTree);
    }

    fn gradients_follow_autodiff(head: Head) {
        let device = test_device();
        let mut rng = Rng(21);
        let mut model = Model::new(Config { head, ..CFG }, &device);
        // a few steps away from init, so attention is not uniform and the head is not symmetric
        for _ in 0..30 {
            let (ctx, tgt) = batch(&mut rng, &device);
            let c = model.forward(ctx, no_guess(&device));
            model.update(&c, tgt);
        }
        let (ctx, tgt) = batch(&mut rng, &device);
        let cache = model.forward(ctx.clone(), no_guess(&device));
        let got = model.grads(&cache, tgt.clone());

        // float twin on the forward (Q8) weights
        let ad = device.clone().autodiff();
        let (b, t, e) = (CFG.batch, CFG.ctx, CFG.emb);
        let leaf = |s: &Tensor<2, Int>| -> Tensor<2> {
            let v: Vec<f32> = ints(&shr_floor(s.clone(), 8)).iter().map(|x| *x as f32).collect();
            Tensor::<2>::from_data(TensorData::new(v, s.dims()), &ad).require_grad()
        };
        let p = &model.p;
        let (emb, pos, wq, wk, wv, w1, w2) =
            (leaf(&p.emb), leaf(&p.pos), leaf(&p.wq), leaf(&p.wk), leaf(&p.wv), leaf(&p.w1), leaf(&p.w2));
        let ctx_ad = Tensor::<2, Int>::from_data(ctx.into_data(), &ad);
        let tgt_ad = Tensor::<1, Int>::from_data(tgt.into_data(), &ad);
        let x = emb.clone().select(0, ctx_ad.reshape([b * t])).reshape([b, t, e]) + pos.clone().unsqueeze::<3>();
        let xf = x.clone().reshape([b * t, e]);
        let proj = |w: &Tensor<2>| (xf.clone().matmul(w.clone()) / 256.0).reshape([b, t, e]);
        let (q, k, v) = (proj(&wq), proj(&wk), proj(&wv));
        let a = burn::tensor::activation::softmax(q.matmul(k.transpose()) / 4096.0 / 256.0, 2);
        let z = (x + a.matmul(v)).reshape([b, t * e]);
        let h = (z.matmul(w1.clone()) / 256.0).clamp(0.0, 1023.0);
        let out = h.matmul(w2.clone()) / 256.0 / 256.0;
        let loss = match &model.tree {
            None => -log_softmax(out, 1).gather(1, tgt_ad.reshape([b, 1])).sum(),
            Some(tree) => {
                // binary cross-entropy at the nodes on each target's path
                let table = |t: &Tensor<2, Int>, scale: f32| -> Tensor<2> {
                    let v: Vec<f32> = ints(t).iter().map(|x| *x as f32 / scale).collect();
                    Tensor::<2>::from_data(TensorData::new(v, t.dims()), &ad).select(0, tgt_ad.clone())
                };
                let (on_path, bit) = (table(&tree.on_path, 1.0), table(&tree.bit_q12, 4096.0));
                let p = burn::tensor::activation::sigmoid(out);
                let ll = bit.clone() * p.clone().log() + (bit.neg() + 1.0) * (p.neg() + 1.0).log();
                -(on_path * ll).sum()
            }
        };
        let grads = loss.backward();

        let names = ["emb", "pos", "wq", "wk", "wv", "w1", "w2"];
        // `each` ends with the copy gain, which this test leaves switched off
        for ((name, want), got) in names.iter().zip([emb, pos, wq, wk, wv, w1, w2]).zip(got.each()) {
            let want: Vec<f32> = want.grad(&grads).unwrap().into_data().convert::<f32>().try_to_vec().unwrap();
            let got: Vec<f64> = ints(got).iter().map(|x| *x as f64).collect();
            let dot: f64 = want.iter().zip(&got).map(|(a, b)| *a as f64 * b).sum();
            let (na, nb) = (
                want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt(),
                got.iter().map(|b| b * b).sum::<f64>().sqrt(),
            );
            let cos = dot / (na * nb);
            println!("{head:?} {name}: cosine {cos:.3}, |int| {nb:.0}, |float| {na:.4}");
            assert!(cos > 0.7, "{head:?} {name}: integer gradient disagrees with autodiff (cosine {cos:.3})");
        }
    }
}
