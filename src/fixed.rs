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

use burn::tensor::ops::IndexingUpdateOp;
use burn::tensor::{Device, Int, Tensor, TensorData};

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
const LUT_ONE: i64 = 1 << 20;
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
    shr_floor(t + (1i32 << (s - 1)), s)
}

/// `x^T @ y` summed over rows in chunks of [`CHUNK`], then averaged over the chunks, so the
/// result is on a per-128-rows scale whatever the row count. Each chunk sum must fit in 2^30.
fn xty(x: Tensor<2, Int>, y: Tensor<2, Int>) -> Tensor<2, Int> {
    let ([n, a], [_, b]) = (x.dims(), y.dims());
    let chunks = n / CHUNK;
    debug_assert!(n % CHUNK == 0 && chunks.is_power_of_two());
    let per = x.reshape([chunks, CHUNK, a]).transpose().matmul(y.reshape([chunks, CHUNK, b]));
    shr_floor(per, chunks.trailing_zeros()).sum_dim(0).reshape([a, b])
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
    /// Told the Q8 code length of every step, trained on or not, so a model can steer itself.
    fn observe(&mut self, _cost_q8: u64) {}
}

/// Which predictor codes the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arch {
    /// One attention layer and a two-layer head ([`Model`]).
    Attn,
    /// A random sum-product network ([`crate::spn::Spn`]). `emb` is the number of distributions
    /// per region and sum nodes per layer, `hid` the number of random replicas.
    Spn,
}

impl Arch {
    pub fn id(self) -> u8 {
        self as u8
    }
    pub fn from_id(id: u8) -> Option<Self> {
        [Self::Attn, Self::Spn].into_iter().find(|a| a.id() == id)
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "attn" => Some(Self::Attn),
            "spn" => Some(Self::Spn),
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
        // sigma(-x) = e^-x / (1 + e^-x), from the same integer exp table as the softmax
        let sigmoid: Vec<i32> = exp_lut().iter().map(|e| ((*e as i64 * 4096) / (LUT_ONE + *e as i64)) as i32).collect();
        Self {
            node,
            sign,
            offset,
            on_path: table(on_path),
            bit_q12: table(bit_q12),
            sigmoid: Tensor::from_data(TensorData::new(sigmoid, [LOGIT_SPAN as usize + 1]), device),
        }
    }

    /// Q8 node logits `[B, 255]` to Q12 probabilities of each node's 1-branch, kept off 0 and 1.
    fn probs(&self, z: Tensor<2, Int>) -> Tensor<2, Int> {
        let [b, n] = z.dims();
        let low = self.sigmoid.clone().select(0, z.clone().abs().clamp(0, LOGIT_SPAN).reshape([b * n])).reshape([b, n]);
        // low is sigma(-|z|); for z >= 0 the answer is its complement
        (low.clone() + z.greater_equal_elem(0).int() * (low * -2 + 4096)).clamp(1, 4095)
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
    /// Feed the head a long-range match proposal (softmax head only).
    pub matcher: bool,
    pub seed: u64,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.arch == Arch::Spn {
            // the batch bound keeps the replica-mix gradient (256 * 2^12 * B) inside i32
            let ok = self.ctx == crate::spn::CTX
                && (2..=8).contains(&self.emb)
                && (1..=crate::spn::MAX_FAN_IN).contains(&self.hid)
                && self.batch.is_power_of_two()
                && (CHUNK..=512).contains(&self.batch)
                && (6..=22).contains(&self.lr)
                && self.head == Head::Softmax
                && !self.matcher;
            return if ok { Ok(()) } else { Err(format!("unsupported sum-product network shape: {self:?}")) };
        }
        let ok = self.ctx.is_power_of_two()
            && self.ctx <= 64
            && self.batch.is_power_of_two()
            && (CHUNK..=1 << 15).contains(&self.batch)
            // head products are bounded by ctx*emb and hid inputs of magnitude 2^10 each
            && self.ctx * self.emb <= 256
            && self.emb <= 64
            && (1..=256).contains(&self.hid)
            && (6..=22).contains(&self.lr)
            && !(self.matcher && self.head != Head::Softmax);
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
    tree: Option<Tree>,
    opt: Optimizer,
    /// The forward pass awaiting its targets.
    pending: Option<Cache>,
    device: Device,
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
    e: Tensor<2, Int>,
    /// Q12: byte probabilities for the softmax head, node probabilities for a tree head.
    prob: Tensor<2, Int>,
    wq: Tensor<2, Int>,
    wk: Tensor<2, Int>,
    wv: Tensor<2, Int>,
    w1: Tensor<2, Int>,
    w2: Tensor<2, Int>,
    /// The match proposal: byte `[B, 1]` and trust level `[B]`.
    guess: (Tensor<2, Int>, Tensor<1, Int>),
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
        Self { cfg, p, lut, tree, opt: Optimizer::new(cfg.rule, cfg.lr, 8), pending: None, device: device.clone() }
    }

    /// Integer softmax over the last dim of Q8 logits: returns (Q20 weights, Q12 probabilities).
    fn softmax<const D: usize>(&self, logits: Tensor<D, Int>) -> (Tensor<D, Int>, Tensor<D, Int>) {
        let dims = logits.dims();
        let n: usize = dims.iter().product();
        let d = (logits.clone().max_dim(D - 1) - logits).clamp(0, LOGIT_SPAN);
        let e = self.lut.clone().select(0, d.reshape([n])).reshape(dims);
        // sum(e) is in [2^20, 2^28]; it is non-negative, so the raw shift is safe
        let p = e.clone() / e.clone().sum_dim(D - 1).bitwise_right_shift_scalar(12);
        (e, p)
    }

    /// `ctx` is `[B, ctx]` byte values. Nothing here depends on the bytes being predicted.
    pub fn forward(&self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Cache {
        let Config { ctx: t, emb: e, batch: b, .. } = self.cfg;
        let (wq, wk, wv) = (shr_floor(self.p.wq.clone(), 8), shr_floor(self.p.wk.clone(), 8), shr_floor(self.p.wv.clone(), 8));
        let (w1, w2) = (shr_floor(self.p.w1.clone(), 8), shr_floor(self.p.w2.clone(), 8));

        let idx = ctx.reshape([b * t]);
        // |x| <= 128 + 128 = 2^8
        let x = shr_floor(self.p.emb.clone(), 8).select(0, idx.clone()).reshape([b, t, e])
            + shr_floor(self.p.pos.clone(), 8).unsqueeze::<3>();
        let xf = x.clone().reshape([b * t, e]);
        // |x @ w| <= 64 * 2^8 * 2^10 = 2^24
        let proj = |w: &Tensor<2, Int>| {
            shr_floor(xf.clone().matmul(w.clone()), 8).clamp(-ACT_MAX, ACT_MAX).reshape([b, t, e])
        };
        let (q, k, v) = (proj(&wq), proj(&wk), proj(&wv));
        // |q @ k^T| <= 64 * 2^10 * 2^10 = 2^26
        let (_, a) = self.softmax(shr_floor(q.clone().matmul(k.clone().transpose()), 12));
        // |a @ v| <= 64 * 2^12 * 2^10 = 2^28
        let y = shr_floor(a.clone().matmul(v.clone()), 12);
        let zf = (x.clone() + y).clamp(-ACT_MAX, ACT_MAX).reshape([b, t * e]);
        // |zf @ w1| <= 256 * 1023 * 1023 < 2^28
        let h = shr_floor(zf.clone().matmul(w1.clone()), 8).clamp(0, ACT_MAX);
        // |h @ w2| <= 256 * 1023 * 1023 < 2^28
        let mut out = shr_floor(h.clone().matmul(w2.clone()), 8);
        let guess = (guess.clone().narrow(1, 0, 1), guess.narrow(1, 1, 1).reshape([b]));
        if self.cfg.matcher {
            // raise the proposed byte's logit by the gain learned for this trust level
            let gain = shr_floor(self.p.copy.clone(), 6).select(0, guess.1.clone());
            out = out + guess.0.clone().reshape([b]).one_hot::<2>(VOCAB) * gain;
        }
        let (e_out, prob) = match &self.tree {
            None => self.softmax(out),
            Some(tree) => {
                let p = tree.probs(out);
                (tree.leaves(&p), p)
            }
        };
        Cache { idx, x, q, k, v, a, zf, h, e: e_out, prob, wq, wk, wv, w1, w2, guess }
    }

    /// Gradients of the summed code length, on a per-128-rows scale, in `Params` layout.
    pub fn grads(&self, c: &Cache, target: Tensor<1, Int>) -> Params {
        let Config { ctx: t, emb: e, batch: b, .. } = self.cfg;
        let clamp_err = |x: Tensor<3, Int>| x.clamp(-ERR_MAX, ERR_MAX);

        // head: dL/dlogits in Q12, |g| <= 2^12. Softmax: p - onehot. Tree: sigma - bit at the 8
        // nodes on the byte's path, zero elsewhere.
        let g = match &self.tree {
            None => c.prob.clone() - target.one_hot::<2>(VOCAB) * 4096,
            Some(tree) => {
                tree.on_path.clone().select(0, target.clone()) * (c.prob.clone() - tree.bit_q12.clone().select(0, target))
            }
        };
        // d/dgain = the proposed byte's error, summed over the rows at each trust level:
        // at most B * 2^12 <= 2^27, then averaged down to the per-128-rows scale
        let dcopy = if self.cfg.matcher {
            let at_guess = g.clone().gather(1, c.guess.0.clone());
            shr_round0(c.guess.1.clone().one_hot::<2>(3).transpose().matmul(at_guess), (b / CHUNK).trailing_zeros())
        } else {
            Tensor::zeros([3, 1], &self.device)
        };
        // chunk sums: 128 * 2^10 * 2^12 = 2^29
        let dw2 = xty(c.h.clone(), g.clone());
        // |g @ w2^T| <= 256 * 4096 * 1023 < 2^30
        let dh = shr_floor(g.matmul(c.w2.clone().transpose()), 8);
        let live = c.h.clone().greater_elem(0).int() * c.h.clone().lower_elem(ACT_MAX).int();
        let dh = (dh * live).clamp(-ERR_MAX, ERR_MAX);
        // chunk sums: 128 * 2^10 * 2^12 = 2^29
        let dw1 = xty(c.zf.clone(), dh.clone());
        // |dh @ w1^T| <= 256 * 4095 * 1023 < 2^30
        let dz = shr_floor(dh.matmul(c.w1.clone().transpose()), 8).clamp(-ERR_MAX, ERR_MAX).reshape([b, t, e]);

        // attention: y = a @ v
        // |dz @ v^T| <= 64 * 2^12 * 2^10 = 2^28
        let da = clamp_err(shr_floor(dz.clone().matmul(c.v.clone().transpose()), 10));
        // |a^T @ dz| <= ctx * 4096 * 4095 < 2^30 for ctx <= 64
        let dv = clamp_err(shr_floor(c.a.clone().transpose().matmul(dz.clone()), 12));
        // softmax: ds = a * (da - sum_j a_j da_j); |sum| <= 2^12 * 2^12 after the row sums to 2^12
        let mean = shr_floor((c.a.clone() * da.clone()).sum_dim(2), 12);
        // |a * (da - mean)| <= 2^12 * 2^13 = 2^25
        let ds = clamp_err(shr_floor(c.a.clone() * (da - mean), 12));
        // |ds @ k| and |ds^T @ q| <= 64 * 2^12 * 2^10 = 2^28
        let dq = clamp_err(shr_floor(ds.clone().matmul(c.k.clone()), 8));
        let dk = clamp_err(shr_floor(ds.transpose().matmul(c.q.clone()), 8));

        let flat = |x: &Tensor<3, Int>| x.clone().reshape([b * t, e]);
        let xf = flat(&c.x);
        let (dqf, dkf, dvf) = (flat(&dq), flat(&dk), flat(&dv));
        // chunk sums: 128 * 2^8 * 2^12 = 2^27
        let (dwq, dwk, dwv) = (xty(xf.clone(), dqf.clone()), xty(xf.clone(), dkf.clone()), xty(xf, dvf.clone()));
        // each term <= 64 * 2^12 * 2^10 = 2^28; the three are shifted before they are added
        let back = |d: Tensor<2, Int>, w: &Tensor<2, Int>| shr_floor(d.matmul(w.clone().transpose()), 8);
        let dx = (flat(&dz) + (back(dqf, &c.wq) + back(dkf, &c.wk) + back(dvf, &c.wv)).clamp(-ERR_MAX, ERR_MAX))
            .clamp(-ERR_MAX, ERR_MAX);

        // Row sums: B * ctx rows of magnitude 2^12. Up to 2^18 rows fit in 2^30 as they are; past
        // that the rows are pre-shifted by the excess. The rest of the averaging xty does (down to
        // a per-128-rows scale) happens after the sum, where it costs no precision.
        let rows_log2 = (b * t / CHUNK).trailing_zeros();
        let pre = (b * t).trailing_zeros().saturating_sub(18);
        let dxs = shr_round0(dx, pre);
        let demb = Tensor::<2, Int>::zeros([VOCAB, e], &self.device).select_assign(
            0,
            c.idx.clone(),
            dxs.clone(),
            IndexingUpdateOp::Add,
        );
        let demb = shr_round0(demb, rows_log2 - pre);
        let dpos = shr_round0(dxs.reshape([b, t, e]).sum_dim(0).reshape([t, e]), rows_log2 - pre);

        Params { emb: demb, pos: dpos, wq: dwq, wk: dwk, wv: dwv, w1: dw1, w2: dw2, copy: dcopy }
    }

    pub fn apply(&mut self, g: Params) {
        // log2 of how much smaller each group's gradient scale is than the head's, measured
        // against autodiff in the test below
        const EMB: i32 = 11;
        const VALUE: i32 = 3;
        const QUERY_KEY: i32 = 1;
        // its input is a bare 1 where the head's inputs are Q8 activations
        const COPY: i32 = 6;
        let o = &mut self.opt;
        o.tick();
        self.p = Params {
            emb: o.step(0, &self.p.emb, g.emb, EMB, SHADOW_EMB_MAX),
            pos: o.step(1, &self.p.pos, g.pos, EMB, SHADOW_EMB_MAX),
            wq: o.step(2, &self.p.wq, g.wq, QUERY_KEY, SHADOW_MAX),
            wk: o.step(3, &self.p.wk, g.wk, QUERY_KEY, SHADOW_MAX),
            wv: o.step(4, &self.p.wv, g.wv, VALUE, SHADOW_MAX),
            w1: o.step(5, &self.p.w1, g.w1, 0, SHADOW_MAX),
            w2: o.step(6, &self.p.w2, g.w2, 0, SHADOW_MAX),
            copy: o.step(7, &self.p.copy, g.copy, COPY, SHADOW_MAX),
        };
    }

    pub fn update(&mut self, c: &Cache, target: Tensor<1, Int>) {
        let g = self.grads(c, target);
        self.apply(g);
    }
}

impl Model {
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
        out.extend(self.opt.state());
        out
    }

    /// Replaces the weights with ones from [`Self::export`] on a model of the same shape.
    pub fn import(&mut self, values: &[i32]) -> Result<(), String> {
        let shapes = Self::shapes(&self.cfg);
        let weights = shapes.iter().map(|s| s[0] * s[1]).sum::<usize>();
        if values.len() != weights + Optimizer::state_len(8) {
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
        self.opt.restore(&values[weights..], &self.device);
        Ok(())
    }
}

impl Predictor for Model {
    fn predict(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Vec<i32> {
        let cache = self.forward(ctx, guess);
        let freqs = cache.freqs();
        self.pending = Some(cache);
        freqs
    }

    fn learn(&mut self, target: Tensor<1, Int>) {
        let cache = self.pending.take().expect("learn without a preceding predict");
        self.update(&cache, target);
    }

    fn observe(&mut self, cost_q8: u64) {
        self.opt.observe(cost_q8, 8 * 256 * self.cfg.batch as u64);
    }
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
        Config { arch: Arch::Attn, ctx: 8, emb: 32, hid: 256, batch: 128, lr: 12, rule: Rule::Sgd, head: Head::Softmax, matcher: false, seed: 7 };

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
        let device = Device::flex();
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
