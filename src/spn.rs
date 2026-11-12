//! A random sum-product network (RAT-SPN, Peharz et al. 2018) as a byte predictor, in integer
//! log-domain arithmetic.
//!
//! The circuit scores the joint assignment of the 8 context bytes and the next byte; evaluating it
//! for all 256 candidates and normalising gives the next byte's distribution, the way Liu, Mandt &
//! Van den Broeck (2021) drive an arithmetic coder from a probabilistic circuit. Product nodes
//! are additions of Q8 log values; sum nodes are integer log-sum-exp from the same exp table the
//! softmax uses. The weights are unnormalised, which the final normalisation over the next byte
//! makes harmless.
//!
//! Structure, repeated in `replicas` copies that a final sum node mixes: the context positions
//! are dealt at random into four leaf regions of two, and the next byte joins the last one. Each
//! leaf region has `comps` factorised distributions. Regions 0 and 1 meet in sum layer A, regions
//! 2 and 3 in sum layer B, and A and B meet at the replica's root. Only the path through region 3
//! depends on the candidate byte, so only it carries the 256-wide dimension; the two sum layers
//! on that path are evaluated as nested log-sum-exps to keep that tensor small.

use burn::tensor::ops::IndexingUpdateOp;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::{Config, LOGIT_SPAN, Predictor, Rng, VOCAB, exp_lut, shr_floor, shr_round, shr_round0};
use crate::optim::Optimizer;

/// Context positions; the region layout below is written for exactly this many.
pub const CTX: usize = 8;
const SHADOW_MAX: i32 = (1 << 18) - 1;
/// Extra bits of precision the backward pass carries. Errors for the 255 wrong candidates are
/// small and must cancel the right one's across several probability multiplies; at Q12 they
/// round away and the gradient loses its direction.
const ERR_BITS: u32 = 5;
/// Bound on error signals: with a Q12 weight the product stays under 2^30.
const ERR_MAX: i32 = (1 << (12 + ERR_BITS)) - 1;
/// Largest number of replicas a config may ask for.
pub const MAX_FAN_IN: usize = 64;
/// Largest fan-in the log table covers: the final normalisation runs over all 256 candidates.
const LOG_FAN_IN: usize = VOCAB;
/// Scale of this model's gradients against the attention head's, for [`crate::optim::Rule::Sgd`].
const SGD_OFFSET: i32 = 6;

/// 256 * ln(i / 256) for i in 256..=256 * LOG_FAN_IN, by inverting the integer exp table.
fn log_lut() -> Vec<i32> {
    let exp = exp_lut();
    let mut out = vec![0i32; 256 * LOG_FAN_IN + 1];
    let mut d = 0usize;
    for (i, slot) in out.iter_mut().enumerate().skip(256) {
        // smallest d with exp(-d/256) <= 256 / i
        while (exp[d] as i64) * (i as i64) > 1 << 28 {
            d += 1;
        }
        *slot = d as i32;
    }
    out
}

pub struct Spn {
    cfg: Config,
    /// Q16 shadows, used as `shadow >> 6` (Q8 logs, up to 16 nats either way).
    /// `[(CTX + 1) * 256, R * I]`: leaf log-potentials by (position, byte value).
    leaf: Tensor<2, Int>,
    /// `[R * K, I * I]`: sum layer A over (region 0, region 1) pairs.
    wa: Tensor<2, Int>,
    /// `[R * K, I * I]`: sum layer B over (region 3, region 2) pairs.
    wb: Tensor<2, Int>,
    /// `[R, K * K]`: each replica's root over (B, A) pairs.
    wroot: Tensor<2, Int>,
    /// `[1, R]`: the mix over replicas.
    wmix: Tensor<2, Int>,
    /// Per region `[1, CTX, R, 1]`: 1 where a context position belongs to it.
    member: Vec<Tensor<4, Int>>,
    /// `[1, CTX]`: each position's first row in `leaf`.
    row_base: Tensor<2, Int>,
    exp: Tensor<1, Int>,
    log: Tensor<1, Int>,
    opt: Optimizer,
    pending: Option<Cache>,
    device: Device,
}

struct Cache {
    idx: Tensor<1, Int>,
    prob: Tensor<2, Int>,
    p_mix: Tensor<3, Int>,
    p_root: Tensor<4, Int>,
    p_root_a: Tensor<4, Int>,
    p_a: Tensor<4, Int>,
    p_b3: Tensor<5, Int>,
    p_b2: Tensor<5, Int>,
}

/// The seven gradients, in the order the optimizer numbers its groups.
pub struct Grads {
    pub leaf: Tensor<2, Int>,
    pub wa: Tensor<2, Int>,
    pub wb: Tensor<2, Int>,
    pub wroot: Tensor<2, Int>,
    pub wmix: Tensor<2, Int>,
}

impl Spn {
    pub fn new(cfg: Config, device: &Device) -> Self {
        let (r, i) = (cfg.hid, cfg.emb);
        let mut rng = Rng(cfg.seed);
        let mut mk = |rows: usize, cols: usize, span: i32| -> Tensor<2, Int> {
            Tensor::from_data(TensorData::new(rng.ints(rows * cols, span << 6), [rows, cols]), device)
        };
        let (leaf, wa, wb, wroot, wmix) =
            (mk((CTX + 1) * VOCAB, r * i, 64), mk(r * i, i * i, 16), mk(r * i, i * i, 16), mk(r, i * i, 16), mk(1, r, 16));
        // the random part: which context positions share a leaf region, drawn per replica
        let mut member = vec![vec![0i32; CTX * r]; 4];
        for rep in 0..r {
            let mut order: Vec<usize> = (0..CTX).collect();
            for at in (1..CTX).rev() {
                order.swap(at, (rng.next() >> 33) as usize % (at + 1));
            }
            for (slot, pos) in order.into_iter().enumerate() {
                member[slot / 2][pos * r + rep] = 1;
            }
        }
        let member = member.into_iter().map(|m| Tensor::from_data(TensorData::new(m, [1, CTX, r, 1]), device)).collect();
        let bases: Vec<i32> = (0..CTX).map(|d| (d * VOCAB) as i32).collect();
        Self {
            cfg,
            leaf,
            wa,
            wb,
            wroot,
            wmix,
            member,
            row_base: Tensor::from_data(TensorData::new(bases, [1, CTX]), device),
            exp: Tensor::from_data(TensorData::new(exp_lut(), [LOGIT_SPAN as usize + 1]), device),
            log: Tensor::from_data(TensorData::new(log_lut(), [256 * LOG_FAN_IN + 1]), device),
            opt: Optimizer::new(cfg.rule, cfg.lr, 5),
            pending: None,
            device: device.clone(),
        }
    }

    /// Integer log-sum-exp over the last dim of Q8 logs. Returns the Q20 exp weights, the result
    /// (last dim kept, size 1) and the Q12 softmax weights, which are its gradient.
    fn lse<const D: usize>(&self, x: Tensor<D, Int>) -> (Tensor<D, Int>, Tensor<D, Int>, Tensor<D, Int>) {
        let dims = x.dims();
        let top = x.clone().max_dim(D - 1);
        let d = (top.clone() - x).clamp(0, LOGIT_SPAN);
        let e = self.exp.clone().select(0, d.reshape([dims.iter().product::<usize>()])).reshape(dims);
        // the sum is in [2^20, fan_in * 2^20] and non-negative, so the raw shift is safe
        let total = e.clone().sum_dim(D - 1).bitwise_right_shift_scalar(12);
        let tdims = total.dims();
        let ln = self.log.clone().select(0, total.clone().reshape([tdims.iter().product::<usize>()])).reshape(tdims);
        (e.clone(), top + ln, e / total)
    }

    fn forward(&self, ctx: Tensor<2, Int>) -> (Tensor<2, Int>, Cache) {
        let (b, r, i) = (self.cfg.batch, self.cfg.hid, self.cfg.emb);
        let w = |t: &Tensor<2, Int>| shr_floor(t.clone(), 6);
        let leaf = w(&self.leaf);

        // leaf regions: each sums its positions' log-potentials, [B, R, I]
        let idx = (ctx + self.row_base.clone()).reshape([b * CTX]);
        let f = leaf.clone().select(0, idx.clone()).reshape([b, CTX, r, i]);
        let region = |l: usize| (f.clone() * self.member[l].clone()).sum_dim(1).reshape([b, r, i]);
        let (l0, l1, l2, l3) = (region(0), region(1), region(2), region(3));
        // region 3 also holds the next byte: one row per candidate, [B, 256, R, I]
        let l3 = l3.reshape([b, 1, r, i]) + leaf.narrow(0, CTX * VOCAB, VOCAB).reshape([1, VOCAB, r, i]);

        // sum layer A over all (i, j) pairs of regions 0 and 1: [B, R, K, I*I] -> [B, R, K]
        let pairs = (l0.reshape([b, r, i, 1]) + l1.reshape([b, r, 1, i])).reshape([b, r, 1, i * i]);
        let (_, a, p_a) = self.lse(pairs + w(&self.wa).reshape([1, r, i, i * i]));

        // sum layer B = lse over (j, i) of wb + l3_j + l2_i, nested: the inner lse over region 2
        // is narrow, and only the outer one over region 3 carries the candidates
        let (_, inner, p_b2) = self.lse(w(&self.wb).reshape([1, r, i, i, i]) + l2.reshape([b, r, 1, 1, i]));
        let (_, bsum, p_b3) = self.lse(l3.reshape([b, VOCAB, r, 1, i]) + inner.reshape([b, 1, r, i, i]));

        // root = lse over (c, a) of wroot + B_c + A_a, nested the same way
        let (_, via_a, p_root_a) = self.lse(w(&self.wroot).reshape([1, r, i, i]) + a.reshape([b, r, 1, i]));
        let (_, root, p_root) = self.lse(bsum.reshape([b, VOCAB, r, i]) + via_a.reshape([b, 1, r, i]));

        // mix the replicas, then normalise over the candidates
        let (_, logit, p_mix) = self.lse(root.reshape([b, VOCAB, r]) + w(&self.wmix).reshape([1, 1, r]));
        let (e, _, prob) = self.lse(logit.reshape([b, VOCAB]));
        (e, Cache { idx, prob, p_mix, p_root, p_root_a, p_a, p_b3, p_b2 })
    }

    /// Gradients of the summed code length, averaged down to a per-128-rows scale.
    fn grads(&self, c: &Cache, target: Tensor<1, Int>) -> Grads {
        let (b, r, i) = (self.cfg.batch, self.cfg.hid, self.cfg.emb);
        let scale = (b / 128).trailing_zeros() + ERR_BITS;
        let narrow = |t: Tensor<4, Int>| t.clamp(-ERR_MAX, ERR_MAX);

        // |g| <= 2^17; every product below is an error (<= 2^17) times a Q12 weight. The shifts
        // round to nearest: a floor's half-unit bias would be summed over 256 candidates and
        // swamp the small errors.
        let g = (c.prob.clone() - target.one_hot::<2>(VOCAB) * 4096) * (1 << ERR_BITS);
        let d_root = shr_round(g.reshape([b, VOCAB, 1]) * c.p_mix.clone(), 12);
        // summed over candidates (clamped back to 2^17), then rows: 2^17 * B <= 2^26
        let d_wmix = d_root.clone().sum_dim(1).clamp(-ERR_MAX, ERR_MAX).sum_dim(0).reshape([1, r]);

        // root: outer lse over c carries the candidates, inner lse over a does not
        let d_b = shr_round(d_root.reshape([b, VOCAB, r, 1]) * c.p_root.clone(), 12);
        let d_via = narrow(d_b.clone().sum_dim(1).reshape([b, r, i, 1]));
        let t_root = shr_round(d_via * c.p_root_a.clone(), 12);
        let d_wroot = t_root.clone().sum_dim(0).reshape([r, i * i]);
        let d_a = narrow(t_root.sum_dim(2).reshape([b, r, i, 1]));

        // sum layer A
        let t_a = shr_round(d_a * c.p_a.clone(), 12);
        let d_wa = t_a.clone().sum_dim(0).reshape([r * i, i * i]);
        let t_a = t_a.sum_dim(2).reshape([b, r, i, i]);
        let d_l0 = narrow(t_a.clone().sum_dim(3).reshape([b, 1, r, i]));
        let d_l1 = narrow(t_a.sum_dim(2).reshape([b, 1, r, i]));

        // sum layer B: outer over region 3 (with candidates), inner over region 2
        let t_b3 = shr_round(d_b.reshape([b, VOCAB, r, i, 1]) * c.p_b3.clone(), 12);
        let d_l3 = t_b3.clone().sum_dim(3).reshape([b, VOCAB, r, i]);
        let d_inner = t_b3.sum_dim(1).reshape([b, r, i, i, 1]).clamp(-ERR_MAX, ERR_MAX);
        let t_b2 = shr_round(d_inner * c.p_b2.clone(), 12);
        let d_wb = t_b2.clone().sum_dim(0).reshape([r * i, i * i]);
        let d_l2 = narrow(t_b2.sum_dim(2).sum_dim(3).reshape([b, 1, r, i]));

        // leaves: the next byte's rows take region 3's error per candidate; each context position
        // takes the error of the region it sits in
        let d_next = d_l3.clone().sum_dim(0).reshape([VOCAB, r * i]);
        let d_l3 = narrow(d_l3.sum_dim(1).reshape([b, 1, r, i]));
        let d_f = [d_l0, d_l1, d_l2, d_l3]
            .into_iter()
            .zip(&self.member)
            .map(|(d, m)| d * m.clone())
            .reduce(|x, y| x + y)
            .unwrap();
        let d_ctx = Tensor::<2, Int>::zeros([CTX * VOCAB, r * i], &self.device).select_assign(
            0,
            c.idx.clone(),
            d_f.reshape([b * CTX, r * i]),
            IndexingUpdateOp::Add,
        );
        Grads {
            leaf: shr_round0(Tensor::cat(vec![d_ctx, d_next], 0), scale),
            wa: shr_round0(d_wa, scale),
            wb: shr_round0(d_wb, scale),
            wroot: shr_round0(d_wroot, scale),
            wmix: shr_round0(d_wmix, scale),
        }
    }

    fn apply(&mut self, g: Grads) {
        let o = &mut self.opt;
        o.tick();
        self.leaf = o.step(0, &self.leaf, g.leaf, SGD_OFFSET, SHADOW_MAX);
        self.wa = o.step(1, &self.wa, g.wa, SGD_OFFSET, SHADOW_MAX);
        self.wb = o.step(2, &self.wb, g.wb, SGD_OFFSET, SHADOW_MAX);
        self.wroot = o.step(3, &self.wroot, g.wroot, SGD_OFFSET, SHADOW_MAX);
        self.wmix = o.step(4, &self.wmix, g.wmix, SGD_OFFSET, SHADOW_MAX);
    }
}

impl Predictor for Spn {
    fn predict(&mut self, ctx: Tensor<2, Int>, _guess: Tensor<2, Int>) -> Vec<i32> {
        let (e, cache) = self.forward(ctx);
        self.pending = Some(cache);
        (e.bitwise_right_shift_scalar(4) + 1).into_data().convert::<i32>().try_to_vec().unwrap()
    }

    fn learn(&mut self, target: Tensor<1, Int>) {
        let cache = self.pending.take().expect("learn without a preceding predict");
        let g = self.grads(&cache, target);
        self.apply(g);
    }

    fn observe(&mut self, cost_q8: u64) {
        self.opt.observe(cost_q8, 8 * 256 * self.cfg.batch as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::{Arch, Head};
    use crate::optim::Rule;
    use burn::tensor::activation::log_softmax;

    const CFG: Config = Config {
        arch: Arch::Spn,
        ctx: CTX,
        emb: 4,
        hid: 3,
        batch: 128,
        lr: 12,
        rule: Rule::Sgd,
        head: Head::Softmax,
        matcher: false,
        seed: 7,
    };

    fn ints(t: &Tensor<2, Int>) -> Vec<i32> {
        t.clone().into_data().convert::<i32>().try_to_vec().unwrap()
    }

    fn batch(rng: &mut Rng, device: &Device) -> (Tensor<2, Int>, Tensor<1, Int>) {
        let ctx: Vec<i32> = (0..CFG.batch * CTX).map(|_| ((rng.next() >> 33) % 23 + 97) as i32).collect();
        let tgt: Vec<i32> = ctx.chunks(CTX).map(|c| c[CTX - 1]).collect();
        (
            Tensor::from_data(TensorData::new(ctx, [CFG.batch, CTX]), device),
            Tensor::from_data(TensorData::new(tgt, [CFG.batch]), device),
        )
    }

    /// The handwritten integer backward pass against float autodiff on the same circuit.
    #[test]
    fn integer_gradients_follow_autodiff() {
        let device = Device::flex();
        let mut rng = Rng(21);
        let mut model = Spn::new(CFG, &device);
        for _ in 0..30 {
            let (ctx, tgt) = batch(&mut rng, &device);
            let (_, c) = model.forward(ctx);
            let g = model.grads(&c, tgt);
            model.apply(g);
        }
        let (ctx, tgt) = batch(&mut rng, &device);
        let (_, cache) = model.forward(ctx.clone());
        let got = model.grads(&cache, tgt.clone());

        // float twin on the forward (Q8 log) weights; lse(x) = 256 * logsumexp(x / 256)
        let ad = device.clone().autodiff();
        let (b, r, i) = (CFG.batch, CFG.hid, CFG.emb);
        let leaf_of = |s: &Tensor<2, Int>| -> Tensor<2> {
            let v: Vec<f32> = ints(&shr_floor(s.clone(), 6)).iter().map(|x| *x as f32).collect();
            Tensor::<2>::from_data(TensorData::new(v, s.dims()), &ad).require_grad()
        };
        fn lse<const D: usize>(x: Tensor<D>) -> Tensor<D> {
            let top = x.clone().max_dim(D - 1).detach();
            ((x - top.clone()) / 256.0).exp().sum_dim(D - 1).log() * 256.0 + top
        }
        let (leaf, wa, wb, wroot, wmix) =
            (leaf_of(&model.leaf), leaf_of(&model.wa), leaf_of(&model.wb), leaf_of(&model.wroot), leaf_of(&model.wmix));
        let member = |l: usize| -> Tensor<4> {
            let v: Vec<f32> = model.member[l].clone().into_data().convert::<i32>().try_to_vec::<i32>().unwrap().iter().map(|x| *x as f32).collect();
            Tensor::<4>::from_data(TensorData::new(v, [1, CTX, r, 1]), &ad)
        };
        let idx = Tensor::<1, Int>::from_data(cache.idx.clone().into_data(), &ad);
        let f = leaf.clone().select(0, idx).reshape([b, CTX, r, i]);
        let region = |l: usize| (f.clone() * member(l)).sum_dim(1).reshape([b, r, i]);
        let (l0, l1, l2, l3) = (region(0), region(1), region(2), region(3));
        let l3 = l3.reshape([b, 1, r, i]) + leaf.clone().narrow(0, CTX * VOCAB, VOCAB).reshape([1, VOCAB, r, i]);
        let pairs = (l0.reshape([b, r, i, 1]) + l1.reshape([b, r, 1, i])).reshape([b, r, 1, i * i]);
        let a = lse(pairs + wa.clone().reshape([1, r, i, i * i]));
        let inner = lse(wb.clone().reshape([1, r, i, i, i]) + l2.reshape([b, r, 1, 1, i]));
        let bsum = lse(l3.reshape([b, VOCAB, r, 1, i]) + inner.reshape([b, 1, r, i, i]));
        let via_a = lse(wroot.clone().reshape([1, r, i, i]) + a.reshape([b, r, 1, i]));
        let root = lse(bsum.reshape([b, VOCAB, r, i]) + via_a.reshape([b, 1, r, i]));
        let logit = lse(root.reshape([b, VOCAB, r]) + wmix.clone().reshape([1, 1, r])).reshape([b, VOCAB]);
        let tgt_ad = Tensor::<1, Int>::from_data(tgt.into_data(), &ad);
        let loss = -log_softmax(logit / 256.0, 1).gather(1, tgt_ad.reshape([b, 1])).sum();
        let grads = loss.backward();

        let names = ["leaf", "wa", "wb", "wroot", "wmix"];
        let mut worst = 1.0f64;
        let got = [&got.leaf, &got.wa, &got.wb, &got.wroot, &got.wmix];
        for ((name, want), got) in names.iter().zip([leaf, wa, wb, wroot, wmix]).zip(got) {
            let want: Vec<f32> = want.grad(&grads).unwrap().into_data().convert::<f32>().try_to_vec().unwrap();
            let got: Vec<f64> = ints(got).iter().map(|x| *x as f64).collect();
            let dot: f64 = want.iter().zip(&got).map(|(a, b)| *a as f64 * b).sum();
            let (na, nb) = (
                want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt(),
                got.iter().map(|b| b * b).sum::<f64>().sqrt(),
            );
            let cos = dot / (na * nb);
            println!("{name}: cosine {cos:.3}, |int| {nb:.0}, |float| {na:.4}");
            worst = worst.min(cos);
        }
        assert!(worst > 0.7, "an integer gradient disagrees with autodiff (worst cosine {worst:.3})");
    }
}
