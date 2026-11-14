//! A random sum-product network (RAT-SPN, Peharz et al. 2018) as a byte predictor, in integer
//! log-domain arithmetic.
//!
//! The circuit scores the joint assignment of the 8 context bytes and the next byte; evaluating it
//! for all 256 candidates gives one log score each, which [`crate::mix::Mix`] normalises into the
//! next byte's distribution, the way Liu, Mandt & Van den Broeck (2021) drive an arithmetic coder
//! from a probabilistic circuit. Product nodes are additions of Q8 log values; sum nodes are
//! integer log-sum-exp from the same exp table the softmax uses. The weights are unnormalised,
//! which the final normalisation over the next byte makes harmless.
//!
//! Structure, repeated in `replicas` copies that a final sum node mixes. Every context position is
//! a leaf with `comps` distributions, and the positions are dealt at random onto the 8 leaves of a
//! binary tree, drawn afresh per replica. Three sum layers merge the leaves pairwise (8 -> 4 -> 2
//! -> 1), each with `comps` sum nodes over all pairs of its children's nodes. The next byte is its
//! own leaf, joined to the context tree's root by a last sum layer, so only that layer carries
//! the 256-wide candidate dimension.

use burn::tensor::ops::IndexingUpdateOp;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::{Config, LOGIT_SPAN, Rng, VOCAB, exp_lut, shr_floor, shr_round, shr_round0};
use crate::optim::Optimizer;

/// Context positions: the leaves of a depth-3 binary tree.
pub const CTX: usize = 8;
/// Sum layers between the leaves and the context tree's root.
const LEVELS: usize = 3;
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
/// Scale of this model's gradients against the attention head's, for [`crate::optim::Rule::Sgd`];
/// set so one `lr` suits both when they are mixed.
const SGD_OFFSET: i32 = 14;
/// Initial spread of the leaf log-potentials and sum weights, in Q8 logs (about 12 and 6 nats).
/// A mixture whose components start alike never separates them: with the small spread the
/// attention model uses, this network learned only the next byte's overall frequencies.
/// Updates between structure adaptations, and the weight gaps they act on, as Q16 shadows:
/// about 8 nats below a sum node's largest weight to prune, and 2 nats below it to regrow.
const ADAPT_EVERY: u64 = 1024;
const PRUNE_GAP: i32 = 2048 << 6;
const REGROW_GAP: i32 = 512 << 6;
const LEAF_SPAN: i32 = 3000;
const WEIGHT_SPAN: i32 = 1500;

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
    /// `[(CTX + 1) * 256, R * I]`: leaf log-potentials by (position, byte value); the last 256
    /// rows are the next byte's.
    leaf: Tensor<2, Int>,
    /// Per sum layer, `[N * R * K, I * I]` for its N = 4, 2, 1 nodes: weights over child pairs.
    layers: Vec<Tensor<2, Int>>,
    /// `[R * J, K]`: the top sum layer, over (next-byte distribution j, context root node k).
    wtop: Tensor<2, Int>,
    /// `[1, R]`: the mix over replicas.
    wmix: Tensor<2, Int>,
    /// `[1, 8, CTX, R, 1]`: 1 where a context position sits on a tree leaf.
    member: Tensor<5, Int>,
    /// `[1, CTX]`: each position's first row in `leaf`.
    row_base: Tensor<2, Int>,
    exp: Tensor<1, Int>,
    log: Tensor<1, Int>,
    opt: Optimizer,
    updates: u64,
    device: Device,
}

pub struct Cache {
    idx: Tensor<1, Int>,
    p_mix: Tensor<3, Int>,
    p_top: Tensor<4, Int>,
    p_join: Tensor<4, Int>,
    /// Per sum layer, `[B, N, R, K, I * I]`.
    p_layers: Vec<Tensor<5, Int>>,
}

/// Gradients in the order the optimizer numbers its groups: leaf, the three layers, top, mix.
pub struct Grads {
    pub leaf: Tensor<2, Int>,
    pub layers: Vec<Tensor<2, Int>>,
    pub wtop: Tensor<2, Int>,
    pub wmix: Tensor<2, Int>,
}

impl Spn {
    pub fn new(cfg: Config, device: &Device) -> Self {
        let (r, i) = (cfg.replicas, cfg.comps);
        let mut rng = Rng(cfg.seed);
        let mut mk = |rows: usize, cols: usize, span: i32| -> Tensor<2, Int> {
            Tensor::from_data(TensorData::new(rng.ints(rows * cols, span << 6), [rows, cols]), device)
        };
        let leaf = mk((CTX + 1) * VOCAB, r * i, LEAF_SPAN);
        let layers = (0..LEVELS).map(|l| mk((CTX >> (l + 1)) * r * i, i * i, WEIGHT_SPAN)).collect();
        let (wtop, wmix) = (mk(r * i, i, WEIGHT_SPAN), mk(1, r, 16));
        // the random part: which context position sits on which leaf of the tree, per replica
        let mut member = vec![0i32; CTX * CTX * r];
        for rep in 0..r {
            let mut order: Vec<usize> = (0..CTX).collect();
            for at in (1..CTX).rev() {
                order.swap(at, (rng.next() >> 33) as usize % (at + 1));
            }
            for (leaf_at, pos) in order.into_iter().enumerate() {
                member[(leaf_at * CTX + pos) * r + rep] = 1;
            }
        }
        let bases: Vec<i32> = (0..CTX).map(|d| (d * VOCAB) as i32).collect();
        Self {
            cfg,
            leaf,
            layers,
            wtop,
            wmix,
            member: Tensor::from_data(TensorData::new(member, [1, CTX, CTX, r, 1]), device),
            row_base: Tensor::from_data(TensorData::new(bases, [1, CTX]), device),
            exp: Tensor::from_data(TensorData::new(exp_lut(), [LOGIT_SPAN as usize + 1]), device),
            log: Tensor::from_data(TensorData::new(log_lut(), [256 * LOG_FAN_IN + 1]), device),
            opt: Optimizer::new(cfg.rule, cfg.lr, LEVELS + 3),
            updates: 0,
            device: device.clone(),
        }
    }

    /// Integer log-sum-exp over the last dim of Q8 logs. Returns the result (last dim kept, size
    /// 1) and the Q12 softmax weights, which are its gradient.
    fn lse<const D: usize>(&self, x: Tensor<D, Int>) -> (Tensor<D, Int>, Tensor<D, Int>) {
        let dims = x.dims();
        let top = x.clone().max_dim(D - 1);
        let d = (top.clone() - x).clamp(0, LOGIT_SPAN);
        let e = self.exp.clone().select(0, d.reshape([dims.iter().product::<usize>()])).reshape(dims);
        // the sum is in [2^20, fan_in * 2^20] and non-negative, so the raw shift is safe
        let total = e.clone().sum_dim(D - 1).bitwise_right_shift_scalar(12);
        let tdims = total.dims();
        let ln = self.log.clone().select(0, total.clone().reshape([tdims.iter().product::<usize>()])).reshape(tdims);
        (top + ln, e / total)
    }

    /// Q8 log score of every candidate next byte, `[B, 256]`.
    pub fn scores(&self, ctx: Tensor<2, Int>) -> (Tensor<2, Int>, Cache) {
        let (b, r, i) = (self.cfg.batch, self.cfg.replicas, self.cfg.comps);
        let w = |t: &Tensor<2, Int>| shr_floor(t.clone(), 6);
        let leaf = w(&self.leaf);

        // tree leaves: each takes the log-potentials of the position dealt to it, [B, 8, R, I]
        let idx = (ctx + self.row_base.clone()).reshape([b * CTX]);
        let f = leaf.clone().select(0, idx.clone()).reshape([b, 1, CTX, r, i]);
        let mut x = (f * self.member.clone()).sum_dim(2).reshape([b, CTX, r, i]);

        // sum layers: node n merges children 2n and 2n+1 over all (i, j) pairs of their nodes
        let mut p_layers = Vec::with_capacity(LEVELS);
        for weights in &self.layers {
            let n = x.dims()[1] / 2;
            let kids = x.reshape([b, n, 2, r, i]);
            let (left, right) = (kids.clone().narrow(2, 0, 1), kids.narrow(2, 1, 1));
            let pairs = (left.reshape([b, n, r, i, 1]) + right.reshape([b, n, r, 1, i])).reshape([b, n, r, 1, i * i]);
            let (out, p) = self.lse(pairs + w(weights).reshape([1, n, r, i, i * i]));
            p_layers.push(p);
            x = out.reshape([b, n, r, i]);
        }

        // top layer = lse over (j, k) of wtop + next_j(candidate) + root_k, nested so that only
        // the outer lse over j carries the candidates
        let (join, p_join) = self.lse(w(&self.wtop).reshape([1, r, i, i]) + x.reshape([b, r, 1, i]));
        let next = leaf.narrow(0, CTX * VOCAB, VOCAB).reshape([1, VOCAB, r, i]);
        let (score, p_top) = self.lse(next + join.reshape([b, 1, r, i]));

        // mix the replicas
        let (logit, p_mix) = self.lse(score.reshape([b, VOCAB, r]) + w(&self.wmix).reshape([1, 1, r]));
        (logit.reshape([b, VOCAB]), Cache { idx, p_mix, p_top, p_join, p_layers })
    }

    /// Gradients from the error at the scores (`g`, Q12, `|g| <= 2^12`), averaged down to a
    /// per-128-rows scale.
    fn grads(&self, c: &Cache, g: Tensor<2, Int>) -> Grads {
        let (b, r, i) = (self.cfg.batch, self.cfg.replicas, self.cfg.comps);
        let scale = (b / 128).trailing_zeros() + ERR_BITS;
        // |g| <= 2^17; every product below is an error (<= 2^17) times a Q12 weight. The shifts
        // round to nearest: a floor's half-unit bias would be summed over 256 candidates and
        // swamp the small errors. Sums over a batch of 512 stay under 2^27.
        let g = g * (1 << ERR_BITS);
        let d_score = shr_round(g.reshape([b, VOCAB, 1]) * c.p_mix.clone(), 12);
        let d_wmix = d_score.clone().sum_dim(1).clamp(-ERR_MAX, ERR_MAX).sum_dim(0).reshape([1, r]);

        // top layer: the next byte's leaf takes the error per candidate; the context side takes
        // it summed over candidates
        let t_top = shr_round(d_score.reshape([b, VOCAB, r, 1]) * c.p_top.clone(), 12);
        let d_next = t_top.clone().sum_dim(0).reshape([VOCAB, r * i]);
        let d_join = t_top.sum_dim(1).reshape([b, r, i, 1]).clamp(-ERR_MAX, ERR_MAX);
        let t_join = shr_round(d_join * c.p_join.clone(), 12);
        let d_wtop = t_join.clone().sum_dim(0).reshape([r * i, i]);
        let mut d_x = t_join.sum_dim(2).reshape([b, 1, r, i]).clamp(-ERR_MAX, ERR_MAX);

        // sum layers, root to leaves
        let mut d_layers = Vec::with_capacity(LEVELS);
        for p in c.p_layers.iter().rev() {
            let n = d_x.dims()[1];
            let t = shr_round(d_x.reshape([b, n, r, i, 1]) * p.clone(), 12);
            d_layers.push(t.clone().sum_dim(0).reshape([n * r * i, i * i]));
            let t = t.sum_dim(3).reshape([b, n, r, i, i]);
            let left = t.clone().sum_dim(4).reshape([b, n, r, i]);
            let right = t.sum_dim(3).reshape([b, n, r, i]);
            d_x = Tensor::stack::<5>(vec![left, right], 2).reshape([b, 2 * n, r, i]).clamp(-ERR_MAX, ERR_MAX);
        }
        d_layers.reverse();

        // leaves: each context position takes the error of the tree leaf it was dealt to
        let d_f = (d_x.reshape([b, CTX, 1, r, i]) * self.member.clone().swap_dims(1, 2)).sum_dim(1);
        let d_ctx = Tensor::<2, Int>::zeros([CTX * VOCAB, r * i], &self.device).select_assign(
            0,
            c.idx.clone(),
            d_f.reshape([b * CTX, r * i]),
            IndexingUpdateOp::Add,
        );
        Grads {
            leaf: shr_round0(Tensor::cat(vec![d_ctx, d_next], 0), scale),
            layers: d_layers.into_iter().map(|d| shr_round0(d, scale)).collect(),
            wtop: shr_round0(d_wtop, scale),
            wmix: shr_round0(d_wmix, scale),
        }
    }

    fn apply(&mut self, g: Grads) {
        let o = &mut self.opt;
        o.tick();
        self.leaf = o.step(0, &self.leaf, g.leaf, SGD_OFFSET, SHADOW_MAX);
        for (l, d) in g.layers.into_iter().enumerate() {
            self.layers[l] = o.step(1 + l, &self.layers[l], d, SGD_OFFSET, SHADOW_MAX);
        }
        self.wtop = o.step(LEVELS + 1, &self.wtop, g.wtop, SGD_OFFSET, SHADOW_MAX);
        self.wmix = o.step(LEVELS + 2, &self.wmix, g.wmix, SGD_OFFSET, SHADOW_MAX);
    }

    /// Trains on the error at the scores of a pass made by [`Self::scores`].
    pub fn learn_from(&mut self, c: &Cache, g: Tensor<2, Int>) {
        let grads = self.grads(c, g);
        self.apply(grads);
        self.updates += 1;
        if self.cfg.adapt > 0 && self.updates % ADAPT_EVERY == 0 {
            self.adapt();
        }
    }

    /// Structure adaptation, from the integer weights alone so a decoder repeats it. In every sum
    /// node, a child whose weight has fallen [`PRUNE_GAP`] below the node's largest is pruned:
    /// pinned to the floor, where it contributes nothing. With regrowth on, it is instead
    /// restarted [`REGROW_GAP`] below the largest, giving the node a fresh child to specialise.
    fn adapt(&mut self) {
        let regrow = self.cfg.adapt == 2;
        let fix = |w: &Tensor<2, Int>| -> Tensor<2, Int> {
            let top = w.clone().max_dim(1);
            let weak = (top.clone() - w.clone()).greater_elem(PRUNE_GAP).int();
            let to = if regrow { top - REGROW_GAP } else { top * 0 - SHADOW_MAX };
            w.clone() * (weak.clone() * -1 + 1) + weak * to
        };
        self.layers = self.layers.iter().map(&fix).collect();
        self.wtop = fix(&self.wtop);
    }

    pub fn observe(&mut self, cost_q8: u64) {
        self.opt.observe(cost_q8, 8 * 256 * self.cfg.batch as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::{Arch, Head, softmax_q12, test_device};
    use crate::optim::Rule;
    use burn::tensor::activation::log_softmax;

    const CFG: Config = Config {
        arch: Arch::Spn,
        ctx: CTX,
        emb: 32,
        hid: 256,
        batch: 128,
        lr: 20,
        rule: Rule::Sgd,
        head: Head::Softmax,
        matcher: false,
        comps: 4,
        replicas: 3,
        short: 0,
        adapt: 0,
        warm: 0,
        norm: false,
        mask: false,
        seed: 7,
    };

    fn ints<const D: usize>(t: &Tensor<D, Int>) -> Vec<i32> {
        t.clone().into_data().convert::<i32>().try_to_vec().unwrap()
    }

    /// The error at the scores for a batch, as [`crate::mix::Mix`] forms it: Q12 and zero-sum.
    fn error(model: &Spn, scores: Tensor<2, Int>, tgt: Tensor<1, Int>) -> Tensor<2, Int> {
        let p = softmax_q12(&model.exp, scores).1;
        p.clone() - tgt.one_hot::<2>(VOCAB) * p.sum_dim(1)
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
        let device = test_device();
        let mut rng = Rng(21);
        let mut model = Spn::new(CFG, &device);
        for _ in 0..30 {
            let (ctx, tgt) = batch(&mut rng, &device);
            let (scores, c) = model.scores(ctx);
            let g = error(&model, scores, tgt);
            model.learn_from(&c, g);
        }
        let (ctx, tgt) = batch(&mut rng, &device);
        let (scores, cache) = model.scores(ctx.clone());
        let got = model.grads(&cache, error(&model, scores, tgt.clone()));

        // float twin on the forward (Q8 log) weights; lse(x) = 256 * logsumexp(x / 256)
        let ad = device.clone().autodiff();
        let (b, r, i) = (CFG.batch, CFG.replicas, CFG.comps);
        let leaf_of = |s: &Tensor<2, Int>| -> Tensor<2> {
            let v: Vec<f32> = ints(&shr_floor(s.clone(), 6)).iter().map(|x| *x as f32).collect();
            Tensor::<2>::from_data(TensorData::new(v, s.dims()), &ad).require_grad()
        };
        fn lse<const D: usize>(x: Tensor<D>) -> Tensor<D> {
            let top = x.clone().max_dim(D - 1).detach();
            ((x - top.clone()) / 256.0).exp().sum_dim(D - 1).log() * 256.0 + top
        }
        let leaf = leaf_of(&model.leaf);
        let layers: Vec<Tensor<2>> = model.layers.iter().map(&leaf_of).collect();
        let (wtop, wmix) = (leaf_of(&model.wtop), leaf_of(&model.wmix));
        let member: Vec<f32> = ints(&model.member).iter().map(|x| *x as f32).collect();
        let member = Tensor::<5>::from_data(TensorData::new(member, [1, CTX, CTX, r, 1]), &ad);
        let idx = Tensor::<1, Int>::from_data(cache.idx.clone().into_data(), &ad);
        let f = leaf.clone().select(0, idx).reshape([b, 1, CTX, r, i]);
        let mut x = (f * member).sum_dim(2).reshape([b, CTX, r, i]);
        for weights in &layers {
            let n = x.dims()[1] / 2;
            let kids = x.reshape([b, n, 2, r, i]);
            let (left, right) = (kids.clone().narrow(2, 0, 1), kids.narrow(2, 1, 1));
            let pairs = (left.reshape([b, n, r, i, 1]) + right.reshape([b, n, r, 1, i])).reshape([b, n, r, 1, i * i]);
            x = lse(pairs + weights.clone().reshape([1, n, r, i, i * i])).reshape([b, n, r, i]);
        }
        let join = lse(wtop.clone().reshape([1, r, i, i]) + x.reshape([b, r, 1, i]));
        let next = leaf.clone().narrow(0, CTX * VOCAB, VOCAB).reshape([1, VOCAB, r, i]);
        let score = lse(next + join.reshape([b, 1, r, i]));
        let logit = lse(score.reshape([b, VOCAB, r]) + wmix.clone().reshape([1, 1, r])).reshape([b, VOCAB]);
        let tgt_ad = Tensor::<1, Int>::from_data(tgt.into_data(), &ad);
        let loss = -log_softmax(logit / 256.0, 1).gather(1, tgt_ad.reshape([b, 1])).sum();
        let grads = loss.backward();

        let mut pairs = vec![("leaf".to_string(), leaf, &got.leaf)];
        for (l, (w, d)) in layers.into_iter().zip(&got.layers).enumerate() {
            pairs.push((format!("layer {l}"), w, d));
        }
        pairs.push(("top".to_string(), wtop, &got.wtop));
        pairs.push(("mix".to_string(), wmix, &got.wmix));
        let mut worst = 1.0f64;
        for (name, want, got) in pairs {
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
