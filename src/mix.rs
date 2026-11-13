//! Combines predictors by adding their Q8 scores for each candidate byte (a product of experts),
//! then the match finder's boost, then one softmax. Every part is optional, so this is where the
//! permutations are tried: attention (with any head), a second attention branch over a shorter
//! context, the sum-product network, and the match finder, in any mix.
//! All parts train on the same error at the summed scores.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fused;
use crate::fixed::{
    self, Arch, Config, LOGIT_SPAN, Model, Predictor, VOCAB, below_table, coded, exp_lut, one_hot_rows, shr_floor, shr_round0, softmax_q12,
};
use crate::optim::Optimizer;
use crate::spn::{self, Spn};

const SHADOW_MAX: i32 = (1 << 18) - 1;
/// Scale of the boost gain's gradient against the attention head's, for plain SGD.
const COPY_OFFSET: i32 = 6;

pub struct Mix {
    cfg: Config,
    attn: Option<Model>,
    /// A second attention model that sees only the last `cfg.short` bytes.
    short: Option<Model>,
    spn: Option<Spn>,
    /// `[3, 1]` Q16 shadow, used as `shadow >> 6`: logit added to the byte the match finder
    /// proposes, per trust level.
    copy: Tensor<2, Int>,
    opt: Optimizer,
    lut: Tensor<1, Int>,
    below: Tensor<2, Int>,
    /// Identity tables, selected by row in place of `one_hot`.
    eye: Tensor<2, Int>,
    eye3: Tensor<2, Int>,
    pending: Option<Pending>,
}

struct Pending {
    attn: Option<fixed::Cache>,
    short: Option<fixed::Cache>,
    spn: Option<spn::Cache>,
    /// Q12 probabilities; on the GPU the Q20 weights, which the error kernel normalises itself.
    prob: Tensor<2, Int>,
    guess_raw: Tensor<2, Int>,
}

impl Mix {
    pub fn new(cfg: Config, device: &Device) -> Self {
        Self {
            cfg,
            attn: (cfg.arch != Arch::Spn).then(|| Model::new(Config { matcher: false, ..cfg }, device)),
            short: (cfg.short > 0).then(|| {
                let branch = Config { arch: Arch::Attn, ctx: cfg.short, matcher: false, seed: cfg.seed + 1, ..cfg };
                Model::new(branch, device)
            }),
            spn: (cfg.arch != Arch::Attn).then(|| Spn::new(cfg, device)),
            copy: Tensor::zeros([3, 1], device),
            opt: Optimizer::new(cfg.rule, cfg.lr, 1).with_offsets(&[COPY_OFFSET]),
            lut: Tensor::from_data(TensorData::new(exp_lut(), [LOGIT_SPAN as usize + 1]), device),
            below: below_table(device),
            eye: one_hot_rows(VOCAB, device),
            eye3: one_hot_rows(3, device),
            pending: None,
        }
    }

    /// Runs every part and the shared head; returns the Q20 byte weights, `[B, VOCAB]`.
    fn weights(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Tensor<2, Int> {

        let b = self.cfg.batch;
        let guess_raw = guess;
        let attn = self.attn.as_ref().map(|m| m.scores(ctx.clone()));
        // the short branch reads the last of the context's bytes itself
        let short = self.short.as_ref().map(|m| m.scores(ctx.clone()));
        let spn = self.spn.as_ref().map(|m| m.scores(ctx));
        let mut total: Option<Tensor<2, Int>> = None;
        let parts = [attn.as_ref().map(|a| &a.0), short.as_ref().map(|a| &a.0), spn.as_ref().map(|s| &s.0)];
        for scores in parts.into_iter().flatten() {
            total = Some(match total {
                None => scores.clone(),
                Some(t) => t + scores.clone(),
            });
        }
        let mut total = total.expect("a mix needs at least one predictor");
        if fused::on_gpu(&total) {
            let e = fused::mix_weights(total, &guess_raw, &self.copy, &self.lut, self.cfg.matcher, LOGIT_SPAN);
            self.pending = Some(Pending { attn: attn.map(|a| a.1), short: short.map(|a| a.1), spn: spn.map(|s| s.1), prob: e.clone(), guess_raw });
            return e;
        }
        if self.cfg.matcher {
            let guess = fixed::split_guess(&guess_raw);
            let gain = shr_floor(self.copy.clone(), 6).select(0, guess.1.clone());
            total = total + self.eye.clone().select(0, guess.0.clone().reshape([b])) * gain;
        }
        let (e, prob) = softmax_q12(&self.lut, total);
        self.pending = Some(Pending { attn: attn.map(|a| a.1), short: short.map(|a| a.1), spn: spn.map(|s| s.1), prob, guess_raw });
        e
    }
}

impl Predictor for Mix {
    fn predict(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>) -> Vec<i32> {
        let e = self.weights(ctx, guess);
        (e.bitwise_right_shift_scalar(4) + 1).into_data().convert::<i32>().try_to_vec().unwrap()
    }

    fn predict_coded(&mut self, ctx: Tensor<2, Int>, guess: Tensor<2, Int>, target: Tensor<1, Int>) -> Tensor<2, Int> {
        let e = self.weights(ctx, guess);
        coded(e, target, &self.below)
    }

    fn learn(&mut self, target: Tensor<1, Int>) {
        let p = self.pending.take().expect("learn without a preceding predict");
        // dL/dscores, Q12; the scores were added, so every part sees the same error. The integer
        // probabilities sum to a little under 4096, so the true byte is charged the row's actual
        // sum: the error then sums to exactly zero and cannot push a sum node's weights all one way.
        let gpu = fused::on_gpu(&p.prob);
        let g = if gpu {
            fused::mix_error(&p.prob, target)
        } else {
            p.prob.clone() - self.eye.clone().select(0, target) * p.prob.sum_dim(1)
        };
        if let (Some(m), Some(c)) = (&mut self.attn, &p.attn) {
            m.learn_from(c, g.clone());
        }
        if let (Some(m), Some(c)) = (&mut self.short, &p.short) {
            m.learn_from(c, g.clone());
        }
        if let (Some(m), Some(c)) = (&mut self.spn, &p.spn) {
            m.learn_from(c, g.clone());
        }
        if self.cfg.matcher {
            // the proposed byte's error, summed over the rows at each trust level: <= B * 2^12
            let shift = (self.cfg.batch / 128).trailing_zeros();
            let d = if gpu {
                fused::guess_sums(&g, &p.guess_raw, shift)
            } else {
                let guess = fixed::split_guess(&p.guess_raw);
                let at_guess = g.gather(1, guess.0);
                shr_round0(self.eye3.clone().select(0, guess.1).transpose().matmul(at_guess), shift)
            };
            self.opt.tick();
            self.copy = self.opt.step(0, &self.copy, d, COPY_OFFSET, SHADOW_MAX);
        }
    }

    fn observe(&mut self, cost_q8: u64) {
        self.opt.observe(cost_q8, 8 * 256 * self.cfg.batch as u64);
        if let Some(m) = &mut self.attn {
            m.observe(cost_q8);
        }
        if let Some(m) = &mut self.short {
            m.observe(cost_q8);
        }
        if let Some(m) = &mut self.spn {
            m.observe(cost_q8);
        }
    }

    fn graphable(&self) -> bool {
        // the sum-product network replaces some of its state each step
        self.spn.is_none()
            && fused::on_gpu(&self.copy)
            && self.opt.sgd(0).is_some()
            && [&self.attn, &self.short].into_iter().flatten().all(|m| m.graphable())
    }

    fn refresh_steps(&mut self) {
        let _ = self.opt.device_steps(&self.copy.device());
        for m in [&mut self.attn, &mut self.short].into_iter().flatten() {
            m.refresh_steps();
        }
    }

    fn health(&self) -> String {
        let parts = [("attention", &self.attn), ("short branch", &self.short)];
        let lines: Vec<String> = parts.iter().filter_map(|(name, m)| m.as_ref().map(|m| format!("{name}: {}", m.health()))).collect();
        lines.join("\n")
    }
}
