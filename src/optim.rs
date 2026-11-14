//! Integer update rules. Each takes a weight group's Q16 shadow and its gradient and returns the
//! new shadow; all state is integer, so the decoder replays it exactly. Add a rule by adding a
//! variant to [`Rule`] and an arm to [`Optimizer::step`].

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::shr_round;

/// Updates before the normalised rule's step size starts to decay.
const DECAY_START: u64 = 256;
/// Steps per window of the cost feedback in [`Rule::Auto`].
const AUTO_WINDOW: u64 = 32;
/// Windows without a blow-up before the auto rule's floor relaxes by one step.
const FLOOR_RELAX: u64 = 16;
/// Temporary A/B switch while the relaxing floor is being evaluated.
fn relax() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GMINT_RELAX").is_some())
}
/// Temporary A/B switch: weight decay of 2^-k per step.
fn decay() -> Option<u32> {
    static K: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *K.get_or_init(|| std::env::var("GMINT_DECAY").ok().and_then(|v| v.parse::<u32>().ok()).filter(|k| *k > 0))
}
/// Slowest rate any rule may reach.
const LR_MAX: i32 = 22;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// `w -= g >> lr`, with a fixed scale correction per group. Best ratio when `lr` is tuned to
    /// the file; diverges and expands the file when it is too fast.
    Sgd,
    /// Each group's gradient is divided by a running average of its own largest magnitude, so the
    /// biggest move starts near `2^(24 - lr)` whatever the data's scale, then decays as
    /// 1/sqrt(updates). One `lr` works across files.
    Norm,
    /// [`Rule::Sgd`] whose rate is steered by the code length the model is producing. Every
    /// [`AUTO_WINDOW`] steps the window's cost is compared with the previous one: clearly worse,
    /// and the step halves; four windows without that and it doubles again. A window that costs
    /// more than storing the bytes raw quarters the step and bars that rate for good. `lr` is
    /// therefore the fastest rate, and the file decides how much of it is used.
    Auto,
}

impl Rule {
    pub fn id(self) -> u8 {
        self as u8
    }
    pub fn from_id(id: u8) -> Option<Self> {
        [Self::Sgd, Self::Norm, Self::Auto].into_iter().find(|r| r.id() == id)
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sgd" => Some(Self::Sgd),
            "norm" => Some(Self::Norm),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

pub struct Optimizer {
    rule: Rule,
    lr: i32,
    /// Per-group gradient scale for [`Rule::Norm`], filled on first use.
    scale: Vec<Option<Tensor<1, Int>>>,
    updates: u64,
    /// [`Rule::Auto`]: the rate in use, the cost of the window being filled and of the one
    /// before it, steps in the current window, and windows since the last slowdown.
    lr_now: i32,
    /// Fastest rate [`Rule::Auto`] may return to; rises when a rate is seen to diverge.
    floor: i32,
    window: u64,
    previous: u64,
    filled: u64,
    calm: u64,
    /// Windows since the last one that cost more than raw.
    quiet: u64,
    /// Windows that cost more than raw, ever: each one raised the floor.
    blowups: u64,
    /// Per group, the log2 offset of its gradient scale (see [`Self::step`]), as last given.
    offsets: Vec<i32>,
    /// The device copy of every group's plain SGD step, `[2 groups]` as (shift, up), and the
    /// values last written to it. Its buffer never moves (see [`Self::device_steps`]).
    steps: Option<(Vec<i32>, Tensor<1, Int>)>,
}

impl Optimizer {
    pub fn new(rule: Rule, lr: u32, groups: usize) -> Self {
        Self {
            rule,
            lr: lr as i32,
            scale: vec![None; groups],
            updates: 0,
            lr_now: lr as i32,
            floor: lr as i32,
            window: 0,
            previous: 0,
            filled: 0,
            calm: 0,
            quiet: 0,
            blowups: 0,
            offsets: vec![0; groups],
            steps: None,
        }
    }

    /// Sets every group's gradient-scale offset up front, as [`Self::step`] would learn them.
    pub fn with_offsets(mut self, offsets: &[i32]) -> Self {
        self.offsets = offsets.to_vec();
        self
    }

    /// Every group's current plain SGD step on the device, `[2 groups]`: (shift, up) per group,
    /// for the fused kernels to read. `None` when the rule is more than plain SGD. The tensor's
    /// buffer is made once and rewritten in place when the rate moves, so a captured graph that
    /// reads it follows the rate; the write happens here, on the next step after the change.
    pub fn device_steps(&mut self, device: &Device) -> Option<Tensor<1, Int>> {
        let mut values = Vec::with_capacity(2 * self.offsets.len());
        for offset in &self.offsets {
            let (shift, up) = self.sgd(*offset)?;
            values.extend([shift as i32, up]);
        }
        match &mut self.steps {
            Some((old, _)) if *old == values => {}
            Some((old, t)) => {
                crate::fused::write_ints(t, values.clone());
                *old = values;
            }
            None => {
                let n = values.len();
                self.steps = Some((values.clone(), Tensor::from_data(TensorData::new(values, [n]), device)));
            }
        }
        self.steps.as_ref().map(|s| s.1.clone())
    }

    /// Number of values [`Self::state`] returns for `groups` weight groups.
    pub fn state_len(groups: usize) -> usize {
        groups + 2
    }

    /// The state worth carrying into later training: [`Rule::Norm`]'s gradient scale per group (0
    /// if unset) and the update count as two 31-bit halves. [`Rule::Auto`]'s rate is left out on
    /// purpose: a rate that collapsed on the training set would stop the model adapting to the
    /// file it is then used on.
    pub fn state(&self) -> Vec<i32> {
        let mut out: Vec<i32> = self
            .scale
            .iter()
            .map(|s| match s {
                Some(t) => t.clone().into_data().convert::<i32>().try_to_vec::<i32>().unwrap()[0],
                None => 0,
            })
            .collect();
        out.extend([(self.updates & 0x7FFF_FFFF) as i32, (self.updates >> 31) as i32]);
        out
    }

    pub fn restore(&mut self, state: &[i32], device: &Device) {
        let groups = self.scale.len();
        for (slot, v) in self.scale.iter_mut().zip(state) {
            *slot = (*v > 0).then(|| Tensor::from_data(TensorData::new(vec![*v], [1]), device));
        }
        self.updates = state[groups] as u64 | (state[groups + 1] as u64) << 31;
    }

    /// Feeds back the Q8 code length of the step just coded, and what storing it raw would cost.
    pub fn observe(&mut self, cost_q8: u64, raw_q8: u64) {
        if self.rule != Rule::Auto {
            return;
        }
        self.window += cost_q8;
        self.filled += 1;
        if self.filled < AUTO_WINDOW {
            return;
        }
        let worse = self.previous > 0 && self.window > self.previous + self.previous / 32;
        let raw = raw_q8 * AUTO_WINDOW;
        if self.window > raw + raw / 16 {
            // Clearly costlier than not compressing at all: the rate is past what this data
            // tolerates. Back off hard and never return to it. The margin keeps incompressible
            // input, which sits just above raw however well the model does, from tripping this.
            self.lr_now = (self.lr_now + 2).min(LR_MAX);
            self.floor = self.lr_now - 1;
            self.calm = 0;
            self.quiet = 0;
            self.blowups += 1;
        } else if worse {
            self.lr_now = (self.lr_now + 1).min(LR_MAX);
            self.calm = 0;
        } else {
            self.calm += 1;
            if self.calm >= 4 && self.lr_now > self.floor {
                self.lr_now -= 1;
                self.calm = 0;
            }
        }
        // The bar on a fast rate lifts again, one step per FLOOR_RELAX windows without a blow-up:
        // a jump to different data looks like a blow-up, and without this every such jump
        // slowed the model for good.
        self.quiet += 1;
        if relax() && self.quiet >= FLOOR_RELAX && self.floor > self.lr {
            self.floor -= 1;
            self.quiet = 0;
        }
        (self.previous, self.window, self.filled) = (self.window, 0, 0);
    }

    /// The rate [`Rule::Auto`] has settled on.
    pub fn rate(&self) -> i32 {
        self.lr_now
    }

    /// [`Rule::Auto`]'s current floor (the fastest rate it may return to) and how many windows
    /// have cost more than raw so far, for the health line.
    pub fn floor(&self) -> (i32, u64) {
        (self.floor, self.blowups)
    }

    /// Call once per training step, before the groups are stepped.
    pub fn tick(&mut self) {
        self.updates += 1;
    }

    /// `sgd_offset` is the log2 of how much smaller this group's gradient scale is than the
    /// head's; only [`Rule::Sgd`] needs it.
    /// The step this rule takes now on a group, when it is plain SGD: the gradient is divided by
    /// `2^shift` to nearest and multiplied by `up`. `None` when the rule needs more than that, so a
    /// kernel can only fuse the step in when this is `Some`.
    pub fn sgd(&self, sgd_offset: i32) -> Option<(u32, i32)> {
        if self.rule == Rule::Norm || decay().is_some() {
            return None;
        }
        Some(match self.lr_now - sgd_offset {
            s @ 1.. => (s as u32, 1),
            // only the embedding groups get here, and their gradients are below 2^20
            s => (0, 1i32 << (-s).min(8)),
        })
    }

    pub fn step(&mut self, group: usize, w: &Tensor<2, Int>, d: Tensor<2, Int>, sgd_offset: i32, max: i32) -> Tensor<2, Int> {
        let decay = decay();
        if crate::fused::on_gpu(w) {
            self.offsets[group] = sgd_offset;
            if let Some(steps) = self.device_steps(&w.device()) {
                // the same arithmetic as below, in one kernel
                return crate::fused::sgd_step(w, d, (&steps, group), max);
            }
        }
        let delta = match self.rule {
            Rule::Sgd | Rule::Auto => match self.lr_now - sgd_offset {
                s @ 1.. => shr_round(d, s as u32),
                // only the embedding groups get here, and their gradients are below 2^20
                s => d * (1i32 << (-s).min(8)),
            },
            Rule::Norm => {
                let peak = d.clone().abs().max();
                // exponential average over 8 steps; values are non-negative, so raw shifts are safe
                let scale = match self.scale[group].take() {
                    None => peak,
                    Some(s) => s.clone() - s.bitwise_right_shift_scalar(3) + peak.bitwise_right_shift_scalar(3),
                };
                self.scale[group] = Some(scale.clone());
                // the step halves each time the update count quadruples
                let decay = (self.updates.max(DECAY_START) / DECAY_START).ilog2() as i32 / 2;
                let step_log2 = (24 - self.lr - decay).max(0);
                d / (scale.bitwise_right_shift_scalar(step_log2) + 1).unsqueeze::<2>()
            }
        };
        // temporary A/B switch: shrink every weight by 2^-decay per step to keep it off its clamp
        let w = match decay {
            Some(k) => w.clone() - shr_round(w.clone(), k),
            None => w.clone(),
        };
        (w - delta).clamp(-max, max)
    }
}
