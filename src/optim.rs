//! Integer update rules. Each takes a weight group's Q16 shadow and its gradient and returns the
//! new shadow; all state is integer, so the decoder replays it exactly. Add a rule by adding a
//! variant to [`Rule`] and an arm to [`Optimizer::step`].

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::shr_round;

/// Updates before the normalised rule's step size starts to decay.
const DECAY_START: u64 = 256;
/// Steps per window of the cost feedback in [`Rule::Auto`].
const AUTO_WINDOW: u64 = 32;
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
}

impl Optimizer {
    pub fn new(rule: Rule, lr: u32, groups: usize) -> Self {
        Self { rule, lr: lr as i32, scale: vec![None; groups], updates: 0, lr_now: lr as i32, floor: lr as i32, window: 0, previous: 0, filled: 0, calm: 0 }
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
        (self.previous, self.window, self.filled) = (self.window, 0, 0);
    }

    /// The rate [`Rule::Auto`] has settled on.
    pub fn rate(&self) -> i32 {
        self.lr_now
    }

    /// Call once per training step, before the groups are stepped.
    pub fn tick(&mut self) {
        self.updates += 1;
    }

    /// `sgd_offset` is the log2 of how much smaller this group's gradient scale is than the
    /// head's; only [`Rule::Sgd`] needs it.
    pub fn step(&mut self, group: usize, w: &Tensor<2, Int>, d: Tensor<2, Int>, sgd_offset: i32, max: i32) -> Tensor<2, Int> {
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
        (w.clone() - delta).clamp(-max, max)
    }
}
