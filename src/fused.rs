//! Hand-written fused kernels, as burn backend extensions. Every tensor op costs a few
//! microseconds to issue whatever it does, and a coding step issued hundreds of them; each op
//! here does a whole layer's worth of work in one launch.
//!
//! Every op has two implementations: a CubeCL kernel for the GPU, and a composition of burn's
//! ordinary integer ops for the CPU backend. They must agree bit for bit, since a file encoded
//! on one is decoded on the other; `gmint probe` checks that they do.

use burn::backend::ops::IntTensorOps;
use burn::backend::tensor::IntTensor;
use burn::backend::{Backend, Dispatch, Flex, backend_extension};
use burn::tensor::{Int, Shape, Tensor};
use burn_cubecl::CubeBackend;

const BIAS_LOG2: u32 = 30;

#[backend_extension(Cube, Flex)]
pub trait FusedOps: Backend {
    /// `clamp(floor((x @ W) / 2^shift), -bound, bound)` where `W` is `floor(w / 2^wshift)`,
    /// transposed first when `wt`. `x` is `[n, k]`; `w` is `[k, m]`, or `[m, k]` when `wt`.
    /// Every product sum must stay inside +-2^30.
    fn matmul_shr_clamp(x: IntTensor<Self>, w: IntTensor<Self>, wt: bool, wshift: u32, shift: u32, bound: i32) -> IntTensor<Self>;

    /// The embedding table's gradient: `[vocab, e]`, row `v` the sum of the rows of `dx` (`[n, e]`)
    /// whose entry in `idx` (`[n]`) is `v`, divided by 2^shift rounded to nearest.
    fn rows_by_value(idx: IntTensor<Self>, dx: IntTensor<Self>, vocab: usize, shift: u32) -> IntTensor<Self>;
}

/// Ops whose CPU side is the ordinary tensor code in the wrappers at the end of this file.
#[backend_extension(Cube)]
pub trait GpuOps: Backend {
    /// Attention weights `[b, t, t]` (Q12) from queries and keys `[b, t, e]`.
    fn attn_probs(q: IntTensor<Self>, k: IntTensor<Self>, lut: IntTensor<Self>, span: i32) -> IntTensor<Self>;

    /// Error at the attention scores `[b, t, t]` from the error at the mixed values `dz`.
    fn attn_ds(dz: IntTensor<Self>, v: IntTensor<Self>, a: IntTensor<Self>, err: i32) -> IntTensor<Self>;

    /// Per stream, `clamp(floor((M @ n) / 2^shift), -bound, bound)` with `m: [b, t, t]` (used
    /// transposed when `mt`) and `n: [b, t, e]`.
    fn small_matmul(m: IntTensor<Self>, n: IntTensor<Self>, mt: bool, shift: u32, bound: i32) -> IntTensor<Self>;

    /// [`GpuOps::small_matmul`], then 0 wherever `|gate| >= act_max`; `gate` is `[b, t, e]`.
    fn small_matmul_gated(m: IntTensor<Self>, n: IntTensor<Self>, gate: IntTensor<Self>, mt: bool, shift: u32, bound: i32, act_max: i32) -> IntTensor<Self>;

    /// [`FusedOps::matmul_shr_clamp`], then 0 wherever `|gate| >= act_max`; `gate` is shaped
    /// like the result.
    #[allow(clippy::too_many_arguments)]
    fn matmul_gated(x: IntTensor<Self>, w: IntTensor<Self>, gate: IntTensor<Self>, wt: bool, wshift: u32, shift: u32, bound: i32, act_max: i32) -> IntTensor<Self>;

    /// `[n / chunk, a, b]`: for every run of `chunk` rows of `x: [n, a]` and `y: [n, b]`, their
    /// `x^T @ y` floor-divided by 2^cshift.
    fn chunked_xty(x: IntTensor<Self>, y: IntTensor<Self>, chunk: usize, cshift: u32) -> IntTensor<Self>;

    /// `clamp(w - round(d / 2^shift) * up, -max, max)`: one weight group's SGD step, written into
    /// `w` itself (see [`xty_sgd`](GpuOps::xty_sgd)). `(shift, up)` is read on the device from
    /// `rates[2 slot ..]`, the optimizer's copy, so a captured step follows the rate as it moves.
    fn sgd_step(w: IntTensor<Self>, d: IntTensor<Self>, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self>;

    /// `[n, e]`: row `r` is `floor(emb[idx[r]] / 2^8) + floor(pos[r mod t] / 2^8)`.
    fn embed(emb: IntTensor<Self>, pos: IntTensor<Self>, idx: IntTensor<Self>) -> IntTensor<Self>;

    /// `[b, 3]` from Q20 byte weights `[b, vocab]` and the true bytes: in the coder's frequencies
    /// (`weight >> 4`, plus 1), the cumulative frequency below the byte, its own, and the total.
    fn coded(e: IntTensor<Self>, target: IntTensor<Self>) -> IntTensor<Self>;

    /// `prob - hot`, where `hot` is `scale` at each row's true byte and 0 elsewhere.
    fn onehot_error(prob: IntTensor<Self>, target: IntTensor<Self>, scale: i32) -> IntTensor<Self>;

    /// Tree head: node logits `[b, 255]` (Q8) to Q12 probabilities of each node's 1-branch.
    fn tree_probs(z: IntTensor<Self>, sigmoid: IntTensor<Self>, span: i32) -> IntTensor<Self>;

    /// Tree head: node probabilities to Q20 byte weights `[b, 256]`, with the match boost when
    /// `matcher`. `leaf` is each byte's leaf position; `guess` is `[b, 2]` (byte, trust level).
    fn tree_leaves(
        p: IntTensor<Self>,
        leaf: IntTensor<Self>,
        guess: IntTensor<Self>,
        copy: IntTensor<Self>,
        lut: IntTensor<Self>,
        matcher: bool,
        span: i32,
    ) -> IntTensor<Self>;

    /// Tree head: the error at the node logits, `p - bit` at the 8 nodes on the true byte's path.
    fn tree_error(p: IntTensor<Self>, target: IntTensor<Self>, leaf: IntTensor<Self>) -> IntTensor<Self>;

    /// `[b]`: each row's probability of its proposed byte, minus 4096 where the proposal was
    /// right. `weights` are Q12 probabilities, or Q20 weights to normalise first when `normalise`.
    fn guess_error(weights: IntTensor<Self>, target: IntTensor<Self>, guess: IntTensor<Self>, normalise: bool) -> IntTensor<Self>;

    /// `[3, 1]`: `values` summed over the rows at each trust level, divided by 2^shift to nearest.
    fn level_sums(values: IntTensor<Self>, guess: IntTensor<Self>, shift: u32) -> IntTensor<Self>;

    /// `clamp(dh, -err, err)` where the unit was neither off nor saturated, 0 elsewhere.
    fn relu_error(dh: IntTensor<Self>, h: IntTensor<Self>, act_max: i32, err: i32) -> IntTensor<Self>;

    /// `[t, e]`: `dx` (`[b * t, e]`) summed over the streams at each context place, divided by
    /// 2^shift to nearest.
    fn place_sums(dx: IntTensor<Self>, t: usize, shift: u32) -> IntTensor<Self>;

    /// Tree head for a mix: node probabilities `[b, 255]` to each byte's Q8 log-probability,
    /// `[b, 256]`, the sum of `logp` over the 8 branches on its path.
    fn tree_log_scores(p: IntTensor<Self>, leaf: IntTensor<Self>, logp: IntTensor<Self>) -> IntTensor<Self>;

    /// A mix's shared softmax: summed Q8 scores `[b, 256]`, plus the match boost when `matcher`,
    /// to Q20 byte weights.
    fn mix_weights(total: IntTensor<Self>, guess: IntTensor<Self>, copy: IntTensor<Self>, lut: IntTensor<Self>, matcher: bool, span: i32) -> IntTensor<Self>;

    /// A mix's error from its Q20 weights: the Q12 probabilities, less their row sum at the
    /// true byte.
    fn mix_error(e: IntTensor<Self>, target: IntTensor<Self>) -> IntTensor<Self>;

    /// `[3, 1]`: each row's value of `g` at its proposed byte, summed per trust level, divided by
    /// 2^shift to nearest.
    fn guess_sums(g: IntTensor<Self>, guess: IntTensor<Self>, shift: u32) -> IntTensor<Self>;

    /// Per-byte errors `[b, 256]` back to the node logits of a tree, `[b, 255]`; `byte_at` is
    /// the byte at each leaf position.
    fn tree_back(g: IntTensor<Self>, prob: IntTensor<Self>, byte_at: IntTensor<Self>) -> IntTensor<Self>;

    /// A weight group's new value from its gradient: [`chunked_xty`] summed over the chunks,
    /// then the SGD step of [`GpuOps::sgd_step`]. The step is written into `w` itself, so its
    /// buffer stays put from step to step (which a replayed CUDA graph needs); every other
    /// handle on the same weights sees the new values.
    fn xty_sgd(x: IntTensor<Self>, y: IntTensor<Self>, w: IntTensor<Self>, chunk: usize, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self>;

    /// The embedding table's (or with `by_place` the position table's) new value: its gradient
    /// from `dx` as [`FusedOps::rows_by_value`] (or [`GpuOps::place_sums`]) forms it, rounded by
    /// `rshift`, then the SGD step. Needs the table's width to divide 256.
    fn embed_sgd(idx: IntTensor<Self>, dx: IntTensor<Self>, w: IntTensor<Self>, t: usize, by_place: bool, rshift: u32, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self>;

    /// The attention block's forward pass, whole streams at a time: from the last `t` bytes of
    /// each row of `ctx` and the Q16 shadows, `(idx [b t], x, q, k, v [b t, e], a [b t, t],
    /// zf [b, t e])`, each as the separate ops would have made it. See [`attention_fits`].
    #[allow(clippy::type_complexity)]
    fn attn_forward(
        ctx: IntTensor<Self>,
        emb: IntTensor<Self>,
        pos: IntTensor<Self>,
        wq: IntTensor<Self>,
        wk: IntTensor<Self>,
        wv: IntTensor<Self>,
        lut: IntTensor<Self>,
        t: usize,
        span: i32,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>);

    /// Overwrites `t`, which must be contiguous, with `values`, in stream order. The buffer stays
    /// where it is, so a replayed graph that reads it reads the new values.
    fn write_ints(t: IntTensor<Self>, values: Vec<i32>) -> IntTensor<Self>;

    /// [`GpuOps::xty_sgd`] for the three attention projections at once: `x^T dq`, `x^T dk`,
    /// `x^T dv` stepping `wq`, `wk`, `wv` in place with rate slots `slot`, `slot + 1`,
    /// `slot + 2`. Needs a width of at most 32 that 4 divides.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn qkv_sgd(
        x: IntTensor<Self>,
        dq: IntTensor<Self>,
        dk: IntTensor<Self>,
        dv: IntTensor<Self>,
        wq: IntTensor<Self>,
        wk: IntTensor<Self>,
        wv: IntTensor<Self>,
        chunk: usize,
        rates: IntTensor<Self>,
        slot: usize,
        max: i32,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>);

    /// [`GpuOps::embed_sgd`] for the byte table and the position table together, both stepped
    /// in place; see [`embed_pos_fits`].
    #[allow(clippy::too_many_arguments)]
    fn embed_pos_sgd(idx: IntTensor<Self>, dx: IntTensor<Self>, emb: IntTensor<Self>, pos: IntTensor<Self>, rshift: u32, rates: IntTensor<Self>, slots: [usize; 2], max: i32) -> (IntTensor<Self>, IntTensor<Self>);

    /// CUDA graph control on the stream of `probe`'s device; see [`graph`]. Returns `probe`.
    fn graph_ctl(probe: IntTensor<Self>, cmd: u32) -> IntTensor<Self>;
}

/// Floor division by 2^s on the CPU backend, the same way [`crate::fixed::shr_floor`] does it.
fn flex_shr_floor(t: IntTensor<Flex>, s: u32) -> IntTensor<Flex> {
    if s == 0 {
        return t;
    }
    let t = Flex::int_add_scalar(t, (1i32 << BIAS_LOG2).into());
    let t = Flex::bitwise_right_shift_scalar(t, (s as i32).into());
    Flex::int_sub_scalar(t, (1i32 << (BIAS_LOG2 - s)).into())
}

impl FusedOps for Flex {
    fn matmul_shr_clamp(x: IntTensor<Self>, w: IntTensor<Self>, wt: bool, wshift: u32, shift: u32, bound: i32) -> IntTensor<Self> {
        let w = flex_shr_floor(w, wshift);
        let w = if wt { Self::int_swap_dims(w, 0, 1) } else { w };
        let p = flex_shr_floor(Self::int_matmul(x, w), shift);
        Self::int_clamp(p, (-bound).into(), bound.into())
    }

    fn rows_by_value(idx: IntTensor<Self>, dx: IntTensor<Self>, vocab: usize, shift: u32) -> IntTensor<Self> {
        use burn::backend::TensorMetadata;
        let e = dx.shape()[1];
        let zeros = Self::int_zeros(Shape::from([vocab, e]), &Default::default(), burn::tensor::IntDType::I32);
        let sums = Self::int_select_assign(zeros, 0, idx, dx, burn::tensor::ops::IndexingUpdateOp::Add);
        flex_round_shift(sums, shift)
    }
}

/// Division by 2^s rounded to nearest on the CPU backend.
fn flex_round_shift(t: IntTensor<Flex>, s: u32) -> IntTensor<Flex> {
    if s == 0 {
        return t;
    }
    flex_shr_floor(Flex::int_add_scalar(t, (1i32 << (s - 1)).into()), s)
}

mod kernel {
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::cubecl::ir::types::MatrixIdent;
    use burn_cubecl::cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};

    /// One thread per output element. Inputs and output go through linear views, as burn's own
    /// kernels do: indexing the raw tensors by position read the wrong elements past row 0.
    ///
    /// Shifts are floor divisions: the value is biased non-negative first, so the shift means the
    /// same whether the target shifts signed values arithmetically or not.
    ///
    /// When `gated`, the result is 0 wherever `|gate| >= act_max` (`gate` is shaped like the
    /// result), as [`masked`] does it; otherwise `gate` is not read.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn matmul_shr_clamp(
        x: LinearView<'_, i32>,
        w: LinearView<'_, i32>,
        gate: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        k: usize,
        m: usize,
        wshift: i32,
        shift: i32,
        bound: i32,
        act_max: i32,
        #[comptime] gated: bool,
    ) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / m;
        let col = ABSOLUTE_POS - row * m;
        let w_bias = 1073741824i32 >> wshift;
        let mut acc = 0i32;
        for i in 0..k {
            let weight = ((w.read(i * m + col) + 1073741824i32) >> wshift) - w_bias;
            acc += x.read(row * k + i) * weight;
        }
        let value = ((acc + 1073741824i32) >> shift) - (1073741824i32 >> shift);
        let low = 0i32 - bound;
        let mut value = clamp(value, low, bound);
        if gated {
            value = masked(value, gate.read(ABSOLUTE_POS), act_max);
        }
        out.write(ABSOLUTE_POS, value);
    }

    /// `value`, or 0 where the forward clamp to +-`act_max` held on `gate`: the clamp's own
    /// derivative, so no error flows back through a value the clamp cut.
    #[cube]
    fn masked(value: i32, gate: i32, act_max: i32) -> i32 {
        let mut out = value;
        if gate >= act_max || gate <= 0i32 - act_max {
            out = 0i32;
        }
        out
    }

    /// Splits four values into their high and low bytes (`v = 128 hi + lo`, `hi` in
    /// -128..=127, `lo` in 0..=127 for `|v| < 2^14`), each packed into a word with the first
    /// value in the lowest byte, as the tensor cores read four consecutive k.
    #[cube]
    fn split4(v0: i32, v1: i32, v2: i32, v3: i32) -> (i32, i32) {
        let h0 = ((v0 + 16384i32) >> 7) - 128i32;
        let h1 = ((v1 + 16384i32) >> 7) - 128i32;
        let h2 = ((v2 + 16384i32) >> 7) - 128i32;
        let h3 = ((v3 + 16384i32) >> 7) - 128i32;
        let hi = (h0 & 255i32) | ((h1 & 255i32) << 8) | ((h2 & 255i32) << 16) | (h3 << 24);
        let lo = (v0 - h0 * 128i32) | ((v1 - h1 * 128i32) << 8) | ((v2 - h2 * 128i32) << 16) | ((v3 - h3 * 128i32) << 24);
        (hi, lo)
    }

    /// [`matmul_shr_clamp`] on the tensor cores. Every operand is split into two bytes,
    /// `v = 128 hi + lo` (see [`split4`]), so `x W` is
    /// `16384 (xh Wh) + 128 (xh Wl + xl Wh) + (xl Wl)`: four exact int8 products summed in i32.
    /// With `|x|, |W| < 2^14` and `k <= 256` no partial sum comes near 2^31, so the total is the
    /// same integer the one-thread-per-output kernel forms.
    ///
    /// A cube of `wr x wc` warps owns `16 wr` rows by `16 wc` columns, each warp 16 by 16 (two
    /// `m16n8k32` tiles). `ks`-deep slices of both operands are staged in shared memory as words
    /// of four k-consecutive bytes, the weight column-major, so every fragment register is one
    /// word; rows are `ks / 4 + 4` words apart, which puts a fragment's 32 words in 32 banks.
    /// Each operand's next slice is loaded into registers while the current one is multiplied.
    /// `gate`, `act_max` and `gated` as in [`matmul_shr_clamp`].
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn matmul_mma(
        x: &Tensor<i32>,
        w: &Tensor<i32>,
        gate: &Tensor<i32>,
        out: &mut Tensor<i32>,
        n: usize,
        k: usize,
        m: usize,
        wshift: i32,
        shift: i32,
        low: i32,
        high: i32,
        act_max: i32,
        #[comptime] wr: usize,
        #[comptime] wc: usize,
        #[comptime] ks: usize,
        #[comptime] gated: bool,
    ) {
        let def = cmma::MmaDefinition::<i8, i8, i32>::new(16usize, 8usize, 32usize);
        let units = comptime!(32 * wr * wc);
        let bm = comptime!(16 * wr);
        let bn = comptime!(16 * wc);
        let kw = comptime!(ks / 4);
        let pitch = comptime!(ks / 4 + 4);
        // words of x and of the weight each unit stages per slice
        let xper = comptime!((16 * wr * ks / 4).div_ceil(32 * wr * wc));
        let wper = comptime!((16 * wc * ks / 4).div_ceil(32 * wr * wc));
        let lane = UNIT_POS_PLANE;
        let tid = UNIT_POS_X as usize;
        let warp = tid / 32;
        let my_r = warp % wr;
        let my_c = warp / wr;
        let row0 = CUBE_POS_X as usize * bm;
        let col0 = CUBE_POS_Y as usize * bn;
        let (sx0, sx1) = (x.stride(0), x.stride(1));
        let (sw0, sw1) = (w.stride(0), w.stride(1));
        let along_k = sw0 == 1 && sw1 != 1;
        let w_bias = 1073741824i32 >> wshift;
        let mut xh = Shared::<[i32]>::new_slice(comptime!(16 * wr * (ks / 4 + 4)));
        let mut xl = Shared::<[i32]>::new_slice(comptime!(16 * wr * (ks / 4 + 4)));
        let mut wh = Shared::<[i32]>::new_slice(comptime!(16 * wc * (ks / 4 + 4)));
        let mut wl = Shared::<[i32]>::new_slice(comptime!(16 * wc * (ks / 4 + 4)));
        let mut xv = Array::<i32>::new(comptime!(4 * xper));
        let mut wv = Array::<i32>::new(comptime!(4 * wper));

        let count_a = def.elems_per_lane(MatrixIdent::A);
        let size_a = def.vector_size(MatrixIdent::A);
        let size!(NA) = size_a;
        let vecs_a = comptime!(count_a / size_a);
        let count_b = def.elems_per_lane(MatrixIdent::B);
        let size_b = def.vector_size(MatrixIdent::B);
        let size!(NB) = size_b;
        let vecs_b = comptime!(count_b / size_b);
        let count_c = def.elems_per_lane(MatrixIdent::Accumulator);
        let size_c = def.vector_size(MatrixIdent::Accumulator);
        let size!(NC) = size_c;
        let vecs_c = comptime!(count_c / size_c);

        let mut hh0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut md0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut ll0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut hh1 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut md1 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut ll1 = Array::<Vector<i32, NC>>::new(vecs_c);
        #[unroll]
        for i in 0..vecs_c {
            let mut zero = Vector::empty();
            #[unroll]
            for q in 0..size_c {
                zero.insert(q, 0i32);
            }
            hh0[i] = zero;
            md0[i] = zero;
            ll0[i] = zero;
            hh1[i] = zero;
            md1[i] = zero;
            ll1[i] = zero;
        }

        let mut k0 = 0usize;
        // the first slice into registers; a unit's word is (row r, word q) of x and (column c,
        // word q) of the weight
        #[unroll]
        for j in 0..xper {
            let e = tid + j * units;
            let r = e / kw;
            let q = e - r * kw;
            #[unroll]
            for b in 0..4usize {
                let kk = k0 + q * 4 + b;
                let mut value = 0i32;
                if e < bm * kw && row0 + r < n && kk < k {
                    value = x[(row0 + r) * sx0 + kk * sx1];
                }
                xv[j * 4 + b] = value;
            }
        }
        #[unroll]
        for j in 0..wper {
            let e = tid + j * units;
            // a transposed view runs along k in memory, so its units walk k fastest
            let mut c = e / kw;
            let mut q = e - c * kw;
            if !along_k {
                q = e / bn;
                c = e - q * bn;
            }
            #[unroll]
            for b in 0..4usize {
                let kk = k0 + q * 4 + b;
                let mut value = 0i32;
                if e < bn * kw && col0 + c < m && kk < k {
                    value = ((w[kk * sw0 + (col0 + c) * sw1] + 1073741824i32) >> wshift) - w_bias;
                }
                wv[j * 4 + b] = value;
            }
        }

        while k0 < k {
            #[unroll]
            for j in 0..xper {
                let e = tid + j * units;
                if e < bm * kw {
                    let r = e / kw;
                    let q = e - r * kw;
                    let (hi, lo) = split4(xv[j * 4], xv[j * 4 + 1], xv[j * 4 + 2], xv[j * 4 + 3]);
                    xh[r * pitch + q] = hi;
                    xl[r * pitch + q] = lo;
                }
            }
            #[unroll]
            for j in 0..wper {
                let e = tid + j * units;
                if e < bn * kw {
                    let mut c = e / kw;
                    let mut q = e - c * kw;
                    if !along_k {
                        q = e / bn;
                        c = e - q * bn;
                    }
                    let (hi, lo) = split4(wv[j * 4], wv[j * 4 + 1], wv[j * 4 + 2], wv[j * 4 + 3]);
                    wh[c * pitch + q] = hi;
                    wl[c * pitch + q] = lo;
                }
            }
            sync_cube();
            // the next slice on its way while this one is multiplied
            let k1 = k0 + ks;
            if k1 < k {
                #[unroll]
                for j in 0..xper {
                    let e = tid + j * units;
                    let r = e / kw;
                    let q = e - r * kw;
                    #[unroll]
                    for b in 0..4usize {
                        let kk = k1 + q * 4 + b;
                        let mut value = 0i32;
                        if e < bm * kw && row0 + r < n && kk < k {
                            value = x[(row0 + r) * sx0 + kk * sx1];
                        }
                        xv[j * 4 + b] = value;
                    }
                }
                #[unroll]
                for j in 0..wper {
                    let e = tid + j * units;
                    let mut c = e / kw;
                    let mut q = e - c * kw;
                    if !along_k {
                        q = e / bn;
                        c = e - q * bn;
                    }
                    #[unroll]
                    for b in 0..4usize {
                        let kk = k1 + q * 4 + b;
                        let mut value = 0i32;
                        if e < bn * kw && col0 + c < m && kk < k {
                            value = ((w[kk * sw0 + (col0 + c) * sw1] + 1073741824i32) >> wshift) - w_bias;
                        }
                        wv[j * 4 + b] = value;
                    }
                }
            }
            #[unroll]
            for sub in 0..comptime!(ks / 32) {
                let kb = sub * 8;
                let mut ah = Array::<Vector<i8, NA>>::new(vecs_a);
                let mut al = Array::<Vector<i8, NA>>::new(vecs_a);
                #[unroll]
                for i in 0..vecs_a {
                    let (row, col) = def.position_of_nth(lane, (i * size_a) as u32, MatrixIdent::A);
                    let at = (my_r * 16 + row as usize) * pitch + kb + col as usize / 4;
                    ah[i] = Vector::reinterpret(xh[at]);
                    al[i] = Vector::reinterpret(xl[at]);
                }
                #[unroll]
                for nt in 0..2usize {
                    let mut bh = Array::<Vector<i8, NB>>::new(vecs_b);
                    let mut bl = Array::<Vector<i8, NB>>::new(vecs_b);
                    #[unroll]
                    for i in 0..vecs_b {
                        let (row, col) = def.position_of_nth(lane, (i * size_b) as u32, MatrixIdent::B);
                        let at = (my_c * 16 + nt * 8 + col as usize) * pitch + kb + row as usize / 4;
                        bh[i] = Vector::reinterpret(wh[at]);
                        bl[i] = Vector::reinterpret(wl[at]);
                    }
                    if nt == 0 {
                        hh0 = def.execute(&ah, &bh, &hh0);
                        md0 = def.execute(&ah, &bl, &md0);
                        md0 = def.execute(&al, &bh, &md0);
                        ll0 = def.execute(&al, &bl, &ll0);
                    } else {
                        hh1 = def.execute(&ah, &bh, &hh1);
                        md1 = def.execute(&ah, &bl, &md1);
                        md1 = def.execute(&al, &bh, &md1);
                        ll1 = def.execute(&al, &bl, &ll1);
                    }
                }
            }
            sync_cube();
            k0 = k1;
        }

        let (so0, so1) = (out.stride(0), out.stride(1));
        let (sg0, sg1) = (gate.stride(0), gate.stride(1));
        #[unroll]
        for nt in 0..2usize {
            #[unroll]
            for i in 0..vecs_c {
                let mut vh = hh0[i];
                let mut vm = md0[i];
                let mut vl = ll0[i];
                if nt == 1 {
                    vh = hh1[i];
                    vm = md1[i];
                    vl = ll1[i];
                }
                #[unroll]
                for q in 0..size_c {
                    let (row, col) = def.position_of_nth(lane, (i * size_c + q) as u32, MatrixIdent::Accumulator);
                    let r = row0 + my_r * 16 + row as usize;
                    let c = col0 + my_c * 16 + nt * 8 + col as usize;
                    if r < n && c < m {
                        let total = vh.extract(q) * 16384i32 + vm.extract(q) * 128i32 + vl.extract(q);
                        let value = ((total + 1073741824i32) >> shift) - (1073741824i32 >> shift);
                        let mut value = clamp(value, low, high);
                        if gated {
                            value = masked(value, gate[r * sg0 + c * sg1], act_max);
                        }
                        out[r * so0 + c * so1] = value;
                    }
                }
            }
        }
    }

    /// [`xty_sgd`] on the tensor cores, by the byte split of [`matmul_mma`]: `x^T y` over each
    /// chunk of rows is four exact int8 products, floor-shifted as the chunk completes. A cube
    /// of 2 x 2 warps owns a 32 x 32 block of the result and a range of rows; both operands are
    /// staged 32 rows at a time as words of four row-consecutive bytes (rows of `x` and `y` are
    /// the shared dimension), loaded along their contiguous axis. The ranges meet as in
    /// [`xty_sgd`]. Needs `|x|, |y| < 2^14` and chunk sums inside 2^29.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn xty_mma(
        x: &Tensor<i32>,
        y: &Tensor<i32>,
        w: &mut Tensor<i32>,
        scratch: &mut Tensor<i32>,
        counter: &mut Tensor<Atomic<i32>>,
        a: usize,
        b: usize,
        chunk: usize,
        span: usize,
        splits: usize,
        cshift: i32,
        rates: &Tensor<i32>,
        slot: usize,
        max: i32,
    ) {
        // the group's step, from the optimizer's device copy: (shift, up)
        let shift = rates[2 * slot * rates.stride(0)];
        let up = rates[(2 * slot + 1) * rates.stride(0)];
        let def = cmma::MmaDefinition::<i8, i8, i32>::new(16usize, 8usize, 32usize);
        let lane = UNIT_POS_PLANE;
        let tid = UNIT_POS_X as usize;
        let warp = tid / 32;
        let my_r = warp % 2;
        let my_c = warp / 2;
        let a0 = CUBE_POS_X as usize * 32;
        let b0 = CUBE_POS_Y as usize * 32;
        let split = CUBE_POS_Z as usize;
        let (sx0, sx1) = (x.stride(0), x.stride(1));
        let (sy0, sy1) = (y.stride(0), y.stride(1));
        // 32 entries of the result's rows (x) and columns (y), 8 words of 4 rows each, 12 apart
        let mut xh = Shared::<[i32]>::new_slice(384usize);
        let mut xl = Shared::<[i32]>::new_slice(384usize);
        let mut yh = Shared::<[i32]>::new_slice(384usize);
        let mut yl = Shared::<[i32]>::new_slice(384usize);
        let mut xv = Array::<i32>::new(8usize);
        let mut yv = Array::<i32>::new(8usize);

        let count_a = def.elems_per_lane(MatrixIdent::A);
        let size_a = def.vector_size(MatrixIdent::A);
        let size!(NA) = size_a;
        let vecs_a = comptime!(count_a / size_a);
        let count_b = def.elems_per_lane(MatrixIdent::B);
        let size_b = def.vector_size(MatrixIdent::B);
        let size!(NB) = size_b;
        let vecs_b = comptime!(count_b / size_b);
        let count_c = def.elems_per_lane(MatrixIdent::Accumulator);
        let size_c = def.vector_size(MatrixIdent::Accumulator);
        let size!(NC) = size_c;
        let vecs_c = comptime!(count_c / size_c);

        let mut hh0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut md0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut ll0 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut hh1 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut md1 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut ll1 = Array::<Vector<i32, NC>>::new(vecs_c);
        let mut total = Array::<i32>::new(comptime!(2 * count_c));
        #[unroll]
        for i in 0..comptime!(2 * count_c) {
            total[i] = 0i32;
        }
        let mut zero = Vector::empty();
        #[unroll]
        for q in 0..size_c {
            zero.insert(q, 0i32);
        }

        let start = split * span;
        let stop = start + span;
        // a unit's two words: entry (e mod 32) of the block, rows 4 (e / 32) .. + 4
        #[unroll]
        for j in 0..2usize {
            let e = tid + j * 128;
            let q = e / 32;
            let i = e - q * 32;
            #[unroll]
            for bb in 0..4usize {
                let r = start + q * 4 + bb;
                let mut xval = 0i32;
                if a0 + i < a {
                    xval = x[r * sx0 + (a0 + i) * sx1];
                }
                let mut yval = 0i32;
                if b0 + i < b {
                    yval = y[r * sy0 + (b0 + i) * sy1];
                }
                xv[j * 4 + bb] = xval;
                yv[j * 4 + bb] = yval;
            }
        }
        let mut r0 = start;
        let mut fresh = true;
        while r0 < stop {
            if fresh {
                #[unroll]
                for i in 0..vecs_c {
                    hh0[i] = zero;
                    md0[i] = zero;
                    ll0[i] = zero;
                    hh1[i] = zero;
                    md1[i] = zero;
                    ll1[i] = zero;
                }
            }
            #[unroll]
            for j in 0..2usize {
                let e = tid + j * 128;
                let q = e / 32;
                let i = e - q * 32;
                let (hi, lo) = split4(xv[j * 4], xv[j * 4 + 1], xv[j * 4 + 2], xv[j * 4 + 3]);
                xh[i * 12 + q] = hi;
                xl[i * 12 + q] = lo;
                let (hi, lo) = split4(yv[j * 4], yv[j * 4 + 1], yv[j * 4 + 2], yv[j * 4 + 3]);
                yh[i * 12 + q] = hi;
                yl[i * 12 + q] = lo;
            }
            sync_cube();
            let r1 = r0 + 32;
            if r1 < stop {
                #[unroll]
                for j in 0..2usize {
                    let e = tid + j * 128;
                    let q = e / 32;
                    let i = e - q * 32;
                    #[unroll]
                    for bb in 0..4usize {
                        let r = r1 + q * 4 + bb;
                        let mut xval = 0i32;
                        if a0 + i < a {
                            xval = x[r * sx0 + (a0 + i) * sx1];
                        }
                        let mut yval = 0i32;
                        if b0 + i < b {
                            yval = y[r * sy0 + (b0 + i) * sy1];
                        }
                        xv[j * 4 + bb] = xval;
                        yv[j * 4 + bb] = yval;
                    }
                }
            }
            let mut ah = Array::<Vector<i8, NA>>::new(vecs_a);
            let mut al = Array::<Vector<i8, NA>>::new(vecs_a);
            #[unroll]
            for i in 0..vecs_a {
                let (row, col) = def.position_of_nth(lane, (i * size_a) as u32, MatrixIdent::A);
                let at = (my_r * 16 + row as usize) * 12 + col as usize / 4;
                ah[i] = Vector::reinterpret(xh[at]);
                al[i] = Vector::reinterpret(xl[at]);
            }
            #[unroll]
            for nt in 0..2usize {
                let mut bh = Array::<Vector<i8, NB>>::new(vecs_b);
                let mut bl = Array::<Vector<i8, NB>>::new(vecs_b);
                #[unroll]
                for i in 0..vecs_b {
                    let (row, col) = def.position_of_nth(lane, (i * size_b) as u32, MatrixIdent::B);
                    let at = (my_c * 16 + nt * 8 + col as usize) * 12 + row as usize / 4;
                    bh[i] = Vector::reinterpret(yh[at]);
                    bl[i] = Vector::reinterpret(yl[at]);
                }
                if nt == 0 {
                    hh0 = def.execute(&ah, &bh, &hh0);
                    md0 = def.execute(&ah, &bl, &md0);
                    md0 = def.execute(&al, &bh, &md0);
                    ll0 = def.execute(&al, &bl, &ll0);
                } else {
                    hh1 = def.execute(&ah, &bh, &hh1);
                    md1 = def.execute(&ah, &bl, &md1);
                    md1 = def.execute(&al, &bh, &md1);
                    ll1 = def.execute(&al, &bl, &ll1);
                }
            }
            sync_cube();
            r0 = r1;
            // a chunk complete: its sums floor-shifted into the total
            fresh = (r0 - start) % chunk == 0;
            if fresh {
                #[unroll]
                for nt in 0..2usize {
                    #[unroll]
                    for i in 0..vecs_c {
                        let mut vh = hh0[i];
                        let mut vm = md0[i];
                        let mut vl = ll0[i];
                        if nt == 1 {
                            vh = hh1[i];
                            vm = md1[i];
                            vl = ll1[i];
                        }
                        #[unroll]
                        for q in 0..size_c {
                            let sum = vh.extract(q) * 16384i32 + vm.extract(q) * 128i32 + vl.extract(q);
                            total[nt * count_c + i * size_c + q] += ((sum + 1073741824i32) >> cshift) - (1073741824i32 >> cshift);
                        }
                    }
                }
            }
        }

        let tile = CUBE_POS_X as usize * CUBE_COUNT_Y as usize + CUBE_POS_Y as usize;
        let mut last = true;
        if splits > 1 {
            #[unroll]
            for nt in 0..2usize {
                #[unroll]
                for p in 0..count_c {
                    let (row, col) = def.position_of_nth(lane, p as u32, MatrixIdent::Accumulator);
                    let i = a0 + my_r * 16 + row as usize;
                    let c = b0 + my_c * 16 + nt * 8 + col as usize;
                    if i < a && c < b {
                        scratch[(split * a + i) * b + c] = total[nt * count_c + p];
                    }
                }
            }
            let mut arrived = Shared::<[i32]>::new_slice(1usize);
            sync_storage();
            if tid == 0 {
                arrived[0] = counter[tile].fetch_add(1i32);
            }
            sync_cube();
            last = arrived[0] == i32::cast_from(splits - 1);
            if last {
                sync_storage();
                #[unroll]
                for nt in 0..2usize {
                    #[unroll]
                    for p in 0..count_c {
                        let (row, col) = def.position_of_nth(lane, p as u32, MatrixIdent::Accumulator);
                        let i = a0 + my_r * 16 + row as usize;
                        let c = b0 + my_c * 16 + nt * 8 + col as usize;
                        let mut sum = 0i32;
                        if i < a && c < b {
                            for sp in 0..splits {
                                sum += scratch[(sp * a + i) * b + c];
                            }
                        }
                        total[nt * count_c + p] = sum;
                    }
                }
                if tid == 0 {
                    counter[tile].store(0i32);
                }
            }
        }
        if last {
            let (sw0, sw1) = (w.stride(0), w.stride(1));
            let half = (1i32 << shift) >> 1;
            let low = 0i32 - max;
            #[unroll]
            for nt in 0..2usize {
                #[unroll]
                for p in 0..count_c {
                    let (row, col) = def.position_of_nth(lane, p as u32, MatrixIdent::Accumulator);
                    let i = a0 + my_r * 16 + row as usize;
                    let c = b0 + my_c * 16 + nt * 8 + col as usize;
                    if i < a && c < b {
                        let delta = (((total[nt * count_c + p] + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
                        let at = i * sw0 + c * sw1;
                        w[at] = clamp(w[at] - delta, low, max);
                    }
                }
            }
        }
    }

    /// [`matmul_shr_clamp`] in tiles: 256 threads, as 16 x 16, share a `(16 tr) x (16 tc)` block
    /// of the output, staging 16-deep slices of `x` and of the shifted weight in shared memory.
    /// Each thread owns rows `ty + 16 i` and columns `tx + 16 j` of the block, so neighbouring
    /// threads read neighbouring words. `w` is read through its strides as `[k, m]`, so a
    /// transposed weight is a view and needs no copy. Integer sums do not depend on the order
    /// they are formed in, so this gives the same bits as the one-thread-per-output kernel.
    /// `gate`, `act_max` and `gated` as in [`matmul_shr_clamp`].
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn matmul_tiled(
        x: &Tensor<i32>,
        w: &Tensor<i32>,
        gate: &Tensor<i32>,
        out: &mut Tensor<i32>,
        n: usize,
        k: usize,
        m: usize,
        wshift: i32,
        shift: i32,
        low: i32,
        high: i32,
        act_max: i32,
        #[comptime] tr: usize,
        #[comptime] tc: usize,
        #[comptime] gated: bool,
    ) {
        let bm = comptime!(16 * tr);
        let bn = comptime!(16 * tc);
        let tid = UNIT_POS_X as usize;
        let ty = tid / 16;
        let tx = tid - ty * 16;
        let row0 = CUBE_POS_X as usize * bm;
        let col0 = CUBE_POS_Y as usize * bn;
        let (sx0, sx1) = (x.stride(0), x.stride(1));
        let (sw0, sw1) = (w.stride(0), w.stride(1));
        // a transposed view walks down k in memory, so its loads go along k
        let along_k = sw0 == 1 && sw1 != 1;
        let w_bias = 1073741824i32 >> wshift;
        let mut xs = Shared::<[i32]>::new_slice(comptime!(16 * tr * 16));
        let mut ws = Shared::<[i32]>::new_slice(comptime!(16 * 16 * tc));
        let mut acc = Array::<i32>::new(comptime!(tr * tc));
        #[unroll]
        for a in 0..comptime!(tr * tc) {
            acc[a] = 0i32;
        }
        let mut k0 = 0usize;
        while k0 < k {
            #[unroll]
            for j in 0..tr {
                let e = tid + j * 256;
                let r = e / 16;
                let kk = e - r * 16;
                let mut value = 0i32;
                if row0 + r < n && k0 + kk < k {
                    value = x[(row0 + r) * sx0 + (k0 + kk) * sx1];
                }
                xs[e] = value;
            }
            #[unroll]
            for j in 0..tc {
                let e = tid + j * 256;
                let mut kk = e / bn;
                let mut c = e - kk * bn;
                if along_k {
                    c = e / 16;
                    kk = e - c * 16;
                }
                let mut value = 0i32;
                if col0 + c < m && k0 + kk < k {
                    value = ((w[(k0 + kk) * sw0 + (col0 + c) * sw1] + 1073741824i32) >> wshift) - w_bias;
                }
                ws[kk * bn + c] = value;
            }
            sync_cube();
            #[unroll]
            for kk in 0..16usize {
                let mut b = Array::<i32>::new(tc);
                #[unroll]
                for j in 0..tc {
                    b[j] = ws[kk * bn + tx + 16 * j];
                }
                #[unroll]
                for i in 0..tr {
                    let a = xs[(ty + 16 * i) * 16 + kk];
                    #[unroll]
                    for j in 0..tc {
                        acc[i * tc + j] += a * b[j];
                    }
                }
            }
            sync_cube();
            k0 += 16;
        }
        let (so0, so1) = (out.stride(0), out.stride(1));
        let (sg0, sg1) = (gate.stride(0), gate.stride(1));
        #[unroll]
        for i in 0..tr {
            let r = row0 + ty + 16 * i;
            #[unroll]
            for j in 0..tc {
                let c = col0 + tx + 16 * j;
                if r < n && c < m {
                    let value = ((acc[i * tc + j] + 1073741824i32) >> shift) - (1073741824i32 >> shift);
                    let mut value = clamp(value, low, high);
                    if gated {
                        value = masked(value, gate[r * sg0 + c * sg1], act_max);
                    }
                    out[r * so0 + c * so1] = value;
                }
            }
        }
    }

    /// A weight group's gradient and SGD step in one: `w - round(D / 2^shift) * up`, clamped to
    /// +-max, where `D` is [`chunked_xty`] summed over the chunks. Tiled like [`matmul_tiled`],
    /// with the rows of `x` and `y` as the shared dimension; each chunk's sum is floor-shifted
    /// as it completes, exactly as the separate kernels did it.
    ///
    /// The rows are split over the cubes along z, a whole number of chunks each. Each cube
    /// publishes its partial sums to `scratch` and counts itself in at its tile's `counter`;
    /// the last to arrive adds the partials, takes the step, and sets the counter back to 0 for
    /// the next launch. Integer sums do not depend on which cube arrives last.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn xty_sgd(
        x: &Tensor<i32>,
        y: &Tensor<i32>,
        w: &mut Tensor<i32>,
        scratch: &mut Tensor<i32>,
        counter: &mut Tensor<Atomic<i32>>,
        a: usize,
        b: usize,
        chunk: usize,
        span: usize,
        splits: usize,
        cshift: i32,
        rates: &Tensor<i32>,
        slot: usize,
        max: i32,
        #[comptime] tr: usize,
        #[comptime] tc: usize,
    ) {
        // the group's step, from the optimizer's device copy: (shift, up)
        let shift = rates[2 * slot * rates.stride(0)];
        let up = rates[(2 * slot + 1) * rates.stride(0)];
        let ba = comptime!(16 * tr);
        let bb = comptime!(16 * tc);
        let tid = UNIT_POS_X as usize;
        let ty = tid / 16;
        let tx = tid - ty * 16;
        let a0 = CUBE_POS_X as usize * ba;
        let b0 = CUBE_POS_Y as usize * bb;
        let split = CUBE_POS_Z as usize;
        let (sx0, sx1) = (x.stride(0), x.stride(1));
        let (sy0, sy1) = (y.stride(0), y.stride(1));
        let mut xs = Shared::<[i32]>::new_slice(comptime!(16 * 16 * tr));
        let mut ys = Shared::<[i32]>::new_slice(comptime!(16 * 16 * tc));
        let mut acc = Array::<i32>::new(comptime!(tr * tc));
        let mut total = Array::<i32>::new(comptime!(tr * tc));
        #[unroll]
        for q in 0..comptime!(tr * tc) {
            total[q] = 0i32;
        }
        let mut r0 = split * span;
        let stop = r0 + span;
        while r0 < stop {
            #[unroll]
            for q in 0..comptime!(tr * tc) {
                acc[q] = 0i32;
            }
            let end = r0 + chunk;
            while r0 < end {
                #[unroll]
                for j in 0..tr {
                    let e = tid + j * 256;
                    let rr = e / ba;
                    let i = e - rr * ba;
                    let mut value = 0i32;
                    if a0 + i < a {
                        value = x[(r0 + rr) * sx0 + (a0 + i) * sx1];
                    }
                    xs[e] = value;
                }
                #[unroll]
                for j in 0..tc {
                    let e = tid + j * 256;
                    let rr = e / bb;
                    let c = e - rr * bb;
                    let mut value = 0i32;
                    if b0 + c < b {
                        value = y[(r0 + rr) * sy0 + (b0 + c) * sy1];
                    }
                    ys[e] = value;
                }
                sync_cube();
                #[unroll]
                for rr in 0..16usize {
                    let mut yv = Array::<i32>::new(tc);
                    #[unroll]
                    for j in 0..tc {
                        yv[j] = ys[rr * bb + tx + 16 * j];
                    }
                    #[unroll]
                    for i in 0..tr {
                        let xv = xs[rr * ba + ty + 16 * i];
                        #[unroll]
                        for j in 0..tc {
                            acc[i * tc + j] += xv * yv[j];
                        }
                    }
                }
                sync_cube();
                r0 += 16;
            }
            #[unroll]
            for q in 0..comptime!(tr * tc) {
                total[q] += ((acc[q] + 1073741824i32) >> cshift) - (1073741824i32 >> cshift);
            }
        }
        let mut last = true;
        if splits > 1 {
            #[unroll]
            for i in 0..tr {
                let r = a0 + ty + 16 * i;
                #[unroll]
                for j in 0..tc {
                    let c = b0 + tx + 16 * j;
                    if r < a && c < b {
                        scratch[(split * a + r) * b + c] = total[i * tc + j];
                    }
                }
            }
            let tile = CUBE_POS_X as usize * CUBE_COUNT_Y as usize + CUBE_POS_Y as usize;
            let mut arrived = Shared::<[i32]>::new_slice(1usize);
            sync_storage();
            if tid == 0 {
                arrived[0] = counter[tile].fetch_add(1i32);
            }
            sync_cube();
            last = arrived[0] == i32::cast_from(splits - 1);
            if last {
                sync_storage();
                #[unroll]
                for i in 0..tr {
                    let r = a0 + ty + 16 * i;
                    #[unroll]
                    for j in 0..tc {
                        let c = b0 + tx + 16 * j;
                        let mut sum = 0i32;
                        if r < a && c < b {
                            for p in 0..splits {
                                sum += scratch[(p * a + r) * b + c];
                            }
                        }
                        total[i * tc + j] = sum;
                    }
                }
                if tid == 0 {
                    counter[tile].store(0i32);
                }
            }
        }
        if last {
            let (sw0, sw1) = (w.stride(0), w.stride(1));
            let half = (1i32 << shift) >> 1;
            let low = 0i32 - max;
            #[unroll]
            for i in 0..tr {
                let r = a0 + ty + 16 * i;
                #[unroll]
                for j in 0..tc {
                    let c = b0 + tx + 16 * j;
                    if r < a && c < b {
                        let delta = (((total[i * tc + j] + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
                        let at = r * sw0 + c * sw1;
                        w[at] = clamp(w[at] - delta, low, max);
                    }
                }
            }
        }
    }




    /// One thread per output element of a per-stream matmul. `gate`, `act_max` and `gated` as in
    /// [`matmul_shr_clamp`].
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn small_matmul(
        m: LinearView<'_, i32>,
        n: LinearView<'_, i32>,
        gate: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        t: usize,
        e: usize,
        mt: usize,
        shift: i32,
        bound: i32,
        act_max: i32,
        #[comptime] gated: bool,
    ) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / e;
        let c = ABSOLUTE_POS - row * e;
        let stream = row / t;
        let i = row - stream * t;
        // m is read at m_at + j * m_step: along its row, or down its column when mt is 1
        let m_at = (1 - mt) * row * t + mt * (stream * t * t + i);
        let m_step = 1 + mt * (t - 1);
        let mut acc = 0i32;
        for j in 0..t {
            acc += m.read(m_at + j * m_step) * n.read((stream * t + j) * e + c);
        }
        let value = ((acc + 1073741824i32) >> shift) - (1073741824i32 >> shift);
        let low = 0i32 - bound;
        let mut value = clamp(value, low, bound);
        if gated {
            value = masked(value, gate.read(ABSOLUTE_POS), act_max);
        }
        out.write(ABSOLUTE_POS, value);
    }

    /// Sum over a cube of 256 units of one value each, handed to every unit: per plane (warp of
    /// 32), then across the 8 planes. Integer sums do not depend on the order. Every unit of the
    /// cube must call it.
    #[cube]
    fn cube_sum(value: i32) -> i32 {
        let mut part = Shared::<[i32]>::new_slice(8usize);
        let s = plane_sum(value);
        if UNIT_POS_PLANE == 0 {
            part[UNIT_POS_X as usize / 32] = s;
        }
        sync_cube();
        let mut total = 0i32;
        #[unroll]
        for p in 0..8usize {
            total += part[p];
        }
        // the next call may reuse the shared memory
        sync_cube();
        total
    }

    /// [`cube_sum`] for the maximum.
    #[cube]
    fn cube_max(value: i32) -> i32 {
        let mut part = Shared::<[i32]>::new_slice(8usize);
        let s = plane_max(value);
        if UNIT_POS_PLANE == 0 {
            part[UNIT_POS_X as usize / 32] = s;
        }
        sync_cube();
        let mut top = part[0];
        #[unroll]
        for p in 1..8usize {
            top = max(top, part[p]);
        }
        // the next call may reuse the shared memory
        sync_cube();
        top
    }



    /// One thread per attention weight, a cube holding whole rows: each thread forms its own
    /// score, then reads its row's from shared memory for the maximum and the sum.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn attn_probs(
        q: LinearView<'_, i32>,
        k: LinearView<'_, i32>,
        lut: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        t: usize,
        e: usize,
        span: i32,
    ) {
        let tid = UNIT_POS_X as usize;
        let at = ABSOLUTE_POS;
        let row = at / t;
        let j = at - row * t;
        let stream = row / t;
        let mut acc = 0i32;
        for c in 0..e {
            acc += q.read(row * e + c) * k.read((stream * t + j) * e + c);
        }
        let mut scores = Shared::<[i32]>::new_slice(256usize);
        scores[tid] = ((acc + 1073741824i32) >> 12) - 262144i32;
        sync_cube();
        let first = tid - j;
        let mut top = scores[first];
        for jj in 1..t {
            top = max(top, scores[first + jj]);
        }
        let mut sum = 0i32;
        let mut mine = 0i32;
        for jj in 0..t {
            let weight = lut.read(usize::cast_from(clamp(top - scores[first + jj], 0i32, span)));
            sum += weight;
            if jj == j {
                mine = weight;
            }
        }
        out.write(at, mine / (sum >> 12));
    }

    /// One thread per attention score's error, a cube holding whole rows: each thread forms its
    /// own error at the weight, then reads its row's for their mean.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn attn_ds(
        dz: LinearView<'_, i32>,
        v: LinearView<'_, i32>,
        a: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        t: usize,
        e: usize,
        err: i32,
    ) {
        let tid = UNIT_POS_X as usize;
        let at = ABSOLUTE_POS;
        let row = at / t;
        let j = at - row * t;
        let stream = row / t;
        let low = 0i32 - err;
        let mut acc = 0i32;
        for c in 0..e {
            acc += dz.read(row * e + c) * v.read((stream * t + j) * e + c);
        }
        let mut errs = Shared::<[i32]>::new_slice(256usize);
        let da = clamp(((acc + 1073741824i32) >> 10) - 1048576i32, low, err);
        errs[tid] = da;
        sync_cube();
        let first = tid - j;
        let mut mean = 0i32;
        for jj in 0..t {
            mean += a.read(row * t + jj) * errs[first + jj];
        }
        let mean = ((mean + 1073741824i32) >> 12) - 262144i32;
        let ds = a.read(at) * (da - mean);
        out.write(at, clamp(((ds + 1073741824i32) >> 12) - 262144i32, low, err));
    }

    /// The embedding (or position) table's gradient and SGD step in one. One thread per element
    /// of `dx` adds it into `acc` at its table row: its byte in `idx`, or with `by_place` its
    /// context place (`r mod t`). The last cube to arrive (see [`xty_sgd`]) rounds each sum by
    /// `rshift` as [`rows_by_value`] and [`place_sums`] round it, steps it like [`sgd_step`],
    /// and sets `acc` and the counter back to 0 for the next launch.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn embed_sgd(
        idx: &Tensor<i32>,
        dx: &Tensor<i32>,
        w: &mut Tensor<i32>,
        acc: &mut Tensor<Atomic<i32>>,
        counter: &mut Tensor<Atomic<i32>>,
        n: usize,
        t: usize,
        e: usize,
        rows: usize,
        by_place: u32,
        rshift: i32,
        rates: &Tensor<i32>,
        slot: usize,
        max: i32,
    ) {
        // the group's step, from the optimizer's device copy: (shift, up)
        let shift = rates[2 * slot * rates.stride(0)];
        let up = rates[(2 * slot + 1) * rates.stride(0)];
        let tid = UNIT_POS_X as usize;
        let at = ABSOLUTE_POS;
        if at < n * e {
            let r = at / e;
            let c = at - r * e;
            let mut key = usize::cast_from(idx[r * idx.stride(0)]);
            if by_place == 1 {
                key = r - (r / t) * t;
            }
            acc[key * e + c].fetch_add(dx[r * dx.stride(0) + c * dx.stride(1)]);
        }
        let mut arrived = Shared::<[i32]>::new_slice(1usize);
        sync_storage();
        if tid == 0 {
            arrived[0] = counter[0].fetch_add(1i32);
        }
        sync_cube();
        if arrived[0] == i32::cast_from(CUBE_COUNT_X as usize - 1) {
            sync_storage();
            let rhalf = (1i32 << rshift) >> 1;
            let half = (1i32 << shift) >> 1;
            let low = 0i32 - max;
            let mut i = tid;
            while i < rows * e {
                let v = i / e;
                let c = i - v * e;
                let sum = acc[i].load();
                acc[i].store(0i32);
                let d = ((sum + rhalf + 1073741824i32) >> rshift) - (1073741824i32 >> rshift);
                let delta = (((d + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
                let at = v * w.stride(0) + c * w.stride(1);
                w[at] = clamp(w[at] - delta, low, max);
                i += 256;
            }
            if tid == 0 {
                counter[0].store(0i32);
            }
        }
    }

    /// [`GpuOps::attn_forward`]. A cube of 256 units holds `256 / (t e)` streams at a time, one
    /// unit per element of their embedded context, and walks groups of streams; the shifted
    /// projection weights sit in shared memory for all of them. Every step is the arithmetic of
    /// the op it replaces: [`embed`], [`matmul_tiled`] with a bound of `ACT_MAX`, [`attn_probs`],
    /// [`small_matmul`] with shift 12, then the clamped residual sum.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn attn_forward(
        ctx: &Tensor<i32>,
        emb: &Tensor<i32>,
        pos: &Tensor<i32>,
        wq: &Tensor<i32>,
        wk: &Tensor<i32>,
        wv: &Tensor<i32>,
        lut: &Tensor<i32>,
        idx_out: &mut Tensor<i32>,
        x_out: &mut Tensor<i32>,
        q_out: &mut Tensor<i32>,
        k_out: &mut Tensor<i32>,
        v_out: &mut Tensor<i32>,
        a_out: &mut Tensor<i32>,
        zf_out: &mut Tensor<i32>,
        streams: usize,
        off: usize,
        span: i32,
        #[comptime] t: usize,
        #[comptime] e: usize,
    ) {
        let te = comptime!(t * e);
        let tt = comptime!(t * t);
        let ee = comptime!(e * e);
        let per = comptime!(256 / (t * e));
        let tid = UNIT_POS_X as usize;
        let ls = tid / te;
        let rem = tid - ls * te;
        let i = rem / e;
        let c = rem - i * e;
        let act = 1023i32;
        let s_idx = idx_out.stride(0);
        let (sx0, sx1) = (x_out.stride(0), x_out.stride(1));
        let (sq0, sq1) = (q_out.stride(0), q_out.stride(1));
        let (sk0, sk1) = (k_out.stride(0), k_out.stride(1));
        let (sv0, sv1) = (v_out.stride(0), v_out.stride(1));
        let (sa0, sa1) = (a_out.stride(0), a_out.stride(1));
        let (sz0, sz1) = (zf_out.stride(0), zf_out.stride(1));
        let mut wsm = Shared::<[i32]>::new_slice(comptime!(3 * e * e));
        let mut u = tid;
        while u < 3 * ee {
            let which = u / ee;
            let rc = u - which * ee;
            let r = rc / e;
            let cc = rc - r * e;
            let mut raw = wv[r * wv.stride(0) + cc * wv.stride(1)];
            if which == 0 {
                raw = wq[r * wq.stride(0) + cc * wq.stride(1)];
            } else if which == 1 {
                raw = wk[r * wk.stride(0) + cc * wk.stride(1)];
            }
            wsm[u] = ((raw + 1073741824i32) >> 8) - 4194304i32;
            u += 256;
        }
        let mut xs = Shared::<[i32]>::new_slice(256usize);
        let mut qs = Shared::<[i32]>::new_slice(256usize);
        let mut ks = Shared::<[i32]>::new_slice(256usize);
        let mut vs = Shared::<[i32]>::new_slice(256usize);
        let mut att = Shared::<[i32]>::new_slice(256usize);
        sync_cube();
        // the units that hold an attention weight: stream ls2, query i2, key j2
        let ls2 = tid / tt;
        let r2 = tid - ls2 * tt;
        let i2 = r2 / t;
        let j2 = r2 - i2 * t;
        let weighs = tid < per * tt;
        let mut grp = CUBE_POS_X as usize;
        while grp * per < streams {
            let s = grp * per + ls;
            let row = s * t + i;
            let byte = ctx[s * ctx.stride(0) + (off + i) * ctx.stride(1)];
            if c == 0 {
                idx_out[row * s_idx] = byte;
            }
            let bi = usize::cast_from(byte);
            let xv = (((emb[bi * emb.stride(0) + c * emb.stride(1)] + 1073741824i32) >> 8) - 4194304i32)
                + (((pos[i * pos.stride(0) + c * pos.stride(1)] + 1073741824i32) >> 8) - 4194304i32);
            xs[tid] = xv;
            x_out[row * sx0 + c * sx1] = xv;
            sync_cube();
            let base = ls * te + i * e;
            let mut aq = 0i32;
            let mut ak = 0i32;
            let mut av = 0i32;
            #[unroll]
            for ii in 0..e {
                let xi = xs[base + ii];
                aq += xi * wsm[ii * e + c];
                ak += xi * wsm[ee + ii * e + c];
                av += xi * wsm[2 * ee + ii * e + c];
            }
            let qv = clamp(((aq + 1073741824i32) >> 8) - 4194304i32, 0i32 - act, act);
            let kv = clamp(((ak + 1073741824i32) >> 8) - 4194304i32, 0i32 - act, act);
            let vv = clamp(((av + 1073741824i32) >> 8) - 4194304i32, 0i32 - act, act);
            qs[tid] = qv;
            ks[tid] = kv;
            vs[tid] = vv;
            q_out[row * sq0 + c * sq1] = qv;
            k_out[row * sk0 + c * sk1] = kv;
            v_out[row * sv0 + c * sv1] = vv;
            sync_cube();
            if weighs {
                let mut acc = 0i32;
                #[unroll]
                for cc in 0..e {
                    acc += qs[ls2 * te + i2 * e + cc] * ks[ls2 * te + j2 * e + cc];
                }
                att[tid] = ((acc + 1073741824i32) >> 12) - 262144i32;
            }
            sync_cube();
            let mut weight = 0i32;
            if weighs {
                let first = tid - j2;
                let mut top = att[first];
                #[unroll]
                for jj in 1..t {
                    top = max(top, att[first + jj]);
                }
                let mut sum = 0i32;
                #[unroll]
                for jj in 0..t {
                    let w = lut[usize::cast_from(clamp(top - att[first + jj], 0i32, span)) * lut.stride(0)];
                    sum += w;
                    if jj == j2 {
                        weight = w;
                    }
                }
                weight = weight / (sum >> 12);
            }
            sync_cube();
            if weighs {
                att[tid] = weight;
                let s2 = grp * per + ls2;
                a_out[(s2 * t + i2) * sa0 + j2 * sa1] = weight;
            }
            sync_cube();
            let mut ay = 0i32;
            #[unroll]
            for j in 0..t {
                ay += att[ls * tt + i * t + j] * vs[ls * te + j * e + c];
            }
            let y = clamp(((ay + 1073741824i32) >> 12) - 262144i32, -1073741824i32, 1073741824i32);
            zf_out[s * sz0 + (i * e + c) * sz1] = clamp(xv + y, 0i32 - act, act);
            sync_cube();
            grp += CUBE_COUNT_X as usize;
        }
    }

    /// The embedding and position tables' gradients and SGD steps in one launch (see
    /// [`embed_sgd`] for the arithmetic). A cube takes `span` rows of `dx` and sums them into
    /// both tables in shared memory first, so the global sums in `acc` (the byte table's
    /// `256 e`, then the position table's `t e`) take one atomic per entry a cube touched rather
    /// than one per element; a common byte would otherwise queue thousands of them on one
    /// address. The last cube to arrive rounds and steps both tables in place and leaves `acc`
    /// and its counter at zero.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn embed_pos_sgd(
        idx: &Tensor<i32>,
        dx: &Tensor<i32>,
        emb: &mut Tensor<i32>,
        pos: &mut Tensor<i32>,
        acc: &mut Tensor<Atomic<i32>>,
        counter: &mut Tensor<Atomic<i32>>,
        rates: &Tensor<i32>,
        n: usize,
        span: usize,
        rshift: i32,
        slot_emb: usize,
        slot_pos: usize,
        max: i32,
        #[comptime] t: usize,
        #[comptime] e: usize,
    ) {
        let tid = UNIT_POS_X as usize;
        let groups = comptime!(256 / e);
        let g = tid / e;
        let c = tid - g * e;
        let table = Shared::<[Atomic<i32>]>::new_slice(comptime!(256 * e));
        let places = Shared::<[Atomic<i32>]>::new_slice(comptime!(t * e));
        let mut i = tid;
        while i < comptime!(256 * e) {
            table[i].store(0i32);
            i += 256;
        }
        let mut i = tid;
        while i < comptime!(t * e) {
            places[i].store(0i32);
            i += 256;
        }
        sync_cube();
        let (sd0, sd1) = (dx.stride(0), dx.stride(1));
        let start = CUBE_POS_X as usize * span;
        let mut r = start + g;
        while r < start + span && r < n {
            let value = dx[r * sd0 + c * sd1];
            let byte = usize::cast_from(idx[r * idx.stride(0)]);
            let place = r - (r / t) * t;
            table[byte * e + c].fetch_add(value);
            places[place * e + c].fetch_add(value);
            r += groups;
        }
        sync_cube();
        let mut i = tid;
        while i < comptime!(256 * e) {
            let sum = table[i].load();
            if sum != 0 {
                acc[i].fetch_add(sum);
            }
            i += 256;
        }
        let mut i = tid;
        while i < comptime!(t * e) {
            let sum = places[i].load();
            if sum != 0 {
                acc[comptime!(256 * e) + i].fetch_add(sum);
            }
            i += 256;
        }
        let mut arrived = Shared::<[i32]>::new_slice(1usize);
        sync_storage();
        if tid == 0 {
            arrived[0] = counter[0].fetch_add(1i32);
        }
        sync_cube();
        if arrived[0] == i32::cast_from(CUBE_COUNT_X as usize - 1) {
            sync_storage();
            let rhalf = (1i32 << rshift) >> 1;
            let low = 0i32 - max;
            let mut i = tid;
            while i < comptime!((256 + t) * e) {
                let sum = acc[i].load();
                acc[i].store(0i32);
                let d = ((sum + rhalf + 1073741824i32) >> rshift) - (1073741824i32 >> rshift);
                let mut slot = slot_emb;
                if i >= comptime!(256 * e) {
                    slot = slot_pos;
                }
                let shift = rates[2 * slot * rates.stride(0)];
                let up = rates[(2 * slot + 1) * rates.stride(0)];
                let half = (1i32 << shift) >> 1;
                let delta = (((d + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
                if i < comptime!(256 * e) {
                    let v = i / e;
                    let col = i - v * e;
                    let at = v * emb.stride(0) + col * emb.stride(1);
                    emb[at] = clamp(emb[at] - delta, low, max);
                } else {
                    let v = (i - comptime!(256 * e)) / e;
                    let col = i - comptime!(256 * e) - v * e;
                    let at = v * pos.stride(0) + col * pos.stride(1);
                    pos[at] = clamp(pos[at] - delta, low, max);
                }
                i += 256;
            }
            if tid == 0 {
                counter[0].store(0i32);
            }
        }
    }

    /// The three attention projections' gradients and SGD steps in one launch, for widths
    /// `e <= 32` and chunks of 128 rows. Cubes stand on a grid of row ranges (`per` chunks each)
    /// by projection (`which`: q, k, v). A cube stages a whole chunk of `x` and of that projection's error in
    /// shared memory at once, so it waits on global memory once per chunk, and forms all `e x e`
    /// sums of the chunk (each unit four of them), floor-shifted by `cshift` as [`chunked_xty`]
    /// does. The ranges' totals are added atomically into `acc` (integer sums, so the order they
    /// land in does not matter); the last cube of a projection steps its weights in place with
    /// rate slot `slot + which`.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn qkv_sgd(
        x: &Tensor<i32>,
        dq: &Tensor<i32>,
        dk: &Tensor<i32>,
        dv: &Tensor<i32>,
        wq: &mut Tensor<i32>,
        wk: &mut Tensor<i32>,
        wv: &mut Tensor<i32>,
        acc: &mut Tensor<Atomic<i32>>,
        counter: &mut Tensor<Atomic<i32>>,
        rates: &Tensor<i32>,
        per: usize,
        cshift: i32,
        slot: usize,
        max: i32,
        #[comptime] e: usize,
    ) {
        let tid = UNIT_POS_X as usize;
        let range = CUBE_POS_X as usize;
        let ranges = CUBE_COUNT_X as usize;
        let which = CUBE_POS_Y as usize;
        let ee = comptime!(e * e);
        // this unit's sums: row i of the result, columns j0 .. j0 + 4
        let quads = comptime!(e / 4);
        let i = tid / quads;
        let j0 = (tid - i * quads) * 4;
        let active = tid < ee / 4;
        // chunks of 128 rows
        let mut xs = Shared::<[i32]>::new_slice(comptime!(128 * e));
        let mut ys = Shared::<[i32]>::new_slice(comptime!(128 * e));
        let mut total = Array::<i32>::new(4usize);
        #[unroll]
        for q in 0..4usize {
            total[q] = 0i32;
        }
        let (sx0, sx1) = (x.stride(0), x.stride(1));
        for p in 0..per {
            let r0 = (range * per + p) * 128;
            // unrolled, so every unit's loads are in flight at once
            #[unroll]
            for l in 0..comptime!(128 * e / 256) {
                let u = tid + l * 256;
                let r = u / e;
                let c = u - r * e;
                xs[u] = x[(r0 + r) * sx0 + c * sx1];
                let mut yv = dv[(r0 + r) * dv.stride(0) + c * dv.stride(1)];
                if which == 0 {
                    yv = dq[(r0 + r) * dq.stride(0) + c * dq.stride(1)];
                } else if which == 1 {
                    yv = dk[(r0 + r) * dk.stride(0) + c * dk.stride(1)];
                }
                ys[u] = yv;
            }
            sync_cube();
            if active {
                let mut acc = Array::<i32>::new(4usize);
                #[unroll]
                for q in 0..4usize {
                    acc[q] = 0i32;
                }
                #[unroll]
                for r in 0..128usize {
                    let xv = xs[r * e + i];
                    #[unroll]
                    for q in 0..4usize {
                        acc[q] += xv * ys[r * e + j0 + q];
                    }
                }
                #[unroll]
                for q in 0..4usize {
                    total[q] += ((acc[q] + 1073741824i32) >> cshift) - (1073741824i32 >> cshift);
                }
            }
            sync_cube();
        }
        // the ranges' totals meet in `acc`, a few atomics per address; the last cube of the
        // projection takes the sums and leaves `acc` and its counter at zero
        let mut last = true;
        if ranges > 1 {
            if active {
                #[unroll]
                for q in 0..4usize {
                    acc[which * ee + i * e + j0 + q].fetch_add(total[q]);
                }
            }
            let mut arrived = Shared::<[i32]>::new_slice(1usize);
            sync_storage();
            if tid == 0 {
                arrived[0] = counter[which].fetch_add(1i32);
            }
            sync_cube();
            last = arrived[0] == i32::cast_from(ranges - 1);
            if last {
                sync_storage();
                if active {
                    #[unroll]
                    for q in 0..4usize {
                        let at = which * ee + i * e + j0 + q;
                        total[q] = acc[at].load();
                        acc[at].store(0i32);
                    }
                }
                if tid == 0 {
                    counter[which].store(0i32);
                }
            }
        }
        if last && active {
            let s = slot + which;
            let shift = rates[2 * s * rates.stride(0)];
            let up = rates[(2 * s + 1) * rates.stride(0)];
            let half = (1i32 << shift) >> 1;
            let low = 0i32 - max;
            #[unroll]
            for q in 0..4usize {
                let delta = (((total[q] + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
                let j = j0 + q;
                if which == 0 {
                    let at = i * wq.stride(0) + j * wq.stride(1);
                    wq[at] = clamp(wq[at] - delta, low, max);
                } else if which == 1 {
                    let at = i * wk.stride(0) + j * wk.stride(1);
                    wk[at] = clamp(wk[at] - delta, low, max);
                } else {
                    let at = i * wv.stride(0) + j * wv.stride(1);
                    wv[at] = clamp(wv[at] - delta, low, max);
                }
            }
        }
    }

    /// One cube per row of 256 byte weights.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn coded(e: LinearView<'_, i32>, target: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>) {
        let row = CUBE_POS_X as usize;
        let v = UNIT_POS_X as usize;
        let byte = usize::cast_from(target.read(row));
        // weights are non-negative, so the plain shift is a floor
        let f = (e.read(row * 256 + v) >> 4) + 1i32;
        let mut low = f * 0i32;
        if v < byte {
            low = f;
        }
        let total = cube_sum(f);
        let below = cube_sum(low);
        if v == byte {
            out.write(row * 3, below);
            out.write(row * 3 + 1, f);
            out.write(row * 3 + 2, total);
        }
    }

    /// One cube per row: the boosted scores' maximum, then each byte's weight.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn mix_weights(
        total: LinearView<'_, i32>,
        guess: LinearView<'_, i32>,
        copy: LinearView<'_, i32>,
        lut: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        matcher: u32,
        span: i32,
    ) {
        let row = CUBE_POS_X as usize;
        let byte = UNIT_POS_X as usize;
        let at = row * 256 + byte;
        let mut own = total.read(at);
        if matcher == 1 {
            let proposed = usize::cast_from(guess.read(row * 2));
            let trust = usize::cast_from(guess.read(row * 2 + 1));
            if byte == proposed {
                own += ((copy.read(trust) + 1073741824i32) >> 6) - 16777216i32;
            }
        }
        let top = cube_max(own);
        out.write(at, lut.read(usize::cast_from(clamp(top - own, 0i32, span))));
    }

    /// One cube per row: Q12 probabilities from the row sum, less their own sum at the true byte.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn mix_error(e: LinearView<'_, i32>, target: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>) {
        let row = CUBE_POS_X as usize;
        let byte = UNIT_POS_X as usize;
        let at = row * 256 + byte;
        let x = e.read(at);
        // the row sums to between 2^20 and 2^28
        let unit = cube_sum(x) >> 12;
        let p = x / unit;
        let all = cube_sum(p);
        let mut value = p;
        if byte == usize::cast_from(target.read(row)) {
            value -= all;
        }
        out.write(at, value);
    }

    /// One cube of 256 units strides over the rows; three sums, one per trust level.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn guess_sums(g: LinearView<'_, i32>, guess: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, rows: usize, shift: i32) {
        let tid = UNIT_POS_X as usize;
        let mut s0 = 0i32;
        let mut s1 = 0i32;
        let mut s2 = 0i32;
        let mut r = tid;
        while r < rows {
            let level = guess.read(r * 2 + 1);
            let value = g.read(r * 256 + usize::cast_from(guess.read(r * 2)));
            if level == 0 {
                s0 += value;
            } else if level == 1 {
                s1 += value;
            } else if level == 2 {
                s2 += value;
            }
            r += 256;
        }
        let t0 = cube_sum(s0);
        let t1 = cube_sum(s1);
        let t2 = cube_sum(s2);
        if tid < 3 {
            let mut acc = t0;
            if tid == 1 {
                acc = t1;
            }
            if tid == 2 {
                acc = t2;
            }
            let half = (1i32 << shift) >> 1;
            out.write(tid, ((acc + half + 1073741824i32) >> shift) - (1073741824i32 >> shift));
        }
    }

    /// One cube per row: the leaves' errors in leaf order, summed up the tree in shared memory
    /// (heap numbering, leaves at 255..511), then each node's error from its subtree's sum and
    /// its 1-branch's.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn tree_back(g: LinearView<'_, i32>, prob: LinearView<'_, i32>, byte_at: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>) {
        let row = CUBE_POS_X as usize;
        let j = UNIT_POS_X as usize;
        let mut sums = Shared::<[i32]>::new_slice(511usize);
        sums[255 + j] = g.read(row * 256 + usize::cast_from(byte_at.read(j)));
        sync_cube();
        #[unroll]
        for step in 0..8usize {
            let width = 1usize << (7 - step);
            if j < width {
                let node = width - 1 + j;
                sums[node] = sums[2 * node + 1] + sums[2 * node + 2];
            }
            sync_cube();
        }
        if j < 255 {
            let under = sums[j];
            let right = sums[2 * j + 2];
            // |prob * under| < 2^26
            let part = ((prob.read(row * 255 + j) * under + 1073743872i32) >> 12) - 262144i32;
            out.write(row * 255 + j, clamp(right - part, -4096i32, 4096i32));
        }
    }

    /// One thread per element of one chunk's product, so the work spreads over many threads
    /// however few elements the product has.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn chunked_xty(
        x: LinearView<'_, i32>,
        y: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        a: usize,
        b: usize,
        chunk: usize,
        cshift: i32,
    ) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let ch = ABSOLUTE_POS / (a * b);
        let rest = ABSOLUTE_POS - ch * a * b;
        let ia = rest / b;
        let ib = rest - ia * b;
        let mut acc = 0i32;
        for r in 0..chunk {
            let at = ch * chunk + r;
            acc += x.read(at * a + ia) * y.read(at * b + ib);
        }
        out.write(ABSOLUTE_POS, ((acc + 1073741824i32) >> cshift) - (1073741824i32 >> cshift));
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn sgd_step(w: &mut Tensor<i32>, d: &Tensor<i32>, rates: &Tensor<i32>, n: usize, cols: usize, slot: usize, max: i32) {
        if ABSOLUTE_POS >= n {
            terminate!();
        }
        // the group's step, from the optimizer's device copy: (shift, up)
        let shift = rates[2 * slot * rates.stride(0)];
        let up = rates[(2 * slot + 1) * rates.stride(0)];
        let r = ABSOLUTE_POS / cols;
        let c = ABSOLUTE_POS - r * cols;
        let half = (1i32 << shift) >> 1;
        let delta = (((d[r * d.stride(0) + c * d.stride(1)] + half + 1073741824i32) >> shift) - (1073741824i32 >> shift)) * up;
        let low = 0i32 - max;
        let at = r * w.stride(0) + c * w.stride(1);
        w[at] = clamp(w[at] - delta, low, max);
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn embed(emb: LinearView<'_, i32>, pos: LinearView<'_, i32>, idx: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, t: usize, e: usize) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let r = ABSOLUTE_POS / e;
        let c = ABSOLUTE_POS - r * e;
        let place = r - (r / t) * t;
        let byte = usize::cast_from(idx.read(r));
        let a = ((emb.read(byte * e + c) + 1073741824i32) >> 8) - 4194304i32;
        let b = ((pos.read(place * e + c) + 1073741824i32) >> 8) - 4194304i32;
        out.write(ABSOLUTE_POS, a + b);
    }


    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn onehot_error(prob: LinearView<'_, i32>, target: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, vocab: usize, scale: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / vocab;
        let v = ABSOLUTE_POS - row * vocab;
        let mut value = prob.read(ABSOLUTE_POS);
        if v == usize::cast_from(target.read(row)) {
            value -= scale;
        }
        out.write(ABSOLUTE_POS, value);
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn tree_probs(z: LinearView<'_, i32>, sigmoid: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, span: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let logit = z.read(ABSOLUTE_POS);
        let mut size = logit;
        if logit < 0 {
            size = 0i32 - logit;
        }
        // sigma(-|z|); for z >= 0 the answer is its complement
        let low = sigmoid.read(usize::cast_from(clamp(size, 0i32, span)));
        let mut p = low;
        if logit >= 0 {
            p = 4096i32 - low;
        }
        out.write(ABSOLUTE_POS, clamp(p, 1i32, 4095i32));
    }

    /// One thread per byte weight: the product of the 8 branch probabilities on the byte's path.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn tree_leaves(
        p: LinearView<'_, i32>,
        leaf: LinearView<'_, i32>,
        guess: LinearView<'_, i32>,
        copy: LinearView<'_, i32>,
        lut: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        matcher: u32,
        span: i32,
    ) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / 256;
        let byte = ABSOLUTE_POS - row * 256;
        let place = usize::cast_from(leaf.read(byte));
        let mut acc = 16i32;
        for level in 0..8usize {
            // heap numbering: level l holds nodes (2^l - 1)..(2^(l+1) - 1)
            let node = (1usize << level) - 1 + (place >> (8 - level));
            let one = p.read(row * 255 + node);
            let mut branch = 4096i32 - one;
            if ((place >> (7 - level)) & 1) == 1 {
                branch = one;
            }
            if level == 0 {
                acc = branch * 16i32;
            } else {
                // acc <= 2^16 and branch <= 2^12, both non-negative
                acc = (acc * branch) >> 12;
            }
        }
        let mut weight = acc * 16i32;
        if matcher == 1 {
            // the boost scales every other byte down by exp(-gain); both factors cut to 10 bits
            let trust = usize::cast_from(guess.read(row * 2 + 1));
            let gain = clamp(((copy.read(trust) + 1073741824i32) >> 6) - 16777216i32, 0i32, span);
            let mut factor = lut.read(usize::cast_from(gain)) >> 10;
            if byte == usize::cast_from(guess.read(row * 2)) {
                factor = 1024i32;
            }
            weight = (weight >> 10) * factor;
        }
        out.write(ABSOLUTE_POS, weight);
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn tree_error(p: LinearView<'_, i32>, target: LinearView<'_, i32>, leaf: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / 255;
        let node = ABSOLUTE_POS - row * 255;
        let place = usize::cast_from(leaf.read(usize::cast_from(target.read(row))));
        let one = p.read(ABSOLUTE_POS);
        let mut value = one * 0i32;
        for level in 0..8usize {
            if (1usize << level) - 1 + (place >> (8 - level)) == node {
                value = one;
                if ((place >> (7 - level)) & 1) == 1 {
                    value = one - 4096i32;
                }
            }
        }
        out.write(ABSOLUTE_POS, value);
    }

    /// One thread per byte: the logs of the 8 branch probabilities on its path, summed.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn tree_log_scores(p: LinearView<'_, i32>, leaf: LinearView<'_, i32>, logp: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS / 256;
        let byte = ABSOLUTE_POS - row * 256;
        let place = usize::cast_from(leaf.read(byte));
        let mut acc = leaf.read(byte) * 0i32;
        for level in 0..8usize {
            let node = (1usize << level) - 1 + (place >> (8 - level));
            let one = p.read(row * 255 + node);
            let mut branch = 4096i32 - one;
            if ((place >> (7 - level)) & 1) == 1 {
                branch = one;
            }
            acc += logp.read(usize::cast_from(branch));
        }
        out.write(ABSOLUTE_POS, acc);
    }





    /// One thread per row.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn guess_error(
        weights: LinearView<'_, i32>,
        target: LinearView<'_, i32>,
        guess: LinearView<'_, i32>,
        mut out: LinearViewMut<'_, i32>,
        vocab: usize,
        normalise: u32,
    ) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let row = ABSOLUTE_POS;
        let byte = guess.read(row * 2);
        let mut value = weights.read(row * vocab + usize::cast_from(byte));
        if normalise == 1 {
            let mut sum = 0i32;
            for v in 0..vocab {
                sum += weights.read(row * vocab + v);
            }
            value = value / clamp(sum >> 12, 1i32, 2147483647i32);
        }
        if byte == target.read(row) {
            value -= 4096i32;
        }
        out.write(ABSOLUTE_POS, value);
    }

    /// One thread per trust level; each walks every row.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn level_sums(values: LinearView<'_, i32>, guess: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, rows: usize, shift: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let level = i32::cast_from(ABSOLUTE_POS);
        let mut acc = 0i32;
        for r in 0..rows {
            if guess.read(r * 2 + 1) == level {
                acc += values.read(r);
            }
        }
        let half = (1i32 << shift) >> 1;
        out.write(ABSOLUTE_POS, ((acc + half + 1073741824i32) >> shift) - (1073741824i32 >> shift));
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn relu_error(dh: LinearView<'_, i32>, h: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, act_max: i32, err: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let unit = h.read(ABSOLUTE_POS);
        let mut value = dh.read(ABSOLUTE_POS);
        if unit <= 0 || unit >= act_max {
            value = value * 0i32;
        }
        let low = 0i32 - err;
        out.write(ABSOLUTE_POS, clamp(value, low, err));
    }

    /// One thread per output element; each walks the streams.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn place_sums(dx: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, streams: usize, t: usize, e: usize, shift: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let place = ABSOLUTE_POS / e;
        let c = ABSOLUTE_POS - place * e;
        let mut acc = 0i32;
        for s in 0..streams {
            acc += dx.read((s * t + place) * e + c);
        }
        let half = (1i32 << shift) >> 1;
        out.write(ABSOLUTE_POS, ((acc + half + 1073741824i32) >> shift) - (1073741824i32 >> shift));
    }

    /// One thread per element of `dx`, each adding itself to its byte's row of the table. Integer
    /// adds commute, so the order the threads land in does not change the sums.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn scatter_rows(out: &mut Tensor<Atomic<i32>>, idx: &Tensor<i32>, dx: &Tensor<i32>, e: usize) {
        if ABSOLUTE_POS >= dx.len() {
            terminate!();
        }
        let r = ABSOLUTE_POS / e;
        let c = ABSOLUTE_POS - r * e;
        let value = usize::cast_from(idx[r * idx.stride(0)]);
        // raw tensors are indexed through their strides: a fresh output's rows are not packed
        out[value * out.stride(0) + c * out.stride(1)].fetch_add(dx[r * dx.stride(0) + c * dx.stride(1)]);
    }

    /// Division by 2^shift rounded to nearest, elementwise; `shift` may be 0.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn round_shift(x: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, shift: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let half = (1i32 << shift) >> 1;
        out.write(ABSOLUTE_POS, ((x.read(ABSOLUTE_POS) + half + 1073741824i32) >> shift) - (1073741824i32 >> shift));
    }

    /// One thread per element of the table: it walks every row and adds the ones for its byte.
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn rows_by_value(idx: LinearView<'_, i32>, dx: LinearView<'_, i32>, mut out: LinearViewMut<'_, i32>, n: usize, e: usize, shift: i32) {
        if !out.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }
        let value = ABSOLUTE_POS / e;
        let col = ABSOLUTE_POS - value * e;
        let wanted = i32::cast_from(value);
        let mut acc = 0i32;
        for r in 0..n {
            if idx.read(r) == wanted {
                acc += dx.read(r * e + col);
            }
        }
        // to nearest: half is 0 when there is no shift
        let half = (1i32 << shift) >> 1;
        out.write(ABSOLUTE_POS, ((acc + half + 1073741824i32) >> shift) - (1073741824i32 >> shift));
    }
}

/// Launches `$kernel` on `$cubes` cubes of `$units` units, passing `$args` after the views.
macro_rules! launch_on {
    ($kernel:path, ($cubes:expr, $units:expr), [$($input:expr),*], $out:expr, $($args:expr),*) => {{
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let address = [$(&$input,)* &$out].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            $kernel(&$out.client, CubeCount::Static($cubes as u32, 1, 1), CubeDim::new_1d($units as u32), address, $($input.into_linear_view(),)* $out.clone().into_linear_view(), $($args),*);
        }
    }};
}

/// Launches `$kernel` with one thread per element of `$out`, passing `$args` after the views.
macro_rules! launch {
    ($kernel:path, [$($input:expr),*], $out:expr, $($args:expr),*) => {{
        #[allow(unused_imports)]
        use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
        let elements = $out.meta.shape().num_elements();
        let cube_dim = CubeDim::new(&$out.client, elements);
        let cube_count = calculate_cube_count_elemwise(&$out.client, elements, cube_dim);
        let address = [$(&$input,)* &$out].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            $kernel(&$out.client, cube_count, cube_dim, address, $($input.into_linear_view(),)* $out.clone().into_linear_view(), $($args),*);
        }
    }};
}

impl FusedOps for CubeBackend {
    fn rows_by_value(idx: IntTensor<Self>, dx: IntTensor<Self>, vocab: usize, shift: u32) -> IntTensor<Self> {
        use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;

        use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};

        let (n, e) = (dx.meta.shape()[0], dx.meta.shape()[1]);
        let sums = Self::int_zeros(Shape::from([vocab, e]), &dx.device, burn::tensor::IntDType::I32);
        let cube_dim = CubeDim::new(&dx.client, n * e);
        let cube_count = calculate_cube_count_elemwise(&dx.client, n * e, cube_dim);
        let address = [&idx, &dx, &sums].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::scatter_rows::launch_unchecked(
                &sums.client,
                cube_count,
                cube_dim,
                address,
                sums.clone().into_tensor_arg(),
                idx.into_tensor_arg(),
                dx.into_tensor_arg(),
                e,
            );
        }
        let out = empty_device_contiguous_dtype(sums.client.clone(), sums.device.clone(), Shape::from([vocab, e]), sums.dtype);
        launch!(kernel::round_shift::launch_unchecked, [sums], out, shift as i32);
        out
    }

    fn matmul_shr_clamp(x: IntTensor<Self>, w: IntTensor<Self>, wt: bool, wshift: u32, shift: u32, bound: i32) -> IntTensor<Self> {
        cube_matmul(x, w, wt, wshift, shift, bound, None)
    }
}

/// [`FusedOps::matmul_shr_clamp`] on the GPU, and with `gate` [`GpuOps::matmul_gated`]: the
/// tensor-core kernel, the tiled one under `GMINT_MMA=0`, or the one-thread-per-output one under
/// `GMINT_NAIVE_MATMUL`.
fn cube_matmul(
    x: IntTensor<CubeBackend>,
    w: IntTensor<CubeBackend>,
    wt: bool,
    wshift: u32,
    shift: u32,
    bound: i32,
    gate: Option<(IntTensor<CubeBackend>, i32)>,
) -> IntTensor<CubeBackend> {
    use burn_cubecl::cubecl::{CubeCount, CubeDim, calculate_cube_count_elemwise};
    use burn_cubecl::kernel::into_contiguous;
    use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;

    // the tiled kernel reads the weight through its strides, so a transpose is only a view
    let w = if wt { CubeBackend::int_swap_dims(w, 0, 1) } else { w };
    // without a gate the kernels are built without its reads, and `x` stands in for it
    let gated = gate.is_some();
    let (gate, act_max) = gate.unwrap_or_else(|| (x.clone(), 0));
    static NAIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*NAIVE.get_or_init(|| std::env::var_os("GMINT_NAIVE_MATMUL").is_some()) {
        let (n, k, m) = (x.meta.shape()[0], x.meta.shape()[1], w.meta.shape()[1]);
        let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::from([n, m]), x.dtype);
        let address = [&x, &w, &gate, &out].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        if use_mma() {
            let (wr, wc, ks) = mma_shape(n, m);
            unsafe {
                kernel::matmul_mma::launch_unchecked(
                    &out.client,
                    CubeCount::Static(n.div_ceil(16 * wr) as u32, m.div_ceil(16 * wc) as u32, 1),
                    CubeDim::new_1d((32 * wr * wc) as u32),
                    address,
                    x.into_tensor_arg(),
                    w.into_tensor_arg(),
                    gate.into_tensor_arg(),
                    out.clone().into_tensor_arg(),
                    n,
                    k,
                    m,
                    wshift as i32,
                    shift as i32,
                    -bound,
                    bound,
                    act_max,
                    wr,
                    wc,
                    ks,
                    gated,
                );
            }
            return out;
        }
        let (tr, tc) = tile_shape(n, m);
        let cube_count = CubeCount::Static(n.div_ceil(16 * tr) as u32, m.div_ceil(16 * tc) as u32, 1);
        unsafe {
            kernel::matmul_tiled::launch_unchecked(
                &out.client,
                cube_count,
                CubeDim::new_1d(256),
                address,
                x.into_tensor_arg(),
                w.into_tensor_arg(),
                gate.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                n,
                k,
                m,
                wshift as i32,
                shift as i32,
                -bound,
                bound,
                act_max,
                tr,
                tc,
                gated,
            );
        }
        return out;
    }
    let (x, w) = (into_contiguous(x), into_contiguous(w));
    let (n, k, m) = (x.meta.shape()[0], x.meta.shape()[1], w.meta.shape()[1]);
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::from([n, m]), x.dtype);
    let cube_dim = CubeDim::new(&x.client, n * m);
    let cube_count = calculate_cube_count_elemwise(&x.client, n * m, cube_dim);
    let address = [&x, &w, &gate, &out].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
    unsafe {
        kernel::matmul_shr_clamp::launch_unchecked(
            &out.client,
            cube_count,
            cube_dim,
            address,
            x.into_linear_view(),
            w.into_linear_view(),
            gate.into_linear_view(),
            out.clone().into_linear_view(),
            k,
            m,
            wshift as i32,
            shift as i32,
            bound,
            act_max,
            gated,
        );
    }
    out
}

thread_local! {
    static MMA: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether matmuls run on the tensor cores ([`kernel::matmul_mma`]); `GMINT_MMA=0` turns them off.
fn use_mma() -> bool {
    MMA.with(|m| *m.get().get_or_insert_with(|| std::env::var("GMINT_MMA").map_or(true, |v| v != "0")))
}

/// Warps down and across, and slice depth, for [`kernel::matmul_mma`] on an `[n, m]` output.
/// 2 x 2 warps over 32-deep slices measured best on every shape the model uses, against 4 x 2,
/// 4 x 4, 2 x 4, 2 x 1 and 1 x 2 warps and 64-deep slices.
fn mma_shape(_n: usize, _m: usize) -> (usize, usize, usize) {
    (2, 2, 32)
}

#[cfg(test)]
fn set_mma(on: bool) {
    MMA.with(|m| m.set(Some(on)));
}

/// Rows and columns per thread for the tiled matmul on an `[n, m]` output: the largest block
/// that still leaves a few cubes per streaming multiprocessor.
/// Rows and columns per thread for the tiled matmul on an `[n, m]` output: the largest block
/// that still leaves a few cubes per streaming multiprocessor.
fn tile_shape(n: usize, m: usize) -> (usize, usize) {
    let cubes = |tr: usize, tc: usize| n.div_ceil(16 * tr) * m.div_ceil(16 * tc);
    let tc = if m <= 32 { 2 } else { 4 };
    for tr in [4, 2] {
        if cubes(tr, tc) >= 96 {
            return (tr, tc);
        }
    }
    (2, 2)
}

/// Rows and columns per thread for [`kernel::xty_sgd`] on an `[a, b]` result.
fn gram_tile(a: usize, b: usize) -> (usize, usize) {
    let cubes = |tr: usize, tc: usize| a.div_ceil(16 * tr) * b.div_ceil(16 * tc);
    for (tr, tc) in [(2, 2), (1, 2)] {
        if cubes(tr, tc) >= 16 {
            return (tr, tc);
        }
    }
    (1, 1)
}

/// Cubes a split reduction aims for: a few per streaming multiprocessor.
const SPREAD: usize = 192;
/// Arrival counters the split reductions share. Kernels run one at a time, and every one leaves
/// the counters it used at zero, so one zeroed buffer serves them all.
const COUNTERS: usize = 4096;

/// Sums the atomic reductions share; like the counters, every kernel leaves them at zero.
const ACCUMULATORS: usize = 1 << 16;

thread_local! {
    static COUNTER_BUF: std::cell::RefCell<Option<IntTensor<CubeBackend>>> = const { std::cell::RefCell::new(None) };
    static ACC_BUF: std::cell::RefCell<Option<IntTensor<CubeBackend>>> = const { std::cell::RefCell::new(None) };
}

/// The shared arrival counters, made (zeroed) on first use.
fn counters(like: &IntTensor<CubeBackend>) -> IntTensor<CubeBackend> {
    COUNTER_BUF.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| CubeBackend::int_zeros(Shape::from([COUNTERS]), &like.device, burn::tensor::IntDType::I32))
            .clone()
    })
}

/// The shared atomic sums and the arrival counters.
fn scratch_sums(like: &IntTensor<CubeBackend>) -> (IntTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let acc = ACC_BUF.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| CubeBackend::int_zeros(Shape::from([ACCUMULATORS]), &like.device, burn::tensor::IntDType::I32))
            .clone()
    });
    (acc, counters(like))
}

/// A fresh output tensor shaped `$shape` on the device of `$like`.
macro_rules! fresh {
    ($like:expr, $shape:expr) => {
        burn_cubecl::ops::numeric::empty_device_contiguous_dtype($like.client.clone(), $like.device.clone(), Shape::from($shape), $like.dtype)
    };
}

impl GpuOps for CubeBackend {
    fn qkv_sgd(
        x: IntTensor<Self>,
        dq: IntTensor<Self>,
        dk: IntTensor<Self>,
        dv: IntTensor<Self>,
        wq: IntTensor<Self>,
        wk: IntTensor<Self>,
        wv: IntTensor<Self>,
        chunk: usize,
        rates: IntTensor<Self>,
        slot: usize,
        max: i32,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>) {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (n, e) = (x.meta.shape()[0], x.meta.shape()[1]);
        let chunks = n / chunk;
        assert!(qkv_fits(e) && chunk == 128 && n % chunk == 0 && chunks.is_power_of_two() && 3 * e * e <= ACCUMULATORS);
        // about 32 cubes per projection, a power of two of chunks each
        let per = chunks.div_ceil(32).next_power_of_two().min(chunks);
        let ranges = chunks / per;
        let (acc, counter) = scratch_sums(&x);
        let address = [&x, &dq, &dk, &dv, &wq].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::qkv_sgd::launch_unchecked(
                &x.client.clone(),
                CubeCount::Static(ranges as u32, 3, 1),
                CubeDim::new_1d(256),
                address,
                x.into_tensor_arg(),
                dq.into_tensor_arg(),
                dk.into_tensor_arg(),
                dv.into_tensor_arg(),
                wq.clone().into_tensor_arg(),
                wk.clone().into_tensor_arg(),
                wv.clone().into_tensor_arg(),
                acc.into_tensor_arg(),
                counter.into_tensor_arg(),
                rates.into_tensor_arg(),
                per,
                chunks.trailing_zeros() as i32,
                slot,
                max,
                e,
            );
        }
        (wq, wk, wv)
    }

    fn embed_pos_sgd(idx: IntTensor<Self>, dx: IntTensor<Self>, emb: IntTensor<Self>, pos: IntTensor<Self>, rshift: u32, rates: IntTensor<Self>, slots: [usize; 2], max: i32) -> (IntTensor<Self>, IntTensor<Self>) {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (n, e, t) = (dx.meta.shape()[0], dx.meta.shape()[1], pos.meta.shape()[0]);
        assert!(embed_pos_fits(t, e) && emb.meta.shape()[0] == 256 && (256 + t) * e <= ACCUMULATORS);
        // ranges of 64 rows: enough cubes to spread the work, few enough partial tables
        let span = 64;
        let (acc, counter) = scratch_sums(&emb);
        let address = [&idx, &dx, &emb, &pos].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::embed_pos_sgd::launch_unchecked(
                &emb.client,
                CubeCount::Static(n.div_ceil(span) as u32, 1, 1),
                CubeDim::new_1d(256),
                address,
                idx.into_tensor_arg(),
                dx.into_tensor_arg(),
                emb.clone().into_tensor_arg(),
                pos.clone().into_tensor_arg(),
                acc.into_tensor_arg(),
                counter.into_tensor_arg(),
                rates.into_tensor_arg(),
                n,
                span,
                rshift as i32,
                slots[0],
                slots[1],
                max,
                t,
                e,
            );
        }
        (emb, pos)
    }

    fn write_ints(t: IntTensor<Self>, values: Vec<i32>) -> IntTensor<Self> {
        assert!(t.is_contiguous() && values.len() == t.meta.shape().num_elements(), "write_ints needs a contiguous tensor of the same size");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        t.client.write(&t.handle, burn_cubecl::cubecl::bytes::Bytes::from_bytes_vec(bytes));
        t
    }

    fn graph_ctl(probe: IntTensor<Self>, cmd: u32) -> IntTensor<Self> {
        let client = &probe.client;
        let done: Result<(), String> = match cmd {
            0 => client.graph_prepare().map_err(|e| format!("graph prepare: {e:?}")),
            1 => client.start_capture().map_err(|e| format!("start capture: {e:?}")),
            2 => client.stop_capture().map(|g| GRAPH.with(|c| *c.borrow_mut() = Some(g))).map_err(|e| format!("stop capture: {e:?}")),
            3 => GRAPH.with(|c| match &*c.borrow() {
                // SAFETY: the captured buffers are the model's weights and the codec's input
                // slots, held for the graph's life, and intermediates the capture pinned; one
                // thread drives this one stream
                Some(g) => unsafe { g.replay() }.map_err(|e| format!("replay: {e:?}")),
                None => Err("replay without a graph".into()),
            }),
            _ => {
                GRAPH.with(|c| c.borrow_mut().take());
                Ok(())
            }
        };
        if let Err(e) = done {
            GRAPH_ERR.with(|c| *c.borrow_mut() = Some(e));
        }
        probe
    }

    fn attn_forward(
        ctx: IntTensor<Self>,
        emb: IntTensor<Self>,
        pos: IntTensor<Self>,
        wq: IntTensor<Self>,
        wk: IntTensor<Self>,
        wv: IntTensor<Self>,
        lut: IntTensor<Self>,
        t: usize,
        span: i32,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>, IntTensor<Self>) {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (b, width, e) = (ctx.meta.shape()[0], ctx.meta.shape()[1], emb.meta.shape()[1]);
        assert!(attention_fits(t, e) && width >= t);
        let rows = fresh!(ctx, [b * t, e]);
        let (idx, x, q, k, v) = (fresh!(ctx, [b * t]), rows, fresh!(ctx, [b * t, e]), fresh!(ctx, [b * t, e]), fresh!(ctx, [b * t, e]));
        let (a, zf) = (fresh!(ctx, [b * t, t]), fresh!(ctx, [b, t * e]));
        let per = 256 / (t * e);
        let groups = b.div_ceil(per);
        assert!(b % per == 0);
        let address = [&ctx, &emb, &pos, &x, &q, &k, &v, &a, &zf].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::attn_forward::launch_unchecked(
                &ctx.client,
                CubeCount::Static(groups.min(attn_cubes()) as u32, 1, 1),
                CubeDim::new_1d(256),
                address,
                ctx.clone().into_tensor_arg(),
                emb.into_tensor_arg(),
                pos.into_tensor_arg(),
                wq.into_tensor_arg(),
                wk.into_tensor_arg(),
                wv.into_tensor_arg(),
                lut.into_tensor_arg(),
                idx.clone().into_tensor_arg(),
                x.clone().into_tensor_arg(),
                q.clone().into_tensor_arg(),
                k.clone().into_tensor_arg(),
                v.clone().into_tensor_arg(),
                a.clone().into_tensor_arg(),
                zf.clone().into_tensor_arg(),
                b,
                width - t,
                span,
                t,
                e,
            );
        }
        (idx, x, q, k, v, a, zf)
    }

    fn xty_sgd(x: IntTensor<Self>, y: IntTensor<Self>, w: IntTensor<Self>, chunk: usize, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self> {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (n, a, b) = (x.meta.shape()[0], x.meta.shape()[1], y.meta.shape()[1]);
        let chunks = n / chunk;
        assert!(n % chunk == 0 && chunk % 16 == 0 && chunks.is_power_of_two());
        // a 32 x 32 result is one tile: the CUDA cores measured faster there (8.9 against 10.1 us)
        if use_mma() && chunk % 32 == 0 && a * b > 32 * 32 {
            let (ta, tb) = (a.div_ceil(32), b.div_ceil(32));
            let splits = (SPREAD.div_ceil(ta * tb)).next_power_of_two().min(chunks);
            assert!(ta * tb <= COUNTERS);
            let scratch = fresh!(w, [splits * a * b]);
            let counter = counters(&w);
            let address = [&x, &y, &w, &scratch].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
            unsafe {
                kernel::xty_mma::launch_unchecked(
                    &w.client,
                    CubeCount::Static(ta as u32, tb as u32, splits as u32),
                    CubeDim::new_1d(128),
                    address,
                    x.into_tensor_arg(),
                    y.into_tensor_arg(),
                    w.clone().into_tensor_arg(),
                    scratch.into_tensor_arg(),
                    counter.into_tensor_arg(),
                    a,
                    b,
                    chunk,
                    n / splits,
                    splits,
                    chunks.trailing_zeros() as i32,
                    rates.clone().into_tensor_arg(),
                    slot,
                    max,
                );
            }
            return w;
        }
        let (tr, tc) = gram_tile(a, b);
        let (ta, tb) = (a.div_ceil(16 * tr), b.div_ceil(16 * tc));
        // enough cubes to go round, each a whole number of chunks
        let splits = (SPREAD.div_ceil(ta * tb)).next_power_of_two().min(chunks);
        assert!(ta * tb <= COUNTERS);
        let scratch = fresh!(w, [splits * a * b]);
        let counter = counters(&w);
        let address = [&x, &y, &w, &scratch].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::xty_sgd::launch_unchecked(
                &w.client,
                CubeCount::Static(ta as u32, tb as u32, splits as u32),
                CubeDim::new_1d(256),
                address,
                x.into_tensor_arg(),
                y.into_tensor_arg(),
                w.clone().into_tensor_arg(),
                scratch.into_tensor_arg(),
                counter.into_tensor_arg(),
                a,
                b,
                chunk,
                n / splits,
                splits,
                chunks.trailing_zeros() as i32,
                rates.into_tensor_arg(),
                slot,
                max,
                tr,
                tc,
            );
        }
        w
    }

    fn embed_sgd(idx: IntTensor<Self>, dx: IntTensor<Self>, w: IntTensor<Self>, t: usize, by_place: bool, rshift: u32, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self> {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (n, e) = (dx.meta.shape()[0], dx.meta.shape()[1]);
        let rows = w.meta.shape()[0];
        assert!(rows * e <= ACCUMULATORS);
        let (acc, counter) = scratch_sums(&w);
        let address = [&idx, &dx, &w].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::embed_sgd::launch_unchecked(
                &w.client,
                CubeCount::Static((n * e).div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                address,
                idx.into_tensor_arg(),
                dx.into_tensor_arg(),
                w.clone().into_tensor_arg(),
                acc.into_tensor_arg(),
                counter.into_tensor_arg(),
                n,
                t,
                e,
                rows,
                by_place as u32,
                rshift as i32,
                rates.into_tensor_arg(),
                slot,
                max,
            );
        }
        w
    }

    fn tree_log_scores(p: IntTensor<Self>, leaf: IntTensor<Self>, logp: IntTensor<Self>) -> IntTensor<Self> {
        let out = fresh!(p, [p.meta.shape()[0], 256]);
        launch!(kernel::tree_log_scores::launch_unchecked, [p, leaf, logp], out,);
        out
    }

    fn mix_weights(total: IntTensor<Self>, guess: IntTensor<Self>, copy: IntTensor<Self>, lut: IntTensor<Self>, matcher: bool, span: i32) -> IntTensor<Self> {
        assert_eq!(total.meta.shape()[1], 256);
        let out = fresh!(total, total.meta.shape().clone());
        let rows = total.meta.shape()[0];
        launch_on!(kernel::mix_weights::launch_unchecked, (rows, 256), [total, guess, copy, lut], out, matcher as u32, span);
        out
    }

    fn mix_error(e: IntTensor<Self>, target: IntTensor<Self>) -> IntTensor<Self> {
        assert_eq!(e.meta.shape()[1], 256);
        let out = fresh!(e, e.meta.shape().clone());
        let rows = e.meta.shape()[0];
        launch_on!(kernel::mix_error::launch_unchecked, (rows, 256), [e, target], out,);
        out
    }

    fn guess_sums(g: IntTensor<Self>, guess: IntTensor<Self>, shift: u32) -> IntTensor<Self> {
        assert_eq!(g.meta.shape()[1], 256);
        let rows = g.meta.shape()[0];
        let out = fresh!(g, [3, 1]);
        launch_on!(kernel::guess_sums::launch_unchecked, (1, 256), [g, guess], out, rows, shift as i32);
        out
    }

    fn tree_back(g: IntTensor<Self>, prob: IntTensor<Self>, byte_at: IntTensor<Self>) -> IntTensor<Self> {
        let out = fresh!(prob, prob.meta.shape().clone());
        let rows = prob.meta.shape()[0];
        launch_on!(kernel::tree_back::launch_unchecked, (rows, 256), [g, prob, byte_at], out,);
        out
    }
    fn tree_probs(z: IntTensor<Self>, sigmoid: IntTensor<Self>, span: i32) -> IntTensor<Self> {
        let out = fresh!(z, z.meta.shape().clone());
        launch!(kernel::tree_probs::launch_unchecked, [z, sigmoid], out, span);
        out
    }

    fn tree_leaves(
        p: IntTensor<Self>,
        leaf: IntTensor<Self>,
        guess: IntTensor<Self>,
        copy: IntTensor<Self>,
        lut: IntTensor<Self>,
        matcher: bool,
        span: i32,
    ) -> IntTensor<Self> {
        let out = fresh!(p, [p.meta.shape()[0], 256]);
        launch!(kernel::tree_leaves::launch_unchecked, [p, leaf, guess, copy, lut], out, matcher as u32, span);
        out
    }

    fn tree_error(p: IntTensor<Self>, target: IntTensor<Self>, leaf: IntTensor<Self>) -> IntTensor<Self> {
        let out = fresh!(p, p.meta.shape().clone());
        launch!(kernel::tree_error::launch_unchecked, [p, target, leaf], out,);
        out
    }

    fn guess_error(weights: IntTensor<Self>, target: IntTensor<Self>, guess: IntTensor<Self>, normalise: bool) -> IntTensor<Self> {
        let (b, vocab) = (weights.meta.shape()[0], weights.meta.shape()[1]);
        let out = fresh!(weights, [b]);
        launch!(kernel::guess_error::launch_unchecked, [weights, target, guess], out, vocab, normalise as u32);
        out
    }

    fn level_sums(values: IntTensor<Self>, guess: IntTensor<Self>, shift: u32) -> IntTensor<Self> {
        let rows = values.meta.shape()[0];
        let out = fresh!(values, [3, 1]);
        launch!(kernel::level_sums::launch_unchecked, [values, guess], out, rows, shift as i32);
        out
    }

    fn relu_error(dh: IntTensor<Self>, h: IntTensor<Self>, act_max: i32, err: i32) -> IntTensor<Self> {
        let out = fresh!(dh, dh.meta.shape().clone());
        launch!(kernel::relu_error::launch_unchecked, [dh, h], out, act_max, err);
        out
    }

    fn place_sums(dx: IntTensor<Self>, t: usize, shift: u32) -> IntTensor<Self> {
        let (n, e) = (dx.meta.shape()[0], dx.meta.shape()[1]);
        let out = fresh!(dx, [t, e]);
        launch!(kernel::place_sums::launch_unchecked, [dx], out, n / t, t, e, shift as i32);
        out
    }

    fn sgd_step(w: IntTensor<Self>, d: IntTensor<Self>, rates: IntTensor<Self>, slot: usize, max: i32) -> IntTensor<Self> {
        use burn_cubecl::cubecl::{CubeCount, CubeDim};
        let (n, cols) = (w.meta.shape().num_elements(), w.meta.shape()[w.meta.shape().len() - 1]);
        assert_eq!(w.meta.shape().len(), 2);
        let address = [&w, &d].into_iter().map(|t| t.required_address_type()).max().unwrap_or_default();
        unsafe {
            kernel::sgd_step::launch_unchecked(
                &w.client,
                CubeCount::Static(n.div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                address,
                w.clone().into_tensor_arg(),
                d.into_tensor_arg(),
                rates.into_tensor_arg(),
                n,
                cols,
                slot,
                max,
            );
        }
        w
    }

    fn embed(emb: IntTensor<Self>, pos: IntTensor<Self>, idx: IntTensor<Self>) -> IntTensor<Self> {
        use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
        let (n, t, e) = (idx.meta.shape()[0], pos.meta.shape()[0], emb.meta.shape()[1]);
        let out = empty_device_contiguous_dtype(emb.client.clone(), emb.device.clone(), Shape::from([n, e]), emb.dtype);
        launch!(kernel::embed::launch_unchecked, [emb, pos, idx], out, t, e);
        out
    }

    fn coded(e: IntTensor<Self>, target: IntTensor<Self>) -> IntTensor<Self> {
        let (b, vocab) = (e.meta.shape()[0], e.meta.shape()[1]);
        assert_eq!(vocab, 256);
        let out = fresh!(e, [b, 3]);
        launch_on!(kernel::coded::launch_unchecked, (b, 256), [e, target], out,);
        out
    }

    fn onehot_error(prob: IntTensor<Self>, target: IntTensor<Self>, scale: i32) -> IntTensor<Self> {
        use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
        let vocab = prob.meta.shape()[1];
        let out = empty_device_contiguous_dtype(prob.client.clone(), prob.device.clone(), prob.meta.shape().clone(), prob.dtype);
        launch!(kernel::onehot_error::launch_unchecked, [prob, target], out, vocab, scale);
        out
    }

    fn attn_probs(q: IntTensor<Self>, k: IntTensor<Self>, lut: IntTensor<Self>, span: i32) -> IntTensor<Self> {
        let (b, t, e) = (q.meta.shape()[0], q.meta.shape()[1], q.meta.shape()[2]);
        // a cube holds whole rows
        assert!(256 % t == 0 && (b * t * t) % 256 == 0);
        let out = fresh!(q, [b, t, t]);
        launch_on!(kernel::attn_probs::launch_unchecked, (b * t * t / 256, 256), [q, k, lut], out, t, e, span);
        out
    }

    fn attn_ds(dz: IntTensor<Self>, v: IntTensor<Self>, a: IntTensor<Self>, err: i32) -> IntTensor<Self> {
        let (b, t, e) = (v.meta.shape()[0], v.meta.shape()[1], v.meta.shape()[2]);
        assert!(256 % t == 0 && (b * t * t) % 256 == 0);
        let out = fresh!(a, [b, t, t]);
        launch_on!(kernel::attn_ds::launch_unchecked, (b * t * t / 256, 256), [dz, v, a], out, t, e, err);
        out
    }

    fn small_matmul(m: IntTensor<Self>, n: IntTensor<Self>, mt: bool, shift: u32, bound: i32) -> IntTensor<Self> {
        let (b, t, e) = (n.meta.shape()[0], n.meta.shape()[1], n.meta.shape()[2]);
        let out = fresh!(n, [b, t, e]);
        // built without the gate's reads; `n` stands in for it
        let gate = n.clone();
        launch!(kernel::small_matmul::launch_unchecked, [m, n, gate], out, t, e, mt as usize, shift as i32, bound, 0, false);
        out
    }

    fn small_matmul_gated(m: IntTensor<Self>, n: IntTensor<Self>, gate: IntTensor<Self>, mt: bool, shift: u32, bound: i32, act_max: i32) -> IntTensor<Self> {
        let (b, t, e) = (n.meta.shape()[0], n.meta.shape()[1], n.meta.shape()[2]);
        assert_eq!(gate.meta.shape().dims::<3>(), [b, t, e]);
        let out = fresh!(n, [b, t, e]);
        launch!(kernel::small_matmul::launch_unchecked, [m, n, gate], out, t, e, mt as usize, shift as i32, bound, act_max, true);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_gated(x: IntTensor<Self>, w: IntTensor<Self>, gate: IntTensor<Self>, wt: bool, wshift: u32, shift: u32, bound: i32, act_max: i32) -> IntTensor<Self> {
        let rows = x.meta.shape()[0];
        let cols = if wt { w.meta.shape()[0] } else { w.meta.shape()[1] };
        assert_eq!(gate.meta.shape().dims::<2>(), [rows, cols]);
        cube_matmul(x, w, wt, wshift, shift, bound, Some((gate, act_max)))
    }

    fn chunked_xty(x: IntTensor<Self>, y: IntTensor<Self>, chunk: usize, cshift: u32) -> IntTensor<Self> {
        use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
        let (n, a, b) = (x.meta.shape()[0], x.meta.shape()[1], y.meta.shape()[1]);
        let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::from([n / chunk, a, b]), x.dtype);
        launch!(kernel::chunked_xty::launch_unchecked, [x, y], out, a, b, chunk, cshift as i32);
        out
    }
}

/// How a fused matmul treats its weight and its result.
#[derive(Clone, Copy)]
pub struct Mat {
    /// Use the weight transposed.
    pub wt: bool,
    /// Floor-shift every weight by this first (8 turns a Q16 shadow into its Q8 forward value).
    pub wshift: u32,
    /// Floor-shift the product by this.
    pub shift: u32,
    /// Clamp the result to +-bound.
    pub bound: i32,
}

/// No clamp: results are bounded by 2^30 already.
pub const UNBOUNDED: i32 = 1 << 30;

/// The fused matmul on ordinary tensors; see [`FusedOps::matmul_shr_clamp`].
pub fn matmul(x: Tensor<2, Int>, w: Tensor<2, Int>, o: Mat) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::matmul_shr_clamp(x.into_dispatch(), w.into_dispatch(), o.wt, o.wshift, o.shift, o.bound))
}

/// [`matmul`], then with a `gate` `(g, act_max)` 0 wherever `|g| >= act_max`: the error passed
/// back through a forward clamp to +-act_max on `g`. See [`GpuOps::matmul_gated`].
pub fn matmul_gated(x: Tensor<2, Int>, w: Tensor<2, Int>, o: Mat, gate: Option<(&Tensor<2, Int>, i32)>) -> Tensor<2, Int> {
    match gate {
        None => matmul(x, w, o),
        Some((g, act_max)) if on_gpu(&x) => Tensor::from_dispatch(Dispatch::matmul_gated(
            x.into_dispatch(),
            w.into_dispatch(),
            g.clone().into_dispatch(),
            o.wt,
            o.wshift,
            o.shift,
            o.bound,
            act_max,
        )),
        Some((g, act_max)) => matmul(x, w, o) * unclamped(g, act_max),
    }
}

/// 1 where `|t| < act_max`, else 0: where a clamp to +-act_max did not hold.
fn unclamped<const D: usize>(t: &Tensor<D, Int>, act_max: i32) -> Tensor<D, Int> {
    (t.clone().abs() * -1 + act_max).clamp(0, 1)
}

/// See [`FusedOps::rows_by_value`].
pub fn rows_by_value(idx: Tensor<1, Int>, dx: Tensor<2, Int>, vocab: usize, shift: u32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::rows_by_value(idx.into_dispatch(), dx.into_dispatch(), vocab, shift))
}

/// Whether a tensor lives on a device the kernels in this file run on. On the CPU backend the
/// wrappers below run the ordinary tensor code they were written to replace.
pub fn on_gpu<const D: usize>(t: &Tensor<D, Int>) -> bool {
    t.device() != burn::tensor::Device::flex()
}

fn attn_cubes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("GMINT_ATTN_CUBES").ok().and_then(|v| v.parse().ok()).unwrap_or(128))
}

thread_local! {
    static GRAPH: std::cell::RefCell<Option<burn_cubecl::cubecl::client::Graph>> = const { std::cell::RefCell::new(None) };
    static GRAPH_ERR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Steps of the CUDA graph a codec captures: [`Prepare`](Graph::Prepare) before a warm-up run
/// of the work, [`Start`](Graph::Start) and [`Stop`](Graph::Stop) around a second run, which is
/// recorded rather than executed, then [`Replay`](Graph::Replay) to execute it as often as
/// needed. One graph per thread; [`Drop`](Graph::Drop) releases it.
#[derive(Clone, Copy)]
pub enum Graph {
    Prepare,
    Start,
    Stop,
    Replay,
    Drop,
}

/// Runs one graph step on the device of `probe`.
pub fn graph<const D: usize>(probe: &Tensor<D, Int>, step: Graph) -> Result<(), String> {
    let _ = Tensor::<D, Int>::from_dispatch(Dispatch::graph_ctl(probe.clone().into_dispatch(), step as u32));
    GRAPH_ERR.with(|c| c.borrow_mut().take()).map_or(Ok(()), Err)
}

/// Overwrites a device tensor in place; see [`GpuOps::write_ints`].
pub fn write_ints<const D: usize>(t: &Tensor<D, Int>, values: Vec<i32>) {
    let _ = Tensor::<D, Int>::from_dispatch(Dispatch::write_ints(t.clone().into_dispatch(), values));
}

/// Whether [`GpuOps::qkv_sgd`] covers projections of width `e`.
pub fn qkv_fits(e: usize) -> bool {
    e <= 32 && e % 4 == 0
}

/// The three attention projections stepped in place; see [`GpuOps::qkv_sgd`].
#[allow(clippy::too_many_arguments)]
pub fn qkv_sgd(x: &Tensor<2, Int>, d: [&Tensor<2, Int>; 3], w: [&Tensor<2, Int>; 3], chunk: usize, rates: &Tensor<1, Int>, slot: usize, max: i32) -> [Tensor<2, Int>; 3] {
    let t = |t: &Tensor<2, Int>| t.clone().into_dispatch();
    let (q, k, v) = Dispatch::qkv_sgd(t(x), t(d[0]), t(d[1]), t(d[2]), t(w[0]), t(w[1]), t(w[2]), chunk, rates.clone().into_dispatch(), slot, max);
    [Tensor::from_dispatch(q), Tensor::from_dispatch(k), Tensor::from_dispatch(v)]
}

/// Whether [`GpuOps::embed_pos_sgd`] covers a model with context `t` and width `e`: both tables
/// fit in shared memory and a cube's units split evenly into rows of `e`.
pub fn embed_pos_fits(t: usize, e: usize) -> bool {
    e <= 32 && 256 % e == 0 && t * e <= 2048
}

/// The byte and position tables stepped in place from the error at the embedded rows; see
/// [`GpuOps::embed_pos_sgd`].
pub fn embed_pos_sgd(idx: &Tensor<1, Int>, dx: &Tensor<2, Int>, emb: &Tensor<2, Int>, pos: &Tensor<2, Int>, rshift: u32, rates: &Tensor<1, Int>, slots: [usize; 2], max: i32) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let d = |t: &Tensor<2, Int>| t.clone().into_dispatch();
    let (e, p) = Dispatch::embed_pos_sgd(idx.clone().into_dispatch(), d(dx), d(emb), d(pos), rshift, rates.clone().into_dispatch(), slots, max);
    (Tensor::from_dispatch(e), Tensor::from_dispatch(p))
}

/// Whether [`GpuOps::attn_forward`] covers an attention block with context `t` and width `e`:
/// a cube holds whole streams and their attention weights, and the three projections fit in
/// shared memory.
pub fn attention_fits(t: usize, e: usize) -> bool {
    t <= e && e <= 32 && 256 % (t * e) == 0
}

/// The attention block's forward pass in one kernel; see [`GpuOps::attn_forward`].
#[allow(clippy::type_complexity)]
pub fn attn_forward(
    ctx: &Tensor<2, Int>,
    emb: &Tensor<2, Int>,
    pos: &Tensor<2, Int>,
    w: [&Tensor<2, Int>; 3],
    lut: &Tensor<1, Int>,
    t: usize,
    span: i32,
) -> (Tensor<1, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>, Tensor<2, Int>) {
    let d = |t: &Tensor<2, Int>| t.clone().into_dispatch();
    let (idx, x, q, k, v, a, zf) = Dispatch::attn_forward(d(ctx), d(emb), d(pos), d(w[0]), d(w[1]), d(w[2]), lut.clone().into_dispatch(), t, span);
    let f = |t| Tensor::<2, Int>::from_dispatch(t);
    (Tensor::from_dispatch(idx), f(x), f(q), f(k), f(v), f(a), f(zf))
}

/// Attention weights (Q12) `[b, t, t]` from queries and keys `[b, t, e]`: the integer softmax of
/// `floor(q @ k^T / 2^12)`.
pub fn attn_probs(q: Tensor<3, Int>, k: Tensor<3, Int>, lut: &Tensor<1, Int>) -> Tensor<3, Int> {
    use crate::fixed::{LOGIT_SPAN, shr_floor, softmax_q12};
    if on_gpu(&q) {
        return Tensor::from_dispatch(Dispatch::attn_probs(q.into_dispatch(), k.into_dispatch(), lut.clone().into_dispatch(), LOGIT_SPAN));
    }
    softmax_q12(lut, shr_floor(q.matmul(k.transpose()), 12)).1
}

/// The error at the attention scores from the error `dz` at the mixed values `a @ v`.
pub fn attn_ds(dz: Tensor<3, Int>, v: Tensor<3, Int>, a: Tensor<3, Int>, err: i32) -> Tensor<3, Int> {
    use crate::fixed::shr_floor;
    if on_gpu(&a) {
        return Tensor::from_dispatch(Dispatch::attn_ds(dz.into_dispatch(), v.into_dispatch(), a.into_dispatch(), err));
    }
    // |dz @ v^T| <= 64 * 2^12 * 2^10 = 2^28
    let da = shr_floor(dz.matmul(v.transpose()), 10).clamp(-err, err);
    // softmax: ds = a * (da - sum_j a_j da_j); |sum| <= 2^12 * 2^12 after the row sums to 2^12
    let mean = shr_floor((a.clone() * da.clone()).sum_dim(2), 12);
    // |a * (da - mean)| <= 2^12 * 2^13 = 2^25
    shr_floor(a * (da - mean), 12).clamp(-err, err)
}

/// Per stream, `clamp(floor((M @ n) / 2^shift), -bound, bound)`, with `m` transposed when `mt`.
pub fn small_matmul(m: Tensor<3, Int>, n: Tensor<3, Int>, mt: bool, shift: u32, bound: i32) -> Tensor<3, Int> {
    if on_gpu(&n) {
        return Tensor::from_dispatch(Dispatch::small_matmul(m.into_dispatch(), n.into_dispatch(), mt, shift, bound));
    }
    let m = if mt { m.transpose() } else { m };
    crate::fixed::shr_floor(m.matmul(n), shift).clamp(-bound, bound)
}

/// [`small_matmul`], then with a `gate` as in [`matmul_gated`].
pub fn small_matmul_gated(m: Tensor<3, Int>, n: Tensor<3, Int>, mt: bool, shift: u32, bound: i32, gate: Option<(&Tensor<3, Int>, i32)>) -> Tensor<3, Int> {
    match gate {
        None => small_matmul(m, n, mt, shift, bound),
        Some((g, act_max)) if on_gpu(&n) => Tensor::from_dispatch(Dispatch::small_matmul_gated(
            m.into_dispatch(),
            n.into_dispatch(),
            g.clone().into_dispatch(),
            mt,
            shift,
            bound,
            act_max,
        )),
        Some((g, act_max)) => small_matmul(m, n, mt, shift, bound) * unclamped(g, act_max),
    }
}

/// `x^T @ y` summed over rows in chunks of `chunk`, then averaged over the chunks, so the result
/// is on a per-`chunk`-rows scale whatever the row count. Each chunk sum must fit in 2^30.
pub fn chunked_xty(x: Tensor<2, Int>, y: Tensor<2, Int>, chunk: usize) -> Tensor<2, Int> {
    let ([n, a], [_, b]) = (x.dims(), y.dims());
    let chunks = n / chunk;
    debug_assert!(n % chunk == 0 && chunks.is_power_of_two());
    if on_gpu(&x) {
        let per: Tensor<3, Int> = Tensor::from_dispatch(Dispatch::chunked_xty(x.into_dispatch(), y.into_dispatch(), chunk, chunks.trailing_zeros()));
        return per.sum_dim(0).reshape([a, b]);
    }
    let per = x.reshape([chunks, chunk, a]).transpose().matmul(y.reshape([chunks, chunk, b]));
    crate::fixed::shr_floor(per, chunks.trailing_zeros()).sum_dim(0).reshape([a, b])
}

/// A weight group's new value from its gradient `x^T y`; see [`GpuOps::xty_sgd`].
pub fn xty_sgd(x: Tensor<2, Int>, y: Tensor<2, Int>, w: &Tensor<2, Int>, chunk: usize, (rates, slot): (&Tensor<1, Int>, usize), max: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::xty_sgd(x.into_dispatch(), y.into_dispatch(), w.clone().into_dispatch(), chunk, rates.clone().into_dispatch(), slot, max))
}

/// An embedding or position table's new value from the error at the embedded rows; see
/// [`GpuOps::embed_sgd`].
pub fn embed_sgd(idx: &Tensor<1, Int>, dx: &Tensor<2, Int>, w: &Tensor<2, Int>, t: usize, by_place: bool, rshift: u32, (rates, slot): (&Tensor<1, Int>, usize), max: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::embed_sgd(idx.clone().into_dispatch(), dx.clone().into_dispatch(), w.clone().into_dispatch(), t, by_place, rshift, rates.clone().into_dispatch(), slot, max))
}

/// One weight group's SGD step on the GPU: `clamp(w - round(d / 2^shift) * up, -max, max)`.
/// Callers check [`on_gpu`] first and keep their own tensor code for the CPU.
pub fn sgd_step(w: &Tensor<2, Int>, d: Tensor<2, Int>, (rates, slot): (&Tensor<1, Int>, usize), max: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::sgd_step(w.clone().into_dispatch(), d.into_dispatch(), rates.clone().into_dispatch(), slot, max))
}

/// The embedded context on the GPU, `[n, e]`: each row's byte embedding plus the embedding of
/// its place in the context, both taken from Q16 shadows.
pub fn embed(emb: &Tensor<2, Int>, pos: &Tensor<2, Int>, idx: &Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::embed(emb.clone().into_dispatch(), pos.clone().into_dispatch(), idx.clone().into_dispatch()))
}

/// What the coder needs for each row's true byte, on the GPU; see [`crate::fixed::coded`].
pub fn coded(e: Tensor<2, Int>, target: Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::coded(e.into_dispatch(), target.into_dispatch()))
}

/// `prob - scale * onehot(target)` on the GPU.
pub fn onehot_error(prob: Tensor<2, Int>, target: Tensor<1, Int>, scale: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::onehot_error(prob.into_dispatch(), target.into_dispatch(), scale))
}

/// GPU forms of the tree head and the smaller backward pieces. Callers check [`on_gpu`] and keep
/// their tensor code for the CPU; each is documented on [`GpuOps`].
pub fn tree_probs(z: Tensor<2, Int>, sigmoid: &Tensor<1, Int>, span: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::tree_probs(z.into_dispatch(), sigmoid.clone().into_dispatch(), span))
}

pub fn tree_leaves(p: &Tensor<2, Int>, leaf: &Tensor<1, Int>, guess: &Tensor<2, Int>, copy: &Tensor<2, Int>, lut: &Tensor<1, Int>, matcher: bool, span: i32) -> Tensor<2, Int> {
    let d = |t: &Tensor<2, Int>| t.clone().into_dispatch();
    Tensor::from_dispatch(Dispatch::tree_leaves(d(p), leaf.clone().into_dispatch(), d(guess), d(copy), lut.clone().into_dispatch(), matcher, span))
}

pub fn tree_error(p: &Tensor<2, Int>, target: &Tensor<1, Int>, leaf: &Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::tree_error(p.clone().into_dispatch(), target.clone().into_dispatch(), leaf.clone().into_dispatch()))
}

/// The copy gain's gradient `[3, 1]`: the proposed byte's error summed per trust level.
pub fn copy_error(weights: &Tensor<2, Int>, target: &Tensor<1, Int>, guess: &Tensor<2, Int>, normalise: bool, shift: u32) -> Tensor<2, Int> {
    let at_guess = Dispatch::guess_error(weights.clone().into_dispatch(), target.clone().into_dispatch(), guess.clone().into_dispatch(), normalise);
    Tensor::from_dispatch(Dispatch::level_sums(at_guess, guess.clone().into_dispatch(), shift))
}

pub fn relu_error(dh: Tensor<2, Int>, h: &Tensor<2, Int>, act_max: i32, err: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::relu_error(dh.into_dispatch(), h.clone().into_dispatch(), act_max, err))
}

pub fn place_sums(dx: &Tensor<2, Int>, t: usize, shift: u32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::place_sums(dx.clone().into_dispatch(), t, shift))
}

/// GPU forms of the mix's shared head ([`crate::mix::Mix`]); each is documented on [`GpuOps`].
pub fn tree_log_scores(p: &Tensor<2, Int>, leaf: &Tensor<1, Int>, logp: &Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::tree_log_scores(p.clone().into_dispatch(), leaf.clone().into_dispatch(), logp.clone().into_dispatch()))
}

pub fn mix_weights(total: Tensor<2, Int>, guess: &Tensor<2, Int>, copy: &Tensor<2, Int>, lut: &Tensor<1, Int>, matcher: bool, span: i32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::mix_weights(total.into_dispatch(), guess.clone().into_dispatch(), copy.clone().into_dispatch(), lut.clone().into_dispatch(), matcher, span))
}

pub fn mix_error(e: &Tensor<2, Int>, target: Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::mix_error(e.clone().into_dispatch(), target.into_dispatch()))
}

pub fn guess_sums(g: &Tensor<2, Int>, guess: &Tensor<2, Int>, shift: u32) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::guess_sums(g.clone().into_dispatch(), guess.clone().into_dispatch(), shift))
}

pub fn tree_back(g: Tensor<2, Int>, prob: &Tensor<2, Int>, byte_at: &Tensor<1, Int>) -> Tensor<2, Int> {
    Tensor::from_dispatch(Dispatch::tree_back(g.into_dispatch(), prob.clone().into_dispatch(), byte_at.clone().into_dispatch()))
}

/// Per-op timings on the GPU, for tuning the kernels above. Each op is queued many times before
/// one wait, so the figure is the larger of the host's cost to issue it and the GPU's to run it.
///   GMINT_TEST_DEVICE=cuda cargo test --release kernel_timings -- --ignored --nocapture
#[cfg(test)]
mod timing {
    use super::*;
    use burn::tensor::{Device, TensorData};
    use std::time::Instant;

    fn ints(device: &Device, shape: [usize; 2], span: i32, seed: u64) -> Tensor<2, Int> {
        let mut rng = crate::fixed::Rng(seed);
        Tensor::from_data(TensorData::new(rng.ints(shape[0] * shape[1], span), shape), device)
    }

    /// A one-group optimizer step on the device: (shift, up).
    fn rates(device: &Device, (shift, up): (u32, i32)) -> Tensor<1, Int> {
        Tensor::from_data(TensorData::new(vec![shift as i32, up], [2]), device)
    }

    /// Issue-or-run time per call (queued, so the larger of host and GPU), then GPU time alone:
    /// the same calls captured as a CUDA graph and replayed.
    fn time(name: &str, device: &Device, mut op: impl FnMut()) {
        for _ in 0..20 {
            op();
        }
        let _ = device.sync();
        let reps = 2000;
        let start = Instant::now();
        for _ in 0..reps {
            op();
        }
        let _ = device.sync();
        let queued = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
        let probe = Tensor::<1, Int>::zeros([1], device);
        let inner = 100;
        let gpu = graph(&probe, Graph::Prepare)
            .map(|_| (0..inner).for_each(|_| op()))
            .and_then(|_| graph(&probe, Graph::Start))
            .map(|_| (0..inner).for_each(|_| op()))
            .and_then(|_| graph(&probe, Graph::Stop))
            .and_then(|_| {
                let _ = device.sync();
                let start = Instant::now();
                for _ in 0..10 {
                    graph(&probe, Graph::Replay)?;
                }
                let _ = device.sync();
                Ok(start.elapsed().as_secs_f64() * 1e6 / (10 * inner) as f64)
            });
        let _ = graph(&probe, Graph::Drop);
        match gpu {
            Ok(g) => println!("{name:<40} {queued:8.2} us queued {g:8.2} us on the GPU"),
            Err(e) => println!("{name:<40} {queued:8.2} us queued (graph: {e})"),
        }
    }

    #[test]
    #[ignore]
    fn mma_matches_tiled() {
        let device = crate::fixed::test_device();
        for (n, k, m, wt, xs, ws) in [(512, 256, 256, false, 1023, 1 << 18), (512, 255, 256, true, 4096, 1 << 18), (4096, 32, 32, false, 1023, 1 << 18), (512, 256, 255, false, 4095, (1 << 18) - 1), (100, 70, 50, true, 4095, 1 << 18)] {
            let x = ints(&device, [n, k], xs, n as u64 + 1);
            let w = if wt { ints(&device, [m, k], ws, 77) } else { ints(&device, [k, m], ws, 78) };
            for shift in [0u32, 8, 12] {
                let o = Mat { wt, wshift: 8, shift, bound: UNBOUNDED };
                set_mma(false);
                let want: Vec<i32> = matmul(x.clone(), w.clone(), o).into_data().convert::<i32>().try_to_vec().unwrap();
                set_mma(true);
                let got: Vec<i32> = matmul(x.clone(), w.clone(), o).into_data().convert::<i32>().try_to_vec().unwrap();
                assert!(want == got, "n {n} k {k} m {m} wt {wt} shift {shift}: tensor-core matmul differs");
            }
        }
    }

    #[test]
    #[ignore]
    fn xty_mma_matches_tiled() {
        let device = crate::fixed::test_device();
        for (n, a, b, xs, ys) in [(512, 256, 255, 1023, 4096), (4096, 32, 32, 256, 4095), (512, 64, 256, 1023, 4095), (1024, 32, 32, 255, 4095), (512, 256, 256, 1023, 4095)] {
            let x = ints(&device, [n, a], xs, n as u64 + 3);
            let y = ints(&device, [n, b], ys, b as u64 + 5);
            // the step writes into the weights, so each run gets its own copy
            let w = || ints(&device, [a, b], 1 << 18, 9);
            for step in [(14u32, 1i32), (0, 4), (20, 1)] {
                let r = rates(&device, step);
                set_mma(false);
                let want: Vec<i32> = xty_sgd(x.clone(), y.clone(), &w(), 128, (&r, 0), (1 << 18) - 1).into_data().convert::<i32>().try_to_vec().unwrap();
                set_mma(true);
                let got: Vec<i32> = xty_sgd(x.clone(), y.clone(), &w(), 128, (&r, 0), (1 << 18) - 1).into_data().convert::<i32>().try_to_vec().unwrap();
                assert!(want == got, "n {n} a {a} b {b} step {step:?}: tensor-core gradient differs");
            }
        }
    }

    /// A whole coding step of the best configuration (attention with a tree head, a 2-byte
    /// branch, the match finder), timed on the GPU as a replayed graph.
    #[test]
    #[ignore]
    fn step_timings() {
        use crate::fixed::{Arch, Config, Head, Predictor};
        let device = crate::fixed::test_device();
        let cfg = Config { arch: Arch::Attn, ctx: 8, emb: 32, hid: 256, batch: 512, lr: 14, rule: crate::optim::Rule::Auto, head: Head::Tree, matcher: true, comps: 8, replicas: 8, short: 2, adapt: 0, warm: 0, norm: false, mask: true, seed: 7 };
        let mut model = crate::mix::Mix::new(cfg, &device);
        let ctx = ints(&device, [512, 8], 127, 1).abs();
        let guess = Tensor::<2, Int>::cat(vec![ints(&device, [512, 1], 127, 2).abs(), ints(&device, [512, 1], 1, 3).abs()], 1);
        let target = ints(&device, [512, 1], 127, 4).abs().reshape([512]);
        for _ in 0..8 {
            let _ = model.predict_coded(ctx.clone(), guess.clone(), target.clone());
            model.learn(target.clone());
        }
        time("full step", &device, || {
            let _ = model.predict_coded(ctx.clone(), guess.clone(), target.clone());
            model.learn(target.clone());
        });
        time("forward only", &device, || {
            let _ = model.predict_coded(ctx.clone(), guess.clone(), target.clone());
        });
    }

    #[test]
    #[ignore]
    fn kernel_timings() {
        let device = crate::fixed::test_device();
        assert!(on_gpu(&Tensor::<1, Int>::zeros([1], &device)), "set GMINT_TEST_DEVICE=cuda");
        let (b, t, e, h) = (512, 8, 32, 256);
        let lut = Tensor::<1, Int>::from_data(TensorData::new(crate::fixed::exp_lut(), [crate::fixed::LOGIT_SPAN as usize + 1]), &device);
        let ctx = ints(&device, [b, t], 0, 1).abs() + ints(&device, [b, t], 127, 2).abs();
        let (emb, pos) = (ints(&device, [256, e], 1 << 14, 3), ints(&device, [t, e], 1 << 12, 4));
        let w = [ints(&device, [e, e], 1 << 14, 5), ints(&device, [e, e], 1 << 14, 6), ints(&device, [e, e], 1 << 14, 7)];
        let flat = ints(&device, [b * t, e], 255, 8);
        let zf = ints(&device, [b, t * e], 1023, 9);
        let (w1, w2) = (ints(&device, [t * e, h], 1 << 14, 10), ints(&device, [h, 255], 1 << 14, 11));
        let g = ints(&device, [b, 255], 4095, 12);
        let layer = Mat { wt: false, wshift: 8, shift: 8, bound: UNBOUNDED };
        let back = Mat { wt: true, ..layer };
        let step = rates(&device, (14, 1));
        let (w3, d3) = (Tensor::zeros([3, 1], &device), Tensor::zeros([3, 1], &device));
        time("noop: empty op (sgd_step on [3,1])", &device, || {
            let _ = sgd_step(&w3, d3.clone(), (&step, 0), 9);
        });
        time("attn_forward t8 e32", &device, || {
            let _ = attn_forward(&ctx, &emb, &pos, [&w[0], &w[1], &w[2]], &lut, t, crate::fixed::LOGIT_SPAN);
        });
        time("matmul x[4096,32] w[32,32]", &device, || {
            let _ = matmul(flat.clone(), w[0].clone(), layer);
        });
        time("matmul zf[512,256] w1[256,256]", &device, || {
            let _ = matmul(zf.clone(), w1.clone(), layer);
        });
        time("matmul h[512,256] w2[256,255]", &device, || {
            let _ = matmul(zf.clone(), w2.clone(), layer);
        });
        time("matmul g[512,255] w2^T", &device, || {
            let _ = matmul(g.clone(), w2.clone(), back);
        });
        time("xty_sgd zf^T g -> [256,255]", &device, || {
            let _ = xty_sgd(zf.clone(), g.clone(), &w2, 128, (&step, 0), (1 << 18) - 1);
        });
        time("xty_sgd x^T dq -> [32,32]", &device, || {
            let _ = xty_sgd(flat.clone(), flat.clone(), &w[0], 128, (&step, 0), (1 << 18) - 1);
        });
        for n in [12usize, 4096, 49152, 1 << 20] {
            let big = Tensor::<1, Int>::zeros([n / 4], &device);
            let _ = device.sync();
            let start = Instant::now();
            for _ in 0..50 {
                let v: Vec<i32> = big.clone().into_data().as_slice::<i32>().unwrap().to_vec();
                assert_eq!(v.len(), n / 4);
            }
            println!("read back {n:>8} bytes {:10.1} us", start.elapsed().as_secs_f64() * 1e6 / 50.0);
        }
        let (wq2, wk2, wv2) = (w[0].clone(), w[1].clone(), w[2].clone());
        let steps3 = Tensor::<1, Int>::from_data(TensorData::new(vec![0, 0, 0, 0, 3, 1, 3, 1, 5, 1], [10]), &device);
        time("qkv_sgd [4096,32]", &device, || {
            let _ = qkv_sgd(&flat, [&flat, &flat, &flat], [&wq2, &wk2, &wv2], 128, &steps3, 2, (1 << 18) - 1);
        });
        let flat2 = ints(&device, [b * 2, e], 255, 8);
        time("qkv_sgd [1024,32]", &device, || {
            let _ = qkv_sgd(&flat2, [&flat2, &flat2, &flat2], [&wq2, &wk2, &wv2], 128, &steps3, 2, (1 << 18) - 1);
        });
        let a8 = ints(&device, [b * t, t], 4095, 13).abs().reshape([b, t, t]);
        let dz3 = ints(&device, [b * t, e], 4095, 14).reshape([b, t, e]);
        time("small_matmul dv", &device, || {
            let _ = small_matmul(a8.clone(), dz3.clone(), true, 12, 4095);
        });
        time("attn_ds", &device, || {
            let _ = attn_ds(dz3.clone(), dz3.clone(), a8.clone(), 4095);
        });
        let h = ints(&device, [b, h], 1023, 15).abs();
        time("relu_error", &device, || {
            let _ = relu_error(zf.clone(), &h, 1023, 4095);
        });
        let gb = ints(&device, [b, 256], 4095, 16);
        let pb = ints(&device, [b, 255], 4095, 17).abs();
        let byte_at = Tensor::<1, Int>::from_data(TensorData::new((0..256).collect::<Vec<i32>>(), [256]), &device);
        time("tree_back", &device, || {
            let _ = tree_back(gb.clone(), &pb, &byte_at);
        });
        let x3 = flat.clone().reshape([b * t, e]);
        time("add + clamp [4096,32]", &device, || {
            let _ = (x3.clone() + x3.clone()).clamp(-4095, 4095);
        });
        let idx = ctx.clone().reshape([b * t]);
        time("embed_sgd emb", &device, || {
            let _ = embed_sgd(&idx, &flat, &emb, t, false, 5, (&step, 0), (1 << 15) - 1);
        });
        // a skewed context, as real data has: half the bytes are 0
        let skew = (ctx.clone() * (ctx.clone() % 2)).reshape([b * t]);
        time("embed_sgd emb, skewed bytes", &device, || {
            let _ = embed_sgd(&skew, &flat, &emb, t, false, 5, (&step, 0), (1 << 15) - 1);
        });
        let steps2 = Tensor::<1, Int>::from_data(TensorData::new(vec![3, 1, 3, 1], [4]), &device);
        time("embed_pos_sgd, skewed bytes", &device, || {
            let _ = embed_pos_sgd(&skew, &flat, &emb, &pos, 5, &steps2, [0, 1], (1 << 15) - 1);
        });
    }
}
