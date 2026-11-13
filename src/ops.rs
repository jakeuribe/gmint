//! Op-by-op agreement between two backends on the integer ops the fixed-point model uses.
//! A mismatch here names the op that would break bit-identical encode/decode.

use burn::tensor::ops::IndexingUpdateOp;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::{Rng, shr_floor};
use crate::fused::{Mat, UNBOUNDED, matmul as fused};

const R: usize = 96;
const K: usize = 128;
const C: usize = 160;

struct Inputs {
    x: Vec<i32>,
    w: Vec<i32>,
    pos: Vec<i32>,
    idx: Vec<i32>,
}

fn out<const D: usize>(t: Tensor<D, Int>) -> Vec<i32> {
    t.into_data().convert::<i32>().try_to_vec().unwrap()
}

fn run(i: &Inputs, d: &Device) -> Vec<(&'static str, Vec<i32>)> {
    let t2 = |v: &[i32], s: [usize; 2]| Tensor::<2, Int>::from_data(TensorData::new(v.to_vec(), s), d);
    let x = || t2(&i.x, [R, K]);
    let w = || t2(&i.w, [K, C]);
    let pos = || t2(&i.pos, [R, C]);
    let idx = || Tensor::<1, Int>::from_data(TensorData::new(i.idx.clone(), [R]), d);
    vec![
        ("matmul", out(x().matmul(w()))),
        ("matmul batched", out(x().reshape([4, R / 4, K]).matmul(w().reshape([4, K, C / 4])))),
        ("transpose matmul", out(x().transpose().matmul(pos()))),
        ("raw shr negative", out(x().bitwise_right_shift_scalar(3))),
        ("shr_floor negative", out(shr_floor(x(), 3))),
        ("add scalar", out(x() + 5)),
        ("add broadcast", out(x().reshape([4, R / 4, K]) + t2(&i.w[..R / 4 * K], [R / 4, K]).unsqueeze::<3>())),
        ("abs max", out(x().abs().max())),
        ("div by scalar tensor", out(x() / (x().abs().max().bitwise_right_shift_scalar(4) + 1).unsqueeze::<2>())),
        ("clamp", out(x().clamp(-300, 300))),
        ("max_dim", out(x().max_dim(1))),
        ("sum_dim", out(x().sum_dim(1))),
        ("sub broadcast", out(x().max_dim(1) - x())),
        ("div broadcast", out(pos() * 4096 / pos().sum_dim(1).bitwise_right_shift_scalar(4))),
        ("div negative", out(x() / t2(&i.pos[..R * K], [R, K]))),
        ("gather", out(x().gather(1, idx().reshape([R, 1])))),
        ("select rows", out(x().select(0, idx()))),
        (
            "select_assign add",
            out(Tensor::<2, Int>::zeros([64, K], d).select_assign(0, idx(), x(), IndexingUpdateOp::Add)),
        ),
        ("one_hot", out(idx().one_hot::<2>(64) * 4096)),
        ("compare->int", out(x().greater_elem(0).int() * x().lower_elem(500).int())),
        ("composed matmul>>8 clamp", out(shr_floor(x().matmul(w()), 8).clamp(-1023, 1023))),
        ("fused matmul>>8 clamp", out(fused(x(), w(), Mat { wt: false, wshift: 0, shift: 8, bound: 1023 }))),
        (
            "composed rows by value",
            out(crate::fixed::shr_round0(Tensor::<2, Int>::zeros([64, K], d).select_assign(0, idx(), x(), IndexingUpdateOp::Add), 2)),
        ),
        ("fused rows by value", out(crate::fused::rows_by_value(idx(), x(), 64, 2))),
        ("composed raw matmul", out(x().matmul(w()))),
        ("fused raw matmul", out(fused(x(), w(), Mat { wt: false, wshift: 0, shift: 0, bound: UNBOUNDED }))),
        (
            "composed x @ (w>>4)^T >>8 clamp",
            out(shr_floor(x().matmul(shr_floor(w().reshape([C, K]), 4).transpose()), 8).clamp(-4095, 4095)),
        ),
        ("fused x @ (w>>4)^T >>8 clamp", out(fused(x(), w().reshape([C, K]), Mat { wt: true, wshift: 4, shift: 8, bound: 4095 }))),
    ]
}

/// Runs every op on both devices and prints the verdicts; returns the ops that disagree.
pub fn compare(a: &Device, b: &Device) -> Vec<&'static str> {
    let mut rng = Rng(11);
    let i = Inputs {
        x: rng.ints(R * K, 1000),
        w: rng.ints(K * C, 1000),
        pos: rng.ints(R * C, 2000).iter().map(|v| v.abs() + 1).collect(),
        idx: rng.ints(R, 60).iter().map(|v| v.abs()).collect(),
    };
    let mut bad: Vec<&'static str> = Vec::new();
    // every fused kernel against the ordinary ops it replaces, on each device
    for (d, results) in [("first", run(&i, a)), ("second", run(&i, b))] {
        for (name, got) in results.iter().filter(|r| r.0.starts_with("fused ")) {
            let want = results.iter().find(|r| r.0.strip_prefix("composed ") == name.strip_prefix("fused "));
            if want.map(|w| &w.1) != Some(got) {
                println!("  {name:<20} DIFFERS from the composed ops on the {d} device");
                bad.push(name);
            }
        }
    }
    for ((name, va), (_, vb)) in run(&i, a).into_iter().zip(run(&i, b)) {
        match va.iter().zip(&vb).position(|(p, q)| p != q) {
            Some(at) => {
                println!("  {name:<20} MISMATCH at {at}: {} vs {}", va[at], vb[at]);
                bad.push(name);
            }
            None if va.len() != vb.len() => {
                println!("  {name:<20} MISMATCH in length: {} vs {}", va.len(), vb.len());
                bad.push(name);
            }
            None => println!("  {name:<20} ok"),
        }
    }
    bad
}

/// Times the operations one coding step is built from, at batch `b`, so a slow one stands out.
/// Each is run `reps` times and then forced to finish by reading one value back.
pub fn bench(d: &Device, b: usize) {
    use std::time::Instant;
    const V: usize = 256;
    let reps = 200;
    let mut rng = Rng(5);
    let wide = Tensor::<2, Int>::from_data(TensorData::new(rng.ints(b * V, 2000), [b, V]), d);
    let hidden = Tensor::<2, Int>::from_data(TensorData::new(rng.ints(b * V, 1000), [b, V]), d);
    let weight = Tensor::<2, Int>::from_data(TensorData::new(rng.ints(V * V, 1000), [V, V]), d);
    let lut = Tensor::<1, Int>::from_data(TensorData::new((0..4096).collect::<Vec<i32>>(), [4096]), d);
    let target = Tensor::<1, Int>::from_data(TensorData::new(rng.ints(b, 100).iter().map(|v| v.abs()).collect::<Vec<i32>>(), [b]), d);
    let host: Vec<i32> = rng.ints(b * 8, 100).iter().map(|v| v.abs()).collect();

    let time = |name: &str, f: &dyn Fn() -> Tensor<2, Int>| {
        let _ = f().sum().into_data();
        let t0 = Instant::now();
        let mut last = f();
        for _ in 1..reps {
            last = f();
        }
        let _ = last.sum().into_data();
        let queued = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        // again, waiting for the device after every call: what one call costs when nothing else
        // is queued behind it
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = f().sum().into_data();
        }
        println!("  {name:<28} {queued:>8.1} us queued {:>8.1} us alone", t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
    };
    println!("batch {b}, per call:");
    time("upload [B, 8] from host", &|| Tensor::from_data(TensorData::new(host.clone(), [b, 8]), d));
    time("matmul [B,256] x [256,256]", &|| hidden.clone().matmul(weight.clone()));
    time("add scalar [B,256]", &|| wide.clone() + 5);
    time("shr_floor [B,256]", &|| shr_floor(wide.clone(), 3));
    time("clamp [B,256]", &|| wide.clone().clamp(-300, 300));
    time("max_dim [B,256]", &|| wide.clone().max_dim(1));
    time("sum_dim [B,256]", &|| wide.clone().sum_dim(1));
    time("lut select B*256", &|| lut.clone().select(0, wide.clone().abs().reshape([b * V])).reshape([b, V]));
    time("div broadcast [B,256]", &|| wide.clone() / (wide.clone().abs().sum_dim(1) + 1));
    time("one_hot [B] -> [B,256]", &|| target.clone().one_hot::<2>(V));
    time("select rows [256,256] by [B]", &|| weight.clone().select(0, target.clone()));
    time("gather [B,256] by [B,1]", &|| wide.clone().gather(1, target.clone().reshape([b, 1])));
    time("transpose matmul (grad)", &|| hidden.clone().transpose().matmul(wide.clone()));
    time(
        "chunked grad matmul",
        &|| hidden.clone().reshape([b / 128, 128, V]).transpose().matmul(wide.clone().reshape([b / 128, 128, V])).sum_dim(0).reshape([V, V]),
    );
    time("compare -> int [B,256]", &|| wide.clone().greater_elem(0).int());

    // candidates for cheaper forms of the same integer results
    time("raw shr [B,256]", &|| wide.clone().abs().bitwise_right_shift_scalar(3));
    time("abs [B,256]", &|| wide.clone().abs());
    time("div scalar [B,256]", &|| wide.clone().abs() / 8);
    time("floor via div [B,256]", &|| (wide.clone() + (1i32 << 30)) / 8 - (1i32 << 27));
    time("mul scalar [B,256]", &|| wide.clone() * 3);
    time("mul elementwise [B,256]", &|| wide.clone() * hidden.clone());
    time("add elementwise [B,256]", &|| wide.clone() + hidden.clone());
    time("greater_elem only", &|| wide.clone().greater_elem(0).int());
    time("mask via clamp", &|| wide.clone().clamp(0, 1));
    time("sub broadcast [B,1]", &|| wide.clone() - wide.clone().narrow(1, 0, 1));
    time("mul broadcast [1,256]", &|| wide.clone() * weight.clone().narrow(0, 0, 1));
    time("narrow [B,1]", &|| wide.clone().narrow(1, 0, 1) + 1);
    time("select cols [B,256] by [256]", &|| wide.clone().select(1, target.clone().narrow(0, 0, 128)));
    time("lut select B*32", &|| lut.clone().select(0, wide.clone().abs().narrow(1, 0, 32).reshape([b * 32])).reshape([b, 32]));
    time("lut gather-free 4096 rows", &|| lut.clone().reshape([4096, 1]).select(0, target.clone()));
    time("sum_dim(0) [B,256]", &|| wide.clone().sum_dim(0));
    time("cat 3 x [B,1]", &|| Tensor::cat(vec![wide.clone().narrow(1, 0, 1), wide.clone().narrow(1, 1, 1), wide.clone().narrow(1, 2, 1)], 1));
    time("zeros [B,256]", &|| Tensor::<2, Int>::zeros([b, V], d) + 1);
    time("transpose only", &|| wide.clone().transpose() + 1);

    // do chains of elementwise ops merge into one kernel (burn's fusion feature)?
    time("chain x12: add mul clamp", &|| {
        let mut t = wide.clone();
        for _ in 0..4 {
            t = ((t + 3) * 3).clamp(-100_000, 100_000);
        }
        t
    });
    time("chain x12: add div clamp", &|| {
        let mut t = wide.clone();
        for _ in 0..4 {
            t = ((t + 3) / 2).clamp(-100_000, 100_000);
        }
        t
    });
    time("chain x12: add shr sub", &|| {
        let mut t = wide.clone();
        for _ in 0..4 {
            t = shr_floor(t, 1);
        }
        t
    });
    let layer = Mat { wt: false, wshift: 0, shift: 8, bound: 1023 };
    time("fused: matmul shr clamp x2", &|| fused(fused(hidden.clone(), weight.clone(), layer), weight.clone(), layer));
    time("composed: matmul shr clamp", &|| shr_floor(hidden.clone().matmul(weight.clone()), 8).clamp(-1023, 1023));
    time("fused: matmul shr clamp", &|| fused(hidden.clone(), weight.clone(), layer));
    time("chain: matmul shr clamp x2", &|| {
        let a = shr_floor(hidden.clone().matmul(weight.clone()), 8).clamp(-1023, 1023);
        shr_floor(a.matmul(weight.clone()), 8).clamp(-1023, 1023)
    });

    // attention-shaped: B small matrices per call, and the embedding gather / scatter
    let (t, e) = (8usize, 32usize);
    let tok = Tensor::<3, Int>::from_data(TensorData::new(rng.ints(b * t * e, 500), [b, t, e]), d);
    let att = Tensor::<3, Int>::from_data(TensorData::new(rng.ints(b * t * t, 500), [b, t, t]), d);
    let small = Tensor::<2, Int>::from_data(TensorData::new(rng.ints(e * e, 500), [e, e]), d);
    let table = Tensor::<2, Int>::from_data(TensorData::new(rng.ints(V * e, 100), [V, e]), d);
    let rows = Tensor::<1, Int>::from_data(TensorData::new(host.clone(), [b * t]), d);
    let flat = |x: Tensor<3, Int>| { let [p, q, r] = x.dims(); x.reshape([p, q * r]) };
    time("embedding select B*8 rows", &|| table.clone().select(0, rows.clone()));
    time("proj [B*8,32] x [32,32]", &|| tok.clone().reshape([b * t, e]).matmul(small.clone()));
    time("batched q k^T [B,8,32]x[B,32,8]", &|| flat(tok.clone().matmul(tok.clone().transpose())));
    time("batched a v [B,8,8]x[B,8,32]", &|| flat(att.clone().matmul(tok.clone())));
    time("batched a^T dz", &|| flat(att.clone().transpose().matmul(tok.clone())));
    time("max_dim last [B,8,8]", &|| flat(att.clone().max_dim(2)));
    time("sum_dim last [B,8,8]", &|| flat(att.clone().sum_dim(2)));
    time("mul + sum_dim [B,8,8]", &|| flat((att.clone() * att.clone()).sum_dim(2)));
    time("chunked grad [B*8,32]", &|| {
        let x = tok.clone().reshape([b * t / 128, 128, e]);
        x.clone().transpose().matmul(x).sum_dim(0).reshape([e, e])
    });
    time("scatter-add B*8 rows -> [256,32]", &|| {
        Tensor::<2, Int>::zeros([V, e], d).select_assign(0, rows.clone(), tok.clone().reshape([b * t, e]), IndexingUpdateOp::Add)
    });
    time("one-hot matmul B*8 rows -> [256,32]", &|| {
        rows.clone().one_hot::<2>(V).transpose().matmul(tok.clone().reshape([b * t, e]))
    });
    time("sum_dim(0) [B,8,32]", &|| flat(tok.clone().sum_dim(0)));
    time("abs().max() [256,256]", &|| weight.clone().abs().max().reshape([1, 1]));
    time("div by scalar tensor", &|| weight.clone() / (weight.clone().abs().max() + 1).unsqueeze::<2>());
}

/// Section timer for `GMINT_PROFILE=1`: [`lap`] waits for the device and charges the time since
/// the previous lap to a named section; [`report`] prints the totals. Off, a lap is one branch.
pub mod prof {
    use std::cell::RefCell;
    use std::time::Instant;

    use burn::tensor::Device;

    struct State {
        last: Instant,
        sections: Vec<(&'static str, f64, u64)>,
    }

    thread_local! {
        static ON: bool = std::env::var_os("GMINT_PROFILE").is_some();
        static STATE: RefCell<State> = RefCell::new(State { last: Instant::now(), sections: Vec::new() });
    }

    pub fn on() -> bool {
        ON.with(|on| *on)
    }

    pub fn lap(device: &Device, name: &'static str) {
        if !on() {
            return;
        }
        let _ = device.sync();
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            let now = Instant::now();
            let spent = (now - s.last).as_secs_f64();
            s.last = now;
            match s.sections.iter_mut().find(|e| e.0 == name) {
                Some(e) => {
                    e.1 += spent;
                    e.2 += 1;
                }
                None => s.sections.push((name, spent, 1)),
            }
        });
    }

    /// Forgets what has been charged so far, so kernel compilation in the first steps is left out.
    pub fn reset() {
        if on() {
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                s.sections.iter_mut().for_each(|e| (e.1, e.2) = (0.0, 0));
                s.last = Instant::now();
            });
        }
    }

    pub fn report() {
        if !on() {
            return;
        }
        STATE.with(|s| {
            let s = s.borrow();
            let total: f64 = s.sections.iter().map(|e| e.1).sum();
            for (name, secs, calls) in &s.sections {
                eprintln!("  {name:<28} {secs:>7.2}s {:>5.1}%  {:>8.1} us/call", 100.0 * secs / total, secs * 1e6 / *calls as f64);
            }
        });
    }
}
