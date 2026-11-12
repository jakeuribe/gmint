//! Op-by-op agreement between two backends on the integer ops the fixed-point model uses.
//! A mismatch here names the op that would break bit-identical encode/decode.

use burn::tensor::ops::IndexingUpdateOp;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::fixed::{Rng, shr_floor};

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
    let mut bad = Vec::new();
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
