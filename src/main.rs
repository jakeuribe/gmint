//! gmint: general minification transformer. An online-trained, integer-only byte model driving a
//! range coder; the decoder retrains the same model, bit for bit, as it decodes.

mod codec;
mod coder;
mod fixed;
mod ops;
mod optim;
mod spn;

use std::collections::HashMap;
use std::process::exit;

use burn::tensor::{Device, IntDType};

use codec::{Options, Prior, Stats};
use fixed::Config;

const USAGE: &str = "usage:
  gmint c <in> <out> [options]     compress
  gmint d <in> <out> [--device D]  decompress
  gmint train <out.prior> <samples...> [options]   pre-train starting weights on sample files
  gmint bench <in> [options]       compress once per --lcr value and report ratio, speed and LCR
  gmint probe <in>                 check integer ops and a cuda-encode / cpu-decode round trip

options:
  --device cuda|flex   backend (default cuda)
  --batch N            streams coded per step, power of two >= 128 (default 512)
  --model attn|spn     predictor: attention model, or random sum-product network (default attn)
  --ctx N --emb N --hid N   model shape (attn default 8, 32, 256; for spn --emb is
                       distributions per region and --hid is replicas, default 4 and 4)
  --head softmax|tree|rtree   output layer: 256-way softmax, bit tree, or randomly permuted tree
  --match on|off       propose bytes from long-range repeats (default off; softmax head only)
  --opt auto|sgd|norm  update rule (default auto: SGD whose rate follows the coded cost)
  --lr N               learning rate, 6..=22; each +1 halves the step (default 14)
  --lcr W[,W...]       latency weight of the update controller, 0..=256 (default 16)
  --prior FILE         start from pre-trained weights; the same file is needed to decompress
  --epochs N           passes over the samples when training (default 2)
  --limit BYTES        only use the first BYTES of the input
  --seed N";

fn device(name: &str) -> Device {
    let mut d = match name {
        "cuda" => Device::cuda(0),
        "flex" => Device::flex(),
        other => fail(&format!("unknown device {other}")),
    };
    // Every tensor in the model is i32. A second configure of the same device in one process
    // reports AlreadyInitialized, which is the state we want anyway.
    let _ = d.configure(IntDType::I32);
    d
}

fn fail(msg: &str) -> ! {
    eprintln!("gmint: {msg}");
    exit(2)
}

struct Args {
    pos: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    fn parse() -> Self {
        let (mut pos, mut flags) = (Vec::new(), HashMap::new());
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.strip_prefix("--") {
                Some(name) => {
                    let v = it.next().unwrap_or_else(|| fail(&format!("--{name} needs a value")));
                    flags.insert(name.to_string(), v);
                }
                None => pos.push(a),
            }
        }
        Self { pos, flags }
    }
    fn num(&self, name: &str, default: usize) -> usize {
        match self.flags.get(name) {
            Some(v) => v.parse().unwrap_or_else(|_| fail(&format!("--{name}: not a number: {v}"))),
            None => default,
        }
    }
    fn device(&self) -> Device {
        device(self.flags.get("device").map(String::as_str).unwrap_or("cuda"))
    }
    fn lcr_weights(&self) -> Vec<u16> {
        let raw = self.flags.get("lcr").map(String::as_str).unwrap_or("16");
        raw.split(',')
            .map(|w| match w.parse::<u16>() {
                Ok(w) if w <= 256 => w,
                _ => fail(&format!("--lcr: expected 0..=256, got {w}")),
            })
            .collect()
    }
    fn prior(&self) -> Option<Prior> {
        self.flags.get("prior").map(|p| Prior::load(p).unwrap_or_else(|e| fail(&e)))
    }
    /// The model shape comes from the prior when there is one.
    fn model(&self) -> Config {
        let shape = self.prior().map(|p| (p.ctx, p.emb, p.hid, p.head, p.matcher));
        let matcher = match self.flags.get("match").map(String::as_str) {
            Some("on") => true,
            Some("off") | None => false,
            Some(v) => fail(&format!("--match: expected on or off, got {v}")),
        };
        let head = match self.flags.get("head") {
            Some(name) => fixed::Head::from_name(name).unwrap_or_else(|| fail(&format!("--head: unknown head {name}"))),
            None => fixed::Head::Softmax,
        };
        let arch = match self.flags.get("model") {
            Some(name) => fixed::Arch::from_name(name).unwrap_or_else(|| fail(&format!("--model: unknown model {name}"))),
            None => fixed::Arch::Attn,
        };
        // for the sum-product network these are distributions per region and replicas
        let (emb, hid) = if arch == fixed::Arch::Spn { (4, 4) } else { (32, 256) };
        let cfg = Config {
            arch,
            ctx: shape.map_or(self.num("ctx", 8), |s| s.0),
            emb: shape.map_or(self.num("emb", emb), |s| s.1),
            hid: shape.map_or(self.num("hid", hid), |s| s.2),
            batch: self.num("batch", 512),
            lr: self.num("lr", 14) as u32,
            rule: match self.flags.get("opt").map(String::as_str).unwrap_or("auto") {
                name => optim::Rule::from_name(name).unwrap_or_else(|| fail(&format!("--opt: unknown rule {name}"))),
            },
            head: shape.map_or(head, |s| s.3),
            matcher: shape.map_or(matcher, |s| s.4),
            seed: self.num("seed", 7) as u64,
        };
        cfg.validate().unwrap_or_else(|e| fail(&e));
        cfg
    }
    fn input(&self) -> Vec<u8> {
        let path = self.pos.get(1).unwrap_or_else(|| fail(USAGE));
        let mut data = std::fs::read(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
        data.truncate(self.num("limit", data.len()));
        data
    }
}

fn report(what: &str, bytes_in: usize, bytes_out: usize, s: &Stats) {
    println!(
        "{what}: {bytes_in} -> {bytes_out} bytes, ratio {:.4}, {:.1}s, {:.0} KB/s, {} of {} steps trained",
        bytes_in as f64 / bytes_out.max(1) as f64,
        s.seconds,
        bytes_in.max(bytes_out) as f64 / 1024.0 / s.seconds.max(1e-9),
        s.updates,
        s.steps
    );
}

fn main() {
    let args = Args::parse();
    match args.pos.first().map(String::as_str) {
        Some("c") if args.pos.len() == 3 => {
            let data = args.input();
            let o = Options { model: args.model(), lcr_weight: args.lcr_weights()[0] };
            let (out, stats) = codec::compress(&data, &o, args.prior().as_ref(), &args.device(), true).unwrap_or_else(|e| fail(&e));
            std::fs::write(&args.pos[2], &out).unwrap_or_else(|e| fail(&e.to_string()));
            report("compressed", data.len(), out.len(), &stats);
            for (done, size) in &stats.curve {
                println!("  after {done:>11} bytes: ratio {:.4}", *done as f64 / *size as f64);
            }
        }
        Some("d") if args.pos.len() == 3 => {
            let file = args.input();
            let (data, stats) = codec::decompress(&file, args.prior().as_ref(), &args.device(), true).unwrap_or_else(|e| fail(&e));
            std::fs::write(&args.pos[2], &data).unwrap_or_else(|e| fail(&e.to_string()));
            report("decompressed", file.len(), data.len(), &stats);
        }
        Some("train") if args.pos.len() >= 3 => {
            let (dev, cfg, epochs) = (args.device(), args.model(), args.num("epochs", 2));
            let mut model = fixed::Model::new(cfg, &dev);
            if let Some(p) = args.prior() {
                model.import(&p.values).unwrap_or_else(|e| fail(&e));
            }
            let limit = args.num("limit", usize::MAX);
            let samples: Vec<Vec<u8>> = args.pos[2..]
                .iter()
                .map(|p| {
                    let mut d = std::fs::read(p).unwrap_or_else(|e| fail(&format!("{p}: {e}")));
                    d.truncate(limit);
                    d
                })
                .collect();
            let bytes: usize = samples.iter().map(|s| s.len() / cfg.batch * cfg.batch).sum();
            println!("{} samples, {bytes} bytes per epoch, {cfg:?}", samples.len());
            for epoch in 1..=epochs {
                let t0 = std::time::Instant::now();
                let cost: u64 = samples.iter().map(|s| codec::train_pass(&mut model, s, &dev)).sum();
                println!(
                    "epoch {epoch}: {:.4} bits/byte (ratio {:.4}), {:.0}s, rate now {}",
                    cost as f64 / 256.0 / bytes as f64,
                    8.0 * 256.0 * bytes as f64 / cost as f64,
                    t0.elapsed().as_secs_f64(),
                    model.rate()
                );
                Prior::save(&args.pos[1], &model).unwrap_or_else(|e| fail(&e.to_string()));
            }
        }
        Some("bench") if args.pos.len() == 2 => {
            let (data, dev, model) = (args.input(), args.device(), args.model());
            // the first GPU use compiles kernels; keep that out of the first row's timing
            let prior = args.prior();
            let _ = codec::compress(&data[..data.len().min(64 << 10)], &Options { model, lcr_weight: 0 }, None, &dev, false);
            println!("{} bytes, {model:?}", data.len());
            println!("{:>4}  {:>9}  {:>8}  {:>8}  {:>8}  {:>10}", "lcr", "bytes", "ratio", "seconds", "trained", "LCR s/ratio");
            let mut base: Option<(f64, f64)> = None;
            for w in args.lcr_weights() {
                let (out, s) = codec::compress(&data, &Options { model, lcr_weight: w }, prior.as_ref(), &dev, false).unwrap_or_else(|e| fail(&e));
                let ratio = data.len() as f64 / out.len() as f64;
                // TRACE eq. 5: latency added per unit of compression ratio gained, against the first row
                let lcr = match base {
                    Some((t0, r0)) if (ratio - r0).abs() > 1e-9 => format!("{:.2}", (s.seconds - t0) / (ratio - r0)),
                    _ => "-".into(),
                };
                base = base.or(Some((s.seconds, ratio)));
                println!(
                    "{w:>4}  {:>9}  {ratio:>8.4}  {:>8.1}  {:>7.0}%  {lcr:>10}",
                    out.len(),
                    s.seconds,
                    100.0 * s.updates as f64 / s.steps.max(1) as f64
                );
            }
        }
        Some("probe") if args.pos.len() == 2 => {
            let (cuda, flex) = (device("cuda"), device("flex"));
            println!("integer ops, flex vs cuda:");
            let bad = ops::compare(&flex, &cuda);
            println!("{} op(s) disagree: {bad:?}\n", bad.len());

            let mut data = args.input();
            data.truncate(args.num("limit", 96 << 10));
            let o = Options { model: Config { batch: 128, ..args.model() }, lcr_weight: args.lcr_weights()[0] };
            let (on_cuda, s) = codec::compress(&data, &o, None, &cuda, false).unwrap();
            report("cuda encode", data.len(), on_cuda.len(), &s);
            let (on_flex, s) = codec::compress(&data, &o, None, &flex, false).unwrap();
            report("flex encode", data.len(), on_flex.len(), &s);
            println!("encodings identical: {}", on_cuda == on_flex);
            match codec::decompress(&on_cuda, None, &flex, false) {
                Ok((back, _)) => println!("cuda-encoded file decoded on flex: {}", if back == data { "exact" } else { "WRONG" }),
                Err(e) => println!("cuda-encoded file decoded on flex: FAILED ({e})"),
            }
        }
        _ => fail(USAGE),
    }
}
