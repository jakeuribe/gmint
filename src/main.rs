//! gmint: general minification transformer. An online-trained, integer-only byte model driving a
//! range coder; the decoder retrains the same model, bit for bit, as it decodes.

mod codec;
mod coder;
mod fixed;
mod fused;
mod mix;
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
  gmint dry <in> [options]         compress without writing anything: ratio, speed, and the
                                   model's health at every tenth of the input
  gmint train <out.prior> <samples...> [options]   pre-train starting weights on sample files
  gmint bench <in> [options]       compress once per --lcr value and report ratio, speed and LCR
  gmint probe <in>                 check integer ops and a cuda-encode / cpu-decode round trip
  gmint opbench                    time the tensor operations a coding step is built from

options:
  --device cuda|flex   backend (default cuda)
  --batch N            streams coded per step, power of two >= 128 (default 512)
  --model attn|spn|both   predictor: attention, random sum-product network, or both with
                       their scores added (default attn; spn and both need --batch <= 512)
  --ctx N --emb N --hid N   attention shape (default 8, 32, 256)
  --comps N --replicas N    sum-product network: distributions per leaf and sum nodes per
                       layer (2..=16), and random replicas (default 8, 8)
  --head softmax|tree|rtree   output layer: 256-way softmax, bit tree, or randomly permuted tree
  --match on|off       propose bytes from long-range repeats (default off)
  --short N            add an attention branch over only the last N bytes (2 or 4)
  --spn-adapt off|prune|grow   prune weak sum-node children, or prune and regrow them
  --warm STEPS         after STEPS steps, train once more on everything coded so far
  --norm on|off        scale each hidden unit's input to a fixed running spread (default off)
  --mask on|off        pass no error back through a forward clamp where it held (default on;
                       off is how files were coded before the option existed)
  --health on          with c: also print the model's health at every tenth of the input
  --scan on            try several configurations on the start of the input, keep the best
  --opt auto|sgd|norm  update rule (default auto: SGD whose rate follows the coded cost)
  --lr N               learning rate, 6..=22; each +1 halves the step (default 14)
  --lcr W[,W...]       latency weight of the update controller, 0..=256 (default 16)
  --prior FILE         start from pre-trained weights; the same file is needed to decompress
  --epochs N           passes over the samples when training (default 2)
  --mix on|off         train on all the samples at once, so every batch mixes them, rather
                       than one sample after another (default on)
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
    /// An on|off flag.
    fn switch(&self, name: &str, default: bool) -> bool {
        match self.flags.get(name).map(String::as_str) {
            Some("on") => true,
            Some("off") => false,
            None => default,
            Some(v) => fail(&format!("--{name}: expected on or off, got {v}")),
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
        let shape = self.prior().map(|p| (p.ctx, p.emb, p.hid, p.head, p.matcher, p.norm, p.mask));
        let (norm, matcher, mask) = (self.switch("norm", false), self.switch("match", false), self.switch("mask", true));
        let head = match self.flags.get("head") {
            Some(name) => fixed::Head::from_name(name).unwrap_or_else(|| fail(&format!("--head: unknown head {name}"))),
            None => fixed::Head::Softmax,
        };
        let arch = match self.flags.get("model") {
            Some(name) => fixed::Arch::from_name(name).unwrap_or_else(|| fail(&format!("--model: unknown model {name}"))),
            None => fixed::Arch::Attn,
        };
        let cfg = Config {
            arch,
            comps: self.num("comps", 8),
            replicas: self.num("replicas", 8),
            short: self.num("short", 0),
            adapt: match self.flags.get("spn-adapt").map(String::as_str) {
                None | Some("off") => 0,
                Some("prune") => 1,
                Some("grow") => 2,
                Some(v) => fail(&format!("--spn-adapt: expected off, prune or grow, got {v}")),
            },
            warm: self.num("warm", 0),
            ctx: shape.map_or(self.num("ctx", 8), |s| s.0),
            emb: shape.map_or(self.num("emb", 32), |s| s.1),
            hid: shape.map_or(self.num("hid", 256), |s| s.2),
            batch: self.num("batch", 512),
            lr: self.num("lr", 14) as u32,
            rule: match self.flags.get("opt").map(String::as_str).unwrap_or("auto") {
                name => optim::Rule::from_name(name).unwrap_or_else(|| fail(&format!("--opt: unknown rule {name}"))),
            },
            head: shape.map_or(head, |s| s.3),
            matcher: shape.map_or(matcher, |s| s.4),
            norm: shape.map_or(norm, |s| s.5),
            mask: shape.map_or(mask, |s| s.6),
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
        Some(cmd @ ("c" | "dry")) if args.pos.len() == if cmd == "c" { 3 } else { 2 } => {
            let dry = cmd == "dry";
            let health = dry || args.flags.get("health").map(String::as_str) == Some("on");
            let data = args.input();
            let mut o = Options { model: args.model(), lcr_weight: args.lcr_weights()[0] };
            if args.flags.get("scan").map(String::as_str) == Some("on") {
                // Pre-scan: the choice is written in the header like any other configuration, so
                // the decoder needs nothing extra.
                let (dev, head_len) = (args.device(), data.len().min(512 << 10));
                let mut best = (usize::MAX, o.model);
                for head in [fixed::Head::Softmax, fixed::Head::Tree] {
                    for matcher in [false, true] {
                        for lr in [12, 14, 16] {
                            let model = Config { head, matcher, lr, ..o.model };
                            if model.validate().is_err() {
                                continue;
                            }
                            let trial = Options { model, lcr_weight: o.lcr_weight };
                            let size = codec::compress(&data[..head_len], &trial, None, &dev, false).map(|r| r.0.len());
                            if let Ok(size) = size {
                                eprintln!("  scan: {head:?} match {matcher} lr {lr}: {size} bytes");
                                if size < best.0 {
                                    best = (size, model);
                                }
                            }
                        }
                    }
                }
                println!("scan chose {:?}, match {}, lr {}", best.1.head, best.1.matcher, best.1.lr);
                o.model = best.1;
            }
            let (out, stats) =
                codec::compress_watched(&data, &o, args.prior().as_ref(), &args.device(), true, health).unwrap_or_else(|e| fail(&e));
            if !dry {
                std::fs::write(&args.pos[2], &out).unwrap_or_else(|e| fail(&e.to_string()));
            }
            report(if dry { "dry run" } else { "compressed" }, data.len(), out.len(), &stats);
            // ratio over everything so far, and over just the tenth that ended here: the second
            // is what shows a model that has stopped learning
            let mut before = (0usize, 0usize);
            for (i, (done, size)) in stats.curve.iter().enumerate() {
                let recent = (*done - before.0) as f64 / (*size - before.1).max(1) as f64;
                println!("  after {done:>11} bytes: ratio {:.4}, this tenth {recent:.4}", *done as f64 / *size as f64);
                for line in stats.health.get(i).map(|h| h.lines()).into_iter().flatten() {
                    println!("      {line}");
                }
                before = (*done, *size);
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
            let count = samples.len();
            // The streams of a training pass are contiguous slices of its data, so a pass per
            // sample trains every batch on one sample: the auto rule then reads each switch to
            // the next sample as the model getting worse and slows down for good, and the prior
            // ends up tuned to the last sample. One pass over all of them joined end to end
            // spreads every batch over all of them.
            let mix = args.switch("mix", true);
            let samples = if mix {
                let mut all = Vec::with_capacity(samples.iter().map(Vec::len).sum());
                samples.into_iter().for_each(|s| all.extend_from_slice(&s));
                vec![all]
            } else {
                samples
            };
            let bytes: usize = samples.iter().map(|s| s.len() / cfg.batch * cfg.batch).sum();
            println!("{count} samples{}, {bytes} bytes per epoch, {cfg:?}", if mix { " mixed" } else { "" });
            for epoch in 1..=epochs {
                let t0 = std::time::Instant::now();
                let (mut cost, mut seen) = (codec::PassCost::default(), 0usize);
                for (n, sample) in samples.iter().enumerate() {
                    // in one long pass, a line at every tenth of it, so the epoch can be watched
                    let (here, mut tenths) = (sample.len() / cfg.batch * cfg.batch, 1);
                    let this = codec::train_pass(&mut model, sample, &dev, |done, so_far, rate| {
                        if samples.len() == 1 && done < here && done * 10 >= here * tenths {
                            tenths = done * 10 / here + 1;
                            let (coded, loss) = so_far.bits_per_byte(done);
                            eprintln!(
                                "  epoch {epoch} at {}%: {coded:.4} bits/byte so far (loss {loss:.4}); rate {rate}, {:.0}s",
                                done * 100 / here,
                                t0.elapsed().as_secs_f64()
                            );
                        }
                    });
                    cost.add(this);
                    seen += here;
                    if samples.len() == 1 {
                        continue;
                    }
                    // otherwise a line per sample: the coded bits and the head's own loss, both
                    // measured before each step's update, over the epoch so far and over this
                    // sample alone
                    let ((coded, loss), (s_coded, s_loss)) = (cost.bits_per_byte(seen), this.bits_per_byte(here));
                    eprintln!(
                        "  epoch {epoch} sample {}/{}: {coded:.4} bits/byte so far (loss {loss:.4}); this sample {s_coded:.4} (loss {s_loss:.4}); rate {}, {:.0}s",
                        n + 1,
                        samples.len(),
                        model.rate(),
                        t0.elapsed().as_secs_f64()
                    );
                }
                let (coded, loss) = cost.bits_per_byte(bytes);
                println!(
                    "epoch {epoch}: {coded:.4} bits/byte (ratio {:.4}), loss {loss:.4} bits/byte, {:.0}s, rate now {}",
                    8.0 / coded,
                    t0.elapsed().as_secs_f64(),
                    model.rate()
                );
                println!("  {}", model.health());
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
        Some("opbench") => {
            let dev = args.device();
            // the first pass compiles the kernels; read the later ones
            for b in [512, 512, 4096] {
                ops::bench(&dev, b);
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
