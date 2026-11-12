//! Carry-less range coder (Subbotin's scheme widened to a 64-bit state), the integer form of
//! arithmetic coding (Witten, Neal & Cleary 1987). Frequencies are plain integers, so the coder
//! never sees a float and totals up to 2^48 are legal.

const TOP: u64 = 1 << 56;
const BOT: u64 = 1 << 48;

pub struct Encoder {
    low: u64,
    range: u64,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self { low: 0, range: u64::MAX, out: Vec::new() }
    }

    /// Codes a symbol occupying `[cum, cum + freq)` of `[0, total)`. `freq` must be non-zero.
    pub fn encode(&mut self, cum: u64, freq: u64, total: u64) {
        debug_assert!(freq > 0 && cum + freq <= total && total < BOT);
        self.range /= total;
        self.low = self.low.wrapping_add(cum * self.range);
        self.range *= freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) >= TOP {
                if self.range >= BOT {
                    break;
                }
                self.range = self.low.wrapping_neg() & (BOT - 1);
            }
            self.out.push((self.low >> 56) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    pub fn len(&self) -> usize {
        self.out.len()
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..8 {
            self.out.push((self.low >> 56) as u8);
            self.low <<= 8;
        }
        self.out
    }
}

pub struct Decoder<'a> {
    low: u64,
    range: u64,
    code: u64,
    src: &'a [u8],
    at: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        let mut d = Self { low: 0, range: u64::MAX, code: 0, src, at: 0 };
        for _ in 0..8 {
            d.code = (d.code << 8) | d.next() as u64;
        }
        d
    }

    /// Past the end the stream reads as zeros; a truncated file then fails the checksum.
    fn next(&mut self) -> u8 {
        let b = self.src.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        b
    }

    /// The cumulative-frequency slot the next symbol falls in. Follow with [`Self::consume`].
    pub fn target(&mut self, total: u64) -> u64 {
        self.range /= total;
        (self.code.wrapping_sub(self.low) / self.range).min(total - 1)
    }

    pub fn consume(&mut self, cum: u64, freq: u64) {
        self.low = self.low.wrapping_add(cum * self.range);
        self.range *= freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) >= TOP {
                if self.range >= BOT {
                    break;
                }
                self.range = self.low.wrapping_neg() & (BOT - 1);
            }
            self.code = (self.code << 8) | self.next() as u64;
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Rng;

    fn roundtrip(tables: &[Vec<u64>], syms: &[usize]) {
        let mut enc = Encoder::new();
        for (t, s) in tables.iter().zip(syms) {
            enc.encode(t[..*s].iter().sum(), t[*s], t.iter().sum());
        }
        let bytes = enc.finish();
        let mut dec = Decoder::new(&bytes);
        for (t, s) in tables.iter().zip(syms) {
            let target = dec.target(t.iter().sum());
            let (mut cum, mut got) = (0, 0);
            while cum + t[got] <= target {
                cum += t[got];
                got += 1;
            }
            assert_eq!(got, *s);
            dec.consume(cum, t[got]);
        }
    }

    #[test]
    fn random_tables_roundtrip() {
        let mut rng = Rng(3);
        let (mut tables, mut syms) = (Vec::new(), Vec::new());
        for _ in 0..20_000 {
            // model-shaped tables: a few large entries up to 2^16, the rest at the floor of 1
            let t: Vec<u64> =
                (0..256).map(|_| if rng.next() % 16 == 0 { rng.next() % 65_536 + 1 } else { 1 }).collect();
            syms.push((rng.next() % 256) as usize);
            tables.push(t);
        }
        roundtrip(&tables, &syms);
    }

    #[test]
    fn extreme_tables_roundtrip() {
        let mut rng = Rng(5);
        let (mut tables, mut syms) = (Vec::new(), Vec::new());
        for i in 0..20_000 {
            let mut t = vec![1u64; 256];
            let hot = (rng.next() % 256) as usize;
            t[hot] = 1 << 24;
            // mostly the likely symbol, sometimes the least likely one
            syms.push(if i % 7 == 0 { (hot + 1) % 256 } else { hot });
            tables.push(t);
        }
        roundtrip(&tables, &syms);
    }
}
