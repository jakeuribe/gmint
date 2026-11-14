//! Interleaved rANS (Duda 2013, "Asymmetric numeral systems"; the interleaving and its GPU form
//! from Giesen 2014, "Interleaved entropy coders").
//!
//! Every stream has its own 64-bit state, and all of them share one buffer of 32-bit words
//! (Giesen's `rans64` layout). A step's symbols are independent given the model the step was
//! predicted with, so a decoder can take all of a step's streams at once: with 32-bit words and a
//! 16-bit frequency scale a stream reads at most one word per symbol, and which word is a prefix
//! sum over the streams that need one. No metadata says where a stream's words are.
//!
//! The state's floor [`L`] sits 2^15 above the frequency scale. With less room (a 32-bit state
//! reading 16-bit words has none at a 16-bit scale) the coder pays visibly for every likely
//! symbol: 1.9% on a held-out file.
//!
//! The encoder runs backwards over the symbols (rANS is last in, first out), so it keeps every
//! symbol's (cumulative frequency, frequency) until the end.
//!
//! Frequencies are integers summing to exactly [`M`], from the model's byte weights by
//! [`quantize`]; the GPU forms the same numbers (see `fused::kernel::coded`).

/// log2 of the frequency scale: every row of frequencies sums to 2^16.
pub const SCALE_BITS: u32 = 16;
pub const M: u32 = 1 << SCALE_BITS;
/// The lower end of a state's normal interval [L, 2^63).
const L: u64 = 1 << 31;

/// A row's coder frequencies from its cost frequencies `f` (the old range coder's,
/// `(weight >> 4) + 1`, any total `T`): each byte gets `max(1, floor(f * 2^16 / T))`, and the
/// largest (the lowest byte of equals) absorbs whatever the row is off from [`M`]. That is the
/// old coder's distribution to 16-bit precision, its floor of 1 per byte included: when the
/// match boost shrinks every byte but the proposed one, `T` is small and that floor is a real
/// hedge against a wrong proposal, which the boost's gain was trained with.
///
/// `f <= 2^16 + 1` and `T < 2^25` (weights need not be normalised: a mix's are relative to its
/// top byte). The GPU forms the same numbers in u64.
pub fn quantize(f: &[i32], out: &mut [u32]) {
    debug_assert_eq!(f.len(), out.len());
    assert!(f.iter().all(|v| (1..=(1 << 16) + 1).contains(v)), "frequencies out of the coder's range: {:?}", f.iter().max());
    let total: u64 = f.iter().map(|v| *v as u64).sum();
    let (mut sum, mut top, mut best) = (0i64, 0usize, 0u32);
    for (v, (fv, q)) in f.iter().zip(out.iter_mut()).enumerate() {
        *q = ((*fv as u64 * M as u64 / total) as u32).max(1);
        sum += *q as i64;
        // strictly larger, so equals leave the lowest byte
        if *q > best {
            (best, top) = (*q, v);
        }
    }
    let adjusted = out[top] as i64 + M as i64 - sum;
    // at most 255 bytes are raised to 1, and then the largest holds nearly all of the row
    assert!(adjusted >= 1, "quantised row cannot sum to M");
    out[top] = adjusted as u32;
}

/// [`quantize`] written plainly, as the reference the fast form is tested on.
#[cfg(test)]
fn quantize_ref(f: &[i32]) -> Vec<u32> {
    let total: f64 = f.iter().map(|v| *v as f64).sum();
    let mut q: Vec<i64> = f.iter().map(|v| ((*v as f64 * M as f64 / total).floor() as i64).max(1)).collect();
    let sum: i64 = q.iter().sum();
    let top = (0..q.len()).fold(0, |best, v| if q[v] > q[best] { v } else { best });
    q[top] += M as i64 - sum;
    q.iter().map(|v| *v as u32).collect()
}

/// Collects every symbol as it is coded, in decoding order, and codes them all at the end: 4
/// bytes of memory per input byte until then.
pub struct Encoder {
    lanes: usize,
    /// Per symbol, `cum | (freq - 1) << 16`.
    syms: Vec<u32>,
    /// The ideal code length so far in Q8 bits, for progress reports.
    bits_q8: u64,
}

impl Encoder {
    pub fn new(lanes: usize) -> Self {
        Self { lanes, syms: Vec::new(), bits_q8: 0 }
    }

    /// The next symbol, `[cum, cum + freq)` of `[0, M)`. Symbols go in stream order within a
    /// step, a whole step at a time.
    pub fn push(&mut self, cum: u32, freq: u32) {
        debug_assert!(freq > 0 && cum + freq <= M);
        self.syms.push(cum | (freq - 1) << 16);
        self.bits_q8 += (SCALE_BITS as u64) << 8;
        self.bits_q8 -= crate::codec::log2_q8(freq as u64);
    }

    /// About how many bytes the symbols so far will take.
    pub fn len(&self) -> usize {
        (self.bits_q8 / 2048) as usize
    }

    /// The coded symbols: each lane's final state (8 bytes, little-endian), then the words.
    pub fn finish(self) -> Vec<u8> {
        if self.syms.is_empty() {
            return Vec::new();
        }
        assert!(self.syms.len() % self.lanes == 0, "a partial step");
        let mut x = vec![L; self.lanes];
        let mut words: Vec<u32> = Vec::with_capacity(self.syms.len() / 8);
        // backwards: the last symbol of the last stream first
        for (i, sym) in self.syms.iter().enumerate().rev() {
            let (cum, freq) = ((sym & 0xFFFF) as u64, ((sym >> 16) + 1) as u64);
            let s = &mut x[i % self.lanes];
            // at most once: after it the state is below freq << 47, where coding keeps it < 2^63
            if *s >= freq << (63 - SCALE_BITS) {
                words.push(*s as u32);
                *s >>= 32;
            }
            *s = ((*s / freq) << SCALE_BITS) + *s % freq + cum;
        }
        let mut out = Vec::with_capacity(8 * self.lanes + 4 * words.len());
        out.extend(x.iter().flat_map(|s| s.to_le_bytes()));
        out.extend(words.iter().rev().flat_map(|w| w.to_le_bytes()));
        out
    }
}

/// The decoder's lane states and its place in the words.
pub struct Decoder<'a> {
    pub x: Vec<u64>,
    words: &'a [u8],
    at: usize,
}

impl<'a> Decoder<'a> {
    /// Past the end the words read as zeros; a truncated file then fails the checksum.
    pub fn new(src: &'a [u8], lanes: usize) -> Self {
        let state = |s: usize| src.get(8 * s..8 * s + 8).map_or(L, |b| u64::from_le_bytes(b.try_into().unwrap()));
        let x = (0..lanes).map(state).collect();
        Self { x, words: src.get(8 * lanes..).unwrap_or(&[]), at: 0 }
    }

    /// The slot lane `s` decodes next, in `[0, M)`; find the symbol whose `[cum, cum + freq)`
    /// holds it, then [`Self::consume`].
    pub fn slot(&self, s: usize) -> u32 {
        (self.x[s] & (M as u64 - 1)) as u32
    }

    /// Takes lane `s`'s symbol off its state. Lanes are consumed in order within a step.
    pub fn consume(&mut self, s: usize, cum: u32, freq: u32) {
        let x = self.x[s];
        let mut x = freq as u64 * (x >> SCALE_BITS) + (x & (M as u64 - 1)) - cum as u64;
        if x < L {
            let w = self.words.get(4 * self.at..4 * self.at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
            self.at += 1;
            x = x << 32 | w as u64;
        }
        self.x[s] = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Rng;

    /// Rows of cost frequencies, `(weight >> 4) + 1`.
    fn rows(rng: &mut Rng, n: usize) -> Vec<Vec<i32>> {
        let weights: Vec<Vec<i32>> = (0..n)
            .map(|i| match i % 4 {
                // model-shaped: a few large weights, the rest near nothing
                0 => (0..256).map(|_| if rng.next() % 16 == 0 { (rng.next() % 65_536) as i32 } else { (rng.next() % 3) as i32 }).collect(),
                // one byte holds nearly everything
                1 => {
                    let hot = (rng.next() % 256) as usize;
                    (0..256).map(|v| if v == hot { 65_000 } else { 0 }).collect()
                }
                // flat, and as a mix leaves them: relative to the top byte, summing far past 2^16
                2 if i % 8 == 2 => vec![256; 256],
                2 => (0..256).map(|_| (30_000 + rng.next() % 35_536) as i32).collect(),
                // all zero
                _ => vec![0; 256],
            })
            .collect();
        weights.into_iter().map(|w| w.iter().map(|g| g + 1).collect()).collect()
    }

    #[test]
    fn quantize_matches_reference_and_sums_to_m() {
        let mut rng = Rng(11);
        let mut q = vec![0u32; 256];
        for g in rows(&mut rng, 4000) {
            quantize(&g, &mut q);
            assert_eq!(q, quantize_ref(&g));
            assert_eq!(q.iter().sum::<u32>(), M);
            assert!(q.iter().all(|f| *f >= 1));
        }
    }

    #[test]
    fn interleaved_round_trip() {
        let mut rng = Rng(7);
        let lanes = 512;
        let steps = 300;
        let tables = rows(&mut rng, 64);
        let mut q = vec![0u32; 256];
        let quantized: Vec<Vec<u32>> = tables.iter().map(|g| {
            quantize(g, &mut q);
            q.clone()
        }).collect();
        let mut plan = Vec::new();
        let mut enc = Encoder::new(lanes);
        for _ in 0..steps {
            for _ in 0..lanes {
                let t = (rng.next() % quantized.len() as u64) as usize;
                // mostly a likely byte, sometimes any byte
                let row = &quantized[t];
                let sym = if rng.next() % 4 == 0 { (rng.next() % 256) as usize } else { (0..256).max_by_key(|v| row[*v]).unwrap() };
                let cum: u32 = row[..sym].iter().sum();
                enc.push(cum, row[sym]);
                plan.push((t, sym));
            }
        }
        let bytes = enc.finish();
        let mut dec = Decoder::new(&bytes, lanes);
        for (i, (t, sym)) in plan.iter().enumerate() {
            let row = &quantized[*t];
            let slot = dec.slot(i % lanes);
            let (mut cum, mut got) = (0u32, 0usize);
            while cum + row[got] <= slot {
                cum += row[got];
                got += 1;
            }
            assert_eq!(got, *sym, "symbol {i}");
            dec.consume(i % lanes, cum, row[got]);
        }
        // every lane ends where the encoder started it
        assert!(dec.x.iter().all(|x| *x == L));
    }
}
