# gmint

General-minification lossless transformer: a lossless compressor whose model learns each file
while coding it. It needs no pre-training and stores no weights in its output.

## How it works

- A small attention model [1] predicts each next byte. A random sum-product network [2] can
  replace it or join it, and a match finder (`--match on`) proposes bytes from earlier repeats.
- After coding each step, the model trains on it. The decoder repeats that training as it
  decodes, so both sides hold the same model at every step.
- Integer-only arithmetic keeps encoder and decoder identical, bit for bit, on any backend.
- The input is split into 512 streams coded in parallel. Streams take 4096-byte blocks in
  turn, so the match finder sees almost all earlier input.
- Each stream has its own rANS state [3], and all states share one stream of words [4], so the
  GPU decodes a whole step at once. With `--lcr 0` every step replays as one CUDA graph, and
  decoding runs as fast as encoding.
- `--lcr` skips training on steps that already code well, after TRACE [5].
- `gmint train` makes a prior (pre-trained starting weights). Its hash goes in the header, and
  decompressing needs it too.

Files in the older range-coded format [6] still decode.

## Requirements

Rust 1.98 or later and an NVIDIA GPU with CUDA. The CPU backend (`--device flex`) is far too
slow for real use.

## Build and use

    cargo build --release
    target/release/gmint c in out.gm --lcr 0     # compress
    target/release/gmint d out.gm in             # decompress
    target/release/gmint dry in                  # ratio, speed, model health; writes nothing
    target/release/gmint train out.prior a b ... # pre-train a prior

`gmint` alone lists every option. Settings go into the file header, so `d` needs only the
`--prior` used, if any.

Best settings so far:

- No prior: `--batch 512 --lcr 0 --head tree --match on --short 2 --warm 1024`
- With a prior: `--prior FILE --lr 15 --seed 42 --block 2048`

## Results

Ratio (original / compressed, higher is better) on six 4 MiB files the prior never saw: a slice
of Zotero's `libxul.so` and five slices of a tar of its install.

|                 | xul   | zot   | z60   | z90   | z120  | z160  | total |
|-----------------|-------|-------|-------|-------|-------|-------|-------|
| xz -9e          | 5.347 | 2.578 | 3.972 | 3.577 | 2.687 | 2.979 | 3.308 |
| zstd -19 --long | 4.537 | 2.001 | 3.658 | 3.391 | 2.540 | 2.765 | 2.936 |
| gmint, no prior | 4.451 | 2.199 | 2.964 | 2.785 | 2.192 | 2.248 | 2.635 |
| gmint, prior    | 3.874 | 2.009 | 2.980 | 2.738 | 2.319 | 2.293 | 2.582 |

gmint does not beat xz yet, and beats zstd on one file (zot). It codes at about 1.3–1.6 MiB/s
each way on an RTX 5070 with `--lcr 0`.

## Tests

    GMINT_TEST_DEVICE=cuda cargo test --release

Without it, tests run on the slow CPU backend.

## Sources

1. A. Vaswani et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. R. Peharz et al. *Probabilistic Deep Learning using Random Sum-Product Networks*. 2018. [arXiv:1806.01910](https://arxiv.org/abs/1806.01910)
3. J. Duda. *Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding*. 2013. [arXiv:1311.2540](https://arxiv.org/abs/1311.2540)
4. F. Giesen. *Interleaved entropy coders*. 2014. [arXiv:1402.3392](https://arxiv.org/abs/1402.3392)
5. Y. Mao, Y. Cui, T.-W. Kuo, C. J. Xue. *A Fast Transformer-based General-Purpose Lossless Compressor* (TRACE). 2022. [arXiv:2203.16114](https://arxiv.org/abs/2203.16114)
6. I. H. Witten, R. M. Neal, J. G. Cleary. *Arithmetic coding for data compression*. Communications of the ACM, 1987. [doi:10.1145/214762.214771](https://doi.org/10.1145/214762.214771)
