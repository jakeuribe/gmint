# gmint

gmint (general-minification lossless transformer) is a lossless compressor. A small neural
network predicts each next byte, and the better it predicts, the fewer bits the byte costs. The
network starts untrained and learns each file while coding it, so the output holds no weights.

It does not yet compress as well as xz or zstd (see Results).

## How it works

- The input is dealt out in 4096-byte blocks to 512 streams. Each step codes one byte of every
  stream.
- The network is a small transformer [1] (`--model spn` swaps in a random sum-product
  network [2]). `--match on` adds a match finder, which proposes the byte that followed the last
  occurrence of the preceding 8 (or 5) bytes.
- After each step the network trains on the bytes just coded. The decoder repeats the same
  predictions and training, so both networks stay identical. All arithmetic is integer, because
  floating-point rounding can differ between devices.
- The entropy coder is rANS [3]: one coder state per stream, all writing into one sequence of
  words [4], so the GPU decodes each step without a round trip to the CPU. With `--lcr 0`,
  each step's GPU work is recorded once and replayed (a CUDA graph).
- `--lcr W` (default 16) skips training on steps cheaper than W/256 of the recent average,
  trading ratio for time, after TRACE [5]; 0 never skips.
- `gmint train` makes a prior: starting weights trained on sample files. The decoder needs the
  same prior and checks its hash.

Older range-coded files [6] still decode.

## Build and use

Needs an NVIDIA GPU with CUDA. Tested with Rust 1.98.

    cargo build --release
    target/release/gmint c in out.gm --lcr 0     # compress
    target/release/gmint d out.gm in             # decompress
    target/release/gmint train out.prior a b ... # train a prior on files a, b
    GMINT_TEST_DEVICE=cuda cargo test --release  # run the tests on the GPU

`gmint` alone lists every option; `d` reads its settings from the file.

Best settings so far: `--lcr 0 --head tree --match on --short 2 --warm 1024`
without a prior; `--prior FILE --lr 15 --seed 42 --block 2048` with one.

## Results

Ratio (original / compressed; higher is better) on six 4 MiB slices of Zotero's install: xul
from `libxul.so`, the rest from a tar of it. The prior was trained on other apps' files.

compressor      | xul   | zot   | z60   | z90   | z120  | z160  | total
----------------|-------|-------|-------|-------|-------|-------|------
xz -9e          | 5.347 | 2.578 | 3.972 | 3.577 | 2.687 | 2.979 | 3.308
zstd -19 --long | 4.537 | 2.001 | 3.658 | 3.391 | 2.540 | 2.765 | 2.936
gmint, no prior | 4.451 | 2.199 | 2.964 | 2.785 | 2.192 | 2.248 | 2.635
gmint, prior    | 3.874 | 2.009 | 2.980 | 2.738 | 2.319 | 2.293 | 2.582

gmint beats zstd only on zot. On an RTX 5070, a 32 MiB file took 20.6 s to compress and 19.8 s
to decompress (`--lcr 0`).

## Sources

1. A. Vaswani et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. R. Peharz et al. *Probabilistic Deep Learning using Random Sum-Product Networks*. 2018. [arXiv:1806.01910](https://arxiv.org/abs/1806.01910)
3. J. Duda. *Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding*. 2013. [arXiv:1311.2540](https://arxiv.org/abs/1311.2540)
4. F. Giesen. *Interleaved entropy coders*. 2014. [arXiv:1402.3392](https://arxiv.org/abs/1402.3392)
5. Y. Mao, Y. Cui, T.-W. Kuo, C. J. Xue. *A Fast Transformer-based General-Purpose Lossless Compressor* (TRACE). 2022. [arXiv:2203.16114](https://arxiv.org/abs/2203.16114)
6. I. H. Witten, R. M. Neal, J. G. Cleary. *Arithmetic coding for data compression*. Communications of the ACM, 1987. [doi:10.1145/214762.214771](https://doi.org/10.1145/214762.214771)
