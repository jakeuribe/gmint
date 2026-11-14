# Priors

A prior is a file of starting weights for gmint's network. Without one, the network starts from
random weights and learns each file from scratch. With `--prior FILE`, it starts from weights
already trained on sample files. Decompressing needs the same prior: the compressed file records
the prior's hash, and gmint stops if they differ.

A prior also fixes the network's shape, so gmint reads `--ctx`, `--emb`, `--hid`, `--head`,
`--match`, `--norm` and `--mask` from it. It does not fix `--lr`, `--seed` or `--block`.

## r17.prior

The best prior so far (639 kB). The network sees the last 4 bytes, turns each into 64 numbers
(an embedding), has 256 hidden units, predicts a byte one bit at a time (`--head tree`), and
uses the match finder (`--match on`).

### How it was trained

On 30 sample files of 4 MiB each from four apps: KiCad, Obsidian, OpenSCAD and Zulip. The
command lists them as KiCad-0, Obsidian-0, OpenSCAD-0, Zulip-0, KiCad-1, and so on. Every
training batch mixes all 30 (`--mix on`, the default), and training makes 8 passes:

    gmint train r17.prior FILES... --batch 512 --ctx 4 --emb 64 --hid 256 --head tree \
      --match on --norm off --opt auto --lr 15 --epochs 8 --seed 42

The cost fell from 2.776 bits per byte in the first pass to 2.656 in the last. Running the same
command on the same code again produced an identical file.

### How to use it

    gmint c in out.gm --prior r17.prior --lr 15 --seed 42 --block 2048
    gmint d out.gm in --prior r17.prior

With this prior, `--block 2048` did better than the default, 4096.

### How good it is

With this prior, gmint compresses worse in total than without one, and worse than xz or
zstd. The test files are six 4 MiB slices of Zotero's install, which r17 never saw in training.
Ratio is original size over compressed size; higher is better.

compressor      | xul   | zot   | z60   | z90   | z120  | z160  | total
----------------|-------|-------|-------|-------|-------|-------|------
xz -9e          | 5.347 | 2.578 | 3.972 | 3.577 | 2.687 | 2.979 | 3.308
zstd -19 --long | 4.537 | 2.001 | 3.658 | 3.391 | 2.540 | 2.765 | 2.936
gmint, no prior | 4.451 | 2.199 | 2.964 | 2.785 | 2.192 | 2.248 | 2.635
gmint, r17      | 3.874 | 2.009 | 2.980 | 2.738 | 2.319 | 2.293 | 2.582

The prior helps on z60, z120 and z160, and hurts on the other three, most of all on xul. The
no-prior row uses `--lcr 0 --head tree --match on --short 2 --warm 1024`, which mixes two
attention networks; priors cannot hold that yet.
