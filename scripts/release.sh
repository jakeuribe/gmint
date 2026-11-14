#!/bin/sh
# Build the gmint binary of a release into dist/: stripped, for x86-64 Linux, with SHA256SUMS.
# The release workflow runs this; so can you. Priors are not in git, so they are added to the
# release afterwards with scripts/add-prior.sh.
#   scripts/release.sh v0.1.0
# The tag must be "v" followed by the version in Cargo.toml.
set -eu
tag=${1:?usage: scripts/release.sh <tag>}
cd "$(dirname "$0")/.."
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
if [ "$tag" != "v$version" ]; then
    echo "release.sh: tag $tag does not match Cargo.toml version $version" >&2
    exit 1
fi
cargo build --release --locked
rm -rf dist
mkdir dist
strip -o "dist/gmint-$tag-x86_64-linux" "${CARGO_TARGET_DIR:-target}/release/gmint"
(cd dist && sha256sum -- * > SHA256SUMS)
ls -l dist
