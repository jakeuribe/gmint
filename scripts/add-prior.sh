#!/bin/sh
# Attach priors to a release made by the release workflow, and add them to its SHA256SUMS.
# Priors are not in git, so run this on the machine that has them, before publishing the draft.
#   scripts/add-prior.sh v0.1.0 priors/r17.prior
# Running it again with the same prior replaces it.
set -eu
tag=${1:?usage: scripts/add-prior.sh <tag> <prior>...}
shift
if [ $# -eq 0 ]; then
    echo "add-prior.sh: no prior given" >&2
    exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
gh release download "$tag" --pattern SHA256SUMS --dir "$tmp"
for p in "$@"; do
    name=$(basename "$p")
    sum=$(sha256sum -- "$p" | cut -d ' ' -f 1)
    grep -v "  $name\$" "$tmp/SHA256SUMS" > "$tmp/sums" || true
    echo "$sum  $name" >> "$tmp/sums"
    mv "$tmp/sums" "$tmp/SHA256SUMS"
done
gh release upload "$tag" "$@" "$tmp/SHA256SUMS" --clobber
cat "$tmp/SHA256SUMS"
