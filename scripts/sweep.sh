#!/bin/sh
# Compress each file under every combination of predictor, match finder, update rule and
# update-skipping weight, and print one row per combination.
#   scripts/sweep.sh <gmint binary> <file>...
# Narrow the sweep by setting MODELS, MATCHES, RULES or LCRS, e.g. RULES="auto:14" LCRS="0 256".
# EXTRA is passed through to every run, e.g. EXTRA="--comps 16 --replicas 8".
bin=$1; shift
: "${MODELS:=attn attn-tree attn-rtree spn both}"
: "${MATCHES:=off on}"
: "${RULES:=auto:14 sgd:15}"
: "${LCRS:=0}"
out=$(mktemp)
printf '%-11s %-5s %-7s %-4s' model match rule lcr
for f in "$@"; do printf ' %14s' "$(basename "$f" | cut -c1-14)"; done
printf ' %8s\n' seconds
for model in $MODELS; do for match in $MATCHES; do for rule in $RULES; do for lcr in $LCRS; do
  case $model in
    attn-tree) flags="--model attn --head tree" ;;
    attn-rtree) flags="--model attn --head rtree" ;;
    both-tree) flags="--model both --head tree" ;;
    both-rtree) flags="--model both --head rtree" ;;
    *) flags="--model $model" ;;
  esac
  printf '%-11s %-5s %-7s %-4s' "$model" "$match" "$rule" "$lcr"
  total=0
  for f in "$@"; do
    line=$("$bin" c "$f" "$out" --batch 512 $flags --match "$match" --opt "${rule%%:*}" --lr "${rule##*:}" --lcr "$lcr" $EXTRA 2>/dev/null | head -1)
    printf ' %14s' "$(echo "$line" | sed -n 's/.*ratio \([0-9.]*\),.*/\1/p')"
    total=$(echo "$line" | sed -n 's/.*ratio [0-9.]*, \([0-9.]*\)s.*/\1/p' | awk -v t="$total" '{print t + $1}')
  done
  printf ' %8s\n' "$total"
done; done; done; done
rm -f "$out"
