#!/bin/bash
# Context-length arc overnight sweep (v2 atoms-only). Trains generalize --tkvocab 0 at
# increasing seq_len (model max_seq_len tracks it via the base.py fix), holding effective
# batch = 32, matched 100 epochs; audits each best ckpt full-eval (bits/canonical-atom +
# content-tier) on the baked :latest (0.2.30 / v2). Sequential, decisive-first, robust to
# per-run failure. Survives session restart (nohup+disown on defroster).
#
#   Progress:   tail -f /scratch/tmp/preframr_experiments/ctx_overnight.log
#   Summary:    /scratch/tmp/preframr_experiments/ctx_overnight_summary.md
#   Done marker:/scratch/tmp/preframr_experiments/ctx_overnight.done
set -u
REPO=/scratch/anarkiwi/preframr-xpt
WORK=/scratch/tmp/preframr_experiments
LOG=$WORK/ctx_overnight.log
DONE=$WORK/ctx_overnight.done
SUMMARY=$WORK/ctx_overnight_summary.md
IMAGE=anarkiwi/preframr:latest
AUDIT=/scratch/tmp/audit_ckpt.py
mkdir -p "$WORK"; rm -f "$DONE"; cd "$REPO"

# seq_len  batch_size  accum   (effective batch = batch_size*accum, held = 32) -- decisive first
RUNS=(
  "16384 4 8"
  "12288 4 8"
  "24576 4 8"
)

{
echo "==== ctx overnight batch started $(date -Iseconds) host=$(hostname) ===="
echo "comparator: v2-8192 baseline content 0.505/0.552/0.485 bits/atom 1.998/2.058/2.272"
for r in "${RUNS[@]}"; do
  set -- $r; SL=$1; BS=$2; AC=$3
  ROOT=$WORK/ctx_atoms_v2_sl${SL}
  rm -rf "$ROOT"
  echo "==== seq_len=$SL bs=$BS accum=$AC (eff $((BS*AC))) START $(date -Iseconds) ===="
  if PYTHONPATH=. python3 -m preframr_experiments.run generalize \
        --root "$ROOT" --tkvocab 0 --seq-len "$SL" --max-epochs 100 \
        --batch-size "$BS" --accumulate-grad-batches "$AC"; then
    CKPT=$(ls -t "$ROOT"/results/generalize/default/seed0/tb_logs/preframr/*/checkpoints/best-*.ckpt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
      echo "==== seq_len=$SL train OK, auditing $CKPT $(date -Iseconds) ===="
      docker run --rm --gpus=all -v /scratch:/scratch "$IMAGE" \
        python3 "$AUDIT" "$CKPT" "$WORK/ctx_audit_sl${SL}.json" 2>&1 | grep -iE "bits/atom|content=|wrote|Error" | grep -iv deprecat
      {
        echo "## seq_len=$SL  (ckpt $(basename "$CKPT"))"
        python3 -c "import json;d=json.load(open('$WORK/ctx_audit_sl${SL}.json'));[print(f\"  {k}: bits/atom={v['bits_per_canonical_atom']:.3f} content={v['content_acc']:.4f} struct={v['structural_acc']:.4f}\") for k,v in d['subsets'].items()]" 2>&1
        echo ""
      } >> "$SUMMARY"
    else
      echo "==== seq_len=$SL train OK but NO CKPT found $(date -Iseconds) ===="
    fi
  else
    echo "==== seq_len=$SL FAILED rc=$? $(date -Iseconds) ===="
  fi
done
echo "==== ctx overnight batch finished $(date -Iseconds) ===="
touch "$DONE"
} >> "$LOG" 2>&1
