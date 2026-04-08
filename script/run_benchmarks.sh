#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

NPROC="${NPROC:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_ITER="${NUM_ITER:-10}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
FORWARD_ONLY="${FORWARD_ONLY:-1}"
SEQ_LENGTHS_STR="${SEQ_LENGTHS:-4096 8192}"
ALGOS_STR="${ALGOS:-ringX_attn.ringX1_attn ringX_attn.ringX2_attn ringX_attn.ringX3_attn ringX_attn.ringX4_attn}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmark_results/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR/raw"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/ringX_attn:$REPO_ROOT/benchmark${RING_FLASH_ATTN_ROOT:+:$RING_FLASH_ATTN_ROOT}:${PYTHONPATH:-}"

if grep -q 'cuda:0' benchmark/benchmark_ringX_func.py; then
  echo "WARNING: benchmark/benchmark_ringX_func.py still hardcodes cuda:0." >&2
  echo "Patch it to use LOCAL_RANK before using NPROC>1." >&2
fi

if ! python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
  echo "Missing Python deps. Make sure flash_attn is importable." >&2
  exit 1
fi

read -r -a SEQ_LENGTHS <<< "$SEQ_LENGTHS_STR"
read -r -a ALGOS <<< "$ALGOS_STR"

{
  echo "algo,impl,causal,batch,seqlen,num_heads,head_dim,mode,ngpus,iter_per_s,total_sec,log_file"
} > "$OUT_DIR/summary.csv"

for algo in "${ALGOS[@]}"; do
  causal_flag=()
  algo_short="${algo##*.}"
  
  case "$algo_short" in
    ringX1_attn|ringX1o_attn)
      causal_flag=()
      ;;
    ringX2_attn|ringX2o_attn|ringX3_attn|ringX3b_attn|ringX4_attn|ringX4o_attn)
      causal_flag=(--causal)
      ;;
    *)
      echo "Unknown algo: $algo" >&2
      exit 1
      ;;
  esac

  for seqlen in "${SEQ_LENGTHS[@]}"; do
    log_file="$OUT_DIR/raw/${algo}.b${BATCH_SIZE}.s${seqlen}.h${NUM_HEADS}.d${HEAD_DIM}.log"

    cmd=(
      torchrun
      --standalone
      --nproc_per_node="$NPROC"
      benchmark/benchmark_ringX_func.py
      --module "$algo"
      --batch_size "$BATCH_SIZE"
      --num_iter "$NUM_ITER"
      --seq_length "$seqlen"
      --num_heads "$NUM_HEADS"
      --head_dim "$HEAD_DIM"
    )

    if [[ ${#causal_flag[@]} -gt 0 ]]; then
      cmd+=("${causal_flag[@]}")
    fi
    if [[ "$FORWARD_ONLY" == "1" ]]; then
      cmd+=(--forward_only)
    fi

    echo ">>> Running: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "$log_file"

    awk -v algo="$algo" \
        -v mode="$([[ "$FORWARD_ONLY" == "1" ]] && echo forward || echo fwd_bwd)" \
        -v logfile="$log_file" '
      /^# / { impl=$2 }
      /ngpus:/ {
        if (match($0, /ngpus: ([0-9]+), causal: (True|False), batch: ([0-9]+), seqlen: ([0-9]+), num_heads: ([0-9]+), head_dim: ([0-9]+)/, m)) {
          ngpus=m[1]
          causal=m[2]
          batch=m[3]
          seqlen=m[4]
          num_heads=m[5]
          head_dim=m[6]
        }
      }
      /iter\/s, .* sec/ {
        if (match($0, /([0-9.]+) iter\/s, ([0-9.]+) sec/, m)) {
          print algo "," impl "," causal "," batch "," seqlen "," num_heads "," head_dim "," mode "," ngpus "," m[1] "," m[2] "," logfile
        }
      }
    ' "$log_file" >> "$OUT_DIR/summary.csv"
  done
done

echo
echo "Done."
echo "Raw logs:    $OUT_DIR/raw"
echo "CSV summary: $OUT_DIR/summary.csv"
