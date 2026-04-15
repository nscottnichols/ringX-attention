#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

NPROC="${NPROC:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_ITER="${NUM_ITER:-10}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
SEQ_LENGTHS_STR="${SEQ_LENGTHS:-4096 8192}"
ALGOS_STR="${ALGOS:-ringX_attn.ringX1_attn ringX_attn.ringX2_attn ringX_attn.ringX3_attn ringX_attn.ringX4_attn}"
DTYPE="${DTYPE:-${BENCHMARK_DTYPE:-bfloat16}}"
BACKEND="${BACKEND:-${RINGX_ATTN_BACKEND:-}}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmark_results/$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${BENCHMARK_MODES:-}" ]]; then
  BENCHMARK_MODES_STR="$BENCHMARK_MODES"
elif [[ -n "${FORWARD_ONLY:-}" ]]; then
  if [[ "$FORWARD_ONLY" == "1" ]]; then
    BENCHMARK_MODES_STR="forward"
  else
    BENCHMARK_MODES_STR="fwd_bwd backward"
  fi
else
  BENCHMARK_MODES_STR="forward fwd_bwd backward"
fi

mkdir -p "$OUT_DIR/raw"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/ringX_attn:$REPO_ROOT/benchmark${RING_FLASH_ATTN_ROOT:+:$RING_FLASH_ATTN_ROOT}:${PYTHONPATH:-}"
if [[ -n "$BACKEND" ]]; then
  export RINGX_ATTN_BACKEND="$BACKEND"
fi
export BENCHMARK_DTYPE="$DTYPE"

if ! python - <<'PY' >/dev/null 2>&1
import ringX_attn.backend
PY
then
  echo "Missing Python deps. Make sure ringX_attn is importable." >&2
  exit 1
fi

read -r -a SEQ_LENGTHS <<< "$SEQ_LENGTHS_STR"
read -r -a ALGOS <<< "$ALGOS_STR"
read -r -a BENCHMARK_MODES <<< "$BENCHMARK_MODES_STR"

HAS_RING_FLASH_ATTN=0
if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ring_flash_attn") is not None else 1)
PY
then
  HAS_RING_FLASH_ATTN=1
fi

{
  echo "algo,impl,mode,status,requested_backend,forward_backend,backward_backend,causal,batch,seqlen,num_heads,head_dim,dtype,ngpus,iter_per_s,total_sec,reason,log_file"
} > "$OUT_DIR/summary.csv"

append_results_from_log() {
  local log_file="$1"
  python - "$log_file" "$OUT_DIR/summary.csv" <<'PY'
import csv
import json
import sys

log_file, summary_file = sys.argv[1], sys.argv[2]
rows = []
with open(log_file, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if not line.startswith("BENCHMARK_RESULT "):
            continue
        payload = json.loads(line[len("BENCHMARK_RESULT "):])
        rows.append([
            payload.get("algo", ""),
            payload.get("impl", ""),
            payload.get("mode", ""),
            payload.get("status", ""),
            payload.get("requested_backend", ""),
            payload.get("forward_backend", ""),
            payload.get("backward_backend", ""),
            payload.get("causal", ""),
            payload.get("batch", ""),
            payload.get("seqlen", ""),
            payload.get("num_heads", ""),
            payload.get("head_dim", ""),
            payload.get("dtype", ""),
            payload.get("ngpus", ""),
            payload.get("iter_per_s", ""),
            payload.get("total_sec", ""),
            (payload.get("reason", "") or "").replace("\n", " ").replace("\r", " "),
            log_file,
        ])

if rows:
    with open(summary_file, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
PY
}

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
    impls=("${algo_short}_func")
    if [[ "$HAS_RING_FLASH_ATTN" == "1" ]]; then
      impls+=(zigzag_ring_flash_attn_func stripe_flash_attn_func ring_flash_attn_func)
    fi

    for mode in "${BENCHMARK_MODES[@]}"; do
      for impl in "${impls[@]}"; do
        log_file="$OUT_DIR/raw/${algo}.impl${impl}.mode${mode}.b${BATCH_SIZE}.s${seqlen}.h${NUM_HEADS}.d${HEAD_DIM}.dtype${DTYPE}.log"

        cmd=(
          torchrun
          --standalone
          --nproc_per_node="$NPROC"
          benchmark/benchmark_ringX_func.py
          --module "$algo"
          --impl "$impl"
          --batch_size "$BATCH_SIZE"
          --num_iter "$NUM_ITER"
          --seq_length "$seqlen"
          --num_heads "$NUM_HEADS"
          --head_dim "$HEAD_DIM"
          --dtype "$DTYPE"
          --modes "$mode"
        )

        if [[ ${#causal_flag[@]} -gt 0 ]]; then
          cmd+=("${causal_flag[@]}")
        fi

        echo ">>> Running: ${cmd[*]}"
        set +e
        "${cmd[@]}" 2>&1 | tee "$log_file"
        cmd_status=${PIPESTATUS[0]}
        set -e

        append_results_from_log "$log_file"

        if [[ $cmd_status -ne 0 ]] && ! grep -q '^BENCHMARK_RESULT ' "$log_file"; then
          python - "$OUT_DIR/summary.csv" "$algo" "$impl" "$mode" "$BATCH_SIZE" "$seqlen" "$NUM_HEADS" "$HEAD_DIM" "$DTYPE" "$NPROC" "$log_file" "$cmd_status" "$BACKEND" <<'PY'
import csv
import sys
summary_file, algo, impl, mode, batch, seqlen, num_heads, head_dim, dtype, ngpus, log_file, status, backend = sys.argv[1:]
with open(summary_file, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow([
        algo,
        impl,
        mode,
        "failed",
        backend or "auto",
        "",
        "",
        "",
        batch,
        seqlen,
        num_heads,
        head_dim,
        dtype,
        ngpus,
        "",
        "",
        f"torchrun exited with status {status}",
        log_file,
    ])
PY
          if [[ "$STOP_ON_ERROR" == "1" ]]; then
            exit "$cmd_status"
          fi
        fi
      done
    done
  done
done

echo
echo "Done."
echo "Raw logs:    $OUT_DIR/raw"
echo "CSV summary: $OUT_DIR/summary.csv"
