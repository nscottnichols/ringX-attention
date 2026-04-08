#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

NPROC="${NPROC:-2}"
TESTS_STR="${TESTS:-test/test_ringX1_attn_func.py test/test_ringX2_attn_func.py test/test_ringX3_attn_func.py}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/test_results/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/shims"

cat > "$OUT_DIR/shims/utils.py" <<'PY'
import random
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log(name, value, rank0_only=False):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return

    if torch.is_tensor(value):
        x = value.detach().float()
        max_abs = x.abs().max().item() if x.numel() else 0.0
        mean_abs = x.abs().mean().item() if x.numel() else 0.0
        print(
            f"[rank {rank}] {name}: shape={list(x.shape)} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
            flush=True,
        )
    else:
        print(f"[rank {rank}] {name}: {value}", flush=True)
PY

export PYTHONUNBUFFERED=1
export PYTHONPATH="$OUT_DIR/shims:$REPO_ROOT/ringX_attn:$REPO_ROOT:${PYTHONPATH:-}"

if ! python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
  echo "Missing Python deps. Make sure flash_attn is importable." >&2
  exit 1
fi

read -r -a TESTS <<< "$TESTS_STR"

{
  echo "test,status,rank,metric,max_abs,mean_abs,log_file"
} > "$OUT_DIR/summary.csv"

failures=0

for test_file in "${TESTS[@]}"; do
  test_name="$(basename "$test_file" .py)"
  log_file="$OUT_DIR/raw/${test_name}.log"

  cmd=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    "$test_file"
  )

  echo ">>> Running: ${cmd[*]}"
  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    failures=$((failures + 1))
  fi

  awk -v test="$test_name" -v status="$status" -v logfile="$log_file" '
    /^\[rank [0-9]+\] .* diff:/ {
      if (match($0, /\[rank ([0-9]+)\] (.*): .*max_abs=([0-9.eE+-]+) mean_abs=([0-9.eE+-]+)/, m)) {
        print test "," status "," m[1] ",\"" m[2] "\"," m[3] "," m[4] "," logfile
      }
    }
  ' "$log_file" >> "$OUT_DIR/summary.csv"

  echo ">>> Exit status for $test_name: $status"
done

echo
echo "Done."
echo "Raw logs:    $OUT_DIR/raw"
echo "CSV summary: $OUT_DIR/summary.csv"

if [[ $failures -ne 0 ]]; then
  echo "$failures test invocation(s) exited non-zero." >&2
  exit 1
fi
