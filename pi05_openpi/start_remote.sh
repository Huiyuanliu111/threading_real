#!/usr/bin/env bash
# Server 10.157.174.249: isolated source/config/cache, shared read-only dependencies.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
mkdir -p logs
exec 9>logs/train.lock
flock -n 9 || { echo 'A training launcher already holds the lock'; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,4,5,6}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONPATH="$HERE/vendor/lerobot${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HERE/.cache/huggingface"
export HF_LEROBOT_HOME="$HERE/.cache/lerobot"
export OPENPI_DATA_HOME="$HERE/.cache/openpi"
export XDG_CACHE_HOME="$HERE/.cache"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python - <<'PY'
import os, shutil, subprocess
free=shutil.disk_usage('.').free / 2**30
if free < 55:
    raise SystemExit(f'Need >=55 GiB free before training; available {free:.1f} GiB')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
selected=set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
for row in rows.splitlines():
    index, memory, util = [v.strip() for v in row.split(',')]
    if index in selected and (int(memory)>1024 or int(util)>10):
        raise SystemExit(f'GPU {index} is busy: {memory} MiB, utilization {util}%')
print(f'Preflight: {free:.1f} GiB free; GPUs {sorted(selected)} idle', flush=True)
PY
printf '%s\n' "$$" > logs/train.pid
exec .venv/bin/python -u run.py train --mode lora --batch-size 12 --fsdp-devices 2 "$@"
