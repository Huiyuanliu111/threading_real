#!/usr/bin/env bash
# v6 recipe; the only training recipe change is action horizon 50 -> 10.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
export HF_HOME="$HERE/.cache/huggingface"
export HF_DATASETS_CACHE=/dev/shm/huiyuan_pi05_v11/hf_datasets
export HF_LEROBOT_HOME="$HERE/.cache/lerobot"
export OPENPI_DATA_HOME="$HERE/.cache/openpi"
export XDG_CACHE_HOME="$HERE/.cache"
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6
export WANDB_MODE=online
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HERE/vendor/lerobot${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  --mode vision_lora_action_full --vision-lora-rank 16
  --image-profile 224 --camera-views both --freeze-vision --freeze-language
  --exp-name threading_tcp6_30hz_h10_vision_lora_action_full_v11
  --dataset-root "$HERE/data/threading_tcp6_nosmooth_30hz"
  --repo-id threading_real/threading_tcp6_nosmooth_30hz
  --horizon 10 --batch-size 12 --fsdp-devices 2 --steps 10000
  --warmup-steps 250 --learning-rate 2.5e-5 --num-workers 4
  --overfit-episodes 0 --val-fraction 0.1 --seed 42
  --eval-interval 1000 --save-interval 2000 --wandb
)
JAX_PLATFORMS=cpu .venv/bin/python -u run.py norm "${ARGS[@]}"
# Optional handoff: wait for the previous experiment's committed checkpoint.
if [[ -n "${V10_HANDOFF_PID:-}" ]]; then
  .venv/bin/python - <<'PY'
import os, pathlib, signal, time
pid = int(os.environ['V10_HANDOFF_PID'])
checkpoint = pathlib.Path('checkpoints/pi05_threading_vision_lora_action_full/threading_tcp6_cam1_native640_overfit1_v10/500')
deadline = time.monotonic() + 1800
while not checkpoint.is_dir():
    os.kill(pid, 0)
    if time.monotonic() > deadline:
        raise TimeoutError('v10 step500 checkpoint not completed; no training was stopped')
    time.sleep(5)
cmd = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
if b'threading_tcp6_cam1_native640_overfit1_v10' not in cmd or os.getpgid(pid) != pid:
    raise RuntimeError('Previous training PID does not match v10')
print(f'v10 checkpoint committed: {checkpoint}; stopping process group {pid}', flush=True)
os.killpg(pid, signal.SIGTERM)
for _ in range(60):
    if not pathlib.Path(f'/proc/{pid}').exists():
        break
    time.sleep(1)
else:
    raise RuntimeError('v10 did not exit; refusing to launch overlapping training')
time.sleep(5)
PY
fi
exec bash start_remote.sh "${ARGS[@]}"
