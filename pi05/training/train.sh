#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU DDP configuration for the combined two-view threading dataset.
# Override variables on the command line, e.g. STEPS=10 ./train.sh.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PI05_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
if [[ -d "${PI05_DIR}/../../data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d" ]]; then
  PROJECT_ROOT=$(cd "${PI05_DIR}/../.." && pwd)
else
  PROJECT_ROOT=$(cd "${PI05_DIR}/.." && pwd)
fi
DATASET_ROOT=${DATASET_ROOT:-${PROJECT_ROOT}/data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d}
REPO_ID=${REPO_ID:-threading_real/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d}
OUTPUT_DIR=${OUTPUT_DIR:-${PI05_DIR}/outputs/threading_combined_pi05_tcp_pose_6d_continuous_v1}
MODEL_ID=${MODEL_ID:-lerobot/pi05_base}
STATE_REPRESENTATION=${STATE_REPRESENTATION:-tcp_pose_6d}
CHUNK_SIZE=${CHUNK_SIZE:-10}
N_ACTION_STEPS=${N_ACTION_STEPS:-${CHUNK_SIZE}}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-80}
EXPECTED_FPS=${EXPECTED_FPS:-15}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
# Full expert run. With six workers and per-rank batch two, every loop step is
# one optimizer update with effective batch 12.
STEPS=${STEPS:-5000}
SAVE_FREQ=${SAVE_FREQ:-1000}
EVAL_FREQ=${EVAL_FREQ:-500}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-512}
EVAL_SPLIT=${EVAL_SPLIT:-0.2}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-false}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-false}
PROPRIOCEPTION_DROPOUT=${PROPRIOCEPTION_DROPOUT:-0.15}
IGNORE_GRIPPER_ACTION=${IGNORE_GRIPPER_ACTION:-true}
FINETUNE_MODE=${FINETUNE_MODE:-visual_full_expert}
VISION_LR=${VISION_LR:-2.5e-6}
PROJECTOR_LR=${PROJECTOR_LR:-1e-5}
EXPERT_ATTENTION_LR=${EXPERT_ATTENTION_LR:-5e-6}
EXPERT_MLP_LR=${EXPERT_MLP_LR:-2.5e-6}
ACTION_LR=${ACTION_LR:-1e-5}
HIGH_NOISE_FRACTION=${HIGH_NOISE_FRACTION:-0.5}
HIGH_NOISE_MIN_TIME=${HIGH_NOISE_MIN_TIME:-0.8}
FIXED_EVAL_SEED=${FIXED_EVAL_SEED:-20260909}
NORMALIZATION_MAPPING=${NORMALIZATION_MAPPING:-'{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}'}
LOG_FREQ=${LOG_FREQ:-20}
SCHEDULER_WARMUP_STEPS=${SCHEDULER_WARMUP_STEPS:-250}
SCHEDULER_DECAY_STEPS=${SCHEDULER_DECAY_STEPS:-5000}
WANDB_ENABLE=${WANDB_ENABLE:-true}
WANDB_PROJECT=${WANDB_PROJECT:-threading_pi05}
JOB_NAME=${JOB_NAME:-threading_combined_pi05_tcp_pose_6d_continuous_v1}
GPU_IDS=${GPU_IDS:-0,1,3,4,5,6}
NUM_PROCESSES=${NUM_PROCESSES:-6}
MIN_GPU_MEMORY_MIB=${MIN_GPU_MEMORY_MIB:-40000}
PRINT_CONFIG_ONLY=${PRINT_CONFIG_ONLY:-false}

# Keep the only copy of downloaded base weights and W&B files inside this
# project. uv is invoked with --no-cache by bootstrap_remote.sh.
export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}
export WANDB_DIR=${WANDB_DIR:-${PROJECT_ROOT}/.cache/wandb}
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

if [[ "${PRINT_CONFIG_ONLY}" == "true" ]]; then
  for setting in \
    DATASET_ROOT REPO_ID OUTPUT_DIR MODEL_ID STATE_REPRESENTATION \
    CHUNK_SIZE N_ACTION_STEPS EXPECTED_EPISODES EXPECTED_FPS \
    BATCH_SIZE NUM_WORKERS GRADIENT_ACCUMULATION STEPS SAVE_FREQ EVAL_FREQ \
    MAX_EVAL_SAMPLES EVAL_SPLIT FINETUNE_MODE PROPRIOCEPTION_DROPOUT \
    IGNORE_GRIPPER_ACTION VISION_LR PROJECTOR_LR EXPERT_ATTENTION_LR \
    EXPERT_MLP_LR ACTION_LR HIGH_NOISE_FRACTION HIGH_NOISE_MIN_TIME \
    NORMALIZATION_MAPPING GPU_IDS NUM_PROCESSES MIN_GPU_MEMORY_MIB \
    WANDB_ENABLE WANDB_PROJECT JOB_NAME; do
    printf '%s=%q\n' "${setting}" "${!setting}"
  done
  exit 0
fi

if ! python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="google/paligemma-3b-pt-224",
    filename="config.json",
)
print("Hugging Face access check passed: google/paligemma-3b-pt-224")
PY
then
  echo "PaliGemma access failed; log in with huggingface-cli or set an authorized HF_TOKEN." >&2
  exit 2
fi

for integer_setting in \
  BATCH_SIZE NUM_WORKERS GRADIENT_ACCUMULATION STEPS SAVE_FREQ EVAL_FREQ \
  MAX_EVAL_SAMPLES LOG_FREQ SCHEDULER_WARMUP_STEPS SCHEDULER_DECAY_STEPS \
  CHUNK_SIZE N_ACTION_STEPS EXPECTED_EPISODES EXPECTED_FPS MIN_GPU_MEMORY_MIB; do
  value=${!integer_setting}
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((value < 1)); then
    echo "${integer_setting} must be a positive integer; got ${value}." >&2
    exit 2
  fi
done
if ((N_ACTION_STEPS > CHUNK_SIZE)); then
  echo "N_ACTION_STEPS must not exceed CHUNK_SIZE." >&2
  exit 2
fi
if [[ "${STATE_REPRESENTATION}" != "joint" && "${STATE_REPRESENTATION}" != "tcp_pose" && "${STATE_REPRESENTATION}" != "tcp_pose_6d" ]]; then
  echo "STATE_REPRESENTATION must be joint, tcp_pose, or tcp_pose_6d; got ${STATE_REPRESENTATION}." >&2
  exit 2
fi
if [[ "${TRAIN_EXPERT_ONLY}" == "true" && "${FREEZE_VISION_ENCODER}" != "true" ]]; then
  echo "TRAIN_EXPERT_ONLY=true already freezes the vision encoder; set FREEZE_VISION_ENCODER=true for an unambiguous run config." >&2
  exit 2
fi
if [[ "${FINETUNE_MODE}" != "default" && "${FINETUNE_MODE}" != "visual_expert" && "${FINETUNE_MODE}" != "visual_full_expert" ]]; then
  echo "FINETUNE_MODE must be default, visual_expert, or visual_full_expert; got ${FINETUNE_MODE}." >&2
  exit 2
fi
if [[ "${FINETUNE_MODE}" != "default" ]] && \
   [[ "${FREEZE_VISION_ENCODER}" != "false" || "${TRAIN_EXPERT_ONLY}" != "false" ]]; then
  echo "visual_expert mode requires both base freeze flags to be false." >&2
  exit 2
fi
python - "${VISION_LR}" "${PROJECTOR_LR}" "${EXPERT_ATTENTION_LR}" "${EXPERT_MLP_LR}" "${ACTION_LR}" \
  "${HIGH_NOISE_FRACTION}" "${HIGH_NOISE_MIN_TIME}" <<'PY'
import sys

for name, raw in zip(
    ("VISION_LR", "PROJECTOR_LR", "EXPERT_ATTENTION_LR", "EXPERT_MLP_LR", "ACTION_LR"),
    sys.argv[1:6], strict=True,
):
    if float(raw) <= 0:
        raise SystemExit(f"{name} must be positive")
fraction, minimum = map(float, sys.argv[6:8])
if not 0 <= fraction <= 1:
    raise SystemExit("HIGH_NOISE_FRACTION must be in [0, 1]")
if not 0 <= minimum < 1:
    raise SystemExit("HIGH_NOISE_MIN_TIME must be in [0, 1)")
PY
python - "${PROPRIOCEPTION_DROPOUT}" <<'PY'
import sys

probability = float(sys.argv[1])
if not 0.0 <= probability < 1.0:
    raise SystemExit("PROPRIOCEPTION_DROPOUT must be in [0, 1)")
PY
if [[ "${IGNORE_GRIPPER_ACTION}" != "true" && "${IGNORE_GRIPPER_ACTION}" != "false" ]]; then
  echo "IGNORE_GRIPPER_ACTION must be true or false; got ${IGNORE_GRIPPER_ACTION}." >&2
  exit 2
fi
GRIPPER_TARGET_NORMALIZED=0.0
if [[ "${IGNORE_GRIPPER_ACTION}" == "true" ]]; then
  GRIPPER_TARGET_NORMALIZED=$(python - "${DATASET_ROOT}" "${NORMALIZATION_MAPPING}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = json.loads(sys.argv[2])["ACTION"]
stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
index = 6
if mode == "QUANTILES":
    low, high = stats["q01"][index], stats["q99"][index]
    value = 2.0 * (0.0 - low) / (high - low) - 1.0
elif mode == "MIN_MAX":
    low, high = stats["min"][index], stats["max"][index]
    value = 2.0 * (0.0 - low) / (high - low) - 1.0
elif mode == "MEAN_STD":
    value = (0.0 - stats["mean"][index]) / stats["std"][index]
else:
    raise SystemExit(f"Unsupported ACTION normalization for gripper no-op: {mode}")
print(value)
PY
  )
fi
IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if ((${#GPU_ARRAY[@]} != NUM_PROCESSES)); then
  echo "GPU_IDS contains ${#GPU_ARRAY[@]} devices but NUM_PROCESSES=${NUM_PROCESSES}." >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
    echo "GPU_IDS must contain unique non-negative integer device IDs; got ${GPU_IDS}." >&2
    exit 2
  fi
  SEEN_GPUS[${gpu_id}]=1
done
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_ARRAY[@]}"; do
    free_mib=$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
      echo "Could not read free memory for GPU ${gpu_id}." >&2
      exit 2
    fi
    if ((free_mib < MIN_GPU_MEMORY_MIB)); then
      echo "GPU ${gpu_id} has only ${free_mib:-unknown} MiB free; refusing to start training." >&2
      exit 2
    fi
  done
fi

python - <<'PY'
try:
    from lerobot.configs.parallelism import ParallelismConfig  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Installed LeRobot lacks the required distributed-training support. Recreate the environment "
        "with bootstrap_remote.sh before training."
    ) from exc
PY

python "${SCRIPT_DIR}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${REPO_ID}" \
  --chunk-size "${CHUNK_SIZE}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --expected-fps "${EXPECTED_FPS}" \
  --state-representation "${STATE_REPRESENTATION}"

python - "${DATASET_ROOT}" "${EVAL_SPLIT}" "${BATCH_SIZE}" \
  "${NUM_PROCESSES}" "${GRADIENT_ACCUMULATION}" "${STEPS}" <<'PY'
import json
import math
import sys
from pathlib import Path

dataset_root, eval_split, batch_size, workers, accumulation, steps = sys.argv[1:]
info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
report = json.loads(
    (Path(dataset_root) / "meta" / "pi05_preparation_report.json").read_text()
)

# LeRobot holds out the last ceil(N * eval_split) episodes for this one-task dataset.
episodes = info["total_episodes"]
held_out = math.ceil(episodes * float(eval_split))
rows = [int(length) for length in report["episode_lengths"]]
if len(rows) != episodes:
    raise SystemExit(
        f"preparation report has {len(rows)} episode lengths, expected {episodes}"
    )
train_frames = sum(rows[: episodes - held_out])
samples_per_microstep = int(batch_size) * int(workers)
optimizer_updates = math.ceil(int(steps) / int(accumulation))
passes = int(steps) * samples_per_microstep / train_frames
print(json.dumps({
    "train_frames": train_frames,
    "samples_per_microstep": samples_per_microstep,
    "effective_optimizer_batch": samples_per_microstep * int(accumulation),
    "optimizer_updates": optimizer_updates,
    "projected_train_passes": passes,
}, indent=2))
PY

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  echo "Set OUTPUT_DIR to a new directory, or use the documented resume command." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PI05_PROPRIO_DROPOUT="${PROPRIOCEPTION_DROPOUT}"
export PI05_IGNORE_GRIPPER_ACTION="${IGNORE_GRIPPER_ACTION}"
export PI05_GRIPPER_TARGET_NORMALIZED="${GRIPPER_TARGET_NORMALIZED}"
export PI05_FINETUNE_MODE="${FINETUNE_MODE}"
export PI05_VISION_LR="${VISION_LR}"
export PI05_PROJECTOR_LR="${PROJECTOR_LR}"
export PI05_EXPERT_ATTENTION_LR="${EXPERT_ATTENTION_LR}"
export PI05_EXPERT_MLP_LR="${EXPERT_MLP_LR}"
export PI05_ACTION_LR="${ACTION_LR}"
export PI05_HIGH_NOISE_FRACTION="${HIGH_NOISE_FRACTION}"
export PI05_HIGH_NOISE_MIN_TIME="${HIGH_NOISE_MIN_TIME}"
export PI05_FIXED_EVAL_SEED="${FIXED_EVAL_SEED}"
export PI05_STATE_REPRESENTATION="${STATE_REPRESENTATION}"

exec torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" \
  "${SCRIPT_DIR}/train_with_state_dropout.py" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --dataset.eval_split="${EVAL_SPLIT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ID}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.normalization_mapping="${NORMALIZATION_MAPPING}" \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --parallelism.dp_replicate="${NUM_PROCESSES}" \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION}" \
  --checkpoint_format=safetensors \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --log_freq="${LOG_FREQ}" \
  --eval_steps="${EVAL_FREQ}" \
  --max_eval_samples="${MAX_EVAL_SAMPLES}" \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --seed=1000
