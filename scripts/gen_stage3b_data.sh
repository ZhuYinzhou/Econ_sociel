#!/usr/bin/env bash
set -euo pipefail

# Generate datasets for Stage3b (core action imitation) in one go.
# Output directories are aligned with ECON/examples/configs/hisim_stage3b.yaml:
# - data/stage_3b_action_<topic>_<event>
#
# Key options:
# - action-imitation-observation-mode:
#   - legacy: use same-stage(t) context (old behavior)
#   - sync_prev_stage: use prev-stage(t-1) context (recommended for sync-stage env)
# - action-imitation-target-mode:
#   - tp1: predict next-stage(t+1) action_type (old behavior)
#   - t:   clone current-stage(t) action_type (recommended for sync-stage env)
# - action-imitation-z-t-source:
#   - macro_secondary_majority_dist: use macro-derived secondary-user dist as z_t
#   - s3a_rollout: rollout z_t using Stage3a belief_encoder.population_update_head
#   - none: no z_t conditioning

ROOT="/home/zhuyinzhou/MAS/ECON"
HISIM_DATA_ROOT="/data/zhuyinzhou/HiSim/data"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

TOPIC="${TOPIC:-metoo}"
EVENTS=(${EVENTS:-e1 e2})

PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-1024}"
PROMPT_TOKENIZER_NAME="${PROMPT_TOKENIZER_NAME:-gpt2}"
GROUP_REPR_DIM="${GROUP_REPR_DIM:-128}"

# S3b semantics (recommended defaults for your current project)
AI_OBS_MODE="${AI_OBS_MODE:-sync_prev_stage}"     # legacy | sync_prev_stage
AI_TARGET_MODE="${AI_TARGET_MODE:-t}"             # tp1 | t
AI_Z_T_SOURCE="${AI_Z_T_SOURCE:-s3a_rollout}"     # macro_secondary_majority_dist | s3a_rollout | none
S3A_CKPT_DIR="${S3A_CKPT_DIR:-/data/zhuyinzhou/ECON/models/checkpoints_s3a/final}"

# Optional: clean existing output dirs (DANGEROUS)
CLEAN="${CLEAN:-0}"

cd "${ROOT}"

run_py () {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    python "$@"
    return 0
  fi
  if [[ "${USE_CONDA:-1}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    ENV_NAME="${ENV_NAME:-HiSim}"
    if conda run -n "${ENV_NAME}" python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
      conda run -n "${ENV_NAME}" python "$@"
      return 0
    else
      echo "[WARN] conda run not usable in this shell; falling back to plain python. (Tip: USE_CONDA=0 to silence this)"
    fi
  fi
  python "$@"
}

maybe_clean_dir () {
  local d="$1"
  if [[ "${CLEAN}" == "1" ]]; then
    echo "[CLEAN] rm -rf ${d}"
    rm -rf "${d}"
  fi
}

echo "=== Config (S3b data) ==="
echo "ROOT=${ROOT}"
echo "HISIM_DATA_ROOT=${HISIM_DATA_ROOT}"
echo "TOPIC=${TOPIC}"
echo "EVENTS=${EVENTS[*]}"
echo "PROMPT_MAX_TOKENS=${PROMPT_MAX_TOKENS}"
echo "PROMPT_TOKENIZER_NAME=${PROMPT_TOKENIZER_NAME}"
echo "GROUP_REPR_DIM=${GROUP_REPR_DIM}"
echo "AI_OBS_MODE=${AI_OBS_MODE}"
echo "AI_TARGET_MODE=${AI_TARGET_MODE}"
echo "AI_Z_T_SOURCE=${AI_Z_T_SOURCE}"
echo "S3A_CKPT_DIR=${S3A_CKPT_DIR}"
echo "CLEAN=${CLEAN}"
echo ""

for EV in "${EVENTS[@]}"; do
  echo "============================"
  echo "Generating Stage3b dataset for ${TOPIC} / ${EV}"
  echo "============================"

  OUT_AI="/home/zhuyinzhou/MAS/ECON/data/stage_3b_action_${TOPIC}_${EV}"
  OUT_BASE="/home/zhuyinzhou/MAS/ECON/data/_tmp_convert_${TOPIC}_${EV}_for_s3b"

  maybe_clean_dir "${OUT_AI}"
  maybe_clean_dir "${OUT_BASE}"

  # Note: convert_hisim_to_econ_dataset.py always generates a belief dataset into --out-dir as well.
  # We isolate it into OUT_BASE to keep Stage3b dataset directory clean.
  cmd=(
    convert_hisim_to_econ_dataset.py
    --hisim-data-root "${HISIM_DATA_ROOT}"
    --topics "${TOPIC}"
    --events "${EV}"
    --out-dir "${OUT_BASE}"
    --export-action-imitation-dataset
    --action-imitation-out-dir "${OUT_AI}"
    --action-imitation-observation-mode "${AI_OBS_MODE}"
    --action-imitation-target-mode "${AI_TARGET_MODE}"
    --action-imitation-z-t-source "${AI_Z_T_SOURCE}"
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}"
    --prompt-tokenizer-name "${PROMPT_TOKENIZER_NAME}"
    --population-scope secondary
    --population-text-source micro
    --nonparam-group-repr-dim "${GROUP_REPR_DIM}"
  )

  # If using s3a_rollout, pass ckpt path
  if [[ "${AI_Z_T_SOURCE}" == "s3a_rollout" ]]; then
    cmd+=( --s3a-belief-encoder-path "${S3A_CKPT_DIR}" )
    cmd+=( --s3a-rollout-belief-dim 128 --s3a-rollout-population-belief-dim 3 --s3a-rollout-n-stages 13 )
  fi

  echo "[S3b] -> ${OUT_AI}"
  run_py "${cmd[@]}"
  echo ""
done

echo "DONE."

