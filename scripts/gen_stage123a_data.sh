#!/usr/bin/env bash
set -euo pipefail

# Generate datasets for Stage1 / Stage2 / Stage3a in one go.
# - Stage1(core): predict self stance(t+1), prev-stage observation in prompt
# - Stage2(noncore): predict self stance(t+1), prev-stage observation in prompt
# - Stage3a(z-transition): predict population z(t+1), prev-stage observation in prompt + B2-2 nonparam group_repr
#
# This script writes to the exact default directories used in the example configs:
# - data/stage_1_dataset_<topic>_<event>
# - data/stage_2_dataset_<topic>_<event>
# - data/stage_3a_dataset_<topic>_<event>_z_transition

ROOT="/home/zhuyinzhou/MAS/ECON"
HISIM_DATA_ROOT="/data/zhuyinzhou/HiSim/data"

TOPIC="${TOPIC:-metoo}"
EVENTS=(${EVENTS:-e1 e2})

# Prompt / truncation
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-1024}"
PROMPT_TOKENIZER_NAME="${PROMPT_TOKENIZER_NAME:-gpt2}"

# Observation semantics
BELIEF_OBS_MODE="${BELIEF_OBS_MODE:-prev_stage}"
ZTRANS_OBS_MODE="${ZTRANS_OBS_MODE:-prev_stage}"

# B2-2 nonparam group repr
GROUP_REPR_DIM="${GROUP_REPR_DIM:-128}"

# Optional: clean existing output dirs (DANGEROUS)
CLEAN="${CLEAN:-0}"

cd "${ROOT}"

run_py () {
  if command -v conda >/dev/null 2>&1; then
    # Default to HiSim env if available
    ENV_NAME="${ENV_NAME:-HiSim}"
    conda run -n "${ENV_NAME}" python "$@"
  else
    python "$@"
  fi
}

maybe_clean_dir () {
  local d="$1"
  if [[ "${CLEAN}" == "1" ]]; then
    echo "[CLEAN] rm -rf ${d}"
    rm -rf "${d}"
  fi
}

echo "=== Config ==="
echo "ROOT=${ROOT}"
echo "HISIM_DATA_ROOT=${HISIM_DATA_ROOT}"
echo "TOPIC=${TOPIC}"
echo "EVENTS=${EVENTS[*]}"
echo "PROMPT_MAX_TOKENS=${PROMPT_MAX_TOKENS}"
echo "PROMPT_TOKENIZER_NAME=${PROMPT_TOKENIZER_NAME}"
echo "BELIEF_OBS_MODE=${BELIEF_OBS_MODE}"
echo "ZTRANS_OBS_MODE=${ZTRANS_OBS_MODE}"
echo "GROUP_REPR_DIM=${GROUP_REPR_DIM}"
echo "CLEAN=${CLEAN}"
echo ""

for EV in "${EVENTS[@]}"; do
  echo "============================"
  echo "Generating datasets for ${TOPIC} / ${EV}"
  echo "============================"

  OUT_S1="/home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_${TOPIC}_${EV}"
  OUT_S2="/home/zhuyinzhou/MAS/ECON/data/stage_2_dataset_${TOPIC}_${EV}"
  OUT_S3A="/home/zhuyinzhou/MAS/ECON/data/stage_3a_dataset_${TOPIC}_${EV}_z_transition"

  maybe_clean_dir "${OUT_S1}"
  maybe_clean_dir "${OUT_S2}"
  maybe_clean_dir "${OUT_S3A}"

  # ---- Stage1 (core stance(t+1)) ----
  echo "[S1] -> ${OUT_S1}"
  run_py convert_hisim_to_econ_dataset.py \
    --hisim-data-root "${HISIM_DATA_ROOT}" \
    --topics "${TOPIC}" \
    --events "${EV}" \
    --user-scope core \
    --out-dir "${OUT_S1}" \
    --core-target-mode self \
    --belief-observation-mode "${BELIEF_OBS_MODE}" \
    --population-scope secondary \
    --population-text-source micro \
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}" \
    --prompt-tokenizer-name "${PROMPT_TOKENIZER_NAME}" \
    --nonparam-group-repr-dim "${GROUP_REPR_DIM}"

  # ---- Stage2 (noncore stance(t+1)) ----
  echo "[S2] -> ${OUT_S2}"
  run_py convert_hisim_to_econ_dataset.py \
    --hisim-data-root "${HISIM_DATA_ROOT}" \
    --topics "${TOPIC}" \
    --events "${EV}" \
    --user-scope noncore \
    --out-dir "${OUT_S2}" \
    --noncore-target-mode self \
    --belief-observation-mode "${BELIEF_OBS_MODE}" \
    --no-user-history \
    --population-scope secondary \
    --population-text-source micro \
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}" \
    --prompt-tokenizer-name "${PROMPT_TOKENIZER_NAME}" \
    --nonparam-group-repr-dim "${GROUP_REPR_DIM}"

  # ---- Stage3a (z-transition) ----
  echo "[S3a] -> ${OUT_S3A}"
  run_py convert_hisim_to_econ_dataset.py \
    --hisim-data-root "${HISIM_DATA_ROOT}" \
    --topics "${TOPIC}" \
    --events "${EV}" \
    --out-dir "${OUT_S3A}" \
    --export-z-transition-dataset \
    --z-transition-out-dir "${OUT_S3A}" \
    --z-transition-population-mode dist \
    --z-transition-conditioning core_user \
    --z-transition-split-strategy random_by_user \
    --z-transition-split-seed 42 \
    --z-transition-observation-mode "${ZTRANS_OBS_MODE}" \
    --population-scope secondary \
    --population-text-source micro \
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}" \
    --prompt-tokenizer-name "${PROMPT_TOKENIZER_NAME}" \
    --nonparam-group-repr-dim "${GROUP_REPR_DIM}"

  echo ""
done

echo "DONE."

