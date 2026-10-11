#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
PYTHON_BIN="${PYTHON_BIN:-$EMBODICHAIN_ROOT/.venv/bin/python}"
TASK_NAME="${1:?Usage: bash policy/ten_task_best/eval.sh TASK [random|clear] [GPU_ID] [extra overrides...]}"
SETTING="${2:-random}"
GPU_ID="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
case "$TASK_NAME" in
  table_rearrangement|click_bell|water_pouring|handle_basket|items_handover|drawer_open_place|mixer_operating|item_assembly|manipulate_pipette|sample_loading) ;;
  *) echo "Unknown task: $TASK_NAME" >&2; exit 2 ;;
esac
test -x "$PYTHON_BIN" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
test -s "$SCRIPT_DIR/checkpoints/$TASK_NAME/model.safetensors" || { echo "Missing checkpoint for $TASK_NAME" >&2; exit 2; }
if [[ -z "${SMOLVLA_PYTHON:-}" && -x "$REPO_ROOT/.venv-smolvla/bin/python" ]]; then
  export SMOLVLA_PYTHON="$REPO_ROOT/.venv-smolvla/bin/python"
fi
unset CUDA_VISIBLE_DEVICES
DEFAULT_OVERRIDES=(--eval_reset_sync_steps 0)
EVAL_ENTRY="$REPO_ROOT/scripts/eval_policy.py"
case "$TASK_NAME" in
  item_assembly|manipulate_pipette)
    DEFAULT_OVERRIDES=(--eval_reset_sync_steps 1 --act_step 10)
    EVAL_ENTRY="$SCRIPT_DIR/run_official_eval.py"
    ;;
esac
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
cd "$REPO_ROOT"
PYTHONWARNINGS=ignore::UserWarning EMBODICHAIN_SIM_EXIT_PROCESS=0 \
  "$PYTHON_BIN" "$EVAL_ENTRY" \
    --config policy/ten_task_best/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --gpu_id "$GPU_ID" \
    --smolvla_cuda_visible_devices "$GPU_ID" \
    --checkpoint_root "$SCRIPT_DIR/checkpoints" \
    --model_name "ten_task_best_${TASK_NAME}" \
    "${DEFAULT_OVERRIDES[@]}" \
    "$@"
