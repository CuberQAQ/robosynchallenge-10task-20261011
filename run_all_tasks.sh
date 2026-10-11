#!/usr/bin/env bash
set -euo pipefail
repo=${1:?Usage: run_all_tasks.sh ROBO_SYN_REPO [EPISODES] [GPU]}
episodes=${2:-100}
gpu=${3:-0}
cd "$repo"
for task in table_rearrangement click_bell water_pouring handle_basket items_handover drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash policy/ten_task_best/eval.sh "$task" random "$gpu" \
    --max_episodes "$episodes" --seed 0 --headless True \
    --eval_expert_check True --eval_video_log False
done
