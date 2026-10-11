# RoboSynChallenge ten-task policy

One evaluation request covers all ten tasks through `policy/ten_task_best`.
Six selected checkpoints are locally trained SmolVLA models; four are released
official checkpoints. `selection_manifest.json` preserves the exact choices
and limitations of the historical selection evidence.

## Source and environment pins

- RoboSynChallenge: `8818c82df5fa2522809b8105e7018950dcc9a9a8`
- EmbodiChain v0.2.4.post1: `89c14da8d0221dc4ee2a4a3e0ba1078b16d447bd`
- Worker LeRobot: `73ab890324b303424a8a5f07272f22c36be5d269`
- Main simulation/ACT: Python 3.11.15; Torch 2.10.0+cu128;
  torchvision 0.25.0+cu128; LeRobot 0.4.4; dexsim-engine 0.4.3.
- Worker: Python 3.12.13; Torch 2.11.0+cu130; LeRobot 0.6.2;
  transformers 5.5.4. The GPU driver must support these CUDA runtimes.

Install the official source environments using the official installation
tutorial. `environment/*_environment_freeze.txt` records every installed package
from the tested fresh environment. `requirements-main.txt` and
`requirements-worker.txt` are derived exact package lists with editable project
entries excluded. Install those pinned project checkouts separately with
`--no-deps`. Rebuild toppra 0.6.3 after the final NumPy 2.2.6 installation. The main requirements use `nvidia-nccl-cu12==2.27.5` instead of the host-local wheel path recorded in the original freeze.
These derived requirements files are supplied for reproduction; an additional
end-to-end installation from those files has not yet been completed.

The main environment dexsim wheel source is the official vendor:
`http://pyp.open3dv.site:2345/packages/dexsim_engine-0.4.3-cp311-cp311-manylinux_2_31_x86_64.whl`.
SmolVLA also downloads the base VLM/processor assets on first use:
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, verified cached revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
Allow official simulator asset downloads and Hugging Face downloads during
setup, or prepopulate the isolated caches. No training dataset is required.

## Install policy and weights

Copy this repository's `policy/ten_task_best` into the pinned official
RoboSynChallenge checkout. Its official `policy/act` and `policy/smolvla`
adapters must be present. Install the pinned worker LeRobot source in the
separate Python 3.12 environment.

Use the private HF repository at this full immutable checkpoint commit (an authorized read token is required):

```bash
python download_checkpoints.py --repo-id CuberQAQ/robosynchallenge-10task-checkpoints-20261011 --revision 3eca6481c2fd642b17a2165ad52ec7aa232a0166 \
  --policy-dir /path/to/RoboSynChallenge/policy/ten_task_best
```

The checkpoint commit is fixed; authenticate using an authorized Hugging Face read token before downloading. Do not place a token in code or run notes. The downloader
checks every checkpoint file's SHA-256 and all ten required model routes.

## Official evaluation entry point

```bash
export EMBODICHAIN_ROOT=/path/to/EmbodiChain
export PYTHON_BIN="$EMBODICHAIN_ROOT/.venv/bin/python"
export SMOLVLA_PYTHON=/path/to/worker-env/bin/python
export SMOLVLA_LEROBOT_ROOT=/path/to/pinned/lerobot
export HF_HOME=/path/to/isolated/hf_cache
export EMBODICHAIN_DATA_ROOT=/path/to/isolated/embodichain_data
export EMBODICHAIN_DATASET_ROOT=/path/to/isolated/dataset_scratch
bash run_all_tasks.sh /path/to/RoboSynChallenge 1 0
# Full local validation:
bash run_all_tasks.sh /path/to/RoboSynChallenge 100 0
```

Single task:

```bash
cd /path/to/RoboSynChallenge
bash policy/ten_task_best/eval.sh sample_loading random 0 \
  --max_episodes 100 --seed 0 --headless True --eval_expert_check True
```

All ten task IDs, architectures, checkpoint steps and fixed official model
revisions are in `selection_manifest.json`. Inputs are the official RGB cameras
and robot state; no depth, intrinsics, point clouds or privileged object state
are consumed by the policy. Output is absolute joint control with official
gripper conversion. Both architectures use 10 executed actions per router call.
ACT reset synchronization is 1; SmolVLA reset synchronization is 0.

The official evaluator and physical task gates are unchanged. The ACT-only
outer runner flushes output and exits only after the unchanged official
evaluator returns normally, avoiding an observed native shutdown GC segfault.
Exceptions, active-evaluation signals and failed task metrics are preserved.
SmolVLA timing covers preprocessing, synchronous worker inference and action
conversion, excluding every `env.step` call.

## Evidence and limitations

Ten routes completed one-episode deployment tests with normal exits in the
fresh environment before the timing-only correction. Eight episodes succeeded;
assembly and sample failed. All ten timing-corrected one-episode tests completed with normal exits by
Oct11 03:41:45HKT: seven successes; bell, handover and sample failed.
The full100/task validation started03:42:01HKT. As of Oct11 12:15HKT, seven tasks completed with normal exits and independently verified raw records: table98%, bell66%, water93%, basket43%, handover38%, drawer72%, mixer83%. Assembly, pipette and sample are not yet complete. The evidence directory records this snapshot. Neither smoke results nor historical offline results
are official held-out competition scores. Report full-test completion separately.
Local timing was on A6000; official efficiency scoring uses RTX 5090.

Training provenance: mixer uses official 1,000 plus independently verified
physical-success-gated synthetic 1,000. Historical table, bell, water, handover
and drawer checkpoints retain training configs and source paths; exact episode
counts/full source QA for these older datasets have not all been independently
re-established. Do not describe them as newly QA-verified 5k datasets. Official
models retain their released training configs and repository revisions.

Official requirements:
- https://robosyn-bench.net/#/faq
- https://robosyn-bench.net/#/evaluation
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html

This private code release pairs with https://huggingface.co/CuberQAQ/robosynchallenge-10task-checkpoints-20261011. No official evaluation request has been submitted.
