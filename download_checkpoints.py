#!/usr/bin/env python3
"""Install all ten checkpoints from an explicit, immutable HF revision."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--revision", required=True, help="Full immutable HF commit SHA")
parser.add_argument("--policy-dir", type=Path, required=True)
args = parser.parse_args()
if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
    parser.error("revision must be a full 40-character commit SHA")
args.policy_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=args.repo_id, revision=args.revision,
                  allow_patterns=["checkpoints/**", "checkpoint_checksums.sha256"],
                  local_dir=str(args.policy_dir))
manifest = args.policy_dir / "checkpoint_checksums.sha256"
if not manifest.is_file():
    raise RuntimeError("HF repository is missing checkpoint_checksums.sha256")
verified = []
for line in manifest.read_text().splitlines():
    expected, relative = line.split("  ", 1)
    path = (args.policy_dir / relative).resolve()
    if not path.is_relative_to(args.policy_dir.resolve()):
        raise RuntimeError(f"Unsafe manifest path: {relative}")
    actual = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA mismatch: {relative}")
    verified.append(relative)
tasks = ["table_rearrangement", "click_bell", "water_pouring", "handle_basket",
         "items_handover", "drawer_open_place", "mixer_operating", "item_assembly",
         "manipulate_pipette", "sample_loading"]
for task in tasks:
    for filename in ("model.safetensors", "config.json"):
        if f"checkpoints/{task}/{filename}" not in verified:
            raise RuntimeError(f"Unverified required file: {task}/{filename}")
print(json.dumps({"repo_id":args.repo_id, "revision":args.revision,
                  "verified_files":len(verified), "tasks":tasks}))
