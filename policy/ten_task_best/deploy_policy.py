"""Task router for the selected RoboSynChallenge checkpoints.

Install this directory under RoboSynChallenge/policy/ten_task_best.  It uses the
official ACT and SmolVLA adapters already supplied by that repository.
"""

from pathlib import Path
import time
import numpy as np


ROUTES = {
    "table_rearrangement": "smolvla",
    "click_bell": "smolvla",
    "water_pouring": "smolvla",
    "handle_basket": "smolvla",
    "items_handover": "smolvla",
    "drawer_open_place": "smolvla",
    "mixer_operating": "smolvla",
    "item_assembly": "act",
    "manipulate_pipette": "act",
    "sample_loading": "smolvla",
}


class RoutedModel:
    def __init__(self, adapter, model, task_name):
        self.adapter = adapter
        self.model = model
        self.task_name = task_name


class FloatImageWorkerClient:
    """Convert official camera bytes to SmolVLA's documented [0, 1] input."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def infer(self, obs, task):
        converted = dict(obs)
        for key, value in obs.items():
            if key.startswith("observation.images.") and value.dtype == np.uint8:
                converted[key] = value.astype(np.float32) / 255.0
        return self.client.infer(converted, task=task)


class LegacyNormalizedACT:
    """Restore checkpoint normalization removed from newer LeRobot ACT code."""

    def __init__(self, policy, input_stats, action_stats):
        self.policy = policy
        self.input_stats = input_stats
        self.action_stats = action_stats

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def select_action(self, batch):
        normalized = dict(batch)
        for key, (mean, std) in self.input_stats.items():
            if key in normalized:
                normalized[key] = (normalized[key] - mean) / (std + 1e-8)
        action = self.policy.select_action(normalized)
        mean, std = self.action_stats
        return action * std + mean


def _load_act(checkpoint, args):
    """Load full ACT weights without fetching unused ImageNet init weights."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = PreTrainedConfig.from_pretrained(
        str(checkpoint), cli_overrides=[f"--device={args['device']}"]
    )
    # A released full checkpoint overwrites the backbone. LeRobot otherwise
    # downloads its ImageNet initialization before loading those saved weights.
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(str(checkpoint), config=config)
    policy.eval()
    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(image_key, image_key.removeprefix("observation.images."))
    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(args.get("act_step", 10))
    policy.state_obs_path = args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map
    if hasattr(policy, "normalize_inputs") and hasattr(policy, "unnormalize_outputs"):
        return policy

    from safetensors import safe_open
    import torch

    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as saved:
        input_stats = {}
        for key, feature in policy.config.input_features.items():
            mode = policy.config.normalization_mapping.get(feature.type)
            if getattr(mode, "value", mode) != "MEAN_STD":
                raise ValueError(f"Unsupported legacy ACT normalization for {key}: {mode}")
            prefix = "normalize_inputs.buffer_" + key.replace(".", "_")
            input_stats[key] = tuple(saved.get_tensor(prefix + "." + stat).to(policy.act_device)
                                     for stat in ("mean", "std"))
        action_stats = tuple(saved.get_tensor("unnormalize_outputs.buffer_action." + stat)
                             .to(policy.act_device) for stat in ("mean", "std"))
    if not all(torch.isfinite(value).all() for stats in [*input_stats.values(), action_stats]
               for value in stats):
        raise ValueError("Legacy ACT normalization contains non-finite statistics")
    return LegacyNormalizedACT(policy, input_stats, action_stats)


def get_model(usr_args):
    task_name = usr_args.get("task_name")
    if task_name not in ROUTES:
        raise ValueError(f"Unsupported task_name: {task_name!r}")
    checkpoint_root = Path(
        usr_args.get("checkpoint_root") or Path(__file__).resolve().parent / "checkpoints"
    ).resolve()
    checkpoint = checkpoint_root / task_name
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing complete checkpoint for {task_name}: {checkpoint}")

    args = dict(usr_args)
    if ROUTES[task_name] == "smolvla":
        from policy.smolvla import deploy_policy as adapter

        args["checkpoint_dir"] = str(checkpoint)
        args["smolvla_cuda_visible_devices"] = int(args.get("gpu_id", 0))
        bundled_lerobot = Path(__file__).resolve().parents[1] / "smolvla" / "lerobot"
        if not args.get("lerobot_root") and (bundled_lerobot / "src" / "lerobot").is_dir():
            args["lerobot_root"] = str(bundled_lerobot)
    else:
        from policy.act import deploy_policy as adapter

        args["checkpoint_path"] = str(checkpoint)
        args["device"] = f"cuda:{int(args.get('gpu_id', 0))}" if args.get("pytorch_device", "cuda") == "cuda" else args.get("pytorch_device", "cpu")

    model = adapter.get_model(args) if ROUTES[task_name] == "smolvla" else _load_act(checkpoint, args)
    if ROUTES[task_name] == "smolvla":
        model = FloatImageWorkerClient(model)
    return RoutedModel(adapter, model, task_name)


def eval(env, model, obs):
    if ROUTES[model.task_name] == "act":
        return model.adapter.eval(env, model.model, obs)

    # The official SmolVLA adapter returns three values and has no timing.
    # Record preprocessing, synchronous worker RPC and executable action
    # conversion here; never include env.step in the inference metric.
    started = time.perf_counter()
    adapter, worker = model.adapter, model.model
    instruction = getattr(env, "_current_instruction", None) or "click the bell"
    encoded_obs = adapter.encode_obs(obs, image_keys=worker.image_keys)
    actions = worker.infer(encoded_obs, task=instruction)
    worker.maybe_print_action_debug(actions)
    # encode_action depends on the fixed action space and device, not dynamic
    # simulation state. Preconversion leaves the official action order intact.
    executable = [adapter.encode_action(
        action, env,
        rescale_gripper=worker.rescale_gripper,
        gripper_indices=worker.gripper_indices,
        gripper_source_range=worker.gripper_source_range,
    ) for action in actions]
    inference_seconds = time.perf_counter() - started
    final_obs, info, truncated = obs, {}, False
    for action_tensor in executable:
        worker.maybe_print_env_action_debug(action_tensor)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if bool(adapter._to_numpy(truncated).reshape(-1)[0]):
            break
    return final_obs, info, truncated, [inference_seconds]


def reset_model(model):
    return model.adapter.reset_model(model.model)
