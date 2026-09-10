from __future__ import annotations

from pathlib import Path


def success_to_go_targets(successes, gamma: float):
    import torch

    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")

    successes = torch.as_tensor(successes, dtype=torch.bool)
    targets = torch.zeros(successes.shape[0], dtype=torch.float32)
    success_indices = torch.nonzero(successes, as_tuple=False).flatten()
    if success_indices.numel() == 0:
        return targets

    first_success = int(success_indices[0].item())
    for step in range(successes.shape[0]):
        future_successes = success_indices[success_indices >= step]
        if future_successes.numel() == 0:
            continue
        distance = int(future_successes[0].item()) - step
        targets[step] = float(gamma**distance)
    targets[first_success:] = 1.0
    return targets


def _as_2d_tensor(value, *, name: str):
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def build_action_chunk_dataset(
    episodes,
    *,
    action_horizon: int,
    stride: int = 1,
    gamma: float = 0.99,
    obs_key: str = "state",
    action_key: str = "action_policy",
    obs_source: str = "state",
    visual_key: str = "visual_features",
    task_feature_source: str = "none",
    task_feature_dim: int = 0,
    task_feature_key: str = "task_features",
):
    import torch

    from guided_action_flow.training.task_features import hash_token_features

    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if obs_source not in {"state", "visual", "state+visual", "visual_tokens"}:
        raise ValueError(
            "obs_source must be one of {'state', 'visual', 'state+visual', "
            f"'visual_tokens'}}, got {obs_source!r}."
        )
    if task_feature_source not in {"none", "tokens", "episode", "vlm_hidden"}:
        raise ValueError(
            "task_feature_source must be one of "
            "{'none', 'tokens', 'episode', 'vlm_hidden'}."
        )
    use_task_features = task_feature_source != "none"
    if task_feature_source == "tokens" and task_feature_dim < 1:
        raise ValueError("task_feature_dim must be >= 1 when task features are enabled.")

    obs_features = []
    proprio_list = []
    action_chunks = []
    targets = []
    episode_indices = []
    frame_indices = []
    task_features = []

    for episode_index, episode in enumerate(episodes):
        states = _as_2d_tensor(episode[obs_key], name=obs_key)
        actions = _as_2d_tensor(episode[action_key], name=action_key)
        successes = torch.as_tensor(episode["success"], dtype=torch.bool).flatten()
        visuals = None
        if obs_source in {"visual", "state+visual", "visual_tokens"}:
            if visual_key not in episode:
                raise ValueError(
                    f"Episodes must contain {visual_key!r} when obs_source={obs_source!r}. "
                    "Re-collect rollouts with --store-visual-features."
                )
            visuals = _as_2d_tensor(episode[visual_key], name=visual_key)
            if visuals.shape[0] != states.shape[0]:
                raise ValueError(
                    f"{visual_key} and {obs_key} must have the same first dimension, "
                    f"got {visuals.shape[0]} and {states.shape[0]}."
                )
        episode_task_features = None
        if task_feature_source == "tokens":
            if "task_tokens" not in episode or "task_attention_mask" not in episode:
                raise ValueError(
                    "Episodes must contain task_tokens and task_attention_mask when "
                    "task_feature_source='tokens'."
                )
            episode_task_features = hash_token_features(
                episode["task_tokens"],
                episode["task_attention_mask"],
                feature_dim=task_feature_dim,
            ).squeeze(0)
        elif task_feature_source in {"episode", "vlm_hidden"}:
            if task_feature_key not in episode:
                raise ValueError(
                    f"Episodes must contain {task_feature_key!r} when "
                    f"task_feature_source={task_feature_source!r}."
                )
            episode_task_features = torch.as_tensor(
                episode[task_feature_key],
                dtype=torch.float32,
            ).flatten()

        if states.shape[0] != actions.shape[0] or states.shape[0] != successes.shape[0]:
            raise ValueError(
                "state, action, and success tensors must have the same first dimension."
            )
        if actions.shape[0] < action_horizon:
            continue

        returns = success_to_go_targets(successes, gamma=gamma)
        for start in range(0, actions.shape[0] - action_horizon + 1, stride):
            if obs_source == "state":
                obs_features.append(states[start])
            elif obs_source in {"visual", "visual_tokens"}:
                # visual_tokens stores the un-pooled spatial tokens flat; the
                # transformer critic reshapes them. Same obs path as 'visual',
                # only the critic differs. Do NOT concat state here.
                obs_features.append(visuals[start])
            else:
                obs_features.append(torch.cat([states[start], visuals[start]], dim=-1))
            proprio_list.append(states[start])
            action_chunks.append(actions[start : start + action_horizon])
            targets.append(returns[start])
            episode_indices.append(episode_index)
            frame_indices.append(start)
            if episode_task_features is not None:
                task_features.append(episode_task_features)

    if not action_chunks:
        raise ValueError(
            "No action chunks were produced. Check horizon, stride, and rollout length."
        )

    dataset = {
        "obs_features": torch.stack(obs_features),
        "proprio": torch.stack(proprio_list),
        "action_chunks": torch.stack(action_chunks),
        "targets": torch.stack(targets).float(),
        "episode_indices": torch.tensor(episode_indices, dtype=torch.long),
        "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
    }
    if use_task_features:
        dataset["task_features"] = torch.stack(task_features)
    return dataset


def split_indices_by_episode(episode_indices, *, val_fraction: float, generator=None):
    import torch

    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    episode_indices = torch.as_tensor(episode_indices, dtype=torch.long).flatten()
    if episode_indices.numel() == 0:
        raise ValueError("episode_indices must not be empty.")

    all_indices = torch.arange(episode_indices.numel(), dtype=torch.long)
    unique_episodes = torch.unique(episode_indices, sorted=True)
    if val_fraction == 0.0 or unique_episodes.numel() < 2:
        return all_indices, all_indices[:0]

    val_episode_count = int(unique_episodes.numel() * val_fraction)
    val_episode_count = max(1, min(val_episode_count, unique_episodes.numel() - 1))
    order = torch.randperm(unique_episodes.numel(), generator=generator)
    val_episodes = unique_episodes[order[:val_episode_count]]
    val_mask = (episode_indices[:, None] == val_episodes[None, :]).any(dim=1)

    return all_indices[~val_mask], all_indices[val_mask]


def load_episode_files(paths):
    import torch

    episodes = []
    for path in paths:
        episodes.append(torch.load(Path(path), map_location="cpu", weights_only=False))
    return episodes


def discover_episode_files(data_dir: str | Path):
    data_path = Path(data_dir)
    return sorted(data_path.rglob("episode_*.pt"))
