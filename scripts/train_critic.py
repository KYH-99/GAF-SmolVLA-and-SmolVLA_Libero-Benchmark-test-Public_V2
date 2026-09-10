#!/usr/bin/env python
"""Train an action-chunk critic from collected rollout data.

v2 supports a multi-modal critic (state + visual/video + language + action, each
encoded separately then fused) trained pooled over a task family, per the
supervisor's design brief. The legacy concat critic is still available via
``--critic-arch concat``. After training we run the Phase-3 ranking checks
(Q-return correlation, success/fail value separation) — that is how we confirm
the critic is *usable*, independent of whether QGF raises success on any task.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _select_checkpoint_metadata(history: list[dict]) -> dict:
    if not history:
        return {"selected_epoch": None, "selected_metric": "none", "selected_val_loss": None}
    val_entries = [entry for entry in history if "val_loss" in entry]
    if not val_entries:
        return {"selected_epoch": history[-1]["epoch"], "selected_metric": "final", "selected_val_loss": None}
    best = min(val_entries, key=lambda entry: entry["val_loss"])
    return {"selected_epoch": best["epoch"], "selected_metric": "val_loss", "selected_val_loss": best["val_loss"]}


def _clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    # critic architecture (v2 default = multimodal)
    parser.add_argument("--critic-arch", choices=["multimodal", "concat", "transformer"], default="multimodal")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-chunk-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--visual-tokens", type=int, default=0, help="0=pooled(StageA); >0=N un-pooled visual tokens(StageB)")
    parser.add_argument("--visual-token-grid", type=int, default=0, help="GxG per camera used at collect time; stored for QGF inference.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--visual-frames", type=int, default=1, help=">1 = video window over recent frames.")
    parser.add_argument("--no-action-temporal", action="store_true", help="Flatten action chunk instead of temporal conv.")
    # inputs
    parser.add_argument(
        "--obs-source", choices=["state", "visual", "state+visual", "visual_tokens"], default="visual",
        help="What goes into the critic's obs/visual channel. 'visual'/'state+visual' "
        "require rollouts collected with --store-visual-features.",
    )
    parser.add_argument(
        "--task-feature-source", choices=["none", "tokens", "episode", "vlm_hidden"], default="tokens",
        help="Language channel. 'tokens' hashes the stored instruction tokens.",
    )
    parser.add_argument("--task-feature-dim", type=int, default=128)
    parser.add_argument("--task-feature-key", default="task_features")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--input-layernorm", action="store_true", default=True)
    # optimization
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--loss", choices=["mse", "bce"], default="bce")
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--neg-target-scale", type=float, default=0.5)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from guided_action_flow.critics.action_chunk_critic import ActionChunkCritic, ActionChunkCriticConfig
    from guided_action_flow.critics.multimodal_critic import MultiModalCritic, MultiModalCriticConfig
    from guided_action_flow.critics.validation import validate_critic
    from guided_action_flow.training.critic_dataset import (
        build_action_chunk_dataset, discover_episode_files, load_episode_files, split_indices_by_episode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = discover_episode_files(args.data_dir)
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.pt files found under {args.data_dir}.")
    episodes = load_episode_files(episode_files)

    dataset = build_action_chunk_dataset(
        episodes,
        action_horizon=args.action_horizon,
        stride=args.stride,
        gamma=args.gamma,
        obs_source=args.obs_source,
        task_feature_source=args.task_feature_source,
        task_feature_dim=args.task_feature_dim,
        task_feature_key=args.task_feature_key,
    )
    obs_features = dataset["obs_features"]
    proprio = dataset["proprio"]
    action_chunks = dataset["action_chunks"]
    targets = dataset["targets"]
    task_features = dataset.get("task_features")

    # per-sample weights: upweight samples from failed episodes
    episode_success = torch.tensor(
        [bool(torch.as_tensor(ep["success"]).any().item()) for ep in episodes], dtype=torch.bool
    )
    sample_episode_success = episode_success[dataset["episode_indices"]]
    sample_weights = torch.where(
        sample_episode_success, torch.ones_like(targets), torch.full_like(targets, float(args.failure_weight))
    )

    generator = torch.Generator().manual_seed(args.seed)
    num_samples = targets.shape[0]
    train_indices, val_indices = split_indices_by_episode(
        dataset["episode_indices"], val_fraction=args.val_fraction, generator=generator
    )

    # dataset columns (fixed order): obs, proprio, action, [task], target, weight
    cols = [obs_features, proprio, action_chunks]
    has_task = task_features is not None
    if has_task:
        cols.append(task_features)
    cols += [targets, sample_weights]
    train_ds = TensorDataset(*[c[train_indices] for c in cols])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)

    visual_dim = obs_features.shape[-1]
    if args.visual_frames > 1:
        if visual_dim % args.visual_frames != 0:
            raise ValueError("obs_features dim must be divisible by --visual-frames.")
        visual_dim = visual_dim // args.visual_frames

    if args.critic_arch == "transformer":
        from guided_action_flow.critics.transformer_critic import (
            TransformerCritic,
            TransformerCriticConfig,
        )

        critic_config = TransformerCriticConfig(
            action_dim=action_chunks.shape[-1],
            action_horizon=action_chunks.shape[1],
            state_dim=proprio.shape[-1],
            visual_dim=obs_features.shape[-1] if args.visual_tokens == 0 else (obs_features.shape[-1] // args.visual_tokens),
            visual_tokens=args.visual_tokens,
            language_dim=0 if not has_task else task_features.shape[-1],
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            depth=args.depth,
            action_chunk_tokens=args.action_chunk_tokens,
            dropout=args.dropout,
            input_layernorm=args.input_layernorm,
        )
        critic = TransformerCritic(critic_config)
    elif args.critic_arch == "multimodal":
        critic_config = MultiModalCriticConfig(
            action_dim=action_chunks.shape[-1],
            action_horizon=action_chunks.shape[1],
            state_dim=proprio.shape[-1],
            visual_dim=obs_features.shape[-1] if args.visual_frames == 1 else visual_dim,
            visual_frames=args.visual_frames,
            language_dim=0 if not has_task else task_features.shape[-1],
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            action_temporal=not args.no_action_temporal,
            input_layernorm=args.input_layernorm,
        )
        critic = MultiModalCritic(critic_config)
    else:
        critic_config = ActionChunkCriticConfig(
            obs_feature_dim=obs_features.shape[-1],
            action_dim=action_chunks.shape[-1],
            action_horizon=action_chunks.shape[1],
            proprio_dim=proprio.shape[-1],
            task_feature_dim=0 if not has_task else task_features.shape[-1],
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            input_layernorm=args.input_layernorm,
        )
        critic = ActionChunkCritic(critic_config)
    critic.module.to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss == "bce":
        _bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        def per_sample_loss(pred, target):
            return _bce(pred, target)
    else:
        def per_sample_loss(pred, target):
            return F.mse_loss(pred, target, reduction="none")

    def weighted_mean(per_sample, weight):
        return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)

    def critic_forward(obs_b, proprio_b, action_b, task_b):
        return critic(
            obs_features=obs_b, action_chunk=action_b, proprio=proprio_b, task_features=task_b,
        )

    history = []
    best_val_loss = None
    best_model_state_dict = None
    for epoch in range(args.epochs):
        critic.module.train()
        train_losses = []
        for batch in train_loader:
            it = iter(batch)
            obs_b = next(it).to(device)
            proprio_b = next(it).to(device)
            action_b = next(it).to(device)
            task_b = next(it).to(device) if has_task else None
            target_b = next(it).to(device)
            weight_b = next(it).to(device)

            pred = critic_forward(obs_b, proprio_b, action_b, task_b)
            loss = weighted_mean(per_sample_loss(pred, target_b), weight_b)

            if args.action_noise_std > 0.0:
                noisy = action_b + torch.randn_like(action_b) * args.action_noise_std
                pred_neg = critic_forward(obs_b, proprio_b, noisy, task_b)
                target_neg = (target_b * args.neg_target_scale).clamp(0.0, 1.0)
                loss = loss + args.neg_weight * weighted_mean(per_sample_loss(pred_neg, target_neg), weight_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        metrics = {"epoch": epoch, "train_loss": float(sum(train_losses) / max(1, len(train_losses)))}
        if val_indices.numel() > 0:
            critic.module.eval()
            with torch.no_grad():
                val_pred = critic_forward(
                    obs_features[val_indices].to(device),
                    proprio[val_indices].to(device),
                    action_chunks[val_indices].to(device),
                    task_features[val_indices].to(device) if has_task else None,
                )
                val_loss = per_sample_loss(val_pred, targets[val_indices].to(device)).mean()
            metrics["val_loss"] = float(val_loss.detach().cpu().item())
            if best_val_loss is None or metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_model_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
        history.append(metrics)
        print(json.dumps(metrics))

    final_model_state_dict = _clone_state_dict_to_cpu(critic.state_dict())
    selection = _select_checkpoint_metadata(history)
    selected_state_dict = (
        best_model_state_dict
        if selection["selected_metric"] == "val_loss" and best_model_state_dict is not None
        else final_model_state_dict
    )

    # Phase-3 ranking validation (does the critic rank action chunks correctly?)
    critic.load_state_dict(selected_state_dict)
    critic.module.to(device).eval()
    ranking = validate_critic(critic, dataset, device=device_name)

    checkpoint = {
        "critic_arch": args.critic_arch,
        "critic_config": asdict(critic_config),
        "model_state_dict": selected_state_dict,
        "final_model_state_dict": final_model_state_dict,
        "training_args": vars(args),
        "num_samples": int(num_samples),
        "num_episodes": len(episodes),
        "num_task_ids": len({int(ep.get("task_id", -1)) for ep in episodes}),
        "task_feature_source": args.task_feature_source,
        "task_feature_key": args.task_feature_key,
        "task_feature_dim": 0 if not has_task else int(task_features.shape[-1]),
        "obs_source": args.obs_source,
        "visual_token_grid": int(args.visual_token_grid),
        "target_mean": float(targets.mean().item()),
        "target_success_fraction": float((targets > 0).float().mean().item()),
        "ranking_validation": ranking,
        "history": history,
        **selection,
    }
    torch.save(checkpoint, output_dir / "critic.pt")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "num_samples": int(num_samples),
                "num_episodes": len(episodes),
                "num_task_ids": checkpoint["num_task_ids"],
                "obs_source": args.obs_source,
                "task_feature_source": args.task_feature_source,
                "target_success_fraction": checkpoint["target_success_fraction"],
                "ranking_validation": ranking,
                "final": history[-1] if history else {},
                "selected": selection,
            },
            f,
            indent=2,
        )
    print("ranking_validation:", json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
