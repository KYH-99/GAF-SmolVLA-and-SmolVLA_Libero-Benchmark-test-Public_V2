#!/usr/bin/env python
"""Train an action-chunk critic with IQL instead of MC/BCE regression.

Motivation (Haichen's next step). Stage B showed that a finer *visual*
representation (spatial tokens + attention) raised the critic's ranking quality
(Q-return corr 0.48->0.62, success/fail sep 3.79->4.0) but did NOT translate
into stable QGF guidance on viewpoint tasks -- "排序对 != 梯度对". The bottleneck
is the *gradient* quality of Q(s,a), not the visual granularity. This script
attacks that by changing how Q is trained: from supervised regression onto the
success-to-go return, to **Implicit Q-Learning** (Kostrikov et al. 2021):

    V(s)     <- expectile_tau( Q_target(s, a) )              (asymmetric L2)
    Q(s, a)  <- r_chunk + gamma^k * (1 - done) * V(s')       (SARSA-style TD)

Crucially the Q network is the *same* TransformerCritic used at Stage B and by
QGF, so the produced checkpoint is a drop-in `critic_arch="transformer"` file:
`critics/checkpoint.load_action_chunk_critic` loads it unchanged, and
`policies/*_qgf.py` / `guidance/qgf.py` are untouched. The ONLY changed variable
vs Stage B is the training objective -- that is the clean ablation.

Chunk MDP. Each training sample is one committed action chunk of `action_horizon`
steps starting at frame `t`. Its transition is:
  s  = obs at t,  a = actions[t:t+H]
  s' = obs at t' where t' = min(t + H, T-1)   (state after committing the chunk)
  r_chunk = sum_{j=0}^{k-1} gamma^j * reward[t+j],  k = t' - t
  done = an episode `done` flag fired inside [t, t')  (no bootstrap past it)
LIBERO reward is sparse (terminal success only), which is exactly why we want a
decoupled V: expectile regression avoids querying Q on OOD actions in the target.

Saved checkpoint mirrors scripts/train_critic.py so downstream tooling is
identical, plus an `iql` block with the extra hyper-parameters and V state dict
(for analysis / resuming; QGF never reads it).
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path


def _clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    # critic architecture (Q net = transformer, matching Stage B / QGF)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-chunk-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--visual-tokens", type=int, default=0, help="0=pooled(StageA); >0=N un-pooled visual tokens(StageB)")
    parser.add_argument("--visual-token-grid", type=int, default=0, help="GxG per camera used at collect time; stored for QGF inference.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    # inputs (same choices/semantics as train_critic.py)
    parser.add_argument(
        "--obs-source", choices=["state", "visual", "state+visual", "visual_tokens"], default="visual_tokens",
    )
    parser.add_argument(
        "--task-feature-source", choices=["none", "tokens", "episode", "vlm_hidden"], default="tokens",
    )
    parser.add_argument("--task-feature-dim", type=int, default=128)
    parser.add_argument("--task-feature-key", default="task_features")
    parser.add_argument("--input-layernorm", action="store_true", default=True)
    # IQL hyper-parameters
    parser.add_argument("--expectile", type=float, default=0.7, help="IQL tau; higher = more optimistic V.")
    parser.add_argument("--polyak", type=float, default=0.005, help="Target-Q Polyak averaging rate.")
    parser.add_argument(
        "--bc-weight", type=float, default=0.0,
        help="Weight of an auxiliary supervised term pulling Q toward the "
        "success-to-go return (the Stage-B objective). 0 = pure IQL; >0 = hybrid "
        "IQL+supervised on the SAME transformer critic. Large = approaches pure supervised.",
    )
    parser.add_argument("--reward-key", default="reward")
    parser.add_argument("--done-key", default="done")
    parser.add_argument(
        "--reward-from-success", dest="reward_from_success", action="store_true", default=True,
        help="Synthesize sparse reward from the success flag (LIBERO's reward field is empty).",
    )
    parser.add_argument(
        "--no-reward-from-success", dest="reward_from_success", action="store_false",
        help="Use the stored reward/done fields verbatim instead.",
    )
    # optimization
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def _episode_tensors(ep, obs_key, visual_key, reward_key, done_key, reward_from_success):
    """Convert one episode's arrays to tensors ONCE (cached per episode).

    In LIBERO the stored ``reward`` field is all-zeros -- the only task-completion
    signal lives in the boolean ``success`` field, and success typically flips
    True only on the FINAL step before the episode ends. With
    ``reward_from_success`` (default) we treat the episode as a sparse-reward MDP:
    reward 1.0 at the first-success step ``fs`` (terminal), else terminal failure
    at the episode's last step. We store ``fs`` (first-success index or -1) and
    ``te`` (terminal index) so the chunk builder can credit success reached at the
    LANDING step too -- keying off a half-open ``[t, tp)`` window drops the
    terminal-step reward and collapses Q to a constant.
    """
    import torch

    states = torch.as_tensor(ep[obs_key], dtype=torch.float32)
    if states.ndim == 1:
        states = states.unsqueeze(-1)
    T = states.shape[0]
    visuals = None
    if visual_key in ep:
        visuals = torch.as_tensor(ep[visual_key], dtype=torch.float32)
        if visuals.ndim == 1:
            visuals = visuals.unsqueeze(-1)

    if reward_from_success:
        succ = torch.as_tensor(ep["success"], dtype=torch.bool).flatten()
        fs = int(succ.nonzero()[0].item()) if succ.any() else -1
        te = fs if fs >= 0 else (T - 1)
    else:
        dones = torch.as_tensor(ep[done_key], dtype=torch.bool).flatten()
        rew = torch.as_tensor(ep[reward_key], dtype=torch.float32).flatten()
        pos = (rew > 0).nonzero()
        fs = int(pos[0].item()) if pos.numel() else -1
        de = dones.nonzero()
        te = int(de[0].item()) if de.numel() else (T - 1)
        if fs >= 0:
            te = min(te, fs)
    return {"states": states, "visuals": visuals, "fs": fs, "te": te, "T": T}


def _obs_row(cache, obs_source, idx):
    """Return the obs vector for one frame, matching build_action_chunk_dataset."""
    import torch

    if obs_source == "state":
        return cache["states"][idx]
    if obs_source in {"visual", "visual_tokens"}:
        return cache["visuals"][idx]
    return torch.cat([cache["states"][idx], cache["visuals"][idx]], dim=-1)


def _build_iql_transitions(episodes, dataset, *, action_horizon, gamma, obs_source,
                           obs_key, visual_key, reward_key, done_key, reward_from_success):
    """Augment the base (s,a,target) dataset with IQL transition tensors.

    Aligned 1:1 with dataset rows via (episode_indices, frame_indices), so the
    same sample ordering / splits apply. Returns next_obs, next_proprio,
    reward_chunk, not_done, and the effective per-sample discount gamma^k.
    Episode arrays are converted once and cached (visual_features is large).
    """
    import torch

    ep_idx = dataset["episode_indices"].tolist()
    fr_idx = dataset["frame_indices"].tolist()

    cache_by_ei: dict[int, dict] = {}
    next_obs, next_proprio, r_chunk, not_done, disc = [], [], [], [], []
    for ei, t in zip(ep_idx, fr_idx):
        cache = cache_by_ei.get(ei)
        if cache is None:
            cache = _episode_tensors(episodes[ei], obs_key, visual_key, reward_key, done_key, reward_from_success)
            cache_by_ei[ei] = cache
        states, fs, te = cache["states"], cache["fs"], cache["te"]

        # land at tp = t + H, but never step past the episode terminal te
        tp = min(t + action_horizon, te)
        if tp >= te:
            # this chunk reaches the terminal step: no bootstrap
            not_done.append(0.0)
            if fs >= 0 and fs <= te:
                r_chunk.append(float(gamma ** max(0, fs - t)))  # discounted success reward (=1 if already solved)
            else:
                r_chunk.append(0.0)                        # terminal failure
            disc.append(0.0)
        else:
            # interior transition: sparse reward is 0, bootstrap from V(s_tp)
            not_done.append(1.0)
            r_chunk.append(0.0)
            disc.append(float(gamma ** (tp - t)))
        next_obs.append(_obs_row(cache, obs_source, tp))
        next_proprio.append(states[tp])

    out = {
        "next_obs_features": torch.stack(next_obs),
        "next_proprio": torch.stack(next_proprio),
        "reward_chunk": torch.tensor(r_chunk, dtype=torch.float32),
        "not_done": torch.tensor(not_done, dtype=torch.float32),
        "discount": torch.tensor(disc, dtype=torch.float32),
    }
    return out


def _make_value_net(cfg, device):
    """V(s): same token trunk as the Q critic but WITHOUT the action tokens.

    Reuses TransformerCritic's projections/encoder shapes for a fair comparison;
    reads a scalar V from the CLS token. Implemented as a real nn.Module.
    """
    import torch
    import torch.nn as nn

    d = cfg.embed_dim

    class ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            if cfg.state_dim > 0:
                self.state_proj = _proj(cfg.state_dim, d, cfg.input_layernorm)
            if cfg.visual_dim > 0:
                self.visual_proj = _proj(cfg.visual_dim, d, cfg.input_layernorm)
            if cfg.language_dim > 0:
                self.language_proj = _proj(cfg.language_dim, d, cfg.input_layernorm)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads, dim_feedforward=4 * d,
                dropout=cfg.dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.type_emb = nn.Parameter(torch.zeros(4, d))  # cls, state, visual, language
            nn.init.normal_(self.cls, std=0.02)
            nn.init.normal_(self.type_emb, std=0.02)

        def forward(self, obs_features=None, proprio=None, task_features=None):
            B = (proprio if proprio is not None else obs_features).shape[0]
            tokens = [self.cls.expand(B, -1, -1) + self.type_emb[0]]
            if hasattr(self, "state_proj"):
                tokens.append(self.state_proj(proprio.reshape(B, -1)).unsqueeze(1) + self.type_emb[1])
            if hasattr(self, "visual_proj") and obs_features is not None:
                if self.cfg.visual_tokens > 0:
                    v = obs_features.reshape(B, self.cfg.visual_tokens, self.cfg.visual_dim)
                else:
                    v = obs_features.reshape(B, 1, -1)
                tokens.append(self.visual_proj(v) + self.type_emb[2])
            if hasattr(self, "language_proj"):
                tokens.append(self.language_proj(task_features.reshape(B, -1)).unsqueeze(1) + self.type_emb[3])
            enc = self.encoder(torch.cat(tokens, dim=1))
            return self.head(enc[:, 0]).squeeze(-1)

    return ValueNet().to(device)


def _proj(in_dim, embed_dim, ln):
    import torch.nn as nn
    layers = []
    if ln:
        layers.append(nn.LayerNorm(in_dim))
    layers.append(nn.Linear(in_dim, embed_dim))
    return nn.Sequential(*layers)


def _expectile_loss(diff, tau):
    import torch
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * diff.pow(2)


def main() -> None:
    args = _build_arg_parser().parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from guided_action_flow.critics.transformer_critic import TransformerCritic, TransformerCriticConfig
    from guided_action_flow.critics.validation import validate_critic
    from guided_action_flow.training.critic_dataset import (
        build_action_chunk_dataset, discover_episode_files, load_episode_files, split_indices_by_episode,
    )

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = discover_episode_files(args.data_dir)
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.pt files found under {args.data_dir}.")
    episodes = load_episode_files(episode_files)

    obs_key = "state"
    visual_key = "visual_features"
    dataset = build_action_chunk_dataset(
        episodes,
        action_horizon=args.action_horizon,
        stride=args.stride,
        gamma=args.gamma,
        obs_key=obs_key,
        obs_source=args.obs_source,
        visual_key=visual_key,
        task_feature_source=args.task_feature_source,
        task_feature_dim=args.task_feature_dim,
        task_feature_key=args.task_feature_key,
    )
    iql = _build_iql_transitions(
        episodes, dataset,
        action_horizon=args.action_horizon, gamma=args.gamma,
        obs_source=args.obs_source, obs_key=obs_key, visual_key=visual_key,
        reward_key=args.reward_key, done_key=args.done_key,
        reward_from_success=args.reward_from_success,
    )
    rsf = float((iql["reward_chunk"] > 0).float().mean().item())
    print(json.dumps({"reward_success_fraction": rsf,
                      "n_reward_pos": int((iql["reward_chunk"] > 0).sum().item()),
                      "n_samples": int(iql["reward_chunk"].numel())}))
    if rsf == 0.0:
        raise SystemExit(
            "reward_success_fraction=0: no positive reward in any transition. "
            "Check --reward-from-success / the success field -- IQL has nothing to learn."
        )

    obs_features = dataset["obs_features"]
    proprio = dataset["proprio"]
    action_chunks = dataset["action_chunks"]
    task_features = dataset.get("task_features")
    has_task = task_features is not None

    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = split_indices_by_episode(
        dataset["episode_indices"], val_fraction=args.val_fraction, generator=generator
    )

    device_name = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    device = torch.device(device_name)

    per_token_visual = obs_features.shape[-1] if args.visual_tokens == 0 else (obs_features.shape[-1] // args.visual_tokens)
    critic_config = TransformerCriticConfig(
        action_dim=action_chunks.shape[-1],
        action_horizon=action_chunks.shape[1],
        state_dim=proprio.shape[-1],
        visual_dim=per_token_visual,
        visual_tokens=args.visual_tokens,
        language_dim=0 if not has_task else task_features.shape[-1],
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        depth=args.depth,
        action_chunk_tokens=args.action_chunk_tokens,
        dropout=args.dropout,
        input_layernorm=args.input_layernorm,
    )
    q_critic = TransformerCritic(critic_config)
    q_critic.module.to(device)
    q_target = TransformerCritic(critic_config)
    q_target.module.to(device)
    q_target.load_state_dict(_clone_state_dict_to_cpu(q_critic.state_dict()))
    for p in q_target.parameters():
        p.requires_grad_(False)
    value_net = _make_value_net(critic_config, device)

    q_opt = torch.optim.AdamW(q_critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    v_opt = torch.optim.AdamW(value_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # fixed column order for the loader
    cols = [obs_features, proprio, action_chunks]
    if has_task:
        cols.append(task_features)
    cols += [
        iql["next_obs_features"], iql["next_proprio"],
        iql["reward_chunk"], iql["not_done"], iql["discount"],
    ]
    if has_task:
        cols.append(task_features)  # next-state task == same task (constant per episode)
    cols.append(dataset["targets"])  # success-to-go return, for the optional BC term
    train_ds = TensorDataset(*[c[train_indices] for c in cols])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    def q_forward(net, o, p, a, tk):
        return net(obs_features=o, action_chunk=a, proprio=p, task_features=tk)

    def v_forward(o, p, tk):
        return value_net(obs_features=o, proprio=p, task_features=tk)

    history = []
    for epoch in range(args.epochs):
        q_critic.module.train(); value_net.train()
        v_losses, q_losses = [], []
        for batch in train_loader:
            it = iter(batch)
            o = next(it).to(device); p = next(it).to(device); a = next(it).to(device)
            tk = next(it).to(device) if has_task else None
            no = next(it).to(device); np_ = next(it).to(device)
            r = next(it).to(device); nd = next(it).to(device); disc = next(it).to(device)
            ntk = next(it).to(device) if has_task else None
            target_b = next(it).to(device)  # success-to-go return (BC term)

            # --- V update: expectile regression toward target Q(s,a) ---
            with torch.no_grad():
                q_targ = q_forward(q_target, o, p, a, tk)
            v = v_forward(o, p, tk)
            v_loss = _expectile_loss(q_targ - v, args.expectile).mean()
            v_opt.zero_grad(set_to_none=True); v_loss.backward(); v_opt.step()

            # --- Q update: TD toward r + gamma^k * (1-done) * V(s') ---
            with torch.no_grad():
                v_next = v_forward(no, np_, ntk)
                q_backup = r + disc * nd * v_next
            q_pred = q_forward(q_critic, o, p, a, tk)
            q_loss = torch.nn.functional.mse_loss(q_pred, q_backup)
            if args.bc_weight > 0.0:
                # auxiliary supervised term: pull Q toward the success-to-go return
                # (same signal the Stage-B critic regressed to), on the SAME network.
                q_loss = q_loss + args.bc_weight * torch.nn.functional.mse_loss(q_pred, target_b)
            q_opt.zero_grad(set_to_none=True); q_loss.backward(); q_opt.step()

            # --- Polyak update of target Q ---
            with torch.no_grad():
                for tp_, sp_ in zip(q_target.parameters(), q_critic.parameters()):
                    tp_.mul_(1.0 - args.polyak).add_(sp_, alpha=args.polyak)

            v_losses.append(float(v_loss.detach().cpu())); q_losses.append(float(q_loss.detach().cpu()))

        metrics = {
            "epoch": epoch,
            "v_loss": sum(v_losses) / max(1, len(v_losses)),
            "q_loss": sum(q_losses) / max(1, len(q_losses)),
        }
        if val_indices.numel() > 0:
            q_critic.module.eval(); value_net.eval()
            with torch.no_grad():
                vi = val_indices
                no = iql["next_obs_features"][vi].to(device)
                np_ = iql["next_proprio"][vi].to(device)
                r = iql["reward_chunk"][vi].to(device); nd = iql["not_done"][vi].to(device)
                disc = iql["discount"][vi].to(device)
                ntk = task_features[vi].to(device) if has_task else None
                tk = task_features[vi].to(device) if has_task else None
                q_backup = r + disc * nd * v_forward(no, np_, ntk)
                q_pred = q_forward(
                    q_critic, obs_features[vi].to(device), proprio[vi].to(device),
                    action_chunks[vi].to(device), tk,
                )
                metrics["val_q_loss"] = float(torch.nn.functional.mse_loss(q_pred, q_backup).cpu())
        history.append(metrics)
        print(json.dumps(metrics))

    # ranking validation on the exact trained Q (same check as Stage B)
    q_critic.module.eval()
    ranking = validate_critic(q_critic, dataset, device=device_name)

    selected_state_dict = _clone_state_dict_to_cpu(q_critic.state_dict())
    checkpoint = {
        "critic_arch": "transformer",
        "critic_config": asdict(critic_config),
        "model_state_dict": selected_state_dict,
        "final_model_state_dict": selected_state_dict,
        "training_args": vars(args),
        "num_samples": int(action_chunks.shape[0]),
        "num_episodes": len(episodes),
        "num_task_ids": len({int(ep.get("task_id", -1)) for ep in episodes}),
        "task_feature_source": args.task_feature_source,
        "task_feature_key": args.task_feature_key,
        "task_feature_dim": 0 if not has_task else int(task_features.shape[-1]),
        "obs_source": args.obs_source,
        "visual_token_grid": int(args.visual_token_grid),
        "target_mean": float(dataset["targets"].mean().item()),
        "target_success_fraction": float((dataset["targets"] > 0).float().mean().item()),
        "ranking_validation": ranking,
        "history": history,
        "selected_epoch": history[-1]["epoch"] if history else None,
        "selected_metric": "final",
        "objective": "iql+bc" if args.bc_weight > 0.0 else "iql",
        "iql": {
            "expectile": args.expectile,
            "polyak": args.polyak,
            "gamma": args.gamma,
            "bc_weight": args.bc_weight,
            "value_state_dict": _clone_state_dict_to_cpu(value_net.state_dict()),
            "reward_success_fraction": float((iql["reward_chunk"] > 0).float().mean().item()),
        },
    }
    torch.save(checkpoint, output_dir / "critic.pt")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "objective": "iql+bc" if args.bc_weight > 0.0 else "iql",
                "num_samples": int(action_chunks.shape[0]),
                "num_episodes": len(episodes),
                "num_task_ids": checkpoint["num_task_ids"],
                "obs_source": args.obs_source,
                "expectile": args.expectile,
                "bc_weight": args.bc_weight,
                "reward_success_fraction": checkpoint["iql"]["reward_success_fraction"],
                "ranking_validation": ranking,
                "final": history[-1] if history else {},
            },
            f, indent=2,
        )
    print("ranking_validation:", json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
