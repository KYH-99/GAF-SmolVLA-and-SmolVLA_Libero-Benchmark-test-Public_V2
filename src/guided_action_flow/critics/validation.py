from __future__ import annotations

"""Critic ranking validation (Haichen's experiment_plan Phase 3).

The supervisor's point: to confirm the critic *works*, don't look at whether QGF
raises success rate — look at whether the critic **ranks action chunks
correctly**. These are the Phase-3 checks from ``docs/experiment_plan.md``:

- correlation between predicted ``Q`` and the rollout success-to-go target
- value separation between successful and failed rollout action chunks
- ensemble disagreement (when multiple critics are given)

All functions take an already-built critic-input dataset (the dict from
``training/critic_dataset.build_action_chunk_dataset``) so they run on the exact
inputs the critic was trained on. A critic is "usable" when Q correlates with
the target and successful chunks score clearly above failed ones — regardless of
whether test-time QGF happens to help on any single task.
"""


def _critic_values(critic, dataset, device="cpu", batch_size=1024):
    import torch

    obs = dataset["obs_features"]
    actions = dataset["action_chunks"]
    proprio = dataset.get("proprio")
    task = dataset.get("task_features")
    n = actions.shape[0]

    values = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            kwargs = {
                "obs_features": obs[start:end].to(device) if obs is not None else None,
                "action_chunk": actions[start:end].to(device),
                "proprio": proprio[start:end].to(device) if proprio is not None else None,
            }
            if task is not None:
                kwargs["task_features"] = task[start:end].to(device)
            v = critic(**kwargs)
            values.append(v.detach().cpu().reshape(-1))
    return torch.cat(values)


def q_return_correlation(critic, dataset, device="cpu"):
    """Pearson correlation between predicted Q and the success-to-go target."""
    import torch

    values = _critic_values(critic, dataset, device=device)
    targets = dataset["targets"].reshape(-1).float()
    vx = values - values.mean()
    tx = targets - targets.mean()
    denom = (vx.norm() * tx.norm()).clamp_min(1.0e-8)
    return float((vx @ tx) / denom)


def success_fail_separation(critic, dataset, device="cpu", success_threshold=0.5):
    """Mean Q on successful vs failed action chunks, and the gap.

    A sample is "successful" when its success-to-go target exceeds the threshold.
    Returns a dict with mean Q for each group and the separation (success - fail).
    """
    import torch

    values = _critic_values(critic, dataset, device=device)
    targets = dataset["targets"].reshape(-1).float()
    succ_mask = targets > success_threshold
    fail_mask = ~succ_mask

    def _mean(mask):
        return float(values[mask].mean()) if int(mask.sum()) > 0 else float("nan")

    q_succ = _mean(succ_mask)
    q_fail = _mean(fail_mask)
    return {
        "q_success_mean": q_succ,
        "q_fail_mean": q_fail,
        "separation": q_succ - q_fail,
        "n_success": int(succ_mask.sum()),
        "n_fail": int(fail_mask.sum()),
    }


def ensemble_disagreement(critics, dataset, device="cpu"):
    """Mean per-sample std of Q across an ensemble of critics (uncertainty proxy)."""
    import torch

    if len(critics) < 2:
        return float("nan")
    stacked = torch.stack(
        [_critic_values(c, dataset, device=device) for c in critics], dim=0
    )
    return float(stacked.std(dim=0, unbiased=False).mean())


def validate_critic(critic, dataset, device="cpu", ensemble=None):
    """Run all Phase-3 ranking checks and return a summary dict."""
    summary = {
        "q_return_correlation": q_return_correlation(critic, dataset, device=device),
        **success_fail_separation(critic, dataset, device=device),
    }
    if ensemble is not None and len(ensemble) > 1:
        summary["ensemble_disagreement"] = ensemble_disagreement(
            ensemble, dataset, device=device
        )
    return summary
