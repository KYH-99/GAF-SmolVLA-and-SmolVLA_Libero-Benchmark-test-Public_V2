#!/usr/bin/env python3
"""
Phase 3: Critic training — two variants per task.

  (A) GAF-Full:    train on ALL 50 collected episodes  (--failure-weight 1.0)
  (B) GAF-Success: train on SUCCESS episodes only       (--failure-weight 0.0)

Uses GAF official train_critic.py.
Output:
  runs/critics/full/{suite}/task{id:02d}/critic.pt
  runs/critics/success_only/{suite}/task{id:02d}/critic.pt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GAF_DIR    = Path(__file__).resolve().parent
COLLECT_DIR = GAF_DIR / "runs" / "collect"
CRITICS_DIR = GAF_DIR / "runs" / "critics"
CFG_DIR    = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON     = sys.executable

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10

# Critic hyperparams (following GAF repo design doc)
CRITIC_EPOCHS   = 30
CRITIC_LR       = 1e-3
ACTION_HORIZON  = 50
CRITIC_ARCH     = "multimodal"    # GAF default
OBS_SOURCE      = "state"         # safe default (no visual feature re-collection needed)
TASK_FEAT_SRC   = "tokens"
TASK_FEAT_DIM   = 128


def _gaf_env() -> dict:
    e = os.environ.copy()
    e["MUJOCO_GL"]          = "egl"
    e["WANDB_MODE"]         = "disabled"
    e["LIBERO_CONFIG_PATH"] = str(CFG_DIR)
    tp = GAF_DIR / "third_party" / "lerobot" / "src"
    e["PYTHONPATH"]         = f"{GAF_DIR / 'src'}:{tp}:" + e.get("PYTHONPATH", "")
    return e


def train_critic(
    suite: str,
    task_id: int,
    variant: str,           # "full" or "success_only"
    failure_weight: float,  # 1.0 = full, 0.0 = success only
) -> dict | None:
    data_dir = COLLECT_DIR / suite / f"task{task_id:02d}"
    out_dir  = CRITICS_DIR / variant / suite / f"task{task_id:02d}"
    ckpt     = out_dir / "critic.pt"

    if ckpt.exists():
        print(f"  [SKIP] {variant}/{suite}/task{task_id:02d} — critic.pt exists")
        meta_path = out_dir / "critic_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {"skipped": True}

    if not data_dir.exists() or not list(data_dir.rglob("episode_*.pt")):
        print(f"  [WARN] No collected data for {suite}/task{task_id:02d}, skipping critic training")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(GAF_DIR / "scripts" / "train_critic.py"),
        "--data-dir",           str(data_dir),
        "--output-dir",         str(out_dir),
        "--epochs",             str(CRITIC_EPOCHS),
        "--lr",                 str(CRITIC_LR),
        "--action-horizon",     str(ACTION_HORIZON),
        "--critic-arch",        CRITIC_ARCH,
        "--obs-source",         OBS_SOURCE,
        "--task-feature-source", TASK_FEAT_SRC,
        "--task-feature-dim",   str(TASK_FEAT_DIM),
        "--failure-weight",     str(failure_weight),
        "--device",             "cuda",
    ]
    t0 = time.time()
    r  = subprocess.run(cmd, env=_gaf_env(), text=True)
    elapsed = time.time() - t0

    if ckpt.exists():
        print(f"  ✓ critic trained [{variant}] {suite}/task{task_id:02d}  ({elapsed:.0f}s)")
        # Try to read training log
        log_path = out_dir / "train_log.json"
        if log_path.exists():
            with open(log_path) as f:
                return json.load(f)
        return {"elapsed_s": elapsed}
    else:
        print(f"  ✗ FAILED [{variant}] {suite}/task{task_id:02d} rc={r.returncode} ({elapsed:.0f}s)")
        return None


def main():
    print("=" * 60)
    print("  Phase 3: Critic Training — Full & Success-Only")
    print("  failure-weight: 1.0 (Full) | 0.0 (Success-Only)")
    print("=" * 60)

    total_start = time.time()
    results = {"full": {}, "success_only": {}}

    for suite in SUITES:
        print(f"\n══ {suite} ══")
        results["full"][suite]         = []
        results["success_only"][suite] = []

        for task_id in range(N_TASKS):
            print(f"\n  [Task {task_id:02d}]")

            # 3-A: Full
            r_full = train_critic(suite, task_id, "full", failure_weight=1.0)
            results["full"][suite].append(r_full)

            # 3-B: Success-only (failure_weight=0.0 suppresses failure samples)
            r_suc = train_critic(suite, task_id, "success_only", failure_weight=0.0)
            results["success_only"][suite].append(r_suc)

    print("\n" + "=" * 60)
    print(f"  Phase 3 complete.  Total time: {(time.time() - total_start) / 60:.1f} min")

    # count trained critics
    full_ckpts = list(CRITICS_DIR.rglob("full/**/critic.pt"))
    suc_ckpts  = list(CRITICS_DIR.rglob("success_only/**/critic.pt"))
    print(f"  Full critics trained:    {len(full_ckpts)} / {N_TASKS * len(SUITES)}")
    print(f"  Success critics trained: {len(suc_ckpts)} / {N_TASKS * len(SUITES)}")

    summary_path = CRITICS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_tasks": N_TASKS * len(SUITES),
            "full_critics": len(full_ckpts),
            "success_critics": len(suc_ckpts),
            "hyperparams": {
                "epochs": CRITIC_EPOCHS, "lr": CRITIC_LR,
                "action_horizon": ACTION_HORIZON,
                "critic_arch": CRITIC_ARCH, "obs_source": OBS_SOURCE,
                "task_feature_source": TASK_FEAT_SRC,
            },
        }, f, indent=2)
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
