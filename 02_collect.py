#!/usr/bin/env python3
"""
Phase 2: Rollout collection — 40 tasks × 50 episodes for Critic training.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GAF_DIR   = Path(__file__).resolve().parent
CKPT_DIR  = GAF_DIR / "checkpoints" / "smolvla_libero"
RUNS_DIR  = GAF_DIR / "runs" / "collect"
LIBERO_DATA = Path.home() / "miniconda3/envs/gaf-exp/lib/python3.12/site-packages/libero/libero"
CFG_DIR   = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON    = sys.executable

SUITES     = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS    = 10
N_EPISODES = 50
SEED       = 2000


def _gaf_env() -> dict:
    e = os.environ.copy()
    e["MUJOCO_GL"]          = "egl"
    e["WANDB_MODE"]         = "disabled"
    e["LIBERO_CONFIG_PATH"] = str(CFG_DIR)
    tp = GAF_DIR / "third_party" / "lerobot" / "src"
    e["PYTHONPATH"]         = f"{GAF_DIR / 'src'}:{tp}:" + e.get("PYTHONPATH", "")
    return e


def _ensure_config():
    cfg_path = CFG_DIR / "config.yaml"
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = (
        f"assets: {LIBERO_DATA}/assets\n"
        f"bddl_files: {LIBERO_DATA}/bddl_files\n"
        f"benchmark_root: {LIBERO_DATA}\n"
        f"datasets: {LIBERO_DATA}/datasets\n"
        f"init_states: {LIBERO_DATA}/init_files\n"
    )
    cfg_path.write_text(cfg)


def collect(suite: str, task_id: int) -> dict | None:
    out = RUNS_DIR / suite / f"task{task_id:02d}"
    summary_path = out / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        n_ep = data.get("aggregated", {}).get("n_episodes", 0)
        pct  = data.get("aggregated", {}).get("pc_success", 0.0)
        print(f"  [SKIP] {suite}/task{task_id:02d} ({n_ep} eps, {pct:.1f}% success)")
        return data

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(GAF_DIR / "scripts" / "collect_rollouts.py"),
        "--policy-path", str(CKPT_DIR),
        "--output-dir",  str(out),
        "--env-type",    "libero",
        "--task",        suite,
        "--task-ids",    f"[{task_id}]",
        "--n-episodes",  str(N_EPISODES),
        "--seed",        str(SEED),
        "--device",      "cuda",
    ]
    t0 = time.time()
    r  = subprocess.run(cmd, env=_gaf_env(), text=True)
    elapsed = time.time() - t0

    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        pct   = data.get("aggregated", {}).get("pc_success", 0.0)
        n_suc = sum(1 for ep in data["per_episode"] if ep["success"])
        print(f"  ✓ {suite}/task{task_id:02d} → {n_suc}/{N_EPISODES} success ({pct:.1f}%)  ({elapsed:.0f}s)")
        return data
    else:
        print(f"  ✗ {suite}/task{task_id:02d} FAILED (rc={r.returncode}, {elapsed:.0f}s)")
        return None


def main():
    print("=" * 60)
    print("  Phase 2: Rollout Collection — 40 tasks × 50 episodes")
    print("=" * 60)

    _ensure_config()
    total_start = time.time()
    stats: dict[str, list] = {}

    for suite in SUITES:
        print(f"\n── {suite} ──")
        stats[suite] = []
        for task_id in range(N_TASKS):
            data = collect(suite, task_id)
            if data:
                pct   = data.get("aggregated", {}).get("pc_success", 0.0)
                n_suc = sum(1 for ep in data["per_episode"] if ep["success"])
                stats[suite].append({"task_id": task_id, "n_success": n_suc,
                                     "n_total": N_EPISODES, "pc_success": pct})

    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    for suite, task_stats in stats.items():
        print(f"\n  {suite}:")
        for ts in task_stats:
            bar = "█" * int(ts["pc_success"] / 10) + "░" * (10 - int(ts["pc_success"] / 10))
            print(f"    task{ts['task_id']:02d}: {bar} {ts['n_success']:2d}/{N_EPISODES} ({ts['pc_success']:5.1f}%)")

    print(f"\n  Total time: {(time.time() - total_start) / 60:.1f} min")

    summary_path = RUNS_DIR / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"n_episodes_per_task": N_EPISODES, "seed": SEED, "stats": stats}, f, indent=2)
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
