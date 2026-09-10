#!/usr/bin/env python3
"""
Phase 1: SmolVLA Baseline evaluation — 40 tasks × 20 episodes.
Uses GAF official eval_policy.py.

Output: runs/baseline/{suite}/task{id:02d}/eval_info.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GAF_DIR  = Path(__file__).resolve().parent
CKPT_DIR = GAF_DIR / "checkpoints" / "smolvla_libero"
RUNS_DIR = GAF_DIR / "runs" / "baseline"
# Corrected: data lives inside libero/libero/ sub-package
LIBERO_DATA = Path.home() / "miniconda3/envs/gaf-exp/lib/python3.12/site-packages/libero/libero"
CFG_DIR  = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON   = sys.executable

SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]
N_TASKS    = 10
N_EPISODES = 50
SEED       = 3000


def _gaf_env() -> dict:
    e = os.environ.copy()
    e["MUJOCO_GL"]          = "egl"
    e["WANDB_MODE"]         = "disabled"
    e["LIBERO_CONFIG_PATH"] = str(CFG_DIR)
    tp = GAF_DIR / "third_party" / "lerobot" / "src"
    e["PYTHONPATH"]         = f"{GAF_DIR / 'src'}:{tp}:" + e.get("PYTHONPATH", "")
    return e


def _ensure_config():
    """Ensure LIBERO config points to correct path."""
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


def run_eval(suite: str, task_id: int) -> float | None:
    out = RUNS_DIR / suite / f"task{task_id:02d}"
    info_path = out / "eval_info.json"
    if info_path.exists():
        with open(info_path) as f:
            data = json.load(f)
        pct = data.get("pc_success", 0.0)
        print(f"  [SKIP] {suite}/task{task_id:02d} already done → {pct:.1f}%")
        return pct

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(GAF_DIR / "scripts" / "eval_policy.py"),
        "--policy-path", str(CKPT_DIR),
        "--output-dir",  str(out),
        "--env-type",    "libero",
        "--task",        suite,
        "--task-ids",    f"[{task_id}]",
        "--n-episodes",  str(N_EPISODES),
        "--seed",        str(SEED),
        "--device",      "cuda",
        "--max-videos",  "3",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, env=_gaf_env(), text=True)
    elapsed = time.time() - t0

    if info_path.exists():
        with open(info_path) as f:
            data = json.load(f)
        pct = data.get("pc_success", 0.0)
        print(f"  ✓ {suite}/task{task_id:02d} → {pct:.1f}%  ({elapsed:.0f}s)")
        return pct
    else:
        print(f"  ✗ {suite}/task{task_id:02d} FAILED (rc={r.returncode}, {elapsed:.0f}s)")
        return None


def main():
    print("=" * 60)
    print("  Phase 1: SmolVLA Baseline — 40 tasks × 20 episodes")
    print("=" * 60)

    _ensure_config()

    results: dict[str, list[float | None]] = {}
    total_start = time.time()

    for suite in SUITES:
        print(f"\n── {suite} ──")
        results[suite] = []
        for task_id in range(N_TASKS):
            pct = run_eval(suite, task_id)
            results[suite].append(pct)

    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    all_vals: list[float] = []
    for suite, vals in results.items():
        valid = [v for v in vals if v is not None]
        avg   = sum(valid) / len(valid) if valid else 0.0
        all_vals.extend(valid)
        row = ", ".join(f"{v:.0f}" if v is not None else "ERR" for v in vals)
        print(f"  {suite:20s}: avg={avg:5.1f}%  [{row}]")

    overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
    elapsed_total = (time.time() - total_start) / 60
    print(f"\n  Overall average : {overall:.1f}%")
    print(f"  Tasks evaluated : {len(all_vals)}/{N_TASKS * len(SUITES)}")
    print(f"  Total time      : {elapsed_total:.1f} min")

    summary_path = RUNS_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "phase": "baseline",
            "n_episodes_per_task": N_EPISODES,
            "seed": SEED,
            "results": results,
            "overall_avg": overall,
        }, f, indent=2)
    print(f"\n  Summary → {summary_path}")


if __name__ == "__main__":
    main()
