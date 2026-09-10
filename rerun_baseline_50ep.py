#!/usr/bin/env python3
"""
Re-run baseline evaluation for tasks that only have 20 episodes.
Deletes the cached eval_info.json and re-runs to 50 episodes.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

GAF_DIR   = Path("/home/inc-kyh/experiments/gaf-comparison")
CKPT_DIR  = GAF_DIR / "checkpoints" / "smolvla_libero"
RUNS_DIR  = GAF_DIR / "runs"
LIBERO_DATA = Path.home() / "miniconda3/envs/gaf-exp/lib/python3.12/site-packages/libero/libero"
CFG_DIR   = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON    = sys.executable
N_EPISODES = 50
SEED = 3000

# Tasks that only have 20 episodes and need rerun
TASKS_TO_RERUN = [
    ("libero_spatial", 0), ("libero_spatial", 2), ("libero_spatial", 3),
    ("libero_spatial", 5), ("libero_spatial", 7), ("libero_spatial", 9),
    ("libero_object", 0), ("libero_object", 2), ("libero_object", 4),
    ("libero_object", 6), ("libero_object", 7), ("libero_object", 9),
    ("libero_goal", 0), ("libero_goal", 2), ("libero_goal", 4),
    ("libero_goal", 5), ("libero_goal", 7), ("libero_goal", 8),
    ("libero_10", 1), ("libero_10", 2), ("libero_10", 3),
    ("libero_10", 5), ("libero_10", 7), ("libero_10", 8),
]

def _gaf_env():
    e = os.environ.copy()
    e["MUJOCO_GL"]          = "egl"
    e["WANDB_MODE"]         = "disabled"
    e["LIBERO_CONFIG_PATH"] = str(CFG_DIR)
    tp = GAF_DIR / "third_party" / "lerobot" / "src"
    e["PYTHONPATH"]         = f"{GAF_DIR / 'src'}:{tp}:" + e.get("PYTHONPATH", "")
    return e

def run_eval(suite: str, task_id: int) -> float:
    out_dir = RUNS_DIR / "baseline" / suite / f"task{task_id:02d}"
    info_path = out_dir / "eval_info.json"

    # Delete old 20-episode cache
    if info_path.exists():
        d = json.loads(info_path.read_text())
        n = d["overall"]["n_episodes"]
        if n >= 50:
            succ = d["overall"]["pc_success"]
            print(f"  [SKIP] {suite}/task{task_id:02d}: already {n} eps → {succ:.1f}%")
            return succ
        else:
            print(f"  [DEL] Removing cached {n}-ep result for {suite}/task{task_id:02d}")
            info_path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(GAF_DIR / "scripts" / "eval_policy.py"),
        "--policy-path", str(CKPT_DIR),
        "--output-dir",  str(out_dir),
        "--env-type",    "libero",
        "--task",        suite,
        "--task-ids",    f"[{task_id}]",
        "--n-episodes",  str(N_EPISODES),
        "--seed",        str(SEED),
        "--device",      "cuda",
        "--max-videos",  "3",
    ]
    t0 = time.time()
    print(f"  [RUN] {suite}/task{task_id:02d}  ({N_EPISODES} eps)...")
    sys.stdout.flush()
    result = subprocess.run(cmd, env=_gaf_env(), capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {suite}/task{task_id:02d} (exit={result.returncode}, {elapsed:.0f}s)")
        return 0.0

    if info_path.exists():
        d = json.loads(info_path.read_text())
        pct = d["overall"].get("pc_success", 0.0)
        print(f"  [OK] {suite}/task{task_id:02d}: {pct:.1f}%  ({elapsed:.0f}s)")
        return pct
    else:
        print(f"  [ERR] No eval_info.json after run for {suite}/task{task_id:02d}")
        return 0.0

print("=" * 60)
print("  Re-running 24 baseline tasks to 50 episodes each")
print("=" * 60)

results = {}
for suite, task_id in TASKS_TO_RERUN:
    if suite not in results:
        results[suite] = {}
    succ = run_eval(suite, task_id)
    results[suite][task_id] = succ

print("\n" + "=" * 60)
print("  RERUN COMPLETE - Summary")
print("=" * 60)

# Re-read all results (including ones not rerun)
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
all_succs = []
suite_avgs = {}
for suite in suites:
    suite_succs = []
    for task_id in range(10):
        info_path = RUNS_DIR / "baseline" / suite / f"task{task_id:02d}" / "eval_info.json"
        if info_path.exists():
            d = json.loads(info_path.read_text())
            pct = d["overall"]["pc_success"]
            n = d["overall"]["n_episodes"]
            suite_succs.append(pct)
            all_succs.append(pct)
            print(f"  {suite}/task{task_id:02d}: {pct:.1f}% ({n} eps)")
    if suite_succs:
        avg = sum(suite_succs) / len(suite_succs)
        suite_avgs[suite] = avg
        print(f"  => {suite} avg: {avg:.1f}%\n")

overall = sum(all_succs) / len(all_succs) if all_succs else 0.0
print(f"\n  *** OVERALL Baseline avg: {overall:.1f}%")

# Update summary.json
summary = {
    "phase": "baseline",
    "n_episodes_per_task": 50,
    "seed": SEED,
    "results": {
        suite: [
            json.loads((RUNS_DIR / "baseline" / suite / f"task{tid:02d}" / "eval_info.json").read_text())["overall"]["pc_success"] / 100.0
            for tid in range(10)
            if (RUNS_DIR / "baseline" / suite / f"task{tid:02d}" / "eval_info.json").exists()
        ]
        for suite in suites
    },
    "overall_avg": overall / 100.0,
    "suite_avgs": {k: v / 100.0 for k, v in suite_avgs.items()}
}
(RUNS_DIR / "baseline" / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n  summary.json updated!")
