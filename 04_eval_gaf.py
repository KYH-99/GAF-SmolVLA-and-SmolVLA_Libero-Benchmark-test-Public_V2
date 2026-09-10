#!/usr/bin/env python3
"""
Phase 4-5: Evaluate GAF (Full Critic) and GAF-Success (Success-Only Critic).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

GAF_DIR     = Path(__file__).resolve().parent
CKPT_DIR    = GAF_DIR / "checkpoints" / "smolvla_libero"
CRITICS_DIR = GAF_DIR / "runs" / "critics"
RUNS_DIR    = GAF_DIR / "runs"
LIBERO_DATA = Path.home() / "miniconda3/envs/gaf-exp/lib/python3.12/site-packages/libero/libero"
CFG_DIR     = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON      = sys.executable

SUITES  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10
N_EPISODES = 50
SEED    = 3000

BETA_SWEEP   = [1.0, 2.0, 3.0, 5.0]
GRAD_CLIP    = 1.0
DEFAULT_BETA = 3.0


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


def run_eval(suite: str, task_id: int, out_dir: Path,
             critic_path: Path | None = None,
             beta: float | None = None) -> float | None:
    info_path = out_dir / "eval_info.json"
    if info_path.exists():
        with open(info_path) as f:
            d = json.load(f)
        pct = d.get("pc_success", 0.0)
        print(f"  [SKIP] → {pct:.1f}%")
        return pct

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
    if critic_path and critic_path.exists():
        cmd += ["--critic-path",        str(critic_path)]
        cmd += ["--qgf-beta",           str(beta or DEFAULT_BETA)]
        cmd += ["--qgf-grad-clip-norm", str(GRAD_CLIP)]

    t0 = time.time()
    r  = subprocess.run(cmd, env=_gaf_env(), text=True)
    elapsed = time.time() - t0

    if info_path.exists():
        with open(info_path) as f:
            d = json.load(f)
        pct = d.get("pc_success", 0.0)
        print(f"  ✓ → {pct:.1f}%  ({elapsed:.0f}s)")
        return pct
    print(f"  ✗ FAILED rc={r.returncode} ({elapsed:.0f}s)")
    return None


def beta_sweep(variant: str = "full") -> float:
    print("\n" + "=" * 60)
    print(f"  Beta Sweep [{variant}] — libero_spatial task 3")
    print("=" * 60)
    suite, task_id = "libero_spatial", 3
    critic_path = CRITICS_DIR / variant / suite / f"task{task_id:02d}" / "critic.pt"
    if not critic_path.exists():
        print(f"  [WARN] No critic at {critic_path}. Using default beta={DEFAULT_BETA}")
        return DEFAULT_BETA

    sweep_results: dict[float, float | None] = {}
    for beta in BETA_SWEEP:
        out = RUNS_DIR / "eval_gaf" / "beta_sweep" / variant / f"spatial_task03_beta{beta}"
        print(f"\n  beta={beta}  ", end="", flush=True)
        pct = run_eval(suite, task_id, out, critic_path=critic_path, beta=beta)
        sweep_results[beta] = pct

    valid = {b: p for b, p in sweep_results.items() if p is not None}
    if not valid:
        return DEFAULT_BETA
    best_beta = max(valid, key=lambda b: valid[b])
    print(f"\n  Best beta={best_beta} → {valid[best_beta]:.1f}%")

    sweep_dir = RUNS_DIR / "eval_gaf" / "beta_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / f"sweep_{variant}.json", "w") as f:
        json.dump({"variant": variant, "sweep": {str(k): v for k, v in sweep_results.items()},
                   "best_beta": best_beta}, f, indent=2)
    return best_beta


def eval_all(variant: str, beta: float, run_subdir: str) -> dict:
    print("\n" + "=" * 60)
    print(f"  Evaluating [{variant}] beta={beta} — 40 tasks × {N_EPISODES} eps")
    print("=" * 60)

    results: dict[str, list[float | None]] = {}
    total_start = time.time()

    for suite in SUITES:
        print(f"\n── {suite} ──")
        results[suite] = []
        for task_id in range(N_TASKS):
            critic_path = CRITICS_DIR / variant / suite / f"task{task_id:02d}" / "critic.pt"
            out = RUNS_DIR / run_subdir / suite / f"task{task_id:02d}"
            tag = f"{suite}/task{task_id:02d}"
            if not critic_path.exists():
                print(f"  [WARN] {tag}: no critic, skipping")
                results[suite].append(None)
                continue
            print(f"  {tag}  ", end="", flush=True)
            pct = run_eval(suite, task_id, out, critic_path=critic_path, beta=beta)
            results[suite].append(pct)

    print("\n" + "=" * 60)
    all_vals = [v for vals in results.values() for v in vals if v is not None]
    overall  = sum(all_vals) / len(all_vals) if all_vals else 0.0
    for suite, vals in results.items():
        valid = [v for v in vals if v is not None]
        avg   = sum(valid) / len(valid) if valid else 0.0
        print(f"  {suite:20s}: {avg:5.1f}%")
    print(f"\n  Overall: {overall:.1f}%  | Time: {(time.time()-total_start)/60:.1f} min")

    summary = {"variant": variant, "beta": beta, "results": results, "overall_avg": overall}
    out_path = RUNS_DIR / run_subdir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    _ensure_config()

    print("\n" + "█" * 60)
    print("  PHASE 4: GAF (Full Critic)")
    print("█" * 60)
    best_beta_full = beta_sweep("full")
    gaf_results    = eval_all("full", best_beta_full, "eval_gaf")

    print("\n" + "█" * 60)
    print("  PHASE 5: GAF-Success (Success-Only Critic)")
    print("█" * 60)
    best_beta_suc  = beta_sweep("success_only")
    gaf_suc_results = eval_all("success_only", best_beta_suc, "eval_gaf_success")

    print("\n" + "=" * 60)
    print("  PHASES 4-5 COMPLETE")
    print(f"  GAF-Full overall:    {gaf_results['overall_avg']:.1f}%")
    print(f"  GAF-Success overall: {gaf_suc_results['overall_avg']:.1f}%")


if __name__ == "__main__":
    main()
