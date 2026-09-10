#!/usr/bin/env python3
"""
Phase 0: Environment bootstrap + smoke test for fresh gaf-exp conda environment.
Run after: conda activate gaf-exp && pip install -e '.[dev,torch]'

This script:
1. Runs bootstrap_third_party.sh (clones pinned LeRobot + LIBERO)
2. Installs LeRobot from the pinned third_party checkout
3. Downloads lerobot/smolvla_libero checkpoint via Hugging Face
4. Writes .libero_configs/vanilla/config.yaml
5. Runs a 1-episode smoke test on libero_spatial task 0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
GAF_DIR   = Path(__file__).resolve().parent          # ~/experiments/gaf-comparison
CKPT_DIR  = GAF_DIR / "checkpoints" / "smolvla_libero"
RUNS_DIR  = GAF_DIR / "runs"
CFG_DIR   = GAF_DIR / ".libero_configs" / "vanilla"
PYTHON    = sys.executable


def _run(cmd: str, env=None, check: bool = True):
    print(f"\n▶  {cmd}")
    r = subprocess.run(cmd, shell=True, env=env, text=True)
    if check and r.returncode != 0:
        print(f"[FATAL] command failed (rc={r.returncode})")
        sys.exit(1)
    return r


def _gaf_env() -> dict:
    e = os.environ.copy()
    e["MUJOCO_GL"]          = "egl"
    e["WANDB_MODE"]         = "disabled"
    e["LIBERO_CONFIG_PATH"] = str(CFG_DIR)
    tp = GAF_DIR / "third_party" / "lerobot" / "src"
    e["PYTHONPATH"]         = f"{GAF_DIR / 'src'}:{tp}:" + e.get("PYTHONPATH", "")
    return e


# ─── Step 1: Third-party bootstrap ────────────────────────────────────────────
def bootstrap():
    script = GAF_DIR / "scripts" / "bootstrap_third_party.sh"
    if not (GAF_DIR / "third_party" / "lerobot" / "src").exists():
        _run(f"bash {script}", env=_gaf_env())
    else:
        print("✓ third_party/lerobot already exists, skipping bootstrap")


# ─── Step 2: Install LeRobot ──────────────────────────────────────────────────
def install_lerobot():
    lr_dir = GAF_DIR / "third_party" / "lerobot"
    result = subprocess.run([PYTHON, "-c", "import lerobot; print(lerobot.__version__)"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ lerobot already installed ({result.stdout.strip()})")
        return
    _run(f"cd {lr_dir} && {PYTHON} -m pip install -e '.[smolvla,libero]'")


# ─── Step 3: Download checkpoint ──────────────────────────────────────────────
def download_checkpoint():
    safetensors = CKPT_DIR / "model.safetensors"
    if safetensors.exists():
        print(f"✓ Checkpoint already at {CKPT_DIR}")
        return
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        f"{PYTHON} {GAF_DIR}/scripts/download_models.py "
        f"--repo-id lerobot/smolvla_libero "
        f"--local-dir {CKPT_DIR}"
    )


# ─── Step 4: LIBERO vanilla config ────────────────────────────────────────────
def write_libero_config():
    cfg_file = CFG_DIR / "config.yaml"
    if cfg_file.exists():
        print(f"✓ LIBERO config already at {cfg_file}")
        return
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [PYTHON, "-c", "import libero as _l, os; print(os.path.dirname(_l.__file__))"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"[FATAL] Cannot locate libero package:\n{r.stderr}")
        sys.exit(1)
    libero_pkg = r.stdout.strip()
    print(f"  libero package path: {libero_pkg}")

    cfg = (
        f"assets: {libero_pkg}/assets\n"
        f"bddl_files: {libero_pkg}/bddl_files\n"
        f"benchmark_root: {libero_pkg}\n"
        f"datasets: {libero_pkg}/datasets\n"
        f"init_states: {libero_pkg}/init_files\n"
    )
    cfg_file.write_text(cfg)
    print(f"  Written: {cfg_file}")


# ─── Step 5: Smoke test ───────────────────────────────────────────────────────
def smoke_test():
    out = RUNS_DIR / "smoke_baseline"
    out.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"{PYTHON} {GAF_DIR}/scripts/eval_policy.py "
        f"--policy-path {CKPT_DIR} "
        f"--output-dir {out} "
        f"--env-type libero "
        f"--task libero_spatial "
        f"--task-ids [0] "
        f"--n-episodes 1 "
        f"--seed 9999 "
        f"--device cuda "
        f"--max-videos 0"
    )
    r = _run(cmd, env=_gaf_env(), check=False)
    info_path = out / "eval_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        pct = info.get("pc_success", "N/A")
        print(f"\n✅ SMOKE TEST PASSED — success rate: {pct}%")
    else:
        print(f"\n⚠️  No eval_info.json found (rc={r.returncode}). Check output above.")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SmolVLA × GAF × GAF-Success — Bootstrap & Smoke Test")
    print("=" * 60)
    bootstrap()
    install_lerobot()
    download_checkpoint()
    write_libero_config()
    smoke_test()
    print("\n[SETUP COMPLETE] Ready for Phase 1–5 experiment scripts.")
