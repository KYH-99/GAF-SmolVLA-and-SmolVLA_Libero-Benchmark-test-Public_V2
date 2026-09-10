#!/usr/bin/env python3
"""
run_gaf_failure_inverted.py
Method B: 실패 에피소드로 학습 + target 반전 (failure=1) + 추론 시 기울기 반전 (β 음수)

- 크리틱: 실패 패턴 → Q 높게 학습 (target=1)
- 추론: v_guided = v_θ + ∇Q/|β|  (기울기 반전 → Q 높은=실패 방향 회피)
- runs/critics_failure_inv/  에 크리틱 저장
- runs/eval_gaf_failure_inv/ 에 평가 결과 저장
"""

import sys, os, json, subprocess, shutil
from pathlib import Path

BASE           = Path("/home/inc-kyh/experiments/gaf-comparison")
COLLECT_DIR    = BASE / "runs" / "collect"
CRITIC_DIR     = BASE / "runs" / "critics_failure_inv"
EVAL_DIR       = BASE / "runs" / "eval_gaf_failure_inv"
TRAIN_SCRIPT   = BASE / "scripts" / "train_critic.py"
EVAL_SCRIPT    = BASE / "scripts" / "eval_policy.py"
LOG_DIR        = BASE / "logs"
POLICY_PATH    = "lerobot/smolvla_libero"

SUITES  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10

import torch

def episode_is_success(ep):
    s = ep.get("success")
    if s is None:
        return False
    t = torch.as_tensor(s)
    return bool(t[-1].item()) if t.numel() > 0 else False

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Method B: GAF-Failure-Inverted 파이프라인")
print("  실패 에피소드 target=1 반전 학습 + 추론 시 기울기 반전")
print("=" * 65)

sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "third_party" / "lerobot" / "src"))

# ── STEP 1: 실패 에피소드 수 파악 ─────────────────────────────────────────────
print("\n[STEP 1] 실패 에피소드 수 파악")
CRITIC_DIR.mkdir(parents=True, exist_ok=True)

fail_info = {}
from guided_action_flow.training.critic_dataset import (
    discover_episode_files, load_episode_files,
)

for suite in SUITES:
    fail_info[suite] = {}
    for j in range(N_TASKS):
        ep_dir = COLLECT_DIR / suite / f"task{j:02d}" / "episodes"
        if not ep_dir.exists():
            fail_info[suite][j] = 0
            continue
        files = discover_episode_files(str(ep_dir))
        eps   = load_episode_files(files)
        n_fail = sum(1 for ep in eps if not episode_is_success(ep))
        fail_info[suite][j] = n_fail
        print(f"  {suite}/task{j:02d}: 실패={n_fail}개")

# ── STEP 2: 크리틱 학습 (failure_only + invert_target) ───────────────────────
print("\n[STEP 2] GAF-Failure-Inverted Q-Critic 학습")

for suite in SUITES:
    for j in range(N_TASKS):
        ckpt_dir  = CRITIC_DIR / suite / f"task{j:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        done_flag = ckpt_dir / "done.flag"

        if done_flag.exists():
            print(f"  [SKIP] {suite}/task{j:02d}: 이미 완료")
            continue

        n_fail = fail_info[suite].get(j, 0)
        if n_fail == 0:
            print(f"  [SKIP] {suite}/task{j:02d}: 실패 에피소드 0개")
            done_flag.touch()
            continue

        print(f"  [TRAIN] {suite}/task{j:02d}: 실패 {n_fail}개, target 반전 학습...", flush=True)

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-dir",          str(COLLECT_DIR / suite / f"task{j:02d}"),
            "--output-dir",        str(ckpt_dir),
            "--obs-source",        "state",
            "--task-feature-source", "tokens",
            "--task-feature-dim",  "128",
            "--epochs",            "30",
            "--lr",                "1e-3",
            "--batch-size",        "256",
            "--episode-filter",    "failure_only",
            "--invert-target",                     # ← target 반전
            "--device",            "cuda",
        ]

        log_path = LOG_DIR / f"train_fail_inv_{suite}_{j:02d}.log"
        ret = subprocess.run(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ,
                 "PYTHONPATH": f"{BASE}/src:{BASE}/third_party/lerobot/src"},
        )

        if ret.returncode == 0:
            done_flag.touch()
            # 로그에서 invert-target 확인
            log_txt = open(log_path).read()
            inv_ok  = "[invert-target]" in log_txt
            print(f"    ✅ 완료 (target 반전 확인: {'✓' if inv_ok else '✗'})")
        else:
            print(f"    ❌ 실패 — 로그: {log_path}")

# ── STEP 3: GAF-Failure-Inverted 평가 (β = -3.0 = 기울기 반전) ───────────────
print("\n[STEP 3] GAF-Failure-Inverted 평가 (β = -3.0)")

# eval_policy.py가 음수 β를 지원하는지 확인
ep_src = open(EVAL_SCRIPT).read()
neg_beta_ok = "qgf_negate" in ep_src or "negate" in ep_src

# 음수 β 지원 여부에 따라 다르게 처리
# eval_policy.py는 내부에서 v_guided = v - grad_Q / β 이므로
# β를 음수로 넣으면: v - grad_Q / (-3) = v + grad_Q/3 → 기울기 반전 ✓
BETA = "-3.0"

for suite in SUITES:
    for j in range(N_TASKS):
        out_dir     = EVAL_DIR / suite / f"task{j:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result_file = out_dir / "eval_info.json"

        if result_file.exists():
            info = json.loads(result_file.read_text())
            pc   = info["overall"]["pc_success"]
            print(f"  [DONE] {suite}/task{j:02d}: {pc:.1f}%")
            continue

        ckpt_dir    = CRITIC_DIR / suite / f"task{j:02d}"
        critic_ckpt = ckpt_dir / "critic.pt"

        if not critic_ckpt.exists():
            # 학습 완료됐지만 파일명 다를 수 있음
            alts = list(ckpt_dir.glob("*.pt"))
            if alts:
                critic_ckpt = alts[0]
            else:
                print(f"  [WARN] {suite}/task{j:02d}: critic.pt 없음")
                continue

        print(f"  [EVAL] {suite}/task{j:02d}: β={BETA} (기울기 반전)...", flush=True)

        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--policy-path", POLICY_PATH,
            "--task",        suite,
            "--task-ids",    str(j),
            "--n-episodes",  "50",
            "--output-dir",  str(out_dir),
            "--critic-path", str(critic_ckpt),
            "--qgf-beta",    BETA,   # 음수 β → 기울기 방향 반전
            "--seed",        "4000",
        ]

        log_path = LOG_DIR / f"eval_fail_inv_{suite}_{j:02d}.log"
        ret = subprocess.run(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ,
                 "MUJOCO_GL": "egl", "WANDB_MODE": "disabled",
                 "PYTHONPATH": f"{BASE}/src:{BASE}/third_party/lerobot/src"},
        )

        if result_file.exists():
            info = json.loads(result_file.read_text())
            pc   = info["overall"]["pc_success"]
            print(f"    ✅ {pc:.1f}%")
        else:
            found = list(out_dir.rglob("eval_info.json"))
            if found:
                shutil.copy(str(found[0]), str(result_file))
                info = json.loads(result_file.read_text())
                pc   = info["overall"]["pc_success"]
                print(f"    ✅ {pc:.1f}% (경로 조정)")
            else:
                print(f"    ❌ 실패 — 로그: {log_path}")

# ── STEP 4: 전체 4개 모델 + Method A + Method B 비교 요약 ─────────────────────
print("\n" + "=" * 65)
print("  최종 결과 요약 (5개 모델 비교)")
print("=" * 65)

models = {
    "Baseline":       BASE / "runs" / "baseline",
    "GAF-Full":       BASE / "runs" / "eval_gaf",
    "GAF-Success":    BASE / "runs" / "eval_gaf_success",
    "GAF-Fail(A)":    BASE / "runs" / "eval_gaf_failure",     # Method A (붕괴된 크리틱)
    "GAF-Fail(B)":    EVAL_DIR,                               # Method B (반전 학습)
}

overall = {m: [] for m in models}
suite_avg = {}

for suite in SUITES:
    suite_avg[suite] = {}
    for model_name, run_dir in models.items():
        vals = []
        for j in range(N_TASKS):
            p = run_dir / suite / f"task{j:02d}" / "eval_info.json"
            if p.exists():
                d  = json.loads(p.read_text())
                pc = d["overall"]["pc_success"]
                vals.append(pc)
                overall[model_name].append(pc)
        suite_avg[suite][model_name] = round(sum(vals)/len(vals), 1) if vals else None

print(f"\n{'Suite':<18} {'Baseline':>9} {'Full':>9} {'Succ':>9} {'Fail-A':>9} {'Fail-B':>9}")
print("-" * 68)
for suite in SUITES:
    row = f"{suite:<18}"
    for m in models:
        v = suite_avg[suite].get(m)
        row += f" {v:>8.1f}%" if v is not None else f" {'N/A':>8}"
    print(row)
print("-" * 68)
row = f"{'Overall (40T)':<18}"
for m in models:
    vals = overall[m]
    row += f" {sum(vals)/len(vals):>8.1f}%" if vals else f" {'N/A':>8}"
print(row)

# JSON 저장
result = {
    "models": list(models.keys()),
    "suite": suite_avg,
    "overall": {m: round(sum(v)/len(v), 1) if v else None for m, v in overall.items()},
    "notes": {
        "GAF-Fail(A)": "failure_only filter, target=0 (naive, critic collapsed)",
        "GAF-Fail(B)": "failure_only filter, target=1 (inverted), beta=-3.0 at inference",
    }
}
json.dump(result, open(EVAL_DIR / "final_summary_5models.json", "w"), indent=2)
print(f"\n저장: {EVAL_DIR}/final_summary_5models.json")
print("✅ Method B 파이프라인 완료!")
