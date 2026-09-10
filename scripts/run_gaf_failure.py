#!/usr/bin/env python3
"""
GAF-Failure 전체 파이프라인
- 실패 에피소드만으로 Q-Critic 학습
- GAF-Failure 모델 평가 (50 episodes / task, β=3.0)
- 결과: runs/eval_gaf_failure/
"""

import sys, os, json, time, shutil, subprocess
from pathlib import Path
import torch

BASE = Path("/home/inc-kyh/experiments/gaf-comparison")
COLLECT_DIR    = BASE / "runs" / "collect"
CRITIC_DIR     = BASE / "runs" / "critics_failure"
EVAL_DIR       = BASE / "runs" / "eval_gaf_failure"
TRAIN_SCRIPT   = BASE / "scripts" / "train_critic.py"
EVAL_SCRIPT    = BASE / "scripts" / "eval_policy.py"
LOG_DIR        = BASE / "logs"
POLICY_PATH    = "lerobot/smolvla_libero"

SUITES  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10

# ──────────────────────────────────────────────────────────────────────────────
def episode_is_success(ep: dict) -> bool:
    s = ep.get("success")
    if s is None:
        return False
    t = torch.as_tensor(s)
    return bool(t[-1].item()) if t.numel() > 0 else False

def count_fail_success(ep_dir: Path):
    from guided_action_flow.training.critic_dataset import discover_episode_files, load_episode_files
    files = discover_episode_files(str(ep_dir))
    eps   = load_episode_files(files)
    succ  = [ep for ep in eps if episode_is_success(ep)]
    fail  = [ep for ep in eps if not episode_is_success(ep)]
    return succ, fail, eps

# ──────────────────────────────────────────────────────────────────────────────
#  단계 1: 실패 에피소드 현황 파악
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: 실패 에피소드 수 파악")
print("=" * 65)

os.makedirs(CRITIC_DIR, exist_ok=True)
fail_info = {}
skip_set  = set()

# activate gaf-exp env path
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "third_party" / "lerobot" / "src"))

for suite in SUITES:
    fail_info[suite] = {}
    for j in range(N_TASKS):
        ep_dir = COLLECT_DIR / suite / f"task{j:02d}" / "episodes"
        if not ep_dir.exists():
            print(f"  [MISS] {suite}/task{j:02d}: episodes/ 없음")
            skip_set.add((suite, j))
            fail_info[suite][j] = {"n_succ": 0, "n_fail": 0}
            continue

        succ_eps, fail_eps, all_eps = count_fail_success(ep_dir)
        n_s, n_f = len(succ_eps), len(fail_eps)
        fail_info[suite][j] = {"n_succ": n_s, "n_fail": n_f}

        if n_f == 0:
            note = " ⚠️  실패 0개 → 수집 성공률 100% (베이스라인 점수 그대로 사용)"
            skip_set.add((suite, j))
        else:
            note = ""
        print(f"  {suite}/task{j:02d}: 전체={n_s+n_f} / 성공={n_s} / 실패={n_f}{note}")

json.dump(fail_info, open(CRITIC_DIR / "fail_info.json", "w"), indent=2)
print(f"\n  ▶ 실패 에피소드 없는 태스크: {len(skip_set)}개 → 스킵")
print(f"  ▶ 학습 대상: {40 - len(skip_set)}개 태스크\n")

# ──────────────────────────────────────────────────────────────────────────────
#  단계 2: failure_only Q-Critic 학습
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 2: GAF-Failure Q-Critic 학습")
print("=" * 65)

for suite in SUITES:
    for j in range(N_TASKS):
        ckpt_dir = CRITIC_DIR / suite / f"task{j:02d}" / "failure_only"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        done_flag = ckpt_dir / "done.flag"
        if done_flag.exists():
            print(f"  [SKIP] {suite}/task{j:02d}: 이미 학습 완료")
            continue

        if (suite, j) in skip_set:
            # 스킵 마커만 남기기
            json.dump({"reason": "no_fail_episodes"}, open(ckpt_dir / "skip.json", "w"))
            print(f"  [SKIP] {suite}/task{j:02d}: 실패 에피소드 0개")
            continue

        n_fail = fail_info[suite][j]["n_fail"]
        data_dir = str(COLLECT_DIR / suite / f"task{j:02d}")

        print(f"  [TRAIN] {suite}/task{j:02d}: 실패 {n_fail}개로 학습 중...", flush=True)

        # 기존 train_critic.py에 --episode-filter=failure_only 인자 추가
        # train_critic.py에 해당 인자가 없다면 wrapper 스크립트 사용
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-dir",          data_dir,
            "--output-dir",        str(ckpt_dir),
            "--obs-source",        "state",
            "--task-feature-source", "tokens",
            "--task-feature-dim",  "128",
            "--epochs",            "30",
            "--lr",                "1e-3",
            "--batch-size",        "256",
            "--episode-filter",    "failure_only",  # 실패만 사용
            "--device",            "cuda",
        ]

        log_path = LOG_DIR / f"train_fail_{suite}_{j:02d}.log"
        ret = subprocess.run(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": f"{BASE}/src:{BASE}/third_party/lerobot/src"},
        )

        if ret.returncode == 0:
            done_flag.touch()
            print(f"    ✅ 완료")
        else:
            # fallback: --episode-filter 미지원 → patch해서 재실행
            print(f"    ⚠️ --episode-filter 미지원 → patch 스크립트로 재실행")
            _train_failure_patched(
                data_dir=data_dir,
                ckpt_dir=ckpt_dir,
                log_path=log_path,
                suite=suite, j=j,
            )
            done_flag.touch()

# ──────────────────────────────────────────────────────────────────────────────
#  (fallback) 내장 학습 함수 — train_critic.py가 --episode-filter 미지원 시 사용
# ──────────────────────────────────────────────────────────────────────────────
def _train_failure_patched(data_dir, ckpt_dir, log_path, suite, j):
    """실패 에피소드만 필터링해 직접 학습"""
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from guided_action_flow.training.critic_dataset import (
        discover_episode_files, load_episode_files,
        build_action_chunk_dataset, split_indices_by_episode,
    )
    from guided_action_flow.critics.action_chunk_critic import (
        ActionChunkCritic, ActionChunkCriticConfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    files   = discover_episode_files(data_dir)
    all_eps = load_episode_files(files)

    # ★ 실패 에피소드만 필터링
    fail_eps = [ep for ep in all_eps if not episode_is_success(ep)]
    if not fail_eps:
        print(f"    [WARN] 실패 에피소드 없음 — 스킵")
        json.dump({"reason": "no_fail_episodes"}, open(ckpt_dir / "skip.json", "w"))
        return

    print(f"    실패 에피소드 {len(fail_eps)}개로 학습 (전체 {len(all_eps)}개 중)", flush=True)

    dataset = build_action_chunk_dataset(
        fail_eps,
        action_horizon=50,
        stride=1,
        gamma=0.99,
        obs_source="state",
        task_feature_source="tokens",
        task_feature_dim=128,
    )

    obs_features   = dataset["obs_features"].to(device)
    proprio        = dataset["proprio"].to(device)
    action_chunks  = dataset["action_chunks"].to(device)
    targets        = dataset["targets"].to(device)
    task_features  = dataset.get("task_features")
    if task_features is not None:
        task_features = task_features.to(device)

    # 실패 에피소드만이므로 모든 target은 0 (실패 신호만 있음)
    # → 크리틱이 "이 행동들은 실패한다"를 배움 (음수 학습)
    print(f"    target 분포: min={targets.min():.2f} max={targets.max():.2f} mean={targets.mean():.3f}")

    n = targets.shape[0]
    idx = torch.randperm(n)
    val_n  = max(1, int(0.2 * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    cfg = ActionChunkCriticConfig(
        obs_dim=obs_features.shape[-1],
        task_feature_dim=128,
        action_dim=action_chunks.shape[-1],
        action_horizon=50,
        hidden_dim=256,
        depth=2,
    )
    critic = ActionChunkCritic(cfg).to(device)
    opt    = torch.optim.AdamW(critic.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, patience, pat_cnt = 1e9, 5, 0
    best_ckpt = None

    for epoch in range(30):
        critic.train()
        perm   = tr_idx[torch.randperm(len(tr_idx))]
        tr_loss = 0.0
        steps   = 0
        for start in range(0, len(perm), 256):
            b = perm[start:start + 256]
            q = critic(
                obs_features[b],
                proprio[b],
                action_chunks[b],
                task_features=task_features[b] if task_features is not None else None,
            ).squeeze(-1)
            loss = F.mse_loss(q, targets[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); steps += 1

        critic.eval()
        with torch.no_grad():
            vb = val_idx
            qv = critic(
                obs_features[vb], proprio[vb], action_chunks[vb],
                task_features=task_features[vb] if task_features is not None else None,
            ).squeeze(-1)
            val_loss = F.mse_loss(qv, targets[vb]).item()

        if epoch % 5 == 0:
            print(f"    epoch {epoch:3d}: tr={tr_loss/max(steps,1):.4f}  val={val_loss:.4f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = {k: v.cpu() for k, v in critic.state_dict().items()}
            pat_cnt   = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f"    Early stop @ epoch {epoch}")
                break

    if best_ckpt:
        torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    else:
        torch.save(critic.cpu().state_dict(), ckpt_dir / "critic_best.pt")

    # config 저장
    json.dump({
        "obs_source": "state",
        "task_feature_dim": 128,
        "episode_filter": "failure_only",
        "n_fail_episodes": len(fail_eps),
        "n_total_episodes": len(all_eps),
        "val_loss_best": best_val,
    }, open(ckpt_dir / "config.json", "w"), indent=2)
    print(f"    ✅ 크리틱 저장: {ckpt_dir / 'critic_best.pt'}")


# ──────────────────────────────────────────────────────────────────────────────
#  단계 3: GAF-Failure 평가
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 3: GAF-Failure 평가 (50 episodes/task, β=3.0)")
print("=" * 65)

for suite in SUITES:
    for j in range(N_TASKS):
        out_dir = EVAL_DIR / suite / f"task{j:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result_file = out_dir / "eval_info.json"
        if result_file.exists():
            info = json.loads(result_file.read_text())
            pc   = info["overall"]["pc_success"]
            print(f"  [DONE] {suite}/task{j:02d}: {pc:.1f}%")
            continue

        ckpt_dir  = CRITIC_DIR / suite / f"task{j:02d}" / "failure_only"
        skip_json = ckpt_dir / "skip.json"

        if skip_json.exists():
            # 실패 에피소드 없음 → baseline 점수 복사
            baseline = BASE / "runs" / "baseline" / suite / f"task{j:02d}" / "eval_info.json"
            if baseline.exists():
                shutil.copy(str(baseline), str(result_file))
                info = json.loads(baseline.read_text())
                pc   = info["overall"]["pc_success"]
                print(f"  [COPY] {suite}/task{j:02d}: {pc:.1f}% (baseline 복사 — 실패 에피소드 없음)")
            continue

        critic_ckpt = ckpt_dir / "critic_best.pt"
        if not critic_ckpt.exists():
            print(f"  [WARN] {suite}/task{j:02d}: critic_best.pt 없음 — 스킵")
            continue

        print(f"  [EVAL] {suite}/task{j:02d}: 평가 중...", flush=True)

        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--policy-path", POLICY_PATH,
            "--task",        suite,
            "--task-ids",    str(j),
            "--n-episodes",  "50",
            "--out-dir",     str(out_dir),
            "--critic-path", str(critic_ckpt),
            "--qgf-beta",    "3.0",
            "--seed",        "3000",
        ]

        log_path = LOG_DIR / f"eval_fail_{suite}_{j:02d}.log"
        ret = subprocess.run(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "MUJOCO_GL": "egl",
                 "PYTHONPATH": f"{BASE}/src:{BASE}/third_party/lerobot/src"},
        )

        if result_file.exists():
            info = json.loads(result_file.read_text())
            pc   = info["overall"]["pc_success"]
            print(f"    ✅ {pc:.1f}%")
        else:
            print(f"    ❌ 실패 — 로그: {log_path}")

# ──────────────────────────────────────────────────────────────────────────────
#  단계 4: 최종 결과 요약
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 4: 최종 결과 요약 (Baseline vs Full vs Success vs Failure)")
print("=" * 65)

models = {
    "Baseline":    BASE / "runs" / "baseline",
    "GAF-Full":    BASE / "runs" / "eval_gaf",
    "GAF-Success": BASE / "runs" / "eval_gaf_success",
    "GAF-Failure": EVAL_DIR,
}

suite_results = {s: {m: [] for m in models} for s in SUITES}

for suite in SUITES:
    for j in range(N_TASKS):
        for model_name, run_dir in models.items():
            p = run_dir / suite / f"task{j:02d}" / "eval_info.json"
            if p.exists():
                d = json.loads(p.read_text())
                pc = d["overall"]["pc_success"]
                suite_results[suite][model_name].append(pc)
            else:
                suite_results[suite][model_name].append(None)

print(f"\n{'Suite':<16} {'Baseline':>10} {'GAF-Full':>10} {'GAF-Succ':>10} {'GAF-Fail':>10}")
print("-" * 60)

overall = {m: [] for m in models}
for suite in SUITES:
    row = f"{suite:<16}"
    for m in models:
        vals = [v for v in suite_results[suite][m] if v is not None]
        avg  = sum(vals) / len(vals) if vals else 0
        row += f" {avg:>9.1f}%"
        overall[m].extend(vals)
    print(row)

print("-" * 60)
row = f"{'Overall (40T)':<16}"
for m in models:
    vals = overall[m]
    avg  = sum(vals) / len(vals) if vals else 0
    row += f" {avg:>9.1f}%"
print(row)

# JSON 저장
summary = {}
for suite in SUITES:
    summary[suite] = {}
    for m in models:
        vals = [v for v in suite_results[suite][m] if v is not None]
        summary[suite][m] = {
            "per_task": suite_results[suite][m],
            "avg": round(sum(vals) / len(vals), 2) if vals else 0,
        }

json.dump(summary, open(EVAL_DIR / "final_summary.json", "w"), indent=2)
print(f"\n결과 저장: {EVAL_DIR / 'final_summary.json'}")
print("\n✅ GAF-Failure 파이프라인 완료!")
