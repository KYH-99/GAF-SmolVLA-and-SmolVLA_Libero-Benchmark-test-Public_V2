#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path("/home/inc-kyh/experiments/gaf-comparison")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10

print("\n" + "=" * 80)
print("  최종 결과 요약 (5개 모델 비교)")
print("=" * 80)

models = {
    "Baseline": BASE / "runs" / "baseline",
    "GAF-Full": BASE / "runs" / "eval_gaf",
    "GAF-Success": BASE / "runs" / "eval_gaf_success",
    "GAF-Fail (Naive)": BASE / "runs" / "eval_gaf_failure",
    "GAF-Fail (Inverted)": BASE / "runs" / "eval_gaf_failure_inv",
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
                try:
                    d = json.loads(p.read_text())
                    pc = d["overall"]["pc_success"]
                    vals.append(pc)
                    overall[model_name].append(pc)
                except Exception:
                    pass
        suite_avg[suite][model_name] = round(sum(vals)/len(vals), 1) if vals else None

header = f"{'Suite':<16}"
for m in models:
    header += f" {m:>20}"
print(header)
print("-" * 120)

for suite in SUITES:
    row = f"{suite:<16}"
    for m in models:
        v = suite_avg[suite].get(m)
        row += f" {v:>19.1f}%" if v is not None else f" {'N/A':>20}"
    print(row)
print("-" * 120)

row = f"{'Overall (40T)':<16}"
for m in models:
    vals = overall[m]
    if vals:
        row += f" {sum(vals)/len(vals):>19.1f}% ({len(vals)}/40)"
    else:
        row += f" {'N/A':>20}"
print(row)

result = {
    "models": list(models.keys()),
    "suite_avg": suite_avg,
    "overall": {m: round(sum(v)/len(v), 1) if v else None for m, v in overall.items()},
    "notes": {
        "GAF-Fail (Naive)": "Failure-only episodes, target=0 (trivial solution / collapsed critic)",
        "GAF-Fail (Inverted)": "Failure-only episodes, inverted target (failure=1), inverted gradient at inference",
    }
}
out_path = BASE / "runs" / "final_summary_5models.json"
json.dump(result, open(out_path, "w"), indent=2)
print(f"\n✅ JSON 저장 완료: {out_path}")
