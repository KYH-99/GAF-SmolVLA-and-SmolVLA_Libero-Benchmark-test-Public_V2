import json
import matplotlib.pyplot as plt

import numpy as np
import os
import torch
from pathlib import Path

# Fix for seaborn module not found - fallback if needed, but we will use matplotlib directly for heatmap
BASE = Path("/home/inc-kyh/experiments/gaf-comparison")
RUNS_DIR = BASE / "runs"
COLLECT_DIR = RUNS_DIR / "collect"
FIG_DIR = BASE / "plots_5models_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = ["Spatial", "Object", "Goal", "Long(10)"]
N_TASKS = 10

models = {
    "Baseline": RUNS_DIR / "baseline",
    "GAF-Full": RUNS_DIR / "eval_gaf",
    "GAF-Success": RUNS_DIR / "eval_gaf_success",
    "GAF-Fail(N)": RUNS_DIR / "eval_gaf_failure",
    "GAF-Fail(I)": RUNS_DIR / "eval_gaf_failure_inv",
}

colors = ['#8C8C8C', '#F39C12', '#27AE60', '#C0392B', '#8E44AD']

# 1. Load Data
mat = {m: np.zeros((4, 10)) for m in models}
success_fraction = np.zeros((4, 10))

for s_idx, suite in enumerate(SUITES):
    for j in range(N_TASKS):
        # 1.1 Load Success Fraction
        ep_dir = COLLECT_DIR / suite / f"task{j:02d}" / "episodes"
        try:
            files = list(ep_dir.glob("episode_*.pt"))
            succ_count = 0
            for f in files:
                ep = torch.load(f, map_location='cpu', weights_only=False)
                t = torch.as_tensor(ep.get("success", [0]))
                if t.numel() > 0 and bool(t[-1].item()):
                    succ_count += 1
            success_fraction[s_idx, j] = succ_count / len(files) if files else 0
        except Exception:
            success_fraction[s_idx, j] = 0.0

        # 1.2 Load Model Scores
        for m, d in models.items():
            p = d / suite / f"task{j:02d}" / "eval_info.json"
            try:
                info = json.loads(p.read_text())
                mat[m][s_idx, j] = info["overall"]["pc_success"]
            except Exception:
                mat[m][s_idx, j] = np.nan

# 2. Overall Bar
plt.figure(figsize=(10, 6))
overall_means = [np.nanmean(mat[m]) for m in models]
bars = plt.bar(list(models.keys()), overall_means, color=colors, edgecolor='black', linewidth=1)
plt.title("Overall Success Rate (40 Tasks)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Success Rate (%)", fontsize=14)
plt.ylim(0, 100)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y+1, f"{y:.1f}%", ha='center', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_overall_bar.png", dpi=180)
plt.close()

# 3. Per-Suite Bar
plt.figure(figsize=(14, 7))
x = np.arange(len(SUITES))
width = 0.16
for i, m in enumerate(models):
    vals = [np.nanmean(mat[m][s,:]) for s in range(4)]
    bars = plt.bar(x + i*width - width*2, vals, width, label=m, color=colors[i], edgecolor='black')
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y + 1, f"{y:.1f}", ha='center', va='bottom', fontsize=10, rotation=90, fontweight='bold')
plt.title("Success Rate by Task Suite", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Success Rate (%)", fontsize=14)
plt.xticks(x, SUITE_LABELS, fontsize=12)
plt.ylim(0, 110) # 텍스트 공간 확보
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_per_suite_bar.png", dpi=180)
plt.close()

# 4. Heatmap
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Task-by-Task Success Rate Heatmap (40 Tasks)", fontsize=18, fontweight='bold', y=1.02)
for i, (m, ax) in enumerate(zip(models, axes)):
    im = ax.imshow(mat[m], cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_title(m, fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(4))
    if i == 0:
        ax.set_yticklabels(SUITE_LABELS, fontsize=12)
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"T{j}" for j in range(10)], rotation=45)
    for y in range(4):
        for x_idx in range(10):
            val = mat[m][y, x_idx]
            color = 'black' if 30 < val < 70 else 'white'
            if not np.isnan(val):
                ax.text(x_idx, y, f"{val:.0f}", ha='center', va='center', color=color, fontsize=10, fontweight='bold')
plt.colorbar(im, ax=axes.ravel().tolist(), label="Success Rate (%)", shrink=0.8, pad=0.02)
plt.savefig(FIG_DIR / "03_heatmap_40tasks.png", dpi=180, bbox_inches='tight')
plt.close()

# 5. Success Fraction vs Delta
plt.figure(figsize=(10, 6))
base_flat = mat["Baseline"].flatten()
succ_frac = success_fraction.flatten() * 100
for i, m in enumerate(list(models.keys())[1:]):
    delta = mat[m].flatten() - base_flat
    plt.scatter(succ_frac, delta, color=colors[i+1], label=m, alpha=0.7, s=80, edgecolors='k')
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title("Impact of Collection Phase Success Rate on Guidance Delta", fontsize=15, fontweight='bold')
plt.xlabel("Collection Phase Success Rate (%)", fontsize=13)
plt.ylabel("Δ Success Rate vs Baseline (%p)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_success_fraction.png", dpi=180)
plt.close()

# 6. Delta Improvement
plt.figure(figsize=(12, 6))
x = np.arange(len(SUITES))
width = 0.2
for i, m in enumerate(list(models.keys())[1:]):
    base_vals = [np.nanmean(mat["Baseline"][s,:]) for s in range(4)]
    vals = [np.nanmean(mat[m][s,:]) for s in range(4)]
    deltas = [v - b for v, b in zip(vals, base_vals)]
    bars = plt.bar(x + i*width - width*1.5, deltas, width, label=m, color=colors[i+1], edgecolor='black')
    for bar, delta in zip(bars, deltas):
        y = bar.get_height()
        offset = 0.5 if delta >= 0 else -0.5
        va = 'bottom' if delta >= 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2, y + offset, f"{delta:+.1f}", ha='center', va=va, fontsize=10, rotation=90, fontweight='bold')
        
plt.axhline(0, color='black', linewidth=1.5)
# Y축 범위 조정 (텍스트가 안 잘리도록)
plt.ylim(min(0, plt.gca().get_ylim()[0]) - 2, max(0, plt.gca().get_ylim()[1]) + 2)
plt.title("Performance Delta vs Baseline (Percentage Points)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Δ Success Rate (%p)", fontsize=14)
plt.xticks(x, SUITE_LABELS, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_delta_improvement.png", dpi=180)
plt.close()

print(f"All plots generated in {FIG_DIR}")
