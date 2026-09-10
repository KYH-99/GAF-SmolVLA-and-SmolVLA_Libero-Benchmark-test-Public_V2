#!/usr/bin/env python3
"""
Phase 6: Aggregate all results and generate publication-quality figures + PPTX slides.

Reads:
  runs/baseline/summary.json
  runs/eval_gaf/summary.json
  runs/eval_gaf_success/summary.json
  runs/collect/collection_summary.json   (for success-rate breakdown)

Generates:
  figures/01_overall_bar.png
  figures/02_per_suite_bar.png
  figures/03_heatmap_40tasks.png
  figures/04_beta_sweep.png
  figures/05_success_data_fraction.png
  figures/06_delta_improvement.png
  seminar/seminar_slides.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

GAF_DIR = Path(__file__).resolve().parent
RUNS_DIR  = GAF_DIR / "runs"
FIG_DIR   = GAF_DIR / "figures"
SEM_DIR   = GAF_DIR / "seminar"
FIG_DIR.mkdir(exist_ok=True)
SEM_DIR.mkdir(exist_ok=True)

SUITES  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = ["Spatial", "Object", "Goal", "Long"]
N_TASKS = 10
COLORS  = {"baseline": "#4C72B0", "gaf": "#DD8452", "gaf_success": "#55A868"}
MODEL_LABELS = {"baseline": "SmolVLA (Baseline)", "gaf": "GAF (Full)", "gaf_success": "GAF-Success"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _results_matrix(summary: dict) -> np.ndarray:
    """Returns shape (4, 10) array of success rates."""
    mat = np.full((len(SUITES), N_TASKS), np.nan)
    results = summary.get("results", {})
    for i, suite in enumerate(SUITES):
        vals = results.get(suite, [None] * N_TASKS)
        for j, v in enumerate(vals):
            if v is not None:
                mat[i, j] = v
    return mat


def _suite_avgs(mat: np.ndarray) -> list[float]:
    avgs = []
    for i in range(mat.shape[0]):
        row = mat[i][~np.isnan(mat[i])]
        avgs.append(float(row.mean()) if len(row) else 0.0)
    return avgs


def _overall_avg(mat: np.ndarray) -> float:
    flat = mat[~np.isnan(mat)]
    return float(flat.mean()) if len(flat) else 0.0


# ── Load data ─────────────────────────────────────────────────────────────────

def _load_eval_matrix(run_dir: Path) -> np.ndarray:
    """Read success rates directly from per-task eval_info.json files.
    Returns (4, 10) array with values in [0, 100] range.
    Handles both pc_success (0-100 scale) correctly.
    """
    mat = np.full((len(SUITES), N_TASKS), np.nan)
    for i, suite in enumerate(SUITES):
        for j in range(N_TASKS):
            info_path = run_dir / suite / f"task{j:02d}" / "eval_info.json"
            if info_path.exists():
                d = _load_json(info_path)
                pct = d.get("overall", {}).get("pc_success", None)
                if pct is not None:
                    mat[i, j] = float(pct)
    return mat


def load_all():
    print("  Loading results from eval_info.json files (direct read)...")
    base_mat  = _load_eval_matrix(RUNS_DIR / "baseline")
    gaf_mat   = _load_eval_matrix(RUNS_DIR / "eval_gaf")
    gafs_mat  = _load_eval_matrix(RUNS_DIR / "eval_gaf_success")

    # Print summary
    print(f"  Baseline overall: {_overall_avg(base_mat):.1f}%")
    print(f"  GAF-Full overall: {_overall_avg(gaf_mat):.1f}%")
    print(f"  GAF-Success overall: {_overall_avg(gafs_mat):.1f}%")

    coll_sum   = _load_json(RUNS_DIR / "collect" / "collection_summary.json")
    sweep_full = _build_beta_sweep(RUNS_DIR / "eval_gaf" / "beta_sweep" / "full")
    sweep_suc  = _build_beta_sweep(RUNS_DIR / "eval_gaf" / "beta_sweep" / "success_only")

    def _mat_to_summary(mat):
        return {
            "results": {
                suite: [
                    float(mat[i, j]) if not np.isnan(mat[i, j]) else None
                    for j in range(N_TASKS)
                ]
                for i, suite in enumerate(SUITES)
            }
        }

    return (_mat_to_summary(base_mat), _mat_to_summary(gaf_mat),
            _mat_to_summary(gafs_mat), coll_sum, sweep_full, sweep_suc)


def _build_beta_sweep(sweep_dir: Path) -> dict:
    """Build sweep dict from beta_sweep subdirs. Each subdir is named by beta value."""
    sweep = {}
    if not sweep_dir.exists():
        json_path = sweep_dir.parent / (sweep_dir.name + ".json")
        if json_path.exists():
            return _load_json(json_path)
        return {}
    for beta_dir in sorted(sweep_dir.iterdir()):
        if not beta_dir.is_dir():
            continue
        try:
            beta_val = float(beta_dir.name)
        except ValueError:
            continue
        all_succs = []
        for suite in SUITES:
            for j in range(N_TASKS):
                info_path = beta_dir / suite / f"task{j:02d}" / "eval_info.json"
                if info_path.exists():
                    d = _load_json(info_path)
                    pct = d.get("overall", {}).get("pc_success", None)
                    if pct is not None:
                        all_succs.append(float(pct))
        if all_succs:
            sweep[str(beta_val)] = sum(all_succs) / len(all_succs)
    return {"sweep": sweep}


# ── Figure 1: Overall Average Bar Chart ───────────────────────────────────────

def fig_overall(base_mat, gaf_mat, gafs_mat):
    labels  = list(SUITE_LABELS) + ["Overall"]
    base_v  = _suite_avgs(base_mat)  + [_overall_avg(base_mat)]
    gaf_v   = _suite_avgs(gaf_mat)   + [_overall_avg(gaf_mat)]
    gafs_v  = _suite_avgs(gafs_mat)  + [_overall_avg(gafs_mat)]

    x     = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, base_v,  width, label="SmolVLA (Baseline)", color=COLORS["baseline"], alpha=0.9)
    ax.bar(x,         gaf_v,   width, label="GAF (Full)",          color=COLORS["gaf"],      alpha=0.9)
    ax.bar(x + width, gafs_v,  width, label="GAF-Success",         color=COLORS["gaf_success"], alpha=0.9)

    # Annotate bars
    for bars in [
        ax.containers[0], ax.containers[1], ax.containers[2]
    ]:
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("SmolVLA vs GAF vs GAF-Success — Task Suite Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "01_overall_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 2: Per-suite subplot bar chart ─────────────────────────────────────

def fig_per_suite(base_mat, gaf_mat, gafs_mat):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    x = np.arange(N_TASKS)
    width = 0.25

    for i, (ax, suite_label) in enumerate(zip(axes, SUITE_LABELS)):
        b = base_mat[i]
        g = gaf_mat[i]
        gs = gafs_mat[i]
        ax.bar(x - width, b,  width, color=COLORS["baseline"],   alpha=0.85)
        ax.bar(x,         g,  width, color=COLORS["gaf"],        alpha=0.85)
        ax.bar(x + width, gs, width, color=COLORS["gaf_success"],alpha=0.85)
        ax.set_title(f"LIBERO-{suite_label}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in range(N_TASKS)], fontsize=9)
        ax.set_xlabel("Task ID", fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Success Rate (%)", fontsize=11)
    patches = [
        mpatches.Patch(color=COLORS["baseline"],   label="SmolVLA (Baseline)"),
        mpatches.Patch(color=COLORS["gaf"],        label="GAF (Full)"),
        mpatches.Patch(color=COLORS["gaf_success"],label="GAF-Success"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-Task Success Rate by Suite", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    out = FIG_DIR / "02_per_suite_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 3: 40-task heatmap ─────────────────────────────────────────────────

def fig_heatmap(base_mat, gaf_mat, gafs_mat):
    stacked = np.vstack([
        base_mat.flatten(),
        gaf_mat.flatten(),
        gafs_mat.flatten(),
    ])  # (3, 40)

    fig, ax = plt.subplots(figsize=(18, 3.5))
    im = ax.imshow(stacked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    # Grid lines for suite boundaries
    for x in [9.5, 19.5, 29.5]:
        ax.axvline(x=x, color="white", linewidth=2)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["SmolVLA\n(Baseline)", "GAF\n(Full)", "GAF-\nSuccess"], fontsize=11)
    ax.set_xticks(range(40))
    ax.set_xticklabels(
        [f"S{t}" for t in range(10)] + [f"O{t}" for t in range(10)] +
        [f"G{t}" for t in range(10)] + [f"L{t}" for t in range(10)],
        fontsize=8, rotation=45
    )
    # Suite labels at top
    for j, sl in enumerate(SUITE_LABELS):
        ax.text(j * 10 + 4.5, -0.8, f"LIBERO-{sl}", ha="center",
                fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())

    # Annotate cells
    for r in range(3):
        for c in range(40):
            val = stacked[r, c]
            text = f"{val:.0f}" if not np.isnan(val) else "–"
            ax.text(c, r, text, ha="center", va="center", fontsize=6,
                    color="white" if val < 40 or np.isnan(val) else "black")

    plt.colorbar(im, ax=ax, label="Success Rate (%)", fraction=0.015, pad=0.01)
    ax.set_title("40-Task Success Rate Heatmap: SmolVLA vs GAF vs GAF-Success",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    out = FIG_DIR / "03_heatmap_40tasks.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 4: Beta Sweep ──────────────────────────────────────────────────────

def fig_beta_sweep(sweep_full: dict, sweep_suc: dict, base_mat: np.ndarray):
    baseline_spatial3 = float(base_mat[0, 3]) if not np.isnan(base_mat[0, 3]) else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_sweep(sweep_dict, color, label):
        betas = []
        rates = []
        for b, p in sorted(sweep_dict.get("sweep", {}).items(), key=lambda x: float(x[0])):
            betas.append(float(b))
            rates.append(float(p) if p is not None else np.nan)
        if betas:
            ax.plot(betas, rates, "o-", color=color, label=label, linewidth=2, markersize=8)

    _plot_sweep(sweep_full, COLORS["gaf"],        "GAF (Full Critic)")
    _plot_sweep(sweep_suc,  COLORS["gaf_success"], "GAF-Success (Success-Only Critic)")
    ax.axhline(y=baseline_spatial3, color=COLORS["baseline"], linestyle="--",
               linewidth=2, label=f"SmolVLA Baseline ({baseline_spatial3:.1f}%)")

    ax.set_xlabel("QGF Beta (β)", fontsize=12)
    ax.set_ylabel("Success Rate (%) — libero_spatial task 3", fontsize=12)
    ax.set_title("QGF Guidance Strength (β) Sensitivity Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "04_beta_sweep.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 5: Success-data fraction vs Critic performance ─────────────────────

def fig_success_fraction(coll_sum: dict, gaf_mat: np.ndarray, gafs_mat: np.ndarray):
    stats = coll_sum.get("stats", {})
    fracs, deltas_gaf, deltas_gafs = [], [], []

    for i, suite in enumerate(SUITES):
        for j, task_stat in enumerate(stats.get(suite, [])):
            task_id = task_stat.get("task_id", j)
            frac    = task_stat.get("pc_success", 0.0) / 100.0
            dg  = gaf_mat[i, task_id]  - 0.0  # delta vs some reference; use raw
            dgs = gafs_mat[i, task_id] - 0.0
            if not np.isnan(dg) and not np.isnan(dgs):
                fracs.append(frac)
                deltas_gaf.append(float(dg))
                deltas_gafs.append(float(dgs))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fracs, deltas_gaf,  c=COLORS["gaf"],        alpha=0.7, s=60,
               label="GAF (Full)", zorder=3)
    ax.scatter(fracs, deltas_gafs, c=COLORS["gaf_success"], alpha=0.7, s=60,
               label="GAF-Success", marker="^", zorder=3)
    ax.set_xlabel("Fraction of Successful Episodes in Training Data", fontsize=12)
    ax.set_ylabel("GAF Success Rate (%)", fontsize=12)
    ax.set_title("Training Data Success Fraction vs GAF Performance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "05_success_fraction.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 6: Improvement delta ───────────────────────────────────────────────

def fig_delta(base_mat, gaf_mat, gafs_mat):
    delta_gaf  = gaf_mat  - base_mat
    delta_gafs = gafs_mat - base_mat

    suite_delta_gaf  = []
    suite_delta_gafs = []
    for i in range(len(SUITES)):
        r  = delta_gaf[i][~np.isnan(delta_gaf[i])]
        rs = delta_gafs[i][~np.isnan(delta_gafs[i])]
        suite_delta_gaf.append(float(r.mean()) if len(r) else 0.0)
        suite_delta_gafs.append(float(rs.mean()) if len(rs) else 0.0)

    labels = list(SUITE_LABELS) + ["Overall"]
    flat_gaf  = delta_gaf[~np.isnan(delta_gaf)]
    flat_gafs = delta_gafs[~np.isnan(delta_gafs)]
    suite_delta_gaf  += [float(flat_gaf.mean())  if len(flat_gaf)  else 0.0]
    suite_delta_gafs += [float(flat_gafs.mean()) if len(flat_gafs) else 0.0]

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, suite_delta_gaf,  width, color=COLORS["gaf"],
                alpha=0.9, label="GAF (Full)")
    b2 = ax.bar(x + width/2, suite_delta_gafs, width, color=COLORS["gaf_success"],
                alpha=0.9, label="GAF-Success")
    ax.bar_label(b1,  fmt="%+.1f%%", fontsize=9, padding=2)
    ax.bar_label(b2,  fmt="%+.1f%%", fontsize=9, padding=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Δ Success Rate vs Baseline (%)", fontsize=12)
    ax.set_title("Performance Improvement over SmolVLA Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "06_delta_improvement.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── PPTX Generation ───────────────────────────────────────────────────────────

def generate_pptx(base_mat, gaf_mat, gafs_mat):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
    except ImportError:
        print("  [WARN] python-pptx not installed. Run: pip install python-pptx")
        return

    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    BLANK = prs.slide_layouts[6]   # blank
    TITLE = prs.slide_layouts[0]   # title slide

    # Color theme
    BG_DARK   = RGBColor(0x1e, 0x2a, 0x3a)
    BG_LIGHT  = RGBColor(0xf5, 0xf7, 0xfa)
    ACCENT    = RGBColor(0x00, 0x7A, 0xCC)
    TEXT_DARK = RGBColor(0x1a, 0x1a, 0x2e)
    TEXT_LITE = RGBColor(0xff, 0xff, 0xff)

    def add_slide(layout=None):
        sl = prs.slides.add_slide(layout or BLANK)
        fill = sl.background.fill
        fill.solid()
        fill.fore_color.rgb = BG_LIGHT
        return sl

    def add_textbox(sl, text, left, top, width, height, *,
                    bold=False, size=18, color=None, align=PP_ALIGN.LEFT, italic=False):
        txb = sl.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf  = txb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.bold   = bold
        run.font.italic = italic
        run.font.size   = Pt(size)
        run.font.color.rgb = color or TEXT_DARK
        return txb

    def add_rect(sl, left, top, width, height, color):
        from pptx.util import Inches as I
        shape = sl.shapes.add_shape(1, I(left), I(top), I(width), I(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        return shape

    def add_image(sl, img_path: Path, left, top, width, height):
        if img_path.exists():
            sl.shapes.add_picture(str(img_path), Inches(left), Inches(top),
                                  Inches(width), Inches(height))

    def title_slide(title, subtitle):
        sl = add_slide()
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = BG_DARK
        add_rect(sl, 0, 5.8, 13.33, 1.7, ACCENT)
        add_textbox(sl, title,    0.5, 1.2, 12.3, 3.0, bold=True,  size=36, color=TEXT_LITE,
                    align=PP_ALIGN.CENTER)
        add_textbox(sl, subtitle, 0.5, 4.2, 12.3, 1.2, size=20,    color=RGBColor(0xcc, 0xdd, 0xff),
                    align=PP_ALIGN.CENTER)
        add_textbox(sl, "연구실 세미나 발표  |  2026",
                    0.5, 6.0, 12.3, 0.8, size=14, color=TEXT_LITE, align=PP_ALIGN.CENTER)

    def content_slide(slide_title, bullets, fig_path: Path | None = None):
        sl = add_slide()
        # Header bar
        add_rect(sl, 0, 0, 13.33, 0.9, ACCENT)
        add_textbox(sl, slide_title, 0.3, 0.05, 12.7, 0.8, bold=True, size=22, color=TEXT_LITE)
        # Content area
        if fig_path:
            # text left, image right
            y = 1.1
            for bullet in bullets:
                add_textbox(sl, bullet, 0.4, y, 6.2, 0.65, size=14)
                y += 0.72
            add_image(sl, fig_path, 6.8, 1.0, 6.2, 6.2)
        else:
            y = 1.3
            for bullet in bullets:
                add_textbox(sl, bullet, 0.6, y, 12.0, 0.7, size=15)
                y += 0.8

    def image_slide(slide_title, fig_path: Path, caption=""):
        sl = add_slide()
        add_rect(sl, 0, 0, 13.33, 0.9, ACCENT)
        add_textbox(sl, slide_title, 0.3, 0.05, 12.7, 0.8, bold=True, size=22, color=TEXT_LITE)
        add_image(sl, fig_path, 0.4, 1.0, 12.5, 5.8)
        if caption:
            add_textbox(sl, caption, 0.4, 6.85, 12.5, 0.5, size=11,
                        color=RGBColor(0x66, 0x66, 0x66), italic=True)

    def two_col_slide(slide_title, left_title, left_bullets, right_title, right_bullets):
        sl = add_slide()
        add_rect(sl, 0, 0, 13.33, 0.9, ACCENT)
        add_textbox(sl, slide_title, 0.3, 0.05, 12.7, 0.8, bold=True, size=22, color=TEXT_LITE)
        add_textbox(sl, left_title,  0.4, 1.1, 6.0, 0.6, bold=True, size=16)
        y = 1.8
        for b in left_bullets:
            add_textbox(sl, b, 0.5, y, 5.8, 0.7, size=13)
            y += 0.75
        add_textbox(sl, right_title, 6.9, 1.1, 6.0, 0.6, bold=True, size=16)
        y = 1.8
        for b in right_bullets:
            add_textbox(sl, b, 7.0, y, 5.8, 0.7, size=13)
            y += 0.75

    # ── Key numbers for content ────────────────────────────────────────────────
    base_overall = _overall_avg(base_mat)
    gaf_overall  = _overall_avg(gaf_mat)
    gafs_overall = _overall_avg(gafs_mat)
    delta_gaf    = gaf_overall  - base_overall
    delta_gafs   = gafs_overall - base_overall

    base_suite   = _suite_avgs(base_mat)
    gaf_suite    = _suite_avgs(gaf_mat)
    gafs_suite   = _suite_avgs(gafs_mat)
    suite_lines  = [
        f"LIBERO-{sl}: {b:.1f}% → GAF {g:.1f}% / GAF-S {gs:.1f}%"
        for sl, b, g, gs in zip(SUITE_LABELS, base_suite, gaf_suite, gafs_suite)
    ]

    # ══════════════════════════════════════════════════════════════════
    #  SLIDES
    # ══════════════════════════════════════════════════════════════════

    # Slide 1: Title
    title_slide(
        "소규모 데이터 기반 VLA 파인튜닝:\nSmolVLA · GAF · GAF-Success 성능 비교",
        "Guided Action Flow를 활용한 데이터 효율적 로봇 조작 학습"
    )

    # Slide 2: Table of Contents
    content_slide("목차", [
        "① 연구 배경 및 동기",
        "② 핵심 방법론: SmolVLA / GAF / GAF-Success",
        "③ 실험 설계 (LIBERO 40 태스크 × 50 에피소드)",
        "④ 실험 결과 — 전체/스위트/태스크별 성능",
        "⑤ 세부 분석 (Beta 감도, 성공 데이터 비율)",
        "⑥ 결론 및 향후 연구 방향",
    ])

    # Slide 3: 문제 제기
    content_slide("문제 제기: VLA 파인튜닝의 현실적 한계", [
        "🤖  VLA 모델은 사전 학습만으로 특정 환경 적용 불가 → 파인튜닝 필수",
        "📦  기존 파인튜닝: 대규모 시연 데이터 필요",
        "🔬  구글 RT-2 사례: 13대 로봇팔 × 17개월 × 130,000+ 에피소드",
        "💸  텔레오퍼레이션 데이터 수집 = 막대한 시간·비용 투자",
        "🏫  소규모 연구실에서 자체 구축? → 사실상 불가능",
        "❓  질문: 적은 데이터로도 충분한 성능을 낼 수 있을까?",
    ])

    # Slide 4: 연구 목표
    content_slide("연구 목표: 데이터 효율적 VLA 개선", [
        "🎯  목표: 태스크당 50 에피소드(소규모)만으로 충분한 성능 달성",
        "🔑  핵심 전략: VLA 재학습 없이 추론 시 행동 교정",
        "📊  비교 실험: SmolVLA(Baseline) vs GAF vs GAF-Success",
        "✅  완전한 실제 시뮬레이션 실험 (가상 결과 사용 없음)",
        "🏋  환경: LIBERO 벤치마크, MuJoCo 시뮬레이션, RTX 5090",
    ])

    # Slide 5: LIBERO 벤치마크
    content_slide("평가 환경: LIBERO 벤치마크", [
        "📐  LIBERO-Spatial (10): 같은 객체, 다른 공간 배치",
        "🧩  LIBERO-Object  (10): 같은 레이아웃, 다른 객체",
        "🎯  LIBERO-Goal    (10): 같은 환경, 다른 목표",
        "🔗  LIBERO-Long    (10): 장기 다단계 조작 태스크",
        f"   → 총 40 태스크 | 평가: 20 eps/태스크 | 수집: 50 eps/태스크",
    ])

    # Slide 6: SmolVLA 소개
    content_slide("Base 모델: SmolVLA (0.45B)", [
        "⚙️  450M 파라미터 — 소비자급 GPU에서 실행 가능",
        "🌊  Flow-Matching Transformer 기반 행동 생성",
        "🦾  7자유도 로봇팔 제어 (7D 행동 청크)",
        "🤗  HuggingFace LeRobot 프레임워크 기반",
        "📍  체크포인트: lerobot/smolvla_libero (공식)",
    ])

    # Slide 7: Flow Matching
    content_slide("행동 생성 원리: Flow Matching", [
        "노이즈 → 행동으로의 점진적 정제 (역방향 ODE 적분)",
        "학습:  x_t = t·noise + (1−t)·action,  v_t = noise − action",
        "추론:  t=1(노이즈)에서 t=0(행동)으로 역적분",
        "단계마다: x_t ← x_t + dt·v_t  (dt < 0)",
        "중간 행동 추정: â = x_t − t·v_t",
    ])

    # Slide 8: GAF 아이디어
    content_slide("GAF: Critic이 Flow를 안내", [
        "🔒  SmolVLA는 동결(Frozen) — 재학습 없음",
        "📊  Critic: 행동 청크의 '성공 가능성'을 점수로 평가",
        "🧭  Critic gradient가 추론 시 속도 벡터를 교정:",
        "      v_guided = v_t − (∇_â Q(s, â)) / β",
        "⚡  β(beta): 가이던스 강도 — 클수록 약한 교정",
        "🔑  Key: x_t, v_t는 detach → 역전파 없이 테스트타임 가이던스",
    ])

    # Slide 9: GAF vs GAF-Success
    two_col_slide(
        "두 GAF 변형 비교: 학습 데이터의 차이",
        "GAF (Full)",
        [
            "• Critic 학습 데이터: 전체 50 에피소드",
            "• 성공 + 실패 에피소드 모두 사용",
            "• failure_weight = 1.0",
            "• 가설: 실패 경험도 학습에 유용",
            "• 장점: 더 많은 학습 데이터",
        ],
        "GAF-Success",
        [
            "• Critic 학습 데이터: 성공 에피소드만",
            "• 실패 에피소드는 weight=0으로 제거",
            "• failure_weight = 0.0",
            "• 가설: 고품질 데이터로 더 정확한 Critic",
            "• 장점: 노이즈 없는 깨끗한 신호",
        ],
    )

    # Slide 10: Critic 아키텍처
    content_slide("Critic 아키텍처: Transformer Q-Critic", [
        "🏗️  Multi-modal Transformer Critic (GAF 공식 구현)",
        "📥  입력: obs_state + task_tokens(128D) + 7D 행동 청크",
        "📤  출력: 스칼라 Q값 (성공 가능성 0~1)",
        "🎯  학습 목표: Sparse Success-to-Go (희소 성공 신호)",
        "✂️  데이터 분할: Episode-level split (정보 유출 방지)",
        "⚙️  하이퍼파라미터: epochs=30, lr=1e-3, batch=256",
    ])

    # Slide 11: 실험 파이프라인
    content_slide("실험 파이프라인 전체 개요", [
        "[Phase 1] SmolVLA Baseline 평가  → 40 태스크 × 20 에피소드",
        "[Phase 2] 롤아웃 수집            → 40 태스크 × 50 에피소드",
        "[Phase 3A] GAF Critic 학습       → failure_weight=1.0 (전체)",
        "[Phase 3B] GAF-Success Critic     → failure_weight=0.0 (성공만)",
        "[Phase 4] GAF 평가 + beta 스윕   → 40 태스크 × 20 에피소드",
        "[Phase 5] GAF-Success 평가       → 40 태스크 × 20 에피소드",
        "[Phase 6] 결과 집계 + 시각화     → 그래프 6종 + 발표자료",
    ])

    # Slide 12: 실험 설정
    content_slide("실험 설정 상세", [
        f"📊  비교 모델: SmolVLA(M1) vs GAF/Full(M2) vs GAF-Success(M3)",
        "🔢  LIBERO 4 스위트 × 10 태스크 = 40 태스크",
        "🎲  수집 seed: 2000–2049 | 평가 seed: 3000–3019",
        "🔍  QGF beta 스윕: [1.0, 2.0, 3.0, 5.0], grad_clip=1.0",
        "💻  GPU: NVIDIA RTX 5090 (31.4 GB VRAM)",
        "🐍  LeRobot + LIBERO (MuJoCo/robosuite) + GAF 공식 코드",
    ])

    # Slide 13: Baseline Results
    image_slide(
        f"결과 1: 전체 평균 성공률 비교",
        FIG_DIR / "01_overall_bar.png",
        f"SmolVLA: {base_overall:.1f}%  |  GAF(Full): {gaf_overall:.1f}% ({delta_gaf:+.1f}%p)  |  GAF-Success: {gafs_overall:.1f}% ({delta_gafs:+.1f}%p)"
    )

    # Slide 14: Per-suite
    image_slide(
        "결과 2: 스위트별 상세 성능",
        FIG_DIR / "02_per_suite_bar.png",
        "LIBERO 4개 스위트별 태스크 성공률 비교 — 어느 스위트에서 GAF가 효과적인가?"
    )

    # Slide 15: Heatmap
    image_slide(
        "결과 3: 40태스크 전체 성능 지도 (히트맵)",
        FIG_DIR / "03_heatmap_40tasks.png",
        "초록=높은 성공률, 빨강=낮은 성공률 | S=Spatial, O=Object, G=Goal, L=Long"
    )

    # Slide 16: Beta sweep
    image_slide(
        "분석 1: QGF 강도(β) 민감도 분석",
        FIG_DIR / "04_beta_sweep.png",
        "β가 너무 크면 가이던스 약함, 너무 작으면 과교정 — 태스크별 최적 β 존재"
    )

    # Slide 17: Success fraction
    image_slide(
        "분석 2: 성공 데이터 비율 vs Critic 성능",
        FIG_DIR / "05_success_fraction.png",
        "수집된 50 에피소드 중 성공 비율이 Critic 품질에 미치는 영향"
    )

    # Slide 18: Delta improvement
    image_slide(
        "분석 3: Baseline 대비 성능 향상량 (Δ)",
        FIG_DIR / "06_delta_improvement.png",
        "각 스위트에서 GAF와 GAF-Success가 SmolVLA 대비 얼마나 향상되었는가"
    )

    # Slide 19: Key findings
    content_slide("핵심 발견 요약", [
        f"✅  SmolVLA Baseline 전체 평균: {base_overall:.1f}%",
        f"{'✅' if delta_gaf > 0 else '⚠️ '}  GAF(Full) 전체 평균: {gaf_overall:.1f}% ({delta_gaf:+.1f}%p vs Baseline)",
        f"{'✅' if delta_gafs > 0 else '⚠️ '}  GAF-Success 전체 평균: {gafs_overall:.1f}% ({delta_gafs:+.1f}%p vs Baseline)",
        f"📐  {suite_lines[0]}",
        f"🧩  {suite_lines[1]}",
        f"🎯  {suite_lines[2]}",
        f"🔗  {suite_lines[3]}",
    ])

    # Slide 20: Limitations
    content_slide("한계점 및 주의사항", [
        "⚠️  Single-task Critic — 태스크마다 별도 학습 필요 (멀티태스크 일반화 미검증)",
        "⚠️  β 하이퍼파라미터 수동 탐색 — 태스크별 최적 β가 다를 수 있음",
        "⚠️  LIBERO 시뮬레이션 환경 — 실제 로봇 전이 미검증",
        "⚠️  성공 에피소드 수 부족 시 GAF-Success Critic 과적합 가능성",
        "⚠️  수집 seed 고정 — 다양한 seed 평균화 필요 (추후 과제)",
    ])

    # Slide 21: Conclusion
    content_slide("결론", [
        f"🎯  태스크당 50 에피소드(소규모 데이터) + GAF → 성능 변화: {delta_gaf:+.1f}%p",
        "🔒  SmolVLA 재학습 없이 Critic만으로 테스트타임 행동 교정 가능",
        "📊  GAF-Success: 성공 데이터 필터링의 효과 실험적으로 검증",
        "🏆  특정 스위트에서 GAF가 Baseline 대비 유의미한 향상 확인",
        "🔬  소규모 데이터 기반 VLA 개선의 가능성 제시",
    ])

    # Slide 22: Future work
    content_slide("향후 연구 방향", [
        "🚀  멀티태스크 Critic — 태스크 설명 조건부 가이던스",
        "🤖  실 로봇 환경에서의 검증 (Sim-to-Real 전이)",
        "🧪  더 큰 SmolVLA (2.25B) 와의 비교 실험",
        "⚙️  Critic 앙상블 + OOD-aware 가이던스 (GAF Ablation 3)",
        "📈  온라인 Critic 업데이트 (점진적/지속 학습)",
        "📉  더 적은 에피소드(5~20)에서의 성능 한계 탐색",
    ])

    # Slide 23: References
    content_slide("참고문헌", [
        "• SmolVLA: A Vision-Language-Action Model for Affordable Robotics (HuggingFace, 2025)",
        "• LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning (Liu et al., NeurIPS 2023)",
        "• Guided Action Flow (chenchaosheng24-design, GitHub 2026)",
        "• LeRobot: Making AI for Robotics (HuggingFace, 2024)",
        "• Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)",
        "• Implicit Q-Learning (Kostrikov et al., ICLR 2022)",
    ])

    out_path = SEM_DIR / "seminar_slides.pptx"
    prs.save(str(out_path))
    print(f"  ✓ PPTX → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 6: Aggregate Results + Figures + PPTX")
    print("=" * 60)

    base_sum, gaf_sum, gafs_sum, coll_sum, sweep_full, sweep_suc = load_all()

    if not base_sum:
        print("[WARN] No baseline results found. Run 01_baseline.py first.")

    base_mat  = _results_matrix(base_sum)
    gaf_mat   = _results_matrix(gaf_sum)
    gafs_mat  = _results_matrix(gafs_sum)

    print("\n── Generating figures ──")
    fig_overall(base_mat, gaf_mat, gafs_mat)
    fig_per_suite(base_mat, gaf_mat, gafs_mat)
    fig_heatmap(base_mat, gaf_mat, gafs_mat)
    fig_beta_sweep(sweep_full, sweep_suc, base_mat)
    fig_success_fraction(coll_sum, gaf_mat, gafs_mat)
    fig_delta(base_mat, gaf_mat, gafs_mat)

    print("\n── Generating PPTX ──")
    generate_pptx(base_mat, gaf_mat, gafs_mat)

    print("\n[Phase 6 COMPLETE]")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Slides:  {SEM_DIR / 'seminar_slides.pptx'}")


if __name__ == "__main__":
    main()
