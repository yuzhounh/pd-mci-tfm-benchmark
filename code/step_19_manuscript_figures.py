"""Step 19: Manuscript figures drawn from saved results.

No model is refitted and nothing new is estimated: the figures redraw the step 21 paired
bootstrap differences and the step 14 test-set predictions of the final model.

Outputs (in ../manuscript/figures/):
- fig_delta_forest.pdf: differences from LR in AUPRC (primary) and AUROC with 95% CIs,
  primary 7-predictor comparison
- fig_calibration_dca.pdf: calibration curve and decision curve of the final model
- fig_nomogram.pdf: copy of step_18_nomogram.pdf
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

fs.apply()

OUT_DIR = os.path.join("..", "manuscript", "figures")
MM = 1 / 25.4
LR_COLOR = "#1f77b4"
ORDER = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "RealMLP", "TabM", "TabICLv2", "TabPFN", "TabFM"]
LABEL = {"TabICLv2": "TabICL v2.0", "TabPFN": "TabPFN v3.5", "TabFM": "TabFM v1.0"}
FAMILY = {
    "SVM": ("Classical machine learning", "#8C8C2A"), "RF": ("Classical machine learning", "#8C8C2A"),
    "XGBoost": ("Gradient-boosted trees", "#d62728"), "LightGBM": ("Gradient-boosted trees", "#d62728"),
    "CatBoost": ("Gradient-boosted trees", "#d62728"),
    "RealMLP": ("Deep learning", "#2ca02c"), "TabM": ("Deep learning", "#2ca02c"),
    "TabICLv2": ("Tabular foundation models", "#9467bd"), "TabPFN": ("Tabular foundation models", "#9467bd"),
    "TabFM": ("Tabular foundation models", "#9467bd"),
}


def parse_ci(s: str) -> tuple[float, float]:
    lo, hi = s.replace("−", "-").split("–")
    return float(lo), float(hi)


def delta_forest(diffs_file: str = "step_21_benchmark_bootstrap_diffs_7feat.csv",
                 out_path: str | None = None, auprc_title: str = "ΔAUPRC vs LR (primary)") -> None:
    """Differences from LR in AUPRC and AUROC; also used for the 28-predictor supplementary figure."""
    diffs = pd.read_csv(diffs_file).set_index("Model").loc[ORDER]
    # Frames of the common size; model names on the left, family legend in a row below the frames
    fig, axes_grid = fs.axes_grid(1, 2, left=23.0, wgap=4.0, bottom=19.0)
    axes = axes_grid[0]
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    ypos = np.arange(len(ORDER))[::-1]
    for ax, metric, title in [(axes[0], "AUPRC", auprc_title),
                              (axes[1], "AUROC", "ΔAUROC vs LR")]:
        for y, model in zip(ypos, ORDER):
            point = diffs.loc[model, f"Delta_{metric}"]
            lo, hi = parse_ci(diffs.loc[model, f"Delta_{metric}_95CI"])
            color = FAMILY[model][1]
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.4)
            ax.plot(point, y, "o", color=color, markersize=4.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel(f"Difference in {metric} (95% CI)")
        ax.grid(axis="x", alpha=0.25)
        # Limits follow the data of each panel (not symmetric about zero): the CI range plus 10% on
        # each side, rounded outward to 0.01, always including zero; ticks every 0.05
        ends = np.array([parse_ci(s) for s in diffs[f"Delta_{metric}_95CI"]]).ravel()
        span = ends.max() - ends.min()
        lo = min(np.floor((ends.min() - 0.1 * span) / 0.01 - 1e-9) * 0.01, 0.0)
        hi = max(np.ceil((ends.max() + 0.1 * span) / 0.01 - 1e-9) * 0.01, 0.0)
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.arange(np.ceil(lo / 0.05 - 1e-9) * 0.05, hi + 1e-9, 0.05))
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([LABEL.get(m, m) for m in ORDER])
    handles = []
    for fam, color in dict(FAMILY.values()).items():
        handles.append(plt.Line2D([], [], color=color, marker="o", linewidth=1.4, markersize=4.5, label=fam))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=7.5)  # family legend below the frames: a little larger than legends inside axes
    fig.savefig(out_path or os.path.join(OUT_DIR, "fig_delta_forest.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def calibration_dca() -> None:
    pred = pd.read_csv("step_14_shrunk_predictions.csv")
    y, p = pred["y_true"].to_numpy(int), pred["y_prob"].to_numpy(float)
    ci = pd.read_csv("step_14_shrunk_bootstrap_ci.csv").set_index("Metric")
    fig, axes_grid = fs.axes_grid(1, 2)
    ax1, ax2 = axes_grid[0]

    # Calibration: quintiles of predicted risk, observed proportion with 95% Wilson interval
    groups = pd.qcut(p, 5, labels=False)
    for g in range(5):
        m = groups == g
        n, k = m.sum(), y[m].sum()
        phat = k / n
        z = 1.96
        centre = (phat + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
        ax1.errorbar(p[m].mean(), phat, yerr=[[phat - (centre - half)], [(centre + half) - phat]],
                     fmt="o", color=LR_COLOR, markersize=4.5, capsize=2, linewidth=1.1)
    lim = 0.45
    ax1.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.9)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xticks(np.arange(0, lim + 1e-9, 0.1))
    ax1.set_yticks(np.arange(0, lim + 1e-9, 0.1))
    ax1.set_xlabel("Predicted probability of PD-MCI")
    ax1.set_ylabel("Observed proportion")
    ax1.set_title("Calibration (test set)")
    slope, s_lo, s_hi = ci.loc["Calibration_Slope", ["Point_Estimate", "CI_Lower", "CI_Upper"]]
    icpt = ci.loc["Joint_Recalibration_Intercept", "Point_Estimate"]
    ax1.text(0.02, 0.43, f"Calibration slope {slope:.2f} (95% CI {s_lo:.2f}–{s_hi:.2f})\n"
             f"Calibration intercept {icpt:.2f}".replace("-", "−"), va="top", fontsize=fs.ANNOTATION_SIZE)
    ax1.grid(alpha=0.25)
    rug = ax1.inset_axes([0, 0, 1, 0.12], sharex=ax1)
    rug.hist(p, bins=40, range=(0, lim), color="#BBBBBB")
    rug.axis("off")

    # Decision curve
    thresholds = np.linspace(0.01, 0.50, 99)
    n = len(y)
    prev = y.mean()
    nb_model = [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in thresholds]
    nb_all = [prev - (1 - prev) * t / (1 - t) for t in thresholds]
    ax2.plot(thresholds, nb_model, color=LR_COLOR, linewidth=1.6, label="Final model")
    ax2.plot(thresholds, nb_all, color="grey", linewidth=1.0, linestyle="--", label="Treat all")
    ax2.axhline(0, color="black", linewidth=0.9, label="Treat none")
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-0.03, prev + 0.02)
    ax2.set_xlabel("Threshold probability")
    ax2.set_ylabel("Net benefit")
    ax2.set_title("Decision curve (test set)")
    # Default light frame, as for the other legends inside axes; text smaller than the axis labels
    ax2.legend(frameon=True, loc="upper right")
    ax2.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT_DIR, "fig_calibration_dca.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(OUT_DIR, exist_ok=True)
    delta_forest()
    calibration_dca()
    shutil.copyfile("step_18_nomogram.pdf", os.path.join(OUT_DIR, "fig_nomogram.pdf"))
    print(f"Saved manuscript figures to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
