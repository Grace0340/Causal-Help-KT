#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the clean, publication-quality paper figures directly from the already
computed summary numbers (no recomputation, <5s). Output (300 dpi):
    fig_pehe.png     RQ1: semi-synthetic sqrt(PEHE), KT vs handcrafted vs oracle, both datasets
    fig_forest.png   RQ2/RQ3: de-confounded ATE forest + E-value, two datasets, three specs
    fig_hetero.png   heterogeneity: ASSISTments subgroup mean CATE + negative share
All labels are in English.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
os.makedirs(_FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

PURPLE, ORANGE, GREEN, RED, GREY = "#5b4b8a", "#e08a2f", "#4a9e6b", "#c0392b", "#7f8c8d"

# ---------------- Fig 1: semi-synthetic sqrt(PEHE) ----------------
pehe = {  # dataset -> (KT64, handcrafted6, oracle)
    "ASSISTments09": (0.0844, 0.1542, 0.0601),
    "Junyi":         (0.0677, 0.1152, 0.0327),
}
red = {"ASSISTments09": 45.2, "Junyi": 41.3}  # % reduction KT vs handcrafted
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (name, vals) in zip(axes, pehe.items()):
    labels = ["KT-state\n(64-d)", "Handcrafted\n(6-d)", "Oracle\n(latent z)"]
    bars = ax.bar(labels, vals, color=[PURPLE, ORANGE, GREEN], width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{name}  (KT -{red[name]:.0f}% vs handcrafted)", fontsize=9.5)
    ax.set_ylabel(r"$\sqrt{\mathrm{PEHE}}$  (lower is better)")
    ax.set_ylim(0, max(vals) * 1.28)
fig.tight_layout()
fig.savefig(os.path.join(_FIG, "fig_pehe.png"))
plt.close(fig)

# ---------------- Fig 2: de-confounded ATE forest + E-value ----------------
# (label, ATE, SE, E-value point)
rows = [
    ("ASSISTments09  full",         -0.274, 0.014, 2.68),
    ("ASSISTments09  overlap-trim", -0.302, 0.012, 3.22),
    ("ASSISTments09  struggling",   -0.287, 0.016, 3.45),
    ("Junyi  full",                 -0.229, 0.006, 2.16),
    ("Junyi  struggling",           -0.338, 0.010, 3.29),
]
naive = {"ASSISTments09": -0.441, "Junyi": -0.243}
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ys = np.arange(len(rows))[::-1]
for y, (lab, ate, se, ev) in zip(ys, rows):
    ax.errorbar(ate, y, xerr=1.96 * se, fmt="o", color=PURPLE, capsize=4, ms=6)
    ax.annotate(f"E={ev:.2f}", (ate, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=GREY)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(naive["ASSISTments09"], color=RED, ls=":", lw=1.1,
           label=f'naive (ASSIST.) {naive["ASSISTments09"]:+.2f}')
ax.axvline(naive["Junyi"], color=ORANGE, ls=":", lw=1.1,
           label=f'naive (Junyi) {naive["Junyi"]:+.2f}')
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlim(-0.49, 0.04)
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_xlabel("De-confounded ATE on next-item success (risk difference)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(_FIG, "fig_forest.png"))
plt.close(fig)

# ---------------- Fig 3: heterogeneity (subgroup CATE, ASSISTments) ----------------
mastery = [-0.299, -0.282, -0.253]
diff = [-0.294, -0.281, -0.261]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, mastery, w, color=PURPLE, label="by baseline mastery")
ax.bar(x + w / 2, diff, w, color=ORANGE, label="by skill difficulty")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(["low", "mid", "high"])
ax.set_ylabel("subgroup mean CATE")
ax.set_title("Who is hurt least (ASSISTments09)", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(min(mastery) * 1.15, 0.02)
fig.tight_layout()
fig.savefig(os.path.join(_FIG, "fig_hetero.png"))
plt.close(fig)

print("saved to results/figures/: fig_pehe.png, fig_forest.png, fig_hetero.png")
