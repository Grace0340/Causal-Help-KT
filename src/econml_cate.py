#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EconML version (with honest confidence intervals):
  CausalForestDML estimates the ATE / CATE of help-seeking (T) on next-item
  success (Y).
  Design:
    W (confounding control) = full 64-d KT state
    X (heterogeneity axis)  = leading 16 PCs of the KT state (compressed to speed
                              up forest splitting while keeping the main signal)
  Reports: ATE [95% CI] / CATE distribution / share of units whose CI excludes 0
  (significant heterogeneity) / heterogeneity-driving dimensions.
Note: for a binary outcome, model_y fits P(Y=1) for residualization (standard DML).
"""
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from econml.dml import CausalForestDML
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
_FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(_FIG, exist_ok=True)

df = pd.read_csv(os.path.join(_DATA, "assist09_with_ktstate.csv"))
KT=[f'h{i}' for i in range(64)]
T = df['T'].values.astype(int); Y = df['Y'].values.astype(int)
W = StandardScaler().fit_transform(df[KT].values)          # confounding control: full KT state
X = PCA(n_components=16, random_state=0).fit_transform(W)   # heterogeneity axis: 16 PCs

print("Fitting CausalForestDML ...")
est = CausalForestDML(
    model_y=HistGradientBoostingRegressor(max_iter=120, max_depth=3),
    model_t=HistGradientBoostingClassifier(max_iter=120, max_depth=3),
    discrete_treatment=True, n_estimators=200, min_samples_leaf=40,
    max_samples=0.35, random_state=0, cv=4)
est.fit(Y, T, X=X, W=W)

ate = est.ate(X); lo,hi = est.ate_interval(X, alpha=0.05)
print("\n=== ATE (honest CI) ===")
print(f"  ATE = {ate:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]")

cate = est.effect(X)
clo,chi = est.effect_interval(X, alpha=0.05)
sig_neg = np.mean(chi < 0); sig_pos = np.mean(clo > 0)
print("\n=== CATE heterogeneity ===")
print(f"  mean {cate.mean():+.3f} | std {cate.std():.3f} | range [{cate.min():+.3f}, {cate.max():+.3f}]")
print(f"  units with 95%CI entirely <0 (significantly negative): {sig_neg*100:.1f}%")
print(f"  units with 95%CI entirely >0 (significantly positive): {sig_pos*100:.1f}%")

# heterogeneity-driving dimensions (which PCs most affect the effect variation)
try:
    imp = est.feature_importances_
    top = np.argsort(-imp)[:5]
    print("\n=== Heterogeneity drivers (top-5 PCs) ===")
    for r in top:
        print(f"  PC{r:02d}  importance={imp[r]:.3f}")
except Exception as ex:
    print("  (feature_importances unavailable:", ex, ")")

# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
ax[0].hist(cate,bins=50,color='#6b5b95',alpha=.85)
ax[0].axvline(0,color='k',ls='--',lw=1); ax[0].axvline(ate,color='#d64161',lw=1.5,label=f'ATE {ate:+.2f}')
ax[0].set_title("CausalForestDML CATE"); ax[0].set_xlabel("effect of help-seeking"); ax[0].legend(fontsize=8)

idx=np.argsort(cate); xs=np.arange(len(cate))
ax[1].fill_between(xs, clo[idx], chi[idx], color='#b8b0d0', alpha=.5, label='95% CI')
ax[1].plot(xs, cate[idx], color='#6b5b95', lw=1.2, label='CATE')
ax[1].axhline(0,color='k',ls='--',lw=1)
ax[1].set_title("Sorted CATE with honest CI"); ax[1].set_xlabel("students (sorted)"); ax[1].set_ylabel("effect"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "econml_cate.png"),dpi=140)
print("\nfigure saved: results/figures/econml_cate.png")
