#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlap fix + sensitivity analysis (sklearn-only):
  1. propensity overlap diagnostics and trimming (e in [0.05, 0.95])
  2. de-confounded ATE under several specifications: full sample / trimmed /
     prior-struggling subgroup (prior_correct == 0) + trimmed
     (the subgroup is defined by a strictly pre-treatment variable, avoiding
     collider bias)
  3. E-value sensitivity: how strong an unmeasured confounder would need to be
     to explain the effect away, with bootstrap confidence intervals; we report
     the E-value at the point estimate and at the confidence limit nearest null.
Outputs: console summary + one two-panel figure (overlap + specification forest).
"""
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
_FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(_FIG, exist_ok=True)

df = pd.read_csv(os.path.join(_DATA, "assist09_with_ktstate.csv"))
KT = [f'h{i}' for i in range(64)]
T = df['T'].values.astype(int); Y = df['Y'].values.astype(int)

# ---- cross-fitted nuisances (KT state) ----
def crossfit(Xcols, seed=0, nsplit=5):
    X = StandardScaler().fit_transform(df[Xcols].values)
    n=len(Y); e=np.zeros(n); mu0=np.zeros(n); mu1=np.zeros(n)
    for tr,te in StratifiedKFold(nsplit,shuffle=True,random_state=seed).split(X,T):
        e[te]=LogisticRegression(max_iter=2000).fit(X[tr],T[tr]).predict_proba(X[te])[:,1]
        i1=tr[T[tr]==1]; i0=tr[T[tr]==0]
        mu1[te]=GradientBoostingClassifier(n_estimators=100,max_depth=3).fit(X[i1],Y[i1]).predict_proba(X[te])[:,1]
        mu0[te]=GradientBoostingClassifier(n_estimators=100,max_depth=3).fit(X[i0],Y[i0]).predict_proba(X[te])[:,1]
    return e,mu0,mu1

print("Cross-fitting ...")
e,mu0,mu1 = crossfit(KT)

# ---- AIPW: returns ATE / RR / per-unit g-formula pseudo-values (for bootstrap) ----
def aipw_components(mask, clip=0.02):
    ec=np.clip(e,clip,1-clip)
    a1 = mu1 + T*(Y-mu1)/ec          # per-unit pseudo-value for E[Y(1)]
    a0 = mu0 + (1-T)*(Y-mu0)/(1-ec)  # E[Y(0)]
    a1,a0 = a1[mask], a0[mask]
    m1,m0 = a1.mean(), a0.mean()
    return m1,m0,(m1-m0),a1,a0

def evalue(rr):
    rr = rr if rr>=1 else 1/rr
    return rr + np.sqrt(rr*(rr-1))

def boot_rr(a1,a0,B=400,seed=1):
    rng=np.random.default_rng(seed); n=len(a1); rrs=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        m1,m0=a1[idx].mean(),a0[idx].mean()
        if m0>0: rrs.append(m1/m0)
    return np.percentile(rrs,[2.5,97.5])

# ---- three specifications ----
trim = (e>=0.05)&(e<=0.95)
strugg = (df['prior_correct'].values==0) & trim
specs = {
    "full sample":            np.ones(len(Y),bool),
    "overlap-trim [.05,.95]": trim,
    "struggling + trim":      strugg,
}
naive = df.loc[T==1,'Y'].mean()-df.loc[T==0,'Y'].mean()
print(f"\nnaive difference (biased): {naive:+.3f}\n")
print(f"overlap-trim kept: {trim.sum()}/{len(Y)}  (treated {T[trim].sum()}, control {trim.sum()-T[trim].sum()})")
print(f"prior-struggling subgroup: {strugg.sum()}  (treated {T[strugg].sum()})\n")

results=[]
print("=== De-confounded ATE + sensitivity per specification ===")
for name,mask in specs.items():
    m1,m0,ate,a1,a0 = aipw_components(mask)
    se = np.sqrt((a1-a0).var()/mask.sum())
    rr = m1/m0
    lo,hi = boot_rr(a1,a0)
    # confidence limit nearest null (take upper limit hi when RR<1)
    near = hi if rr<1 else lo
    ev_pt, ev_ci = evalue(rr), evalue(near)
    results.append((name,ate,se,rr,ev_pt,ev_ci))
    print(f"  [{name}]")
    print(f"     ATE={ate:+.3f} (SE {se:.3f}) | RR={rr:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"     E-value point={ev_pt:.2f} | near-null CI limit={ev_ci:.2f}")

print("\nReading: a larger E-value is more robust. An E-value of X means an")
print("unmeasured confounder associated with both help-seeking and next-item")
print("success at risk-ratio strength ~X would be needed to explain the effect away.")

# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].hist(e[T==1],bins=40,alpha=.6,density=True,label='treated (help)',color='#d64161')
ax[0].hist(e[T==0],bins=40,alpha=.6,density=True,label='control',color='#6b5b95')
ax[0].axvline(.05,color='k',ls='--',lw=1); ax[0].axvline(.95,color='k',ls='--',lw=1)
ax[0].set_title("Propensity overlap (dashed = trim bounds)")
ax[0].set_xlabel("estimated propensity e(x)"); ax[0].legend(fontsize=8)

ys=np.arange(len(results))[::-1]
for y,(name,ate,se,rr,evp,evc) in zip(ys,results):
    ax[1].errorbar(ate,y,xerr=1.96*se,fmt='o',color='#6b5b95',capsize=4)
    ax[1].annotate(f"E={evp:.2f}",(ate,y),textcoords="offset points",xytext=(6,6),fontsize=8)
ax[1].axvline(naive,color='#d64161',ls=':',lw=1.2,label=f'naive {naive:+.2f}')
ax[1].axvline(0,color='k',lw=.8)
ax[1].set_yticks(ys); ax[1].set_yticklabels([r[0] for r in results],fontsize=9)
ax[1].set_title("De-confounded ATE across specifications"); ax[1].set_xlabel("ATE (risk difference)")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "sensitivity_analysis.png"),dpi=140)
print("\nfigure saved: results/figures/sensitivity_analysis.png")
