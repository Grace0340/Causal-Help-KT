#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal analysis on real data (sklearn-only, no EconML required):
  - propensity score + overlap diagnostics
  - de-confounded ATE (AIPW/DR, cross-fitted): naive vs KT-state vs handcrafted
  - heterogeneous effects CATE (T-learner on KT state): distribution / negative
    share / subgroup "who benefits"
  - RQ1 real-data signal: does the KT-state or handcrafted CATE ranking target
    help better (DR-Qini)
Outputs: console summary + one three-panel figure.
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
HAND = ['prior_correct','prior_attempt','prior_hint','prior_log_ms','opportunity','skill_difficulty']
KT   = [f'h{i}' for i in range(64)]
T = df['T'].values.astype(int); Y = df['Y'].values.astype(int)

def crossfit(Xcols, seed=0, nsplit=5):
    X = StandardScaler().fit_transform(df[Xcols].values)
    n=len(Y); e=np.zeros(n); mu0=np.zeros(n); mu1=np.zeros(n)
    skf=StratifiedKFold(nsplit, shuffle=True, random_state=seed)
    for tr,te in skf.split(X,T):
        ps=LogisticRegression(max_iter=2000).fit(X[tr],T[tr])
        e[te]=ps.predict_proba(X[te])[:,1]
        i1=tr[T[tr]==1]; i0=tr[T[tr]==0]
        m1=GradientBoostingClassifier(n_estimators=100,max_depth=3).fit(X[i1],Y[i1])
        m0=GradientBoostingClassifier(n_estimators=100,max_depth=3).fit(X[i0],Y[i0])
        mu1[te]=m1.predict_proba(X[te])[:,1]; mu0[te]=m0.predict_proba(X[te])[:,1]
    return e,mu0,mu1

def aipw(e,mu0,mu1,clip=0.02):
    e=np.clip(e,clip,1-clip)
    psi=(mu1-mu0)+T*(Y-mu1)/e-(1-T)*(Y-mu0)/(1-e)
    return psi.mean(), psi.std()/np.sqrt(len(psi)), psi

print("Cross-fitting nuisances on KT state ...")
e_kt,mu0_kt,mu1_kt = crossfit(KT)
print("Cross-fitting nuisances on handcrafted features ...")
e_h,mu0_h,mu1_h = crossfit(HAND)

naive = df.loc[T==1,'Y'].mean()-df.loc[T==0,'Y'].mean()
ate_kt,se_kt,phi = aipw(e_kt,mu0_kt,mu1_kt)
ate_h ,se_h ,_   = aipw(e_h ,mu0_h ,mu1_h)

print("\n=== ATE: naive vs de-confounded ===")
print(f"  naive (biased)        : {naive:+.3f}")
print(f"  AIPW handcrafted      : {ate_h:+.3f}  (SE {se_h:.3f})")
print(f"  AIPW KT state         : {ate_kt:+.3f}  (SE {se_kt:.3f})")

# overlap
print("\n=== Overlap (KT propensity) ===")
print(f"  e range [{e_kt.min():.3f}, {e_kt.max():.3f}] | e>0.95: {np.mean(e_kt>0.95)*100:.2f}%  e<0.05: {np.mean(e_kt<0.05)*100:.1f}%")

# CATE (T-learner)
tau_kt = mu1_kt - mu0_kt
tau_h  = mu1_h  - mu0_h
print("\n=== Heterogeneous effects CATE (KT-state T-learner) ===")
print(f"  mean {tau_kt.mean():+.3f} | std {tau_kt.std():.3f} | range [{tau_kt.min():+.3f}, {tau_kt.max():+.3f}]")
print(f"  share predicted negative (help harmful): {np.mean(tau_kt<0)*100:.1f}%")
print(f"  share predicted positive (help helpful): {np.mean(tau_kt>0)*100:.1f}%")

# Subgroups: baseline mastery (mu0, continuous) / skill difficulty, split into tertiles
def terc(x):
    q=np.quantile(x,[1/3,2/3])
    if q[0]==q[1]:  # degenerate discrete variable -> split at median
        return (x>np.median(x)).astype(int)*2
    return np.digitize(x,q)
ability = mu0_kt  # predicted control success rate = baseline-mastery proxy
print("\n=== Who benefits (subgroup mean CATE) ===")
sub = [("baseline mastery", ability), ("skill difficulty", df['skill_difficulty'].values)]
for name,vals in sub:
    g=terc(vals); means=[tau_kt[g==k].mean() for k in range(3)]
    print(f"  {name:18s} low/mid/high: {means[0]:+.3f} / {means[1]:+.3f} / {means[2]:+.3f}")

# RQ1: DR-Qini curve using the common KT nuisance phi, comparing the two rankings
def qini_curve(rank, phi):
    order=np.argsort(-rank); cum=np.cumsum(phi[order])/len(phi)
    frac=np.arange(1,len(phi)+1)/len(phi)
    return frac, cum
fk,ck = qini_curve(tau_kt, phi)
fh,chd= qini_curve(tau_h , phi)
rand = fk*phi.mean()
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
def auuc(frac,cum,rand): return _trapz(cum-rand,frac)
print("\n=== RQ1: targeting ability of the CATE ranking (DR-Qini, higher is better) ===")
print(f"  KT-state ranking    AUUC = {auuc(fk,ck,rand):+.4f}")
print(f"  handcrafted ranking AUUC = {auuc(fh,chd,rand):+.4f}")

# ---- figure ----
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].hist(tau_kt,bins=50,color='#6b5b95',alpha=.85)
ax[0].axvline(0,color='k',lw=1,ls='--'); ax[0].axvline(tau_kt.mean(),color='#d64161',lw=1.5)
ax[0].set_title("CATE distribution (KT-state)"); ax[0].set_xlabel("estimated individual effect of help-seeking")

labels=['low','mid','high']
xa=np.arange(3); w=.38
for j,(nm,vals,cc) in enumerate([("baseline mastery",ability,'#6b5b95'),
                                 ("skill difficulty",df['skill_difficulty'].values,'#feb236')]):
    g=terc(vals); m=[tau_kt[g==k].mean() for k in range(3)]
    ax[1].bar(xa+(j-0.5)*w,m,w,label=nm,color=cc)
ax[1].axhline(0,color='k',lw=.8); ax[1].set_xticks(xa); ax[1].set_xticklabels(labels)
ax[1].set_title("Who benefits (subgroup mean CATE)"); ax[1].legend(fontsize=8)

ax[2].plot(fk,ck,label='KT-state ranking',color='#6b5b95',lw=2)
ax[2].plot(fh,chd,label='handcrafted ranking',color='#feb236',lw=2)
ax[2].plot(fk,rand,label='random',color='gray',ls='--',lw=1)
ax[2].set_title("DR-Qini (targeting ability)"); ax[2].set_xlabel("fraction targeted"); ax[2].set_ylabel("cumulative gain"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "cate_analysis.png"),dpi=140)
print("\nfigure saved: results/figures/cate_analysis.png")
