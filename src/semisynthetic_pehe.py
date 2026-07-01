#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-synthetic benchmark (RQ1 with known ground truth):
  Assumption: the true heterogeneous effect tau(x) is driven by the latent
  knowledge state (we build z from the leading principal components of the KT
  state). With tau_true known, we compare the sqrt(PEHE) of a T-learner using
  three covariate sets:
    - KT state (64-d, rich knowledge-state representation)
    - handcrafted features (6-d, information-poor view)
    - oracle (the true latent z, an upper bound on performance)
  If KT << handcrafted, it supports "the knowledge-state representation recovers
  individual effects better".
Note: this DGP assumes heterogeneity comes from the knowledge state, which the
paper states explicitly; the real-data Qini provides complementary evidence.
"""
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
_FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(_FIG, exist_ok=True)

df = pd.read_csv(os.path.join(_DATA, "assist09_with_ktstate.csv"))
KT=[f'h{i}' for i in range(64)]
HAND=['prior_correct','prior_attempt','prior_hint','prior_log_ms','opportunity','skill_difficulty']
Xkt = StandardScaler().fit_transform(df[KT].values)
Xhand = StandardScaler().fit_transform(df[HAND].values)
rng = np.random.default_rng(0)
def sig(x): return 1/(1+np.exp(-x))

# ---- latent ability z (leading two PCs of the KT state) ----
pcs = PCA(n_components=2, random_state=0).fit_transform(Xkt)
z1 = StandardScaler().fit_transform(pcs[:,[0]]).ravel()
z2 = StandardScaler().fit_transform(pcs[:,[1]]).ravel()

# ---- data-generating process ----
e   = sig(-1.6 - 1.0*z1)                 # confounding: lower ability is likelier treated
Ts  = (rng.random(len(z1)) < e).astype(int)
mu0 = sig(0.8 + 0.9*z1 + 0.3*z2)         # baseline success rises with ability
tau = 0.10 + 0.12*z1 - 0.10*(z1*z2)      # nonlinear heterogeneous effect
mu1 = np.clip(mu0 + tau, 0.01, 0.99)
tau_true = mu1 - mu0
Y0 = (rng.random(len(z1)) < mu0).astype(int)
Y1 = (rng.random(len(z1)) < mu1).astype(int)
Ys = np.where(Ts==1, Y1, Y0)
print(f"synthetic treated share {Ts.mean()*100:.1f}% | true ATE {tau_true.mean():+.3f} | tau range [{tau_true.min():+.3f},{tau_true.max():+.3f}]")

# ---- T-learner (cross-fitted) ----
def tlearner(X, seed=0):
    n=len(Ys); m1=np.zeros(n); m0=np.zeros(n)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(X):
        i1=tr[Ts[tr]==1]; i0=tr[Ts[tr]==0]
        m1[te]=HistGradientBoostingRegressor(max_iter=120,max_depth=3).fit(X[i1],Ys[i1]).predict(X[te])
        m0[te]=HistGradientBoostingRegressor(max_iter=120,max_depth=3).fit(X[i0],Ys[i0]).predict(X[te])
    return m1-m0

print("Fitting T-learners (KT / handcrafted / oracle) ...")
tau_kt = tlearner(Xkt)
tau_hd = tlearner(Xhand)
tau_or = tlearner(np.c_[z1,z2])

pehe = lambda th: np.sqrt(np.mean((th-tau_true)**2))
aerr = lambda th: abs(th.mean()-tau_true.mean())
res = [("KT state (64)",tau_kt),("handcrafted (6)",tau_hd),("oracle (true z)",tau_or)]
print("\n=== sqrt(PEHE) (lower is more accurate) ===")
for nm,th in res:
    print(f"  {nm:16s}  sqrt(PEHE)={pehe(th):.4f}   |ATE error|={aerr(th):.4f}")
imp=(pehe(tau_hd)-pehe(tau_kt))/pehe(tau_hd)*100
print(f"\nKT state vs handcrafted: sqrt(PEHE) reduced by {imp:.1f}%  -> supports RQ1")

# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
names=[r[0] for r in res]; vals=[pehe(r[1]) for r in res]
ax[0].bar(names,vals,color=['#6b5b95','#feb236','#88b04b'])
ax[0].set_ylabel("sqrt(PEHE)"); ax[0].set_title("CATE estimation error (lower=better)")
for i,v in enumerate(vals): ax[0].annotate(f"{v:.3f}",(i,v),ha='center',va='bottom',fontsize=9)
s=rng.choice(len(tau_true),3000,replace=False)
ax[1].scatter(tau_true[s],tau_hd[s],s=4,alpha=.3,color='#feb236',label='handcrafted')
ax[1].scatter(tau_true[s],tau_kt[s],s=4,alpha=.3,color='#6b5b95',label='KT-state')
lim=[tau_true.min(),tau_true.max()]; ax[1].plot(lim,lim,'k--',lw=1)
ax[1].set_xlabel("true tau(x)"); ax[1].set_ylabel("estimated tau_hat(x)")
ax[1].set_title("Recovered vs true effect"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "semisynthetic_pehe.png"),dpi=140)
print("figure saved: results/figures/semisynthetic_pehe.png")
