#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Junyi replication: ATE / CATE / DR-Qini / semi-synthetic PEHE / sensitivity,
mirroring the ASSISTments pipeline on the larger, more skill-diverse Junyi data.
Outputs: console summary + one two-panel figure (DR-Qini + semi-synthetic PEHE).
"""
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")
_FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(_FIG, exist_ok=True)

df = pd.read_csv(os.path.join(_DATA, "junyi_with_ktstate.csv"))
KT=[f'h{i}' for i in range(64)]
HAND=['prior_correct','prior_attempt','prior_hint','prior_log_ms','opportunity','skill_difficulty']
T=df['T'].values.astype(int); Y=df['Y'].values.astype(int)

def crossfit(cols,seed=0):
    X=StandardScaler().fit_transform(df[cols].values); n=len(Y)
    e=np.zeros(n);m0=np.zeros(n);m1=np.zeros(n)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,T):
        e[te]=LogisticRegression(max_iter=2000).fit(X[tr],T[tr]).predict_proba(X[te])[:,1]
        i1=tr[T[tr]==1];i0=tr[T[tr]==0]
        m1[te]=HistGradientBoostingClassifier(max_iter=120,max_depth=3).fit(X[i1],Y[i1]).predict_proba(X[te])[:,1]
        m0[te]=HistGradientBoostingClassifier(max_iter=120,max_depth=3).fit(X[i0],Y[i0]).predict_proba(X[te])[:,1]
    return e,m0,m1
def aipw(e,m0,m1,mask=None,clip=0.02):
    if mask is None: mask=np.ones(len(Y),bool)
    ec=np.clip(e,clip,1-clip)
    a1=m1+T*(Y-m1)/ec; a0=m0+(1-T)*(Y-m0)/(1-ec)
    return (a1-a0)[mask].mean(),a1,a0
def evalue(rr):
    rr=rr if rr>=1 else 1/rr; return rr+np.sqrt(rr*(rr-1))

print("Cross-fitting ..."); e,m0,m1=crossfit(KT); eh,m0h,m1h=crossfit(HAND)
naive=df.loc[T==1,'Y'].mean()-df.loc[T==0,'Y'].mean()
ate,a1,a0=aipw(e,m0,m1); ateh,_,_=aipw(eh,m0h,m1h)
tau=m1-m0; tauh=m1h-m0h
print("\n== ATE ==")
print(f"  naive {naive:+.3f} | AIPW handcrafted {ateh:+.3f} | AIPW-KT {ate:+.3f}")
print(f"  overlap: e<0.05 share {np.mean(e<0.05)*100:.1f}%")
print(f"  CATE(KT): mean {tau.mean():+.3f} std {tau.std():.3f} negative share {np.mean(tau<0)*100:.1f}%")

# Qini
phi=a1-a0
def qini(rank):
    o=np.argsort(-rank); cum=np.cumsum(phi[o])/len(phi); fr=np.arange(1,len(phi)+1)/len(phi)
    return fr,cum
_trapz=getattr(np,"trapezoid",getattr(np,"trapz",None))
fk,ck=qini(tau); fh,ch=qini(tauh); rand=fk*phi.mean()
print(f"\n== RQ1 Qini (AUUC) ==\n  KT {_trapz(ck-rand,fk):+.4f} | handcrafted {_trapz(ch-rand,fh):+.4f}")

# Sensitivity: trimming + struggling subgroup + E-value
print("\n== Sensitivity ==")
for nm,mask in [("full sample",np.ones(len(Y),bool)),
                ("overlap-trim",(e>=0.05)&(e<=0.95)),
                ("struggling+trim",(df['prior_correct'].values==0)&(e>=0.05)&(e<=0.95))]:
    at,aa1,aa0=aipw(e,m0,m1,mask); mm1,mm0=aa1[mask].mean(),aa0[mask].mean(); rr=mm1/mm0
    print(f"  {nm:16s} ATE {at:+.3f} | RR {rr:.3f} | E-value {evalue(rr):.2f} | n={mask.sum()}")

# Semi-synthetic PEHE
print("\n== Semi-synthetic PEHE ==")
rng=np.random.default_rng(0)
Xkt=StandardScaler().fit_transform(df[KT].values); Xh=StandardScaler().fit_transform(df[HAND].values)
pcs=PCA(2,random_state=0).fit_transform(Xkt)
z1=StandardScaler().fit_transform(pcs[:,[0]]).ravel(); z2=StandardScaler().fit_transform(pcs[:,[1]]).ravel()
es=1/(1+np.exp(-(-1.6-1.0*z1))); Ts=(rng.random(len(z1))<es).astype(int)
mu0=1/(1+np.exp(-(0.8+0.9*z1+0.3*z2))); taut=0.10+0.12*z1-0.10*(z1*z2)
mu1=np.clip(mu0+taut,0.01,0.99); tt=mu1-mu0
Y0=(rng.random(len(z1))<mu0).astype(int); Y1=(rng.random(len(z1))<mu1).astype(int); Ys=np.where(Ts==1,Y1,Y0)
def tlearn(X):
    n=len(Ys);p1=np.zeros(n);p0=np.zeros(n)
    for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
        i1=tr[Ts[tr]==1];i0=tr[Ts[tr]==0]
        p1[te]=HistGradientBoostingRegressor(max_iter=120,max_depth=3).fit(X[i1],Ys[i1]).predict(X[te])
        p0[te]=HistGradientBoostingRegressor(max_iter=120,max_depth=3).fit(X[i0],Ys[i0]).predict(X[te])
    return p1-p0
pehe=lambda th:np.sqrt(np.mean((th-tt)**2))
pk=tlearn(Xkt); ph=tlearn(Xh); po=tlearn(np.c_[z1,z2])
print(f"  KT {pehe(pk):.4f} | handcrafted {pehe(ph):.4f} | oracle {pehe(po):.4f}")
print(f"  KT vs handcrafted: reduced by {(pehe(ph)-pehe(pk))/pehe(ph)*100:.1f}%")

# figure: Qini + PEHE
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
ax[0].plot(fk,ck,label='KT-state',color='#6b5b95',lw=2); ax[0].plot(fh,ch,label='handcrafted',color='#feb236',lw=2)
ax[0].plot(fk,rand,'--',color='gray',lw=1,label='random'); ax[0].set_title("Junyi DR-Qini"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("fraction targeted"); ax[0].set_ylabel("cumulative gain")
ax[1].bar(['KT(64)','hand(6)','oracle'],[pehe(pk),pehe(ph),pehe(po)],color=['#6b5b95','#feb236','#88b04b'])
ax[1].set_ylabel("sqrt(PEHE)"); ax[1].set_title("Junyi semi-synthetic PEHE")
for i,v in enumerate([pehe(pk),pehe(ph),pehe(po)]): ax[1].annotate(f"{v:.3f}",(i,v),ha='center',va='bottom',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(_FIG, "junyi_replication.png"),dpi=140)
print("\nfigure: results/figures/junyi_replication.png")
