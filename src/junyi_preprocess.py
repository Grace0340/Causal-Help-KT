#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Junyi (subsample of Log_Problem.csv) -> two tables with the same schema as ASSISTments.
Column mapping:
  uuid -> user_id (int) | ucid -> skill_id (int) | timestamp_TW -> time order
  is_correct -> correct | is_hint_used -> treatment (help) | total_attempt_cnt -> attempt
  used_hint_cnt -> hint_count | total_sec_taken -> response time
Design matches the ASSISTments version (single decision point, 2nd same-KC
interaction, Y = next item, X strictly pre-treatment).
Outputs: junyi_sequences.csv / junyi_decision_points.csv
Then run, e.g.:
  python src/train_dkt_extract_fast.py --seq data/junyi_sequences.csv \
      --decisions data/junyi_decision_points.csv --out data/junyi_with_ktstate.csv --device cuda
"""
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

# place the raw/subsampled data under data/ (see data/README.md)
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = os.path.join(_DATA, "junyi_sample.csv")
OUT_SEQ = os.path.join(_DATA, "junyi_sequences.csv")
OUT_TAB = os.path.join(_DATA, "junyi_decision_points.csv")

df = pd.read_csv(SRC, low_memory=False)
df["correct"] = df["is_correct"].astype(int)
df["first_action"] = df["is_hint_used"].astype(int)        # 1 = help, 0 = no
df["user_id"] = pd.factorize(df["uuid"])[0]
df["skill_id"] = pd.factorize(df["ucid"])[0]
df["ts"] = pd.to_datetime(df["timestamp_TW"], errors="coerce")
df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)
df["order_id"] = np.arange(len(df))                         # global unique time-order key
df["opportunity"] = df.groupby(["user_id", "skill_id"]).cumcount()  # prior practice count on this KC

print(f"rows {len(df)} | students {df.user_id.nunique()} | skills (ucid) {df.skill_id.nunique()}")
print(f"help share {df.first_action.mean()*100:.1f}% | overall accuracy {df.correct.mean():.3f}")

# skill difficulty
skill_diff = (1 - df.groupby("skill_id")["correct"].mean()).rename("skill_difficulty")

# sequence table (for DKT)
seq = df[["user_id","skill_id","order_id","correct","first_action",
          "total_attempt_cnt","used_hint_cnt","total_sec_taken","opportunity"]].rename(
          columns={"total_attempt_cnt":"attempt_count","used_hint_cnt":"hint_count",
                   "total_sec_taken":"ms_first_response"})
seq.to_csv(OUT_SEQ, index=False)

# single decision points (T/Y/X)
rows=[]
for (uid,sid),g in df.groupby(["user_id","skill_id"], sort=False):
    if len(g)<3: continue
    g=g.reset_index(drop=True)
    prev,dec,nxt = g.iloc[0],g.iloc[1],g.iloc[2]
    rows.append(dict(
        user_id=uid, skill_id=sid,
        T=int(dec["first_action"]==1), Y=int(nxt["correct"]),
        prior_correct=int(prev["correct"]),
        prior_attempt=float(prev["total_attempt_cnt"]),
        prior_hint=float(prev["used_hint_cnt"]),
        prior_log_ms=float(np.log1p(max(prev["total_sec_taken"],0))),
        opportunity=float(dec["opportunity"]),
        skill_difficulty=float(skill_diff.loc[sid]),
        decision_order_id=int(dec["order_id"])))
tab=pd.DataFrame(rows); tab.to_csv(OUT_TAB,index=False)

# diagnostics
n1=(tab["T"]==1).sum(); n0=(tab["T"]==0).sum()
y1=tab.loc[tab["T"]==1,"Y"].mean(); y0=tab.loc[tab["T"]==0,"Y"].mean()
print(f"\ndecision points {len(tab)} | treated {n1} control {n0} | treated share {n1/len(tab)*100:.1f}%")
print(f"naive E[Y|T=1]={y1:.3f} E[Y|T=0]={y0:.3f} naive diff={y1-y0:+.3f}")
print(f"\noutput: {OUT_SEQ}\n        {OUT_TAB}")
