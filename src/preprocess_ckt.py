#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASSISTments 2009-2010 skill-builder -> single-decision-point causal table (T / Y / X)

Design (cross-sectional single decision point, avoids time-varying confounding):
  unit          = one decision point per (user_id, skill_id) sequence
  decision point= the 2nd interaction on that skill (index 1); sequence length >= 3
                  (fixing the position makes unit selection independent of T)
  treatment T   = whether help was requested first, first_action in {1,2}
                  (1 = hint, 2 = scaffold)
  outcome Y     = whether the next same-skill item (index 2) is answered correctly
                  on the first attempt (avoids leakage)
  covariates X  = strictly pre-treatment features (from index 0) + stable
                  attributes; never the decision point's own post-treatment info
                  (its attempt/hint/response time are consequences of the treatment)
Notes:
  - rows with missing skill_id are dropped; only original==1 items are kept
    (removing scaffold sub-items, standard KT practice)
  - one order_id split across multiple skills becomes multiple rows -> building
    sequences per (user, skill) handles this naturally
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# place the raw dataset under data/ (public dataset; see data/README.md)
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = os.path.join(_DATA, "skill_builder_data.csv")
OUT_TABLE = os.path.join(_DATA, "assist09_decision_points.csv")
OUT_SEQ   = os.path.join(_DATA, "assist09_sequences.csv")

# ---------- 1. load + clean ----------
df = pd.read_csv(SRC, encoding="ISO-8859-15", low_memory=False)
df = df.dropna(subset=["skill_id"])          # drop rows without a skill tag (~12.6%)
df = df[df["original"] == 1].copy()          # original items only
df["correct"] = df["correct"].astype(int)
df["first_action"] = df["first_action"].astype(int)
df["skill_id"] = df["skill_id"].astype(int)
df = df.sort_values(["user_id", "skill_id", "order_id"]).reset_index(drop=True)

# global skill difficulty (stable covariate): 1 - mean correctness of the skill
skill_diff = (1 - df.groupby("skill_id")["correct"].mean()).rename("skill_difficulty")

# ---------- 2. save the cleaned interaction sequences (for KT pretraining) ----------
seq_cols = ["user_id", "skill_id", "order_id", "correct", "first_action",
            "attempt_count", "hint_count", "ms_first_response", "opportunity"]
df[seq_cols].to_csv(OUT_SEQ, index=False)

# ---------- 3. build single decision points (T/Y/X) ----------
rows = []
for (uid, sid), g in df.groupby(["user_id", "skill_id"], sort=False):
    if len(g) < 3:
        continue
    g = g.reset_index(drop=True)
    prev = g.iloc[0]   # index 0: before the decision (supplies pre-treatment X)
    dec  = g.iloc[1]   # index 1: the decision point (defines T)
    nxt  = g.iloc[2]   # index 2: the outcome Y

    T = int(dec["first_action"] in (1, 2))          # treatment: help requested first
    Y = int(nxt["correct"])                          # outcome: next item correct on first try
    rows.append({
        "user_id": uid,
        "skill_id": sid,
        "T": T,
        "Y": Y,
        # X: strictly pre-treatment (from index 0) ----------
        "prior_correct":  int(prev["correct"]),
        "prior_attempt":  float(prev["attempt_count"]),
        "prior_hint":     float(prev["hint_count"]),
        "prior_log_ms":   float(np.log1p(max(prev["ms_first_response"], 0))),
        # the decision point's "opportunity" counts prior practice -> pre-treatment, usable
        "opportunity":    float(dec["opportunity"]),
        "skill_difficulty": float(skill_diff.loc[sid]),
        "decision_order_id": int(dec["order_id"]),   # for later alignment with the KT state
    })

tab = pd.DataFrame(rows)
tab.to_csv(OUT_TABLE, index=False)

# ---------- 4. diagnostics ----------
print("=" * 56)
print(f"total decision points (unit = user x skill, length>=3): {len(tab)}")
n1, n0 = (tab["T"] == 1).sum(), (tab["T"] == 0).sum()
print(f"treated T=1 (help first): {n1}   control T=0: {n0}   treated share: {n1/len(tab)*100:.1f}%")
print()

# naive difference (unadjusted for confounding -> exactly what the causal method corrects)
y1, y0 = tab.loc[tab["T"] == 1, "Y"].mean(), tab.loc[tab["T"] == 0, "Y"].mean()
print(f"naive E[Y|T=1]={y1:.3f}   E[Y|T=0]={y0:.3f}   naive diff={y1-y0:+.3f}")
print("  (note: this is a confounded association, not a causal effect; weaker")
print("   students seek help more, biasing it downward)")
print()

# standardized mean differences (SMD) across arms -> evidence of confounding
print("Covariate balance (|SMD|>0.1 is imbalanced, i.e. causal adjustment is needed):")
covs = ["prior_correct", "prior_attempt", "prior_hint",
        "prior_log_ms", "opportunity", "skill_difficulty"]
for c in covs:
    a, b = tab.loc[tab["T"] == 1, c], tab.loc[tab["T"] == 0, c]
    sd = np.sqrt((a.var() + b.var()) / 2)
    smd = (a.mean() - b.mean()) / sd if sd > 0 else 0
    flag = "  <-- imbalanced" if abs(smd) > 0.1 else ""
    print(f"  {c:18s} SMD={smd:+.3f}{flag}")

print()
print("output files:")
print(" ", OUT_TABLE, "(decision-point table: T/Y/X)")
print(" ", OUT_SEQ,   "(interaction sequences: for KT pretraining and state extraction)")
