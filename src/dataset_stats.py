#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for name, f in [("ASSISTments09", "assist09_with_ktstate.csv"),
                ("Junyi", "junyi_with_ktstate.csv")]:
    d = pd.read_csv(os.path.join(_DATA, f))
    print(f"\n### {name}  ({f})")
    print(f"  decision points N = {len(d)}")
    print(f"  unique students   = {d['user_id'].nunique()}")
    print(f"  unique skills     = {d['skill_id'].nunique()}")
    print(f"  treated (help) T=1 = {int((d['T']==1).sum())}  ({(d['T']==1).mean()*100:.1f}%)")
    print(f"  outcome base rate P(Y=1) = {d['Y'].mean():.3f}")
    print(f"  P(Y=1|T=1)={d.loc[d['T']==1,'Y'].mean():.3f}  P(Y=1|T=0)={d.loc[d['T']==0,'Y'].mean():.3f}")
