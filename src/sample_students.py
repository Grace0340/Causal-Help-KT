#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subsample a very large log file (e.g. Junyi Log_Problem.csv, tens of millions of
rows) by student, producing a smaller, uploadable subset that keeps every
interaction of the sampled students (complete sequences). Low-memory: reads in
chunks, never loading the whole table at once.

Usage:
  python src/sample_students.py --src data/Log_Problem.csv --out data/junyi_sample.csv \
         --id-col uuid --n-students 5000

Note: we sample "students", not "rows", so each student's full sequence is
preserved (both KT and the causal design need intact sequences).
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="dataset_sample.csv")
ap.add_argument("--id-col", default="uuid", help="student id column (Junyi = uuid)")
ap.add_argument("--n-students", type=int, default=5000)
ap.add_argument("--chunk", type=int, default=1_000_000)
ap.add_argument("--encoding", default="utf-8")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# Pass 1: collect all student ids
print("[1/2] scanning student ids ...")
ids = set()
for ch in pd.read_csv(args.src, usecols=[args.id_col], chunksize=args.chunk,
                      encoding=args.encoding, low_memory=False):
    ids.update(ch[args.id_col].dropna().unique().tolist())
ids = np.array(sorted(ids))
print(f"      {len(ids)} students total")

rng = np.random.default_rng(args.seed)
keep = set(rng.choice(ids, size=min(args.n_students, len(ids)), replace=False).tolist())
print(f"      sampled {len(keep)}")

# Pass 2: write out all rows of the selected students
print("[2/2] writing subset ...")
first = True; n_rows = 0
for ch in pd.read_csv(args.src, chunksize=args.chunk, encoding=args.encoding, low_memory=False):
    sub = ch[ch[args.id_col].isin(keep)]
    if len(sub):
        sub.to_csv(args.out, mode="w" if first else "a", header=first, index=False)
        first = False; n_rows += len(sub)
print(f"done: {args.out}  ({n_rows} rows, {len(keep)} students)")
