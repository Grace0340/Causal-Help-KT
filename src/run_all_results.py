#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce all results reported in the paper in one command (CPU-only, sklearn).

Prerequisite: the processed data are under the repository's data/ folder:
    data/assist09_with_ktstate.csv   (T/Y/handcrafted features/h0..h63)
    data/junyi_with_ktstate.csv      (same)

Runs the four analysis scripts in turn, echoing their console output to the
screen and to results/logs/paper_results.log; figures are saved to
results/figures/:
    cate_analysis.png / sensitivity_analysis.png
    semisynthetic_pehe.png / junyi_replication.png

Usage (from the repository root):
    python src/run_all_results.py
"""
import subprocess
import sys
import os
import time

SCRIPTS = [
    ("ASSISTments09 | ATE/CATE/Qini",                 "cate_analysis.py"),
    ("ASSISTments09 | overlap + sensitivity + E-value","sensitivity_analysis.py"),
    ("ASSISTments09 | semi-synthetic PEHE (RQ1)",      "semisynthetic_pehe.py"),
    ("Junyi | ATE/CATE/Qini/PEHE/sensitivity",         "junyi_full_analysis.py"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)
LOG = os.path.join(LOGDIR, "paper_results.log")


def run_one(title, script, logf):
    header = "\n" + "=" * 70 + f"\n### {title}   [{script}]\n" + "=" * 70 + "\n"
    print(header, flush=True)
    logf.write(header)
    logf.flush()
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        msg = f"[skip] not found: {script}\n"
        print(msg); logf.write(msg); return
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, path], cwd=HERE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        logf.write(line)
    proc.wait()
    tail = f"\n[{script} done, exit code {proc.returncode}, elapsed {time.time()-t0:.1f}s]\n"
    print(tail); logf.write(tail); logf.flush()


def main():
    print(f"Python: {sys.version.split()[0]}")
    for pkg in ("numpy", "pandas", "sklearn", "matplotlib"):
        try:
            m = __import__(pkg)
            print(f"  {pkg:12s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  [missing dependency] {pkg}: {e}  ->  pip install numpy pandas scikit-learn matplotlib")
    with open(LOG, "w", encoding="utf-8") as logf:
        logf.write(f"Python {sys.version}\n")
        for title, script in SCRIPTS:
            run_one(title, script, logf)
    print(f"\nAll done. Text results saved to: {LOG}")
    print("Figures saved to: results/figures/")


if __name__ == "__main__":
    main()
