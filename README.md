# Causal-Help-KT

Code for the study *"When Does Help Actually Help? Causal Estimation of
Help-Seeking Effects Using Deep Knowledge-State Representations for AI Tutoring."*

> **No datasets are distributed in this repository.** Both datasets are
> third-party and governed by their own licenses, so please download them from
> their original sources and regenerate the processed tables with the scripts in
> `src/` (see [Data](#data) and `data/README.md`).

We treat confounding as a representation problem: the hidden state of a pretrained
deep knowledge-tracing (DKT) model, read out **before** a help/no-help decision,
is used as a high-dimensional adjustment set inside cross-fitted, doubly-robust
causal estimators. We estimate the average and heterogeneous effect of
help-seeking on next-item success, stress-test it (overlap trimming, subgroups,
E-value), and benchmark representations on a semi-synthetic design with known
counterfactuals.

> The paper (manuscript/PDF) is **not** included in this repository while it is
> under submission. A link will be added here after publication.

## Key results (reproducible from this repo)

| | ASSISTments09 | Junyi |
|---|---|---|
| Decision points | 28,282 | 104,241 |
| Naive help vs. no-help | -0.441 | -0.243 |
| De-confounded ATE (KT-state) | **-0.275** | **-0.229** |
| CATE negative share | 93.8% | 100% |
| E-value (full / struggling) | 2.68 / 3.45 | 2.16 / 3.29 |
| sqrt(PEHE): KT vs. handcrafted | 0.084 vs. 0.154 | 0.068 vs. 0.115 |

## Repository layout

```
Causal-Help-KT/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/                         # NO data is committed here
│   └── README.md                 # how to download the raw data and regenerate the tables
├── src/
│   ├── preprocess_ckt.py         # ASSISTments -> decision-point table (T/Y/X)
│   ├── junyi_preprocess.py       # Junyi -> decision-point table (same schema)
│   ├── sample_students.py        # student-level subsampling for huge raw logs
│   ├── train_dkt_extract.py      # DKT pretraining + pre-decision state extraction
│   ├── train_dkt_extract_fast.py # faster DKT (embedding + packed sequences)
│   ├── cate_analysis.py          # ASSISTments: AIPW ATE, CATE, DR-Qini
│   ├── sensitivity_analysis.py   # overlap trimming + E-value sensitivity
│   ├── semisynthetic_pehe.py     # semi-synthetic sqrt(PEHE) benchmark
│   ├── junyi_full_analysis.py    # Junyi: ATE/CATE/Qini/PEHE/sensitivity
│   ├── econml_cate.py            # optional: CausalForestDML with honest CIs
│   ├── dataset_stats.py          # descriptive statistics
│   ├── make_paper_figures.py     # publication figures from summary numbers
│   └── run_all_results.py        # one command: run all analyses
└── results/
    ├── figures/                  # fig_pehe.png, fig_forest.png, fig_hetero.png
    └── logs/                     # paper_results.log, junyi_dkt.log
```

All scripts resolve `data/` and `results/` relative to the repository root, so
they can be launched from anywhere (e.g. `python src/run_all_results.py`).

## Reproducing the results

No data ships with this repository, so first download the raw datasets and
rebuild the processed tables (see [Data](#data) and `data/README.md`). Once the
tables are in `data/`, run, from the repository root:

```bash
pip install -r requirements.txt
python src/run_all_results.py      # -> results/logs/paper_results.log + results/figures/
python src/make_paper_figures.py   # regenerates the three paper figures
python src/dataset_stats.py        # dataset descriptive statistics
```

Rebuilding the tables from the raw logs (a GPU is recommended for the DKT step):

```bash
python src/preprocess_ckt.py       # or src/junyi_preprocess.py
python src/train_dkt_extract_fast.py --device cuda --hidden 64 --epochs 30
```

## Data

**No datasets are included in this repository.** Both are third-party datasets
released for **non-commercial research** under their own terms; the ASSISTments
terms in particular permit sharing processing *code* but **not redistributing
the data**. Download the raw files from their original sources and regenerate
the processed tables with the scripts in `src/` (full instructions in
`data/README.md`):

- **ASSISTments 2009-2010 "skill builder"** — obtain `skill_builder_data.csv`
  from the official ASSISTments data page.
- **Junyi Academy** — obtain `Log_Problem.csv`, then subsample students with
  `src/sample_students.py`.

Please cite the original dataset papers (Feng et al., 2009; Chang et al., 2015)
if you use them.

## Design notes

- One decision point per (learner, skill) sequence, fixed at the second
  same-skill interaction (position is independent of treatment).
- Treatment `T`: help-seeking (hint/scaffold) at the decision point.
- Outcome `Y`: correctness of the **next** same-skill item (no leakage).
- Covariates: strictly pre-treatment only; the DKT state is read out before the
  decision, so it never sees the treatment or the outcome.

## License

MIT (see `LICENSE`).
