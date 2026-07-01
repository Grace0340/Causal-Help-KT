# Data

**No datasets are distributed in this repository.** Both datasets are
third-party, released for **non-commercial research use only**, and governed by
their own terms of use. Download the raw files from their original sources and
regenerate the processed tables locally with the scripts in `src/`.

## What the scripts expect in this folder

Place the raw files here, then run the preprocessing/extraction steps below to
(re)create the processed tables used by the analysis code:

| Processed file (generated locally) | Rows | Description |
|---|---|---|
| `*_decision_points.csv` | — | One decision point per (learner, skill): treatment `T`, outcome `Y`, and 6 handcrafted, strictly pre-treatment covariates |
| `*_sequences.csv` | — | Cleaned interaction sequences used to pretrain the DKT encoder |
| `*_with_ktstate.csv` | — | Decision points joined with the 64-d pre-decision DKT state (`h0`…`h63`) |

Column conventions:

- `T` — help-seeking at the decision point (`1` = hint/scaffold requested, `0` = not).
- `Y` — correctness of the **next** same-skill item (first attempt).
- Handcrafted covariates: `prior_correct`, `prior_attempt`, `prior_hint`,
  `prior_log_ms`, `opportunity`, `skill_difficulty`.
- `h0`…`h63` — recurrent DKT hidden state read out **before** the decision.

## How to obtain the raw data and rebuild the tables

### ASSISTments 2009–2010 "skill builder"

Download `skill_builder_data.csv` from the official ASSISTments data page, place
it in this `data/` folder, then:

```bash
python src/preprocess_ckt.py            # -> assist09_sequences.csv + assist09_decision_points.csv
python src/train_dkt_extract_fast.py \
    --seq data/assist09_sequences.csv \
    --decisions data/assist09_decision_points.csv \
    --out data/assist09_with_ktstate.csv --device cuda
```

### Junyi Academy

Download `Log_Problem.csv` (tens of millions of rows). Subsample students first,
then preprocess and extract DKT states:

```bash
python src/sample_students.py --src data/Log_Problem.csv --out data/junyi_sample.csv \
    --id-col uuid --n-students 5000
python src/junyi_preprocess.py          # -> junyi_sequences.csv + junyi_decision_points.csv
python src/train_dkt_extract_fast.py \
    --seq data/junyi_sequences.csv \
    --decisions data/junyi_decision_points.csv \
    --out data/junyi_with_ktstate.csv --device cuda
```

## Sources, licensing and citation

Both datasets are for **non-commercial research use only**. Cite the original
papers if you use them.

- **ASSISTments 2009–2010 "skill builder".** Free for research; **redistribution
  of the data is not permitted** (share only code that processes it — each user
  must obtain the raw file from the original source), no commercial use, no
  de-anonymization. Cite: M. Feng, N. Heffernan, and K. Koedinger, "Addressing
  the assessment challenge with an online system that tutors as it assesses,"
  *User Modeling and User-Adapted Interaction*, 19(3):243–266, 2009.
- **Junyi Academy.** Non-commercial use only; attribution required. Cite:
  H.-S. Chang, H.-J. Hsu, and K.-T. Chen, "Modeling exercise relationships in
  e-learning: A unified approach," in *Proc. Int. Conf. Educational Data Mining
  (EDM)*, 2015, pp. 532–535.
