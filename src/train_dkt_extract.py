#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKT pretraining + pre-decision knowledge-state h_t extraction -> causal table with KT features

Pipeline:
  1. read assist09_sequences.csv, reorder into per-student time series by (user, order_id)
  2. train a standard DKT (Piech et al. 2015): an LSTM encodes the (skill, correct)
     stream and predicts the next step
  3. for each decision point (user, skill, decision_order_id), find its position p in
     the student's sequence and take the strictly pre-decision state state[p]
     (= encodes history for step<p, not the decision point itself); this is the
     confounder representation X in the causal graph
  4. join h0..h{H-1} back to assist09_decision_points.csv -> table with the KT state

Design note (no leakage):
  state[p] only encodes interactions before the decision point; it excludes the
  decision's treatment action and outcome, and the outcome item Y. Thus h_t is a
  clean pre-treatment covariate.

Run on a GPU box:
  python src/train_dkt_extract.py --device cuda --hidden 64 --epochs 30
"""
import argparse
import os
import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ---- torch is imported lazily inside training so the pure-data functions work without it ----


# ======================= pure data / alignment logic (no torch) =======================
def load_sequences(seq_path):
    df = pd.read_csv(seq_path)
    df["skill_id"] = df["skill_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    # DKT needs each student's real time order (order_id is the time key)
    df = df.sort_values(["user_id", "order_id", "skill_id"]).reset_index(drop=True)
    skills = sorted(df["skill_id"].unique())
    skill2idx = {s: i for i, s in enumerate(skills)}
    df["sidx"] = df["skill_id"].map(skill2idx)
    return df, skill2idx


def build_user_steps(df):
    """per student -> time-ordered (sidx, correct) + a (skill_id, order_id)->position map"""
    user_steps, user_pos = {}, {}
    for uid, g in df.groupby("user_id", sort=False):
        sidx = g["sidx"].to_numpy()
        corr = g["correct"].to_numpy()
        skid = g["skill_id"].to_numpy()
        oid = g["order_id"].to_numpy()
        user_steps[uid] = (sidx, corr)
        # (skill_id, order_id) -> global position p (first occurrence)
        pos = {}
        for p in range(len(skid)):
            key = (int(skid[p]), int(oid[p]))
            if key not in pos:
                pos[key] = p
        user_pos[uid] = pos
    return user_steps, user_pos


def extract_state_matrix(decision_df, user_steps, user_pos, get_states, H):
    """get_states(uid, sidx, corr) -> ndarray [L, H], where state[t] encodes history for step<t"""
    X = np.zeros((len(decision_df), H), dtype=np.float32)
    matched = 0
    cache = {}
    for i, row in enumerate(decision_df.itertuples(index=False)):
        uid = row.user_id
        key = (int(row.skill_id), int(row.decision_order_id))
        pos = user_pos.get(uid, {}).get(key, None)
        if pos is None:
            continue
        if uid not in cache:
            sidx, corr = user_steps[uid]
            cache[uid] = get_states(uid, sidx, corr)   # [L, H]
        states = cache[uid]
        X[i] = states[pos]                              # strictly pre-decision state
        matched += 1
    return X, matched


# ============================ DKT model and training ============================
def train_dkt(df, skill2idx, hidden=64, epochs=30, batch=64, lr=1e-3,
              max_len=200, device="cpu", seed=42):
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    S = len(skill2idx)

    # assemble each student's time-series tensor
    seqs = []
    for uid, g in df.groupby("user_id", sort=False):
        s = g["sidx"].to_numpy(); c = g["correct"].to_numpy()
        if len(s) < 2:
            continue
        seqs.append((s[:max_len], c[:max_len]))
    rng = np.random.default_rng(seed); rng.shuffle(seqs)
    n_val = max(1, int(0.1 * len(seqs)))
    val_seqs, tr_seqs = seqs[:n_val], seqs[n_val:]

    def to_tensors(batch_seqs):
        xs, masks, tgt_s, tgt_y = [], [], [], []
        for s, c in batch_seqs:
            L = len(s)
            x = np.zeros((L, 2 * S), dtype=np.float32)
            x[np.arange(L), s * 2 + c] = 1.0          # one-hot(skill, correct)
            xs.append(torch.tensor(x))
            # predict step t+1: inputs 0..L-2, targets 1..L-1
            tgt_s.append(torch.tensor(s[1:], dtype=torch.long))
            tgt_y.append(torch.tensor(c[1:], dtype=torch.float32))
            masks.append(torch.ones(L - 1))
        X = pad_sequence(xs, batch_first=True)
        TS = pad_sequence(tgt_s, batch_first=True)
        TY = pad_sequence(tgt_y, batch_first=True)
        M = pad_sequence(masks, batch_first=True)
        return X, TS, TY, M

    class DKT(nn.Module):
        def __init__(self, in_dim, hid, n_skill):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
            self.out = nn.Linear(hid, n_skill)
        def forward(self, x):
            h, _ = self.lstm(x)                        # [B, L, H]
            return torch.sigmoid(self.out(h)), h       # prediction + hidden state

    model = DKT(2 * S, hidden, S).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction="none")

    def run(seqs_, train=True):
        model.train(train)
        order = np.arange(len(seqs_))
        if train: np.random.shuffle(order)
        tot, ys, ps = 0.0, [], []
        for k in range(0, len(order), batch):
            bs = [seqs_[j] for j in order[k:k + batch]]
            X, TS, TY, M = [t.to(device) for t in to_tensors(bs)]
            pred, _ = model(X)                          # [B, L, S]
            pred = pred[:, :-1, :]                      # align prediction for step t+1
            p_next = pred.gather(2, TS.unsqueeze(2)).squeeze(2)  # [B, L-1]
            loss = (bce(p_next, TY) * M).sum() / M.sum()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bs)
            m = M.bool()
            ys.append(TY[m].detach().cpu().numpy())
            ps.append(p_next[m].detach().cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        return tot / len(seqs_), auc

    for ep in range(1, epochs + 1):
        tr_loss, tr_auc = run(tr_seqs, True)
        with torch.no_grad():
            va_loss, va_auc = run(val_seqs, False)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d} | train AUC {tr_auc:.3f} | val AUC {va_auc:.3f}")

    # build get_states: state[t] = history for step<t (state[0]=0)
    def get_states(uid, sidx, corr):
        model.eval()
        L = len(sidx)
        x = np.zeros((L, 2 * S), dtype=np.float32)
        x[np.arange(L), sidx * 2 + corr] = 1.0
        with torch.no_grad():
            _, h = model(torch.tensor(x).unsqueeze(0).to(device))  # [1, L, H]
        h = h.squeeze(0).cpu().numpy()                  # [L, H]: h[t] = output including step t
        states = np.zeros_like(h)
        states[1:] = h[:-1]                             # shift right -> state[t] only sees step<t
        return states

    return get_states, hidden


# ================================ main ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=os.path.join(_DATA, "assist09_sequences.csv"))
    ap.add_argument("--decisions", default=os.path.join(_DATA, "assist09_decision_points.csv"))
    ap.add_argument("--out", default=os.path.join(_DATA, "assist09_with_ktstate.csv"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("[1/4] loading sequences ...")
    df, skill2idx = load_sequences(args.seq)
    dec = pd.read_csv(args.decisions)
    print(f"      sequence rows {len(df)} | skills {len(skill2idx)} | decision points {len(dec)}")

    print("[2/4] training DKT ...")
    get_states, H = train_dkt(df, skill2idx, hidden=args.hidden, epochs=args.epochs,
                              batch=args.batch, lr=args.lr, device=args.device)

    print("[3/4] extracting pre-decision knowledge state h_t ...")
    user_steps, user_pos = build_user_steps(df)
    X, matched = extract_state_matrix(dec, user_steps, user_pos, get_states, H)
    print(f"      aligned {matched}/{len(dec)} decision points")

    print("[4/4] joining and saving ...")
    h_cols = [f"h{i}" for i in range(H)]
    out = pd.concat([dec.reset_index(drop=True),
                     pd.DataFrame(X, columns=h_cols)], axis=1)
    out.to_csv(args.out, index=False)
    print(f"      saved {args.out}  (shape={out.shape})")
    print("\nnext: estimate CATE on (X=h0..h{}, T, Y) with EconML and compare sqrt(PEHE) "
          "against the handcrafted-covariate version.".format(H - 1))


if __name__ == "__main__":
    main()
