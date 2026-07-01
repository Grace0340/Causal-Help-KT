#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKT pretraining + pre-decision knowledge-state h_t extraction (faster version).

Speed-ups over train_dkt_extract.py:
  1. nn.Embedding lookup instead of building a 246-d one-hot
  2. sequences pre-encoded and cached once, not rebuilt every epoch
  3. pack_padded_sequence skips padding time steps
  4. automatic device detection: falls back to CPU if --device cuda is
     unavailable; enables cudnn.benchmark
The alignment/extraction/join logic is identical to the original (verified to
align 100% of decision points on the real data).

GPU:  python src/train_dkt_extract_fast.py --device cuda --hidden 64 --epochs 30
CPU:  python src/train_dkt_extract_fast.py --device cpu  --hidden 64 --epochs 30
"""
import argparse
import os
import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ===================== pure data / alignment logic (no torch needed) =====================
def load_sequences(seq_path):
    df = pd.read_csv(seq_path)
    df["skill_id"] = df["skill_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df = df.sort_values(["user_id", "order_id", "skill_id"]).reset_index(drop=True)
    skills = sorted(df["skill_id"].unique())
    skill2idx = {s: i for i, s in enumerate(skills)}
    df["sidx"] = df["skill_id"].map(skill2idx)
    return df, skill2idx


def build_user_steps(df):
    user_steps, user_pos = {}, {}
    for uid, g in df.groupby("user_id", sort=False):
        sidx = g["sidx"].to_numpy(); corr = g["correct"].to_numpy()
        skid = g["skill_id"].to_numpy(); oid = g["order_id"].to_numpy()
        user_steps[uid] = (sidx, corr)
        pos = {}
        for p in range(len(skid)):
            key = (int(skid[p]), int(oid[p]))
            if key not in pos:
                pos[key] = p
        user_pos[uid] = pos
    return user_steps, user_pos


def extract_state_matrix(decision_df, user_steps, user_pos, get_states, H):
    X = np.zeros((len(decision_df), H), dtype=np.float32)
    matched = 0; cache = {}
    for i, row in enumerate(decision_df.itertuples(index=False)):
        uid = row.user_id
        key = (int(row.skill_id), int(row.decision_order_id))
        pos = user_pos.get(uid, {}).get(key, None)
        if pos is None:
            continue
        if uid not in cache:
            sidx, corr = user_steps[uid]
            cache[uid] = get_states(uid, sidx, corr)
        X[i] = cache[uid][pos]
        matched += 1
    return X, matched


# ============================ DKT (Embedding + pack) ============================
def train_dkt(df, skill2idx, hidden=64, emb=100, epochs=30, batch=128, lr=1e-3,
              max_len=200, device="cpu", seed=42):
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    from sklearn.metrics import roc_auc_score

    # ---- device detection ----
    if device == "cuda" and not torch.cuda.is_available():
        print("  [warn] no usable CUDA detected, falling back to CPU")
        device = "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"  device: {device}")

    torch.manual_seed(seed); np.random.seed(seed)
    S = len(skill2idx); N_INTER = 2 * S

    # ---- pre-encode each student's sequence, cache as LongTensor (once) ----
    cached = []
    for uid, g in df.groupby("user_id", sort=False):
        s = g["sidx"].to_numpy()[:max_len]
        c = g["correct"].to_numpy()[:max_len]
        if len(s) < 2:
            continue
        inter = torch.tensor(s * 2 + c, dtype=torch.long)     # interaction id = skill*2 + correct
        tgt_s = torch.tensor(s[1:], dtype=torch.long)
        tgt_y = torch.tensor(c[1:], dtype=torch.float32)
        cached.append((inter, tgt_s, tgt_y))
    rng = np.random.default_rng(seed); rng.shuffle(cached)
    n_val = max(1, int(0.1 * len(cached)))
    val_set, tr_set = cached[:n_val], cached[n_val:]

    def collate(items):
        inter = pad_sequence([x[0] for x in items], batch_first=True)
        ts = pad_sequence([x[1] for x in items], batch_first=True)
        ty = pad_sequence([x[2] for x in items], batch_first=True)
        lengths = torch.tensor([len(x[0]) for x in items])
        mask = pad_sequence([torch.ones(len(x[1])) for x in items], batch_first=True)
        return inter, ts, ty, lengths, mask

    class DKT(nn.Module):
        def __init__(self, n_inter, emb_dim, hid, n_skill):
            super().__init__()
            self.emb = nn.Embedding(n_inter, emb_dim)
            self.lstm = nn.LSTM(emb_dim, hid, batch_first=True)
            self.out = nn.Linear(hid, n_skill)
        def forward(self, inter, lengths=None):
            x = self.emb(inter)
            if lengths is not None:
                packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                              enforce_sorted=False)
                h, _ = self.lstm(packed)
                h, _ = pad_packed_sequence(h, batch_first=True)
            else:
                h, _ = self.lstm(x)
            return torch.sigmoid(self.out(h)), h

    model = DKT(N_INTER, emb, hidden, S).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction="none")

    def run(data, train=True):
        model.train(train)
        idx = np.arange(len(data))
        if train: np.random.shuffle(idx)
        tot, ys, ps = 0.0, [], []
        for k in range(0, len(idx), batch):
            items = [data[j] for j in idx[k:k + batch]]
            inter, ts, ty, lengths, mask = collate(items)
            inter, ts, ty, mask = inter.to(device), ts.to(device), ty.to(device), mask.to(device)
            pred, _ = model(inter, lengths)            # [B, Lmax, S]
            Lt = ts.size(1)
            pred = pred[:, :Lt, :]                      # align to steps 1..L-1
            p_next = pred.gather(2, ts.unsqueeze(2)).squeeze(2)
            loss = (bce(p_next, ty) * mask).sum() / mask.sum()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(items)
            m = mask.bool()
            ys.append(ty[m].detach().cpu().numpy())
            ps.append(p_next[m].detach().cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        return tot / len(data), auc

    for ep in range(1, epochs + 1):
        tr_loss, tr_auc = run(tr_set, True)
        with torch.no_grad():
            _, va_auc = run(val_set, False)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d} | train AUC {tr_auc:.3f} | val AUC {va_auc:.3f}")

    # ---- get_states: state[t] encodes history for step<t (state[0]=0), no leakage ----
    def get_states(uid, sidx, corr):
        model.eval()
        inter = torch.tensor(sidx * 2 + corr, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            _, h = model(inter, None)                  # [1, L, H]
        h = h.squeeze(0).cpu().numpy()
        states = np.zeros_like(h); states[1:] = h[:-1]
        return states

    return get_states, hidden


# ================================ main ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=os.path.join(_DATA, "assist09_sequences.csv"))
    ap.add_argument("--decisions", default=os.path.join(_DATA, "assist09_decision_points.csv"))
    ap.add_argument("--out", default=os.path.join(_DATA, "assist09_with_ktstate.csv"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("[1/4] loading sequences ...")
    df, skill2idx = load_sequences(args.seq)
    dec = pd.read_csv(args.decisions)
    print(f"      sequence rows {len(df)} | skills {len(skill2idx)} | decision points {len(dec)}")

    print("[2/4] training DKT ...")
    get_states, H = train_dkt(df, skill2idx, hidden=args.hidden, emb=args.emb,
                              epochs=args.epochs, batch=args.batch, lr=args.lr,
                              device=args.device)

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


if __name__ == "__main__":
    main()
