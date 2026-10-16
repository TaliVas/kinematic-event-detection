# HFS vs control decoding from 3D reaching kinematics (within-animal).
# Reproduces the reported decoding accuracies over 10 runs, and a
# session-grouped evaluation that keeps whole sessions out of training.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

DATA = {
    'Thina': 'data/trajectories_thina_subset.pkl',
    'Nana':  'data/trajectories_nana_subset.pkl',
}
WIN_START, WIN_END = 200, 600      # adjusted_time window, -100 to +300 ms around onset
MIN_LEN, MAX_LEN_FILTER = 35, 50
EPOCHS = 70
LR = 0.001
BATCH = 32
SEEDS = list(range(42, 52))
FEATS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']


def load_trials(pkl):
    df = pd.read_pickle(pkl)
    rows = []
    for tid, g in df.groupby('id'):
        g = g.sort_values('time_milisecond')
        w = g[(g['adjusted_time'] >= WIN_START) & (g['adjusted_time'] <= WIN_END)]
        if len(w) < 10:
            continue
        rows.append({
            'id': tid, 'type': g['type'].iloc[0], 'target': g['id_target'].iloc[0],
            'session': g['file_name'].iloc[0], 'seq_length': len(w),
            'pos_x': w['centered_x'].values, 'pos_y': w['centered_y'].values, 'pos_z': w['centered_z'].values,
            'vel_x': w['dx'].values, 'vel_y': w['dy'].values, 'vel_z': w['dz'].values,
        })
    t = pd.DataFrame(rows)
    t = t[(t['seq_length'] >= MIN_LEN) & (t['seq_length'] <= MAX_LEN_FILTER)].reset_index(drop=True)
    return t


def balance_per_target(t):
    parts = []
    for tg in sorted(t['target'].unique()):
        sub = t[t['target'] == tg]
        c = sub[sub['type'] == 'Control']
        h = sub[sub['type'] == 'HFS']
        n = min(len(c), len(h))
        parts.append(c.sample(n, random_state=42))
        parts.append(h.sample(n, random_state=42))
    return pd.concat(parts).reset_index(drop=True)


def build_arrays(t):
    maxlen = t['seq_length'].max()
    X = np.zeros((len(t), maxlen, len(FEATS)), dtype=np.float32)
    lengths = np.zeros(len(t), dtype=np.int64)
    for i, (_, r) in enumerate(t.iterrows()):
        lengths[i] = r['seq_length']
        for j, f in enumerate(FEATS):
            v = r[f]
            if f.startswith('pos'):
                v = v - v[0]               # re-center so each reach starts at the origin
            X[i, :r['seq_length'], j] = v
    y = (t['type'] == 'HFS').astype(int).values
    return X, lengths, y, t['target'].values, t['session'].values




def zscore_fit(X, lengths, train_idx):
    # per-feature mean/std over valid (non-padded) timesteps of the training trials
    vals = [[] for _ in range(X.shape[2])]
    for i in train_idx:
        for j in range(X.shape[2]):
            vals[j].append(X[i, :lengths[i], j])
    mean = np.array([np.concatenate(v).mean() for v in vals], dtype=np.float32)
    std = np.array([np.concatenate(v).std() or 1.0 for v in vals], dtype=np.float32)
    return mean, std


def zscore_apply(X, lengths, mean, std):
    Xs = X.copy()
    for i in range(len(X)):
        Xs[i, :lengths[i]] = (X[i, :lengths[i]] - mean) / std
    return Xs


class TrajDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.lengths = torch.from_numpy(np.asarray(lengths, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.lengths[i]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size * 2, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        pooled = self.pool(out.permute(0, 2, 1)).squeeze(-1)
        return self.fc2(self.relu(self.fc1(pooled)))


def train_eval(X, y, lengths, train_idx, test_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = zscore_fit(X, lengths, train_idx)      # fit on training trials only
    Xs = zscore_apply(X, lengths, mean, std)
    tr = DataLoader(TrajDataset(Xs[train_idx], y[train_idx], lengths[train_idx]), batch_size=BATCH, shuffle=True)
    te = DataLoader(TrajDataset(Xs[test_idx], y[test_idx], lengths[test_idx]), batch_size=BATCH)
    model = LSTMClassifier(len(FEATS))
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb, lb in tr:
            opt.zero_grad()
            loss = crit(model(xb, lb).squeeze(), yb)
            loss.backward()
            opt.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb, lb in te:
            p = (torch.sigmoid(model(xb, lb).squeeze()) >= 0.5).float()
            preds.extend(p.numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


def run_within_animal(monkey):
    t = balance_per_target(load_trials(DATA[monkey]))
    X, lengths, y, targets, sessions = build_arrays(t)
    strat = pd.Series(y).astype(str) + '_' + pd.Series(targets).astype(str)
    accs, per_target = [], {}
    for seed in SEEDS:
        tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                          random_state=seed, stratify=strat)
        preds, labels = train_eval(X, y, lengths, tr_idx, te_idx, seed)
        accs.append(accuracy_score(labels, preds))
        tt = targets[te_idx]
        for tg in np.unique(tt):
            per_target.setdefault(int(tg), []).append(accuracy_score(labels[tt == tg], preds[tt == tg]))
        print(f'{monkey} seed {seed}: accuracy {accs[-1]:.3f}')
    print(f'\n{monkey}: overall {np.mean(accs):.3f} +/- {np.std(accs, ddof=1):.3f} (10 runs)')
    for tg in sorted(per_target):
        a = np.array(per_target[tg])
        print(f'  target {tg}: {a.mean():.3f} +/- {a.std(ddof=1):.3f}')


def run_session_grouped(monkey):
    t = balance_per_target(load_trials(DATA[monkey]))
    X, lengths, y, targets, sessions = build_arrays(t)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=sessions))
    assert set(sessions[tr_idx]).isdisjoint(set(sessions[te_idx]))
    preds, labels = train_eval(X, y, lengths, tr_idx, te_idx, 42)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    acc = accuracy_score(labels, preds)
    # session-aware permutation test: shuffle labels within each test session
    test_sessions = sessions[te_idx]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(1000):
        yperm = labels.copy()
        for s in np.unique(test_sessions):
            m = np.where(test_sessions == s)[0]
            yperm[m] = rng.permutation(labels[m])
        null.append(accuracy_score(yperm, preds))
    p = (np.sum(np.array(null) >= acc) + 1) / 1001
    print(f'{monkey} session-grouped: accuracy {acc:.3f}  '
          f'balanced {balanced_accuracy_score(labels, preds):.3f}  '
          f'sensitivity {tp/(tp+fn):.3f}  specificity {tn/(tn+fp):.3f}  '
          f'permutation p={p:.3f}')


if __name__ == '__main__':
    for m in ['Thina', 'Nana']:
        run_within_animal(m)
        run_session_grouped(m)
