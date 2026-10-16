# BiLSTM event detection (movement onset / turning point) with per-trial uncertainty.
# Trains within-animal and evaluates within- and between-animal (after inter-animal
# alignment). Features are z-scored using training-set statistics only.

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

FEATURES = ['rotation_x', 'rotation_y', 'rotation_z', 'normalized_time',
            'velocity', 'angle', 'distance_to_end']
EPOCHS = 100
LR = 0.001
BATCH = 32
SEED = 42


def load_features(path, label_key='start_move'):
    # path points to a pickle holding (df_features, start_move, stop_turn).
    # label_key selects the event: 'start_move' (onset) or 'stop_turn' (turning point).
    with open(path, 'rb') as f:
        df_features, start_move, stop_turn = pickle.load(f)
    labels = {'start_move': start_move, 'stop_turn': stop_turn}[label_key]
    labels = {k: v for k, v in labels.items() if v is not None and not pd.isna(v) and v > 0.0005}
    df_features = df_features[df_features['id'].isin(labels)]
    df_features = df_features.dropna(subset=FEATURES)
    return df_features, labels


def to_sequences(df_features):
    tmax = df_features.groupby('id')['adjusted_time'].max()
    tmin = df_features.groupby('id')['adjusted_time'].min()
    grouped = df_features.groupby('id')
    ids = list(grouped.groups.keys())
    seqs = [g[FEATURES].values.astype(np.float32) for _, g in grouped]
    ttype = df_features.groupby('id')['type_trajectory'].first().values
    uniq = np.unique(ttype)
    t2i = {t: i for i, t in enumerate(uniq)}
    type_idx = np.array([t2i[t] for t in ttype])
    return ids, seqs, type_idx, len(uniq), tmax, tmin


def zscore(train_seqs, seqs):
    allv = np.concatenate(train_seqs, axis=0)
    mean = allv.mean(axis=0)
    std = allv.std(axis=0)
    std[std == 0] = 1.0
    return [((s - mean) / std).astype(np.float32) for s in seqs]


def pad(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen, len(FEATURES)), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


class TrajDataset(Dataset):
    def __init__(self, X, types, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.types = torch.from_numpy(np.asarray(types, dtype=np.int64))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.types[i], self.y[i]


class TrajectoryModel(nn.Module):
    def __init__(self, num_types, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 50, bidirectional=True, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Embedding(num_types, 2)
        self.fc1 = nn.Linear(100 + 2, 100)
        self.fc_mean = nn.Linear(100, 1)
        self.fc_logvar = nn.Linear(100, 1)

    def forward(self, x, types):
        out, _ = self.lstm(x)
        pooled = self.pool(out.permute(0, 2, 1)).squeeze(-1)
        emb = self.embedding(types).flatten(1)
        h = torch.relu(self.fc1(torch.cat([pooled, emb], dim=1)))
        return self.fc_mean(h), self.fc_logvar(h)


def gaussian_nll(mu, logvar, target):
    return (torch.exp(-logvar) * (target - mu) ** 2 + logvar).mean()


def train(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for x, types, y in loader:
            opt.zero_grad()
            mu, logvar = model(x, types)
            loss = gaussian_nll(mu.squeeze(), logvar.squeeze(), y)
            loss.backward()
            opt.step()


def mae_ms(ids, mu, labels, tmax, tmin):
    e = []
    for _id, m in zip(ids, mu):
        true_ms = labels[_id] * (tmax[_id] - tmin[_id]) + tmin[_id]
        pred_ms = m * (tmax[_id] - tmin[_id]) + tmin[_id]
        e.append(abs(pred_ms - true_ms))
    return float(np.mean(e))


def predict(model, X, types):
    model.eval()
    with torch.no_grad():
        mu, _ = model(torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(types, dtype=np.int64)))
    return mu.squeeze().numpy()


def run_within_animal(path, event='start_move'):
    df, labels = load_features(path, event)
    ids, seqs, types, n_types, tmax, tmin = to_sequences(df)
    y = np.array([labels[i] for i in ids], dtype=np.float32)
    idx = np.arange(len(ids))
    counts = np.bincount(types)
    safe = len(counts[counts > 0]) > 1 and counts[counts > 0].min() >= 2
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED,
                              stratify=types if safe else None)
    seqs_z = zscore([seqs[i] for i in tr], seqs)       # fit z-score on training trials only
    maxlen = max(len(s) for s in seqs_z)
    X = pad(seqs_z, maxlen)
    model = TrajectoryModel(n_types, len(FEATURES))
    train(model, DataLoader(TrajDataset(X[tr], types[tr], y[tr]), batch_size=BATCH, shuffle=True))
    mu = predict(model, X[te], types[te])
    test_ids = [ids[i] for i in te]
    print(f'within-animal {event}: MAE = {mae_ms(test_ids, mu, labels, tmax, tmin):.2f} ms  (n={len(te)})')
    return model


if __name__ == '__main__':
    # onset:
    # Event-detection subset: pickle holding (df_features, start_move, stop_turn).
    run_within_animal('data/features_nana_aligned_subset.pkl', event='start_move')   # movement onset
    # run_within_animal('data/features_nana_aligned_subset.pkl', event='stop_turn')  # turning point
