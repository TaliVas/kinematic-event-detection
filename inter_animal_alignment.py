# Inter-animal alignment (IAA): rotation-constrained Procrustes (Kabsch) that maps
# one animal's workspace onto the other using per-target endpoint directions.

import numpy as np
import pandas as pd


def target_direction_vectors(df, pos_cols=('rotation_x', 'rotation_y', 'rotation_z'),
                              target_col='id_target', origin=None):
    # For each target, average final hand position relative to the origin, normalized to unit length.
    if origin is None:
        origin = np.zeros(3)
    vecs = {}
    for tg, g in df.groupby(target_col):
        ends = []
        for _, trial in g.groupby('id'):
            p = trial[list(pos_cols)].values[-1]
            ends.append(p)
        v = np.mean(ends, axis=0) - origin
        n = np.linalg.norm(v)
        vecs[int(tg)] = v / n if n > 0 else v
    return vecs


def kabsch_rotation(V_source, V_target):
    # Optimal rotation R in SO(3) mapping V_source onto V_target (both k x 3, matched order).
    M = V_target.T @ V_source
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R


def fit_alignment(df_source, df_target, pos_cols=('rotation_x', 'rotation_y', 'rotation_z')):
    vs = target_direction_vectors(df_source, pos_cols)
    vt = target_direction_vectors(df_target, pos_cols)
    targets = sorted(set(vs) & set(vt))
    V_source = np.array([vs[t] for t in targets])
    V_target = np.array([vt[t] for t in targets])
    return kabsch_rotation(V_source, V_target)


def apply_rotation(df, R, pos_cols=('rotation_x', 'rotation_y', 'rotation_z')):
    out = df.copy()
    out[list(pos_cols)] = df[list(pos_cols)].values @ R.T
    return out


if __name__ == '__main__':
    import pickle
    # Each subset is a pickle (df_features, start_move, stop_turn); the rotation is fit
    # from all direct (non-update) trials and maps the source animal onto the target.
    df_source = pickle.load(open('data/features_nana_aligned_subset.pkl', 'rb'))[0]
    df_target = pickle.load(open('data/features_thina_aligned_subset.pkl', 'rb'))[0]
    direct_source = df_source[df_source['id_update'].isna()]
    direct_target = df_target[df_target['id_update'].isna()]
    R = fit_alignment(direct_source, direct_target)
    print('rotation matrix:\n', R)
    print('det:', round(float(np.linalg.det(R)), 4))
    df_source_aligned = apply_rotation(df_source, R)
