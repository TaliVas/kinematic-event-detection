# Feature extraction for 3D reaching trajectories.
# Input: a per-timestep dataframe with raw hand positions (x, y, z), trial timing
# columns (RT, RTu, MT, MTu, go_time, cue_time, Cue2Go, Cue2GoU, Go2U, update_time,
# updTime) and trial metadata (id, id_target, id_update, type, file_name).
# Output: (df_features, start_move, stop_turn) saved as a pickle, where df_features
# holds the per-timestep features and start_move / stop_turn map trial id to the
# normalized event times. Features are z-scored at training time (not here) to avoid
# leakage across the train/test split.

import numpy as np
import pandas as pd
import pickle
from scipy.signal import savgol_filter

LEFT_WINDOW, RIGHT_WINDOW = 200, 1500
MAX_MOVEMENT_DURATION = 1200
TARGET_ANGLES = {1: 270, 2: 315, 3: 0, 4: 45, 5: 90, 6: 135, 7: 180, 8: 225}


def three_point_angles(df):
    p1 = df[['x', 'y', 'z']].shift(1)
    p2 = df[['x', 'y', 'z']]
    p3 = df[['x', 'y', 'z']].shift(-1)
    v1 = (p2 - p1).values
    v2 = (p3 - p2).values
    dot = (v1 * v2).sum(axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norms = np.where(norms == 0, 1e-10, norms)
    return np.arccos(np.clip(dot / norms, -1.0, 1.0))


def compute_velocity(df, signal_cue):
    d = df[df['time_milisecond'] >= signal_cue].copy()
    dx, dy, dz = d['x'].diff(), d['y'].diff(), d['z'].diff()
    dt = d['time_milisecond'].diff()
    d['velocity'] = np.sqrt(dx**2 + dy**2 + dz**2) / (dt + 1e-10)
    d = d.dropna(subset=['velocity'])
    d['three_point_angle'] = three_point_angles(d)
    ddx, ddy, ddz = dx.diff().fillna(0), dy.diff().fillna(0), dz.diff().fillna(0)
    d['curvature'] = (np.sqrt((ddy*dz - ddz*dy)**2 + (ddz*dx - ddx*dz)**2 + (ddx*dy - ddy*dx)**2)
                      / (np.power(dx**2 + dy**2 + dz**2, 1.5) + 1e-10))
    return d.iloc[1:]


def process_trial(trial):
    t = trial.copy()
    t['time_milisecond'] = t['time_milisecond'] * 1000
    t['go_milisecond'] = t.iloc[0]['go_time'] * 1000
    if not pd.isna(t.iloc[0]['RTu']):
        t['update_milisecond'] = t.iloc[0]['update_time'] * 1000
        t['start_movement_miliseconds'] = t.iloc[0]['RTu'] * 1000 + t.iloc[0]['go_milisecond']
        t['movement_duration_miliseconds'] = t.iloc[0]['MTu'] * 1000
        if not (np.isnan(t.iloc[0]['updTime']) or t.iloc[0]['updTime'] == -100):
            t['stop_turn_miliseconds'] = t.iloc[0]['updTime'] * 1000 + t.iloc[0]['update_milisecond']
    else:
        t['start_movement_miliseconds'] = t.iloc[0]['RT'] * 1000 + t.iloc[0]['go_milisecond']
        t['movement_duration_miliseconds'] = t.iloc[0]['MT'] * 1000
    return compute_velocity(t, t.iloc[0]['cue_time'] * 1000)


def reaction_time_ok(group):
    return ((pd.isna(group['RT']).all() and (-0.320 <= group['RTu']).all() and (group['RTu'] <= 0.5).all()) or
            (pd.isna(group['RTu']).all() and (-0.2 <= group['RT']).all() and (group['RT'] <= 0.5).all()))


def window_and_normalize(trial):
    go = trial.iloc[0]['go_milisecond']
    w = trial[(trial['time_milisecond'] >= go - LEFT_WINDOW) &
              (trial['time_milisecond'] <= go + RIGHT_WINDOW)].copy()
    t0 = w.iloc[0]['time_milisecond']
    w['adjusted_time'] = w['time_milisecond'] - t0
    tmax = w['adjusted_time'].max()
    w['normalized_time'] = w['adjusted_time'] / tmax
    w['normalized_start_movement'] = (trial['start_movement_miliseconds'].iloc[0] - t0) / tmax
    if 'stop_turn_miliseconds' in trial.columns and not pd.isna(trial['stop_turn_miliseconds'].iloc[0]):
        w['normalized_stop_turn'] = (trial['stop_turn_miliseconds'].iloc[0] - t0) / tmax
    return w


def smooth(group):
    n = min(5, len(group) - (1 - len(group) % 2))
    if n >= 3:
        for c in ['x', 'y', 'z']:
            group['smooth_' + c] = savgol_filter(group[c], n, 1)
    else:
        for c in ['x', 'y', 'z']:
            group['smooth_' + c] = group[c].values
    return group


def recenter(group):
    for c in ['x', 'y', 'z']:
        group['centered_' + c] = group['smooth_' + c] - group['smooth_' + c].iloc[0]
    return group


def rotate_xy(x, y, angle_deg):
    a = np.radians(angle_deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def rotate(group):
    tg = int(group.iloc[0]['id_target'])
    updated = not pd.isna(group.iloc[0]['id_update'])
    up = group.iloc[0]['id_update'] if updated else None
    if tg == 5:
        rx, ry, rz = group['centered_x'], group['centered_y'], group['centered_z']
        if updated and up > tg:
            rx = -rx
    else:
        rx, ry = rotate_xy(group['centered_x'], group['centered_y'], 90 - TARGET_ANGLES[tg])
        rz = group['centered_z']
        if updated and up > tg:
            rx = -rx
    group['rotation_x'], group['rotation_y'], group['rotation_z'] = rx, ry, rz
    return group


def map_type(id_target, id_update):
    if not pd.isna(id_update):
        diff = abs(round(id_update) - round(id_target))
        if diff > 4:
            diff = 8 - diff
        return {1: 45, 2: 90, 3: 135, 4: 180}.get(diff, 0)
    return 0


def distance_and_angle_to_end(group):
    ex, ey, ez = group['rotation_x'].iloc[-1], group['rotation_y'].iloc[-1], group['rotation_z'].iloc[-1]
    group['distance_to_end'] = np.sqrt((group['rotation_x'] - ex)**2 +
                                       (group['rotation_y'] - ey)**2 +
                                       (group['rotation_z'] - ez)**2)
    return group


def extract_features(df):
    trials = [process_trial(df[df['id'] == i]) for i in df['id'].unique()]
    proc = pd.concat(trials, ignore_index=True).dropna(subset=['three_point_angle'])
    proc = proc.groupby('id').filter(reaction_time_ok)
    proc = proc.groupby('id').filter(lambda g: g['movement_duration_miliseconds'].max() <= MAX_MOVEMENT_DURATION)
    win = pd.concat([window_and_normalize(proc[proc['id'] == i]) for i in proc['id'].unique()], ignore_index=True)
    win = win.groupby('id', group_keys=False).apply(smooth)
    win = win.groupby('id', group_keys=False).apply(recenter)
    win = win.groupby('id', group_keys=False).apply(rotate)
    win['type_trajectory'] = win.apply(lambda r: map_type(r['id_target'], r['id_update']), axis=1)
    win = win.groupby('id', group_keys=False).apply(distance_and_angle_to_end)
    win['angle'] = win['three_point_angle']

    start_move, stop_turn = {}, {}
    for i in win['id'].unique():
        tr = win[win['id'] == i]
        sm = tr.iloc[0]['normalized_start_movement']
        if not pd.isna(sm):
            idx = (tr['normalized_time'] - sm).abs().idxmin()
            start_move[i] = tr.at[idx, 'normalized_time']
        if 'normalized_stop_turn' in tr.columns and not pd.isna(tr.iloc[0]['normalized_stop_turn']):
            idx = (tr['normalized_time'] - tr.iloc[0]['normalized_stop_turn']).abs().idxmin()
            stop_turn[i] = tr.at[idx, 'normalized_time']
    return win, start_move, stop_turn


if __name__ == '__main__':
    raw = pd.read_pickle('data/raw_example_subset.pkl')   # per-timestep raw dataframe
    df_features, start_move, stop_turn = extract_features(raw)
    with open('data/features_example.pkl', 'wb') as f:
        pickle.dump((df_features, start_move, stop_turn), f)
    print('trials:', df_features['id'].nunique(), '| onset labels:', len(start_move),
          '| turning labels:', len(stop_turn))
