# Kinematic event detection and HFS decoding in primate reaching

Code for detecting movement onset and corrective turning points from 3D reaching
trajectories, aligning trajectories across animals, and decoding control vs. HFS
trials.

## Pipeline

`feature_extraction.py` (raw tracking -> features) -> `event_detection.py` /
`hfs_decoding.py` (models). `inter_animal_alignment.py` is used for between-animal
evaluation.

## Scripts

- `feature_extraction.py` -- computes trajectory features (velocity, three-point angle,
  curvature, distance-to-endpoint, normalized time) from raw hand positions, applies the
  reaction-time and movement-duration filters, windows and normalizes each trial, smooths,
  re-centers, and rotates to a common within-animal reference, and produces the
  movement-onset and turning-point reference labels. Outputs `(df_features, start_move,
  stop_turn)`.
- `event_detection.py` — BiLSTM event detector (movement onset and turning point)
  with per-trial uncertainty (Gaussian negative log-likelihood). Trains within an
  animal and evaluates within- and between-animal. Features are z-scored using
  training-set statistics only.
- `inter_animal_alignment.py` — rotation-constrained Procrustes (Kabsch) alignment
  that maps one animal's workspace onto the other using per-target endpoint directions.
- `hfs_decoding.py` — within-animal decoding of HFS vs. control trials (10 runs),
  plus a session-grouped evaluation in which no recording session appears in both
  training and test.

## Data

A small example subset is provided under `data/` to illustrate the expected format and
let the scripts run. The complete dataset is available from the corresponding author on
request; the reported values are obtained when the scripts are run on the full dataset.

Expected inputs:
- Event detection: a pickle containing `(df_features, start_move, stop_turn)`, where
  `df_features` has one row per timestep with the feature columns, `id`, `id_target`,
  `type_trajectory`, and `adjusted_time`; `start_move` / `stop_turn` map trial `id` to
  the normalized event time.
- HFS decoding: the processed trajectory pickle with per-trial position and velocity.

## Requirements

See `requirements.txt`.

```
pip install -r requirements.txt
```

## Running

```
python event_detection.py
python hfs_decoding.py
```
