"""Build training windows from the Kaggle exercise-recognition time-series dataset.

Dataset: muhannadtuameh/exercise-recognition-time-series (via kagglehub).
Landmarks are hip-centered world coords, y down (same orientation as image
coords), ~30 fps. We downsample x2 to ~15 fps to match inference, run the
same feature pipeline as inference, and window per video (no window ever
spans two videos; train/val split is by video to avoid leakage).
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from excal.features.build import frame_features, make_windows
from excal.model.net import CLASSES

DATASET_FPS = 30.0
TARGET_FPS = 15.0
N_LANDMARKS = 33


def load_dataset(root: str | Path, window: int = 45, stride: int = 15):
    """Returns X (n, window, d), y (n,), groups (n,) video ids."""
    root = Path(root)
    with open(root / "labels.csv") as f:
        labels = {r["vid_id"]: r["class"] for r in csv.DictReader(f)}

    frames = defaultdict(list)
    with open(root / "landmarks.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        # columns: vid_id, frame_order, then x_/y_/z_ per landmark in order
        assert len(header) == 2 + 3 * N_LANDMARKS, header[:5]
        for row in reader:
            frames[row[0]].append((int(row[1]), [float(v) for v in row[2:]]))

    step = round(DATASET_FPS / TARGET_FPS)
    X, y, groups = [], [], []
    for vid, rows in frames.items():
        rows.sort(key=lambda r: r[0])
        coords = np.asarray([r[1] for r in rows], dtype=np.float32)
        coords = coords.reshape(-1, N_LANDMARKS, 3)[::step]
        # append visibility=1 to match the (33, 4) landmark format
        lms = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
        feats = frame_features(lms, TARGET_FPS)
        wins = make_windows(feats, window, stride)
        X.append(wins)
        y.extend([CLASSES.index(labels[vid])] * len(wins))
        groups.extend([vid] * len(wins))

    return np.concatenate(X), np.asarray(y), np.asarray(groups)


def split_by_video(y, groups, val_frac: float = 0.2, seed: int = 0):
    """Boolean masks (train, val) splitting whole videos, stratified by class."""
    rng = np.random.default_rng(seed)
    vids = np.unique(groups)
    val_vids = set()
    vid_cls = {v: y[groups == v][0] for v in vids}
    for c in np.unique(y):
        cvids = [v for v in vids if vid_cls[v] == c]
        rng.shuffle(cvids)
        val_vids.update(cvids[: max(1, int(len(cvids) * val_frac))])
    val = np.isin(groups, list(val_vids))
    return ~val, val
