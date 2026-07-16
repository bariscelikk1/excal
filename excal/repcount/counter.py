"""Rep counting via cycle detection on the exercise's primary signal.

Each exercise has one scalar signal per frame (e.g. mean elbow angle for
push-ups). A rep = one full valley in that signal: we detect minima with
scipy.signal.find_peaks on the inverted, smoothed signal, requiring a
minimum prominence so small wobbles don't count.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from excal.features.build import FEATURE_NAMES

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# exercise -> (feature columns averaged into the signal, min prominence,
#              min distance between reps in seconds, invert)
# invert=False counts valleys (angle dips), True counts peaks (signal rises).
REP_CONFIG = {
    "push_up": (["l_elbow", "r_elbow"], 30.0, 0.8, False),
    "pull_up": (["l_elbow", "r_elbow"], 30.0, 0.8, False),
    "squat": (["l_knee", "r_knee"], 30.0, 0.8, False),
    "situp": (["l_hip", "r_hip"], 25.0, 0.8, False),
    "jumping_jack": (["arm_spread"], 0.8, 0.4, True),
}


def rep_signal(feats: np.ndarray, exercise: str) -> np.ndarray:
    cols, *_ = REP_CONFIG[exercise]
    return feats[:, [_IDX[c] for c in cols]].mean(axis=1)


def count_reps(feats: np.ndarray, exercise: str, fps: float) -> dict:
    """Count reps in a contiguous segment of frame features.

    Returns {"reps": int, "rep_frames": [frame indices of rep bottoms]}.
    """
    cols, prominence, min_gap_s, invert = REP_CONFIG[exercise]
    sig = rep_signal(feats, exercise)
    sig = uniform_filter1d(sig, size=max(3, int(fps * 0.2)))
    peaks, _ = find_peaks(
        sig if invert else -sig,
        prominence=prominence,
        distance=max(1, int(min_gap_s * fps)),
    )
    return {"reps": int(len(peaks)), "rep_frames": peaks.tolist()}
