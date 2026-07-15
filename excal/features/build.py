"""Turn raw pose landmark sequences into classifier features.

Input: landmarks (n_frames, 33, 4) from excal.pose.extract.
Feature vector per frame (FEATURE_NAMES order):
  8 joint angles (degrees), torso inclination, 2 spread ratios,
  then per-frame angle velocities (deg/s) for the 8 joints.
"""

import numpy as np

# MediaPipe pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# (name, a, b, c): angle at b between rays b->a and b->c
ANGLE_DEFS = [
    ("l_elbow", L_SHOULDER, L_ELBOW, L_WRIST),
    ("r_elbow", R_SHOULDER, R_ELBOW, R_WRIST),
    ("l_knee", L_HIP, L_KNEE, L_ANKLE),
    ("r_knee", R_HIP, R_KNEE, R_ANKLE),
    ("l_hip", L_SHOULDER, L_HIP, L_KNEE),
    ("r_hip", R_SHOULDER, R_HIP, R_KNEE),
    ("l_shoulder", L_ELBOW, L_SHOULDER, L_HIP),
    ("r_shoulder", R_ELBOW, R_SHOULDER, R_HIP),
]

FEATURE_NAMES = (
    [name for name, *_ in ANGLE_DEFS]
    + ["torso_incline", "arm_spread", "leg_spread"]
    + [f"{name}_vel" for name, *_ in ANGLE_DEFS]
)
N_STATIC = len(ANGLE_DEFS) + 3


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (deg) at b between rays b->a, b->c. Inputs (n, 2)."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (
        np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def frame_features(landmarks: np.ndarray, fps: float) -> np.ndarray:
    """(n_frames, 33, 4) -> (n_frames, len(FEATURE_NAMES)) float32."""
    xy = landmarks[..., :2].astype(np.float64)

    static = np.empty((len(xy), N_STATIC))
    for i, (_, a, b, c) in enumerate(ANGLE_DEFS):
        static[:, i] = _angle(xy[:, a], xy[:, b], xy[:, c])

    mid_shoulder = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    torso = mid_shoulder - mid_hip
    # 0 deg = upright (image y grows downward), 90 = horizontal (e.g. push-up plank)
    static[:, 8] = np.degrees(np.arctan2(np.abs(torso[:, 0]), -torso[:, 1] + 1e-8))

    torso_len = np.linalg.norm(torso, axis=-1) + 1e-8
    static[:, 9] = np.linalg.norm(xy[:, L_WRIST] - xy[:, R_WRIST], axis=-1) / torso_len
    static[:, 10] = np.linalg.norm(xy[:, L_ANKLE] - xy[:, R_ANKLE], axis=-1) / torso_len

    vel = np.gradient(static[:, : len(ANGLE_DEFS)], axis=0) * fps
    return np.concatenate([static, vel], axis=1).astype(np.float32)


def make_windows(
    feats: np.ndarray, window: int = 45, stride: int = 15
) -> np.ndarray:
    """(n_frames, d) -> (n_windows, window, d). At 15 fps, 45 frames = 3 s."""
    if len(feats) < window:
        return np.empty((0, window, feats.shape[1]), dtype=feats.dtype)
    starts = range(0, len(feats) - window + 1, stride)
    return np.stack([feats[s : s + window] for s in starts])


def window_starts(n_frames: int, window: int = 45, stride: int = 15) -> np.ndarray:
    return np.arange(0, max(n_frames - window + 1, 0), stride)
