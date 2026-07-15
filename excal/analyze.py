"""End-to-end workout video analysis.

video -> pose keypoints -> features -> windowed classification (with idle
gating + temporal smoothing) -> contiguous exercise segments -> rep counts
-> MET calorie estimates -> report dict.

Usage:
  python -m excal.analyze video.mp4 --weight-kg 70 [--max-seconds 120]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from excal.calories.met import calories
from excal.features.build import FEATURE_NAMES, frame_features
from excal.model.net import ExerciseNet
from excal.pose.extract import extract_keypoints
from excal.repcount.counter import count_reps

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "exercise_net.pt"
WINDOW, STRIDE = 45, 15
CONF_MIN = 0.6  # softmax confidence below this -> idle
VEL_MIN = 15.0  # mean |angle velocity| (deg/s) below this -> idle
MIN_SEGMENT_S = 4.0


def load_model(weights: Path = WEIGHTS):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = ExerciseNet(len(ckpt["mean"]), len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def classify_windows(feats: np.ndarray, model, ckpt) -> list[str | None]:
    """Label per window start (None = idle/unknown)."""
    n_vel = len([n for n in FEATURE_NAMES if n.endswith("_vel")])
    labels = []
    for s in range(0, len(feats) - WINDOW + 1, STRIDE):
        win = feats[s : s + WINDOW]
        if np.abs(win[:, -n_vel:]).mean() < VEL_MIN:
            labels.append(None)
            continue
        x = (win - ckpt["mean"]) / ckpt["std"]
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(x[None])), 1)[0]
        conf, cls = probs.max(0)
        labels.append(ckpt["classes"][cls] if conf >= CONF_MIN else None)
    return labels


def smooth(labels: list, k: int = 2) -> list:
    """Majority vote over a +-k neighborhood (idle wins ties)."""
    out = []
    for i in range(len(labels)):
        votes = [l for l in labels[max(0, i - k) : i + k + 1] if l]
        if not votes:
            out.append(None)
            continue
        best = max(set(votes), key=votes.count)
        out.append(best if votes.count(best) > k else None)
    return out


def segments_from_labels(labels: list, fps: float) -> list[tuple[str, int, int]]:
    """Merge consecutive same-label windows -> (label, start_frame, end_frame)."""
    segs = []
    for i, lab in enumerate(labels):
        start, end = i * STRIDE, i * STRIDE + WINDOW
        if lab and segs and segs[-1][0] == lab and start <= segs[-1][2]:
            segs[-1] = (lab, segs[-1][1], end)
        elif lab:
            segs.append((lab, start, end))
    return [s for s in segs if (s[2] - s[1]) / fps >= MIN_SEGMENT_S]


def analyze(video_path: str | Path, weight_kg: float, max_seconds: float | None = None) -> dict:
    data = extract_keypoints(video_path, sample_fps=15.0, max_seconds=max_seconds)
    fps = float(data["fps"])
    if not len(data["timestamps"]):
        return {"video": str(video_path), "segments": [], "totals": {"calories": 0.0}}

    feats = frame_features(data["landmarks"], fps)
    model, ckpt = load_model()
    labels = smooth(classify_windows(feats, model, ckpt))

    segments, total_cal, total_reps = [], 0.0, 0
    per_exercise: dict[str, dict] = {}
    for lab, s, e in segments_from_labels(labels, fps):
        dur = (e - s) / fps
        reps = count_reps(feats[s:e], lab, fps)["reps"] if lab != "plank" else 0
        cal = calories(lab, weight_kg, dur)
        segments.append(
            {
                "exercise": lab,
                "start_s": round(float(data["timestamps"][s]), 1),
                "end_s": round(float(data["timestamps"][min(e, len(feats) - 1)]), 1),
                "duration_s": round(dur, 1),
                "reps": reps,
                "calories": round(cal, 2),
            }
        )
        total_cal += cal
        total_reps += reps
        agg = per_exercise.setdefault(lab, {"reps": 0, "duration_s": 0.0, "calories": 0.0})
        agg["reps"] += reps
        agg["duration_s"] = round(agg["duration_s"] + dur, 1)
        agg["calories"] = round(agg["calories"] + cal, 2)

    return {
        "video": str(video_path),
        "weight_kg": weight_kg,
        "segments": segments,
        "per_exercise": per_exercise,
        "totals": {"reps": total_reps, "calories": round(total_cal, 2)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--weight-kg", type=float, default=70.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()
    print(json.dumps(analyze(args.video, args.weight_kg, args.max_seconds), indent=2))


if __name__ == "__main__":
    main()
