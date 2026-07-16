"""Render an annotated copy of the analyzed video.

Draws the pose skeleton (green tracking lines) on every analyzed frame plus
a live HUD: current exercise, rep count for the segment, and running totals
(reps + kcal). Reuses the landmarks from the analysis pass, so the video is
re-read but pose estimation is not re-run.
"""

from pathlib import Path

import cv2
import numpy as np

from excal.calories.met import MET
from excal.pose.extract import SKELETON

GREEN = (0, 255, 0)
RED = (0, 60, 230)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
REP_FLASH_S = 0.35  # highlight window after each counted rep

NICE = {
    "jumping_jack": "Jumping jacks",
    "pull_up": "Pull-ups",
    "push_up": "Push-ups",
    "situp": "Sit-ups",
    "squat": "Squats",
}


def _per_frame_state(n_frames: int, seg_infos: list[dict], weight_kg: float, fps: float):
    """Precompute HUD values for each sampled frame index.

    seg_infos: [{"label", "start", "end", "rep_frames"}] in sampled-frame coords.
    Returns (label, seg_reps, total_reps, kcal, rep_flash) arrays.
    """
    label = np.full(n_frames, None, dtype=object)
    seg_reps = np.zeros(n_frames, dtype=int)
    total_reps = np.zeros(n_frames, dtype=int)
    kcal = np.zeros(n_frames)
    rep_flash = np.zeros(n_frames, dtype=bool)
    flash_n = max(1, int(REP_FLASH_S * fps))

    cal_rate = {ex: MET[ex] * weight_kg / 3600.0 for ex in MET}  # kcal per second
    reps_done, cal_done = 0, 0.0
    for seg in sorted(seg_infos, key=lambda s: s["start"]):
        s, e, lab = seg["start"], min(seg["end"], n_frames), seg["label"]
        idx = np.arange(s, e)
        label[idx] = lab
        kcal[idx] = cal_done + (idx - s) / fps * cal_rate[lab]
        reps_in_seg = np.zeros(e - s, dtype=int)
        for rf in seg["rep_frames"]:
            if rf < e - s:
                reps_in_seg[rf:] += 1
                rep_flash[s + rf : min(s + rf + flash_n, n_frames)] = True
        seg_reps[idx] = reps_in_seg
        total_reps[idx] = reps_done + reps_in_seg
        reps_done += int(reps_in_seg[-1]) if len(reps_in_seg) else 0
        cal_done += (e - s) / fps * cal_rate[lab]
        if e < n_frames:
            total_reps[e:] = reps_done
            kcal[e:] = cal_done
    return label, seg_reps, total_reps, kcal, rep_flash


def _draw_skeleton(frame: np.ndarray, lms: np.ndarray, flash: bool) -> None:
    """lms: (33, 4) landmark array with normalized x, y."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lms[:, :2]]
    thick = 3 if flash else 2
    for a, b in SKELETON:
        cv2.line(frame, pts[a], pts[b], GREEN, thick, cv2.LINE_AA)
    for a, b in SKELETON:  # joints only where the skeleton connects
        for p in (pts[a], pts[b]):
            cv2.circle(frame, p, 4, RED, -1, cv2.LINE_AA)


def _draw_hud(frame, lab, reps, total, cal, flash, scale):
    h, w = frame.shape[:2]
    pad, lh = int(12 * scale), int(26 * scale)
    lines = (
        [(NICE.get(lab, lab), GREEN), (f"Reps: {reps}", WHITE)]
        if lab
        else [("idle", (180, 180, 180))]
    )
    lines.append((f"Total: {total} reps  |  {cal:.1f} kcal", WHITE))

    box_w = int(310 * scale)
    box_h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = pad * 2 + lh * i + int(14 * scale)
        cv2.putText(frame, text, (pad * 2, y), FONT, 0.62 * scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
    if flash and lab:
        cv2.circle(frame, (pad + box_w - int(20 * scale), pad + int(24 * scale)), int(9 * scale), GREEN, -1, cv2.LINE_AA)


def render_annotated(
    video_path: str | Path,
    out_path: str | Path,
    landmarks: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    seg_infos: list[dict],
    weight_kg: float,
) -> Path:
    """Write an H.264 mp4 of the sampled frames with skeleton + HUD overlay."""
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    label, seg_reps, total_reps, kcal, rep_flash = _per_frame_state(
        len(frame_indices), seg_infos, weight_kg, fps
    )

    writer, ptr, frame_idx = None, 0, -1
    while ptr < len(frame_indices):
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx != frame_indices[ptr]:
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h)
            )
            scale = max(0.75, min(w, h) / 720)
        _draw_skeleton(frame, landmarks[ptr], bool(rep_flash[ptr]))
        _draw_hud(
            frame, label[ptr], int(seg_reps[ptr]), int(total_reps[ptr]),
            float(kcal[ptr]), bool(rep_flash[ptr]), scale,
        )
        writer.write(frame)
        ptr += 1

    cap.release()
    if writer is not None:
        writer.release()
    return out_path
