"""Video -> pose keypoint extraction using MediaPipe PoseLandmarker (Tasks API).

Output: .npz with
  landmarks: (n_frames, 33, 4) float32 — x, y, z, visibility (x/y normalized to [0,1])
  timestamps: (n_frames,) float64 — seconds from video start
  frame_indices: (n_frames,) int64 — source frame index (frames with no detection are skipped)
  fps: float — effective sampling fps

Requires the model file weights/pose_landmarker_full.task:
  curl -sL -o weights/pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

N_LANDMARKS = 33
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "weights" / "pose_landmarker_full.task"

# Subset of the POSE_CONNECTIONS skeleton for the debug overlay.
SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]


def _draw_skeleton(frame: np.ndarray, lms: list) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in SKELETON:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def extract_keypoints(
    video_path: str | Path,
    sample_fps: float | None = None,
    overlay_path: str | Path | None = None,
    max_seconds: float | None = None,
    model_path: str | Path = DEFAULT_MODEL,
) -> dict:
    """Run PoseLandmarker over a video.

    sample_fps: downsample to this rate (None = every frame).
    overlay_path: if set, write a debug video with the skeleton drawn.
    max_seconds: stop after this many seconds of video (useful for long files).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1 if sample_fps is None else max(1, round(src_fps / sample_fps))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    writer = None
    landmarks, timestamps, frame_indices = [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = frame_idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break
            if frame_idx % step:
                continue

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(mp_img, int(t * 1000))
            detected = bool(result.pose_landmarks)

            if overlay_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(overlay_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        src_fps / step,
                        (w, h),
                    )
                if detected:
                    _draw_skeleton(frame, result.pose_landmarks[0])
                writer.write(frame)

            if not detected:
                continue
            landmarks.append(
                [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks[0]]
            )
            timestamps.append(t)
            frame_indices.append(frame_idx)

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "landmarks": np.asarray(landmarks, dtype=np.float32).reshape(-1, N_LANDMARKS, 4),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": src_fps / step,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out-dir", default="data/keypoints")
    ap.add_argument("--sample-fps", type=float, default=15.0)
    ap.add_argument("--overlay", action="store_true", help="write debug overlay video")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    overlay = out_dir / f"{stem}_overlay.mp4" if args.overlay else None

    data = extract_keypoints(
        args.video,
        sample_fps=args.sample_fps,
        overlay_path=overlay,
        max_seconds=args.max_seconds,
    )
    out = out_dir / f"{stem}.npz"
    np.savez_compressed(out, **data)
    n = len(data["timestamps"])
    dur = data["timestamps"][-1] if n else 0.0
    print(f"{n} frames with pose over {dur:.1f}s -> {out}")
    if overlay:
        print(f"overlay -> {overlay}")


if __name__ == "__main__":
    main()
