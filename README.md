# excal

Analyze workout videos: detect exercises (push-ups, squats, sit-ups, jumping jacks), count reps, and estimate calories burned.

## How it works

```
video → MediaPipe pose keypoints → joint-angle features → exercise classifier (PyTorch)
      → rep counter (angle cycle detection) → MET-based calorie estimate → report
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Extract keypoints + debug overlay video
.venv/bin/python -m excal.pose.extract path/to/video.mp4 -o data/keypoints/

# Full analysis (classifier + reps + calories)
.venv/bin/python -m excal.analyze path/to/video.mp4 --weight-kg 70
```

## Project layout

- `excal/pose/` — video → keypoint extraction (MediaPipe)
- `excal/features/` — joint angles, normalization, windowing
- `excal/repcount/` — rep counting via angle cycle detection
- `excal/model/` — exercise classifier (training + inference)
- `excal/calories/` — MET-based calorie calculation
- `excal/api/` — FastAPI backend
- `web/` — React frontend
