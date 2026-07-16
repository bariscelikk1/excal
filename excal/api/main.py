"""FastAPI backend: upload a workout video, get an analysis report.

Run:
  .venv/bin/uvicorn excal.api.main:app --reload

POST /api/analyze  (multipart: file, weight_kg, optional max_seconds) -> {job_id}
GET  /api/jobs/{job_id} -> {status: queued|running|done|error, result?}
GET  /api/jobs/{job_id}/video -> annotated mp4 (skeleton tracking + reps/kcal HUD)
GET  /       -> web UI
"""

import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from excal.analyze import analyze

app = FastAPI(title="excal")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
OVERLAY_DIR = Path(tempfile.gettempdir()) / "excal_overlays"
OVERLAY_DIR.mkdir(exist_ok=True)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _run_job(job_id: str, video: Path, weight_kg: float, max_seconds: float | None) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"
    overlay = OVERLAY_DIR / f"{job_id}.mp4"
    try:
        result = analyze(video, weight_kg, max_seconds, overlay_path=overlay)
        result["video"] = _jobs[job_id]["filename"]
        with _lock:
            _jobs[job_id].update(
                status="done", result=result,
                overlay=str(overlay) if overlay.exists() else None,
            )
    except Exception as e:  # surfaced to the client via job status
        with _lock:
            _jobs[job_id].update(status="error", error=str(e))
    finally:
        video.unlink(missing_ok=True)


@app.post("/api/analyze")
async def submit(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    weight_kg: float = Form(70.0),
    max_seconds: float | None = Form(None),
):
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    with _lock:
        _jobs[job_id] = {"status": "queued", "filename": file.filename}
    background.add_task(_run_job, job_id, tmp, weight_kg, max_seconds)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id, {"status": "unknown"})
        return {k: v for k, v in job.items() if k != "overlay"} | {
            "has_video": bool(job.get("overlay"))
        }


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    with _lock:
        overlay = _jobs.get(job_id, {}).get("overlay")
    if not overlay or not Path(overlay).exists():
        raise HTTPException(404, "no annotated video for this job")
    return FileResponse(overlay, media_type="video/mp4", filename="excal_tracked.mp4")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
