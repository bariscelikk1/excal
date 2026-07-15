"""FastAPI backend: upload a workout video, get an analysis report.

Run:
  .venv/bin/uvicorn excal.api.main:app --reload

POST /api/analyze  (multipart: file, weight_kg, optional max_seconds) -> {job_id}
GET  /api/jobs/{job_id} -> {status: queued|running|done|error, result?}
GET  /       -> web UI
"""

import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from excal.analyze import analyze

app = FastAPI(title="excal")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _run_job(job_id: str, video: Path, weight_kg: float, max_seconds: float | None) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = analyze(video, weight_kg, max_seconds)
        result["video"] = _jobs[job_id]["filename"]
        with _lock:
            _jobs[job_id].update(status="done", result=result)
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
        return _jobs.get(job_id, {"status": "unknown"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
