# app/jobs.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import uuid4
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

executor = ThreadPoolExecutor(max_workers=2)

_jobs: Dict[str, "Job"] = {}
_jobs_lock = Lock()


@dataclass
class Job:
    id: str
    name: str
    status: str  # queued | running | success | failed
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _log(job: Job, message: str) -> None:
    job.logs.append(f"[{_now()}] {message}")


def submit_job(name: str, fn: Callable[[], Any]) -> Job:
    job_id = str(uuid4())
    job = Job(id=job_id, name=name, status="queued", created_at=_now())

    with _jobs_lock:
        _jobs[job_id] = job

    def _run():
        job.status = "running"
        job.started_at = _now()
        _log(job, "Job started")

        try:
            job.result = fn()
            job.status = "success"
            _log(job, "Job finished successfully")
        except Exception:
            job.status = "failed"
            job.error = traceback.format_exc()
            _log(job, "Job failed")
        finally:
            job.finished_at = _now()

    executor.submit(_run)
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> Dict[str, Dict[str, Any]]:
    with _jobs_lock:
        return {jid: j.to_dict() for jid, j in _jobs.items()}