# app/jobs.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import uuid4
import traceback
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)  # ajust according CPU

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

_jobs: Dict[str, Job] = {}

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def submit_job(name: str, fn: Callable[[], Any]) -> Job:
    job_id = str(uuid4())
    job = Job(id=job_id, name=name, status="queued", created_at=_now())
    _jobs[job_id] = job

    def _run():
        job.status = "running"
        job.started_at = _now()
        try:
            job.result = fn()
            job.status = "success"
        except Exception:
            job.status = "failed"
            job.error = traceback.format_exc()
        finally:
            job.finished_at = _now()

    executor.submit(_run)
    return job

def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)

def list_jobs() -> Dict[str, Dict[str, Any]]:
    # uuseful in debugging mode
    return {jid: j.to_dict() for jid, j in _jobs.items()}
