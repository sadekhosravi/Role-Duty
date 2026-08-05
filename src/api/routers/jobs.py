"""Watching the background work.

The other half of the ingest endpoints: they hand back a job id, this is where
it means something. Read-only — there is no cancel, because the only cancellable
work is an ingest and stopping one halfway leaves a partly-indexed document that
is harder to reason about than one that finished.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import JobsDep
from ..schemas import ErrorResponse, JobList, JobView

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "",
    response_model=JobList,
    summary="Every job still remembered, newest first",
    description=(
        "The registry is bounded and in memory: finished jobs are dropped "
        "oldest-first once it is full, and everything is gone on restart."
    ),
)
async def list_jobs(jobs: JobsDep) -> JobList:
    return JobList(jobs=[JobView.of(job) for job in jobs.list()])


@router.get(
    "/{job_id}",
    response_model=JobView,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="One job — poll this after an ingest",
    description=(
        "`status` goes queued -> running -> succeeded or failed. On success "
        "`result` holds what was ingested and where; on failure `error` holds "
        "the exception, since there was no request left to raise it into.\n\n"
        "A 404 means the job never existed or has been evicted — not that it "
        "failed."
    ),
)
async def get_job(job_id: str, jobs: JobsDep) -> JobView:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such job: {job_id}")
    return JobView.of(job)
