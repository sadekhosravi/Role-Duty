"""Background jobs, for the work that cannot finish inside a request.

Ingesting one PDF means Docling parsing it and then — on the graph side — an LLM
extracting entities from every section. That is tens of seconds at best and many
minutes for a folder. Answering it synchronously would mean a request held open
past every proxy's idle timeout, a Swagger tab that looks hung, and a client
with no way to ask how far along it is or whether it failed.

So ingest returns `202 Accepted` with a job id, and the job is polled. That is
the ordinary shape for this, and it is worth being clear about what this
implementation is and is not:

* Jobs run as asyncio tasks in the serving process. There is no queue and no
  worker — restart the server and running jobs are gone, unrecorded.
* The registry is in memory and bounded. Finished jobs are evicted oldest-first
  once `max_jobs` is reached, so a job id is not a permanent handle.
* Nothing is deduplicated. Submitting the same ingest twice runs it twice.

Every one of those is fine for a single-node learning project and none of them
is fine for a real one; the upgrade is a task queue (Celery, arq, RQ) behind the
same two endpoints, which is why the endpoints look like this now.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Where a job is. `str` so it serializes as its own name, not as an int."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class Job:
    """One unit of background work, and everything a poller needs to know."""

    id: str
    kind: str  # what was asked for, e.g. "ingest.graph"
    target: str  # what it was asked to do it to, e.g. "data/raw/x.pdf"
    status: JobStatus = JobStatus.queued
    submitted_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """How long it ran, once it has. None while queued."""
        if self.started_at is None:
            return None
        return ((self.finished_at or _now()) - self.started_at).total_seconds()


class JobRegistry:
    """Submits jobs, tracks them, and keeps the last `max_jobs` of them.

    Kept as a class with an instance on the app rather than module globals, so
    the app can be constructed twice — in a test, or beside another — without
    the two sharing a job list.
    """

    def __init__(self, max_jobs: int = 100) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}
        self._max_jobs = max_jobs

    def submit(
        self,
        kind: str,
        target: str,
        work: Callable[[], Awaitable[dict[str, Any]]],
    ) -> Job:
        """Start `work` in the background and return the job tracking it.

        `work` is a zero-argument coroutine function rather than a coroutine,
        so nothing is created until there is a job to attribute it to — a
        coroutine built and then dropped on an error here would warn about
        never being awaited, which is a confusing way to report a failed
        submission.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, target=target)
        self._jobs[job.id] = job
        self._evict()

        task = asyncio.create_task(self._run(job, work), name=f"{kind}:{job.id}")
        # Held for the job's lifetime. asyncio keeps only a weak reference to a
        # running task, so a task nothing else refers to can be garbage
        # collected mid-flight and the ingest simply stops happening.
        self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.id, None))
        return job

    async def _run(self, job: Job, work: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        """Run one job, and record how it ended whichever way that was."""
        job.status = JobStatus.running
        job.started_at = _now()
        try:
            job.result = await work()
            job.status = JobStatus.succeeded
        except asyncio.CancelledError:
            job.status = JobStatus.cancelled
            job.error = "cancelled"
            job.finished_at = _now()
            raise
        except Exception as error:  # noqa: BLE001 - recorded on the job, not raised
            # There is no request left to raise into: the client was answered at
            # submission. The failure has to land on the job or it lands nowhere,
            # so it is stored for the poller and logged for whoever is watching.
            job.status = JobStatus.failed
            job.error = f"{type(error).__name__}: {error}"
            log.exception("job %s (%s) failed", job.id, job.kind)
        finally:
            job.finished_at = job.finished_at or _now()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Every job still remembered, newest first."""
        return sorted(self._jobs.values(), key=lambda job: job.submitted_at, reverse=True)

    def _evict(self) -> None:
        """Drop the oldest finished jobs once the registry is full.

        Only finished ones: evicting a job that is still running would leave it
        executing with nothing tracking it, which is worse than the memory.
        """
        while len(self._jobs) > self._max_jobs:
            for job_id, job in self._jobs.items():
                if job.status in {JobStatus.queued, JobStatus.running}:
                    continue
                del self._jobs[job_id]
                break
            else:
                return  # everything left is in flight

    async def shutdown(self) -> None:
        """Cancel anything still running, and wait for it to unwind.

        Called from the app's lifespan. Without it the process can exit with
        tasks mid-write, and a half-written LightRAG store is a worse outcome
        than a cancelled ingest the user can simply run again.
        """
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
