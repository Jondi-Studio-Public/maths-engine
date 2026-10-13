"""Worker processes that run engine calls with a time limit.

Generating or grading can, rarely, run for a very long time: SymPy simplification has inputs that never finish
(`mathgen/grading/safe_parse.py` lists some), and nothing inside Python can stop a running thread. So the server
runs every call that generates or grades in a separate process, and a process that misses its deadline is killed
and replaced. The caller gets `Timeout`, the other workers carry on, and the server stays up.

Each worker loads the skill library once when it starts (a second or so), so a deadline counts only the call.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import queue
import sys
import threading
import time

from .. import service

_JOBS = {}
log = logging.getLogger("mathgen.server")


def job(fn):
    """Register `fn(lib, **kwargs)` as a job a worker can run by name."""
    _JOBS[fn.__name__] = fn
    return fn


class Timeout(Exception):
    """A call ran past its deadline; its worker was stopped and replaced."""


class WorkerError(Exception):
    """A call failed inside a worker for a reason that is not the caller's: a bug, an exhausted generator."""


class Unavailable(Exception):
    """No worker became free before the deadline: the server is busy."""


def _limit_memory(megabytes: int) -> None:
    """Cap this process's address space, so one runaway call raises `MemoryError` instead of taking the host's
    memory. POSIX only: `resource` does not exist on Windows, where the cap is skipped."""
    if megabytes <= 0:
        return
    try:
        import resource
    except ImportError:
        return
    limit = megabytes * 1024 * 1024
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ValueError, OSError):  # a platform that refuses (macOS, say): run uncapped rather than not at all
        log.warning("could not cap a worker's memory at %d MB", megabytes)


def _worker_main(conn, settings):
    # Anything a job prints goes to stderr, never to the parent's stdout (which a stdio caller may be reading).
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    root, extra, id_secret, memory_mb = settings
    _limit_memory(memory_mb)
    lib = service.Library.load(root, extra, id_secret=id_secret)
    from . import jobs  # noqa: F401  (registers the jobs)

    conn.send(("ready", lib.version()))
    while True:
        try:
            name, kwargs = conn.recv()
        except EOFError:
            return
        try:
            conn.send(("ok", _JOBS[name](lib, **kwargs)))
        except service.QuestionChanged as exc:
            conn.send(("changed", {"old_id": exc.old_id, "new_id": exc.new_id}))
        except service.NotFound as exc:
            conn.send(("not_found", str(exc)))
        except service.BadRequest as exc:
            conn.send(("bad_request", str(exc)))
        except MemoryError:  # the cap was hit; the worker's heap may be fragmented, so the pool replaces it
            conn.send(("memory", "the request needed more memory than a worker may use"))
        except Exception as exc:  # anything else is a bug; report it without the traceback's local values
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class _Worker:
    def __init__(self, ctx, settings, start_timeout):
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker_main, args=(child, settings), daemon=True)
        self.process.start()
        child.close()
        try:
            if not self.conn.poll(start_timeout):
                raise WorkerError("a worker did not start in time")
            status, self.version = self.conn.recv()
            if status != "ready":
                raise WorkerError("a worker failed to start")
        except (WorkerError, EOFError, OSError, ValueError):
            self.stop()
            raise WorkerError("a worker failed to start") from None

    def alive(self) -> bool:
        return self.process.is_alive()

    def stop(self):
        self.process.kill()
        self.process.join(5)
        self.conn.close()


class WorkerPool:
    """A fixed number of worker processes. Thread-safe: each call borrows one idle worker for its duration.

    A worker that misses a deadline or dies is replaced in the background, retrying until a replacement starts, so
    the pool comes back to full size on its own. `live` says how many workers are running now."""

    def __init__(
        self,
        workers: int = 2,
        *,
        root=None,
        extra_roots=None,
        id_secret: str | None = None,
        memory_mb: int = 0,
        start_timeout: float = 120.0,
    ):
        if workers < 1:
            raise ValueError("a pool needs at least one worker")
        # `spawn` everywhere, so a worker never inherits the server's threads or locks.
        self._ctx = multiprocessing.get_context("spawn")
        # Handed to each worker as a process argument (a pipe, not the command line or the environment).
        self._settings = (root, extra_roots, id_secret, memory_mb)
        self._start_timeout = start_timeout
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._all: list[_Worker] = []
        self._lock = threading.Lock()
        self._closed = False
        self.size = workers
        # Start the workers side by side: each spends a few seconds importing the engine and loading skills.
        starters = [threading.Thread(target=self._try_add) for _ in range(workers)]
        for t in starters:
            t.start()
        for t in starters:
            t.join()
        if len(self._all) < workers:
            self.close()
            raise WorkerError(f"only {len(self._all)} of {workers} workers started")
        self.version = self._all[0].version

    @property
    def live(self) -> int:
        with self._lock:
            return sum(1 for w in self._all if w.alive())

    def _try_add(self) -> bool:
        try:
            worker = _Worker(self._ctx, self._settings, self._start_timeout)
        except WorkerError as exc:
            log.error("could not start a worker: %s", exc)
            return False
        with self._lock:
            if self._closed:
                worker.stop()
                return True
            self._all.append(worker)
        self._idle.put(worker)
        return True

    def _respawn(self):
        delay = 1.0
        while not self._closed and not self._try_add():
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    def _replace(self, worker):
        worker.stop()
        with self._lock:
            if worker in self._all:
                self._all.remove(worker)
            if self._closed:
                return
        # In the background, so the caller gets its answer now.
        threading.Thread(target=self._respawn, daemon=True).start()

    def _borrow(self, wait: float) -> _Worker:
        """An idle worker that is still running. One that died while idle (killed from outside, out of memory) is
        replaced, and the next is tried: nothing had been sent to it, so the call has lost nothing."""
        deadline = time.monotonic() + wait
        while True:
            try:
                worker = self._idle.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise Unavailable("every worker is busy") from None
            if worker.alive():
                return worker
            self._replace(worker)

    def run(self, name: str, timeout: float, *, wait: float = 30.0, **kwargs):
        """Run job `name` with `kwargs` in a worker and return its result, or raise: `Timeout`, `Unavailable`,
        `service.NotFound`, `service.BadRequest`, `Changed` (`service.QuestionChanged` across processes), or
        `WorkerError`. `wait` is how long to queue for a free worker; `timeout` starts once one has the job."""
        if name not in _JOB_NAMES:
            raise ValueError(f"unknown job {name!r}")
        worker = self._borrow(wait)
        dropped = None
        try:
            worker.conn.send((name, kwargs))
            if not worker.conn.poll(timeout):
                self._replace(worker)
                worker = None
                raise Timeout(f"{name} took longer than {timeout:g} s")
            status, value = worker.conn.recv()
            if status == "memory":  # keep the worker out of the idle queue; it is replaced below
                dropped, worker = worker, None
        except (EOFError, OSError, BrokenPipeError):
            if worker is not None:
                self._replace(worker)
                worker = None
            raise WorkerError("a worker stopped unexpectedly") from None
        finally:
            if worker is not None:
                self._idle.put(worker)
        if status == "ok":
            return value
        if status == "memory":
            self._replace(dropped)
            raise WorkerError(value)
        if status == "changed":
            raise Changed(**value)
        if status == "not_found":
            raise service.NotFound(value)
        if status == "bad_request":
            raise service.BadRequest(value)
        raise WorkerError(value)

    def close(self):
        with self._lock:
            self._closed = True
            workers = list(self._all)
            self._all.clear()
        for worker in workers:
            worker.stop()


class Changed(Exception):
    """The library no longer generates the question an id names (`service.QuestionChanged`, across processes)."""

    def __init__(self, old_id: str, new_id: str):
        super().__init__(f"question {old_id} now generates different content ({new_id})")
        self.old_id = old_id
        self.new_id = new_id


def _cgroup_cpus() -> int | None:
    """The CPU quota of this container (cgroup v2 `cpu.max`), rounded up, or None when there is none."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota == "max":
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        return None


def default_workers() -> int:
    """Workers to start: the CPUs this process may use (Python 3.13's `process_cpu_count`, else the affinity
    mask), within a container's CPU quota, at most 8. `os.cpu_count()` alone counts the host's CPUs, not the
    container's. Each worker holds the loaded library in memory (see `docs/server.md`)."""
    counter = getattr(os, "process_cpu_count", None)
    if counter is not None:
        cpus = counter()
    elif hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count()
    quota = _cgroup_cpus()
    if quota is not None:
        cpus = min(cpus or quota, quota)
    return max(1, min(8, cpus or 2))


# The job names a caller may ask for, fixed here so the parent process need not import the jobs module first.
_JOB_NAMES = frozenset({"make_questions", "question", "paper", "grade", "figure_svg"})
