"""Grade a reply with a time limit, in a worker process that is killed when it runs over.

    from mathgen.grading.deadline import grade_with_deadline

    result = grade_with_deadline("expression", question, "sqrt(acos(x)^2)", timeout=2.0)

`grade()` bounds the size and shape of what it parses (`safe_parse.py`), not the time SymPy then takes, so a
reply built to stay inside every limit can still run for minutes (`safe_parse.py`'s docstring lists some). A
service that grades replies from people it does not trust should grade them here, not with `grade()`.

Why a process: a thread cannot be stopped from outside, `signal.alarm` only works in the main thread and only on
POSIX, and both can only interrupt Python code, not a long call into C. A process can always be killed. The
standard library's `ProcessPoolExecutor` cannot kill one worker that is running over (3.14 added
`kill_workers()`, which stops all of them; this package supports 3.12), so `GradingPool` keeps its own workers:
each one answers on its own pipe, and a worker that misses its deadline is killed and replaced.

A reply that runs out of time is graded wrong, the same as any reply the grader cannot read, and the
result has `timed_out` set to True. The worker that ran over is killed and a fresh one starts in its place. A
deadline only counts the grading itself: a call waits for a worker to be ready before its clock starts.

Where the OS has it (Linux, macOS), workers start from a "forkserver" that has already imported the grader, so a
replacement is ready in milliseconds; elsewhere (Windows) each one is spawned and imports the engine itself, which
takes seconds. Either way a call that finds every worker busy waits for one, so a host that grades replies from
the public should limit how many replies one person can send at once: with the default 2 workers and 2 s, six
replies built to hang, sent together, finish at about 2, 2, 4, 4, 6 and 6 s, and honest replies queue behind them.

Workers are separate processes started with "spawn" or "forkserver", so a script that grades with a deadline must
start from `if __name__ == "__main__":`, as `multiprocessing` requires; without it no worker can start and the call
raises `WorkerStartError`. A web server or test runner imports the code as a module, so needs nothing.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import queue
import threading
import time
from dataclasses import dataclass

from .typed import GradeResult, grade

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 2.0  # seconds
# Replies one worker grades before the pool replaces it, as `maxtasksperchild` does for `multiprocessing.Pool`: it
# bounds the worker's loop and any memory SymPy's caches gather over a long-running service.
MAX_REPLIES_PER_WORKER = 10_000
_STARTUP_TIMEOUT = 120.0  # seconds a new worker may take to import the engine before it is given up on


class WorkerStartError(RuntimeError):
    """Worker processes started but none became ready to grade, twice running, so the reply was not graded.

    Almost always the calling script lacks `if __name__ == "__main__":`: each worker runs the script again and fails.
    Raised, not graded wrong, because the fault is in the host and nothing is known about the reply."""


@dataclass
class DeadlineResult(GradeResult):
    """`GradeResult` with `timed_out`: True when the grader ran out of time and the reply was graded wrong."""

    timed_out: bool = False


def _serve(conn, grade_fn) -> None:
    """A worker: say ready, then grade each (kind, question, given, skill) it is sent until the pipe closes or it has
    graded `MAX_REPLIES_PER_WORKER` replies (the pool retires it at that count, so the bound is never what ends it)."""
    conn.send("ready")
    for _ in range(MAX_REPLIES_PER_WORKER):
        try:
            job = conn.recv()
        except EOFError:
            return
        kind, question, given, skill = job
        try:
            conn.send(("ok", grade_fn(kind, question, given, skill=skill)))
        except Exception as exc:  # noqa: BLE001 - `grade` raises only for a bad call; the caller gets it back
            try:
                conn.send(("error", exc))
            except Exception:  # noqa: BLE001 - an exception that does not pickle
                conn.send(("error", RuntimeError(f"{type(exc).__name__}: {exc}")))


class _Worker:
    def __init__(self, context, grade_fn):
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, grade_fn), daemon=True)
        self.process.start()
        child.close()
        self.ready = False
        self.served = 0

    def wait_ready(self) -> None:
        if self.ready:
            return
        try:
            started = self.conn.poll(_STARTUP_TIMEOUT) and self.conn.recv() == "ready"
        except (EOFError, OSError):  # the worker died while starting
            started = False
        if not started:
            raise WorkerStartError("a grading worker did not start")
        self.ready = True

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.kill()
        self.process.join()
        self.conn.close()


class GradingPool:
    """A fixed number of worker processes that grade with a deadline. Safe to share between threads: a call waits
    for a free worker, so `workers` is also the most replies graded at once.

    `grade_fn` is what a worker calls, `grade` by default; it must be importable by name (a module-level function),
    because a worker imports it. `start_method` is a `multiprocessing` start method. The default is "forkserver",
    with the grader preloaded so a replacement worker starts at once, where the OS has it, and "spawn" elsewhere:
    both are safe in a process that already runs threads, which "fork" is not.
    """

    def __init__(
        self, workers: int = 2, *, timeout: float = DEFAULT_TIMEOUT, grade_fn=grade, start_method: str | None = None
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.timeout = timeout
        if start_method is None:
            start_method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        self._context = multiprocessing.get_context(start_method)
        if start_method == "forkserver":
            _preload_grader(self._context, grade_fn)
        self._grade_fn = grade_fn
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._lock = threading.Lock()
        self._all: set[_Worker] = set()
        self._closed = False
        for _ in range(workers):
            self._idle.put(self._start())

    def _start(self) -> _Worker:
        worker = _Worker(self._context, self._grade_fn)
        with self._lock:
            self._all.add(worker)
        return worker

    def _replace(self, worker: _Worker | None) -> None:
        """Stop `worker` and put a fresh one in its place. If a new process cannot be started (out of memory, too
        many processes), the place is held by None and the next call to take it tries again, so the pool never
        loses a place and a caller never waits on a queue nothing will refill."""
        if worker is not None:
            worker.stop()
        with self._lock:
            self._all.discard(worker)
            closed = self._closed
        if closed:
            return
        try:
            fresh = self._start()
        except Exception:  # noqa: BLE001 - logged; the place is kept for the next call to retry
            logger.exception("could not start a grading worker; the next reply will try again")
            fresh = None
        self._idle.put(fresh)

    def grade(self, kind: str, question, given: str, *, skill=None, timeout: float | None = None) -> DeadlineResult:
        """`grade(kind, question, given, skill=skill)` in a worker, wrong and `timed_out` if it takes longer than
        `timeout` seconds (the pool's own when None). Raises what `grade` raises for a bad call (an unknown kind)."""
        if self._closed:
            raise RuntimeError("the grading pool is closed")
        limit = self.timeout if timeout is None else timeout
        # A worker that dies under a reply (killed from outside, out of memory) is replaced and the reply tried once
        # more on the new one, so one lost process does not mark a learner wrong. A second loss grades it wrong.
        for attempt in (1, 2):
            worker = self._idle.get()
            start_failed = False
            try:
                if worker is None:
                    worker = self._start()
                try:
                    worker.wait_ready()
                except WorkerStartError:
                    start_failed = True
                    raise
                worker.conn.send((kind, question, given, skill))
                started = time.monotonic()
                if worker.conn.poll(limit):
                    status, value = worker.conn.recv()
                else:
                    status, value = "timeout", None
            except (EOFError, OSError, RuntimeError) as exc:
                logger.error(
                    "a grading worker failed (%s) on attempt %d; kind=%r given=%.200r", exc, attempt, kind, given
                )
                self._replace(worker)
                continue
            except BaseException:
                # Interrupted mid-call (KeyboardInterrupt): the worker may still be grading, so it cannot be reused.
                self._replace(worker)
                raise
            break
        else:
            if start_failed:
                raise WorkerStartError(
                    "no grading worker could start, so the reply was not graded; a script that grades with a deadline "
                    'must run from `if __name__ == "__main__":` (see mathgen.grading.deadline)'
                )
            return _wrong(question)
        if status == "timeout":
            logger.warning(
                "grading kind=%r given=%.200r ran past %.1f s; worker killed, graded wrong", kind, given, limit
            )
            self._replace(worker)
            return _wrong(question, timed_out=True)
        worker.served += 1
        if worker.served >= MAX_REPLIES_PER_WORKER:
            self._replace(worker)  # retired at its bound; a fresh one takes its place
        else:
            self._idle.put(worker)
        if status == "error":
            raise value  # `grade` raises only for a bad call (an unknown kind), as the in-process call would
        logger.debug("graded kind=%r in %.3f s", kind, time.monotonic() - started)
        return DeadlineResult(**vars(value))

    def close(self) -> None:
        """Stop every worker. A pool cannot be used after this."""
        with self._lock:
            self._closed = True
            workers = list(self._all)
            self._all.clear()
        for worker in workers:
            worker.stop()

    def __enter__(self) -> GradingPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _preload_grader(context, grade_fn) -> None:
    """Have the forkserver import the grader once, so each worker forked from it starts with the engine loaded.

    The preload list is the process's own, shared by every forkserver context, and only read when the server starts:
    this adds the grader's modules to the list a host may already have set, and changes nothing once the server
    runs. `__main__` is left out: preloading it would run the host's script again in the server."""
    from multiprocessing import forkserver

    current = getattr(forkserver._forkserver, "_preload_modules", [])  # no public getter in 3.12-3.14
    wanted = [__name__, grade_fn.__module__]
    merged = [m for m in current if m != "__main__"] + [m for m in wanted if m not in current and m != "__main__"]
    context.set_forkserver_preload(merged)


def _wrong(question, *, timed_out: bool = False) -> DeadlineResult:
    return DeadlineResult(correct=False, canonical=getattr(question, "answer_text", ""), timed_out=timed_out)


_default_pool: GradingPool | None = None
_default_lock = threading.Lock()


def default_pool() -> GradingPool:
    """The shared pool `grade_with_deadline` uses, started on first use with two workers and closed at exit."""
    global _default_pool
    with _default_lock:
        if _default_pool is None:
            _default_pool = GradingPool()
            atexit.register(_default_pool.close)
        return _default_pool


def grade_with_deadline(
    kind: str, question, given: str, *, skill=None, timeout: float = DEFAULT_TIMEOUT
) -> DeadlineResult:
    """`grade()` with a time limit, in the shared pool: a reply that takes longer than `timeout` seconds is graded
    wrong (`timed_out` is True). For more workers or another default, make a `GradingPool` of your own."""
    return default_pool().grade(kind, question, given, skill=skill, timeout=timeout)
