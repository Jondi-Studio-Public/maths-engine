"""Fail when a generator is slow enough to tax the whole suite.

WHY THIS EXISTS, measured at `-n 8` when the suite had about 12000 tests and took about 344 s:

In the suite this script was written for, one file of per-topic sweeps was 44% of the CPU. It held 19 parametrized sweeps, each looping
`_TEXT_SEEDS` (60) seeds across every tier, so ONE generator is called on the order of a thousand times. The
eight slowest individual tests in the whole suite were the same topic --
`use_partial_fractions_where_necessary` -- at 15-28 s each. A slow generator is therefore not a local cost: it
is multiplied by every per-topic sweep, and every future round inherits it.

WHY IT IS A SCRIPT AND NOT A TEST: it measures wall clock. Inside the suite it would run, and a timing
assertion there fails at random -- which is worse than no check, because a flaky gate teaches people to re-run
until green.

THE CEILING IS DERIVED, NOT CHOSEN. Per-topic median make() time today:
    p50 ~0.0 ms   p90 1.7 ms   p95 4.2 ms   p99 49.9 ms   worst 58.7 ms
The six topics above 15 ms are all legitimately heavy symbolic work (partial
fractions, Fourier coefficients, integration by parts, ODEs). Condemning them
would need a grandfathered list, and a list with no forcing function is the weak
form of this idea. So the hard ceiling (CEILING_MS, 250 ms) sits at about 4x today's worst --
where nothing legitimate lives and only a pathological new generator lands.
ADVISORY_MS flags the slowest 1% so an author has to justify it in the pull request
without the gate going red.

WHAT IT TIMES. A generator is not called with its defaults in the library: a skill's source names a topic, a
difficulty preset and parameters, and those can make the same function far slower or faster than its default
call. So each row below is one distinct (topic, difficulty, params) combination that an active skill source
uses, plus the bare default call of every registered topic. Each is timed as the median of 8 calls (seeds
0..7). Nothing is swallowed:

* a call that raises is an error that names the topic, difficulty and params, with the exception;
* a call still running after CALL_TIMEOUT_S (5 s) is an error, and the rest of that combination is skipped;
* a median over the ceiling is an error; a median over the advisory threshold is only listed.

Each combination gets one untimed warm-up call (a seed outside the timed ones) before the timed calls. The
first call of a topic in a process fills sympy's caches, and a heavy generator can read 3x slower cold than warm.
The suite calls each generator hundreds of times in one process, so the warm figure is the one that taxes it.
There are no retries: the median of the 8 timed calls is the figure.

The (topic, difficulty, params) rows are measured one after another in each process, and spread over
`--workers` processes. Output ends in `errors: []` when there is nothing to fix.
"""

import argparse
import json
import multiprocessing
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathgen import engine  # noqa: E402
from mathgen.core import make, topics  # noqa: E402

CEILING_MS = 250.0
ADVISORY_MS = 15.0
SEEDS = 8
CALL_TIMEOUT_S = 5.0

# (topic, difficulty or None, params as a sorted tuple of (name, value)); hashable and printable.
Combo = tuple[str, "str | None", tuple]


class CallTimeout(BaseException):
    """Raised by the alarm. A BaseException so a generator's own `except Exception` cannot swallow it."""


def _on_alarm(_signum, _frame):
    raise CallTimeout


def _at_least_one(text: str) -> int:
    """argparse `type=` for --seeds: a median over no samples is an error, not a result."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def describe(combo: Combo) -> str:
    """`topic`, or `topic [easy, kinds=('a', 'b')]` when a difficulty or parameters are part of the call."""
    topic, difficulty, params = combo
    extras = ([difficulty] if difficulty else []) + [f"{k}={v!r}" for k, v in params]
    return f"{topic} [{', '.join(extras)}]" if extras else topic


def combinations() -> list[Combo]:
    """Every bare topic default, then every distinct (topic, difficulty, params) an active skill source uses."""
    seen: set[Combo] = {(topic, None, ()) for _level, topic in topics()}
    for skill in engine.load_skills()[1]:
        if skill.status != "active":
            continue
        for src in skill.sources:
            params = tuple(sorted(engine._tuplify(src.params).items()))
            seen.add((src.topic, src.difficulty, params))
    return sorted(seen, key=lambda c: (c[0], c[1] or "", json.dumps(c[2], default=repr)))


def time_combo(combo: Combo, seeds: int = SEEDS, timeout_s: float = CALL_TIMEOUT_S) -> dict:
    """Median make() time in ms over seeds 0..seeds-1, plus any problems: `{"ms", "problems"}`.

    `ms` is None when no call finished. A timeout stops the combination (the rest would only repeat it).
    The warm-up call is not timed, but its exception or timeout is a problem like any other."""
    topic, difficulty, params = combo
    kwargs = dict(params)
    use_timer = hasattr(signal, "setitimer")
    if use_timer:
        previous = signal.signal(signal.SIGALRM, _on_alarm)
    times: list[float] = []
    problems: list[str] = []
    try:
        for seed in [seeds, *range(seeds)]:  # the first entry is the untimed warm-up
            warmup = seed == seeds
            start = time.perf_counter()
            try:
                if use_timer:
                    signal.setitimer(signal.ITIMER_REAL, timeout_s)
                try:
                    make(topic, seed=seed, difficulty=difficulty, **kwargs)
                finally:
                    if use_timer:
                        signal.setitimer(signal.ITIMER_REAL, 0)
            except CallTimeout:
                problems.append(f"{'warm-up' if warmup else f'seed {seed}'}: no answer after {timeout_s:g} s")
                break
            except Exception as exc:  # noqa: BLE001 - reported below, never swallowed
                problems.append(
                    f"{'warm-up' if warmup else f'seed {seed}'}: {type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
                )
                continue
            if not warmup:
                times.append((time.perf_counter() - start) * 1000.0)
    finally:
        if use_timer:
            signal.signal(signal.SIGALRM, previous)
    return {"ms": statistics.median(times) if times else None, "problems": problems}


def _time_task(task: tuple[Combo, int, float]) -> tuple[Combo, dict]:
    combo, seeds, timeout_s = task
    return combo, time_combo(combo, seeds, timeout_s)


def measure(seeds: int = SEEDS, workers: int = 1, timeout_s: float = CALL_TIMEOUT_S) -> list[tuple[Combo, dict]]:
    """Every combination's result, slowest first (a combination with no finished call sorts last)."""
    tasks = [(combo, seeds, timeout_s) for combo in combinations()]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_time_task, tasks, chunksize=16)
    else:
        results = [_time_task(task) for task in tasks]
    results.sort(key=lambda r: (-(r[1]["ms"] if r[1]["ms"] is not None else -1.0), describe(r[0])))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=_at_least_one, default=SEEDS)
    ap.add_argument("--ceiling-ms", type=float, default=CEILING_MS)
    ap.add_argument("--advisory-ms", type=float, default=ADVISORY_MS)
    ap.add_argument("--call-timeout", type=float, default=CALL_TIMEOUT_S, help="seconds before one call is an error")
    ap.add_argument(
        "--workers", type=_at_least_one, default=1, help="processes (default 1; timings are less steady above 1)"
    )
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    start = time.perf_counter()
    results = measure(args.seeds, args.workers, args.call_timeout)
    elapsed = time.perf_counter() - start
    timed = [(r["ms"], combo) for combo, r in results if r["ms"] is not None]
    bare = sum(1 for combo, _ in results if combo[1] is None and not combo[2])
    print(
        f"{len(results)} combinations ({bare} bare topic defaults, the rest from skill sources), "
        f"median make() over {args.seeds} seeds, {elapsed:.0f} s"
    )
    if timed:
        print(
            f"  p50 {statistics.median(m for m, _ in timed):6.1f} ms    worst {timed[0][0]:6.1f} ms ({describe(timed[0][1])})"
        )
    print(f"\n  slowest {args.top}:")
    for m, combo in timed[: args.top]:
        mark = "  OVER CEILING" if m > args.ceiling_ms else ("  advisory" if m > args.advisory_ms else "")
        print(f"    {m:8.1f} ms  {describe(combo)}{mark}")

    over = [(m, combo) for m, combo in timed if m > args.ceiling_ms]
    advisory = [(m, combo) for m, combo in timed if args.advisory_ms < m <= args.ceiling_ms]
    failed = [(combo, r["problems"]) for combo, r in results if r["problems"]]
    if advisory:
        print(
            f"\nadvisory ({len(advisory)} combinations over {args.advisory_ms:g} ms): heavy but "
            f"permitted. A NEW generator landing here should say why in its pull request."
        )
    if over or failed:
        print(f"\nerrors: {len(over) + len(failed)}")
        for m, combo in over:
            print(f"  {describe(combo)}: {m:.1f} ms median per make(), over the {args.ceiling_ms:g} ms ceiling")
        for combo, problems in failed:
            print(f"  {describe(combo)}: {len(problems)} failed call(s)")
            for problem in problems[:3]:
                print(f"      {problem}")
        if over:
            print(
                "\nA generator this slow is called hundreds of times by every test that sweeps "
                "seeds, and by the regrade sweep. Cache what you can "
                "compute once at module scope, and prefer exact integer/Fraction "
                "arithmetic over sympy where the mathematics allows it."
            )
        return 1
    print("\nerrors: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
