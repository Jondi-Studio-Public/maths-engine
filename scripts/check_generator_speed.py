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
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathgen.core import make, topics  # noqa: E402

CEILING_MS = 250.0
ADVISORY_MS = 15.0
SEEDS = 8


def _at_least_one(text: str) -> int:
    """argparse `type=` for --seeds: a median over no samples is an error, not a result."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def measure(seeds: int = SEEDS) -> list[tuple[float, str]]:
    rows = []
    for _level, topic in topics():
        times = []
        for seed in range(seeds):
            start = time.perf_counter()
            try:
                make(topic, seed=seed)
            except Exception:  # noqa: BLE001 - a broken generator is another check's job
                pass
            times.append((time.perf_counter() - start) * 1000.0)
        rows.append((statistics.median(times), topic))
    rows.sort(reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=_at_least_one, default=SEEDS)
    ap.add_argument("--ceiling-ms", type=float, default=CEILING_MS)
    ap.add_argument("--advisory-ms", type=float, default=ADVISORY_MS)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = measure(args.seeds)
    medians = [m for m, _ in rows]
    print(f"{len(rows)} topics, median make() over {args.seeds} seeds")
    print(f"  p50 {statistics.median(medians):6.1f} ms    worst {rows[0][0]:6.1f} ms ({rows[0][1]})")
    print(f"\n  slowest {args.top}:")
    for m, topic in rows[: args.top]:
        mark = "  OVER CEILING" if m > args.ceiling_ms else ("  advisory" if m > args.advisory_ms else "")
        print(f"    {m:8.1f} ms  {topic}{mark}")

    over = [(m, t) for m, t in rows if m > args.ceiling_ms]
    advisory = [(m, t) for m, t in rows if args.advisory_ms < m <= args.ceiling_ms]
    if advisory:
        print(
            f"\nadvisory ({len(advisory)} topics over {args.advisory_ms:g} ms): heavy but "
            f"permitted. A NEW generator landing here should say why in its pull request."
        )
    if over:
        print(f"\nerrors: {len(over)} generator(s) over the {args.ceiling_ms:g} ms ceiling")
        for m, t in over:
            print(f"  {t}: {m:.1f} ms median per make()")
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
