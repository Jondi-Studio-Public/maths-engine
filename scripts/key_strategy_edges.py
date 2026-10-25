"""Print, for every part of every source, how far a no-maths learner strategy beats chance.

The strategies are those of `tests/test_key_not_lone_of_kind.py`, which gates on them (6 points from chance): eliminate
every option that is the only holder of some visible feature, pick the most central option, pick the most outlying
option, pick the option nearest the median. Read-only.

    PYTHONPATH=. python scripts/key_strategy_edges.py                       # the six PR skills, 1200 seeds
    PYTHONPATH=. python scripts/key_strategy_edges.py --skills a,b,c --seeds 400 --summary
        # any skill ids (every source of each), and the distribution of the per-part best edge
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import test_key_not_lone_of_kind as T  # noqa: E402

SEEDS_ARG: int | None = None


def _one(job):
    skill, src = job
    seeds = SEEDS_ARG or T.SEEDS_FOR.get(src["topic"], T.SEEDS)
    result = T.survey(src["topic"], src.get("difficulty"), dict(src.get("params") or {}), range(seeds))
    rows = []
    for (label, strategy), (total, squares, draws) in sorted(result["elim"].items()):
        if draws >= 100:
            mean = total / draws
            rows.append((label, strategy, mean, math.sqrt(max(squares / draws - mean * mean, 0) / draws), draws))
    return skill, src.get("difficulty"), rows


def _jobs(skills: list[str] | None) -> list:
    if not skills:
        return [p.values for p in T._sources()]
    jobs = []
    for skill in skills:
        path = next(ROOT.glob(f"skills/*/{skill}.yaml"))
        jobs += [(skill, s) for s in yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]]
    return jobs


def main() -> None:
    global SEEDS_ARG
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", help="comma-separated skill ids (default: the six PR skills)")
    ap.add_argument("--seeds", type=int, help="seeds per source (default: the test's own)")
    ap.add_argument("--summary", action="store_true", help="also print the distribution of the per-part best edge")
    args = ap.parse_args()
    SEEDS_ARG = args.seeds
    edges = []
    with Pool() as pool:
        for skill, difficulty, rows in pool.map(
            _one, _jobs(args.skills.split(",") if args.skills else None), chunksize=1
        ):
            best: dict = {}
            for label, strategy, mean, se, draws in rows:
                if label not in best or mean > best[label][1]:
                    best[label] = (strategy, mean, se, draws)
            for label, (strategy, mean, se, draws) in sorted(best.items()):
                over = mean > T.ELIM_POINTS and mean > T.ELIM_SIGMAS * se
                edges.append((mean, over))
                print(
                    f"{skill} {difficulty} part {label}: best {strategy} {mean:+.3f} (se {se:.3f}, {draws} draws)"
                    + ("  <-- over the bound" if over else "")
                )
    if args.summary and edges:
        xs = sorted(e for e, _ in edges)
        p90 = xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)]
        print(
            f"\nparts {len(xs)}: median {statistics.median(xs):+.3f}, 90th percentile {p90:+.3f}, max {xs[-1]:+.3f}, "
            f"above +{T.ELIM_POINTS:.2f}: {sum(e > T.ELIM_POINTS for e in xs)} ({100 * sum(e > T.ELIM_POINTS for e in xs) / len(xs):.0f}%), "
            f"over the bound with 3 se: {sum(o for _, o in edges)}"
        )


if __name__ == "__main__":
    main()
