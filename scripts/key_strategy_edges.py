"""Print, for every part of every source of the six PR skills, how far a no-maths learner strategy beats chance.

The strategies are those of `tests/test_key_not_lone_of_kind.py`, which gates on them (6 points from chance): eliminate
every option that is the only holder of some visible feature, pick the most central option, pick the most outlying
option, pick the option nearest the median. Read-only. Usage: `PYTHONPATH=. python scripts/key_strategy_edges.py [seeds]`.
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import test_key_not_lone_of_kind as T  # noqa: E402


def _one(param):
    skill, src = param.values
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else T.SEEDS_FOR.get(src["topic"], T.SEEDS)
    result = T.survey(src["topic"], src["difficulty"], dict(src.get("params") or {}), range(seeds))
    rows = []
    for (label, strategy), (total, squares, draws) in sorted(result["elim"].items()):
        if draws >= 100:
            mean = total / draws
            rows.append((label, strategy, mean, math.sqrt(max(squares / draws - mean * mean, 0) / draws), draws))
    return skill, src["difficulty"], rows


def main() -> None:
    with Pool() as pool:
        for skill, difficulty, rows in pool.map(_one, T._sources(), chunksize=1):
            best: dict = {}
            for label, strategy, mean, se, draws in rows:
                if label not in best or mean > best[label][1]:
                    best[label] = (strategy, mean, se, draws)
            for label, (strategy, mean, se, draws) in sorted(best.items()):
                flag = "  <-- over the bound" if mean > T.ELIM_POINTS and mean > T.ELIM_SIGMAS * se else ""
                print(
                    f"{skill} {difficulty} part {label}: best {strategy} {mean:+.3f} (se {se:.3f}, {draws} draws){flag}"
                )


if __name__ == "__main__":
    main()
