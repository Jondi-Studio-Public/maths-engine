"""Write tests/skill_contract_answer_floors.yaml: every live skill's tiers that give fewer than
`MIN_DISTINCT_ANSWERS` distinct answers over the 60-seed sweep, at the count they give now.

Usage: python scripts/build_answer_floors.py [--workers N]

Run it when a tier on the list has been widened, or to record the list again from scratch; the
file is not hand-edited. `scripts/skill_contract.py`'s `check_answer_variety` holds each listed
tier at its count and fails a listed tier that has reached the minimum, so after a fix the entry
has to go. Running it twice on the same tree writes the same bytes. Takes about ten minutes of CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import skill_contract as sc  # noqa: E402

HEADER = f"""\
# Generated floor data, do not hand-edit: python scripts/build_answer_floors.py
# skill: {{topic/difficulty: distinct answers over the {sc.SEEDS}-seed sweep}}. Tiers below
# {sc.MIN_DISTINCT_ANSWERS} distinct answers, the backlog of tiers a learner can pass by guessing; the
# answers check holds each at its count and fails once it reaches {sc.MIN_DISTINCT_ANSWERS} (remove it).
# May only shrink or rise.
"""


def _low_tiers(skill_id: str) -> tuple[str, dict[str, int]]:
    need = min(sc.MIN_DISTINCT_ANSWERS, sc.SEEDS)
    return skill_id, {k: n for k, n in sorted(sc.distinct_answers(skill_id).items()) if n < need}


def render(floors: dict[str, dict[str, int]]) -> str:
    lines = [HEADER.rstrip("\n")]
    for skill_id in sorted(floors):
        lines.append(f"{skill_id}:")
        lines.extend(f"  {key}: {count}" for key, count in sorted(floors[skill_id].items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    with ProcessPoolExecutor(args.workers) as pool:
        found = dict(pool.map(_low_tiers, sc.live_ids(), chunksize=4))
    floors = {sid: tiers for sid, tiers in found.items() if tiers}
    sc.ANSWER_FLOORS_PATH.write_text(render(floors), encoding="utf-8", newline="\n")
    print(f"{sum(map(len, floors.values()))} tiers in {len(floors)} skills -> {sc.ANSWER_FLOORS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
