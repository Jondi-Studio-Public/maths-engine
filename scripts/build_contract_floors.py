"""Write the two backlogs the library-wide contract test holds the skills to:

- tests/skill_contract_answer_floors.yaml: every live skill's tiers that give fewer than
  `MIN_DISTINCT_ANSWERS` distinct answers over the 60-seed sweep, at the count they give now.
- tests/skill_contract_option_baseline.yaml: the tiers with an option fault (`option_faults`:
  option count, middle answer, impossible probability), by family.

Usage: python scripts/build_contract_floors.py [--workers N]

Run it after fixing a listed tier, or to record both lists again from scratch; neither file is
hand-edited. `tests/test_skill_option_quality.py` fails a listed tier that no longer has its fault,
so after a fix the entry has to go. Running it twice on the same tree writes the same bytes. Takes
about ten minutes of CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import skill_contract as sc  # noqa: E402

ANSWER_HEADER = f"""\
# Generated floor data, do not hand-edit: python scripts/build_contract_floors.py
# skill: {{topic/difficulty: distinct answers over the {sc.SEEDS}-seed sweep}}. Tiers below
# {sc.MIN_DISTINCT_ANSWERS} distinct answers, the backlog of tiers a learner can pass by guessing; the
# answers check holds each at its count and fails once it reaches {sc.MIN_DISTINCT_ANSWERS} (remove it).
# May only shrink or rise.
"""

OPTION_HEADER = f"""\
# Generated baseline data, do not hand-edit: python scripts/build_contract_floors.py
# family: {{skill: [topic/difficulty]}}. The tiers with each option fault over the {sc.SEEDS}-seed sweep
# (scripts/skill_contract.py, option_faults): option_count, a multiple-choice tier that does not
# always show the same number of options, at least {sc.MIN_OPTIONS}; middle_answer, all-number options
# where the answer is the lowest or highest in under {sc.MIN_EXTREME_SHARE:.0%} of draws; probability, a
# probability offered a wrong option above 1 or below 0. May only shrink.
"""


def _measure(skill_id: str):
    need = min(sc.MIN_DISTINCT_ANSWERS, sc.SEEDS)
    low = {k: n for k, n in sorted(sc.distinct_answers(skill_id).items()) if n < need}
    return skill_id, low, {family: sorted(tiers) for family, tiers in sc.option_faults(skill_id).items()}


def render_floors(floors: dict[str, dict[str, int]]) -> str:
    lines = [ANSWER_HEADER.rstrip("\n")]
    for skill_id in sorted(floors):
        lines.append(f"{skill_id}:")
        lines.extend(f"  {key}: {count}" for key, count in sorted(floors[skill_id].items()))
    return "\n".join(lines) + "\n"


def render_baseline(baseline: dict[str, dict[str, list[str]]]) -> str:
    lines = [OPTION_HEADER.rstrip("\n")]
    for family in sorted(baseline):
        lines.append(f"{family}:")
        for skill_id in sorted(baseline[family]):
            lines.append(f"  {skill_id}:")
            lines.extend(f"  - {key}" for key in baseline[family][skill_id])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    with ProcessPoolExecutor(args.workers) as pool:
        measured = list(pool.map(_measure, sc.live_ids(), chunksize=4))
    floors = {sid: low for sid, low, _ in measured if low}
    baseline: dict[str, dict[str, list[str]]] = {"middle_answer": {}, "option_count": {}, "probability": {}}
    for sid, _, faults in measured:
        for family, tiers in faults.items():
            if tiers:
                baseline[family][sid] = tiers
    sc.ANSWER_FLOORS_PATH.write_text(render_floors(floors), encoding="utf-8", newline="\n")
    sc.OPTION_BASELINE_PATH.write_text(render_baseline(baseline), encoding="utf-8", newline="\n")
    print(f"answers: {sum(map(len, floors.values()))} tiers in {len(floors)} skills")
    for family, by_skill in baseline.items():
        print(f"{family}: {sum(map(len, by_skill.values()))} tiers in {len(by_skill)} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
