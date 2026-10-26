"""`hecs-help-repayment` must build for every seed: the years-to-clear question skips debts that never clear.

`scripts/build_banks.py` failed on this skill with "debt does not clear" (hard years source, seeds 181 and 294),
because a draw whose indexation outran the repayment for 60 years raised instead of being redrawn.
"""

import pytest

from mathgen import engine

_, _SKILLS = engine.load_skills()
SKILL = {s.id: s for s in _SKILLS}["hecs-help-repayment"]


@pytest.mark.parametrize("seed", [181, 294, *range(0, 60)])
def test_every_source_builds_and_marks_its_key(seed):
    for source in range(len(SKILL.sources)):
        q = engine.make_for_skill(SKILL, seed, source=source)
        assert engine.grade(SKILL, q, q.answer_text).correct
