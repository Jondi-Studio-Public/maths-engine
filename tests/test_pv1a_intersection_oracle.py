"""Independent oracle for the `pv1a_intersection` questions: it re-derives each answer from the rendered question
text.

Imports neither the generator module nor the shared line helper. It calls
`mathgen.make` through the skill's own sources and re-derives the meeting point
from `question.figure` alone, in exact arithmetic: the two drawn lines' two
through points each, solved for the common point. >= 1000 seeds per tier.
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-interpret-the-point-of-intersection-and-other-important"
SEEDS = 1000


@pytest.fixture(scope="module")
def sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {k.id: k for k in skills}[SKILL_ID].sources}


def _meet(fig):
    coef = []
    for ln in fig["lines"]:
        (x1, y1), (x2, y2) = ln["through"]
        m = Fraction(y2 - y1, x2 - x1)
        coef.append((m, y1 - m * x1))
    (m1, c1), (m2, c2) = coef
    assert m1 != m2
    x = (c2 - c1) / (m1 - m2)
    return x, m1 * x + c1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_key_is_where_the_drawn_lines_meet(sources, difficulty):
    src = sources[difficulty]
    seen = set()
    for seed in range(SEEDS):
        q = make(src.topic, seed=seed, difficulty=difficulty, **(src.params or {}))
        x, y = _meet(q.figure)
        assert x.denominator == 1 and y.denominator == 1, (difficulty, seed)
        assert q.answer_text == f"({int(x)}, {int(y)})", (difficulty, seed)
        assert grade_pair(q, f"{int(x)}, {int(y)}").correct
        # the meeting point is never a point a line was drawn through
        drawn = {tuple(p) for ln in q.figure["lines"] for p in ln["through"]}
        assert (x, y) not in drawn
        # no distractor is the meeting point, in either order spelling
        assert all(d != q.answer_text for d in q.distractors)
        for d in q.distractors:
            assert not grade_pair(q, d).correct, (difficulty, seed, d)
        seen.add((int(x), int(y)))
    assert len(seen) >= 20
