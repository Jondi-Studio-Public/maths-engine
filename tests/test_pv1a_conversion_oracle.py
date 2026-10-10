"""Independent oracle for the `pv1a_conversion` questions: it re-derives each answer from the rendered question
text.

Imports neither the generator module nor the shared line helper. It calls
`mathgen.make` through the skill's own sources, reads the line from
`question.figure` (its two through points), and reads the stated input and the
direction from `question.text` (the numeral and the axis unit it is followed
by), then converts in exact arithmetic. >= 1000 seeds per tier.
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-interpret-and-use-graphs-in-practical-situations-including"
SEEDS = 1000


@pytest.fixture(scope="module")
def sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {k.id: k for k in skills}[SKILL_ID].sources}


def _convert(q):
    fig = q.figure
    (x1, y1), (x2, y2) = fig["lines"][0]["through"]
    m = Fraction(y2 - y1, x2 - x1)
    c = y1 - m * x1
    ux = re.search(r"\(([^)]+)\)$", fig["x_label"]).group(1)
    uy = re.search(r"\(([^)]+)\)$", fig["y_label"]).group(1)
    nums = re.findall(r"\d+", q.text)
    assert len(nums) == 1
    n = int(nums[0])
    after = q.text[q.text.index(nums[0]) + len(nums[0]) :].lstrip()
    if after.startswith(ux):
        return m * n + c, "forward"
    assert after.startswith(uy), q.text
    return (n - c) / m, "reverse"


@pytest.mark.parametrize(
    "difficulty,direction", [("easy", "forward"), ("medium", "reverse"), ("hard", "forward"), ("exam", "reverse")]
)
def test_key_is_the_converted_value(sources, difficulty, direction):
    src = sources[difficulty]
    seen = set()
    for seed in range(SEEDS):
        q = make(src.topic, seed=seed, difficulty=difficulty, **(src.params or {}))
        value, got_direction = _convert(q)
        assert got_direction == direction, (difficulty, seed)
        assert value.denominator == 1 and value >= 1, (difficulty, seed)
        assert q.answer_text == str(int(value)), (difficulty, seed)
        assert grade_number(q, str(int(value))).correct
        assert str(int(value)) not in q.distractors
        for d in q.distractors:
            assert not grade_number(q, d).correct, (difficulty, seed, d)
        seen.add(int(value))
    assert len(seen) >= 8
