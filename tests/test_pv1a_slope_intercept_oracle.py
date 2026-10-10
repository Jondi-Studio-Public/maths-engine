"""Independent oracle for the `pv1a_slope_intercept` questions: it re-derives each answer from the rendered
question text.

Imports neither the generator module nor the shared line helper. It calls
`mathgen.make` through the skill's own sources and re-derives (m, c) from
`question.figure` alone, by two routes (real coordinates of the two through
points; and squares-counted slope times the axis scale), over 1000 seeds per
tier in four chunks.
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-determine-the-slope-and-intercepts-of-a-straight"
CHUNK = 250


@pytest.fixture(scope="module")
def sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {k.id: k for k in skills}[SKILL_ID].sources}


def _parse(text):
    nums = re.fullmatch(r"\((-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?)\)", text)
    assert nums, text
    return Fraction(nums.group(1)), Fraction(nums.group(2))


@pytest.mark.parametrize("chunk", range(4))
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_key_equals_slope_and_intercept_of_the_drawn_line(sources, difficulty, chunk):
    src = sources[difficulty]
    for seed in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        q = make(src.topic, seed=seed, difficulty=difficulty, **(src.params or {}))
        fig = q.figure
        (x1, y1), (x2, y2) = fig["lines"][0]["through"]
        assert x1 != x2
        m = Fraction(y2 - y1, x2 - x1)
        c = y1 - m * x1
        # second route: squares, then the axis scale
        gx, gy = fig["x_grid"], fig["y_grid"]
        m2 = Fraction(y2 - y1, gy) / Fraction(x2 - x1, gx) * Fraction(gy, gx)
        assert m == m2
        assert c.denominator == 1
        key = _parse(q.answer_text)
        assert key == (m, c), (difficulty, seed, q.answer_text, m, c)
        assert grade_pair(q, q.answer_text).correct
        assert grade_pair(q, f"({m.numerator}/{m.denominator}, {c})").correct
        # no distractor is the key, and every one is a wrong answer for the real grader
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert _parse(d) != key
            assert not grade_pair(q, d).correct, (difficulty, seed, d)
        # the two through points are grid intersections, at least two columns apart
        assert abs(x2 - x1) >= 2 * gx
        for x, y in ((x1, y1), (x2, y2)):
            assert x % gx == 0 and y % gy == 0
        if difficulty == "exam":
            assert gx != gy
        else:
            assert gx == gy == 1
