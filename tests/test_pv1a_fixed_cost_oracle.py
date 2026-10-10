"""Independent oracle for the `pv1a_fixed_cost` questions: it re-derives each answer from the rendered question
text.

Imports neither the generator module nor the shared line helper. It calls
`mathgen.make` through the skill's own sources and re-derives the fixed cost
from `question.figure` alone: the drawn line's value at x = 0 from its two
through points, in exact arithmetic. >= 1000 seeds per tier.
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-interpret-in-context-the-slope-and-intercept-of"
SEEDS = 1000


@pytest.fixture(scope="module")
def sources():
    _, skills = load_skills()
    return {s.difficulty: s for s in {k.id: k for k in skills}[SKILL_ID].sources}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_key_equals_intercept_of_the_drawn_line(sources, difficulty):
    src = sources[difficulty]
    for seed in range(SEEDS):
        q = make(src.topic, seed=seed, difficulty=difficulty, **(src.params or {}))
        (x1, y1), (x2, y2) = q.figure["lines"][0]["through"]
        slope = Fraction(y2 - y1, x2 - x1)
        intercept = Fraction(y1) - slope * x1
        assert intercept.denominator == 1
        assert q.answer_text == str(int(intercept)), (difficulty, seed)
        assert grade_number(q, str(int(intercept))).correct
        # no distractor is the intercept, and the rate / point readings stay wrong answers
        assert str(int(intercept)) not in q.distractors
        # the line is a rising cost line and the fixed cost is positive
        assert slope > 0 and intercept > 0
