"""W5b 1174: triangle-area-base-height hard tier supplies a slanted side as a distractor length."""

import re

import pytest

from mathgen.grading import GRADERS
from mathgen import make

TOPIC = "triangle_area_base_height"
HARD = dict(base_hi=30, height_hi=24, slant=True)

OLD_TEXTS = {
    ("easy", 0): "A triangle has a base of 6 cm and a perpendicular height of 3 cm. Calculate its area in cm².",
    ("easy", 1): "A triangle has a base of 4 cm and a perpendicular height of 2 cm. Calculate its area in cm².",
    ("easy", 2): "A triangle has a base of 2 cm and a perpendicular height of 4 cm. Calculate its area in cm².",
    ("medium", 0): "A triangle has a base of 2 cm and a perpendicular height of 5 cm. Calculate its area in cm².",
    ("medium", 1): "A triangle has a base of 15 cm and a perpendicular height of 12 cm. Calculate its area in cm².",
    ("medium", 2): "A triangle has a base of 15 cm and a perpendicular height of 10 cm. Calculate its area in cm².",
    ("hard", 0): "A triangle has a base of 8 cm and a perpendicular height of 14 cm. Calculate its area in cm².",
    ("hard", 1): "A triangle has a base of 13 cm and a perpendicular height of 10 cm. Calculate its area in cm².",
    ("hard", 2): "A triangle has a base of 16 cm and a perpendicular height of 11 cm. Calculate its area in cm².",
}


@pytest.mark.parametrize(("difficulty", "seed"), sorted(OLD_TEXTS))
def test_default_path_unchanged(difficulty, seed):
    assert make(TOPIC, seed=seed, difficulty=difficulty).text == OLD_TEXTS[(difficulty, seed)]


def _parse(text):
    m = re.fullmatch(
        r"A triangle has a base of (\d+) cm, a slanted side of (\d+) cm and a perpendicular height of (\d+) cm\. "
        r"Calculate its area in cm²\.",
        text,
    )
    assert m, text
    return tuple(int(x) for x in m.groups())


@pytest.mark.parametrize("seed", range(300))
def test_answer_rederived_and_distractors(seed):
    q = make(TOPIC, seed=seed, difficulty="hard", **HARD)
    base, side, height = _parse(q.text)
    assert side > height
    # the slant is a real hypotenuse: horizontal offset is a whole number fitting in the base
    offset_sq = side * side - height * height
    offset = round(offset_sq**0.5)
    assert offset * offset == offset_sq and offset <= base
    assert base * height % 2 == 0
    assert int(q.answer_text) == base * height // 2 == q.answer
    assert len(q.distractors) >= 3
    for d in q.distractors:
        assert int(d) != int(q.answer_text)
    assert len(set(q.distractors)) == len(q.distractors)
    assert str(base * height) in q.distractors  # omitted the half


@pytest.mark.parametrize("seed", range(100))
def test_grading(seed):
    q = make(TOPIC, seed=seed, difficulty="hard", **HARD)
    grader = GRADERS["number"]
    assert grader(q, q.answer_text).correct
    for d in q.distractors:
        assert not grader(q, d).correct
