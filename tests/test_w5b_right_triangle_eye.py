"""W5b 1175: right-triangle-angle hard tier subtracts the observer's eye height."""

import math
import re

import pytest

from mathgen.grading import GRADERS
from mathgen import make

TOPIC = "trigonometry"
HARD = dict(kinds=("angle",), angle_style="elevation_eye", angle_lo=81, angle_hi=150)
OLD = {
    ("easy", 0): dict(angle_style="legs", angle_lo=3, angle_hi=40),
    ("easy", 1): dict(angle_style="legs", angle_lo=3, angle_hi=40),
    ("easy", 2): dict(angle_style="legs", angle_lo=3, angle_hi=40),
    ("medium", 0): dict(angle_style="leg_hyp", angle_lo=41, angle_hi=80),
    ("medium", 1): dict(angle_style="leg_hyp", angle_lo=41, angle_hi=80),
    ("medium", 2): dict(angle_style="leg_hyp", angle_lo=41, angle_hi=80),
    ("hard", 0): dict(angle_style="mixed", angle_lo=81, angle_hi=150),
    ("hard", 1): dict(angle_style="mixed", angle_lo=81, angle_hi=150),
    ("hard", 2): dict(angle_style="mixed", angle_lo=81, angle_hi=150),
}
OLD_TEXTS = {
    (
        "easy",
        0,
    ): "A right-angled triangle has opposite side 15 and adjacent side 25. Find the angle θ in degrees (1 d.p.).",
    (
        "easy",
        1,
    ): "A right-angled triangle has opposite side 40 and adjacent side 9. Find the angle θ in degrees (1 d.p.).",
    (
        "easy",
        2,
    ): "A right-angled triangle has opposite side 21 and adjacent side 19. Find the angle θ in degrees (1 d.p.).",
    (
        "medium",
        0,
    ): "A right-angled triangle has hypotenuse 122 and the side opposite angle θ is 74. Find θ in degrees (1 d.p.).",
    (
        "medium",
        1,
    ): "A right-angled triangle has hypotenuse 118 and the side adjacent to angle θ is 47. Find θ in degrees (1 d.p.).",
    (
        "medium",
        2,
    ): "A right-angled triangle has hypotenuse 106 and the side adjacent to angle θ is 55. Find θ in degrees (1 d.p.).",
    (
        "hard",
        0,
    ): "A right-angled triangle has hypotenuse 165 and the side adjacent to angle θ is 83. Find θ in degrees (1 d.p.).",
    (
        "hard",
        1,
    ): "A right-angled triangle has hypotenuse 238 and the side opposite angle θ is 99. Find θ in degrees (1 d.p.).",
    (
        "hard",
        2,
    ): "A right-angled triangle has opposite side 82 and adjacent side 121. Find the angle θ in degrees (1 d.p.).",
}


@pytest.mark.parametrize(("difficulty", "seed"), sorted(OLD_TEXTS))
def test_existing_styles_unchanged(difficulty, seed):
    q = make(TOPIC, seed=seed, difficulty=difficulty, kinds=("angle",), **OLD[(difficulty, seed)])
    assert q.text == OLD_TEXTS[(difficulty, seed)]


def test_unknown_style_still_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, difficulty="hard", kinds=("angle",), angle_style="nope")


def _parse(text):
    m = re.fullmatch(
        r"A student whose eyes are (\d\.\d) m above the ground stands (\d+) m from the base of "
        r"a vertical tower that is (\d+) m tall\. Find the angle of elevation of the top of "
        r"the tower from the student's eyes \(1 d\.p\.\)\.",
        text,
    )
    assert m, text
    return float(m.group(1)), int(m.group(2)), int(m.group(3))


def _deg(opp, adj):
    return round(math.degrees(math.atan(opp / adj)), 1)


@pytest.mark.parametrize("seed", range(300))
def test_answer_rederived_and_distractors(seed):
    q = make(TOPIC, seed=seed, difficulty="hard", **HARD)
    eye, dist, tower = _parse(q.text)
    ans = _deg(tower - eye, dist)
    assert 10.0 <= ans <= 80.0
    assert q.answer == ans
    assert q.answer_text == f"θ = {ans}°"
    assert len(q.distractors) == 3
    expected = {
        f"θ = {_deg(tower, dist)}°",
        f"θ = {_deg(tower + eye, dist)}°",
        f"θ = {round(90 - ans, 1)}°",
    }
    assert set(q.distractors) == expected
    values = [float(re.search(r"-?\d+\.\d", d).group()) for d in q.distractors]
    assert all(abs(v - ans) >= 0.1 - 1e-9 for v in values)
    assert len(set(values)) == len(values)
    assert q.answer_text not in q.distractors


@pytest.mark.parametrize("seed", range(100))
def test_grading(seed):
    q = make(TOPIC, seed=seed, difficulty="hard", **HARD)
    grader = GRADERS["number"]
    assert grader(q, q.answer_text).correct
    for d in q.distractors:
        assert not grader(q, d).correct
