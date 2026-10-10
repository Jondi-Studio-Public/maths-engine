"""Independent oracle for senior-solve-problems-involving-bearings.

Derives the expected answer purely from the rendered `question.text`, using
only basic bearing/trigonometry rules re-implemented here from scratch. Does
NOT import `mathgen.generators.p7a_bearings_build` -- no helper, no constant,
no formula shared with the generator.

- easy: mod-360 arithmetic on the stated bearing.
- medium: Pythagoras on the two stated leg lengths.
- hard: a full 2D vector-sum + atan2 bearing conversion (the "opposite" route
  from the generator's own in-generator check, which uses a closed-form
  reference-bearing + inverse-tan lookup instead).
"""

import math
import re

import pytest

from mathgen import make

TOPIC = "bearings_problem"

_EASY_RE = re.compile(
    r"The bearing from point A to point B is (\d{3})°\. "
    r"What is the bearing from point B to point A\?"
)
_MEDIUM_RE = re.compile(
    r"Starting at point A, a hiker walks on a bearing of (\d{3})° for (\d+) km, "
    r"arriving at point M, then turns and walks on a bearing of (\d{3})° for (\d+) km, "
    r"arriving at point B\. Find the straight-line distance from A to B, to 1 decimal place\."
)
_HARD_RE = re.compile(
    r"Starting at point A, a surveyor walks on a bearing of (\d{3})° for (\d+) m, "
    r"arriving at point M, then turns and walks on a bearing of (\d{3})° for (\d+) m, "
    r"arriving at point B\. Find the bearing (from A to B|from B to A), to the nearest degree\."
)


def _oracle_easy(text: str) -> int:
    m = _EASY_RE.match(text)
    assert m, f"could not parse easy question: {text!r}"
    bearing_ab = int(m.group(1))
    # A bearing and its reverse point in opposite directions, 180 degrees apart.
    return (bearing_ab + 180) % 360


def _oracle_medium(text: str) -> float:
    m = _MEDIUM_RE.match(text)
    assert m, f"could not parse medium question: {text!r}"
    L1, L2 = int(m.group(2)), int(m.group(4))
    # The two legs run on bearings 90 degrees apart, so AB is the hypotenuse
    # of a right triangle with those two legs.
    hyp = math.sqrt(L1 * L1 + L2 * L2)
    return round(hyp, 1)


def _oracle_hard(text: str) -> int:
    m = _HARD_RE.match(text)
    assert m, f"could not parse hard question: {text!r}"
    b1, L1, b2, L2, direction = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        m.group(5),
    )
    # Full 2D vector-sum: East/North displacement of each leg, added, then
    # converted back to a compass bearing with atan2(east, north).
    theta1, theta2 = math.radians(b1), math.radians(b2)
    east = L1 * math.sin(theta1) + L2 * math.sin(theta2)
    north = L1 * math.cos(theta1) + L2 * math.cos(theta2)
    bearing_ab = round(math.degrees(math.atan2(east, north))) % 360
    if direction == "from A to B":
        return bearing_ab
    return (bearing_ab + 180) % 360


@pytest.mark.parametrize("seed", range(60))
def test_easy_back_bearing_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="easy")
    expected = _oracle_easy(q.text)
    assert expected == int(q.answer_text), f"seed {seed}: oracle {expected} != reported {q.answer_text}. {q.text}"
    for d in q.distractors:
        assert int(d) != expected, f"seed {seed}: distractor {d} equals oracle answer {expected}"


@pytest.mark.parametrize("seed", range(60))
def test_medium_leg_distance_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="medium")
    expected = _oracle_medium(q.text)
    reported = float(q.answer_text)
    assert abs(expected - reported) < 1e-9, f"seed {seed}: oracle {expected} != reported {reported}. {q.text}"
    for d in q.distractors:
        assert abs(float(d) - expected) > 0.05, (
            f"seed {seed}: distractor {d} within grading tolerance of oracle answer {expected}"
        )


@pytest.mark.parametrize("seed", range(60))
def test_hard_leg_bearing_oracle(seed):
    q = make(TOPIC, seed=seed, difficulty="hard")
    expected = _oracle_hard(q.text)
    assert expected == int(q.answer_text), f"seed {seed}: oracle {expected} != reported {q.answer_text}. {q.text}"
    for d in q.distractors:
        assert int(d) != expected, f"seed {seed}: distractor {d} equals oracle answer {expected}"


def test_all_four_bearing_pair_combinations_seen_by_the_oracle():
    """The oracle's own sweep independently confirms the required
    coverage: all four bearing-pair combinations occur over 60 seeds."""
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        m = _HARD_RE.match(q.text)
        seen.add((int(m.group(1)), int(m.group(3))))
    assert seen == {(0, 90), (90, 180), (180, 270), (270, 0)}, seen
