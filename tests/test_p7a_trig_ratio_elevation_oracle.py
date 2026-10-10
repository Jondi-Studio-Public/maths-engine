"""Independent oracle for the `p7a_trig_ratio_elevation` questions: it re-derives each answer from the rendered
question text.

Parses only ``question.text`` / ``answer_text`` / ``distractors`` and
re-derives the answer from scratch. Does NOT import
``mathgen.generators.p7a_trig_ratio_elevation_build`` -- if this file ever
needs a name from that module, the check is not independent.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC_A = "trig_ratio_review_tan"
TOPIC_B = "elevation_depression_problem"

# The skill's own `sources` (skills/differentiation/senior-review-sine-cosine-
# and-tangent-as-ratios-of.yaml / skills/senior-mathematics/senior-work-with-
# the-concepts-of-angle-of-elevation.yaml) drive each tier purely through
# `difficulty`, whose preset already carries the right `variant` -- no extra
# params needed, so `make(topic, seed=seed, difficulty=difficulty)` alone
# reaches every tier a learner sees.


# --------------------------------------------------------------------------
# Skill A oracle
# --------------------------------------------------------------------------

_EASY_A_RE = re.compile(
    r"The side opposite theta is (\d+) units, the side adjacent to theta is (\d+) units, "
    r"and the hypotenuse is (\d+) units\. What is (sin|cos|tan)\(theta\)"
)
_MEDIUM_A_LEGS_RE = re.compile(
    r"The side opposite theta is (\d+) units and the side adjacent to theta is (\d+) units\. "
    r"\(The hypotenuse is not given\.\) What is (sin|cos|tan)\(theta\)"
)
_MEDIUM_A_OPP_HYP_RE = re.compile(
    r"The side opposite theta is (\d+) units and the hypotenuse is (\d+) units\. "
    r"\(The adjacent side is not given\.\) What is (sin|cos|tan)\(theta\)"
)
_MEDIUM_A_ADJ_HYP_RE = re.compile(
    r"The side adjacent to theta is (\d+) units and the hypotenuse is (\d+) units\. "
    r"\(The opposite side is not given\.\) What is (sin|cos|tan)\(theta\)"
)
_HARD_A_RE = re.compile(
    r"For an acute angle theta in a right-angled triangle, (sin|cos|tan)\(theta\) = "
    r"(\d+/\d+|\d+)\. Find (sin|cos|tan)\(theta\)"
)


def _ratio(name: str, opp: int, adj: int, hyp: int) -> Fraction:
    if name == "sin":
        return Fraction(opp, hyp)
    if name == "cos":
        return Fraction(adj, hyp)
    return Fraction(opp, adj)


def _expected_a(text: str) -> Fraction:
    m = _EASY_A_RE.search(text)
    if m:
        opp, adj, hyp, name = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        return _ratio(name, opp, adj, hyp)

    m = _MEDIUM_A_LEGS_RE.search(text)
    if m:
        opp, adj, name = int(m.group(1)), int(m.group(2)), m.group(3)
        hyp_sq = opp * opp + adj * adj
        hyp = math.isqrt(hyp_sq)
        assert hyp * hyp == hyp_sq, f"legs {opp},{adj} give no integer hypotenuse"
        return _ratio(name, opp, adj, hyp)

    m = _MEDIUM_A_OPP_HYP_RE.search(text)
    if m:
        opp, hyp, name = int(m.group(1)), int(m.group(2)), m.group(3)
        adj_sq = hyp * hyp - opp * opp
        adj = math.isqrt(adj_sq)
        assert adj * adj == adj_sq, f"opp/hyp {opp}/{hyp} gives no integer adjacent"
        return _ratio(name, opp, adj, hyp)

    m = _MEDIUM_A_ADJ_HYP_RE.search(text)
    if m:
        adj, hyp, name = int(m.group(1)), int(m.group(2)), m.group(3)
        opp_sq = hyp * hyp - adj * adj
        opp = math.isqrt(opp_sq)
        assert opp * opp == opp_sq, f"adj/hyp {adj}/{hyp} gives no integer opposite"
        return _ratio(name, opp, adj, hyp)

    m = _HARD_A_RE.search(text)
    if m:
        given_name, given_frac, asked_name = m.group(1), m.group(2), m.group(3)
        if "/" in given_frac:
            p, q = (int(x) for x in given_frac.split("/"))
        else:
            p, q = int(given_frac), 1
        if given_name == "sin":
            opp, hyp = p, q
            adj_sq = hyp * hyp - opp * opp
            adj = math.isqrt(adj_sq)
            assert adj * adj == adj_sq
        elif given_name == "cos":
            adj, hyp = p, q
            opp_sq = hyp * hyp - adj * adj
            opp = math.isqrt(opp_sq)
            assert opp * opp == opp_sq
        else:
            opp, adj = p, q
            hyp_sq = opp * opp + adj * adj
            hyp = math.isqrt(hyp_sq)
            assert hyp * hyp == hyp_sq
        return _ratio(asked_name, opp, adj, hyp)

    raise AssertionError(f"could not parse Skill A question: {text!r}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_trig_ratio_review_tan_oracle(seed: int, difficulty: str):
    q = make(TOPIC_A, seed=seed, difficulty=difficulty)
    expected = _expected_a(q.text)
    given = Fraction(q.answer_text)
    assert expected == given, (
        f"seed {seed}/{difficulty}: expected {expected}, answer_text says {given}. Question: {q.text}"
    )
    for d in q.distractors:
        assert Fraction(d) != expected, (
            f"seed {seed}/{difficulty}: distractor {d} equals expected {expected}. Question: {q.text}"
        )


# --------------------------------------------------------------------------
# Skill B oracle
# --------------------------------------------------------------------------

_EASY_B_RE = re.compile(
    r"From a point on level ground (\d+) m from the base of a tower, the angle of "
    r"elevation to the top of the tower is (\d+) degrees\."
)
_MEDIUM_B_RE = re.compile(
    r"An observer stands at the top of a cliff (\d+) m high\. The angle of depression from "
    r"the observer to a boat on the water is (\d+) degrees\."
)
_HARD_B_RE = re.compile(
    r"an observer measures the angle of elevation to the top of a tower as (\d+) degrees\. "
    r"Walking directly away from the tower, along a straight line through the tower's base, "
    r"the observer stops (\d+) m further away and measures the angle of elevation again, "
    r"now (\d+) degrees\."
)


def _expected_b(text: str) -> float:
    m = _EASY_B_RE.search(text)
    if m:
        d, theta = int(m.group(1)), int(m.group(2))
        return d * math.tan(math.radians(theta))

    m = _MEDIUM_B_RE.search(text)
    if m:
        H, theta = int(m.group(1)), int(m.group(2))
        return H / math.tan(math.radians(theta))

    m = _HARD_B_RE.search(text)
    if m:
        theta1, d, theta2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Solve the two tan equations directly (not the generator's closed
        # form): x tan(theta2) = (x - d) tan(theta1), where x is the far
        # point's distance to the base.
        t1, t2 = math.radians(theta1), math.radians(theta2)
        x = d * math.tan(t1) / (math.tan(t1) - math.tan(t2))
        return x * math.tan(t2)

    raise AssertionError(f"could not parse Skill B question: {text!r}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_elevation_depression_oracle(seed: int, difficulty: str):
    q = make(TOPIC_B, seed=seed, difficulty=difficulty)
    expected = _expected_b(q.text)
    expected_text = f"{expected:.1f}"
    given = float(q.answer_text)
    assert abs(expected - given) < 0.05, (
        f"seed {seed}/{difficulty}: expected {expected_text}, answer_text says {q.answer_text}. Question: {q.text}"
    )
    for d in q.distractors:
        assert abs(float(d) - expected) > 0.051, (
            f"seed {seed}/{difficulty}: distractor {d} within tolerance of expected {expected_text}. Question: {q.text}"
        )
