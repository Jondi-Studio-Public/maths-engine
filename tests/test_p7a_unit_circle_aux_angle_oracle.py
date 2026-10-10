"""Independent oracle for the `p7a_unit_circle_aux_angle` questions: it re-derives each answer from the rendered
question text.

Parses only `question.text` / `answer_text` / `distractors` -- never imports
anything from `mathgen.generators.p7a_unit_circle_aux_angle_build`. If the
answer cannot be derived from the question text alone, that is itself the
finding.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

VARIETY_SEEDS = range(60)

TIERS_A = [
    ("easy", {"variant": "quadrant_one"}),
    ("medium", {"variant": "signed_quadrant"}),
    ("hard", {"variant": "mixed_with_quadrantal"}),
]
TIERS_B = [
    ("easy", {"variant": "cos_form_q1"}),
    ("medium", {"variant": "cos_form_mixed_sign"}),
    ("hard", {"variant": "sin_form_mixed_sign"}),
]

_QUADRANTAL_RE = re.compile(r"Theta = (-?\d+) degrees, with terminal point \((-?\d+), (-?\d+)\)")
_COORD_RE = re.compile(r"terminal point on the unit circle is \((-?\d+/\d+|-?\d+), (-?\d+/\d+|-?\d+)\)")


def _parse_a(text: str):
    """Return (x, y) as Fractions -- the stated cos(theta), sin(theta) -- for
    either question shape skill A uses."""
    m = _QUADRANTAL_RE.search(text)
    if m:
        return Fraction(int(m.group(2))), Fraction(int(m.group(3)))
    m = _COORD_RE.search(text)
    assert m, f"cannot parse terminal point from: {text}"
    return Fraction(m.group(1)), Fraction(m.group(2))


@pytest.mark.parametrize("difficulty,params", TIERS_A)
@pytest.mark.parametrize("seed", VARIETY_SEEDS)
def test_oracle_a_tan_matches_stated_coordinates(difficulty, params, seed):
    q = make("unit_circle_tan_evaluate", seed=seed, difficulty=difficulty, **params)
    x, y = _parse_a(q.text)

    # The biconditional both directions: undefined iff x == 0, both checked.
    is_undefined = q.answer_text == "undefined"
    assert is_undefined == (x == 0), (
        f"seed={seed} {difficulty}: x={x}, y={y}, answer_text={q.answer_text!r} -- undefined-ness disagrees with x == 0"
    )
    if is_undefined:
        return
    expected = y / x
    actual = Fraction(q.answer_text)
    assert expected == actual, f"seed={seed} {difficulty}: expected tan={expected}, got {actual} ({q.text})"
    for d in q.distractors:
        if d == "undefined":
            continue
        assert Fraction(d) != expected, f"seed={seed} {difficulty}: distractor {d} equals the true tan"


# --- Skill B --------------------------------------------------------------

_AB_RE = re.compile(r"for a = (-?\d+) and b = (-?\d+)")
_PAIR_RE = re.compile(r"R = (-?\d+), alpha = (-?\d+\.\d+)")


def _parse_pair(s: str):
    m = _PAIR_RE.search(s)
    assert m, f"cannot parse R/alpha pair from: {s}"
    return int(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("difficulty,params", TIERS_B)
@pytest.mark.parametrize("seed", VARIETY_SEEDS)
def test_oracle_b_r_and_alpha_match_stated_a_b(difficulty, params, seed):
    q = make("aux_angle_r_cos_conversion", seed=seed, difficulty=difficulty, **params)
    m = _AB_RE.search(q.text)
    assert m, f"cannot parse a, b from: {q.text}"
    a, b = int(m.group(1)), int(m.group(2))
    form = "cos" if "R cos" in q.text else "sin"

    expected_r = math.hypot(a, b)
    if form == "cos":
        expected_alpha = math.degrees(math.atan2(b, a))
    else:
        expected_alpha = math.degrees(math.atan2(a, b))
    expected_alpha = round(expected_alpha, 1) + 0.0

    r_val, alpha_val = _parse_pair(q.answer_text)
    assert abs(r_val - expected_r) < 0.5, f"seed={seed} {difficulty}: R={r_val} vs expected {expected_r}"
    # Match within the question's own stated rounding precision (1 d.p.) plus
    # the pair grader's own tolerance, since alpha was rounded before display.
    assert abs(alpha_val - expected_alpha) <= 0.05 + 1e-9, (
        f"seed={seed} {difficulty}: alpha={alpha_val} vs expected {expected_alpha} ({q.text})"
    )

    for d in q.distractors:
        dr, dalpha = _parse_pair(d)
        same_r = abs(dr - expected_r) < 0.5
        same_alpha = abs(dalpha - expected_alpha) <= 0.051
        assert not (same_r and same_alpha), f"seed={seed} {difficulty}: distractor {d} matches the true pair"
