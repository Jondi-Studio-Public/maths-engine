"""Independent oracle for the `p10d_alternate_segment_converses` questions: it re-derives each answer from the
rendered question text.

Parses `question.text` with its own regexes and re-derives the answer from scratch -- it must NOT import
anything from `mathgen.generators.p10d_alternate_segment_converses_build`, not a helper, not a constant, not
the fixed MCQ option strings (those are copied here as literals because they are part of the skill's public
contract, the same way `mcq_options` is part of what a learner is given -- not a re-use of generator
internals).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

import mathgen

ALT_SEG_TOPIC = "alternate_segment_theorem"
CONVERSE_TOPIC = "circle_theorem_converses_classify"

OPTION_DIAMETER = "The stated side is a diameter of the circle (converse of: an angle in a semicircle is a right angle)"
OPTION_CONCYCLIC = "The four points lie on a common circle (converse of: angles subtended by the same arc are equal)"
OPTION_CYCLIC_QUAD = "The quadrilateral is a cyclic quadrilateral (converse of: opposite angles of a cyclic quadrilateral are supplementary)"
OPTION_TANGENT = "The line is a tangent to the circle at that point (converse of: the alternate segment theorem)"
OPTION_NONE = "None of these conclusions can be established from the information given"


# ============================================================================
# Skill 1 oracle: alternate_segment_theorem
# ============================================================================


def _oracle_direct(text: str) -> int:
    m = re.search(r"measured at T, is (\d+) degrees\. Point Q", text)
    assert m, f"cannot parse direct: {text!r}"
    return int(m.group(1))


def _oracle_composed(text: str) -> int:
    m = re.search(r"angle QTP is (\d+) degrees and angle TPQ is (\d+) degrees", text)
    assert m, f"cannot parse composed: {text!r}"
    a, b = int(m.group(1)), int(m.group(2))
    return 180 - a - b


def _extract_linear_exprs(text: str, var: str) -> list[tuple[int, int]]:
    """Find every ``(p{var} [+ q | - q])``-shaped parenthesised expression
    and return its ``(p, q)`` coefficients. Handles a bare unit coefficient
    (mathgen.core.fmt_signed renders coefficient 1 as ``x``, not ``1x``) and
    a bare linear term with no constant at all (mathgen.core.join_terms
    drops a zero-coefficient term, so ``q == 0`` renders as e.g. ``(2n)``
    with no trailing ``+ 0``)."""
    pattern = re.compile(rf"\((-?\d*){var}(?:\s*([+-])\s*(\d+))?\)")
    out = []
    for p_str, sign, c_str in pattern.findall(text):
        if p_str in ("", "+"):
            p = 1
        elif p_str == "-":
            p = -1
        else:
            p = int(p_str)
        q = 0 if not sign else (int(c_str) if sign == "+" else -int(c_str))
        out.append((p, q))
    return out


def _oracle_algebraic(text: str) -> int:
    exprs = _extract_linear_exprs(text, "x")
    assert len(exprs) == 2, f"expected 2 linear expressions in x, got {exprs} in {text!r}"
    (p1, q1), (p2, q2) = exprs
    assert p1 != p2, f"coefficients must differ for a unique solution: {text!r}"
    x = Fraction(q2 - q1, p1 - p2)
    assert x.denominator == 1, f"x does not resolve to an integer: {x} in {text!r}"
    assert x > 0, f"x must be positive: {x} in {text!r}"
    angle = p1 * int(x) + q1
    assert angle == p2 * int(x) + q2
    return angle


_ALT_SEG_ORACLES = {"easy": _oracle_direct, "medium": _oracle_composed, "hard": _oracle_algebraic}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(200))
def test_alternate_segment_oracle_matches(difficulty, seed):
    q = mathgen.make(ALT_SEG_TOPIC, seed=seed, difficulty=difficulty)
    expected = _ALT_SEG_ORACLES[difficulty](q.text)
    actual = int(q.answer_text)
    assert expected == actual, (
        f"oracle mismatch {difficulty} seed={seed}: expected {expected}, got {actual}\nQuestion: {q.text}"
    )


# ============================================================================
# Skill 2 oracle: circle_theorem_converses_classify
# ============================================================================


def _classify_family_a(text: str) -> str:
    theta = int(re.search(r"Angle XYZ = (\d+) degrees", text).group(1))
    return OPTION_DIAMETER if theta == 90 else OPTION_NONE


def _classify_family_b(text: str) -> str:
    m = re.search(r"Angle ACB = (\d+) degrees and angle ADB = (\d+) degrees", text)
    alpha, beta = int(m.group(1)), int(m.group(2))
    return OPTION_CONCYCLIC if alpha == beta else OPTION_NONE


def _classify_family_c(text: str) -> str:
    m = re.search(r"angle P = (\d+) degrees and angle R = (\d+) degrees", text)
    a, b = int(m.group(1)), int(m.group(2))
    return OPTION_CYCLIC_QUAD if a + b == 180 else OPTION_NONE


def _classify_family_d(text: str) -> str:
    exprs = _extract_linear_exprs(text, "n")
    assert len(exprs) == 2, f"expected 2 linear expressions in n, got {exprs} in {text!r}"
    (p1, q1), (p2, q2) = exprs
    x_value = int(re.search(r"known to be (\d+) degrees", text).group(1))
    n = Fraction(x_value - q1, p1)
    assert n.denominator == 1, f"n does not resolve to an integer: {n} in {text!r}"
    assert n > 0, f"n must be positive: {n} in {text!r}"
    y_value = p2 * int(n) + q2
    return OPTION_TANGENT if x_value == y_value else OPTION_NONE


def _classify_from_text(text: str) -> str:
    """Dispatch on the configuration's own wording, not on which tier asked
    for it -- exactly what a learner (and this independent oracle) has to
    go on."""
    if "Triangle XYZ has all three vertices on a circle" in text:
        return _classify_family_a(text)
    if "same side of line AB" in text:
        return _classify_family_b(text)
    if "opposite angles of the quadrilateral" in text:
        return _classify_family_c(text)
    if "A line meets a circle at point T" in text:
        return _classify_family_d(text)
    raise AssertionError(f"unrecognised configuration family in {text!r}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(400))
def test_converses_classify_oracle_matches_across_full_draw_space(difficulty, seed):
    q = mathgen.make(CONVERSE_TOPIC, seed=seed, difficulty=difficulty)
    expected = _classify_from_text(q.text)
    assert expected == q.answer_text, (
        f"oracle mismatch {difficulty} seed={seed}: expected {expected!r}, got {q.answer_text!r}\nQuestion: {q.text}"
    )


@pytest.mark.parametrize("seed", range(400))
def test_converses_classify_easy_is_always_family_a(seed):
    q = mathgen.make(CONVERSE_TOPIC, seed=seed, difficulty="easy")
    assert "Triangle XYZ has all three vertices on a circle" in q.text


@pytest.mark.parametrize("seed", range(400))
def test_converses_classify_medium_is_always_family_b_or_c(seed):
    q = mathgen.make(CONVERSE_TOPIC, seed=seed, difficulty="medium")
    is_b = "same side of line AB" in q.text
    is_c = "opposite angles of the quadrilateral" in q.text
    assert is_b or is_c
    assert not (is_b and is_c)


@pytest.mark.parametrize("seed", range(400))
def test_converses_classify_hard_is_always_family_d(seed):
    q = mathgen.make(CONVERSE_TOPIC, seed=seed, difficulty="hard")
    assert "A line meets a circle at point T" in q.text
