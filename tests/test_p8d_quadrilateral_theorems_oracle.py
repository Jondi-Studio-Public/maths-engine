"""Re-derives every answer by parsing ``question.text``/``answer_text`` alone.
Never imports anything from ``mathgen.generators.p8d_quadrilateral_theorems_build``.
"""

import math
import re
from fractions import Fraction

import mathgen
import pytest

CYCLIC_TOPIC = "the_opposite_angles_of_a_cyclic_quadrilateral_are"
RHOMBUS_TOPIC = "the_diagonals_of_a_parallelogram_meet_at_right"

_LINEAR_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def parse_linear(text: str) -> tuple[Fraction, Fraction]:
    """Parse a rendered linear-in-x expression like 'x - 1', '4x + 2', '5x'
    into (coefficient, constant), both exact Fractions."""
    m = _LINEAR_RE.match(text.strip())
    if not m:
        raise ValueError(f"could not parse linear expression: {text!r}")
    coef_str, sign, mag = m.group(1), m.group(2), m.group(3)
    if coef_str in ("", "+"):
        coef = Fraction(1)
    elif coef_str == "-":
        coef = Fraction(-1)
    else:
        coef = Fraction(int(coef_str))
    if sign is None:
        const = Fraction(0)
    else:
        const = Fraction(int(mag)) if sign == "+" else -Fraction(int(mag))
    return coef, const


# ============================================================================
# Skill 1: the_opposite_angles_of_a_cyclic_quadrilateral_are
# ============================================================================


def oracle_cyclic_easy(text: str) -> int:
    m = re.search(r"Angle A is (\d+) degrees", text)
    a = int(m.group(1))
    return 180 - a


def oracle_cyclic_medium(text: str) -> int:
    m = re.search(r"Angle A = \(?([^()]+?)\)? degrees and angle C = \(?([^()]+?)\)? degrees", text)
    a1, a2 = parse_linear(m.group(1))
    c1, c2 = parse_linear(m.group(2))
    ask = re.search(r"Find angle ([AC]), in degrees", text).group(1)
    x_sym_coef = a1 + c1
    x_sym_const = a2 + c2
    x = (Fraction(180) - x_sym_const) / x_sym_coef
    a_val = a1 * x + a2
    c_val = c1 * x + c2
    result = a_val if ask == "A" else c_val
    assert result.denominator == 1
    return int(result)


def oracle_cyclic_hard(text: str) -> int:
    m = re.search(r"Angle A = \(?([^()]+?)\)? degrees and angle B = \(?([^()]+?)\)? degrees", text)
    a1, a2 = parse_linear(m.group(1))
    b1, b2 = parse_linear(m.group(2))
    k_match = re.search(r"Angle C exceeds angle D by (\d+) degrees?", text)
    k = Fraction(int(k_match.group(1)))
    ask = re.search(r"Find angle ([ABCD]), in degrees", text).group(1)

    # C - D = k  =>  (180-A) - (180-B) = k  =>  B - A = k
    x = (k - b2 + a2) / (b1 - a1)
    a_val = a1 * x + a2
    b_val = b1 * x + b2
    c_val = 180 - a_val
    d_val = 180 - b_val
    values = {"A": a_val, "B": b_val, "C": c_val, "D": d_val}
    result = values[ask]
    assert result.denominator == 1
    return int(result)


@pytest.mark.parametrize("seed", range(60))
def test_cyclic_easy_oracle(seed):
    q = mathgen.make(CYCLIC_TOPIC, seed=seed, difficulty="easy")
    assert oracle_cyclic_easy(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_cyclic_medium_oracle(seed):
    q = mathgen.make(CYCLIC_TOPIC, seed=seed, difficulty="medium")
    assert oracle_cyclic_medium(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_cyclic_hard_oracle(seed):
    q = mathgen.make(CYCLIC_TOPIC, seed=seed, difficulty="hard")
    assert oracle_cyclic_hard(q.text) == int(q.answer_text)


# ============================================================================
# Skill 2: the_diagonals_of_a_parallelogram_meet_at_right
# ============================================================================


def oracle_rhombus_easy(text: str) -> int:
    m = re.search(r"lengths (\d+) cm and (\d+) cm", text)
    p, q = int(m.group(1)), int(m.group(2))
    total = p * p + q * q
    doubled_side = math.isqrt(total)
    assert doubled_side * doubled_side == total
    assert doubled_side % 2 == 0
    return doubled_side // 2


def oracle_rhombus_medium(text: str) -> int:
    m = re.search(r"One diagonal is (\d+) cm and the side length is (\d+) cm", text)
    p, s = int(m.group(1)), int(m.group(2))
    half_p = Fraction(p, 2)
    half_q_sq = Fraction(s) ** 2 - half_p**2
    assert half_q_sq > 0
    numer, denom = half_q_sq.numerator, half_q_sq.denominator
    assert denom == 1
    half_q = math.isqrt(numer)
    assert half_q * half_q == numer
    return 2 * half_q


def oracle_rhombus_hard(text: str) -> int:
    m = re.search(r"AB = \(?([^()]+?)\)? cm and BC = \(?([^()]+?)\)? cm", text)
    a1, a2 = parse_linear(m.group(1))
    b1, b2 = parse_linear(m.group(2))
    x = (b2 - a2) / (a1 - b1)
    side = a1 * x + a2
    assert side.denominator == 1
    return int(side)


@pytest.mark.parametrize("seed", range(60))
def test_rhombus_easy_oracle(seed):
    q = mathgen.make(RHOMBUS_TOPIC, seed=seed, difficulty="easy")
    assert oracle_rhombus_easy(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_rhombus_medium_oracle(seed):
    q = mathgen.make(RHOMBUS_TOPIC, seed=seed, difficulty="medium")
    assert oracle_rhombus_medium(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_rhombus_hard_oracle(seed):
    q = mathgen.make(RHOMBUS_TOPIC, seed=seed, difficulty="hard")
    assert oracle_rhombus_hard(q.text) == int(q.answer_text)
