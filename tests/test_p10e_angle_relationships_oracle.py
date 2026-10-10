"""Re-derives every answer by parsing ``question.text``/``answer_text`` alone.
Never imports anything from ``mathgen.generators.p10e_angle_relationships_build``.
"""

import re
from fractions import Fraction

import mathgen
import pytest

TOPIC = "p10e_angle_relationships"

_LINEAR_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def parse_linear(text: str) -> tuple[Fraction, Fraction]:
    """Parse a rendered linear-in-x expression like '3x + 10', '5x - 30',
    'x', '-x' into (coefficient, constant), both exact Fractions."""
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


def oracle_easy(text: str) -> int:
    m = re.search(r"measure (\d+) degrees and (\d+) degrees\. Find the size of the third angle", text)
    g1, g2 = int(m.group(1)), int(m.group(2))
    return 360 - g1 - g2


def oracle_medium(text: str) -> int:
    m = re.search(r"measure (\d+) degrees, (\d+) degrees and (\d+) degrees\. Find the size of the fourth angle", text)
    g1, g2, g3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 360 - g1 - g2 - g3


def oracle_hard(text: str) -> int:
    m = re.search(r"One angle measures \(([^)]+)\) degrees and the other measures \(([^)]+)\) degrees", text)
    a1, a2 = parse_linear(m.group(1))
    c1, c2 = parse_linear(m.group(2))
    assert "always equal" in text  # the question must state the equality fact itself
    x = (c2 - a2) / (a1 - c1)
    angle = a1 * x + a2
    assert angle.denominator == 1
    return int(angle)


@pytest.mark.parametrize("seed", range(60))
def test_easy_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="easy")
    assert oracle_easy(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_medium_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="medium")
    assert oracle_medium(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_hard_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="hard")
    assert oracle_hard(q.text) == int(q.answer_text)
