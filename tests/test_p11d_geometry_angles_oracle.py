"""Re-derives every answer by parsing ``question.text``/``question.answer_text``
alone. Never imports anything from
``mathgen.generators.p11d_geometry_angles_build`` -- every regex and lookup
table here is written fresh, so a bug shared between the generator's own
in-generator check and this file would have to be typed twice independently.
"""

import re
from fractions import Fraction

import mathgen
import pytest

TOPIC1 = "p11d_parallel_line_angle_theorems"
TOPIC2 = "p11d_compare_angle_names"

_EQUAL_TYPES = ("corresponding", "alternate")

_EASY_MEDIUM_RE = re.compile(
    r"One angle measures (\d+) degrees, and it is (corresponding|alternate|co-interior) "
    r"(?:to|with) another angle"
)
_HARD_REL_RE = re.compile(r"forming two (corresponding|alternate|co-interior) angles")
_HARD_EXPR_RE = re.compile(r"One angle measures \(([^)]+)\) degrees and the other measures \(([^)]+)\) degrees")
_LINEAR_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def parse_linear(text: str) -> tuple[Fraction, Fraction]:
    """Parse a rendered linear-in-x expression like '6x - 110', '3x + 10',
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
    const = Fraction(0) if sign is None else (Fraction(int(mag)) if sign == "+" else -Fraction(int(mag)))
    return coef, const


def oracle_easy_medium(text: str) -> int:
    m = _EASY_MEDIUM_RE.search(text)
    assert m is not None, f"could not parse stated angle/relationship: {text!r}"
    x, relationship = int(m.group(1)), m.group(2)
    if relationship in _EQUAL_TYPES:
        return x
    return 180 - x


def oracle_hard(text: str) -> int:
    m_rel = _HARD_REL_RE.search(text)
    m_expr = _HARD_EXPR_RE.search(text)
    assert m_rel is not None and m_expr is not None, f"could not parse relationship/expressions: {text!r}"
    relationship = m_rel.group(1)
    a1, a2 = parse_linear(m_expr.group(1))
    c1, c2 = parse_linear(m_expr.group(2))
    if relationship in _EQUAL_TYPES:
        # a1*x + a2 = c1*x + c2
        x = (c2 - a2) / (a1 - c1)
    else:
        # (a1*x + a2) + (c1*x + c2) = 180
        x = (180 - a2 - c2) / (a1 + c1)
    angle = a1 * x + a2
    assert angle.denominator == 1
    return int(angle)


@pytest.mark.parametrize("seed", range(60))
def test_easy_oracle(seed):
    q = mathgen.make(TOPIC1, seed=seed, difficulty="easy")
    assert oracle_easy_medium(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_medium_oracle(seed):
    q = mathgen.make(TOPIC1, seed=seed, difficulty="medium")
    assert oracle_easy_medium(q.text) == int(q.answer_text)


@pytest.mark.parametrize("seed", range(60))
def test_hard_oracle(seed):
    q = mathgen.make(TOPIC1, seed=seed, difficulty="hard")
    assert oracle_hard(q.text) == int(q.answer_text)


# -- Skill 2: p11d_compare_angle_names -------------------------------------------

_ORDER_TEXT_RE = re.compile(r": (.+)\.$")
_CANONICAL_ORDER = ["acute", "right", "obtuse", "straight", "reflex", "revolution"]
_RANK = {name: i for i, name in enumerate(_CANONICAL_ORDER)}


def oracle_order(text: str) -> str:
    m = _ORDER_TEXT_RE.search(text)
    assert m is not None, f"could not parse drawn names: {text!r}"
    names = [n.strip() for n in m.group(1).split(",")]
    return ", ".join(sorted(names, key=lambda n: _RANK[n]))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_order_oracle(seed, difficulty):
    q = mathgen.make(TOPIC2, seed=seed, difficulty=difficulty)
    assert oracle_order(q.text) == q.answer_text
