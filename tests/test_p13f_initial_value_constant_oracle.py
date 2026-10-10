"""Re-derives each rendered question's answer from `question.text` alone, via a fresh term-by-term parse and
substitution (never importing anything from `mathgen.generators.p13f_initial_value_constant_build`).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p13f_initial_value_constant"

# One F(x) term: an optional sign, an optional leading coefficient, "x" or
# "x^n", and an optional "/d" denominator -- e.g. "x^3/3", "-2x", "3x^2",
# "5x^4/4".
_TERM_RE = re.compile(r"([+-]?)\s*(\d*)x(?:\^(\d+))?(?:/(\d+))?")
_F_RE = re.compile(r"y = (.+?) \+ C")
_ZERO_X0_RE = re.compile(r"y\(0\) = (-?\d+)")
_XY_RE = re.compile(r"y\((-?\d+)\) = (-?\d+)")
_CROSSES_RE = re.compile(r"crosses the x-axis (?:at|when) x = (-?\d+)")
_POINT_RE = re.compile(r"passes through the point where x = (-?\d+) and y = (-?\d+)")


def _parse_F(f_text: str) -> list[tuple[Fraction, int]]:
    """[(coefficient, exponent), ...] parsed from the rendered F(x) text."""
    terms = []
    for m in _TERM_RE.finditer(f_text):
        sign, coef_digits, exp_digits, denom_digits = m.groups()
        sign_val = -1 if sign == "-" else 1
        coef = int(coef_digits) if coef_digits else 1
        exp = int(exp_digits) if exp_digits else 1
        denom = int(denom_digits) if denom_digits else 1
        terms.append((Fraction(sign_val * coef, denom), exp))
    return terms


def _eval_F(terms: list[tuple[Fraction, int]], x0: int) -> Fraction:
    return sum(coef * Fraction(x0) ** exp for coef, exp in terms)


def _parse_condition(text: str) -> tuple[int, int]:
    m = _ZERO_X0_RE.search(text)
    if m:
        return 0, int(m.group(1))
    m = _XY_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _CROSSES_RE.search(text)
    if m:
        return int(m.group(1)), 0
    m = _POINT_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"could not find an initial condition in {text!r}")


def _oracle(text: str) -> Fraction:
    f_text = _F_RE.search(text).group(1)
    terms = _parse_F(f_text)
    x0, y0 = _parse_condition(text)
    return Fraction(y0) - _eval_F(terms, x0)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard", "exam"))
def test_oracle_recomputes(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = _oracle(q.text)
        expected = Fraction(q.answer["p"], q.answer["q"])
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Proven red first: the oracle actually checks something ---------------


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="medium")
    recomputed = _oracle(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted


def test_oracle_parses_all_four_condition_shapes():
    """Sanity check on the oracle's own parser, independent of the
    generator: all four condition renderings it must handle."""
    assert _parse_condition("... y(0) = 3 ...") == (0, 3)
    assert _parse_condition("... y(-2) = -5 ...") == (-2, -5)
    assert _parse_condition("... crosses the x-axis at x = 4 ...") == (4, 0)
    assert _parse_condition("... passes through the point where x = -3 and y = 7 ...") == (-3, 7)
