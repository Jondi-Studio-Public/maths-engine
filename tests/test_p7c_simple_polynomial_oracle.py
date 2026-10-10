"""Independent oracle for senior-solve-simple-polynomial-equations.

Re-derives the root from `question.text` alone, with a general quadratic
formula (`cmath.sqrt`) that never touches how the generator constructed the
question from p, m. Never imports `mathgen.generators.p7c_polynomial_solve_build`.
"""

from __future__ import annotations

import cmath
import re

import pytest

from mathgen import make

TOPIC = "simple_polynomial_equation"

_EASY_RE = re.compile(r"z² \+ (\d+) = 0")
_MEDIUM_RE = re.compile(r"\(z ([+-]) (\d+)\)² \+ (\d+) = 0")
_HARD_RE = re.compile(r"z² ([+-]) (\d+)z \+ (\d+) = 0")
_ANSWER_RE = re.compile(r"^(-?\d+)\s*\+\s*(\d*)i$|^(\d*)i$")


def _solve_monic_quadratic(B: float, C: float) -> complex:
    """Root with positive imaginary part of z^2 + Bz + C = 0."""
    disc = B * B - 4 * C
    sq = cmath.sqrt(disc)
    r1 = (-B + sq) / 2
    r2 = (-B - sq) / 2
    return r1 if r1.imag > 0 else r2


def _parse_answer(answer_text: str) -> complex:
    m = _ANSWER_RE.match(answer_text.strip())
    if not m:
        raise ValueError(f"could not parse answer_text: {answer_text!r}")
    if m.group(3) is not None:
        coef = m.group(3)
        return complex(0, int(coef) if coef else 1)
    p = int(m.group(1))
    coef = m.group(2)
    return complex(p, int(coef) if coef else 1)


def _derive_B_C(text: str):
    m = _EASY_RE.search(text)
    if m:
        return 0, int(m.group(1))
    m = _MEDIUM_RE.search(text)
    if m:
        sign, absp, k = m.groups()
        p = -int(absp) if sign == "+" else int(absp)
        k = int(k)
        return -2 * p, p * p + k
    m = _HARD_RE.search(text)
    if m:
        sign, b, c = m.groups()
        b = int(b)
        B = b if sign == "+" else -b
        return B, int(c)
    raise ValueError(f"could not parse equation from: {text!r}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_root_matches_answer_text(difficulty):
    mismatches = []
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        B, C = _derive_B_C(q.text)
        derived = _solve_monic_quadratic(B, C)
        stated = _parse_answer(q.answer_text)
        if abs(derived - stated) > 1e-6:
            mismatches.append((seed, q.text, q.answer_text, derived, stated))
    assert not mismatches, f"oracle mismatches: {mismatches}"


def test_oracle_covers_all_tiers():
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(5):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            B, C = _derive_B_C(q.text)
            _parse_answer(q.answer_text)
