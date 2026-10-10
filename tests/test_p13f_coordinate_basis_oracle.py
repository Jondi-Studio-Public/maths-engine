"""Re-derives each rendered question's answer from `question.text` alone, via a fresh elimination-based 2x2 solve
(never importing anything from `mathgen.generators.p13f_coordinate_basis_build`).
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p13f_coordinate_basis"

_B1_RE = re.compile(r"b1 = \((-?\d+), (-?\d+)\)")
_B2_RE = re.compile(r"b2 = \((-?\d+), (-?\d+)\)")
_V_RE = re.compile(r"v = \((-?\d+), (-?\d+)\)")
_C1_RE = re.compile(r"c1 = (-?\d+)\b")
_C2_RE = re.compile(r"c2 = (-?\d+)\b")


def _int_pair(match: re.Match) -> tuple[int, int]:
    a, b = match.groups()
    return int(a), int(b)


def _oracle_solve(b1: tuple[int, int], b2: tuple[int, int], v: tuple[int, int]) -> tuple[Fraction, Fraction]:
    """Solve c1*b1 + c2*b2 = v by elimination, written fresh (never Cramer's
    rule, never sympy): eliminate c1 by scaling each row so its b1 column
    matches, then subtract."""
    b1x, b1y = b1
    b2x, b2y = b2
    vx, vy = v
    # Two equations: b1x*c1 + b2x*c2 = vx ; b1y*c1 + b2y*c2 = vy.
    # Eliminate c1: multiply eqn1 by b1y, eqn2 by b1x, subtract.
    lhs_c2 = b1y * b2x - b1x * b2y
    rhs_c2 = b1y * vx - b1x * vy
    c2 = Fraction(rhs_c2, lhs_c2)
    # Back-substitute into eqn1 for c1 (b1x and b1y cannot both be zero on a
    # valid basis vector, so at least one division below is well-defined).
    if b1x != 0:
        c1 = Fraction(vx - b2x * c2, b1x)
    else:
        c1 = Fraction(vy - b2y * c2, b1y)
    return c1, c2


def _oracle_solve_from_text(text: str) -> tuple[Fraction, Fraction]:
    b1 = _int_pair(_B1_RE.search(text))
    b2 = _int_pair(_B2_RE.search(text))
    v = _int_pair(_V_RE.search(text))
    return _oracle_solve(b1, b2, v)


def _oracle_reconstruct_from_text(text: str) -> tuple[int, int]:
    b1 = _int_pair(_B1_RE.search(text))
    b2 = _int_pair(_B2_RE.search(text))
    c1 = int(_C1_RE.search(text).group(1))
    c2 = int(_C2_RE.search(text).group(1))
    vx = c1 * b1[0] + c2 * b2[0]
    vy = c1 * b1[1] + c2 * b2[1]
    return vx, vy


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_recomputes_solve_tiers(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        c1, c2 = _oracle_solve_from_text(q.text)
        expected_c1 = Fraction(q.answer["c1"])
        expected_c2 = Fraction(q.answer["c2"])
        assert c1 == expected_c1 and c2 == expected_c2, (
            f"{tier} seed={seed}: oracle ({c1}, {c2}) != canonical ({expected_c1}, {expected_c2}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam_tier():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        vx, vy = _oracle_reconstruct_from_text(q.text)
        assert (vx, vy) == (q.answer["vx"], q.answer["vy"]), (
            f"exam seed={seed}: oracle ({vx}, {vy}) != canonical ({q.answer['vx']}, {q.answer['vy']}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Proven red first: the oracle actually checks something ---------------


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    c1, c2 = _oracle_solve_from_text(q.text)
    assert (c1, c2) != (c1 + 1, c2)


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    vx, vy = _oracle_reconstruct_from_text(q.text)
    assert (vx, vy) != (vx + 1, vy)
