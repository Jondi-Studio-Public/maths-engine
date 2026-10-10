"""Re-derives each rendered question's [b0, b1, b2] from `question.text` alone by re-solving the normal equations
with its own Fraction Gaussian elimination, never importing anything from
`mathgen.generators.p14e_multivariate_regression_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14e_multivariate_regression"

_POINTS_RE = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")
_HARD_S_RE = re.compile(
    r"XtX = \[\[(-?\d+), (-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+), (-?\d+)\], "
    r"\[(-?\d+), (-?\d+), (-?\d+)\]\]"
)
_HARD_T_RE = re.compile(r"Xty[^\[]*\[(-?\d+), (-?\d+), (-?\d+)\]")
_EXAM_RE = re.compile(r"XtX = \[\[(-?\d+), 0, 0\], \[0, (-?\d+), (-?\d+)\], \[0, (-?\d+), (-?\d+)\]\]")
_EXAM_U_RE = re.compile(r"Xty[^\[]*\[(-?\d+), (-?\d+)\]")
_EXAM_MEANS_RE = re.compile(r"x1bar = (-?\d+), x2bar = (-?\d+), ybar = (-?\d+)")


def _solve_gauss(rows: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Plain Fraction-exact Gaussian elimination for a small square system,
    written without reference to the generator's own sympy-based solve."""
    n = len(rows)
    a = [list(map(Fraction, row)) + [Fraction(b)] for row, b in zip(rows, rhs)]
    for col in range(n):
        pivot_row = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[pivot_row] = a[pivot_row], a[col]
        pivot = a[col][col]
        a[col] = [v / pivot for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [v - factor * w for v, w in zip(a[r], a[col])]
    return [a[i][n] for i in range(n)]


def _oracle_raw_data(text: str) -> tuple[int, int, int]:
    points = [(int(x1), int(x2), int(y)) for x1, x2, y in _POINTS_RE.findall(text)]
    assert len(points) == 4, f"expected 4 (x1, x2, y) triples, found {len(points)}: {text!r}"
    n = len(points)
    sx1 = sum(p[0] for p in points)
    sx2 = sum(p[1] for p in points)
    sx1x1 = sum(p[0] * p[0] for p in points)
    sx1x2 = sum(p[0] * p[1] for p in points)
    sx2x2 = sum(p[1] * p[1] for p in points)
    sy = sum(p[2] for p in points)
    sx1y = sum(p[0] * p[2] for p in points)
    sx2y = sum(p[1] * p[2] for p in points)
    rows = [[n, sx1, sx2], [sx1, sx1x1, sx1x2], [sx2, sx1x2, sx2x2]]
    rhs = [sy, sx1y, sx2y]
    b0, b1, b2 = _solve_gauss(rows, rhs)
    return int(b0), int(b1), int(b2)


def _oracle_gram(text: str) -> tuple[int, int, int]:
    ms = _HARD_S_RE.search(text)
    assert ms is not None, f"could not find Gram matrix in {text!r}"
    s00, s01, s02, s01b, s11, s12, s02b, s12b, s22 = (int(g) for g in ms.groups())
    mt = _HARD_T_RE.search(text)
    assert mt is not None, f"could not find Xty vector in {text!r}"
    t0, t1, t2 = (int(g) for g in mt.groups())
    rows = [[s00, s01, s02], [s01b, s11, s12], [s02b, s12b, s22]]
    rhs = [t0, t1, t2]
    b0, b1, b2 = _solve_gauss(rows, rhs)
    return int(b0), int(b1), int(b2)


def _oracle_exam(text: str) -> tuple[int, int, int]:
    m = _EXAM_RE.search(text)
    assert m is not None, f"could not find block Gram matrix in {text!r}"
    n, a, b_off, b_off2, c = (int(g) for g in m.groups())
    mu = _EXAM_U_RE.search(text)
    assert mu is not None, f"could not find Xty lower entries in {text!r}"
    u1, u2 = int(mu.group(1)), int(mu.group(2))
    means = _EXAM_MEANS_RE.search(text)
    assert means is not None, f"could not find sample means in {text!r}"
    x1bar, x2bar, ybar = (int(g) for g in means.groups())
    b1, b2 = _solve_gauss([[a, b_off], [b_off2, c]], [u1, u2])
    b0 = Fraction(ybar) - b1 * x1bar - b2 * x2bar
    return int(b0), int(b1), int(b2)


@pytest.mark.parametrize(
    "tier,fn",
    [
        ("easy", _oracle_raw_data),
        ("medium", _oracle_raw_data),
        ("hard", _oracle_gram),
    ],
)
def test_oracle_recomputes(tier, fn):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = fn(q.text)
        expected = tuple(q.answer)
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        recomputed = _oracle_exam(q.text)
        expected = tuple(q.answer)
        assert recomputed == expected, (
            f"exam seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_raw_data(q.text)
    corrupted = (recomputed[0] + 1, recomputed[1], recomputed[2])
    assert recomputed != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = (recomputed[0] + 1, recomputed[1], recomputed[2])
    assert recomputed != corrupted


def test_gram_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="hard")
    recomputed = _oracle_gram(q.text)
    corrupted = (recomputed[0] + 1, recomputed[1], recomputed[2])
    assert recomputed != corrupted
