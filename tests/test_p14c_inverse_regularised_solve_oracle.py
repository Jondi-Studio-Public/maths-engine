"""Re-derives each rendered question's answer from `question.text` alone via the closed-form regularised normal
equations, never importing anything from `mathgen.generators.p14c_inverse_regularised_solve_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14c_inverse_regularised_solve"

_ROW_RE = re.compile(r"\((-?\d+), (-?\d+)\)")
_B_RE = re.compile(r"b = \[([^\]]+)\]")
_LAM_RE = re.compile(r"lambda = (-?\d+)")


def _parse_rows_b(text: str):
    rows = [(int(a), int(c)) for a, c in _ROW_RE.findall(text)]
    b = [int(v.strip()) for v in _B_RE.search(text).group(1).split(",")]
    return rows, b


def _ata_atb(rows, b):
    ata11 = sum(a1 * a1 for a1, _a2 in rows)
    ata12 = sum(a1 * a2 for a1, a2 in rows)
    ata22 = sum(a2 * a2 for _a1, a2 in rows)
    atb1 = sum(a1 * bi for (a1, _a2), bi in zip(rows, b))
    atb2 = sum(a2 * bi for (_a1, a2), bi in zip(rows, b))
    return ata11, ata12, ata22, atb1, atb2


def _solve(m11, m12, m22, r1, r2):
    det = m11 * m22 - m12 * m12
    x1 = Fraction(r1 * m22 - r2 * m12, det)
    x2 = Fraction(m11 * r2 - m12 * r1, det)
    return x1, x2


def _oracle_compute(text: str) -> tuple[Fraction, Fraction]:
    rows, b = _parse_rows_b(text)
    lam = int(_LAM_RE.search(text).group(1))
    ata11, ata12, ata22, atb1, atb2 = _ata_atb(rows, b)
    return _solve(ata11 + lam, ata12, ata22 + lam, atb1, atb2)


_X_RE = re.compile(r"x = \((-?\d+(?:/\d+)?), (-?\d+(?:/\d+)?)\)")


def _frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _oracle_exam(text: str) -> Fraction:
    rows, b = _parse_rows_b(text)
    ata11, ata12, ata22, atb1, atb2 = _ata_atb(rows, b)
    x1_s, x2_s = _X_RE.search(text).groups()
    x1, x2 = _frac(x1_s), _frac(x2_s)
    return Fraction(atb1 - ata11 * x1 - ata12 * x2, x1)


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_recomputes(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        rx1, rx2 = _oracle_compute(q.text)
        ex1, ex2 = Fraction(q.answer["x1"]), Fraction(q.answer["x2"])
        assert (rx1, rx2) == (ex1, ex2), (
            f"{tier} seed={seed}: oracle ({rx1}, {rx2}) != canonical ({ex1}, {ex2}) (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        recomputed = _oracle_exam(q.text)
        expected = Fraction(q.answer["lambda"])
        assert recomputed == expected, (
            f"exam seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    rx1, rx2 = _oracle_compute(q.text)
    corrupted = (rx1 + 1, rx2)
    assert (rx1, rx2) != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted
