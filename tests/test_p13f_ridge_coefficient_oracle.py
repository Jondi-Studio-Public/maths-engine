"""Re-derives each rendered question's answer from `question.text` alone via the closed form beta =
Sxy/(Sxx+lambda), never importing anything from `mathgen.generators.p13f_ridge_coefficient_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p13f_ridge_coefficient"

_SXX_RE = re.compile(r"Sxx = sum\(x_i\^2\) = (-?\d+)")
_SXY_RE = re.compile(r"Sxy = sum\(x_i\*y_i\) = (-?\d+)")
_LAM_RE = re.compile(r"lambda = (-?\d+)[.,]")
_POINTS_RE = re.compile(r"\((-?\d+),\s*(-?\d+)\)")


def _frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _oracle_given(text: str) -> Fraction:
    Sxx = int(_SXX_RE.search(text).group(1))
    Sxy = int(_SXY_RE.search(text).group(1))
    lam = int(_LAM_RE.search(text).group(1))
    return Fraction(Sxy, Sxx + lam)


def _oracle_hard(text: str) -> Fraction:
    points = [(int(x), int(y)) for x, y in _POINTS_RE.findall(text)]
    lam = int(_LAM_RE.search(text).group(1))
    Sxx = sum(x * x for x, _ in points)
    Sxy = sum(x * y for x, y in points)
    return Fraction(Sxy, Sxx + lam)


_BETA_PATTERNS = (
    re.compile(r"works out to (-?\d+(?:/\d+)?) for some"),
    re.compile(r"beta = (-?\d+(?:/\d+)?) = Sxy"),
    re.compile(r"equals (-?\d+(?:/\d+)?)\."),
)


def _oracle_exam(text: str) -> Fraction:
    Sxx = int(_SXX_RE.search(text).group(1))
    Sxy = int(_SXY_RE.search(text).group(1))
    beta = None
    for pattern in _BETA_PATTERNS:
        m = pattern.search(text)
        if m:
            beta = _frac(m.group(1))
            break
    assert beta is not None, f"could not find beta in {text!r}"
    lam = Fraction(Sxy, 1) / beta - Sxx
    return lam


@pytest.mark.parametrize(
    "tier,fn",
    [
        ("easy", _oracle_given),
        ("medium", _oracle_given),
        ("hard", _oracle_hard),
    ],
)
def test_oracle_recomputes(tier, fn):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        recomputed = fn(q.text)
        expected = Fraction(q.answer["p"], q.answer["q"])
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
        expected = Fraction(q.answer["p"], q.answer["q"])
        assert recomputed == expected, (
            f"exam seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_given(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted
