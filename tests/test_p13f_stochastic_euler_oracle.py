"""Re-derives each rendered question's answer from `question.text` alone via a hand-written Euler-Maruyama step,
never importing anything from `mathgen.generators.p13f_stochastic_euler_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p13f_stochastic_euler"

_LAM_RE = re.compile(r"\((-?\d+)\)Y dt")
_FORCING_RE = re.compile(r"Y dt \+ \((-?\d+)\) dt")
_Y0_RE = re.compile(r"Y\(0\) = (-?\d+)")
_H_RE = re.compile(r"h = (-?\d+(?:/\d+)?)")
_SIGMA_CONST_RE = re.compile(r"σ = (-?\d+(?:/\d+)?)")
_K_RE = re.compile(r"σ\(Y\) = \((-?\d+)\)Y")
_DW_RE = re.compile(r"ΔW = (-?\d+(?:/\d+)?)")
_DW1_RE = re.compile(r"ΔW1 = (-?\d+(?:/\d+)?)")
_DW2_RE = re.compile(r"ΔW2 = (-?\d+(?:/\d+)?)")


def _frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _common(text: str):
    lam = int(_LAM_RE.search(text).group(1))
    forcing = int(_FORCING_RE.search(text).group(1))
    y0 = int(_Y0_RE.search(text).group(1))
    h = _frac(_H_RE.search(text).group(1))
    return lam, forcing, y0, h


def _oracle_additive(text: str) -> Fraction:
    lam, forcing, y0, h = _common(text)
    sigma = _frac(_SIGMA_CONST_RE.search(text).group(1))
    dw = _frac(_DW_RE.search(text).group(1))
    y1 = Fraction(y0) + h * (lam * y0 + forcing) + sigma * dw
    return y1


def _oracle_multiplicative(text: str) -> Fraction:
    lam, forcing, y0, h = _common(text)
    k = int(_K_RE.search(text).group(1))
    dw = _frac(_DW_RE.search(text).group(1))
    sigma = Fraction(k) * y0
    y1 = Fraction(y0) + h * (lam * y0 + forcing) + sigma * dw
    return y1


def _oracle_exam(text: str) -> Fraction:
    lam, forcing, y0, h = _common(text)
    k = int(_K_RE.search(text).group(1))
    dw1 = _frac(_DW1_RE.search(text).group(1))
    dw2 = _frac(_DW2_RE.search(text).group(1))
    sigma0 = Fraction(k) * y0
    y1 = Fraction(y0) + h * (lam * y0 + forcing) + sigma0 * dw1
    sigma1 = Fraction(k) * y1
    y2 = y1 + h * (lam * y1 + forcing) + sigma1 * dw2
    return y2


@pytest.mark.parametrize(
    "tier,fn",
    [
        ("easy", _oracle_additive),
        ("medium", _oracle_additive),
        ("hard", _oracle_multiplicative),
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
    recomputed = _oracle_additive(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = recomputed + 1
    assert recomputed != corrupted
