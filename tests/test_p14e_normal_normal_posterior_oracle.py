"""Re-derives each rendered question's posterior mean and variance from `question.text` alone via the closed-form
weighted-average update, never importing anything from
`mathgen.generators.p14e_normal_normal_posterior_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14e_normal_normal_posterior"

_MU0_RE = re.compile(r"mu0 = (-?\d+)")
_TAU0_SQ_RE = re.compile(r"tau0\^2 = (-?\d+)")
_LAMBDA0_RE = re.compile(r"lambda0 = (-?\d+)")
_SIGMA_SQ_RE = re.compile(r"sigma\^2 = (-?\d+)")
_N_RE = re.compile(r"\bn = (-?\d+)\b")
_XBAR_RE = re.compile(r"x-bar = (-?\d+(?:/\d+)?)")
_N1_RE = re.compile(r"\bn1 = (-?\d+)\b")
_N2_RE = re.compile(r"\bn2 = (-?\d+)\b")
_XBAR1_RE = re.compile(r"x-bar_1 = (-?\d+(?:/\d+)?)")
_XBAR2_RE = re.compile(r"x-bar_2 = (-?\d+(?:/\d+)?)")
_N_PAREN_RE = re.compile(r"N\((-?\d+),\s*(-?\d+)\)")
_MEAN_RE = re.compile(r"mean = (-?\d+(?:/\d+)?)")
_VARIANCE_RE = re.compile(r"variance = (-?\d+(?:/\d+)?)")


def _frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _update(mu0: Fraction, tau0_sq: Fraction, sigma_sq: Fraction, n: int, xbar: Fraction):
    precision1 = Fraction(1, 1) / tau0_sq + Fraction(n) / sigma_sq
    tau1_sq = Fraction(1, 1) / precision1
    mu1 = tau1_sq * (mu0 / tau0_sq + Fraction(n) * xbar / sigma_sq)
    return mu1, tau1_sq


def _oracle_given(text: str) -> tuple[Fraction, Fraction]:
    """easy/medium: prior stated as a variance, single update."""
    mu0 = Fraction(int(_MU0_RE.search(text).group(1)))
    tau0_sq = Fraction(int(_TAU0_SQ_RE.search(text).group(1)))
    sigma_sq = Fraction(int(_SIGMA_SQ_RE.search(text).group(1)))
    n = int(_N_RE.search(text).group(1))
    xbar = _frac(_XBAR_RE.search(text).group(1))
    return _update(mu0, tau0_sq, sigma_sq, n, xbar)


def _oracle_hard(text: str) -> tuple[Fraction, Fraction]:
    """hard: prior stated as a precision lambda0 = 1/tau0^2."""
    mu0 = Fraction(int(_MU0_RE.search(text).group(1)))
    lambda0 = Fraction(int(_LAMBDA0_RE.search(text).group(1)))
    tau0_sq = Fraction(1, 1) / lambda0
    sigma_sq = Fraction(int(_SIGMA_SQ_RE.search(text).group(1)))
    n = int(_N_RE.search(text).group(1))
    xbar = _frac(_XBAR_RE.search(text).group(1))
    return _update(mu0, tau0_sq, sigma_sq, n, xbar)


def _oracle_exam(text: str) -> tuple[Fraction, Fraction]:
    """exam: two sequential updates, chaining the first posterior into the
    second update's prior. All three exam templates state the prior as
    "N(mu0, tau0^2)" -- only some also repeat mu0/tau0^2 with their own
    labels, so this reads the N(...) parenthesised form directly rather than
    relying on the "mu0 = "/"tau0^2 = " labels the easy/medium/hard oracles
    use."""
    paren = _N_PAREN_RE.search(text)
    mu0 = Fraction(int(paren.group(1)))
    tau0_sq = Fraction(int(paren.group(2)))
    sigma_sq = Fraction(int(_SIGMA_SQ_RE.search(text).group(1)))
    n1 = int(_N1_RE.search(text).group(1))
    xbar1 = _frac(_XBAR1_RE.search(text).group(1))
    n2 = int(_N2_RE.search(text).group(1))
    xbar2 = _frac(_XBAR2_RE.search(text).group(1))
    mu_a, tau_a_sq = _update(mu0, tau0_sq, sigma_sq, n1, xbar1)
    return _update(mu_a, tau_a_sq, sigma_sq, n2, xbar2)


def _canonical_pair(text: str) -> tuple[Fraction, Fraction]:
    return _frac(_MEAN_RE.search(text).group(1)), _frac(_VARIANCE_RE.search(text).group(1))


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
        expected = (Fraction(q.answer["mean"]), Fraction(q.answer["variance"]))
        assert recomputed == expected, (
            f"{tier} seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        # Also confirm the answer_text a learner actually reads matches --
        # verifies the artefact the learner receives, not just `q.answer`.
        assert recomputed == _canonical_pair(q.answer_text)
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        recomputed = _oracle_exam(q.text)
        expected = (Fraction(q.answer["mean"]), Fraction(q.answer["variance"]))
        assert recomputed == expected, (
            f"exam seed={seed}: oracle {recomputed} != canonical {expected} (text={q.text!r})"
        )
        assert recomputed == _canonical_pair(q.answer_text)
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    recomputed = _oracle_given(q.text)
    corrupted = (recomputed[0] + 1, recomputed[1])
    assert recomputed != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    recomputed = _oracle_exam(q.text)
    corrupted = (recomputed[0], recomputed[1] + 1)
    assert recomputed != corrupted
