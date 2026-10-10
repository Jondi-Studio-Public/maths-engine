"""Re-derives each rendered question's answer from `question.text` alone via the closed form rel_out_bound =
kappa * rel_in, never importing anything from `mathgen.generators.p14c_condition_sensitivity_build`.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14c_condition_sensitivity"

_KAPPA_RE = re.compile(r"kappa = (-?\d+(?:/\d+)?)")
_REL_IN_RE = re.compile(r"relative input error (?:of )?(-?\d+(?:/\d+)?)")

_HARD_PATTERNS = (
    re.compile(r"true value (-?\d+) and absolute error (-?\d+(?:/\d+)?)"),
    re.compile(r"input value is (-?\d+) with an absolute error of (-?\d+(?:/\d+)?)"),
    re.compile(r"input value (-?\d+) and absolute input error (-?\d+(?:/\d+)?)"),
)

_BOUND_PATTERNS = (
    re.compile(r"works out to (-?\d+(?:/\d+)?) for some"),
    re.compile(r"rel_in = (-?\d+(?:/\d+)?) for some"),
    re.compile(r"equals (-?\d+(?:/\d+)?)\."),
)


def _frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _oracle_given(text: str) -> Fraction:
    kappa = _frac(_KAPPA_RE.search(text).group(1))
    rel_in = _frac(_REL_IN_RE.search(text).group(1))
    return kappa * rel_in


def _oracle_hard(text: str) -> Fraction:
    kappa = _frac(_KAPPA_RE.search(text).group(1))
    input_val = abs_err = None
    for pattern in _HARD_PATTERNS:
        m = pattern.search(text)
        if m:
            input_val = int(m.group(1))
            abs_err = _frac(m.group(2))
            break
    assert input_val is not None, f"could not find input value/abs error in {text!r}"
    rel_in = abs_err / abs(input_val)
    return kappa * rel_in


def _oracle_exam(text: str) -> Fraction:
    kappa = _frac(_KAPPA_RE.search(text).group(1))
    bound = None
    for pattern in _BOUND_PATTERNS:
        m = pattern.search(text)
        if m:
            bound = _frac(m.group(1))
            break
    assert bound is not None, f"could not find bound in {text!r}"
    return bound / kappa


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
