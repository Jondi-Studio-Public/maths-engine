"""Independent oracle for senior-use-numerical-integration-using-technology-
f-left-t (ACMSM127, topic ``p13c_exponential_pdf_numint``).

Parses ONLY ``question.text``/``question.answer_text`` -- never imports
anything from ``mathgen.generators.p13c_exponential_pdf_numint_build`` (not a
helper, not a constant, not a table). Re-derives the value of the definite
integral from scratch with ``sympy.integrate`` acting on the stated lambda
and bounds pulled out of the rendered prompt, then rounds it to the STATED
precision itself (via ``mathgen.core``'s shared, generic rounding helpers --
never the generator's own construction arithmetic) and compares against
``question.answer_text``.
"""

import re
from fractions import Fraction

import pytest
import sympy as sp

from mathgen import make
from mathgen.core import round_half_up, stated_precision_string

TOPIC = "p13c_exponential_pdf_numint"
SEEDS = range(60)
TIERS = ("easy", "medium", "hard", "exam")
TIER_DP = {"easy": 2, "medium": 2, "hard": 4, "exam": 4}

_t = sp.symbols("t")

_LAM_RE = re.compile(r"f\(t\) = (?P<lam>[\d.]*)e\^\(-(?P=lam)t\)")

_SINGLE_BOUNDS_PATTERNS = (
    re.compile(r"from t = (?P<a>-?\d+(?:\.\d+)?) to t = (?P<b>-?\d+(?:\.\d+)?), correct to (?P<dp>\d+) decimal places"),
    re.compile(
        r"between t = (?P<a>-?\d+(?:\.\d+)?) and t = (?P<b>-?\d+(?:\.\d+)?), correct to (?P<dp>\d+) decimal places"
    ),
    re.compile(
        r"over the interval from t = (?P<a>-?\d+(?:\.\d+)?) to t = (?P<b>-?\d+(?:\.\d+)?)\. "
        r"Give your answer correct to (?P<dp>\d+) decimal places"
    ),
)

_EXAM_BOUNDS_PATTERNS = (
    re.compile(
        r"from t = (?P<a1>-?\d+(?:\.\d+)?) to t = (?P<b1>-?\d+(?:\.\d+)?) and "
        r"(?:from t = |the definite integral of f\(t\) from t = )"
        r"(?P<a2>-?\d+(?:\.\d+)?) to t = (?P<b2>-?\d+(?:\.\d+)?)[.,] "
        r"(?:correct to |Give your answer correct to )(?P<dp>\d+) decimal places"
    ),
    re.compile(
        r"over \[(?P<a1>-?\d+(?:\.\d+)?), (?P<b1>-?\d+(?:\.\d+)?)\] together with \[(?P<a2>-?\d+(?:\.\d+)?), (?P<b2>-?\d+(?:\.\d+)?)\], "
        r"correct to (?P<dp>\d+) decimal places"
    ),
)


def _parse_lambda(text: str) -> sp.Rational:
    m = _LAM_RE.search(text)
    assert m is not None, f"could not parse lambda out of {text!r}"
    return sp.Rational(m.group("lam") or "1")


def _parse_single_bounds(text: str):
    for pattern in _SINGLE_BOUNDS_PATTERNS:
        m = pattern.search(text)
        if m:
            return sp.Rational(m.group("a")), sp.Rational(m.group("b")), int(m.group("dp"))
    raise AssertionError(f"could not parse a/b bounds out of {text!r}")


def _parse_exam_bounds(text: str):
    for pattern in _EXAM_BOUNDS_PATTERNS:
        m = pattern.search(text)
        if m:
            return (
                sp.Rational(m.group("a1")),
                sp.Rational(m.group("b1")),
                sp.Rational(m.group("a2")),
                sp.Rational(m.group("b2")),
                int(m.group("dp")),
            )
    raise AssertionError(f"could not parse the two exam intervals out of {text!r}")


def _oracle_integral(lam: sp.Rational, a: sp.Rational, b: sp.Rational) -> Fraction:
    result = sp.integrate(lam * sp.exp(-lam * _t), (_t, a, b))
    return Fraction(str(result.evalf(40)))


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_matches_single_interval_answer(tier):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        lam = _parse_lambda(q.text)
        a, b, dp = _parse_single_bounds(q.text)
        assert dp == TIER_DP[tier], f"{tier} seed={seed}: parsed dp={dp} != expected {TIER_DP[tier]}"

        raw = _oracle_integral(lam, a, b)
        rounded = round_half_up(raw, dp)
        expected_text = stated_precision_string(rounded, dp)

        assert expected_text == q.answer_text, (
            f"{tier} seed={seed}: oracle computed {expected_text!r} (lam={lam}, a={a}, b={b}), "
            f"generator said {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_matches_exam_tier_disjoint_union_answer():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        lam = _parse_lambda(q.text)
        a1, b1, a2, b2, dp = _parse_exam_bounds(q.text)
        assert dp == TIER_DP["exam"], f"seed={seed}: parsed dp={dp} != expected {TIER_DP['exam']}"

        raw = _oracle_integral(lam, a1, b1) + _oracle_integral(lam, a2, b2)
        rounded = round_half_up(raw, dp)
        expected_text = stated_precision_string(rounded, dp)

        assert expected_text == q.answer_text, (
            f"exam seed={seed}: oracle computed {expected_text!r} "
            f"(lam={lam}, a1={a1}, b1={b1}, a2={a2}, b2={b2}), generator said {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Premise consistency: the stated lambda is genuinely > 0, bounds finite -


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_premise_is_a_bounded_finite_definite_integral(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        lam = _parse_lambda(q.text)
        a, b, _dp = _parse_single_bounds(q.text)
        assert lam > 0, f"{tier} seed={seed}: lambda not positive: {lam}"
        assert a < b, f"{tier} seed={seed}: bounds not ordered a < b: a={a}, b={b}"
