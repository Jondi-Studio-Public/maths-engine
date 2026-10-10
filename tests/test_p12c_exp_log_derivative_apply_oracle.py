"""Independent oracle for the `p12c_exp_log_derivative_apply` questions: it re-derives each answer from the
rendered question text.

Imports nothing from `mathgen.generators.p12c_exp_log_derivative_apply_build`:
not the formula helpers, not the templates, not the misconception constants,
not the boundary/rounding guards. Parses `N0`, `k`, and whichever of `t0`/`R`/
`M` the tier states straight out of the RENDERED `question.text`, and
re-derives the correct answer from scratch via plain `math.log`/`math.exp`
calls and `round_half_up`, a route independent of however the generator's
own construction built the scenario (construct-to-target may work backward
from a chosen answer; this file always works FORWARD from the stated numbers
alone, using its own re-implementation of half-up rounding rather than the
generator's shared helper).

Run across >= 1000 seeds total, spread over the eight (topic, tier) pairs.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

EXP_TOPIC = "p12c_exp_deriv_apply"
LOG_TOPIC = "p12c_log_deriv_apply"
SEEDS_PER_TIER = 130  # 8 (topic, tier) pairs x 130 = 1040 >= 1000


def _round_half_up_2dp(value: float) -> str:
    """Independent re-implementation of half-up rounding to 2 decimal places
    -- deliberately NOT a call to `mathgen.core.round_half_up`, so a bug in
    the shared helper (used by the generator itself) would not silently
    cancel out here."""
    scaled = math.floor(abs(value) * 100 + 0.5)
    sign = "-" if value < 0 and scaled != 0 else ""
    whole, frac = divmod(scaled, 100)
    return f"{sign}{whole}.{frac:02d}"


# --- Exponential: N(t) = N0 * e^(k*t) ---------------------------------------

_EXP_EASY_RE = re.compile(
    r"N\(t\) = (?P<N0>\d+)·e\^\((?P<k>-?\d+\.\d+)t\), where t >= 0\. "
    r"Find the instantaneous rate of change of N with respect to t at t = 0\. "
    r"Give your answer correct to 2 decimal places\.$"
)
_EXP_MEDIUM_RE = re.compile(
    r"N\(t\) = (?P<N0>\d+)·e\^\((?P<k>-?\d+\.\d+)t\), where t >= 0\. "
    r"Find the instantaneous rate of change of N with respect to t at t = (?P<t0>\d+)\. "
    r"Give your answer correct to 2 decimal places\.$"
)
_EXP_HARD_RE = re.compile(
    r"N\(t\) = (?P<N0>\d+)·e\^\((?P<k>-?\d+\.\d+)t\), where t >= 0\. "
    r"The rate of change of N first reaches (?P<R>\d+) \(the same units as N, per \w+\) at some time t\. "
    r"Find this value of t, in \w+\. Give your answer correct to 2 decimal places\.$"
)
_EXP_EXAM_RE = re.compile(
    r"N\(t\) = (?P<N0>\d+)·e\^\((?P<k>-?\d+\.\d+)t\), where t >= 0\. "
    r"At some time t, N\(t\) = (?P<M>\d+)\. "
    r"Find the instantaneous rate of change of N with respect to t at that moment\. "
    r"Give your answer correct to 2 decimal places\.$"
)


def test_oracle_exp_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(EXP_TOPIC, seed=seed, difficulty="easy", mode="rate_at_zero")
        m = _EXP_EASY_RE.search(q.text)
        assert m, (seed, q.text)
        N0, k = int(m["N0"]), float(m["k"])
        rate = N0 * k
        assert _round_half_up_2dp(rate) == q.answer_text, (seed, q.text, q.answer_text, rate)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exp_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(EXP_TOPIC, seed=seed, difficulty="medium", mode="rate_at_t")
        m = _EXP_MEDIUM_RE.search(q.text)
        assert m, (seed, q.text)
        N0, k, t0 = int(m["N0"]), float(m["k"]), int(m["t0"])
        rate = N0 * k * math.exp(k * t0)
        assert _round_half_up_2dp(rate) == q.answer_text, (seed, q.text, q.answer_text, rate)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exp_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(EXP_TOPIC, seed=seed, difficulty="hard", mode="time_for_rate")
        m = _EXP_HARD_RE.search(q.text)
        assert m, (seed, q.text)
        N0, k, R = int(m["N0"]), float(m["k"]), int(m["R"])
        base_rate = N0 * k
        assert base_rate > 0, (seed, q.text, base_rate)
        t = math.log(R / base_rate) / k
        assert _round_half_up_2dp(t) == q.answer_text, (seed, q.text, q.answer_text, t)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exp_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(EXP_TOPIC, seed=seed, difficulty="exam", mode="rate_at_value")
        m = _EXP_EXAM_RE.search(q.text)
        assert m, (seed, q.text)
        N0, k, M = int(m["N0"]), float(m["k"]), int(m["M"])
        # dN/dt = kN, so at the moment N(t) = M, dN/dt = k*M -- the same
        # identity the generator uses, but re-derived here from the
        # rendered numbers alone rather than trusted from construction.
        rate = k * M
        assert _round_half_up_2dp(rate) == q.answer_text, (seed, q.text, q.answer_text, rate)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Logarithmic: V(x) = a*ln(x) + b -----------------------------------------

_LOG_EASY_RE = re.compile(
    r"V\(x\) = (?P<a>\d+)·ln\(x\) (?P<bsign>[+-]) (?P<b>\d+), for x > 0\. "
    r"Find the instantaneous rate of change of V with respect to x at x = (?P<x>\d+)\. "
    r"Give your answer correct to 2 decimal places\.$"
)
_LOG_MEDIUM_RE = re.compile(
    r"V\(x\) = (?P<a>\d+)·ln\(x\) (?P<bsign>[+-]) (?P<b>\d+), for x > 0\. "
    r"The value of x starts at (?P<x_base>\d+) and is (?P<mult>\d+) times that once the system settles\. "
    r"Find the instantaneous rate of change of V with respect to x once the system has settled\. "
    r"Give your answer correct to 2 decimal places\.$"
)
_LOG_HARD_RE = re.compile(
    r"V\(x\) = (?P<a>\d+)·ln\(x\) (?P<bsign>[+-]) (?P<b>\d+), for x > 0\. "
    r"The rate of change of V first equals (?P<R>\d+\.\d+) at some value of x\. "
    r"Find this value of x\. Give your answer correct to 2 decimal places\.$"
)
_LOG_EXAM_RE = re.compile(
    r"V\(x\) = (?P<a>\d+)·ln\(x\) (?P<bsign>[+-]) (?P<b>\d+), for x > 0\. "
    r"The rate of change of V first equals (?P<R>\d+\.\d+) at some value of x\. "
    r"Find V\(x\) at that value of x\. Give your answer correct to 2 decimal places\.$"
)


def test_oracle_log_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(LOG_TOPIC, seed=seed, difficulty="easy", mode="rate_at_x")
        m = _LOG_EASY_RE.search(q.text)
        assert m, (seed, q.text)
        a, x = int(m["a"]), int(m["x"])
        rate = Fraction(a, x)
        assert _round_half_up_2dp(float(rate)) == q.answer_text, (seed, q.text, q.answer_text, rate)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_log_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(LOG_TOPIC, seed=seed, difficulty="medium", mode="rate_at_resolved_x")
        m = _LOG_MEDIUM_RE.search(q.text)
        assert m, (seed, q.text)
        a, x_base, mult = int(m["a"]), int(m["x_base"]), int(m["mult"])
        x = x_base * mult
        rate = Fraction(a, x)
        assert _round_half_up_2dp(float(rate)) == q.answer_text, (seed, q.text, q.answer_text, rate)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_log_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(LOG_TOPIC, seed=seed, difficulty="hard", mode="x_for_rate")
        m = _LOG_HARD_RE.search(q.text)
        assert m, (seed, q.text)
        a, R = int(m["a"]), float(m["R"])
        x = a / R
        assert _round_half_up_2dp(x) == q.answer_text, (seed, q.text, q.answer_text, x)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_log_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(LOG_TOPIC, seed=seed, difficulty="exam", mode="value_at_rate")
        m = _LOG_EXAM_RE.search(q.text)
        assert m, (seed, q.text)
        a, b_sign, b, R = int(m["a"]), m["bsign"], int(m["b"]), float(m["R"])
        b_signed = b if b_sign == "+" else -b
        x = a / R
        V = a * math.log(x) + b_signed
        assert _round_half_up_2dp(V) == q.answer_text, (seed, q.text, q.answer_text, V)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Prove the oracle is red first: a corrupted answer must be caught -------


def test_oracle_catches_a_corrupted_answer():
    """The oracle's own assertion, re-run against a deliberately corrupted
    `answer_text` (an off-by-one), must actually raise -- proving the check
    fires rather than trivially passing (oracles are proven red first)."""
    q = make(EXP_TOPIC, seed=0, difficulty="easy", mode="rate_at_zero")
    m = _EXP_EASY_RE.search(q.text)
    N0, k = int(m["N0"]), float(m["k"])
    rate = N0 * k
    corrupted_answer_text = _round_half_up_2dp(rate + 1.0)
    try:
        assert _round_half_up_2dp(rate) == corrupted_answer_text
    except AssertionError:
        pass
    else:
        raise AssertionError("oracle failed to catch a corrupted answer_text")
