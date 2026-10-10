"""Re-derives the answer from the numbers stated in `question.text` alone, via a units table hand-written here --
does NOT import `mathgen.generators.p19h_energy_build` (no catalogue, no helper, no constant shared with
construction)."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

from mathgen import make

KWH_TOPIC = "p19h_energy_kwh_consumption"
KJ_TOPIC = "p19h_energy_kj_amount"
SEEDS = range(40)


def _round(value: Fraction, dp: int) -> Decimal:
    d = Decimal(value.numerator) / Decimal(value.denominator)
    quant = Decimal(1).scaleb(-dp)
    return d.quantize(quant, rounding=ROUND_HALF_UP)


def _expect_text(value: Fraction, dp: int) -> str:
    d = _round(value, dp)
    return f"{d:.{dp}f}" if dp > 0 else str(int(d))


# --- kWh consumption: power (kW or W) x duration (hours or minutes) ---------

_KWH_ONE_ITEM_RE = re.compile(
    r"power rating of ([\d.]+) (kW|W).*?(?:runs for|switched on for|in use for) (\d+) (hours?|minutes)"
)
_KWH_TWO_ITEM_RE = re.compile(
    r"power rating of ([\d.]+) (kW|W) and runs for (\d+) (hours?|minutes)\. "
    r".*? power rating of ([\d.]+) (kW|W) and runs for (\d+) (hours?|minutes)\."
)


def _power_to_kw(value_str: str, unit: str) -> Fraction:
    value = Fraction(value_str)
    return value if unit == "kW" else value / 1000


def _time_to_hours(value: int, unit: str) -> Fraction:
    return Fraction(value) if unit.startswith("hour") else Fraction(value, 60)


def test_oracle_kwh_single_item_tiers():
    """Covers kwh_easy and kwh_medium -- ONE power+duration clause."""
    checked = 0
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(KWH_TOPIC, seed=seed, difficulty=tier)
            m = _KWH_ONE_ITEM_RE.search(q.text)
            assert m, f"{tier}/{seed}: unexpected text: {q.text!r}"
            power_str, power_unit, time_val, time_unit = m.groups()
            kw = _power_to_kw(power_str, power_unit)
            hours = _time_to_hours(int(time_val), time_unit)
            expected = _expect_text(kw * hours, 1)
            assert expected == q.answer_text, (
                f"{tier}/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
            )
            checked += 1
    assert checked == 2 * len(SEEDS)


def test_oracle_kwh_two_item_tiers():
    """hard (both durations in hours) and exam (second in minutes): sum of the
    two appliances' own kW x hours."""
    checked = 0
    for tier in ("hard", "exam"):
        for seed in SEEDS:
            q = make(KWH_TOPIC, seed=seed, difficulty=tier)
            m = _KWH_TWO_ITEM_RE.search(q.text)
            assert m, f"{tier}/{seed}: unexpected text: {q.text!r}"
            p1, u1, t1, tu1, p2, u2, t2, tu2 = m.groups()
            total = _power_to_kw(p1, u1) * _time_to_hours(int(t1), tu1) + _power_to_kw(p2, u2) * _time_to_hours(
                int(t2), tu2
            )
            expected = _expect_text(total, 1)
            assert expected == q.answer_text, (
                f"{tier}/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
            )
            checked += 1
    assert checked == 2 * len(SEEDS)


# --- kJ activity amount: rate (per min/hour/sec) x duration ------------------

_KJ_ONE_ACTIVITY_RE = re.compile(
    r"energy-use rate of ([\d.]+) kJ per (minute|hour)\. "
    r"The activity lasts (\d+) minutes\."
)
_KJ_TWO_ACTIVITY_RE = re.compile(
    r"a rate of (\d+) kJ per minute for (\d+) minutes\. "
    r".*? a rate of (\d+) kJ per hour for (\d+) minutes\."
)
_KJ_EXAM_RE = re.compile(
    r"energy-use rate of ([\d.]+) kJ per second\. "
    r"The activity lasts (\d+) hours? (\d+) minutes\."
)


def test_oracle_kj_easy_and_medium_tiers():
    """kj_easy: rate per minute, duration in minutes -- direct multiply.
    kj_medium: rate per HOUR, duration in minutes -- convert rate to
    per-minute (divide by 60) before multiplying."""
    checked = 0
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(KJ_TOPIC, seed=seed, difficulty=tier)
            m = _KJ_ONE_ACTIVITY_RE.search(q.text)
            assert m, f"{tier}/{seed}: unexpected text: {q.text!r}"
            rate_str, rate_unit, duration_str = m.groups()
            rate = Fraction(rate_str)
            duration = int(duration_str)
            rate_per_min = rate if rate_unit == "minute" else rate / 60
            expected = _expect_text(rate_per_min * duration, 0)
            assert expected == q.answer_text, (
                f"{tier}/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
            )
            checked += 1
    assert checked == 2 * len(SEEDS)


def test_oracle_kj_hard_two_activity_tier():
    checked = 0
    for seed in SEEDS:
        q = make(KJ_TOPIC, seed=seed, difficulty="hard")
        m = _KJ_TWO_ACTIVITY_RE.search(q.text)
        assert m, f"hard/{seed}: unexpected text: {q.text!r}"
        rate1, dur1, rate2_hour, dur2 = (int(g) for g in m.groups())
        energy1 = rate1 * dur1
        energy2 = Fraction(rate2_hour, 60) * dur2
        expected = _expect_text(energy1 + energy2, 0)
        assert expected == q.answer_text, f"hard/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_kj_exam_tier():
    """Rate per second, duration in mixed "H hour(s) M minutes" form --
    convert the duration to seconds (via minutes) and multiply."""
    checked = 0
    for seed in SEEDS:
        q = make(KJ_TOPIC, seed=seed, difficulty="exam")
        m = _KJ_EXAM_RE.search(q.text)
        assert m, f"exam/{seed}: unexpected text: {q.text!r}"
        rate_sec_str, hours_str, minutes_str = m.groups()
        rate_sec = Fraction(rate_sec_str)
        total_seconds = (int(hours_str) * 60 + int(minutes_str)) * 60
        expected = _expect_text(rate_sec * total_seconds, 0)
        assert expected == q.answer_text, f"exam/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        checked += 1
    assert checked == len(SEEDS)
