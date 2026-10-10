"""Deliberately does NOT import anything from
``mathgen.generators.p19i_transport_ear_build`` -- it parses
``question.text`` with its own regexes and recomputes the answer from
scratch using its own ``fractions.Fraction`` arithmetic and its own
rounding helper, then compares against ``question.answer_text``.
"""

import math
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC_A = "p19i_transport_time_compare"
TOPIC_B = "p19i_effective_annual_rate"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


def _round_half_up(value: Fraction, places: int) -> Fraction:
    """Ties away from zero, entirely independent of mathgen.core's own
    round_half_up (same convention, separately implemented)."""
    scale = 10**places
    scaled = value * scale
    sign = -1 if scaled < 0 else 1
    magnitude = math.floor(abs(scaled) + Fraction(1, 2))
    return Fraction(sign * magnitude, scale)


def _fmt(value: Fraction, places: int) -> str:
    scaled = value * (10**places)
    assert scaled.denominator == 1, value
    n = scaled.numerator
    if places == 0:
        return str(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole, frac = divmod(n, 10**places)
    return f"{sign}{whole}.{str(frac).zfill(places)}"


# =========================================================================
# Skill A oracle: distance / speed for two named transport modes
# =========================================================================

_LEG_RE = re.compile(r"By ([a-z]+), the average speed is ([\d.]+) (km/h|m/s|km/min)")
_DISTANCE_RE = re.compile(r"distance of (\d+) km")
_FINAL_UNIT_RE = re.compile(r"each in (hours|minutes)")


def _to_kmh(value: Fraction, unit: str) -> Fraction:
    if unit == "km/h":
        return value
    if unit == "m/s":
        return value * Fraction(36, 10)
    if unit == "km/min":
        return value * 60
    raise ValueError(unit)  # pragma: no cover -- exhaustive above


def _oracle_a(text: str) -> str:
    distance = Fraction(int(_DISTANCE_RE.search(text).group(1)))
    legs = _LEG_RE.findall(text)
    assert len(legs) == 2, text
    final_unit = _FINAL_UNIT_RE.search(text).group(1)
    times_hours = []
    for _name, raw_speed, unit in legs:
        speed_kmh = _to_kmh(Fraction(raw_speed), unit)
        times_hours.append(distance / speed_kmh)
    if final_unit == "hours":
        rounded = [_round_half_up(t, 1) for t in times_hours]
        return ", ".join(_fmt(r, 1) for r in rounded)
    rounded = [_round_half_up(t * 60, 0) for t in times_hours]
    return ", ".join(_fmt(r, 0) for r in rounded)


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_agrees_with_the_shipped_answer(difficulty):
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        checked += 1
        oracle_text = _oracle_a(q.text)
        if oracle_text != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, oracle_text))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill A {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_fires_on_a_corrupted_answer(difficulty):
    real = make(TOPIC_A, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer_text=real.answer_text)
    a, b = real.answer_text.split(", ")
    corrupted_text = f"{b}, {a}"  # swap is a real, always-different corruption
    corrupted = SimpleNamespace(text=real.text, answer_text=corrupted_text)
    assert _oracle_a(correct.text) == correct.answer_text
    assert _oracle_a(corrupted.text) != corrupted.answer_text


def test_skill_a_oracle_fires_on_a_broken_question():
    """A deliberately-mangled question (distance changed) makes the
    oracle disagree with the ORIGINAL answer_text -- proves the oracle is
    actually reading the question, not just echoing the canonical."""
    real = make(TOPIC_A, seed=0, difficulty="easy")
    broken_text = re.sub(r"distance of \d+ km", "distance of 999999 km", real.text)
    assert _oracle_a(broken_text) != real.answer_text


# =========================================================================
# Skill B oracle: EAR = (1 + r/n)^n - 1 at two named compounding frequencies
# =========================================================================

_N_FOR_FREQUENCY = {"daily": 365, "monthly": 12, "quarterly": 4, "six-monthly": 2}
_NOUN_TO_FREQUENCY = {"day": "daily", "month": "monthly", "quarter": "quarterly", "six months": "six-monthly"}

_NOMINAL_RE = re.compile(
    r"nominal annual interest rate of ([\d.]+)% is applied.*?"
    r"compounded ([a-z-]+) and when compounded ([a-z-]+)"
)
# hard tier: frequencies are stated as intervals; n = 12 / months.
_HARD_RE = re.compile(
    r"nominal annual interest rate of ([\d.]+)%\. Interest is added "
    r"every (day|month|3 months|6 months) in the first account and "
    r"every (day|month|3 months|6 months) in the second"
)
_N_FOR_INTERVAL = {"day": 365, "month": 12, "3 months": 4, "6 months": 2}
_EXAM_RE = re.compile(
    r"A rate of ([\d.]+)% is applied every (day|month|quarter|six months) "
    r"\(([a-z-]+) compounding\).*?instead compounded ([a-z-]+)"
)


def _ear_oracle(r: Fraction, n: int) -> Fraction:
    return (1 + r / n) ** n - 1


def _ear_oracle_from_period_rate(p: Fraction, n: int) -> Fraction:
    """`p` is already the per-period rate (not a nominal annual rate to be divided by `n`): EAR = (1 + p)^n - 1.
    `_ear_oracle` would divide `p` by `n` a second time."""
    return (1 + p) ** n - 1


def _oracle_b(text: str) -> str:
    m = _EXAM_RE.search(text)
    if m:
        per_period_pct, noun, name1, name2 = m.group(1), m.group(2), m.group(3), m.group(4)
        assert _NOUN_TO_FREQUENCY[noun] == name1, text
        p = Fraction(per_period_pct) / 100
        n1 = _N_FOR_FREQUENCY[name1]
        r_nominal = p * n1
        ear1 = _ear_oracle_from_period_rate(p, n1)
        ear2 = _ear_oracle(r_nominal, _N_FOR_FREQUENCY[name2])
    else:
        m = _HARD_RE.search(text)
        if m:
            r = Fraction(m.group(1)) / 100
            n1, n2 = _N_FOR_INTERVAL[m.group(2)], _N_FOR_INTERVAL[m.group(3)]
        else:
            m = _NOMINAL_RE.search(text)
            assert m, text
            r = Fraction(m.group(1)) / 100
            n1, n2 = _N_FOR_FREQUENCY[m.group(2)], _N_FOR_FREQUENCY[m.group(3)]
        ear1 = _ear_oracle(r, n1)
        ear2 = _ear_oracle(r, n2)
    r1 = _round_half_up(ear1 * 100, 2)
    r2 = _round_half_up(ear2 * 100, 2)
    return ", ".join([_fmt(r1, 2), _fmt(r2, 2)])


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_agrees_with_the_shipped_answer(difficulty):
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        checked += 1
        oracle_text = _oracle_b(q.text)
        if oracle_text != q.answer_text:
            challenged.append((seed, q.text, q.answer_text, oracle_text))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill B {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_fires_on_a_corrupted_answer(difficulty):
    real = make(TOPIC_B, seed=0, difficulty=difficulty)
    a, b = real.answer_text.split(", ")
    corrupted_text = f"{b}, {a}"
    assert _oracle_b(real.text) == real.answer_text
    assert _oracle_b(real.text) != corrupted_text


def test_skill_b_oracle_fires_on_a_broken_question():
    real = make(TOPIC_B, seed=0, difficulty="easy")
    broken_text = re.sub(r"rate of [\d.]+%", "rate of 99%", real.text)
    assert _oracle_b(broken_text) != real.answer_text
