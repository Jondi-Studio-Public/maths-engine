"""Independent oracle for the rate calculation and rate comparison questions of p7d."""

import functools

"""Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p7d_rates_build` and recomputes the expected answer from
first principles (rate x time, or a rate difference/comparison), then
compares it against `answer_text`'s own displayed numeric value, including
its stated decimal precision.

Generates through each skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

CALC_TOPIC = "p7d_complete_rate_calculations"
COMPARE_TOPIC = "p7d_use_rates_to_compare"

_CALC_ID = "senior-complete-calculations-with-rates-including-solving-problems-involving"
_COMPARE_ID = "senior-use-rates-to-make-comparisons"

SEEDS = range(60)


# `load_skills()` rescans every skill YAML from disk and costs ~560 ms. These
# helpers are called once per SEED, so an uncached lookup spent ~290 s of the
# suite reloading the same library. Cached: `make()` still runs per seed and
# every assertion is unchanged -- only the repeated disk scan is removed.
@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _num(token: str) -> Fraction:
    """A plain or "$"-prefixed decimal literal -> exact Fraction value."""
    return Fraction(token.lstrip("$"))


def _answer_value(answer_text: str) -> Fraction:
    return _num(answer_text)


# =============================================================================
# Skill 1: p7d_complete_rate_calculations (ACMEM073)
# =============================================================================

_CALC_EASY_PATTERNS = [
    re.compile(r"A car travels at (\d+) km/h\. How far does it travel in (\d+) hours\?"),
    re.compile(r"Water flows from a tap at (\d+) litres per minute\. How many litres flow in (\d+) minutes\?"),
    re.compile(r"A tool-hire company charges \$([\d.]+) per hour\. How much does (\d+) hours of hire cost\?"),
    re.compile(r"A machine produces (\d+) items per hour\. How many items does it produce in (\d+) hours\?"),
]

_CALC_MEDIUM_PATTERNS = [
    # (regex, which slot is the unknown: "n" means solve total/known_rate=n; "r" means solve total/known_n=r)
    (
        re.compile(r"A car travels (\d+) km in (\d+) hours at a constant speed\. What is its speed, in km/h\?"),
        "rate_given_total_and_n",
    ),
    (
        re.compile(r"A car travels at (\d+) km/h and covers (\d+) km\. How many hours does the journey take\?"),
        "time_given_r_and_total",
    ),
    (
        re.compile(
            r"A tap fills a tank with (\d+) litres in (\d+) minutes at a constant rate\. "
            r"What is the flow rate, in litres per minute\?"
        ),
        "rate_given_total_and_n",
    ),
    (
        re.compile(
            r"A tap flows at (\d+) litres per minute and fills a container with (\d+) litres\. "
            r"How many minutes does this take\?"
        ),
        "time_given_r_and_total",
    ),
    (
        re.compile(
            r"A worker earns \$([\d.]+) for (\d+) hours of work at a constant hourly rate\. "
            r"What is their hourly pay rate\?"
        ),
        "rate_given_total_and_n",
    ),
    (
        re.compile(
            r"A worker is paid \$([\d.]+) per hour and earns \$([\d.]+) in total\. "
            r"How many hours did they work\?"
        ),
        "time_given_r_and_total",
    ),
    (
        re.compile(
            r"A machine produces (\d+) items in (\d+) hours at a constant rate\. "
            r"What is its production rate, in items per hour\?"
        ),
        "rate_given_total_and_n",
    ),
    (
        re.compile(
            r"A machine produces (\d+) items per hour and produces (\d+) items in total\. "
            r"How many hours does this take\?"
        ),
        "time_given_r_and_total",
    ),
]

_CALC_HARD_PATTERNS = [
    re.compile(
        r"Tap A fills a tank at (\d+) litres per minute\. Tap B fills the same tank at (\d+) litres "
        r"per minute\. Working together, starting at the same time, they fill a (\d+) litre tank from "
        r"empty\. How many minutes does it take to fill the tank\?"
    ),
    re.compile(
        r"Worker A completes a job at (\d+) items per hour\. Worker B completes the same kind of job "
        r"at (\d+) items per hour\. Working together, starting at the same time, they must complete a "
        r"job requiring (\d+) items in total\. How many hours does it take them to finish\?"
    ),
]


def _oracle_calc_easy(text: str) -> Fraction:
    for pat in _CALC_EASY_PATTERNS:
        m = pat.search(text)
        if m:
            rate, qty = _num(m.group(1)), _num(m.group(2))
            return rate * qty
    raise AssertionError(f"unparseable easy-tier calc question: {text!r}")


def _oracle_calc_medium(text: str) -> Fraction:
    for pat, kind in _CALC_MEDIUM_PATTERNS:
        m = pat.search(text)
        if m:
            first, second = _num(m.group(1)), _num(m.group(2))
            if kind == "rate_given_total_and_n":
                # "total ... in n ..." -> rate = total / n
                total, n = first, second
                return total / n
            # kind == "time_given_r_and_total": "at r ... covers total ..." -> time = total / r
            r, total = first, second
            return total / r
    raise AssertionError(f"unparseable medium-tier calc question: {text!r}")


def _oracle_calc_hard(text: str) -> Fraction:
    for pat in _CALC_HARD_PATTERNS:
        m = pat.search(text)
        if m:
            a, b, total = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
            return total / (a + b)
    raise AssertionError(f"unparseable hard-tier calc question: {text!r}")


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_calc_easy),
        ("medium", _oracle_calc_medium),
        ("hard", _oracle_calc_hard),
    ],
)
def test_calc_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _CALC_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _answer_value(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# Skill 2: p7d_use_rates_to_compare (ACMEM074)
# =============================================================================

_COMPARE_EASY_PATTERNS = [
    re.compile(
        r"Car A travels at (\d+) km/h and Car B at (\d+) km/h\. "
        r"How many km/h faster is Car ([AB])\?"
    ),
    re.compile(
        r"Tap A flows at (\d+) litres per minute and Tap B at (\d+) litres per minute\. "
        r"How many litres per minute faster is Tap ([AB])\?"
    ),
    re.compile(
        r"Worker A is paid \$([\d.]+) per hour and Worker B is paid \$([\d.]+) per hour\. "
        r"How much more per hour does Worker ([AB]) earn\?"
    ),
    re.compile(
        r"Machine A produces (\d+) items per hour and Machine B produces (\d+) items per hour\. "
        r"How many more items per hour does Machine ([AB]) produce\?"
    ),
]

_COMPARE_MEDIUM_PATTERNS = [
    re.compile(
        r"Car A travels (\d+) km in (\d+) hours\. Car B travels (\d+) km in (\d+) hours\. "
        r"Based on their average speeds, how many km/h faster is Car ([AB])\?"
    ),
    re.compile(
        r"Tap A releases (\d+) litres in (\d+) minutes\. Tap B releases (\d+) litres in "
        r"(\d+) minutes\. How many litres per minute faster is Tap ([AB])\?"
    ),
    re.compile(
        r"Worker A earns \$([\d.]+) for (\d+) hours of work\. Worker B earns \$([\d.]+) for "
        r"(\d+) hours of work\. How much more per hour does Worker ([AB]) earn\?"
    ),
    re.compile(
        r"Machine A produces (\d+) items in (\d+) hours\. Machine B produces (\d+) items in "
        r"(\d+) hours\. How many more items per hour does Machine ([AB]) produce\?"
    ),
]

_COMPARE_HARD_PATTERN = re.compile(
    r"Brand A sells a (\d+) mL bottle for \$([\d.]+)\. Brand B sells a (\d+) L bottle for \$([\d.]+)\."
)


def _oracle_compare_easy(text: str) -> Fraction:
    for pat in _COMPARE_EASY_PATTERNS:
        m = pat.search(text)
        if m:
            rA, rB = _num(m.group(1)), _num(m.group(2))
            return abs(rA - rB)
    raise AssertionError(f"unparseable easy-tier compare question: {text!r}")


def _oracle_compare_medium(text: str) -> Fraction:
    for pat in _COMPARE_MEDIUM_PATTERNS:
        m = pat.search(text)
        if m:
            totalA, hoursA, totalB, hoursB = (_num(m.group(1)), _num(m.group(2)), _num(m.group(3)), _num(m.group(4)))
            rateA, rateB = totalA / hoursA, totalB / hoursB
            return abs(rateA - rateB)
    raise AssertionError(f"unparseable medium-tier compare question: {text!r}")


def _oracle_compare_hard(text: str) -> Fraction:
    m = _COMPARE_HARD_PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier compare question: {text!r}")
    volA_mL, packA_price, volB_L, packB_price = (_num(m.group(1)), _num(m.group(2)), _num(m.group(3)), _num(m.group(4)))
    # mL -> L by dividing by 1000, per the question's own stated conversion.
    priceA_perL = packA_price / (volA_mL / 1000)
    priceB_perL = packB_price / volB_L
    diff = abs(priceA_perL - priceB_perL)
    # "to the nearest cent" -- round half up to 2dp, matching the canonical
    # answer's own stated precision.
    cents = int((diff * 100 + Fraction(1, 2)))
    return Fraction(cents, 100)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_compare_easy),
        ("medium", _oracle_compare_medium),
        ("hard", _oracle_compare_hard),
    ],
)
def test_compare_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, _COMPARE_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _answer_value(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- Distractors, both skills: the oracle's own recomputation must never
# equal a distractor either (a stronger check than "the generator's own
# check_distinct passed") --------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,topic,difficulty,oracle_fn",
    [
        (_CALC_ID, CALC_TOPIC, "easy", _oracle_calc_easy),
        (_CALC_ID, CALC_TOPIC, "medium", _oracle_calc_medium),
        (_CALC_ID, CALC_TOPIC, "hard", _oracle_calc_hard),
        (_COMPARE_ID, COMPARE_TOPIC, "easy", _oracle_compare_easy),
        (_COMPARE_ID, COMPARE_TOPIC, "medium", _oracle_compare_medium),
        (_COMPARE_ID, COMPARE_TOPIC, "hard", _oracle_compare_hard),
    ],
)
def test_oracle_answer_never_matches_a_distractor(skill_id, topic, difficulty, oracle_fn):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, skill_id, difficulty, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            assert expected != _num(d), (
                f"seed {seed} {difficulty} {topic}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
