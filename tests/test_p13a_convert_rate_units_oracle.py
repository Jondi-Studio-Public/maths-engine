"""Independent oracle for the `p13a_convert_rate_units` questions: it re-derives each answer from the rendered
question text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p13a_convert_rate_units_build` and recomputes the
expected answer from first principles, using its OWN unit-conversion
factor table (1 km = 1000 m, 1 h = 3600 s, 1 L = 1000 mL -- none of these
constants are imported from the generator module).

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "senior-convert-units-of-rates-occurring-in-practical-situations"
SEEDS = range(60)

# The oracle's own conversion-factor table -- never imported from the
# generator module.
_M_PER_KM = 1000
_S_PER_H = 3600
_ML_PER_L = 1000  # unused directly here (no mL appears in this skill's text)
# but kept for parity with the generator's table.


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _num(token: str) -> Fraction:
    return Fraction(token.lstrip("$"))


def _kmh_to_ms(kmh: Fraction) -> Fraction:
    """The oracle's own km/h -> m/s conversion, built from km=1000m, h=3600s."""
    return kmh * _M_PER_KM / _S_PER_H


# --- easy: distance = converted-speed x time -----------------------------

_EASY_RE = re.compile(r"at ([\d.]+) km/h\..*?in (\d+) seconds\?")


def _oracle_easy(text: str) -> Fraction:
    m = _EASY_RE.search(text)
    assert m, f"unparseable easy question: {text!r}"
    speed_kmh, t = _num(m.group(1)), _num(m.group(2))
    speed_ms = _kmh_to_ms(speed_kmh)
    return speed_ms * t


# --- medium: time = distance / converted-speed ---------------------------

_MEDIUM_RE = re.compile(r"at ([\d.]+) km/h\..*?travel (\d+) metres\?")


def _oracle_medium(text: str) -> Fraction:
    m = _MEDIUM_RE.search(text)
    assert m, f"unparseable medium question: {text!r}"
    speed_kmh, distance = _num(m.group(1)), _num(m.group(2))
    speed_ms = _kmh_to_ms(speed_kmh)
    return distance / speed_ms


# --- hard: litres = (consumption / 100) x (distance_m / 1000) ------------

_HARD_RE = re.compile(
    r"consumes (\d+) litres of fuel per 100 km\. For a trip of (\d+) m, "
    r"how many litres of fuel does the \w+(?: \w+)? use, to 2 decimal places\?"
)


def _oracle_hard(text: str) -> Fraction:
    m = _HARD_RE.search(text)
    assert m, f"unparseable hard question: {text!r}"
    consumption_per_100km, distance_m = _num(m.group(1)), _num(m.group(2))
    distance_km = distance_m / _M_PER_KM
    consumption_per_km = consumption_per_100km / 100
    return consumption_per_km * distance_km


# --- exam: total cost = price-per-L x consumption-per-km x distance_km ---

_EXAM_RE = re.compile(
    r"Fuel costs (\d+) cents per litre\. A \w+(?: \w+)? consumes (\d+) litres of fuel per 100 km\. "
    r"For a trip of (\d+) km, find the total cost of fuel, to the nearest cent\."
)


def _oracle_exam(text: str) -> Fraction:
    m = _EXAM_RE.search(text)
    assert m, f"unparseable exam question: {text!r}"
    price_cents_per_L, consumption_per_100km, distance_km = (_num(m.group(1)), _num(m.group(2)), _num(m.group(3)))
    price_per_L = price_cents_per_L / 100  # cents -> dollars
    consumption_per_km = consumption_per_100km / 100
    exact_dollars = price_per_L * consumption_per_km * distance_km
    # "to the nearest cent": round half up.
    cents = int(exact_dollars * 100 + Fraction(1, 2))
    return Fraction(cents, 100)


_ORACLES = {
    "easy": _oracle_easy,
    "medium": _oracle_medium,
    "hard": _oracle_hard,
    "exam": _oracle_exam,
}


@pytest.mark.parametrize("difficulty,oracle_fn", sorted(_ORACLES.items()))
def test_oracle_matches_answer(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _num(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty,oracle_fn", sorted(_ORACLES.items()))
def test_oracle_answer_never_matches_a_distractor(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            assert expected != _num(d), (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )
