"""Independent oracle for the decimal rounding and ordering skills of p9e.

Independent oracle for two skills.

Parses the RENDERED question text and imports NOTHING from
`mathgen.generators.p9e_decimal_accuracy_order_build` -- every recompute here
takes its own route (a fresh half-up-on-`Fraction` rounding for the accuracy
skill, a fresh `sorted()`-free ascending check for the ordering skill),
re-deriving the answer from what a learner is actually given.

Generates through each skill's own `sources`, never
`make(topic, seed, difficulty=tier)` alone.
"""

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

ROUND_SKILL_ID = "f10-round-decimals-accuracy"
ORDER_SKILL_ID = "f10-decimal-order-numberline"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(60)

_PLACE_DP = {"hundredth": 2, "tenth": 1, "whole number": 0}
_RAW_RE = re.compile(r"[$]?(\d+\.\d+)")
_PLACE_RE = re.compile(r"nearest (hundredth|tenth|whole number)")
_DECIMAL_RE = re.compile(r"\d+\.\d+")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ---------------------------------------------------------------------------
# f10-round-decimals-accuracy
# ---------------------------------------------------------------------------


def _oracle_round(text: str) -> Fraction:
    """Recompute the expected rounded value from the rendered text alone:
    extract the one unrounded numeral and the recognised rounding phrase,
    then round half-up via exact `Fraction` arithmetic -- a route that never
    touches the generator's own integer-digit construction."""
    raw_match = _RAW_RE.search(text)
    place_match = _PLACE_RE.search(text)
    if raw_match is None or place_match is None:
        raise AssertionError(f"unparseable round-decimal-accuracy question: {text!r}")
    dp = _PLACE_DP[place_match.group(1)]
    exact = Fraction(raw_match.group(1))
    scale = 10**dp
    units = math.floor(exact * scale + Fraction(1, 2))
    return Fraction(units, scale)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        expected = _oracle_round(q.text)
        actual = Fraction(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_oracle_answer_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        expected = _oracle_round(q.text)
        for d in q.distractors:
            given = Fraction(d)
            assert expected != given, (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_round_oracle_is_purely_textual():
    """A hand-built question, fed straight to the oracle with no generator
    involvement, confirming the recompute comes from the text alone."""
    text = "A digital calliper reads the length of a beetle as 3.95 cm. Round this length to the nearest tenth of a centimetre."
    assert _oracle_round(text) == Fraction("4.0")
    text2 = "Splitting the cost of a shared takeaway order evenly left each of several friends owing $2.995. Round this amount to the nearest hundredth of a dollar."
    assert _oracle_round(text2) == Fraction("3.00")


# ---------------------------------------------------------------------------
# f10-decimal-order-numberline
# ---------------------------------------------------------------------------


def _oracle_sort(text: str) -> list[Fraction]:
    """Recompute the ascending order from the rendered text alone: parse
    every decimal string from the question with `Fraction` (never reusing
    the construction's own `Fraction`/list objects), and sort. Deliberately
    does NOT reuse `sorted()` for the verification step below -- the ORDER
    computed here uses `sorted()` once as a reference, and the pairwise walk
    is what actually proves ascending order, independent of how this list
    was produced."""
    numbers = _DECIMAL_RE.findall(text)
    if len(numbers) != 4:
        raise AssertionError(f"unparseable decimal-order question: {text!r}")
    return sorted(Fraction(n) for n in numbers)


def _is_strictly_ascending(values: list[Fraction]) -> bool:
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(ORDER_SKILL_ID, difficulty, seed)
        expected = _oracle_sort(q.text)
        actual = [Fraction(t) for t in q.answer_text.split(", ")]
        assert _is_strictly_ascending(actual), (seed, difficulty, q.text, q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        # And the answer is a genuine permutation of the values shown in
        # the question (same multiset, parsed independently).
        shown = sorted(Fraction(n) for n in _DECIMAL_RE.findall(q.text))
        assert shown == sorted(actual), (seed, difficulty, q.text, q.answer_text)
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_oracle_answer_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ORDER_SKILL_ID, difficulty, seed)
        expected = _oracle_sort(q.text)
        for d in q.distractors:
            given = [Fraction(t) for t in d.split(", ")]
            assert expected != given, (
                f"seed {seed} {difficulty}: oracle order {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_order_oracle_is_purely_textual():
    text = "Arrange these decimals from smallest to largest: 2.451, 0.4499, 2.4602, 0.451."
    assert _oracle_sort(text) == [Fraction("0.4499"), Fraction("0.451"), Fraction("2.451"), Fraction("2.4602")]


def test_order_oracle_fires_on_a_corrupted_answer():
    text = "Arrange these decimals from smallest to largest: 0.3, 0.1, 0.4, 0.2."
    expected = _oracle_sort(text)
    corrupted = list(reversed(expected))
    assert expected != corrupted
    assert not _is_strictly_ascending(corrupted)
