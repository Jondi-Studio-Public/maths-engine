"""Independent oracle for `f10-fraction-equivalence-numberline`.

Independent oracle for f10-fraction-equivalence-numberline.

Parses the RENDERED question text and imports NOTHING from
`mathgen.generators.p10e_fraction_equivalence_numberline_build` -- sorts the
fractions found in the text with Python's own `fractions.Fraction`
(auto-reducing on parse, so an unreduced hard-tier display item is compared
by value), and separately confirms strict ascending order with a
`sorted()`-free pairwise walk.

Generates through the skill's own `sources`, never
`make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "f10-fraction-equivalence-numberline"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(60)

_QUESTION_RE = re.compile(r"^Arrange these fractions from smallest to largest: (.+)\.$")
_FRACTION_TOKEN_RE = re.compile(r"\d+/\d+")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_sort(text: str) -> list[Fraction]:
    """Recompute the ascending order from the rendered text alone: parse
    every fraction token from the question with `Fraction` (auto-reducing),
    and sort. `sorted()` is used once here as a reference; the pairwise walk
    below is what actually proves ascending order."""
    match = _QUESTION_RE.match(text)
    if match is None:
        raise AssertionError(f"unparseable fraction_equivalence_numberline question: {text!r}")
    tokens = _FRACTION_TOKEN_RE.findall(match.group(1))
    if not tokens:
        raise AssertionError(f"no fraction tokens found in {text!r}")
    return sorted(Fraction(t) for t in tokens)


def _is_strictly_ascending(values: list[Fraction]) -> bool:
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        expected = _oracle_sort(q.text)
        actual = [Fraction(t) for t in q.answer_text.split(", ")]
        assert _is_strictly_ascending(actual), (seed, difficulty, q.text, q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        # the answer is a genuine permutation of the same multiset of VALUES
        # shown in the question, parsed independently.
        shown = sorted(Fraction(t) for t in _FRACTION_TOKEN_RE.findall(q.text))
        assert shown == sorted(actual), (seed, difficulty, q.text, q.answer_text)
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_answer_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        expected = _oracle_sort(q.text)
        for d in q.distractors:
            given = sorted(Fraction(t) for t in d.split(", "))
            given_ordered = [Fraction(t) for t in d.split(", ")]
            assert expected != given_ordered, (
                f"seed {seed} {difficulty}: oracle order {expected} matches distractor {d!r} -- {q.text!r}"
            )
            assert sorted(expected) == given  # same multiset of values regardless


def test_oracle_is_purely_textual():
    text = "Arrange these fractions from smallest to largest: 3/4, 1/4, 2/6."
    assert _oracle_sort(text) == [Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)]


def test_oracle_fires_on_a_corrupted_answer():
    text = "Arrange these fractions from smallest to largest: 1/4, 1/2, 1/3."
    expected = _oracle_sort(text)
    corrupted = list(reversed(expected))
    assert expected != corrupted
    assert not _is_strictly_ascending(corrupted)
