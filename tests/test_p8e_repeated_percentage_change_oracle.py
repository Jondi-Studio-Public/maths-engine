"""Independent oracle for the `p8e_repeated_percentage_change` questions: it re-derives each answer from the
rendered question text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8e_repeated_percentage_change_build` and recomputes the
expected final quantity from first principles (compounding each stated
percentage change onto the previous running value, as an exact Fraction
product), then compares it against `answer_text`'s own displayed value.

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

TOPIC = "repeated_percentage_change"
SKILL_ID = "senior-determine-the-overall-change-in-a-quantity-following"
SEEDS = range(60)

_START_RE = re.compile(r"starts at (\d+)\.")
_CHANGE_RE = re.compile(r"(?:First|Then) it (increases|decreases) by (\d+)%\.")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle(text: str) -> int:
    start_match = _START_RE.search(text)
    assert start_match, f"unparseable starting quantity: {text!r}"
    value = Fraction(int(start_match.group(1)))
    changes = _CHANGE_RE.findall(text)
    assert changes, f"unparseable percentage changes: {text!r}"
    for direction, pct in changes:
        signed_pct = int(pct) if direction == "increases" else -int(pct)
        value *= Fraction(100 + signed_pct, 100)
    assert value.denominator == 1, f"oracle expected an exact integer, got {value}: {text!r}"
    return int(value)


@pytest.mark.parametrize("difficulty,expected_changes", [("easy", 2), ("medium", 2), ("hard", 3)])
def test_oracle_matches_answer(difficulty, expected_changes):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        changes = _CHANGE_RE.findall(q.text)
        assert len(changes) == expected_changes, (seed, difficulty, q.text)
        expected = _oracle(q.text)
        actual = int(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_answer_never_matches_a_distractor(difficulty):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = _oracle(q.text)
        for d in q.distractors:
            assert expected != int(d), (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


# --- The naive "adds the percentages" answer is never the true answer ------
# (this is the property the skill exists to test; swept over a large sample
# rather than sampled: "adds-the-percentages never
# equals the true answer over >= 500 seeds per tier").


def _naive_additive(text: str) -> Fraction:
    start_match = _START_RE.search(text)
    q = Fraction(int(start_match.group(1)))
    changes = _CHANGE_RE.findall(text)
    total_pct = sum(int(pct) if direction == "increases" else -int(pct) for direction, pct in changes)
    return q * Fraction(100 + total_pct, 100)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_naive_additive_answer_never_equals_true_answer(difficulty):
    skills_dict = _skills_dict()
    n = 0
    for seed in range(500):
        q = _make_from_sources(skills_dict, difficulty, seed)
        naive = _naive_additive(q.text)
        true = int(q.answer_text)
        assert naive != true, (seed, difficulty, q.text, q.answer_text)
        n += 1
    assert n == 500
