"""Independent oracle for `f10-compound-outcomes-replacement`.

Independent oracle for f10-compound-outcomes-replacement.

Parses the RENDERED question text and imports NOTHING from
`mathgen.generators.p11d_probability_build`. Recovers R, B, T and the two
colour names from the question header, the stated outcome order from the
question's own tail clause, recomputes the with-replacement probability list
from scratch, and confirms it matches `question.answer_text` positionally.
Also independently recomputes the mandatory `no-replacement-values`
distractor and confirms it is offered.

Generates through the skill's own `sources`, never
`make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "f10-compound-outcomes-replacement"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(60)

_HEADER_RE = re.compile(
    r"^A bag contains (\d+) (\w+) balls and (\d+) (\w+) balls, (\d+) in total\. "
    r"A ball is drawn at random, its colour is noted, then it is returned to the bag "
    r"before a second ball is drawn\. List the probability of each of these four "
    r"outcomes, in this order: (.+)\.$"
)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_parse(text: str):
    """Recover (R, B, T, colour_a, colour_b, stated_labels) from the rendered
    question text alone."""
    m = _HEADER_RE.match(text)
    if m is None:
        raise AssertionError(f"unparseable compound_outcomes_replacement question: {text!r}")
    R, color_a, B, color_b, T, order_text = m.groups()
    R, B, T = int(R), int(B), int(T)
    if R + B != T:
        raise AssertionError(f"stated counts do not sum to the stated total: {text!r}")
    stated_labels = [x.strip() for x in order_text.split(", ")]
    return R, B, T, color_a, color_b, stated_labels


def _oracle_with_replacement(
    R: int, B: int, T: int, color_a: str, color_b: str, stated_labels: list[str]
) -> list[Fraction]:
    """Recompute the with-replacement probability list, in the stated order,
    from the raw counts alone -- via `Fraction(count_of_first, T) *
    Fraction(count_of_second, T)` from a colour-name-keyed count dict, never
    reusing the generator's own p/q variables or code structure."""
    count_of = {color_a: R, color_b: B}
    values = []
    for label in stated_labels:
        first, second = label.split("-then-")
        values.append(Fraction(count_of[first], T) * Fraction(count_of[second], T))
    return values


def _oracle_no_replacement(
    R: int, B: int, T: int, color_a: str, color_b: str, stated_labels: list[str]
) -> list[Fraction]:
    """Recompute the mandatory `no-replacement-values` distractor list, in
    the stated order, from the raw counts alone."""
    count_of = {color_a: R, color_b: B}
    values = []
    for label in stated_labels:
        first, second = label.split("-then-")
        first_count, second_count = count_of[first], count_of[second]
        second_count_after_first_draw = second_count - 1 if second == first else second_count
        values.append(Fraction(first_count, T) * Fraction(second_count_after_first_draw, T - 1))
    return values


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        R, B, T, color_a, color_b, stated_labels = _oracle_parse(q.text)
        expected = _oracle_with_replacement(R, B, T, color_a, color_b, stated_labels)
        actual = [Fraction(t) for t in q.answer_text.split(", ")]
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        assert all(0 < v < 1 for v in actual), (seed, difficulty, actual)
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_no_replacement_distractor_is_offered(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        R, B, T, color_a, color_b, stated_labels = _oracle_parse(q.text)
        expected_no_replace = _oracle_no_replacement(R, B, T, color_a, color_b, stated_labels)
        distractor_lists = [[Fraction(t) for t in d.split(", ")] for d in q.distractors]
        assert expected_no_replace in distractor_lists, (
            f"seed {seed} {difficulty}: {expected_no_replace} not among {distractor_lists} -- {q.text!r}"
        )


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_answer_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        R, B, T, color_a, color_b, stated_labels = _oracle_parse(q.text)
        expected = _oracle_with_replacement(R, B, T, color_a, color_b, stated_labels)
        for d in q.distractors:
            given = [Fraction(t) for t in d.split(", ")]
            assert expected != given, (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_oracle_hand_worked_example():
    text = (
        "A bag contains 3 red balls and 5 blue balls, 8 in total. "
        "A ball is drawn at random, its colour is noted, then it is returned to the bag "
        "before a second ball is drawn. List the probability of each of these four "
        "outcomes, in this order: red-then-red, red-then-blue, blue-then-red, blue-then-blue."
    )
    R, B, T, color_a, color_b, labels = _oracle_parse(text)
    assert (R, B, T, color_a, color_b) == (3, 5, 8, "red", "blue")
    expected = _oracle_with_replacement(R, B, T, color_a, color_b, labels)
    assert expected == [Fraction(9, 64), Fraction(15, 64), Fraction(15, 64), Fraction(25, 64)]
    no_replace = _oracle_no_replacement(R, B, T, color_a, color_b, labels)
    assert no_replace == [
        Fraction(3, 8) * Fraction(2, 7),
        Fraction(3, 8) * Fraction(5, 7),
        Fraction(5, 8) * Fraction(3, 7),
        Fraction(5, 8) * Fraction(4, 7),
    ]


def test_oracle_fires_on_a_corrupted_answer():
    text = (
        "A bag contains 3 red balls and 5 blue balls, 8 in total. "
        "A ball is drawn at random, its colour is noted, then it is returned to the bag "
        "before a second ball is drawn. List the probability of each of these four "
        "outcomes, in this order: red-then-red, red-then-blue, blue-then-red, blue-then-blue."
    )
    R, B, T, color_a, color_b, labels = _oracle_parse(text)
    expected = _oracle_with_replacement(R, B, T, color_a, color_b, labels)
    corrupted = list(reversed(expected))
    assert expected != corrupted
