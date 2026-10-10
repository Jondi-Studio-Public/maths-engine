"""Independent oracle for `f10-fraction-add-subtract-equivalent`.

Independent oracle for f10-fraction-add-subtract-equivalent.

Parses the RENDERED question text and imports NOTHING from
`mathgen.generators.p10e_fraction_add_subtract_build` -- recomputes the
expression's value with Python's own `fractions.Fraction` arithmetic, reading
the terms and operators straight out of the text a learner is actually
given.

Generates through the skill's own `sources`, never
`make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "f10-fraction-add-subtract-equivalent"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(60)

_EXPR_RE = re.compile(r"^Calculate (.+)\.$")


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_evaluate(text: str) -> Fraction:
    """Recompute the expression's value purely from the rendered text:
    strip "Calculate "/"." and read the leading fraction plus each
    subsequent (operator, fraction) pair left to right -- a route that
    never touches the generator's own LCD-finding/scaling code."""
    match = _EXPR_RE.match(text)
    if match is None:
        raise AssertionError(f"unparseable fraction_add_subtract question: {text!r}")
    tokens = match.group(1).split()
    if len(tokens) % 2 == 0 or len(tokens) < 1:
        raise AssertionError(f"unexpected token count in {text!r}: {tokens!r}")
    total = Fraction(tokens[0])
    i = 1
    while i < len(tokens):
        op, term = tokens[i], tokens[i + 1]
        value = Fraction(term)
        total = total + value if op == "+" else total - value
        i += 2
    return total


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        expected = _oracle_evaluate(q.text)
        actual = Fraction(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_oracle_answer_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(SKILL_ID, difficulty, seed)
        expected = _oracle_evaluate(q.text)
        for d in q.distractors:
            given = Fraction(d)
            assert expected != given, (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_oracle_is_purely_textual():
    assert _oracle_evaluate("Calculate 1/3 + 1/6.") == Fraction(1, 2)
    assert _oracle_evaluate("Calculate 3/4 - 1/6.") == Fraction(7, 12)
    assert _oracle_evaluate("Calculate 1/2 + 1/3 - 1/4.") == Fraction(7, 12)


def test_oracle_fires_on_a_corrupted_answer():
    expected = _oracle_evaluate("Calculate 1/3 + 1/6.")
    corrupted = Fraction(2, 9)  # the denominators-added misconception value
    assert expected != corrupted
