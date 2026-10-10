"""Independent oracle for senior-review-rates-and-percentages.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8e_rates_percentages_build` and recomputes the expected
final amount from first principles: a rate x hours subtotal, then a
percentage adjustment applied to that subtotal (increase for easy/medium,
decrease for hard, both hard-coded here from the tier itself, never parsed
from the generator's own text or reusing its private helpers).

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

TOPIC = "rate_percentage_combined"
SKILL_ID = "senior-review-rates-and-percentages"
SEEDS = range(60)

_DIRECTION = {"easy": 1, "medium": 1, "hard": -1}

_PATTERN = re.compile(r"\$([\d.]+)/hour\. A job takes ([\d.]+) hours.*?(\d+)%")


def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_final_amount(text: str, difficulty: str) -> Fraction:
    m = _PATTERN.search(text)
    if not m:
        raise AssertionError(f"unparseable question: {text!r}")
    rate = Fraction(m.group(1))
    hours = Fraction(m.group(2))
    p = int(m.group(3))
    subtotal = rate * hours
    direction = _DIRECTION[difficulty]
    return subtotal * Fraction(100 + direction * p, 100)


def _answer_value(answer_text: str) -> Fraction:
    return Fraction(answer_text.lstrip("$"))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle(difficulty):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = _oracle_final_amount(q.text, difficulty)
        actual = _answer_value(q.answer_text)
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
        expected = _oracle_final_amount(q.text, difficulty)
        for d in q.distractors:
            assert expected != _answer_value(d), (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_oracle_rejects_direction_mismatch():
    """A hard-coded direction sanity check: applying the WRONG tier's
    direction to a hard-tier (discount) question must NOT match the
    generator's own answer, proving the oracle's direction table is doing
    real work rather than always agreeing by coincidence."""
    skills_dict = _skills_dict()
    q = _make_from_sources(skills_dict, "hard", 0)
    correct = _oracle_final_amount(q.text, "hard")
    wrong_direction = _oracle_final_amount(q.text, "easy")  # applies +p instead of -p
    assert correct != wrong_direction
    assert _answer_value(q.answer_text) == correct
