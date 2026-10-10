"""Independent oracle for the `p7e_exponent_laws` questions: it re-derives each answer from the rendered question
text.

Parses each rendered `question.text` with a regex and recomputes the final
exponent from scratch with plain integer add/subtract/multiply -- never
imports anything from `mathgen.generators.p7e_exponent_laws_build`. Sweeps
>=40 seeds x 3 tiers (120 questions total).
"""

import re

import pytest

import mathgen

TOPIC = "exponent_laws_integer_variable"
SEEDS = range(40)

_EASY_RE = re.compile(
    r"^Simplify x\^(\(-?\d+\)|-?\d+) . x\^(\(-?\d+\)|-?\d+), "
    r"giving your answer as a single power of x\.$"
)
_MEDIUM_RE = _EASY_RE  # same shape, product symbol swapped for division at runtime
_HARD_RE = re.compile(
    r"^Simplify \(x\^(-?\d+)\)\^(\d+) . x\^(\(-?\d+\)|-?\d+) . x\^(\(-?\d+\)|-?\d+), "
    r"giving your answer as a single power of x\.$"
)


def _int(token: str) -> int:
    return int(token.strip("()"))


def _oracle_easy(text: str) -> int:
    match = _EASY_RE.match(text)
    assert match, f"could not parse easy text: {text!r}"
    a, b = _int(match.group(1)), _int(match.group(2))
    return a + b


def _oracle_medium(text: str) -> int:
    match = _MEDIUM_RE.match(text)
    assert match, f"could not parse medium text: {text!r}"
    a, b = _int(match.group(1)), _int(match.group(2))
    return a - b


def _oracle_hard(text: str) -> int:
    match = _HARD_RE.match(text)
    assert match, f"could not parse hard text: {text!r}"
    a, b, c, d = (
        _int(match.group(1)),
        _int(match.group(2)),
        _int(match.group(3)),
        _int(match.group(4)),
    )
    return a * b + c - d


def _answer_exponent(question) -> int:
    assert question.answer_text.startswith("x^"), question.answer_text
    return int(question.answer_text[2:])


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_easy(seed):
    question = mathgen.make(TOPIC, seed=seed, difficulty="easy")
    expected = _oracle_easy(question.text)
    assert _answer_exponent(question) == expected, question.text


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_medium(seed):
    question = mathgen.make(TOPIC, seed=seed, difficulty="medium")
    expected = _oracle_medium(question.text)
    assert _answer_exponent(question) == expected, question.text


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_hard(seed):
    question = mathgen.make(TOPIC, seed=seed, difficulty="hard")
    expected = _oracle_hard(question.text)
    assert _answer_exponent(question) == expected, question.text


@pytest.mark.parametrize(
    "difficulty,oracle",
    [
        ("easy", _oracle_easy),
        ("medium", _oracle_medium),
        ("hard", _oracle_hard),
    ],
)
def test_oracle_distractors_never_equal_the_recomputed_answer(difficulty, oracle):
    for seed in SEEDS:
        question = mathgen.make(TOPIC, seed=seed, difficulty=difficulty)
        expected = oracle(question.text)
        for distractor in question.distractors:
            assert distractor.startswith("x^"), distractor
            distractor_exponent = int(distractor[2:])
            assert distractor_exponent != expected, question.text
