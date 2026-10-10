"""Independent oracle for the `p11e_bayes` questions: it re-derives each answer from the rendered question text.

Parses `question.text` alone -- never imports anything from
`mathgen.generators.p11e_bayes_build` -- and recomputes the posterior from
first principles (population x rate -> joint counts -> ratio, or for hard,
ratio+total -> populations first). Confirms `answer_text`'s displayed value
against that independent recomputation, and that the oracle's own answer
never coincides with a shipped distractor.

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone with an assumed default.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

TOPIC = "bayes_finite_counts"
SKILL_ID = "ucv1-bayes-finite-counts"
SEEDS = range(60)

_DIRECT_RE = re.compile(
    r"Machine 1 produces (\d+) items and Machine 2 produces (\d+) items\. "
    r"(\d+) out of every (\d+) items from Machine 1 are faulty, and (\d+) out of every (\d+) "
    r"items from Machine 2 are faulty\. An item is picked at random from the combined output "
    r"and found to (not )?be faulty\."
)

_HARD_RE = re.compile(
    r"Machine 1 and Machine 2 produce items in the ratio (\d+) : (\d+), "
    r"with (\d+) items produced in total\. "
    r"(\d+) out of every (\d+) items from Machine 1 are faulty, and (\d+) out of every (\d+) "
    r"items from Machine 2 are faulty\. An item is picked at random from the combined output "
    r"and found to (not )?be faulty\."
)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_direct(text: str) -> Fraction:
    m = _DIRECT_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable direct-tier question: {text!r}")
    n1, n2, a1, b1, a2, b2 = (int(g) for g in m.groups()[:6])
    complement = m.group(7) is not None
    if (n1 * a1) % b1 or (n2 * a2) % b2:
        raise AssertionError(f"non-exact joint count from question alone: {text!r}")
    f1, f2 = n1 * a1 // b1, n2 * a2 // b2
    counts = (n1 - f1, n2 - f2) if complement else (f1, f2)
    return Fraction(counts[0], counts[0] + counts[1])


def _oracle_hard(text: str) -> Fraction:
    m = _HARD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier question: {text!r}")
    k1, k2, total, a1, b1, a2, b2 = (int(g) for g in m.groups()[:7])
    complement = m.group(8) is not None
    derived_n1 = Fraction(total * k1, k1 + k2)
    derived_n2 = Fraction(total * k2, k1 + k2)
    if derived_n1.denominator != 1 or derived_n2.denominator != 1:
        raise AssertionError(f"ratio-derived population not an exact integer: {text!r}")
    n1, n2 = int(derived_n1), int(derived_n2)
    if (n1 * a1) % b1 or (n2 * a2) % b2:
        raise AssertionError(f"non-exact joint count from question alone: {text!r}")
    f1, f2 = n1 * a1 // b1, n2 * a2 // b2
    counts = (n1 - f1, n2 - f2) if complement else (f1, f2)
    return Fraction(counts[0], counts[0] + counts[1])


def _answer_value(answer_text: str) -> Fraction:
    parts = answer_text.split("/")
    return Fraction(int(parts[0]), int(parts[1])) if len(parts) == 2 else Fraction(int(parts[0]))


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_direct),
        ("medium", _oracle_direct),
        ("hard", _oracle_hard),
    ],
)
def test_bayes_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, SKILL_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = _answer_value(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_direct),
        ("medium", _oracle_direct),
        ("hard", _oracle_hard),
    ],
)
def test_oracle_answer_never_matches_a_distractor(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, SKILL_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        for d in q.distractors:
            assert expected != _answer_value(d), (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_hard_tier_ratio_derived_populations_are_always_exact_integers():
    """Confirms the hard tier's own claim (never a rounded population) purely
    from the rendered question text, independent of the generator."""
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, SKILL_ID, "hard", seed)
        m = _HARD_RE.search(q.text)
        assert m, q.text
        k1, k2, total = (int(g) for g in m.groups()[:3])
        derived = Fraction(total * k1, k1 + k2)
        assert derived.denominator == 1, f"seed {seed}: {q.text!r} implies a fractional population"
