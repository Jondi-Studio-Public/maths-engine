"""Independent oracle for the `p8d_circle_theorems` questions: it re-derives each answer from the rendered
question text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8d_circle_theorems_build` and recomputes the expected
answer from first principles, then compares it against `answer_text`.

Generates through each skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

ANGLE_ID = "senior-the-angle-at-the-centre-subtended-by-an"
CHORD_ID = "senior-when-two-chords-of-a-circle-intersect-the"

SEEDS = range(60)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, skill_id, difficulty, seed):
    skill = skills_dict[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# =============================================================================
# Skill 1: the_angle_at_the_centre_subtended_by_an_arc (ACMSM030)
# =============================================================================

_EASY_ANGLE_RE = re.compile(
    r"point P lies on the major arc AB\. Angle AOB, the non-reflex angle at the "
    r"centre, is (\d+) degrees\."
)
_MEDIUM_ANGLE_RE = re.compile(r"angle AOC is (\d+) degrees and angle COB is (\d+) degrees")
_HARD_ANGLE_RE = re.compile(r"reflex angle AOB, measured the long way round from OA to OB, is (\d+) degrees")


def _oracle_angle_easy(text: str) -> int:
    m = _EASY_ANGLE_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier angle question: {text!r}")
    x = int(m.group(1))
    assert 20 < x < 160 and x % 2 == 0
    return x // 2


def _oracle_angle_medium(text: str) -> int:
    m = _MEDIUM_ANGLE_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier angle question: {text!r}")
    p, q = int(m.group(1)), int(m.group(2))
    total = p + q
    assert 20 < total < 160 and total % 2 == 0
    return total // 2


def _oracle_angle_hard(text: str) -> int:
    m = _HARD_ANGLE_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier angle question: {text!r}")
    r = int(m.group(1))
    assert 200 < r < 340 and r % 2 == 0
    return r // 2


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_angle_easy),
        ("medium", _oracle_angle_medium),
        ("hard", _oracle_angle_hard),
    ],
)
def test_angle_at_centre_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, ANGLE_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = int(q.answer_text)
        assert expected == actual, f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r}"
        for d in q.distractors:
            assert int(d) != expected, f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r}"
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# Skill 2: when_two_chords_of_a_circle_intersect_the_product (ACMSM035)
# =============================================================================

_EASY_CHORD_RE = re.compile(r"AP is (\d+) cm, PB is (\d+) cm and CP is (\d+) cm\. Find the length of PD, in cm\.")
_MEDIUM_CHORD_RE = re.compile(r"AP is (\d+) mm, PB is (\d+) cm and CP is (\d+) cm\. Find the length of PD, in cm\.")
_HARD_CHORD_RE = re.compile(
    r"AP is x cm and PB is \(x \+ (\d+)\) cm, where x is a positive integer\. "
    r"CP is (\d+) cm and PD is (\d+) cm\. Find the length of AP, in cm\."
)


def _oracle_chord_easy(text: str) -> Fraction:
    m = _EASY_CHORD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable easy-tier chord question: {text!r}")
    ap, pb, cp = (Fraction(m.group(i)) for i in (1, 2, 3))
    pd = ap * pb / cp
    assert pd.denominator == 1, f"non-integer PD: {pd}"
    return pd


def _oracle_chord_medium(text: str) -> Fraction:
    m = _MEDIUM_CHORD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable medium-tier chord question: {text!r}")
    ap_mm, pb, cp = (Fraction(m.group(i)) for i in (1, 2, 3))
    ap_cm = ap_mm / 10  # the oracle's own unit conversion, independent of the generator
    pd = ap_cm * pb / cp
    assert pd.denominator == 1, f"non-integer PD: {pd}"
    return pd


def _oracle_chord_hard(text: str) -> Fraction:
    m = _HARD_CHORD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier chord question: {text!r}")
    k, c, d = (int(m.group(i)) for i in (1, 2, 3))
    cd = c * d
    # Solve x(x + k) = cd via the quadratic formula, using exact integer
    # arithmetic -- a route independent of however the generator built x.
    disc = k * k + 4 * cd
    root = math.isqrt(disc)
    assert root * root == disc, f"discriminant {disc} is not a perfect square"
    assert (root - k) % 2 == 0
    x = (root - k) // 2
    assert x > 0
    return Fraction(x)


@pytest.mark.parametrize(
    "difficulty,oracle_fn",
    [
        ("easy", _oracle_chord_easy),
        ("medium", _oracle_chord_medium),
        ("hard", _oracle_chord_hard),
    ],
)
def test_when_two_chords_oracle(difficulty, oracle_fn):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, CHORD_ID, difficulty, seed)
        expected = oracle_fn(q.text)
        actual = Fraction(q.answer_text)
        assert expected == actual, f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r}"
        for dist in q.distractors:
            assert expected != Fraction(dist), (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {dist!r}"
            )
        checked += 1
    assert checked == len(SEEDS)
