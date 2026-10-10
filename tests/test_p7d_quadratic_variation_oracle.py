"""Independent oracle for `p7d_examine_quadratic_variation`, recomputing each tier's answer."""

import functools

"""Independent oracle for senior-examine-examples-of-quadratically-related-variables.

Parses the stated relation and numbers straight out of ``question.text``,
recomputes the expected answer from the relation ``y = kx^2`` a second way,
and asserts it matches ``question.answer_text``. Reads only ``question.text``/
``answer_text``/``distractors`` -- must not import anything from
``p7d_quadratic_variation_build``.
"""
import math
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
import mathgen

SKILL_ID = "senior-examine-examples-of-quadratically-related-variables"
SEEDS = range(40)

_EASY_RE = re.compile(r"When x = (\d+), y = (\d+)\. If x increases to (\d+), find the new value of y\.")
_MEDIUM_RE = re.compile(r"When x = (\d+), y = (\d+)\. If y increases to (\d+), find the new positive value of x\.")
_HARD_RE = re.compile(
    r"when v = (\d+) m/s, E = (\d+) J\. If the speed is multiplied by (\d+), "
    r"by what factor is the kinetic energy multiplied\?"
)


# `load_skills()` rescans every skill YAML from disk and costs ~560 ms. These
# helpers are called once per SEED, so an uncached lookup spent ~290 s of the
# suite reloading the same library. Cached: `make()` still runs per seed and
# every assertion is unchanged -- only the repeated disk scan is removed.
@functools.lru_cache(maxsize=None)
def _sources_by_difficulty():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {s.difficulty: s for s in skill.sources}


SOURCES = _sources_by_difficulty()


def _questions(difficulty):
    source = SOURCES[difficulty]
    for seed in SEEDS:
        yield mathgen.make(source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))


def _isqrt_fraction(frac: Fraction) -> Fraction:
    """Exact square root of a Fraction known to be a perfect square, without
    calling any helper from the generator under test."""
    num_root = math.isqrt(frac.numerator)
    den_root = math.isqrt(frac.denominator)
    assert num_root * num_root == frac.numerator, f"{frac} numerator is not a perfect square"
    assert den_root * den_root == frac.denominator, f"{frac} denominator is not a perfect square"
    return Fraction(num_root, den_root)


def test_oracle_easy_tier_recomputes_shipped_answer():
    checked = 0
    for q in _questions("easy"):
        match = _EASY_RE.search(q.text)
        assert match is not None, f"could not parse easy question: {q.text!r}"
        x1, y1, x2 = (int(g) for g in match.groups())
        k = Fraction(y1, x1 * x1)
        y2_derived = k * (x2 * x2)
        assert y2_derived.denominator == 1, f"non-integer derived y2 for {q.text!r}"
        assert int(y2_derived) == int(q.answer_text), (
            f"easy: derived {int(y2_derived)} != shipped {q.answer_text} for {q.text!r}"
        )
        checked += 1
    assert checked >= 20


def test_oracle_medium_tier_recomputes_shipped_answer():
    checked = 0
    for q in _questions("medium"):
        match = _MEDIUM_RE.search(q.text)
        assert match is not None, f"could not parse medium question: {q.text!r}"
        x1, y1, y2 = (int(g) for g in match.groups())
        ratio = Fraction(y2, y1)
        scale = _isqrt_fraction(ratio)
        x2_derived = Fraction(x1) * scale
        assert x2_derived.denominator == 1, f"non-integer derived x2 for {q.text!r}"
        assert x2_derived > 0
        assert int(x2_derived) == int(q.answer_text), (
            f"medium: derived {int(x2_derived)} != shipped {q.answer_text} for {q.text!r}"
        )
        checked += 1
    assert checked >= 20


def test_oracle_hard_tier_recomputes_shipped_answer():
    checked = 0
    for q in _questions("hard"):
        match = _HARD_RE.search(q.text)
        assert match is not None, f"could not parse hard question: {q.text!r}"
        v1, e1, t = (int(g) for g in match.groups())
        k = Fraction(e1, v1 * v1)
        v2 = v1 * t
        e2_derived = k * (v2 * v2)
        factor_derived = e2_derived / e1
        assert factor_derived.denominator == 1, f"non-integer derived factor for {q.text!r}"
        assert int(factor_derived) == int(q.answer_text), (
            f"hard: derived {int(factor_derived)} != shipped {q.answer_text} for {q.text!r}"
        )
        checked += 1
    assert checked >= 20


def test_oracle_distractors_do_not_recompute_to_the_shipped_answer():
    """Every distractor, independently recomputed via the wrong-relation
    arithmetic a learner would actually perform, must differ from the
    shipped answer -- confirms the misconceptions are genuine, not just
    that the grader happens to reject the string."""
    for q in _questions("easy"):
        match = _EASY_RE.search(q.text)
        x1, y1, x2 = (int(g) for g in match.groups())
        # "as if direct": y2 = y1 * (x2 / x1)
        as_if_direct = Fraction(y1) * Fraction(x2, x1)
        # "as if inverse": y2 = y1 * (x1 / x2)
        as_if_inverse = Fraction(y1) * Fraction(x1, x2)
        for wrong in (as_if_direct, as_if_inverse):
            if wrong.denominator == 1:
                assert str(int(wrong)) != q.answer_text

    for q in _questions("hard"):
        match = _HARD_RE.search(q.text)
        v1, e1, t = (int(g) for g in match.groups())
        assert str(t) != q.answer_text
        assert str(t**3) != q.answer_text
