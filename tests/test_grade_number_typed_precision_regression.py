"""Regression sweep for the `grade_number` typed-tolerance fix.

Every *active* `answer: number` skill routes through the same `grading.grade_number`, so this
sweeps all of them.

Three checks:

1. Every active number-kind skill, every source, 40 seeds: the real grader
   accepts the canonical `answer_text`.
2. The 16 affected skills: a value differing from the canonical `answer_text`'s own literal by
   exactly one unit in its *last stated decimal place* is REJECTED, every seed/tier.
3. The 6 clean skills: a non-regression pin -- canonical still accepted, a
   genuinely wrong value still rejected.
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make

SEEDS = range(40)

# A plain decimal literal: optional sign, digits, optional ".digits" -- the
# same shape `question.answer_text` templates actually emit (see
# `mathgen/grading/typed.py`'s `_NUMERIC_LITERAL_RE`), used here only to *construct* a
# deliberately-wrong given text, not to assert anything about the
# implementation directly.
_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _perturb_last_digit(literal: str) -> str:
    """Change the last digit of `literal` (after the decimal point if one is
    present, otherwise the last digit of the integer) by one unit in its own
    place -- the "one-least-significant-digit error" every affected skill must reject."""
    sign = ""
    body = literal
    if body.startswith("-"):
        sign, body = "-", body[1:]
    whole, _, frac = body.partition(".")
    target = list(frac) if frac else list(whole)
    last = int(target[-1])
    target[-1] = "8" if last == 9 else str(last + 1)
    if frac:
        return f"{sign}{whole}.{''.join(target)}"
    return f"{sign}{''.join(target)}"


def _one_lsd_wrong_given(answer_text: str) -> str:
    """Build a typed answer that differs from `answer_text` by exactly one
    unit in its last stated decimal place. Every one of the 16 affected
    skills' `answer_text` carries exactly one numeric literal -- if that assumption
    ever stops holding for a given skill, this raises rather than silently
    constructing a meaningless perturbation."""
    literals = _LITERAL_RE.findall(answer_text)
    assert len(literals) == 1, (
        f"expected exactly one numeric literal in {answer_text!r}, found {literals!r} "
        "-- this skill's answer_text shape has changed; "
        "the perturbation below is no longer meaningful for it and needs re-deriving"
    )
    literal = literals[0]
    perturbed = _perturb_last_digit(literal)
    return answer_text.replace(literal, perturbed, 1)


@functools.cache
def _skills_by_id() -> dict:
    # Loaded once: load_skills() costs ~1 s and this module calls _sources()
    # 22 times at collection.
    _domains, loaded = skills_mod.load_skills()
    return {s.id: s for s in loaded}


def _sources(skill_id: str):
    skill = _skills_by_id()[skill_id]
    assert skill.status == "active", f"{skill_id} is not active"
    assert skill.answer == "number", f"{skill_id} is not answer: number"
    return skill.sources


def _make_params(source) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (source.params or {}).items()}


# -- 1. every active number-kind skill accepts its own canonical answer ----

_domains, _loaded_skills = skills_mod.load_skills()
_ALL_NUMBER_SKILLS = [s for s in _loaded_skills if s.status == "active" and s.answer == "number"]
_ALL_NUMBER_SOURCE_CASES = [(skill.id, source) for skill in _ALL_NUMBER_SKILLS for source in skill.sources]


@pytest.mark.parametrize(
    "skill_id,source",
    _ALL_NUMBER_SOURCE_CASES,
    ids=[f"{skill_id}[{source.topic}/{source.difficulty}]" for skill_id, source in _ALL_NUMBER_SOURCE_CASES],
)
def test_every_active_number_skill_accepts_its_canonical_answer(skill_id, source):
    params = _make_params(source)
    for seed in SEEDS:
        question = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
        result = grading.grade_number(question, question.answer_text)
        assert result.correct, (
            f"{skill_id} (topic={source.topic}, difficulty={source.difficulty}, seed={seed}): "
            f"grade_number rejected its own canonical answer_text {question.answer_text!r} "
            f"(answer={question.answer!r})"
        )


# -- 2. the 16 affected skills reject a one-LSD error -----------------------

AFFECTED_SKILLS = [
    "round-decimals",
    "decimal-arithmetic",
    "decimal-place-value",
    "money-change",
    "compound-interest",
    "flat-loan-repayment",
    "right-triangle-angle",
    "binomial-point-probability",
    "strain-from-extension",
    "hookes-law",
    "moment-of-a-force",
    "bending-moment-point-load",
    "directional-derivative",
    "f10-decimal-notation-tenths-hundredths",
    "f10-division-remainder-context",
    "f10-powers-ten-multiply-divide",
]

_AFFECTED_SOURCE_CASES = [(skill_id, source) for skill_id in AFFECTED_SKILLS for source in _sources(skill_id)]


@pytest.mark.parametrize(
    "skill_id,source",
    _AFFECTED_SOURCE_CASES,
    ids=[f"{skill_id}[{source.topic}/{source.difficulty}]" for skill_id, source in _AFFECTED_SOURCE_CASES],
)
def test_affected_skill_rejects_one_last_significant_digit_error(skill_id, source):
    params = _make_params(source)
    for seed in SEEDS:
        question = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
        if not isinstance(question.answer, (int, float)) or isinstance(question.answer, bool):
            continue  # this source's answer isn't a plain scalar this seed; nothing to perturb
        wrong = _one_lsd_wrong_given(question.answer_text)
        result = grading.grade_number(question, wrong)
        assert not result.correct, (
            f"{skill_id} (topic={source.topic}, difficulty={source.difficulty}, seed={seed}): "
            f"grade_number wrongly accepted a one-LSD-off value {wrong!r} for canonical "
            f"{question.answer_text!r}"
        )


# -- 3. the 6 clean skills stay clean (non-regression pin) ------------------

CLEAN_SKILLS = [
    "unit-pricing",
    "cosine-rule",
    "sine-rule",
    "triangle-area-sine",
    "calculate-overtime",
    "f10-decimal-add-subtract-estimate",
]

_CLEAN_SOURCE_CASES = [(skill_id, source) for skill_id in CLEAN_SKILLS for source in _sources(skill_id)]


@pytest.mark.parametrize(
    "skill_id,source",
    _CLEAN_SOURCE_CASES,
    ids=[f"{skill_id}[{source.topic}/{source.difficulty}]" for skill_id, source in _CLEAN_SOURCE_CASES],
)
def test_clean_skill_still_accepts_canonical_and_rejects_wrong_value(skill_id, source):
    params = _make_params(source)
    for seed in SEEDS:
        question = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
        if not isinstance(question.answer, (int, float)) or isinstance(question.answer, bool):
            continue
        assert grading.grade_number(question, question.answer_text).correct, (
            f"{skill_id} (topic={source.topic}, difficulty={source.difficulty}, seed={seed}): "
            f"canonical answer_text {question.answer_text!r} was rejected"
        )
        wrong = str(float(question.answer) + 1000.0)
        assert not grading.grade_number(question, wrong).correct, (
            f"{skill_id} (topic={source.topic}, difficulty={source.difficulty}, seed={seed}): "
            f"a genuinely wrong value {wrong!r} was wrongly accepted"
        )
