"""Independent oracle for T3 applied-pythagoras-problems.

Re-derives every answer by parsing ``question.text`` alone, with
``math.sqrt`` -- never importing anything from
``mathgen.generators.t3_applied_pythagoras_problems_build``. Rounds the
same way (``round(v, 1)``) and compares against
``float(question.answer_text)`` within a tight tolerance (1e-9): both sides
round the same way, so this is oracle-vs-generator agreement on an
identical rounding rule, not the grader's 0.051 typed-answer tolerance.

Covers all four variants, 60 seeds each (240 questions minimum), plus a
fifth sweep that dispatches by parsing the real `hard`-tier text (the
artefact a learner actually receives, coin flip included) rather than only
the direct `variant=` override (the artefact the learner receives, not the one the code holds).
"""

import math
import re

import pytest

import mathgen

TOPIC = "applied_pythagoras"

_LADDER_LEN_RE = re.compile(r"ladder of length (\d+\.\d) m")
_LADDER_HEIGHT_GIVEN_RE = re.compile(r"reaching (\d+\.\d) m up the wall")
_LADDER_DISTANCE_GIVEN_RE = re.compile(r"is (\d+\.\d) m from the base of the wall")
_DISPLACEMENT_RE = re.compile(r"walks (\d+\.\d) km (?:north|south), then (\d+\.\d) km (?:east|west)")
_EQUILATERAL_SIDE_RE = re.compile(r"side length (\d+\.\d) units")
_RECT_DIMS_RE = re.compile(r"length (\d+\.\d) m, width (\d+\.\d) m and height (\d+\.\d) m")

_TOL = 1e-9


def oracle_ladder(text: str) -> float:
    ladder = float(_LADDER_LEN_RE.search(text).group(1))
    m = _LADDER_HEIGHT_GIVEN_RE.search(text) or _LADDER_DISTANCE_GIVEN_RE.search(text)
    given = float(m.group(1))
    return round(math.sqrt(ladder**2 - given**2), 1)


def oracle_displacement(text: str) -> float:
    m = _DISPLACEMENT_RE.search(text)
    p, q = float(m.group(1)), float(m.group(2))
    return round(math.sqrt(p**2 + q**2), 1)


def oracle_equilateral_altitude(text: str) -> float:
    side = float(_EQUILATERAL_SIDE_RE.search(text).group(1))
    half_base = side / 2
    return round(math.sqrt(side**2 - half_base**2), 1)


def oracle_rect_solid_diagonal(text: str) -> float:
    m = _RECT_DIMS_RE.search(text)
    length, width, height = (float(g) for g in m.groups())
    return round(math.sqrt(length**2 + width**2 + height**2), 1)


def _variant_of(text: str) -> str:
    if "ladder" in text:
        return "ladder"
    if "hiker" in text:
        return "displacement"
    if "equilateral" in text:
        return "equilateral_altitude"
    if "rectangular box" in text:
        return "rect_solid_diagonal"
    raise AssertionError(f"could not classify question text: {text!r}")


_ORACLES = {
    "ladder": oracle_ladder,
    "displacement": oracle_displacement,
    "equilateral_altitude": oracle_equilateral_altitude,
    "rect_solid_diagonal": oracle_rect_solid_diagonal,
}


# ============================================================================
# Direct-variant sweeps: 60 seeds each, via the explicit `variant=` override
# (bypasses the hard tier's coin flip -- an independent rng stream per
# `mathgen.core.make`'s seed-string construction).
# ============================================================================


@pytest.mark.parametrize("seed", range(60))
def test_ladder_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, variant="ladder")
    expected = oracle_ladder(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL, (
        f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r}"
    )


@pytest.mark.parametrize("seed", range(60))
def test_displacement_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, variant="displacement")
    expected = oracle_displacement(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL, (
        f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r}"
    )


@pytest.mark.parametrize("seed", range(60))
def test_equilateral_altitude_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, variant="equilateral_altitude")
    expected = oracle_equilateral_altitude(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL, (
        f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r}"
    )


@pytest.mark.parametrize("seed", range(60))
def test_rect_solid_diagonal_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, variant="rect_solid_diagonal")
    expected = oracle_rect_solid_diagonal(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL, (
        f"seed {seed}: oracle {expected} != answer_text {q.answer_text!r}"
    )


# ============================================================================
# Real hard-tier sweep: the artefact a learner actually receives, coin flip
# included -- dispatches by parsing the rendered text, not by trusting
# which sub-variant "should" have been drawn.
# ============================================================================


@pytest.mark.parametrize("seed", range(60))
def test_real_hard_tier_oracle_dispatches_by_text(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="hard")
    variant = _variant_of(q.text)
    expected = _ORACLES[variant](q.text)
    assert abs(expected - float(q.answer_text)) < _TOL, (
        f"seed {seed} ({variant}): oracle {expected} != answer_text {q.answer_text!r}"
    )


@pytest.mark.parametrize("seed", range(40))
def test_real_easy_tier_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="easy")
    assert _variant_of(q.text) == "ladder"
    expected = oracle_ladder(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL


@pytest.mark.parametrize("seed", range(40))
def test_real_medium_tier_oracle(seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty="medium")
    assert _variant_of(q.text) == "displacement"
    expected = oracle_displacement(q.text)
    assert abs(expected - float(q.answer_text)) < _TOL
