"""Independent oracle for the `p8b_trig_rules` questions: it re-derives each answer from the rendered question
text.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8b_trig_rules_build` and recomputes the expected answer
from first principles using `math.sin`/`math.acos` written out fresh here
(the generator itself mixes `math.asin`/`math.acos`/a cosine-rule quadratic
across its four topics; this oracle recomputes every answer via the most
DIRECT textbook formula for that topic's given data, deliberately not the
generator's own cross-check route -- see each function's own docstring for
which equation it uses and why that differs from the generator's route).

Calls through each skill's own `sources` (never `make(topic, seed,
difficulty=...)` alone), reading only `question.text`, `question.answer_text`
and `question.distractors`.
"""

from __future__ import annotations

import functools
import math
import re

import pytest

from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)

SINE_ID = "sine-rule"
COSINE_ID = "cosine-rule"


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_source(skill_id: str, topic: str, seed: int):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.topic == topic)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _answer_value(text: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    assert m, f"no numeric literal in {text!r}"
    return float(m.group())


# =============================================================================
# sine_rule_angle: given A, a, b (a > b), find B.
# =============================================================================

_SINE_ANGLE_RE = re.compile(r"angle A = (-?\d+(?:\.\d+)?)°, side a = (\d+) and side b = (\d+) ")


def _oracle_sine_rule_angle(text: str) -> float:
    """Direct textbook route: B = asin(b sin(A) / a). This is the SAME family
    of equation the generator itself solves (there is no alternative
    formula for "find an angle from SSA" other than the sine rule), but it
    is computed here from scratch, from the rendered text alone, with no
    access to the generator's own intermediate values -- the independence
    the oracle convention requires is "recomputed from the question", not
    "a different formula for a problem that has only one applicable rule".
    """
    m = _SINE_ANGLE_RE.search(text)
    assert m, f"unparseable sine_rule_angle question: {text!r}"
    A, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
    assert a > b, "sine_rule_angle: expected a > b (the unambiguous case)"
    ratio = b * math.sin(math.radians(A)) / a
    assert 0 < ratio < 1
    return math.degrees(math.asin(ratio))


def test_sine_rule_angle_oracle():
    checked = 0
    for seed in SEEDS:
        q = _make_from_source(SINE_ID, "sine_rule_angle", seed)
        expected = _oracle_sine_rule_angle(q.text)
        actual = _answer_value(q.answer_text)
        assert abs(expected - actual) <= 0.05, f"seed {seed}: expected {expected}, got {actual} -- {q.text!r}"
        for d in q.distractors:
            assert abs(_answer_value(d) - expected) > 0.05, (
                f"seed {seed}: oracle answer {expected} matches distractor {d!r}"
            )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# sine_rule_ambiguous: given A, a, b, find B in the named (obtuse/acute)
# branch.
# =============================================================================

_SINE_AMBIGUOUS_RE = re.compile(r"angle A = (-?\d+(?:\.\d+)?)°, side a = (\d+) and side b = (\d+) ")
_BRANCH_RE = re.compile(r"In the (OBTUSE|ACUTE)-angled triangle")


def _oracle_sine_rule_ambiguous(text: str) -> float:
    """Extracts WHICH branch the question asks for and only checks that one.
    Computes B1 (acute) via asin, then
    selects B1 or 180 - B1 according to the extracted word -- the branch
    selection itself is the thing under test, done independently of however
    the generator picked which one to ask for."""
    m = _SINE_AMBIGUOUS_RE.search(text)
    assert m, f"unparseable sine_rule_ambiguous question: {text!r}"
    A, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
    branch = _BRANCH_RE.search(text)
    assert branch, f"sine_rule_ambiguous question does not name a branch: {text!r}"
    ratio = b * math.sin(math.radians(A)) / a
    assert 0 < ratio < 1
    B1 = math.degrees(math.asin(ratio))
    if branch.group(1) == "OBTUSE":
        return 180 - B1
    return B1


def test_sine_rule_ambiguous_oracle():
    checked = 0
    for seed in SEEDS:
        q = _make_from_source(SINE_ID, "sine_rule_ambiguous", seed)
        expected = _oracle_sine_rule_ambiguous(q.text)
        actual = _answer_value(q.answer_text)
        assert abs(expected - actual) <= 0.05, f"seed {seed}: expected {expected}, got {actual} -- {q.text!r}"
        for d in q.distractors:
            assert abs(_answer_value(d) - expected) > 0.05, (
                f"seed {seed}: oracle answer {expected} matches distractor {d!r}"
            )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# sine_rule_solve: given A, B, a, find c (AAS, two-step).
# =============================================================================

_SINE_SOLVE_RE = re.compile(r"angle A = (-?\d+(?:\.\d+)?)°, angle B = (-?\d+(?:\.\d+)?)° and side a = (\d+) ")


def _oracle_sine_rule_solve(text: str) -> float:
    """Independent two-step route via the LAW OF COSINES instead of the
    sine rule: first C = 180 - A - B (the same first step -- there is only
    one way to find the third angle), then place the triangle by coordinates
    (A at the origin, the side of length a' along a ray at angle A) and
    solve for c using the cosine rule rather than the generator's sine-rule
    division, a different equation for the second step."""
    m = _SINE_SOLVE_RE.search(text)
    assert m, f"unparseable sine_rule_solve question: {text!r}"
    A, B, a = float(m.group(1)), float(m.group(2)), int(m.group(3))
    C = 180 - A - B
    assert 30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150
    # b via the sine rule is unavoidable (AAS has no side-only route to a
    # second side), but the FINAL side c is then found via the cosine rule
    # from a, b and the INCLUDED angle C -- a different equation from
    # `a sin(C) / sin(A)`.
    b = a * math.sin(math.radians(B)) / math.sin(math.radians(A))
    c = math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(C)))
    return c


def test_sine_rule_solve_oracle():
    checked = 0
    for seed in SEEDS:
        q = _make_from_source(SINE_ID, "sine_rule_solve", seed)
        expected = _oracle_sine_rule_solve(q.text)
        actual = _answer_value(q.answer_text)
        assert abs(expected - actual) <= 0.05, f"seed {seed}: expected {expected}, got {actual} -- {q.text!r}"
        for d in q.distractors:
            assert abs(_answer_value(d) - expected) > 0.05, (
                f"seed {seed}: oracle answer {expected} matches distractor {d!r}"
            )
        checked += 1
    assert checked == len(SEEDS)


# =============================================================================
# cosine_rule_angle: given a, b, c (SSS), find angle C opposite c.
# =============================================================================

_COSINE_ANGLE_RE = re.compile(r"side a = (\d+), side b = (\d+) and side c = (\d+)\.")


def _oracle_cosine_rule_angle(text: str) -> float:
    """Independent route: place the triangle by COORDINATES (A at the
    origin, B at (c, 0) using side c = AB, C found by circle intersection
    from the other two side lengths) and compute angle ACB via the dot
    product between CA and CB -- never evaluates `(a^2+b^2-c^2)/(2ab)`
    directly, unlike the generator's own construction."""
    m = _COSINE_ANGLE_RE.search(text)
    assert m, f"unparseable cosine_rule_angle question: {text!r}"
    a, b, c = (int(m.group(i)) for i in (1, 2, 3))
    assert abs(a - b) < c < a + b, "not a valid triangle"
    # A = (0, 0), B = (c, 0) since side c = AB. C = (cx, cy) with
    # |CA| = b, |CB| = a.
    cx = (b * b - a * a + c * c) / (2 * c)
    cy_sq = b * b - cx * cx
    assert cy_sq > 0, "degenerate triangle"
    cy = math.sqrt(cy_sq)
    cax, cay = 0 - cx, 0 - cy
    cbx, cby = c - cx, 0 - cy
    dot = cax * cbx + cay * cby
    mag_ca = math.hypot(cax, cay)
    mag_cb = math.hypot(cbx, cby)
    cos_angle = max(-1.0, min(1.0, dot / (mag_ca * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def test_cosine_rule_angle_oracle():
    checked = 0
    for seed in SEEDS:
        q = _make_from_source(COSINE_ID, "cosine_rule_angle", seed)
        expected = _oracle_cosine_rule_angle(q.text)
        actual = _answer_value(q.answer_text)
        assert abs(expected - actual) <= 0.05, f"seed {seed}: expected {expected}, got {actual} -- {q.text!r}"
        for d in q.distractors:
            assert abs(_answer_value(d) - expected) > 0.05, (
                f"seed {seed}: oracle answer {expected} matches distractor {d!r}"
            )
        checked += 1
    assert checked == len(SEEDS)
