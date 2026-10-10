"""A NEW `kind="division_two_step"` on the `linear_equation` topic (`mathgen/generators/algebra.py`).

Every existing `kind` on this topic (`standard`, `brackets`, `brackets_leading`, `double_brackets`,
`both_sides`, `division_form`) that renders a TWO-step equation is solved by subtracting a constant, THEN
DIVIDING the variable's coefficient away (`standard`: `a*x + b = c`). `division_two_step` renders `x/a + b =
c` (or `x/a - b = c` when `b` is drawn negative) and is solved the opposite way for its final step -- subtract
`b`, then MULTIPLY both sides by `a` -- the genuine reasoning escalation the range-only hard tier was missing.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "linear_equation"

# Matches the skill's hard source exactly: `{kind: division_two_step}`, with
# the hard preset's own coef_hi=12, sol_hi=30, b_hi=50.
_HARD_PARAMS = dict(kind="division_two_step")


def _make_division_two_step(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_division_two_step_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_division_two_step(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_division_two_step_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone."""
    shapes = {prompt_shape(_make_division_two_step(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_division_two_step_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium tiers' shapes -- a genuine escalation in kind, not
    merely a bigger number. Easy/medium mirror the skill YAML's own sources
    (unaffected by this change: both still pass no `kind`, so `kind="standard"`
    by default)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_division_two_step(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_division_two_step_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_division_two_step(seed).to_json()
        second = _make_division_two_step(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive x0 from the rendered prompt text ALONE, with
# Fraction arithmetic, never importing mathgen.generators.algebra.
# ---------------------------------------------------------------------------

_DIV_TWO_STEP_RE = re.compile(r"^Solve x/(-?\d+) ([+-]) (\d+) = (-?\d+(?:/\d+)?)$")


def _independent_division_two_step_solution(question_text: str) -> Fraction:
    m = _DIV_TWO_STEP_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a_str, sign, b_str, c_str = m.groups()
    a = int(a_str)
    b_signed = int(b_str) if sign == "+" else -int(b_str)
    c = Fraction(c_str)
    # x/a + b = c  =>  x/a = c - b  =>  x = a * (c - b)
    x0 = a * (c - b_signed)
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return x0


def test_division_two_step_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_division_two_step(seed)
        oracle_x0 = _independent_division_two_step_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


import mathgen.generators.algebra as _alg  # noqa: E402


def test_division_two_step_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _alg.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_alg, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        _make_division_two_step(0)


def test_division_two_step_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    _make_division_two_step(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_division_two_step_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_division_two_step(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_division_two_step_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_division_two_step(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_division_two_step_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_division_two_step(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_division_two_step_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard: 'Solve x/4 + 35 = 42', x0 = 28 (see the render
    divided_instead = p/a = 7/4; added_instead = a*(c+b) = 4*77 = 308;
    wrong_order = a*c - b = 4*42 - 35 = 133.
    """
    q = _make_division_two_step(0)
    x0 = q.answer
    assert x0 == 28
    wrong_candidates = ["7/4", "308", "133", str(x0 + 1)]
    for wrong in wrong_candidates:
        wrong_val = Fraction(wrong)
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kinds unaffected
# ---------------------------------------------------------------------------


def test_original_standard_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "standard"` branch (the easy/medium
    sources and the mental block all use it implicitly, never passing
    `kind`) still matches its own long-standing shape."""
    pattern = re.compile(r"^Solve -?\d*x( [+-] \d+)? = -?\d+$")
    for seed in range(300):
        for difficulty in ("easy", "medium"):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert pattern.match(q.text), (seed, difficulty, q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "two-step-equations" in ids
    skill = ids["two-step-equations"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "division_two_step"
