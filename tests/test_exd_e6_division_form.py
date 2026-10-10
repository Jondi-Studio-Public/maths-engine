"""A NEW `kind="division_form"` on the `linear_equation` topic (`mathgen/generators/algebra.py`).

Every existing `kind` on this topic (`standard`, `brackets`, `brackets_leading`, `double_brackets`,
`both_sides`) is solved by DIVIDING the variable's coefficient away. `division_form` renders `x/a = c` and is
solved the opposite way -- MULTIPLY both sides by `a` -- the genuine reasoning escalation the range-only hard
tier was missing.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "linear_equation"

# Matches the skill's hard source exactly: `{kind: division_form,
# allow_negative: true, coef_hi: 15, sol_hi: 15}`.
_HARD_PARAMS = dict(kind="division_form", allow_negative=True, coef_hi=15, sol_hi=15)


def _make_division(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_division_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_division(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_division_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone."""
    shapes = {prompt_shape(_make_division(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_division_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium tiers' shapes -- a genuine escalation in kind, not
    merely a bigger number. Easy/medium mirror the skill YAML's own sources
    (unaffected by this change)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", a_fixed=1)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="easy", b_zero=True)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", allow_negative=True)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_division(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_division_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_division(seed).to_json()
        second = _make_division(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive x0 from the rendered prompt text ALONE, with
# Fraction arithmetic, never importing mathgen.generators.algebra.
# ---------------------------------------------------------------------------

_DIV_RE = re.compile(r"^Solve x/(-?\d+) = (-?\d+(?:/\d+)?)$")


def _independent_division_solution(question_text: str) -> Fraction:
    m = _DIV_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a_str, c_str = m.groups()
    a = int(a_str)
    c = Fraction(c_str)
    # x/a = c  =>  multiply both sides by a  =>  x = a * c
    x0 = a * c
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return x0


def test_division_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_division(seed)
        oracle_x0 = _independent_division_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


import mathgen.generators.algebra as _alg  # noqa: E402


def test_division_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _alg.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_alg, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        _make_division(0)


def test_division_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    _make_division(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_division_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_division(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_division_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_division(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_division_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_division(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_division_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=1, difficulty=hard: a=4, c=1/2, x0=2 (see the render output quoted
    numerator_only = a*c.p = 4*1 = 4.
    """
    q = _make_division(1)
    x0 = q.answer
    assert x0 == 2
    wrong_candidates = ["1/8", "1/2", "4", str(x0 + 1)]
    for wrong in wrong_candidates:
        wrong_val = Fraction(wrong)
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kinds unaffected
# ---------------------------------------------------------------------------


def test_original_standard_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "standard"` branch (the mental block
    and easy/medium sources all use it implicitly, never passing `kind`)
    still matches its own long-standing shape."""
    pattern = re.compile(r"^Solve -?\d*x( [+-] \d+)? = -?\d+$")
    for seed in range(300):
        for params in (dict(a_fixed=1), dict(b_zero=True), dict(allow_negative=True)):
            q = make(TOPIC, seed=seed, difficulty="medium", **params)
            assert pattern.match(q.text), (seed, params, q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "one-step-equations" in ids
    skill = ids["one-step-equations"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "division_form"
