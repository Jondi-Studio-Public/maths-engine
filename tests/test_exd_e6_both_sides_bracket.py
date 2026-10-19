"""A NEW `kind="both_sides_bracket"` on the `linear_equation` topic (`mathgen/generators/algebra.py`).

Every existing `kind="both_sides"` question is `a1*x + b1 = a2*x + b2` -- variables on both sides, but no
bracket. `both_sides_bracket` renders `a1(x + h) = a2*x + b2` -- ONE side (the left) carries a bracket that
must be distributed before the two sides' variable terms can be collected, combining "distribute" with
"variables on both sides" in one question, the genuine reasoning escalation the range-only hard tier was
missing.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "linear_equation"

# Matches the skill's hard source exactly: `{kind: both_sides_bracket}`, with
# the hard preset's own coef_hi=12, sol_hi=30, b_hi=50.
_HARD_PARAMS = dict(kind="both_sides_bracket")


def _make_bracket(seed, **overrides):
    params = {**_HARD_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_both_sides_bracket_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_bracket(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_both_sides_bracket_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone."""
    shapes = {prompt_shape(_make_bracket(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_both_sides_bracket_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium tiers' shapes -- a genuine escalation in kind, not
    merely a bigger number. Easy/medium mirror the skill YAML's own sources
    (unaffected by this change: both still pass `kind: both_sides`, no
    bracket)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kind="both_sides")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kind="both_sides")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_bracket(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_both_sides_bracket_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_bracket(seed).to_json()
        second = _make_bracket(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive x0 from the rendered prompt text ALONE, with
# Fraction arithmetic, never importing mathgen.generators.algebra.
# ---------------------------------------------------------------------------

_BOTH_SIDES_BRACKET_RE = re.compile(r"^Solve (\d+)\(x ([+-]) (\d+)\) = (\d*)x ([+-]) (\d+)$")


def _independent_both_sides_bracket_solution(question_text: str) -> Fraction:
    m = _BOTH_SIDES_BRACKET_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a1_str, sign_h, h_str, a2_str, sign_b2, b2_str = m.groups()
    a1 = int(a1_str)
    h = int(h_str) if sign_h == "+" else -int(h_str)
    a2 = int(a2_str) if a2_str else 1
    b2 = int(b2_str) if sign_b2 == "+" else -int(b2_str)
    # a1(x + h) = a2*x + b2  =>  a1*x + a1*h = a2*x + b2
    #                        =>  (a1 - a2)*x = b2 - a1*h
    x0 = Fraction(b2 - a1 * h, a1 - a2)
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return x0


def test_both_sides_bracket_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_bracket(seed)
        oracle_x0 = _independent_both_sides_bracket_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


import mathgen.generators.algebra as _alg  # noqa: E402


def test_both_sides_bracket_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _alg.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_alg, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        _make_bracket(0)


def test_both_sides_bracket_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    _make_bracket(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_both_sides_bracket_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_bracket(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_both_sides_bracket_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_bracket(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_both_sides_bracket_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_bracket(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_both_sides_bracket_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard: 'Solve 4(x + 12) = 9x - 92', x0 = 28 (see the
    partial_distribution = (b2 - h)/(a1 - a2) = (-92 - 12)/(4 - 9) = 104/5;
    wrong_side = (a1*h - b2)/(a1 - a2) = (48 - (-92))/(4 - 9) = -28;
    dropped_bracket = b2/(a1 - a2) = -92/(-5) = 92/5.
    """
    q = _make_bracket(0)
    x0 = q.answer
    assert x0 == 28
    wrong_candidates = ["104/5", "-28", "92/5", str(x0 + 1)]
    for wrong in wrong_candidates:
        wrong_val = Fraction(wrong)
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kinds unaffected
# ---------------------------------------------------------------------------


def test_original_both_sides_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "both_sides"` branch (the skill's own
    easy/medium sources, unaffected by this change) still matches its own
    long-standing, bracket-free shape."""
    pattern = re.compile(r"^Solve -?\d*x( [+-] \d+)? = -?\d*x( [+-] \d+)?$")
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty, kind="both_sides")
            assert pattern.match(q.text), (seed, difficulty, q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "variables-on-both-sides" in ids
    skill = ids["variables-on-both-sides"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "both_sides_bracket"
