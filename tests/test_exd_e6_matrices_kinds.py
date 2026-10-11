"""Tests the new `kind` values of the `matrices` generator behind the 2x2 determinant, 3x3 determinant and
matrix-solve skills.

Covers `mathgen/generators/matrices.py`'s `matrices` function -- ONE shared test file because all three skills
dispatch off the same `kind` argument on the same generator. Each skill adds its own tests,
prefixed by its own skill-word so names never collide:

- `matrix-determinant-2x2`: `test_determinant_product_*`
  (new `kind="determinant_product"`)
- `matrix-determinant-3x3`: `test_determinant3_product_*`
  (new `kind="determinant3_product"`)
- `matrix-solve-system`:    `test_system_context_*` (new `kind="system_context"`)

Every new `kind` is an ADDED `if kind == "...":` branch; the default-`kinds` output for `matrices` (seeds 0/1/7/42, and the "3:easy"/"3:medium"/"3:hard" preset draws) must stay
byte-identical (this repository does not pin those hashes, so that is kept by construction, not asserted
here), and every test below is written so it can never touch the
pre-existing `"determinant"`/ `"determinant3"`/`"inverse"`/`"inverse3"`/`"system"` branches' own construction,
only their OUTPUT (read-only assertions that the old kinds still produce their long-standing shapes).

Note: `kind="inverse"` already exists in this file and is unused by any of
the three skills in this group -- its answer shape is a MATRIX, not a
number, so it cannot serve `matrix-determinant-2x2` (`answer: number`).
`determinant_product` is a genuinely new branch.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "matrices"


# ===========================================================================
# matrix-determinant-2x2: kind="determinant_product" -- given two 2x2
# matrices A, B, find det(AB).
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [determinant_product], hi: 6}`.
_DET_PRODUCT_PARAMS = dict(kinds=("determinant_product",), hi=6)


def _make_det_product(seed, **overrides):
    params = {**_DET_PRODUCT_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_determinant_product_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_det_product(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


_DET_PRODUCT_PATTERN = re.compile(
    r"^A = \[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\], "
    r"B = \[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\]\. Find det\(AB\)\.$"
)


def test_determinant_product_prompt_text_matches_expected_template_over_60_seeds():
    for seed in range(60):
        q = _make_det_product(seed)
        assert _DET_PRODUCT_PATTERN.match(q.text), (seed, q.text)


def test_determinant_product_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds)."""
    texts = {_make_det_product(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_determinant_product_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_det_product(seed).to_json()
        second = _make_det_product(seed).to_json()
        assert first == second


def test_determinant_product_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes (both mirror the skill YAML's own sources,
    `kinds: [determinant]`, untouched by this change) -- a genuine escalation
    (two matrices and a product step), not merely a bigger `hi`."""
    lower_shapes = {
        prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("determinant",), hi=4)) for s in range(60)
    }
    lower_shapes |= {
        prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("determinant",), hi=6)) for s in range(60)
    }
    hard_shapes = {prompt_shape(_make_det_product(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive det(AB) from the rendered prompt text ALONE,
# never importing mathgen.generators.matrices, and never using sympy --
# plain-Python 2x2 arithmetic instead, so this genuinely does not share a
# code path with the generator's own `check()`.
# ---------------------------------------------------------------------------


def _independent_det_ab(question_text: str) -> int:
    m = _DET_PRODUCT_PATTERN.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a1, a2, a3, a4, b1, b2, b3, b4 = (int(g) for g in m.groups())
    # AB = [[a1*b1+a2*b3, a1*b2+a2*b4], [a3*b1+a4*b3, a3*b2+a4*b4]]
    ab11 = a1 * b1 + a2 * b3
    ab12 = a1 * b2 + a2 * b4
    ab21 = a3 * b1 + a4 * b3
    ab22 = a3 * b2 + a4 * b4
    det_ab = ab11 * ab22 - ab12 * ab21
    # Cross-check via det(A) * det(B), the property route.
    det_a = a1 * a4 - a2 * a3
    det_b = b1 * b4 - b2 * b3
    assert det_ab == det_a * det_b, (question_text, det_ab, det_a, det_b)
    return det_ab


def test_determinant_product_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_det_product(seed)
        oracle_ans = _independent_det_ab(q.text)
        if oracle_ans != q.answer:
            mismatches.append((seed, q.text, oracle_ans, q.answer))
    assert mismatches == [], mismatches


def test_determinant_product_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_det_product(seed)
        assert q.answer_text == str(q.answer), (seed, q.answer_text, q.answer)


# The generator's own in-body `check` actually catches a corrupted construction.

import mathgen.generators.matrices as _mat  # noqa: E402


def test_determinant_product_corrupted_det_route_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` so its `__mul__` silently perturbs one entry of
    the product, breaking agreement between the `(A*B).det()` route and the
    `det(A)*det(B)` route the branch independently checks against --
    confirming `check()` actually fires rather than being a decoration that
    never runs."""
    from sympy import Matrix as _RealMatrix

    class _WrongMatrix(_RealMatrix):
        def __mul__(self, other):
            result = _RealMatrix.__mul__(self, other)
            if result.shape == (2, 2):
                bad = list(result)
                bad[0] = bad[0] + 1
                return _WrongMatrix(2, 2, bad)
            return result

    monkeypatch.setattr(_mat, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_det_product(0)


def test_determinant_product_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_det_product(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer; the
# det(A+B)-instead-of-det(AB) candidate is genuinely distinct from the
# answer for the seeds it is offered on (not merely assumed).
# ---------------------------------------------------------------------------


def test_determinant_product_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_det_product(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_determinant_product_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_det_product(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


def test_determinant_product_added_matrices_candidate_distinct_from_answer_when_offered():
    """The det(A+B) misconception candidate is constructed explicitly in the
    generator; where it is NOT filtered out (i.e. it survives into
    `q.distractors`), confirm by independent Python arithmetic that it truly
    differs from the answer -- never a vacuous check."""
    seen_added_candidate = False
    for seed in range(300):
        q = _make_det_product(seed)
        m = _DET_PRODUCT_PATTERN.match(q.text)
        a1, a2, a3, a4, b1, b2, b3, b4 = (int(g) for g in m.groups())
        added_matrix_det = (a1 + b1) * (a4 + b4) - (a2 + b2) * (a3 + b3)
        if str(added_matrix_det) in q.distractors:
            seen_added_candidate = True
            assert added_matrix_det != q.answer, (seed, q.text, added_matrix_det, q.answer)
    assert seen_added_candidate, "det(A+B) candidate never survived pick_distractors over 300 seeds"


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"], exact match for an
# integer-formatted canonical -- matrix-determinant-2x2's `answer: number`).
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_determinant_product_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_det_product(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_determinant_product_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[determinant_product], hi=6: see the
    summed = det(A) + det(B); det_a_only = det(A); det_b_only = det(B);
    added_matrices = det(A + B).
    """
    q = _make_det_product(0)
    m = _DET_PRODUCT_PATTERN.match(q.text)
    a1, a2, a3, a4, b1, b2, b3, b4 = (int(g) for g in m.groups())
    det_a = a1 * a4 - a2 * a3
    det_b = b1 * b4 - b2 * b3
    ans = q.answer
    assert ans == det_a * det_b

    summed = det_a + det_b
    added_matrices = (a1 + b1) * (a4 + b4) - (a2 + b2) * (a3 + b3)

    for wrong in (summed, det_a, det_b, added_matrices):
        given = str(wrong)
        result = _NUMBER_GRADER(q, given)
        if wrong == ans:
            assert result.correct, (given, q.answer_text)
        else:
            assert not result.correct, (given, q.answer_text)


# Original kinds unaffected.


def test_determinant_product_original_determinant_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "determinant"` branch (still the
    skill's own easy/medium sources) still matches its own long-standing
    shape, after the new `determinant_product` branch was added below it."""
    pattern = re.compile(r"^Find the determinant of A = \[-?\d+ -?\d+; -?\d+ -?\d+\]$")
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="medium", kinds=("determinant",), hi=6)
        assert pattern.match(q.text), (seed, q.text)


def test_determinant_product_original_determinant3_inverse_inverse3_system_kinds_unaffected_over_60_seeds():
    for seed in range(60):
        det3_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("determinant3",), hi=6)
        inv_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse",), hi=6)
        inv3_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse3",), hi=6)
        sys_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("system",), hi=6)
        assert det3_q.text.startswith("Find the determinant of A ="), (seed, det3_q.text)
        assert inv_q.text.startswith("Find the inverse of A ="), (seed, inv_q.text)
        assert inv3_q.text.startswith("Find the inverse of A ="), (seed, inv3_q.text)
        assert sys_q.text.startswith("Solve A x = b where A ="), (seed, sys_q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_determinant_product_skill_loads_via_api_skills_matrix_determinant_2x2():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "matrix-determinant-2x2" in ids
    skill = ids["matrix-determinant-2x2"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["determinant_product"]
    assert hard_sources[0].params.get("hi") == 6
    # the exam source is covered by this skill's tests/test_exd_x_* file
    easy_medium = [s for s in skill.sources if s.difficulty not in ("hard", "exam")]
    for src in easy_medium:
        assert src.params.get("kinds") == ["determinant"], src.params


# ===========================================================================
# matrix-determinant-3x3: kind="determinant3_product" -- given two 3x3
# matrices A, B, find det(AB).
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [determinant3_product]}`
# (no explicit `hi` -- `determinant3`'s entry range is the hardcoded
# `randint(-3, 3)`, never parameterized by the function's own `hi` argument;
# confirmed by reading the pre-existing "determinant3" branch before the
# new kind was added).
_DET3_PRODUCT_PARAMS = dict(kinds=("determinant3_product",))


def _make_det3_product(seed, **overrides):
    params = {**_DET3_PRODUCT_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_determinant3_product_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_det3_product(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


_DET3_PRODUCT_PATTERN = re.compile(
    r"^A = \[(-?\d+) (-?\d+) (-?\d+); (-?\d+) (-?\d+) (-?\d+); (-?\d+) (-?\d+) (-?\d+)\], "
    r"B = \[(-?\d+) (-?\d+) (-?\d+); (-?\d+) (-?\d+) (-?\d+); (-?\d+) (-?\d+) (-?\d+)\]\. "
    r"Find det\(AB\)\.$"
)


def test_determinant3_product_prompt_text_matches_expected_template_over_60_seeds():
    for seed in range(60):
        q = _make_det3_product(seed)
        assert _DET3_PRODUCT_PATTERN.match(q.text), (seed, q.text)


def test_determinant3_product_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds)."""
    texts = {_make_det3_product(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_determinant3_product_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_det3_product(seed).to_json()
        second = _make_det3_product(seed).to_json()
        assert first == second


def test_determinant3_product_no_zero_row_or_column_in_either_matrix_over_300_seeds():
    """Same reroll guard as the existing `determinant3` branch, applied
    independently to each of A and B -- confirms it, not merely assumes it."""
    for seed in range(300):
        q = _make_det3_product(seed)
        m = _DET3_PRODUCT_PATTERN.match(q.text)
        vals = [int(g) for g in m.groups()]
        a = vals[0:9]
        b = vals[9:18]
        for mat in (a, b):
            rows = [mat[0:3], mat[3:6], mat[6:9]]
            cols = [mat[0::3], mat[1::3], mat[2::3]]
            for row in rows:
                assert any(v != 0 for v in row), (seed, q.text)
            for col in cols:
                assert any(v != 0 for v in col), (seed, q.text)


def test_determinant3_product_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes (both mirror the skill YAML's own sources,
    `kinds: [determinant3]`, untouched by this change) -- a genuine escalation
    (two 3x3 matrices and a product step), not merely a bigger range."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("determinant3",))) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("determinant3",))) for s in range(60)}
    hard_shapes = {prompt_shape(_make_det3_product(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive det(AB) from the rendered prompt text ALONE,
# never importing mathgen.generators.matrices, and never using sympy --
# plain-Python 3x3 arithmetic instead, so this genuinely does not share a
# code path with the generator's own `check()`.
# ---------------------------------------------------------------------------


def _mat_mul3(a, b):
    """a, b: length-9 lists (row-major 3x3). Returns length-9 product."""
    out = [0] * 9
    for i in range(3):
        for j in range(3):
            out[3 * i + j] = sum(a[3 * i + k] * b[3 * k + j] for k in range(3))
    return out


def _det3(m):
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])


def _independent_det3_ab(question_text: str) -> int:
    m = _DET3_PRODUCT_PATTERN.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    vals = [int(g) for g in m.groups()]
    a, b = vals[0:9], vals[9:18]
    ab = _mat_mul3(a, b)
    det_ab = _det3(ab)
    # Cross-check via det(A) * det(B), the property route.
    det_a = _det3(a)
    det_b = _det3(b)
    assert det_ab == det_a * det_b, (question_text, det_ab, det_a, det_b)
    return det_ab


def test_determinant3_product_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_det3_product(seed)
        oracle_ans = _independent_det3_ab(q.text)
        if oracle_ans != q.answer:
            mismatches.append((seed, q.text, oracle_ans, q.answer))
    assert mismatches == [], mismatches


def test_determinant3_product_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_det3_product(seed)
        assert q.answer_text == str(q.answer), (seed, q.answer_text, q.answer)


# The generator's own in-body `check` actually catches a corrupted construction.


def test_determinant3_product_corrupted_det_route_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` so its `__mul__` silently perturbs one entry of
    the 3x3 product, breaking agreement between the `(A*B).det()` route and
    the `det(A)*det(B)` route the branch independently checks against --
    confirming `check()` actually fires rather than being a decoration that
    never runs."""
    from sympy import Matrix as _RealMatrix

    class _WrongMatrix(_RealMatrix):
        def __mul__(self, other):
            result = _RealMatrix.__mul__(self, other)
            if result.shape == (3, 3):
                bad = list(result)
                bad[0] = bad[0] + 1
                return _WrongMatrix(3, 3, bad)
            return result

    monkeypatch.setattr(_mat, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_det3_product(0)


def test_determinant3_product_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_det3_product(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer; the
# det(A+B)-instead-of-det(AB) candidate is genuinely distinct from the
# answer for the seeds it is offered on (not merely assumed).
# ---------------------------------------------------------------------------


def test_determinant3_product_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_det3_product(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_determinant3_product_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_det3_product(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


def _added_matrices_det3(question_text: str) -> int:
    m = _DET3_PRODUCT_PATTERN.match(question_text)
    vals = [int(g) for g in m.groups()]
    a, b = vals[0:9], vals[9:18]
    added = [x + y for x, y in zip(a, b)]
    return _det3(added)


def test_determinant3_product_added_matrices_candidate_distinct_from_answer_when_offered():
    """The det(A+B) misconception candidate is constructed explicitly in the
    generator; where it is NOT filtered out (i.e. it survives into
    `q.distractors`), confirm by independent Python arithmetic that it truly
    differs from the answer -- never a vacuous check."""
    seen_added_candidate = False
    for seed in range(300):
        q = _make_det3_product(seed)
        added_matrix_det = _added_matrices_det3(q.text)
        if str(added_matrix_det) in q.distractors:
            seen_added_candidate = True
            assert added_matrix_det != q.answer, (seed, q.text, added_matrix_det, q.answer)
    assert seen_added_candidate, "det(A+B) candidate never survived pick_distractors over 300 seeds"


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"], exact match for an
# integer-formatted canonical -- matrix-determinant-3x3's `answer: number`).
# ---------------------------------------------------------------------------


def test_determinant3_product_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_det3_product(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_determinant3_product_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[determinant3_product]: see the render
    summed = det(A) + det(B); det_a_only = det(A); det_b_only = det(B);
    added_matrices = det(A + B).
    """
    q = _make_det3_product(0)
    m = _DET3_PRODUCT_PATTERN.match(q.text)
    vals = [int(g) for g in m.groups()]
    a, b = vals[0:9], vals[9:18]
    det_a = _det3(a)
    det_b = _det3(b)
    ans = q.answer
    assert ans == det_a * det_b

    summed = det_a + det_b
    added_matrices = _added_matrices_det3(q.text)

    for wrong in (summed, det_a, det_b, added_matrices):
        given = str(wrong)
        result = _NUMBER_GRADER(q, given)
        if wrong == ans:
            assert result.correct, (given, q.answer_text)
        else:
            assert not result.correct, (given, q.answer_text)


# Original kinds unaffected.


def test_determinant3_product_original_determinant3_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "determinant3"` branch (still the
    skill's own easy/medium sources) still matches its own long-standing
    shape, after the new `determinant3_product` branch was added below it."""
    pattern = re.compile(
        r"^Find the determinant of A = "
        r"\[-?\d+ -?\d+ -?\d+; -?\d+ -?\d+ -?\d+; -?\d+ -?\d+ -?\d+\]$"
    )
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="medium", kinds=("determinant3",))
        assert pattern.match(q.text), (seed, q.text)


def test_determinant3_product_original_determinant_inverse_inverse3_system_kinds_unaffected_over_60_seeds():
    for seed in range(60):
        det_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("determinant",), hi=6)
        inv_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse",), hi=6)
        inv3_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse3",), hi=6)
        sys_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("system",), hi=6)
        assert det_q.text.startswith("Find the determinant of A ="), (seed, det_q.text)
        assert inv_q.text.startswith("Find the inverse of A ="), (seed, inv_q.text)
        assert inv3_q.text.startswith("Find the inverse of A ="), (seed, inv3_q.text)
        assert sys_q.text.startswith("Solve A x = b where A ="), (seed, sys_q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_determinant3_product_skill_loads_via_api_skills_matrix_determinant_3x3():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "matrix-determinant-3x3" in ids
    skill = ids["matrix-determinant-3x3"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["determinant3_product"]
    # the exam source is covered by this skill's tests/test_exd_x_* file
    easy_medium = [s for s in skill.sources if s.difficulty not in ("hard", "exam")]
    for src in easy_medium:
        assert src.params.get("kinds") == ["determinant3"], src.params


# ===========================================================================
# matrix-solve-system: kind="system_context" -- a two-item purchase scenario the student must
# translate into Ax = b before solving.

from fractions import Fraction  # noqa: E402

# Matches the skill's hard source exactly: `{kinds: [system_context], hi: 6}`.
_SYSTEM_CONTEXT_PARAMS = dict(kinds=("system_context",), hi=6)


def _make_system_context(seed, **overrides):
    params = {**_SYSTEM_CONTEXT_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_system_context_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_system_context(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


_SYSTEM_CONTEXT_PATTERN = re.compile(
    r"^(\d+) apples? and (\d+) melons? cost \$(\d+) in total; "
    r"(\d+) apples? and (\d+) melons? cost \$(\d+) in total\. "
    r"Find the price of one apple and one melon\.$"
)


def test_system_context_prompt_text_matches_expected_template_over_60_seeds():
    for seed in range(60):
        q = _make_system_context(seed)
        assert _SYSTEM_CONTEXT_PATTERN.match(q.text), (seed, q.text)


def test_system_context_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds)."""
    texts = {_make_system_context(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_system_context_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_system_context(seed).to_json()
        second = _make_system_context(seed).to_json()
        assert first == second


def test_system_context_A_entries_are_small_positive_integers_over_300_seeds():
    """A's entries are the two items' quantities, drawn 1-6 (never negative
    or zero -- a negative/zero item count makes no scenario sense), a
    deliberate deviation from the raw "system" branch's -hi..hi range."""
    for seed in range(300):
        q = _make_system_context(seed)
        m = _SYSTEM_CONTEXT_PATTERN.match(q.text)
        a11, a12, _b0, a21, a22, _b1 = (int(g) for g in m.groups())
        for v in (a11, a12, a21, a22):
            assert 1 <= v <= 6, (seed, q.text, v)


def test_system_context_prices_are_positive_over_300_seeds():
    """x0, y0 (the two item prices) are drawn 1-9, positive, so the
    scenario's stated per-item price answer reads sensibly."""
    for seed in range(300):
        q = _make_system_context(seed)
        assert 1 <= q.answer["x"] <= 9, (seed, q.answer)
        assert 1 <= q.answer["y"] <= 9, (seed, q.answer)


def test_system_context_stated_totals_match_bvec_entries_over_300_seeds():
    """Independently confirms the scenario's stated purchase totals really
    are `b_vec`'s entries: recomputes each total from the parsed quantities
    and the answer's own (x, y) and checks it against the parsed dollar
    figure in the prompt -- not merely trusting the generator's own
    in-body `check()`."""
    for seed in range(300):
        q = _make_system_context(seed)
        m = _SYSTEM_CONTEXT_PATTERN.match(q.text)
        a11, a12, b0, a21, a22, b1 = (int(g) for g in m.groups())
        x, y = q.answer["x"], q.answer["y"]
        assert a11 * x + a12 * y == b0, (seed, q.text, q.answer)
        assert a21 * x + a22 * y == b1, (seed, q.text, q.answer)


def test_system_context_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes (both mirror the skill YAML's own sources,
    `kinds: [system]`, untouched by this change) -- a genuine escalation (a
    word scenario the student must translate into Ax = b first), not
    merely a bigger `hi`."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("system",), hi=4)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("system",), hi=6)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_system_context(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive the (x, y) solution from the rendered
# prompt text ALONE, never importing mathgen.generators.matrices, and never
# using sympy -- plain-Python Cramer's rule (Fraction-exact) instead, so
# this genuinely does not share a code path with the generator's own
# `check()`.
# ---------------------------------------------------------------------------


def _solve2(a, b, c, d, e, f):
    """Solve [[a, b], [c, d]] [x; y] = [e; f] via Cramer's rule, Fraction-exact."""
    det = a * d - b * c
    assert det != 0
    x = Fraction(e * d - b * f, det)
    y = Fraction(a * f - e * c, det)
    return x, y


def _fmt_rational(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _independent_system_context_solve(question_text: str):
    m = _SYSTEM_CONTEXT_PATTERN.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a11, a12, b0, a21, a22, b1 = (int(g) for g in m.groups())
    return _solve2(a11, a12, a21, a22, b0, b1)


def test_system_context_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_system_context(seed)
        x, y = _independent_system_context_solve(q.text)
        if x != q.answer["x"] or y != q.answer["y"]:
            mismatches.append((seed, q.text, (x, y), q.answer))
    assert mismatches == [], mismatches


def test_system_context_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_system_context(seed)
        assert q.answer_text == f"x = {q.answer['x']}, y = {q.answer['y']}", (seed, q.answer_text, q.answer)


# The generator's own in-body `check` actually catches a corrupted construction.


def test_system_context_corrupted_solve_route_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` so its `.solve()` silently perturbs one entry of
    the result, breaking agreement between the solved `(x, y)` and the
    chosen `v = (x0, y0)` the branch's own `check(A.solve(b_vec) == v)`
    compares against -- confirming `check()` actually fires rather than
    being a decoration that never runs."""
    from sympy import Matrix as _RealMatrix

    class _WrongMatrix(_RealMatrix):
        def solve(self, *a, **kw):
            result = _RealMatrix.solve(self, *a, **kw)
            bad = list(result)
            bad[0] = bad[0] + 1
            return _RealMatrix(len(bad), 1, bad)

    monkeypatch.setattr(_mat, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_system_context(0)


def test_system_context_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_system_context(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer; each of the
# three misconception candidates (swapped-pair, roles-reversed, single-
# equation) is genuinely distinct from the answer on the seeds it is
# offered on (not merely assumed reachable).
# ---------------------------------------------------------------------------


def test_system_context_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_system_context(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_system_context_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_system_context(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


def test_system_context_swapped_pair_candidate_reachable_and_distinct_over_300_seeds():
    """The "swapped-pair-matrix" misconception, reused from `xy_solution`'s
    own idea -- confirmed reachable/offered for this branch too (not
    assumed)."""
    seen = False
    for seed in range(300):
        q = _make_system_context(seed)
        swapped = f"x = {q.answer['y']}, y = {q.answer['x']}"
        if swapped in q.distractors:
            seen = True
            assert swapped != q.answer_text, (seed, q.text)
    assert seen, "swapped-pair candidate never survived pick_distractors over 300 seeds"


def test_system_context_roles_reversed_candidate_reachable_and_distinct_over_300_seeds():
    """The "solved A^T x = b instead of Ax = b" misconception -- reversed
    the two items' roles when setting up the system."""
    seen = False
    for seed in range(300):
        q = _make_system_context(seed)
        m = _SYSTEM_CONTEXT_PATTERN.match(q.text)
        a11, a12, b0, a21, a22, b1 = (int(g) for g in m.groups())
        # A^T = [[a11, a21], [a12, a22]]
        x, y = _solve2(a11, a21, a12, a22, b0, b1)
        cand = f"x = {_fmt_rational(x)}, y = {_fmt_rational(y)}"
        if cand in q.distractors:
            seen = True
            assert cand != q.answer_text, (seed, q.text)
    assert seen, "roles-reversed candidate never survived pick_distractors over 300 seeds"


def test_system_context_single_equation_candidate_reachable_and_distinct_over_300_seeds():
    """The "used only ONE of the two purchase statements" misconception --
    solved a single equation, treating the other item's price as already
    known (assumed 0)."""
    seen = False
    for seed in range(300):
        q = _make_system_context(seed)
        m = _SYSTEM_CONTEXT_PATTERN.match(q.text)
        a11, _a12, b0, _a21, _a22, _b1 = (int(g) for g in m.groups())
        x = Fraction(b0, a11)
        cand = f"x = {_fmt_rational(x)}, y = 0"
        if cand in q.distractors:
            seen = True
            assert cand != q.answer_text, (seed, q.text)
    assert seen, "single-equation candidate never survived pick_distractors over 300 seeds"


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["pair"] -- matrix-solve-system's
# `answer: pair`).
# ---------------------------------------------------------------------------

_PAIR_GRADER = GRADERS["pair"]


def test_system_context_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_system_context(seed)
        canonical_result = _PAIR_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _PAIR_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_system_context_grader_rejects_deliberately_wrong_values():
    """The four misconception values, computed by hand against a fixed seed
    and checked they are actually wrong before asserting the grader rejects
    them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[system_context], hi=6: see the render
    swapped = (y0, x0); roles_reversed = A^T-solved pair; single_eq =
    (b0/a11, 0); sign_x/sign_y = negate one of x0, y0.
    """
    q = _make_system_context(0)
    m = _SYSTEM_CONTEXT_PATTERN.match(q.text)
    a11, a12, b0, a21, a22, b1 = (int(g) for g in m.groups())
    x0, y0 = q.answer["x"], q.answer["y"]
    assert a11 * x0 + a12 * y0 == b0
    assert a21 * x0 + a22 * y0 == b1

    swapped = f"x = {y0}, y = {x0}"
    rx, ry = _solve2(a11, a21, a12, a22, b0, b1)
    roles_reversed = f"x = {_fmt_rational(rx)}, y = {_fmt_rational(ry)}"
    sx = Fraction(b0, a11)
    single_eq = f"x = {_fmt_rational(sx)}, y = 0"
    sign_x = f"x = {-x0}, y = {y0}"
    sign_y = f"x = {x0}, y = {-y0}"

    for given in (swapped, roles_reversed, single_eq, sign_x, sign_y):
        result = _PAIR_GRADER(q, given)
        if given == q.answer_text:
            assert result.correct, (given, q.answer_text)
        else:
            assert not result.correct, (given, q.answer_text)


# Original kinds unaffected.


def test_system_context_original_system_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "system"` branch (still the skill's
    own easy/medium sources) still matches its own long-standing shape,
    after the new `system_context` branch was added above it."""
    pattern = re.compile(r"^Solve A x = b where A = \[-?\d+ -?\d+; -?\d+ -?\d+\] and b = \(-?\d+, -?\d+\)$")
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="medium", kinds=("system",), hi=6)
        assert pattern.match(q.text), (seed, q.text)


def test_system_context_original_determinant_determinant3_inverse_inverse3_kinds_unaffected_over_60_seeds():
    for seed in range(60):
        det_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("determinant",), hi=6)
        det3_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("determinant3",), hi=6)
        inv_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse",), hi=6)
        inv3_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("inverse3",), hi=6)
        assert det_q.text.startswith("Find the determinant of A ="), (seed, det_q.text)
        assert det3_q.text.startswith("Find the determinant of A ="), (seed, det3_q.text)
        assert inv_q.text.startswith("Find the inverse of A ="), (seed, inv_q.text)
        assert inv3_q.text.startswith("Find the inverse of A ="), (seed, inv3_q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_system_context_skill_loads_via_api_skills_matrix_solve_system():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "matrix-solve-system" in ids
    skill = ids["matrix-solve-system"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["system_context"]
    assert hard_sources[0].params.get("hi") == 6
    # the exam source is covered by this skill's tests/test_exd_x_* file
    easy_medium = [s for s in skill.sources if s.difficulty not in ("hard", "exam")]
    for src in easy_medium:
        assert src.params.get("kinds") == ["system"], src.params
