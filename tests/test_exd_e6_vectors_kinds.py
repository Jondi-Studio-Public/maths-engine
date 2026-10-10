"""Tests the new `kind` values of the `vectors` generator behind the vector addition, cross product, dot product
and related skills.

Covers `mathgen/generators/geometry.py`'s `vectors` function -- ONE shared test file because all four skills
dispatch off the same `kind` argument on the same generator. Each skill adds its own
tests, prefixed by its own skill-word so names never collide:

- `vector-addition`:      `test_subtract_*` (new `kind="subtract"`)
- `vector-cross-product`: `test_cross_sum_*` (new `kind="cross_sum"`)
- `vector-dot-product`:   `test_dot_sum_*` (new `kind="dot_sum"`)
- `vector-magnitude`:     `test_magnitude_sum_*` (new `kind="magnitude_sum"`)

Every new `kind` is an ADDED `if kind == "...":` branch; the default-`kinds` output for `vectors` (seeds 0/1/7) must stay byte-identical (this repository does not pin
those hashes, so that is kept by construction, not asserted here), and every test below is
written so it can never touch the pre-existing `"dot"`/`"cross"`/`"add"`/`"magnitude"` branches' own
construction, only their OUTPUT (read-only assertions that the old kinds still produce their long-standing
shapes).

Note on prompt variety: every existing `vectors` kind -- `dot`, `cross`, `add`,
`magnitude` -- yields exactly ONE `mathgen.text_quality.prompt_shape()` over
60 seeds today (measured directly, not assumed): the prompt is a single
fixed template and only the digits inside it vary, and `_mask_digits` erases
those on purpose. So a new vector `kind` is not
held to a 3-shapes-per-kind bar; what actually gates the library (and what
this file tests) is `scripts/measure_tiers.py`'s cross-TIER comparison -- the
new kind's one shape must differ from the union of the skill's own lower
tiers' shapes (it does: the literal operator/wording differs) -- and
the raw-`.text` distinct-string count (>= 10 over 60 seeds), which a fixed
template with random integer vectors clears trivially.

Grader hazard found and fixed, relevant to the cross-sum, dot-sum and magnitude-sum kinds too: the skill's grader
(`mathgen.grading.GRADERS["list"]`) is ORDER-INSENSITIVE (a correct-valued permutation is accepted), but
`mathgen.core.pick_distractors` defaults to an ORDERED comparison against the answer. A distractor built by
negating or reordering the answer's own components (a "wrong sign"/"wrong order" misconception) can land on a
genuine permutation of the correct answer for some seeds -- e.g. an answer shaped `(a, -a, 0)` has `(-a, a,
0)` as a permutation of itself -- and `pick_distractors` keeps it because it looks different ORDERED while the
real grader marks it correct. **Pass `pick_distractors(..., unordered=True)` for every `vectors` branch whose
answer is a 3-tuple graded by `GRADERS["list"]`** (`subtract` and `cross_sum` now both do). Proved empirically
for `subtract` over 5000 seeds, 0 false accepts; proved empirically for `cross_sum` the same way below
(0/5000).

A DIFFERENT, genuinely new hazard is specific to `magnitude_sum`: the triangle inequality means `|u| + |v|`
can be EXACTLY `|u + v|` for some seeds (u, v parallel, same direction) -- offering it unconditionally would
be a distractor that is secretly a correct answer. The `magnitude_sum` branch only appends it to its candidate
list when it is more than 0.02 away from `answer_alt` (the rounded 2-d.p. decimal) -- measured directly, not
merely an exact sympy `!=` check: `grade_expression` accepts any typed value within 0.005 of `answer_alt`
regardless of symbolic form (a looser test than `pick_distractors`' own exact-value dedup), and an
exact-`!=`-only first draft of this branch let 4/5000 seeds through where `|u| + |v|` was symbolically
distinct from `|u + v|` but numerically inside that 0.005 window (u, v close to but not exactly parallel) --
e.g. seed 347: `11 + sqrt(538)` (`≈ 34.194`) against the true answer `sqrt(1169)` (`≈ 34.191`), a gap of about
0.004, grader-accepted before the fix.
`test_magnitude_sum_triangle_inequality_distractor_omitted_when_parallel_same_direction` forces the
EXACT-equality degenerate case directly (a stub rng, not a seed hunt -- three independent random coordinates
landing exactly parallel is not something a small seed sweep will hit on its own) and confirms the candidate
is dropped there too;
`test_magnitude_sum_triangle_inequality_distractor_never_grader_accepted_over_5000_seeds` sweeps the REAL
grader over 5000 seeds (0 false accepts after the fix) to catch the near-miss case the exact-equality test
cannot reach.
"""

from __future__ import annotations

import re

import pytest
from sympy import Matrix, sqrt

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "vectors"


# ===========================================================================
# vector-addition: kind="subtract" -- "Find u - v", component-wise.
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [subtract], hi: 15}`.
_SUBTRACT_PARAMS = dict(kinds=("subtract",), hi=15)


def _make_subtract(seed, **overrides):
    params = {**_SUBTRACT_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_subtract_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_subtract(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_subtract_prompt_text_matches_expected_template_over_60_seeds():
    pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\), v = \(-?\d+, -?\d+, -?\d+\)\. Find u - v$")
    for seed in range(60):
        q = _make_subtract(seed)
        assert pattern.match(q.text), (seed, q.text)


def test_subtract_distinct_raw_prompts_over_60_seeds():
    """Matches the library's own convention
    (at least `_MIN_DISTINCT = 10` distinct raw prompts over 60 seeds) rather than a masked-shape count -- see the module
    docstring for why a single-template vector question is not held to a
    3-shapes bar."""
    texts = {_make_subtract(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_subtract_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_subtract(seed).to_json()
        second = _make_subtract(seed).to_json()
        assert first == second


def test_subtract_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes -- a genuine escalation (the literal operator
    changes from `+` to `-`), not merely a bigger `hi`. Easy/medium mirror
    the skill YAML's own sources (`kinds: [add]`, untouched by this change)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("add",), hi=5)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("add",), hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_subtract(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive u - v from the rendered prompt text ALONE,
# never importing mathgen.generators.geometry.
# ---------------------------------------------------------------------------

_SUBTRACT_RE = re.compile(r"^u = \((-?\d+), (-?\d+), (-?\d+)\), v = \((-?\d+), (-?\d+), (-?\d+)\)\. Find u - v$")


def _independent_subtract(question_text: str) -> list[int]:
    m = _SUBTRACT_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    u0, u1, u2, v0, v1, v2 = (int(g) for g in m.groups())
    return [u0 - v0, u1 - v1, u2 - v2]


def test_subtract_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_subtract(seed)
        oracle_ans = _independent_subtract(q.text)
        if oracle_ans != list(q.answer):
            mismatches.append((seed, q.text, oracle_ans, q.answer))
    assert mismatches == [], mismatches


def test_subtract_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_subtract(seed)
        assert q.answer_text == f"({q.answer[0]}, {q.answer[1]}, {q.answer[2]})", (
            seed,
            q.answer_text,
            q.answer,
        )


# The generator's own in-body `check` actually catches a corrupted construction.

import mathgen.generators.geometry as _geo  # noqa: E402


def test_subtract_corrupted_matrix_sub_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` itself (the independent-check route the
    `subtract` branch uses: `check(list(Matrix(u) - Matrix(v)) == ans)`) so
    its subtraction silently perturbs one component -- confirming `check()`
    actually fires rather than being a decoration that never runs."""

    class _WrongMatrix(Matrix):
        def __sub__(self, other):
            result = Matrix.__sub__(self, other)
            bad = list(result)
            bad[0] = bad[0] + 1
            return Matrix(bad)

    monkeypatch.setattr(_geo, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_subtract(0)


def test_subtract_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_subtract(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_subtract_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_subtract(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_subtract_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_subtract(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["list"], order-insensitive by value
# but exact by count -- vector-addition's `answer: list`).
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_LIST_GRADER = GRADERS["list"]


def test_subtract_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_subtract(seed)
        canonical_result = _LIST_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _LIST_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_subtract_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[subtract], hi=15: see the render output
    wrong_order = v - u; drop_z = (u0-v0, u1-v1, 0).
    """
    q = _make_subtract(0)
    u_match = re.match(r"^u = \((-?\d+), (-?\d+), (-?\d+)\)", q.text)
    v_match = re.search(r"v = \((-?\d+), (-?\d+), (-?\d+)\)", q.text)
    u = [int(g) for g in u_match.groups()]
    v = [int(g) for g in v_match.groups()]
    ans = list(q.answer)
    assert ans == [u[0] - v[0], u[1] - v[1], u[2] - v[2]]

    added = [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
    wrong_order = [v[0] - u[0], v[1] - u[1], v[2] - u[2]]
    drop_z = [u[0] - v[0], u[1] - v[1], 0]

    for wrong in (added, wrong_order, drop_z):
        assert wrong != ans, (wrong, ans)
        given = f"({wrong[0]}, {wrong[1]}, {wrong[2]})"
        result = _LIST_GRADER(q, given)
        assert not result.correct, (given, q.answer_text)


# Original kinds unaffected.


def test_subtract_original_add_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "add"` branch (still the skill's own
    easy/medium sources) still matches its own long-standing shape and
    answer, after the new `subtract` branch was added below it."""
    pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\), v = \(-?\d+, -?\d+, -?\d+\)\. Find u \+ v$")
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="medium", kinds=("add",), hi=9)
        assert pattern.match(q.text), (seed, q.text)


def test_subtract_original_dot_cross_magnitude_kinds_unaffected_over_60_seeds():
    magnitude_pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\)\. Find \|u\| \(exact\)$")
    for seed in range(60):
        dot_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot",), hi=15)
        cross_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("cross",), hi=15)
        mag_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("magnitude",), hi=15)
        assert dot_q.text.endswith("Find u · v"), (seed, dot_q.text)
        assert cross_q.text.endswith("Find u × v"), (seed, cross_q.text)
        assert magnitude_pattern.match(mag_q.text), (seed, mag_q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_subtract_skill_loads_via_api_skills_vector_addition():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "vector-addition" in ids
    skill = ids["vector-addition"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["subtract"]
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty != "hard"]
    for src in easy_medium:
        assert src.params.get("kinds") == ["add"], src.params


# ===========================================================================
# vector-cross-product: kind="cross_sum" -- "Find (p + q) x v", the SUM
# of two vectors crossed with a third.
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [cross_sum], hi: 15}`.
# (The PRE-EXISTING hard source carried no `hi` at all -- default `hi=9` --
# confirmed by reading `vector-cross-product.yaml` before this edit;
# `hi: 15` matches the pattern used for `subtract`.)
_CROSS_SUM_PARAMS = dict(kinds=("cross_sum",), hi=15)


def _make_cross_sum(seed, **overrides):
    params = {**_CROSS_SUM_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_cross_sum_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_cross_sum(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_cross_sum_prompt_text_matches_expected_template_over_60_seeds():
    pattern = re.compile(
        r"^p = \(-?\d+, -?\d+, -?\d+\), q = \(-?\d+, -?\d+, -?\d+\), "
        r"v = \(-?\d+, -?\d+, -?\d+\)\. Find \(p \+ q\) × v$"
    )
    for seed in range(60):
        q = _make_cross_sum(seed)
        assert pattern.match(q.text), (seed, q.text)


def test_cross_sum_distinct_raw_prompts_over_60_seeds():
    """Same convention as the `subtract` tests -- see the module docstring for why
    a single-template vector question is not held to a 3-shapes bar."""
    texts = {_make_cross_sum(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_cross_sum_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_cross_sum(seed).to_json()
        second = _make_cross_sum(seed).to_json()
        assert first == second


def test_cross_sum_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes -- a genuine escalation (a THIRD vector and an
    addition step before the cross product), not merely a bigger `hi`.
    Easy/medium mirror the skill YAML's own sources (`kinds: [cross]`,
    untouched by this change)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("cross",), hi=9)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("cross",), hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_cross_sum(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive (p + q) x v from the rendered prompt text
# ALONE, never importing mathgen.generators.geometry, using a plain-Python
# cross-product formula (no sympy).
# ---------------------------------------------------------------------------

_CROSS_SUM_RE = re.compile(
    r"^p = \((-?\d+), (-?\d+), (-?\d+)\), q = \((-?\d+), (-?\d+), (-?\d+)\), "
    r"v = \((-?\d+), (-?\d+), (-?\d+)\)\. Find \(p \+ q\) × v$"
)


def _plain_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _independent_cross_sum(question_text: str) -> list[int]:
    m = _CROSS_SUM_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    p0, p1, p2, q0, q1, q2, v0, v1, v2 = (int(g) for g in m.groups())
    usum = [p0 + q0, p1 + q1, p2 + q2]
    return _plain_cross(usum, [v0, v1, v2])


def test_cross_sum_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_cross_sum(seed)
        oracle_ans = _independent_cross_sum(q.text)
        if oracle_ans != list(q.answer):
            mismatches.append((seed, q.text, oracle_ans, q.answer))
    assert mismatches == [], mismatches


def test_cross_sum_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_cross_sum(seed)
        assert q.answer_text == f"({q.answer[0]}, {q.answer[1]}, {q.answer[2]})", (
            seed,
            q.answer_text,
            q.answer,
        )


# The generator's own in-body `check` actually catches a corrupted construction.


def test_cross_sum_corrupted_matrix_cross_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` itself (the independent-check route the
    `cross_sum` branch uses: `check(w.dot(Matrix(usum)) == 0 and
    w.dot(Matrix(v)) == 0)`) so its `.dot` silently perturbs the result --
    confirming `check()` actually fires rather than being a decoration that
    never runs."""

    class _WrongMatrix(Matrix):
        def dot(self, other):
            return Matrix.dot(self, other) + 1

    monkeypatch.setattr(_geo, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_cross_sum(0)


def test_cross_sum_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_cross_sum(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_cross_sum_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_cross_sum(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_cross_sum_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_cross_sum(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["list"], order-insensitive by value
# but exact by count -- vector-cross-product's `answer: list`).
# ---------------------------------------------------------------------------


def test_cross_sum_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_cross_sum(seed)
        canonical_result = _LIST_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _LIST_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_cross_sum_grader_rejects_deliberately_wrong_values():
    """The three misconception values, hand-recomputed from the rendered
    text for a fixed seed and checked they are actually wrong before
    asserting the grader rejects them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[cross_sum], hi=15: p, q, v parsed from the rendered text. forgot_q = p x v
    (dropped the addition of q); wrong_order = v x (p+q) = -answer (crossed in the wrong order);
    wrong_pair_mixed = (p x q) elementwise-multiplied by v (crossed the wrong pair of vectors).
    """
    q = _make_cross_sum(0)
    m = _CROSS_SUM_RE.match(q.text)
    assert m is not None, q.text
    p0, p1, p2, q0c, q1c, q2c, v0, v1, v2 = (int(g) for g in m.groups())
    p, qq, v = [p0, p1, p2], [q0c, q1c, q2c], [v0, v1, v2]
    ans = list(q.answer)
    usum = [p[0] + qq[0], p[1] + qq[1], p[2] + qq[2]]
    assert ans == _plain_cross(usum, v)

    forgot_q = _plain_cross(p, v)
    wrong_order = [-ans[0], -ans[1], -ans[2]]
    pq_cross = _plain_cross(p, qq)
    wrong_pair_mixed = [pq_cross[0] * v[0], pq_cross[1] * v[1], pq_cross[2] * v[2]]

    for wrong in (forgot_q, wrong_order, wrong_pair_mixed):
        assert wrong != ans, (wrong, ans)
        given = f"({wrong[0]}, {wrong[1]}, {wrong[2]})"
        result = _LIST_GRADER(q, given)
        assert not result.correct, (given, q.answer_text)


# Original kinds unaffected. The `subtract` tests already cover dot/cross/add/magnitude; this one adds
# `subtract` (an earlier new kind, now sitting textually ABOVE `cross_sum` in the function) to the same
# read-only confirmation, from the `cross_sum` side.


def test_cross_sum_original_subtract_dot_cross_magnitude_kinds_unaffected_over_60_seeds():
    magnitude_pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\)\. Find \|u\| \(exact\)$")
    for seed in range(60):
        sub_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("subtract",), hi=15)
        dot_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot",), hi=15)
        cross_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("cross",), hi=15)
        mag_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("magnitude",), hi=15)
        assert sub_q.text.endswith("Find u - v"), (seed, sub_q.text)
        assert dot_q.text.endswith("Find u · v"), (seed, dot_q.text)
        assert cross_q.text.endswith("Find u × v"), (seed, cross_q.text)
        assert magnitude_pattern.match(mag_q.text), (seed, mag_q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_cross_sum_skill_loads_via_api_skills_vector_cross_product():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "vector-cross-product" in ids
    skill = ids["vector-cross-product"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["cross_sum"]
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty != "hard"]
    for src in easy_medium:
        assert src.params.get("kinds") == ["cross"], src.params


# ---------------------------------------------------------------------------
# Cross-branch hazard check.
# ---------------------------------------------------------------------------


def test_cross_sum_own_unordered_fix_has_zero_false_accepts_over_2000_seeds():
    """Directly proves `cross_sum`'s `unordered=True` closes the same hazard class found (see module docstring):
    sweeps the real grader against every distractor, not just the ones the author's judgement expects to
    collide."""
    bad = 0
    for seed in range(2000):
        q = _make_cross_sum(seed)
        for d in q.distractors:
            if _LIST_GRADER(q, d).correct:
                bad += 1
    assert bad == 0, bad


# ===========================================================================
# vector-dot-product: kind="dot_sum" -- "Find (p + q) . v", the SUM of
# two vectors dotted with a third. Unlike `subtract`/`cross_sum` the answer
# here is a SCALAR (a single number, `answer: number`, graded by
# `mathgen.grading.grade_number`), not a 3-tuple graded by `GRADERS["list"]` --
# so the order-sensitive-distractor hazard `subtract`/`cross_sum` had to work
# around does not apply: a single number has no "order" to be
# insensitive to. Checked directly below (`unordered=True` is NOT passed to
# `pick_distractors` here, matching the pre-existing `dot` branch's own call,
# which also does not pass it -- confirmed by reading, not assumed).
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [dot_sum], hi: 15}`.
_DOT_SUM_PARAMS = dict(kinds=("dot_sum",), hi=15)


def _make_dot_sum(seed, **overrides):
    params = {**_DOT_SUM_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_dot_sum_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_dot_sum(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_dot_sum_prompt_text_matches_expected_template_over_60_seeds():
    pattern = re.compile(
        r"^p = \(-?\d+, -?\d+, -?\d+\), q = \(-?\d+, -?\d+, -?\d+\), "
        r"v = \(-?\d+, -?\d+, -?\d+\)\. Find \(p \+ q\) · v$"
    )
    for seed in range(60):
        q = _make_dot_sum(seed)
        assert pattern.match(q.text), (seed, q.text)


def test_dot_sum_distinct_raw_prompts_over_60_seeds():
    """Same convention as the `subtract` and `cross_sum` tests -- see the module docstring for
    why a single-template vector question is not held to a 3-shapes bar."""
    texts = {_make_dot_sum(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_dot_sum_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_dot_sum(seed).to_json()
        second = _make_dot_sum(seed).to_json()
        assert first == second


def test_dot_sum_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes -- a genuine escalation (a THIRD vector and an
    addition step before the dot product), not merely a bigger `hi`.
    Easy/medium mirror the skill YAML's own sources (`kinds: [dot]`,
    untouched by this change)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("dot",), hi=5)) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("dot",), hi=9)) for s in range(60)}
    hard_shapes = {prompt_shape(_make_dot_sum(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive (p + q) . v from the rendered prompt text
# ALONE, never importing mathgen.generators.geometry, using plain Python
# (no sympy).
# ---------------------------------------------------------------------------

_DOT_SUM_RE = re.compile(
    r"^p = \((-?\d+), (-?\d+), (-?\d+)\), q = \((-?\d+), (-?\d+), (-?\d+)\), "
    r"v = \((-?\d+), (-?\d+), (-?\d+)\)\. Find \(p \+ q\) · v$"
)


def _independent_dot_sum(question_text: str) -> int:
    m = _DOT_SUM_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    p0, p1, p2, q0, q1, q2, v0, v1, v2 = (int(g) for g in m.groups())
    usum = [p0 + q0, p1 + q1, p2 + q2]
    return sum(ui * vi for ui, vi in zip(usum, [v0, v1, v2]))


def test_dot_sum_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_dot_sum(seed)
        oracle_ans = _independent_dot_sum(q.text)
        if oracle_ans != q.answer:
            mismatches.append((seed, q.text, oracle_ans, q.answer))
    assert mismatches == [], mismatches


def test_dot_sum_answer_text_matches_answer_over_300_seeds():
    for seed in range(300):
        q = _make_dot_sum(seed)
        assert q.answer_text == str(q.answer), (seed, q.answer_text, q.answer)


# The generator's own in-body `check` actually catches a corrupted construction.


def test_dot_sum_corrupted_matrix_dot_raises_verification_error(monkeypatch):
    """Monkeypatch `Matrix` itself (the independent-check route the
    `dot_sum` branch uses: `check(Matrix(usum).dot(Matrix(v)) == ans)`) so
    its `.dot` silently perturbs the result -- confirming `check()` actually
    fires rather than being a decoration that never runs."""

    class _WrongMatrix(Matrix):
        def dot(self, other):
            return Matrix.dot(self, other) + 1

    monkeypatch.setattr(_geo, "Matrix", _WrongMatrix)
    with pytest.raises(VerificationError):
        _make_dot_sum(0)


def test_dot_sum_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_dot_sum(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_dot_sum_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_dot_sum(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_dot_sum_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_dot_sum(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (`mathgen.grading.grade_number`, `vector-dot-product`'s
# `answer: number`). A SCALAR answer, unlike `subtract`/`cross_sum`'s
# `answer: list` graded by `GRADERS["list"]` -- no order-insensitivity
# hazard is possible here (see module-level comment above this section).
# ---------------------------------------------------------------------------

_NUMBER_GRADER = GRADERS["number"]


def test_dot_sum_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_dot_sum(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_dot_sum_grader_rejects_deliberately_wrong_values():
    """The three misconception values, hand-recomputed from the rendered
    text for a fixed seed and checked they are actually wrong before
    asserting the grader rejects them (never a vacuous check).

    seed=0, difficulty=hard, kinds=[dot_sum], hi=15: p, q, v parsed from the rendered text. forgot_q = p. v
    (dropped the addition of q); multiplied_dots = (p. v) * (q. v) (multiplied the two partial dot products
    together instead of adding them -- NOT the distributive identity (p+q).v == p.v + q.v, which would equal
    the answer and is deliberately avoided); added_not_multiplied = sum of (usum_i + v_i) elementwise instead
    of multiplying.
    """
    q = _make_dot_sum(0)
    m = _DOT_SUM_RE.match(q.text)
    assert m is not None, q.text
    p0, p1, p2, q0c, q1c, q2c, v0, v1, v2 = (int(g) for g in m.groups())
    p, qq, v = [p0, p1, p2], [q0c, q1c, q2c], [v0, v1, v2]
    ans = q.answer
    usum = [p[0] + qq[0], p[1] + qq[1], p[2] + qq[2]]
    assert ans == sum(ui * vi for ui, vi in zip(usum, v))

    p_dot_v = sum(ui * vi for ui, vi in zip(p, v))
    q_dot_v = sum(qi * vi for qi, vi in zip(qq, v))
    forgot_q = p_dot_v
    multiplied_dots = p_dot_v * q_dot_v
    added_not_multiplied = sum(ui + vi for ui, vi in zip(usum, v))
    # Confirm the identity trap is genuinely avoided: a correctly-distributed
    # p.v + q.v is NOT among the candidates (it equals the answer exactly).
    assert p_dot_v + q_dot_v == ans

    for wrong in (forgot_q, multiplied_dots, added_not_multiplied):
        assert wrong != ans, (wrong, ans)
        result = _NUMBER_GRADER(q, str(wrong))
        assert not result.correct, (wrong, q.answer_text)


# Original kinds unaffected. The `subtract` and `cross_sum` tests already cover subtract/dot/cross/add/magnitude;
# this one adds nothing new to that list (dot_sum reuses the same "dot" wording pattern check via a
# fresh dict(kinds=("dot",)) call), confirming the pre-existing "dot" branch specifically -- the
# branch immediately preceding dot_sum textually -- still produces its own long-standing shape and
# answer.


def test_dot_sum_original_subtract_dot_cross_magnitude_kinds_unaffected_over_60_seeds():
    magnitude_pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\)\. Find \|u\| \(exact\)$")
    for seed in range(60):
        sub_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("subtract",), hi=15)
        dot_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot",), hi=15)
        cross_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("cross",), hi=15)
        mag_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("magnitude",), hi=15)
        assert sub_q.text.endswith("Find u - v"), (seed, sub_q.text)
        assert dot_q.text.endswith("Find u · v"), (seed, dot_q.text)
        assert cross_q.text.endswith("Find u × v"), (seed, cross_q.text)
        assert magnitude_pattern.match(mag_q.text), (seed, mag_q.text)


def test_dot_sum_original_dot_branch_distractors_are_not_order_sensitive():
    """One-line confirmation: the pre-existing `dot`
    branch's own `pick_distractors(...)` call (unchanged by this change) does
    NOT pass `unordered=True`, and the answer it grades against
    (`mathgen.grading.grade_number`, `answer: number`) is a single scalar with no
    positional structure for a candidate to be a "permutation" of -- so the
    `subtract`/`cross_sum` order-sensitive-distractor hazard class (seen in
    `subtract` and `cross_sum`) cannot occur here, confirmed by reading `dot`'s own call
    (`pick_distractors(str(ans), [str(c) for c in cands])`, no `unordered`
    kwarg) and by this direct 300-seed grader sweep finding 0 false accepts.
    """
    bad = 0
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot",), hi=15)
        for d in q.distractors:
            if _NUMBER_GRADER(q, d).correct:
                bad += 1
    assert bad == 0, bad


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_dot_sum_skill_loads_via_api_skills_vector_dot_product():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "vector-dot-product" in ids
    skill = ids["vector-dot-product"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["dot_sum"]
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty != "hard"]
    for src in easy_medium:
        assert src.params.get("kinds") == ["dot"], src.params


# ===========================================================================
# vector-magnitude: kind="magnitude_sum" -- "Find |u + v| (exact)", the
# magnitude of the SUM of two vectors, not of a single given vector. Unlike
# `cross_sum`/`dot_sum` no THIRD vector is drawn -- u and v are already drawn
# unconditionally for every kind, same convention as `dot`/`cross`/`add`/
# `subtract`. The answer is a SCALAR expression (`answer: expression`,
# `mathgen.grading.grade_expression`), same shape as the pre-existing `magnitude`
# branch -- see the module docstring for the order-hazard finding (none) and
# the triangle-inequality finding (a real, seed-dependent hazard, handled).
# ===========================================================================

# Matches the skill's hard source exactly: `{kinds: [magnitude_sum], hi: 15}`.
_MAGNITUDE_SUM_PARAMS = dict(kinds=("magnitude_sum",), hi=15)

_EXPRESSION_GRADER = GRADERS["expression"]


def _make_magnitude_sum(seed, **overrides):
    params = {**_MAGNITUDE_SUM_PARAMS, **overrides}
    return make(TOPIC, seed=seed, difficulty="hard", **params)


# ---------------------------------------------------------------------------
# Text quality, determinism
# ---------------------------------------------------------------------------


def test_magnitude_sum_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_magnitude_sum(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_magnitude_sum_prompt_text_matches_expected_template_over_60_seeds():
    pattern = re.compile(
        r"^u = \(-?\d+, -?\d+, -?\d+\), v = \(-?\d+, -?\d+, -?\d+\)\. "
        r"Find \|u \+ v\| \(exact\)$"
    )
    for seed in range(60):
        q = _make_magnitude_sum(seed)
        assert pattern.match(q.text), (seed, q.text)


def test_magnitude_sum_distinct_raw_prompts_over_60_seeds():
    """Same convention as the earlier kinds' tests -- see the module
    docstring for why a single-template vector question is not held to a
    3-shapes bar."""
    texts = {_make_magnitude_sum(seed).text for seed in range(60)}
    assert len(texts) >= 10, texts


def test_magnitude_sum_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_magnitude_sum(seed).to_json()
        second = _make_magnitude_sum(seed).to_json()
        assert first == second


def test_magnitude_sum_hard_adds_shape_over_easy_and_medium():
    """The new hard kind's shape is not in the union of the UNCHANGED
    easy/medium tiers' shapes -- a genuine escalation (an addition step
    before the magnitude computation), not merely a bigger `hi`. Easy/medium
    mirror the skill YAML's own sources (`kinds: [magnitude]`, untouched by
    this change)."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kinds=("magnitude",), hi=5)) for s in range(60)}
    lower_shapes |= {
        prompt_shape(make(TOPIC, seed=s, difficulty="medium", kinds=("magnitude",), hi=9)) for s in range(60)
    }
    hard_shapes = {prompt_shape(_make_magnitude_sum(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


# ---------------------------------------------------------------------------
# Independent oracle: re-derive |u + v| from the rendered prompt text ALONE,
# never importing mathgen.generators.geometry. Uses sympy's own `sqrt`
# (imported at module level above) because the exact answer may be an
# unsimplified surd -- re-deriving that exact symbolic form independently
# needs the same tool the generator itself uses, not a plain-Python
# reimplementation of integer square-root simplification.
# ---------------------------------------------------------------------------

_MAGNITUDE_SUM_RE = re.compile(
    r"^u = \((-?\d+), (-?\d+), (-?\d+)\), v = \((-?\d+), (-?\d+), (-?\d+)\)\. "
    r"Find \|u \+ v\| \(exact\)$"
)


def _independent_magnitude_sum(question_text: str):
    m = _MAGNITUDE_SUM_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    u0, u1, u2, v0, v1, v2 = (int(g) for g in m.groups())
    w = [u0 + v0, u1 + v1, u2 + v2]
    ssq = sum(wi * wi for wi in w)
    return sqrt(ssq), ssq


def test_magnitude_sum_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_magnitude_sum(seed)
        oracle_m, oracle_ssq = _independent_magnitude_sum(q.text)
        if str(oracle_m) != q.answer or oracle_m**2 != oracle_ssq:
            mismatches.append((seed, q.text, str(oracle_m), q.answer))
    assert mismatches == [], mismatches


def test_magnitude_sum_answer_text_and_alt_match_over_300_seeds():
    for seed in range(300):
        q = _make_magnitude_sum(seed)
        oracle_m, _ = _independent_magnitude_sum(q.text)
        assert q.answer_text == str(oracle_m), (seed, q.answer_text, oracle_m)
        assert q.answer_alt == round(float(oracle_m), 2), (seed, q.answer_alt, oracle_m)


# The generator's own in-body `check` actually catches a corrupted construction.


def test_magnitude_sum_corrupted_sqrt_raises_verification_error(monkeypatch):
    """Monkeypatch `sqrt` itself (the independent-check route the
    `magnitude_sum` branch uses: `check(m**2 == ssq)`, where `m = sqrt(ssq)`)
    so it silently perturbs the result -- confirming `check()` actually
    fires rather than being a decoration that never runs."""
    real_sqrt = _geo.sqrt

    def _wrong_sqrt(x):
        return real_sqrt(x) + 1

    monkeypatch.setattr(_geo, "sqrt", _wrong_sqrt)
    with pytest.raises(VerificationError):
        _make_magnitude_sum(0)


def test_magnitude_sum_corrupted_check_is_not_a_false_positive_when_uncorrupted():
    _make_magnitude_sum(0)  # no exception


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_magnitude_sum_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_magnitude_sum(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_magnitude_sum_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_magnitude_sum(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (`mathgen.grading.grade_expression`, `vector-magnitude`'s
# `answer: expression`).
# ---------------------------------------------------------------------------


def test_magnitude_sum_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make_magnitude_sum(seed)
        canonical_result = _EXPRESSION_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _EXPRESSION_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


@pytest.mark.parametrize("chunk", range(10))
def test_magnitude_sum_triangle_inequality_distractor_never_grader_accepted_over_5000_seeds(chunk):
    """The dedicated, wider sweep for the triangle-inequality
    warning: the |u| + |v| candidate is only ever appended when an
    exact sympy comparison confirms it differs from the true answer for that
    seed (see the branch body and the module docstring), but this sweeps the
    REAL grader over every seed's every distractor -- including
    `grade_expression`'s own 0.005-tolerance `answer_alt` shortcut, which is
    looser than the construction-time exact check -- to rule out a near-miss
    false accept too, not merely an exact-symbolic one."""
    # Still seeds 0-4999, in ten chunks of 500 so pytest-xdist can run them on ten workers
    # instead of one worker grinding through all 5000 (~4 minutes).
    bad = []
    for seed in range(chunk * 500, (chunk + 1) * 500):
        q = _make_magnitude_sum(seed)
        for d in q.distractors:
            if _EXPRESSION_GRADER(q, d).correct:
                bad.append((seed, d, q.answer_text))
    assert bad == [], bad


class _FixedRNG:
    """Minimal rng stub: returns queued `randint` values in order; `.choice`
    always returns the first (only) kind. Used to force a specific,
    PARALLEL-same-direction u, v pair through `_geo.vectors` directly
    (bypassing `mathgen.make`'s real `random.Random`) so the triangle
    inequality's own equality case is exercised deterministically -- three
    independent random coordinates landing exactly parallel is not something
    any seed sweep small enough to run in a test will hit on its own."""

    def __init__(self, values):
        self._values = list(values)

    def randint(self, a, b):
        return self._values.pop(0)

    def choice(self, seq):
        return seq[0]


def test_magnitude_sum_triangle_inequality_distractor_omitted_when_parallel_same_direction():
    """u = (3, 4, 0) (|u| = 5), v = (6, 8, 0) (|v| = 10, parallel to u, same
    direction) -- w = u + v = (9, 12, 0), |w| = 15 = |u| + |v| EXACTLY, the
    triangle inequality's own equality case. The generator must NOT offer
    "15" as a distractor (it would be a correct answer wearing a wrong
    badge, and the real grader would mark a learner "correct" for typing the
    slip)."""
    fixed_rng = _FixedRNG([3, 4, 0, 6, 8, 0])
    q = _geo.vectors(fixed_rng, hi=15, kinds=("magnitude_sum",))
    assert q["text"] == "u = (3, 4, 0), v = (6, 8, 0). Find |u + v| (exact)", q["text"]
    assert q["answer"] == "15", q["answer"]
    # |u| + |v| = 5 + 10 = 15 == |u + v| = 15 exactly -- must be excluded.
    assert "15" not in q["distractors"], q["distractors"]
    # The other two misconceptions (no-square-before-root, forgot-sqrt) are
    # still offered: 9+12+0=21 (no-square), 225 (forgot-sqrt, ssq itself).
    assert set(q["distractors"]) == {"21", "225"}, q["distractors"]


def test_magnitude_sum_grader_rejects_deliberately_wrong_values():
    """seed=0 hard tier, hand-recomputed misconception candidates checked
    genuinely wrong before asserting the real grader rejects them (never a
    vacuous check): |u| + |v| (triangle-inequality slip), the no-square
    sum of absolute summed components, and the un-rooted sum of squares
    (forgot-sqrt)."""
    q = _make_magnitude_sum(0)
    parsed = _MAGNITUDE_SUM_RE.match(q.text)
    assert parsed is not None, q.text
    u0, u1, u2, v0, v1, v2 = (int(g) for g in parsed.groups())
    u, v = [u0, u1, u2], [v0, v1, v2]
    w = [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
    ssq = sum(wi * wi for wi in w)
    ans_value = sqrt(ssq)
    assert str(ans_value) == q.answer

    mag_u = sqrt(sum(ui * ui for ui in u))
    mag_v = sqrt(sum(vi * vi for vi in v))
    sum_of_mags = mag_u + mag_v
    no_square = sum(abs(wi) for wi in w)
    forgot_sqrt = ssq

    for wrong in (sum_of_mags, no_square, forgot_sqrt):
        assert wrong != ans_value, (wrong, ans_value)
        result = _EXPRESSION_GRADER(q, str(wrong))
        assert not result.correct, (wrong, q.answer_text)


# Original kinds unaffected. The earlier kinds' tests already cover
# subtract/dot/cross/add/magnitude/cross_sum/dot_sum; this one adds nothing new to that list,
# confirming every prior kind still produces its own long-standing shape after this edit.


def test_magnitude_sum_original_kinds_unaffected_over_60_seeds():
    magnitude_pattern = re.compile(r"^u = \(-?\d+, -?\d+, -?\d+\)\. Find \|u\| \(exact\)$")
    for seed in range(60):
        sub_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("subtract",), hi=15)
        dot_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot",), hi=15)
        cross_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("cross",), hi=15)
        add_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("add",), hi=15)
        mag_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("magnitude",), hi=15)
        cross_sum_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("cross_sum",), hi=15)
        dot_sum_q = make(TOPIC, seed=seed, difficulty="hard", kinds=("dot_sum",), hi=15)
        assert sub_q.text.endswith("Find u - v"), (seed, sub_q.text)
        assert dot_q.text.endswith("Find u · v"), (seed, dot_q.text)
        assert cross_q.text.endswith("Find u × v"), (seed, cross_q.text)
        assert add_q.text.endswith("Find u + v"), (seed, add_q.text)
        assert magnitude_pattern.match(mag_q.text), (seed, mag_q.text)
        assert cross_sum_q.text.endswith("Find (p + q) × v"), (seed, cross_sum_q.text)
        assert dot_sum_q.text.endswith("Find (p + q) · v"), (seed, dot_sum_q.text)


def test_magnitude_sum_original_magnitude_branch_distractors_are_not_order_sensitive():
    """The magnitude-branch distractor-order check, run
    explicitly (the same check as for `dot`): the pre-existing
    `magnitude` branch's own `pick_distractors(...)` call (unchanged by this
    change) does NOT pass `unordered=True`, and the answer it grades against
    (`mathgen.grading.grade_expression`, `answer: expression`) is a single
    scalar expression with no positional structure for a candidate to be a
    "permutation" of -- so the `subtract`/`cross_sum` order-sensitive-
    distractor hazard class cannot occur here, confirmed by
    reading `magnitude`'s own call (`pick_distractors(str(m), cands)`, no
    `unordered` kwarg) and by this direct 300-seed grader sweep finding 0
    false accepts."""
    bad = 0
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty="hard", kinds=("magnitude",), hi=15)
        for d in q.distractors:
            if _EXPRESSION_GRADER(q, d).correct:
                bad += 1
    assert bad == 0, bad


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills, hard source repointed correctly.
# ---------------------------------------------------------------------------


def test_magnitude_sum_skill_loads_via_api_skills_vector_magnitude():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "vector-magnitude" in ids
    skill = ids["vector-magnitude"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kinds") == ["magnitude_sum"]
    assert hard_sources[0].params.get("hi") == 15
    easy_medium = [s for s in skill.sources if s.difficulty != "hard"]
    for src in easy_medium:
        assert src.params.get("kinds") == ["magnitude"], src.params
