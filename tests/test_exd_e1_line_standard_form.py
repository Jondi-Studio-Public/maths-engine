"""A new exam source for `senior-find-the-equation-of-a-straight-line-given` (`variant="standard_form"`,
`difficulty="exam"`) -- given two points, the answer is required as `ax + by + c = 0` with integer
coefficients, `a > 0`, `gcd(|a|, |b|, |c|) == 1` (an exam-style target).

It also adds a new, additive `answer_form == "line_standard_form"` branch on
`grade_expression`. This file exercises that grading branch directly
(`mathgen.grading.GRADERS["expression"]`/`grading._line_standard_form_coeffs`), not just the generator's own
self-consistency checks.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape
from mathgen.grading import typed as grading
from mathgen.skills import load_skills

TOPIC = "find_the_equation_of_a_straight_line_given"
DIFFICULTY = "exam"
SEEDS = range(60)  # 60 seeds, matching measure_tiers.py's own sweep

grade_expression = grading.GRADERS["expression"]


def _mk(seed):
    return make(TOPIC, seed=seed, difficulty=DIFFICULTY)


def test_determinism_same_seed_is_byte_identical():
    for seed in SEEDS:
        first = _mk(seed).to_json()
        second = _mk(seed).to_json()
        assert first == second, f"seed={seed}: not deterministic"


def test_no_bad_text_patterns_over_60_seeds():
    for seed in SEEDS:
        q = _mk(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects}"


def test_prompt_variety_exam_tier():
    shapes = {prompt_shape(_mk(seed)) for seed in SEEDS}
    assert len(shapes) >= 3, shapes


def test_exam_tier_adds_a_shape_no_lower_tier_has():
    """Mirrors `scripts/measure_tiers.py`'s own escalation check, proved
    independently here rather than trusted by eye."""
    easy_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in SEEDS}
    medium_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="medium")) for s in SEEDS}
    hard_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="hard")) for s in SEEDS}
    exam_shapes = {prompt_shape(_mk(s)) for s in SEEDS}
    new_shapes = exam_shapes - easy_shapes - medium_shapes - hard_shapes
    assert new_shapes, "exam tier adds no shape over easy/medium/hard"


def test_at_least_20_distinct_rendered_questions_over_60_seeds():
    texts = {_mk(seed).text for seed in SEEDS}
    assert len(texts) >= 20, len(texts)


# ---------------------------------------------------------------------------
# Answer shape.


def test_answer_form_is_line_standard_form_every_seed():
    for seed in SEEDS:
        assert _mk(seed).answer_form == "line_standard_form"


_ANSWER_RE = re.compile(r"^(-?\d*)x ([+-]) (\d*)y(?: ([+-]) (\d+))? = 0$")


def _coeffs_from_answer_text(text: str) -> tuple[int, int, int]:
    """Independent re-derivation of (a, b, c) from `answer_text` ALONE, via
    a hand-written regex -- never `grading._line_standard_form_coeffs`
    itself, so this is a genuinely separate route from the grader under
    test, and never the generator's own a/b/c variables."""
    m = _ANSWER_RE.match(text)
    assert m, f"unrecognised answer_text shape: {text!r}"
    a_txt, y_sign, b_txt, c_sign, c_txt = m.groups()
    a = int(a_txt) if a_txt else 1
    b_mag = int(b_txt) if b_txt else 1
    b = b_mag if y_sign == "+" else -b_mag
    if c_sign is None:
        c = 0
    else:
        c = int(c_txt) if c_sign == "+" else -int(c_txt)
    return a, b, c


def test_answer_text_matches_ax_by_c_shape_and_is_valid_every_seed():
    """`answer_text` always parses under `_ANSWER_RE` (a > 0 rendered
    without a redundant leading "+"), and the independently re-derived
    (a, b, c) is always integer, a > 0, gcd 1 -- proved by construction, not
    trusted from the generator's own internal variables."""
    for seed in SEEDS:
        q = _mk(seed)
        a, b, c = _coeffs_from_answer_text(q.answer_text)
        assert a > 0, (seed, q.answer_text)
        assert math.gcd(math.gcd(abs(a), abs(b)), abs(c)) == 1, (seed, q.answer_text)


def test_answer_and_every_distractor_share_the_ax_by_c_shape():
    """No option is identifiable by shape alone -- every distractor also parses under the same regex as the
    answer."""
    for seed in SEEDS:
        q = _mk(seed)
        for option in [q.answer_text, *q.distractors]:
            _coeffs_from_answer_text(option)  # raises via assert if the shape differs


def test_two_distractors_distinct_nonempty_and_not_the_answer():
    for seed in SEEDS:
        q = _mk(seed)
        assert len(q.distractors) == 2, (seed, q.distractors)
        assert len(set(q.distractors)) == 2, (seed, q.distractors)
        assert q.answer_text not in q.distractors, (seed, q.distractors)
        assert all(d for d in q.distractors), (seed, q.distractors)


# ---------------------------------------------------------------------------
# The real grader, called directly -- not the generator's own check().


def test_real_grader_accepts_canonical_and_rejects_every_distractor():
    for seed in SEEDS:
        q = _mk(seed)
        accepted = grade_expression(q, q.answer_text)
        assert accepted.correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            rejected = grade_expression(q, d)
            assert not rejected.correct, (seed, q.text, "distractor wrongly accepted:", d)


def test_a_deliberately_wrong_line_is_rejected():
    """A genuinely different line (different (a, b, c) up to no scalar --
    shift the constant term by 1) must be rejected, independent of whichever
    misconception distractors the generator happens to draw for a seed."""
    for seed in SEEDS:
        q = _mk(seed)
        a, b, c = _coeffs_from_answer_text(q.answer_text)
        wrong = f"{a}x + {'+' if b >= 0 else '-'} {abs(b)}y + {c + 1} = 0"
        result = grade_expression(q, wrong)
        assert not result.correct, (seed, q.text, q.answer_text, wrong)


def test_wrong_sign_a_negative_is_rejected():
    """`-2x - 3y + 6 = 0` style: same line, but a < 0 -- the exact opposite
    of the required normalisation."""
    for seed in SEEDS:
        q = _mk(seed)
        a, b, c = _coeffs_from_answer_text(q.answer_text)
        wrong_sign = f"{-a}x {'+' if -b >= 0 else '-'} {abs(b)}y {'+' if -c >= 0 else '-'} {abs(c)} = 0"
        result = grade_expression(q, wrong_sign)
        assert not result.correct, (seed, q.text, q.answer_text, wrong_sign)


def test_not_reduced_double_every_coefficient_is_rejected():
    """`4x + 6y - 12 = 0` style: same line, doubled coefficients, gcd 2 --
    value-equal, not fully reduced."""
    for seed in SEEDS:
        q = _mk(seed)
        a, b, c = _coeffs_from_answer_text(q.answer_text)
        doubled = f"{2 * a}x {'+' if 2 * b >= 0 else '-'} {abs(2 * b)}y {'+' if 2 * c >= 0 else '-'} {abs(2 * c)} = 0"
        result = grade_expression(q, doubled)
        assert not result.correct, (seed, q.text, q.answer_text, doubled)


def test_tripled_and_negated_mis_normalisation_is_rejected():
    """A third, genuinely different mis-normalisation: same line, scaled by
    -3 (still value-equal, still not the required a > 0 / gcd-1 form)."""
    for seed in SEEDS:
        q = _mk(seed)
        a, b, c = _coeffs_from_answer_text(q.answer_text)
        sa, sb, sc = -3 * a, -3 * b, -3 * c
        scaled = f"{sa}x {'+' if sb >= 0 else '-'} {abs(sb)}y {'+' if sc >= 0 else '-'} {abs(sc)} = 0"
        result = grade_expression(q, scaled)
        assert not result.correct, (seed, q.text, q.answer_text, scaled)


def test_missing_equals_zero_is_rejected():
    """The bare expression, with no "= 0" at all, is not an equation and
    must be rejected -- `_parse_expression` would happily parse it as a
    plain expression, so the form check must catch it, not rely on that
    failing to parse."""
    for seed in SEEDS:
        q = _mk(seed)
        bare = q.answer_text.split("=")[0].strip()
        result = grade_expression(q, bare)
        assert not result.correct, (seed, q.text, q.answer_text, bare)


# ---------------------------------------------------------------------------
# `_line_standard_form_coeffs` / `grade_expression` unit-level checks
# (grading internals, not just end-to-end via a generated question).


def test_line_standard_form_coeffs_accepts_canonical_shape():
    assert grading._line_standard_form_coeffs("2x + 3y - 6 = 0") == (2, 3, -6)


def test_line_standard_form_coeffs_rejects_wrong_sign():
    assert grading._line_standard_form_coeffs("-2x - 3y + 6 = 0") is None


def test_line_standard_form_coeffs_rejects_not_reduced():
    assert grading._line_standard_form_coeffs("4x + 6y - 12 = 0") is None


def test_line_standard_form_coeffs_rejects_non_integer_coefficient():
    assert grading._line_standard_form_coeffs("2.5x + 3y - 6 = 0") is None
    assert grading._line_standard_form_coeffs("x/2 + 3y - 6 = 0") is None


def test_line_standard_form_coeffs_rejects_extra_symbol():
    assert grading._line_standard_form_coeffs("2x + 3y + z - 6 = 0") is None


def test_line_standard_form_coeffs_rejects_quadratic_term():
    assert grading._line_standard_form_coeffs("x**2 + 3y - 6 = 0") is None


def test_line_standard_form_coeffs_rejects_non_equation_text():
    assert grading._line_standard_form_coeffs("2x + 3y - 6") is None
    assert grading._line_standard_form_coeffs("2x + 3y - 6 = 0 = 0") is None


def test_line_standard_form_coeffs_handles_a_equals_1_and_b_equals_1():
    assert grading._line_standard_form_coeffs("x + y - 3 = 0") == (1, 1, -3)
    assert grading._line_standard_form_coeffs("x - y = 0") == (1, -1, 0)


def test_grade_expression_same_line_different_valid_looking_scale_rejected():
    """Two DIFFERENT normalised-and-valid-looking lines (both individually
    pass `_line_standard_form_coeffs`, i.e. neither is malformed) must still
    be told apart by an exact tuple comparison to the canonical's own
    (a, b, c) -- not merely "is this text in the required form"."""
    q = _mk(0)
    a, b, c = _coeffs_from_answer_text(q.answer_text)
    other_valid_line = f"{a}x {'+' if b >= 0 else '-'} {abs(b)}y {'+' if (c + 1) >= 0 else '-'} {abs(c + 1)} = 0"
    # sanity: the "other" text is itself well-formed (a > 0, gcd 1) --
    # otherwise this test would trivially pass via the normalisation check,
    # not the same-line check it claims to exercise.
    assert grading._line_standard_form_coeffs(other_valid_line) is not None
    result = grade_expression(q, other_valid_line)
    assert not result.correct


# ---------------------------------------------------------------------------
# The independent, in-generator check firing on a corrupted value.

import mathgen.generators.p6e_linear_construction_build as _p6e  # noqa: E402


def test_corrupted_standard_form_derivation_raises_verification_error(monkeypatch):
    """Corrupts `_standard_form_from_two_points`'s FIRST call (the branch's
    own primary derivation) by shifting `c` by 1 -- the in-body substitution
    check (both original points must satisfy `a*x + b*y + c == 0`) must then
    raise."""
    original = _p6e._standard_form_from_two_points
    call_count = {"n": 0}

    def _corrupted(x1, y1, x2, y2):
        a, b, c = original(x1, y1, x2, y2)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return a, b, c + 1  # corrupt only the branch's own primary derivation
        return a, b, c

    monkeypatch.setattr(_p6e, "_standard_form_from_two_points", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty=DIFFICULTY)


def test_corrupted_derivation_is_not_a_false_positive_when_uncorrupted():
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous."""
    make(TOPIC, seed=1, difficulty=DIFFICULTY)  # must not raise


# ---------------------------------------------------------------------------
# Existing variants untouched.


def test_easy_medium_hard_variants_unaffected():
    """The pre-existing three variants are provably unchanged: same
    `answer_form` (`None`), same answer-text pattern (`"y = ..."`, never the
    new `"ax + by + c = 0"` shape)."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in (0, 1, 2, 7, 42):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_form is None, (difficulty, seed, q.answer_form)
            assert q.answer_text.startswith("y = "), (difficulty, seed, q.answer_text)


# ---------------------------------------------------------------------------
# The skill loads with the expected shape.


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-find-the-equation-of-a-straight-line-given")
    assert skill.domain == "linear-relations"
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert set(skill.prerequisites) == {"rate-of-change"}
    stages_covered = set()
    exam_sources = 0
    for source in skill.sources:
        assert source.topic == TOPIC
        stages_covered.update(source.stages)
        if source.difficulty == "exam":
            exam_sources += 1
            assert source.params.get("variant") == "standard_form"
            assert source.stages == ["exam"]
    assert exam_sources == 1
    assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
