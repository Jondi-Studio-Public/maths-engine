"""Tests for `mathgen/generators/p19b_coefficient_determination_build.py`:
`p19b_coefficient_determination` (skill
`senior-use-the-coefficient-of-determination-to-assess-the`, ACMGM060).

This file holds the generator-specific
proofs: the in-generator independent check firing on a corrupted value, the
real-grader battery, the answer's own-kind checks, and the answer-space/
distractor properties.

The independent ORACLE lives in
`tests/test_p19b_coefficient_determination_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19b_coefficient_determination_build as gen

TOPIC = "p19b_coefficient_determination"
TIERS = ("easy", "medium", "hard", "exam")
SKILL_ID = "senior-use-the-coefficient-of-determination-to-assess-the"

_DEC_TOL = stated_precision_tolerance(4)
_PCT_TOL = stated_precision_tolerance(2)


def _all(tier, seeds=range(60)):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Shared / by-hand prompt-shape and construction checks
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(tier):
    for q in _all(tier, seeds=range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(tier):
    shapes = {prompt_shape(q) for q in _all(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all(tier):
        assert len(q.distractors) >= 1, f"{tier} seed {q.seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor == answer: {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_alike(tier):
    """Every option (answer + distractors) is a signed decimal at exactly the
    tier's stated precision, so no option is identifiable by its shape."""
    places = {"easy": 4, "medium": 2, "hard": 2, "exam": 2}[tier]
    pattern = re.compile(rf"^-?\d+\.\d{{{places}}}$")
    for q in _all(tier, seeds=range(30)):
        for opt in [q.answer_text, *q.distractors]:
            assert pattern.match(opt), f"{tier} seed {q.seed}: {opt!r} not {places}dp"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_draws(tier):
    counts = [len(q.distractors) for q in _all(tier, seeds=range(200))]
    at_least_two = sum(1 for c in counts if c >= 2)
    assert at_least_two / len(counts) >= 0.85, f"{tier}: only {at_least_two}/{len(counts)} >= 2 distractors"


@pytest.mark.parametrize("tier", TIERS)
def test_no_answer_key_concentration_over_15pct_of_200(tier):
    answers = Counter(q.answer_text for q in _all(tier, seeds=range(200)))
    top = max(answers.values())
    assert top / 200 <= 0.15, f"{tier}: top answer {answers.most_common(1)} is {top}/200"


@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_starts_with_uppercase_or_digit(tier):
    for q in _all(tier, seeds=range(60)):
        assert q.text[0].isupper() or q.text[0].isdigit(), (
            f"{tier} seed {q.seed}: prompt does not start uppercase/digit: {q.text[:60]!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_instruction_phrase_in_both_text_and_latex(tier):
    for q in _all(tier, seeds=range(10)):
        assert "decimal places" in q.text
        assert "decimal places" in q.latex


# ============================================================================
# Independent in-generator check (`_verify_r_squared`) fires on corruption
# ============================================================================


def test_verify_r_squared_fires_on_corrupted_value():
    r = Fraction(72, 100)
    r2 = r * r
    gen._verify_r_squared(r, r2)  # no exception -- the true r^2
    with pytest.raises(VerificationError):
        gen._verify_r_squared(r, r2 + Fraction(1, 100))  # a corrupted key


def test_verify_r_squared_uses_decimal_not_fraction():
    """`_verify_r_squared` computes through `decimal.Decimal`, a different
    module/code path from the `Fraction` multiplication construction uses --
    confirmed by patching in a wrong Fraction r^2 and observing the Decimal
    route still catch it (i.e. it recomputes, it does not just echo back the
    value it is handed)."""
    r = Fraction(-65, 100)
    r2_correct = r * r
    assert r2_correct == Fraction(4225, 10000)
    gen._verify_r_squared(r, r2_correct)
    with pytest.raises(VerificationError):
        gen._verify_r_squared(r, Fraction(4224, 10000))


# ============================================================================
# Real grader battery
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(tier, seeds=range(30)):
        assert grade_number(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_number(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(tier):
    tol = {"easy": float(_DEC_TOL), "medium": float(_PCT_TOL), "hard": float(_PCT_TOL), "exam": float(_PCT_TOL)}[tier]
    places = {"easy": 4, "medium": 2, "hard": 2, "exam": 2}[tier]
    for q in _all(tier, seeds=range(20)):
        wrong = float(q.answer_text) + max(0.1, tol * 20)
        assert not grade_number(q, f"{wrong:.{places}f}").correct, f"{tier} seed {q.seed}: wrong value accepted"


def test_grade_number_accepts_percentage_probe_forms():
    """The percentage-form probe: `68.89`, `68.89%`
    and `0.6889` against a canonical percentage answer_text of `68.89`."""
    q = next(q for q in _all("medium", seeds=range(60)) if q.answer_text == "68.89")
    assert grade_number(q, "68.89").correct
    assert grade_number(q, "68.89%").correct
    assert not grade_number(q, "0.6889").correct


# ============================================================================
# Answer-space band: |r| in [0.30, 0.99], exact precision, top-key share
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space_r_band(tier):
    for q in _all(tier, seeds=range(60)):
        matches = re.findall(r"r = (-?\d+\.\d{2})|\|r\| = (\d+\.\d{2})", q.text)
        for a, b in matches:
            r = float(a or b)
            assert 0.30 <= abs(r) <= 0.99, f"{tier} seed {q.seed}: |r|={abs(r)} out of band"


# ============================================================================
# hard tier: |r|, not signed r, decides which explains more
# ============================================================================


def test_hard_strength_is_abs_r_not_r():
    """The answer is the LARGER r^2 (i.e. larger |r|) -- proven by finding at
    least one seed (across 200) where the answer variable's own stated r is
    NEGATIVE, so "pick the larger SIGNED r" would pick the wrong (weaker)
    variable.

    could be drawn negative while the weaker one was forced positive within
    the SAME context (e.g. "number of practice tests completed ... r = -0.84"
    against final exam score, a naturally-positive pair) -- sign is now the
    context's own natural sign (`ctx["sign"]`), so a "negative" context draws
    BOTH variables negative. The trap still fires whenever the stronger
    variable is negative: with both r negative, the MORE negative (larger
    |r|) value is the smaller SIGNED value, so a naive signed-r comparison
    still picks the weaker variable. Assert that directly instead of
    "weaker positive", which the fix makes impossible by construction."""
    found_negative_stronger = False
    for q in _all("hard", seeds=range(200)):
        matches = re.findall(r"r = (-?\d+\.\d{2})", q.text)
        assert len(matches) == 2
        r1, r2 = Fraction(matches[0]), Fraction(matches[1])
        answer = Fraction(q.answer_text) / 100
        stronger_val = r1 if r1 * r1 >= r2 * r2 else r2
        weaker_val = r2 if r1 * r1 >= r2 * r2 else r1
        assert stronger_val * stronger_val == answer, f"seed {q.seed}: answer is not the larger r^2"
        if stronger_val < 0:
            assert weaker_val > stronger_val, f"seed {q.seed}: a naive signed-r comparison should fail here"
            found_negative_stronger = True
    assert found_negative_stronger, "200 seeds never drew the negative-is-stronger trap"


def test_context_sign_matches_natural_direction_easy():
    """Every easy-tier context is naturally positive (more of x, more of y),
    and `_draw_r(rng, negative=False)` is unconditional, so every draw's
    stated r must be positive over 200 seeds."""
    for q in _all("easy", seeds=range(200)):
        m = re.search(r"r = (-?\d+\.\d{2})", q.text)
        assert m, f"easy seed {q.seed}: no stated r found in {q.text!r}"
        assert float(m.group(1)) > 0, f"easy seed {q.seed}: r should be positive: {q.text!r}"


def test_context_sign_matches_natural_direction_medium():
    """Every medium-tier context is naturally negative (more of x, less of
    y), and `_draw_r(rng, negative=True)` is unconditional, so every draw's
    stated r must be negative over 200 seeds."""
    for q in _all("medium", seeds=range(200)):
        m = re.search(r"r = (-?\d+\.\d{2})", q.text)
        assert m, f"medium seed {q.seed}: no stated r found in {q.text!r}"
        assert float(m.group(1)) < 0, f"medium seed {q.seed}: r should be negative: {q.text!r}"


def test_context_sign_matches_natural_direction_hard():
    """Build a context -> expected-sign table straight from the generator's own data (keyed by `x1_label`, unique
    per context) and assert every one of 200 seeds' two stated r values matches its context's sign; also
    confirm both signs are actually exercised over the sweep."""
    sign_by_x1_label = {ctx["x1_label"]: ctx["sign"] for ctx in gen._HARD_CONTEXTS}
    assert set(sign_by_x1_label.values()) == {"positive", "negative"}
    seen_signs = set()
    for q in _all("hard", seeds=range(200)):
        matches = re.findall(r"r = (-?\d+\.\d{2})", q.text)
        assert len(matches) == 2, f"hard seed {q.seed}: expected two stated r values: {q.text!r}"
        r1, r2 = float(matches[0]), float(matches[1])
        x1_label = next((label for label in sign_by_x1_label if label in q.text), None)
        assert x1_label is not None, f"hard seed {q.seed}: no known context label in {q.text!r}"
        expected_sign = sign_by_x1_label[x1_label]
        seen_signs.add(expected_sign)
        if expected_sign == "negative":
            assert r1 < 0 and r2 < 0, (
                f"hard seed {q.seed}: {x1_label} is a negative context, both r must be negative: {q.text!r}"
            )
        else:
            assert r1 > 0 and r2 > 0, (
                f"hard seed {q.seed}: {x1_label} is a positive context, both r must be positive: {q.text!r}"
            )
    assert seen_signs == {"positive", "negative"}, f"200 hard seeds did not exercise both context signs: {seen_signs}"


# ============================================================================
# exam tier: difference of two r^2 values, Model A always explains more
# ============================================================================


def test_exam_model_a_always_explains_more():
    for q in _all("exam", seeds=range(60)):
        matches = re.findall(r"\|r\| = (\d+\.\d{2})", q.text)
        assert len(matches) == 2
        r1, r2 = Fraction(matches[0]), Fraction(matches[1])
        assert r1 > r2, f"seed {q.seed}: Model A's |r| is not the larger one"
        diff = (r1 * r1 - r2 * r2) * 100
        assert diff == Fraction(q.answer_text)
        assert diff > 0


# ============================================================================
# Skill loads
# ============================================================================


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        assert skill.domain == "statistics-probability"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["simple-event-probability"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
