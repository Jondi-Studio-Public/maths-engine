"""Tests for `mathgen/generators/p19e_degree_discriminant_build.py`:
`p19e_poly_degree_leading` (skill `senior-identify-the-coefficients-and-the-
degree-of-a`, ACMMM015) and `p19e_discriminant_zero_count` (skill
`senior-find-turning-points-and-zeros-of-quadratics-and`, ACMMM010).

This file holds the generator-
specific proofs a generic check cannot: the independent in-generator check firing
on a corrupted value, forced-condition sweeps, the real-grader battery, and
the answer's own-kind checks.

The independent ORACLE lives in `tests/test_p19e_degree_discriminant_oracle.py`.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p19e_degree_discriminant_build as gen

DEG_TOPIC = "p19e_poly_degree_leading"
DEG_SKILL = "senior-identify-the-coefficients-and-the-degree-of-a"
DISC_TOPIC = "p19e_discriminant_zero_count"
DISC_SKILL = "senior-find-turning-points-and-zeros-of-quadratics-and"
TIERS = ("easy", "medium", "hard", "exam")


def _all(topic, tier, seeds=range(60)):
    return [make(topic, seed=s, difficulty=tier) for s in seeds]


# ============================================================================
# Shared / by-hand prompt-shape and construction checks -- both topics
# ============================================================================


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns(topic, tier):
    for q in _all(topic, tier, seeds=range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{topic} {tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_at_least_3_shapes(topic, tier):
    shapes = {prompt_shape(q) for q in _all(topic, tier)}
    assert len(shapes) >= 3, f"{topic} {tier}: only {len(shapes)} shapes: {shapes}"


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_determinism(topic, tier):
    for seed in (0, 1, 2, 40, 59):
        first = make(topic, seed=seed, difficulty=tier).to_json()
        second = make(topic, seed=seed, difficulty=tier).to_json()
        assert first == second


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {q.text for q in _all(topic, tier)}
    assert len(texts) >= 20, f"{topic} {tier}: only {len(texts)} distinct"


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for q in _all(topic, tier):
        assert len(q.distractors) >= 1, f"{topic} {tier} seed {q.seed}: no distractors"
        for d in q.distractors:
            assert d != q.answer_text, f"{topic} {tier}: distractor == answer: {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{topic} {tier}: duplicate distractors"


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_200_draws(topic, tier):
    seeds = _all(topic, tier, seeds=range(200))
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"{topic} {tier}: only {at_least_two}/200 have >= 2 distractors"


@pytest.mark.parametrize("topic", [DEG_TOPIC, DISC_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_default_params_also_have_two_distractors_on_85pct(topic, tier):
    # `make(topic, seed=s)` with NO difficulty -- the default-params path.
    seeds = [make(topic, seed=s) for s in range(200)]
    at_least_two = sum(1 for q in seeds if len(q.distractors) >= 2)
    assert at_least_two / len(seeds) >= 0.85, f"{topic} default: only {at_least_two}/200 have >= 2 distractors"


# ============================================================================
# p19e_poly_degree_leading: answer-space concentration + properties
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_deg_top_key_pair_under_15pct_of_200_draws(tier):
    seeds = _all(DEG_TOPIC, tier, seeds=range(200))
    counts = Counter(q.answer_text for q in seeds)
    top_share = counts.most_common(1)[0][1] / len(seeds)
    assert top_share < 0.15, f"{tier}: top pair {counts.most_common(1)} share {top_share}"


@pytest.mark.parametrize("tier", TIERS)
def test_deg_answer_is_pair_of_ints(tier):
    for q in _all(DEG_TOPIC, tier, seeds=range(40)):
        assert isinstance(q.answer["degree"], int)
        assert isinstance(q.answer["coefficient"], int)
        assert q.answer["coefficient"] != 0


def _extract_medium_poly(text: str) -> str:
    for prefix, end in (
        ("P(x) = ", ", with its terms"),
        ("order of degree: ", ". What"),
        ("polynomial ", " (terms"),
    ):
        if prefix in text and end in text:
            return text.split(prefix, 1)[1].split(end, 1)[0]
    raise AssertionError(f"unrecognised medium-tier template: {text!r}")


def _first_term_degree(poly_text: str) -> int:
    first = re.split(r" (?=[+-] )", poly_text)[0]
    m = re.search(r"x\^(\d+)", first)
    if m:
        return int(m.group(1))
    return 1 if "x" in first else 0


def test_medium_first_term_is_never_the_leading_term():
    """The medium-tier construction's own DECISION: the first-written term
    must never be the highest-degree one -- the learner cannot just read
    the degree off the first term's position, unlike at the easy tier."""
    for q in _all(DEG_TOPIC, "medium", seeds=range(60)):
        poly = _extract_medium_poly(q.text)
        first_deg = _first_term_degree(poly)
        assert first_deg != q.answer["degree"], f"seed {q.seed}: first term IS the leading term in {poly!r}"


def _extract_easy_poly(text: str) -> str:
    for prefix, end in (
        ("P(x) = ", ". State"),
        ("polynomial ", ", give"),
        ("polynomial ", " is written"),
    ):
        if prefix in text and end in text:
            return text.split(prefix, 1)[1].split(end, 1)[0]
    raise AssertionError(f"unrecognised easy-tier template: {text!r}")


def test_easy_first_term_is_always_the_leading_term():
    """Contrast case: at easy, the polynomial is always written in
    descending order, so the first term IS the leading term."""
    for q in _all(DEG_TOPIC, "easy", seeds=range(30)):
        poly = _extract_easy_poly(q.text)
        first_deg = _first_term_degree(poly)
        assert first_deg == q.answer["degree"], f"seed {q.seed}: {poly!r}"


def test_hard_tier_is_a_product_of_factors():
    """>= 2 factors always, but the `linear_squared` variant may render its
    bare linear factor with no parens at all ("3x(-2x + 4)^2") -- so the
    text-level proof is >= 1 bracketed factor group, not >= 2."""
    for q in _all(DEG_TOPIC, "hard", seeds=range(60)):
        assert q.text.count("(") >= 1, q.text


def test_exam_tier_shows_a_cancelling_pair():
    """Every exam-tier draw carries two occurrences of the same power with
    opposite coefficients (the pair that cancels)."""
    term_re = re.compile(r"(-?\d*)x\^(\d+)")
    for q in _all(DEG_TOPIC, "exam", seeds=range(60)):
        powers = Counter(int(p) for _, p in term_re.findall(q.text))
        assert any(n >= 2 for n in powers.values()), f"seed {q.seed}: no repeated power in {q.text!r}"


# ============================================================================
# p19e_poly_degree_leading: independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_degree():
    coeffs = {4: 3, 2: 1, 0: -5}
    gen._verify_additive_poly(coeffs, 4, 3)  # no exception when correct
    with pytest.raises(VerificationError):
        gen._verify_additive_poly(coeffs, 3, 3)
    with pytest.raises(VerificationError):
        gen._verify_additive_poly(coeffs, 4, 4)


# ============================================================================
# p19e_poly_degree_leading: real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_deg_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(DEG_TOPIC, tier, seeds=range(20)):
        assert grade_pair(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_pair(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_deg_real_grader_accepts_reordered_labels_and_alt_forms(tier):
    for q in _all(DEG_TOPIC, tier, seeds=range(10)):
        d, c = q.answer["degree"], q.answer["coefficient"]
        reordered = f"coefficient = {c}, degree = {d}"
        assert grade_pair(q, reordered).correct, f"{tier} seed {q.seed}: reordered rejected"
        bare = f"{d}, {c}"
        assert grade_pair(q, bare).correct, f"{tier} seed {q.seed}: bare positional rejected"


@pytest.mark.parametrize("tier", TIERS)
def test_deg_real_grader_rejects_wrong_and_values_swapped(tier):
    for q in _all(DEG_TOPIC, tier, seeds=range(10)):
        d, c = q.answer["degree"], q.answer["coefficient"]
        wrong = f"degree = {d + 100}, coefficient = {c}"
        assert not grade_pair(q, wrong).correct, f"{tier} seed {q.seed}: wrong degree wrongly accepted"
        swapped = f"degree = {c}, coefficient = {d}"
        if c != d:
            assert not grade_pair(q, swapped).correct, f"{tier} seed {q.seed}: swapped values wrongly accepted"
        word = "two"
        assert not grade_pair(q, word).correct


# ============================================================================
# p19e_discriminant_zero_count: answer-space concentration + properties
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_disc_each_zero_count_between_15_and_55pct_of_200_draws(tier):
    seeds = _all(DISC_TOPIC, tier, seeds=range(200))
    counts = Counter(q.answer["zeros"] for q in seeds)
    for cls in (0, 1, 2):
        share = counts.get(cls, 0) / len(seeds)
        assert 0.15 <= share <= 0.55, f"{tier}: zero-count {cls} share {share} ({counts})"


@pytest.mark.parametrize("tier", TIERS)
def test_disc_zero_count_is_consistent_with_discriminant_sign(tier):
    for q in _all(DISC_TOPIC, tier, seeds=range(60)):
        d, z = q.answer["D"], q.answer["zeros"]
        if d > 0:
            assert z == 2
        elif d == 0:
            assert z == 1
        else:
            assert z == 0


def _extract_disc_poly(text: str) -> str:
    """The `ax^2 + bx + c` expression as written, minus the trailing ' = 0'."""
    for marker in ("quadratic ", "f(x) = ", "y = "):
        if marker in text:
            after = text.split(marker, 1)[1]
            for end in (" = 0", "."):
                if end in after:
                    return after.split(end, 1)[0]
    raise AssertionError(f"could not extract polynomial from {text!r}")


def test_disc_medium_tier_includes_negative_a_or_c_sign_traps():
    """`a` or `c` is negative often enough that -4ac is positive by a
    genuine double-negative, not merely by chance on a handful of draws."""
    hits = 0
    for q in _all(DISC_TOPIC, "medium", seeds=range(200)):
        poly = _extract_disc_poly(q.text)
        leading_negative = poly.lstrip().startswith("-")
        constant_negative = bool(re.search(r"- \d+$", poly))
        if leading_negative or constant_negative:
            hits += 1
    assert hits / 200 >= 0.2, f"only {hits}/200 medium draws show a sign trap"


def test_disc_hard_tier_has_a_missing_term_when_zero_count_is_not_1():
    """Every hard-tier draw reconstructs a, b, c from `working[0]`
    ("a = .., b = .., c = .."). Draws with zero-count 0 or 2 have exactly
    one of b, c missing (the DECISION: identify a, b, c before
    substituting). Zero-count 1 draws are the perfect-square fix
    and are a full trinomial instead -- see
    `test_disc_hard_tier_count1_is_never_the_degenerate_ax2_eq_0`."""
    checked = 0
    for q in _all(DISC_TOPIC, "hard", seeds=range(60)):
        m = re.search(r"a = (-?\d+), b = (-?\d+), c = (-?\d+)", q.working[0])
        assert m, q.working
        b, c = int(m.group(2)), int(m.group(3))
        if q.answer["zeros"] == 1:
            assert b != 0 and c != 0, f"seed {q.seed}: count-1 draw is degenerate ({q.working[0]!r})"
        else:
            assert b == 0 or c == 0, f"seed {q.seed}: neither b nor c is missing ({q.working[0]!r})"
        checked += 1
    assert checked == 60


def test_disc_hard_tier_count1_is_never_the_degenerate_ax2_eq_0():
    """At 200 seeds, the OLD generator's count-1 hard draws were ALWAYS `a*x^2 = 0` (b=c=0 both missing) --
    58/200 draws overall. After the fix, `ax^2=0` must never appear on any hard draw."""
    for q in _all(DISC_TOPIC, "hard", seeds=range(200)):
        m = re.search(r"a = (-?\d+), b = (-?\d+), c = (-?\d+)", q.working[0])
        assert m, q.working
        b, c = int(m.group(2)), int(m.group(3))
        assert not (b == 0 and c == 0), f"seed {q.seed}: degenerate ax^2=0 ({q.working[0]!r})"


def test_disc_no_distractor_has_a_zero_count_outside_0_1_2():
    """The old `(count, disc)` "swapped (zeros, D)" fallback put the raw discriminant into the ZEROS slot,
    producing nonsense pairs like "D=0, zeros=-48" or "D=1, zeros=0" that no real misconception produces --
    measured 137/200 hard, 16-18/200 easy/medium, 8/200 exam draws with at least one such distractor. Every
    distractor's zeros value must be in {0, 1, 2} on every draw, every tier."""
    for tier in TIERS:
        for q in _all(DISC_TOPIC, tier, seeds=range(200)):
            for d in q.distractors:
                m = re.search(r"zeros = (-?\d+)", d)
                assert m, d
                assert int(m.group(1)) in (0, 1, 2), f"{tier} seed {q.seed}: nonsense distractor {d!r}"


def test_disc_exam_wording_does_not_say_crosses_the_x_axis():
    """ "how many times it crosses the x-axis" is a defensible-to-reject reading when D=0 (the curve TOUCHES, not
    crosses, the axis) -- measured on the third exam template, 64/200 draws. Reworded to "how many
    x-intercepts it has", which does not depend on crossing vs touching."""
    for q in _all(DISC_TOPIC, "exam", seeds=range(200)):
        assert "crosses the x-axis" not in q.text, f"seed {q.seed}: {q.text!r}"


def test_disc_exam_tier_includes_boundary_and_near_boundary_deltas():
    seen_small = 0
    for q in _all(DISC_TOPIC, "exam", seeds=range(200)):
        if abs(q.answer["D"]) <= 4:
            seen_small += 1
    assert seen_small > 0


# ============================================================================
# p19e_discriminant_zero_count: independent in-generator check
# ============================================================================


def test_independent_check_fires_on_corrupted_discriminant():
    gen._verify_discriminant(1, 5, 2, 17)  # no exception when correct (5^2-4*1*2=17)
    with pytest.raises(VerificationError):
        gen._verify_discriminant(1, 5, 2, 18)
    with pytest.raises(VerificationError):
        gen._verify_discriminant(1, 5, 2, -17)


def test_classify_matches_definition():
    assert gen._classify(5) == 2
    assert gen._classify(0) == 1
    assert gen._classify(-5) == 0


# ============================================================================
# p19e_discriminant_zero_count: real grader
# ============================================================================


@pytest.mark.parametrize("tier", TIERS)
def test_disc_real_grader_accepts_canonical_and_rejects_distractors(tier):
    for q in _all(DISC_TOPIC, tier, seeds=range(20)):
        assert grade_pair(q, q.answer_text).correct, f"{tier} seed {q.seed}: canonical rejected"
        for d in q.distractors:
            assert not grade_pair(q, d).correct, f"{tier} seed {q.seed}: distractor {d!r} wrongly accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_disc_real_grader_accepts_reordered_labels_and_alt_forms(tier):
    for q in _all(DISC_TOPIC, tier, seeds=range(10)):
        d, z = q.answer["D"], q.answer["zeros"]
        reordered = f"zeros = {z}, D = {d}"
        assert grade_pair(q, reordered).correct, f"{tier} seed {q.seed}: reordered rejected"
        bare = f"{d}, {z}"
        assert grade_pair(q, bare).correct, f"{tier} seed {q.seed}: bare positional rejected"


@pytest.mark.parametrize("tier", TIERS)
def test_disc_real_grader_rejects_wrong_and_values_swapped_and_word(tier):
    for q in _all(DISC_TOPIC, tier, seeds=range(10)):
        d, z = q.answer["D"], q.answer["zeros"]
        wrong = f"D = {d + 1000}, zeros = {z}"
        assert not grade_pair(q, wrong).correct, f"{tier} seed {q.seed}: wrong D wrongly accepted"
        if d != z:
            swapped = f"D = {z}, zeros = {d}"
            assert not grade_pair(q, swapped).correct, f"{tier} seed {q.seed}: swapped values wrongly accepted"
        assert not grade_pair(q, "two").correct


# ============================================================================
# Skill load
# ============================================================================


class TestSkillLoad:
    def test_degree_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == DEG_SKILL)
        assert skill.domain == "functions-sequences"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["variable-concept", "evaluate-powers"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == DEG_TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]

    def test_discriminant_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == DISC_SKILL)
        assert skill.domain == "quadratics"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["quadratic-vertex"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == DISC_TOPIC
            stages_covered.update(source.stages)
        assert "exam" in stages_covered
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]


def test_topics_registered_once():
    from mathgen.core import REGISTRY

    _, fn1 = REGISTRY[DEG_TOPIC]
    _, fn2 = REGISTRY[DISC_TOPIC]
    assert fn1.__module__.endswith("p19e_degree_discriminant_build")
    assert fn2.__module__.endswith("p19e_degree_discriminant_build")
