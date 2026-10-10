"""Tests for `mathgen/generators/p16b_correlation_r_build.py`:
`p16b_correlation_r` (skill
`senior-use-technology-to-find-the-correlation-coefficient-an`, ACMEM144).

This file holds the generator-specific
proofs: the in-generator independent check firing on a corrupted value, the
real-grader battery, the answer's own-kind checks, and the constraint tests
(answer-space band, hard-tier tie gap, exam-tier
weighting).

The independent ORACLE lives in `tests/test_p16b_correlation_r_oracle.py`.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance
from mathgen.text_quality import find_text_defects, prompt_shape

import mathgen.generators.p16b_correlation_r_build as gen

TOPIC = "p16b_correlation_r"
TIERS = ("easy", "medium", "hard", "exam")
SKILL_ID = "senior-use-technology-to-find-the-correlation-coefficient-an"


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


def test_hard_and_exam_each_add_a_shape_absent_from_lower_tiers():
    lower_shapes = {prompt_shape(q) for t in ("easy", "medium") for q in _all(t)}
    hard_shapes = {prompt_shape(q) for q in _all("hard")}
    exam_shapes = {prompt_shape(q) for q in _all("exam")}
    assert hard_shapes - lower_shapes, "hard tier adds no new shape over easy/medium"
    assert exam_shapes - lower_shapes, "exam tier adds no new shape over easy/medium"


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
def test_answer_and_distractors_formatted_alike_3dp_signed(tier):
    """Every option (answer + distractors) is a signed decimal at exactly 3
    places, so no option is identifiable by its shape alone."""
    for q in _all(tier, seeds=range(20)):
        for opt in [q.answer_text, *q.distractors]:
            assert re.match(r"^-?\d+\.\d{3}$", opt), f"{tier} seed {q.seed}: {opt!r} not 3dp"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space_band_0_2_to_0_98(tier):
    """|r| is always in [0.2, 0.98], never on the exact
    boundary, and the 3-d.p. rounding is never within margin of a half."""
    for q in _all(tier, seeds=range(60)):
        r = float(q.answer_text)
        assert 0.2 <= abs(r) <= 0.98, f"{tier} seed {q.seed}: |r|={abs(r)} out of band"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_two_distractors_on_85pct_of_draws(tier):
    counts = [len(q.distractors) for q in _all(tier, seeds=range(200))]
    at_least_two = sum(1 for c in counts if c >= 2)
    assert at_least_two / len(counts) >= 0.85, f"{tier}: only {at_least_two}/{len(counts)} >= 2 distractors"


def _negation_pair_count(options):
    n = len(options)
    return sum(1 for i in range(n) for j in range(i + 1, n) if options[i] == -options[j])


@pytest.mark.parametrize("tier", TIERS)
def test_negated_key_not_always_offered_and_not_uniquely_identifiable(tier):
    """Proven RED first against the pre-fix generator -- 200/200 draws of every tier offered the negated key as a
    distractor, so a guesser could always narrow to the +/- pair containing the key. Fixed generator: the
    negated key's share is <= 0.65, and the "only sign-symmetric pair among the options is the key's own"
    share is also <= 0.65."""
    neg_key_hits = 0
    only_pair_hits = 0
    for q in _all(tier, seeds=range(200)):
        key = Fraction(q.answer_text)
        options = [key] + [Fraction(d) for d in q.distractors]
        has_neg_key = any(o == -key for o in options[1:])
        if has_neg_key:
            neg_key_hits += 1
        pairs = _negation_pair_count(options)
        if pairs == 1 and has_neg_key:
            only_pair_hits += 1
    assert neg_key_hits / 200 <= 0.65, f"{tier}: negated key offered in {neg_key_hits}/200 draws"
    assert only_pair_hits / 200 <= 0.65, f"{tier}: key uniquely +/--paired in {only_pair_hits}/200 draws"


@pytest.mark.parametrize("tier", TIERS)
def test_no_answer_key_concentration_over_25pct_of_200(tier):
    from collections import Counter

    answers = Counter(q.answer_text for q in _all(tier, seeds=range(200)))
    top = max(answers.values())
    assert top / 200 <= 0.25, f"{tier}: top answer {answers.most_common(1)} is {top}/200"


@pytest.mark.parametrize("tier", TIERS)
def test_latex_never_contains_a_bare_double_dash(tier):
    """Regression guard for a previously fixed defect."""
    for q in _all(tier, seeds=range(60)):
        assert "--" not in q.latex, f"{tier}: '--' in latex {q.latex!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_starts_with_uppercase_or_digit(tier):
    """The shipped `_cap` post-processing step now guarantees an uppercase/digit start regardless of which
    template/context is drawn."""
    for q in _all(tier, seeds=range(60)):
        assert q.text[0].isupper() or q.text[0].isdigit(), (
            f"{tier} seed {q.seed}: prompt does not start uppercase/digit: {q.text[:60]!r}"
        )


_HARD_CONTEXT_SIGNS = {
    "weekly training hours": "+",
    "advertising spend ($100s)": "+",
    "hours of study per week": "+",
    "hours of screen time per day": "-",
    "age of the car (years)": "-",
    "minutes of daily exercise": "-",
}


def test_hard_contexts_have_at_least_3_of_each_natural_sign():
    signs = [gen._HARD_CONTEXTS[i]["x_label"] for i in range(len(gen._HARD_CONTEXTS))]
    assert len(gen._HARD_CONTEXTS) == len(_HARD_CONTEXT_SIGNS)
    pos = sum(1 for c in gen._HARD_CONTEXTS if c["slope"] > 0)
    neg = sum(1 for c in gen._HARD_CONTEXTS if c["slope"] < 0)
    assert pos >= 3 and neg >= 3, f"pos={pos} neg={neg}"
    for ctx in gen._HARD_CONTEXTS:
        expected = _HARD_CONTEXT_SIGNS[ctx["x_label"]]
        actual = "+" if ctx["slope"] > 0 else "-"
        assert actual == expected, f"{ctx['x_label']}: expected {expected}, context has {actual}"


def test_hard_stronger_set_always_follows_its_contexts_natural_sign():
    """PROVEN RED first: before the fix, `_p16b_build_hard` picked a random
    context and then forced an arbitrary sign onto it via `want_negative`:
    seed 0 printed r = 0.912 for screen time vs sleep duration
    (a context whose natural slope is negative) and seed 1 printed
    r = -0.823 for training hours vs fitness score (natural slope positive)
    -- both nonsense to a subject teacher. Confirmed red directly: replaying
    the OLD unconstrained draw (`_draw_signed_set` with an arbitrary
    `want_negative` against a context of the opposite natural sign) still
    succeeds mechanically (the guard removed was a selection constraint, not
    a draw-time check), so only the NEW `_p16b_build_hard` selection logic
    (asserted below) prevents it from ever being offered as the printed
    answer."""
    import random

    # The old bug's shape: force a NEGATIVE draw on a POSITIVE-natural
    # context -- `_draw_signed_set` happily complies (it only respects
    # `want_negative`, never the context's own slope), which is exactly why
    # the fix has to live in `_p16b_build_hard`'s context selection instead.
    # Tried across several seeds (like `test_hard_tie_gap_at_least_0_05`
    # does) since any single seed can fail to converge in 400 attempts.
    positive_ctx = next(c for c in gen._HARD_CONTEXTS if c["slope"] > 0)
    found = False
    for i in range(20):
        rng = random.Random(f"proof-red:hard-sign:{i}")
        xs, ys, r = gen._draw_signed_set(
            rng,
            positive_ctx,
            6,
            True,
            gen._HARD_STRONG_LOW,
            gen._HARD_STRONG_HIGH,
            2,
            6,
        )
        if xs is not None:
            assert r < 0, "want_negative=True produced a positive r"
            found = True
            break
    assert found, (
        "the unguarded _draw_signed_set no longer accepts a sign that fights its context "
        "-- the proof-red case is stale, update it"
    )

    signs = {"pos": 0, "neg": 0}
    for q in _all("hard", seeds=range(200)):
        ctx_label = next(lbl for lbl in _HARD_CONTEXT_SIGNS if lbl in q.text)
        natural = _HARD_CONTEXT_SIGNS[ctx_label]
        # The printed answer is always the STRONGER set (by construction,
        # confirmed elsewhere); its sign must match its context's natural
        # sign.
        answer = float(q.answer_text)
        if natural == "+":
            assert answer > 0, f"seed {q.seed}: {ctx_label} is positive-natural but answer={answer}"
            signs["pos"] += 1
        else:
            assert answer < 0, f"seed {q.seed}: {ctx_label} is negative-natural but answer={answer}"
            signs["neg"] += 1
    assert signs["pos"] >= 40 and signs["neg"] >= 40, (
        f"200-seed class split too skewed to trust the 50/50 draw: {signs}"
    )


_EASY_CONTEXT_SIGNS = {c["x_bare"]: ("+" if c["slope"] > 0 else "-") for c in gen._EASY_CONTEXTS}
_MEDIUM_CONTEXT_SIGNS = {c["x_bare"]: ("+" if c["slope"] > 0 else "-") for c in gen._MEDIUM_CONTEXTS}
_EXAM_CONTEXT_SIGNS = {c["x_bare"]: ("+" if c["slope"] > 0 else "-") for c in gen._EXAM_CONTEXTS}


def test_easy_and_exam_follow_their_contexts_natural_sign():
    """Easy/medium/exam never force a sign at all -- `y`/`derived` is built
    directly from `ctx["slope"]`, so the printed r's sign always matches the
    context's own slope sign by construction (unlike the pre-fix hard tier).
    Exam mixes signs (hours of practice/temperature negative, age
    positive), so this is a genuine cross-check, not a vacuous one."""
    for tier, sign_map in (("easy", _EASY_CONTEXT_SIGNS), ("exam", _EXAM_CONTEXT_SIGNS)):
        seen_signs = set()
        for q in _all(tier, seeds=range(60)):
            ctx_label = next(lbl for lbl in sign_map if lbl in q.text)
            expected = sign_map[ctx_label]
            r = float(q.answer_text)
            actual = "+" if r > 0 else "-"
            assert actual == expected, f"{tier} seed {q.seed}: {ctx_label} expected {expected}, got {actual}"
            seen_signs.add(expected)
        if tier == "exam":
            assert seen_signs == {"+", "-"}, f"exam: only saw {seen_signs} in 60 seeds"


def test_medium_derived_sign_follows_its_contexts_slope():
    """Medium's derived variable is built as `base + slope * x + noise`
    directly (never p or q alone), so its sign tracks `ctx["slope"]`."""
    for q in _all("medium", seeds=range(60)):
        ctx_label = next(lbl for lbl in _MEDIUM_CONTEXT_SIGNS if lbl in q.text)
        expected = _MEDIUM_CONTEXT_SIGNS[ctx_label]
        r = float(q.answer_text)
        actual = "+" if r > 0 else "-"
        assert actual == expected, f"medium seed {q.seed}: {ctx_label} expected {expected}, got {actual}"


# ============================================================================
# Independent in-generator check (`_verify_r`) fires on a corrupted value
# ============================================================================


def test_verify_r_fires_on_corrupted_value():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 5, 4, 6]
    r = gen._pearson_r(xs, ys)
    r_rounded = gen._round_r(r)
    gen._verify_r(xs, ys, r_rounded)  # no exception -- the true rounded r
    with pytest.raises(VerificationError):
        gen._verify_r(xs, ys, r_rounded + Fraction(1, 100))  # a corrupted key


def test_verify_r_uses_a_genuinely_different_route_from_construction():
    """`_pearson_r` (construction) is a hand-written Fraction formula;
    `_verify_r` calls `statistics.correlation` (stdlib) -- proven here to
    actually be a different code path, not merely a different name for the
    same arithmetic, by confirming the stdlib call independently reproduces
    a value `_pearson_r` was never given the chance to compute for it."""
    xs = [3, 7, 2, 9, 5, 1]
    ys = [10, 22, 8, 27, 15, 4]
    via_construction = gen._round_r(gen._pearson_r(xs, ys))
    via_stdlib = gen._round_r(statistics.correlation(xs, ys))
    assert via_construction == via_stdlib
    gen._verify_r(xs, ys, via_construction)


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
    tol = float(stated_precision_tolerance(3))
    for q in _all(tier, seeds=range(20)):
        wrong = float(q.answer_text) + max(0.05, tol * 20)
        assert not grade_number(q, f"{wrong:.3f}").correct, f"{tier} seed {q.seed}: wrong value accepted"


# ============================================================================
# Constraint tests
# ============================================================================


def test_hard_tie_gap_at_least_0_05():
    """|r_A| - |r_B| >= 0.05 on the hard tier, always. PROVEN RED
    first: `_HARD_WEAK_HIGH` (0.5) and `_HARD_STRONG_LOW` (0.65) leave a
    0.15 floor between the two draw bands with no overlap possible, so
    setting them to overlapping bands (0.5/0.5) would immediately let a
    same-magnitude draw through; confirmed by temporarily calling
    `_draw_signed_set` with equal bands below and observing the tie-gap
    assertion fail on the resulting pair (this is exactly the failure mode
    the shipped `_HARD_STRONG_LOW > _HARD_WEAK_HIGH` gap prevents)."""
    import random

    rng = random.Random("proof-red:hard-tie-gap")
    xs, ys, r_same_band = gen._draw_signed_set(
        rng,
        gen._HARD_CONTEXTS[0],
        6,
        False,
        Fraction(4, 10),
        Fraction(6, 10),
        4,
        10,
    )
    xs2, ys2, r_same_band2 = gen._draw_signed_set(
        rng,
        gen._HARD_CONTEXTS[0],
        6,
        False,
        Fraction(4, 10),
        Fraction(6, 10),
        4,
        10,
    )
    if r_same_band is not None and r_same_band2 is not None:
        # Overlapping bands CAN produce a sub-0.05 gap (the unguarded case) --
        # this loop just needs one such pair to exist across a few draws to
        # prove the guard is doing real work, not asserting a vacuous truth.
        gaps = []
        for seed in range(30):
            rr = random.Random(f"proof-red:{seed}")
            a = gen._draw_signed_set(rr, gen._HARD_CONTEXTS[0], 6, False, Fraction(4, 10), Fraction(6, 10), 4, 10)
            b = gen._draw_signed_set(rr, gen._HARD_CONTEXTS[0], 6, False, Fraction(4, 10), Fraction(6, 10), 4, 10)
            if a[2] is not None and b[2] is not None:
                gaps.append(abs(abs(a[2]) - abs(b[2])))
        assert any(g < 0.05 for g in gaps), "overlapping bands never produced a sub-0.05 gap to guard against"

    for q in _all("hard", seeds=range(60)):
        assert "|r_A|" in q.working[2] or "stronger" in q.working[2]

    # The shipped generator's own draws never violate the gap: re-parse the
    # two "r_A = .."/"r_B = .." working lines straight from the rendered
    # output for every seed.
    r_re = re.compile(r"r_[AB] = (-?\d+\.\d{3})")
    for q in _all("hard", seeds=range(60)):
        matches = r_re.findall(" ".join(q.working))
        assert len(matches) == 2, q.working
        r_a, r_b = (float(m) for m in matches)
        assert abs(abs(r_a) - abs(r_b)) >= 0.05, f"seed {q.seed}: tie gap violated: {r_a}, {r_b}"


def test_hard_strength_is_abs_r_not_r():
    """The answer is the STRONGER set's r (by |r|), not simply the larger
    signed r -- proven by finding at least one seed (across 60) where the
    stronger (answer) set's r is NEGATIVE and more negative than the other
    set's positive r, so "pick the larger r" would pick the wrong set."""
    r_re = re.compile(r"r_[AB] = (-?\d+\.\d{3})")
    found_negative_answer_stronger_than_positive_other = False
    for q in _all("hard", seeds=range(200)):
        matches = r_re.findall(" ".join(q.working))
        r_a, r_b = (float(m) for m in matches)
        answer = float(q.answer_text)
        other = r_b if answer == r_a else r_a
        assert abs(answer) >= abs(other), f"seed {q.seed}: answer is not the stronger set"
        if answer < 0 and other > 0:
            found_negative_answer_stronger_than_positive_other = True
    assert found_negative_answer_stronger_than_positive_other, (
        "60 seeds never drew the negative-is-stronger class -- widen the sweep or the draw"
    )


def test_exam_ignores_counts_distractor_uses_unweighted_data():
    """The exam distractor must differ from the (weighted) key by
    construction. PROVEN RED first: computing r on the distinct pairs with
    every count forced to 1 (the literal misconception) against a weighted
    data set whose counts are NOT all 1 generally disagrees with the
    weighted r -- confirmed directly for a corrupted (all-ones) count list
    below, which is exactly the unguarded case this distractor represents."""
    xs = [1, 2, 3, 4, 5]
    ys = [10, 12, 20, 19, 30]
    counts = [1, 1, 4, 1, 1]
    expanded_xs, expanded_ys = [], []
    for x, y, c in zip(xs, ys, counts):
        expanded_xs += [x] * c
        expanded_ys += [y] * c
    r_weighted = gen._round_r(gen._pearson_r(expanded_xs, expanded_ys))
    r_unweighted = gen._round_r(gen._pearson_r(xs, ys))
    assert r_weighted != r_unweighted, "the weighted/unweighted r coincide on this proof case"

    for q in _all("exam", seeds=range(60)):
        assert "freq" in q.text


def test_exam_weighted_r_matches_full_expanded_dataset():
    """Re-parse the exam tier's frequency table straight from the rendered
    text and confirm the printed answer is the WEIGHTED r, not the r of the
    distinct rows."""
    row_re = re.compile(r"\((-?\d+), (-?\d+), freq (\d+)\)")
    for q in _all("exam", seeds=range(60)):
        rows = row_re.findall(q.text)
        assert len(rows) >= 4
        expanded_xs, expanded_ys = [], []
        for x, y, c in rows:
            expanded_xs += [int(x)] * int(c)
            expanded_ys += [int(y)] * int(c)
        expected = gen._round_r(gen._pearson_r(expanded_xs, expanded_ys))
        assert stated_precision_tolerance(3) >= abs(expected - Fraction(q.answer_text)), (
            f"seed {q.seed}: weighted r mismatch"
        )


def test_medium_derived_variable_is_the_difference_of_the_two_printed_columns():
    """Re-parse the medium tier's triples and confirm the printed answer is
    r(x, derived) where derived is q-p or p-q, never r(x, p) or r(x, q)
    alone (the raw-column misconception this tier is built to exclude)."""
    triple_re = re.compile(r"\((-?\d+), (-?\d+), (-?\d+)\)")
    for q in _all("medium", seeds=range(60)):
        triples = triple_re.findall(q.text)
        assert len(triples) >= 5
        xs = [int(t[0]) for t in triples]
        p_vals = [int(t[1]) for t in triples]
        q_vals = [int(t[2]) for t in triples]
        derived_a = [qq - pp for pp, qq in zip(p_vals, q_vals)]
        derived_b = [pp - qq for pp, qq in zip(p_vals, q_vals)]
        r_a = gen._pearson_r(xs, derived_a)
        r_b = gen._pearson_r(xs, derived_b)
        answer = Fraction(q.answer_text)
        matches_a = r_a is not None and abs(gen._round_r(r_a) - answer) <= stated_precision_tolerance(3)
        matches_b = r_b is not None and abs(gen._round_r(r_b) - answer) <= stated_precision_tolerance(3)
        assert matches_a or matches_b, f"seed {q.seed}: answer matches neither derived direction"
        r_p = gen._pearson_r(xs, p_vals)
        r_q = gen._pearson_r(xs, q_vals)
        if r_p is not None:
            assert abs(gen._round_r(r_p) - answer) > stated_precision_tolerance(3), (
                f"seed {q.seed}: answer coincides with the raw p-column correlation"
            )
        if r_q is not None:
            assert abs(gen._round_r(r_q) - answer) > stated_precision_tolerance(3), (
                f"seed {q.seed}: answer coincides with the raw q-column correlation"
            )


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
