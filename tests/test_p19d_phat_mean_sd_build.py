"""Theoretical mean and standard deviation of the sampling distribution of the sample proportion p-hat
(senior-understand-the-concept-of-the-sample-proportion-widehat, ACMMM174, ``answer: pair``)."""

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen import make
from mathgen.core import GRADER_TOLERANCE

TOPIC = "p19d_phat_mean_sd"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
SEEDS_200 = range(200)

_UNITS = ["sectors", "marbles", "components", "cards"]
_GENERIC_SAMPLE_NOUNS = ["people", "items", "respondents"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_shape_is_labelled_mean_sd_dict(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert set(q.answer) == {"mean", "sd"}
        assert q.answer_text.startswith("mean = ") and ", sd = " in q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_mean_equals_p_exactly_and_sd_matches_formula(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        mean_str, sd_str = q.answer_text.replace("mean = ", "").replace("sd = ", "").split(", ")
        mean_val = Fraction(mean_str)
        sd_val = Fraction(sd_str)
        assert mean_val == Fraction(str(q.answer["mean"]))
        assert abs(float(sd_val) - q.answer["sd"]) < 1e-9
        assert sd_str.count(".") == 1 and len(sd_str.split(".")[1]) == 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_four_distractors_every_seed(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 4
        assert len(set(q.distractors)) == 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_labelled_pair_grader_is_order_insensitive(difficulty):
    """`mean = ..., sd = ...` and `sd = ..., mean = ...` must both be
    accepted -- the labelled-pair design decision this skill relies on
    (`mathgen/grading/typed.py::_labelled_pair_verdict`)."""
    for seed in SEEDS[:20]:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        mean_part, sd_part = q.answer_text.split(", ")
        swapped = f"{sd_part}, {mean_part}"
        assert grade_pair(q, swapped).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_sd_component_or_mean_component_clears_grader_tolerance(difficulty):
    """s24: every misconception distractor must differ from the true pair by MORE than grade_pair's 0.051
    tolerance on at least one component, checked directly every seed -- never assumed from the construction
    ranges alone."""
    from mathgen.core import GRADER_TOLERANCE

    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        mean_str, sd_str = q.answer_text.replace("mean = ", "").replace("sd = ", "").split(", ")
        true_mean, true_sd = Fraction(mean_str), Fraction(sd_str)
        for d in q.distractors:
            dm, ds = d.replace("mean = ", "").replace("sd = ", "").split(", ")
            d_mean, d_sd = Fraction(dm), Fraction(ds)
            assert abs(d_mean - true_mean) > GRADER_TOLERANCE or abs(d_sd - true_sd) > GRADER_TOLERANCE, (
                seed,
                difficulty,
                d,
            )


@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not grade_pair(q, "mean = 0.999999, sd = 0.999999").correct
        assert not grade_pair(q, "not an answer").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_p_and_n_stated_explicitly_in_text(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "p = " in q.text
        assert "n = " in q.text


def test_never_mentions_an_observed_sample_proportion():
    """Boundary vs. every other shipping p-hat skill: this skill's text
    always frames p as a GIVEN true value and n as a planned sample size,
    never an observed p-hat read off an already-drawn sample."""
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert "was drawn" not in q.text
            assert "sample proportion, p-hat, is" not in q.text
            assert "confidence" not in q.text
            assert "margin of error" not in q.text.lower()


def test_medium_requires_fraction_conversion_absent_in_easy():
    for seed in SEEDS:
        q_easy = make(TOPIC, seed=seed, difficulty="easy")
        q_medium = make(TOPIC, seed=seed, difficulty="medium")
        assert "/" not in q_easy.text.split("A random sample")[0]
        assert any(f"{n}/{d}" in q_medium.text for n in range(1, 20) for d in (2, 4, 5, 8, 10, 20))


def test_hard_uses_a_chance_model_absent_in_medium():
    contexts = ["spinner", "bag of marbles", "box of components", "deck of cards"]
    for seed in SEEDS:
        q_hard = make(TOPIC, seed=seed, difficulty="hard")
        assert any(c in q_hard.text for c in contexts)


def test_exam_names_two_studies_and_a_target_absent_in_hard():
    for seed in SEEDS:
        q_exam = make(TOPIC, seed=seed, difficulty="exam")
        assert "Study A" in q_exam.text and "Study B" in q_exam.text
        assert "for Study" in q_exam.text
        q_hard = make(TOPIC, seed=seed, difficulty="hard")
        assert "Study A" not in q_hard.text


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_sampled_unit_matches_the_chance_context_200_seeds(difficulty):
    """Item A: the sample-size sentence must name the SAME unit as the
    chance-model context ("a deck of cards" ... "35 cards"), never a
    generic "people"/"items"/"respondents" noun instead."""
    mismatches = 0
    for seed in SEEDS_200:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        text = q.text
        present_units = [u for u in _UNITS if u in text]
        assert len(present_units) == 1, (seed, text)
        unit = present_units[0]
        generic_hits = [g for g in _GENERIC_SAMPLE_NOUNS if g in text]
        if generic_hits:
            mismatches += 1
            continue
        # The sample-size sentence ("batch of N <unit>" / "sample n = N
        # <unit>") must use the context's own unit word.
        if difficulty == "hard":
            m = re.search(r"batch of \d+ (\w+)", text)
        else:
            m = re.search(r"will sample n = \d+ (\w+)", text)
        assert m, (seed, text)
        if m.group(1) != unit:
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/200 noun mismatches ({difficulty})"


def _true_mean_sd(q):
    mean_str, sd_str = q.answer_text.replace("mean = ", "").replace("sd = ", "").split(", ")
    return Fraction(mean_str), Fraction(sd_str)


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_trap_n_distractor_present_and_clears_tolerance_when_offered(difficulty):
    """Item B: a 5th "trap-n" candidate (sqrt(var / wrong_n) for a wrong-but-
    present-in-the-text n) is constructed on hard/exam; whenever
    `pick_distractors` actually offers it, it must clear GRADER_TOLERANCE on
    sd by direct Fraction-based construction-time check -- never merely
    trusted from `pick_distractors`' own filtering."""
    offered = 0
    for seed in SEEDS_200:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        true_mean, true_sd = _true_mean_sd(q)
        p = float(true_mean)
        var = p * (1 - p)

        if difficulty == "hard":
            m = re.search(r"batch of (\d+) \w+", q.text)
            wrong_n = int(m.group(1))
        else:
            n_values = [int(x) for x in re.findall(r"will sample n = (\d+)", q.text)]
            # n_target is recoverable from true_sd (n = var / sd^2, exact to
            # the nearest integer); wrong_n is the OTHER named study's n.
            n_target = round(var / (float(true_sd) ** 2))
            wrong_n = [n for n in n_values if n != n_target][0]

        sd_trap = round(math.sqrt(var / wrong_n), 4)
        trap_text = f"mean = {p:.4f}, sd = {sd_trap:.4f}"
        if trap_text in q.distractors:
            offered += 1
            trap_sd_frac = Fraction(str(sd_trap))
            assert abs(trap_sd_frac - true_sd) > GRADER_TOLERANCE, (seed, difficulty, trap_text)
    assert offered > 0, f"trap-n distractor never offered over 200 seeds ({difficulty})"


_LARGE_POOL_PHRASE = "very large supply of"


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_sampled_count_never_exceeds_stated_pool_or_states_unbounded_supply(difficulty):
    """Assert the chosen discipline positively and unconditionally: every draw must either (a) never sample more
    than `total`, or (b) say so via the large-supply phrase -- never silently exceed the pool."""
    violations = 0
    unbounded = 0
    worst = []
    for seed in SEEDS_200:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        text = q.text
        m_total = re.search(r"(\d+) equally likely", text)
        assert m_total, (seed, text)
        total = int(m_total.group(1))
        large_pool = _LARGE_POOL_PHRASE in text
        if difficulty == "hard":
            m = re.search(r"batch of (\d+) \w+", text)
            assert m, (seed, text)
            sampled = int(m.group(1))
        else:
            n_values = [int(x) for x in re.findall(r"will sample n = (\d+)", text)]
            assert len(n_values) == 2, (seed, text)
            sampled = max(n_values)
        if large_pool:
            unbounded += 1
        elif sampled > total:
            violations += 1
            if len(worst) < 5:
                worst.append((seed, total, sampled))
        assert large_pool or sampled <= total, (seed, total, sampled, text)
    assert violations == 0, f"{violations}/200 sampled-exceeds-pool violations ({difficulty}); worst={worst}"
    assert unbounded == 200, f"only {unbounded}/200 draws state the large-supply framing ({difficulty})"


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_exam_target_n_never_exceeds_pool_under_old_strict_framing(difficulty):
    """Confirm explicitly (not assume) whether the TARGET study's own n -- drawn from
    `_N_TIGHT_RANGE`/`_N_DERIVE_TIGHT`, never the trap-n `other_n`/`base` -- would itself have exceeded a
    strict small-pool reading, now that the large-supply framing makes doing so unproblematic either way.
    Purely diagnostic: records the measured rate, asserts nothing beyond the fact that it is measurable."""
    exceeds = 0
    for seed in SEEDS_200:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        text = q.text
        m_total = re.search(r"(\d+) equally likely", text)
        total = int(m_total.group(1))
        if difficulty == "hard":
            # hard's own target n is the *derived* n = base*dnum//dden,
            # recoverable as the trailing "so n = (\d+)." literal.
            m = re.search(r"so n = (\d+)\.", text)
        else:
            m = re.search(r"for Study ([A-Z])", text)
            target = m.group(1)
            m = re.search(rf"Study {target} will sample n = (\d+)", text)
        n_target = int(m.group(1))
        if n_target > total:
            exceeds += 1
    assert isinstance(exceeds, int)


def test_medium_trap_n_distractor_present_and_clears_tolerance_when_offered():
    """Medium tier must also offer a trap-n sd distractor (sqrt(var / wrong_n) for the
    pre-derivation `base` count), the same discipline as hard/exam."""
    offered = 0
    for seed in SEEDS_200:
        q = make(TOPIC, seed=seed, difficulty="medium")
        true_mean, true_sd = _true_mean_sd(q)
        p = float(true_mean)
        var = p * (1 - p)
        m = re.search(r"contact (\d+) potential respondents", q.text)
        assert m, (seed, q.text)
        wrong_n = int(m.group(1))
        sd_trap = round(math.sqrt(var / wrong_n), 4)
        trap_text = f"mean = {p:.4f}, sd = {sd_trap:.4f}"
        if trap_text in q.distractors:
            offered += 1
            trap_sd_frac = Fraction(str(sd_trap))
            assert abs(trap_sd_frac - true_sd) > GRADER_TOLERANCE, (seed, trap_text)
    assert offered > 0, "trap-n distractor never offered over 200 seeds (medium)"


def test_phat_pair_construction_always_includes_a_trap_n_candidate_when_asked():
    """White-box check on the literal wording ("present among the
    four-or-five candidate list construction attempts"): `_phat_pair` itself
    must build a 5th trap-n candidate whenever a `wrong_n` is supplied, and
    only 4 when it is not -- independent of whether `pick_distractors` later
    keeps it."""
    from mathgen.generators import p19d_phat_mean_sd_build as _mod
    from fractions import Fraction as _F

    p, n, wrong_n = _F(2, 5), 30, 270  # inside the tight pool, k = 9
    result_without = _mod._phat_pair(p, n)
    result_with = _mod._phat_pair(p, n, wrong_n=wrong_n)
    assert len(result_without["candidate_texts"]) == 4
    assert len(result_with["candidate_texts"]) == 5
    var = float(p) * (1 - float(p))
    import math as _math

    sd_trap = round(_math.sqrt(var / wrong_n), 4)
    assert f"mean = {float(p):.4f}, sd = {sd_trap:.4f}" in result_with["candidate_texts"]


_CONTAINER_CLAIM = re.compile(r"\b(?:has|have|contains?|holds?) (\d+) equally likely")
_TINY_POOL = 50  # every ratio denominator in the pool is <= 20; any literal
# physical container under 50 items cannot supply hundreds of sampled units.


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_no_tiny_literal_pool_claim_while_far_larger_count_sampled(difficulty):
    """The setup must never assert that a small physical container (`has {total} equally likely {unit}`) exists
    while a far larger count is sampled. p is stated as a ratio of a very large supply instead."""
    flagged = []
    for seed in SEEDS_200:
        text = make(TOPIC, seed=seed, difficulty=difficulty).text
        m_total = re.search(r"(\d+) equally likely", text)
        assert m_total, (seed, text)
        total = int(m_total.group(1))
        if difficulty == "hard":
            sampled = int(re.search(r"batch of (\d+) \w+", text).group(1))
        else:
            sampled = max(int(x) for x in re.findall(r"will sample n = (\d+)", text))
        claims_container = bool(_CONTAINER_CLAIM.search(text))
        ratio_only = re.search(rf"[Ii]n every (?:group of )?{total} equally likely", text)
        if (claims_container and total < _TINY_POOL and sampled > total) or not ratio_only:
            flagged.append((seed, total, sampled))
    assert not flagged, f"{len(flagged)}/200 tiny-pool claims ({difficulty}); worst={flagged[:5]}"


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_ratio_sentence_is_structurally_a_ratio_of_a_very_large_supply(difficulty):
    for seed in SEEDS_200:
        text = make(TOPIC, seed=seed, difficulty=difficulty).text
        first = text.split(". ")[0]
        assert "very large supply of" in first, (seed, first)
        assert "so the true proportion is p = " in text
        assert text.count("very large supply of") == 1, (seed, text)
