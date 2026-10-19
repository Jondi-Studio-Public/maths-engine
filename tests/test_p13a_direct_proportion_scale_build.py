"""Tests `f10-direct-proportion-scale-model` (AC9M9M05/AC9M10M05): direct proportion in scale models."""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p13a_direct_proportion_scale"
SKILL = "f10-direct-proportion-scale-model"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

_LABEL_YES = "Yes, directly proportional"
_LABEL_NO = "No, not directly proportional (a fixed fee applies)"
_OPTION_RE = re.compile(
    r"^(Yes, directly proportional|No, not directly proportional \(a fixed fee applies\)) -- "
    r"(?:the total cost is \$(\d+\.\d{2})|you can buy (\d+) [a-zA-Z]+)\.$"
)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(difficulty, seed):
    skill = _skills_dict()[SKILL]
    if difficulty == "exam":
        # The skill's exam tier is now the multi-part chain `exd_x_r2x_direct_proportion_table`; this module
        # tests this topic's own single-step exam preset, which the generator still provides.
        return make(TOPIC, seed=seed, difficulty="exam")
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _label_of(option: str) -> str:
    m = _OPTION_RE.match(option)
    assert m, option
    return m.group(1)


# ============================================================================= Required test
# coverage, adapted to mcq
# =============================================================================


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(difficulty, seed)
        assert len(q.distractors) == 3, (difficulty, seed, q.distractors)
        assert len(set(q.distractors)) == 3, (difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    assert GRADERS["mcq"] is grade_mcq
    for seed in SEEDS:
        q = _make_from_sources(difficulty, seed)
        opts = mcq_options(q)
        correct_idx = opts.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct, (difficulty, seed, q.text, q.answer_text)
        for idx, opt in enumerate(opts):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, idx).correct, (difficulty, seed, q.text, opt)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(difficulty, seed)
        opts = mcq_options(q)
        # an out-of-range index is unambiguously wrong regardless of shuffle
        assert not grade_mcq(q, len(opts) + 5).correct, (difficulty, seed, q.text)
        assert not grade_mcq(q, "not-a-number").correct, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_corrupted_canonical(difficulty):
    """A corrupted canonical text (one-cent/one-unit nudge, or a flipped
    classification label) never happens to equal a real rendered option --
    the real grader (matching by index, not by re-parsing given text) would
    never even see this string, but the underlying VALUE must still not
    collide with a real option for `pick_distractors`'s own dedup to have
    done its job. Checked here directly against the four rendered option
    texts."""
    for seed in SEEDS[:15]:
        q = _make_from_sources(difficulty, seed)
        opts = mcq_options(q)
        flipped = q.answer_text.replace(_LABEL_YES, "\0").replace(_LABEL_NO, _LABEL_YES).replace("\0", _LABEL_NO)
        assert flipped not in opts, (difficulty, seed, q.text, q.answer_text)


def test_exam_independent_check_fires_on_a_corrupted_second_data_point():
    """`_solve_two_point` is the exam tier's own independent-route check
    (solves the 2x2 system from the two data points rather than replaying
    the `fixed + rate*quantity` formula forward) -- exercised directly here
    with a corrupted second total, proving `check_equivalent` actually
    fires rather than merely being called."""
    from fractions import Fraction as F

    from mathgen.core import VerificationError, check_equivalent
    from mathgen.generators.p13a_direct_proportion_scale_build import _solve_two_point

    q1, t1 = 5, F(4000)  # fixed=1000, rate=600 (cents): t = 1000 + 600*q
    q2, t2 = 10, F(7000)
    solved_rate, solved_fixed = _solve_two_point(q1, t1, q2, t2)
    assert solved_rate == F(600) and solved_fixed == F(1000)

    corrupted_t2 = t2 + 1  # a one-cent nudge on the second data point
    corrupted_rate, corrupted_fixed = _solve_two_point(q1, t1, q2, corrupted_t2)
    with pytest.raises(VerificationError):
        check_equivalent(corrupted_fixed, solved_fixed, "must not match after corruption")
    with pytest.raises(VerificationError):
        check_equivalent(corrupted_rate, solved_rate, "must not match after corruption")


def test_medium_inverse_independent_check_fires_on_a_corrupted_total():
    """The medium-inverse tier's own independent route is a repeated-
    subtraction loop over `total - fixed` (never the division the
    construction itself used) -- corrupting the total by one cent must make
    it NOT land on an exact multiple of the rate after removing any fixed
    fee, which the generator's own `check()` call would catch (proved
    directly here rather than only via the real grader)."""
    from mathgen.core import VerificationError, check

    rate_cents = 350
    fixed_cents = 1000
    total_cents = Fraction(fixed_cents + rate_cents * 12)
    corrupted_total_cents = total_cents + 1

    remaining = corrupted_total_cents - fixed_cents
    count = 0
    while remaining > 0:
        remaining -= rate_cents
        count += 1
    with pytest.raises(VerificationError):
        check(remaining == 0, "medium inverse: total is not an exact multiple of the rate")


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_satisfies_its_own_declared_kind(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(difficulty, seed)
        assert isinstance(q.answer, str), (difficulty, seed, q.answer)
        assert q.answer == q.answer_text
        assert _OPTION_RE.match(q.answer_text), (difficulty, seed, q.answer_text)
        for d in q.distractors:
            assert _OPTION_RE.match(d), (difficulty, seed, d)


def test_no_resample_deadlock_over_seed_sweep():
    for difficulty in DIFFICULTIES:
        for seed in range(300):
            make(TOPIC, seed=seed, difficulty=difficulty)  # must not raise


def test_skill_loads_through_api():
    skill = _skills_dict()[SKILL]
    assert skill is not None
    assert skill.domain == "measurement"
    assert skill.prerequisites == ["ratio-rate-proportion", "f10-ratio-rate-financial-model"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


# =============================================================================
# The classification kernel -- now explicitly asked and graded
# =============================================================================


def test_every_tier_asks_the_classification_question_explicitly():
    for difficulty in DIFFICULTIES:
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "Is this relationship directly proportional, and" in q.text, (difficulty, seed, q.text)


def test_word_proportional_never_appears_outside_the_asked_question():
    """The word "proportional" (or "proportion") must appear ONLY inside the
    fixed classification-question clause this module bolts onto every stem
    -- never asserted elsewhere in the stem itself, and the mcq OPTIONS are
    a separate field from `q.text` entirely, so this checks the stem never
    tips the learner off ahead of the explicit question."""
    clause = "Is this relationship directly proportional, and"
    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            stripped = q.text.replace(clause, "")
            assert "proportion" not in stripped.lower(), (difficulty, seed, q.text)


def test_easy_never_states_a_fixed_component():
    """easy is the deliberate, disclosed pure_rate-only baseline tier: there
    is nothing else to compare against, so it must never mention a fixed/
    delivery/call-out/booking fee."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "fee" not in q.text.lower(), (seed, q.text)
        assert "plus" not in q.text.lower(), (seed, q.text)
        assert q.answer_text.startswith(_LABEL_YES), (seed, q.text, q.answer_text)


def test_easy_always_offers_four_yes_labelled_options():
    """easy's classification is a constant, disclosed "yes" -- every option (answer and all three distractors)
    carries the same label; only the computed total varies. This is the deliberate baseline, not an oversight."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        opts = mcq_options(q)
        assert len(opts) == 4
        labels = {_label_of(o) for o in opts}
        assert labels == {_LABEL_YES}, (seed, q.text, opts)


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_non_easy_tiers_offer_at_least_two_distinct_labels(difficulty):
    """The classification must be a LIVE decision from medium onward: at
    least one option must carry the opposite label from the correct one, so
    a learner reading the tier (rather than the scenario) cannot reliably
    guess the label."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        opts = mcq_options(q)
        labels = {_label_of(o) for o in opts}
        assert labels == {_LABEL_YES, _LABEL_NO}, (difficulty, seed, q.text, opts)


def test_exam_offers_at_least_two_distinct_labels():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        opts = mcq_options(q)
        labels = {_label_of(o) for o in opts}
        assert labels == {_LABEL_YES, _LABEL_NO}, (seed, q.text, opts)


def test_hard_rate_plus_fixed_draw_states_a_fixed_component_and_a_rate():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "plus" not in q.text:
            continue  # pure_rate draw
        assert re.search(r"\$\d+(\.\d{2})? (call-out fee|flat booking fee|a month)", q.text), (seed, q.text)


def test_hard_pure_rate_draw_never_states_a_fixed_component():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "plus" in q.text:
            continue  # rate_plus_fixed draw
        assert "fee" not in q.text.lower(), (seed, q.text)


def test_hard_omits_fixed_fee_distractor_present_on_rate_plus_fixed_draws():
    """The required distractor: `rate * quantity` alone, omitting the fixed
    fee, paired with the "Yes" label -- the textbook direct-proportion-vs-
    not error."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "plus" not in q.text:
            continue
        fixed = Fraction(re.search(r"\$(\d+\.\d{2}) (?:call-out fee|flat booking fee|a month)", q.text).group(1))
        rate = Fraction(re.search(r"\$(\d+\.\d{2}) per (?:hour|GB|day)", q.text).group(1))
        quantity = int(re.search(r"for (\d+) (?:hours|GB|days)", q.text).group(1))
        omits_fixed = rate * quantity
        expected_opt = f"{_LABEL_YES} -- the total cost is ${omits_fixed:.2f}."
        if expected_opt == q.answer_text:
            continue  # collision resampled away in the generator
        assert expected_opt in q.distractors, (seed, q.text, q.distractors, expected_opt)


def test_hard_applies_fixed_fee_per_unit_distractor_present_on_rate_plus_fixed_draws():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "plus" not in q.text:
            continue
        fixed = Fraction(re.search(r"\$(\d+\.\d{2}) (?:call-out fee|flat booking fee|a month)", q.text).group(1))
        rate = Fraction(re.search(r"\$(\d+\.\d{2}) per (?:hour|GB|day)", q.text).group(1))
        quantity = int(re.search(r"for (\d+) (?:hours|GB|days)", q.text).group(1))
        applies_per_unit = (fixed + rate) * quantity
        expected_opt = f"{_LABEL_NO} -- the total cost is ${applies_per_unit:.2f}."
        if expected_opt == q.answer_text:
            continue
        assert expected_opt in q.distractors, (seed, q.text, q.distractors, expected_opt)


def test_hard_and_medium_draw_both_kinds_over_400_seeds():
    for difficulty, marker_present in (("hard", lambda t: "plus" in t), ("medium", lambda t: "delivery" in t.lower())):
        fixed_count = 0
        pure_count = 0
        for seed in range(400):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if marker_present(q.text):
                fixed_count += 1
            else:
                pure_count += 1
        assert fixed_count > 100, (difficulty, fixed_count)
        assert pure_count > 100, (difficulty, pure_count)


# =============================================================================
# Medium tier: both variants, and both kinds within each variant
# =============================================================================


def _is_medium_inverse(text: str) -> bool:
    return "how many" in text.lower()


def test_medium_extract_variant_states_a_group_cost_never_a_per_unit_rate():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if _is_medium_inverse(q.text):
            continue
        assert "per" not in q.text, (seed, q.text)
        assert re.search(r"costs? \$\d+\.\d{2}", q.text), (seed, q.text)


def test_medium_inverse_variant_answer_is_a_plain_whole_number():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if not _is_medium_inverse(q.text):
            continue
        m = re.search(r"you can buy (\d+) [a-zA-Z]+\.$", q.answer_text)
        assert m, (seed, q.text, q.answer_text)


def test_medium_draws_both_variants_over_a_seed_sweep():
    extract_count = 0
    inverse_count = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if _is_medium_inverse(q.text):
            inverse_count += 1
        else:
            extract_count += 1
    assert extract_count > 30, extract_count
    assert inverse_count > 30, inverse_count


def test_medium_extract_variant_draws_both_kinds_over_a_seed_sweep():
    fixed_count = 0
    pure_count = 0
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if _is_medium_inverse(q.text):
            continue
        if "delivery" in q.text.lower():
            fixed_count += 1
        else:
            pure_count += 1
    assert fixed_count > 30, fixed_count
    assert pure_count > 30, pure_count


def test_medium_inverse_variant_draws_both_kinds_over_a_seed_sweep():
    fixed_count = 0
    pure_count = 0
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if not _is_medium_inverse(q.text):
            continue
        if "delivery" in q.text.lower():
            fixed_count += 1
        else:
            pure_count += 1
    assert fixed_count > 30, fixed_count
    assert pure_count > 30, pure_count


# =============================================================================
# Exam tier: two data points, no formula named
# =============================================================================


def test_exam_states_two_data_points_and_no_formula():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.text.count("the total is $") == 2, (seed, q.text)
        assert "rate" not in q.text.lower()
        assert "fee" not in q.text.lower()
        assert "fixed" not in q.text.lower()


def test_exam_draws_both_pure_and_fixed_kinds_over_a_seed_sweep():
    pure = 0
    fixed = 0
    pattern = re.compile(r"for (\d+) \S+ the total is \$(\d+\.\d{2}); for (\d+) \S+ the total is \$(\d+\.\d{2})")
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = pattern.search(q.text)
        q1, t1, q2, t2 = int(m.group(1)), Fraction(m.group(2)), int(m.group(3)), Fraction(m.group(4))
        rate = (t2 - t1) / (q2 - q1)
        implied_fixed = t1 - rate * q1
        if implied_fixed == 0:
            pure += 1
        else:
            fixed += 1
    assert pure > 50, pure
    assert fixed > 50, fixed


def test_exam_wrong_data_point_ratio_distractor_present_when_ratios_disagree():
    pattern = re.compile(
        r"for (\d+) \S+ the total is \$(\d+\.\d{2}); for (\d+) \S+ the total is \$(\d+\.\d{2}).*for (\d+) \S+\?"
    )
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        m = pattern.search(q.text)
        q1, t1, q2, t2, q3 = (
            int(m.group(1)),
            Fraction(m.group(2)),
            int(m.group(3)),
            Fraction(m.group(4)),
            int(m.group(5)),
        )
        rate = (t2 - t1) / (q2 - q1)
        implied_fixed = t1 - rate * q1
        if implied_fixed == 0:
            continue  # pure_rate: the two ratios agree, this distractor is not offered
        wrong = Fraction(t1, q1) * q3
        expected_opt = f"{_LABEL_YES} -- the total cost is ${round(wrong, 2):.2f}."
        if expected_opt == q.answer_text:
            continue
        assert expected_opt in q.distractors or any(
            abs(Fraction(re.search(r"\$(\d+\.\d{2})", d).group(1)) - wrong) <= Fraction(1, 200)
            and _label_of(d) == _LABEL_YES
            for d in q.distractors
        ), (seed, q.text, q.distractors, wrong)
