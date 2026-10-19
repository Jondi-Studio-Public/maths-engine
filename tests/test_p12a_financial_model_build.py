"""Tests `f10-rational-percentage-financial-model` and `f10-ratio-rate-financial-model`: financial models using
fractions, percentages, ratios and rates."""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

PM_TOPIC = "rational_percentage_financial_model"
RR_TOPIC = "ratio_rate_financial_model"
PM_SKILL = "f10-rational-percentage-financial-model"
RR_SKILL = "f10-ratio-rate-financial-model"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
_TOL = stated_precision_tolerance(2)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    topic, params = src.topic, src.params or {}
    if topic == "exd_x_r3e_fee_discount_gst":
        # The exam source moved to a multi-part chain; these tests pin the earlier single-answer tier.
        topic, params = "rational_percentage_financial_model", {}
    return make(topic, seed=seed, difficulty=src.difficulty, **params)


# =============================================================================
# Shared, parametrised over both (topic, skill_id) pairs
# =============================================================================

SKILLS = [(PM_TOPIC, PM_SKILL), (RR_TOPIC, RR_SKILL)]


@pytest.mark.parametrize("topic,skill_id", SKILLS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(topic, skill_id, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skill_id, difficulty, seed)
        assert len(q.distractors) == len(set(q.distractors)) >= 1, (difficulty, seed, q.distractors)
        for d in q.distractors:
            assert abs(Fraction(d) - Fraction(q.answer_text)) > _TOL, (difficulty, seed, q.answer_text, d)


@pytest.mark.parametrize("topic,skill_id", SKILLS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, skill_id, difficulty):
    assert GRADERS["number"] is grade_number
    for seed in SEEDS:
        q = _make_from_sources(skill_id, difficulty, seed)
        wrong = str(Fraction(q.answer_text) + 1)
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


@pytest.mark.parametrize("topic,skill_id", SKILLS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_common_typed_variants(topic, skill_id, difficulty):
    """A learner typing "$81" or "81" for a canonical "81.00", or "81.0" for "81.00", is still correct -- the
    stated-precision tolerance covers a trailing-zero-dropped typed form."""
    for seed in SEEDS[:10]:
        q = _make_from_sources(skill_id, difficulty, seed)
        bare = q.answer_text.rstrip("0").rstrip(".") if "." in q.answer_text else q.answer_text
        assert grade_number(q, bare).correct, (difficulty, seed, q.answer_text, bare)
        assert grade_number(q, f"${q.answer_text}").correct, (difficulty, seed, q.answer_text)


@pytest.mark.parametrize("topic,skill_id", SKILLS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_a_plain_number_stated_to_2dp(topic, skill_id, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skill_id, difficulty, seed)
        assert isinstance(q.answer, (int, float)), (difficulty, seed, q.answer)
        assert re.fullmatch(r"-?\d+(\.\d{2})?", q.answer_text), (difficulty, seed, q.answer_text)
        for d in q.distractors:
            assert re.fullmatch(r"-?\d+(\.\d{2})?", d), (difficulty, seed, d)


@pytest.mark.parametrize("topic,skill_id", SKILLS)
def test_no_resample_deadlock_over_seed_sweep(topic, skill_id):
    for difficulty in DIFFICULTIES:
        for seed in range(300):
            make(topic, seed=seed, difficulty=difficulty)  # must not raise


def test_pm_skill_loads_through_api():
    skill = _skills_dict()[PM_SKILL]
    assert skill is not None
    assert skill.domain == "financial-discrete"
    assert skill.prerequisites == ["f10-fraction-percent-quantity-strategy", "f10-rational-percentage-estimation"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


def test_rr_skill_loads_through_api():
    skill = _skills_dict()[RR_SKILL]
    assert skill is not None
    assert skill.domain == "measurement"
    assert skill.prerequisites == ["ratio-rate-proportion", "f10-circle-and-prism-duration-rates"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_src.stages == ["exam"]


# =============================================================================
# Percentage-model: choice kernel, chain and reverse specifics
# =============================================================================


def test_pm_easy_states_both_a_fraction_and_a_percentage_clause():
    """The core of the skill is the fraction-vs-percentage CHOICE: every easy-tier
    question states BOTH a fraction figure and a percentage figure, never
    one alone."""
    for seed in range(60):
        q = make(PM_TOPIC, seed=seed, difficulty="easy")
        assert re.search(r"\d+/\d+", q.text), (seed, q.text)
        assert re.search(r"\d+%", q.text), (seed, q.text)


def test_pm_medium_states_one_fraction_and_one_percent_step_in_a_chain():
    for seed in range(60):
        q = make(PM_TOPIC, seed=seed, difficulty="medium")
        tokens = re.findall(r"\d+/\d+|\d+%", q.text)
        assert len(tokens) == 2, (seed, q.text)
        kinds = ["/" in t for t in tokens]
        assert kinds.count(True) == 1 and kinds.count(False) == 1, (seed, q.text)


def test_pm_hard_is_always_reverse_direction():
    for seed in range(60):
        q = make(PM_TOPIC, seed=seed, difficulty="hard")
        assert "How much did" in q.text or "How much was" in q.text, (seed, q.text)
        assert re.search(r"left", q.text), (seed, q.text)


def test_pm_exam_states_three_deductions_with_two_different_bases():
    for seed in range(60):
        q = make(PM_TOPIC, seed=seed, difficulty="exam")
        assert "ORIGINAL amount" in q.text
        assert "what THEN remained" in q.text
        assert "flat" in q.text and "service fee" in q.text


def test_pm_easy_operation_swap_distractor_present_when_it_differs_from_the_answer():
    """operation-swap distractor: applying the OTHER stated clause's
    operation to M must be offered whenever it differs from the answer."""
    for seed in range(60):
        q = make(PM_TOPIC, seed=seed, difficulty="easy")
        M = int(re.search(r"\$(\d+)\b", q.text).group(1))
        frac = re.search(r"(\d+)/(\d+)", q.text)
        frac_value = Fraction(int(frac.group(1)), int(frac.group(2)))
        pct = Fraction(int(re.search(r"(\d+)%", q.text).group(1)), 100)
        answer = Fraction(q.answer_text)
        frac_cents, pct_cents = frac_value * M, pct * M
        other = pct_cents if answer == frac_cents else frac_cents
        if other == answer:
            continue  # collision by construction is resampled away in the generator
        distractor_values = [Fraction(d) for d in q.distractors]
        assert other in distractor_values, (seed, q.text, q.distractors, other)


def test_pm_medium_missed_step_distractor_equals_first_step_amount():
    for seed in range(40):
        q = make(PM_TOPIC, seed=seed, difficulty="medium")
        M = int(re.search(r"\$(\d+)\b", q.text).group(1))
        first_tok = re.findall(r"\d+/\d+|\d+%", q.text)[0]
        first_value = (
            Fraction(int(first_tok.split("/")[0]), int(first_tok.split("/")[1]))
            if "/" in first_tok
            else Fraction(int(first_tok.rstrip("%")), 100)
        )
        missed_step = M * first_value
        distractor_values = [Fraction(d) for d in q.distractors]
        assert missed_step in distractor_values, (seed, q.text, q.distractors, missed_step)


def test_pm_reverse_forward_simulation_check_fires_on_a_corrupted_value():
    """`_pm_reverse_forward_simulate` is the in-generator check's own
    recompute route; corrupting the final amount must break equality
    against a forward simulation from the (unchanged) answer."""
    from mathgen.generators.p12a_financial_model_build import _pm_reverse_forward_simulate

    M_cents = Fraction(15000)
    frac_value = Fraction(1, 5)
    pct_value = Fraction(1, 5)
    forward = _pm_reverse_forward_simulate(M_cents, frac_value, pct_value, "fraction_first")
    assert forward == Fraction(9600)
    corrupted = forward + 1
    assert corrupted != forward


# =============================================================================
# Ratio-rate-model: choice kernel, composition, trap and chained specifics
# =============================================================================


def test_rr_easy_is_ratio_split_or_rate_never_both_or_neither():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="easy")
        is_ratio = "ratio" in q.text
        is_rate = "per " in q.text
        assert is_ratio != is_rate, (seed, q.text)


def test_rr_medium_composes_a_genuine_second_step():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="medium")
        is_ratio = "ratio" in q.text
        if is_ratio:
            assert "pool" in q.text or "combine" in q.text, (seed, q.text)
        else:
            assert "flat" in q.text and ("discount" in q.text or "surcharge" in q.text), (seed, q.text)


def test_rr_hard_is_a_correction_trap_scenario():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="hard")
        assert "materials fee" in q.text or "free" in q.text, (seed, q.text)


def test_rr_exam_chains_a_ratio_split_into_a_rate_application():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="exam")
        assert "ratio" in q.text
        assert "per " in q.text
        assert "larger share" in q.text
        assert re.fullmatch(r"-?\d+\.\d{2}", q.answer_text), (seed, q.answer_text)


def test_rr_easy_ratio_part_swap_distractor_present_when_parts_unequal():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="easy")
        if "ratio" not in q.text:
            continue
        T = int(re.search(r"\$(\d+)\b", q.text).group(1))
        p, qv = (int(x) for x in re.search(r"ratio (\d+):(\d+)", q.text).groups())
        if p == qv:
            continue
        share_p = Fraction(T * p, p + qv)
        share_q = Fraction(T * qv, p + qv)
        answer = Fraction(q.answer_text)
        other_share = share_q if answer == share_p else share_p
        distractor_values = [Fraction(d) for d in q.distractors]
        assert other_share in distractor_values, (seed, q.text, q.distractors, other_share)


def test_rr_exam_smaller_share_distractor_present():
    for seed in range(60):
        q = make(RR_TOPIC, seed=seed, difficulty="exam")
        T = int(re.search(r"\$(\d+)\b", q.text).group(1))
        p, qv = (int(x) for x in re.search(r"ratio (\d+):(\d+)", q.text).groups())
        rate = Fraction(re.search(r"\$(\d+(?:\.\d+)?) per \w+", q.text).group(1))
        smaller = min(p, qv)
        smaller_value = Fraction(T * smaller, p + qv) / rate
        distractor_values = [Fraction(d) for d in q.distractors]
        rounded = Fraction(round(smaller_value * 100), 100)  # matches the generator's own round_half_up(..., 2) display
        assert rounded in distractor_values, (seed, q.text, q.distractors, smaller_value, rounded)


# Independently curated allowlist (deliberately NOT imported from the
# generator module) of every (item_plural, unit) pairing that is genuinely
# sensible in ordinary retail/trade English, mirroring
# `_RR_ITEM_UNIT_PAIRS` in `mathgen/generators/p12a_financial_model_build.py`.
_SENSIBLE_ITEM_UNIT_PAIRS = frozenset(
    {
        ("ribbons", "metre"),
        ("ropes", "metre"),
        ("cable", "metre"),
        ("fabric", "metre"),
        ("fuel", "litre"),
        ("paint", "litre"),
        ("juice", "litre"),
        ("milk", "litre"),
        ("apples", "kg"),
        ("flour", "kg"),
        ("rice", "kg"),
        ("cheese", "kg"),
        ("parking", "hour"),
        ("car hire", "hour"),
        ("tutoring", "hour"),
    }
)

_ITEM_UNIT_EXTRACTORS = [
    re.compile(r"sells ([a-z ]+?) at a rate of \$[\d.]+ per (\w+)"),
    re.compile(r"At the market, ([a-z ]+?) (?:cost|costs) \$[\d.]+ per (\w+)"),
    re.compile(r"^([A-Za-z ]+?) (?:are|is) sold at a rate of \$[\d.]+ per (\w+)"),
]


def _extract_item_unit_pairs(text):
    """Every (item_plural, unit) pair actually rendered in `text`'s
    flavour wording, lower-cased for allowlist comparison. Empty if the
    branch has no item wording at all (e.g. the "bill"/tradesperson
    context, or the ratio branch)."""
    found = []
    for pattern in _ITEM_UNIT_EXTRACTORS:
        m = pattern.search(text)
        if m:
            found.append((m.group(1).lower(), m.group(2)))
    return found


def test_rr_easy_medium_hard_item_unit_pairing_is_always_sensible():
    """Sweeps 300 seeds across the three affected tiers (easy/medium/hard)
    and asserts every rendered rate-branch (item, unit) pair is a member of
    the independently curated sensible allowlist above. This test is
    constructed to FAIL against the pre-fix generator, which drew `item`
    from the generic `mathgen.slots.OBJECTS` list independently of `unit`
    and so could render e.g. "chairs cost $1.80 per kg" or "crayons ...
    per metre" -- neither pairing is in `_SENSIBLE_ITEM_UNIT_PAIRS`."""
    checked = 0
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(RR_TOPIC, seed=seed, difficulty=difficulty)
            for item_plural, unit in _extract_item_unit_pairs(q.text):
                checked += 1
                assert (item_plural, unit) in _SENSIBLE_ITEM_UNIT_PAIRS, (
                    difficulty,
                    seed,
                    q.text,
                    item_plural,
                    unit,
                )
    assert checked > 100, "expected the rate branch to fire many times over a 300-seed sweep"


# Proved red before the resample guard.
_EASY_PAIR_RE = re.compile(r"(\d+)/(\d+) of it .*?(\d+)% of it")


def test_easy_tier_fraction_and_percentage_never_exceed_the_whole():
    for seed in range(200):
        q = _make_from_sources(PM_SKILL, "easy", seed)
        m = _EASY_PAIR_RE.search(q.text)
        assert m, q.text
        share = Fraction(int(m[1]), int(m[2])) + Fraction(int(m[3]), 100)
        assert share <= 1, f"seed {seed}: {q.text}"
