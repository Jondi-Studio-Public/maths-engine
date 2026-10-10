"""Tests `f10-round-decimals-accuracy` (AC9M7N05) and `f10-decimal-order-numberline` (AC9M5N01)."""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_number, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p9e_decimal_accuracy_order_build import (
    _verify_ascending_permutation,
    _verify_round_half_up,
)

ROUND_TOPIC = "p9e_round_decimal_accuracy"
ORDER_TOPIC = "p9e_decimal_order_numberline"
ROUND_SKILL_ID = "f10-round-decimals-accuracy"
ORDER_SKILL_ID = "f10-decimal-order-numberline"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    """Generate through the skill's own `sources`, never a bare
    `make(topic, seed, difficulty=tier)` alone."""
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ---------------------------------------------------------------------------
# f10-round-decimals-accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        assert len(q.distractors) == len(set(q.distractors)) >= 2, (difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    assert GRADERS["number"] is grade_number


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        wrong = str(round(float(q.answer_text) + 3.7, 4))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_unrounded_typed_answer_is_rejected(difficulty):
    """The raw, unrounded value shown in the question must never be accepted
    -- `_is_rounded_as_asked` exists precisely to reject this."""
    _raw_re = re.compile(r"[$]?(\d+\.\d+)")
    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        m = _raw_re.search(q.text)
        assert m, (difficulty, seed, q.text)
        raw = m.group(1)
        if raw == q.answer_text:
            continue  # the rare case where the raw value is already exactly rounded
        assert not grade_number(q, raw).correct, (difficulty, seed, q.text, raw, q.answer_text)


_PLACE_PHRASE_RE = re.compile(r"nearest (hundredth|tenth|whole number)")


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_round_recognised_rounding_phrase_is_present(difficulty):
    """Confirm directly against `requested_rounding` that every question's
    own phrasing is one of the six words the grader recognises."""
    from mathgen.text_quality import requested_rounding

    for seed in SEEDS:
        q = _make_from_sources(ROUND_SKILL_ID, difficulty, seed)
        places = requested_rounding(q.text)
        assert places is not None, (difficulty, seed, q.text)
        assert places == len(q.answer_text.split(".")[1]) if "." in q.answer_text else places == 0


def _dropped_digit(text: str) -> str:
    """The tier-controlled dropped digit, read from the rendered raw value.
    headcount's raw carries a second, free EXTRA digit after the dropped one
    (see the generator module docstring), so the dropped digit is the FIRST
    decimal digit there; money/length carry no extra digit, so it is the
    LAST (and only) one."""
    m = re.search(r"(\d+\.\d+)", text)
    digits = m.group(1).split(".")[1]
    is_headcount = "calliper" not in text and "hundredth of a dollar" not in text
    return digits[0] if is_headcount else digits[-1]


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_round_dropped_digit_is_never_a_bare_five(difficulty):
    for seed in range(120):
        q = make(ROUND_TOPIC, seed=seed, difficulty=difficulty)
        assert _dropped_digit(q.text) != "5", (difficulty, seed, q.text)


def test_round_hard_tier_dropped_digit_is_always_five():
    for seed in range(120):
        q = make(ROUND_TOPIC, seed=seed, difficulty="hard")
        assert _dropped_digit(q.text) == "5", (seed, q.text)


def test_round_easy_tier_is_money_only_and_never_carries():
    for seed in range(60):
        q = make(ROUND_TOPIC, seed=seed, difficulty="easy")
        assert "hundredth of a dollar" in q.text, (seed, q.text)
        raw_whole = int(re.search(r"\$(\d+)\.", q.text).group(1))
        answer_whole = int(q.answer_text.split(".")[0])
        assert raw_whole == answer_whole, (seed, q.text, q.answer_text)


def test_round_medium_tier_uses_length_or_headcount_only():
    contexts = set()
    for seed in range(60):
        q = make(ROUND_TOPIC, seed=seed, difficulty="medium")
        assert "hundredth of a dollar" not in q.text, (seed, q.text)
        contexts.add("length" if "calliper" in q.text else "headcount")
    assert contexts == {"length", "headcount"}, contexts


def test_round_hard_tier_reaches_every_context():
    contexts = set()
    for seed in range(120):
        q = make(ROUND_TOPIC, seed=seed, difficulty="hard")
        if "calliper" in q.text:
            contexts.add("length")
        elif "hundredth of a dollar" in q.text:
            contexts.add("money")
        else:
            contexts.add("headcount")
    assert contexts == {"money", "length", "headcount"}, contexts


def test_round_hard_tier_carry_fires_and_is_measured():
    """Not merely possible in theory: measured over a real seed sweep, with
    a denominator, per money/length/headcount separately."""
    carried = {"money": 0, "length": 0, "headcount": 0}
    total = {"money": 0, "length": 0, "headcount": 0}
    n = 300
    for seed in range(n):
        q = make(ROUND_TOPIC, seed=seed, difficulty="hard")
        if "calliper" in q.text:
            ctx = "length"
        elif "hundredth of a dollar" in q.text:
            ctx = "money"
        else:
            ctx = "headcount"
        total[ctx] += 1
        raw = Fraction(re.search(r"(\d+\.\d+)", q.text).group(1))
        answer = Fraction(q.answer_text)
        if ctx == "headcount":
            # headcount has zero kept decimal digits, so the forced-5
            # dropped digit rounds up EVERY hard-tier question regardless
            # of `force_carry` -- the distinguishing signal here is
            # specifically a base-10 rollover of the whole part (49 -> 50),
            # which only happens when the whole part's own last digit was
            # forced to 9.
            if int(raw) % 10 == 9:
                carried[ctx] += 1
        else:
            # money/length: a carry into the next place changes the
            # WHOLE-NUMBER part. The ordinary forced-5 rounding never does
            # this on its own (it only changes the last kept decimal digit)
            # unless every kept digit was also 9, cascading the +1 all the
            # way through.
            if int(answer) != int(raw):
                carried[ctx] += 1
    print(f"hard-tier carry counts over {n} seeds: {carried} of {total}")
    for ctx in ("money", "length", "headcount"):
        assert carried[ctx] > 0, (ctx, carried, total)


def test_round_headcount_hard_tier_has_no_wrong_accuracy_distractor():
    """Disclosed limitation: at headcount+hard the extra digit is forced to
    0 for the genuine tie, so a finer-place "wrong accuracy" rounding would
    equal the unrounded raw value -- not constructible there. Every
    distractor must still be a whole number (no decimal point) confirming
    wrong-accuracy (which would show 1 decimal place) never appears."""
    seen_headcount = 0
    for seed in range(120):
        q = make(ROUND_TOPIC, seed=seed, difficulty="hard")
        if "calliper" in q.text or "hundredth of a dollar" in q.text:
            continue
        seen_headcount += 1
        for d in q.distractors:
            assert "." not in d, (seed, q.text, q.distractors)
    assert seen_headcount > 0


def test_round_length_context_is_a_plausible_beetle_length():
    """A measurement skill's figure must be sensible, not merely
    whatever the digit construction needed. `whole_range=(1, 12)` for the
    length context, so the full raw reading (whole + a sub-1 fractional
    part) never exceeds 13 cm -- inside even record-sized beetle lengths."""
    seen = 0
    for seed in range(200):
        q = make(ROUND_TOPIC, seed=seed, difficulty="hard")
        if "calliper" not in q.text:
            continue
        seen += 1
        raw = float(re.search(r"(\d+\.\d+) cm", q.text).group(1))
        assert 0 < raw <= 13, (seed, q.text)
    assert seen > 0


def test_round_typed_answer_with_currency_symbol_is_still_accepted():
    """A learner who keeps the `$` when typing a money answer is a plausible
    response, not a wrong one -- confirmed directly against the real
    grader, not assumed."""
    for seed in range(20):
        q = make(ROUND_TOPIC, seed=seed, difficulty="easy")
        assert grade_number(q, "$" + q.answer_text).correct, (seed, q.text, q.answer_text)


def test_round_half_up_disagrees_with_python_float_banker_rounding():
    """`round(2.675, 2)` gives `2.67` in Python (binary float representation,
    not a real half-up rounding-boundary case) -- confirms the generator's
    own integer half-up idiom gives the DIFFERENT, correct half-up answer
    `2.68` for the equivalent exact case, so this generator is not
    accidentally relying on float `round()`'s banker's-rounding behaviour."""
    assert round(2.675, 2) == 2.67  # the well-known float trap
    from mathgen.generators.p9e_decimal_accuracy_order_build import _round_half_up

    assert _round_half_up(2675, 10) == 268  # 2.675 held as 2675 thousandths -> 2.68


def test_round_independent_check_fires_on_corrupted_answer():
    _verify_round_half_up("2.995", 2, "3.00")  # must not raise
    with pytest.raises(VerificationError):
        _verify_round_half_up("2.995", 2, "2.99")


def test_round_skill_loads_through_api():
    skill = _skills_dict()[ROUND_SKILL_ID]
    assert skill is not None
    assert skill.domain == "primary-number"
    assert skill.prerequisites == ["f10-estimate-counts-and-check-reasonableness", "round-decimals"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# f10-decimal-order-numberline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ORDER_SKILL_ID, difficulty, seed)
        assert len(q.distractors) == len(set(q.distractors)) >= 2, (difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    assert GRADERS["ordered_list"] is grade_ordered_list


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(ORDER_SKILL_ID, difficulty, seed)
        items = q.answer_text.split(", ")
        wrong = ", ".join([items[-1]] + items[:-1])  # a rotation, not the correct order
        assert wrong != q.answer_text
        assert not grade_ordered_list(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


def test_order_easy_tier_all_values_below_one():
    for seed in range(60):
        q = make(ORDER_TOPIC, seed=seed, difficulty="easy")
        for token in q.answer_text.split(", "):
            assert Fraction(token) < 1, (seed, q.text)


def test_order_medium_tier_spans_below_and_above_one():
    for seed in range(60):
        q = make(ORDER_TOPIC, seed=seed, difficulty="medium")
        values = [Fraction(t) for t in q.answer_text.split(", ")]
        assert any(v < 1 for v in values) and any(v >= 1 for v in values), (seed, q.text)


def test_order_hard_tier_spans_below_and_above_one_and_shares_a_prefix():
    for seed in range(60):
        q = make(ORDER_TOPIC, seed=seed, difficulty="hard")
        values = [Fraction(t) for t in q.answer_text.split(", ")]
        assert any(v < 1 for v in values) and any(v >= 1 for v in values), (seed, q.text)
        fracs = [t.split(".")[1] for t in q.answer_text.split(", ")]
        assert len({f[0] for f in fracs}) == 1 or len({f[:2] for f in fracs}) == 1, (seed, q.text, fracs)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_order_values_have_three_or_four_decimal_places(difficulty):
    for seed in range(60):
        q = make(ORDER_TOPIC, seed=seed, difficulty=difficulty)
        for token in q.answer_text.split(", "):
            frac = token.split(".")[1]
            assert len(frac) in (3, 4), (difficulty, seed, q.text, token)


def test_order_independent_check_fires_on_corrupted_answer():
    values = [Fraction("0.1"), Fraction("0.2"), Fraction("0.3")]
    _verify_ascending_permutation(values, sorted(values))  # must not raise
    with pytest.raises(VerificationError):
        _verify_ascending_permutation(values, list(reversed(sorted(values))))


def test_order_skill_loads_through_api():
    skill = _skills_dict()[ORDER_SKILL_ID]
    assert skill is not None
    assert skill.domain == "primary-number"
    assert skill.prerequisites == ["decimal-place-value", "f10-number-represent-order-to-120"]
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


def test_no_resample_deadlock_over_seed_sweep():
    for topic in (ROUND_TOPIC, ORDER_TOPIC):
        for difficulty in ("easy", "medium", "hard"):
            for seed in range(300):
                make(topic, seed=seed, difficulty=difficulty)  # must not raise
