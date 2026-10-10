"""Tests the volume-to-capacity conversion skills `p19h_volcap_general_relationship` and
`p19h_volcap_named_equivalence`.

Senior volume-to-capacity conversion, general relationship + named equivalence
(senior-understand-the-relationship-between-volume-and-capacity,
senior-recognise-relations-between-volume-and-capacity-recognising-that)."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
WIDE_SEEDS = range(60)
CONCENTRATION_SEEDS = range(200)

VC1_TOPIC = "p19h_volcap_general_relationship"
VC2_TOPIC = "p19h_volcap_named_equivalence"
TIERS = ["easy", "medium", "hard", "exam"]


# --- determinism / distractor hygiene, both topics ----------------------------


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) == 3


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_and_distractors_are_always_bare_whole_numbers(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        for opt in [question.answer_text] + question.distractors:
            assert "." not in opt, opt
            int(opt)


# --- real grader accept/reject, both topics -----------------------------------


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong = int(question.answer_text) + 1
        assert not grade_number(question, str(wrong)).correct


# --- variety, both topics -----------------------------------------------------


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in WIDE_SEEDS}
    assert len(seen) >= 20, f"{topic}/{difficulty}: only {len(seen)} distinct questions over 60 seeds"


# --- skill 1 target-unit/direction boundary: never kL, never reverse ----------


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc1_never_mentions_kilolitres(difficulty):
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty=difficulty)
        assert "kL" not in question.text and "kilolitre" not in question.text.lower()


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc1_never_asks_for_the_reverse_direction(difficulty):
    """Every vc1 question gives a volume and asks for a capacity -- never the
    reverse ("has a capacity of ... what is its volume")."""
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty=difficulty)
        assert "what is its volume" not in question.text.lower()
        assert "what is their combined" not in question.text.lower()


def test_vc1_hard_targets_millilitres():
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty="hard")
        assert "millilitres (mL)" in question.text


def test_vc1_easy_targets_millilitres_or_litres_never_kilolitres():
    """Easy is no longer mL throughout -- part of the draws need the real 1000 cm3 = 1 L factor."""
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty="easy")
        assert "millilitres (mL)" in question.text or "litres (L)" in question.text


@pytest.mark.parametrize("difficulty", ["medium", "exam"])
def test_vc1_targets_litres_never_states_the_1000_factor(difficulty):
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty=difficulty)
        assert "litres (L)" in question.text
        assert "1000" not in question.text and "1,000" not in question.text


def test_vc1_easy_numeral_unchanged_only_for_the_ml_draws():
    """This test used to assert the key is in the prompt for EVERY easy draw (an earlier defect: 200/200 echo). Now
    only the cm3 -> mL kernel draws echo; the litres draws divide by 1000."""
    for seed in SEEDS:
        question = make(VC1_TOPIC, seed=seed, difficulty="easy")
        volume = int(re.search(r"(\d+) cm³", question.text).group(1))
        if "millilitres (mL)" in question.text:
            assert int(question.answer_text) == volume
        else:
            assert int(question.answer_text) * 1000 == volume


def test_vc1_hard_and_exam_state_three_dimensions():
    for seed in SEEDS:
        hard_q = make(VC1_TOPIC, seed=seed, difficulty="hard")
        assert hard_q.text.count(" cm") >= 3
        exam_q = make(VC1_TOPIC, seed=seed, difficulty="exam")
        assert exam_q.text.count(" m") >= 3


# --- skill 2 target-unit/direction boundary: never L, both directions --------


@pytest.mark.parametrize("difficulty", TIERS)
def test_vc2_never_mentions_bare_litres_as_a_target(difficulty):
    for seed in SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty=difficulty)
        assert "litres (L)" not in question.text
        assert "in litres" not in question.text.lower()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_vc2_states_at_least_one_named_equivalence(difficulty):
    for seed in SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty=difficulty)
        assert "1 cm³ = 1 mL" in question.text or "1 m³ = 1 kL" in question.text


def test_vc2_easy_numeral_never_changes():
    for seed in SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty="easy")
        assert question.answer_text in question.text


def test_vc2_medium_both_directions_represented():
    forward = reverse = 0
    for seed in WIDE_SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty="medium")
        if "what is its capacity in kilolitres" in question.text:
            forward += 1
        elif "what is its volume in cubic metres" in question.text:
            reverse += 1
        else:
            raise AssertionError(question.text)
    assert forward > 0 and reverse > 0, f"forward={forward}, reverse={reverse}"


def test_vc2_hard_both_directions_represented():
    forward = reverse = 0
    for seed in WIDE_SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty="hard")
        if "in kilolitres (kL)" in question.text:
            forward += 1
        elif "in cubic centimetres (cm³)" in question.text:
            reverse += 1
        elif "in millilitres (mL)" in question.text:  # M3 -> mL form
            forward += 1
        else:
            raise AssertionError(question.text)
    assert forward > 0 and reverse > 0, f"forward={forward}, reverse={reverse}"


def test_vc2_hard_states_both_named_facts():
    for seed in SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty="hard")
        assert "1 cm³ = 1 mL" in question.text
        assert "1 m³ = 1 kL" in question.text


def test_vc2_exam_two_containers_both_target_units_represented():
    ml_target = kl_target = 0
    for seed in WIDE_SEEDS:
        question = make(VC2_TOPIC, seed=seed, difficulty="exam")
        assert "Container A" in question.text and "Container B" in question.text
        if "combined capacity in millilitres" in question.text:
            ml_target += 1
        elif "combined capacity in kilolitres" in question.text:
            kl_target += 1
        else:
            raise AssertionError(question.text)
    assert ml_target > 0 and kl_target > 0, f"mL={ml_target}, kL={kl_target}"


def test_vc2_never_computes_a_volume_from_dimensions():
    """Every vc2 question gives a volume/capacity DIRECTLY -- never three
    dimensions to multiply (that step belongs only to the sibling skill)."""
    for difficulty in TIERS:
        for seed in SEEDS:
            question = make(VC2_TOPIC, seed=seed, difficulty=difficulty)
            assert " by " not in question.text, question.text


def test_vc1_never_states_a_named_equivalence():
    """vc1 never restates '1 cm3 = 1 mL' or '1 m3 = 1 kL' as a given fact --
    the sibling skill's own kernel, not this one's."""
    for difficulty in TIERS:
        for seed in SEEDS:
            question = make(VC1_TOPIC, seed=seed, difficulty=difficulty)
            assert "using the fact" not in question.text.lower()


# --- hard/exam side-by-side: the duplicate-risk table's own separation ------


def test_vc1_hard_and_vc2_hard_never_ask_numerically_identical_questions():
    """The duplicate-risk table's own bar: vc1's hard tier always requires a
    three-dimension multiplication vc2's hard tier never asks for, and vc2's
    hard tier always requires the x1,000,000 mL<->kL ladder vc1's hard tier
    never asks for (vc1's hard-tier conversion is always the trivial 1:1)."""
    for seed in SEEDS:
        vc1_q = make(VC1_TOPIC, seed=seed, difficulty="hard")
        vc2_q = make(VC2_TOPIC, seed=seed, difficulty="hard")
        assert " by " in vc1_q.text  # three dimensions, always multiplied
        assert " by " not in vc2_q.text  # a single given volume, never dimensions
        assert "kL" not in vc1_q.text
        assert "1 m³ = 1 kL" in vc2_q.text


# --- answer concentration @200 seeds, per tier, both topics ------------------


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_under_15_percent_at_200_seeds(topic, difficulty):
    from collections import Counter

    answers = [make(topic, seed=s, difficulty=difficulty).answer_text for s in CONCENTRATION_SEEDS]
    counts = Counter(answers)
    top_key, top_count = counts.most_common(1)[0]
    assert top_count / len(answers) < 0.15, (
        f"{topic}/{difficulty}: top answer {top_key!r} is {top_count}/{len(answers)} = {top_count / len(answers):.1%}"
    )


# Medium's key is no longer the prompt's own bare numeral -----


def test_vc2_medium_answer_is_not_a_bare_stated_numeral_over_200_seeds():
    """The fix states TWO separate quantities and sums them, so the answer should equal one of the stated
    numerals only by the rarest fluke (never, since both quantities are always >= 1)."""
    bad = 0
    for seed in range(200):
        question = make(VC2_TOPIC, seed=seed, difficulty="medium")
        stated_numbers = re.findall(r"\d+", question.text)
        if question.answer_text in stated_numbers:
            bad += 1
    assert bad == 0, (
        f"medium: answer_text matched a bare stated numeral in {bad}/200 draws "
        f"(the original defect rate was 200/200; fixed rate must be far below that)"
    )


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_cubic_units_are_superscripted_never_bare_3_after_cm_or_m(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert not re.search(r"\bcm3\b|\bm3\b", question.text), (
            f"{topic}/{difficulty}/{seed}: bare 'cm3'/'m3' in text: {question.text!r}"
        )
        # vc1 hard/exam state raw dimensions ("27 cm by 27 cm by 27 cm") with
        # no cubic-unit phrase in the prompt text at all (the volume itself
        # is computed, never stated) -- superscript only applies where a
        # cubic unit is actually displayed.
        if not (topic == VC1_TOPIC and difficulty in ("hard", "exam")):
            assert "³" in question.text, (
                f"{topic}/{difficulty}/{seed}: expected a superscripted cubic unit in text: {question.text!r}"
            )
        for w in question.working:
            assert not re.search(r"\bcm3\b|\bm3\b", w), (
                f"{topic}/{difficulty}/{seed}: bare 'cm3'/'m3' in working: {w!r}"
            )


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
def test_latex_field_wraps_text_in_proper_latex(topic):
    for difficulty in TIERS:
        question = make(topic, seed=1, difficulty=difficulty)
        assert question.latex == rf"\text{{{question.text}}}"


# --- independent in-generator check fires on a corrupted value ---------------


def test_independent_check_fires_on_corrupted_value():
    with pytest.raises(VerificationError):
        check(999 == 5000, "independent check should have failed")


def test_raises_on_unknown_mode_both_topics():
    with pytest.raises(ValueError):
        make(VC1_TOPIC, seed=1, mode="unknown")
    with pytest.raises(ValueError):
        make(VC2_TOPIC, seed=1, mode="unknown")


# --- skill loading -------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id",
    [
        "senior-understand-the-relationship-between-volume-and-capacity",
        "senior-recognise-relations-between-volume-and-capacity-recognising-that",
    ],
)
def test_skill_loads_with_expected_properties(skill_id):
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == skill_id), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 4
    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert len(exam_sources) == 1
    assert exam_sources[0].stages == ["exam"]


# Fixes -------------- Swept (300 seeds per tier), so
# each fails on the earlier tree.

FIX_SEEDS = range(300)


def _sorted_rank(question) -> int:
    options = sorted(int(x) for x in [question.answer_text] + list(question.distractors))
    return options.index(int(question.answer_text))


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_key_rank_among_sorted_options_is_not_constant(topic, difficulty):
    """No rank (0 = smallest) may hold more than 45% of draws in any tier: the
    an earlier review measured a constant rank (200/200) in six of the eight sets."""
    from collections import Counter

    ranks = Counter(_sorted_rank(make(topic, seed=s, difficulty=difficulty)) for s in FIX_SEEDS)
    rank, count = ranks.most_common(1)[0]
    assert count / len(FIX_SEEDS) <= 0.45, (
        f"{topic}/{difficulty}: rank {rank} in {count}/{len(FIX_SEEDS)} {dict(ranks)}"
    )


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
def test_easy_key_is_not_the_smallest_option_every_time(topic):
    smallest = sum(_sorted_rank(make(topic, seed=s, difficulty="easy")) == 0 for s in FIX_SEEDS)
    assert smallest < len(FIX_SEEDS) * 0.45, smallest


def test_vc1_easy_involves_the_real_factor_in_a_real_share_of_draws():
    """The relationship skill's easy tier may not be a 1:1 echo throughout: at
    least 30% of draws need the 1000 cm3 = 1 L factor, so the key is not the
    prompt's own number (review: 200/200 echo)."""
    not_echo = 0
    for s in FIX_SEEDS:
        q = make(VC1_TOPIC, seed=s, difficulty="easy")
        if q.answer_text not in re.findall(r"\d+", q.text):
            not_echo += 1
            assert "litres (L)" in q.text
    assert not_echo >= len(FIX_SEEDS) * 0.30, not_echo


def _numbers(question):
    return [int(x) for x in [question.answer_text] + list(question.distractors)]


def test_easy_distractors_include_divide_by_1000_and_divide_by_100_misconceptions():
    """vc1/vc2 easy cm3 -> mL: the litres-number slip (key/1000) and the
    divide-by-100 slip (key/100) both appear, not just x10^k shifts upward."""
    for topic in (VC1_TOPIC, VC2_TOPIC):
        by_1000 = by_100 = mL_form = 0
        for s in FIX_SEEDS:
            q = make(topic, seed=s, difficulty="easy")
            if "millilitres (mL)" not in q.text:
                continue
            mL_form += 1
            key = int(q.answer_text)
            options = _numbers(q)
            by_1000 += key // 1000 in options
            by_100 += key // 100 in options
        assert mL_form >= 100, (topic, mL_form)
        assert by_1000 >= mL_form * 0.4 and by_100 >= mL_form * 0.4, (topic, by_1000, by_100, mL_form)


def test_vc1_medium_and_exam_include_the_100_litre_per_cubic_metre_misconception():
    for tier in ("medium", "exam"):
        hits = 0
        for s in FIX_SEEDS:
            q = make(VC1_TOPIC, seed=s, difficulty=tier)
            key = int(q.answer_text)
            hits += key // 10 in _numbers(q)  # 1 m3 = 100 L
        assert hits >= len(FIX_SEEDS) * 0.25, (tier, hits)


def test_vc2_medium_includes_the_mixed_unit_slip():
    """Convert one addend by x1000 (m3 read as L) before adding: the option is
    above the key, which is what breaks the 'key is the largest' tell."""
    hits = 0
    for s in FIX_SEEDS:
        q = make(VC2_TOPIC, seed=s, difficulty="medium")
        a, b = (int(x) for x in re.findall(r"(\d+) (?:m³|kL)", q.text)[:2])
        options = _numbers(q)
        hits += (1000 * a + b in options) or (a + 1000 * b in options)
    assert hits >= len(FIX_SEEDS) * 0.4, hits


# Litre band each named container can physically hold (lower, upper).
BANDS_L = {
    "jug": (1, 3),
    "kettle": (1, 2),
    "bucket": (5, 15),
    "watering can": (5, 12),
    "drink dispenser": (5, 20),
    "fish tank": (20, 200),
    "aquarium display tank": (20000, 200000),
    "aquarium": (20, 200),
    "plastic tub": (10, 60),
    "open container": (5, 100),
    "water tank": (1000, 500000),
    "rainwater tank": (1000, 30000),
    "garden pond": (2000, 20000),
    "swimming pool": (25000, 150000),
    "irrigation tank": (5000, 50000),
    "header tank": (1000, 15000),
    "fuel storage tank": (20000, 150000),
    "town water reservoir": (100000, 400000),
}
# A name may be listed under several scales (a fish tank is 100 L, an aquarium
# display 100,000 L); the band used is the widest listed for that name.
NOT_LIQUID_CONTAINERS = (
    "concrete block",
    "crate",
    "storage box",
    "ingot",
    "pencil case",
    "lunchbox",
    "coffee mug",
    "cooler",
    "shipping container",
    "storage room",
    "storage silo",
    "flower pot",
    "paint tin",
    "fish bowl",
)


def _named_context(text: str) -> str:
    lowered = text.lower()
    found = [name for name in sorted(BANDS_L, key=len, reverse=True) if name in lowered]
    return found[0] if found else ""


def _litres_stated(topic, difficulty, question) -> float | None:
    text = question.text
    if topic == VC1_TOPIC:
        if difficulty == "easy":
            return int(re.search(r"(\d+) cm³", text).group(1)) / 1000
        if difficulty == "medium":
            return int(re.search(r"(\d+) m³", text).group(1)) * 1000
        if difficulty == "hard":
            a, b, c = (int(x) for x in re.findall(r"(\d+) cm", text))
            return a * b * c / 1000
        a, b, c = (int(x) for x in re.findall(r"(\d+) m\b", text))
        return a * b * c * 1000
    if difficulty == "easy":
        m = re.search(r"(\d+) cm³", text)
        if m:
            return int(m.group(1)) / 1000
        return int(re.search(r"(\d+) m³", text).group(1)) * 1000
    if difficulty == "medium":
        return int(question.answer_text) * 1000  # total kL (or m3) held
    if difficulty == "hard":
        m = re.search(r"holds (\d+) cm³", text)
        if m:
            return int(m.group(1)) / 1_000_000 * 1000
        m = re.search(r"capacity of (\d+) kL", text)
        if m:
            return int(m.group(1)) * 1000
        return int(re.search(r"holds (\d+) m³", text).group(1)) * 1000
    return None  # exam: generic Container A / B


@pytest.mark.parametrize("topic", [VC1_TOPIC, VC2_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_every_named_context_is_a_liquid_container_with_a_feasible_volume(topic, difficulty):
    if topic == VC2_TOPIC and difficulty == "exam":
        pytest.skip("exam uses the unnamed Container A / Container B")
    for s in FIX_SEEDS:
        q = make(topic, seed=s, difficulty=difficulty)
        lowered = q.text.lower()
        for bad in NOT_LIQUID_CONTAINERS:
            assert bad not in lowered, (bad, q.text)
        name = _named_context(q.text)
        assert name, q.text
        lo, hi = BANDS_L[name]
        litres = _litres_stated(topic, difficulty, q)
        assert lo <= litres <= hi, (name, litres, q.text)


def test_vc1_exam_dimensions_are_not_a_cube_and_pools_are_full_size():
    for s in FIX_SEEDS:
        q = make(VC1_TOPIC, seed=s, difficulty="exam")
        dims = [int(x) for x in re.findall(r"(\d+) m\b", q.text)]
        assert len(dims) == 3 and len(set(dims)) > 1, q.text
        if "swimming pool" in q.text:
            assert dims[0] * dims[1] * dims[2] >= 25, q.text
