"""Tests the energy-unit skills `p19h_energy_kwh_consumption` and `p19h_energy_kj_amount`.

Senior energy computations, kWh consumption + kJ activity amount
(senior-use-units-of-energy-to-describe-consumption-of, senior-use-units-of-energy-to-describe-the-amount)."""

import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check, stated_precision_string
from mathgen.generators.p19h_energy_build import ACTIVITIES, APPLIANCES

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)
WIDE_SEEDS = range(60)

KWH_TOPIC = "p19h_energy_kwh_consumption"
KJ_TOPIC = "p19h_energy_kj_amount"
TIERS = ["easy", "medium", "hard", "exam"]


# --- catalogue sanity ---------------------------------------------------------


def test_appliance_power_bands_are_realistic():
    """Each appliance has a
    power BAND of at least three values, and a whole-hours range only if the
    appliance legitimately runs for hours."""
    for name, lo, hi, step, hours, minutes in APPLIANCES:
        assert 15 <= lo < hi <= 3600, f"{name}: band {lo}-{hi} W outside realistic household range"
        assert len(range(lo, hi + 1, step)) >= 3, name
        assert hours is not None or minutes is not None
        if hours is not None:
            assert hours[1] <= 24


def test_activity_rate_bands_are_realistic():
    for name, source, lo, hi in ACTIVITIES:
        assert 5 <= lo < hi <= 45, f"{name}: {lo}-{hi} kJ/min outside realistic range"
        assert source


# --- determinism / distractor hygiene, both topics ----------------------------


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generator_is_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_kwh_answer_and_distractors_always_state_exactly_one_decimal_place(difficulty):
    for seed in SEEDS:
        question = make(KWH_TOPIC, seed=seed, difficulty=difficulty)
        for opt in [question.answer_text] + question.distractors:
            assert "." in opt, opt
            assert len(opt.split(".")[1]) == 1, opt


@pytest.mark.parametrize("difficulty", TIERS)
def test_kj_answer_and_distractors_are_always_bare_integers(difficulty):
    for seed in SEEDS:
        question = make(KJ_TOPIC, seed=seed, difficulty=difficulty)
        for opt in [question.answer_text] + question.distractors:
            assert "." not in opt, opt
            int(opt)


# --- real grader accept/reject, both topics -----------------------------------


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_kwh_answer(difficulty):
    for seed in SEEDS:
        question = make(KWH_TOPIC, seed=seed, difficulty=difficulty)
        wrong = Fraction(question.answer_text) + 1
        wrong_text = stated_precision_string(wrong, 1)
        assert not grade_number(question, wrong_text).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_kj_answer(difficulty):
    for seed in SEEDS:
        question = make(KJ_TOPIC, seed=seed, difficulty=difficulty)
        wrong = int(question.answer_text) + 1
        assert not grade_number(question, str(wrong)).correct


# --- variety, both topics -----------------------------------------------------


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(topic, difficulty):
    seen = {make(topic, seed=s, difficulty=difficulty).text for s in WIDE_SEEDS}
    assert len(seen) >= 20, f"{topic}/{difficulty}: only {len(seen)} distinct questions over 60 seeds"


# --- context boundary: no food/Calorie framing on either skill ---------------


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_never_mentions_food_or_calories(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        text_lower = question.text.lower()
        assert "calorie" not in text_lower
        assert "food" not in text_lower


@pytest.mark.parametrize("difficulty", TIERS)
def test_kwh_never_asks_to_convert_a_single_stated_energy_value(difficulty):
    for seed in SEEDS:
        question = make(KWH_TOPIC, seed=seed, difficulty=difficulty)
        assert "convert" not in question.text.lower()


@pytest.mark.parametrize("difficulty", TIERS)
def test_kj_never_asks_to_convert_a_single_stated_energy_value(difficulty):
    for seed in SEEDS:
        question = make(KJ_TOPIC, seed=seed, difficulty=difficulty)
        assert "convert" not in question.text.lower()


# --- tier escalation shape (hand checks; measure_tiers.py is the instrument) --


def test_kwh_hard_never_draws_the_same_appliance_twice():
    for seed in WIDE_SEEDS:
        question = make(KWH_TOPIC, seed=seed, difficulty="hard")
        assert question.text.count(" and ") >= 1


def test_kj_hard_never_draws_the_same_activity_twice():
    for seed in WIDE_SEEDS:
        question = make(KJ_TOPIC, seed=seed, difficulty="hard")
        assert question.text.count("While ") == 2


_KWH_EXAM_RE = re.compile(
    r"power rating of (\d+) W and runs for (\d+) hours?\. .*?power rating of (\d+) W and runs for (\d+) minutes\."
)


def test_kwh_exam_never_lands_on_the_rounding_boundary():
    """Every kWh tier's raw answer can land off the 1-d.p. grid; confirm the
    generator's redraw guard keeps every seed's RAW (pre-round) answer strictly
    off the halfway point. Recomputes the raw value from the rendered text,
    independent of the already-rounded `answer_text`."""
    for seed in range(300):
        question = make(KWH_TOPIC, seed=seed, difficulty="exam")
        m = _KWH_EXAM_RE.search(question.text)
        assert m, question.text
        w1, h1, w2, m2 = (int(g) for g in m.groups())
        raw = Fraction(w1, 1000) * h1 + Fraction(w2, 1000) * Fraction(m2, 60)
        scaled = raw * 20  # 2 * 10**dp, dp=1
        assert not (scaled.denominator == 1 and scaled.numerator % 2 == 1), (
            f"seed={seed}: {question.text!r} raw answer {raw} sits on the rounding boundary"
        )


def test_kwh_exam_states_second_duration_in_minutes():
    for seed in SEEDS:
        question = make(KWH_TOPIC, seed=seed, difficulty="exam")
        assert _KWH_EXAM_RE.search(question.text), question.text


def test_kj_exam_states_duration_in_mixed_hours_and_minutes_form():
    for seed in SEEDS:
        question = make(KJ_TOPIC, seed=seed, difficulty="exam")
        assert "hour" in question.text and "minutes" in question.text
        assert "per second" in question.text


def test_kj_easy_no_distractor_rounds_to_zero_or_one():
    """Over >= 200 draws, no offered option (answer or distractor) may round to 0 or 1."""
    for seed in range(200):
        question = make(KJ_TOPIC, seed=seed, difficulty="easy")
        for opt in [question.answer_text] + question.distractors:
            assert int(opt) not in (0, 1), f"seed={seed}: implausible option {opt!r} in {question.text!r}"


def test_kj_exam_distinct_prompt_pool_widened():
    """Assert the widened bar and report the 200-seed count for direct comparison with the old ~64."""
    seen60 = {make(KJ_TOPIC, seed=s, difficulty="exam").text for s in range(60)}
    assert len(seen60) >= 35, f"kj_exam: only {len(seen60)} distinct prompts over 60 seeds"

    seen200 = {make(KJ_TOPIC, seed=s, difficulty="exam").text for s in range(200)}
    assert len(seen200) > 64, f"kj_exam: only {len(seen200)} distinct prompts over 200 seeds (old ceiling ~64)"


# --- independent in-generator check fires on a corrupted value ---------------


def test_independent_check_fires_on_corrupted_kwh_value():
    corrupted = Fraction(999)
    true_value = Fraction(3, 5)  # 0.6, plausible real kWh value
    with pytest.raises(VerificationError):
        check(corrupted == true_value, "independent check should have failed")


def test_independent_check_fires_on_corrupted_kj_value():
    corrupted = Fraction(999)
    true_value = Fraction(240)
    with pytest.raises(VerificationError):
        check(corrupted == true_value, "independent check should have failed")


def test_raises_on_unknown_mode_both_topics():
    with pytest.raises(ValueError):
        make(KWH_TOPIC, seed=1, mode="unknown")
    with pytest.raises(ValueError):
        make(KJ_TOPIC, seed=1, mode="unknown")


# --- skill loading -------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,expected_sources",
    [
        ("senior-use-units-of-energy-to-describe-consumption-of", 4),
        ("senior-use-units-of-energy-to-describe-the-amount", 4),
    ],
)
def test_skill_loads_with_expected_properties(skill_id, expected_sources):
    _, skills = load_skills(ROOT / "skills")
    skill = next((s for s in skills if s.id == skill_id), None)
    assert skill is not None
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["convert-metric-length"]
    assert len(skill.sources) == expected_sources
    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert len(exam_sources) == 1
    assert exam_sources[0].stages == ["exam"]


# Fixes (swept, fail on the earlier tree) ---------

import collections

SWEEP = 250
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# name substring -> (max realistic watts, max realistic hours of one run)
REAL_APPLIANCES = {
    "laptop charger": (100, 14),
    "desk fan": (80, 14),
    "ceiling fan": (100, 14),
    "set-top box": (40, 24),
    "television": (200, 10),
    "games console": (250, 6),
    "desktop computer": (400, 10),
    "electric blanket": (200, 10),
    "air conditioner": (1600, 10),
    "bar heater": (2500, 6),
    "electric oven": (3200, 3.5),
    "clothes dryer": (3600, 3.5),
    "dishwasher": (2200, 3),
    "washing machine": (1000, 3),
    "vacuum cleaner": (1500, 2),
    "microwave": (1600, 1),
    "hair dryer": (2200, 1),
    "toaster": (1600, 1),
    "kettle": (2400, 1),
}


def _stem(text):
    return text.split("To 1 decimal place")[0].split("To the nearest kilojoule")[0]


def _rank_shares(topic, tier):
    counts = collections.Counter()
    for seed in range(SWEEP):
        q = make(topic, seed=seed, difficulty=tier)
        opts = sorted(float(o) for o in [q.answer_text] + q.distractors)
        counts[opts.index(float(q.answer_text))] += 1
    return {r: c / SWEEP for r, c in counts.items()}


@pytest.mark.parametrize("difficulty", TIERS)
def test_kwh_no_zero_or_subplausible_option(difficulty):
    for seed in range(SWEEP):
        q = make(KWH_TOPIC, seed=seed, difficulty=difficulty)
        for opt in [q.answer_text] + q.distractors:
            assert float(opt) >= 0.2, f"seed={seed}: option {opt!r} in {q.text!r}"


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_within_factor_30_of_key(topic, difficulty):
    for seed in range(SWEEP):
        q = make(topic, seed=seed, difficulty=difficulty)
        key = float(q.answer_text)
        for d in q.distractors:
            ratio = float(d) / key
            assert 1 / 30 - 1e-9 <= ratio <= 30 + 1e-9, f"seed={seed}: {d!r} vs key {key} in {q.text!r}"


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_key_rank_varies_no_rank_over_45_percent(topic, difficulty):
    shares = _rank_shares(topic, difficulty)
    assert len(shares) >= 3, shares
    assert max(shares.values()) <= 0.45, shares


@pytest.mark.parametrize("topic", [KWH_TOPIC, KJ_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_prompt_never_restates_the_key(topic, difficulty):
    for seed in range(SWEEP):
        q = make(topic, seed=seed, difficulty=difficulty)
        nums = {Fraction(t) for t in _NUM_RE.findall(_stem(q.text))}
        assert Fraction(q.answer_text) not in nums, f"seed={seed}: {q.text!r}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_kwh_appliance_powers_and_durations_are_realistic(difficulty):
    for seed in range(SWEEP):
        q = make(KWH_TOPIC, seed=seed, difficulty=difficulty)
        stem = _stem(q.text)
        low = stem.lower()
        hits = sorted((low.find(n), n) for n in REAL_APPLIANCES if n in low)
        assert hits, f"appliance not covered by REAL_APPLIANCES: {q.text!r}"
        for i, (pos, name) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(stem)
            seg = stem[pos:end]
            pm = re.search(r"([\d.]+) (kW|W)\b", seg)
            dm = re.search(r"(\d+) (hours?|minutes)", seg)
            assert pm and dm, f"{name}: {seg!r}"
            watts = float(pm.group(1)) * (1000 if pm.group(2) == "kW" else 1)
            hours = int(dm.group(1)) / (60 if dm.group(2) == "minutes" else 1)
            max_w, max_h = REAL_APPLIANCES[name]
            assert watts <= max_w, f"seed={seed}: {name} {watts} W > {max_w} W ({q.text!r})"
            assert hours <= max_h, f"seed={seed}: {name} {hours} h > {max_h} h ({q.text!r})"


def test_kwh_power_varies_within_each_appliance():
    powers = collections.defaultdict(set)
    for seed in range(400):
        q = make(KWH_TOPIC, seed=seed, difficulty="medium")
        low = _stem(q.text).lower()
        name = next(n for n in REAL_APPLIANCES if n in low)
        powers[name].add(re.search(r"([\d.]+) W\b", _stem(q.text)).group(1))
    for name, seen in powers.items():
        assert len(seen) >= 3, f"{name}: only powers {sorted(seen)}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_kwh_prompt_pool_at_least_170_of_200(difficulty):
    seen = {make(KWH_TOPIC, seed=s, difficulty=difficulty).text for s in range(200)}
    assert len(seen) >= 170, f"only {len(seen)} distinct prompts over 200 seeds"


def _kwh_tokens(text):
    stem = _stem(text)
    powers = re.findall(r"([\d.]+) (kW|W)\b", stem)
    durs = re.findall(r"(\d+) (hours?|minutes)", stem)
    return powers, durs


def test_kwh_tiers_stack_one_step_at_a_time():
    for seed in range(SWEEP):
        p, d = _kwh_tokens(make(KWH_TOPIC, seed=seed, difficulty="easy").text)
        assert len(p) == 1 and p[0][1] == "kW" and d[0][1].startswith("hour")
        p, d = _kwh_tokens(make(KWH_TOPIC, seed=seed, difficulty="medium").text)
        assert len(p) == 1 and p[0][1] == "W" and d[0][1].startswith("hour")
        # hard = medium + a second appliance summed (both in W, both hours)
        p, d = _kwh_tokens(make(KWH_TOPIC, seed=seed, difficulty="hard").text)
        assert len(p) == 2 and all(u == "W" for _, u in p), p
        assert all(u.startswith("hour") for _, u in d), d
        # exam = hard + a time conversion (second appliance stated in minutes)
        p, d = _kwh_tokens(make(KWH_TOPIC, seed=seed, difficulty="exam").text)
        assert len(p) == 2 and all(u == "W" for _, u in p), p
        assert [u.startswith("hour") for _, u in d] == [True, False], d


_MACHINE_WORDS = ("machine", "trainer", "treadmill", "bike")


@pytest.mark.parametrize("difficulty", TIERS)
def test_kj_source_fits_the_activity(difficulty):
    for seed in range(SWEEP):
        text = make(KJ_TOPIC, seed=seed, difficulty=difficulty).text
        assert "lab sensor" not in text, text
        for activity, source in re.findall(r"While ([^,]+), ([^.]*?) shows", text):
            if "machine display" in source:
                assert any(w in activity for w in _MACHINE_WORDS), text
            if any(w in activity for w in ("swimming", "jogging", "hiking", "home workout", "dancing", "walking")):
                assert "machine display" not in source, text


def test_kj_exam_uses_at_least_8_activities():
    seen = set()
    for seed in range(200):
        seen.add(re.search(r"While ([^,]+),", make(KJ_TOPIC, seed=seed, difficulty="exam").text).group(1))
    assert len(seen) >= 8, sorted(seen)
