"""Tests the periodic-motion skills, including `p19g_sinusoid_amplitude_period`: determinism, distractors and
grading.

Focused contract tests for the periodic-motion modelling / practical
evaluation skills:

- ``senior-model-periodic-motion-using-sine-and-cosine-functions``
  (topic ``p19g_sinusoid_amplitude_period``, `answer: pair`)
- ``senior-identify-contexts-suitable-for-modelling-by-trigonometric-functions``
  (topic ``p19g_sinusoid_evaluate_context``, `answer: number`)

The independent oracles below parse ONLY `question.text`/`answer_text`
(never anything from `mathgen.generators.p19g_trig_model_build`) and
re-derive the expected answer via a fresh route -- regex extraction of the
stated coefficients/scenario values, then plain Python arithmetic/`math`
(never the generator's own helpers)."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p19g_trig_model_build import (
    _verify_amplitude_period,
    _quantity,
    _quantity_independent,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

AMP_TOPIC = "p19g_sinusoid_amplitude_period"
EVAL_TOPIC = "p19g_sinusoid_evaluate_context"
TOPICS = [AMP_TOPIC, EVAL_TOPIC]

AMP_ID = "senior-model-periodic-motion-using-sine-and-cosine-functions"
EVAL_ID = "senior-identify-contexts-suitable-for-modelling-by-trigonometric-functions"


# ===========================================================================
# Determinism, shape, distractor hygiene (both topics, every tier)
# ===========================================================================


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_non_empty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.distractors, f"seed {seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)), q.distractors
        assert len(q.distractors) >= 3, f"seed {seed}: only {len(q.distractors)} distractors"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{topic} [{difficulty}]: only {len(texts)} distinct questions"


# ===========================================================================
# Skill 1: p19g_sinusoid_amplitude_period (answer: pair)
# ===========================================================================

_AMP_PAIR_RE = re.compile(r"^amplitude = (-?\d+(?:\.\d+)?), period = (\d+)$")
#: A distractor's period field may also be the "B/pi" period-formula
#: misread (`misread_period`, an exact fraction like "1/12"), or a decimal
#: (the swapped-amplitude-period distractor puts the decimal-formatted
#: `amplitude_str` in the period slot on hard/exam) -- see
#: `p19g_sinusoid_amplitude_period`'s distractor construction. The canonical
#: answer's own period (always a plain positive integer, load-bearing for
#: `grade_pair`) is never either shape.
_AMP_PAIR_DISTRACTOR_RE = re.compile(r"^amplitude = (-?\d+(?:\.\d+)?), period = (-?\d+(?:\.\d+)?|-?\d+/\d+)$")


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_answer_is_labelled_pair(difficulty):
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert set(q.answer) == {"amplitude", "period"}
        assert _AMP_PAIR_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _AMP_PAIR_DISTRACTOR_RE.match(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_never_asks_at_t_equals(difficulty):
    """Boundary against the sibling skill: skill 1 never substitutes a time."""
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        assert "t =" not in q.text
        assert "at t" not in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_never_leaks_abcd_coefficients_as_labels(difficulty):
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        assert "A =" not in q.text and "B =" not in q.text
        assert "C =" not in q.text and "D =" not in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"seed {seed}: canonical {q.answer_text!r} not accepted"
        for i, distractor in enumerate(q.distractors):
            result = grade_pair(q, distractor)
            assert not result.correct, f"seed {seed}: distractor {i} {distractor!r} accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_labelled_pair_order_independence(difficulty):
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        swapped = f"period = {q.answer['period']}, amplitude = {q.answer['amplitude']}"
        result = grade_pair(q, swapped)
        assert result.correct, f"seed {seed}: swapped-label {swapped!r} not accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in range(20):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        wrong = f"amplitude = {q.answer['amplitude']}, period = {q.answer['period'] + 1}"
        result = grade_pair(q, wrong)
        assert not result.correct, f"seed {seed}: {wrong!r} incorrectly accepted"


def test_amplitude_independent_check_fires_on_corrupted_period():
    with pytest.raises(VerificationError):
        _verify_amplitude_period(Fraction(3), 0, 1, 10, 21, Fraction(3))  # period should be 20


def test_amplitude_independent_check_fires_on_corrupted_amplitude():
    with pytest.raises(VerificationError):
        _verify_amplitude_period(Fraction(3), 0, 1, 10, 20, Fraction(4))  # amplitude should be 3


def test_amplitude_independent_check_fires_on_wrong_rounding():
    # amplitude_source = 2.67 rounded to 1 d.p. should be 2.7, not 2.6.
    with pytest.raises(VerificationError):
        _verify_amplitude_period(Fraction("2.67"), 1, 1, 10, 20, Fraction("2.6"))


# --- Independent oracle: parse the RENDERED TEXT, never the generator ------

_AMP_FN_RE = re.compile(r"y = (-?\d+(?:\.\d+)?)(sin|cos)\((\d*)pi/(\d+)\)t(?: [+-] [\w/]+)?\)?")


def _coefficient(text: str) -> str:
    """The stem leaves a coefficient of 1 or -1 unwritten (sin(...), -cos(...))."""
    return {"": "1", "-": "-1"}.get(text, text)


def _oracle_amplitude_period(text: str, places: int) -> tuple[str, int]:
    """Recompute (amplitude_str, period) from the rendered `y = ...` text
    alone, via `sympy`'s own pi-cancellation on the (possibly unsimplified)
    B fraction parsed out of the text -- never via the generator's own `k`/
    `m`/`n` variables."""
    from sympy import Rational, pi as spi, simplify

    match = re.search(r"y = (-?\d*(?:\.\d+)?)(sin|cos)\(\((\d*)pi/(\d+)\)t", text)
    assert match, text
    a_str, _kind, num_str, den_str = match.groups()
    m = int(num_str) if num_str else 1
    n = int(den_str)
    period_expr = simplify(2 * spi / (Rational(m, n) * spi))
    period = int(period_expr)
    amplitude_exact = abs(Fraction(_coefficient(a_str)))
    if places == 0:
        amplitude_str = str(int(amplitude_exact))
    else:
        scale = 10**places
        scaled = math.floor(float(amplitude_exact) * scale + 0.5)
        amplitude_str = f"{scaled / scale:.{places}f}"
    return amplitude_str, period


@pytest.mark.parametrize("difficulty,places", [("easy", 0), ("medium", 0), ("hard", 1), ("exam", 1)])
def test_amplitude_oracle_matches_generator_over_60_seeds(difficulty, places):
    for seed in SEEDS:
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        amplitude_str, period = _oracle_amplitude_period(q.text, places)
        assert period == q.answer["period"], (seed, q.text, period, q.answer)
        assert float(amplitude_str) == pytest.approx(float(q.answer["amplitude"]), abs=1e-9), (
            seed,
            q.text,
            amplitude_str,
            q.answer,
        )


# Measurement: the OLD distractor set here always left the correct amplitude (or
# period) value matching 2+ of the other 3 offered options, so a learner could win by reading every
# option's amplitude/period column and picking whichever value repeated across the most options --
# 200/200 draws per tier on the amplitude column, reproduced before this fix. This sweep asserts the key is never
# identifiable that way: the correct amplitude is never the value shared by a MAJORITY (>= 2) of the
# other three offered options, and separately for the period, over 200 seeds per tier.
_AMP_PAIR_VALUE_RE = re.compile(r"^amplitude = (-?\d+(?:\.\d+)?), period = (-?\d+(?:\.\d+)?|-?\d+/\d+)$")


def _extract_amp_period(text: str) -> tuple[str, str]:
    m = _AMP_PAIR_VALUE_RE.match(text)
    assert m, text
    return m.group(1), m.group(2)


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_key_is_never_the_column_consensus(difficulty):
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        ans_amp, ans_period = _extract_amp_period(q.answer_text)
        others = [_extract_amp_period(d) for d in q.distractors]
        assert len(others) >= 3, (seed, q.distractors)
        amp_matches = sum(1 for a, _p in others if a == ans_amp)
        period_matches = sum(1 for _a, p in others if p == ans_period)
        assert amp_matches < 2, (
            f"seed {seed} [{difficulty}]: correct amplitude {ans_amp!r} is the "
            f"column-wise consensus across the other options {others!r}"
        )
        assert period_matches < 2, (
            f"seed {seed} [{difficulty}]: correct period {ans_period!r} is the "
            f"column-wise consensus across the other options {others!r}"
        )


_AMP_EQ_RE = re.compile(r"y = (-?\d*(?:\.\d+)?)(?:sin|cos)\(.*?\)(?: ([+-]) (\d+))?\. State")


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_height_contexts_never_model_a_negative_height(difficulty):
    heights = 0
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        m = _AMP_EQ_RE.search(q.text)
        assert m, q.text
        amp = abs(float(_coefficient(m.group(1))))
        d = 0 if m.group(2) is None else int(m.group(3)) * (1 if m.group(2) == "+" else -1)
        if re.search(r"height above", q.text.split(", is modelled")[0]):
            heights += 1
            assert d - amp >= 0, (difficulty, seed, q.text)
    if difficulty in ("hard", "exam"):
        assert heights > 20, f"{difficulty}: only {heights} height-context draws"


# ===========================================================================
# Skill 2: p19g_sinusoid_evaluate_context (answer: number)
# ===========================================================================

_STARTS = ["starting at its mean height and rising", "starting at its maximum", "starting at its minimum"]
_C_MULT = {
    "starting at its mean height and rising": 0.0,
    "starting at its maximum": 0.5,
    "starting at its minimum": -0.5,
}
_CONVERSION_FACTORS = {("minutes", "seconds"): 60, ("hours", "minutes"): 60, ("seconds", "milliseconds"): 1000}


def _parse_scenario(text: str) -> dict:
    amp = re.search(r"amplitude (\d+(?:\.\d+)?) (\w+)", text)
    mean = re.search(r"about a mean of (\d+(?:\.\d+)?) \w+", text)
    period = re.search(r"period of (\d+(?:\.\d+)?) (\w+)", text)
    start = next(s for s in _STARTS if s in text)
    return {
        "a": float(amp.group(1)),
        "unit_len": amp.group(2),
        "d": float(mean.group(1)),
        "period": float(period.group(1)),
        "period_unit": period.group(2),
        "start": start,
    }


def _oracle_forward_value(text: str, dp: int) -> float:
    """Independent forward evaluation: parse amplitude/mean/period/start
    condition and the stated time (converting units if the text says to),
    then evaluate with plain `math.sin`/`math.pi`."""
    scenario = _parse_scenario(text)
    conv = re.search(r"t = (\d+(?:\.\d+)?) (\w+).*convert to (\w+) first", text)
    if conv:
        t_value, t_unit, target_unit = float(conv.group(1)), conv.group(2), conv.group(3)
        factor = _CONVERSION_FACTORS.get((target_unit, t_unit))
        assert factor is not None, text
        t = t_value / factor
    else:
        t_match = re.search(r"t = (\d+(?:\.\d+)?) \w+", text)
        assert t_match, text
        t = float(t_match.group(1))
    c_mult = _C_MULT[scenario["start"]]
    angle = 2 * math.pi * t / scenario["period"] + c_mult * math.pi
    value = scenario["d"] + scenario["a"] * math.sin(angle)
    return round(value, dp)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_evaluate_oracle_matches_generator_over_60_seeds(difficulty):
    for seed in SEEDS:
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        oracle_value = _oracle_forward_value(q.text, dp=1)
        assert oracle_value == pytest.approx(q.answer, abs=0.05), (seed, q.text, oracle_value, q.answer)


def _oracle_inverse_time(text: str) -> float:
    """Independent inverse-solve oracle for the exam tier: parse the
    scenario and target value, then re-derive the smallest positive t via a
    fresh `math.asin`-based computation (never the generator's own
    `_evaluate_context_exam`)."""
    scenario = _parse_scenario(text)
    target_match = re.search(r"first reaches (\d+(?:\.\d+)?) \w+", text)
    assert target_match, text
    target = float(target_match.group(1))
    a, d, period = scenario["a"], scenario["d"], scenario["period"]
    c_mult = _C_MULT[scenario["start"]]
    s = (target - d) / a
    theta1 = math.asin(s)
    theta2 = math.pi - theta1
    offset = c_mult * math.pi
    t1 = (period * ((theta1 - offset) / (2 * math.pi))) % period
    t2 = (period * ((theta2 - offset) / (2 * math.pi))) % period
    if t1 <= 1e-9:
        t1 += period
    if t2 <= 1e-9:
        t2 += period
    return round(min(t1, t2), 1)


def test_evaluate_exam_oracle_matches_generator_over_60_seeds():
    for seed in SEEDS:
        q = make(EVAL_TOPIC, seed=seed, difficulty="exam")
        oracle_t = _oracle_inverse_time(q.text)
        assert oracle_t == pytest.approx(q.answer, abs=0.05), (seed, q.text, oracle_t, q.answer)


# --- D >= A (never-negative) and starting-condition unambiguity -----------


def test_evaluate_d_never_less_than_a_and_quantity_never_negative():
    for difficulty in ["easy", "medium", "hard", "exam"]:
        for seed in SEEDS:
            q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
            scenario = _parse_scenario(q.text)
            assert scenario["d"] >= scenario["a"], (difficulty, seed, q.text)
            # The min of the model is D - A >= 0: never negative by construction.
            assert scenario["d"] - scenario["a"] >= 0


def test_evaluate_starting_condition_phase_is_fixed_and_unambiguous():
    """Each of the three starting-condition phrases pins exactly one phase
    against the sine base model -- confirmed here by checking value(t=0)
    against the phrase's own defining property (mean/rising, max, min)."""
    for difficulty in ["easy", "medium"]:
        for seed in SEEDS:
            q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
            scenario = _parse_scenario(q.text)
            a, d, period, start = scenario["a"], scenario["d"], scenario["period"], scenario["start"]
            c_mult = _C_MULT[start]
            value_at_0 = d + a * math.sin(c_mult * math.pi)
            derivative_at_0 = a * (2 * math.pi / period) * math.cos(c_mult * math.pi)
            if start == "starting at its mean height and rising":
                assert abs(value_at_0 - d) < 1e-9
                assert derivative_at_0 > 0
            elif start == "starting at its maximum":
                assert abs(value_at_0 - (d + a)) < 1e-9
            else:
                assert abs(value_at_0 - (d - a)) < 1e-9


# --- Grader-kind and answer shape ------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_evaluate_answer_is_plain_float_with_stated_precision(difficulty):
    for seed in SEEDS:
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, float)
        assert re.fullmatch(r"-?\d+\.\d \w+", q.answer_text), q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_evaluate_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed {seed}: canonical {q.answer_text!r} not accepted"
        for i, distractor in enumerate(q.distractors):
            result = grade_number(q, distractor)
            assert not result.correct, f"seed {seed}: distractor {i} {distractor!r} accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_evaluate_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in range(20):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        wrong_value = q.answer + 1.0
        wrong_text = q.answer_text.replace(re.search(r"-?\d+\.\d", q.answer_text).group(0), f"{wrong_value:.1f}")
        result = grade_number(q, wrong_text)
        assert not result.correct, f"seed {seed}: {wrong_text!r} incorrectly accepted"


def test_evaluate_never_asks_for_bare_amplitude_or_period():
    """Boundary against the sibling skill: skill 2 never asks the learner
    to state the bare amplitude or period as the final answer."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
            assert "state the amplitude" not in q.text.lower()
            assert "state the period" not in q.text.lower()


# --- construction-vs-independent-sympy-route agreement (internal check) ---


def test_quantity_math_and_sympy_routes_agree():
    for a, d, period, c_mult, t in [
        (5, 12, 10, Fraction(0), Fraction(3)),
        (7, 20, 8, Fraction(1, 2), Fraction(13, 10)),
        (3, 9, 15, Fraction(-1, 2), Fraction(11, 2)),
    ]:
        math_route = _quantity(a, d, period, c_mult, float(t))
        sympy_route = _quantity_independent(a, d, Fraction(period), c_mult, t)
        assert abs(math_route - sympy_route) < 1e-9


# ===========================================================================
# Skill contract loading tests
# ===========================================================================


def test_both_skills_load_with_correct_answer_kind_and_prerequisites():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    amp = by_id[AMP_ID]
    ev = by_id[EVAL_ID]
    assert amp.answer == "pair"
    assert amp.prerequisites == ["right-triangle-angle"]
    assert ev.answer == "number"
    assert ev.prerequisites == [
        "senior-understand-the-unit-circle-definition-of-cos-theta",
        "angle-degrees-radians-pi-form",
    ]


def test_both_skills_are_active_placement_ineligible_and_general_band():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (AMP_ID, EVAL_ID):
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert getattr(skill, "learner_band", "general") == "general"


def test_both_skills_have_sources_covering_all_four_tiers():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (AMP_ID, EVAL_ID):
        difficulties = {src.difficulty for src in by_id[skill_id].sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}
    # The exam source is `stages: [exam]`, never `stages: []`.
    for skill_id in (AMP_ID, EVAL_ID):
        skill = by_id[skill_id]
        exam_src = next(s for s in skill.sources if s.difficulty == "exam")
        assert exam_src.stages == ["exam"]


# =========================================================================== Regression guard: each
# scenario's period must land in a physically realistic band, in its own unit -- never a single
# shared `k` range applied regardless of which real object is named. Proved RED against the pre-fix generator
# (556/800 violations, a scratch copy of that generator loaded into an
# isolated registry, never a checkout/stash) before this fix; green here, every tier, 200
# seeds/tier. ===========================================================================

_PERIOD_BANDS = {
    "ferris": (4, 30, "minutes"),
    "tide": (10, 14, "hours"),
    "pendulum": (4, 30, "seconds"),
    "piston": (20, 100, "milliseconds"),
}


def _scenario_of(text: str) -> str:
    if "Ferris" in text:
        return "ferris"
    if "tide" in text.lower():
        return "tide"
    if "pendulum" in text.lower():
        return "pendulum"
    if "piston" in text.lower():
        return "piston"
    raise AssertionError(f"unrecognised scenario in: {text!r}")


@pytest.mark.parametrize("difficulty", TIERS)
def test_periodic_motion_scenario_period_is_physically_realistic(difficulty):
    """Real-world anchors: a piston cycles many times per second (milliseconds, 20-100 here); a tide's period is
    about 12.4 hours (10-14 here, the closest an integer-period design gets); a Ferris wheel rotates in a
    couple of minutes (4-30 minutes); a large pendulum swings over several seconds (4-30 seconds, unchanged)."""
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        m = re.search(r"period = (\d+)", q.answer_text)
        assert m, q.answer_text
        period = int(m.group(1))
        m2 = re.search(r"t (\w+) after tracking begins", q.text)
        assert m2, q.text
        unit = m2.group(1)
        scenario = _scenario_of(q.text)
        lo, hi, expected_unit = _PERIOD_BANDS[scenario]
        assert unit == expected_unit, (scenario, unit, expected_unit, seed, difficulty)
        assert lo <= period <= hi, (scenario, period, lo, hi, seed, difficulty)


def _first_number(s: str) -> float:
    return float(re.search(r"-?\d+(?:\.\d+)?", s).group(0))


def test_easy_tier_key_rarely_equals_the_mean():
    """Measured: key == stated mean on ~60% of easy draws (the
    quarter/half fractions land on sin = 0). The key must stay a minority."""
    hits = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty="easy")
        d = _parse_scenario(q.text)["d"]
        hits += _first_number(q.answer_text) == d
    assert hits / 200 < 0.25, f"key equals the mean on {hits}/200 easy seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_stays_within_the_physical_range(difficulty):
    """Forward tiers ask for a height/distance, so every offered option must
    lie in [d - a, d + a]. The exam tier asks for a TIME: its options must
    be positive and no later than two periods."""
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        s = _parse_scenario(q.text)
        for opt in q.distractors:
            v = _first_number(opt)
            if difficulty == "exam":
                assert 0 < v <= 2 * s["period"] + 0.05, (seed, q.text, opt)
            else:
                assert s["d"] - s["a"] - 1e-9 <= v <= s["d"] + s["a"] + 1e-9, (difficulty, seed, q.text, opt)


# Distractors are independent wrong routes ---------------------
#
# An earlier version perturbed BOTH fields of every distractor, so the key was the only option showing the
# amplitude readable straight off the function (and it offered absurd reciprocal periods like
# "1/5"). Contract now: in every tier, on every seed, some distractor keeps the key's amplitude and
# some keeps its period, every period is a plain positive number >= 1, and the "most frequent value
# in each column" heuristic does not single out the key.


def _options_2134(q):
    return [_extract_amp_period(t) for t in [q.answer_text] + list(q.distractors)]


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_some_distractor_shares_key_amplitude_and_some_shares_period(difficulty):
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options_2134(q)
        key_a, key_p = opts[0]
        assert any(a == key_a for a, _p in opts[1:]), (seed, difficulty, opts)
        assert any(p == key_p for _a, p in opts[1:]), (seed, difficulty, opts)
        assert len(set(opts)) == len(opts), (seed, difficulty, opts)


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_no_absurd_or_fractional_period_values(difficulty):
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        for _a, p in _options_2134(q):
            assert "/" not in p, (seed, difficulty, p)
            assert re.fullmatch(r"\d+", p), (seed, difficulty, p)
            assert int(p) >= 1, (seed, difficulty, p)


@pytest.mark.parametrize("difficulty", TIERS)
def test_amplitude_period_column_mode_heuristic_rarely_wins(difficulty):
    """The heuristic 'pick the option whose amplitude AND period are each the
    most frequent value in their column' picks the key on < 45% of seeds. It
    only counts as a win when the key is the SOLE option it selects."""
    wins = 0
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty=difficulty)
        opts = _options_2134(q)
        a_counts = [sum(x[0] == o[0] for x in opts) for o in opts]
        p_counts = [sum(x[1] == o[1] for x in opts) for o in opts]
        top = [i for i in range(len(opts)) if a_counts[i] == max(a_counts) and p_counts[i] == max(p_counts)]
        wins += top == [0]
    assert wins / 200 < 0.45, f"[{difficulty}] heuristic wins {wins}/200"


_AMP2139_FN_RE = re.compile(r"y = (-?\d*(?:\.\d+)?)(?:sin|cos)\(")
_AMP2139_WORD_RE = re.compile(r"amplitude (\d+(?:\.\d+)?) (?:metres|centimetres)")


def _stated_amplitude_2139(topic: str, text: str) -> str:
    m = (_AMP2139_FN_RE if topic == AMP_TOPIC else _AMP2139_WORD_RE).search(text)
    assert m, text
    return _coefficient(m.group(1))


def _scenario_2139(text: str):
    head = text.split(" varies sinusoidally")[0].split(", is modelled")[0]
    if "Ferris" in head:
        return "ferris"
    if "tide" in head.lower():
        return "tide"
    return "other"


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_2139_tide_amplitude_at_most_6_and_ferris_at_least_3(topic, difficulty):
    seen = {"tide": 0, "ferris": 0}
    for seed in range(200):
        q = make(topic, seed=seed, difficulty=difficulty)
        amp = abs(float(_stated_amplitude_2139(topic, q.text)))
        kind = _scenario_2139(q.text)
        if kind == "tide":
            seen["tide"] += 1
            assert amp <= 6, (topic, difficulty, seed, q.text)
        elif kind == "ferris":
            seen["ferris"] += 1
            assert amp >= 3, (topic, difficulty, seed, q.text)
    assert seen["tide"] > 10 and seen["ferris"] > 10, seen


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_2139_no_stated_amplitude_with_three_decimal_places(topic, difficulty):
    for seed in range(200):
        q = make(topic, seed=seed, difficulty=difficulty)
        stated = _stated_amplitude_2139(topic, q.text)
        places = len(stated.split(".")[1]) if "." in stated else 0
        assert places <= 2, (topic, difficulty, seed, stated)


def test_2139_exam_amplitude_still_requires_rounding_to_one_place():
    for seed in range(200):
        q = make(AMP_TOPIC, seed=seed, difficulty="exam")
        stated = _stated_amplitude_2139(AMP_TOPIC, q.text)
        assert "." in stated and len(stated.split(".")[1]) == 2, (seed, stated)
        assert not stated.endswith(("0", "5")), (seed, stated)


# 200 seeds per tier of real `make` output.
# ===========================================================================


def _opts_2141(q):
    return [_first_number(q.answer_text)] + [_first_number(o) for o in q.distractors]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_2141_reflection_attack_scores_low(difficulty):
    """'key = 2D - (option farthest from D)' must not find the key."""
    hits = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        d = _parse_scenario(q.text)["d"]
        vals = _opts_2141(q)
        far = max(vals[1:], key=lambda v: abs(v - d))
        hits += abs((2 * d - far) - vals[0]) <= 0.051
    assert hits / 200 <= 0.30, f"[{difficulty}] reflection attack hits {hits}/200"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_2141_key_is_not_usually_the_extreme_option(difficulty):
    hits = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        vals = _opts_2141(q)
        hits += vals[0] in (max(vals), min(vals))
    assert hits / 200 <= 0.55, f"[{difficulty}] key extreme on {hits}/200"


def test_2141_exam_pick_the_minimum_is_not_a_winning_strategy():
    hits = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty="exam")
        vals = _opts_2141(q)
        hits += vals[0] == min(vals)
    assert hits / 200 <= 0.45, f"exam key is the minimum on {hits}/200"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_2141_stated_mean_is_a_distractor_in_a_minority_of_rows(difficulty):
    hits = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        d = _parse_scenario(q.text)["d"]
        hits += any(abs(v - d) <= 0.051 for v in _opts_2141(q)[1:])
    assert hits / 200 <= 0.30, f"[{difficulty}] mean offered as distractor on {hits}/200"


def test_2141_exam_no_p_plus_minus_key_pair_and_no_half_period_habit():
    pair = half = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty="exam")
        p = _parse_scenario(q.text)["period"]
        vals = _opts_2141(q)
        key, ds = vals[0], vals[1:]
        has = lambda x: any(abs(v - x) <= 0.051 for v in ds)  # noqa: E731
        pair += has(p - key) and has(p + key)
        half += has(p / 2)
    assert pair / 200 <= 0.15, f"P-key and P+key both offered on {pair}/200"
    assert half / 200 <= 0.25, f"P/2 offered on {half}/200"


@pytest.mark.parametrize("difficulty", TIERS)
def test_2141_tide_period_is_8_to_14_hours(difficulty):
    seen = 0
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        if "tide" not in q.text.split(" varies sinusoidally")[0].lower():
            continue
        seen += 1
        period = _parse_scenario(q.text)["period"]
        assert 8 <= period <= 14, (difficulty, seed, q.text)
    assert seen > 10


@pytest.mark.parametrize("difficulty", TIERS)
def test_2141_options_in_range_and_exactly_one_correct_through_grader(difficulty):
    for seed in range(200):
        q = make(EVAL_TOPIC, seed=seed, difficulty=difficulty)
        s = _parse_scenario(q.text)
        assert len(q.distractors) == 3, (seed, q.distractors)
        for opt in q.distractors:
            v = _first_number(opt)
            if difficulty == "exam":
                assert 0 < v < s["period"] + 1e-9, (seed, q.text, opt)
            else:
                assert s["d"] - s["a"] - 1e-9 <= v <= s["d"] + s["a"] + 1e-9, (seed, opt)
        if seed < 50:
            assert grade_number(q, q.answer_text).correct
            for opt in q.distractors:
                assert not grade_number(q, opt).correct, (seed, q.text, opt)


def _derivation_score_2145(seeds=200):
    """Expected success of 'pick the option the others derive from': the
    option maximising how many others sit within 0.06 of v/2, 2v, P-v, P+v
    (ties share the pick equally)."""
    total = 0.0
    for seed in range(seeds):
        q = make(EVAL_TOPIC, seed=seed, difficulty="exam")
        p = _parse_scenario(q.text)["period"]
        vals = _opts_2141(q)
        scores = []
        for i, v in enumerate(vals):
            others = [w for j, w in enumerate(vals) if j != i]
            scores.append(sum(any(abs(w - t) <= 0.06 for t in (v / 2, 2 * v, p - v, p + v)) for w in others))
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        total += (1 / len(winners)) if 0 in winners else 0.0
    return total / seeds


def test_2145_exam_derivation_attack_scores_low():
    score = _derivation_score_2145()
    assert score <= 0.40, f"derivation attack scores {score:.3f}"


def test_2145_exam_no_distractor_is_half_or_double_the_key():
    bad = total = 0
    for seed in range(200):
        vals = _opts_2141(make(EVAL_TOPIC, seed=seed, difficulty="exam"))
        key = vals[0]
        for v in vals[1:]:
            total += 1
            bad += abs(v - key / 2) <= 0.06 or abs(v - 2 * key) <= 0.06
    assert bad == 0, f"{bad}/{total} exam distractors are key/2 or 2*key"


def test_2145_exam_grader_accepts_key_and_rejects_distractors():
    for seed in range(200, 260):
        q = make(EVAL_TOPIC, seed=seed, difficulty="exam")
        assert grade_number(q, q.answer_text).correct, (seed, q.text)
        for opt in q.distractors:
            assert not grade_number(q, opt).correct, (seed, q.text, opt)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_2149_evaluate_distractors_are_clear_of_key_and_of_each_other(difficulty):
    min_key = min_pair = 10.0
    for seed in range(200):
        vals = _opts_2141(make(EVAL_TOPIC, seed=seed, difficulty=difficulty))
        key, ds = vals[0], vals[1:]
        min_key = min(min_key, min(abs(v - key) for v in ds))
        min_pair = min(min_pair, min(abs(ds[i] - ds[j]) for i in range(3) for j in range(i + 1, 3)))
    assert min_key >= 0.3 - 1e-9, f"min |distractor - key| = {min_key:.2f}"
    assert min_pair >= 0.2 - 1e-9, f"min pairwise gap = {min_pair:.2f}"
