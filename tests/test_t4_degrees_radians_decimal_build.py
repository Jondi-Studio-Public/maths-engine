"""Degree<->radian conversion, decimal answers.

Expected, not an error.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

TOPIC1 = "t4_degrees_to_radians_decimal"
TOPIC2 = "t4_radians_to_degrees_decimal"

TOPIC1_VARIANTS = ["forward_whole", "forward_signed", "dms_forward"]
TOPIC2_VARIANTS = ["forward_basic", "forward_signed", "pi_fraction_input"]

TOPIC1_PLACES = {"forward_whole": 2, "forward_signed": 3, "dms_forward": 4}
TOPIC2_PLACES = {"forward_basic": 1, "forward_signed": 2, "pi_fraction_input": 2}

SEEDS_40 = range(40)
SEEDS_60 = range(60)
SEEDS_100 = range(100)


def _tolerance(places: int) -> Fraction:
    return Fraction(1, 2 * 10**places)


def _wrong_answer_text(q, places: int) -> str:
    """A deliberately wrong answer, formatted at the SAME stated precision as
    the canonical -- not a raw float repr, which `_is_rounded_as_asked` would
    reject before the tolerance comparison even runs (a false reject for the
    wrong reason)."""
    return f"{q.answer + 10:.{places}f}"


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", TOPIC1_VARIANTS)
def test_topic1_deterministic(variant):
    for seed in SEEDS_40:
        q1 = make(TOPIC1, seed=seed, variant=variant)
        q2 = make(TOPIC1, seed=seed, variant=variant)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("variant", TOPIC2_VARIANTS)
def test_topic2_deterministic(variant):
    for seed in SEEDS_40:
        q1 = make(TOPIC2, seed=seed, variant=variant)
        q2 = make(TOPIC2, seed=seed, variant=variant)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_topic1_deterministic_by_difficulty(difficulty):
    for seed in SEEDS_40:
        q1 = make(TOPIC1, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC1, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_topic2_deterministic_by_difficulty(difficulty):
    for seed in SEEDS_40:
        q1 = make(TOPIC2, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC2, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


# ---------------------------------------------------------------------------
# distractor hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,variant", [(TOPIC1, v) for v in TOPIC1_VARIANTS] + [(TOPIC2, v) for v in TOPIC2_VARIANTS]
)
def test_distractors_distinct_nonempty_never_equal_answer(topic, variant):
    for seed in SEEDS_60:
        q = make(topic, seed=seed, variant=variant)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


# ---------------------------------------------------------------------------
# the grader's exact tolerance -- asserted directly in the test, every seed,
# for every one of the six tiers, over >=100 seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,variant,places",
    [(TOPIC1, v, TOPIC1_PLACES[v]) for v in TOPIC1_VARIANTS] + [(TOPIC2, v, TOPIC2_PLACES[v]) for v in TOPIC2_VARIANTS],
)
def test_every_distractor_clears_the_grader_tolerance(topic, variant, places):
    tolerance = _tolerance(places)
    for seed in SEEDS_100:
        q = make(topic, seed=seed, variant=variant)
        answer_value = Fraction(q.answer_text)
        for d in q.distractors:
            gap = abs(Fraction(d) - answer_value)
            assert gap > tolerance, (
                f"{topic}/{variant} seed={seed}: distractor {d!r} is within tolerance "
                f"{tolerance} of answer {q.answer_text!r} (gap={gap})"
            )


# ---------------------------------------------------------------------------
# real grader accept/reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,variant,places",
    [(TOPIC1, v, TOPIC1_PLACES[v]) for v in TOPIC1_VARIANTS] + [(TOPIC2, v, TOPIC2_PLACES[v]) for v in TOPIC2_VARIANTS],
)
def test_real_grader_accepts_canonical_and_rejects_distractors_and_wrong(topic, variant, places):
    for seed in SEEDS_40:
        q = make(topic, seed=seed, variant=variant)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        assert not grade_number(q, _wrong_answer_text(q, places)).correct


# ---------------------------------------------------------------------------
# independent check fires on corrupted input -- monkeypatch the standard
# library route the generator cross-checks against
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", TOPIC1_VARIANTS)
def test_topic1_independent_check_fires_on_corrupted(monkeypatch, variant):
    monkeypatch.setattr(math, "radians", lambda x: x + 999.0)
    with pytest.raises(VerificationError):
        make(TOPIC1, seed=0, variant=variant)


@pytest.mark.parametrize("variant", TOPIC2_VARIANTS)
def test_topic2_independent_check_fires_on_corrupted(monkeypatch, variant):
    monkeypatch.setattr(math, "degrees", lambda x: x + 999.0)
    with pytest.raises(VerificationError):
        make(TOPIC2, seed=0, variant=variant)


# ---------------------------------------------------------------------------
# answer is a float; answer_text matches f"{value:.{places}f}" exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,variant,places",
    [(TOPIC1, v, TOPIC1_PLACES[v]) for v in TOPIC1_VARIANTS] + [(TOPIC2, v, TOPIC2_PLACES[v]) for v in TOPIC2_VARIANTS],
)
def test_answer_is_float_and_answer_text_matches_stated_places(topic, variant, places):
    for seed in SEEDS_60:
        q = make(topic, seed=seed, variant=variant)
        assert isinstance(q.answer, float)
        assert q.answer_text == f"{q.answer:.{places}f}"
        m = re.match(r"^-?\d+\.(\d+)$", q.answer_text)
        assert m, f"{topic}/{variant} seed={seed}: {q.answer_text!r} is not a plain decimal"
        assert len(m.group(1)) == places


# ---------------------------------------------------------------------------
# variety: >=20 distinct rendered questions per tier over 60 seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,variant", [(TOPIC1, v) for v in TOPIC1_VARIANTS] + [(TOPIC2, v) for v in TOPIC2_VARIANTS]
)
def test_at_least_20_distinct_questions_per_tier(topic, variant):
    texts = {make(topic, seed=s, variant=variant).text for s in SEEDS_60}
    assert len(texts) >= 20, f"{topic}/{variant}: only {len(texts)} distinct questions over 60 seeds"


# ---------------------------------------------------------------------------
# the DMS tier's text carries the literal degree/prime codepoints
# ---------------------------------------------------------------------------


def test_dms_tier_text_contains_degree_and_prime_symbols():
    for seed in SEEDS_60:
        q = make(TOPIC1, seed=seed, variant="dms_forward")
        assert "°" in q.text, f"seed={seed}: missing degree sign in {q.text!r}"
        assert "′" in q.text, f"seed={seed}: missing prime mark in {q.text!r}"


# ---------------------------------------------------------------------------
# unknown variant raises
# ---------------------------------------------------------------------------


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make(TOPIC1, seed=1, variant="unknown")
    with pytest.raises(ValueError):
        make(TOPIC2, seed=1, variant="unknown")


# ---------------------------------------------------------------------------
# structural traps forced every seed
# ---------------------------------------------------------------------------


def test_dms_forward_minutes_never_zero():
    for seed in SEEDS_60:
        q = make(TOPIC1, seed=seed, variant="dms_forward")
        m = re.search(r"°(\d+)′", q.text)
        assert m, q.text
        assert int(m.group(1)) != 0


def test_medium_tiers_are_always_negative_and_non_whole():
    for seed in SEEDS_60:
        q1 = make(TOPIC1, seed=seed, variant="forward_signed")
        m = re.search(r"Convert (-?[\d.]+) degrees", q1.text)
        value = float(m.group(1))
        assert value < 0, q1.text
        assert value != int(value), q1.text

        q2 = make(TOPIC2, seed=seed, variant="forward_signed")
        m2 = re.search(r"Convert (-?[\d.]+) rad", q2.text)
        value2 = float(m2.group(1))
        assert value2 < 0, q2.text


def test_pi_fraction_input_denominator_never_divides_180():
    for seed in SEEDS_60:
        q = make(TOPIC2, seed=seed, variant="pi_fraction_input")
        m = re.search(r"Convert -?(\d*)pi/(\d+) rad", q.text)
        assert m, q.text
        denominator = int(m.group(2))
        assert 180 % denominator != 0, q.text


# ---------------------------------------------------------------------------
# answer and every distractor formatted alike
# ---------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"^-?\d+\.\d+$")


@pytest.mark.parametrize(
    "topic,variant", [(TOPIC1, v) for v in TOPIC1_VARIANTS] + [(TOPIC2, v) for v in TOPIC2_VARIANTS]
)
def test_answer_and_distractors_formatted_alike(topic, variant):
    for seed in SEEDS_60:
        q = make(topic, seed=seed, variant=variant)
        assert _NUMERIC_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _NUMERIC_RE.match(d), d


# ---------------------------------------------------------------------------
# skills load correctly
# ---------------------------------------------------------------------------


def test_skills_load_with_correct_properties():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}

    deg_to_rad = by_id.get("angle-degrees-to-radians-decimal")
    assert deg_to_rad is not None
    assert deg_to_rad.prerequisites == []
    assert deg_to_rad.answer == "number"
    assert deg_to_rad.placement_eligible is False
    assert len(deg_to_rad.sources) == 3

    rad_to_deg = by_id.get("angle-radians-to-degrees-decimal")
    assert rad_to_deg is not None
    assert rad_to_deg.prerequisites == []
    assert rad_to_deg.answer == "number"
    assert rad_to_deg.placement_eligible is False
    assert len(rad_to_deg.sources) == 3
