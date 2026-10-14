"""Task 603 (packet T4, build 3/3): radians -> degrees/minutes (DMS) pair.

Reverse direction of task 601's DMS->radians hard tier. This skill
(``angle-radians-to-dms``, topic ``t4_radians_to_dms``, ``answer: pair``)
converts a radian value to degrees and minutes, rounded to the nearest
minute, across three input-shape tiers.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import Question

import mathgen.generators.t4_dms_build  # noqa: F401  (registers the topic)

TOPIC = "t4_radians_to_dms"
SKILL_ID = "angle-radians-to-dms"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_DECIMAL_R_RE = re.compile(r"(-?\d+\.\d+) rad")
_FRACTION_R_RE = re.compile(r"(\d+)/(\d+) rad")
_DMS_RE = re.compile(r"^(\d+)°(\d{2})′$")


def _r_from_text(text: str, tier: str) -> float:
    if tier == "hard":
        m = _FRACTION_R_RE.search(text)
        assert m, f"no k/m fraction found in {text!r}"
        return int(m.group(1)) / int(m.group(2))
    m = _DECIMAL_R_RE.search(text)
    assert m, f"no decimal radian value found in {text!r}"
    return float(m.group(1))


def _minutes_exact(R: float) -> float:
    degree_decimal = R * 180 / math.pi
    D = math.floor(degree_decimal)
    return (degree_decimal - D) * 60


def _q(**kwargs) -> Question:
    defaults = dict(
        topic="test",
        level=0,
        seed=1,
        text="q",
        latex="q",
        answer=None,
        answer_text="",
        answer_latex="",
        working=[],
        difficulty=None,
        distractors=[],
    )
    defaults.update(kwargs)
    return Question(**defaults)


# --- determinism -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_across_seeds(tier):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=tier)
        q2 = make(TOPIC, seed=seed, difficulty=tier)
        assert q1.to_json() == q2.to_json()


# --- minute-to-60 carry --------------------------------------------------
# Seeds found by direct search (found by search): a random 60-seed sweep is
# not guaranteed to hit the carry, so these are pinned exact seeds where the
# pre-carry minutes_exact rounds to 60.

_CARRY_SEEDS = {"easy": 24, "medium": 61, "hard": 19}


@pytest.mark.parametrize("tier", TIERS)
def test_minute_to_60_carry_fires_and_result_is_valid(tier):
    seed = _CARRY_SEEDS[tier]
    q = make(TOPIC, seed=seed, difficulty=tier)
    R = _r_from_text(q.text, tier)
    minutes_exact = _minutes_exact(R)
    assert round(minutes_exact) == 60, (
        f"{tier} seed={seed}: expected this pinned seed to land the pre-carry "
        f"minute value at exactly 60 (got {minutes_exact!r}); the carry-firing "
        f"seed for this tier needs re-searching"
    )
    assert q.answer["minutes"] == 0
    assert 0 <= q.answer["minutes"] < 60
    assert q.answer_text.endswith("00′")


# --- boundary-resample guard ---------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_boundary_guard_never_returns_a_half_adjacent_minute(tier):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        R = _r_from_text(q.text, tier)
        minutes_exact = _minutes_exact(R)
        assert abs((minutes_exact % 1) - 0.5) >= 0.05, (
            f"{tier} seed={seed}: minutes_exact={minutes_exact!r} sits on the "
            f".5 rounding boundary the guard is supposed to reject"
        )


# --- zero-padded single-digit minutes round-trip --------------------------

_PAD_SEEDS = {"easy": 2, "medium": 1, "hard": 11}


@pytest.mark.parametrize("tier", TIERS)
def test_zero_padded_minutes_round_trip_through_extract_values(tier):
    from mathgen.grading.spoken import extract_values

    seed = _PAD_SEEDS[tier]
    q = make(TOPIC, seed=seed, difficulty=tier)
    assert 1 <= q.answer["minutes"] <= 9, (
        f"{tier} seed={seed}: expected a single-digit minute value for this "
        f"pinned seed (got {q.answer['minutes']}); the pad seed needs re-searching"
    )
    values = extract_values(q.answer_text)
    assert values == [q.answer["degrees"], q.answer["minutes"]]


# --- distractor hygiene ----------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_differ_from_answer_in_at_least_one_component(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        answer_tuple = (q.answer["degrees"], q.answer["minutes"])
        for distractor in q.distractors:
            m = _DMS_RE.match(distractor)
            assert m, f"{tier} seed={seed}: unparseable distractor {distractor!r}"
            d_tuple = (int(m.group(1)), int(m.group(2)))
            assert d_tuple != answer_tuple, f"{tier} seed={seed}: distractor {distractor!r} equals the answer"


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_from_each_other_and_nonempty_distribution(tier):
    counts = {}
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert len(q.distractors) == len(set(q.distractors)), (
            f"{tier} seed={seed}: duplicate distractors {q.distractors!r}"
        )
        assert q.answer_text not in q.distractors
        n = len(q.distractors)
        counts[n] = counts.get(n, 0) + 1
    # Every seed must produce at least one distractor; the great majority
    # produce all three (some legitimately fewer when a candidate collides
    # with the answer).
    assert min(counts) >= 1, f"{tier}: some seed produced zero distractors: {counts}"
    assert counts.get(3, 0) >= 0.9 * len(SEEDS), f"{tier}: fewer than 90% of seeds produced all 3 distractors: {counts}"


# --- real grader: accepts canonical, rejects distractors and wrong answer --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_rejects_distractors_and_wrong(tier):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert grade_pair(q, q.answer_text).correct, (
            f"{tier} seed={seed}: canonical answer_text {q.answer_text!r} rejected"
        )
        for distractor in q.distractors:
            assert not grade_pair(q, distractor).correct, f"{tier} seed={seed}: distractor {distractor!r} accepted"
        wrong_minutes = (q.answer["minutes"] + 3) % 60
        wrong_text = f"{q.answer['degrees']}°{wrong_minutes:02d}′"
        assert not grade_pair(q, wrong_text).correct, (
            f"{tier} seed={seed}: deliberately wrong answer {wrong_text!r} accepted"
        )


# --- positive/negative control on the already-settled grade_pair behaviour -


def test_grade_pair_positive_control_accepts_the_briefs_typed_strings():
    q = _q(answer={"degrees": 43, "minutes": 16}, answer_text="43°16′")
    for given in ["43°16'", "43° 16′", "43 16", "43,16"]:
        assert grade_pair(q, given).correct, f"expected {given!r} to be accepted"


def test_grade_pair_negative_control_rejects_these_strings():
    q = _q(answer={"degrees": 43, "minutes": 16}, answer_text="43°16′")
    for given in ["16, 43", "43, 17", "43, 15.5", "43.27"]:
        assert not grade_pair(q, given).correct, f"expected {given!r} to be rejected"


# --- symbol presence -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_text_carries_degree_and_minute_symbols(tier):
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert "°" in q.answer_text
        assert "′" in q.answer_text


def test_easy_and_medium_question_text_carries_no_dms_symbols():
    for seed in range(20):
        for tier in ["easy", "medium"]:
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert "°" not in q.text and "′" not in q.text


def test_hard_question_text_carries_no_degree_or_minute_symbol():
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert "°" not in q.text and "′" not in q.text


# --- distinct question texts ------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(tier):
    texts = {make(TOPIC, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct question texts over 60 seeds"


# --- skill loading -----------------------------------------------------


def test_skill_loads_with_expected_shape():
    _domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None, f"skill {SKILL_ID!r} not found"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert len(skill.sources) == 4
    variants = {src.params.get("variant") for src in skill.sources}
    assert variants == {"basic", "context", "fraction_input", "exam_pi_fraction"}
