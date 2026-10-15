"""A fourth, ``exam``-tier source for ``angle-radians-to-dms`` (topic ``t4_radians_to_dms``, ``answer: pair``).

The existing three tiers (``easy``/``basic``, ``medium``/``context``, ``hard``/``fraction_input``) are covered
by ``tests/test_t4_dms_oracle.py``, a separate file.
This file covers only the new ``exam`` tier (``variant="exam_pi_fraction"``), plus the skill-level
checks that changed shape by adding a fourth source.

Independent oracle discipline: ``_oracle_dms``/``_radian_value_exam`` below read only
``question.text``/``question.answer_text`` and re-derive the DMS pair from scratch (floor + round + carry,
written independently of the generator's own ``_degree_decimal_and_minutes``/``_finalize_carry``); they import
nothing from ``mathgen.generators.t4_dms_build``. A handful of tests that need to exercise the generator's own
internal ``_independent_check``/constant table directly (the "two in-generator checks" section) import the
module deliberately and are labelled as such.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import prompt_shape

import mathgen.generators.t4_dms_build as t4_dms_build  # noqa: F401  (registers the topic)

TOPIC = "t4_radians_to_dms"
SKILL_ID = "angle-radians-to-dms"
TIER = "exam"
SEEDS = range(60)

_PI_FRACTION_RE = re.compile(r"(\d*)pi/(\d+) rad")
_DMS_RE = re.compile(r"^(\d+)°(\d{2})′$")


def _radian_value_exam(text: str) -> tuple[float, int, int]:
    """Parse the exam tier's own pi-fraction shape ("5pi/6 rad", "pi/13
    rad") out of the rendered question text; returns (R, k, m)."""
    m = _PI_FRACTION_RE.search(text)
    assert m, f"no pi-fraction found in {text!r}"
    num_str, den_str = m.groups()
    k = int(num_str) if num_str else 1
    denom = int(den_str)
    return k * math.pi / denom, k, denom


def _oracle_dms(R: float) -> tuple[int, int]:
    """Own, from-scratch computation: floor/round/carry, no calls into the generator module at all --
    deliberately duplicates the same small routine ``tests/test_t4_dms_oracle.py`` already wrote for the other
    three tiers, independently"""
    degrees_decimal = R * 180 / math.pi
    degrees = math.floor(degrees_decimal)
    minutes = round((degrees_decimal - degrees) * 60)
    if minutes == 60:
        degrees += 1
        minutes = 0
    return degrees, minutes


# --- determinism -------------------------------------------------------


# --- input shape: a genuine pi-multiple, not a bare rational -------------


def test_question_text_uses_ascii_pi_and_a_context_wrapper():
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=TIER)
        assert "pi" in q.text, f"seed={seed}: expected an ASCII 'pi' token in {q.text!r}"
        assert "π" not in q.text, f"seed={seed}: unicode pi found, ASCII 'pi' required"
        assert not q.text.startswith("Convert "), (
            f"seed={seed}: exam text {q.text!r} still opens with the hard "
            "(fraction_input) tier's bare 'Convert k/m rad ...' shape"
        )
        assert "Express this" in q.text, f"seed={seed}: expected a context-template sentence, got {q.text!r}"


def test_denominator_family_disjoint_from_fraction_input_pool():
    fraction_input_pool = {2, 3, 4, 5, 6, 7, 8, 9}
    exam_pool = set(t4_dms_build._EXAM_PI_DENOMINATORS)
    assert exam_pool.isdisjoint(fraction_input_pool), (
        f"exam denominator family {exam_pool} overlaps fraction_input's {fraction_input_pool}"
    )
    seen = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=TIER)
        _, k, m = _radian_value_exam(q.text)
        seen.add(m)
        assert m in exam_pool, f"seed={seed}: denominator {m} not in {exam_pool}"
        assert math.gcd(k, m) == 1, f"seed={seed}: {k}/{m} not in lowest terms"
    assert len(seen) >= 3, f"only {len(seen)} distinct denominators drawn over 60 seeds: {seen}"


def test_none_of_the_classic_special_angle_denominators_are_drawn():
    """Regression guard for the design defect found while building this
    generator: {3,4,6,12} all divide 180 exactly, so every draw from that family
    lands on a whole degree (minutes always 0) and {8} always lands on
    exactly 30 minutes -- neither exercises the rounding-to-minutes
    machinery this skill teaches, nor leaves two of the three misconception
    distractors constructible."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=TIER)
        _, _k, m = _radian_value_exam(q.text)
        assert m not in (3, 4, 6, 8, 12), f"seed={seed}: drew retired denominator {m}"


def test_minutes_component_is_genuinely_exercised_not_trivial():
    minutes_seen = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=TIER)
        minutes_seen.add(q.answer["minutes"])
    assert len(minutes_seen) >= 10, (
        f"only {len(minutes_seen)} distinct minute values over 60 seeds: {minutes_seen} "
        "-- the DMS rounding machinery is not being genuinely exercised"
    )
    assert minutes_seen != {0}, "every draw rounded to exactly 0 minutes"


# --- boundary-resample guard ---------------------------------------------


def test_boundary_guard_never_returns_a_half_adjacent_minute():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=TIER)
        R, _k, _m = _radian_value_exam(q.text)
        degree_decimal = R * 180 / math.pi
        D = math.floor(degree_decimal)
        minutes_exact = (degree_decimal - D) * 60
        assert abs((minutes_exact % 1) - 0.5) >= 0.05, (
            f"seed={seed}: minutes_exact={minutes_exact!r} sits on the .5 "
            "rounding boundary the guard is supposed to reject"
        )


# --- distractor hygiene ----------------------------------------------------


def test_distractors_distinct_nonempty_and_never_equal_the_answer():
    counts = {}
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=TIER)
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: duplicate distractors {q.distractors!r}"
        assert q.answer_text not in q.distractors
        answer_tuple = (q.answer["degrees"], q.answer["minutes"])
        for distractor in q.distractors:
            m = _DMS_RE.match(distractor)
            assert m, f"seed={seed}: unparseable distractor {distractor!r}"
            d_tuple = (int(m.group(1)), int(m.group(2)))
            assert d_tuple != answer_tuple, f"seed={seed}: distractor {distractor!r} equals the answer"
        n = len(q.distractors)
        counts[n] = counts.get(n, 0) + 1
    assert min(counts) >= 1, f"some seed produced zero distractors: {counts}"
    assert counts.get(3, 0) >= 0.9 * len(SEEDS), f"fewer than 90% of seeds produced all 3 distractors: {counts}"


# --- real grader: accepts canonical, rejects distractors and wrong answer --


def test_real_grader_accepts_canonical_rejects_distractors_and_wrong():
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=TIER)
        assert grade_pair(q, q.answer_text).correct, f"seed={seed}: canonical answer_text {q.answer_text!r} rejected"
        for distractor in q.distractors:
            assert not grade_pair(q, distractor).correct, f"seed={seed}: distractor {distractor!r} accepted"
        wrong_minutes = (q.answer["minutes"] + 3) % 60
        wrong_text = f"{q.answer['degrees']}°{wrong_minutes:02d}′"
        assert not grade_pair(q, wrong_text).correct, f"seed={seed}: deliberately wrong answer {wrong_text!r} accepted"


# --- answer-kind checks ----------------------------------------------------


def test_answer_is_a_well_formed_dms_pair():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=TIER)
        D, minutes = q.answer["degrees"], q.answer["minutes"]
        assert isinstance(D, int)
        assert isinstance(minutes, int)
        assert 0 <= minutes < 60
        assert q.answer_text == f"{D}°{minutes:02d}′"
        assert "°" in q.answer_text and "′" in q.answer_text


def test_question_text_carries_no_dms_symbols():
    for seed in range(20):
        q = make(TOPIC, seed=seed, difficulty=TIER)
        assert "°" not in q.text and "′" not in q.text


# --- independent oracle -----------------------------------------------------


def test_oracle_recomputes_the_dms_pair_from_the_question_alone():
    disagreements = []
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=TIER)
        R, _k, _m = _radian_value_exam(q.text)
        m = _DMS_RE.match(q.answer_text)
        assert m, f"seed={seed}: unparseable answer_text {q.answer_text!r}"
        rendered = (int(m.group(1)), int(m.group(2)))
        oracle = _oracle_dms(R)
        if oracle != rendered:
            disagreements.append((seed, R, rendered, oracle))
    assert not disagreements, (
        f"oracle disagreed with the rendered answer on {len(disagreements)} of 40 seeds: {disagreements}"
    )


def test_oracle_catches_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty=TIER)
    R, _k, _m = _radian_value_exam(q.text)
    oracle = _oracle_dms(R)
    corrupted_degrees = oracle[0] + 5
    corrupted_text = f"{corrupted_degrees}°{oracle[1]:02d}′"
    m = _DMS_RE.match(corrupted_text)
    corrupted = (int(m.group(1)), int(m.group(2)))
    assert corrupted != oracle, "oracle failed to disagree with a deliberately wrong answer"


# --- the in-generator independent check, exercised directly ----------------
# (deliberately imports the generator module -- this is the internal-check
# test, not the independent oracle above)


def test_independent_check_fires_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty=TIER)
    R, _k, _m = _radian_value_exam(q.text)
    correct_D, correct_minutes = q.answer["degrees"], q.answer["minutes"]
    # correct call does not raise
    t4_dms_build._independent_check(R, correct_D, correct_minutes)
    with pytest.raises(VerificationError):
        t4_dms_build._independent_check(R, correct_D + 1, correct_minutes)
    with pytest.raises(VerificationError):
        t4_dms_build._independent_check(R, correct_D, (correct_minutes + 5) % 60)


# text variety --------------


def test_at_least_20_distinct_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty=TIER).text for seed in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct question texts over 60 seeds"


def test_at_least_3_distinct_prompt_shapes_over_60_seeds():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=TIER)) for seed in SEEDS}
    assert len(shapes) >= 3, f"prompt variety: only {len(shapes)} distinct prompt_shape() outputs: {shapes}"


def test_exam_prompt_shapes_are_new_relative_to_every_lower_tier():
    exam_shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=TIER)) for seed in SEEDS}
    for lower_tier in ("easy", "medium", "hard"):
        lower_shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=lower_tier)) for seed in SEEDS}
        assert exam_shapes.isdisjoint(lower_shapes), (
            f"exam tier shares a prompt_shape() with {lower_tier!r}: {exam_shapes & lower_shapes}"
        )


# --- skill loading -----------------------------------------------------


def test_skill_loads_with_four_sources_including_exam():
    _domains, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None, f"skill {SKILL_ID!r} not found"
    assert skill.answer == "pair"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["angle-radians-to-degrees-decimal"]
    assert len(skill.sources) == 4
    variants = {src.params.get("variant") for src in skill.sources}
    assert variants == {"basic", "context", "fraction_input", "exam_pi_fraction"}
    exam_source = next(src for src in skill.sources if src.params.get("variant") == "exam_pi_fraction")
    assert exam_source.difficulty == "exam"
    assert exam_source.stages == ["exam"]
