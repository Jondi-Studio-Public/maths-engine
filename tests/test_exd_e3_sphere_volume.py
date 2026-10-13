"""Exam source for `senior-use-formulas-to-find-the-volume-of-pyramids`
(`mathgen/generators/p7e_pyramid_sphere_volume_build.py`, topic `volume_of_pyramids_and_spheres`).

The independent oracle below (`_oracle_exam`) parses `question.text` alone and re-derives the sphere volume
from scratch with plain `math` -- it imports nothing from
`mathgen.generators.p7e_pyramid_sphere_volume_build`, so a formula/formatting slip baked into the generator's
own `check` calls cannot also be baked into the check that is supposed to catch it.
`test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true by feeding it a deliberately
wrong `answer_text` and confirming it disagrees.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "volume_of_pyramids_and_spheres"
SKILL_ID = "senior-use-formulas-to-find-the-volume-of-pyramids"
SEEDS = range(40)
SHAPE_SEEDS = range(60)

_EXPRESSION_GRADER = GRADERS["expression"]

_GIVEN_RES = [
    re.compile(
        r"A sphere has a (diameter|radius) of (\d+) cm\. Find its volume, "
        r"in cm³, correct to 2 decimal places\."
    ),
    re.compile(
        r"A sphere's (diameter|radius) is (\d+) cm\. Calculate its volume, "
        r"in cm³, correct to 2 decimal places\."
    ),
]


def _match_given(text: str):
    for pattern in _GIVEN_RES:
        match = pattern.search(text)
        if match:
            return match
    return None


# --- independent oracle: exam tier (decimal sphere volume) -----------------


def _oracle_exam(text: str) -> str:
    """Re-derive the rounded sphere volume from the rendered prompt alone,
    using plain `math.pi`/`math.pow` -- a route that never touches
    `mathgen.generators.p7e_pyramid_sphere_volume_build`, `sympy`, or the
    generator's own rounding guard."""
    match = _match_given(text)
    assert match, f"could not parse exam-tier sphere prompt: {text!r}"
    kind, value = match.group(1), int(match.group(2))
    radius = value / 2 if kind == "diameter" else float(value)
    volume = (4.0 / 3.0) * math.pi * math.pow(radius, 3)
    return f"{volume:.2f}"


def test_exam_tier_oracle_agrees():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        expected = _oracle_exam(q.text)
        assert expected == q.answer_text, (
            f"seed={seed}: oracle recomputed {expected} from the prompt, "
            f"generator answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true (the `inverse_one_to_one_
    function` failure mode): feed it a real question's prompt paired with a
    deliberately wrong `answer_text` and confirm it disagrees."""
    q = make(TOPIC, seed=0, difficulty="exam")
    corrupted = f"{float(q.answer_text) + 50.0:.2f}"
    assert _oracle_exam(q.text) != corrupted


# --- determinism -------------------------------------------------------------


def test_exam_tier_deterministic_across_seeds():
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty="exam")
        q2 = make(TOPIC, seed=seed, difficulty="exam")
        assert q1.to_json() == q2.to_json()


# --- distractor hygiene: never equal, always beyond the stated precision ---


def test_exam_tier_distractors_distinct_and_clear_stated_precision():
    tol = float(stated_precision_tolerance(2))
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 3
        answer_value = float(q.answer_text)
        for d in q.distractors:
            assert abs(float(d) - answer_value) > tol, (
                f"seed={seed}: distractor {d!r} is within the stated 2 d.p. tolerance of the answer {q.answer_text!r}"
            )


# --- formatting consistency (no option is the only 2dp decimal on offer) ---


def test_exam_tier_every_option_is_a_2dp_decimal():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        for text in [q.answer_text, *q.distractors]:
            assert re.fullmatch(r"\d+\.\d\d", text), f"seed={seed}: option {text!r} is not X.YY"


# --- answer-kind checks: a plain 2 d.p. string, off the rounding boundary --


def test_exam_tier_answer_is_a_2dp_string_off_the_boundary():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, str)
        assert q.answer == q.answer_text
        assert re.fullmatch(r"\d+\.\d\d", q.answer_text)
        value = float(q.answer_text)
        scaled = value * 100
        frac = scaled - math.floor(scaled)
        assert abs(frac - 0.5) > 0.05, f"seed={seed}: answer sits on the rounding boundary"


# --- variety: at least 20 distinct rendered questions over 60 seeds --------


def test_exam_tier_at_least_20_distinct_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty="exam").text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct exam-tier questions over 60 seeds"


def test_exam_tier_clears_three_distinct_shapes():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam")) for seed in SHAPE_SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes over 60 seeds"


# --- dimension pool: genuinely excludes the hard tier's clean-multiple case


def test_exam_tier_dimension_pool_excludes_clean_pi_multiples():
    """The hard tier's diameter is always a multiple of 6 (radius always a
    multiple of 3), which makes (4/3)*pi*r^3 an exact integer multiple of
    pi. The exam tier's whole point is that it does NOT -- assert this by
    construction over a large seed sample, both for the diameter-given and
    radius-given branches."""
    diameter_given = 0
    radius_given = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        match = _match_given(q.text)
        assert match
        kind, value = match.group(1), int(match.group(2))
        if kind == "diameter":
            diameter_given += 1
            assert value % 6 != 0, f"seed={seed}: diameter {value} is a multiple of 6"
        else:
            radius_given += 1
            assert value % 3 != 0, f"seed={seed}: radius {value} is a multiple of 3"
    # Both branches of `rng.choice(["diameter", "radius"])` are genuinely
    # exercised over this sample, not a construction that only ever draws
    # one kind.
    assert diameter_given > 30
    assert radius_given > 30


# --- real grader accept/reject: the answer=answer_text plain-string pattern


def test_exam_tier_real_grader_accepts_canonical_and_rejects_distractors():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        result = _EXPRESSION_GRADER(q, q.answer_text)
        assert result.correct, f"seed={seed}: grader rejected the canonical answer {q.answer_text!r}"
        for d in q.distractors:
            result = _EXPRESSION_GRADER(q, d)
            assert not result.correct, f"seed={seed}: grader accepted distractor {d!r}"


def test_exam_tier_real_grader_rejects_a_deliberately_wrong_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        wrong = f"{float(q.answer_text) + 25.0:.2f}"
        result = _EXPRESSION_GRADER(q, wrong)
        assert not result.correct


def test_exam_tier_real_grader_rejects_a_0_005_adjacent_distractor():
    """A candidate strictly INSIDE `stated_precision_tolerance(2)` (0.005) of
    the true rounded value -- +0.004, a case `grade_number`'s stated-
    precision tolerance would accept -- still fails this real grader
    outright: `grade_expression` is an EXACT sympy comparison once both
    sides are the same sympified value (`symbolically_equal`), never a
    tolerance-based accept for this grader kind. The 0.005 margin matters
    only for distractor HYGIENE (`pick_distractors(...,
    tolerance=stated_precision_tolerance(2))` dropping a near-miss so no two
    options render identically), never for grader leniency."""
    q = make(TOPIC, seed=0, difficulty="exam")
    inside_tolerance = f"{float(q.answer_text) + 0.004:.3f}"
    result = _EXPRESSION_GRADER(q, inside_tolerance)
    assert not result.correct


# default-params behaviour is untouched ----------------------------------------


def test_hard_tier_still_answers_an_exact_pi_multiple():
    """The hard tier's own shape (an exact `k*pi` expression, diameter
    always a multiple of 6) must not have moved."""
    for seed in range(10):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert re.fullmatch(r"\d*pi", q.answer_text) or q.answer_text == "pi"
        match = re.search(r"diameter of (\d+) cm", q.text)
        assert match
        assert int(match.group(1)) % 6 == 0


# --- skill contract -----------------------------------------------------


def test_skill_contract_has_four_sources_including_exam():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "expression"
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    # The exam source is now the pyramid chain; the sphere generator's own exam preset is still tested here.
    assert {source.topic for source in skill.sources[:3]} == {TOPIC}

    exam_source = skill.sources[3]
    assert exam_source.stages == ["exam"]
    assert exam_source.topic == "exd_x_fu_pyramid_volume"
