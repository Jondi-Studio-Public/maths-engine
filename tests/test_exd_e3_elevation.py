"""Exam source for `senior-work-with-the-concepts-of-angle-of-elevation` (topic `elevation_depression_problem`,
variant `twin_base_distance`, `mathgen/generators/p7a_trig_ratio_elevation_build.py`).

The existing hard tier (`variant="two_observation"`) gives the horizontal gap
between two observation points plus both elevation angles and asks for the
tower's HEIGHT. The exam tier runs the SAME two right triangles the OTHER
direction: height and both angles are given, and the question asks for the
HORIZONTAL DISTANCE between the two observation points -- a genuine
"which quantity is unknown" escalation, not a bigger number.

The independent oracle below (`_oracle_exam`) parses `question.text` alone and re-derives the distance from
scratch with plain `math` -- it imports nothing from `mathgen.generators.p7a_trig_ratio_elevation_build`, so a
sign/ formatting slip in the generator's own `check` calls cannot also be baked into the check that is
supposed to catch it. `test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true by
feeding it a deliberately wrong `answer_text` and confirming it disagrees.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "elevation_depression_problem"
SKILL_ID = "senior-work-with-the-concepts-of-angle-of-elevation"
SEEDS = range(40)
SHAPE_SEEDS = range(60)


# --- independent oracle: exam tier -----------------------------------------

# One regex per phrasing `phrase()` can pick (mathgen/generators/
# p7a_trig_ratio_elevation_build.py's `_EXAM_TWIN_BASE_DISTANCE_TEMPLATES`).
# Each captures H (tower height), theta1 (near/steeper observer) and theta2
# (far/shallower observer) -- in whatever ORDER that template states them --
# and the oracle reorders them itself before computing.

_TEMPLATE_A_RE = re.compile(
    r"A tower is (\d+) m tall\. Observers A and B stand on level ground on the same side "
    r"of the tower, with A, B and the base of the tower collinear\. From A, the angle of "
    r"elevation to the top of the tower is (\d+) degrees\. From B, further from the "
    r"tower than A, the angle of elevation to the top is (\d+) degrees\. Find the "
    r"distance AB, to 1 decimal place\."
)
_TEMPLATE_B_RE = re.compile(
    r"Two survey markers, P and Q, lie on level ground on the same side of a flagpole, "
    r"with P, Q and the base of the flagpole collinear\. The angle of elevation from P to "
    r"the top of the flagpole is (\d+) degrees, and from Q, which is further from the "
    r"flagpole than P, it is (\d+) degrees\. The flagpole is (\d+) m tall\. Find the "
    r"distance PQ, to 1 decimal place\."
)
_TEMPLATE_C_RE = re.compile(
    r"A lighthouse is (\d+) m tall\. A ship's navigator measures the angle of elevation to "
    r"the top of the lighthouse while sailing directly away from it along a single straight "
    r"line, on the same side: (\d+) degrees from the nearer position, then (\d+) "
    r"degrees from a farther position\. How far apart, to 1 decimal place, are the two "
    r"positions\?"
)


def _oracle_exam(text: str) -> str:
    """Re-derive the horizontal distance between the two observers from the
    rendered prompt alone: each observer's own distance to the base is
    H/tan(theta) (first principles, the same relationship the easy tier's
    single-triangle question teaches), and the answer is their difference."""
    m = _TEMPLATE_A_RE.search(text)
    if m:
        H, theta1, theta2 = (int(v) for v in m.groups())
    else:
        m = _TEMPLATE_B_RE.search(text)
        if m:
            theta1, theta2, H = (int(v) for v in m.groups())
        else:
            m = _TEMPLATE_C_RE.search(text)
            if m:
                H, theta1, theta2 = (int(v) for v in m.groups())
            else:
                raise AssertionError(f"could not parse exam-tier prompt: {text!r}")

    x1 = H / math.tan(math.radians(theta1))
    x2 = H / math.tan(math.radians(theta2))
    d = x2 - x1
    return f"{d:.1f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_exam_tier_oracle_agrees(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    expected = _oracle_exam(q.text)
    assert expected == q.answer_text, (
        f"seed={seed}: oracle recomputed {expected} from the prompt, generator answer_text is {q.answer_text}\n{q.text}"
    )


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true (the `inverse_one_to_one_
    function` failure mode): feed it a real question's prompt paired with a
    deliberately wrong `answer_text` and confirm it disagrees."""
    q = make(TOPIC, seed=0, difficulty="exam")
    corrupted = f"{round(float(q.answer_text) + 5.0, 1):.1f}"
    assert _oracle_exam(q.text) != corrupted


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_exam_tier_deterministic_across_seeds(seed):
    q1 = make(TOPIC, seed=seed, difficulty="exam")
    q2 = make(TOPIC, seed=seed, difficulty="exam")
    assert q1.to_json() == q2.to_json()


# --- distractor hygiene ----------------------------------------------------


def test_exam_tier_distractors_distinct_and_clear_stated_precision():
    # `pick_distractors` silently drops a candidate that lands within the
    # stated-precision tolerance of the true answer -- the same behaviour
    # the pre-existing easy/hard tiers of this generator already exhibit
    # (measured over 300 seeds: easy drops to 2 distractors 5/300, hard
    # 1/300, this exam tier 2/300 -- e.g. seed 92: H=15, theta1=43,
    # theta2=25 gives x1=H/tan(43)=16.1 and d=x2-x1=16.1, so the
    # single-observation "near_only" candidate coincides with the true
    # answer by construction and is correctly dropped rather than offered
    # as a second correct option). >= 2 is the measured floor, not a
    # guaranteed 3.
    tol = stated_precision_tolerance(1)
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 2
        answer_value = float(q.answer_text)
        for d in q.distractors:
            assert abs(float(d) - answer_value) > tol, (
                f"seed={seed}: distractor {d!r} is within the stated 1 d.p. tolerance of the answer {q.answer_text!r}"
            )


def test_exam_tier_seed_92_hits_the_measured_2_distractor_floor():
    """Pins the specific seed the module docstring above measures (H=15,
    theta1=43, theta2=25: the near-only candidate coincides with the true
    answer and is dropped), and confirms the real grader is still correct
    with only 2 options on offer."""
    q = make(TOPIC, seed=92, difficulty="exam")
    assert q.answer_text == "16.1"
    assert len(q.distractors) == 2
    assert grade_number(q, q.answer_text).correct
    for d in q.distractors:
        assert not grade_number(q, d).correct


# --- formatting consistency (no option is the only decimal/integer on offer)


def test_exam_tier_every_option_is_a_1dp_decimal():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        for text in [q.answer_text, *q.distractors]:
            assert re.fullmatch(r"\d+\.\d", text), f"seed={seed}: option {text!r} is not X.Y"
        assert not q.answer_text.endswith(".0"), (
            f"seed={seed}: answer {q.answer_text!r} displays as a whole number, hiding "
            f"the decimal reasoning step this skill's precision convention exists to add"
        )


# --- answer-kind checks ------------------------------------------------------


def test_exam_tier_answer_is_a_float_rounded_to_1dp_and_off_the_boundary():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, float)
        assert q.answer == round(q.answer, 1)
        scaled = abs(q.answer) * 10
        frac = scaled - math.floor(scaled)
        assert abs(frac - 0.5) > 0.05, f"seed={seed}: answer sits on the rounding boundary"
        assert q.answer > 0


# --- variety: at least 20 distinct rendered questions over 60 seeds --------


def test_exam_tier_at_least_20_distinct_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty="exam").text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct exam-tier questions over 60 seeds"


def test_exam_tier_clears_three_distinct_shapes():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam")) for seed in SHAPE_SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes over 60 seeds"


def test_exam_tier_every_template_is_reachable_over_60_seeds():
    """Regression guard for a previously fixed defect."""
    matched = {"a": 0, "b": 0, "c": 0}
    for seed in SHAPE_SEEDS:
        text = make(TOPIC, seed=seed, difficulty="exam").text
        if _TEMPLATE_A_RE.search(text):
            matched["a"] += 1
        elif _TEMPLATE_B_RE.search(text):
            matched["b"] += 1
        elif _TEMPLATE_C_RE.search(text):
            matched["c"] += 1
    assert all(count > 0 for count in matched.values()), matched


# --- real grader accept/reject ----------------------------------------------


def test_exam_tier_real_grader_accepts_canonical_and_rejects_distractors():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


def test_exam_tier_real_grader_rejects_a_deliberately_wrong_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        wrong = round(q.answer + 5.0, 1)
        assert not grade_number(q, f"{wrong:.1f}").correct


# default-params behaviour is untouched -----------------------


def test_easy_medium_hard_tiers_are_still_unaffected_by_the_new_preset():
    """Spot check that adding the "exam" preset key did not disturb the existing three presets' own rendered
    output -- the render-baseline replay is the authoritative proof of this across the whole library."""
    exam_shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam")) for seed in range(20)}
    for difficulty in ["easy", "medium", "hard"]:
        shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=difficulty)) for seed in range(20)}
        assert shapes, f"{difficulty} tier produced no renders"
        assert not (shapes & exam_shapes), (
            f"{difficulty} tier's prompt shape collides with the new exam tier's: {shapes & exam_shapes}"
        )


# --- skill contract -----------------------------------------------------


def test_skill_contract_has_four_sources_including_exam():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "number"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC}

    exam_source = skill.sources[3]
    assert exam_source.params.get("variant") == "twin_base_distance"
    assert exam_source.stages == ["exam"]


def test_skill_still_declares_the_two_misconceptions_the_exam_tier_reuses():
    """Regression guard for a previously fixed defect."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]
    ids = {m.id for m in skill.misconceptions}
    assert {"single-observation-shortcut", "opposite-sides-misreading"} <= ids
