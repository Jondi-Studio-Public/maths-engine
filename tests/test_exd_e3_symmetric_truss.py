"""Exam source for `pythagoras-truss-chain`
(topic `symmetric_truss`, `mathgen/generators/exd_e3_symmetric_truss_build.py`).

A symmetric truss (half-span, apex height, panel count) is a genuinely
different shape from the skill's existing three "chain of dependent right
triangles" sources: the target is one of three siblings read off the SAME
pair of nested/similar right triangles -- the top chord (a direct Pythagorean
hypotenuse), the apex web at an interior panel point (a NEW hypotenuse), or
the perpendicular web at that panel point (a similar-triangles ratio, not a
second Pythagoras application).

The independent oracle below (`_oracle_symmetric_truss`) parses
`question.text` alone and re-derives the target quantity from scratch with
plain `math`/`fractions.Fraction` -- it imports nothing from
`mathgen.generators.exd_e3_symmetric_truss_build`, so a slip in the
generator's own `check()` calls cannot also be baked into the check that is
supposed to catch it (the failure mode where one wrong formula is baked into both
the generator and its check). `test_oracle_catches_a_corrupted_answer`
proves the oracle is not vacuously true.

`test_hand_verified_instance_*` reproduce two hand-verified
constructions directly (both with plain arithmetic in this file, and by
confirming the generator's own precomputed table contains a matching row) --
these are NOT part of the independent-oracle discipline above and may import
the generator module.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "symmetric_truss"
SKILL_ID = "pythagoras-truss-chain"
SEEDS = range(200)
SHAPE_SEEDS = range(60)


# --- independent oracle -------------------------------------------------

_PROMPT_RE = re.compile(
    r"A symmetric truss has a half-span of (\d+) m, an apex height of (\d+) m, and is "
    r"divided into (\d+) equal panels along each sloped half\. Panel point (\d+) \(out of "
    r"(\d+), counting from the support\) lies (\d+) m from the support, measured along the "
    r"base\. Find the length, in m, of (the top chord|the apex web at panel point (\d+)|"
    r"the perpendicular web at panel point (\d+))"
)


def _oracle_symmetric_truss(text: str) -> str:
    """Re-derive the requested quantity from the rendered prompt alone, using
    the similar-triangles algebra directly,
    never the generator's precomputed table. Verifies the WHOLE artefact the
    learner receives, not just the arithmetic downstream of one stated
    number: the stated panel offset `x` must itself equal `A*j/m` exactly
    (span/height/panel-count/panel-index consistency), and the panel index
    named inside the target phrase ("...at panel point N") must equal the
    one named in the "Panel point j (out of m...)" clause -- a generator bug
    that stated the wrong panel index in the closing sentence would still be
    caught even though this oracle does not itself recompute `x` from
    scratch (the question states it; recomputing it independently is what
    the `x*panels == A*j` check below does)."""
    m = _PROMPT_RE.search(text)
    assert m, f"could not parse symmetric_truss prompt: {text!r}"
    A, H, panels, j, panels2, x = (int(v) for v in m.groups()[:6])
    target_phrase = m.group(7)
    apex_j = m.group(8)
    perp_j = m.group(9)

    assert panels == panels2, f"panel count stated inconsistently: {panels} vs {panels2}"
    assert 1 <= j < panels, f"panel index {j} out of range for {panels} panels"
    # x must equal A*j/m EXACTLY (the construction's own integrality
    # requirement) -- re-derived independently of whatever value the prompt
    # states, not merely trusted.
    assert x * panels == A * j, f"stated panel offset x={x} does not equal A*j/m: A={A}, j={j}, m={panels}"

    chord_sq = A * A + H * H
    L = math.isqrt(chord_sq)
    assert L * L == chord_sq, f"A={A}, H={H} do not form an exact chord"

    if target_phrase.startswith("the top chord"):
        return str(L)

    d = A - x
    if target_phrase.startswith("the apex web"):
        assert apex_j is not None and int(apex_j) == j, (
            f"apex web target names panel point {apex_j}, prompt states panel point {j}"
        )
        diff = d * d + H * H
        root = math.isqrt(diff)
        assert root * root == diff, f"A={A}, H={H}, x={x}, j={j}: apex web is not exact"
        return str(root)

    # "the perpendicular web at panel point N"
    assert perp_j is not None and int(perp_j) == j, (
        f"perpendicular web target names panel point {perp_j}, prompt states panel point {j}"
    )
    frac = Fraction(x * H, L)
    assert frac.denominator == 1, f"A={A}, H={H}, x={x}, L={L}: perpendicular web is not exact"
    return str(int(frac))


@pytest.mark.parametrize("seed", SEEDS)
def test_exam_tier_oracle_agrees(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    expected = _oracle_symmetric_truss(q.text)
    assert expected == q.answer_text, (
        f"seed={seed}: oracle recomputed {expected} from the prompt, generator answer_text is {q.answer_text}\n{q.text}"
    )


def test_oracle_catches_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    corrupted = str(int(q.answer_text) + 7)
    assert _oracle_symmetric_truss(q.text) != corrupted


# --- the two hand-verified instances ----------------------------------------


def test_hand_verified_instance_1_apex_web():
    """(a,b,c)=(4,3,5), t=1 -> A=20, H=15, L=25, m=10. At j=6: x=12, d=8,
    apex web = sqrt(8^2+15^2) = sqrt(289) = 17."""
    A, H, L, m, j = 20, 15, 25, 10, 6
    x = A * j // m
    d = A - x
    assert (x, d) == (12, 8)
    diff = d * d + H * H
    root = math.isqrt(diff)
    assert root * root == diff and root == 17

    from mathgen.generators.exd_e3_symmetric_truss_build import _TABLE

    matches = [e for e in _TABLE["apex_web"] if (e.A, e.H, e.L, e.m, e.j) == (A, H, L, m, j)]
    assert matches, "table has no apex_web row for the hand-verified instance"
    assert matches[0].answer == 17


def test_hand_verified_instance_1_perpendicular_web():
    """Same triple/scale, a different panel point: j=5, x=10, perpendicular
    web = 10*15/25 = 6."""
    A, H, L, m, j = 20, 15, 25, 10, 5
    x = A * j // m
    assert x == 10
    frac = Fraction(x * H, L)
    assert frac.denominator == 1 and int(frac) == 6

    from mathgen.generators.exd_e3_symmetric_truss_build import _TABLE

    matches = [e for e in _TABLE["perp_web"] if (e.A, e.H, e.L, e.m, e.j) == (A, H, L, m, j)]
    assert matches, "table has no perp_web row for the hand-verified instance"
    assert matches[0].answer == 6


def test_hand_verified_instance_2_apex_web_and_perpendicular_web_same_point():
    """(a,b,c)=(4,3,5), t=5 -> A=100, H=75, L=125, m=5. At j=3: x=60, d=40,
    apex web = sqrt(40^2+75^2) = sqrt(7225) = 85; the SAME point's
    perpendicular web = 60*75/125 = 36."""
    A, H, L, m, j = 100, 75, 125, 5, 3
    x = A * j // m
    d = A - x
    assert (x, d) == (60, 40)

    apex_diff = d * d + H * H
    apex_root = math.isqrt(apex_diff)
    assert apex_root * apex_root == apex_diff and apex_root == 85

    perp_frac = Fraction(x * H, L)
    assert perp_frac.denominator == 1 and int(perp_frac) == 36

    from mathgen.generators.exd_e3_symmetric_truss_build import _TABLE

    apex_matches = [e for e in _TABLE["apex_web"] if (e.A, e.H, e.L, e.m, e.j) == (A, H, L, m, j)]
    perp_matches = [e for e in _TABLE["perp_web"] if (e.A, e.H, e.L, e.m, e.j) == (A, H, L, m, j)]
    assert apex_matches and apex_matches[0].answer == 85
    assert perp_matches and perp_matches[0].answer == 36


def test_table_is_healthy_size():
    """Documents the actual pool size the enumeration produced, so a future
    change to the bank (triples/t/m ranges) has a number to compare against."""
    from mathgen.generators.exd_e3_symmetric_truss_build import _TABLE

    assert len(_TABLE["chord"]) >= 500
    assert len(_TABLE["perp_web"]) >= 500
    # apex_web is the sparsest slice by construction (the task file's own
    # warning: "much sparser" than the perpendicular-web condition) -- still
    # comfortably enough for a healthy draw pool.
    assert len(_TABLE["apex_web"]) >= 20


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_exam_tier_deterministic_across_seeds(seed):
    q1 = make(TOPIC, seed=seed, difficulty="exam")
    q2 = make(TOPIC, seed=seed, difficulty="exam")
    assert q1.to_json() == q2.to_json()


# --- distractor hygiene / answer-kind checks --------------------------------


def test_exam_tier_distractors_distinct_and_never_equal_the_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) == 3, f"seed={seed}: expected 3 distractors, got {q.distractors}"


def test_exam_tier_answer_and_every_option_is_a_plain_integer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, int)
        for text in [q.answer_text, *q.distractors]:
            assert re.fullmatch(r"\d+", text), f"seed={seed}: option {text!r} is not a plain integer"
        assert q.answer > 0


# --- variety: at least 20 distinct rendered questions over 60 seeds --------


def test_exam_tier_at_least_20_distinct_questions_over_60_seeds():
    texts = {make(TOPIC, seed=seed, difficulty="exam").text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct exam-tier questions over 60 seeds"


# --- DCM-001: >= 3 distinct prompt shapes over 60 seeds, all 3 targets hit -


def test_exam_tier_clears_dcm_001_three_distinct_shapes():
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam")) for seed in SHAPE_SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes over 60 seeds"


def test_exam_tier_all_three_targets_reachable_over_60_seeds():
    seen = {"chord": 0, "apex_web": 0, "perp_web": 0}
    for seed in SHAPE_SEEDS:
        text = make(TOPIC, seed=seed, difficulty="exam").text
        tail = text.split("Find the length, in m, of ")[-1]
        if tail.startswith("the top chord"):
            seen["chord"] += 1
        elif tail.startswith("the apex web"):
            seen["apex_web"] += 1
        elif tail.startswith("the perpendicular web"):
            seen["perp_web"] += 1
        else:
            raise AssertionError(f"unrecognised target phrase: {tail!r}")
    assert all(count > 0 for count in seen.values()), seen


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
        wrong = str(q.answer + 11)
        assert not grade_number(q, wrong).correct


# --- lower tiers of the skill are untouched by the new topic ---------------


def test_lower_tiers_of_the_skill_still_use_the_chain_topic_unaffected():
    """The new exam source is a NEW topic (`symmetric_truss`), never a new
    preset on the existing `pythagoras_truss_chain` topic -- so the easy/
    medium/hard tiers' own generator module (`t3_pythagoras_truss_chain_
    build.py`) is untouched by construction. Spot check the three lower
    tiers still resolve through their own topic and produce non-empty,
    mutually non-colliding prompt shapes from the new exam-tier ones."""
    exam_shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam")) for seed in range(20)}
    for difficulty in ["easy", "medium", "hard"]:
        q = make("pythagoras_truss_chain", seed=0, difficulty=difficulty)
        assert q.text
        shapes = {prompt_shape(make("pythagoras_truss_chain", seed=seed, difficulty=difficulty)) for seed in range(20)}
        assert shapes, f"{difficulty} tier produced no renders"
        assert not (shapes & exam_shapes), (
            f"{difficulty} tier's prompt shape collides with the new exam tier's: {shapes & exam_shapes}"
        )


# --- skill contract -----------------------------------------------------


def test_skill_contract_has_four_sources_including_exam_on_a_new_topic():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]

    assert skill.answer == "number"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.topic for source in skill.sources] == [
        "pythagoras_truss_chain",
        "pythagoras_truss_chain",
        "pythagoras_truss_chain",
        "symmetric_truss",
    ]

    exam_source = skill.sources[3]
    assert exam_source.topic == TOPIC
    assert exam_source.stages == ["exam"]


def test_skill_still_declares_the_three_misconceptions_the_exam_tier_reuses():
    """Confirms the three reused ids (`stopped-at-intermediate`,
    `added-instead-of-pythagoras`, `wrong-stage-value`) are still declared
    after their descriptions were broadened to cover the exam tier -- does
    not itself check the description WORDING."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[SKILL_ID]
    ids = {m.id for m in skill.misconceptions}
    assert {"stopped-at-intermediate", "added-instead-of-pythagoras", "wrong-stage-value"} <= ids
