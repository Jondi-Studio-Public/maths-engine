"""Tests `uc2-spherical-triple-integral`: the topic `s2a1_spherical` (easy, medium, hard) and the exam chain
`exd_x_s2a1_spherical`.

The keys are checked against a Cartesian Monte Carlo estimate that reads only the question text (see
`tests/s2a1_oracle.py`), and against the closed forms the families reduce to.
"""

from __future__ import annotations

import re

import pytest
import sympy

import mathgen.generators.s2a1_spherical_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests import s2a1_oracle as oracle

SKILL_ID = "uc2-spherical-triple-integral"
TOPIC = "s2a1_spherical"
EXAM_TOPIC = "exd_x_s2a1_spherical"
SK = oracle.skill(SKILL_ID)
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return oracle.question(TOPIC, tier, seed)


def test_source_wired_last():
    last = SK.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in SK.sources] == ["easy", "medium", "hard", "exam"]
    assert list(SK.year_level) == ["university"] and "university-year-2" in SK.tags


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_cartesian_monte_carlo(tier):
    for seed in range(24):
        q = _q(tier, seed)
        assert oracle.mc_matches(q.text, oracle.key_value(q.answer_text)), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    oracle.check_key_and_distractors(SK, TOPIC, tier, range(60))


@pytest.mark.parametrize("tier", (*TIERS, "exam"))
def test_distractors_are_tagged_with_declared_errors(tier):
    oracle.check_tags(SK, EXAM_TOPIC if tier == "exam" else TOPIC, tier, range(30))


@pytest.mark.parametrize("tier", TIERS)
def test_answers_are_exact_and_positive(tier):
    for seed in range(40):
        q = _q(tier, seed)
        value = oracle.parse_typed(q.answer_text)
        assert value.is_real and value > 0, (seed, q.answer_text)
        assert value.has(sympy.pi) or "xyz" in q.text or "x ≥ 0" in q.text, (seed, q.text, q.answer_text)


def _shape(text: str) -> dict:
    return dict(
        ball="z ≥ 0" not in text
        and "x ≥ 0" not in text
        and re.search(r"^[^≤]*x² \+ y² \+ z² ≤ \d+\.?$|≤ \d+\.$", text) is not None,
        cone="z² , z ≥ 0" in text or re.search(r"z², z ≥ 0", text) is not None,
        octant="x ≥ 0, y ≥ 0, z ≥ 0" in text,
        origin=re.search(r"x² \+ y² \+ z² ≤ \d+z", text) is not None,
        cap=re.search(r"z ≥ [1-9]\d*(?:»|\.|,)", text) is not None and "z ≥ 0" not in text.split("z ≥")[-1][:3],
        shell=re.search(r"\b\d+ ≤ x² \+ y² \+ z² ≤", text) is not None,
    )


def test_each_tier_has_the_shapes_the_design_asks_for():
    medium = [_shape(_q("medium", s).text) for s in range(80)]
    assert sum(f["cone"] for f in medium) >= 20 and sum(f["octant"] for f in medium) >= 15
    hard = [_q("hard", s).text for s in range(90)]
    assert sum(re.search(r"x² \+ y² \+ z² ≤ \d+z", t) is not None for t in hard) >= 12
    assert sum(re.search(r"z ≥ [1-9]\d*", t) is not None for t in hard) >= 12
    assert sum("z ≥ 0" in t and "≤ z²" in t or "z², z ≥ 0" in t for t in hard) >= 12
    easy = [_q("easy", s).text for s in range(60)]
    assert (
        sum("z ≥ 0" in t for t in easy) >= 10
        and sum(re.search(r"\b\d+ ≤ x² \+ y² \+ z² ≤", t) is not None for t in easy) >= 10
    )


def test_the_stems_never_give_the_coordinates_or_the_jacobian():
    for tier in TIERS:
        for seed in range(40):
            text = _q(tier, seed).text
            assert "spherical" not in text and "sin φ" not in text and "θ" not in text and "φ" not in text, text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 14


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


def test_closed_forms():
    pi = sympy.pi
    found_cone = found_cap = found_origin = 0
    for seed in range(200):
        for tier in ("medium", "hard"):
            q = _q(tier, seed)
            value = oracle.parse_typed(q.answer_text)
            cone = re.match(
                r"^Find the volume of the solid (\d+ ≤ )?x² \+ y² \+ z² ≤ (\d+), (x² \+ y² ≤ z²|x² \+ y² ≤ 3z²|3\(x² \+ y²\) ≤ z²), z ≥ 0\.$",
                q.text,
            )
            if cone:
                inner = sympy.sqrt(int(cone.group(1)[:-3])) if cone.group(1) else 0
                outer = sympy.sqrt(int(cone.group(2)))
                cos = {
                    "x² + y² ≤ z²": sympy.sqrt(2) / 2,
                    "x² + y² ≤ 3z²": sympy.Rational(1, 2),
                    "3(x² + y²) ≤ z²": sympy.sqrt(3) / 2,
                }[cone.group(3)]
                assert sympy.simplify(value - 2 * pi * (outer**3 - inner**3) * (1 - cos) / 3) == 0, q.text
                found_cone += 1
            cap = re.match(r"^Find the volume of the solid x² \+ y² \+ z² ≤ (\d+), z ≥ (\d+)\.$", q.text)
            if cap:
                a, h = sympy.sqrt(int(cap.group(1))), int(cap.group(2))
                assert value == pi * (a - h) ** 2 * (2 * a + h) / 3, q.text
                found_cap += 1
            origin = re.match(r"^Find the volume of the solid x² \+ y² \+ z² ≤ (\d+)z\.$", q.text)
            if origin:
                assert value == sympy.Rational(4, 3) * pi * sympy.Rational(int(origin.group(1)), 2) ** 3
                found_origin += 1
    assert found_cone >= 8 and found_cap >= 5 and found_origin >= 5


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def test_exam_shape_and_marks():
    oracle.check_exam_shape(
        SK, EXAM_TOPIC, range(40), [3, 3, 3, 1], [{"expression"}, {"expression"}, {"fraction"}, {"fraction"}]
    )
    for seed in range(20):
        assert find_text_defects(oracle.question(EXAM_TOPIC, "exam", seed)) == []


def test_exam_keys_match_cartesian_monte_carlo():
    for seed in range(20):
        q = oracle.question(EXAM_TOPIC, "exam", seed)
        stem = q.stem
        constraints = oracle.region_constraints(stem)
        k, m = re.search(r"δ = (\d+) \+ (\d*)d", stem).groups()
        rho = int(k) + int(m or 1) * oracle.sympy.sqrt(oracle.x**2 + oracle.y**2 + oracle.z**2)
        volume, mass, moment = (oracle.mc_integral(constraints, f) for f in (1 + 0 * oracle.z, rho, rho * oracle.z))
        keys = [oracle.key_value(p.answer_text) for p in q.parts]
        for estimate, key in zip((volume, mass, moment / mass, mass / volume), keys):
            assert abs(estimate - key) <= 0.04 * abs(key), (seed, stem, estimate, key)


def test_exam_key_accepted_every_distractor_rejected():
    oracle.check_exam_distractors(SK, EXAM_TOPIC, range(40))


def test_follow_through_carries_a_wrong_answer_forward():
    oracle.check_follow_through(SK, EXAM_TOPIC, range(40), minimum=30)


def test_exam_covers_both_solids():
    stems = [oracle.question(EXAM_TOPIC, "exam", s).stem for s in range(60)]
    assert sum("hemispherical" in t for t in stems) >= 15 and sum("conical" in t for t in stems) >= 15


def test_exam_answer_space_and_determinism():
    answers = [oracle.question(EXAM_TOPIC, "exam", seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 40
    assert oracle.question(EXAM_TOPIC, "exam", 7).to_json() == oracle.question(EXAM_TOPIC, "exam", 7).to_json()


def test_a_wrong_reply_is_not_accepted_for_the_whole_chain():
    q = oracle.question(EXAM_TOPIC, "exam", 3)
    assert not grade(SK, q, {p.label: "0" for p in q.parts}).correct
