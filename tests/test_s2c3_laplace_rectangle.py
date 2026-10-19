"""Tests `uc2-laplace-equation-rectangle` (topic `s2c3_laplace_rectangle`) and its exam chain.

The oracle reads the plate and the edge profile back out of the wording and re-solves it edge by edge from
Simpson-integrated coefficients, then (for a sample) by Jacobi iteration on a grid.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest

import mathgen.generators.s2c3_laplace_rectangle_build  # noqa: F401  (registers the topics)
from _s2c3_helpers import (
    assert_exam_grading,
    assert_follow_through,
    assert_key_accepted_and_distractors_rejected,
    assert_text_quality_and_determinism,
    skill_by_id,
    spread,
    tier_question,
)
from _s2c3_laplace_oracle import Plate
from mathgen import make

TOPIC = "s2c3_laplace_rectangle"
EXAM = "exd_x_s2c3_laplace_rectangle"
SKILL = skill_by_id("uc2-laplace-equation-rectangle")
TIERS = ("easy", "medium", "hard")
_FIND = re.compile(r"Find u\(([\d.]+), ([\d.]+)\)")
_SUB = {c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}


def test_skill_wiring():
    assert SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]
    assert "uc2-fourier-half-range" in SKILL.prerequisites
    assert SKILL.year_level == ("university",)


def test_easy_matches_the_edge_by_edge_solution():
    edges = Counter()
    for seed in range(60):
        q = tier_question(TOPIC, "easy", seed)
        plate = Plate(q.text)
        assert len(plate.edges) == 1
        edges.update(list(plate.edges))
        x0, y0 = (float(v) for v in _FIND.search(q.text).groups())
        assert abs(plate.u(x0, y0) - float(q.answer)) < 6e-4, (seed, q.text, q.answer_text)
    assert len(edges) == 4


def test_medium_coefficient_is_the_sine_integral_of_the_edge():
    for seed in range(80):
        q = tier_question(TOPIC, "medium", seed)
        plate = Plate(q.text)
        (edge,) = plate.edges
        n = int(_SUB[re.search(r"Find B([₀-₉])", q.text).group(1)])
        assert abs(plate.coefficient(edge, n) - float(q.answer)) < 6e-4, (seed, q.text, q.answer_text)


def test_hard_covers_two_hot_edges_and_the_centre():
    kinds = Counter()
    for seed in range(80):
        q = tier_question(TOPIC, "hard", seed)
        plate = Plate(q.text)
        if len(plate.edges) == 2:
            kinds["two"] += 1
            x0, y0 = (float(v) for v in _FIND.search(q.text).groups())
            tol = 6e-4
        else:
            kinds["centre"] += 1
            x0, y0 = plate.a / 2, plate.b / 2
            tol = 6e-3
            assert "centre" in q.text and q.answer_text.split(".")[1].__len__() == 2
        assert abs(plate.u(x0, y0) - float(q.answer)) < tol, (seed, q.text, q.answer_text)
    assert kinds["two"] >= 15 and kinds["centre"] >= 15


@pytest.mark.parametrize("tier", ("easy", "hard"))
def test_jacobi_iteration_agrees_on_a_sample(tier):
    checked = 0
    for seed in range(0, 30, 6):
        q = tier_question(TOPIC, tier, seed)
        plate = Plate(q.text)
        m = _FIND.search(q.text)
        x0, y0 = (float(v) for v in m.groups()) if m else (plate.a / 2, plate.b / 2)
        # grid points only: the question points are on quarter lengths, which the 40-cell grid contains
        assert abs(plate.jacobi(x0, y0) - float(q.answer)) < 0.03 * (1 + abs(float(q.answer))) + 0.5, (seed, q.text)
        checked += 1
    assert checked == 5


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    assert_key_accepted_and_distractors_rejected(SKILL, TOPIC, tier, minimum=3)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    assert_text_quality_and_determinism(TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    spread(TOPIC, tier, minimum_answers=20, minimum_texts=50)


def test_exam_shape_and_keys():
    assert_exam_grading(SKILL, EXAM, [3, 2, 2, 3], ["number"] * 4)
    q = make(EXAM, seed=1, difficulty="exam")
    assert q.parts[1].follow_through == "part_a/3"
    assert q.parts[2].follow_through is not None and q.parts[0].follow_through is None
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        plate = Plate(q.stem)
        assert list(plate.edges) == ["top"]
        assert abs(plate.coefficient("top", 1) - float(q.parts[0].answer)) < 6e-4, seed
        assert abs(plate.coefficient("top", 3) - float(q.parts[1].answer)) < 6e-4, seed
        first = (
            plate.coefficient("top", 1)
            * math.sinh(math.pi * plate.b / 2 / plate.a)
            / math.sinh(math.pi * plate.b / plate.a)
        )
        assert abs(first - float(q.parts[2].answer)) < 6e-4, seed
        assert abs(plate.u(plate.a / 2, plate.b / 2) - float(q.parts[3].answer)) < 6e-3, seed


def test_exam_follow_through():
    def wrong_a(q):
        return "50.000" if abs(float(q.parts[0].answer) - 50) > 1 else "70.000"

    assert_follow_through(SKILL, EXAM, "b", {"a": wrong_a}, lambda q, r: f"{float(r['a']) / 3:.3f}")

    def first_mode(q, r):
        plate = Plate(q.stem)
        return (
            f"{float(r['a']) * math.sinh(math.pi * plate.b / 2 / plate.a) / math.sinh(math.pi * plate.b / plate.a):.3f}"
        )

    assert_follow_through(SKILL, EXAM, "c", {"a": wrong_a}, first_mode)
