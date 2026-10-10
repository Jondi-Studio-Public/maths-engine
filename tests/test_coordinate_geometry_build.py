"""Focused contract tests for the coordinate-geometry pilot
(midpoint-two-points, distance-two-points -- AC9M9A03)."""

import re
from pathlib import Path

import pytest
import sympy
from sympy import Rational, sqrt

from mathgen.grading.typed import grade_expression, grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.coordinate_geometry_build import _verify_distance, _verify_midpoint

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


@pytest.mark.parametrize("topic", ["midpoint_two_points", "distance_two_points"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- midpoint_two_points -----------------------------------------------


def test_midpoint_grades_correct_answer_and_rejects_distractors():
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("midpoint_two_points", seed=seed, difficulty=difficulty)
            assert grade_pair(question, question.answer_text).correct
            for distractor in question.distractors:
                assert not grade_pair(question, distractor).correct


def test_midpoint_swapped_pair_is_rejected():
    """grade_pair compares positionally: swapping x/y must not grade correct
    unless the two components happen to coincide."""
    swaps_checked = 0
    for seed in SEEDS:
        question = make("midpoint_two_points", seed=seed, difficulty="medium")
        mx, my = question.answer["x"], question.answer["y"]
        if mx == my:
            continue
        swapped = f"x = {my}, y = {mx}"
        assert not grade_pair(question, swapped).correct
        swaps_checked += 1
    assert swaps_checked > 0


def test_midpoint_grades_correct_from_bracketed_pair_notation():
    """Learners write midpoints as "(3.5, 3)", not the canonical "x = 3.5,
    y = 3" -- confirm the grader accepts that shape too."""
    question = make("midpoint_two_points", seed=1, difficulty="easy")
    mx, my = question.answer["x"], question.answer["y"]
    assert grade_pair(question, f"({mx}, {my})").correct


def test_midpoint_both_integer_and_half_integer_cases_occur():
    """A preset that only ever produced integer midpoints would hide the
    interesting exact-half-integer case -- confirm both appear across a
    broad seed sample at every difficulty. Judged from the canonical
    answer_text (rational_text emits "p/q" only when q != 1), not from the
    stored answer's float type."""
    for difficulty in ("easy", "medium", "hard"):
        saw_integer = False
        saw_half_integer = False
        for seed in range(200):
            question = make("midpoint_two_points", seed=seed, difficulty=difficulty)
            if "/" in question.answer_text:
                saw_half_integer = True
            else:
                saw_integer = True
            if saw_integer and saw_half_integer:
                break
        assert saw_integer, f"{difficulty}: never saw an integer midpoint"
        assert saw_half_integer, f"{difficulty}: never saw a half-integer midpoint"


def test_midpoint_known_half_integer_case_stays_exact():
    """A half-integer midpoint (at least one component with denominator 2)
    must stay exact in its canonical text form ("7/2", never "3.5" or a
    rounded decimal), and must grade correct both from that exact text and
    from an equivalent decimal. `Question.answer` stores a plain float for
    JSON-serialisability, but every value here is n/2 for integer n, which a
    binary float represents exactly -- so this loses no precision; confirmed
    directly against a sympy.Rational parse of the canonical text."""
    for seed in range(200):
        question = make("midpoint_two_points", seed=seed, difficulty="easy")
        if "/" in question.answer_text:
            break
    else:
        pytest.fail("no half-integer midpoint found in 200 easy seeds")
    mx, my = question.answer["x"], question.answer["y"]
    exact_x = sympy.Rational(mx)
    exact_y = sympy.Rational(my)
    assert exact_x.q == 2 or exact_y.q == 2
    assert float(exact_x) == mx and float(exact_y) == my  # no rounding loss
    assert grade_pair(question, question.answer_text).correct
    assert grade_pair(question, f"x = {mx}, y = {my}").correct


def test_midpoint_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_midpoint(0, 0, 4, 4, Rational(3), Rational(2))
    # A genuinely correct midpoint must still pass.
    _verify_midpoint(0, 0, 4, 4, Rational(2), Rational(2))


# --- distance_two_points -------------------------------------------------


def test_distance_grades_correct_answer_and_rejects_distractors():
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("distance_two_points", seed=seed, difficulty=difficulty)
            assert grade_expression(question, question.answer_text).correct
            for distractor in question.distractors:
                assert not grade_expression(question, distractor).correct


def test_distance_both_rational_and_surd_cases_occur():
    """Sample both Pythagorean-triple (rational) separations and genuine
    surds across a broad seed sample at every difficulty. `Question.answer`
    is the same canonical text sympy.sympify parses inside grade_expression
    itself, so this exercises the real grading path, not a shortcut."""
    for difficulty in ("easy", "medium", "hard"):
        saw_rational = False
        saw_surd = False
        for seed in range(200):
            question = make("distance_two_points", seed=seed, difficulty=difficulty)
            if sympy.sympify(question.answer).is_rational:
                saw_rational = True
            else:
                saw_surd = True
            if saw_rational and saw_surd:
                break
        assert saw_rational, f"{difficulty}: never saw a rational (Pythagorean-triple) distance"
        assert saw_surd, f"{difficulty}: never saw a surd distance"


def test_distance_known_surd_case_stays_a_surd():
    """A surd distance must stay an exact simplified radical expression,
    never a rounded decimal like 2.236."""
    for seed in range(200):
        question = make("distance_two_points", seed=seed, difficulty="easy")
        if not sympy.sympify(question.answer).is_rational:
            break
    else:
        pytest.fail("no surd distance found in 200 easy seeds")
    assert "sqrt" in question.answer_text
    assert "." not in question.answer_text
    assert grade_expression(question, question.answer_text).correct


_SIMPLIFIED_SURD_RE = re.compile(r"^\d+\*sqrt\(\d+\)$")


def test_distance_surd_is_simplified_not_left_as_a_bare_radicand():
    """A radicand with a square factor (e.g. dx^2+dy^2 == 8) must simplify
    to a coefficient times a smaller surd (2*sqrt(2)), never appear as the
    unsimplified sqrt(8)."""
    found = False
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(300):
            question = make("distance_two_points", seed=seed, difficulty=difficulty)
            if _SIMPLIFIED_SURD_RE.match(question.answer_text):
                found = True
                assert grade_expression(question, question.answer_text).correct
                assert "sqrt(8)" not in question.answer_text
                break
        if found:
            break
    assert found, "never observed a square-factor radicand to confirm surd simplification"


def test_distance_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_distance(3, 4, sqrt(24))  # should be sqrt(25) = 5
    # A genuinely correct distance must still pass.
    _verify_distance(3, 4, sqrt(25))


# --- skill contracts -------------------------------------------------


def test_coordinate_pilot_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    midpoint = by_id["midpoint-two-points"]
    distance = by_id["distance-two-points"]

    assert midpoint.prerequisites == ["coordinate-plane-readiness", "integer-arithmetic"]
    assert midpoint.answer == "pair"
    assert midpoint.placement_eligible is False
    assert midpoint.status == "active"
    assert [source.stages for source in midpoint.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert [source.difficulty for source in midpoint.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in midpoint.sources if source.difficulty != "exam"} == {"midpoint_two_points"}

    assert distance.prerequisites == ["coordinate-plane-readiness", "pythagorean-hypotenuse"]
    assert distance.answer == "expression"
    assert distance.placement_eligible is False
    assert distance.status == "active"
    assert [source.stages for source in distance.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert [source.difficulty for source in distance.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in distance.sources if source.difficulty != "exam"} == {"distance_two_points"}
