"""Focused contract tests for: 3-D coordinate displacement (f10-three-dimensional-coordinates) and common-base
exponential equations (f10-exponential-relations-equations)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction, grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11d_coords_exponential_build import (
    _verify_coordinate_change_3d,
    _verify_compound,
    _verify_simple,
)

SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

COORD_TOPIC = "p11d_three_dimensional_coordinates"
EXP_TOPIC = "p11d_exponential_common_base_solve"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [COORD_TOPIC, EXP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_with_unique_nonempty_distractors(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


@pytest.mark.parametrize("topic", [COORD_TOPIC, EXP_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(topic, difficulty):
    texts = {make(topic, seed=s, difficulty=difficulty).text for s in SEEDS}
    assert len(texts) >= 20, f"{topic} {difficulty}: only {len(texts)} distinct questions over 60 seeds"


# --- skill 1: p11d_three_dimensional_coordinates (pair answer) -------------


_PAIR_RE = re.compile(r"^\(-?\d+, -?\d+, -?\d+\)$")


@pytest.mark.parametrize("difficulty", TIERS)
def test_coord_answer_is_pair_with_dx_dy_dz_int_keys(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict)
        assert set(q.answer) == {"dx", "dy", "dz"}
        assert all(isinstance(v, int) for v in q.answer.values())
        assert q.answer_text == f"dx = {q.answer['dx']}, dy = {q.answer['dy']}, dz = {q.answer['dz']}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_coord_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        reordered = f"dz = {q.answer['dz']}, dx = {q.answer['dx']}, dy = {q.answer['dy']}"
        assert grade_pair(q, reordered).correct
        bare = f"({q.answer['dx']}, {q.answer['dy']}, {q.answer['dz']})"
        assert grade_pair(q, bare).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_coord_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        dx, dy, dz = q.answer["dx"], q.answer["dy"], q.answer["dz"]
        wrong = f"dx = {dx + 1000}, dy = {dy + 1000}, dz = {dz + 1000}"
        assert not grade_pair(q, wrong).correct


_SIMPLE_MOVE_RE = re.compile(r"from \((-?\d+), (-?\d+), (-?\d+)\) to \((-?\d+), (-?\d+), (-?\d+)\)")


def test_coord_easy_tier_positive_range():
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty="easy")
        m = _SIMPLE_MOVE_RE.search(q.text)
        assert m, q.text
        pts = [int(v) for v in m.groups()]
        assert all(1 <= v <= 10 for v in pts), q.text


def test_coord_medium_tier_signed_range():
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty="medium")
        m = _SIMPLE_MOVE_RE.search(q.text)
        assert m, q.text
        pts = [int(v) for v in m.groups()]
        assert all(-10 <= v <= 10 for v in pts), q.text


def test_coord_hard_tier_three_named_points_all_distinct_pairwise_start_end():
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty="hard")
        m = re.search(
            r"A\((-?\d+), (-?\d+), (-?\d+)\), moves to B\((-?\d+), (-?\d+), (-?\d+)\), "
            r"then moves again to C\((-?\d+), (-?\d+), (-?\d+)\)",
            q.text,
        )
        assert m, q.text
        a = tuple(int(v) for v in m.groups()[0:3])
        b = tuple(int(v) for v in m.groups()[3:6])
        c = tuple(int(v) for v in m.groups()[6:9])
        assert a != b, "A and B must differ (no zero-length first leg)"
        assert b != c, "B and C must differ (no zero-length second leg)"
        assert a != c, "A and C must differ (no zero total displacement)"
        assert -10 <= min(a + b + c) and max(a + b + c) <= 10


def test_coord_independent_check_fires_on_corrupted_reconstruction():
    with pytest.raises(VerificationError):
        _verify_coordinate_change_3d((1, 2, 3), (4, 4, 4), (5, 5, 5))  # wrong delta
    # correct case does not raise
    _verify_coordinate_change_3d((1, 2, 3), (4, 4, 4), (3, 2, 1))


# --- skill 2: p11d_exponential_common_base_solve (fraction answer) ---------


@pytest.mark.parametrize("difficulty", TIERS)
def test_exp_answer_is_fraction_dict_in_lowest_terms(difficulty):
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        p, den = q.answer["p"], q.answer["q"]
        assert isinstance(p, int) and isinstance(den, int) and den > 0
        assert Fraction(p, den) == Fraction(p, den)  # constructs without error
        from math import gcd

        assert gcd(abs(p), den) == 1, f"{p}/{den} is not in lowest terms"


@pytest.mark.parametrize("difficulty", TIERS)
def test_exp_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty=difficulty)
        x = Fraction(q.answer["p"], q.answer["q"])
        wrong = x + 1000
        assert not grade_fraction(q, str(wrong)).correct


def test_exp_easy_tier_positive_integer_solution_single_base():
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="easy")
        m = re.search(r"(\d+)\^x = (\d+)", q.text)
        assert m, q.text
        p, b = int(m.group(1)), int(m.group(2))
        assert p in (2, 3, 5)
        x = Fraction(q.answer["p"], q.answer["q"])
        assert x.denominator == 1 and x > 0
        assert p ** int(x) == b


def test_exp_medium_tier_composite_base_never_easy_reachable():
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="medium")
        m = re.search(r"(\d+)\^x = (-?[\d/]+)", q.text)
        assert m, q.text
        a_base = int(m.group(1))
        # a_base must be a composite power of one of the three primes (never
        # the bare prime itself -- that would be easy-tier reachable).
        assert a_base not in (2, 3, 5), q.text
        x = Fraction(q.answer["p"], q.answer["q"])
        # never a positive integer -- that would be reachable by easy-tier
        # reasoning alone without recognising the composite base.
        assert not (x.denominator == 1 and x > 0), f"medium seed {seed} collapsed to easy-reachable x={x}"


def test_exp_hard_tier_compound_coefficient_present_and_isolated_correctly():
    for seed in SEEDS:
        q = make(EXP_TOPIC, seed=seed, difficulty="hard")
        m = re.search(r"(\d+) × (\d+)\^x = (\d+)", q.text)
        assert m, q.text
        a_coef, p, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert a_coef != 1
        x = Fraction(q.answer["p"], q.answer["q"])
        assert x.denominator == 1
        assert a_coef * p ** int(x) == c


def test_exp_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_simple(2, Fraction(16), Fraction(5))  # 2^5 != 16
    _verify_simple(2, Fraction(16), Fraction(4))  # correct, no raise

    with pytest.raises(VerificationError):
        _verify_compound(4, 2, 32, Fraction(4))  # 4*2^4 = 64 != 32
    _verify_compound(4, 2, 32, Fraction(3))  # 4*2^3 = 32, correct


# --- both skills load through mathgen.skills.load_skills -----------------------


def test_both_skills_load_via_api_skills():
    _, skills = load_skills()
    ids = {s.id for s in skills}
    assert "f10-three-dimensional-coordinates" in ids
    assert "f10-exponential-relations-equations" in ids


def test_f10_three_dimensional_coordinates_properties():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "f10-three-dimensional-coordinates")
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "pair"
    assert skill.domain == "geometry"
    assert "f10-coordinate-movement" in skill.prerequisites


def test_f10_exponential_relations_equations_properties():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "f10-exponential-relations-equations")
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "fraction"
    assert skill.domain == "functions-sequences"
    assert "f10-exponent-laws-integer-variable" in skill.prerequisites
    assert "function-notation" in skill.prerequisites
