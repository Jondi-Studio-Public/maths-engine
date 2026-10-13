"""Tests for T5 circle-reverse-from-area + cone-volume skills."""

import re
from fractions import Fraction

import pytest
from sympy import sqrt as sympy_sqrt

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent, round_half_up
from mathgen.generators.t5_circle_cone_build import (
    PI,
    _MAX_CONE_VOLUME,
    _aspect_ok,
    _bisection_sqrt,
    _boundary_margin,
    _newton_sqrt,
    _pi_precision_safe,
    _route_a_circumference,
    _route_a_diameter,
    _route_b_circumference,
    _route_b_diameter,
    _verify_sqrt,
    _volume,
    _volume_check,
)

CIRCLE_TOPIC = "circle_reverse_from_area"
CONE_TOPIC = "cone_volume"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)

_AREA_RE = re.compile(r"area of ([\d.]+) cm²")
_CONE_EASY_RE = re.compile(r"radius of (\d+) cm and a height of (\d+) cm")
_CONE_MEDIUM_RE = re.compile(r"diameter of (\d+) cm and a height of (\d+) cm")
_CONE_HARD_RE = re.compile(r"radius of (\d+) cm and a slant height of (\d+) cm")


def _area_from_text(text: str) -> Fraction:
    m = _AREA_RE.search(text)
    assert m, f"could not find a stated area in: {text!r}"
    return Fraction(m.group(1))


# ---------------------------------------------------------------------------
# Shared checks, both topics
# ---------------------------------------------------------------------------


class _CommonChecks:
    TOPIC = None

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_within_tolerance(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1
            answer = Fraction(q.answer_text)
            for d in q.distractors:
                assert abs(Fraction(d) - answer) > Fraction(1, 20), (
                    f"seed {seed}: distractor {d} within 0.05 of answer {q.answer_text}"
                )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            wrong = f"{float(Fraction(q.answer_text) + Fraction(137, 10)):.1f}"
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_and_distractors_formatted_alike(self, difficulty):
        pattern = re.compile(r"^-?\d+\.\d$")
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert pattern.match(q.answer_text), f"seed {seed}: answer {q.answer_text!r} not 1dp"
            for d in q.distractors:
                assert pattern.match(d), f"seed {seed}: distractor {d!r} not 1dp"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(self.TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step


# ---------------------------------------------------------------------------
# circle_reverse_from_area
# ---------------------------------------------------------------------------


class TestCircleReverseFromArea(_CommonChecks):
    TOPIC = CIRCLE_TOPIC

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_not_sitting_on_rounding_boundary(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            area = _area_from_text(q.text)
            if difficulty == "easy":
                exact = _newton_sqrt(area / PI)
            elif difficulty == "medium":
                exact = 2 * _newton_sqrt(area / PI)
            else:
                exact = 2 * _newton_sqrt(PI * area)
            margin = _boundary_margin(exact, 1)
            assert margin > Fraction(1, 1000), (
                f"seed {seed} [{difficulty}]: exact value {exact} sits on the 1dp rounding boundary (margin {margin})"
            )

    def test_circle_reverse_medium_diameter_routes_agree(self):
        """The single most important test in this file: for the diameter
        tier, route A (round the radius to 1dp first, then double) and
        route B (2*sqrt(area/pi) directly, rounded once) must produce the
        IDENTICAL 1dp string on every seed, swept over 60 seeds."""
        checked = 0
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            area = _area_from_text(q.text)
            route_a = _route_a_diameter(area)
            route_b = _route_b_diameter(area)
            assert route_a == route_b, f"seed {seed}: routes disagree, A={route_a} B={route_b}"
            assert f"{float(route_a):.1f}" == f"{float(route_b):.1f}" == q.answer_text
            checked += 1
        assert checked == 60

    def test_circle_reverse_hard_circumference_routes_agree(self):
        """Same two-route agreement proof for the circumference tier: route
        A rounds the radius to 1dp first then applies C=2*pi*r, rounded
        again; route B computes 2*sqrt(pi*area) directly, rounded once.
        Swept over 60 seeds."""
        checked = 0
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            area = _area_from_text(q.text)
            route_a = _route_a_circumference(area)
            route_b = _route_b_circumference(area)
            assert route_a == route_b, f"seed {seed}: routes disagree, A={route_a} B={route_b}"
            assert f"{float(route_a):.1f}" == f"{float(route_b):.1f}" == q.answer_text
            checked += 1
        assert checked == 60

    def test_area_stated_to_two_decimal_places(self):
        for seed in SEEDS:
            for tier in TIERS:
                q = make(self.TOPIC, seed=seed, difficulty=tier)
                m = _AREA_RE.search(q.text)
                assert m and re.match(r"^\d+\.\d\d$", m.group(1))

    def test_easy_asks_for_radius_medium_diameter_hard_circumference(self):
        for seed in range(10):
            assert "find the radius" in make(self.TOPIC, seed=seed, difficulty="easy").text
            assert "find the diameter" in make(self.TOPIC, seed=seed, difficulty="medium").text
            assert "find the circumference" in make(self.TOPIC, seed=seed, difficulty="hard").text


# ---------------------------------------------------------------------------
# cone_volume
# ---------------------------------------------------------------------------


class TestConeVolume(_CommonChecks):
    TOPIC = CONE_TOPIC

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_not_sitting_on_rounding_boundary(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "easy":
                r, h = map(int, _CONE_EASY_RE.search(q.text).groups())
            elif difficulty == "medium":
                d, h = map(int, _CONE_MEDIUM_RE.search(q.text).groups())
                r = Fraction(d, 2)
            else:
                r, l = map(int, _CONE_HARD_RE.search(q.text).groups())
                h_sq = l * l - r * r
                h = int(round(h_sq**0.5))
                assert h * h == h_sq
            exact = _volume(r, h)
            margin = _boundary_margin(exact, 1)
            assert margin > Fraction(1, 1000), (
                f"seed {seed} [{difficulty}]: volume {exact} sits on the 1dp rounding boundary"
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_volume_is_pi_precision_safe_and_capped(self, difficulty):
        """The finding this test guards: an unconstrained hard-tier draw
        could reach volumes (~563000 cm^3) where the stated pi (3.14159)
        and a calculator's true pi round the 1dp answer DIFFERENTLY -- by
        0.5, ten times the grading tolerance. `_MAX_CONE_VOLUME` keeps every
        shipped volume at a magnitude comparable to easy/medium, and
        `_pi_precision_safe` is the direct proof the disagreement cannot
        happen for the (r, h) actually shipped; swept, not spot-checked."""
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "easy":
                r, h = map(int, _CONE_EASY_RE.search(q.text).groups())
            elif difficulty == "medium":
                d, h = map(int, _CONE_MEDIUM_RE.search(q.text).groups())
                r = Fraction(d, 2)
            else:
                r, l = map(int, _CONE_HARD_RE.search(q.text).groups())
                h_sq = l * l - r * r
                h = int(round(h_sq**0.5))
                assert h * h == h_sq
            assert _volume(r, h) <= _MAX_CONE_VOLUME, f"seed {seed} [{difficulty}]: volume exceeds the safe cap"
            assert _pi_precision_safe(r, h), f"seed {seed} [{difficulty}]: r={r} h={h} is pi-precision unsafe"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_aspect_ratio_plausible_band(self, difficulty):
        """Lesson (possible is not the same as plausible): every drawn cone's height:radius ratio stays within
        [1/4, 4] so no seed ships a near-flat disc or a near-spike. Swept, not spot-checked."""
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "easy":
                r, h = map(int, _CONE_EASY_RE.search(q.text).groups())
            elif difficulty == "medium":
                d, h = map(int, _CONE_MEDIUM_RE.search(q.text).groups())
                r = Fraction(d, 2)
            else:
                r, l = map(int, _CONE_HARD_RE.search(q.text).groups())
                h_sq = l * l - r * r
                h = int(round(h_sq**0.5))
                assert h * h == h_sq
            assert _aspect_ok(r, h), f"seed {seed} [{difficulty}]: r={r} h={h} outside the aspect band"

    def test_hard_height_is_exact_pythagorean_integer(self):
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            r, l = map(int, _CONE_HARD_RE.search(q.text).groups())
            h_sq = l * l - r * r
            assert h_sq > 0
            h = round(h_sq**0.5)
            assert h * h == h_sq, f"seed {seed}: l^2 - r^2 = {h_sq} is not a perfect square"
            assert l > r and l > h, f"seed {seed}: slant height {l} is not the hypotenuse"

    def test_easy_always_has_exactly_two_distractors(self):
        # Honest omission: easy gives the radius
        # directly, so uses-diameter-as-radius has nothing to bite on --
        # only omits-one-third-factor and squares-wrong-dimension are ever
        # possible. The generator's own h == 3*r guard rules out the one
        # deterministic collision between those two misconceptions (both
        # equal pi*r^2*h when h == 3*r), so every seed must clear both.
        counts = [len(make(self.TOPIC, seed=seed, difficulty="easy").distractors) for seed in range(60)]
        assert all(c == 2 for c in counts), counts

    def test_medium_always_has_exactly_three_distractors(self):
        # Medium has three possible misconceptions (omits-one-third-factor,
        # squares-wrong-dimension, uses-diameter-as-radius). The generator's
        # own h == 3*r and h == 4*r guards rule out the two deterministic
        # pairwise collisions, so every seed must clear all three.
        counts = [len(make(self.TOPIC, seed=seed, difficulty="medium").distractors) for seed in range(60)]
        assert all(c == 3 for c in counts), counts


# ---------------------------------------------------------------------------
# Independent-check helpers, exercised directly with corrupted values
# ---------------------------------------------------------------------------


def test_newton_and_bisection_sqrt_agree():
    for n in [Fraction(4), Fraction(2), Fraction(100, 7), Fraction(31417, 10000)]:
        newton = _newton_sqrt(n)
        bisected = _bisection_sqrt(n)
        assert abs(newton - bisected) < Fraction(1, 10**6)
        assert abs(newton * newton - n) < Fraction(1, 10**10)


def test_verify_sqrt_fires_on_corrupted_claim():
    _verify_sqrt(Fraction(4), Fraction(2))  # must not raise
    with pytest.raises(VerificationError):
        _verify_sqrt(Fraction(4), Fraction(2) + Fraction(1, 100))


def test_volume_cross_check_fires_on_corrupted_answer():
    r, h = 6, 9
    exact = _volume(r, h)
    check_equivalent(exact, _volume_check(r, h), "sanity: cross-check should agree")
    with pytest.raises(VerificationError):
        check_equivalent(exact, _volume_check(r, h) + Fraction(1, 1000), "corrupted volume")


def test_check_distinct_fires_when_distractor_equals_answer():
    check_distinct(Fraction(5), Fraction(6))  # must not raise
    with pytest.raises(VerificationError):
        check_distinct(Fraction(5), Fraction(5))


def test_sympy_pythagoras_check_fires_on_corrupted_height():
    # Exercises the one independent check in _cone_hard (sympy re-derivation
    # of h from l and r) directly with a corrupted claimed height, the same
    # way the other independent checks above are exercised.
    r, l, h = 12, 20, 16
    check_equivalent(sympy_sqrt(l**2 - r**2), h, "sanity: sympy Pythagoras should agree")
    with pytest.raises(VerificationError):
        check_equivalent(sympy_sqrt(l**2 - r**2), h + 1, "corrupted height")


def test_round_half_up_boundary_cases():
    assert round_half_up(Fraction(125, 100), 1) == Fraction(13, 10)  # 1.25 -> 1.3
    assert round_half_up(Fraction(124, 100), 1) == Fraction(12, 10)  # 1.24 -> 1.2


# ---------------------------------------------------------------------------
# Skill YAML loading
# ---------------------------------------------------------------------------


def test_skills_load_with_expected_shape():
    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}

    circle = by_id["circle-reverse-from-area"]
    assert circle.prerequisites == ["f10-circle-measures"]
    assert circle.answer == "number"
    assert circle.placement_eligible is False
    difficulties = {src.difficulty for src in circle.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    stages_by_difficulty = {src.difficulty: src.stages for src in circle.sources}
    assert stages_by_difficulty["hard"] and "maintenance" in stages_by_difficulty["hard"]

    cone = by_id["cone-volume"]
    assert cone.prerequisites == ["senior-use-formulas-to-find-the-volume-of-pyramids"]
    assert cone.answer == "number"
    assert cone.placement_eligible is False
    difficulties = {src.difficulty for src in cone.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    stages_by_difficulty = {src.difficulty: src.stages for src in cone.sources}
    assert stages_by_difficulty["hard"] and "maintenance" in stages_by_difficulty["hard"]
