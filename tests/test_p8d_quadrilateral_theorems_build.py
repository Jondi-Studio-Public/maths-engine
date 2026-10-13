"""Tests the cyclic-quadrilateral and rhombus-diagonal theorem generators and that their skills load.

Quadrilateral theorems build."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

CYCLIC_TOPIC = "the_opposite_angles_of_a_cyclic_quadrilateral_are"
RHOMBUS_TOPIC = "the_diagonals_of_a_parallelogram_meet_at_right"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]

_BAD_SIGN_PATTERNS = [re.compile(r"\+ -"), re.compile(r"- -")]


def _sweep(topic, seeds=range(60)):
    for tier in TIERS:
        for seed in seeds:
            yield tier, seed, make(topic, seed=seed, difficulty=tier)


class _CommonChecks:
    """Mixin-style shared assertions parametrised by TOPIC on the subclass."""

    TOPIC = None

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

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
            wrong = str(int(q.answer_text) + 137)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_integer(self, difficulty):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            int(q.answer_text)  # raises if not a plain integer string
            for dist in q.distractors:
                int(dist)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(self.TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_bad_sign_patterns(self, difficulty):
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty=difficulty)
            for field in [q.text, q.latex, *q.working]:
                for pat in _BAD_SIGN_PATTERNS:
                    assert not pat.search(field), f"seed {seed}: bad sign pattern in {field!r}"


_LINEAR_EXPR_RE = re.compile(r"^(-?\d*)x(?:\s*([+-])\s*(\d+))?$")


def _parse_linear(text):
    """Parse a rendered linear-in-x expression ('x - 1', '4x + 2', '5x')
    into (coefficient, constant) ints. Standalone, parses only the rendered
    text -- does not import anything from the generator module."""
    m = _LINEAR_EXPR_RE.match(text.strip())
    assert m, f"could not parse linear expression: {text!r}"
    coef_str, sign, mag = m.group(1), m.group(2), m.group(3)
    coef = 1 if coef_str in ("", "+") else (-1 if coef_str == "-" else int(coef_str))
    const = 0 if sign is None else (int(mag) if sign == "+" else -int(mag))
    return coef, const


def _medium_all_angles(text):
    """Both opposite-pair angles (A, C) for the cyclic-quadrilateral medium
    tier, derived from the rendered text alone."""
    m = re.search(r"Angle A = \(([^)]+)\) degrees and angle C = \(([^)]+)\) degrees", text)
    a1, a2 = _parse_linear(m.group(1))
    c1, c2 = _parse_linear(m.group(2))
    remaining = 180 - (a2 + c2)
    assert remaining % (a1 + c1) == 0
    x = remaining // (a1 + c1)
    return [a1 * x + a2, c1 * x + c2]


def _hard_all_angles(text):
    """All four angles (A, B, C, D) for the cyclic-quadrilateral hard tier,
    derived from the rendered text alone."""
    m = re.search(r"Angle A = \(([^)]+)\) degrees and angle B = \(([^)]+)\) degrees", text)
    a1, a2 = _parse_linear(m.group(1))
    b1, b2 = _parse_linear(m.group(2))
    k = int(re.search(r"Angle C exceeds angle D by (\d+) degrees?", text).group(1))
    # C - D = k  =>  (180-A) - (180-B) = k  =>  B - A = k
    numer, denom = k - b2 + a2, b1 - a1
    assert numer % denom == 0
    x = numer // denom
    a_val, b_val = a1 * x + a2, b1 * x + b2
    return [a_val, b_val, 180 - a_val, 180 - b_val]


class TestCyclicQuadrilateral(_CommonChecks):
    TOPIC = CYCLIC_TOPIC

    def test_easy_angles_plausible_range(self):
        """No angle so small the figure would look like a sliver to a
        learner. Easy draws `a` directly from a fixed, designed 21-159
        range rather
        than deriving it algebraically from independently-drawn
        coefficients -- there is no free-floating combination that can
        silently land small, so this tier's own designed floor (20) is the
        right bound here, not the tighter 30 the medium/hard fix below
        needed. Swept, not spot-checked."""
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="easy")
            a = int(re.search(r"Angle A is (\d+) degrees", q.text).group(1))
            assert 20 <= a <= 160, f"seed {seed}: angle A={a} outside [20, 160]"
            assert 20 <= 180 - a <= 160, f"seed {seed}: angle C={180 - a} outside [20, 160]"

    def test_medium_angles_plausible_range(self):
        """Same plausibility sweep for the medium tier's two given angles."""
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            for label, val in zip("AC", _medium_all_angles(q.text)):
                assert 30 <= val <= 150, f"seed {seed}: angle {label}={val} outside [30, 150]"

    def test_hard_angles_plausible_range(self):
        """Every one of the four hard-tier angles (A, B, C, D) stays within a plausible range across >=60 seeds
        -- not a spot check on one seed."""
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            for label, val in zip("ABCD", _hard_all_angles(q.text)):
                assert 30 <= val <= 150, f"seed {seed}: angle {label}={val} outside [30, 150]"

    def test_easy_excludes_ninety(self):
        for seed in range(200):
            q = make(self.TOPIC, seed=seed, difficulty="easy")
            m = re.search(r"Angle A is (\d+) degrees", q.text)
            assert m is not None
            assert m.group(1) != "90"

    def test_medium_asks_named_angle(self):
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            assert re.search(r"Find angle [AC], in degrees\.$", q.text)

    def test_hard_asks_one_of_four_named_angles(self):
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            assert re.search(r"Find angle [ABCD], in degrees\.$", q.text)


class TestRhombusDiagonals(_CommonChecks):
    TOPIC = RHOMBUS_TOPIC

    def test_easy_states_bisect_as_given_fact(self):
        for seed in range(10):
            q = make(self.TOPIC, seed=seed, difficulty="easy")
            assert "bisect each other at right angles" in q.text
            assert "property of every rhombus" in q.text

    def test_medium_states_bisect_as_given_fact(self):
        for seed in range(10):
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            assert "bisect each other at right angles" in q.text

    def test_hard_never_asserts_bisection_as_this_theorem(self):
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            assert "bisect" not in q.text
            assert "diagonals" in q.text and "right angles" in q.text

    def test_hard_no_sum_of_squares_of_diagonals_collision(self):
        """The hard-tier converse question never states a diagonal length at all
        (only side-length expressions), so there is no sum-of-squares-of-diagonals
        (ACMSM041) value in play for a learner to collide with."""
        length_pattern = re.compile(r"\d+\s*cm")
        for seed in range(60):
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            assert "diagonals are given to meet at right angles" in q.text
            # The only "cm" lengths in the text are the two side expressions
            # (AB, BC), never a bare numeric diagonal length.
            assert "bisect" not in q.text
            assert not length_pattern.search(q.text)


class TestSkillsLoad:
    def test_cyclic_quadrilateral_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-the-opposite-angles-of-a-cyclic-quadrilateral-are"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-the-angle-at-the-centre-subtended-by-an"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4  # three practice tiers plus the exam source

    def test_rhombus_diagonals_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-the-diagonals-of-a-parallelogram-meet-at-right"), None)
        assert skill is not None
        assert skill.domain == "senior-mathematics"
        assert skill.answer == "number"
        assert skill.prerequisites == ["f10-quadrilateral-properties", "f10-congruence-similarity-criteria"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4  # three practice tiers plus the exam source
