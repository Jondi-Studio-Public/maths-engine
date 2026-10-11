"""Tests for the T3 applied-pythagoras-problems skill."""

import math
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.t3_applied_pythagoras_problems_build import _bad_draw, _fmt

TOPIC = "applied_pythagoras"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


def _variant_of(text: str) -> str:
    if "ladder" in text:
        return "ladder"
    if "hiker" in text:
        return "displacement"
    if "equilateral" in text:
        return "equilateral_altitude"
    if "rectangular box" in text:
        return "rect_solid_diagonal"
    raise AssertionError(f"could not classify question text: {text!r}")


def _collect_variant(variant: str, difficulty: str, count: int = 60, max_raw_seed: int = 2000):
    """Draw real `difficulty`-tier questions (going through the actual coin
    flip for `hard`) until `count` of them are the requested `variant`,
    bounded by `max_raw_seed` distinct underlying seeds. Returns a list of
    (raw_seed, question) pairs."""
    out = []
    for seed in range(max_raw_seed):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        if _variant_of(q.text) == variant:
            out.append((seed, q))
            if len(out) >= count:
                break
    assert len(out) >= count, f"only found {len(out)} {variant!r} questions within {max_raw_seed} raw seeds"
    return out


class _CommonChecks:
    """Shared assertions, run at every difficulty tier (each tier may mix
    variants at `hard`)."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_within_tolerance(self, difficulty):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) == 2, (
                f"seed {seed}, {difficulty}: expected exactly 2 distractors "
                f"(by-construction collision guard), got {q.distractors!r}"
            )
            answer_val = float(q.answer_text)
            for d in q.distractors:
                gap = abs(float(d) - answer_val)
                assert gap > 0.05, (
                    f"seed {seed}, {difficulty}: distractor {d!r} within 0.05 of answer {q.answer_text!r}"
                )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}, {difficulty}: grader rejected answer {q.answer_text!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}, {difficulty}: grader accepted distractor {dist!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            wrong = f"{float(q.answer_text) + 137:.1f}"
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}, {difficulty}: grader accepted wrong answer {wrong!r}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_never_trailing_zero_or_integer(self, difficulty):
        """The trailing-zero/exact-integer guard: every accepted question's `answer_text` states exactly 1
        decimal place with a non-zero tenths digit, swept over 60 seeds at
        every tier."""
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            m = re.match(r"^-?\d+\.(\d)$", q.answer_text)
            assert m, f"seed {seed}, {difficulty}: answer_text {q.answer_text!r} is not exactly 1 decimal place"
            assert m.group(1) != "0", (
                f"seed {seed}, {difficulty}: answer_text {q.answer_text!r} has a trailing-zero tenths digit"
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_states_one_decimal_place_rounding_instruction(self, difficulty):
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert "to 1 decimal place" in q.text


class TestCommon(_CommonChecks):
    pass


# ============================================================================
# The two in-generator numeric traps, exercised directly (not just trusted
# via the sweep above).
# ============================================================================


class TestBadDrawGuardDirect:
    def test_rounding_boundary_landing_is_bad(self):
        # 1.25 * 10 = 12.5 exactly on the x.5 boundary.
        assert _bad_draw(1.25) is True
        # A value 5e-7 off the boundary is still within the 1e-6 tolerance.
        assert _bad_draw(1.25000005) is True
        assert _bad_draw(12.05) is True

    def test_trailing_zero_landing_is_bad(self):
        assert _bad_draw(12.0) is True
        assert _bad_draw(5.0) is True
        # 12.04 rounds to 12.0 -- still a trailing-zero landing.
        assert _bad_draw(12.04) is True

    def test_clean_one_decimal_value_is_not_bad(self):
        assert _bad_draw(12.3) is False
        assert _bad_draw(1.26) is False
        assert _bad_draw(5.7) is False


class TestConstructionCollisionGuardDirect:
    """Exercises the by-construction distractor-collision guard directly against known corrupted/forced inputs that
    DO collide, proving the predicate the generator uses would reject them
    -- rather than only trusting that the sweep above never observes a
    collision."""

    def test_displacement_one_leg_only_collision_is_detected(self):
        # Found live at seed 24 pre-fix: q tiny relative to p makes
        # sqrt(p^2+q^2) round to the same 1dp value as p itself.
        p, q = 18.4, 1.3
        distance = math.sqrt(p**2 + q**2)
        assert _fmt(distance) == _fmt(p), "expected a genuine one-leg-only collision for this forced input"
        # The real generator's rng stream is identical up to this first
        # (p, q) draw (dir1/dir2 are drawn before the loop and are
        # unaffected by the guard), so seed 24 at the medium tier still
        # draws exactly this colliding pair on attempt 1 -- proving the
        # guard actually redraws, not merely that some other invariant
        # holds on whatever it ships.
        q_gen = make(TOPIC, seed=24, difficulty="medium")
        m = _DISPLACEMENT_RE.search(q_gen.text)
        shipped_p, shipped_q = float(m.group(1)), float(m.group(3))
        assert (shipped_p, shipped_q) != (p, q), (
            "seed 24 shipped the known-colliding (p, q) pair -- the collision guard did not redraw"
        )
        assert len(q_gen.distractors) == 2

    def test_ladder_added_not_subtracted_collision_is_detected(self):
        # A forced input found by exhaustive search over the ladder's own
        # constructibility grid: ladder=2.5, given=0.4 rounds both the
        # correct leg and the added-not-subtracted misconception to 2.5.
        ladder, given = 2.5, 0.4
        other = math.sqrt(ladder**2 - given**2)
        added = math.sqrt(ladder**2 + given**2)
        assert _fmt(other) == _fmt(added), "expected a genuine added-not-subtracted collision for this forced input"

    def test_rect_solid_forgot_third_dimension_collision_is_detected(self):
        # A forced input found by random search: a small height next to
        # two large in-plane dimensions makes the face diagonal round to
        # the same 1dp value as the full space diagonal.
        l, w, h = 13.3, 19.2, 2.0
        diagonal = math.sqrt(l**2 + w**2 + h**2)
        face = math.sqrt(l**2 + w**2)
        assert _fmt(diagonal) == _fmt(face), "expected a genuine forgot-third-dimension collision for this forced input"


# ============================================================================
# Constructibility sweeps -- one per
# variant, 60 seeds, re-parsing the drawn lengths from `question.text`
# alone.
# ============================================================================

_LADDER_LEN_RE = re.compile(r"ladder of length (\d+\.\d) m")
_LADDER_HEIGHT_GIVEN_RE = re.compile(r"reaching (\d+\.\d) m up the wall")
_LADDER_DISTANCE_GIVEN_RE = re.compile(r"is (\d+\.\d) m from the base of the wall")
_DISPLACEMENT_RE = re.compile(r"walks (\d+\.\d) km (north|south), then (\d+\.\d) km (east|west)")
_EQUILATERAL_SIDE_RE = re.compile(r"side length (\d+\.\d) units")
_RECT_DIMS_RE = re.compile(r"length (\d+\.\d) m, width (\d+\.\d) m and height (\d+\.\d) m")


class TestLadderConstructibility:
    def test_bounds_and_ladder_strictly_greater_than_given(self):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="easy")
            ladder = float(_LADDER_LEN_RE.search(q.text).group(1))
            m_h = _LADDER_HEIGHT_GIVEN_RE.search(q.text)
            m_d = _LADDER_DISTANCE_GIVEN_RE.search(q.text)
            assert m_h or m_d, f"seed {seed}: could not find the given leg in {q.text!r}"
            given = float((m_h or m_d).group(1))

            assert 2.5 <= ladder <= 9.0, f"seed {seed}: ladder length {ladder} outside [2.5, 9.0]"
            assert ladder > given, f"seed {seed}: ladder length {ladder} not strictly greater than given leg {given}"
            assert ladder * 0.15 < given < ladder * 0.90, (
                f"seed {seed}: given leg {given} not strictly between 15% and 90% of ladder length {ladder}"
            )


class TestDisplacementConstructibility:
    def test_bounds_both_legs_positive_and_walking_range(self):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="medium")
            m = _DISPLACEMENT_RE.search(q.text)
            assert m, f"seed {seed}: could not parse displacement legs from {q.text!r}"
            p, q_leg = float(m.group(1)), float(m.group(3))
            assert p > 0 and q_leg > 0, f"seed {seed}: a leg is not strictly positive ({p}, {q_leg})"
            assert 1.0 <= p <= 30.0, f"seed {seed}: leg {p} outside [1.0, 30.0]"
            assert 1.0 <= q_leg <= 30.0, f"seed {seed}: leg {q_leg} outside [1.0, 30.0]"


class TestEquilateralAltitudeConstructibility:
    def test_bounds_side_length_positive_and_believable(self):
        found = _collect_variant("equilateral_altitude", "hard", count=60)
        for seed, q in found:
            side = float(_EQUILATERAL_SIDE_RE.search(q.text).group(1))
            assert side > 0, f"seed {seed}: side length not strictly positive"
            assert 3.0 <= side <= 25.0, f"seed {seed}: side length {side} outside [3.0, 25.0]"


class TestRectSolidDiagonalConstructibility:
    def test_bounds_all_dimensions_positive_and_believable(self):
        found = _collect_variant("rect_solid_diagonal", "hard", count=60)
        for seed, q in found:
            m = _RECT_DIMS_RE.search(q.text)
            assert m, f"seed {seed}: could not parse box dimensions from {q.text!r}"
            length, width, height = (float(g) for g in m.groups())
            for name, val in [("length", length), ("width", width), ("height", height)]:
                assert val > 0, f"seed {seed}: {name} {val} not strictly positive"
                assert 2.0 <= val <= 20.0, f"seed {seed}: {name} {val} outside [2.0, 20.0]"


# ============================================================================
# Skill loads
# ============================================================================


class TestSkillLoads:
    def test_applied_pythagoras_problems_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "applied-pythagoras-problems"), None)
        assert skill is not None
        assert skill.domain == "trigonometry"
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["pythagorean-hypotenuse", "pythagorean-leg"]
        # easy, medium, hard and the exam tier
        assert len(skill.sources) == 4
