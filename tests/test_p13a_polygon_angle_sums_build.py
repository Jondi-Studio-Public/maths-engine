"""Tests `p13a_polygon_angle_sum` (`f10-polygon-angle-sums`): determinism, tier shapes, plausibility and an
independent check.

Tests for f10-polygon-angle-sums."""

import collections
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p13a_polygon_angle_sums_build import (
    _check_easy,
    _check_medium,
    _check_hard,
    _check_exam,
    _known_angle_band,
    _NAME_TO_N,
)

TOPIC = "p13a_polygon_angle_sum"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

_MISSING_LO, _MISSING_HI = 15, 170


class TestDeterminismAndShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2, f"seed {seed} {difficulty}: fewer than 2 distractors survived"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_and_distractors_formatted_alike(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            int(q.answer_text)  # raises if not a plain integer string
            for dist in q.distractors:
                int(dist)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            wrong = str(int(q.answer_text) + 137)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


class TestTierShapeContent:
    def test_easy_states_numeral_polygon_and_individual_angles(self):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="easy")
            m = re.search(r"A polygon has (\d+) sides\.", q.text)
            assert m is not None
            n = int(m.group(1))
            assert 3 <= n <= 6
            assert "of the angles measure" not in q.text  # never the grouped shape
            for name in ("triangle", "quadrilateral", "pentagon", "hexagon", "heptagon", "octagon"):
                assert name not in q.text  # never named

    def test_medium_states_numeral_polygon_and_grouped_restatement(self):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="medium")
            assert re.search(r"A polygon has (\d+) sides\.", q.text) is not None
            assert re.search(r"\d+ of the angles measure \d+ degrees each", q.text) is not None
            assert "and the rest measure" in q.text
            for name in ("triangle", "quadrilateral", "pentagon", "hexagon", "heptagon", "octagon"):
                assert name not in q.text  # never named

    def test_hard_states_named_polygon_and_individual_angles(self):
        seen_names = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="hard")
            m = re.search(r"An? (\w+) has \d+ known interior angles:", q.text)
            assert m is not None
            seen_names.add(m.group(1))
            assert "of the angles measure" not in q.text  # never grouped
            assert re.search(r"has \d+ sides", q.text) is None  # never numeral
        assert seen_names <= {"triangle", "quadrilateral", "pentagon", "hexagon", "heptagon", "octagon"}
        assert len(seen_names) >= 3, f"hard: only saw {seen_names} over 60 seeds"

    def test_exam_states_named_polygon_and_grouped_restatement(self):
        seen_names = set()
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="exam")
            m = re.search(r"An? (\w+) has \d+ angles measuring \d+ degrees each", q.text)
            assert m is not None
            seen_names.add(m.group(1))
            assert "and the rest measuring" in q.text
        assert seen_names <= {"quadrilateral", "pentagon", "hexagon", "heptagon", "octagon"}
        assert "triangle" not in seen_names  # never named for exam (needs a non-empty group + remainder)
        assert len(seen_names) >= 3, f"exam: only saw {seen_names} over 60 seeds"


class TestPlausibility:
    """A figure must be constructible AND sensible: every known angle and the resulting missing angle
    must sit in a sensible convex-polygon range. Proven RED first on an unconstrained draft, then constrained
    by resampling the whole draw -- never clamping -- until the computed missing angle lands in [15, 170]."""

    def test_missing_angle_within_sensible_range_200_seeds_every_tier(self):
        for difficulty in TIERS:
            for seed in range(200):
                q = make(TOPIC, seed=seed, difficulty=difficulty)
                missing = int(q.answer_text)
                assert _MISSING_LO <= missing <= _MISSING_HI, (
                    f"{difficulty} seed {seed}: missing angle {missing} outside [{_MISSING_LO}, {_MISSING_HI}]"
                )

    # Per-tier (n, known-angle-list) extraction, written directly against this
    # test's own regexes (not the generator's private `_EASY_RE` etc.) so a
    # regression in either the generator's parsing or this test's own is not
    # invisible to the other.
    _EASY_RE = re.compile(r"A polygon has (\d+) sides\. \d+ of its interior angles measure (.+?) degrees\.")
    _MEDIUM_RE = re.compile(
        r"A polygon has (\d+) sides\. (\d+) of the angles measure (\d+) degrees each, "
        r"and the rest measure (.+?) degrees\."
    )
    _HARD_RE = re.compile(r"An? (\w+) has \d+ known interior angles: (.+?) degrees, with one angle still unknown\.")
    _EXAM_RE = re.compile(r"An? (\w+) has (\d+) angles measuring (\d+) degrees each and the rest measuring (.+?)\.")
    _NUM_RE = re.compile(r"\d+")

    def _known_angles_and_n(self, difficulty, text):
        pattern = {"easy": self._EASY_RE, "medium": self._MEDIUM_RE, "hard": self._HARD_RE, "exam": self._EXAM_RE}[
            difficulty
        ]
        m = pattern.search(text)
        assert m is not None, f"{difficulty}: could not parse {text!r}"
        if difficulty == "easy":
            n = int(m.group(1))
            knowns = [int(x) for x in self._NUM_RE.findall(m.group(2))]
        elif difficulty == "medium":
            n, g, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rest = [int(x) for x in self._NUM_RE.findall(m.group(4))]
            knowns = [v] * g + rest
        elif difficulty == "hard":
            n = _NAME_TO_N[m.group(1)]
            knowns = [int(x) for x in self._NUM_RE.findall(m.group(2))]
        else:  # exam
            n, g, v = _NAME_TO_N[m.group(1)], int(m.group(2)), int(m.group(3))
            rest = [int(x) for x in self._NUM_RE.findall(m.group(4))]
            knowns = [v] * g + rest
        return n, knowns

    def test_every_known_angle_within_its_own_n_dependent_band_200_seeds(self):
        """The OTHER half: not just the missing angle (already checked
        above), but every stated KNOWN angle too, sits in the same
        n-dependent plausibility band the generator itself draws from
        (`_known_angle_band`) -- this failed on the unconstrained draft in
        exactly the same red-first proof as the missing-angle bound (a
        fixed 80-160 band routinely drew a triangle's two known angles
        summing past 180 before the resample fix)."""
        for difficulty in TIERS:
            for seed in range(200):
                q = make(TOPIC, seed=seed, difficulty=difficulty)
                n, knowns = self._known_angles_and_n(difficulty, q.text)
                lo, hi = _known_angle_band(n)
                for val in knowns:
                    assert lo <= val <= hi, (
                        f"{difficulty} seed {seed}: known angle {val} outside band [{lo}, {hi}] for n={n}"
                    )
                    assert 0 < val < 180, f"{difficulty} seed {seed}: implausible known angle {val}"


class TestIndependentCheck:
    def test_check_easy_fires_on_corrupted_answer(self):
        text = "A polygon has 4 sides. 3 of its interior angles measure 80, 90 and 100 degrees. What is the size of the missing interior angle, in degrees?"
        _check_easy(text, "90")  # 360 - 270 = 90, correct
        with pytest.raises(VerificationError):
            _check_easy(text, "91")

    def test_check_medium_fires_on_corrupted_answer(self):
        text = (
            "A polygon has 5 sides. 3 of the angles measure 100 degrees each, "
            "and the rest measure 90 degrees. What is the size of the missing interior angle, in degrees?"
        )
        # sum_formula = (5-2)*180 = 540; known = 3*100 + 90 = 390; missing = 150
        _check_medium(text, "150")
        with pytest.raises(VerificationError):
            _check_medium(text, "151")

    def test_check_hard_fires_on_corrupted_answer(self):
        text = "A hexagon has 5 known interior angles: 100, 110, 120, 130 and 100 degrees, with one angle still unknown. What is the size of the missing interior angle, in degrees?"
        # sum = 720 - 560 = 160
        _check_hard(text, "160")
        with pytest.raises(VerificationError):
            _check_hard(text, "161")

    def test_check_exam_fires_on_corrupted_answer(self):
        text = (
            "A pentagon has 2 angles measuring 100 degrees each and the rest measuring "
            "110 and 90 degrees. What is the size of the missing interior angle, in degrees?"
        )
        # sum_formula = (5-2)*180 = 540; known = 2*100 + 110 + 90 = 400; missing = 140.
        _check_exam(text, "140")
        with pytest.raises(VerificationError):
            _check_exam(text, "141")


class TestLoadSkills:
    def test_skill_loads_with_expected_shape(self):
        _domains, skills = load_skills()
        by_id = {s.id: s for s in skills}
        skill = by_id["f10-polygon-angle-sums"]
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["f10-parallel-line-angle-theorems"]
        tiers_covered = {src.difficulty for src in skill.sources}
        assert tiers_covered == {"easy", "medium", "hard", "exam"}
        exam_sources = [src for src in skill.sources if src.difficulty == "exam"]
        assert exam_sources and exam_sources[0].stages == ["exam"]


class TestAnswerConcentration:
    """The 200-seed answer-concentration measure: no single answer value above 25% of draws on any tier (the
    answer space here is numeric -- dozens of reachable missing-angle values -- not a small closed class set)."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_single_answer_dominates_200_seeds(self, difficulty):
        counts = collections.Counter(make(TOPIC, seed=seed, difficulty=difficulty).answer_text for seed in range(200))
        total = sum(counts.values())
        top_answer, top_n = counts.most_common(1)[0]
        share = top_n / total
        assert share < 0.25, (
            f"{difficulty}: answer {top_answer!r} is {share:.0%} of 200 draws ({len(counts)} distinct answers)"
        )
        assert len(counts) >= 10, f"{difficulty}: only {len(counts)} distinct answers over 200 seeds"


class TestDistractorPlausibility:
    """A distractor that is a negative number or a 300+ 'degrees' value is eliminable by SHAPE alone, without
    doing any maths -- the same "no option identifiable. Measured and disclosed here rather than assumed:
    every tier prioritises in-range distractors first (`_order_in_range_first`), but a misconception-derived
    value can still be the only survivor once in-range candidates run out."""

    _ANGLE_LO, _ANGLE_HI = 1, 179

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_measure_in_range_distractor_share_200_seeds(self, difficulty):
        total_distractors = 0
        out_of_range_distractors = 0
        seeds_all_out_of_range = 0
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            in_range_count = 0
            for d in q.distractors:
                total_distractors += 1
                val = int(d)
                if self._ANGLE_LO <= val <= self._ANGLE_HI:
                    in_range_count += 1
                else:
                    out_of_range_distractors += 1
            if in_range_count == 0:
                seeds_all_out_of_range += 1
        share = out_of_range_distractors / total_distractors
        print(
            f"{difficulty}: {out_of_range_distractors}/{total_distractors} "
            f"({share:.1%}) distractors outside [{self._ANGLE_LO}, {self._ANGLE_HI}]; "
            f"{seeds_all_out_of_range}/200 seeds have ALL distractors out of range"
        )
        # Disclosed, not silently accepted: every seed must have at least one
        # in-range (non-eliminable-by-shape) distractor, per _order_in_range_first's
        # own guarantee (>= 2 bounded offsets are always constructible for any
        # missing angle in this skill's own [15, 170] range).
        assert seeds_all_out_of_range == 0, (
            f"{difficulty}: {seeds_all_out_of_range}/200 seeds have every distractor "
            f"eliminable by shape alone (outside [{self._ANGLE_LO}, {self._ANGLE_HI}])"
        )
