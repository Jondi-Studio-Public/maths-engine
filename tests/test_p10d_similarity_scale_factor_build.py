"""Focused contract tests for: senior similarity conditions
(senior-review-the-conditions-for-similarity-of-two-dimensional, ACMGM021) and linear scale-factor
length/perimeter (senior-use-the-scale-factor-for-two-similar-figures, ACMGM022)."""

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, grade_number, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p10d_similarity_scale_factor_build import (
    ALL_OPTIONS,
    OPTION_AAA,
    OPTION_NOT_SIMILAR,
    OPTION_SAS,
    OPTION_SSS,
    _check_aaa,
    _check_sas,
    _check_sss,
)
from mathgen.core import VerificationError

CLASSIFY_TOPIC = "similarity_conditions_classify"
LENGTH_TOPIC = "similar_figure_scale_factor_length"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


def _angles_from_sides(sides):
    """Independent (test-only, not generator-reused) Law-of-Cosines angle
    computation from three side lengths, in degrees."""
    a, b, c = sides

    def ang(x, y, z):
        cosv = (x * x + y * y - z * z) / (2 * x * y)
        cosv = max(-1.0, min(1.0, cosv))
        return math.degrees(math.acos(cosv))

    return ang(b, c, a), ang(a, c, b), ang(a, b, c)


def _law_of_cosines_third_side(a, b, angle_deg):
    c2 = a * a + b * b - 2 * a * b * math.cos(math.radians(angle_deg))
    return math.sqrt(max(c2, 0.0))


# --- determinism / distractor hygiene, both topics --------------------------


@pytest.mark.parametrize("topic", [CLASSIFY_TOPIC, LENGTH_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_and_distractors_hygienic(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


# --- Skill 1: similarity_conditions_classify --------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_answer_is_one_of_four_fixed_options(difficulty):
    for seed in SEEDS:
        q = make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text in ALL_OPTIONS
        options = mcq_options(q)
        assert q.answer_text in options
        assert set(options) == set(ALL_OPTIONS)


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_only_tier_criterion_or_not_similar_is_ever_correct(difficulty):
    tier_criterion = {"easy": OPTION_AAA, "medium": OPTION_SSS, "hard": OPTION_SAS}[difficulty]
    for seed in SEEDS:
        q = make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text in (tier_criterion, OPTION_NOT_SIMILAR)


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_real_grader_accepts_answer_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for i, opt in enumerate(options):
            if opt != q.answer_text:
                assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_independent_check_fires_on_corrupted_answer(difficulty):
    """Each independent check must reject a deliberately wrong expected label."""
    if difficulty == "easy":
        with pytest.raises(VerificationError):
            _check_aaa(50, 60, 70, 50, 60, 70, expected_similar=False)  # actually similar
        with pytest.raises(VerificationError):
            _check_aaa(50, 60, 70, 50, 61, 69, expected_similar=True)  # actually not similar
    elif difficulty == "medium":
        with pytest.raises(VerificationError):
            _check_sss((6, 9, 12), (12, 18, 24), expected_similar=False)  # actually similar
        with pytest.raises(VerificationError):
            _check_sss((6, 9, 12), (12, 18, 23), expected_similar=True)  # actually not similar
    else:
        with pytest.raises(VerificationError):
            _check_sas((6, 9, 50), (12, 18, 50), expected_similar=False)  # actually similar
        with pytest.raises(VerificationError):
            _check_sas((6, 9, 50), (12, 18, 80), expected_similar=True)  # actually not similar


def test_classify_aaa_ground_truth_is_sorted_not_positional():
    """A DEF triangle whose angle MULTISET matches ABC's is similar even if the
    positionally-stated correspondence in the prose does not literally pair
    the same values -- ground truth is the sorted-angle-set comparison, not the
    letters. Regression guard for a false reject on a swapped correspondence."""
    # ABC = {50, 60, 70}. DEF stated as D=50, E=70 (swapped from the "natural"
    # A<->D, B<->E correspondence) -- still similar, because {50, 70, 60} ==
    # {50, 60, 70} as multisets.
    _check_aaa(50, 60, 70, 50, 70, 60, expected_similar=True)


def test_classify_sss_ground_truth_is_sorted_not_positional():
    # ABC sides (6, 8, 10); DEF sides stated in a different order (16, 12, 20)
    # -- still similar (k=2) once sorted: (6,8,10) vs (12,16,20).
    _check_sss((6, 8, 10), (16, 12, 20), expected_similar=True)


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_plausibility_no_sliver(difficulty):
    """Every triangle shown to the learner must be plausible: no angle under
    20 degrees, longest/shortest side ratio <= 3, no side under 4 -- asserted
    directly from the rendered numbers."""
    for seed in range(2000):
        q = make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
        text = q.text
        if difficulty == "easy":
            angles = [int(x) for x in re.findall(r"= (\d+) degrees", text)]
            assert len(angles) == 4
            a1, a2, d1, d2 = angles
            a3 = 180 - a1 - a2
            f3 = 180 - d1 - d2
            for ang in (a1, a2, a3, d1, d2, f3):
                assert 20 <= ang <= 130, f"sliver angle {ang} in {text!r}"
        elif difficulty == "medium":
            sides = [int(x) for x in re.findall(r"= (\d+) cm", text)]
            assert len(sides) == 6
            abc, defs = sides[:3], sides[3:]
            for tri in (abc, defs):
                lo, hi = min(tri), max(tri)
                assert lo >= 4, f"side {lo} < 4 in {text!r}"
                assert hi <= 3 * lo, f"ratio {hi}/{lo} > 3 in {text!r}"
                # The ratio bound and the 15% triangle-inequality margin are
                # NOT a proxy for "no sliver angle" -- (4, 10, 12) clears both
                # yet has an 18.2 degree angle. Recompute all three angles via
                # Law of Cosines directly from the rendered sides.
                for ang in _angles_from_sides(tri):
                    assert 20 <= ang <= 140, f"sliver angle {ang:.1f} in triangle {tri} ({text!r})"
        else:  # hard
            nums = [int(x) for x in re.findall(r"(\d+) cm", text)]
            angs = [int(x) for x in re.findall(r"= (\d+) degrees", text)]
            assert len(nums) == 4 and len(angs) == 2
            a, b, d, e = nums
            angle_c, angle_f = angs
            for side_pair, angle_deg in (((a, b), angle_c), ((d, e), angle_f)):
                lo, hi = min(side_pair), max(side_pair)
                assert lo >= 4 and hi <= 3 * lo, f"sliver sides {side_pair} in {text!r}"
                assert 20 <= angle_deg <= 140, f"included angle {angle_deg} out of bounds in {text!r}"
                # Derive the third side and the other two angles independently
                # and check the FULL triangle, not just the two given sides.
                third = _law_of_cosines_third_side(lo, hi, angle_deg)
                full_sides = sorted([lo, hi, third])
                assert full_sides[2] <= 3 * full_sides[0] + 1e-6, f"derived-side sliver ratio {full_sides} in {text!r}"
                # +/- 1e-6 tolerance: forward (given angle -> third side) then
                # backward (three sides -> angle) Law of Cosines is not
                # bit-exact, so an angle that is mathematically exactly 20
                # degrees can round-trip to 19.999999999999993. This is a
                # float precision allowance, not a widened plausibility bound
                # -- confirmed by re-deriving the same row with the sides in
                # the opposite order, which lands on 20.00000000000001.
                for ang in _angles_from_sides([side_pair[0], side_pair[1], third]):
                    assert 20 - 1e-6 <= ang <= 140 + 1e-6, f"derived sliver angle {ang:.6f} in {text!r}"


def test_classify_balance_at_400_seeds():
    """SIMILAR/NOT_SIMILAR is drawn 50/50 by construction; report the actual figure at 400 seeds per tier."""
    for difficulty in TIERS:
        similar = 0
        for seed in range(400):
            q = make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty)
            if q.answer_text != OPTION_NOT_SIMILAR:
                similar += 1
        pct = similar / 400
        # Generous band: a fair coin's 400-draw share deviates from 50% by
        # more than 15 points about 1-in-1000 times.
        assert 0.35 <= pct <= 0.65, f"{difficulty} similar share {pct} outside [0.35, 0.65]"


# --- Skill 2: similar_figure_scale_factor_length ----------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_answer_is_exact_integer(difficulty):
    for seed in SEEDS:
        q = make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)
        assert "." not in q.answer_text
        for d in q.distractors:
            assert "." not in d and "/" not in d


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_real_grader_accepts_answer_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
        wrong = q.answer + 1
        assert not grade_number(q, str(wrong)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_never_mentions_area_surface_area_or_volume(difficulty):
    """The whole boundary against scale_factor_similar_areas/_solids: assert it
    structurally across 500 seeds, not by reading a handful of rows."""
    hits = 0
    for seed in range(500):
        q = make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
        low = q.text.lower()
        if "area" in low or "volume" in low:
            hits += 1
    assert hits == 0, f"{difficulty}: {hits}/500 questions mentioned area/volume"


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_plausibility_ratio_and_bounds(difficulty):
    """Ratio pool bounded away from k=1 and extremes; every length in [3, 60]
    for easy/medium's stated inputs; perimeter consistent with a real polygon
    (hard). Run directly against rendered numbers."""
    for seed in range(200):
        q = make(LENGTH_TOPIC, seed=seed, difficulty=difficulty)
        text = q.text
        if difficulty == "easy":
            ratio_match = re.search(r"scale factor from Figure A to Figure B is (\d+)/(\d+)", text)
            p, den = int(ratio_match.group(1)), int(ratio_match.group(2))
            ratio = Fraction(p, den)
            # Away from 1 and away from extremes: the easy pool's own values
            # (matching p6f's RATIO_PAIRS range) never exceed a 6:5 closeness
            # to 1 or a 3:1 spread.
            assert ratio != 1
            assert Fraction(6, 5) <= ratio <= 3 or Fraction(1, 3) <= ratio <= Fraction(5, 6)
        elif difficulty == "medium":
            lengths = [int(x) for x in re.findall(r"(\d+) cm", text)]
            for length in lengths:
                assert 3 <= length <= 60, f"length {length} outside [3,60] in {text!r}"
        else:  # hard
            side_lengths = [int(x) for x in re.findall(r"measures (\d+) cm", text)]
            perimeter = int(re.search(r"perimeter of (\d+) cm", text).group(1))
            assert perimeter >= 3 * side_lengths[0], f"perimeter floor violated in {text!r}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_length_variety_at_least_20_distinct_over_60_seeds(difficulty):
    texts = {make(LENGTH_TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_classify_variety_at_least_20_distinct_over_60_seeds(difficulty):
    texts = {make(CLASSIFY_TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


# --- skills loader -----------------------------------------------------------


def test_both_skills_load_via_api_skills():
    domains, skills = load_skills()
    skills_dict = {s.id: s for s in skills}

    classify_skill = skills_dict.get("senior-review-the-conditions-for-similarity-of-two-dimensional")
    assert classify_skill is not None
    assert classify_skill.answer == "mcq"
    assert classify_skill.prerequisites == []
    assert classify_skill.placement_eligible is False

    length_skill = skills_dict.get("senior-use-the-scale-factor-for-two-similar-figures")
    assert length_skill is not None
    assert length_skill.answer == "number"
    assert length_skill.prerequisites == []
    assert length_skill.placement_eligible is False
