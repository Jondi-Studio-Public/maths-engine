"""Surface area of pyramids and addition-of-faces for irregular solids
(senior-find-the-surface-area-of-pyramids-such-as, senior-use-addition-of-the-area-of-the-faces)."""

import math
from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p6f_measurement_faces_build import (
    _verify_pyramid_square_base,
    _verify_pyramid_rectangular_base,
    _verify_pyramid_right_triangle_base,
    _verify_faces_easy,
    _verify_faces_medium,
    _verify_faces_hard,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

PYRAMID_TOPIC = "find_the_surface_area_of_pyramids_such_as"
FACES_TOPIC = "use_addition_of_the_area_of_the_faces"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors `test_all_topics_usually_have_enough_distractors`'s own bar
    (>=2 distractors on >=85% of seeds) for a bare `make(topic, seed=s)`
    call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_text_never_uses_comma_separator(difficulty):
    """Ensure answer_text never contains comma (thousands separator).
    Commas are fine in prose question text but not in the canonical answer."""
    for seed in SEEDS:
        for topic in [PYRAMID_TOPIC, FACES_TOPIC]:
            question = make(topic, seed=seed, difficulty=difficulty)
            assert "," not in question.answer_text


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_integer(difficulty):
    """Ensure answer is a plain int (not Fraction)."""
    for seed in SEEDS:
        for topic in [PYRAMID_TOPIC, FACES_TOPIC]:
            question = make(topic, seed=seed, difficulty=difficulty)
            assert isinstance(question.answer, int)


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- pyramid-specific tests --------------------------------------------------


def test_pyramid_square_base_independent_check_fires_on_corrupted():
    """Test that _verify_pyramid_square_base fires on a mismatch."""
    from fractions import Fraction

    # 5² + 2*5*10 = 25 + 100 = 125
    _verify_pyramid_square_base(5, 10, Fraction(125))  # must not raise
    with pytest.raises(VerificationError):
        _verify_pyramid_square_base(5, 10, Fraction(124))


def test_pyramid_rectangular_base_independent_check_fires_on_corrupted():
    """Test that _verify_pyramid_rectangular_base fires on a mismatch."""
    from fractions import Fraction

    # 6*3 + 6*5 + 3*4 = 18 + 30 + 12 = 60
    _verify_pyramid_rectangular_base(6, 3, 5, 4, Fraction(60))  # must not raise
    with pytest.raises(VerificationError):
        _verify_pyramid_rectangular_base(6, 3, 5, 4, Fraction(59))


def test_pyramid_right_triangle_base_independent_check_fires_on_corrupted():
    """Test that _verify_pyramid_right_triangle_base fires on a mismatch."""
    from fractions import Fraction

    # Base: 6*8/2 = 24; Laterals: 6*3/2 + 8*4/2 + 10*5/2 = 9 + 16 + 25 = 50; Total: 74
    _verify_pyramid_right_triangle_base(6, 8, 10, 3, 4, 5, Fraction(74))  # must not raise
    with pytest.raises(VerificationError):
        _verify_pyramid_right_triangle_base(6, 8, 10, 3, 4, 5, Fraction(73))


def test_pyramid_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(PYRAMID_TOPIC, seed=1, mode="unknown")


def test_pyramid_rectangular_base_slant_heights_resampled_when_equal():
    """Test that when s_l == s_w, resampling occurs (needed for swap distractor).
    Count how many times resampling fires across 200 seeds."""
    # This is a statistical test: with randint(3, 15) drawn twice, there's a 1/13
    # probability they're equal on each attempt. We expect to see some seeds hit this.
    # Over 200 seeds, we expect roughly 15 hits.
    # We just verify that resampling doesn't hang and produces valid output.
    for seed in range(200):
        question = make(PYRAMID_TOPIC, seed=seed, difficulty="medium")
        # If we get here, resampling didn't hang
        assert question.answer >= 0


def test_pyramid_medium_always_l_base_greater_than_w():
    """Medium tier: l_base > w always (for correct 'longer sides' wording)."""
    for seed in SEEDS:
        question = make(PYRAMID_TOPIC, seed=seed, difficulty="medium")
        # Extract dimensions from text: "rectangular base measuring {l_base} cm by {w} cm"
        match = re.search(r"base measuring (\d+) cm by (\d+) cm", question.text)
        assert match
        l_base, w = int(match.group(1)), int(match.group(2))
        assert l_base > w, f"seed {seed}: l_base={l_base} not > w={w}"


def test_square_pyramid_slant_height_is_realizable():
    """Every rendered square-base pyramid must correspond to a real solid: the
    stated slant height l and base side s must imply a non-negative integer
    apex height h with l^2 = h^2 + (s/2)^2 (single shared apex)."""
    for seed in range(60):
        question = make(PYRAMID_TOPIC, seed=seed, difficulty="easy")
        match = re.search(
            r"square base with side length (\d+) cm\. Each of its four "
            r"triangular faces has a slant height of (\d+) cm",
            question.text,
        )
        assert match, f"seed {seed}: could not parse s, l from {question.text!r}"
        s, l = int(match.group(1)), int(match.group(2))
        h_squared = l * l - (s // 2) * (s // 2)
        assert h_squared >= 0, f"seed {seed}: s={s}, l={l} gives negative h^2"
        h = math.isqrt(h_squared)
        assert h * h == h_squared, f"seed {seed}: s={s}, l={l} does not imply an integer apex height"


def test_rectangular_pyramid_slant_heights_share_one_apex():
    """Every rendered rectangular-base pyramid's two distinct slant heights
    must imply the SAME apex height h: s_l^2 = h^2 + (w/2)^2 and
    s_w^2 = h^2 + (l_base/2)^2 must agree on h^2."""
    for seed in range(60):
        question = make(PYRAMID_TOPIC, seed=seed, difficulty="medium")
        match = re.search(
            r"base measuring (\d+) cm by (\d+) cm\. The two triangular faces "
            r"on the longer sides have a slant height of (\d+) cm each; the "
            r"two triangular faces on the shorter sides have a slant height "
            r"of (\d+) cm each",
            question.text,
        )
        assert match, f"seed {seed}: could not parse dims from {question.text!r}"
        l_base, w, s_l, s_w = (int(match.group(i)) for i in range(1, 5))
        h_squared_from_l = s_l * s_l - (w // 2) * (w // 2)
        h_squared_from_w = s_w * s_w - (l_base // 2) * (l_base // 2)
        assert h_squared_from_l >= 0, f"seed {seed}: s_l implies negative h^2"
        assert h_squared_from_w >= 0, f"seed {seed}: s_w implies negative h^2"
        assert h_squared_from_l == h_squared_from_w, (
            f"seed {seed}: s_l and s_w imply different apex heights "
            f"({h_squared_from_l} vs {h_squared_from_w}) -- not one real pyramid"
        )


def test_right_triangle_pyramid_slant_heights_share_one_apex():
    """Every rendered right-triangle-base pyramid must state the SAME slant
    height on all three lateral faces, and that value must correspond to a
    non-negative integer apex height h at the base's incentre:
    s^2 = h^2 + r^2, r = (a+b-c)/2."""
    for seed in range(60):
        question = make(PYRAMID_TOPIC, seed=seed, difficulty="hard")
        match = re.search(
            r"legs (\d+) cm and (\d+) cm and hypotenuse (\d+) cm\. The "
            r"triangular face on the \d+ cm edge has a slant height of "
            r"(\d+) cm, the face on the \d+ cm edge has a slant height of "
            r"(\d+) cm, and the face on the \d+ cm edge has a slant height "
            r"of (\d+) cm",
            question.text,
        )
        assert match, f"seed {seed}: could not parse dims from {question.text!r}"
        a, b, c, s_a, s_b, s_c = (int(match.group(i)) for i in range(1, 7))
        assert s_a == s_b == s_c, (
            f"seed {seed}: lateral slant heights are not all equal "
            f"({s_a}, {s_b}, {s_c}) -- not realizable at a single incentre"
        )
        s = s_a
        r = (a + b - c) // 2
        assert (a + b - c) % 2 == 0, f"seed {seed}: r=(a+b-c)/2 not an integer"
        assert r >= 0, f"seed {seed}: negative inradius r={r}"
        h_squared = s * s - r * r
        assert h_squared >= 0, f"seed {seed}: s={s}, r={r} gives negative h^2"
        h = math.isqrt(h_squared)
        assert h * h == h_squared, f"seed {seed}: s={s}, r={r} does not imply an integer apex height"


# --- faces-specific tests ----------------------------------------------------


def test_faces_easy_independent_check_fires_on_corrupted():
    """Test that _verify_faces_easy fires on a mismatch."""
    from fractions import Fraction

    # 3 rectangles: 5*6 + 7*8 + 9*10 = 30 + 56 + 90 = 176
    # 2 squares: 4² + 5² = 16 + 25 = 41
    # Total: 217
    _verify_faces_easy(5, 6, 7, 8, 9, 10, 4, 5, Fraction(217))  # must not raise
    with pytest.raises(VerificationError):
        _verify_faces_easy(5, 6, 7, 8, 9, 10, 4, 5, Fraction(216))


def test_faces_medium_independent_check_fires_on_corrupted():
    """Test that _verify_faces_medium fires on a mismatch."""
    from fractions import Fraction

    # 2 rectangles: 5*6 + 7*8 = 30 + 56 = 86
    # 2 triangles: (1/2)*4*6 + (1/2)*8*10 = 12 + 40 = 52
    # 1 square: 9² = 81
    # Total: 219
    _verify_faces_medium(5, 6, 7, 8, 4, 6, 8, 10, 9, Fraction(219))  # must not raise
    with pytest.raises(VerificationError):
        _verify_faces_medium(5, 6, 7, 8, 4, 6, 8, 10, 9, Fraction(218))


def test_faces_hard_independent_check_fires_on_corrupted():
    """Test that _verify_faces_hard fires on a mismatch."""
    from fractions import Fraction

    # 3 rectangles: 5*6 + 7*8 + 9*10 = 30 + 56 + 90 = 176
    # 2 triangles: (1/2)*4*6 + (1/2)*8*10 = 12 + 40 = 52
    # Total: 228
    # Lure: 6² = 36 (must not be included)
    _verify_faces_hard(5, 6, 7, 8, 9, 10, 4, 6, 8, 10, 6, Fraction(228))  # must not raise
    with pytest.raises(VerificationError):
        _verify_faces_hard(5, 6, 7, 8, 9, 10, 4, 6, 8, 10, 6, Fraction(227))


def test_faces_hard_includes_lure_distractor_on_every_seed():
    """THE hard-tier trap: verify includes_the_excluded_face distractor is present
    at every one of 40+ swept seeds (not merely usually)."""
    for seed in SEEDS:
        question = make(FACES_TOPIC, seed=seed, difficulty="hard")
        # Extract the lure square's side length from the text
        match = re.search(r"square face of side (\d+) cm", question.text)
        assert match, f"seed {seed}: could not find lure square in text"
        s_x = int(match.group(1))
        lure_area = s_x * s_x
        expected_lure_distractor = str(int(question.answer) + lure_area)
        assert expected_lure_distractor in question.distractors, (
            f"seed {seed}: includes_the_excluded_face distractor {expected_lure_distractor} "
            f"not found in {question.distractors}"
        )


def test_faces_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(FACES_TOPIC, seed=1, mode="unknown")


# --- face phrasing consistency (the oracle regex-parses these) ---------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_face_easy_phrasing_exact_match(difficulty):
    """Verify easy tier uses exact face-sentence phrasing."""
    if difficulty != "easy":
        return
    question = make(FACES_TOPIC, seed=1, difficulty="easy")
    # Should have Face 1 is a rectangle, Face 2 is a rectangle, etc.
    assert re.search(r"Face 1 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 2 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 3 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 4 is a square with side \d+ cm\.", question.text)
    assert re.search(r"Face 5 is a square with side \d+ cm\.", question.text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_face_medium_phrasing_exact_match(difficulty):
    """Verify medium tier uses exact face-sentence phrasing."""
    if difficulty != "medium":
        return
    question = make(FACES_TOPIC, seed=1, difficulty="medium")
    # Should have rectangles, triangles, square
    assert re.search(r"Face 1 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 2 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 3 is a triangle with base \d+ cm and height \d+ cm\.", question.text)
    assert re.search(r"Face 4 is a triangle with base \d+ cm and height \d+ cm\.", question.text)
    assert re.search(r"Face 5 is a square with side \d+ cm\.", question.text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_face_hard_phrasing_exact_match(difficulty):
    """Verify hard tier uses exact face-sentence phrasing."""
    if difficulty != "hard":
        return
    question = make(FACES_TOPIC, seed=1, difficulty="hard")
    # Should have 3 rectangles and 2 triangles
    assert re.search(r"Face 1 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 2 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 3 is a rectangle \d+ cm by \d+ cm\.", question.text)
    assert re.search(r"Face 4 is a triangle with base \d+ cm and height \d+ cm\.", question.text)
    assert re.search(r"Face 5 is a triangle with base \d+ cm and height \d+ cm\.", question.text)
    # Should have the lure sentence
    assert "The solid is open on one side" in question.text
    assert re.search(r"square face of side \d+ cm", question.text)


# --- variety: at least 20 distinct questions per tier over 60 seeds -------


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_at_least_20_distinct_questions_per_tier(topic, difficulty):
    questions = []
    for seed in range(60):
        question = make(topic, seed=seed, difficulty=difficulty)
        questions.append(question.text)
    distinct = len(set(questions))
    assert distinct >= 20, f"{topic} {difficulty}: only {distinct} distinct questions over 60 seeds (need >=20)"


# --- answer and distractors formatted alike ---


@pytest.mark.parametrize("topic", [PYRAMID_TOPIC, FACES_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_and_distractors_formatted_alike(topic, difficulty):
    """All answers and distractors are plain integers (no decimals or special formatting)."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # Answer should be all digits
        assert question.answer_text.isdigit(), (
            f"{topic} {difficulty} seed {seed}: answer_text={question.answer_text} is not all digits"
        )
        # Every distractor should also be all digits
        for distractor in question.distractors:
            assert distractor.isdigit(), (
                f"{topic} {difficulty} seed {seed}: distractor={distractor} "
                f"is not all digits (answer_text={question.answer_text})"
            )


# --- skills loading -------


def test_skills_load_with_correct_properties():
    """Verify both skills load through mathgen.skills.load_skills with correct
    prerequisites, answer type, and placement_eligible."""
    _, skills = load_skills()
    skills_by_id = {skill.id: skill for skill in skills}

    # Check skill 1: pyramid
    skill1 = skills_by_id.get("senior-find-the-surface-area-of-pyramids-such-as")
    assert skill1 is not None, "pyramid skill not found in loaded skills"
    assert skill1.prerequisites == ["convert-metric-length"]
    assert skill1.answer == "number"
    assert skill1.placement_eligible == False

    # Check skill 2: faces
    skill2 = skills_by_id.get("senior-use-addition-of-the-area-of-the-faces")
    assert skill2 is not None, "faces skill not found in loaded skills"
    assert skill2.prerequisites == ["convert-metric-length"]
    assert skill2.answer == "number"
    assert skill2.placement_eligible == False
