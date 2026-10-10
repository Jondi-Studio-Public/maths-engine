"""Map-scale distance problems (senior-use-scales-to-find-distances-such-as-on / ACMEM083)."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, stated_precision_tolerance

SEEDS = range(40)
TOPIC = "map_scale_distance"
DIFFICULTIES = ["easy", "medium", "hard"]

# (topic, params) per difficulty, matching the skill's own `sources` entries
# -- generating through the skill's own params, never `make(topic, seed,
# difficulty=tier)` alone.
PARAMS = {
    "easy": dict(mode="verbal_forward"),
    "medium": dict(mode="ratio_forward"),
    "hard": dict(mode="ratio_reverse"),
}


def _make(seed, difficulty):
    return make(TOPIC, seed=seed, difficulty=difficulty, **PARAMS[difficulty])


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generator_is_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        repeat = _make(seed, difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_text_never_uses_comma_separator(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert "," not in question.answer_text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_correct_type(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert isinstance(question.answer, (int, float))
        assert not isinstance(question.answer, bool)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_formatted_alike(difficulty):
    """No option is identifiable by its shape (decimal point present or not)."""
    for seed in SEEDS:
        question = _make(seed, difficulty)
        is_answer_decimal = "." in question.answer_text
        for distractor in question.distractors:
            is_distractor_decimal = "." in distractor
            assert is_answer_decimal == is_distractor_decimal, (
                f"{difficulty} seed {seed}: answer '{question.answer_text}' "
                f"format mismatch with distractor '{distractor}'"
            )


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_indefinite_article_error(difficulty):
    """ "a online map" / "an bushwalking map" would be a wording defect --
    every context noun pairs with the grammatically correct article."""
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert not re.search(r"\ba [aeiou]", question.text, re.IGNORECASE), question.text
        assert not re.search(r"\ban [^aeiou\s]", question.text, re.IGNORECASE), question.text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_never_converges_on_the_similar_figures_skill_wording(difficulty):
    """This skill's context is always a MAP with a scale statement, never
    two similar geometric figures -- the word "similar" (the other skill's
    entire framing, "Figure A and Figure B are similar") must never appear."""
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert "similar" not in question.text.lower()


def test_at_least_20_distinct_questions_over_60_seeds_per_tier():
    for difficulty in DIFFICULTIES:
        texts = {_make(seed, difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct texts over 60 seeds"


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"{difficulty} seed {seed}: real grader rejected its own canonical answer_text {question.answer_text!r}"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, (
                f"{difficulty} seed {seed}: real grader accepted distractor {distractor!r} as correct"
            )


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = _make(seed, difficulty)
        if "." in question.answer_text:
            dp = len(question.answer_text.split(".")[1])
            offset = Fraction(1, 10**dp) * 5  # clears the stated-precision tolerance
        else:
            offset = 1
        wrong = Fraction(question.answer_text) + offset
        assert not grade_number(question, str(float(wrong))).correct


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_decimal_answers_meet_stated_precision_tolerance(difficulty):
    """Every distractor differs from the answer by more than half a unit in
    the LAST decimal place answer_text states (or, for an integer canonical,
    by more than the grader's own exact-match margin)."""
    for seed in SEEDS:
        question = _make(seed, difficulty)
        answer_text = question.answer_text
        if "." not in answer_text:
            tol = Fraction(1, 2)  # a looser but sufficient margin for an
            # integer canonical, which the real grader matches EXACTLY
            # (stated_precision_tolerance(0) is NOT that margin -- see its
            # own docstring).
        else:
            places = len(answer_text.split(".")[1])
            tol = stated_precision_tolerance(places)
        answer_frac = Fraction(answer_text)
        for distractor in question.distractors:
            gap = abs(Fraction(distractor) - answer_frac)
            assert gap > tol, (
                f"{difficulty} seed {seed}: distractor '{distractor}' too close "
                f"to answer '{answer_text}' (gap={gap}, tol={tol})"
            )


# --- independent-check-in-generator fires on corruption ---------------------


def test_verify_forward_fires_on_corrupted_total():
    from mathgen.generators.p10d_map_scale_distance_build import _verify_forward

    with pytest.raises(VerificationError):
        _verify_forward(100000, 6, Fraction(999), "km")


def test_verify_reverse_fires_on_corrupted_total():
    from mathgen.generators.p10d_map_scale_distance_build import _verify_reverse

    with pytest.raises(VerificationError):
        _verify_reverse(100000, 6, Fraction(999), "km")


# Map scale ratios and distances must stay in the range a real map actually
# uses: ratio n in [5000, 250000]; map distances in [1, 30] cm; the resulting
# real-world figure must be sensible (never 0, never needing more than 3
# significant figures to state honestly). Proved RED against an unconstrained
# draft FIRST, then GREEN against the generator's own curated pools -- not
# merely asserted against the pools alone.


def _assert_bounds(ratios, map_distances_cm, real_world_values):
    for n in ratios:
        assert 5000 <= n <= 250000, n
    for d in map_distances_cm:
        assert 1 <= d <= 30, d
    for value in real_world_values:
        assert 1 <= value <= 999, value


def test_plausibility_bounds_red_then_green():
    import mathgen.generators.p10d_map_scale_distance_build as g

    # RED: an UNCONSTRAINED first draft -- a huge ratio far outside a real
    # map ("1 : 5000000"), and a map distance no ruler could measure (500 cm)
    # -- both drawn from a naive `randint` range with no bound at all, the
    # literal risk this section warns against.
    unconstrained_ratios = [5000000, 20000000]
    unconstrained_distances = [500, 1000]
    unconstrained_real_values = [d * n / 100 for d, n in zip(unconstrained_distances, unconstrained_ratios)]
    with pytest.raises(AssertionError):
        _assert_bounds(unconstrained_ratios, unconstrained_distances, unconstrained_real_values)

    # GREEN: the real, curated pools this generator actually draws from.
    easy_reals = [scale * md for scale, md in g.EASY_KM_OPTIONS + g.EASY_M_OPTIONS]
    easy_distances = [md for _, md in g.EASY_KM_OPTIONS + g.EASY_M_OPTIONS]
    ratio_reals = []
    for n in g.RATIO_SCALES:
        for d in g.MAP_DISTANCES_INT:
            value, _unit = g._real_distance_and_unit(n, d)
            ratio_reals.append(value)
    _assert_bounds(g.RATIO_SCALES, easy_distances + g.MAP_DISTANCES_INT, easy_reals + ratio_reals)


def test_plausibility_real_world_values_never_zero_over_full_draw_space():
    import mathgen.generators.p10d_map_scale_distance_build as g

    for scale, md in g.EASY_KM_OPTIONS + g.EASY_M_OPTIONS:
        assert scale * md > 0
    for n in g.RATIO_SCALES:
        for d in g.MAP_DISTANCES_INT:
            value, _unit = g._real_distance_and_unit(n, d)
            assert value > 0


# --- load through mathgen.skills --------------------------------------------------


def test_skill_loads_with_correct_prerequisites_and_placement():
    domains, skills = load_skills()
    skill = next(
        (s for s in skills if s.id == "senior-use-scales-to-find-distances-such-as-on"),
        None,
    )
    assert skill is not None
    assert skill.prerequisites == []
    assert skill.placement_eligible is False
    assert skill.answer == "number"
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}
