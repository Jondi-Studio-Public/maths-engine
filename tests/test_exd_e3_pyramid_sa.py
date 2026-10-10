"""Exam-tier source for senior-find-the-surface-area-of-pyramids-such-as (topic
find_the_surface_area_of_pyramids_such_as, mode=square_base_from_height).

SURFACE AREA ONLY. This test file asserts only on what this change builds -- the exam-tier source that gives a
square pyramid's base side and apex (perpendicular) height instead of a slant height, requiring the learner to
derive the slant height via Pythagoras before applying the surface-area formula. It does not add, assert on,
or reference any volume part or the separate volume-and-surface-area pairing skill.

The generator, test file and this oracle share one author's reading of the
contract for everything except the oracle section at the bottom, which
re-derives the answer purely from the rendered question text (never imports
`SQUARE_PYRAMID_TABLE`, `_build_pyramid_square_base_from_height`, or any
other generator-internal helper) so a slip shared between generator and test
cannot hide from it.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p6f_measurement_faces_build import (
    _verify_pyramid_square_base_from_height,
)

TOPIC = "find_the_surface_area_of_pyramids_such_as"
SKILL_ID = "senior-find-the-surface-area-of-pyramids-such-as"
SEEDS = range(40)

# Two independent anchor regexes, not one rigid full-sentence match: the exam tier renders one of
# THREE structurally distinct phrasings that vary clause order, vocabulary and closing verb, but all
# three keep these two anchor phrases identical so s and h can always be recovered this way
# regardless of which variant rendered.
SIDE_RE = re.compile(r"side length (\d+) cm")
HEIGHT_RE = re.compile(r"perpendicular height of (\d+) cm")


def _parse_s_h(text: str) -> tuple[int, int]:
    side_match = SIDE_RE.search(text)
    height_match = HEIGHT_RE.search(text)
    assert side_match and height_match, f"could not parse s, h from {text!r}"
    return int(side_match.group(1)), int(height_match.group(1))


# --- determinism / basic hygiene --------------------------------------------


def test_exam_tier_is_deterministic_across_seeds():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        repeat = make(TOPIC, seed=seed, difficulty="exam")
        assert q.to_json() == repeat.to_json()


def test_exam_tier_distractors_distinct_nonempty_never_equal_answer():
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1
        if len(q.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


def test_exam_tier_answer_is_plain_int():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert isinstance(q.answer, int)


def test_exam_tier_answer_text_never_uses_comma_separator():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "," not in q.answer_text


def test_exam_tier_answer_and_distractors_formatted_alike():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert q.answer_text.isdigit()
        for d in q.distractors:
            assert d.isdigit()


# --- real grader accept/reject ----------------------------------------------


def test_exam_tier_real_grader_accepts_canonical_and_rejects_every_distractor():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


def test_exam_tier_real_grader_rejects_a_deliberately_wrong_answer():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        wrong = q.answer + 1
        assert not grade_number(q, str(wrong)).correct


# --- shape: h and s given, l never printed directly -------------------------


def test_exam_tier_gives_apex_height_not_slant_height():
    """The prompt must give the base side and the apex (perpendicular)
    height -- never a slant height directly (that is what every lower tier
    already gives; the escalation is the extra derivation step)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert "perpendicular height" in q.text
        assert "slant height" not in q.text
        assert "volume" not in q.text.lower()


def test_exam_tier_h_and_s_imply_a_realizable_integer_slant_height():
    """Every rendered exam-tier question's stated s and h must imply an
    exact integer slant height l via l^2 = h^2 + (s/2)^2 -- the same
    realizability identity the lower tiers already prove, just derived from
    h instead of stored directly as l."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        s, h = _parse_s_h(q.text)
        l_squared = h * h + (s // 2) * (s // 2)
        l = math.isqrt(l_squared)
        assert l * l == l_squared, f"seed {seed}: s={s}, h={h} does not imply an integer slant height"


# --- independent check (in-generator) ---------------------------------------


def test_exam_tier_independent_check_fires_on_corrupted():
    """_verify_pyramid_square_base_from_height derives l from h and s via
    math.isqrt (never reading the generator's stored l), then sums the four
    triangular faces plus the base via an explicit loop."""
    # s=6, h=4: half_side=3, l^2 = 16+9=25, l=5. base=36, lateral=4*(3*5)=60. total=96.
    _verify_pyramid_square_base_from_height(6, 4, Fraction(96))  # must not raise
    with pytest.raises(VerificationError):
        _verify_pyramid_square_base_from_height(6, 4, Fraction(95))


def test_exam_tier_independent_check_fires_on_non_realizable_height():
    """A (s, h) pair that does not imply an integer slant height must fail
    the independent check regardless of what `expected` claims."""
    # s=6, h=1: half_side=3, l^2 = 1+9=10, not a perfect square.
    with pytest.raises(VerificationError):
        _verify_pyramid_square_base_from_height(6, 1, Fraction(1000))


def test_exam_tier_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="unknown")


# --- the new misconception: exercisable on every seed -----------------------


def test_exam_tier_uses_apex_height_as_slant_height_distractor_always_present():
    """The tier's defining trap -- plugging h straight into the lateral-face
    formula instead of deriving l first -- must be an exercisable distractor
    on every seed (l > h always, since l^2 = h^2 + (s/2)^2 with s > 0, so
    this differs from the true answer by construction, never merely by
    chance)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        s, h = _parse_s_h(q.text)
        uses_h_as_l = s * s + 2 * s * h
        assert str(uses_h_as_l) in q.distractors, (
            f"seed {seed}: uses_apex_height_as_slant_height distractor {uses_h_as_l} not found in {q.distractors}"
        )
        # And it must actually be wrong (the grader rejects it).
        assert not grade_number(q, str(uses_h_as_l)).correct


# --- variety: at least 20 distinct questions over 60 seeds -------------------


def test_exam_tier_at_least_20_distinct_questions_over_60_seeds():
    questions = [make(TOPIC, seed=s, difficulty="exam").text for s in range(60)]
    distinct = len(set(questions))
    assert distinct >= 20, f"only {distinct} distinct exam-tier questions over 60 seeds"


def test_exam_tier_at_least_3_distinct_masked_prompt_shapes_over_60_seeds():
    """Unlike the easy/medium/hard tiers on this topic, the exam tier renders one of three structurally distinct
    phrasings (`_EXAM_SQUARE_BASE_TEMPLATES`), so a masked-shape sweep sees >= 3 distinct shapes, not only the
    drawn numbers."""
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="exam")) for s in range(60)}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct masked shapes over 60 seeds: {shapes}"


# --- skill loading ------------------------------------------------------------


def test_skill_loads_with_exam_source():
    _, skills = load_skills()
    skills_by_id = {s.id: s for s in skills}
    skill = skills_by_id.get(SKILL_ID)
    assert skill is not None, f"{SKILL_ID} not found in loaded skills"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert len(skill.sources) == 4

    exam_sources = [src for src in skill.sources if src.difficulty == "exam"]
    assert len(exam_sources) == 1, f"expected exactly one exam source, got {exam_sources}"
    exam_src = exam_sources[0]
    assert exam_src.topic == TOPIC
    assert exam_src.params == {"mode": "square_base_from_height"}
    assert exam_src.stages == ["exam"]

    misconception_ids = {m.id for m in skill.misconceptions}
    assert "uses_apex_height_as_slant_height" in misconception_ids


# --- independent oracle: re-derive from rendered text only, no generator import ----


def _oracle_answer_from_text(text: str) -> Fraction:
    """Derive the exam-tier answer purely from the rendered question text --
    parses s and h, derives l via Pythagoras (math.isqrt), and sums the base
    plus four lateral triangles. Reads no generator internals."""
    s_int, h_int = _parse_s_h(text)
    s = Fraction(s_int)
    h = Fraction(h_int)
    half_side = s / 2
    l_squared = h * h + half_side * half_side
    assert l_squared.denominator == 1
    l = math.isqrt(int(l_squared))
    assert l * l == int(l_squared), f"non-integer slant height implied by: {text}"
    base_area = s * s
    lateral_area = 4 * (s * Fraction(l) / 2)
    return base_area + lateral_area


@pytest.mark.parametrize("seed", range(60))
def test_oracle_exam_tier_answer_matches_rendered_text(seed):
    q = make(TOPIC, seed=seed, difficulty="exam")
    expected = _oracle_answer_from_text(q.text)
    assert expected.denominator == 1
    actual = Fraction(q.answer_text)
    assert expected == actual, (
        f"seed {seed}: oracle expected {expected}, generator answer_text was {actual}\nQuestion: {q.text}"
    )
    for distractor_text in q.distractors:
        distractor = Fraction(distractor_text)
        assert distractor != expected, (
            f"seed {seed}: distractor {distractor} matches the oracle's expected answer\nQuestion: {q.text}"
        )
