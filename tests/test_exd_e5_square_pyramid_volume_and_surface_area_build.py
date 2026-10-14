"""Tests for square-pyramid-volume-and-surface-area.

The checks for `mathgen/generators/exd_e5_square_pyramid_volume_and_surface_area_build.py` are:

1. An INDEPENDENT ORACLE (`_independent_answer_text` below) that parses
   `question.text` alone -- the base side (or, at the `hard` tier, the base
   AREA, recovered to a side via `math.isqrt`) and the perpendicular height
   -- and re-derives `l`/volume/surface_area from scratch with plain `math`/
   `Fraction` arithmetic. It does NOT import anything from
   `mathgen.generators.exd_e5_square_pyramid_volume_and_surface_area_build`
   -- not a helper, not a constant, not `_TRIPLES` (confirmed by this file's
   own import list: only `re`, `math`, `fractions`, `mathgen.make`,
   `mathgen.core`, `mathgen.text_quality`, `mathgen.grading`, `mathgen.skills` and --
   in the corruption-firing tests and the OWN-TABLE transcription check
   ONLY -- the generator module itself).
2. A firing test: corrupting one part (slant, then volume, then
   surface_area) to return a wrong value and confirming the GENERATOR's own
   in-body independent recomputation (`_final`, run every time `make()` is
   called) raises `VerificationError`.

In the SHIPPED code, `_final` recomputes `l`, `volume` and `surface_area` all a SECOND way, straight from
`s`/`h` (never from a part's own already-returned value), and raises via `check`/`check_equivalent` on any
disagreement. The test that DOES distinguish the two is the corruption-firing family below: it monkeypatches
`_part_slant`/ `_part_volume`/`_part_surface_area` (module-level functions, not closures, exactly so this
monkeypatch is possible) to return a wrong-by-1 value and asserts `make` raises `VerificationError`. A
`_final` that TRUSTED a part's own value instead of rechecking it against `s`/`h` would emit the corrupted
value in `answer_text` with no exception at all, and this test would fail to see one raised.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_square_pyramid_volume_and_surface_area"
SKILL_ID = "square-pyramid-volume-and-surface-area"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)


def _all_questions(tier, seeds=SEEDS):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


# -- by-hand text-quality, variety and distractor checks --------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_bad_text_patterns_over_a_seed_sweep(tier):
    for q in _all_questions(tier, seeds=range(30)):
        defects = find_text_defects(q)
        assert defects == [], f"{tier}: {defects} for {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_variety_per_tier(tier):
    shapes = {prompt_shape(q) for q in _all_questions(tier)}
    assert len(shapes) >= 3, f"{tier}: only {len(shapes)} distinct prompt shapes over 60 seeds: {shapes}"


def test_tiers_are_masking_safe_shapes_never_collide_across_tiers():
    """Every tier draws (s, h) from the same curated triple table -- escalation
    is entirely through added template wording (easy -> medium) and, at
    `hard`, a genuinely different GIVEN quantity (base area instead of base
    side). Checked directly here, pairwise, rather than merely assumed from
    the wording design."""
    shapes = {tier: {prompt_shape(q) for q in _all_questions(tier)} for tier in TIERS}
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1 :]:
            overlap = shapes[a] & shapes[b]
            assert overlap == set(), f"{a} and {b} share shape(s): {overlap}"


@pytest.mark.parametrize("tier", TIERS)
def test_determinism_same_seed_and_tier_is_byte_identical(tier):
    for seed in (0, 1, 2, 3, 40, 59):
        first = make(TOPIC, seed=seed, difficulty=tier).to_json()
        second = make(TOPIC, seed=seed, difficulty=tier).to_json()
        assert first == second


# -- the OWN-TABLE transcription check: every entry in
# `_TRIPLES` actually satisfies leg_a**2 + leg_b**2 == hyp**2 exactly. This
# imports `_TRIPLES` directly -- deliberately not part of the independent
# oracle above, which never imports the generator -- because the whole point
# here is to catch a transcription error in the generator's OWN table, which
# only reading the table itself can do. -------------------------------------

import mathgen.generators.exd_e5_square_pyramid_volume_and_surface_area_build as _gen  # noqa: E402


def test_every_triple_is_genuinely_pythagorean():
    for leg_a, leg_b, hyp in _gen._TRIPLES:
        assert leg_a * leg_a + leg_b * leg_b == hyp * hyp, (
            f"({leg_a}, {leg_b}, {hyp}) is not a Pythagorean triple: "
            f"{leg_a}**2 + {leg_b}**2 = {leg_a * leg_a + leg_b * leg_b} != {hyp * hyp}"
        )


def test_volume_is_always_an_exact_integer_for_every_triple():
    """Every entry has 3 | leg_a or 3 | leg_b (the module docstring's own
    claim) -- proved directly here for every table entry, not merely
    asserted, since volume = (4/3) * leg_a**2 * leg_b."""
    for leg_a, leg_b, hyp in _gen._TRIPLES:
        volume = Fraction(4, 3) * leg_a * leg_a * leg_b
        assert volume.denominator == 1, f"({leg_a}, {leg_b}, {hyp}): volume {volume} is not an exact integer"


def test_every_drawn_seed_uses_a_table_triple_and_stays_exact():
    """Every LIVE draw over 60 seeds x 4 tiers still satisfies
    l**2 == h**2 + (s/2)**2 exactly -- a required test, run
    against the rendered question (parsed independently, not against the
    generator's internal state)."""
    for tier in TIERS:
        for q in _all_questions(tier):
            s, h = _parse_s_h(tier, q.text)
            m = _ANSWER_RE.match(q.answer_text)
            assert m is not None, q.answer_text
            half_s = s // 2
            assert s % 2 == 0, f"{tier} seed {q.seed}: base side {s} is not even"
            l_squared = h * h + half_s * half_s
            l = math.isqrt(l_squared)
            assert l * l == l_squared, f"{tier} seed {q.seed}: h={h}, s={s} do not imply an exact slant height"


# the independent oracle ---------------------------
#
# Parses the rendered `question.text` -- the base side (easy/medium/exam) or base area (hard,
# recovered to a side via math.isqrt) and the perpendicular height -- and recomputes
# l/volume/surface_area from scratch with plain math/Fraction arithmetic. Shares no code, constant
# or helper with `mathgen.generators.exd_e5_square_pyramid_volume_and_surface_area_build` --
# confirmed by this file's own import list at the top of the module (the `_gen` import above is used
# ONLY by the corruption-firing tests and the own-table transcription check, both explicitly NOT the
# independent oracle).

_S_RE = re.compile(r"base side length of (\d+) cm")
_AREA_RE = re.compile(r"base area of (\d+) cm²")
_H_RE = re.compile(r"perpendicular height of (\d+) cm")
_ANSWER_RE = re.compile(r"^volume = (\d+), surface_area = (\d+)$")


def _parse_s_h(tier: str, question_text: str) -> tuple[int, int]:
    h_match = _H_RE.search(question_text)
    assert h_match is not None, f"oracle: no perpendicular height found in {question_text!r}"
    h = int(h_match.group(1))
    if tier == "hard":
        area_match = _AREA_RE.search(question_text)
        assert area_match is not None, f"oracle: no base area found in {question_text!r}"
        area = int(area_match.group(1))
        s = math.isqrt(area)
        assert s * s == area, f"oracle: base area {area} is not a perfect square"
    else:
        s_match = _S_RE.search(question_text)
        assert s_match is not None, f"oracle: no base side found in {question_text!r}"
        s = int(s_match.group(1))
    return s, h


def _independent_answer_text(tier: str, question_text: str) -> str:
    """Re-derive `answer_text` from the rendered prompt alone -- the
    oracle. Never reads `question.answer`/`question.working`."""
    s, h = _parse_s_h(tier, question_text)
    half_s = s // 2
    l_squared = h * h + half_s * half_s
    l = math.isqrt(l_squared)
    assert l * l == l_squared, f"oracle: h={h}, s={s} do not imply an exact slant height"
    volume = Fraction(s * s * h, 3)
    assert volume.denominator == 1, f"oracle: volume {volume} is not an exact integer"
    volume = int(volume)
    surface_area = s * s + 2 * s * l
    return f"volume = {volume}, surface_area = {surface_area}"


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _all_questions(tier):
        oracle_answer = _independent_answer_text(tier, q.text)
        if oracle_answer != q.answer_text:
            mismatches.append((q.seed, q.text, oracle_answer, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_answer_shape_is_recognised(tier):
    for q in _all_questions(tier, seeds=range(30)):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_slant_height_derivation_line_present_in_working(tier):
    """The slant-height derivation is not itself a graded label, but the
    the skill requires it shown as its own step in `working` at
    every seed."""
    for q in _all_questions(tier, seeds=range(20)):
        assert len(q.working) == 3, q.working
        assert "Slant height" in q.working[0], q.working[0]
        assert "√" in q.working[0], q.working[0]


@pytest.mark.parametrize("tier", TIERS)
def test_every_part_value_appears_in_working(tier):
    """Every part's own value appears somewhere in `working`, so a learner
    can check their intermediate steps -- not only the final answer."""
    for q in _all_questions(tier, seeds=range(20)):
        s, h = _parse_s_h(tier, q.text)
        half_s = s // 2
        l = math.isqrt(h * h + half_s * half_s)
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        volume_text, sa_text = m.groups()
        assert str(l) in q.working[0], q.working[0]
        assert volume_text in q.working[1], q.working[1]
        assert sa_text in q.working[2], q.working[2]


# -- the independent-verification-firing tests --------------------------------
#
# These tests (only, alongside the own-table transcription check above)
# import the generator module directly, to monkeypatch a module-level part
# function into returning a corrupted value -- exercises the GENERATOR's own
# in-body independent recomputation (`_final`, run on every real call to
# `make()`), not the oracle above.


def test_corrupted_slant_part_raises_verification_error(monkeypatch):
    original = _gen._part_slant

    def _corrupted(rng, values, *, s, h, l):
        value, working = original(rng, values, s=s, h=h, l=l)
        return value + 1, working  # wrong-by-1 slant height

    monkeypatch.setattr(_gen, "_part_slant", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_slant_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="hard")  # no exception


def test_corrupted_volume_part_raises_verification_error(monkeypatch):
    original = _gen._part_volume

    def _corrupted(rng, values, *, s, h, volume):
        value, working = original(rng, values, s=s, h=h, volume=volume)
        return value + 1, working  # wrong-by-1 volume

    monkeypatch.setattr(_gen, "_part_volume", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


def test_corrupted_volume_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="easy")  # no exception


def test_corrupted_surface_area_part_raises_verification_error(monkeypatch):
    original = _gen._part_surface_area

    def _corrupted(rng, values, *, s, l, surface_area):
        value, working = original(rng, values, s=s, l=l, surface_area=surface_area)
        return value + 1, working  # wrong-by-1 surface area

    monkeypatch.setattr(_gen, "_part_surface_area", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="exam")


def test_corrupted_surface_area_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="exam")  # no exception


# -- every distractor's gap explicitly, every seed/tier/part ----------------


@pytest.mark.parametrize("tier", TIERS)
def test_apex_height_as_slant_height_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        s, h = _parse_s_h(tier, q.text)
        half_s = s // 2
        l = math.isqrt(h * h + half_s * half_s)
        true_sa = s * s + 2 * s * l
        wrong_sa = s * s + 2 * s * h
        gap = abs(true_sa - wrong_sa)
        assert gap > 0.5, f"{tier} seed {q.seed}: apex-as-slant gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_omits_base_square_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        s, h = _parse_s_h(tier, q.text)
        half_s = s // 2
        l = math.isqrt(h * h + half_s * half_s)
        true_sa = s * s + 2 * s * l
        wrong_sa = 2 * s * l
        gap = abs(true_sa - wrong_sa)
        assert gap > 0.5, f"{tier} seed {q.seed}: omits-base gap {gap} <= 0.5"
        assert gap > 0.051
        assert gap == s * s  # exactly the base area, by construction


@pytest.mark.parametrize("tier", TIERS)
def test_no_half_factor_on_laterals_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        s, h = _parse_s_h(tier, q.text)
        half_s = s // 2
        l = math.isqrt(h * h + half_s * half_s)
        true_sa = s * s + 2 * s * l
        wrong_sa = s * s + 4 * s * l
        gap = abs(true_sa - wrong_sa)
        assert gap > 0.5, f"{tier} seed {q.seed}: no-half-factor gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_no_third_factor_on_volume_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        s, h = _parse_s_h(tier, q.text)
        true_volume = Fraction(s * s * h, 3)
        assert true_volume.denominator == 1
        true_volume = int(true_volume)
        wrong_volume = s * s * h
        gap = abs(true_volume - wrong_volume)
        assert gap > 0.5, f"{tier} seed {q.seed}: no-third-factor gap {gap} <= 0.5"
        assert gap > 0.051
        assert wrong_volume == 3 * true_volume  # exactly 3x, by construction


# -- real grader accept/reject, INCLUDING row 12's label-order battery ------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_canonical_order(tier):
    for q in _all_questions(tier, seeds=range(10)):
        result = grade_pair(q, q.answer_text)
        assert result.correct, f"{tier} seed {q.seed}: canonical order rejected: {q.answer_text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_reordered_labels(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        volume_text, sa_text = m.groups()
        reordered = f"surface_area = {sa_text}, volume = {volume_text}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed {q.seed}: reordered labels rejected: {reordered!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_wrong_value_under_a_correct_label(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        volume_text, sa_text = m.groups()
        wrong_sa = int(sa_text) + 1000  # far outside any 0.051 tolerance
        given = f"volume = {volume_text}, surface_area = {wrong_sa}"
        result = grade_pair(q, given)
        assert not result.correct, f"{tier} seed {q.seed}: wrong surface_area value wrongly accepted: {given!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_correct_values_under_swapped_labels(tier):
    """volume and surface_area VALUES swapped between their own labels --
    same value set, wrong pairing. Skipped on the rare seed where the two
    values happen to coincide (never demonstrable there)."""
    for q in _all_questions(tier, seeds=range(20)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        volume_text, sa_text = m.groups()
        if volume_text == sa_text:
            continue
        swapped = f"volume = {sa_text}, surface_area = {volume_text}"
        result = grade_pair(q, swapped)
        assert not result.correct, f"{tier} seed {q.seed}: swapped volume/surface_area wrongly accepted: {swapped!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_dropped_label(tier):
    """A two-label canonical against a learner answer with only one label
    falls through to the positional path (row 12's 'not measured' note) --
    still correctly rejected here since the positional value list lengths
    differ."""
    for q in _all_questions(tier, seeds=range(5)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        volume_text, _sa_text = m.groups()
        dropped = f"volume = {volume_text}"
        result = grade_pair(q, dropped)
        assert not result.correct, f"{tier} seed {q.seed}: dropped label wrongly accepted: {dropped!r}"


# -- distractor hygiene -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    for q in _all_questions(tier, seeds=range(40)):
        for d in q.distractors:
            assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
        assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"


# -- at least 20 distinct rendered questions per tier over 60 seeds ---------


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(tier):
    texts = {q.text for q in _all_questions(tier)}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"


# -- skill loads --------------------------------------------------------------


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        assert skill.domain == "measurement"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {
            "senior-use-formulas-to-find-the-volume-of-pyramids",
            "pythagorean-hypotenuse",
        }
        stages_covered = set()
        for source in skill.sources:
            # The default exam tier is the multi-part chain (tests/test_exam_parts_square_pyramid.py); the old
            # single-part generator is the extra `exam-single` source.
            is_chain = source.difficulty == "exam" and source.effective_key == "exam"
            assert source.topic == ("exam_parts_square_pyramid" if is_chain else TOPIC)
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam" and s.effective_key == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
        single = skill.sources[-1]  # the old single-part generator, wired as the extra `exam-single` source
        assert (single.topic, single.difficulty, single.effective_key, single.stages) == (
            TOPIC,
            "exam",
            "exam-single",
            ["exam"],
        )
