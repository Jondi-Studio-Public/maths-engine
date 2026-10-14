"""Tests for circle-area-to-diameter-and-circumference.

`mathgen/generators/exd_e5_circle_area_to_diameter_and_circumference_build.py`
is checked directly here for text quality, prompt variety and distractor
count, plus:

1. An INDEPENDENT ORACLE (`_independent_answer_text` below) that parses
   `question.text` alone -- the DISPLAYED (rounded) area, via the literal
   "<area> cm²" substring every template embeds -- and recomputes
   `r' = sqrt(area_display / pi)`, `d' = 2 * r'`, `C' = pi * d'` from
   scratch, rounding each to its own stated precision ONLY at the very
   end. This is the round-trip proof, run for real over every
   seed/tier via `make()`, never imported from
   `mathgen.generators.exd_e5_circle_area_to_diameter_and_circumference_build`
   (confirmed by this file's own import list: only `math`, `re`,
   `decimal`, `mathgen.make`, `mathgen.core`, `mathgen.text_quality`,
   `mathgen.grading`, `mathgen.skills` and -- in the corruption-firing tests
   ONLY -- the generator module itself, to monkeypatch it).
2. A SECOND, fully independent enumeration
   (`test_round_trip_recovers_canonical_values_for_every_possible_radius`)
   over all 19 possible radii (2..20) directly in plain `math`/`Decimal`
   arithmetic -- no `mathgen.make()` call at all -- proving the chosen
   radius domain (2..20) and precision combination (area 2dp, diameter/
   circumference 1dp) round-trips cleanly for EVERY possible draw, not a
   seed sample.
3. A firing test: corrupting one part (diameter, then circumference) to
   return a wrong value and confirming the GENERATOR's own in-body
   independent recomputation (`_final`, run every time `make()` is called)
   raises `VerificationError`.

In the SHIPPED code, `_final` recomputes both parts independently from `r` (never from a part's own
already-returned value, and never by re-deriving from the rounded displayed area) and raises via
`check`/`check_equivalent` on any disagreement. The test that DOES distinguish the two is the
corruption-firing family below.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_e5_circle_area_to_diameter_and_circumference"
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
    """This skill has no structural (odd/even-style) tier lever -- every
    tier draws the radius from the same range and escalates ENTIRELY
    through added template wording (the generator module's own docstring).
    Checked directly here, pairwise, rather than merely assumed from the
    wording design."""
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


# the independent oracle ----------------------------------------------------------
#
# Parses the rendered `question.text` -- the DISPLAYED (rounded) area, via the literal "<area> cm²"
# substring every template embeds -- and recomputes r'/d'/C' from scratch with plain
# `math`/`Decimal` arithmetic, rounding only at the very end. Shares no code, constant or helper
# with `mathgen.generators.exd_e5_circle_area_to_diameter_and_circumference_build` -- confirmed by
# this file's own import list at the top of the module.

_AREA_RE = re.compile(r"(\d+\.\d\d)\s*cm²")
_ANSWER_RE = re.compile(r"^diameter = (\d+\.\d), circumference = (\d+\.\d)$")


def _parse_area(question_text: str) -> Decimal:
    m = _AREA_RE.search(question_text)
    assert m is not None, f"oracle: no '<area> cm²' found in {question_text!r}"
    return Decimal(m.group(1))


def _round_half_up_str(value, places: int) -> str:
    """Independent half-up ("away from zero") rounding to `places` decimal
    places, formatted with EXACTLY `places` digits -- NOT imported from
    `mathgen.core.round_half_up`/`stated_precision_string`."""
    quant = Decimal(1).scaleb(-places)
    d = value if isinstance(value, Decimal) else Decimal(repr(value))
    return str(d.quantize(quant, rounding=ROUND_HALF_UP))


def _independent_answer_text(question_text: str) -> str:
    """Re-derive `answer_text` from the rendered prompt alone -- the
    round-trip proof. Never reads `question.answer`/`question.working`.
    Computes r'/d'/C' at full precision from the DISPLAYED area and rounds
    each of d'/C' only once, at the very end -- exactly the route a learner
    with a calculator would take."""
    area_display = _parse_area(question_text)
    r_prime = math.sqrt(float(area_display) / math.pi)
    d_prime = 2 * r_prime
    c_prime = math.pi * d_prime
    d_text = _round_half_up_str(d_prime, 1)
    c_text = _round_half_up_str(c_prime, 1)
    return f"diameter = {d_text}, circumference = {c_text}"


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_matches_answer_text_over_60_seeds(tier):
    mismatches = []
    for q in _all_questions(tier):
        oracle_answer = _independent_answer_text(q.text)
        if oracle_answer != q.answer_text:
            mismatches.append((q.seed, q.text, oracle_answer, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_independent_oracle_answer_shape_is_recognised(tier):
    for q in _all_questions(tier, seeds=range(30)):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


def test_round_trip_recovers_canonical_values_for_every_possible_radius():
    """A SECOND, fully independent enumeration -- no `mathgen.make()` call
    at all -- over EVERY one of the 19 possible radii (2..20), the
    generator's own chosen domain. Proves the chosen precision combination
    (area to 2dp, diameter/circumference to 1dp) round-trips cleanly for
    every possible draw, not a seed sample: rounding the true area to 2dp
    and then recomputing r'/d'/C' from that rounded value alone reproduces
    the true diameter/circumference at their own 1dp precision, for all 19
    radii."""
    mismatches = []
    for r in range(2, 21):
        area_true = math.pi * r * r
        area_display = Decimal(_round_half_up_str(area_true, 2))
        d_true = 2 * r
        c_true = math.pi * d_true
        d_true_text = _round_half_up_str(d_true, 1)
        c_true_text = _round_half_up_str(c_true, 1)

        r_prime = math.sqrt(float(area_display) / math.pi)
        d_prime = 2 * r_prime
        c_prime = math.pi * d_prime
        d_prime_text = _round_half_up_str(d_prime, 1)
        c_prime_text = _round_half_up_str(c_prime, 1)

        if d_prime_text != d_true_text or c_prime_text != c_true_text:
            mismatches.append(
                (r, float(area_true), str(area_display), d_true_text, d_prime_text, c_true_text, c_prime_text)
            )
    assert mismatches == [], mismatches


# -- the independent-verification-firing tests --------------------------------
#
# These tests (only) import the generator module directly, to monkeypatch a
# module-level part function into returning a corrupted value -- exercises
# the GENERATOR's own in-body independent recomputation (`_final`, run on
# every real call to `make()`), not the oracle above.

import mathgen.generators.exd_e5_circle_area_to_diameter_and_circumference_build as _gen  # noqa: E402


def test_corrupted_diameter_part_raises_verification_error(monkeypatch):
    original = _gen._part_diameter

    def _corrupted(rng, values, *, r, diameter_rounded):
        value, working = original(rng, values, r=r, diameter_rounded=diameter_rounded)
        return value + 1, working  # wrong-by-1 diameter

    monkeypatch.setattr(_gen, "_part_diameter", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard")


def test_corrupted_diameter_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="hard")  # no exception


def test_corrupted_circumference_part_raises_verification_error(monkeypatch):
    original = _gen._part_circumference

    def _corrupted(rng, values, *, r, circumference_rounded):
        value, working = original(rng, values, r=r, circumference_rounded=circumference_rounded)
        return value + 1, working  # wrong-by-1 circumference

    monkeypatch.setattr(_gen, "_part_circumference", _corrupted)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="easy")


def test_corrupted_circumference_part_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=1, difficulty="easy")  # no exception


# -- the magnitude constraint: diameter >= 4, every seed/tier ---------------
#
# Proven here as an EXPLICIT, UNCONDITIONAL assertion over every seed/tier tested, not merely
# asserted by the radius domain's own construction.


@pytest.mark.parametrize("tier", TIERS)
def test_diameter_never_below_four_and_circumference_never_below_floor(tier):
    for q in _all_questions(tier):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        diameter = float(m.group(1))
        circumference = float(m.group(2))
        assert diameter >= 4, f"{tier} seed {q.seed}: diameter {diameter} < 4"
        assert circumference >= 4 * math.pi, f"{tier} seed {q.seed}: circumference {circumference} below its floor"
        assert diameter > 0.051 and circumference > 0.051


# -- every distractor's gap explicitly, every seed/tier ----------------------
#
# Every gap is computed directly from the rendered question/answer text alone (never from the
# generator's own internals), per the `> 0.5` floor requirement for every `answer: pair`
# skill


@pytest.mark.parametrize("tier", TIERS)
def test_reports_radius_as_diameter_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        m = _ANSWER_RE.match(q.answer_text)
        true_diameter = float(m.group(1))
        r = true_diameter / 2  # diameter is always exactly 2 * r, so this recovers r exactly
        wrong_diameter = r  # reports the (undoubled) radius as the diameter
        gap = abs(true_diameter - wrong_diameter)
        assert gap > 0.5, f"{tier} seed {q.seed}: reports-radius-as-diameter gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_circumference_direct_from_area_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        area_display = float(_parse_area(q.text))
        m = _ANSWER_RE.match(q.answer_text)
        true_circumference = float(m.group(2))
        wrong_circumference = 2 * math.pi * area_display  # dimensionally wrong: applies
        # circumference's 2*pi factor directly to the AREA instead of going through r/d
        gap = abs(true_circumference - wrong_circumference)
        assert gap > 0.5, f"{tier} seed {q.seed}: circumference-direct-from-area gap {gap} <= 0.5"
        assert gap > 0.051


@pytest.mark.parametrize("tier", TIERS)
def test_diameter_formula_on_radius_distractor_clears_gap_floor(tier):
    for q in _all_questions(tier, seeds=range(40)):
        m = _ANSWER_RE.match(q.answer_text)
        true_diameter = float(m.group(1))
        true_circumference = float(m.group(2))
        r = true_diameter / 2
        wrong_circumference = math.pi * r  # uses the diameter-based formula on the radius
        gap = abs(true_circumference - wrong_circumference)
        assert gap > 0.5, f"{tier} seed {q.seed}: diameter-formula-on-radius gap {gap} <= 0.5"
        assert gap > 0.051


# -- every part's value appears in working -----------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_every_part_value_appears_in_working(tier):
    for q in _all_questions(tier, seeds=range(20)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        diameter_text, circumference_text = m.groups()
        assert len(q.working) == 2, q.working
        assert diameter_text in q.working[0], q.working[0]
        assert circumference_text in q.working[1], q.working[1]


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
        diameter_text, circumference_text = m.groups()
        reordered = f"circumference = {circumference_text}, diameter = {diameter_text}"
        result = grade_pair(q, reordered)
        assert result.correct, f"{tier} seed {q.seed}: reordered labels rejected: {reordered!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_wrong_value_under_a_correct_label(tier):
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        diameter_text, circumference_text = m.groups()
        wrong_circumference = float(circumference_text) + 50  # far outside any 0.051 tolerance
        given = f"diameter = {diameter_text}, circumference = {wrong_circumference}"
        result = grade_pair(q, given)
        assert not result.correct, f"{tier} seed {q.seed}: wrong circumference value wrongly accepted: {given!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_correct_values_under_swapped_labels(tier):
    """diameter and circumference VALUES swapped between their own labels --
    same value set, wrong pairing. circumference is always ~pi times the
    diameter at this skill's radius range, so the two never coincide -- a
    genuine test on every seed, not merely on a lucky one."""
    for q in _all_questions(tier, seeds=range(10)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        diameter_text, circumference_text = m.groups()
        assert diameter_text != circumference_text, (
            f"{tier} seed {q.seed}: diameter == circumference, cannot demonstrate a swap"
        )
        swapped = f"diameter = {circumference_text}, circumference = {diameter_text}"
        result = grade_pair(q, swapped)
        assert not result.correct, f"{tier} seed {q.seed}: swapped diameter/circumference wrongly accepted: {swapped!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_rejects_dropped_label(tier):
    """A two-label canonical against a learner answer with only one label
    falls through to the positional path -- still correctly rejected here
    since the positional value list lengths differ."""
    for q in _all_questions(tier, seeds=range(5)):
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None
        diameter_text, _circumference_text = m.groups()
        dropped = f"diameter = {diameter_text}"
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
        skill = next(s for s in skills if s.id == "circle-area-to-diameter-and-circumference")
        assert skill.domain == "measurement"
        assert skill.answer == "pair"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"f10-circle-measures"}
        stages_covered = set()
        for source in skill.sources:
            # The default exam tier is the multi-part chain (tests/test_exam_parts_circle_radius_arc.py); the old
            # single-part generator is the extra `exam-single` source.
            is_chain = source.difficulty == "exam" and source.effective_key == "exam"
            assert source.topic == ("exam_parts_circle_radius_arc" if is_chain else TOPIC)
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
