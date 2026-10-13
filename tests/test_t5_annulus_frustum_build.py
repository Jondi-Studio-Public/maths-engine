"""Constructibility/plausibility values (R, r[, h/l]) are pulled back out of the
RENDERED question text via regex, the same convention
``test_p8d_quadrilateral_theorems_build.py`` uses for its own
``test_*_angles_plausible_range`` sweeps -- not read off the generator's
internal locals -- so these tests exercise what a learner is actually shown.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import round_half_up
from mathgen.generators.t5_annulus_frustum_build import (
    PI,
    _tenths_tie_distance,
)

ANNULUS = "annulus_cross_section_area"
FRUSTUM = "frustum_volume"
TIERS = ["easy", "medium", "hard"]
SEEDS_40 = range(40)
SEEDS_60 = range(60)


# ---------------------------------------------------------------------------
# Text parsers (independent of the generator's internal variables -- regex
# over the rendered text only, exactly as p8d's own test file does).
# ---------------------------------------------------------------------------


def _annulus_R_r(text: str) -> tuple[Fraction, Fraction]:
    """(R, r) in a single common unit, derived from the rendered annulus text."""
    m = re.search(r"outer radius of (\d+) cm and an inner radius of (\d+) cm", text)
    if m:
        return Fraction(int(m.group(1))), Fraction(int(m.group(2)))
    m = re.search(r"outer diameter of (\d+) cm and an inner radius of (\d+) cm", text)
    if m:
        return Fraction(int(m.group(1)), 2), Fraction(int(m.group(2)))
    m = re.search(r"outer radius of (\d+) cm and an inner diameter of (\d+) cm", text)
    if m:
        return Fraction(int(m.group(1))), Fraction(int(m.group(2)), 2)
    m = re.search(r"outer diameter of (\d+) mm and a wall thickness of ([\d.]+) cm", text)
    if m:
        R = Fraction(int(m.group(1)), 2)
        t_mm = Fraction(m.group(2)) * 10
        return R, R - t_mm
    raise AssertionError(f"could not parse annulus R/r from: {text!r}")


def _frustum_R_r_h(text: str) -> tuple[Fraction, Fraction, Fraction]:
    """(R, r, h) in a single common unit, derived from the rendered frustum
    text alone -- the hard tier's h is independently re-derived via
    Pythagoras from the stated slant height, a different route from
    whatever the generator's own triple-drawing did."""
    m = re.search(r"base radius of (\d+) cm, a top radius of (\d+) cm and a height of (\d+) cm", text)
    if m:
        return Fraction(int(m.group(1))), Fraction(int(m.group(2))), Fraction(int(m.group(3)))
    m = re.search(r"base diameter of (\d+) cm, a top diameter of (\d+) cm and a height of (\d+) cm", text)
    if m:
        return Fraction(int(m.group(1)), 2), Fraction(int(m.group(2)), 2), Fraction(int(m.group(3)))
    m = re.search(r"base radius of (\d+) cm, a top radius of (\d+) cm and a slant height of (\d+) cm", text)
    if m:
        R, r, l = Fraction(int(m.group(1))), Fraction(int(m.group(2))), Fraction(int(m.group(3)))
        h_sq = l * l - (R - r) * (R - r)
        import math

        root = math.isqrt(int(h_sq))
        assert root * root == h_sq, f"l^2 - (R-r)^2 not a perfect square: {h_sq}"
        return R, r, Fraction(root)
    raise AssertionError(f"could not parse frustum R/r/h from: {text!r}")


# ---------------------------------------------------------------------------
# Mandatory constructibility/plausibility sweeps (60 seeds x 3 tiers = 180
# checks minimum, per topic).
# ---------------------------------------------------------------------------


def test_annulus_inner_strictly_less_than_outer():
    for tier in TIERS:
        for seed in SEEDS_60:
            q = make(ANNULUS, seed=seed, difficulty=tier)
            R, r = _annulus_R_r(q.text)
            assert r < R, f"{tier} seed={seed}: r={r} not strictly less than R={R} ({q.text!r})"


def test_annulus_inner_radius_within_plausible_band():
    for tier in TIERS:
        for seed in SEEDS_60:
            q = make(ANNULUS, seed=seed, difficulty=tier)
            R, r = _annulus_R_r(q.text)
            assert Fraction(1, 5) * R <= r <= Fraction(9, 10) * R, (
                f"{tier} seed={seed}: r={r} outside [0.2R, 0.9R] for R={R} ({q.text!r})"
            )


def test_frustum_top_radius_strictly_less_than_base():
    for tier in TIERS:
        for seed in SEEDS_60:
            q = make(FRUSTUM, seed=seed, difficulty=tier)
            R, r, h = _frustum_R_r_h(q.text)
            assert r < R, f"{tier} seed={seed}: r={r} not strictly less than R={R} ({q.text!r})"
            assert h > 0, f"{tier} seed={seed}: h={h} not strictly positive ({q.text!r})"


def test_frustum_plausibility_band():
    for tier in TIERS:
        for seed in SEEDS_60:
            q = make(FRUSTUM, seed=seed, difficulty=tier)
            R, r, h = _frustum_R_r_h(q.text)
            assert Fraction(1, 5) * R <= r <= Fraction(9, 10) * R, (
                f"{tier} seed={seed}: r={r} outside [0.2R, 0.9R] for R={R} ({q.text!r})"
            )
            assert h >= Fraction(3, 10) * R, f"{tier} seed={seed}: h={h} below 0.3R for R={R} ({q.text!r})"


# ---------------------------------------------------------------------------
# Determinism, distractors, real grader, answer precision, distinct
# questions, uniform formatting -- required for every build topic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_determinism_across_seeds(topic, tier):
    for seed in SEEDS_40:
        q1 = make(topic, seed=seed, difficulty=tier)
        q2 = make(topic, seed=seed, difficulty=tier)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_within_tolerance(topic, tier):
    for seed in SEEDS_40:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.distractors, f"{topic} {tier} seed={seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        answer_val = Fraction(q.answer_text)
        for d in q.distractors:
            d_val = Fraction(d)
            assert abs(d_val - answer_val) > Fraction(1, 20), (
                f"{topic} {tier} seed={seed}: distractor {d} within 0.05 of answer {q.answer_text}"
            )


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_answer_rejects_distractors_and_wrong(topic, tier):
    grader = GRADERS["number"]
    for seed in SEEDS_40:
        q = make(topic, seed=seed, difficulty=tier)
        result = grader(q, q.answer_text)
        assert result.correct, f"{topic} {tier} seed={seed}: grader rejected answer {q.answer_text}"
        for d in q.distractors:
            r = grader(q, d)
            assert not r.correct, f"{topic} {tier} seed={seed}: grader accepted distractor {d}"
        wrong = f"{float(q.answer) + 1.0:.1f}"
        rw = grader(q, wrong)
        assert not rw.correct, f"{topic} {tier} seed={seed}: grader accepted wrong answer {wrong}"


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_identically(topic, tier):
    one_dp = re.compile(r"^-?\d+\.\d$")
    for seed in SEEDS_40:
        q = make(topic, seed=seed, difficulty=tier)
        assert one_dp.match(q.answer_text), f"{topic} {tier} seed={seed}: {q.answer_text!r} not 1dp"
        for d in q.distractors:
            assert one_dp.match(d), f"{topic} {tier} seed={seed}: distractor {d!r} not 1dp"


def _raw_answer_fraction(topic: str, text: str) -> Fraction:
    """The exact, unrounded Fraction the generator's own formula would
    produce, recomputed from values parsed out of the rendered text alone
    (never the generator's internal locals)."""
    if topic == ANNULUS:
        R, r = _annulus_R_r(text)
        return PI * R**2 - PI * r**2
    R, r, h = _frustum_R_r_h(text)
    return Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_answer_not_on_rounding_boundary(topic, tier):
    """The exact Fraction value the question's own stated quantities produce
    is kept away from the x.x5 half-up tie by more than `Fraction(1, 1000)`
    -- re-derived here from values parsed out of the rendered text (an
    independent recomputation of the generator's own formula, not a read of
    its internal locals), matching the p8d resample-loop pattern's own
    "prove the bound, don't just trust it" discipline. Also confirms
    `answer_text` is the correctly-rounded 1dp display of that exact value."""
    for seed in SEEDS_40:
        q = make(topic, seed=seed, difficulty=tier)
        raw = _raw_answer_fraction(topic, q.text)
        distance = _tenths_tie_distance(raw)
        assert distance > Fraction(1, 1000), (
            f"{topic} {tier} seed={seed}: raw value {raw} sits within 1/1000 of "
            f"the 1dp rounding tie (distance={distance})"
        )
        assert round_half_up(raw, 1) == Fraction(q.answer_text)


@pytest.mark.parametrize("topic", [ANNULUS, FRUSTUM])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS_60}
    assert len(texts) >= 20, f"{topic} {tier}: only {len(texts)} distinct questions over 60 seeds"


# ---------------------------------------------------------------------------
# Independent-check firing on a corrupted value, exercised directly.
# ---------------------------------------------------------------------------


def test_check_equivalent_fires_on_corrupted_annulus_value():
    from mathgen.core import VerificationError, check_equivalent

    R, r = 20, 8
    raw = PI * R**2 - PI * r**2
    corrupted = raw + Fraction(1, 1000)
    with pytest.raises(VerificationError):
        check_equivalent(raw, corrupted, "expected mismatch")


def test_check_distinct_fires_when_candidate_equals_answer():
    from mathgen.core import VerificationError, check_distinct

    R, r = 20, 8
    raw = PI * R**2 - PI * r**2
    with pytest.raises(VerificationError):
        check_distinct(raw, raw, msg="candidate equals the answer")


def test_check_equivalent_fires_on_corrupted_frustum_value():
    from mathgen.core import VerificationError, check_equivalent

    R, r, h = 20, 8, 10
    raw = Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)
    corrupted = raw + Fraction(1, 1000)
    with pytest.raises(VerificationError):
        check_equivalent(raw, corrupted, "expected mismatch")


# ---------------------------------------------------------------------------
# Rounding helper sanity (used throughout the module).
# ---------------------------------------------------------------------------


def test_round_half_up_tenths_basic():
    assert round_half_up(Fraction(1235, 100), 1) == Fraction(124, 10)  # 12.35 -> 12.4
    assert round_half_up(Fraction(1234, 100), 1) == Fraction(123, 10)  # 12.34 -> 12.3


def test_tenths_tie_distance_zero_at_tie():
    assert _tenths_tie_distance(Fraction(1235, 100)) == 0


# ---------------------------------------------------------------------------
# Skill loading.
# ---------------------------------------------------------------------------


class TestSkillsLoad:
    def test_annulus_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "annulus-cross-section-area"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "number"
        assert skill.prerequisites == ["f10-circle-measures"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4
        stages = {tuple(src.stages) for src in skill.sources}
        assert (6,) in stages
        assert (6, "maintenance") in stages

    def test_frustum_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "frustum-volume"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-use-formulas-to-find-the-volume-of-pyramids"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4
        stages = {tuple(src.stages) for src in skill.sources}
        assert (6,) in stages
        assert (6, "maintenance") in stages
