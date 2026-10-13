"""Build tests for PV1B: senior-interpret-the-ratios-frac-f-left-x-h."""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1b_chord_slope_build as gen
from mathgen.grading.typed import grade
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-interpret-the-ratios-frac-f-left-x-h"
TOPIC = "pv1b_chord_slope"
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=None)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = {s.difficulty: s for s in _skill().sources}[tier]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def test_skill_loads_active_with_four_sources():
    s = _skill()
    assert s.status == "active" and s.deferred_reason is None
    assert s.needs_figure is True
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.placement_eligible is False
    assert s.answer == "fraction" and s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert {x.topic for x in s.sources} == {TOPIC}
    assert {m.id for m in s.misconceptions} == {
        "tangent-slope-at-first-point",
        "reciprocal",
        "sign-flipped",
        "rise-only",
        "curve-height-difference-over-wrong-run",
        "average-of-heights",
    }


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in range(20):
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_every_seed_has_a_valid_figure_without_leaks(tier):
    for seed in range(300):
        q = _make(tier, seed)
        assert q.figure is not None
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert q.figure["alt"] == (gen.ALT_TWO if tier == "exam" else gen.ALT_ONE)
        assert len(q.figure["curves"][0]["pieces"]) >= 1
        assert sum(len(p) for p in q.figure["curves"][0]["pieces"]) <= 200
        assert "*" not in q.text


def test_reader_is_registered():
    assert figures.get_reader(TOPIC) is gen.read_chord_slope


@pytest.mark.parametrize("tier", TIERS)
def test_reader_agrees_with_key_on_1000_seeds(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(1000):
        q = _make(tier, seed)
        assert reader(q) == q.answer_text, f"{tier} seed {seed}"


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_key_and_rejects_distractors_and_wrong_values(tier):
    for seed in range(200):
        q = _make(tier, seed)
        assert grade("fraction", q, q.answer_text).correct, (tier, seed)
        key = Fraction(q.answer_text)
        if key.denominator in (1, 2, 4, 5, 8, 10):
            assert grade("fraction", q, str(float(key))).correct, (tier, seed, "decimal")
        for d in q.distractors:
            assert not grade("fraction", q, d).correct, (tier, seed, d)
        for wrong in (key + 1, key - 1, key + Fraction(1, 7)):
            assert not grade("fraction", q, str(wrong)).correct, (tier, seed, wrong)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_distinct_values_and_mostly_two_or_more(tier):
    short = 0
    for seed in range(200):
        q = _make(tier, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        assert len(set(values)) == len(values), (tier, seed, values)
        assert len(q.distractors) <= 3
        if len(q.distractors) < 2:
            short += 1
    assert short / 200 <= 0.15, f"{tier}: {short}/200 draws have fewer than two distractors"


def test_in_generator_check_fires_on_a_corrupted_answer():
    f = gen._make_f(1, 3, 2, Fraction(2), Fraction(1), Fraction(0), 0)  # chord (1,2)-(3,6)
    gen._verify(f, 1, 3, 2, 6, Fraction(2))  # correct
    with pytest.raises(VerificationError):
        gen._verify(f, 1, 3, 2, 6, Fraction(3))  # corrupted key
    with pytest.raises(VerificationError):
        gen._verify(f, 1, 3, 2, 7, Fraction(5, 2))  # through-point off the curve
    with pytest.raises(VerificationError):
        gen._verify(f, 1, 1, 2, 2, Fraction(0))  # zero run


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_twenty_distinct_questions_and_three_shapes_over_60_seeds(tier):
    qs = [_make(tier, s) for s in range(60)]
    assert len({(q.text, str(q.figure["curves"]), str(q.figure["lines"])) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


def _stated(q):
    xs = [int(v) for v in re.findall(r"\b[xa] = (-?\d+)", q.text)]
    hs = re.findall(r"\bh = (-?\d+)", q.text)
    x1 = xs[0]
    return x1, (x1 + int(hs[0]) if hs else xs[1])


def _line_slope(q):
    p, qq = q.figure["lines"][0]["through"]
    return Fraction(qq[1] - p[1], qq[0] - p[0])


def _turns_between(q):
    """The curve heads the other way (than the chord) at one of the two ends."""
    pts = {}
    for piece in q.figure["curves"][0]["pieces"]:
        for x, y in piece:
            pts[x] = y
    x1, x2 = sorted(_stated(q))
    m = _line_slope(q)
    ends = []
    for x in (x1, x2):
        fwd = (pts[round(x + 0.1, 6)] - pts[x]) / 0.1 if round(x + 0.1, 6) in pts else None
        bwd = (pts[x] - pts[round(x - 0.1, 6)]) / 0.1 if round(x - 0.1, 6) in pts else None
        ends.append(fwd if fwd is not None else bwd)
    return any(e * float(m) < 0 for e in ends if e is not None)


def test_medium_curve_turns_between_the_points():
    for seed in range(200):
        assert _turns_between(_make("medium", seed)), seed


def test_turning_property_is_not_vacuous():
    """Easy's curve keeps rising across its interval, so the property is a real constraint."""
    assert not any(_turns_between(_make("easy", s)) for s in range(60))


def test_hard_h_is_negative_on_every_draw():
    for seed in range(300):
        q = _make("hard", seed)
        x1, x2 = _stated(q)
        assert x2 < x1, (seed, q.text)


def test_negative_h_property_is_not_vacuous():
    assert not all(_stated(_make("easy", s))[1] < _stated(_make("easy", s))[0] for s in range(60))


def test_exam_tangent_slope_is_a_distinct_distractor_value():
    for seed in range(200):
        q = _make("exam", seed)
        key = Fraction(q.answer_text)
        assert len(q.figure["lines"]) == 2
        slopes = [
            Fraction(l["through"][1][1] - l["through"][0][1], l["through"][1][0] - l["through"][0][0])
            for l in q.figure["lines"]
        ]
        assert slopes.count(key) == 1 and len(set(slopes)) == 2
        tangent = next(s for s in slopes if s != key)
        assert tangent != key
