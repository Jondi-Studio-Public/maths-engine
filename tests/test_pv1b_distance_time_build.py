"""Build tests for PV1B: senior-interpret-distance-versus-time-graphs."""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.grading.typed import grade
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1b_distance_time_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-interpret-distance-versus-time-graphs"
TOPIC = "pv1b_distance_time"
TIERS = ["easy", "medium", "hard", "exam"]


@functools.lru_cache(maxsize=None)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = {s.difficulty: s for s in _skill().sources}[tier]
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def test_skill_loads_deferred_with_four_sources():
    s = _skill()
    assert s.status == "deferred" and s.deferred_reason == "visual"
    assert s.needs_figure is True
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "fraction" and s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert {x.topic for x in s.sources} == {TOPIC}
    assert {m.id for m in s.misconceptions} == {
        "rise-only",
        "inverted-slope",
        "neighbouring-segment",
        "average-over-whole-graph",
        "sign-flipped",
        "unconverted-unit",
        "height-instead-of-speed",
        "segment-duration",
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
        assert q.figure["alt"] == gen.ALT
        assert "*" not in q.text and "**" not in q.text


def test_reader_is_registered():
    assert figures.get_reader(TOPIC) is gen.read_distance_time


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
    ts, ds = [0, 3, 6], [0, 6, 6]
    gen._verify(ts, ds, 1, 2, 1, Fraction(2))  # correct: rise 6 over run 3
    with pytest.raises(VerificationError):
        gen._verify(ts, ds, 1, 2, 1, Fraction(3))  # corrupted key
    with pytest.raises(VerificationError):
        gen._verify(ts, ds, 2, 4, 1, Fraction(0))  # interval spans two segments
    with pytest.raises(VerificationError):
        gen._verify([0, 15, 30], [0, 2, 5], 5, 10, 60, Fraction(2, 15))  # unit not converted


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_twenty_distinct_questions_and_three_shapes_over_60_seeds(tier):
    qs = [_make(tier, s) for s in range(60)]
    assert len({(q.text, str(q.figure["segments"])) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


def _interval_strictly_inside(q):
    a, b = (int(x) for x in re.findall(r"\d+", q.text)[:2])
    return any(s["from"][0] < a < b < s["to"][0] for s in q.figure["segments"])


@pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
def test_interval_ends_are_never_vertices(tier):
    for seed in range(300):
        assert _interval_strictly_inside(_make(tier, seed)), (tier, seed)


def test_interval_property_is_not_vacuous():
    """The unconstrained draw (easy's whole-segment interval) violates the
    property, so the constrained tiers' pass above is a real constraint."""
    assert not any(_interval_strictly_inside(_make("easy", s)) for s in range(60))


def test_hard_states_the_sign_convention_on_every_draw():
    for seed in range(300):
        q = _make("hard", seed)
        assert "positive velocity" in q.text and "away from home" in q.text.lower()


def test_exam_converts_a_km_per_minute_slope():
    for seed in range(100):
        q = _make("exam", seed)
        seg = next(s for s in q.figure["segments"] if s["from"][0] <= int(re.findall(r"\d+", q.text)[0]) < s["to"][0])
        raw = Fraction(seg["to"][1] - seg["from"][1], seg["to"][0] - seg["from"][0])
        assert Fraction(q.answer_text) == raw * 60
        assert str(raw) not in [q.answer_text]
