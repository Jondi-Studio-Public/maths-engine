"""Build test for the translated-point skill (ACMMM025, y = f(x) + a)."""

from __future__ import annotations

import functools
import re
from fractions import Fraction as F

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1c_translated_point_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-translations-and-the-graphs-of-y-f"
TOPIC = "pv1c_translated_point"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _grade(q, given):
    return GRADERS["pair"](q, given).correct


def _xy(q):
    return q.answer["x"], q.answer["y"]


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "pair"
    assert s.mental is None
    assert len(s.sources) == 4
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {"opposite-direction", "horizontal-by-a", "horizontal-slide", "gap-as-y"}


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_translated_point


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_and_no_leak(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert q.figure is not None
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert q.figure["alt"] == "two curves on a coordinate grid"
        assert len(q.figure["curves"]) == 2 and len(q.figure["points"]) == 1
        assert q.figure["points"][0]["label"] == "P"
        assert q.figure["lines"] == [] and q.figure["segments"] == []
        for c in q.figure["curves"]:
            assert 2 <= sum(len(p) for p in c["pieces"]) <= 200
        assert "**" not in q.text and "*" not in q.text
        assert "same x-coordinate" in q.text or "vertical line through" in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        x, y = _xy(q)
        assert _grade(q, q.answer_text)
        assert _grade(q, f"{x}, {y}")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, f"({x + 1}, {y})")
        assert not _grade(q, f"({x}, {y + 1})")
        assert not _grade(q, f"({y}, {x})") or x == y


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(re.fullmatch(r"\(-?\d+, -?\d+\)", v) for v in vals)
        counts.append(len(q.distractors))
    assert sum(c >= 2 for c in counts) >= 0.85 * len(counts)
    assert sum(c == 3 for c in counts) >= 0.8 * len(counts)


@pytest.mark.parametrize("tier", TIERS)
def test_reader_agrees_with_key(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(1000):
        q = _make(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    qs = [_make(tier, seed) for seed in range(60)]
    assert len({q.text for q in qs}) >= 10
    assert len({(q.text, q.answer_text, repr(q.figure["curves"])) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration(tier):
    counts = {}
    for seed in range(200):
        a = _make(tier, seed).answer_text
        counts[a] = counts.get(a, 0) + 1
    assert max(counts.values()) <= 0.25 * 200


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._sample_fn

    def wrong_curve(co, p, q, shift):  # the drawn gap is no longer the key's a
        return real(co, p, q, 2 * shift)

    monkeypatch.setattr(gen, "_sample_fn", wrong_curve)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


def test_prompts_name_the_family_and_never_state_a():
    for tier in TIERS:
        for seed in range(40):
            q = _make(tier, seed)
            assert "y = f(x) + a" in q.text
            assert "1 unit" in q.text or "one unit" in q.text
            assert r"\(y = f(x) + a\)" in q.latex and r"\(P\)" in q.latex
            assert "translat" not in q.text.lower()
            assert not re.search(r"a = ", q.text)


# ------------------------------------------------ the properties checked here
def _maps(q):
    return [{x: y for piece in c["pieces"] for x, y in piece} for c in q.figure["curves"]]


def _split(q):
    px, py = q.figure["points"][0]["at"]
    m = _maps(q)
    i = next(i for i, mp in enumerate(m) if mp.get(px) == py)
    return px, py, m[i], m[1 - i]


def _has_trap(q) -> bool:
    px, py, f, g = _split(q)
    return any(g.get(px + h) == py for h in (-4, -3, -2, -1, 1, 2, 3, 4))


def _line_with_integer_offset(q) -> bool:
    px, py, f, g = _split(q)
    ks = [k for k in (-2, -1, 1, 2) if (px + k) in f]
    m = (f[px + 1] - f[px]) if (px + 1) in f else (f[px] - f[px - 1])
    if any(abs((f[px + k] - f[px]) - m * k) > 1e-4 for k in ks) or m == 0:
        return False
    a = g[px] - py
    b = F(a).limit_denominator(100) / F(m).limit_denominator(100)
    return b.denominator == 1 and b != 0 and b != a


def test_hard_always_has_a_same_row_crossing():
    for s in range(300):
        assert _has_trap(_make("hard", s))


def test_exam_always_has_a_whole_horizontal_offset():
    for s in range(300):
        assert _line_with_integer_offset(_make("exam", s))


def test_hard_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """An independent steep draw (a not tied to a crossing) has no same-row grid crossing."""
    monkeypatch.setattr(gen, "_verify", lambda *a, **k: None)  # the in-generator check would refuse it
    monkeypatch.setattr(gen, "_draw", lambda rng, variant: ((F(4), F(1), F(0)), 1, (0,)))
    assert not all(_has_trap(_make("hard", s)) for s in range(30))


def test_exam_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """A line whose vertical shift is not a whole horizontal shift (b = 1/2) fails."""
    monkeypatch.setattr(gen, "_verify", lambda *a, **k: None)
    monkeypatch.setattr(gen, "_draw", lambda rng, variant: ((F(2), F(0), F(0)), 1, (0,)))
    assert not all(_line_with_integer_offset(_make("exam", s)) for s in range(30))


def test_direction_by_tier():
    for s in range(200):
        px, py, f, g = _split(_make("easy", s))
        assert g[px] > py
    signs = {(_split(q)[3][_split(q)[0]] > _split(q)[1]) for q in (_make("medium", s) for s in range(100))}
    assert signs == {True, False}


def test_steepness_and_gentleness_by_tier():
    for tier, ok in (
        ("easy", lambda m: abs(m) <= 1 + 1e-3),
        ("medium", lambda m: abs(m) <= 1 + 1e-3),
        ("hard", lambda m: round(abs(m)) in (2, 3, 4)),
    ):
        for s in range(100):
            px, py, f, g = _split(_make(tier, s))
            hi_, lo_ = round(px + 1 / 12, 6), round(px - 1 / 12, 6)
            assert hi_ in f and lo_ in f
            slope = (f[hi_] - f[lo_]) * 6
            assert ok(slope), (tier, s, slope)
