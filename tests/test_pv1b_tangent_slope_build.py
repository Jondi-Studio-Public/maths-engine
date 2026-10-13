"""Build test for the tangent-slope skill (ACMMM085)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1b_tangent_slope_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-interpret-the-derivative-as-the-slope-or-gradient"
TOPIC = "pv1b_tangent_slope"
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
    return GRADERS["fraction"](q, given).correct


def test_skill_shape():
    s = _skill()
    assert s.status == "active" and s.deferred_reason is None and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.placement_eligible is False
    assert s.answer == "fraction"
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert len(s.sources) == 4
    assert s.mental is None
    assert {m.id for m in s.misconceptions} == {
        "reciprocal-slope",
        "sign-flipped",
        "f-of-a",
        "chord-slope",
        "x-coordinate-a",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_tangent_slope


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
        assert "tangent" not in q.figure["alt"] and "f(" not in q.figure["alt"]
        assert len(q.figure["lines"]) == (2 if tier == "exam" else 1)
        assert 2 <= sum(len(p) for c in q.figure["curves"] for p in c["pieces"]) <= 200
        assert not any(p.get("label") for p in q.figure["points"])


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert _grade(q, q.answer_text)
        if _value(q).denominator == 1:
            assert _grade(q, f"{int(_value(q))}.0")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, str(float(_value(q)) + 1))
        assert not _grade(q, str(float(_value(q)) + 100))


def _value(q):
    from fractions import Fraction

    return Fraction(q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(re.fullmatch(r"-?\d+(/\d+)?", v) for v in vals)
        counts.append(len(q.distractors))
    assert sum(c >= 2 for c in counts) >= 0.85 * len(counts)


@pytest.mark.parametrize("tier", TIERS)
def test_reader_agrees_with_key(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(1000):
        q = _make(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    qs = [_make(tier, seed) for seed in range(60)]
    assert len({q.text for q in qs}) >= 3
    assert len({(q.text, q.answer_text, repr(q.figure["curves"]), repr(q.figure["lines"])) for q in qs}) >= 20
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
    real = gen._tangent_points

    def off_the_tangent(*a, **kw):  # the drawn tangent no longer has the key's slope
        pts = real(*a, **kw)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts
        return ([x1, y1], [x2, y2 - 1 if y2 > -6 else y2 + 1])

    monkeypatch.setattr(gen, "_tangent_points", off_the_tangent)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the property checked here
def zero_share(tier, n=200):
    return sum(_make(tier, s).answer_text == "0" for s in range(n)) / n


def test_medium_horizontal_tangent_is_present_but_at_most_a_fifth():
    share = zero_share("medium")
    assert 0 < share <= 0.20, share


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """With the zero weight lifted the share breaks the 20% cap: the cap is the constraint."""
    monkeypatch.setitem(gen._ZERO_WEIGHT, "sign_and_zero", 0.9)
    assert zero_share("medium") > 0.20


def test_exam_secant_is_visibly_not_the_tangent_and_negatives_appear():
    neg = 0
    for seed in range(200):
        q = _make("exam", seed)
        g = sorted(float(gen._line_slope(ln)) for ln in q.figure["lines"])
        assert g[0] != g[1]
        neg += q.answer_text.startswith("-")
    assert neg > 0
    medium_neg = sum(_make("medium", s).answer_text.startswith("-") for s in range(200))
    assert medium_neg > 0
    assert all(not _make("easy", s).answer_text.startswith("-") for s in range(200))


def test_prompts_state_the_scale_and_no_python_notation():
    for tier in TIERS:
        for seed in range(40):
            q = _make(tier, seed)
            assert "1 unit" in q.text and "grid square" in q.text or "grid" in q.text
            assert "**" not in q.text and "*" not in q.text
