"""Build test for the dilation skill (ACMMM026, y = c f(x))."""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1c_dilation_factor_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-dilations-and-the-graphs-of-y-cf"
TOPIC = "pv1c_dilation_factor"
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


def _value(q):
    return Fraction(q.answer_text)


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "fraction"
    assert s.mental is None
    assert len(s.sources) == 4
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "reciprocal",
        "sign-flipped",
        "difference",
        "height-of-q",
        "extra-not-total",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_dilation_factor


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
        assert [p["label"] for p in q.figure["points"]] == ["P", "Q"]
        assert len(q.figure["curves"]) == 2 and q.figure["lines"] == []
        for c in q.figure["curves"]:
            assert 2 <= sum(len(p) for p in c["pieces"]) <= 200
        assert "*" not in q.text and "*" not in q.latex
        assert "y = f(x)" in q.text and "y = cf(x)" in q.text and "P" in q.text and "Q" in q.text
        assert r"\(y = f(x)\)" in q.latex and r"\(y = cf(x)\)" in q.latex


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert _grade(q, q.answer_text)
        assert _grade(q, str(float(_value(q))) if _value(q).denominator in (1, 2, 4, 5) else q.answer_text)
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, str(float(_value(q)) + 1))
        assert not _grade(q, str(float(_value(q)) + 100))


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len({Fraction(v) for v in vals}) == len(vals)
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

    def wrong_curve(e, c):  # the Q curve is no longer c times the P curve
        return real(e, c * 2 if c != 1 else c)

    monkeypatch.setattr(gen, "_sample_fn", wrong_curve)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
def _side_pairs(tier, n=400):
    """{(Q is higher than P, c is negative)} over n seeds."""
    out = set()
    for s in range(n):
        q = _make(tier, s)
        p, qq = (pt["at"][1] for pt in q.figure["points"])
        out.add((qq > p, _value(q) < 0))
    return out


def test_exam_sign_is_not_decided_by_which_point_is_higher():
    """P below the axis: Q higher than P goes with c > 0 (c < 1) AND with c < 0 (Q above the
    axis), so 'Q is lower, so negative' and 'Q is higher, so positive' are both wrong."""
    pairs = _side_pairs("exam")
    assert (True, False) in pairs and (True, True) in pairs and (False, False) in pairs


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """With P above the axis (the unconstrained, hard-style draw) Q higher than P always means
    c > 0: the exam constraint is real."""
    monkeypatch.setattr(gen, "_EXAM_SIGN", 1)
    assert (True, True) not in _side_pairs("exam")


def test_signs_and_sides_by_tier():
    def ys(q):
        return [pt["at"][1] for pt in q.figure["points"]]

    for s in range(150):
        e, m, h, x = (_make(t, s) for t in TIERS)
        assert all(y > 0 for y in ys(e)) and _value(e) >= 2
        assert all(y > 0 for y in ys(m)) and 0 < _value(m) != 1 and _value(m).denominator != 1
        p, q = ys(h)
        assert p > 0 > q and _value(h) < 0
        p, q = ys(x)
        assert p < 0 and (q < 0) == (_value(x) > 0)
    neg = sum(_value(_make("exam", s)) < 0 for s in range(200))
    assert 60 <= neg <= 140


def test_prompts_state_the_convention_and_the_scale():
    for tier in TIERS:
        for seed in range(40):
            q = _make(tier, seed)
            assert re.search(r"x = -?\d+", q.text) and "1 unit" in q.text
            assert not re.search(r"c = ", q.text)


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_positive_tiers_offer_no_distractor_the_prompt_rules_out(tier):
    """The prompt says c is positive, so a negative option is not a live misconception there."""
    saw_extra = False
    for seed in range(100):
        q = _make(tier, seed)
        assert all(not d.startswith("-") for d in q.distractors), (tier, seed, q.distractors)
        saw_extra |= len(q.distractors) == 3
    assert saw_extra
