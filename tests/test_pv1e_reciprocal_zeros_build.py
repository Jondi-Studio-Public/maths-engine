"""Build test for the 1/f vertical-asymptote skill (ACMSM099)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1e_reciprocal_zeros_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-the-relationship-between-the-graph-of-y"
TOPIC = "pv1e_reciprocal_zeros"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
LIST_RE = re.compile(r"-?\d+(, -?\d+)*")


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _grade(q, given):
    return GRADERS["list"](q, given).correct


def _values(text):
    return [int(v) for v in text.split(", ")]


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.deferred_reason == "visual" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "list" and s.prerequisites == ["function-notation"]
    assert s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "sign-flipped",
        "y-intercept",
        "gradient",
        "off-by-one",
        "only-one-zero",
        "vertex-x",
        "zeros-and-vertex",
        "y-axis",
        "omits-origin",
        "origin-only",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_zeros


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_nothing_printed_and_1_over_f_never_drawn(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["points"] == [] and fig["segments"] == []
        assert len(fig["lines"]) + len(fig["curves"]) == 1  # one graph: f, never 1/f
        assert fig["alt"] in ("a straight line on a coordinate grid", "a parabola on a coordinate grid")
        assert (fig["x_range"], fig["y_range"], fig["equal_scale"]) == ([-6, 6], [-6, 6], True)
        for c in fig["curves"]:
            assert 2 <= sum(len(p) for p in c["pieces"]) <= 200


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_values(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        vals = _values(q.answer_text)
        assert _grade(q, q.answer_text)
        assert _grade(q, ", ".join(str(v) for v in reversed(vals)))  # a list: order is free
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, ", ".join(str(v + 1) for v in vals))
        assert not _grade(q, ", ".join([*map(str, vals), str(vals[0])]))  # a repeated value is a longer list
        assert not _grade(q, ", ".join(str(v) for v in vals[:-1]) or "99")  # one value short, or a plain wrong value


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(LIST_RE.fullmatch(v) for v in vals)
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
    assert len({(q.text, q.answer_text, repr(q.figure["lines"]), repr(q.figure["curves"])) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration(tier):
    counts = {}
    for seed in range(200):
        a = _make(tier, seed).answer_text
        counts[a] = counts.get(a, 0) + 1
    assert max(counts.values()) <= 0.25 * 200


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_the_conventions_and_never_names_the_zeros(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert "each x-value once" in q.text or "(each x-value once)" in q.text
        assert "grid square" in q.text or "per grid square" in q.text
        assert r"\frac{1}{f(x)}" in q.latex
        assert (r"\(g(x) = \frac{1}{f(x)}\)" in q.latex) == (tier == "exam")
        assert "zero" not in q.text.lower() and "f(x) = 0" not in q.text and "x-intercept" not in q.text
        assert "**" not in q.text and "*" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._build_figure

    def shifted(rng, shape):  # draw a different graph from the one the key describes
        moved = dict(shape)
        for name in ("r", "p", "q"):
            if name in moved and (name != "p" or tier != "exam"):
                moved[name] = moved[name] + 1 if moved[name] < 4 else moved[name] - 1
        if moved == shape:
            moved["r"] = 1
        return real(rng, moved)

    monkeypatch.setattr(gen, "_build_figure", shifted)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
def test_hard_touches_and_easy_exam_line_cross():
    for seed in range(100):
        q = _make("hard", seed)
        (r,) = _values(q.answer_text)
        ys = [y for p in q.figure["curves"][0]["pieces"] for x, y in p if abs(x - r) < 0.2]
        assert min(ys) >= 0 or max(ys) <= 0  # one side of the axis: a touch, not a crossing


def test_origin_is_a_zero_in_every_exam_draw_and_in_no_other_tier():
    assert all(0 in _values(_make("exam", s).answer_text) for s in range(200))
    for tier in ("easy", "medium", "hard"):
        assert all(0 not in _values(_make(tier, s).answer_text) for s in range(200))


def test_exam_mixes_a_single_zero_and_two_zeros_but_mostly_two():
    sizes = [len(_values(_make("exam", s).answer_text)) for s in range(200)]
    assert 0 < sizes.count(1) <= 0.25 * 200 < sizes.count(2)


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """With the line allowed to leave the tier's single-zero rule, exam would be 'zero at the origin' only."""

    def always_line(rng, tier):
        return [0], {"kind": "line", "m": gen.F(1), "r": 0}

    monkeypatch.setattr(gen, "_draw", always_line)
    sizes = [len(_values(_make("exam", s).answer_text)) for s in range(50)]
    assert not (0 < sizes.count(1) <= 0.25 * 50 < sizes.count(2))
