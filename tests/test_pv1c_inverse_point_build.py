"""Build test for the inverse-point skill (ACMSM096, reflection of a function and its inverse)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1c_inverse_point_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-the-reflection-property-of-the-graph-of"
TOPIC = "pv1c_inverse_point"
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
    assert s.status == "active" and s.deferred_reason is None and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.placement_eligible is False
    assert s.answer == "pair"
    assert s.mental is None
    assert len(s.sources) == 4
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "unswapped",
        "negated-swap",
        "sign-flipped-second",
        "sign-flipped-first",
        "reflected-in-axis",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_inverse_point


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
        assert q.figure["segments"] == []
        assert len(q.figure["lines"]) == (1 if tier == "easy" else 0)
        for c in q.figure["curves"]:
            assert 2 <= sum(len(p) for p in c["pieces"]) <= 200
        assert "**" not in q.text and "*" not in q.text
        assert "y = f(x)" in q.text and "y = f⁻¹(x)" in q.text
        assert r"{y = f^{-1}(x)}" in q.latex and r"{P}" in q.latex
        assert ("y = x" in q.text) == (tier == "easy")


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


def test_every_misconception_is_reachable():
    seen = set()
    for tier in TIERS:
        for seed in range(100):
            q = _make(tier, seed)
            p, qq = q.figure["points"][0]["at"]
            for name, v in (
                ("unswapped", (p, qq)),
                ("negated-swap", (-qq, -p)),
                ("sign-flipped-second", (qq, -p)),
                ("sign-flipped-first", (-qq, p)),
                ("reflected-in-axis", (p, -qq)),
            ):
                if f"({v[0]}, {v[1]})" in q.distractors:
                    seen.add(name)
    assert seen == {m.id for m in _skill().misconceptions}


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._invert

    def wrong_inverse(fn, increasing):
        inv = real(fn, increasing)
        return lambda y: None if inv(y) is None else inv(y) + 1  # the drawn inverse is shifted

    monkeypatch.setattr(gen, "_invert", wrong_inverse)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
def _maps(q):
    return [{x: y for piece in c["pieces"] for x, y in piece} for c in q.figure["curves"]]


def _decreasing(m) -> bool:
    ys = [m[x] for x in sorted(m)]
    return all(b < a for a, b in zip(ys, ys[1:]))


def _fixed_on_both(q) -> bool:
    px, py = q.figure["points"][0]["at"]
    return px == py and all(abs(m.get(px, 1e9) - py) < 1e-6 for m in _maps(q))


def _hard_property(q) -> bool:
    px, py = q.figure["points"][0]["at"]
    return all(_decreasing(m) for m in _maps(q)) and (px < 0 or py < 0)


def test_hard_always_decreasing_with_a_negative_coordinate():
    for s in range(300):
        assert _hard_property(_make("hard", s))


def _crossing_pair(q) -> bool:
    """P is on both curves, NOT on y = x, so its image (b, a) is a different point (also on both)."""
    px, py = q.figure["points"][0]["at"]
    maps = _maps(q)
    return (
        px != py
        and all(abs(m.get(px, 1e9) - py) < 1e-6 for m in maps)
        and all(abs(m.get(py, 1e9) - px) < 1e-6 for m in maps)
        and all(_decreasing(m) for m in maps)
    )


def test_exam_point_is_a_crossing_off_the_mirror_line():
    for s in range(300):
        q = _make("exam", s)
        assert _crossing_pair(q)
        px, py = q.figure["points"][0]["at"]
        assert f"({px}, {py})" in q.distractors  # copying P is a live, distinct distractor
        assert q.answer_text == f"({py}, {px})" != f"({px}, {py})"


def test_hard_figures_are_distinct_up_to_curve_order():
    figs = []
    for s in range(30):  # pool is ~700 figures: a repeat inside 30 seeds is excluded, not left to chance
        q = _make("hard", s)
        figs.append((tuple(q.figure["points"][0]["at"]), frozenset(repr(c) for c in q.figure["curves"])))
    assert len(set(figs)) == len(figs)


def test_easy_p_not_on_mirror_line_and_positive():
    for s in range(300):
        q = _make("easy", s)
        px, py = q.figure["points"][0]["at"]
        assert px != py and px > 0 and py > 0
        assert q.figure["lines"][0]["through"] == [[0, 0], [1, 1]]


def test_hard_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """An unconstrained draw (an increasing f through a positive point) fails the hard property."""
    monkeypatch.setattr(gen, "_verify", lambda *a, **k: None)  # the in-generator check would refuse it
    monkeypatch.setattr(gen, "_draw_point", lambda rng, variant: (3, 2))
    monkeypatch.setattr(gen, "_build_f", lambda rng, variant, p, q: (gen._line(p, q, gen.F(2)), True, "line"))
    assert not all(_hard_property(_make("hard", s)) for s in range(10))


def test_exam_property_is_red_against_the_unconstrained_draw(monkeypatch):
    monkeypatch.setattr(gen, "_verify", lambda *a, **k: None)
    monkeypatch.setattr(gen, "_draw_point", lambda rng, variant: (3, 2))
    monkeypatch.setattr(gen, "_build_f", lambda rng, variant, p, q: (gen._line(p, q, gen.F(2)), True, "line"))
    assert not all(_crossing_pair(_make("exam", s)) for s in range(10))


def test_exam_curves_are_labelled_and_the_prompt_says_which_is_f():
    for s in range(60):
        q = _make("exam", s)
        labels = [c.get("label") for c in q.figure["curves"]]
        assert sorted(labels) == ["f", "f⁻¹"]  # distinct, both present
        assert "curve labelled f is y = f(x)" in q.text
        assert "curve labelled f⁻¹ is y = f⁻¹(x)" in q.text
        # the label 'f' really sits on the curve the prompt calls y = f(x): the other is its mirror image
        by_label = {c["label"]: {x: y for pc in c["pieces"] for x, y in pc} for c in q.figure["curves"]}
        px, py = q.figure["points"][0]["at"]
        assert abs(by_label["f"][px] - py) < 1e-6 and abs(by_label["f⁻¹"][py] - px) < 1e-6
        assert q.answer_text == f"({q.figure['points'][0]['at'][1]}, {q.figure['points'][0]['at'][0]})"


def test_other_tiers_keep_unlabelled_curves():
    for tier in ("easy", "medium", "hard"):
        assert all("label" not in c for c in _make(tier, 3).figure["curves"])
