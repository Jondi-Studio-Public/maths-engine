"""Build test for the hyperbola-asymptote skill (ACMMM013)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1e_hyperbola_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-frac1x"
TOPIC = "pv1e_hyperbola_asymptotes"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
KEY_RE = re.compile(r"x = (-?\d+), y = (-?\d+)")


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _hk(text):
    m = KEY_RE.fullmatch(text)
    assert m, text
    return int(m.group(1)), int(m.group(2))


def _grade(q, given):
    return GRADERS["pair"](q, given).correct


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.deferred_reason == "visual" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "pair"
    assert s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "swapped",
        "sign-h",
        "sign-k",
        "branch-point",
        "axis-next-to",
        "squares-not-units",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_hyperbola


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_nothing_printed_and_no_asymptote_drawn(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert q.figure is not None
        assert figures.validate(q.figure) == []
        h, k = _hk(q.answer_text)
        figures.assert_no_leak(q.figure, [q.answer_text, f"x = {h}", f"y = {k}", *q.distractors])
        fig = q.figure
        assert fig["alt"] == "a curve with two branches on a coordinate grid"
        assert fig["lines"] == [] and fig["segments"] == [] and fig["points"] == []
        assert len(fig["curves"]) == 1 and len(fig["curves"][0]["pieces"]) == 2
        assert 2 <= sum(len(p) for p in fig["curves"][0]["pieces"]) <= 200
        assert "label" not in fig["curves"][0]


@pytest.mark.parametrize("tier", TIERS)
def test_window_matches_the_tier(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        f = q.figure
        (x0, x1), (y0, y1) = f["x_range"], f["y_range"]
        assert x1 - x0 == 12
        if tier in ("easy", "medium"):
            assert (x0, x1, y0, y1) == (-6, 6, -6, 6) and f["equal_scale"] is True
        elif tier == "hard":
            assert (y0, y1, f["y_grid"], f["x_grid"]) == (-8, 8, 2, 1) and f["equal_scale"] is False
        else:
            assert not (x0 <= 0 <= x1) and not (y0 <= 0 <= y1)  # no axis is drawn


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_values(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        h, k = _hk(q.answer_text)
        assert _grade(q, q.answer_text)
        assert _grade(q, f"y = {k}, x = {h}")  # labelled, so order is free
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        for wrong in (f"x = {h + 1}, y = {k}", f"x = {h}, y = {k + 1}", f"x = {h}", f"y = {k}"):
            assert not _grade(q, wrong), (tier, seed, wrong)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(KEY_RE.fullmatch(v) for v in vals)
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
    assert len({(q.text, q.answer_text) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration(tier):
    counts = {}
    for seed in range(200):
        a = _make(tier, seed).answer_text
        counts[a] = counts.get(a, 0) + 1
    assert max(counts.values()) <= 0.25 * 200


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_the_form_and_the_scale(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert "in the form x = h, y = k" in q.text
        assert r"in the form \(x = h,\ y = k\)" in q.latex
        assert r"\frac{1}{x - h}" in q.latex
        assert "grid square" in q.text or "square is" in q.text or "per grid" in q.text
        assert "**" not in q.text and "*" not in q.text
    q = _make("hard", 0)
    assert "2 units" in q.text or "2s" in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._build_figure

    def shifted(h, k, x_range, y_range, y_grid, equal_scale):  # the drawn curve no longer has the key's h
        return real(h + 1, k, x_range, y_range, y_grid, equal_scale)

    monkeypatch.setattr(gen, "_build_figure", shifted)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
def _far_end_offsets(fig):
    left, right = sorted(fig["curves"][0]["pieces"], key=lambda p: p[0][0])
    k = round(left[0][1])
    return abs(left[0][1] - k), abs(right[-1][1] - k)


@pytest.mark.parametrize("tier", TIERS)
def test_both_far_ends_run_within_a_third_of_the_horizontal_asymptote(tier):
    for seed in range(200):
        a, b = _far_end_offsets(_make(tier, seed).figure)
        assert a <= 1 / 3 + 1e-6 and b <= 1 / 3 + 1e-6, (tier, seed, a, b)


def test_medium_always_names_an_axis_and_the_others_never_do():
    zero = sum(0 in _hk(_make("medium", s).answer_text) for s in range(200))
    assert zero == 200
    assert all(0 not in _hk(_make("easy", s).answer_text) for s in range(200))
    assert all(0 not in _hk(_make("hard", s).answer_text) for s in range(200))


def test_hard_horizontal_asymptote_is_even_so_squares_and_units_differ():
    for s in range(200):
        _, k = _hk(_make("hard", s).answer_text)
        assert k % 2 == 0 and k != 0


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """With h placed near the side edge the far end sits more than a third from y = k: the cap is the constraint."""

    def near_edge(rng, tier):
        return 5, 2, (-6, 6), (-6, 6), 1, True

    monkeypatch.setattr(gen, "_draw", near_edge)
    with pytest.raises(VerificationError):
        gen.pv1e_hyperbola_asymptotes(__import__("random").Random(1), tier="easy")
    # and the figure that would have been drawn really is outside the bound
    fig = gen._build_figure(5, 2, (-6, 6), (-6, 6), 1, True)
    assert max(_far_end_offsets(fig)) > 1 / 3
