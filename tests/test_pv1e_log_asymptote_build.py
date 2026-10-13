"""Build test for the logarithm-asymptote skill (ACMMM156)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1e_log_asymptote_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-recognise-the-qualitative-features-of-the-graph-of-acmmm156"
TOPIC = "pv1e_log_asymptote"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
KEY_RE = re.compile(r"x = (-?\d+)")


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _v(text):
    m = KEY_RE.fullmatch(text)
    assert m, text
    return int(m.group(1))


def _grade(q, given):
    return GRADERS["pair"](q, given).correct


def _piece(q):
    (curve,) = q.figure["curves"]
    (piece,) = curve["pieces"]
    return piece


def test_skill_shape():
    s = _skill()
    assert s.status == "active" and s.deferred_reason is None and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.placement_eligible is False
    assert s.answer == "pair" and s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "x-intercept",
        "sign-flipped",
        "y-axis",
        "off-by-one",
        "off-by-two",
        "squares-not-units",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_log_asymptote


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_nothing_printed_and_no_asymptote_drawn(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["lines"] == [] and fig["segments"] == [] and fig["points"] == []
        assert len(fig["curves"]) == 1 and "label" not in fig["curves"][0]
        assert len(fig["curves"][0]["pieces"]) == 1
        assert 2 <= len(_piece(q)) <= 200
        assert fig["alt"] == "a curve climbing from the bottom of the grid on a coordinate grid"


@pytest.mark.parametrize("tier", TIERS)
def test_window_matches_the_tier(tier):
    for seed in SEEDS:
        f = _make(tier, seed).figure
        (x0, x1), (y0, y1) = f["x_range"], f["y_range"]
        assert (y0, y1) == (-6, 6)
        if tier in ("easy", "medium"):
            assert (x0, x1, f["x_grid"], f["equal_scale"]) == (-6, 6, 1, True)
        elif tier == "hard":
            assert (x0, x1, f["x_grid"], f["y_grid"], f["equal_scale"]) == (-12, 12, 2, 1, False)
        else:
            assert x1 - x0 == 12 and not (x0 <= 0 <= x1) and f["equal_scale"] is True


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_values(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        v = _v(q.answer_text)
        assert _grade(q, q.answer_text) and _grade(q, f"x={v}")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        for wrong in (f"x = {v + 1}", f"x = {v - 1}", f"x = {v + 7}", f"x = {-v - 3}"):
            assert not _grade(q, wrong), (tier, seed, wrong)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals) and all(KEY_RE.fullmatch(v) for v in vals)
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
def test_prompt_names_the_family_without_its_values(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert "log_a(x + c)" in q.text and r"\log_{a}(x + c)" in q.latex
        assert "vertical asymptote" in q.text
        assert "**" not in q.text and "*" not in q.text
    assert (
        "2 units across" in _make("hard", 0).text
        or "2 units per grid square" in _make("hard", 0).text
        or "one grid square is 2 units" in _make("hard", 0).text
    )


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._curve

    def shifted(base, v, x_range):  # the drawn curve runs down along a different gridline
        return real(base, v + 1, x_range)

    monkeypatch.setattr(gen, "_curve", shifted)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
@pytest.mark.parametrize("tier", TIERS)
def test_the_curve_starts_within_a_hundredth_of_the_key_and_runs_down_to_the_bottom(tier):
    for seed in range(200):
        q = _make(tier, seed)
        piece, v = _piece(q), _v(q.answer_text)
        assert 0 < piece[0][0] - v <= 0.011 + 1e-6, (tier, seed, piece[0])
        assert piece[0][1] <= -5.5  # leaves the window through the bottom edge
        assert piece[-1][0] - v >= 3  # and at least three units of it lie to the right


def test_easy_is_a_gridline_off_the_axis_and_medium_sometimes_is_the_axis():
    assert all(_v(_make("easy", s).answer_text) != 0 for s in range(200))
    assert any(_v(_make("medium", s).answer_text) == 0 for s in range(200))
    assert all(_v(_make("hard", s).answer_text) % 2 == 0 and _v(_make("hard", s).answer_text) != 0 for s in range(200))


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """Allowing the asymptote at x = 5 leaves one unit of curve in the window: the rule is the constraint."""

    def near_right_edge(rng, tier):
        return 2, 5, (-6, 6), 1, True

    monkeypatch.setattr(gen, "_draw", near_right_edge)
    with pytest.raises(VerificationError):
        gen.pv1e_log_asymptote(__import__("random").Random(1), tier="easy")
