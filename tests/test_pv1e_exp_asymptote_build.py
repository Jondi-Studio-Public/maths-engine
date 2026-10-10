"""Build test for the exponential-asymptote skill (ACMMM065)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1e_exp_asymptote_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-recognise-the-qualitative-features-of-the-graph-of"
TOPIC = "pv1e_exp_asymptote"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
KEY_RE = re.compile(r"y = (-?\d+)")


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _k(text):
    m = KEY_RE.fullmatch(text)
    assert m, text
    return int(m.group(1))


def _grade(q, given):
    return GRADERS["pair"](q, given).correct


def _piece(q):
    (curve,) = q.figure["curves"]
    (piece,) = curve["pieces"]
    return piece


def _level_end(piece):
    left, right = abs(piece[1][1] - piece[0][1]), abs(piece[-1][1] - piece[-2][1])
    return "left" if left < right else "right"


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.deferred_reason == "visual" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "pair" and s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {
        "y-intercept",
        "x-axis",
        "sign-flipped",
        "off-by-one",
        "off-by-two",
        "squares-not-units",
    }


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_exp_asymptote


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
        assert "label" not in fig["curves"][0] and len(fig["curves"]) == 1
        assert len(fig["curves"][0]["pieces"]) == 1
        assert 2 <= len(_piece(q)) <= 200
        assert fig["alt"] == "a curve rising steeply at one end on a coordinate grid"


@pytest.mark.parametrize("tier", TIERS)
def test_window_matches_the_tier(tier):
    for seed in SEEDS:
        f = _make(tier, seed).figure
        (x0, x1), (y0, y1) = f["x_range"], f["y_range"]
        assert (x0, x1) == (-6, 6)
        if tier in ("easy", "medium"):
            assert (y0, y1, f["y_grid"], f["equal_scale"]) == (-6, 6, 1, True)
        elif tier == "hard":
            assert (y0, y1, f["y_grid"], f["x_grid"], f["equal_scale"]) == (-12, 12, 2, 1, False)
        else:
            assert y1 - y0 == 12 and not (y0 <= 0 <= y1) and f["equal_scale"] is True


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_values(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        k = _k(q.answer_text)
        assert _grade(q, q.answer_text) and _grade(q, f"y={k}")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        for wrong in (f"y = {k + 1}", f"y = {k - 1}", f"y = {k + 7}", f"y = {-k - 3}"):
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
        assert "a^x + k" in q.text and r"\(y = a^{x} + k\)" in q.latex
        assert "horizontal asymptote" in q.text
        assert "**" not in q.text and "*" not in q.text
    assert "2 units" in _make("hard", 0).text


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen.cf.pieces

    def lifted(f, x_range, y_range, n=gen.cf.N_SAMPLES):  # the drawn curve no longer levels off at the key
        return real(lambda x: f(x) + 1, x_range, y_range, n)

    monkeypatch.setattr(gen.cf, "pieces", lifted)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
@pytest.mark.parametrize("tier", TIERS)
def test_the_level_end_is_within_a_sixtieth_of_the_key_and_the_other_end_climbs(tier):
    for seed in range(200):
        q = _make(tier, seed)
        piece, k = _piece(q), _k(q.answer_text)
        end = piece[0][1] if _level_end(piece) == "left" else piece[-1][1]
        assert abs(end - k) <= 1 / 60, (tier, seed)
        ys = [y for _, y in piece]
        window = q.figure["y_range"][1] - q.figure["y_range"][0]
        assert max(ys) - min(ys) >= 0.2 * window


def test_easy_levels_on_the_left_medium_on_both_sides_and_k_zero_appears_only_there():
    assert all(_level_end(_piece(_make("easy", s))) == "left" for s in range(200))
    sides = {_level_end(_piece(_make("medium", s))) for s in range(200)}
    assert sides == {"left", "right"}
    assert any(_k(_make("medium", s).answer_text) == 0 for s in range(200))
    for tier in ("easy", "hard"):
        assert all(_k(_make(tier, s).answer_text) != 0 for s in range(200))


def test_hard_levels_on_even_gridlines():
    assert all(_k(_make("hard", s).answer_text) % 2 == 0 for s in range(200))


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """Without the climb rule a=3, k=5 would be drawn: a window of almost flat curve, not a plausible figure."""

    def flat_corner(rng, tier):
        return 3.0, 5, (-6, 6), (-6, 6), 1, True

    pcs = gen.cf.pieces(lambda x: 3.0**x + 5, (-6, 6), (-6, 6))
    assert not gen._shows_a_climb(pcs, (-6, 6))
    monkeypatch.setattr(gen, "_draw", flat_corner)
    with pytest.raises(RuntimeError):
        gen.pv1e_exp_asymptote(__import__("random").Random(1), tier="easy")
