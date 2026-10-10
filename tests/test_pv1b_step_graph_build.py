"""Build test for the step-graph skill (ACMGM047)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1b_step_graph_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-interpret-piece-wise-linear-and-step-graphs-used"
TOPIC = "pv1b_step_graph"
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
    assert s.status == "deferred" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "fraction"
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert s.mental is None


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_pv1b_step_graph


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
        assert q.figure["alt"] == "a step graph made of horizontal segments on a coordinate grid"


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert _grade(q, q.answer_text)
        assert _grade(q, f"{q.answer}.0")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, str(q.answer + 1))
        assert not _grade(q, str(q.answer + 100))


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
    assert len({q.text for q in qs}) >= 20
    assert len({(q.text, q.answer_text, repr(q.figure["segments"])) for q in qs}) >= 20
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
    if tier == "exam":
        real = gen._figure

        def shut_the_dots(*a, **kw):  # every step's right end becomes open: no greatest value
            fig = real(*a, **kw)
            for s in fig["segments"]:
                s["open_end"] = True
            return fig

        monkeypatch.setattr(gen, "_figure", shut_the_dots)
    else:
        real = gen.step_value
        monkeypatch.setattr(gen, "step_value", lambda fig, x: real(fig, x) + 1)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


def _input(q):
    return int(re.findall(r"\d+", q.text)[0])


def _ends(q):
    return {s["to"][0] for s in q.figure["segments"]}


def _starts(q):
    return {s["from"][0] for s in q.figure["segments"]}


def property_medium_input_is_a_shared_step_end(q):
    x = _input(q)
    return x in (_ends(q) & _starts(q))


def property_easy_input_is_strictly_inside_a_step(q):
    x = _input(q)
    return any(s["from"][0] < x < s["to"][0] for s in q.figure["segments"])


def test_medium_input_is_a_shared_step_end_on_every_draw():
    for seed in range(300):
        assert property_medium_input_is_a_shared_step_end(_make("medium", seed)), seed


def test_easy_input_is_strictly_inside_a_step_on_every_draw():
    for seed in range(300):
        assert property_easy_input_is_strictly_inside_a_step(_make("easy", seed)), seed


def test_closed_side_is_drawn_both_ways_on_medium_and_easy_and_hard():
    for tier in ("easy", "medium", "hard"):
        sides = [_make(tier, s).figure["segments"][0]["open_start"] for s in range(200)]
        share = sum(sides) / len(sides)
        assert 0.3 <= share <= 0.7, (tier, share)
    # medium's key is the lower step when the right end is closed, the upper step otherwise
    lower = upper = 0
    for s in range(200):
        q = _make("medium", s)
        closed_right = q.figure["segments"][0]["open_start"]
        x = _input(q)
        lo = next(g["from"][1] for g in q.figure["segments"] if g["to"][0] == x)
        if q.answer == lo:
            lower += 1
            assert closed_right
        else:
            upper += 1
            assert not closed_right
    assert lower > 40 and upper > 40


def test_exam_every_charge_appears_on_exactly_one_step():
    for seed in range(200):
        q = _make("exam", seed)
        heights = [s["from"][1] for s in q.figure["segments"]]
        assert len(set(heights)) == len(heights)
        assert heights == sorted(heights)
        assert all(s["open_start"] and not s["open_end"] for s in q.figure["segments"])


def test_prompt_states_the_convention_without_naming_a_closed_end():
    for tier in TIERS:
        for seed in range(40):
            t = _make(tier, seed).text
            assert "use the value shown by the filled circle" in t
            assert "closed" not in t and "open" not in t and "inclusive" not in t
            assert "**" not in t and "*" not in t
