"""Build test for the end-behaviour skill (ACMMM014), topic `pv1e_power_end_behaviour`."""

from __future__ import annotations

import functools
from collections import Counter

import pytest

import mathgen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1e_power_end_behaviour_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-x-acmmm014"
TOPIC = "pv1e_power_end_behaviour"
TIERS = ["easy", "medium", "hard", "exam"]
LABELS = {gen.INC, gen.DEC, gen.APP}
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed, notes=True):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, notes=notes, **(src.params or {}))


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.deferred_reason == "visual" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "mcq"
    assert s.mental is None
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    ids = {m.id for m in s.misconceptions}
    assert ids == set(gen._NOTE_ID.values())
    for t in TIERS:
        for seed in range(60):
            assert set(_make(t, seed).option_notes.values()) <= ids


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_end_behaviour


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_nothing_printed_and_nothing_drawn_but_the_curve(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["alt"] == "a curve on a coordinate grid"
        assert fig["lines"] == [] and fig["segments"] == [] and fig["points"] == []
        assert len(fig["curves"]) == 1 and "label" not in fig["curves"][0]
        assert 2 <= sum(len(p) for p in fig["curves"][0]["pieces"]) <= 200


@pytest.mark.parametrize("tier", TIERS)
def test_reader_grader_and_options(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(300):
        q = _make(tier, seed)
        options = mcq_options(q)
        assert reader(q) == q.answer_text
        assert set(options) == LABELS and len(options) == 3
        assert len(q.distractors) == 2 and q.answer_text not in q.distractors
        for i, opt in enumerate(options):
            assert grade_mcq(q, str(i)).correct is (opt == q.answer_text)
        assert set(q.option_notes) == set(q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_class_and_key_position_balance_at_400_seeds(tier):
    classes, position = Counter(), Counter()
    for seed in range(400):
        q = _make(tier, seed, notes=False)
        classes[q.answer_text] += 1
        position[mcq_options(q).index(q.answer_text)] += 1
    assert set(classes) == LABELS and set(position) == {0, 1, 2}
    for n in [*classes.values(), *position.values()]:
        assert 100 <= n <= 170, (classes, position)  # about a third each


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    qs = [_make(tier, seed, notes=False) for seed in range(60)]
    assert len({q.text for q in qs}) >= 20
    assert len({(q.text, q.answer_text) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3
    figs = {str(q.figure["curves"]) for q in qs}
    assert len(figs) >= 10  # the learner sees different pictures, not three


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_names_the_family_and_the_end_but_never_the_answer(tier):
    both = set()
    for seed in range(60):
        q = _make(tier, seed)
        assert "y = x^n" in q.text and r"\(y = x^n\)" in q.latex
        assert "tends to infinity" in q.text or "tends to negative infinity" in q.text
        assert r"\(x \to \infty\)" in q.latex or r"\(x \to -\infty\)" in q.latex
        for lab in LABELS:
            assert lab not in q.text and lab not in q.latex
        assert "without bound" not in q.text and "fixed value" not in q.text
        assert "**" not in q.text and "$" not in q.text
        both.add("negative infinity" in q.text)
    assert both == {True, False}


def test_hard_and_exam_add_a_shape():
    def shapes(t):
        return {prompt_shape(_make(t, s, notes=False)) for s in range(60)}

    easy, medium, hard, exam = (shapes(t) for t in TIERS)
    assert hard - easy - medium
    assert exam - easy - medium - hard


def test_n_half_is_only_ever_asked_at_the_right_end():
    for tier in TIERS:
        for seed in range(300):
            q = _make(tier, seed, notes=False)
            n = gen._exponent(q.figure)
            if n == gen._HALF:
                assert "negative infinity" not in q.text


def test_far_end_of_every_approach_question_is_level():
    for tier in TIERS:
        for seed in range(300):
            q = _make(tier, seed, notes=False)
            if q.answer_text != gen.APP:
                continue
            end = -1 if "negative infinity" in q.text else 1
            outer = gen._end_points(q.figure, end)[-1]
            assert abs(outer[1]) <= 0.21 and abs(outer[0]) >= 4.5


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._build_figure

    def other_curve(n, x_range, y_range):  # the drawn curve is not the key's n
        return real(1 if n != 1 else 2, x_range, y_range)

    monkeypatch.setattr(gen, "_build_figure", other_curve)
    with pytest.raises(VerificationError):
        for seed in range(20):
            _make(tier, seed)


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        mathgen.make(TOPIC, seed=1, tier="bogus")
