"""Build test for the end-behaviour skill (ACMMM014), topic `pv1e_power_end_behaviour`."""

from __future__ import annotations

import functools
from collections import Counter

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1e_power_end_behaviour_build as gen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-recognise-features-of-the-graphs-of-y-x-acmmm014"
TOPIC = "pv1e_power_end_behaviour"
TIERS = ["easy", "medium", "hard", "exam"]
LABELS = {gen.INC, gen.DEC, gen.APP}
SEEDS = range(40)
PHRASES = ("tends to infinity", "tends to negative infinity", "tends to 0 from the right", "tends to 0 from the left")
MATHS = (r"{x \to \infty}", r"{x \to -\infty}", r"{x \to 0^{+}}", r"{x \to 0^{-}}")


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed, notes=True):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, notes=notes, **(src.params or {}))


def test_skill_shape():
    s = _skill()
    assert s.status == "active" and s.deferred_reason is None and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
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
        assert "y = x^n" in q.text and r"{y = x^n}" in q.latex
        assert any(p in q.text for p in PHRASES)
        assert any(m in q.latex for m in MATHS)
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


def _ask(q):
    return gen.ask_of(q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_class_balance_at_1000_seeds_with_the_near_zero_questions(tier):
    classes = Counter(_make(tier, seed, notes=False).answer_text for seed in range(1000))
    assert set(classes) == LABELS
    for n in classes.values():
        assert 290 <= n <= 380, classes  # about a third each


@pytest.mark.parametrize("tier", TIERS)
def test_near_zero_questions_are_the_hyperbola_only_and_never_in_easy(tier):
    asks = Counter()
    for seed in range(400):
        q = _make(tier, seed, notes=False)
        ask = _ask(q)
        asks[(gen._exponent(q.figure), ask)] += 1
        if ask in (gen.ZERO_R, gen.ZERO_L):
            assert gen._exponent(q.figure) == -1
            assert q.answer_text == (gen.INC if ask == gen.ZERO_R else gen.DEC)
    near = {k for k in asks if k[1] in (gen.ZERO_R, gen.ZERO_L)}
    assert (near == set()) if tier == "easy" else (near == {(-1, gen.ZERO_R), (-1, gen.ZERO_L)})


@pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
def test_the_hyperbola_has_all_three_answers_and_the_two_sides_are_balanced(tier):
    answers, sides = Counter(), Counter()
    for seed in range(1000):
        q = _make(tier, seed, notes=False)
        if gen._exponent(q.figure) != -1:
            continue
        answers[q.answer_text] += 1
        if _ask(q) in (gen.ZERO_R, gen.ZERO_L):
            sides[_ask(q)] += 1
    assert set(answers) == LABELS
    # each side is drawn with the same chance (binomial noise only), so neither dominates
    assert min(sides.values()) >= 0.3 * sum(sides.values()), sides


def test_a_near_zero_question_draws_the_branch_up_to_the_edge_of_the_window():
    for tier in ("medium", "hard", "exam"):
        for seed in range(400):
            q = _make(tier, seed, notes=False)
            ask = _ask(q)
            if ask not in (gen.ZERO_R, gen.ZERO_L):
                continue
            fig = q.figure
            pts = gen._end_points(fig, 1 if ask == gen.ZERO_R else -1)
            assert abs(pts[0][0]) <= 0.3
            edge = fig["y_range"][1] if ask == gen.ZERO_R else fig["y_range"][0]
            assert abs(pts[0][1] - edge) <= 0.9
            assert abs(pts[0][1]) >= 4


def test_near_zero_options_are_tagged_with_the_right_misconception():
    for tier in ("medium", "hard", "exam"):
        for seed in range(400):
            q = _make(tier, seed)
            if _ask(q) not in (gen.ZERO_R, gen.ZERO_L):
                continue
            for d, note in q.option_notes.items():
                assert note == (
                    "reads-the-far-end-instead-of-the-axis" if d == gen.APP else "reads-the-other-side-of-the-axis"
                )
