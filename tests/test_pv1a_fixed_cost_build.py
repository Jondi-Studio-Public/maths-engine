"""Tests `pv1a_fixed_cost` (ACMGM042): the fixed cost read off a contextual line."""

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "pv1a_fixed_cost"
SKILL_ID = "senior-interpret-in-context-the-slope-and-intercept-of"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(skill, difficulty, seed):
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def test_skill_yaml_shape(skill):
    assert skill.status == "active"
    assert skill.deferred_reason is None
    assert skill.needs_figure is True
    assert list(skill.capability_gates) == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.mental is None
    assert {s.difficulty for s in skill.sources} == set(DIFFICULTIES)
    exam = next(s for s in skill.sources if s.difficulty == "exam")
    assert list(exam.stages) == ["exam"]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_figure_valid_no_leak_and_plausible(skill, difficulty):
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["alt"] == "a straight line on a coordinate grid"
        assert fig["equal_scale"] is False
        (x1, y1), (x2, y2) = fig["lines"][0]["through"]
        xg, yg = fig["x_grid"], fig["y_grid"]
        xr, yr = fig["x_range"], fig["y_range"]
        assert (xr[1] - xr[0]) / xg + 1 <= 10 and (yr[1] - yr[0]) / yg + 1 <= 10
        for x, y in ((x1, y1), (x2, y2)):
            assert xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]
            assert x % xg == 0 and y % yg == 0  # lattice points
        assert abs(x2 - x1) / xg >= 2  # two grid columns apart
        assert x1 >= 1  # the intercept itself is never a through point


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tier_decisions(skill, difficulty):
    for seed in SEEDS:
        fig = _make(skill, difficulty, seed).figure
        x0, y0 = fig["x_range"][0], fig["y_range"][0]
        if difficulty == "easy":
            assert x0 == 0 and y0 == 0
        elif difficulty == "medium":
            assert x0 == 0 and y0 > 0
        else:
            assert x0 > 0  # y-axis not drawn
        if difficulty == "exam":
            assert fig["x_grid"] > 1 and fig["y_grid"] > 1
            (xa, ya), (xb, yb) = fig["lines"][0]["through"]
            assert ((yb - ya) / (xb - xa)) % 1 != 0  # fractional rate per real unit


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_reader_grader_and_distractors(skill, difficulty):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        assert reader(q) == q.answer_text
        assert isinstance(q.answer, int) and q.answer >= 1
        assert grade_number(q, f"${q.answer_text}").correct
        assert not grade_number(q, str(q.answer + 1)).correct
        assert len(q.distractors) >= 2
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        for d in q.distractors:
            assert d.isdigit() and int(d) > 0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompt_never_states_answer_or_rate_and_shapes(difficulty):
    texts, shapes = set(), set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not any(ch.isdigit() for ch in q.text), q.text
        assert "graph" in q.text
        assert q.text in q.latex and "$" not in q.text
        texts.add(q.text)
        shapes.add(prompt_shape(q))
    assert len(texts) >= 20, len(texts)
    assert len(shapes) >= 3


def test_exam_names_the_fee_and_hard_adds_a_shape():
    exam = {make(TOPIC, seed=s, difficulty="exam").text for s in range(60)}
    assert all(
        any(
            f in t
            for f in (
                "hire fee",
                "joining fee",
                "call-out fee",
                "set-up cost",
                "connection fee",
                "flagfall",
                "delivery fee",
                "booking fee",
            )
        )
        for t in exam
    )
    easy = {prompt_shape(make(TOPIC, seed=s, difficulty="easy")) for s in range(60)}
    hard = {prompt_shape(make(TOPIC, seed=s, difficulty="hard")) for s in range(60)}
    exam_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="exam")) for s in range(60)}
    assert hard - easy and exam_shapes - easy - hard


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")
