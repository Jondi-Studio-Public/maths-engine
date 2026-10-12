"""Sign of the gradient off a drawn line (ACMMM003), topic pv1a_gradient_sign."""

from collections import Counter
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "pv1a_gradient_sign"
SKILL_ID = "senior-recognise-features-of-the-graph-of-y-mx"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
LABELS = {"positive", "negative", "zero"}
SEEDS = range(40)


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(skill, difficulty, seed):
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, notes=True, **(src.params or {}))


def _mc(fig):
    (x1, y1), (x2, y2) = fig["lines"][0]["through"]
    m = Fraction(y2 - y1, x2 - x1)
    return m, Fraction(y1) - m * x1


def test_skill_yaml_shape(skill):
    assert skill.status == "active"
    assert skill.deferred_reason is None
    assert skill.needs_figure is True
    assert list(skill.capability_gates) == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.mental is None
    assert skill.answer == "mcq"
    assert {s.difficulty for s in skill.sources} == set(DIFFICULTIES)
    exam = next(s for s in skill.sources if s.difficulty == "exam")
    assert list(exam.stages) == ["exam"]
    ids = {m.id for m in skill.misconceptions}
    for d in DIFFICULTIES:
        for seed in range(60):
            assert set(_make(skill, d, seed).option_notes.values()) <= ids


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_figure_valid_no_leak_and_plausible(skill, difficulty):
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["alt"] == "a straight line on a coordinate grid"
        assert fig["equal_scale"] is (difficulty != "exam")
        (x1, y1), (x2, y2) = fig["lines"][0]["through"]
        xg, yg = fig["x_grid"], fig["y_grid"]
        xr, yr = fig["x_range"], fig["y_range"]
        assert (xr[1] - xr[0]) / xg + 1 <= 10 and (yr[1] - yr[0]) / yg + 1 <= 10
        for x, y in ((x1, y1), (x2, y2)):
            assert xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]
            assert x % xg == 0 and y % yg == 0
        assert abs(x2 - x1) / xg >= 2


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tier_decisions(skill, difficulty):
    for seed in range(80):
        q = _make(skill, difficulty, seed)
        fig = q.figure
        m, c = _mc(fig)
        x_lo, x_hi = fig["x_range"]
        y_lo, y_hi = fig["y_range"]
        y_axis_drawn = x_lo <= 0 <= x_hi
        if m == 0:
            assert c != 0  # never the x-axis itself
        if difficulty == "easy":
            assert (x_lo, x_hi, y_lo, y_hi) == (-4, 4, -4, 4)
            assert m == 0 or abs(m) in (1, 2)
        elif difficulty == "medium":
            assert y_axis_drawn and y_lo <= c <= y_hi
            if m > 0:
                assert c < 0  # intercept contradicts the gradient
            if m < 0:
                assert c > 0
        elif difficulty == "hard":
            assert not y_axis_drawn
        else:
            assert fig["x_grid"] != fig["y_grid"] and fig["equal_scale"] is False


def test_exam_picture_steepness_is_independent_of_sign():
    seen = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        (x1, y1), (x2, y2) = q.figure["lines"][0]["through"]
        gx, gy = q.figure["x_grid"], q.figure["y_grid"]
        slope_in_squares = Fraction(y2 - y1, gy) / Fraction(x2 - x1, gx)
        if slope_in_squares != 0:
            seen.add((slope_in_squares > 0, abs(slope_in_squares) < 1))
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_reader_grader_and_options(skill, difficulty):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        options = mcq_options(q)
        assert reader(q) == q.answer_text
        assert grade_mcq(q, str(options.index(reader(q)))).correct
        assert set(options) == LABELS and len(options) == 3
        assert len(q.distractors) == 2 and q.answer_text not in q.distractors
        for i, opt in enumerate(options):
            assert grade_mcq(q, str(i)).correct is (opt == q.answer_text)
        assert set(q.option_notes) == set(q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_class_and_key_position_balance_at_400_seeds(difficulty):
    classes, position = Counter(), Counter()
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        classes[q.answer_text] += 1
        position[mcq_options(q).index(q.answer_text)] += 1
    assert set(classes) == LABELS and set(position) == {0, 1, 2}
    for n in [*classes.values(), *position.values()]:
        assert 100 <= n <= 170, (classes, position)  # about a third each


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompt_states_no_value_and_has_shapes(difficulty):
    texts, shapes = set(), set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not any(ch.isdigit() for ch in q.text), q.text
        assert q.text in q.latex and "$" not in q.text
        assert "graph" in q.text.lower() or "grid" in q.text.lower() or "diagram" in q.text.lower()
        texts.add(q.text)
        shapes.add(prompt_shape(q))
    assert len(texts) >= 20, len(texts)
    assert len(shapes) >= 3


def test_hard_and_exam_add_a_shape():
    def shapes(d):
        return {prompt_shape(make(TOPIC, seed=s, difficulty=d)) for s in range(60)}

    easy, medium, hard, exam = (shapes(d) for d in DIFFICULTIES)
    assert medium - easy
    assert exam - easy - medium - hard
    assert hard - easy - medium


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")
