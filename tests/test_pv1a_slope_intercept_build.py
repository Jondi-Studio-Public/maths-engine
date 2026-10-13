"""Tests `pv1a_slope_intercept` (ACMGM041): reading (m, c) off a drawn line."""

from collections import Counter
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "pv1a_slope_intercept"
SKILL_ID = "senior-determine-the-slope-and-intercepts-of-a-straight"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
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
    assert skill.answer == "pair"
    assert {s.difficulty for s in skill.sources} == set(DIFFICULTIES)
    exam = next(s for s in skill.sources if s.difficulty == "exam")
    assert list(exam.stages) == ["exam"]
    assert {m.id for m in skill.misconceptions} >= {
        "swapped-m-and-c",
        "sign-error-on-m",
        "c-read-from-a-point",
        "m-counted-in-squares",
    }


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
        assert abs(x2 - x1) / xg >= 2  # two grid columns apart
        assert x1 != x2  # never vertical
        m, c = _mc(fig)
        assert c.denominator == 1  # c is an integer
        assert m.denominator in (1, 2, 3, 4, 5)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tier_decisions(skill, difficulty):
    for seed in SEEDS:
        fig = _make(skill, difficulty, seed).figure
        m, c = _mc(fig)
        x_lo, x_hi = fig["x_range"]
        y_lo, y_hi = fig["y_range"]
        y_axis_drawn = x_lo <= 0 <= x_hi
        c_visible = y_lo <= c <= y_hi
        if difficulty == "easy":
            assert y_axis_drawn and c_visible and m.denominator == 1
        elif difficulty == "medium":
            assert not y_axis_drawn and m.denominator in (1, 2) and c_visible
        elif difficulty == "hard":
            assert m.denominator > 1 and not c_visible
        else:
            assert fig["x_grid"] != fig["y_grid"] and fig["equal_scale"] is False


def test_hard_has_more_than_halves():
    dens = {_mc(make(TOPIC, seed=s, difficulty="hard").figure)[0].denominator for s in range(80)}
    assert dens - {1, 2}


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_reader_grader_and_distractors(skill, difficulty):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        assert reader(q) == q.answer_text
        assert grade_pair(q, q.answer_text).correct
        m, c = _mc(q.figure)
        assert grade_pair(
            q, f"{m}, {c}" if m.denominator == 1 else f"{float(m)}, {c}"
        ).correct or m.denominator not in (1, 2, 4)
        assert grade_pair(
            q, f"({m.numerator}/{m.denominator}, {int(c)})" if m.denominator > 1 else f"({m}, {int(c)})"
        ).correct
        assert len(q.distractors) == 3
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        for d in q.distractors:
            assert not grade_pair(q, d).correct, (difficulty, seed, d)
        assert set(q.option_notes) == set(q.distractors)


def test_swapped_and_sign_errors_are_offered_and_rejected():
    seen = Counter()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy", notes=True)
        for d, note in q.option_notes.items():
            seen[note] += 1
            assert not grade_pair(q, d).correct
    assert {"swapped-m-and-c", "sign-error-on-m", "c-read-from-a-point"} <= set(seen)


def test_exam_distractor_counts_squares():
    notes = Counter()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam", notes=True)
        notes.update(q.option_notes.values())
    assert notes["m-counted-in-squares"] > 0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompt_never_states_answer_and_shapes(difficulty):
    texts, shapes = set(), set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "=" not in q.text and "graph" in q.text.lower() or "grid" in q.text.lower()
        assert not any(ch.isdigit() for ch in q.text), q.text
        assert q.text in q.latex and "$" not in q.text
        texts.add(q.text)
        shapes.add(prompt_shape(q))
    assert len(texts) >= 20, len(texts)
    assert len(shapes) >= 3


def test_hard_and_exam_add_a_shape():
    def shapes(d):
        return {prompt_shape(make(TOPIC, seed=s, difficulty=d)) for s in range(60)}

    easy, medium, hard, exam = (shapes(d) for d in DIFFICULTIES)
    assert hard - easy - medium
    assert exam - easy - medium - hard


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_concentration(difficulty):
    answers = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(200))
    assert len(answers) >= 20, len(answers)
    assert max(answers.values()) / 200 <= 0.25


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")
