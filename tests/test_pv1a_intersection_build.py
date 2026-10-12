"""The point where two drawn lines meet (ACMEM126), topic pv1a_intersection."""

import math
import re
from fractions import Fraction

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "pv1a_intersection"
SKILL_ID = "senior-interpret-the-point-of-intersection-and-other-important"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(skill, difficulty, seed):
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, notes=True, **(src.params or {}))


def _lines(fig):
    out = []
    for ln in fig["lines"]:
        (x1, y1), (x2, y2) = ln["through"]
        m = Fraction(y2 - y1, x2 - x1)
        out.append((m, Fraction(y1) - m * x1, ln))
    return out


def _angle_grid_units(fig):
    xg, yg = fig["x_grid"], fig["y_grid"]
    (m1, _, _), (m2, _, _) = _lines(fig)
    a1 = math.atan(float(m1 * xg / yg))
    a2 = math.atan(float(m2 * xg / yg))
    return abs(math.degrees(a1 - a2))


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


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_figure_valid_no_leak_and_plausible(skill, difficulty):
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        fig = q.figure
        assert fig is not None and figures.validate(fig) == []
        figures.assert_no_leak(fig, [q.answer_text, *q.distractors])
        assert fig["alt"] == "two straight lines on a coordinate grid"
        assert [ln.get("label") for ln in fig["lines"]] == ["A", "B"]
        xg, yg = fig["x_grid"], fig["y_grid"]
        xr, yr = fig["x_range"], fig["y_range"]
        assert (xr[1] - xr[0]) / xg + 1 <= 10 and (yr[1] - yr[0]) / yg + 1 <= 10
        through = []
        for ln in fig["lines"]:
            pts = [tuple(p) for p in ln["through"]]
            for x, y in pts:
                assert xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]
                assert x % xg == 0 and y % yg == 0  # lattice points
            assert abs(pts[1][0] - pts[0][0]) / xg >= 2  # two grid columns apart
            through += pts
        (m1, c1, _), (m2, c2, _) = _lines(fig)
        assert m1 != m2  # never parallel
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        assert x.denominator == 1 and y.denominator == 1
        assert x % xg == 0 and y % yg == 0  # a grid intersection
        assert (x, y) not in {(Fraction(a), Fraction(b)) for a, b in through}
        assert xr[0] < x < xr[1] and yr[0] < y < yr[1]
        assert _angle_grid_units(fig) >= 15
        assert len(set(through)) == 4


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tier_decisions(skill, difficulty):
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        fig = q.figure
        (m1, c1, _), (m2, c2, _) = _lines(fig)
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        xr, yr = fig["x_range"], fig["y_range"]
        if difficulty == "easy":
            assert x > 0 and y > 0
            assert xr[0] <= 0 <= xr[1] and yr[0] <= 0 <= yr[1]  # both axes drawn
            assert fig["x_grid"] == fig["y_grid"] == 1 and fig["equal_scale"] is True
        elif difficulty == "medium":
            assert x <= 0 or y <= 0
            assert fig["x_grid"] == fig["y_grid"] == 1
            assert xr[0] <= 0 <= xr[1] and yr[0] <= 0 <= yr[1]
        elif difficulty == "hard":
            xg, yg = fig["x_grid"], fig["y_grid"]
            assert (xg == 1 and yg in (2, 5)) or (yg == 1 and xg in (2, 5))
            assert fig["equal_scale"] is False
        else:
            assert fig["x_grid"] != fig["y_grid"] and fig["equal_scale"] is False
            assert xr[0] == 0 and yr[0] == 0  # a context graph: quadrant I only
            assert fig["x_label"] not in ("", "x") and fig["y_label"].endswith("(dollars)")  # quantities and units
            assert m1 > 0 and m2 > 0 and c1 >= 0 and c2 >= 0


def test_medium_draws_a_horizontal_line_sometimes():
    flat = sum(
        any(
            ln["through"][0][1] == ln["through"][1][1]
            for ln in make(TOPIC, seed=s, difficulty="medium").figure["lines"]
        )
        for s in range(60)
    )
    assert flat >= 10


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_reader_grader_and_distractors(skill, difficulty):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    notes = set()
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        assert reader(q) == q.answer_text
        X, Y = q.answer["x"], q.answer["y"]
        assert isinstance(X, int) and isinstance(Y, int)
        assert q.answer_text == f"({X}, {Y})"
        for form in (q.answer_text, f"{X}, {Y}", f"({X},{Y})", f"{X},{Y}", f"({X}.0, {Y}.0)", f"( {X} , {Y} )"):
            assert grade_pair(q, form).correct, (difficulty, seed, form)
        assert not grade_pair(q, f"({Y}, {X})").correct or X == Y
        assert len(q.distractors) >= 2
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        for d in q.distractors:
            assert re.fullmatch(r"\(-?\d+, -?\d+\)", d), d
            assert q.option_notes[d]
            notes.add(q.option_notes[d])
    assert len(notes) >= 3


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompts_have_no_numbers_and_enough_variety(difficulty):
    texts, shapes = set(), set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert not any(ch.isdigit() for ch in q.text), q.text
        assert "A" in q.text and "B" in q.text
        assert q.text in q.latex and "$" not in q.text
        texts.add(q.text)
        shapes.add(prompt_shape(q))
    assert len(texts) >= 20, len(texts)
    assert len(shapes) >= 3


def test_hard_and_exam_add_a_shape():
    shape = lambda d: {prompt_shape(make(TOPIC, seed=s, difficulty=d)) for s in range(60)}
    easy, medium, hard, exam = shape("easy"), shape("medium"), shape("hard"), shape("exam")
    assert medium - easy
    assert hard - easy - medium
    assert exam - easy - medium - hard


def test_exam_asks_where_the_plans_agree_and_names_the_pair_order():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        assert re.search(r"\((hours|classes|gigabytes|weeks|posters|kilometres), dollars\)", q.text), q.text
        assert q.figure["x_label"] and q.figure["y_label"]


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# --- label readability: the renderer draws each label at the SECOND clipped end of its line ---


def _renderer_label_anchor(through, xr, yr):
    """The figure renderer's clipLineToPlane, re-implemented: the t-high end of the clipped line."""
    (x1, y1), (x2, y2) = [(Fraction(a), Fraction(b)) for a, b in through]
    dx, dy = x2 - x1, y2 - y1
    t_hi = None
    for p, q in ((dx, xr[1] - x1), (-dx, x1 - xr[0]), (dy, yr[1] - y1), (-dy, y1 - yr[0])):
        if p > 0:
            t_hi = q / p if t_hi is None else min(t_hi, q / p)
    return x1 + t_hi * dx, y1 + t_hi * dy


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_labels_are_visible_apart_and_lines_are_not_axes(skill, difficulty):
    for seed in range(200):
        fig = _make(skill, difficulty, seed).figure
        xr, yr = fig["x_range"], fig["y_range"]
        xspan, yspan = Fraction(xr[1] - xr[0]), Fraction(yr[1] - yr[0])
        anchors = []
        for ln in fig["lines"]:
            (x1, y1), (x2, y2) = ln["through"]
            assert not (y1 == y2 == 0) and not (x1 == x2 == 0), (difficulty, seed)  # not an axis
            bx, by = _renderer_label_anchor(ln["through"], xr, yr)
            assert (yr[1] - by) / yspan >= Fraction(12, 100), (difficulty, seed, "label under the top edge")
            assert (bx - xr[0]) / xspan >= Fraction(15, 100), (difficulty, seed, "label past the left edge")
            anchors.append((bx, by))
        (ax, ay), (bx, by) = anchors
        assert abs(ax - bx) / xspan >= Fraction(12, 100) or abs(ay - by) / yspan >= Fraction(12, 100), (
            difficulty,
            seed,
        )
