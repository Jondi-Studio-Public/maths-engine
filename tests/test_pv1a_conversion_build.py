"""Use a conversion graph (ACMEM124), topic pv1a_conversion."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import figures, make
from mathgen.text_quality import prompt_shape

TOPIC = "pv1a_conversion"
SKILL_ID = "senior-interpret-and-use-graphs-in-practical-situations-including"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(skill, difficulty, seed):
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, notes=True, **(src.params or {}))


def _line(fig):
    (x1, y1), (x2, y2) = fig["lines"][0]["through"]
    m = Fraction(y2 - y1, x2 - x1)
    return m, Fraction(y1) - m * x1


def _units(fig):
    return [re.search(r"\(([^)]+)\)\s*$", fig[k]).group(1) for k in ("x_label", "y_label")]


def test_skill_yaml_shape(skill):
    assert skill.status == "deferred"
    assert skill.deferred_reason == "visual"
    assert skill.needs_figure is True
    assert list(skill.capability_gates) == ["figure_stimulus"]
    assert skill.mental is None
    assert skill.answer == "number"
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
        assert len(fig["lines"]) == 1 and not fig["points"]
        xg, yg = fig["x_grid"], fig["y_grid"]
        xr, yr = fig["x_range"], fig["y_range"]
        assert (xr[1] - xr[0]) / xg + 1 <= 10 and (yr[1] - yr[0]) / yg + 1 <= 10
        (x1, y1), (x2, y2) = fig["lines"][0]["through"]
        for x, y in ((x1, y1), (x2, y2)):
            assert xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]
            assert x % xg == 0 and y % yg == 0
        assert abs(x2 - x1) / xg >= 2
        assert len(_units(fig)) == 2


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tier_decisions_and_prompt_names_units_and_direction(skill, difficulty):
    reverse_seen = forward_seen = 0
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        fig = q.figure
        m, c = _line(fig)
        xu, yu = _units(fig)
        assert len(re.findall(r"\d+", q.text)) == 1, q.text  # exactly one numeral: the input
        assert xu in q.text and yu in q.text  # both units named
        n = int(re.search(r"\d+", q.text).group())
        xin = re.search(rf"\b{n}\s*{re.escape(xu)}", q.text) is not None
        yin = re.search(rf"\b{n}\s*{re.escape(yu)}", q.text) is not None
        assert xin != yin  # the input has exactly one unit
        forward = xin
        # the stated input sits on a grid line of its own axis; so does the answer on the other
        in_step, out_step = (fig["x_grid"], fig["y_grid"]) if forward else (fig["y_grid"], fig["x_grid"])
        assert n % in_step == 0 and q.answer % out_step == 0
        assert n >= 2
        if difficulty in ("easy", "hard"):
            assert forward
            assert (m * n + c) == q.answer
        else:
            assert not forward
            assert (n - c) / m == q.answer
        if difficulty in ("easy", "medium"):
            assert c == 0  # proportional, through the origin
        else:
            assert c != 0  # an offset conversion
        if difficulty == "exam":
            assert fig["x_grid"] > 1 and fig["y_grid"] > 1 and fig["x_grid"] != fig["y_grid"]
        forward_seen += forward
        reverse_seen += not forward
    assert (forward_seen > 0) == (difficulty in ("easy", "hard"))
    assert (reverse_seen > 0) == (difficulty in ("medium", "exam"))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_reader_grader_and_distractors(skill, difficulty):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    notes = set()
    for seed in SEEDS:
        q = _make(skill, difficulty, seed)
        assert reader(q) == q.answer_text
        assert isinstance(q.answer, int) and q.answer >= 1
        assert not grade_number(q, str(q.answer + 1)).correct
        assert not grade_number(q, str(q.answer - 1)).correct
        assert len(q.distractors) >= 2
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        for d in q.distractors:
            assert d.isdigit() and int(d) >= 1
            notes.add(q.option_notes[d])
    assert len(notes) >= 3


def test_only_true_or_scenario_specific_conversions():
    true_units = {
        ("hours", "minutes"),
        ("feet", "inches"),
        ("metres", "centimetres"),
        ("kilometres", "metres"),
        ("kilograms", "grams"),
        ("litres", "millilitres"),
        ("minutes", "seconds"),
        ("days", "hours"),
        ("weeks", "days"),
        ("yards", "feet"),
    }
    for difficulty in DIFFICULTIES:
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            xu, yu = _units(q.figure)
            m, c = _line(q.figure)
            if (xu, yu) in true_units:
                # a real conversion factor: the slope in real units is the exact factor
                factor = {
                    ("hours", "minutes"): 60,
                    ("feet", "inches"): 12,
                    ("metres", "centimetres"): 100,
                    ("kilometres", "metres"): 1000,
                    ("kilograms", "grams"): 1000,
                    ("litres", "millilitres"): 1000,
                    ("minutes", "seconds"): 60,
                    ("days", "hours"): 24,
                    ("weeks", "days"): 7,
                    ("yards", "feet"): 3,
                }[(xu, yu)]
                assert m == factor and c == 0
                assert difficulty in ("easy", "medium")
            else:
                # a scenario-specific rule: the prompt says so, and never as a general fact
                assert "exchange desk today" in q.text or "rule of thumb" in q.text, q.text
                if "rule of thumb" in q.text:
                    assert "treat the rule as exact" in q.text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_prompt_variety(difficulty):
    texts, shapes = set(), set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.text in q.latex and "$" not in q.text
        assert "graph" in q.text
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


def test_bad_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")
