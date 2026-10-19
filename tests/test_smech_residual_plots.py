"""Tests `residual-plot-linear-model-check`: the verdict is re-derived from the plotted points or the listed residuals."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
from smech_support import (
    assert_exam_rules,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    fixed,
    frac,
    questions,
    skill,
)

from mathgen import figures, make
from mathgen.engine import grade

SK = skill("residual-plot-linear-model-check")
SEEDS = range(60)
NUM = r"-?\d+(?:\.\d+)?"


def runs(res):
    return 1 + sum(1 for a, b in zip(res, res[1:]) if (a > 0) != (b > 0))


def fans(res):
    third = len(res) // 3
    first = sum(abs(r) for r in res[:third]) / third
    last = sum(abs(r) for r in res[-third:]) / third
    return last / first >= 2.5


def verdict(res):
    """What a careful reader says about the residuals: the same rule a teacher would state, written afresh."""
    n = runs(res)
    if n <= 3:
        return "curved"
    if fans(res):
        return "fan"
    assert n >= (5 if len(res) > 5 else 4), res
    return "random"


def plotted(q):
    pts = sorted(q.figure["points"], key=lambda p: p["at"][0])
    return [Fraction(str(p["at"][1])) for p in pts]


WORDS = {"random": "scattered at random", "curved": "curved pattern", "fan": "fan out"}


def test_grading_rules():
    for d in ("easy", "medium", "hard"):
        assert_single_answer_rules(SK, d, SEEDS, minimum_distractors=3)


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_the_figure_shows_the_keyed_pattern(difficulty):
    seen = set()
    for q in questions(SK, difficulty, range(150)):
        res = plotted(q)
        assert len(res) == 9 and all(r != 0 for r in res)
        kind = verdict(res)
        seen.add(kind)
        assert WORDS[kind] in q.answer_text, (res, q.answer_text)
        assert q.answer_text.startswith("Appropriate") == (kind == "random")
    assert seen == ({"random", "curved"} if difficulty == "easy" else {"random", "curved", "fan"})


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_the_figure_does_not_leak_the_pattern(difficulty):
    for q in questions(SK, difficulty, range(60)):
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        alt = q.figure["alt"].lower()
        assert not any(w in alt for w in ("curve", "fan", "random", "pattern", "linear")), alt
        assert figures.get_reader(q.topic)(q) == q.answer_text


def test_the_reader_works_from_the_plot_alone():
    q = questions(SK, "medium", [3])[0]
    other = next(x for x in questions(SK, "medium", range(40)) if x.answer_text != q.answer_text)
    q.figure = other.figure
    assert figures.get_reader(q.topic)(q) == other.answer_text


def test_hard_tier_has_a_table_and_a_stated_r_squared_and_no_figure():
    seen = set()
    for q in questions(SK, "hard", range(150)):
        assert q.figure is None
        res = [frac(v) for v in re.findall(rf"x = \d+: ({NUM})", q.text)]
        assert len(res) == 9
        r2 = frac(re.search(r"r² = (\d\.\d+)", q.text).group(1))
        kind = verdict(res)
        seen.add(kind)
        assert kind in {"curved", "random"}
        if kind == "curved":
            assert r2 >= Fraction(9, 10) and q.answer_text.startswith("Not appropriate") and "curved" in q.answer_text
            assert any(d.startswith("Appropriate: r²") for d in q.distractors)  # the high r² trap is offered
        else:
            assert r2 <= Fraction(11, 20) and q.answer_text.startswith("Appropriate") and "no pattern" in q.answer_text
            assert any("is low" in d for d in q.distractors)  # the low r² trap is offered
        assert f"{r2}".replace("0.", "0.") in q.answer_text or str(float(r2)) in q.answer_text
    assert seen == {"curved", "random"}


def test_tiers_differ_in_shape():
    easy = {q.answer_text for q in questions(SK, "easy", range(80))}
    medium = {q.answer_text for q in questions(SK, "medium", range(80))}
    assert len(easy) == 2 and len(medium) == 3
    assert any("fan out" in a for a in medium) and not any("fan out" in a for a in easy)


def _exam():
    return [make("smech_residual_plots_exam", seed=s, difficulty="exam") for s in range(120)]


def points(t):
    return [(int(x), int(y)) for x, y in re.findall(r"\((\d+), (\d+)\)", t)]


def exam_oracle(q):
    t = q.text
    pts = points(t)
    m = re.search(rf"predicted [^=]+ = ({NUM}) ([+-]) ({NUM}) ×", t)
    a, b = frac(m.group(1)), frac(m.group(3)) * (1 if m.group(2) == "+" else -1)
    x = int(re.search(r"equal to (\d+),", q.parts[1].prompt + ",").group(1))
    res = [y - (a + b * xx) for xx, y in pts]
    sse = sum(r * r for r in res)
    kind = "curved" if runs(res) <= 3 else "random"  # five points: the sign pattern is all there is to read
    key = {
        "curved": "Not appropriate: the residuals form a curved pattern",
        "random": "Appropriate: the residuals are scattered at random about zero, with no pattern",
    }[kind]
    listed = {int(k): frac(v) for k, v in re.findall(rf"x = (\d+): ({NUM})", t)}
    assert all(listed[xx] == r for xx, r in zip([p[0] for p in pts], res) if xx in listed)
    return {"a": fixed(a + b * x, 1), "b": fixed(dict(pts)[x] - (a + b * x), 1), "c": key, "d": fixed(sse, 2)}


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[1, 1, 2, 2])
    kinds = set()
    for q in _exam():
        got = exam_oracle(q)
        assert got == {p.label: p.answer_text for p in q.parts}, q.text
        kinds.add(got["c"][:3])
    assert kinds == {"Not", "App"}


def test_exam_follows_a_wrong_residual_through_to_the_sum_of_squares():
    checked = 0
    for q in _exam()[:80]:
        x = int(re.search(r"equal to (\d+),", q.parts[1].prompt + ",").group(1))
        listed = [frac(v) for v in re.findall(rf"x = \d+: ({NUM})", q.text)]
        wrong = frac(q.parts[1].answer_text) + Fraction(6, 5)
        if wrong == 0:
            continue
        reply = {
            "a": q.parts[0].answer_text,
            "b": fixed(wrong, 1),
            "c": q.parts[2].answer_text,
            "d": fixed(wrong * wrong + sum(r * r for r in listed), 2),
        }
        if reply["d"] in q.parts[3].distractors or reply["d"] == q.parts[3].answer_text:
            continue
        result = grade(SK, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through
        checked += 1
    assert checked >= 40


def test_exam_distractors_keep_the_precision_and_text_varies():
    for q in _exam()[:40]:
        for p in q.parts:
            if p.answer_kind == "number":
                assert_number_distractors_match_places(p)
    assert len({q.text for q in _exam()}) >= 60


def test_deterministic():
    for topic, d in (("smech_residual_plots", "medium"), ("smech_residual_plots_exam", "exam")):
        assert make(topic, seed=7, difficulty=d).to_dict() == make(topic, seed=7, difficulty=d).to_dict()


def test_plotted_and_listed_residuals_add_to_zero():
    """The residuals of a least-squares line sum to zero, so the plotted points and the tabled values must too."""
    for difficulty in ("easy", "medium"):
        for seed in SEEDS:
            q = make("smech_residual_plots", seed=seed, difficulty=difficulty)
            assert sum(plotted(q)) == 0, (difficulty, seed, plotted(q))
    for seed in SEEDS:
        q = make("smech_residual_plots", seed=seed, difficulty="hard")
        listed = [Fraction(v) for v in re.findall(r"x = \d+: (" + NUM + ")", q.text)]
        assert len(listed) == 9 and sum(listed) == 0, (seed, listed)


def test_exam_five_residuals_add_to_zero():
    for seed in SEEDS:
        q = make("smech_residual_plots_exam", seed=seed, difficulty="exam")
        listed = [Fraction(v) for v in re.findall(r"x = -?\d+: (" + NUM + ")", q.text)]
        hidden = Fraction(q.parts[1].answer_text)
        assert len(listed) == 4 and sum(listed) + hidden == 0, (seed, listed, hidden)
