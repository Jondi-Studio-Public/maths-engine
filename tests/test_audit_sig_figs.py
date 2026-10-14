"""A question that asks for N significant figures is graded to N, and follow-through rounds to the count asked."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from mathgen import engine, make
from mathgen.grading import grade
from mathgen.text_quality import requested_significant_figures, significant_figure_count

_SKILL = next(s for s in engine.load_skills()[1] if s.id == "significant-figures")
_EXAM = next(i for i, s in enumerate(_SKILL.sources) if s.difficulty == "exam")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Round 4333 to 2 significant figures.", 2),
        ("Give x to three sig. figs.", 3),
        ("Round to 3 s.f. and 2 dp", None),
        ("Hence write the area to that number of significant figures.", None),
        ("Estimate the cost by rounding each of the three numbers to 1 significant figure.", None),
        ("Each value is given to 4 significant figures. Work out the total.", None),
    ],
)
def test_requested_significant_figures(text, expected):
    assert requested_significant_figures(text) == expected


@pytest.mark.parametrize(
    ("token", "count"),
    [
        ("23.0", 3),
        ("0.550", 3),
        ("0.55", 2),
        ("230", 2),
        ("4300", 2),
        ("100", 1),
        ("0", 1),
        ("0.0070", 2),
        ("-12.50", 4),
    ],
)
def test_significant_figure_count(token, count):
    assert significant_figure_count(token) == count


def test_practice_tiers_refuse_extra_figures_and_keep_the_canonical():
    checked = 0
    for source in range(3):
        for seed in range(10):
            q = engine.make_for_skill(_SKILL, seed, source=source)
            assert engine.grade(_SKILL, q, q.answer_text).correct, (source, seed, q.answer_text)
            padded = q.answer_text + ("0" if "." in q.answer_text else ".0")
            assert not engine.grade(_SKILL, q, padded).correct, (source, seed, padded)
            checked += 1
    assert checked == 30


def test_whole_number_trailing_zeros_stay_ambiguous():
    q = make("significant_figures", seed=2, variant="large_trailing_zero")
    assert requested_significant_figures(q.text) is not None
    assert grade("number", q, q.answer_text).correct


def test_exam_reply_with_extra_figures_is_wrong_and_exact_figures_right():
    for seed in range(1, 11):
        q = engine.make_for_skill(_SKILL, seed, source=_EXAM)
        full = {p.label: p.answer_text for p in q.parts}
        assert engine.grade(_SKILL, q, full).marks == sum(p.marks for p in q.parts)
        c = q.parts[2].answer_text
        padded = c + ("0" if "." in c else ".0")
        assert not engine.grade(_SKILL, q, {**full, "c": padded}).parts["c"].correct, (seed, padded)


def _round_sig(value: Fraction, figures: int) -> str:
    """`value` half-up to `figures` significant figures, written with its trailing zeros."""
    exponent = 0
    while value >= 10 ** (exponent + 1):
        exponent += 1
    while value < Fraction(10) ** exponent:
        exponent -= 1
    unit = Fraction(10) ** (exponent - figures + 1)
    rounded = math.floor(value / unit + Fraction(1, 2)) * unit
    places = max(0, -(exponent - figures + 1))
    return f"{float(rounded):.{places}f}"


def test_exam_follow_through_rounds_to_the_learners_own_count():
    credited = 0
    for seed in range(1, 21):
        q = engine.make_for_skill(_SKILL, seed, source=_EXAM)
        product = q.parts[0].answer_text
        k = int(q.parts[1].answer_text)
        for wrong in (k - 1, k + 1):
            if wrong < 2:
                continue  # one figure is a single digit that the junk guard (0, 1, -1) may refuse
            reply = {"a": product, "b": str(wrong), "c": _round_sig(Fraction(product), wrong)}
            result = engine.grade(_SKILL, q, reply)
            assert result.parts["c"].correct and result.parts["c"].follow_through, (seed, reply)
            assert not result.parts["b"].correct
            # the count the learner chose is the count enforced: one figure more is refused
            more = _round_sig(Fraction(product), wrong) + ("0" if "." in reply["c"] else ".0")
            assert not engine.grade(_SKILL, q, {**reply, "c": more}).parts["c"].correct
            credited += 1
    assert credited >= 20


def test_a_canonical_written_to_more_places_than_the_count_is_still_accepted():
    # "Round 0.009819 to 1 significant figure" is answered 0.010 so that its precision tells 0.009 apart.
    for seed in range(60):
        q = engine.make_for_skill(_SKILL, seed, source=1)
        assert engine.grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
