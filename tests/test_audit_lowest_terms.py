"""Where the stem asks for lowest terms, a decimal or an unreduced fraction is wrong; elsewhere nothing changes."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


def _frac(text, p, q, shown=None):
    return Question(
        topic="t", level=0, seed=1, text=text, latex="", answer={"p": p, "q": q}, answer_text=shown or f"{p}/{q}"
    )


ASKS = "Write 0.568 as a fraction in lowest terms."


@pytest.mark.parametrize("reply", ["71/125", " 71 / 125 "])
def test_reduced_fraction_is_accepted(reply):
    assert grade("fraction", _frac(ASKS, 71, 125), reply).correct


@pytest.mark.parametrize("reply", ["0.568", ".568", "142/250", "568/1000", "0.568 "])
def test_decimal_and_unreduced_are_wrong_when_lowest_terms_is_asked(reply):
    assert not grade("fraction", _frac(ASKS, 71, 125), reply).correct


@pytest.mark.parametrize("reply", ["3", "3.0", "6/2"])
def test_whole_number_key(reply):
    question = _frac("Give x as an integer or a fraction in lowest terms.", 3, 1, "3")
    assert grade("fraction", question, reply).correct is (reply != "6/2")


@pytest.mark.parametrize("reply", ["0.568", "142/250", "71/125"])
def test_without_the_wording_every_equal_form_is_still_accepted(reply):
    assert grade("fraction", _frac("Write 0.568 as a fraction.", 71, 125), reply).correct


def test_a_key_shown_as_a_decimal_is_not_refused_its_own_form():
    question = _frac("Give the answer in simplest form.", 1, 8, "0.125")
    assert grade("fraction", question, "0.125").correct


def test_mixed_number_keeps_its_current_behaviour():
    question = _frac("Write 1.5 as a fraction in lowest terms.", 3, 2)
    assert grade("fraction", question, "3/2").correct
    assert grade("fraction", question, "1 1/2").correct


def _exam_question(skill_id, seed):
    sk = _SKILLS[skill_id]
    source = next(i for i, s in enumerate(sk.sources) if s.difficulty == "exam")
    return sk, engine.make_for_skill(sk, seed, source=source)


@pytest.mark.parametrize(
    "skill_id, label",
    [("f10-terminating-recurring-decimals", "a"), ("f10-percentage-equivalence-comparison", "a")],
)
def test_real_parts_refuse_the_decimal_and_keep_the_fraction(skill_id, label):
    sk, q = _exam_question(skill_id, 1)
    part = next(p for p in q.parts if p.label == label)
    answer = part.answer
    value = Fraction(answer["p"], answer["q"])
    decimal = f"{float(value):.10g}"
    assert Fraction(decimal) == value, "the repro needs a terminating decimal"
    full = {p.label: p.answer_text for p in q.parts}
    assert engine.grade(sk, q, full).parts[label].correct
    assert not engine.grade(sk, q, {**full, label: decimal}).parts[label].correct


def test_each_item_of_an_ordered_list_of_fractions_is_held_to_it():
    sk, q = _exam_question("senior-use-relative-frequencies-obtained-from-data-to-obtain", 1)
    part = next(p for p in q.parts if p.answer_kind == "ordered_list")
    full = {p.label: p.answer_text for p in q.parts}
    assert engine.grade(sk, q, full).parts[part.label].correct
    unreduced = ", ".join(f"{x.numerator * 2}/{x.denominator * 2}" for x in part.answer)
    decimals = ", ".join(f"{float(x):.10g}" for x in part.answer)
    assert not engine.grade(sk, q, {**full, part.label: unreduced}).parts[part.label].correct
    assert not engine.grade(sk, q, {**full, part.label: decimals}).parts[part.label].correct
