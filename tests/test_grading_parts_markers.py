"""Part markers in a one-string reply to a multi-part question, on real multi-part skill questions.

A learner who types "a. 6.1 b. 0.98" in one box must be marked part by part, a marker must never be read
inside an answer (a decimal point, a variable), and a reply that cannot be split must say so instead of
scoring a silent 0. See `mathgen/grading/parts.py`.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading import grade
from mathgen.grading.parts import UNREADABLE_NOTE_KIND, split_reply

SKILL_TOPIC = "exam_parts_circle_radius_arc"  # skill circle-area-to-diameter-and-circumference: parts a, b, c (numbers)
EXPRESSION_TOPIC = "exam_parts_implicit_tangent"  # skill senior-use-implicit-differentiation-...: parts a-d


def _question(topic=SKILL_TOPIC, seed=3, labels=None):
    question = make(topic, seed=seed)
    assert question.parts
    if labels is not None:
        parts = [dataclasses.replace(p, label=label, follow_through=None) for p, label in zip(question.parts, labels)]
        question = dataclasses.replace(question, parts=parts)
    return question


def _answers(question):
    return [p.answer_text for p in question.parts]


def _wrong(question):
    return [str(float(p.answer_text) + 100) for p in question.parts]


def _forms(labels, values):
    a, b, c = labels
    x, y, z = values
    return {
        "dot": f"{a}. {x} {b}. {y} {c}. {z}",
        "dot-upper": f"{a.upper()}. {x} {b.upper()}. {y} {c.upper()}. {z}",
        "paren": f"({a}) {x} ({b}) {y} ({c}) {z}",
        "paren-upper": f"({a.upper()}) {x} ({b.upper()}) {y} ({c.upper()}) {z}",
        "close-paren-commas": f"{a}) {x}, {b}) {y}, {c}) {z}",
        "equals": f"{a}={x} {b}={y} {c}={z}",
        "equals-spaced": f"{a} = {x}  {b} = {y}  {c} = {z}",
        "colon-semicolons": f"{a}: {x}; {b}: {y}; {c}: {z}",
        "newlines": f"{a}. {x}\n{b}. {y}\n{c}. {z}",
        "bare-newlines": f"{x}\n{y}\n{z}",
        "bare-semicolons": f"{x}; {y}; {z}",
        "lead-in": f"My answers: {a}) {x} {b}) {y} {c}) {z}",
        "paren-colon": f"({a}): {x} ({b}): {y} ({c}): {z}",
    }


LABEL_SETS = [("a", "b", "c"), ("i", "ii", "iii"), ("1", "2", "3")]


@pytest.mark.parametrize("labels", LABEL_SETS, ids=["abc", "roman", "digits"])
def test_every_marker_form_is_graded_right(labels):
    question = _question(labels=labels)
    for name, text in _forms(labels, _answers(question)).items():
        if name.startswith("bare") and labels != LABEL_SETS[0]:
            continue
        result = grade("number", question, text)
        assert result.correct, (name, text, result.marks)
        assert result.marks == result.max_marks, (name, text)
        assert result.note_kind != UNREADABLE_NOTE_KIND, name


@pytest.mark.parametrize("labels", LABEL_SETS, ids=["abc", "roman", "digits"])
def test_wrong_values_stay_wrong_in_every_form(labels):
    question = _question(labels=labels)
    for name, text in _forms(labels, _wrong(question)).items():
        result = grade("number", question, text)
        assert not result.correct, (name, text)
        assert result.marks == 0, (name, text)


def test_marker_forms_on_an_unrelabelled_real_skill_question():
    # No relabelling: the generator's own follow-through parts, graded as the API would.
    for seed in range(5):
        question = _question(seed=seed)
        for name, text in _forms(("a", "b", "c"), _answers(question)).items():
            assert grade("number", question, text).marks == question.parts[0].marks + sum(
                p.marks for p in question.parts[1:]
            ), (seed, name)


def test_a_reply_missing_a_part_gets_partial_marks():
    question = _question()
    a, b, c = _answers(question)
    for text in (f"a. {a} c. {c}", f"(a) {a} (c) {c}", f"a={a} c={c}", f"a) {a}\nc) {c}", f"c: {c}; a: {a}"):
        result = grade("number", question, text)
        assert not result.correct, text
        assert result.marks == question.parts[0].marks + question.parts[2].marks, text
        assert result.parts["b"].marks == 0
        assert result.note_kind != UNREADABLE_NOTE_KIND


def test_an_answer_in_one_part_is_not_marked_in_another():
    question = _question()
    a, b, c = _answers(question)
    result = grade("number", question, f"a. {b} b. {a} c. {c}")
    assert result.parts["c"].correct
    assert not result.parts["a"].correct and not result.parts["b"].correct


def test_decimal_points_are_not_markers():
    question = _question(labels=("1", "2", "3"))
    # "1.5" must not open part 1, and "2.8" must not be read as the marker "2." plus "8".
    assert split_reply(question.parts, "1. 2.8 2. 17.6 3. 1.5") == {"1": "2.8", "2": "17.6", "3": "1.5"}
    assert split_reply(question.parts, "1) 2.8 2) 1.5") == {"1": "2.8", "2": "1.5", "3": ""}


def test_roman_labels_are_not_confused_with_each_other():
    question = _question(labels=("i", "ii", "iii"))
    assert split_reply(question.parts, "i. 4 ii. 5 iii. 6") == {"i": "4", "ii": "5", "iii": "6"}
    assert split_reply(question.parts, "(iii) 6 (i) 4") == {"i": "4", "ii": "", "iii": "6"}


def test_a_variable_inside_an_answer_is_not_a_marker():
    question = _question(EXPRESSION_TOPIC)
    a, b, c, d = _answers(question)
    text = f"a) {a} b) {b} c) {c} d) {d}"
    replies = split_reply(question.parts, text)
    assert replies == {"a": a, "b": b, "c": c, "d": d}
    # The tangent reply "y = ..." and a bare "x = 3" stay inside their part: x and y are not labels here.
    assert split_reply(question.parts, "a) x = 3 b) y = 2x + 1") == {"a": "x = 3", "b": "y = 2x + 1", "c": "", "d": ""}
    # A label letter glued to a coefficient is part of the expression, not a marker.
    assert split_reply(question.parts, "a) 2b = 4 c) 7") == {"a": "2b = 4", "b": "", "c": "7", "d": ""}
    assert grade("expression", question, text).marks == sum(p.marks for p in question.parts)


@pytest.mark.parametrize(
    "text",
    [
        "6.1 and 0.98",
        "I don't know",
        "6.1, 0.98, 12.3",
        "6.1 0.98",
        "the radius is 6.1",
        "a",
    ],
)
def test_an_unreadable_reply_says_so(text):
    question = _question()
    result = grade("number", question, text)
    assert result.correct is False
    assert result.marks == 0 and result.max_marks == sum(p.marks for p in question.parts)
    assert result.note_kind == UNREADABLE_NOTE_KIND
    assert "Could not read your answer as separate parts" in result.note
    assert "(a)" in result.note
    assert set(result.parts) == {"a", "b", "c"}


def test_an_empty_or_dict_reply_is_not_called_unreadable():
    question = _question()
    assert grade("number", question, "").note_kind != UNREADABLE_NOTE_KIND
    assert grade("number", question, "   ").note is None
    result = grade("number", question, dict(zip("abc", _answers(question))))
    assert result.correct and result.marks == sum(p.marks for p in question.parts)
    assert result.marks == Fraction(result.max_marks)
