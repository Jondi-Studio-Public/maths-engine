"""Stray letters beside a number, a leading minus that belongs to nothing, SI prefixes on a unit the question names,
and a Python None printed into question text. Grade checks call `grading.grade`, as every caller does."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathgen import make
from mathgen.grading import grade
from mathgen.text_quality import find_text_defects


def _q(answer, answer_text, text="Find it."):
    return SimpleNamespace(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_alt=None,
        distractors=[],
        option_notes={},
        answer_form=None,
        distractor_values={},
        distractor_credit={},
        parts=None,
    )


SEVEN = _q(7, "7")
LABELLED = _q(7, "x = 7")
FORCE = _q(80, "F = 80 N", "Find the force.")
MONEY = _q(352.5, "A$352.50", "Find the cost in Australian dollars.")
POINT = _q([-6, 4], "(-6, 4)", "Find the point.")
HALF = _q({"p": 1, "q": 2}, "1/2", "Write as a fraction.")


@pytest.mark.parametrize("reply", ["7x", "7/x", "x7", "7 x", "7y", "7/y", "x/7"])
def test_a_stray_letter_beside_a_number_is_wrong(reply):
    assert not grade("number", SEVEN, reply).correct


@pytest.mark.parametrize("reply", ["7", "7.0", "x = 7", "x=7", "7 dogs", "7 cm", "7 m", "7 N", "7 s", "7 J", "7 p.m."])
def test_a_number_still_passes_with_a_label_or_a_unit_word(reply):
    assert grade("number", SEVEN, reply).correct


def test_a_label_the_answer_shows_and_a_unit_word_still_pass():
    assert grade("number", LABELLED, "7").correct
    assert grade("number", LABELLED, "x = 7").correct
    assert not grade("number", LABELLED, "7x").correct
    assert grade("number", _q(0.064, "0.064"), "0.064 cm").correct


def test_a_letter_after_a_slash_is_wrong_but_a_rate_unit_is_not():
    assert not grade("fraction", HALF, "1/x").correct
    assert not grade("number", SEVEN, "7/s").correct
    assert grade("number", _q(12, "12 m/s", "Find the speed."), "12 m/s").correct
    assert grade("number", _q(12, "12 km/h", "Find the speed."), "12 km/h").correct


@pytest.mark.parametrize("reply", ["-(-6, 4)", "-(6, -4)", "- (-6, 4)", "(-6, 4)x", "(-6, 4)/x", "(-6, 4)y"])
def test_a_pair_with_a_stray_mark_is_wrong(reply):
    assert not grade("pair", POINT, reply).correct


@pytest.mark.parametrize("reply", ["(-6, 4)", "(-6,4)", "-6, 4", "x = -6, y = 4", "(x, y) = (-6, 4)", "P(-6, 4)"])
def test_a_pair_is_still_read_with_its_labels(reply):
    assert grade("pair", POINT, reply).correct


@pytest.mark.parametrize("reply", ["-A$352.50", "-F = 10.0 N", "-x = 7", "-(7)", "-x7", "- (7)", "-y", "-A352.50"])
def test_a_minus_that_is_followed_by_no_number_is_dropped_nowhere(reply):
    assert not grade("number", MONEY, reply).correct
    assert not grade("number", SEVEN, reply).correct
    assert not grade("number", FORCE, reply).correct


def test_a_leading_minus_before_a_number_is_judged_by_its_value_as_before():
    assert grade("number", _q(-7, "-7"), "-7").correct
    assert grade("number", _q(-7, "-7"), "- 7").correct
    assert grade("number", _q(-7, "-7"), "-7 dogs").correct
    assert not grade("number", SEVEN, "-7").correct


def test_a_letter_in_the_canonical_value_is_still_allowed():
    assert grade("number", MONEY, "A$352.50").correct
    assert grade("number", MONEY, "352.50").correct


@pytest.mark.parametrize("reply", ["80 mN", "80 MN", "80 µN", "80 nN", "80 GN", "80 kN"])
def test_an_si_prefix_on_the_asked_unit_is_a_different_unit(reply):
    assert not grade("number", FORCE, reply).correct


@pytest.mark.parametrize("reply", ["80 N", "F = 80 N", "80", "80 newtons", "80 dogs"])
def test_the_asked_unit_and_unit_words_still_pass(reply):
    assert grade("number", FORCE, reply).correct


def test_the_canonicals_own_prefixed_unit_is_accepted_and_the_other_case_is_not():
    current = _q(5, "I = 5 mA", "Find the current.")
    assert grade("number", current, "5 mA").correct
    assert not grade("number", current, "5 MA").correct


def test_a_prefixed_unit_is_not_judged_when_the_question_asks_for_none():
    assert grade("number", SEVEN, "7 mN").correct
    assert grade("number", SEVEN, "7 dm").correct


def test_a_converted_length_refuses_a_prefixed_unit_it_does_not_list():
    convert = _q(30, "30 mm", "Convert 3 cm to mm.")
    assert grade("number", convert, "30 mm").correct
    assert not grade("number", convert, "30 dm").correct
    assert not grade("number", convert, "30 nm").correct


# -- None in learner text -----------------------------------------------------------------------------------


def test_the_function_options_never_print_none():
    for seed in range(60):
        question = make("p15a_definition_quantifier_check", seed=seed, difficulty="easy")
        assert find_text_defects(question) == [], (seed, question.text)
        assert "None" not in question.text


def test_the_exam_function_options_never_print_none():
    for seed in range(60):
        question = make("exd_x_r2g_relation_repair", seed=seed, difficulty="exam", tier="exam")
        assert find_text_defects(question) == [], (seed, question.text)


def test_text_quality_flags_a_printed_none_but_not_a_category_called_none():
    shown = SimpleNamespace(text="NOT a function: None has no image in R.", answer_text="x", distractors=[])
    assert find_text_defects(shown) == ["python_null_in_text in text: 'None h'"]
    category = SimpleNamespace(
        text="Table (home / pet — None — Cat): House — 3", answer_text="House None = 3", distractors=["None of these"]
    )
    assert find_text_defects(category) == []
    nan = SimpleNamespace(text="The mean is NaN.", answer_text="x", distractors=[])
    assert find_text_defects(nan) == ["python_null_in_text in text: 'NaN'"]
