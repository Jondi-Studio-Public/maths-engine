"""Four grader fixes from the 2026-10-14 bug hunt: mixed numbers, e^2 and log10/log2 in expressions, the M/m SI
prefix clash in units, zero vectors in a scalar-multiple matrix answer, and part labels with no space between."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy

from mathgen.core import Part
from mathgen.grading import grade
from mathgen.grading.parts import _split
from mathgen.grading.units import unit_is_acceptable
from mathgen.grading.values import read_value


def _q(answer, answer_text, form=None, text="Find it."):
    return SimpleNamespace(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_alt=None,
        distractors=[],
        option_notes={},
        answer_form=form,
        distractor_values={},
        distractor_credit={},
        parts=None,
    )


# -- expression: mixed numbers, e^2, log10 / log2 ------------------------------------------------------------


def test_a_mixed_number_in_an_expression_is_its_value():
    assert grade("expression", _q("3/2", "3/2"), "1 1/2").correct
    assert not grade("expression", _q("1/2", "1/2"), "1 1/2").correct
    assert grade("expression", _q("-3/2", "-3/2"), "-1 1/2").correct
    assert grade("expression", _q("x+5/2", "x + 5/2"), "x + 2 1/2").correct


def test_two_spaced_numbers_are_still_a_product():
    assert grade("expression", _q("6", "6"), "2 3").correct
    assert not grade("expression", _q("5", "5"), "2 3").correct


@pytest.mark.parametrize("reply, expected", [("e^2", sympy.exp(2)), ("e^(1/2)", sympy.exp(sympy.Rational(1, 2)))])
def test_read_value_reads_e_to_a_power(reply, expected):
    assert read_value(reply) == expected


def test_read_value_still_refuses_a_lone_e_and_a_variable():
    assert read_value("e") is None
    assert read_value("x^2") is None


@pytest.mark.parametrize(
    "reply, expected",
    [("log10(100)", 2), ("log2(8)", 3), ("log10(1000)", 3), ("log2(2*(1+3))", 3), ("3 + log10(10)", 4)],
)
def test_log_with_the_base_in_its_name(reply, expected):
    assert read_value(reply) == expected
    assert grade("expression", _q(str(expected), str(expected)), reply).correct


def test_log_with_an_underscored_base_in_an_expression():
    assert grade("expression", _q("3", "3"), "log_10(1000)").correct


def test_log_with_a_base_is_not_graded_as_a_different_number():
    assert not grade("expression", _q("3", "3"), "log10(100)").correct
    assert read_value("log10(100") is None


# -- units: M (mega) and m (milli) are different prefixes -------------------------------------------------------


@pytest.mark.parametrize("reply", ["2000 mL", "2000 ml", "2000 millilitres", "2000 Millilitres"])
def test_millilitre_spellings_still_pass(reply):
    assert unit_is_acceptable(_q(2000, "2000 mL", text="How many millilitres in 2 L?"), reply)


@pytest.mark.parametrize("reply", ["2000 ML", "2000 Ml"])
def test_a_megalitre_is_not_a_millilitre(reply):
    question = _q(2000, "2000 mL", text="How many millilitres in 2 L?")
    assert not unit_is_acceptable(question, reply)
    assert not grade("number", question, reply).correct
    assert grade("number", question, "2000 mL").correct


@pytest.mark.parametrize("given, shown", [("5 Mm", "5 mm"), ("5 Mg", "5 mg"), ("5 MS", "5 ms"), ("5 mJ", "5 MJ")])
def test_the_other_m_prefix_is_a_different_unit(given, shown):
    question = _q(5, shown)
    assert unit_is_acceptable(question, shown)
    assert not unit_is_acceptable(question, given)


@pytest.mark.parametrize(
    "given, shown", [("5 KM", "5 km"), ("30 MM", "30 mm"), ("5 Km", "5 km"), ("5 MHZ", "5 MHz"), ("5 MPA", "5 MPa")]
)
def test_case_variants_without_a_prefix_clash_still_pass(given, shown):
    assert unit_is_acceptable(_q(5, shown), given)


def test_a_clashing_compound_unit_is_refused():
    question = _q(2, "2 g/mL")
    assert unit_is_acceptable(question, "2 g/mL")
    assert unit_is_acceptable(question, "2 g/ml")
    assert not unit_is_acceptable(question, "2 g/ML")


# -- matrix: zero vector scalar multiple ------------------------------------------------------------------------


def test_a_zero_key_matches_a_zero_reply_only():
    key = _q([[0, 0]], "(0, 0)", form="scalar_multiple")
    assert grade("matrix", key, "[0,0]").correct
    assert grade("matrix", key, "(0, 0)").correct
    assert not grade("matrix", key, "(1, 0)").correct
    assert not grade("matrix", key, "(0, 2)").correct


def test_a_zero_reply_is_still_not_a_multiple_of_a_non_zero_key():
    key = _q([[1], [2]], "(1, 2)", form="scalar_multiple")
    assert not grade("matrix", key, "(0, 0)").correct
    assert grade("matrix", key, "(2, 4)").correct
    assert not grade("matrix", key, "(2, 5)").correct


def test_a_zero_column_in_a_column_scalar_multiple_key():
    key = _q([[1, 0], [2, 0]], "[[1, 0], [2, 0]]", form="column_scalar_multiple")
    assert grade("matrix", key, "[[1,0],[2,0]]").correct
    assert grade("matrix", key, "[[2,0],[4,0]]").correct
    assert not grade("matrix", key, "[[2,1],[4,0]]").correct
    assert not grade("matrix", key, "[[0,0],[0,0]]").correct


# -- parts: labels with no space between ------------------------------------------------------------------------


def _parts(*labels):
    return [
        Part(label=label, prompt="Find it", answer_kind="number", answer=i + 5, answer_text=str(i + 5))
        for i, label in enumerate(labels)
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "(a)5(b)6(c)7",
        "(a) 5 (b) 6 (c) 7",
        "(a)5 (b)6 (c)7",
        "(A)5(B)6(C)7",
        "(a)5;(b)6;(c)7",
        "(a):5(b):6(c):7",
        "(a)-5(b)-6(c)-7",
    ],
)
def test_glued_labels_split_like_spaced_ones(reply):
    sign = "-" if "-" in reply else ""
    assert _split(_parts("a", "b", "c"), reply) == {"a": f"{sign}5", "b": f"{sign}6", "c": f"{sign}7"}


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("a) 5 b) 6 c) 7", {"a": "5", "b": "6", "c": "7"}),
        ("a=5 b=6 c=7", {"a": "5", "b": "6", "c": "7"}),
        ("a: 5, b: 6, c: 7", {"a": "5", "b": "6", "c": "7"}),
        ("5; 6; 7", {"a": "5", "b": "6", "c": "7"}),
        ("5\n6\n7", {"a": "5", "b": "6", "c": "7"}),
        ("(b) 6 (a) 5", {"a": "5", "b": "6", "c": ""}),
        ("My answers: a) 5 b) 6", {"a": "5", "b": "6", "c": ""}),
        ("(a) 1.5 (b) 2. (c) 3", {"a": "1.5", "b": "2.", "c": "3"}),
    ],
)
def test_other_formats_split_as_before(reply, expected):
    assert _split(_parts("a", "b", "c"), reply) == expected


def test_a_bracketed_letter_inside_a_working_is_not_a_label_unless_the_reply_opens_with_one():
    assert _split(_parts("a", "b"), "3(b)") is None
    assert _split(_parts("a", "b"), "(a) 3(b)") == {"a": "3", "b": ""}
