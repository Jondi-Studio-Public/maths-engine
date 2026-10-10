"""Marking correctness on messy typed replies: a right answer marked right and a wrong one marked wrong.

Each case here was marked the other way before: a reply with two minus signs, an unfactored expression in a
bracket for a factorising question, a surd that skipped the question's form through its decimal alternative,
the constant of integration as "c" or "K", square brackets and braces, a decimal power for a root, the empty set
as "∅", and an ordered list separated by semicolons."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import typed as grading


def _q(answer, answer_text, **extra):
    return Question(topic="test", level=0, seed=1, text="q", latex="", answer=answer, answer_text=answer_text, **extra)


def _correct(kind, question, given) -> bool:
    return grading.grade(kind, question, given).correct


# -- two minus signs: "--3" is 3, never -3 ---------------------------------------------


@pytest.mark.parametrize(
    "kind, question, given",
    [
        ("number", _q(-3, "-3"), "--3"),
        ("number", _q(-3, "-3"), "- -3"),
        ("number", _q(-3, "-3"), "−−3"),
        ("number", _q(3, "3"), "--3"),
        ("fraction", _q({"p": -1, "q": 2}, "-1/2"), "--1/2"),
        ("pair", _q({"x": 3, "y": -6}, "(3, -6)"), "(3, --6)"),
        ("list", _q([-1, 3], "-1, 3"), "--1, 3"),
        ("set", _q([-2, 5], "{-2, 5}"), "--2, 5"),
        ("ordered_list", _q([-3, 5], "-3, 5"), "--3, 5"),
        (
            "table",
            _q({"columns": ["x", "1"], "rows": [["y", None]], "blanks": {"0,1": -2}}, "y 1 = -2"),
            '{"0,1": "--2"}',
        ),
    ],
)
def test_two_minus_signs_are_wrong(kind, question, given):
    assert not _correct(kind, question, given)


@pytest.mark.parametrize("given", ["-3", "- 3", "−3", "(-3)"])
def test_one_minus_sign_still_reads(given):
    assert _correct("number", _q(-3, "-3"), given)


def test_expression_reads_two_minus_signs_as_plus():
    assert _correct("expression", _q("x + 3", "x + 3"), "x--3")


# -- factored form ---------------------------------------------------------------------

_FACTORED = _q("2*(4*x - 3)", "2(4x - 3)", answer_form="factored")


@pytest.mark.parametrize("given", ["(8x - 6)", "((8x-6))", " (8x - 6) "])
def test_unfactored_expression_in_a_bracket_is_not_factored(given):
    assert not _correct("expression", _FACTORED, given)


@pytest.mark.parametrize("given", ["2(4x - 3)", "(4x - 3)2", "(2)(4x - 3)", "2[4x - 3]", "-2(3 - 4x)"])
def test_factored_reply_still_accepted(given):
    assert _correct("expression", _FACTORED, given)


def _factored(answer, answer_text):
    return _q(answer, answer_text, answer_form="factored")


@pytest.mark.parametrize(
    "answer, answer_text, given",
    [
        ("2*(4*x - 3)", "2(4x - 3)", "1(8x - 6)"),  # a factor of 1 factors nothing
        ("(2*x - 11)*(x - 1)", "(2x - 11)(x - 1)", "2(x - 3)^2 - 7(x - 3) - 30"),  # the question restated
        ("(4*x + y)*(x + 3*y)", "(4x + y)(x + 3y)", "(4x^2 + 13xy + 3y^2)"),
        ("6*(2*x + 3)", "6(2x + 3)", "3(4x + 6)"),  # a common factor left inside the bracket
        ("9*x*(x - 8)", "9x(x - 8)", "x(9x - 72)"),
        ("9*x*(x - 8)", "9x(x - 8)", "9(x^2 - 8x)"),
        ("2*(x - 3)*(x**2 + 6*x - 4)", "2(x - 3)(x^2 + 6x - 4)", "(2x - 6)(x^2 + 6x - 4)"),
        ("(x - 1)*(x + 1)*(x**2 + 1)", "(x - 1)(x + 1)(x^2 + 1)", "(x^2 - 1)(x^2 + 1)"),  # fewer factors
    ],
)
def test_not_factorised_far_enough(answer, answer_text, given):
    assert not _correct("expression", _factored(answer, answer_text), given)


@pytest.mark.parametrize(
    "answer, answer_text, given",
    [
        ("(x + 7)**2", "(x + 7)(x + 7)", "(x + 7)^2"),
        ("(x - 3)*(x + 3)", "(x - 3)(x + 3)", "-(3 - x)(x + 3)"),
        ("(2*x - 11)*(x - 1)", "(2x - 11)(x - 1)", "(2(x - 3) - 5)(x - 3 + 2)"),
        ("9*x*(x - 8)", "9x(x - 8)", "9x(x - 8)"),
        # Factoring further than the canonical is still factorised.
        ("2*(x + 2)*(x**2 + 2*x)", "2(x + 2)(x^2 + 2x)", "2x(x + 2)^2"),
    ],
)
def test_factorised_in_another_order_or_further(answer, answer_text, given):
    assert _correct("expression", _factored(answer, answer_text), given)


# -- answer_alt does not skip the answer form -----------------------------------------

_SURD = _q("2*sqrt(3)", "2√3", answer_form="simplified_surd", answer_alt=3.4641016)


@pytest.mark.parametrize("given", ["√12", "sqrt(12)"])
def test_unsimplified_surd_is_not_accepted_through_its_decimal(given):
    assert not _correct("expression", _SURD, given)


@pytest.mark.parametrize("given", ["2√3", "3.46", "3.464"])
def test_surd_and_its_decimal_still_accepted(given):
    assert _correct("expression", _SURD, given)


# -- the constant of integration ---------------------------------------------------------

_INTEGRAL = _q("x**3/3 + C", "x^3/3 + C")


@pytest.mark.parametrize("given", ["x^3/3 + C", "x^3/3 + c", "x^3/3 + K", "x^3/3 + k", "(1/3)x^3 + c"])
def test_constant_of_integration_in_any_usual_letter(given):
    assert _correct("expression", _INTEGRAL, given)


@pytest.mark.parametrize("given", ["x^3/3", "x^3/3 + d", "x^3/3 + c + k", "x^3/3 + cx", "x^3/3 + 5"])
def test_wrong_or_missing_constant_still_wrong(given):
    assert not _correct("expression", _INTEGRAL, given)


def test_a_letter_the_answer_uses_is_not_the_constant():
    question = _q("k*x**3/3 + C", "kx^3/3 + C")
    assert _correct("expression", question, "kx^3/3 + c")
    assert not _correct("expression", question, "kx^3/3 + k")


# -- square brackets and braces ------------------------------------------------------------


def test_square_brackets_group():
    question = _q("2*(x + 3*(x - 1))", "2[x + 3(x - 1)]")
    assert _correct("expression", question, "2[x + 3(x - 1)]")
    assert _correct("expression", question, "8x - 6")


def test_braces_group():
    assert _correct("expression", _q("exp(2*x)", "e^(2x)"), "e^{2x}")


@pytest.mark.parametrize(
    "answer, given",
    [("2*x", "e^{2x}"), ("2*x", "\\sqrt{2x}"), ("12", "\\sqrt{12}")],
)
def test_braced_text_is_not_half_read(answer, given):
    """The spoken-word fallback read "e^{2x}" as 2x and "\\sqrt{12}" as 12."""
    assert not _correct("expression", _q(answer, answer), given)


# -- decimals for exact values -------------------------------------------------------------


@pytest.mark.parametrize("given", ["x^0.5", "x^.5", "sqrt(x)"])
def test_decimal_power_for_a_root(given):
    assert _correct("expression", _q("sqrt(x)", "√x"), given)


def test_decimal_coefficient_for_a_half():
    assert _correct("expression", _q("x/2", "x/2"), "0.5x")


@pytest.mark.parametrize("given", ["x^0.6", "x^0.33"])
def test_a_decimal_that_is_not_the_value_is_wrong(given):
    assert not _correct("expression", _q("x**(1/3)", "x^(1/3)"), given)


def test_rounded_decimal_for_a_third_is_wrong():
    assert not _correct("expression", _q("x/3", "x/3"), "0.333333x")


# -- the empty set -------------------------------------------------------------------------

_EMPTY = _q([], grading.set_answer_text([]))


@pytest.mark.parametrize("given", ["{}", "{ }", "∅", "Ø", "empty set", "The empty set"])
def test_empty_set_spellings(given):
    assert _correct("set", _EMPTY, given)


@pytest.mark.parametrize("given", ["{0}", "0", "{∅}"])
def test_not_the_empty_set(given):
    assert not _correct("set", _EMPTY, given)


# -- semicolons in an ordered list ---------------------------------------------------------

_SEQUENCE = _q([3, 5, 8, 13], "3, 5, 8, 13")


@pytest.mark.parametrize("given", ["3; 5; 8; 13", "3;5;8;13"])
def test_ordered_list_with_semicolons(given):
    assert _correct("ordered_list", _SEQUENCE, given)


def test_ordered_list_with_semicolons_keeps_order():
    assert not _correct("ordered_list", _SEQUENCE, "13; 8; 5; 3")
