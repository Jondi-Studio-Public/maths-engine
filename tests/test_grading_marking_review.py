"""Marking correctness on messy typed replies: a right answer marked right and a wrong one marked wrong.

Each case here was marked the other way before: a reply with two minus signs, an unfactored expression in a
bracket for a factorising question, a surd that skipped the question's form through its decimal alternative,
the constant of integration as "c" or "K", square brackets and braces, a decimal power for a root, the empty set
as "∅", and an ordered list separated by semicolons."""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy

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


@pytest.mark.parametrize(
    "answer, given",
    [
        ("x/3", "x/3.0"),
        ("x/3", "x/3."),
        ("pi/3", "pi/3.0"),
        ("1/6", "1/6.0"),
        ("1/6", "0.5/3"),
        ("2/3", "2/3."),
        ("2/3", "2.0/3"),
        ("x**3/3 + C", "x^3/3.0 + c"),
    ],
)
def test_a_whole_number_written_as_a_decimal_is_read_exactly(answer, given):
    """The decimal is read as written, before sympy divides by it: 1/3.0 is a third, not 0.333... made exact."""
    assert _correct("expression", _q(answer, answer), given)


def test_a_decimal_divisor_that_is_not_the_value_is_wrong():
    assert not _correct("expression", _q("x/3", "x/3"), "x/3.1")


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


# -- an angle typed with its unit, in an expression answer -------------------------------

_ANGLE = _q("30", "30")


@pytest.mark.parametrize("given", ["30", "30°", "30 °", "30º", "30 degrees", "30 Degrees", "30 deg"])
def test_angle_with_its_unit(given):
    assert _correct("expression", _ANGLE, given)


@pytest.mark.parametrize("given", ["31°", "30 radians", "°30"])
def test_wrong_angle_with_a_unit_is_still_wrong(given):
    assert not _correct("expression", _ANGLE, given)


def test_degree_sign_is_not_dropped_from_a_symbolic_answer():
    assert not _correct("expression", _q("x", "x"), "x°")


# -- a vector in component form for a pair answer -----------------------------------------

_VECTOR3 = _q({"x": -4, "y": -3, "z": 0}, "(-4, -3, 0)")


@pytest.mark.parametrize(
    "given", ["-4i-3j+0k", "-4i - 3j + 0k", "−4i − 3j + 0k", "-4 i - 3 j + 0 k", "-3j - 4i", "-4i - 3j"]
)
def test_component_form_spaced_or_not(given):
    assert _correct("pair", _VECTOR3, given)


@pytest.mark.parametrize("given", ["-4i + 3j", "-4i - 3j + k", "-4i - 3i", "4i - 3j", "-4i - 3j + 0k + 2"])
def test_wrong_component_form_is_wrong(given):
    assert not _correct("pair", _VECTOR3, given)


@pytest.mark.parametrize("given", ["i - 2j", "1i - 2j", "-2j + i", "i - 2j + 0k"])
def test_unit_coefficient_and_missing_k(given):
    assert _correct("pair", _q({"x": 1, "y": -2}, "(1, -2)"), given)


def test_a_k_component_is_not_dropped_from_a_two_component_answer():
    assert not _correct("pair", _q({"x": 1, "y": -2}, "(1, -2)"), "i - 2j + k")


# -- the question pasted back ----------------------------------------------------------------


def _asked(answer, answer_text, text):
    question = _q(answer, answer_text)
    question.text = text
    return question


_SCALARS = _asked(
    -2,
    "-2",
    "The vectors u = -5i + 2j and w = -3i + 2j satisfy p·u + q·w = i - 2j, where p and q are scalars. "
    "Find the value of q.",
)


@pytest.mark.parametrize("given", [_SCALARS.text, "p·u + q·w = i - 2j", "satisfy p·u + q·w = i - 2j"])
def test_question_pasted_back_is_wrong(given):
    assert not _correct("number", _SCALARS, given)


@pytest.mark.parametrize("given", ["-2", "q = -2"])
def test_answer_to_that_question_is_right(given):
    assert _correct("number", _SCALARS, given)


def test_equation_copied_from_the_question_is_wrong():
    question = _asked(4, "x = 4", "Solve 2x/2 = 4")
    assert not _correct("number", question, "2x/2 = 4")
    assert _correct("number", question, "x = 4")


@pytest.mark.parametrize(
    "answer, answer_text, text, given",
    [
        (9, "9", "Find the area of a 3 by 3 square.", "area = 3 x 3 = 9"),
        (3, "x = 3", "Solve 3x = 9", "3x = 9, x = 3"),
    ],
)
def test_working_that_restates_part_of_the_question_is_still_right(answer, answer_text, text, given):
    assert _correct("number", _asked(answer, answer_text, text), given)


def test_a_coordinate_answer_is_still_read():
    """The degree check reads the expected answer, which can be a tuple rather than a number."""
    assert _correct("expression", _q("(1, 2)", "(1, 2)"), "(1, 2)")


def test_a_common_factor_is_allowed_when_the_canonical_keeps_one():
    """A question about the split into a linear and a quadratic factor, not about the common factor."""
    question = _q("(3*x - 3)*(x**2 + 9)", "(3x - 3)(x^2 + 9)", answer_form="factored")
    for given in ["(3x - 3)(x^2 + 9)", "3(x - 1)(x^2 + 9)", "(x - 1)(3x^2 + 27)"]:
        assert _correct("expression", question, given)
    assert not _correct("expression", question, "(3x^3 - 3x^2 + 27x - 27)")


# -- infinity in words, for an answer shown as ∞ -------------------------------------------


@pytest.mark.parametrize("given", ["∞", "+∞", "infinity", "Infinity", "inf", "oo", "+infinity"])
def test_infinity_in_words(given):
    assert _correct("expression", _q("∞", "∞"), given)


@pytest.mark.parametrize("given", ["-∞", "-infinity", "∞∞", "infinity squared", "2"])
def test_not_infinity_is_wrong(given):
    assert not _correct("expression", _q("∞", "∞"), given)


def test_negative_infinity_in_words():
    assert _correct("expression", _q("-∞", "-∞"), "-infinity")
    assert not _correct("expression", _q("-∞", "-∞"), "infinity")


# -- an exact fraction answer, with the variable named ---------------------------------------


@pytest.mark.parametrize("answer", [Fraction(50, 7), sympy.Rational(50, 7), {"p": 50, "q": 7}])
@pytest.mark.parametrize("kind", ["number", "fraction"])
@pytest.mark.parametrize("given", ["x = 50/7", "x=50/7", "50/7", "7 1/7", "100/14"])
def test_exact_fraction_answer_with_or_without_the_variable(answer, kind, given):
    assert _correct(kind, _q(answer, "x = 50/7"), given)


@pytest.mark.parametrize("answer", [Fraction(50, 7), sympy.Rational(50, 7), {"p": 50, "q": 7}])
@pytest.mark.parametrize("kind", ["number", "fraction"])
@pytest.mark.parametrize("given", ["x = 50/8", "7.14", "x = 7", "x = -50/7"])
def test_exact_fraction_answer_wrong_values(answer, kind, given):
    assert not _correct(kind, _q(answer, "x = 50/7"), given)


def test_a_labelled_reply_needs_a_labelled_canonical_with_the_same_variable():
    assert not _correct("fraction", _q({"p": 7, "q": 1}, "7"), "x = 7")
    assert not _correct("fraction", _q({"p": 50, "q": 7}, "x = 50/7"), "y = 50/7")


def test_degree_sign_is_not_dropped_from_an_answer_in_radians():
    assert not _correct("expression", _q("9*pi/4", "9π/4"), "9*pi/4°")
    assert _correct("expression", _q("9*pi/4", "9π/4"), "9pi/4")


# -- an exact answer is marked exactly, not as a half-unit range ----------------------


@pytest.mark.parametrize(
    "answer, answer_text, given",
    [
        (7.7 - 5.1, "2.6", "2.64"),  # float noise: 2.6000000000000005
        (7.7 - 5.1, "2.6", "2.56"),
        (0.3, "0.3", "0.34"),
        (0.5, "0.5", "0.51"),
        (19.05, "19.05", "19.054"),
        (60, "F = 60.00 N", "60.004"),
    ],
)
def test_near_miss_on_an_exact_answer_is_wrong(answer, answer_text, given):
    assert not _correct("number", _q(answer, answer_text), given)


@pytest.mark.parametrize(
    "answer, answer_text, given",
    [
        (7.7 - 5.1, "2.6", "2.6"),
        (7.7 - 5.1, "2.6", "2.60"),
        (0.5, "0.5", ".5"),
        (60, "F = 60.00 N", "60"),
        (19.05, "19.05", "$19.05"),
    ],
)
def test_exact_answer_in_any_form_still_right(answer, answer_text, given):
    assert _correct("number", _q(answer, answer_text), given)


def test_rounded_answer_keeps_half_a_unit():
    # 1/3 shown as 0.33 is a rounding, not the value, so a more precise reply is still right.
    question = _q(1 / 3, "0.33")
    assert _correct("number", question, "0.333")
    assert _correct("number", question, "0.33")
    assert not _correct("number", question, "0.34")


# -- a multiple-choice reply passed as an int, as the docs say -------------------------


def test_mcq_index_as_an_int_is_read():
    from mathgen.core import mcq_options

    question = _q("4", "4", distractors=["3", "5", "6"])
    index = mcq_options(question).index("4")
    assert _correct("mcq", question, index)
    assert _correct("mcq", question, str(index))
    assert not _correct("mcq", question, (index + 1) % 4)
    assert not _correct("mcq", question, True)


# -- units on a typed number ------------------------------------------------------------


def _with_text(text, answer, answer_text):
    return Question(topic="test", level=0, seed=1, text=text, latex="", answer=answer, answer_text=answer_text)


_CM_TO_MM = _with_text("Convert 3 cm to mm.", 30, "30")


@pytest.mark.parametrize("given", ["30 kg", "30 furlongs", "30 cm", "30 km"])
def test_wrong_unit_is_wrong(given):
    assert not _correct("number", _CM_TO_MM, given)


@pytest.mark.parametrize("given", ["30", "30 mm", "30mm", "30 millimetres", "30 MM"])
def test_asked_unit_or_none_is_right(given):
    assert _correct("number", _CM_TO_MM, given)


def test_unit_shown_by_the_canonical_is_the_one_asked():
    question = _with_text(
        "A spring with stiffness k = 500 N/m is stretched by 0.05 m. Find the force.", 25.0, "F = 25.00 N"
    )
    assert _correct("number", question, "25 newtons")
    assert not _correct("number", question, "25 kg")


def test_a_unit_that_only_describes_the_data_is_not_the_one_asked():
    question = _with_text(
        "The model y = -2 ln(x) + 33 gives the time, in seconds, a worker takes after x hours of practice. "
        "Find x (hours of practice) when y = 28, to 1 decimal place.",
        12.2,
        "12.2",
    )
    assert _correct("number", question, "12.2 hours")


def test_nearest_cent_is_a_precision_not_the_unit():
    question = _with_text("Find the balance. Give your answer to the nearest cent.", 4988.28, "4988.28")
    assert _correct("number", question, "4988.28 dollars")


def test_vector_component_is_not_joules():
    question = _with_text("Find the j-component of the velocity, in m/s, at t = 1.", -4, "-4")
    assert _correct("number", question, "-4 m/s")
    assert not _correct("number", question, "-4 km/h")


# -- a variable label on a fraction answer ----------------------------------------------


def test_fraction_label_naming_the_question_variable_is_dropped():
    question = Question(
        topic="test", level=0, seed=1, text="Find x if 5^x = 125.", latex="", answer={"p": 3, "q": 1}, answer_text="3"
    )
    assert _correct("fraction", question, "x = 3")
    assert _correct("fraction", question, "3")
    assert not _correct("fraction", question, "y = 3")
    assert not _correct("fraction", question, "x = 4")
