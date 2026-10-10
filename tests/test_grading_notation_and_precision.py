"""Learner notation in typed expressions, and stated precision in pair and
list answers.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen.core import Question

from mathgen.grading import typed as grading


def _q(answer, answer_text, text="q") -> Question:
    return Question(topic="test", level=0, seed=1, text=text, latex="q", answer=answer, answer_text=answer_text)


def _correct(kind, q, given) -> bool:
    return grading.grade(kind, q, given).correct


# -- expression: learner notation -------------------------------------------


@pytest.mark.parametrize("given", ["6*pi*i", "6πi", "6 π i", "6*pi*I"])
def test_imaginary_unit_and_pi_as_typed(given):
    assert _correct("expression", _q("6*pi*I", "6*pi*I"), given)


@pytest.mark.parametrize("given", ["6*pi", "6*pi*j", "-6πi"])
def test_imaginary_unit_wrong_answers_stay_wrong(given):
    assert not _correct("expression", _q("6*pi*I", "6*pi*I"), given)


def test_i_stays_a_variable_when_the_answer_is_real():
    # No I in the expected expression: a learner's i is not the imaginary unit.
    q = _q("3*i + 4", "3*i + 4")
    assert _correct("expression", q, "3i + 4")
    assert not _correct("expression", q, "3I + 4")


@pytest.mark.parametrize("given", ["-5e^(2x)", "-5*e^(2*x)", "y = -5e^(2x)", "-5exp(2x)"])
def test_e_power_reads_as_exp(given):
    assert _correct("expression", _q("-5*exp(2*x)", "y = -5*exp(2*x)"), given)


def test_e_power_wrong_answer_stays_wrong():
    assert not _correct("expression", _q("-5*exp(2*x)", "y = -5*exp(2*x)"), "-5e^(3x)")


def test_e_stays_a_variable_when_the_answer_uses_it():
    q = _q("2*e + 1", "2e + 1")
    assert _correct("expression", q, "2e + 1")
    assert not _correct("expression", q, "2*exp(1) + 1")


@pytest.mark.parametrize("given", ["2·x + 1", "2×x + 1", "2⋅x + 1"])
def test_multiplication_signs(given):
    assert _correct("expression", _q("2*x + 1", "2x + 1"), given)


@pytest.mark.parametrize("given", ["ln|x| + C", "ln(|x|) + C", "log(Abs(x)) + C"])
def test_absolute_value_bars(given):
    assert _correct("expression", _q("log(Abs(x)) + C", "ln|x| + C"), given)


def test_bars_never_silently_become_a_different_expression():
    # "ln|x| + C" used to parse, via the spoken-word fallback, as `x`.
    assert not _correct("expression", _q("x", "x"), "ln|x| + C")
    assert grading._parse_expression("ln|x") is None


@pytest.mark.parametrize(
    "text, parsed",
    [
        ("|x| + |y|", "Abs(x) + Abs(y)"),
        ("||x| - 1|", "Abs(Abs(x) - 1)"),
        ("|x||y|", "Abs(x)*Abs(y)"),
        ("2|x - 1|", "2*Abs(x - 1)"),
        ("3ln|2x - 1|", "3*log(Abs(2*x - 1))"),
    ],
)
def test_bar_pairing(text, parsed):
    assert str(grading._parse_expression(text)) == parsed


def test_ln_abs_against_ln_is_unchanged_policy():
    assert not _correct("expression", _q("log(x) + C", "ln(x) + C"), "ln|x| + C")


# -- pair: stated precision ---------------------------------------------------


def _volume_area():
    return _q({"volume": 12.34, "area": 5.67}, "volume = 12.34, area = 5.67")


@pytest.mark.parametrize(
    "given",
    ["volume = 12.34, area = 5.67", "area = 5.67, volume = 12.34", "12.34, 5.67", "volume = 12.340, area = 5.67"],
)
def test_pair_at_stated_precision_accepted(given):
    assert _correct("pair", _volume_area(), given)


@pytest.mark.parametrize(
    "given", ["volume = 12.39, area = 5.67", "volume = 12.34, area = 5.72", "12.3, 5.67", "area = 5.63, volume = 12.34"]
)
def test_pair_outside_stated_precision_rejected(given):
    # All of these were accepted by the flat ±0.051 tolerance.
    assert not _correct("pair", _volume_area(), given)


def test_pair_unrounded_value_within_half_a_unit_accepted():
    q = _q({"r": 3.6056, "theta": 56.3}, "r = 3.61, theta = 56.3")
    assert _correct("pair", q, "r = 3.6056, theta = 56.31")
    assert not _correct("pair", q, "r = 3.62, theta = 56.3")


def test_proportion_interval_is_graded_to_its_hundredths():
    q = _q({"lower": 0.31, "upper": 0.49}, "lower = 0.31, upper = 0.49")
    assert _correct("pair", q, "lower = 0.31, upper = 0.49")
    assert not _correct("pair", q, "lower = 0.35, upper = 0.49")
    assert not _correct("pair", q, "lower = 0.31, upper = 0.53")


def test_pair_integer_component_is_exact():
    q = _q({"x": 3, "y": -6}, "x = 3, y = -6")
    assert _correct("pair", q, "x = 3.0, y = -6")
    assert not _correct("pair", q, "x = 3.04, y = -6")


def test_pair_fraction_component_is_exact():
    q = _q({"x": Fraction(-4, 21), "y": Fraction(4, 21)}, "x = -4/21, y = 4/21")
    assert _correct("pair", q, "x = -8/42, y = 4/21")
    assert not _correct("pair", q, "x = -0.19, y = 0.19")


def test_pair_with_units_and_brackets():
    assert _correct("pair", _q({"R_A": 2.5, "R_B": 7.5}, "R_A = 2.5 kN, R_B = 7.5 kN"), "R_B = 7.5 kN, R_A = 2.5 kN")
    assert not _correct("pair", _q({"R_A": 2.5, "R_B": 7.5}, "R_A = 2.5 kN, R_B = 7.5 kN"), "R_A = 2.56, R_B = 7.5")
    assert _correct("pair", _q({"x": 63, "y": 81}, "(63, 81)"), "(63, 81)")
    assert not _correct("pair", _q({"x": 63, "y": 81}, "(63, 81)"), "(63.04, 81)")


def test_pair_without_a_readable_literal_keeps_the_old_tolerance():
    # "96°36′" is one segment for two components: nothing to read precision
    # from, so the transcript tolerance still applies.
    q = _q({"degrees": 96, "minutes": 36}, "96°36′")
    assert _correct("pair", q, "96°36′")
    assert _correct("pair", q, "96°36.04′")


# -- list: stated precision ---------------------------------------------------


def test_list_at_stated_precision():
    q = _q([-2.62, -0.38], "x = -2.62 or x = -0.38")
    assert _correct("list", q, "x = -0.38 or x = -2.62")
    assert _correct("list", q, "-2.62, -0.38")
    assert not _correct("list", q, "-2.6, -0.4")
    assert not _correct("list", q, "-2.66, -0.38")


def test_list_of_integers_is_exact():
    q = _q([1, 17], "1, 17")
    assert _correct("list", q, "17, 1")
    assert not _correct("list", q, "1.04, 17")


# -- expression: text too costly to evaluate ----------------------------------


@pytest.mark.parametrize(
    "given",
    ["2^2^2^2^2^2^2^2", "2^(2^(2^(2^(2^5))))", "9^9999999", "2^-99999999", "99999!", "x^(10^(10^10))"],
)
def test_answers_that_would_hang_the_parser_are_marked_wrong_quickly(given):
    # sympy evaluates integer powers and factorials while parsing; each of these
    # used to hang the request.
    import time

    start = time.perf_counter()
    assert not _correct("expression", _q("2*x", "2x"), given)
    assert time.perf_counter() - start < 2


@pytest.mark.parametrize(
    "given, answer",
    [("2^3^2", "512"), ("(2^3)^2", "64"), ("x^2^3", "x**8"), ("(x^2)^3", "x**6"), ("5!", "120"), ("10^4", "10000")],
)
def test_ordinary_powers_and_factorials_still_parse(given, answer):
    assert _correct("expression", _q(answer, answer), given)


# a pair typed without a space after the comma ----------


@pytest.mark.parametrize("given", ["316,1564", "316, 1564", "316 , 1564"])
def test_pair_typed_without_a_space(given):
    assert _correct("pair", _q({"n": 316, "N": 1564}, "n = 316, N = 1564"), given)


def test_pair_of_three_digit_values_typed_without_a_space():
    # "316,156" looks like one grouped numeral, but a pair needs two values.
    assert _correct("pair", _q({"n": 316, "N": 156}, "n = 316, N = 156"), "316,156")
    assert not _correct("pair", _q({"n": 316, "N": 156}, "n = 316, N = 156"), "316,157")


def test_grouped_numerals_still_read_as_one_value():
    assert _correct("pair", _q({"a": 1234, "b": 5}, "a = 1234, b = 5"), "1,234, 5")
    assert grading._strip_comma_groups("316,1564") == "316,1564"
    assert grading._strip_comma_groups("12,345,678") == "12345678"
