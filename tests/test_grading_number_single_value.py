"""A typed `number` reply states one value, and that value must be the answer.

The grader used to read every number out of the reply and accept it when ANY of them matched, so "5 9",
"9 or 7", "not 9" and a counted run "0 1 2 ... 29 9" were all correct for 9: a skill could be passed by
listing numbers. `grade_number` now needs exactly one number after the normalisation it already did, and
reads a negation, a comparison or a "plus or minus" as not stating the answer at all.

The accept table is the other half of the contract: every ordinary way of writing the answer stays correct.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from mathgen.core import Question
from mathgen.grading import grade


def _question(answer, answer_text=None, text="Work it out.") -> Question:
    shown = answer_text if answer_text is not None else str(answer)
    return Question(
        topic="test",
        level=0,
        seed=1,
        text=text,
        latex=text,
        answer=answer,
        answer_text=shown,
        answer_latex=shown,
        working=[],
        difficulty="easy",
        distractors=[],
    )


def _correct(answer, reply, answer_text=None) -> bool:
    return grade("number", _question(answer, answer_text), reply).correct


# -- replies that are still right ---------------------------------------------------------------

_NINE = [
    "9",
    " 9 ",
    "9.",
    "9.0",
    "9.00",
    "+9",
    "09",
    "x = 9",
    "x=9",
    "answer: 9",
    "The answer is 9",
    "it is 9.",
    "9 cm",
    "9cm",
    "9 cm^2",
    "9 cm^2.",
    "9cm2",
    "9 m/s^2",
    "9 m s^-1",
    "9 m^(-1)",
    "9 m3",
    "9 sq cm",
    "9 cm squared",
    "$9",
    "$9.00",
    "9 dollars",
    "9 AUD",
    "A$9",
    "9%",
    "9 degrees",
    "9°",
    "nine",
    "Nine.",
    "NINE",
    "nine apples",
    "9 apples.",
    "9 ones",
    "x = nine",
    "f(2) = 9",
    "f(0)=9",
    "area = 3 x 3 = 9",
    "3 × 3 = 9",
    "y = 2x + 3 = 9",
    "about 9",
    "~9",
    "approximately 9",
    "9 (approx)",
    "9?",
    "(9)",
    "[9]",
    "9 o'clock",
    "9th",
    "18/2",
    "9/1",
    "9 cm",
    " 9",
]


@pytest.mark.parametrize("reply", _NINE)
def test_ordinary_ways_of_writing_nine_are_correct(reply):
    assert _correct(9, reply) is True


@pytest.mark.parametrize(
    "answer, reply",
    [
        (-9, "-9"),
        (-9, "−9"),  # the unicode minus
        (-9, "–9"),  # an en dash typed for it
        (-9, "minus nine"),
        (-9, "negative nine"),
        (-9, "(-9)"),
        (-9, "x = -9"),
        (-9, "-9 m"),
        (1000, "1000"),
        (1000, "1,000"),
        (1000, "$1,000"),
        (1000, "1,000 m"),
        (1000, "one thousand"),
        (9.5, "9.5"),
        (9.5, "9.50"),
        (9.5, "nine point five"),
        (9.5, "19/2"),
        (0, "0"),
        (0, "zero"),
        (0.25, "0.25"),
        (0.25, "1/4"),
        (0.25, "a quarter"),
    ],
)
def test_other_targets_keep_every_ordinary_spelling(answer, reply):
    assert _correct(answer, reply) is True


# -- replies that are not -----------------------------------------------------------------------

_NOT_NINE = [
    # more than one number: even when one of them is the answer
    "5 9",
    "9 5",
    "9 or 7",
    "9, 7",
    "9 and 7",
    "5-9",
    "9 - 5",
    "2 + 2",
    "9,5",
    " ".join(str(i) for i in range(30)) + " 9",
    ", ".join(str(i) for i in range(30)) + ", 9",
    "9 " * 40,
    # negated or hedged
    "not 9",
    "Not 9",
    "isn't 9",
    "it isn’t 9",
    "is not 9",
    "never 9",
    "neither 9",
    "cannot be 9",
    "±9",
    "+-9",
    "plus or minus 9",
    "plus-minus 9",
    ">9",
    "<9",
    "9 > 8",
    "≥9",
    "≤9",
    "≠9",
    "9 != 7",
    # not one plain number
    "x^2",
    "10^3",
    "1e3",
    "9e0",
    "",
    "   ",
    "x",
    "nine nine",
    "nine or seven",
    # a call is not a unit: its bracket and argument are numbers
    "prime(10**9)",
    "fibonacci(10**9)",
    "f(10)(9)",
]


@pytest.mark.parametrize("reply", _NOT_NINE)
def test_a_reply_that_is_not_one_value_is_wrong(reply):
    assert _correct(9, reply) is False


@pytest.mark.parametrize(
    "answer, reply",
    [
        (9, "8"),
        (9, "10"),
        (9, "-9"),
        (9, "nine hundred"),
        (-9, "9"),
        (-9, "minus nine or nine"),
        (1000, "1,001"),
        (1000, "100"),
        (9.5, "9"),
        (9.5, "9.6"),
        (0, "1"),
        (0, "0 or 1"),
    ],
)
def test_a_wrong_single_value_is_still_wrong(answer, reply):
    assert _correct(answer, reply) is False


def test_a_long_run_of_numbers_is_wrong_for_every_answer():
    guess = " ".join(str(i) for i in range(200))
    for answer in range(0, 200, 7):
        assert _correct(answer, guess) is False


def test_the_unit_exponent_rule_does_not_hide_a_second_value():
    # "m3" is a unit, "9 3" and "9 m 3" are two numbers.
    assert _correct(9, "9 3") is False
    assert _correct(9, "9 m 3") is False
    assert _correct(9, "3 9 m2") is False
    assert _correct(3, "3 m2") is True
    assert _correct(9, "9 m^(-1)") is True
    assert _correct(9, "9 m(2)") is False  # a bracket after a letter is not a unit exponent


# -- a numeric answer a generator writes as text ------------------------------------------------
#
# 22 questions across eight senior skills carry `answer="10"` (a str), which was graded by the spoken
# transcript reader instead of the number grader, so "not 10" was correct there. They are numbers.


@pytest.mark.parametrize("answer", ["10", "3.5", "0.116", "1306"])
def test_a_numeric_answer_written_as_text_is_graded_as_a_number(answer):
    question = _question(answer)
    assert grade("number", question, answer).correct is True
    assert grade("number", question, f"{answer} degrees").correct is True
    assert grade("number", question, f"x = {answer}").correct is True
    assert grade("number", question, f"not {answer}").correct is False
    assert grade("number", question, f"5 {answer}").correct is False
    assert grade("number", question, f"{answer} or 7").correct is False
    assert grade("number", question, "0 1 2 3 4 5 6 7 8 9 " + answer).correct is False


def test_a_fraction_answer_written_as_text_is_graded_exactly():
    question = _question("3/20", "3/20")
    assert grade("fraction", question, "3/20").correct is True
    assert grade("fraction", question, "6/40").correct is True  # same value
    assert grade("fraction", question, "not 3/20").correct is False
    assert grade("fraction", question, "1/7 3/20").correct is False
    assert grade("fraction", question, "3/21").correct is False
    assert grade("fraction", question, "2/3").correct is False


# -- the pair and list graders: "not (28, 24)" holds exactly the right numbers ---------------------


def test_a_negated_pair_is_wrong():
    question = _question({"a": 28, "b": 24}, "(28, 24)")
    assert grade("pair", question, "(28, 24)").correct is True
    assert grade("pair", question, "28, 24").correct is True
    for reply in ("not (28, 24)", "never (28, 24)", "it isn't (28, 24)", "(28, 24) is not it"):
        assert grade("pair", question, reply).correct is False, reply


def test_a_negated_list_is_wrong():
    question = _question([3, 6], "x = 3 and x = 6")
    assert grade("list", question, "x = 3 and x = 6").correct is True
    assert grade("list", question, "3, 6").correct is True
    assert grade("list", question, "6 and 3").correct is True
    for reply in ("not x = 3 and x = 6", "never 3 and 6", "3 and 6 are not the answer", "3, 6 != answer"):
        assert grade("list", question, reply).correct is False, reply


def test_the_negation_check_stands_down_when_the_answer_itself_uses_the_sign():
    question = _question([2, 5], "x < 2 or x > 5")
    assert grade("list", question, "x < 2 or x > 5").correct is True


# -- a minus is a sign wherever it is not a subtraction ----------------------------------------
#
# `extract_values` keeps a minus only when it touches the first digit, so "- 9" and "-$542.00" were read as
# the positive number: a learner's -9 was marked wrong for 9's question and right for -9's, the other way round.

_SPACED_MINUS = ["- 9", "− 9", "−9", "– 9", "— 9", "-  9", "-\t9", "x = - 9", "(- 9)", "-9 cm", "- 9 cm", "minus 9"]


@pytest.mark.parametrize("reply", _SPACED_MINUS)
def test_a_minus_followed_by_a_space_is_a_sign(reply):
    assert _correct(-9, reply) is True
    assert _correct(9, reply) is False


@pytest.mark.parametrize(
    "reply", ["-$542.00", "−$542.00", "- $542.00", "-$ 542", "$-542.00", "$ - 542", "−$542", "-£542", "-€ 542"]
)
def test_a_negative_amount_keeps_its_sign(reply):
    owed = _question(-542.0, "-$542.00")
    paid = _question(542.0, "$542.00")
    assert grade("number", owed, reply).correct is True
    assert grade("number", paid, reply).correct is False


@pytest.mark.parametrize("reply", ["$542.00", "$ 542", "$542", "542", "542.00", "542 dollars", "A$542.00"])
def test_a_positive_amount_is_not_a_negative_one(reply):
    assert grade("number", _question(542.0, "$542.00"), reply).correct is True
    assert grade("number", _question(-542.0, "-$542.00"), reply).correct is False


def test_a_minus_after_a_digit_a_letter_or_a_bracket_is_still_a_subtraction():
    for reply in ("5-9", "5 -9", "5 - 9", "(5)-9", "5.-9"):
        assert _correct(9, reply) is False, reply
        assert _correct(-9, reply) is False, reply
        assert _correct(-4, reply) is False, reply


# -- right replies that were turned away: "no" is not a negation, and notes, units and currency are not values --

_ACCEPTED = [
    # "no" is not a negation: it is in "no units", "No. of sides" and "no, it's 9"
    (9, "9 (no units)"),
    (0, "0 (no real solutions)"),
    (1, "1 (no remainder)"),
    (9, "No. of sides = 9"),
    (9, "no, it's 9"),
    (9, "no it is 9"),
    (9, "nope, 9"),
    (9, "9, no units"),
    # a precision note says nothing about the value
    (9.5, "9.5 (1 dp)"),
    (9.5, "9.5 (to 1 d.p.)"),
    (9.5, "9.5 to 1 decimal place"),
    (9.5, "9.5 (2 s.f.)"),
    (1200, "1200 (2 s.f.)"),
    (9.5, "9.5 (1 d.p.)"),
    (9.5, "9.5 (1 decimal place)"),
    (9.5, "9.5 to one decimal place"),
    (9.5, "9.5 correct to 1 decimal place"),
    (9.5, "9.5 (3 sig figs)"),
    (9.5, "9.5 (2 significant figures)"),
    (1200, "1200 to 2 significant figures"),
    (9.5, "9.5 (1dp)"),
    (1200, "1200 (2sf)"),
    (1200, "1200 (2 s.f)"),
    (9.5, "x = 9.5 (1 dp)"),
    # and so does a bracketed remark with no digit in it
    (9.5, "9.5 (rounded)"),
    (9.5, "9.5 (approx.)"),
    (9, "9 (units)"),
    (9, "9 (nine)"),
    (9, "(nine)"),
    # a number written straight after letters is the number, not the exponent of a unit
    (9, "AUD9"),
    (9, "AUD 9"),
    (9.5, "AUD9.50"),
    (9, "Rs9"),
    (95, "Rs95"),
    (5, "A$5"),
    (-5, "minus5"),
    (-5, "minus 5"),
    (99, "AUD99"),
    (9, "$9"),
    # but an exponent after the first number is still a unit's
    (9, "9 cm2"),
    (9, "9cm2"),
    (9, "AUD9 cm2"),
    (9, "9 m/s2"),
    (9, "9 kg m s-2"),
    # a decimal with no leading zero
    (0.5, ".5"),
    (0.5, ".50"),
    (0.5, "$.50"),
    (0.25, ".25"),
    (-0.5, "-.5"),
    (-0.5, "−.5"),
    (-0.5, "- .5"),
    (9.5, "9.5"),
    (0.5, "x = .5"),
    # a unit written in LaTeX has its exponent dropped like any other
    (9, "9\\ \\text{cm}^2"),
    (9, "9\\text{ cm}^2"),
    (9, "9\\,\\text{cm}^2"),
    (9, "9 \\text{cm}^2"),
    (9, "9 \\mathrm{m}^3"),
    (9, "9\\text{ m}^2."),
    (9, "$9\\ \\text{cm}^2$"),
    (9, "9\\text{cm}2"),
    (9, "area = 9\\ \\text{cm}^2"),
    (-9, "-9\\ \\text{cm}^2"),
]


@pytest.mark.parametrize("answer, reply", _ACCEPTED)
def test_a_legitimate_reply_that_was_turned_away_is_accepted(answer, reply):
    assert _correct(answer, reply) is True


_STILL_REFUSED = [
    (5, ".5"),
    (0.5, "5"),
    (-0.5, ".5"),
    (0.5, "-.5"),
    (0.5, "−.5"),
    (9, "9 (or 7)"),
    (9, "9 (not 7)"),
    (9, "9 (2 dp) 5"),
    (9.5, "9.5 (1 dp) 3"),
    (9.55, "9.5 (2 dp)"),
    (9, "no 9 or 7"),
    (9, "AUD9 7"),
    (9, "9 m2 7"),
    (12345, "12 345"),
    (9.5, "9 1/2"),
    (1000, "1e3"),
    (1000, "1.0e3"),
    (9, "9 \\text{ cm}^{2}"),  # a braced exponent is not read, braced or not
    (5, "{5}^2"),  # the "}" closes no unit
    (9, "9 {cm}^2"),
    (2, "{5}^2"),
    (9, "+-9"),
    (9, "9 +- 1"),
    (9, "9 +/- 1"),
    (9, "±9"),
    (9, "9 plus or minus 1"),
]


@pytest.mark.parametrize("answer, reply", _STILL_REFUSED)
def test_what_must_stay_wrong_is_still_wrong(answer, reply):
    assert _correct(answer, reply) is False


def test_a_unit_exponent_only_counts_after_the_first_number():
    # "m2" before the number is the number 2 (it is how "AUD9" and "Rs9" keep their 9), so two values here.
    assert _correct(9, "m2 9") is False
    assert _correct(2, "m2") is True
    assert _correct(9, "9 m2") is True
    assert _correct(9, "nine m2") is True
    assert _correct(9, "nine m2 m3") is True
    assert _correct(9, "nine 2") is False


def test_a_leading_point_decimal_is_checked_for_the_rounding_asked():
    question = _question(0.3, "0.3", text="Round 0.26 to the nearest tenth.")
    assert grade("number", question, ".3").correct is True
    assert grade("number", question, "0.3").correct is True
    assert grade("number", question, ".26").correct is False  # not rounded
    assert grade("number", question, "0.26").correct is False


def test_a_plus_and_a_minus_in_an_equation_are_not_plus_or_minus():
    question = _question({"m": 2, "b": -3}, "y = 2x - 3")
    assert grade("pair", question, "y=2x+-3").correct is True
    assert grade("pair", question, "y = 2x + -3").correct is True
    assert grade("pair", question, "y = 2x - 3").correct is True
    assert grade("pair", question, "y = 2x +/- 3").correct is False
    assert grade("pair", question, "y = 2x ± 3").correct is False
    assert grade("pair", question, "y = 2x + 3").correct is False


# -- a question is left as it was given (it may not be a Question at all) ------------------------


@pytest.mark.parametrize("answer, text", [("10", "10"), ("3.5", "3.5"), (7, "7"), (7.25, "7.25"), (-2, "-2")])
def test_a_duck_typed_question_is_graded(answer, text):
    question = SimpleNamespace(answer=answer, answer_text=text)
    assert grade("number", question, text).correct is True
    assert grade("number", question, f"x = {text}").correct is True
    assert grade("number", question, "11").correct is False
    assert grade("number", question, f"not {text}").correct is False


def test_grading_a_text_answer_does_not_change_the_question():
    question = _question("10", "10")
    before = dataclasses.asdict(question)
    assert grade("number", question, "10").correct is True
    assert dataclasses.asdict(question) == before
    assert question.answer == "10"
