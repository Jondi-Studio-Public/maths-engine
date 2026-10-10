"""The minus sign: every typed-answer kind reads the typographic minus and its lookalikes as "-".

A phone keyboard, a word processor and a pasted worksheet type U+2212 (the real minus sign) where a person
means "-". Only the number grader translated it. The others read it as nothing, so a minus was lost
(`x = 3, y = -6` typed with U+2212 was accepted for y = 6, `{-1, 2, 3}` for {1, 2, 3}) or the whole reply was
unreadable (`-3/4`, `-x`, `x - 2`, `-3, 1, 2`). `grade()` now spells the sign "-" before any grader reads the
reply, for every kind but a multiple-choice one; the en and em dashes become a minus where they are a sign
(and, in a number or an expression, anywhere: a range is not a thing those say). The spoken readers in
`mathgen.grading.spoken` are not part of this and read what they always read.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mathgen.core import Question
from mathgen.grading import grade, typed
from mathgen.grading.spoken import extract_values

MINUS = "−"  # U+2212 MINUS SIGN
SMALL = "﹣"  # U+FE63 SMALL HYPHEN-MINUS
FULL = "－"  # U+FF0D FULLWIDTH HYPHEN-MINUS
EN = "–"  # U+2013 EN DASH
EM = "—"  # U+2014 EM DASH
SIGNS = [MINUS, SMALL, FULL]


def _question(answer, answer_text, **fields) -> Question:
    return Question(
        topic="test",
        level=0,
        seed=1,
        text="Work it out.",
        latex="Work it out.",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[],
        difficulty="easy",
        distractors=[],
        **fields,
    )


def _ok(kind, question, reply) -> bool:
    return grade(kind, question, reply).correct


# -- the translation itself ---------------------------------------------------------------------


def test_the_minus_and_its_lookalikes_become_a_hyphen_minus():
    for sign in SIGNS:
        assert typed._spell_minus_as_ascii("pair", f"x = 3, y = {sign}6") == "x = 3, y = -6"
        assert typed._spell_minus_as_ascii("expression", f"x {sign} 2") == "x - 2"


def test_the_dashes_are_a_minus_in_a_number_or_an_expression_wherever_they_are():
    for dash in (EN, EM, "‒", "―"):
        assert typed._spell_minus_as_ascii("expression", f"x {dash} 2") == "x - 2"
        assert typed._spell_minus_as_ascii("number", f"5{dash}9") == "5-9"


def test_the_dashes_are_a_minus_in_other_kinds_only_where_they_are_a_sign():
    for kind in ("pair", "list", "set", "fraction", "ordered_list", "table", "points"):
        assert typed._spell_minus_as_ascii(kind, f"{EN}5") == "-5"
        assert typed._spell_minus_as_ascii(kind, f"x = {EM}5") == "x = -5"
        assert typed._spell_minus_as_ascii(kind, f"({EN}5, {EM}6)") == "(-5, -6)"
        assert typed._spell_minus_as_ascii(kind, f"{EN} 5") == "- 5"
        assert typed._spell_minus_as_ascii(kind, f"{EN}3/4") == "-3/4"
        assert typed._spell_minus_as_ascii(kind, f"5{EN}9") == f"5{EN}9"  # a range stays a range
        assert typed._spell_minus_as_ascii(kind, f"a{EN}b") == f"a{EN}b"
        assert typed._spell_minus_as_ascii(kind, f"(1){EN}2") == f"(1){EN}2"


def test_a_multiple_choice_reply_is_left_alone():
    assert typed._spell_minus_as_ascii("mcq", f"{MINUS}3") == f"{MINUS}3"


def test_the_spoken_readers_are_untouched():
    # `grade()` does the translating; the transcript path reads what it always read.
    assert extract_values(f"{MINUS}3") == [Fraction(3)]
    assert extract_values("-3") == [Fraction(-3)]
    assert extract_values("minus three") == [Fraction(-3)]


# -- number -------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_number(sign):
    negative, positive = _question(-9, "-9"), _question(9, "9")
    for reply in (f"{sign}9", f"{sign} 9", f"x = {sign}9", f"({sign}9)"):
        assert _ok("number", negative, reply) is True, reply
        assert _ok("number", positive, reply) is False, reply


def test_a_dash_between_two_numbers_is_still_two_numbers():
    assert _ok("number", _question(9, "9"), f"5{EN}9") is False
    assert _ok("number", _question(9, "9"), f"5{MINUS}9") is False


# -- pair ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_a_labelled_pair(sign):
    negative = _question({"x": 3, "y": -6}, "x = 3, y = -6")
    positive = _question({"x": 3, "y": 6}, "x = 3, y = 6")
    for reply in (f"x = 3, y = {sign}6", f"y = {sign}6, x = 3", f"x=3,y={sign}6"):
        assert _ok("pair", negative, reply) is True, reply
        assert _ok("pair", positive, reply) is False, reply


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_an_unlabelled_pair(sign):
    negative = _question({"x": -5, "y": 5}, "(-5, 5)")
    positive = _question({"x": 5, "y": 5}, "(5, 5)")
    for reply in (f"({sign}5, 5)", f"{sign}5, 5", f"({sign}5,5)"):
        assert _ok("pair", negative, reply) is True, reply
        assert _ok("pair", positive, reply) is False, reply


def test_an_equation_pair():
    question = _question({"m": -1, "b": 7}, "y = -x + 7")
    for sign in SIGNS:
        assert _ok("pair", question, f"y = {sign}x + 7") is True
        assert _ok("pair", question, f"y = 7 {sign} x") is True
        assert _ok("pair", question, f"y = x {sign} 7") is False


# -- list ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_list(sign):
    negative = _question([-1, 2, 3], "-1, 2, 3")
    positive = _question([1, 2, 3], "1, 2, 3")
    for reply in (f"{sign}1, 2, 3", f"3, 2, {sign}1", f"{sign}1 and 2 and 3", f"x = {sign}1 or x = 2 or x = 3"):
        assert _ok("list", negative, reply) is True, reply
        assert _ok("list", positive, reply) is False, reply


def test_a_range_written_with_a_dash_is_still_two_numbers():
    question = _question([5, 9], "5, 9")
    assert _ok("list", question, f"5{EN}9") is True
    assert _ok("list", question, f"5 {EN} 9") is True
    assert _ok("list", question, "5, 9") is True
    pair = _question({"a": 5, "b": 9}, "(5, 9)")
    assert _ok("pair", pair, f"5{EN}9") is True


# -- set ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_set(sign):
    negative = _question([-1, 2, 3], "{-1, 2, 3}")
    positive = _question([1, 2, 3], "{1, 2, 3}")
    for reply in (f"{{{sign}1, 2, 3}}", f"{sign}1, 2, 3", f"{{3, 2, {sign}1}}"):
        assert _ok("set", negative, reply) is True, reply
        assert _ok("set", positive, reply) is False, reply


def test_a_set_of_pairs():
    question = _question([[-1, 2], [3, -4]], "{(-1, 2), (3, -4)}")
    assert _ok("set", question, f"{{({MINUS}1, 2), (3, {MINUS}4)}}") is True
    assert _ok("set", question, f"{{(1, 2), (3, {MINUS}4)}}") is False


# -- fraction -----------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_fraction(sign):
    negative = _question({"p": -3, "q": 4}, "-3/4")
    positive = _question({"p": 3, "q": 4}, "3/4")
    for reply in (f"{sign}3/4", f" {sign}3/4 ", f"{sign}6/8"):
        assert _ok("fraction", negative, reply) is True, reply
        assert _ok("fraction", positive, reply) is False, reply


def test_a_mixed_number_and_a_ratio():
    mixed = _question({"p": -7, "q": 2}, "-3 1/2")
    assert _ok("fraction", mixed, f"{MINUS}3 1/2") is True
    assert _ok("fraction", mixed, "3 1/2") is False
    assert _ok("fraction", mixed, f"{MINUS}7/2") is True
    ratio = _question({"p": -12, "q": 5}, "-12:5")
    assert _ok("fraction", ratio, f"{MINUS}12:5") is True
    assert _ok("fraction", ratio, "12:5") is False


# -- expression ---------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_expression(sign):
    question = _question("x - 2", "x - 2")
    assert _ok("expression", question, f"x {sign} 2") is True
    assert _ok("expression", question, f"x{sign}2") is True
    assert _ok("expression", question, f"{sign}2 + x") is True
    assert _ok("expression", question, f"x {sign} 3") is False
    assert _ok("expression", question, f"x + 2") is False
    negation = _question("-x", "-x")
    assert _ok("expression", negation, f"{sign}x") is True
    assert _ok("expression", negation, f"{sign}x{sign}0") is True
    assert _ok("expression", negation, "x") is False
    exponent = _question("x**(-2)", "x^(-2)")
    assert _ok("expression", exponent, f"x^({sign}2)") is True
    assert _ok("expression", exponent, f"1/x^2") is True


# -- ordered_list -------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", SIGNS + [EN, EM])
def test_ordered_list(sign):
    negative = _question([-3, 1, 2], "-3, 1, 2")
    positive = _question([3, 1, 2], "3, 1, 2")
    for reply in (f"{sign}3, 1, 2", f"({sign}3, 1, 2)", f"{sign}3 , 1 , 2"):
        assert _ok("ordered_list", negative, reply) is True, reply
        assert _ok("ordered_list", positive, reply) is False, reply
    assert _ok("ordered_list", negative, f"{sign}3, {sign}1, 2") is False


# -- the drawn kinds and the table read it too ---------------------------------------------------


def test_points():
    question = _question([[-1, 2], [3, -4]], "(-1, 2) (3, -4)")
    assert _ok("points", question, f"({MINUS}1, 2) (3, {MINUS}4)") is True
    assert _ok("points", question, f"({MINUS}1, 2) (3, 4)") is False
    assert _ok("points", question, f"({EN}1, 2) (3, {EN}4)") is True


def test_table():
    answer = {"columns": ["", "A"], "rows": [["r1", None], ["r2", None]], "blanks": {"0,1": -5, "1,1": 7}}
    question = _question(answer, typed.table_answer_text(answer))
    assert _ok("table", question, f"r1 A = {MINUS}5, r2 A = 7") is True
    assert _ok("table", question, f"r1 A = 5, r2 A = 7") is False
    assert _ok("table", question, '{"0,1": "' + MINUS + '5", "1,1": "7"}') is True
    assert _ok("table", question, f"{MINUS}5, 7") is True
    assert _ok("table", question, f"{MINUS}5, {MINUS}7") is False


# -- a space between the sign and its number ----------------------------------------------------
#
# "y = - 6", "(3, - 6)" and "{- 1, 2, 3}" are negative numbers, as "- 6" always was in a number reply. The rule is
# the number grader's: a minus that does not follow a value is a sign. "5 - 6" and "x - 6" stay a subtraction.


@pytest.mark.parametrize(
    ("given", "read_as"),
    [
        ("- 6", "-6"),
        ("x = 3, y = - 6", "x = 3, y = -6"),
        ("(3, - 6)", "(3, -6)"),
        ("(1, - 2)", "(1, -2)"),
        ("{- 1, 2, 3}", "{-1, 2, 3}"),
        ("x = 3 or x = - 3", "x = 3 or x = -3"),
        ("3 and - 3", "3 and -3"),
        ("3 or - 3", "3 or -3"),
        ("3, - 6", "3, -6"),
        ("3\n- 6", "3\n-6"),
        ("x = - .5", "x = -.5"),
        ("-   6", "-6"),
    ],
)
def test_a_minus_apart_from_its_number_is_a_sign(given, read_as):
    for kind in ("pair", "list", "set", "ordered_list", "points"):
        assert typed._attach_spaced_sign(kind, given) == read_as, (kind, given)


@pytest.mark.parametrize(
    "given",
    [
        "5 - 6",
        "x - 6",
        "(5) - 6",
        "[5] - 6",
        "{5} - 6",
        "5. - 6",
        "for - 6",
        "- x",
        "x = 3, y = x - 6",
        "-6",
        "3 -6",
        "y = 7 - x",
    ],
)
def test_a_minus_that_follows_a_value_is_left_a_subtraction(given):
    for kind in ("pair", "list", "set", "ordered_list", "points", "line", "fraction"):
        assert typed._attach_spaced_sign(kind, given) == given, (kind, given)


@pytest.mark.parametrize("kind", ["expression", "mcq", "number"])
def test_a_spaced_sign_is_left_alone_where_another_reader_owns_it(kind):
    # An expression keeps its subtraction ("x - 6", "- 6 + x" is the sum it says); a multiple-choice reply is an
    # option's text; a number reply has its own sign rule.
    assert typed._attach_spaced_sign(kind, "x = 3, y = - 6") == "x = 3, y = - 6"


def test_a_spaced_sign_in_a_pair():
    negative = _question({"x": 3, "y": -6}, "x = 3, y = -6")
    positive = _question({"x": 3, "y": 6}, "x = 3, y = 6")
    for reply in (
        "x = 3, y = - 6",
        "y = - 6, x = 3",
        "x = 3 and y = - 6",
        f"x = 3, y = {MINUS} 6",
        f"x = 3, y = {EN} 6",
    ):
        assert _ok("pair", negative, reply) is True, reply
        assert _ok("pair", positive, reply) is False, reply
    unlabelled = _question({"x": 3, "y": -6}, "(3, -6)")
    assert _ok("pair", unlabelled, "(3, - 6)") is True
    assert _ok("pair", unlabelled, "3, - 6") is True
    assert _ok("pair", unlabelled, "(3, 6)") is False


def test_a_spaced_sign_in_a_list_a_set_and_an_ordered_list():
    solutions = _question([3, -3], "x = 3 or x = -3")
    assert _ok("list", solutions, "x = 3 or x = - 3") is True
    assert _ok("list", solutions, "3, - 3") is True
    assert _ok("list", solutions, "x = 3 or x = 3") is False
    members = _question([-1, 2, 3], "{-1, 2, 3}")
    assert _ok("set", members, "{- 1, 2, 3}") is True
    assert _ok("set", members, "{1, 2, 3}") is False
    ordered = _question([-3, 1, 2], "-3, 1, 2")
    assert _ok("ordered_list", ordered, "- 3, 1, 2") is True
    assert _ok("ordered_list", ordered, "3, 1, 2") is False


def test_a_spaced_sign_in_the_drawn_kinds_and_the_table():
    points = _question([[-1, 2], [3, -4]], "(-1, 2) (3, -4)")
    assert _ok("points", points, "(- 1, 2) (3, - 4)") is True
    assert _ok("points", points, "(1, 2) (3, - 4)") is False
    line = _question([[0, -1], [1, 1]], typed.line_answer_text([[0, -1], [1, 1]]))
    assert _ok("line", line, "(0, - 1) (1, 1)") is True
    assert _ok("line", line, "(0, 1) (1, 1)") is False
    polyline_points = [[-2, 0], [0, 3]]
    polyline = _question(polyline_points, typed.polyline_answer_text(polyline_points))
    assert _ok("polyline", polyline, "(- 2, 0) (0, 3)") is True
    assert _ok("polyline", polyline, "(2, 0) (0, 3)") is False
    answer = {"columns": ["", "A"], "rows": [["r1", None], ["r2", None]], "blanks": {"0,1": -5, "1,1": 7}}
    table = _question(answer, typed.table_answer_text(answer))
    assert _ok("table", table, "r1 A = - 5, r2 A = 7") is True
    assert _ok("table", table, "r1 A = 5, r2 A = 7") is False


def test_a_spaced_sign_in_a_fraction():
    question = _question({"p": -3, "q": 4}, "-3/4")
    assert _ok("fraction", question, "- 3/4") is True
    assert _ok("fraction", question, "3/4") is False


def test_a_subtraction_is_not_turned_into_a_negative_number():
    # "5 - 6" in a list is still the two numbers 5 and 6 (a range or a slip), never 5 and -6.
    assert _ok("list", _question([5, 6], "5, 6"), "5 - 6") is True
    assert _ok("list", _question([5, -6], "5, -6"), "5 - 6") is False
    assert _ok("list", _question([5, -6], "5, -6"), "5, - 6") is True


# -- a long gap reads as a short one ---------------------------------------------------------------


def test_a_run_of_whitespace_is_one_gap():
    wide = " " * 5000
    assert typed._WHITESPACE_RUN_RE.sub(typed._one_gap, f"a{wide}b") == "a b"
    assert typed._WHITESPACE_RUN_RE.sub(typed._one_gap, f"a{wide}\n{wide}b") == "a\nb"
    assert typed._WHITESPACE_RUN_RE.sub(typed._one_gap, "a \t \r\n \t b") == "a\nb"
    assert typed._WHITESPACE_RUN_RE.sub(typed._one_gap, "a\t\tb") == "a b"


def test_a_long_gap_grades_as_a_short_one_in_every_kind_that_splits_on_one():
    wide = " " * 3000
    points = _question([[1, 2], [3, -4]], "(1, 2) (3, -4)")
    assert _ok("points", points, f"({wide}1{wide},{wide}2{wide}){wide}({wide}3{wide},{wide}-4{wide})") is True
    assert _ok("points", points, f"({wide}1{wide},{wide}2{wide}){wide}({wide}3{wide},{wide}4{wide})") is False
    members = _question([1, 2, 3], "{1, 2, 3}")
    assert _ok("set", members, f"{{{wide}1{wide},{wide}2{wide},{wide}3{wide}}}") is True
    assert _ok("set", members, f"{{{wide}1{wide},{wide}2{wide},{wide}4{wide}}}") is False
    answer = {"columns": ["", "A"], "rows": [["r1", None], ["r2", None]], "blanks": {"0,1": -5, "1,1": 7}}
    table = _question(answer, typed.table_answer_text(answer))
    assert _ok("table", table, f"r1{wide}A{wide}={wide}-5,{wide}r2{wide}A{wide}={wide}7") is True
    assert _ok("table", table, f"r1{wide}A{wide}={wide}5,{wide}r2{wide}A{wide}={wide}7") is False
    vertices = [[0, 0], [2, 4], [5, 4]]
    polyline = _question(vertices, typed.polyline_answer_text(vertices))
    assert _ok("polyline", polyline, f"(0,{wide}0){wide}(2,{wide}4){wide}(5,{wide}4)") is True
    assert _ok("polyline", polyline, f"(0,{wide}0){wide}(2,{wide}4){wide}(5,{wide}5)") is False
    ordered = _question([-3, 1, 2], "-3, 1, 2")
    assert _ok("ordered_list", ordered, f"-3{wide},{wide}1{wide},{wide}2") is True
    assert _ok("ordered_list", ordered, f"3{wide},{wide}1{wide},{wide}2") is False
