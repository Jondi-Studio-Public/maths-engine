"""A reply's text becomes a number only through a pattern that cannot ask for a huge one.

`Fraction("1e9999999")` works out 10**9999999, which takes seconds; `1e99999999`, ten characters of reply,
takes minutes. The fraction grader, every item of an ordered list and a number skill whose answer is a
fraction all handed the reply's text to `Fraction`. `typed._exact_fraction` is now the only place that
does, and only after a pattern that has no exponent and no underscore (and a length cap) has matched.
"""

from __future__ import annotations

import logging
import signal
import time
from fractions import Fraction

import pytest

from mathgen.core import Question
from mathgen.grading import grade, typed

#: A test still running after this is stopped (where the platform allows), so a regression fails instead of
#: freezing the run: `Fraction("1e99999999")` runs for minutes.
HARD_STOP_SECONDS = 20


class _StillRunning(BaseException):
    """Raised by the alarm. Not an `Exception`, so `grade()`'s own backstop cannot swallow it."""


@pytest.fixture(autouse=True)
def _hard_stop():
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise _StillRunning(f"still running after {HARD_STOP_SECONDS} s")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, HARD_STOP_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


_BASE = dict(
    topic="test", level=0, seed=1, text="q", latex="q", answer_latex="", working=[], difficulty="easy", distractors=[]
)


def _question(answer, answer_text, **fields) -> Question:
    return Question(answer=answer, answer_text=answer_text, **_BASE, **fields)


# -- the reader itself ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, value",
    [
        ("3/4", Fraction(3, 4)),
        ("-3/4", Fraction(-3, 4)),
        ("+3/4", Fraction(3, 4)),
        ("6/8", Fraction(3, 4)),
        ("0.75", Fraction(3, 4)),
        (".75", Fraction(3, 4)),
        ("-.5", Fraction(-1, 2)),
        ("5.", Fraction(5)),
        ("12", Fraction(12)),
        ("007", Fraction(7)),
        ("-0", Fraction(0)),
        ("1000000", Fraction(10**6)),
        ("  3/4  ", Fraction(3, 4)),
        ("7 / 13", Fraction(7, 13)),
        ("3 /4", Fraction(3, 4)),
        ("3/ 4", Fraction(3, 4)),
        ("3\t/\t4", Fraction(3, 4)),
        ("-3 / 4", Fraction(-3, 4)),
        ("9" * 100, Fraction(10**100 - 1)),
    ],
)
def test_a_plain_number_is_read_exactly(text, value):
    assert typed._exact_fraction(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "1e5",
        "1E5",
        "1e+5",
        "1e-5",
        "1.5e3",
        "1e9999999",
        "1e99999999",
        "-1e9999999",
        "1e-9999999",
        "1_000",
        "1_0e5",
        "0x10",
        "inf",
        "nan",
        "",
        " ",
        "1/0",
        "1/",
        "/2",
        "--1",
        "1.2.3",
        "1/2/3",
        "1 / 2 / 3",
        "1 /",
        "/ 2",
        "1 / 0",
        "1 \n/ 2",
        "3/4e5",
        "9" * 101,
        "1" + "0" * 200,
        "." * 5,
        ".",
        "-",
        "+",
    ],
)
def test_anything_else_is_not_a_number(text):
    started = time.perf_counter()
    assert typed._exact_fraction(text) is None
    assert time.perf_counter() - started < 0.1


def test_a_spaced_slash_is_read_by_every_fraction_form():
    three_quarters = _question({"p": 3, "q": 4}, "3/4")
    for reply in ("3 / 4", "3 /4", "3/ 4", "6 / 8", " 3 / 4 "):
        assert grade("fraction", three_quarters, reply).correct is True, reply
    assert grade("fraction", three_quarters, "3 / 5").correct is False
    assert grade("number", _question({"p": 7, "q": 13}, "7/13"), "7 / 13").correct is True
    assert grade("number", _question({"p": 7, "q": 13}, "7/13"), "7 / 12").correct is False
    fractions = _question([0.5, 0.75], "1/2, 3/4")
    assert grade("ordered_list", fractions, "1 / 2, 3 / 4").correct is True
    assert grade("ordered_list", fractions, "(1 / 2, 3 /4)").correct is True
    assert grade("ordered_list", fractions, "1 / 2, 3 / 5").correct is False
    assert typed._parse_typed_fraction("7 / 13") == Fraction(7, 13)
    assert typed._parse_numeric_item("1 / 2") == Fraction(1, 2)


def test_the_fraction_parser_and_the_list_item_reader_use_it():
    for text in ("1e9999999", "1e99999999", "-1e9999999", "1e-9999999", "1.5e9999999", "1/1e9999999"):
        started = time.perf_counter()
        assert typed._parse_typed_fraction(text) is None
        assert typed._parse_numeric_item(text) is None
        assert time.perf_counter() - started < 0.1, text


def test_a_long_run_of_digits_is_refused_by_every_fraction_form():
    digits = "9" * 5000
    for text in (digits, f"{digits}/2", f"1/{digits}", f"{digits} 1/2", f"1 1/{digits}", f"{digits}:3", f"3:{digits}"):
        started = time.perf_counter()
        assert typed._parse_typed_fraction(text) is None
        assert time.perf_counter() - started < 0.1


# -- through grade() -------------------------------------------------------------------------

_EXPONENTS = [
    "1e9999999",
    "1e99999999",
    "-1e9999999",
    "1e-9999999",
    "1.5e9999999",
    "1E9999999",
    "1e+99999999",
]


def _graded_wrong_quickly(kind, question, reply):
    started = time.perf_counter()
    result = grade(kind, question, reply)
    assert time.perf_counter() - started < 0.5, f"{kind}: {reply!r}"
    assert result.correct is False, f"{kind}: {reply!r}"


@pytest.mark.parametrize("reply", _EXPONENTS)
def test_an_exponent_is_wrong_and_quick_for_a_fraction(reply):
    _graded_wrong_quickly("fraction", _question({"p": 3, "q": 4}, "3/4"), reply)
    _graded_wrong_quickly("fraction", _question("3/20", "3/20"), reply)
    _graded_wrong_quickly("fraction", _question({"p": 12, "q": 5}, "12:5"), reply)
    # a number skill whose answer is an exact fraction goes to the same grader
    _graded_wrong_quickly("number", _question({"p": 125, "q": 64}, "125/64"), reply)


@pytest.mark.parametrize("reply", _EXPONENTS)
def test_an_exponent_is_wrong_and_quick_for_an_ordered_list(reply):
    question = _question([1, 2, 3], "1, 2, 3")
    for text in (f"{reply}, 2, 3", f"1, {reply}, 3", f"1, 2, {reply}", f"({reply}, 2, 3)", f"{reply} < 2 < 3"):
        _graded_wrong_quickly("ordered_list", question, text)
    _graded_wrong_quickly("ordered_list", _question([0.5, 2], "0.5, 2"), f"{reply}, 2")


def test_ordinary_fractions_and_lists_are_read_as_before():
    three_quarters = _question({"p": 3, "q": 4}, "3/4")
    for reply in ("3/4", "6/8", "0.75", ".75", " 3/4 ", "+3/4", "75/100"):
        assert grade("fraction", three_quarters, reply).correct is True, reply
    for reply in ("3/5", "0.8", "3/4e0", "7.5e-1", "3/4 or 1", ""):
        assert grade("fraction", three_quarters, reply).correct is False, reply
    mixed = _question({"p": 7, "q": 2}, "3 1/2")
    assert grade("fraction", mixed, "3 1/2").correct is True
    assert grade("fraction", mixed, "7/2").correct is True
    assert grade("fraction", mixed, "3.5").correct is True
    ratio = _question({"p": 12, "q": 5}, "12:5")
    assert grade("fraction", ratio, "12:5").correct is True
    assert grade("fraction", ratio, "24:10").correct is False  # not in lowest terms, as before
    assert grade("fraction", ratio, "12/5").correct is True
    numbers = _question([1, 2, 3], "1, 2, 3")
    for reply in ("1, 2, 3", "(1, 2, 3)", "1 2 3", "01, 2.0, 3", "1\n2\n3", "1 < 2 < 3"):
        assert grade("ordered_list", numbers, reply).correct is True, reply
    thousands = _question([1234, 5], "1,234, 5")
    assert grade("ordered_list", thousands, "1,234, 5").correct is True
    assert grade("ordered_list", thousands, "1234, 5").correct is True
    assert grade("ordered_list", _question([0.5, 0.25], "0.5, 0.25"), ".5, .25").correct is True


# -- the other conversions of a reply's text ---------------------------------------------------


def test_an_enormous_option_number_does_not_raise(caplog):
    question = _question("1", "1")
    with caplog.at_level(logging.ERROR, logger="mathgen.grading"):
        for reply in ("9" * 5000, "-" + "9" * 5000, "1" + "0" * 20, "option " + "7" * 100_000):
            started = time.perf_counter()
            result = grade("mcq", question, reply)
            assert time.perf_counter() - started < 0.5
            assert result.correct is False
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_an_integer_past_a_float_does_not_raise_when_a_decimal_answer_is_offered(caplog):
    question = _question("sqrt(115)", "sqrt(115)", answer_alt=10.72)
    with caplog.at_level(logging.ERROR, logger="mathgen.grading"):
        for reply in ("10^400", "10**400", "9" * 400, "10.72", "sqrt(115)"):
            result = grade("expression", question, reply)
            assert result.correct is (reply in ("10.72", "sqrt(115)")), reply
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
