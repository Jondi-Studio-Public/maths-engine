"""Learner notation the expression parser misread, second round."""

from __future__ import annotations

import pytest

from mathgen.core import Question

from mathgen.grading import typed as grading


def _q(answer, answer_text, text="q") -> Question:
    return Question(topic="test", level=0, seed=1, text=text, latex="q", answer=answer, answer_text=answer_text)


def _correct(q, given) -> bool:
    return grading.grade("expression", q, given).correct


@pytest.mark.parametrize(
    ("typed", "reads_as"),
    [
        ("x²", "x**2"),
        ("2x²", "2*x**2"),
        ("x³+1", "x**3 + 1"),
        ("x⁻¹", "1/x"),
        ("sin²x", "sin(x)**2"),
        ("sin²(x)", "sin(x)**2"),
        ("cos(theta)²", "cos(theta)**2"),
        ("cosec(x)", "csc(x)"),
        ("cosec^2 x", "csc(x)**2"),
        ("arcsin(x)", "asin(x)"),
        ("arctan x", "atan(x)"),
        ("sinx", "sin(x)"),
        ("xsinx", "x*sin(x)"),
        ("3sinθ", "3*sin(θ)"),
        ("sin 2x", "sin(2*x)"),
        ("sin x/2", "sin(x)/2"),
        ("x sin x", "x*sin(x)"),
        ("e^x sin x", "e**x*sin(x)"),
        ("sin(x^2)", "sin(x**2)"),
        ("sin^2 x", "sin(x)**2"),
        ("sin(x)cos(x)", "sin(x)*cos(x)"),
        ("sin x + cos x", "sin(x) + cos(x)"),
    ],
)
def test_notation_reads_as_written(typed, reads_as):
    import sympy

    assert (
        sympy.simplify(grading._parse_expression(typed) - sympy.sympify(reads_as, locals={"θ": sympy.Symbol("θ")})) == 0
    )


@pytest.mark.parametrize(
    "typed",
    [
        "2 sin x cos x",  # was 2*sin(x*cos(x))
        "sin 2x cos x",  # was sin(2*x*cos(x))
        "sin x cos x",
        "2sinxcosx",
        "sin x cos(x)",
        "sin^-1(x)",  # arcsin to a learner, 1/sin(x) to sympy
        "sin⁻¹x",
        "sin -x",  # was x
        "sin x^2",  # sin(x**2) to sympy, (sin x)**2 to many learners
        "sinx^2",
        "sin",
    ],
)
def test_notation_without_one_reading_fails_closed(typed):
    assert grading._parse_expression(typed) is None


def test_superscript_is_never_dropped_by_the_spoken_fallback():
    # "x²" used to fail the parse and reach the spoken-word fallback, which
    # dropped the "²" and graded the answer as x.
    assert not _correct(_q("x", "x"), "x²")
    assert _correct(_q("x**2", "x^2"), "x²")


@pytest.mark.parametrize("typed", ["2 sin x cos x", "sin 2x cos x"])
def test_ambiguous_double_angle_is_not_marked_correct(typed):
    assert not _correct(_q("sin(2*x)", "sin(2x)"), typed)
    assert _correct(_q("sin(2*x)", "sin(2x)"), "2 sin(x) cos(x)")


@pytest.mark.parametrize(
    ("answer", "typed"),
    [
        ("sin(theta)**2", "sin²θ"),
        ("sin(theta)**2", "sin(θ)^2"),
        ("2*theta", "2θ"),
        ("cos(alpha)", "cos α"),
    ],
)
def test_greek_character_matches_a_spelled_out_key(answer, typed):
    assert _correct(_q(answer, answer), typed)


def test_greek_name_matches_a_character_key():
    import sympy

    theta = sympy.Symbol("θ")
    assert _correct(_q(2 * theta, "2θ"), "2 theta")
    assert _correct(_q(2 * theta, "2θ"), "2θ")


def test_greek_letter_is_not_read_as_a_different_variable():
    assert not _correct(_q("2*theta", "2*theta"), "2φ")
    assert not _correct(_q("2*x", "2*x"), "2θ")


@pytest.mark.parametrize(
    ("answer", "typed"),
    [
        ("csc(x)", "cosec x"),
        ("-csc(x)*cot(x)", "-cosec(x) cot(x)"),
        ("asin(x)", "arcsin(x)"),
    ],
)
def test_function_spellings_are_accepted(answer, typed):
    assert _correct(_q(answer, answer), typed)
