"""The `inequality` and `matrix` answer kinds: what each reads as the answer, and what it refuses."""

from __future__ import annotations

import itertools
import time
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen.answers import ANSWER_FORMS, ANSWER_KINDS
from mathgen.grading import GRADERS, grade
from mathgen.grading.inequality import read_solution_set, set_text
from mathgen.grading.matrix import matrix_latex, matrix_text, read_matrix, vector_text


def _question(answer, answer_text=None, form=None):
    text = answer_text if answer_text is not None else answer
    return SimpleNamespace(answer=answer, answer_text=text, answer_form=form, text="Solve.")


def test_both_kinds_are_registered_and_in_the_contract():
    assert {"inequality", "matrix"} <= set(GRADERS) <= ANSWER_KINDS | {"inequality", "matrix"}
    assert {"inequality", "matrix"} <= ANSWER_KINDS
    assert {"scalar_multiple", "column_scalar_multiple"} <= ANSWER_FORMS


# -- inequality -----------------------------------------------------------------------------------------------------

INEQUALITY_CASES = [
    (
        "x > 3",
        [
            "x>3",
            "3 < x",
            "(3, ∞)",
            "(3,inf)",
            "(3, infinity)",
            "x ∈ (3, ∞)",
            "x in (3, oo)",
            "{x : x > 3}",
            "{x | x > 3}",
            "x > 3.0",
            "x > 6/2",
            "x > 3.",
            "x \\in (3, \\infty)",
            "x \\in \\left(3, \\infty\\right)",
        ],
        [
            "x >= 3",
            "x ≥ 3",
            "x > 3 or x < 3",
            "t > 3",
            "[3, oo)",
            "x < 3",
            "3",
            "x = 3",
            "x > 4",
            "2x > 6",
            "x > 3 or x > 2",
            "x > 2",
            "x > 3 and x < 10",
            "",
            "x > ",
            "> 3",
        ],
    ),
    (
        "x <= -5/2",
        [
            "x ≤ -5/2",
            "x <= -2.5",
            "-5/2 >= x",
            "(-∞, -5/2]",
            "x =< -2.5",
            "x ≤ - 5/2",
            "x ⩽ −5/2",
            "x ≤ -2 1/2",
            "x \\leq -\\frac{5}{2}",
            "$x \\le -2.5$",
        ],
        ["x < -5/2", "x <= -2.4", "x ≥ -5/2", "x <= 5/2", "x <= --5/2"],
    ),
    (
        "-2 < x <= 7",
        [
            "-2 < x ≤ 7",
            "x > -2 and x <= 7",
            "x > -2, x <= 7",
            "(-2, 7]",
            "7 >= x > -2",
            "(x > -2) and (x <= 7)",
            "x > -2 ∧ x ≤ 7",
            "(-2, 7] ∩ (-5, 9)",
        ],
        ["-2 <= x <= 7", "x > -2 or x <= 7", "(-2,7)", "-2 < x > 7", "7 < x <= -2", "(7, -2]"],
    ),
    (
        "x < -1 or x > 3",
        [
            "x > 3 or x < -1",
            "(-∞,-1) ∪ (3,∞)",
            "(-inf,-1) U (3,inf)",
            "x < -1, x > 3",
            "x < -1 || x > 3",
            "(-oo, -1) or (3, oo)",
        ],
        ["-1 < x < 3", "x < -1", "x<=-1 or x>3", "x < -1 and x > 3", "x < 3 or x > -1"],
    ),
    ("x != 4", ["x ≠ 4", "x < 4 or x > 4", "(-∞, 4) ∪ (4, ∞)", "4 != x"], ["x > 4", "all real numbers", "x = 4"]),
    (
        "all real numbers",
        ["R", "ℝ", "all real x", "(-∞, ∞)", "x ∈ ℝ", "all x", "x > 1 or x <= 1"],
        ["no solution", "x > 0"],
    ),
    ("no solution", ["∅", "{}", "none", "no real solutions", "x > 3 and x < 1"], ["all real numbers", "x = 0"]),
    ("x > √2", ["x > sqrt(2)", "x>√2", "(√2, ∞)", "x > 2/sqrt(2)"], ["x > 1.41", "x > 1.414", "x > 2", "x > √3"]),
    ("t >= 12", ["t ≥ 12", "12 <= t", "[12, ∞)"], ["x ≥ 12", "t > 12"]),
]


@pytest.mark.parametrize("answer, right, wrong", INEQUALITY_CASES)
def test_inequality_reads_every_spelling_of_the_set_and_nothing_else(answer, right, wrong):
    question = _question(answer)
    for reply in right:
        assert grade("inequality", question, reply).correct, (answer, reply)
    for reply in wrong:
        assert not grade("inequality", question, reply).correct, (answer, reply)


def test_the_canonical_text_grades_correct_and_round_trips():
    for answer, _, _ in INEQUALITY_CASES:
        solution, variable = read_solution_set(answer)
        assert grade("inequality", _question(answer), answer).correct
        assert grade("inequality", _question(answer), set_text(solution, variable or "x")).correct
        assert grade("inequality", _question(answer), set_text(solution, None)).correct


def test_a_reply_naming_two_variables_is_unreadable():
    assert read_solution_set("x > 3 or y < 1") is None
    assert not grade("inequality", _question("x > 3"), "x > 3 and y > 3").correct


def test_an_unreadable_canonical_is_a_logged_failure_not_a_crash(caplog):
    result = grade("inequality", _question("whatever"), "x > 3")
    assert not result.correct
    assert "raised" in caplog.text


@pytest.mark.parametrize(
    "reply",
    [
        "x > " + "9" * 290,
        "x > 1 or " * 33 + "x > 2",
        "(" * 150 + "x > 3" + ")" * 145,
        "x > 10^10^10^10",
        "x > factorial(10000)",
        "x > __import__('os')",
        "x > 1e999999999",
        "[" * 300,
    ],
)
def test_pathological_replies_are_quick_and_wrong(reply):
    start = time.perf_counter()
    assert not grade("inequality", _question("x > 3"), reply).correct
    assert time.perf_counter() - start < 2


# -- matrix ---------------------------------------------------------------------------------------------------------

M = [[1, -2], [{"p": 3, "q": 5}, 1]]
M_TEXT = "[[1, -2], [3/5, 1]]"


@pytest.mark.parametrize(
    "reply",
    [
        "[[1, -2], [3/5, 1]]",
        "[1 -2; 3/5 1]",
        "A⁻¹ = [1 -2; 0.6 1]",
        "A^-1 = [[1,-2],[0.6,1]]",
        "A^(-1) = [1 -2; 3/5 1]",
        "1/5 [[5, -10], [3, 5]]",
        "(1/5)[5 -10; 3 5]",
        "1/5 × [5 -10; 3 5]",
        "[[5,-10],[3,5]]/5",
        "((1, -2), (3/5, 1))",
        "[1, -2] [3/5, 1]",
        "1 -2\n3/5 1",
        "1, -2; 3/5, 1",
        "\\begin{pmatrix} 1 & -2 \\\\ \\frac{3}{5} & 1 \\end{pmatrix}",
        "\\begin{bmatrix} 1 & -2 \\\\ 0.6 & 1 \\end{bmatrix}",
        "[1 −2; 3/5 1]",
        "[1, - 2; 3 / 5, 1]",
    ],
)
def test_matrix_reads_every_spelling(reply):
    assert grade("matrix", _question(M, M_TEXT), reply).correct, reply


@pytest.mark.parametrize(
    "reply",
    [
        "[[1, 2], [3/5, 1]]",
        "[[1, -2], [0.61, 1]]",
        "[1, -2, 3/5, 1]",
        "[[1, -2], [3/5, 1], [0, 0]]",
        "[[1, 3/5], [-2, 1]]",
        "[[-1, 2], [-3/5, -1]]",
        "[[5, -10], [3, 5]]",
        "1/5 [[1, -2], [3/5, 1]]",
        "x",
        "[[1, -2], [3/5, 1]] or [[1, 2], [3/5, 1]]",
        "[[1, -2], [3/5]]",
        "[[1, -2], [3/5, 1]]/0",
        "0 [[1,1],[1,1]]",
        "[[1, --2], [3/5, 1]]",
        "[[1 - 2], [3/5, 1]]",
    ],
)
def test_matrix_refuses_a_different_matrix(reply):
    assert not grade("matrix", _question(M, M_TEXT), reply).correct, reply


def test_a_vector_reads_in_either_orientation():
    column = _question([[1], [-2], [{"p": 1, "q": 2}]], "(1, -2, 1/2)")
    for reply in (
        "(1, -2, 1/2)",
        "[1, -2, 0.5]",
        "<1, -2, 1/2>",
        "[[1], [-2], [1/2]]",
        "[1 -2 1/2]",
        "v = (1, -2, 1/2)",
    ):
        assert grade("matrix", column, reply).correct, reply
    for reply in ("(1, 1/2, -2)", "(2, -4, 1)", "(1, -2)"):
        assert not grade("matrix", column, reply).correct, reply


def test_scalar_multiple_accepts_any_nonzero_multiple_only():
    eigenvector = _question([[1], [2]], "(1, 2)", form="scalar_multiple")
    for reply in ("(1, 2)", "(2, 4)", "(-1, -2)", "[[3], [6]]", "<1/2, 1>", "(√2, 2√2)"):
        assert grade("matrix", eigenvector, reply).correct, reply
    for reply in ("(0, 0)", "(1, 3)", "(2, 1)", "(1, 2, 0)"):
        assert not grade("matrix", eigenvector, reply).correct, reply
    with_zero = _question([[0], [1], [-1]], "(0, 1, -1)", form="scalar_multiple")
    assert grade("matrix", with_zero, "(0, -3, 3)").correct
    assert not grade("matrix", with_zero, "(1, -3, 3)").correct


def test_column_scalar_multiple_scales_each_column_on_its_own():
    p_matrix = _question([[1, 1], [2, -1]], "[[1, 1], [2, -1]]", form="column_scalar_multiple")
    for reply in ("[[1, 1], [2, -1]]", "[[2, -1], [4, 1]]", "[[-1, 3], [-2, -3]]"):
        assert grade("matrix", p_matrix, reply).correct, reply
    for reply in ("[[1, 1], [-1, 2]]", "[[0, 1], [0, -1]]", "[[1, 2], [1, -1]]"):
        assert not grade("matrix", p_matrix, reply).correct, reply


def test_closed_form_entries_compare_exactly():
    question = _question([["sqrt(2)/2", "-sqrt(2)/2"], ["sqrt(2)/2", "sqrt(2)/2"]], "[[√2/2, -√2/2], [√2/2, √2/2]]")
    assert grade("matrix", question, "[[1/√2, -1/√2], [1/√2, 1/√2]]").correct
    assert not grade("matrix", question, "[[0.707, -0.707], [0.707, 0.707]]").correct


def test_helpers_spell_what_the_grader_reads():
    assert matrix_text(M) == M_TEXT
    assert vector_text([1, -2, {"p": 1, "q": 2}]) == "(1, -2, 1/2)"
    assert matrix_latex(M) == r"\begin{pmatrix} 1 & -2 \\ \frac{3}{5} & 1 \end{pmatrix}"
    assert read_matrix(matrix_text(M)) is not None
    assert grade("matrix", _question(M, M_TEXT), matrix_text(M)).correct


def test_a_form_that_does_not_apply_is_a_logged_failure(caplog):
    assert not grade("matrix", _question(M, M_TEXT, form="factored"), M_TEXT).correct
    assert "raised" in caplog.text


@pytest.mark.parametrize(
    "reply",
    [
        "[" + ", ".join(["1"] * 590) + "]",
        "[" * 300 + "]" * 300,
        "[[10^10^10, 1], [1, 1]]",
        "[[__import__('os'), 1], [1, 1]]",
        "[[factorial(100000), 1], [1, 1]]",
        "[[1e9999999, 1], [1, 1]]",
        ";" * 500,
    ],
)
def test_pathological_matrix_replies_are_quick_and_wrong(reply):
    start = time.perf_counter()
    assert not grade("matrix", _question(M, M_TEXT), reply).correct
    assert time.perf_counter() - start < 2


SIGNED = [[{"p": -5, "q": 4}, 1], [{"p": 3, "q": 2}, -1]]


def _spellings(rows):
    """A matrix written the ways a learner might, each entry's sign kept on the entry."""

    def entry(v, latex=False):
        v = Fraction(v["p"], v["q"]) if isinstance(v, dict) else Fraction(v)
        sign, v = ("-" if v < 0 else ""), abs(v)
        if v.denominator == 1:
            return f"{sign}{v.numerator}"
        if latex:
            return f"{sign}\\frac{{{v.numerator}}}{{{v.denominator}}}"
        return f"{sign}({v.numerator}/{v.denominator})"

    return [
        r"\begin{pmatrix} " + r" \\ ".join(" & ".join(entry(v, True) for v in r) for r in rows) + r" \end{pmatrix}",
        "; ".join(", ".join(entry(v) for v in r) for r in rows),
        "\n".join(" ".join(entry(v) for v in r) for r in rows),
        "[" + ", ".join("[" + ", ".join(entry(v) for v in r) + "]" for r in rows) + "]",
    ]


def _negated(v):
    return {"p": -v["p"], "q": v["q"]} if isinstance(v, dict) else -v


def test_a_leading_minus_belongs_to_the_first_entry_not_the_matrix():
    question = _question(SIGNED, "[[-5/4, 1], [3/2, -1]]")
    for reply in _spellings(SIGNED):
        assert grade("matrix", question, reply).correct, reply
    for flips in itertools.product([False, True], repeat=4):
        if not any(flips):
            continue
        wrong = [[_negated(v) if flips[2 * i + j] else v for j, v in enumerate(r)] for i, r in enumerate(SIGNED)]
        for reply in _spellings(wrong):
            assert not grade("matrix", question, reply).correct, reply


def test_a_divisor_on_the_last_entry_does_not_scale_the_matrix():
    question = _question([[1, 2], [3, {"p": 4, "q": 5}]], "[[1, 2], [3, 4/5]]")
    assert grade("matrix", question, "1, 2; 3, (4)/5").correct
    assert not grade("matrix", question, "5, 10; 15, (4)/5").correct
    assert grade("matrix", question, "[[5, 10], [15, 4]]/5").correct
