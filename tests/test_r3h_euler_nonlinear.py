"""``euler-method-nonlinear-table``: tables and errors grade right, every distractor wrong, and the numbers are Euler's."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_euler_nonlinear"
EXAM = "exd_x_r3h_euler_nonlinear"
add_common_tests(globals(), PRACTICE, EXAM)

x, y = sympy.symbols("x y")


def _slope_function(text):
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    return parse_expr(
        text,
        local_dict={"x": x, "y": y},
        transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
    )


def _euler_values(f, x0, y0, h, n):
    ys = [Fraction(y0)]
    xs = [Fraction(x0)]
    for _ in range(n):
        s = Fraction(
            str(
                f.subs(
                    {
                        x: sympy.Rational(xs[-1].numerator, xs[-1].denominator),
                        y: sympy.Rational(ys[-1].numerator, ys[-1].denominator),
                    }
                )
            )
        )
        ys.append(ys[-1] + h * s)
        xs.append(xs[-1] + h)
    return xs, ys


def test_table_entries_are_euler_estimates():
    """Re-run Euler's method from the stem alone and compare with every blank."""
    for tier in ("easy", "medium"):
        for q in questions(PRACTICE, tier, range(1, 17)):
            h = Fraction(re.search(r"h = (\d+(?:\.\d+)?)", q.text).group(1))
            f = _slope_function(re.search(r"dy/dx = (.+?) with", q.text).group(1))
            x0, y0 = (Fraction(v) for v in re.search(r"y\((\d+(?:\.\d+)?)\) = (\d+(?:\.\d+)?)", q.text).groups())
            blanks = q.answer["blanks"]
            n = max(int(k.split(",")[0]) for k in blanks)
            xs, ys = _euler_values(f, x0, y0, h, n)
            for key, value in blanks.items():
                r, c = map(int, key.split(","))
                if c == 2:
                    assert abs(float(ys[r]) - float(value)) < 6e-4, (q.text, key, value)
                else:
                    slope = f.subs(
                        {
                            x: sympy.Rational(xs[r].numerator, xs[r].denominator),
                            y: sympy.Rational(ys[r].numerator, ys[r].denominator),
                        }
                    )
                    assert abs(float(slope) - float(value)) < 6e-4, (q.text, key, value)


def test_hard_error_uses_the_given_exact_solution():
    for q in questions(PRACTICE, "hard", range(1, 13)):
        h = Fraction(re.search(r"h = (\d+(?:\.\d+)?)", q.text).group(1))
        f = _slope_function(re.search(r"dy/dx = (.+?) with", q.text).group(1))
        x0, y0 = (Fraction(v) for v in re.search(r"y\((\d+(?:\.\d+)?)\) = (\d+(?:\.\d+)?)", q.text).groups())
        n = int(re.search(r"table of (\d+) steps", q.text).group(1))
        xs, ys = _euler_values(f, x0, y0, h, n)
        exact = _slope_function(
            re.search(r"exact solution is y = (.+?)\. Complete", q.text).group(1).replace("e^", "E^")
        )
        blanks = q.answer["blanks"]
        for i in range(1, n + 1):
            assert abs(float(blanks[f"{i},2"]) - float(ys[i])) < 6e-5, q.text
        ex = float(exact.subs(x, float(xs[-1])))
        assert abs(float(blanks[f"{n},3"]) - ex) < 6e-5, q.text
        assert abs(float(blanks[f"{n},4"]) - (ex - float(ys[-1]))) < 1.1e-4, q.text
