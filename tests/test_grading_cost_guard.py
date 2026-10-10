"""The cost guard in `mathgen.grading.safe_parse`: what it refuses, and that it never refuses a real answer.

A typed expression is parsed by SymPy, and a short one can ask for a computation that never finishes.
The guard reads the shape of the expression (before the parse, unevaluated, and again on the result) and
raises `TooCostly` for one that is too big, too deep or too tall a tower. Every limit sits several times
above anything the library's own answers reach, so a right answer in any ordinary spelling is never turned
away; these tests pin both sides.
"""

from __future__ import annotations

import contextlib
import signal
import time

import pytest
import sympy

from mathgen.grading import grade, typed
from mathgen.grading.safe_parse import (
    MAX_LENGTH,
    TooCostly,
    check_expression_cost,
    parse_expression_safely,
    sympify_safely,
    too_costly,
)

_T = typed._TRANSFORMATIONS


def _costly(text: str) -> bool:
    started = time.perf_counter()
    refused = too_costly(text, transformations=_T)
    assert time.perf_counter() - started < 1.0, f"measuring {text[:60]!r} took too long"
    return refused


# Each of these used to take SymPy seconds, minutes, or all the memory.
@pytest.mark.parametrize(
    "text",
    [
        # towers of powers, with positive, negative and fractional exponents
        "10**10**10",
        "9**9**9**9",
        "2**2**2**2**2**2**2**2",
        "2**-" * 40 + "2",
        "x**-" * 40 + "2",
        "x**2**2**2**2**2",
        "2**(1/2)**(1/2)**(1/2)**(1/2)**(1/2)",
        # a huge power of a number or of a symbol
        "x**(10**10)",
        "(x+1)**(10**6)",
        "x**1000000",
        "2**(10**7)",
        "sqrt(10**1000000)",
        "(10**50)**-1 + x",
        "(x+y+z+w)**60",
        # factorials
        "(10**6)!",
        "99999999!",
        "((((9!)!)!)!)!",
        "factorial(10**7)",
        "(684 + z - tan(11))!",
        "((4**4 + z*z**5)**11)!",
        "(t + 733)!",
        # functions of something huge, and nesting
        "exp(849146628) - x",
        "floor(exp(10**7*x))",
        "gcd(x**100000, x**99999)",
        "gcd(10**5000, 10**4999)",
        "ln(ln(ln(ln(ln(ln(x))))))",
        "sin(sin(sin(sin(sin(sin(x))))))",
        "sqrt(" * 10 + "x" + ")" * 10,
        "(243 + Abs((pi + 11)!*tan(8/11)))**50",
        "10**300 + pi",
        "3**(135*x*y**50)",
        "2**(123456*x)",
        "1e41*x",
        "sec(oo*x)",
    ],
)
def test_expression_that_costs_sympy_seconds_is_refused_unevaluated(text):
    assert _costly(text) is True
    with pytest.raises(TooCostly):
        parse_expression_safely(text, transformations=_T)


def test_the_cost_guard_refuses_a_nested_square_twenty_deep():
    text = "x"
    for _ in range(20):
        text = f"({text}+1)**2+1"
    assert _costly(text) is True


# What a learner may write, in any of the ordinary spellings, is read as before.
@pytest.mark.parametrize(
    "text",
    [
        "x**2 + 3*x - 5",
        "12*x**2 - 4*x + 1",
        "(x+1)**2",
        "(x+1)**50",
        "(x+y)**10",
        "(a+b+c)**8",
        "x**100",
        "2**10",
        "2**100",
        "2**-3",
        "10**-6",
        "(2/3)**5",
        "100!",
        "20!",
        "n!/(k!*(n-k)!)",
        "(n+1)!/(n-1)!",
        "(2*n)!",
        "n**2",
        "sqrt(50)",
        "6*sqrt(21)",
        "3*sqrt(2) + sqrt(8)",
        "sqrt(1 - x**2)",
        "sqrt(x**2 + 1)/x",
        "cbrt(27)",
        "e**(2*x)",
        "exp(-x**2/2)",
        "2**(x+1)",
        "1.05**(12*t)",
        "25*exp(-45*pi**2*t/2)",
        "sin(x)**2 + cos(x)**2",
        "sin(2*x)*cos(x)",
        "sin(30) + cos(180)",
        "tan(pi/4)",
        "ln(x)/x",
        "log(100)",
        "log(10**12)",
        "Abs(x - 3)",
        "(1 + sqrt(2))**5",
        "pi*r**2",
        "4/3*pi*r**3",
        "1/(x-1) + 1/(x+1)",
        "(x**2 - 1)/(x - 1)",
        "x/y + y/x",
        "(a*b)/(c*d)",
        "(" * 60 + "x" + ")" * 60,  # brackets that wrap nothing cost nothing
        "1000000*x",
        "86400*t",
        "0.000001*x",
        "123456789",
        "x**2*y**3*z",
        "3.14159*r",
        "gcd(12, 18)",
        "floor(7/2)",
        "oo",
        "-oo",
    ],
)
def test_ordinary_answers_are_not_refused(text):
    assert _costly(text) is False
    parse_expression_safely(text, transformations=_T)  # must not raise


def test_a_long_digit_run_is_measured_in_linear_time():
    # `\d{4,}\s*!` rescanned a digit run from every digit: 100,000 digits was 10^10 steps.
    started = time.perf_counter()
    assert too_costly("9" * 100_000, transformations=_T) is False  # over the length cap: the parse refuses it
    assert too_costly("9" * (MAX_LENGTH - 1), transformations=_T) in (True, False)
    assert time.perf_counter() - started < 1.0


def test_the_guard_checks_the_parsed_tree_as_well_as_the_text():
    # A text that is small but whose parse is large: the check on the result is what stops it.
    expr = sympy.Pow(sympy.Add(*sympy.symbols("a b c d e f g h")), 40, evaluate=False)
    with pytest.raises(TooCostly):
        check_expression_cost(expr)
    check_expression_cost(sympy.sympify("(x+1)**3"))


def test_a_pair_is_measured_item_by_item():
    check_expression_cost((sympy.Integer(1), sympy.Symbol("x")))
    with pytest.raises(TooCostly):
        check_expression_cost((sympy.Integer(1), sympy.Pow(sympy.Symbol("x"), 10**9, evaluate=False)))


def test_unevaluated_parse_is_never_cost_checked():
    # The cost guard itself reads the unevaluated tree; that parse must not call itself.
    tree = parse_expression_safely("2**2**2**2**2**2**2**2", transformations=_T, evaluate=False)
    assert isinstance(tree, sympy.Pow)


def test_the_guard_does_not_change_what_a_refusal_looks_like_to_callers():
    assert issubclass(TooCostly, ValueError)
    assert typed._parse_expression("10^10^10") is None
    assert typed._parse_expression("x^(10^10)") is None
    assert typed._parse_expression("(x+1)^2") == sympy.sympify("(x+1)**2")


@pytest.mark.parametrize(
    "reply",
    ["12x^2 - 4x + 1", "1 - 4x + 12x^2", "4x(3x - 1) + 1", "12*x**2-4*x+1", "x(12x - 4) + 1", " 12 x^2 - 4 x + 1 "],
)
def test_correct_expression_in_ordinary_spellings_still_grades_correct(reply):
    from mathgen.core import Question

    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer="12*x**2 - 4*x + 1", answer_text="12x^2 - 4x + 1",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    assert grade("expression", question, reply).correct is True


def test_sympify_safely_reads_what_generators_write_and_nothing_else():
    assert sympify_safely("x^2 + 1") == sympy.sympify("x**2 + 1")
    assert sympify_safely("(2, -5)") == (2, -5)
    assert sympify_safely(["1/2", "0.5"]) == [sympy.Rational(1, 2), sympy.Float("0.5")]
    assert sympify_safely(7) == 7
    for bad in ("__import__('os')", "x.__class__", "open('f')", "lambda: 1", "x; y", "a b c +"):
        with pytest.raises(sympy.SympifyError):
            sympify_safely(bad)


# -- limits that reach past the library, and the nesting and size rules added with them --------
#
# The first limits sat a few times above the library's own answers, and refused a right answer one step
# beyond it: `299792458t` (nine digits next to a variable), `x^300`, `sqrt(10^15)`, `(a+b+c+d)^10`. Each limit is
# now raised to the largest value at which the worst expression that fits is still graded in a few
# milliseconds (measured, with the probe's float ceiling gone: see `core._probe_differs`).
#
#   MAX_COEFFICIENT_DIGITS       9 -> 19 -> 40  (the longest number `_check_tokens` reads: `6.022e23*n`, `10^23*n`)
#   MAX_SYMBOLIC_EXPONENT      256 -> 1000
#   MAX_EXPANDED_TERMS        1000 -> 5000, for a polynomial counted from its terms: (a+b+c+d)^15 is 816, (a+b)^100 is
#                             101 (anything with a function of a symbol or a root in it is counted as before)
#   MAX_DEGREE                (new) 5000, the most the old monomial count allowed one symbol
#   MAX_RADICAND_DIGITS         12 -> 19 -> 100  (`cbrt(10^21)`)
#   MAX_TRANSCENDENTAL_ARGUMENT 10^4 -> 700 (exp(700) is the largest e^x a float holds), for exp, sinh, cosh,
#                               sech and csch only; sin, cos, tan, atan, tanh and the rest stop at 10^12
#   MAX_PERIODIC_DEGREE         40 -> 1000 for a polynomial argument (40 stays for one with a function or a root in it),
#                               and 8 when the argument has a factorial of a variable in it
#
# and the new rules: transcendental functions (and powers with a variable in the exponent, and roots of
# anything that has one in it) nest at most three deep, and a chain of three holds no number of five digits or more,
# an inverse function of a real number outside its domain
# (`asin(2)`) and a negative irrational number to a variable power (`sec(2)^x`) are refused, and a number with
# pi, e, a root or a function in it stays below 10^306.

_WORST_INSIDE_THE_LIMITS = [
    "x**300",
    "x**1000",
    "(x+1)**300",
    "(x+1)**1000",
    "(2*x+1)**1000",
    "(5*x+3)**1000",
    "(99999999*x+1)**1000",
    "(123456789012345678*x+1)**1000",
    "(a+b+c+d)**10",
    "(a+b+c+d)**15",
    "(x+y)**70",
    "(x+y+z)**25",
    "299792458*t",
    "123456789012345678*t",
    "sqrt(10**15)",
    "sqrt(10**18)",
    "sqrt(999999866000004473)",
    "sqrt(999999866000004473*x)",
    "cosh(700)",
    "exp(699)",
    "sinh(10)*x",
    "sin(ln(x))",
    "e**(sin(x))",
    "exp(sin(x)**2)",
    "exp(-x**2/2)",
    "ln(1 + e**x)",
    "x**x",
    "2**sin(x)",
    "sin(30) + cos(180)",
    # a part of a complex number, of an expression of any degree an answer has
    "Abs(x**100)",
    "Abs((x+1)**100)",
    "sign((x+1)**100)",
    "re((x+I)**12)",
    "sign(x)*x",
    # an exponential function of the degrees an answer has, a trigonometric function of an angle of degree 40
    "exp(x**4)",
    "exp(-x**2/2)",
    "cosh(x**4)",
    "exp((x+9)**3)",
    "sin(x**40)",
    "tan(x**40)",
    "gcd(x**2 - 1, x**2 + x)",
    "lcm(6*x, 9*x**2)",
    "gcd(12, 18)",
    # the imaginary unit beside one transcendental function
    "exp(I*x)",
    "exp(I*x) + exp(-I*x)",
    "cos(x) + I*sin(x)",
    "exp(I*pi*x)",
    "exp(2*I*x)",
    # the check for an infinity used to ask SymPy whether this power was one, and it took seconds to say
    # (with asin(6), a number outside the domain, which is now refused: see the next list)
    "ln(2*pi/e + sqrt(asin(1/6)))**2",
    "(ln(2*pi/exp(1) + sqrt(asin(1/6))))**2",
    # -- three levels of transcendental functions: the composite and its chain-rule derivative
    "sin(sin(sin(x)))",
    "ln(ln(ln(x)))",
    "sin(cos(e**x))",
    "ln(sin(e**x))",
    "ln(cos(sin(x)))",
    "e**(cos(sin(x)))",
    "ln(1 + ln(1 + ln(x)))",
    "sin(2*sin(2*sin(2*x)))",
    "cos(sin(sin(x)))*cos(sin(x))*cos(x)",
    "-e**(cos(sin(x)))*sin(sin(x))*cos(x)",
    "exp(sqrt(ln(x/10)))",
    # -- a number past 700 in a function that repeats or stays bounded
    "sin(720)",
    "tan(765)",
    "cos(1080)",
    "sin(701)",
    "sin(360*3)",
    "sin(2*pi*1000)",
    "cos(1000*pi)",
    "sin(x)*cos(720)",
    "atan(1000)",
    "tanh(1000)",
    "sin(10**12)",
    "tan(10**11/7)",
    "atan(10**11*pi)",
    "sin(x**50)",
    "cos(x**1000)",
    # -- the composed and the nested: the limit is on the degree for a polynomial, 40 when a function or a root is in it
    "sin(sqrt(x**40))",
    "sin(exp(x)**40)",
    "cos(sin(cos(10000*x)))",
    "cos(sin(cos(sqrt(x))))",
    "sin(cos(tan(x)))",
    "exp(sin(cos(x)))",
    "exp(y*(3 - 10*pi))",
    "y*(3 - pi)**6",
    "tan(((12*z)!)**8)",
    # -- a coefficient, a radicand and a power of a constant a little past what the library writes
    "6.022e23*n",
    "10**23*n",
    "6.02214076e23*n + 1.380649e-23",
    "cbrt(10**21)",
    "sqrt(2**100)",
    "sqrt(10**99)",
    "sqrt(2)**100",
    "pi**25",
    "(2*pi)**30",
    "(1 + sqrt(2))**20",
    # -- how many terms it multiplies out to, not the degree and the symbols: (a+b)^100 has 101
    "(a+b)**100",
    "(a+b)**1000",
    "(a+b+c)**60",
    "x**300*y**300",
    "(x*y)**400",
    "(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10)*(x-11)*(x-12)*(x-13)",
    # -- a negative number to a power with a variable in it, when it is whole or algebraic
    "(-2)**n",
    "(-1/2)**n",
    "(1 - sqrt(5))**n",
    "((1 + sqrt(5))**n - (1 - sqrt(5))**n)/(2**n*sqrt(5))",
    # -- an inverse function of a number inside its domain
    "asin(1/2)*x",
    "acos(-1/3) + x",
    "atan(5)*x",
    "ln(asin(x))",
]


@pytest.mark.parametrize("text", _WORST_INSIDE_THE_LIMITS)
def test_a_big_but_ordinary_expression_is_neither_refused_nor_slow(text):
    from mathgen.core import Question

    assert _costly(text) is False
    parse_expression_safely(text, transformations=_T)  # must not raise
    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer="12*x**2 - 4*x + 1", answer_text="12x^2 - 4x + 1",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    started = time.perf_counter()
    assert grade("expression", question, text).correct is False
    assert time.perf_counter() - started < 1.0, f"{text} took {time.perf_counter() - started:.2f} s to grade"


_JUST_PAST_THE_LIMITS = [
    "x**1001",
    "(x+1)**1001",
    "x**100000",
    "1e41*t",
    "10**41*t",
    "x*exp(60)",
    "x*exp(100)",
    "sqrt(10**101)",
    "sqrt(2**400)",
    "cbrt(10**150)",
    "(a+b+c+d)**30",
    "(a+b+c+d+e)**20",
    "(x+y+z+w)**60",
    # a function of a number past what a float holds, alone or beside anything
    "cosh(4096)",
    "sinh(4096)",
    "cosh(8000)",
    "cosh(710)",
    "exp(701)",
    "exp(1000) - x",
    "sin(10**13)",
    "tan(10**13)",
    "atan(10**13)",
    "tanh(10**13)",
    "sech(701)",
    "csch(1000)",
    "cosh(700)**2",
    "cosh(700)*cosh(700)",
    "exp(700)*x",
    "exp(100000*x)",
    "exp(99999*x)",
    "sinh(10**6*x)",
    # transcendental functions nested four deep (a root of something with one in it counts as a level)
    "exp((y*ln((asin(x))/(x*10)))**(1/2))",
    "exp(sqrt(ln(sin(x))))",
    "exp(cosh(((e*x)**(x+1))**4))",
    "exp(exp(exp(exp(x))))",
    "ln(ln(ln(ln(x))))",
    "sin(cos(tan(sin(x))))",
    "exp(sin(cos(exp(x))))",
    "e**(e**(e**(e**x)))",
    "x**(x**(x**(x**x)))",
    "2**(2**(2**(2**x)))",
    # an inverse function of a real number outside its domain is a complex number SymPy keeps as it is, and
    # anything built on one (`sqrt(log(asin(2) + x)^2)`) is where `simplify` stops finishing
    "asin(2)",
    "acos(-3)",
    "asin(2)*x",
    "sqrt(log(asin(2) + x)**2)",
    "sqrt(log(asin(5) + x)**-4)",
    "sqrt(log(asin(2)*x)**-4)",
    "asin(sqrt(2))",
    "asin(5 - 1)",
    "log(asin(E))*x",
    "ln(2*pi/e + sqrt(asin(6)))**2",
    "(ln(2*pi/exp(1) + sqrt(asin(6))))**2",
    # a negative number that is not whole or algebraic, to a power with a variable in it: complex, with a branch
    # of the logarithm to pick
    "sec(2)**x",
    "cos(3)**x",
    "(1 - pi)**n",
    "Abs((sec(2)**x + atan(x) + 2 - e)**(1/2))",
    # a factorial of something with pi in it, or of a complex number
    "(28*x - pi)!",
    "sign(cbrt(sqrt(I) + (I)!))",
    "(1 + 2*I)!",
    # exp or a hyperbolic function of a high degree (`e^(x^9)` was refused already), and of a big coefficient
    # once it is raised to the power
    "exp(x**5)",
    "sinh(x**9)",
    "cot(x + exp((x**9)**pi))",
    "exp((x+99)**3)",
    "exp((x+100000)**2)",
    "lcm(558, exp((733*cot(z))**3))",
    # a trigonometric function of a high degree: its argument is a number with hundreds of digits at a probe point
    "sin(x**1001)",
    "cos(x**501*y**501)",
    "tan(((12*z)!)**9)",
    "sin(((12*z)!*(12*w)!)**5)",
    "tan(((12*z)!)**653)",
    # two transcendental functions nested around the imaginary unit: sin(i*y) is sinh(y)
    "sin(exp(I*x))",
    "exp(I*sin(x))",
    "sec(sin((883*I + x)**9))",
    "abs(E**12*Min(E, log(z))**-5*((I + z)**x)**x)",
    # gcd or lcm of something that is not a polynomial
    "floor(lcm(831, cos(asin(t)**10))) + log(x)",
    "gcd(sin(x), cos(x))",
    "lcm(2, sqrt(x))",
    # the real or imaginary part (or the size, or the sign) of a high power: SymPy takes it apart term by term
    "Abs(exp(cos(t)**1500))",
    "Abs(exp(((cos(t))**3)**500))",
    "Abs((x+1)**101)",
    "re((x+I)**101)",
    "im(x**500)",
    "sign((x+y)**200)",
    "conjugate(exp(x)**300)",
    "arg((x+1)**1000)",
    # held back by the measuring parse, so its value is not known, and SymPy would multiply out its power
    "cos(factorial(1)**500)",
    "asin(1)**1000000",
    "floor(7.5)**500",
    # the degree of a power of a power of a power (`(((x/27)^272)^997)^1000` is `x^271184000` over a number of 388 million digits)
    "(((x/27)**272)**997)**1000",
    # a function or a root in the argument of a trigonometric function: a lower degree than for a polynomial
    "tan(sqrt((t**229 + 1)**3))",
    "sin(sqrt(x**41))",
    "cot(cos((288578324*z+pi**7)**30.22)**10)",
    # a large number in a chain of three
    "cos(sin(cos(100000*x)))",
    "sec((t+y)*(365931662*E+x-10)**(596**t-E))",
    "cos(sin(2 + cos(e) + 864076265/sqrt(z)))",
    # three deep with two exponentials in the chain (`exp`, or a power with a variable in the exponent)
    "exp(exp(exp(x)))",
    "e**(e**(e**x))",
    "x**(x**(x**x))",
    "2**(2**(2**x))",
    # a number with pi, e, a root or a function in it next to a variable: its digits are an upper bound, so 19, not 40
    "sec(exp(y*(3 - 10*pi)**6))",
    "y*(3 - 10*pi)**6",
]


@pytest.mark.parametrize("text", _JUST_PAST_THE_LIMITS)
def test_one_step_past_a_limit_is_refused(text):
    started = time.perf_counter()
    with pytest.raises(TooCostly):
        parse_expression_safely(text, transformations=_T)
    assert typed._parse_expression(text.replace("**", "^")) is None
    assert time.perf_counter() - started < 1.0


def test_the_transcendental_limit_counts_a_power_with_a_variable_in_the_exponent():
    for text in (
        "exp(2**x)", "e**(2**x)", "sin(2**x)", "2**sin(x)", "x**(sin(x))", "ln(x**x)",
        "e**(sin(ln(x)))", "ln(sin(exp(x)))",
    ):  # fmt: skip
        assert _costly(text) is False, text
    for text in (
        "exp(sin(ln(2**x)))", "exp(sin(x**x)**(sin(x)))", "exp(exp(x**x)**x)", "e**(sin(ln(x**x)))",
        # three deep with two exponentials in the chain (`exp`, or a power with a variable in the exponent)
        "exp(sin(2**x))", "sin(x**x)**(sin(x))", "exp(x**x)**x", "e**(e**(e**x))", "exp(exp(exp(x)))", "exp(sin(exp(x)))",
        "exp(((x**(1/8))**z)**((x - 12)**-3))",
    ):  # fmt: skip
        with pytest.raises(TooCostly):
            parse_expression_safely(text, transformations=_T)


def test_the_size_limit_of_a_number_in_a_function_depends_on_how_the_function_grows():
    # exp, sinh, cosh, sech and csch grow or shrink like exp of the argument: 700 is where a float ends.
    for text in ("exp(700)", "exp(-700)", "sinh(700)", "cosh(-700)", "sech(700)", "csch(700)"):
        assert _costly(text) is False, text
    for text in ("exp(701)", "exp(-701)", "sinh(701)", "cosh(750)", "cosh(-750)", "sech(701)", "csch(-1000)"):
        assert _costly(text) is True, text
    # the rest repeat, stay bounded or grow like a logarithm: the limit is a million million
    for name in ("sin", "cos", "tan", "cot", "sec", "csc", "atan", "acot", "tanh", "coth", "asinh", "acosh"):
        assert _costly(f"{name}(10**12)") is False, name
        assert _costly(f"{name}(10**13)") is True, name


def test_an_inverse_function_is_refused_only_for_a_real_number_outside_its_domain():
    for text in (
        "asin(1)", "asin(-1)", "acos(1)", "asin(1/2)", "acos(-0.5)", "asin(sqrt(2)/2)", "asin(sqrt(3)/2)*x",
        "acosh(1)", "acosh(2)", "atanh(1/2)", "asec(2)", "acsc(-2)", "acsc(1)", "acosh(10**6)",
        "asin(x)", "acos(2*x)", "asin(x + 2)", "ln(asin(x))", "asin(2*I)",
    ):  # fmt: skip
        assert _costly(text) is False, text
    for text in (
        "asin(2)", "asin(-1.5)", "acos(3)", "acos(sqrt(2))", "acosh(0)", "acosh(1/2)", "acosh(-2)", "atanh(2)",
        "atanh(-3/2)", "asec(1/2)", "asec(0.9)", "acsc(1/2)", "asin(2) + x", "x*acos(10)", "sin(asin(5))",
        "sqrt(5 - asin(2))", "asin(pi)", "asin(1 + 1/1000)", "acos(-2)**2",
    ):  # fmt: skip
        assert _costly(text) is True, text


def test_a_negative_number_that_is_not_whole_or_algebraic_is_refused_to_a_variable_power():
    for text in (
        "(-2)**n", "(-1/2)**n", "(-1.5)**n", "(1 - sqrt(5))**n", "((1 - sqrt(5))/2)**n", "(-sqrt(2))**n", "2**x",
        "sec(1)**x", "cos(1)**n", "(1 + sqrt(2))**n", "e**x", "pi**x", "(pi - 3)**n", "(-1)**n*x",
    ):  # fmt: skip
        assert _costly(text) is False, text
    for text in (
        "sec(2)**x",
        "cos(3)**n",
        "(1 - pi)**n",
        "(-pi)**x",
        "(2 - exp(2))**n",
        "(ln(1/2))**x",
        "sin(5)**(2*x)",
    ):
        assert _costly(text) is True, text


def test_the_expanded_terms_are_counted_from_the_terms_and_not_the_degree():
    # (a+b)^100 is 101 terms; the degree and the two symbols would have said 5,151 monomials. A polynomial is counted
    # from its terms; anything with a function of a symbol, a root or a variable exponent in it still from its degree
    # and symbols (`sin(x)*(a+b)^100` is 182,104), because SymPy's work on those is not in the terms.
    for text in (
        "(a+b)**100", "(x+y)**200", "(a+b)**1000", "(a+b+c)**60", "(a+b+c+d)**15", "x**300*y**300", "(x*y)**400",
        "(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10)*(x-11)*(x-12)*(x-13)",
        "(x+1)**100 + (x-1)**100", "(x+y)**30 * (x-y)**30", "(sin(x) + cos(x))**100",
    ):  # fmt: skip
        assert _costly(text) is False, text
    for text in (
        "(a+b)**100*(c+d)**100", "(a+b+c+d)**60", "(a+b+c+d+e+f)**15", "(a+b+c)**100",
        "(sin(x) + cos(x))**100 * (sin(y) + cos(y))**100", "sin(x)*(a+b)**100", "(a+b)*(c+d)*(e+f)*(g+h)*(i+j)*(k+l)*(m+n)*(o+p)*(q+r)*(s+t)*(u+v)*(w+z)*(aa+ab)",
    ):  # fmt: skip
        assert _costly(text) is True, text


def test_a_power_of_an_irrational_constant_is_refused_only_when_it_is_a_sum_or_a_function_value():
    for text in (
        "pi**25",
        "pi**500",
        "sqrt(2)**100",
        "(2*pi)**30",
        "(sqrt(2)*pi)**50",
        "(1 + sqrt(2))**20",
    ):
        assert _costly(text) is False, text
    for text in (
        "(pi + 11)**50", "(1 + sqrt(2))**25", "(2*(pi + 1))**30", "(243 + pi)**21", "cos(1)**200", "asin(1)**1000000", "floor(7.5)**500",
    ):  # fmt: skip
        assert _costly(text) is True, text


def test_the_float_ceiling_no_longer_sends_the_probe_to_simplify():
    # `cosh(4096) - 12x^2 + 4x - 1` is far past a float, so the numeric probe used to skip every point and leave
    # the comparison to `simplify`, which does not finish. The probe now compares SymPy numbers.
    from mathgen.core import _probe_differs, symbolically_equal

    x = sympy.Symbol("x")
    started = time.perf_counter()
    assert _probe_differs(sympy.cosh(4096), 12 * x**2 - 4 * x + 1) is True
    assert _probe_differs((x + 1) ** 3000, 12 * x**2 - 4 * x + 1) is True
    assert _probe_differs(sympy.exp(10**6 * x), x) is True
    assert symbolically_equal(sympy.cosh(4096) + x, sympy.cosh(4096) + x) is True
    assert symbolically_equal((x + 1) ** 3000, (x + 1) ** 3000) is True
    assert time.perf_counter() - started < 2.0


def test_the_probe_still_proves_what_it_proved_and_still_declines_what_it_declined():
    from mathgen.core import _probe_differs

    x, y = sympy.symbols("x y")
    assert _probe_differs(x**2 + 3 * x - 5, x**2 + 3 * x - 4) is False  # the difference has no symbol: inconclusive
    assert _probe_differs(x**2 + y, x**2 + 2 * y) is True
    assert _probe_differs(sympy.sin(x) ** 2, 1 - sympy.cos(x) ** 2) is False  # equal: noise is not a difference
    assert _probe_differs(sympy.sqrt(x - 5), sympy.sqrt(x - 5) + x) is True  # complex at every point
    pole = 1 / (x - sympy.Rational(3, 7))
    assert _probe_differs(pole, pole + x) is True  # one of the points is the pole, and is skipped
    assert _probe_differs(sympy.Function("f")(x), sympy.Function("f")(x) + 1) is False  # not a number at any point


def test_a_bare_infinity_is_still_an_answer_and_an_infinity_inside_one_is_not():
    for bare in (sympy.oo, -sympy.oo, sympy.Mul(-1, sympy.oo, evaluate=False)):
        check_expression_cost(bare)  # must not raise
    check_expression_cost((-sympy.oo, sympy.Integer(3)))
    for text in ("oo + x", "x*oo", "sec(oo*x)", "ln(oo)*x"):
        with pytest.raises(TooCostly):
            parse_expression_safely(text, transformations=_T)


def test_the_infinity_check_asks_atoms_only():
    # `is_infinite` on a power or a function is an assumptions query, and for this one it took longer than 25 s.
    from mathgen.grading.safe_parse import _is_bare_infinity

    x = sympy.Symbol("x")
    tree = sympy.log(2 * sympy.pi / sympy.E + sympy.sqrt(sympy.asin(6)), evaluate=False) ** 2 + x
    started = time.perf_counter()
    assert _is_bare_infinity(tree) is False
    with contextlib.suppress(TooCostly):  # the walker refuses asin(6) first now; the infinity check is asked too
        check_expression_cost(tree)
    assert time.perf_counter() - started < 1.0


@contextlib.contextmanager
def _stopped_after(seconds: int):
    """Cut a call that has stopped finishing off after `seconds` (where the platform has `SIGALRM`), so that
    a regression fails the test instead of freezing the run."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    class StillRunning(BaseException):  # not an Exception: grade()'s own backstop must not swallow it
        pass

    def stop(signum, frame):
        raise StillRunning(f"still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "reply", ["cot(e)^1000", "(cot(e))^1000", "tan(e)^900", "sec(e)^1000 + 1", "cot(e)^1000 - x + x"]
)
def test_the_cost_is_measured_again_once_e_is_read_as_eulers_number(reply):
    # `e` is a variable when the reply is parsed, so `cot(e)^1000` is a thousand-term polynomial and passes;
    # once the question says it is Euler's number it is a power of a constant, and `simplify` takes minutes.
    from mathgen.core import Question

    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer="6*sqrt(21)", answer_text="6sqrt(21)",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    started = time.perf_counter()
    with _stopped_after(20):
        assert grade("expression", question, reply).correct is False
    assert time.perf_counter() - started < 1.0


def test_reading_e_as_eulers_number_still_works_for_an_ordinary_answer():
    from mathgen.core import Question

    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer="exp(2*x)", answer_text="e^(2x)",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    for reply in ("e^(2x)", "exp(2x)", "(e^x)^2", "e^(2*x)"):
        assert grade("expression", question, reply).correct is True, reply


def test_a_reply_that_cannot_be_built_once_e_is_a_number_is_wrong_and_not_an_error(caplog):
    # `Max(e*I, 2)` is fine while `e` is a variable; with `e` as Euler's number SymPy cannot order `E*I` and 2.
    import logging

    from mathgen.core import Question

    question = Question(
        topic="test", level=0, seed=1, text="q", latex="q", answer="x + 1", answer_text="x + 1",
        answer_latex="", working=[], difficulty="easy", distractors=[],
    )  # fmt: skip
    with caplog.at_level(logging.ERROR, logger="mathgen.grading"):
        assert grade("expression", question, "Max(e*I, 2)").correct is False
        assert grade("expression", question, "x + 1").correct is True
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
