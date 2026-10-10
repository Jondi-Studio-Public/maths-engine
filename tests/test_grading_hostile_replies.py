"""Replies that used to keep the grader busy for seconds or minutes, and must not again.

The answer graders read text from a learner, and several of them hand it to SymPy or to `Fraction`. A short
reply can ask for something astronomically large: a tower of powers (`10^10^10`, `2^-2^-2^-...`), a factorial
of a million, a power with a symbolic base and a ten-digit exponent, a spoken "five to the power of fifty
million", the exponent in `1e99999999`, a hyperbolic function of 4096, a cosh of an exp of a power, an inverse
function of a number outside its domain, and thousands of spaces between two numbers. The cost
guard in `mathgen.grading.safe_parse` reads the *shape* of an expression and refuses such a reply unevaluated;
`Fraction` is only ever handed a plain number. The checks here send every reply in `HOSTILE` through
`grade()` for every kind that reads text, and fail if any takes two seconds or comes back correct.

This is a regression list, not a proof. Each entry is a reply that was found to hang or to take seconds, plus
the families around it (and `tests/test_grading_cost_guard.py` pins the limits). The guard bounds the shape of
an expression, not the time `simplify` takes on it: an expression built to stay inside every limit and still
be slow can exist (see the "known limit" paragraph of `safe_parse`), and nothing here says it cannot. A service
that grades replies it does not trust should also give each call a deadline of its own.

Each case is timed on its own. On a platform with `SIGALRM` a case that has stopped finishing is also cut
off after 20 s, so a regression fails here instead of freezing the run.
"""

from __future__ import annotations

import dataclasses
import signal
import time

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade, typed
from mathgen.grading.spoken import check_answer, spoken_to_expr
from mathgen.skills import load_skills

#: A reply that finishes within this is fine; the guard's own cost is milliseconds.
BUDGET_SECONDS = 2.0
#: A reply still running after this is stopped (where the platform allows), and the case fails.
HARD_STOP_SECONDS = 20


class _StillRunning(BaseException):
    """Raised by the alarm. Not an `Exception`, so `grade()`'s own backstop cannot swallow it."""


@pytest.fixture(autouse=True)
def _hard_stop():
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise _StillRunning(f"still grading after {HARD_STOP_SECONDS} s")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, HARD_STOP_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _nested(depth: int, inner: str = "x") -> str:
    text = inner
    for _ in range(depth):
        text = f"({text}+1)^2+1"
    return text


#: name -> reply. Every one of them is something a person could paste or type.
HOSTILE = {
    # -- powers of powers, including the negative exponents that used to slip past the guard
    "negative-exponent tower of numbers": "2^-" * 40 + "2",
    "negative-exponent tower of x": "x^-" * 40 + "2",
    "negative-exponent tower with **": "2**-" * 40 + "2",
    "tower 10^10^10": "10^10^10",
    "tower 9**9**9**9": "9**9**9**9",
    "tower of eight twos": "2**2**2**2**2**2**2**2",
    "x to a tower": "x^2^2^2^2^2",
    "fractional tower": "2^(1/2)^(1/2)^(1/2)^(1/2)^(1/2)",
    # -- a huge exponent or base
    "x^(10^10)": "x^(10^10)",
    "(x+1)^(10^6)": "(x+1)^(10^6)",
    "x^1000000": "x^1000000",
    "2**(10**7)": "2**(10**7)",
    "sqrt(10**1000000)": "sqrt(10**1000000)",
    "1e500000": "1e500000",
    "(x+y+z+w)^60": "(x+y+z+w)^60",
    "(10^20)^-1": "(10^20)^-1 + x",
    "number to a symbolic tower": "3^(135*x*y^50)*e^(x^-2)",
    "irrational constant to the 50th": "(243 + Abs((pi + 11)!*tan(8/11)))^50",
    "digit run of 100000": "9" * 100_000,
    # -- an exponent in a number: `Fraction("1e99999999")` works out 10**99999999
    "exponent 1e9999999": "1e9999999",
    "exponent 1e99999999": "1e99999999",
    "negative 1e9999999": "-1e9999999",
    "tiny 1e-9999999": "1e-9999999",
    "decimal 1.5e9999999": "1.5e9999999",
    "capital 1E9999999": "1E9999999",
    "in a list 1e9999999, 2, 3": "1e9999999, 2, 3",
    "in a list 1, 1e9999999, 3": "1, 1e9999999, 3",
    "in a bracketed list": "(1e9999999, 2, 3)",
    "fraction over an exponent": "1/1e9999999",
    "exponent with a sign": "1e+99999999",
    "underscores": "1_0e99999999",
    "ten thousand digits over one": "9" * 10_000 + "/1",
    "ten thousand digits point five": "9" * 10_000 + ".5",
    "digit run of 200000 ones": "1" * 200_000,
    "a 300-digit coefficient": "7" * 300 + "*x^2 + 1",
    # -- factorials and the functions that grow like them
    "(10^6)!": "(10^6)!",
    "99999999!": "99999999!",
    "factorial(10**7)": "factorial(10**7)",
    "((((9!)!)!)!)!": "((((9!)!)!)!)!",
    "binomial(10**12, 5*10**11)": "binomial(10**12, 5*10**11)",
    "fibonacci(10**9)": "fibonacci(10**9)",
    "primorial(10**6)": "primorial(10**6)",
    "prime(10**9)": "prime(10**9)",
    "factorial of a function of x": "(684 + z - tan(11))!",
    "factorial of a polynomial": "((4^4 + z*z^5)^11)!",
    "factorial inside exp": "exp(y + (z*536)!)",
    # -- other functions of a huge argument, and nesting
    "exp(849146628)": "exp(849146628) - x",
    "cosh(4096)": "cosh(4096)",
    "sinh(4096)": "sinh(4096)",
    "cosh(8000)": "cosh(8000)",
    "exp of a root of a log of an asin": "exp((y*ln((asin(x))/(x*10)))^(1/2))",
    "exp of cosh of a variable power": "exp(cosh(((e*x)^(x+1))^4))",
    "exp(1000) squared": "exp(1000)^2",
    "cosh(710) plus x": "cosh(710) + x",
    "exp of exp of exp": "exp(exp(exp(x)))",
    "a power with a coefficient past a float": "(99999999*x+1)^1000",
    "a power of a long coefficient": "(123456789012345678*x+1)^1000",
    "a power of 9x+9": "(9*x+9)^1000",
    "a power of 5x+3": "(5*x+3)^1000",
    "a power of 2x+1": "(2*x+1)^1000",
    "four symbols to the 15th": "(a+b+c+d)^15",
    # found by the random fuzz: each took SymPy over eight seconds (or never finished)
    "abs of a power of a power of cos": "Abs(exp(((cos(t))^3)^500))",
    "sign of a root of a factorial of i": "sign(cbrt((sqrt(I) + (I)!)))",
    "a log of pi, e and a root of asin(6), squared": "(ln((((2*pi)/exp(1)) + sqrt(asin(6)))))^2",
    "re of a high power": "re((x+I)^500)",
    "cot of x plus exp of x^(9 pi)": "(((((806580212 + y)*x) + ln(sin((y + y))))*pi) - cot((x + exp(((x)^9)^(pi)))))",
    "tan of a factorial to the 653": "tan((((z*12))!)^(653))",
    "floor of lcm of 831 and cos": "(floor(lcm(831, cos((asin(t))^10))) + log(x))",
    "lcm of 558 and exp of a cube": "lcm(558, exp(((733*(cot(z))^1))^3))",
    "abs of a power of a power with i": "abs(((E)^12*((Min(E, log(z)))^-5*(((I + z))^(x))^x)))",
    "sec of sin of a ninth power with i": "sec(sin((((883*I) + x))^9))",
    "cot of e to the 1000": "(cot(e))^1000",
    "floor(exp(10^7 x))": "floor(exp(10^7*x))",
    "gcd of two huge powers": "gcd(x^100000, x^99999)",
    "gcd of huge integers": "gcd(10**5000, 10**4999)",
    "log nested six deep": "ln(ln(ln(ln(ln(ln(x))))))",
    "sin nested six deep": "sin(sin(sin(sin(sin(sin(x))))))",
    "sqrt nested ten deep": "sqrt(" * 10 + "x" + ")" * 10,
    "squares nested twenty deep": _nested(20),
    "five thousand brackets": "(" * 5000,
    "five thousand brackets closed": "(" * 5000 + "x" + ")" * 5000,
    "infinity inside a function": "sec(oo*x)",
    # found by the second round of the fuzz: a complex number SymPy keeps as it is (`asin(2)`, `sec(2)^x`)
    # inside a function of a variable. The last one took 4.4 s, the others never finished.
    "sqrt of a log of asin(2) plus x, squared": "sqrt(log(asin(2)+x)^2)",
    "sqrt of a log of asin(5) plus x, to the -4": "sqrt(log(asin(5)+x)^-4)",
    "sqrt of a log of asin(2) times x, to the -4": "sqrt(log(asin(2)*x)^-4)",
    "abs of a root with sec(2) to the x": "Abs(((sec(2))^x+atan(x)+2-e)^(1/2))",
    "one over a root with asin(4) and asin(e)": (
        "1/sqrt(((cos(tan(x)))+(asin(5-1)))*((log(asin(e)))^3)-((tanh(e))+(x-x))^3/(1)^2)"
    ),
    "abs of a cos of a root of asin(3)": "Abs(cos((sqrt(asin(3)))*(cot((y)^-1))*x))",
    "cos of a root with asin(5), 4.4 s": "cos(sqrt(((sqrt((2)*(2)))+((3)*(3)/asin(5)))^3))",
    # -- long runs of whitespace: a separator written `\s*,\s*` was tried from every space of the run, and
    # several of them in a row made a reply of a few thousand characters take minutes
    "spaces around a comma in a point": "(" + " " * 100 + "," + " " * 100 + "x",
    "tabs around a comma in a point": "(" + "\t" * 130 + "," + "\t" * 130 + "x" + "\t" * 130 + "1,1)",
    "a bracket of 8000 spaces": "(" + " " * 8000 + ")",
    "two numbers 16000 spaces apart": "1" + " " * 16000 + "1",
    "two numbers 99990 spaces apart": "1" + " " * 99_990 + "1",
    "an arrow in 32000 spaces": "1" + " " * 16000 + "->" + " " * 16000 + "2",
    "a less-than in 32000 spaces": "1" + " " * 16000 + "<" + " " * 16000 + "2",
    "a comma in 32000 spaces": "1" + " " * 16000 + "," + " " * 16000 + "2",
    "a semicolon in 32000 spaces": "1" + " " * 16000 + ";" + " " * 16000 + "2",
    "an and in 32000 spaces": "1" + " " * 16000 + "and" + " " * 16000 + "2",
    "a line break in 32000 spaces": "1" + " " * 16000 + "\n" + " " * 16000 + "2",
    "a pair with 32000 spaces inside": "(1" + " " * 16000 + ",2" + " " * 16000 + ")",
    "spaces and tabs alternating": "(" + " \t" * 5000 + "1,2)",
    "no-break spaces": "1" + "\u00a0" * 16000 + "1",
    "opening brackets after a gap": "(1, " + " " * 10 + ")" + " (" * 2000,
    # found by the third round of the fuzz, on limits that had been raised for right answers (a deeper nest, a
    # higher degree, a power of a constant): each used to be refused for another reason and, with that limit
    # raised, ran for seconds or minutes
    "a function the parse holds back, to the 500th, in a secant": "((sec(cos(((1)!)^500)) - e))^97.820",
    "a lcm to the 500th in a cotangent": "(940 - ((((y - 46.278))^-3)^-4 + cot((lcm(9, 0))^500)))",
    "an arcsine of a lcm to the 1000th": "(asin((lcm(oo, 800))^1000)/6)",
    "a factorial of a huge arcsine": "(asin((((pi)^-4)!)^7478739283270786649969907953098309213884))!",
    "a held-back factorial to the ten-billionth": (
        "(((((cot(7))!)^10000000000)!*cot(((floor(x) + (pi)^e))^5))*cot(((11*x)*(((pi)^(11) + 5)*(pi)^10000000000))))"
    ),
    "a tangent of an abs to the 427th": "(0/(t - tan((abs(abs(11)))^427)))",
    "asin(1) to the 7 millionth": "(asin(1))^7242873",
    "a power of a power of a power of x over 27": "((((((x)^1/3)^1/9)^2)^272)^997)^1000",
    "a tangent of a root of a cube of a degree 229": "(34.76 + ((tan(sqrt(((E - (4*((7)^(e) + (t)^229))))^3))*287))^4)",
    "a cotangent of a cosine of a root of degree 31": "((cot((cos((((z*288578324) + (pi)^(7)))^30.22))^10))^1/4)^-4",
    "a secant over an exponential tower with big numbers": "sec(((t + y)*(((365931662*E) + (x - 10)))^(((596)^t - E))))",
    "a cosine of a sine of a root with a big number": "cos(sin(((2 + cos(e))/(sqrt(z)/abs(864076265)))))",
    "a floor of a gamma and a power of 121": "floor(((Abs((e)!)*(cot(z))^1/9)*(((11*11) - (t/11)))^413))",
    "a max to the 500th of a degree 793": "(Max(((z)^792*(x)^1/4), sign(Min((t + 1), (x)^-2))))^500",
    "exp of a power of a power with a variable in both exponents": "exp((((x)^1/8)^(z))^(((x - 12))^-3))",
    "a secant of an exp of y times a constant to the sixth": "sec(exp((y*((Min(7, E) - (10*pi)))^6)))",
    # -- words: the spoken reader, which a reply that is not valid maths falls back to
    "spoken power of fifty million": "5 to the power of 50000000",
    "spoken power of fifty million, with and": "and 5 to the power of 50000000",
    "spoken power in words": "ten to the power of ninety nine million",
    "spoken power of fifty million words": "five to the power of fifty million",
    "spoken tower": "2 to the power of " * 30 + "2",
    "spoken tower of x": "x to the power of 2 " * 30,
    "spoken, a thousand-digit exponent": "2 to the power of " + "9" * 1000,
    "twenty thousand millions": "nine million " * 20_000,
    "fifty thousand decimal places": "zero point " + "nine " * 50_000,
}


def _hand_built(**fields) -> Question:
    base = dict(
        topic="test", level=0, seed=1, text="q", latex="q", answer_latex="", working=[], difficulty="easy",
        distractors=[],
    )  # fmt: skip
    return Question(**{**base, **fields})


@pytest.fixture(scope="module")
def questions() -> dict[str, tuple[str, Question]]:
    """One question per grader (and per expression form), as (kind, question)."""
    _, skills = load_skills()
    by_id = {skill.id: skill for skill in skills}

    def from_skill(skill_id, **fields):
        question = engine.make_for_skill(by_id[skill_id], seed=1)
        return dataclasses.replace(question, **fields) if fields else question

    expression = _hand_built(answer="12*x**2 - 4*x + 1", answer_text="12x^2 - 4x + 1")
    table = {"columns": ["", "A", "B"], "rows": [["r1", 1, None], ["r2", None, 4]], "blanks": {"0,2": 5, "1,1": 3}}
    out = {
        # An answer no payload happens to contain: "((9!)!)" is one number, 9, and 9 is a fine answer.
        "number": ("number", _hand_built(answer=7919, answer_text="7919")),
        "number, text answer": ("number", _hand_built(answer="7919", answer_text="7919")),
        "number, p/q answer": ("number", _hand_built(answer={"p": 125, "q": 64}, answer_text="125/64")),
        "fraction": ("fraction", _first_fraction(skills)),
        "fraction, text answer": ("fraction", _hand_built(answer="3/20", answer_text="3/20")),
        "fraction, ratio": ("fraction", _hand_built(answer={"p": 12, "q": 5}, answer_text="12:5")),
        "pair": ("pair", from_skill("divide-quantity-in-ratio")),
        "list": ("list", from_skill("stationary-points")),
        "ordered_list": ("ordered_list", from_skill("permutation-composition")),
        "mcq": ("mcq", from_skill("add-within-10")),
        "expression": ("expression", expression),
        "table": ("table", _hand_built(answer=table, answer_text=typed.table_answer_text(table))),
        "set": ("set", _hand_built(answer=[1, 2, 3], answer_text="{1, 2, 3}")),
        "points": ("points", _hand_built(answer=[[1, 2], [3, 4]], answer_text="(1, 2) (3, 4)")),
        "polyline": (
            "polyline",
            _hand_built(answer={"pieces": [{"vertices": [[0, 0], [1, 1], [2, 0]]}]}, answer_text="x"),
        ),
        "line": ("line", _hand_built(answer=[[0, 0], [1, 1]], answer_text="x")),
    }
    for form in ("simplified_surd", "factored", "expanded", "single_fraction", "rationalised"):
        out[f"expression, {form}"] = ("expression", dataclasses.replace(expression, answer_form=form))
    out["expression, line_standard_form"] = (
        "expression",
        dataclasses.replace(expression, answer_form="line_standard_form", answer="2*x+3*y-6", answer_text="2x + 3y - 6 = 0"),
    )  # fmt: skip
    out["pair, equation"] = ("pair", from_skill("equation-from-table"))
    return out


def _first_fraction(skills):
    for skill in skills:
        if skill.status == "active" and skill.answer == "fraction":
            question = engine.make_for_skill(skill, seed=1)
            if isinstance(question.answer, dict) and {"p", "q"} <= set(question.answer):
                return question
    raise AssertionError("no fraction skill with a p/q answer")


CASES = [
    "number", "number, text answer", "number, p/q answer", "fraction", "fraction, text answer", "fraction, ratio",
    "pair", "pair, equation", "list", "ordered_list", "mcq",
    "expression", "expression, simplified_surd", "expression, factored", "expression, expanded",
    "expression, single_fraction", "expression, rationalised", "expression, line_standard_form",
    "table", "set", "points", "polyline", "line",
]  # fmt: skip


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_reply_is_graded_wrong_and_quickly(questions, case, name):
    kind, question = questions[case]
    started = time.perf_counter()
    result = grade(kind, question, HOSTILE[name])
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, f"{kind} ({case}) took {elapsed:.2f} s on {name!r}"
    if kind != "mcq":  # an mcq reply is an option number, so "(x+1)^(10^6)" is "1" by design
        assert result.correct is False


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_reply_is_never_correct_by_the_spoken_reader(questions, name):
    # `check_answer` is the spoken-transcript path: the same text, read as speech.
    _, question = questions["expression"]
    started = time.perf_counter()
    correct = check_answer(question, HOSTILE[name])
    spoken_to_expr(HOSTILE[name])
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, f"the spoken reader took {elapsed:.2f} s on {name!r}"
    assert correct is False


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_reply_is_refused_before_it_is_parsed(name):
    # The two parse entry points. A reply either parses to something cheap or is refused as `None`.
    started = time.perf_counter()
    typed._parse_expression(HOSTILE[name])
    typed._too_costly_to_parse(HOSTILE[name].strip())
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, f"parsing took {elapsed:.2f} s on {name!r}"


def test_the_spoken_fallback_reads_a_huge_power_as_nothing():
    assert spoken_to_expr("five to the power of fifty million") is None
    assert spoken_to_expr("5 to the power of 50000000") is None
    assert spoken_to_expr("2 to the power of " * 30 + "2") is None


def test_the_spoken_reader_still_reads_a_polynomial():
    import sympy

    x = sympy.Symbol("x")
    assert spoken_to_expr("twelve x squared minus four x plus one") == 12 * x**2 - 4 * x + 1
    assert spoken_to_expr("x to the power of 5") == x**5
    assert spoken_to_expr("two to the power of ten") == 1024


def test_a_correct_expression_is_still_correct_after_hostile_ones(questions):
    # The guard refuses by shape; it keeps no state that could turn a good answer away afterwards.
    kind, question = questions["expression"]
    for name in ("negative-exponent tower of x", "(10^6)!", "spoken power of fifty million"):
        assert grade(kind, question, HOSTILE[name]).correct is False
    assert grade(kind, question, "12x^2 - 4x + 1").correct is True
    assert grade(kind, question, "1 - 4x + 12x^2").correct is True


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("filler", ["9 ", "9,", "(1,2) ", "x ", "nine ", "-"])
def test_a_multi_megabyte_reply_is_refused_at_the_door(questions, case, filler):
    kind, question = questions[case]
    started = time.perf_counter()
    result = grade(kind, question, filler * (3_000_000 // len(filler)))
    assert time.perf_counter() - started < BUDGET_SECONDS
    assert result.correct is False
    assert result.canonical == question.answer_text


def test_a_reply_at_a_sensible_length_is_graded_as_always(questions):
    kind, question = questions["number"]
    assert grade(kind, question, "x = 7919 " + "." * 5_000).correct is True
    assert grade(kind, question, "x = 7919" + " " * 90_000).correct is True  # under the limit, still read
