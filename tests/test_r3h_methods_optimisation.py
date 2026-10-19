"""``methods-optimisation-context``: answers grade right, every distractor wrong, and the optimum is the true optimum."""

from __future__ import annotations

import math
import re

import sympy

from mathgen.generators import r3h_methods_optimisation_build as _build
from r3h_support import add_common_tests, questions

# the generator skips its brute-force grid search for speed; every question these tests draw runs it
_build.SECOND_ROUTE = True

PRACTICE = "r3h_methods_optimisation"
EXAM = "exd_x_r3h_methods_optimisation"
add_common_tests(globals(), PRACTICE, EXAM)


def _search(fn, lo, hi, maximise, n=200000):
    """Brute-force best (argument, value) of fn on a fine grid over (lo, hi)."""
    best = None
    for i in range(1, n):
        x = lo + (hi - lo) * i / n
        v = fn(x)
        if best is None or (v > best[1] if maximise else v < best[1]):
            best = (x, v)
    return best


def _model(text):
    """(function, lo, hi, maximise, kind) rebuilt from the numbers printed in the stem alone."""
    if "tin" in text:
        v = float(re.search(r"volume of (\d+) cm", text).group(1))
        return (lambda r: 2 * math.pi * r * r + 2 * v / r), 0.01, 40, False
    if "open-topped box" in text:
        v = float(re.search(r"(?:hold|volume of) (\d+) cm", text).group(1))
        return (lambda x: x * x + 4 * v / x), 0.01, 60, False
    if "parabola y = " in text and "rectangle" in text:
        k = float(re.search(r"y = (\d+) - x\^2", text).group(1))
        return (lambda x: 2 * x * (k - x * x)), 0, math.sqrt(k), True
    if "cardboard" in text:
        a, b = (float(v) for v in re.search(r"sheet of cardboard (\d+) cm by (\d+) cm", text).groups())
        return (lambda x: x * (a - 2 * x) * (b - 2 * x)), 0, a / 2, True
    if "business" in text:
        a, b = (float(v) for v in re.search(r"\((\d+) - (\d+)q\)", text).groups())
        cost = re.search(r"production is \(([^)]*)\)", text).group(1)
        c0, c1, c2 = re.match(r"(\d+) \+ (\d+)q \+ (\d*)q\^3", cost).groups()
        c0, c1, c2 = float(c0), float(c1), float(c2 or 1)
        return (lambda q: (a - b * q) * q - (c0 + c1 * q + c2 * q**3)), 0, a / b, True
    c = float(re.search(r"\(0, (\d+)\)", text).group(1))
    return (lambda x: x * x + (x * x - c) ** 2), 0, 12, False


def test_medium_and_hard_answers_are_the_true_optimum():
    for tier in ("medium", "hard"):
        for q in questions(PRACTICE, tier, range(1, 21)):
            fn, lo, hi, maximise = _model(q.text)
            x_best, v_best = _search(fn, lo, hi, maximise)
            asked_value = tier == "hard"
            answer = float(q.answer)
            if asked_value:
                if "least distance" in q.text:
                    v_best = math.sqrt(v_best)
                assert abs(answer - v_best) < max(0.06, 2e-4 * abs(v_best)), (q.text, answer, v_best)
            else:
                assert abs(answer - x_best) < 0.011, (q.text, answer, x_best)


def test_easy_function_matches_the_situation():
    for q in questions(PRACTICE, "easy", range(1, 25)):
        fn, lo, hi, maximise = _model(q.text)
        expr = sympy.sympify(q.answer)
        sym = sorted(expr.free_symbols, key=str)[0]
        for p in (1.3, 2.1, 3.7):
            if p < hi:
                assert abs(float(expr.subs(sym, p)) - fn(p)) < 1e-6 * max(1, abs(fn(p))), (q.text, q.answer)


def test_exam_second_derivative_shows_a_maximum():
    for q in questions(EXAM, "exam", range(1, 13)):
        assert float(q.parts[2].answer_text) < 0
