"""Derives the correct answer for both skills from the rendered question TEXT
ALONE, using only `mathgen.make` and `question.text`/`answer_text`. Imports
nothing from `mathgen.generators.p9c_leibniz_derivative_build`.
"""

import re
from fractions import Fraction

from sympy import Eq, Symbol, diff, solve, sympify

from mathgen import make

x = Symbol("x")

TOPIC_4 = "p9c_leibniz_derivative_correspondence"
TOPIC_5 = "p9c_increments_formula_estimate"


def _parse_f(f_text: str):
    """A textbook-style polynomial like "5x^2 - 3x + 9" -> a sympy Expr in x."""
    explicit = re.sub(r"(\d)(x)", r"\1*\2", f_text)
    return sympify(explicit.replace("^", "**"), locals={"x": x})


# ---------------------------------------------------------------------------
# Skill 4: senior-use-the-leibniz-notation-for-the-derivative-frac

_EVAL_RE = re.compile(
    r"y = f\(x\) = (?P<f>[^.]+)\. Using the correspondence dy/dx = f'\(x\), "
    r"find the value of dy/dx when x = (?P<p>-?\d+)\.$"
)
_INVERT_RE = re.compile(
    r"y = f\(x\) = (?P<f>[^.]+)\. dy/dx = f'\(x\) = (?P<T>-?\d+) when x takes "
    r"some value\. Using the correspondence dy/dx = f'\(x\), find that value of x\.$"
)


class TestLeibnizCorrespondenceOracle:
    def test_easy_and_medium_all_seeds(self):
        examined = 0
        for tier in ("easy", "medium"):
            for seed in range(60):
                q = make(TOPIC_4, seed=seed, difficulty=tier)
                m = _EVAL_RE.match(q.text)
                assert m, f"{tier} seed {seed}: could not parse {q.text!r}"
                f_expr = _parse_f(m.group("f"))
                p = int(m.group("p"))
                oracle_value = diff(f_expr, x).subs(x, p)
                assert oracle_value == sympify(q.answer_text), (
                    f"{tier} seed {seed}: oracle {oracle_value} != answer_text {q.answer_text!r}\ntext={q.text!r}"
                )
                examined += 1
        assert examined == 120

    def test_hard_all_seeds(self):
        examined = 0
        for seed in range(60):
            q = make(TOPIC_4, seed=seed, difficulty="hard")
            m = _INVERT_RE.match(q.text)
            assert m, f"hard seed {seed}: could not parse {q.text!r}"
            f_expr = _parse_f(m.group("f"))
            T = int(m.group("T"))
            derivative = diff(f_expr, x)
            solutions = solve(Eq(derivative, T), x)
            assert len(solutions) == 1, f"hard seed {seed}: expected a unique solution, got {solutions}"
            assert solutions[0] == sympify(q.answer_text), (
                f"hard seed {seed}: oracle {solutions[0]} != answer_text {q.answer_text!r}\ntext={q.text!r}"
            )
            examined += 1
        assert examined == 60


# ---------------------------------------------------------------------------
# Skill 5: senior-use-the-increments-formula-delta-y-cong-frac

_ESTIMATE_CHANGE_RE = re.compile(
    r"y = f\(x\) = (?P<f>[^.]+)\. Estimate the change in y, δy, using the "
    r"increments formula δy ≅ dy/dx × δx, for a change of δx = (?P<dx>\S+) "
    r"from x0 = (?P<x0>-?\d+)\.$"
)
_ESTIMATE_NEW_VALUE_RE = re.compile(
    r"y = f\(x\) = (?P<f>[^.]+)\. Estimate the new value of y when x changes "
    r"from x0 = (?P<x0>-?\d+) to x0 \+ δx, using the increments formula "
    r"δy ≅ dy/dx × δx, given δx = (?P<dx>\S+)\.$"
)


def _answer_value(q):
    return Fraction(q.answer["p"], q.answer["q"])


class TestIncrementsFormulaOracle:
    def test_easy_and_medium_all_seeds(self):
        examined = 0
        for tier in ("easy", "medium"):
            for seed in range(60):
                q = make(TOPIC_5, seed=seed, difficulty=tier)
                m = _ESTIMATE_CHANGE_RE.match(q.text)
                assert m, f"{tier} seed {seed}: could not parse {q.text!r}"
                f_expr = _parse_f(m.group("f"))
                dx = Fraction(m.group("dx"))
                x0 = int(m.group("x0"))
                deriv_value = diff(f_expr, x).subs(x, x0)
                oracle_value = Fraction(str(deriv_value)) * dx
                assert oracle_value == _answer_value(q), (
                    f"{tier} seed {seed}: oracle {oracle_value} != answer {q.answer}\ntext={q.text!r}"
                )
                examined += 1
        assert examined == 120

    def test_hard_all_seeds(self):
        examined = 0
        for seed in range(60):
            q = make(TOPIC_5, seed=seed, difficulty="hard")
            m = _ESTIMATE_NEW_VALUE_RE.match(q.text)
            assert m, f"hard seed {seed}: could not parse {q.text!r}"
            f_expr = _parse_f(m.group("f"))
            dx = Fraction(m.group("dx"))
            x0 = int(m.group("x0"))
            deriv_value = diff(f_expr, x).subs(x, x0)
            f_x0 = f_expr.subs(x, x0)
            oracle_value = Fraction(str(f_x0)) + Fraction(str(deriv_value)) * dx
            assert oracle_value == _answer_value(q), (
                f"hard seed {seed}: oracle {oracle_value} != answer {q.answer}\ntext={q.text!r}"
            )
            examined += 1
        assert examined == 60
