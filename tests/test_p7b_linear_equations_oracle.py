"""Independent oracle for the `p7b_linear_equations` questions: it re-derives each answer from the rendered
question text.

Parses each rendered question and derives the answer from `question.text` alone -- never imports
`mathgen.generators.p7b_linear_equations_build`."""

import re
from fractions import Fraction

from sympy import Eq, Poly, PolynomialError, Rational, solve, symbols

from mathgen.skills import load_skills
from mathgen import make

x = symbols("x")


def _frac_from_text(t: str) -> Fraction:
    t = t.strip()
    if "/" in t:
        num, den = t.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(t))


class TestSolveLinearEquationsOracle:
    """Computational oracle for senior-solve-linear-equations: re-derives x0
    from the rendered equation text with sympy, independent of the generator's
    own hand-Fraction construction route."""

    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-solve-linear-equations")

        # "Solve for x: (<numer1>)/<p> = <numer2>" (easy)
        easy_re = re.compile(r"^Solve for x: \(([^)]+)\)/(\d+) = (.+)$")
        # "Solve for x: (<numer1>)/<p> - (<numer2>)/<q> = <d>" (medium)
        medium_re = re.compile(r"^Solve for x: \(([^)]+)\)/(\d+) - \(([^)]+)\)/(\d+) = (-?\d+)$")
        # "Solve for x: (2x - a)/3 + (x + b)/5 = (x - c)/2 <sign> <d>" (hard)
        hard_re = re.compile(r"^Solve for x: \(([^)]+)\)/3 \+ \(([^)]+)\)/5 = \(([^)]+)\)/2 ([+-]) (\d+)$")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
                examined += 1
                text, answer_text = q.text, q.answer_text

                m = hard_re.match(text)
                if m:
                    n1, n2, n3, sign, d_abs = m.groups()
                    lhs = parse_binomial(n1) / 3 + parse_binomial(n2) / 5
                    d = int(d_abs) if sign == "+" else -int(d_abs)
                    rhs = parse_binomial(n3) / 2 + d
                    _check(lhs, rhs, answer_text, text, challenged)
                    continue

                m = medium_re.match(text)
                if m:
                    n1, p, n2, q_, d = m.groups()
                    lhs = parse_binomial(n1) / int(p) - parse_binomial(n2) / int(q_)
                    rhs = int(d)
                    _check(lhs, rhs, answer_text, text, challenged)
                    continue

                m = easy_re.match(text)
                if m:
                    n1, p, n2 = m.groups()
                    lhs = parse_binomial(n1) / int(p)
                    rhs = parse_binomial(n2)
                    _check(lhs, rhs, answer_text, text, challenged)
                    continue

                challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"senior-solve-linear-equations oracle: examined {examined}, challenged 0")


def parse_binomial(text: str):
    """Parse a rendered ``join_terms`` binomial like 'x + 5', '-2x - 3', 'x^2 - 4'
    into a sympy expression. The only variable terms the generator ever emits
    here are 'x', '-x', 'Nx', '-Nx' or a bare constant."""
    text = text.strip()
    text = text.replace("x^2", "x**2")
    # implicit multiplication ("3x" -> "3*x") and unary leading minus are both
    # handled by sympy's parser via a direct string substitution for the one
    # shape this generator ever emits: an optional signed coefficient directly
    # against "x", nothing else compound.
    text = re.sub(
        r"(?<![\w.])(-?\d*)x\b",
        lambda m: (
            f"({m.group(1) or '1'})*x" if m.group(1) not in ("", "-") else ("(-1)*x" if m.group(1) == "-" else "(1)*x")
        ),
        text,
    )
    return Rational(1) * symbols_eval(text)


def symbols_eval(text: str):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(text, local_dict={"x": x}, transformations=transformations)


def _check(lhs, rhs, answer_text, text, challenged):
    computed = solve(Eq(lhs, rhs), x)
    if len(computed) != 1:
        challenged.append({"issue": f"expected 1 solution, sympy found {computed}", "text": text})
        return
    expected = _frac_from_text(answer_text)
    if Rational(computed[0]) != Rational(expected.numerator, expected.denominator):
        challenged.append(
            {
                "issue": f"answer mismatch: oracle computed x={computed[0]}, answer_text says {answer_text}",
                "text": text,
            }
        )


class TestIdentifyAndSolveOracle:
    """Enumerative/computational oracle for senior-identify-and-solve-linear-
    equations: for every rendered equation, independently re-derive whether it
    is linear (via sympy.Poly on the expanded lhs - rhs, catching
    PolynomialError for a reciprocal-in-the-unknown shape) and, if linear, the
    solved value -- then checks the rendered answer_text against that."""

    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-identify-and-solve-linear-equations")

        linear_re = re.compile(r"^Linear; x = (-?\d+(?:/\d+)?)$")
        not_linear_text = "Not linear"

        # ax + b = c
        simple_re = re.compile(r"^(.+) = (-?\d+)\. Is this equation")
        # a/(x + d) = c  (reciprocal)
        reciprocal_re = re.compile(r"^(-?\d+)/\(([^)]+)\) = (-?\d+)\. Is this equation")
        # (numer1)/p = c   OR   (n1)(n2) [+-] x^2 = c
        bracket_frac_re = re.compile(r"^\(([^)]+)\)/(\d+) = (-?\d+)\. Is this equation")
        bracket_pair_re = re.compile(r"^\(([^)]+)\)\(([^)]+)\) ([+-]) x\^2 = (-?\d+)\. Is this equation")

        examined = 0
        challenged = []
        for source in skill.sources:
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
                examined += 1
                text, answer_text = q.text, q.answer_text

                m = bracket_pair_re.match(text)
                if m:
                    n1, n2, sign, c = m.groups()
                    lhs = parse_binomial(n1) * parse_binomial(n2)
                    lhs = lhs + x**2 if sign == "+" else lhs - x**2
                    rhs = int(c)
                    _check_classification(lhs, rhs, answer_text, text, challenged)
                    continue

                m = reciprocal_re.match(text)
                if m:
                    a, denom, c = m.groups()
                    lhs = int(a) / parse_binomial(denom)
                    rhs = int(c)
                    _check_classification(lhs, rhs, answer_text, text, challenged)
                    continue

                m = bracket_frac_re.match(text)
                if m:
                    n1, p, c = m.groups()
                    lhs = parse_binomial(n1) / int(p)
                    rhs = int(c)
                    _check_classification(lhs, rhs, answer_text, text, challenged)
                    continue

                m = simple_re.match(text)
                if m:
                    lhs_str, c = m.groups()
                    lhs = parse_binomial(lhs_str)
                    rhs = int(c)
                    _check_classification(lhs, rhs, answer_text, text, challenged)
                    continue

                challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"senior-identify-and-solve-linear-equations oracle: examined {examined}, challenged 0")


def _check_classification(lhs, rhs, answer_text, text, challenged):
    expr = (lhs - rhs).expand()
    try:
        poly = Poly(expr, x)
    except PolynomialError:
        oracle_label, oracle_value = "not_linear", None
    else:
        deg = poly.degree()
        if deg == 1:
            solved = solve(Eq(lhs, rhs), x)
            oracle_label = "linear"
            oracle_value = solved[0] if solved else None
        else:
            oracle_label, oracle_value = "not_linear", None

    if oracle_label == "not_linear":
        if answer_text != "Not linear":
            challenged.append({"issue": f"oracle says not_linear, answer_text was {answer_text!r}", "text": text})
        return

    m = re.match(r"^Linear; x = (-?\d+(?:/\d+)?)$", answer_text)
    if not m:
        challenged.append(
            {"issue": f"oracle says linear (x={oracle_value}), answer_text was {answer_text!r}", "text": text}
        )
        return
    stated = _frac_from_text(m.group(1))
    if Rational(oracle_value) != Rational(stated.numerator, stated.denominator):
        challenged.append(
            {
                "issue": f"oracle computed x={oracle_value}, answer_text stated x={stated}",
                "text": text,
            }
        )
