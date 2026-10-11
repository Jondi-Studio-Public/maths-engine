"""Independent oracle for the `p7b_quadratics` questions: it re-derives each answer from the rendered question
text.

Parses each rendered question and derives the answer from `question.text` alone -- never imports
`mathgen.generators.p7b_quadratics_build`."""

import cmath
import re
from fractions import Fraction

from sympy import expand, simplify, symbols, sympify
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make

x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse_frac(s: str) -> Fraction:
    s = s.strip()
    if "/" in s:
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def _parse_pqi(text: str):
    """Parse 'p + qi' / 'p - m/ni' rendered text into (Fraction p, Fraction q).

    The magnitude is bracket-free by construction (`_complex_text` in the
    generator was fixed to drop the parenthesised '(m/n)i' form -- confirmed
    by direct parse that 'm/ni' already reads left-to-right as (m/n)*i under
    the grader's own transformations, and a bracket in the canonical answer
    would make `grade_expression` reject a learner's bracket-free typed
    answer). Both forms are accepted here regardless, since this oracle must
    keep working whichever form a future edit renders."""
    m = re.match(r"^(-?\d+(?:/\d+)?) ([+-]) (\(\d+/\d+\)|\d+/\d+|\d+)?i$", text.strip())
    if not m:
        raise ValueError(f"cannot parse p+qi shape: {text!r}")
    p_str, sign, mag_str = m.groups()
    p = _parse_frac(p_str)
    mag = Fraction(1) if mag_str is None else _parse_frac(mag_str.strip("()"))
    q = mag if sign == "+" else -mag
    return p, q


class TestExpandBinomialProductMonicOracle:
    """Re-expands the two factors parsed straight out of the rendered
    question text with sympy, independent of the generator's own construction
    route, and compares to the rendered answer_text."""

    PATTERN = re.compile(r"^Expand \(x ([+-]) (\d+)\)\(x ([+-]) (\d+)\) into a trinomial in standard form\.$")

    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "f10-quadratic-algebra-expansion")

        examined = 0
        challenged = []
        for source in skill.sources:
            if source.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                m = self.PATTERN.match(q.text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": q.text})
                    continue
                s1, a_str, s2, b_str = m.groups()
                a = int(a_str) if s1 == "+" else -int(a_str)
                b = int(b_str) if s2 == "+" else -int(b_str)
                expected = expand((x + a) * (x + b))
                got = parse_expr(
                    q.answer_text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": x}
                )
                if simplify(expected - got) != 0:
                    challenged.append(
                        {
                            "issue": f"oracle expanded {expected}, answer_text stated {q.answer_text}",
                            "text": q.text,
                        }
                    )

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"f10-quadratic-algebra-expansion oracle: examined {examined}, challenged 0")


class TestQuadraticComplexConjugateSolutionsOracle:
    """Re-derives each root with cmath/complex arithmetic (never sympy.I,
    never the generator's Fraction/i-Symbol construction route) straight from
    the rendered equation text, and compares to the rendered answer_text."""

    FORWARD = re.compile(
        r"^The equation (\d*)x\^2 ([+-]) (\d+)x ([+-]) (\d+) = 0 has complex roots\. "
        r"State the solution with the positive imaginary part, in the form p \+ qi\.$"
    )
    # Hard tier: one root and b are given, c is withheld,
    # and the question asks for the value of c -- not "the other solution".
    HARD = re.compile(
        r"^(.+?) is one solution of x\^2 ([+-]) (\d+)x \+ c = 0, "
        r"where the coefficients are real\. Find the value of c\.$"
    )
    GIVEN_ROOT = re.compile(r"^(-?\d+) ([+-]) (\d+)?i$")

    def test_oracle(self):
        _, skills = load_skills()
        skill = next(
            s for s in skills if s.id == "senior-determine-complex-conjugate-solutions-of-real-quadratic-equations"
        )

        examined = 0
        challenged = []
        for source in skill.sources:
            if source.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                text, answer_text = q.text, q.answer_text

                m = self.HARD.match(text)
                if m:
                    # Hard tier: one root and b are given,
                    # c is withheld and asked for. Re-derive independently via
                    # the conjugate-root theorem (other root = conjugate of the
                    # given one) and Vieta's relations for a monic quadratic --
                    # b = -(sum of roots), c = product of roots -- using
                    # Python's complex arithmetic, never the generator's own
                    # p*p + q*q formula.
                    given_str, s1, b_str = m.groups()
                    gm = self.GIVEN_ROOT.match(given_str)
                    if not gm:
                        challenged.append({"issue": "unparsed given-root shape", "text": text})
                        continue
                    gp, gsign, gq = gm.groups()
                    gq_val = int(gq) if gq else 1
                    given_complex = complex(int(gp), (1 if gsign == "+" else -1) * gq_val)
                    b = (1 if s1 == "+" else -1) * int(b_str)

                    other_root = given_complex.conjugate()
                    expected_b = -(given_complex + other_root).real
                    if abs(expected_b - b) > 1e-9:
                        challenged.append(
                            {
                                "issue": f"stated b={b} inconsistent with the given root's own "
                                f"real part (expected b={expected_b})",
                                "text": text,
                            }
                        )
                        continue

                    product = given_complex * other_root
                    if abs(product.imag) > 1e-9:
                        challenged.append(
                            {"issue": f"product of the conjugate roots is not real: {product}", "text": text}
                        )
                        continue
                    expected_c = product.real

                    try:
                        c_ans = int(answer_text)
                    except ValueError:
                        challenged.append(
                            {"issue": f"hard-tier answer_text is not a plain integer: {answer_text!r}", "text": text}
                        )
                        continue
                    if abs(expected_c - c_ans) > 1e-9:
                        challenged.append(
                            {"issue": f"oracle c={expected_c} != answer_text {answer_text}", "text": text}
                        )
                    continue

                m = self.FORWARD.match(text)
                if m:
                    coef_s, s1, b_str, s2, c_str = m.groups()
                    a = int(coef_s) if coef_s else 1
                    b = (1 if s1 == "+" else -1) * int(b_str)
                    c = (1 if s2 == "+" else -1) * int(c_str)
                    disc = b * b - 4 * a * c
                    if disc >= 0:
                        challenged.append({"issue": f"discriminant not negative: {disc}", "text": text})
                        continue
                    sqrt_disc = cmath.sqrt(disc)
                    root1 = (-b + sqrt_disc) / (2 * a)
                    root2 = (-b - sqrt_disc) / (2 * a)
                    oracle_root = root1 if root1.imag > 0 else root2
                    p_ans, q_ans = _parse_pqi(answer_text)
                    if abs(float(p_ans) - oracle_root.real) > 1e-6 or abs(float(q_ans) - oracle_root.imag) > 1e-6:
                        challenged.append(
                            {"issue": f"oracle root {oracle_root} != answer_text {answer_text}", "text": text}
                        )
                    continue

                challenged.append({"seed": seed, "issue": "unparsed question shape", "text": text})

        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(
            f"senior-determine-complex-conjugate-solutions-of-real-quadratic-equations "
            f"oracle: examined {examined}, challenged 0"
        )
