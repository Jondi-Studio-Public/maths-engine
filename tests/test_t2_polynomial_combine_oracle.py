"""Independent oracle for the `t2_polynomial_combine` questions: it re-derives each answer from the rendered
question text.

Parses each rendered question with regexes written from scratch against the
pinned template strings, extracts each labelled polynomial's coefficient-by-
degree dict with its own small term-parser (a two-line difference from the
build test's parser is expected and fine -- independently written, not
imported), recomputes the combined result with plain dict arithmetic (no
sympy needed for the arithmetic itself), and only then converts both the
oracle's own rendering and the generator's `answer_text` to sympy expressions
to compare by VALUE. Never imports
`mathgen.generators.t2_polynomial_combine_build`.
"""

import re

import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import join_terms

x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_ADD_SUB_RE = re.compile(r"^A = (.+?)\. B = (.+?)\. C = (.+?)\. Find A \+ B - C\.$")
_LINCOMB_RE = re.compile(r"^P = (.+?)\. Q = (.+?)\. R = (.+?)\. Find (.+?)\.$")
_TERM_RE = re.compile(r"\s*([+-])\s*(\d+)?(x(?:\^(\d+))?)?")
_SIGNED_LETTER_RE = re.compile(r"([+-])\s*(\d+)?([PQR])")


def _parse_poly(display: str) -> dict:
    """{degree: coefficient} from a rendered polynomial string, built purely
    from the text -- never from the generator's own coefficient arrays."""
    padded = display if display[0] in "+-" else "+" + display
    coeffs: dict[int, int] = {}
    for m in _TERM_RE.finditer(padded):
        sign_s, mag_s, xpart, deg_s = m.groups()
        sign = 1 if sign_s == "+" else -1
        if xpart is None:
            degree, mag = 0, int(mag_s) if mag_s is not None else 0
        elif deg_s is not None:
            degree, mag = int(deg_s), int(mag_s) if mag_s is not None else 1
        else:
            degree, mag = 1, int(mag_s) if mag_s is not None else 1
        coeffs[degree] = coeffs.get(degree, 0) + sign * mag
    return coeffs


def _parse_multipliers(combo_text: str):
    padded = "+" + combo_text
    mult = {}
    for sign_s, mag_s, letter in _SIGNED_LETTER_RE.findall(padded):
        mag = int(mag_s) if mag_s else 1
        mult[letter] = mag * (1 if sign_s == "+" else -1)
    return mult["P"], mult["Q"], -mult["R"]


def _render(coeffs: dict) -> str:
    """Canonical string for a combined coefficient dict, using the shared
    mathgen.core.join_terms helper -- a library helper, not the generator
    under test, so calling it here is not an independence violation."""
    nonzero_degrees = [d for d, c in coeffs.items() if c != 0]
    if not nonzero_degrees:
        return "0"
    max_d = max(nonzero_degrees)
    terms = [(coeffs.get(d, 0), f"x^{d}" if d >= 2 else ("x" if d == 1 else "")) for d in range(max_d, -1, -1)]
    return join_terms(terms)


def _sympify(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": x})


class TestPolynomialAddSubtractOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "add-subtract-polynomials")
        examined = 0
        challenged = []
        for source in skill.sources:
            # This oracle's regex and `_parse_poly` are single-variable by construction (matching
            # the ORIGINAL three tiers' own template) -- the exam source has its own from-scratch,
            # two-variable-aware independent oracle in tests/test_exd_e1_polynomial_exam_sources.py
            # (test_oracle_agrees_with_every_source_every_seed), so skipping it here is a scope
            # boundary, not a coverage gap.
            if source.difficulty == "exam":
                continue
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                m = _ADD_SUB_RE.match(q.text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": q.text})
                    continue
                a, b, c = (_parse_poly(g) for g in m.groups())
                degrees = set(a) | set(b) | set(c)
                combined = {d: a.get(d, 0) + b.get(d, 0) - c.get(d, 0) for d in degrees}
                oracle_expr = _sympify(_render(combined))
                stated_expr = _sympify(q.answer_text)
                if sympy.simplify(oracle_expr - stated_expr) != 0:
                    challenged.append(
                        {
                            "seed": seed,
                            "issue": f"oracle {oracle_expr} != stated {stated_expr}",
                            "text": q.text,
                        }
                    )
        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"add-subtract-polynomials oracle: examined {examined}, challenged 0")


class TestPolynomialLinearCombinationOracle:
    def test_oracle(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "polynomial-linear-combination")
        examined = 0
        challenged = []
        for source in skill.sources:
            # See the matching comment in TestPolynomialAddSubtractOracle
            # above: the exam source is genuinely two-variable and has its
            # own independent oracle in
            # tests/test_exd_e1_polynomial_exam_sources.py.
            if source.difficulty == "exam":
                continue
            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
                examined += 1
                m = _LINCOMB_RE.match(q.text)
                if not m:
                    challenged.append({"seed": seed, "issue": "unparsed question shape", "text": q.text})
                    continue
                p_disp, q_disp, r_disp, combo_text = m.groups()
                p, qq, r = _parse_poly(p_disp), _parse_poly(q_disp), _parse_poly(r_disp)
                m1, m2, m3 = _parse_multipliers(combo_text)
                degrees = set(p) | set(qq) | set(r)
                combined = {d: m1 * p.get(d, 0) + m2 * qq.get(d, 0) - m3 * r.get(d, 0) for d in degrees}
                oracle_expr = _sympify(_render(combined))
                stated_expr = _sympify(q.answer_text)
                if sympy.simplify(oracle_expr - stated_expr) != 0:
                    challenged.append(
                        {
                            "seed": seed,
                            "issue": f"oracle {oracle_expr} != stated {stated_expr}",
                            "text": q.text,
                        }
                    )
        assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"
        print(f"polynomial-linear-combination oracle: examined {examined}, challenged 0")
