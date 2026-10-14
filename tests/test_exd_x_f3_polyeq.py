"""Exam tier of senior-solve-simple-polynomial-equations.

The oracle parses the equation out of the rendered prompt, collects it with sympy and applies the
quadratic formula with `cmath`; it never imports the generator.
"""

from __future__ import annotations

import cmath
import re

from sympy import I, Poly, Symbol, sympify

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_polyeq"
_GRADER = GRADERS["expression"]
_Z = Symbol("z")


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


_EQ = re.compile(r"Solve (.+?) = (.+?) for z\.")


def _term(text: str):
    t = text.replace("²", "**2").replace("z", "*z")
    t = re.sub(r"(^|[+\-(])\s*\*z", r"\g<1>1*z", t)  # bare z -> 1*z
    return sympify(t, locals={"z": _Z})


def _roots(text: str):
    lhs, rhs = _EQ.search(text).groups()
    poly = Poly(_term(lhs) - _term(rhs), _Z)
    a, b, c = (int(poly.coeff_monomial(m)) for m in (_Z**2, _Z, 1))
    s = cmath.sqrt(b * b - 4 * a * c)
    return (-b + s) / (2 * a), (-b - s) / (2 * a)


def _value(answer_text: str) -> complex:
    return complex(sympify(answer_text.replace("i", "*I").replace("-*I", "-I").replace("+ *I", "+ I")).evalf())


def test_key_matches_independent_oracle_over_400_seeds():
    bad = []
    for s in range(400):
        q = _q(s)
        pos = max(_roots(q.text), key=lambda r: r.imag)
        assert pos.imag > 0
        got = _value(q.answer_text)
        if abs(got - pos) > 1e-9:
            bad.append((s, q.text, pos, q.answer_text))
    assert bad == [], bad[:3]


def test_roots_are_fractional_somewhere_and_both_forms_occur():
    qs = [_q(s) for s in range(60)]
    assert any("/" in q.answer_text for q in qs)
    assert any(" = 0 for z" in q.text for q in qs) and any(" = 0 for z" not in q.text for q in qs)


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for s in range(150):
        q = _q(s)
        assert len(q.distractors) == 3
        assert _GRADER(q, q.answer_text).correct, s
        bad += [(s, d) for d in q.distractors if _GRADER(q, d).correct]
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(4)
    z = _value(q.answer_text)
    for wrong in (f"{z.real + 1} + {z.imag}i", f"{z.real} + {z.imag + 1}i", f"{z.real} - {z.imag}i", ""):
        assert not _GRADER(q, wrong).correct, wrong
