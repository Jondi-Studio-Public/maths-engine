"""Tests `complex-exponential-form-conversions`: every key is recomputed from the complex number printed in the stem.

The oracle reads the stem's own spelling (``3e^(-i2π/3)``, ``2(cos(π/3) + i sin(π/3))``, ``-√3 - i``) with SymPy's
parser, evaluates the operation the question names (a part, a product, a quotient, a power, an Euler combination) in
exact complex arithmetic, and compares it with the stored key.
"""

from __future__ import annotations

import math
import re

import sympy
from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_single_answer_rules,
    questions,
    skill,
    sources,
)
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

SK = skill("complex-exponential-form-conversions")
SEEDS = range(60)
_T = standard_transformations + (implicit_multiplication_application,)
_SUPER = {"²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8}


def cx(text: str):
    """The complex number a stem's formula stands for."""
    t = text.strip().rstrip(".")
    t = t.replace("√", "sqrt").replace("π", " pi ").replace("e^(", "exp(")
    t = re.sub(r"sqrt\s*(\d+)", r"sqrt(\1)", t)
    t = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", " I ", t)
    return sympy.sympify(parse_expr(t, transformations=_T, local_dict={"I": sympy.I, "pi": sympy.pi}))


def value_of(expr):
    return complex(sympy.N(sympy.expand_complex(expr), 30))


def key_of(q_or_part) -> float:
    answer = q_or_part.answer
    return float(sympy.N(sympy.sympify(answer.replace("^", "**") if isinstance(answer, str) else answer), 30))


def check_key(q):
    text = q.text
    if m := re.match(r"Let z = (.*?)\. Find the (real|imaginary) part of z, (?:Re|Im)\(z\), as an exact value\.", text):
        z = value_of(cx(m.group(1)))
        expected = z.real if m.group(2) == "real" else z.imag
    elif m := re.match(
        r"Let z1 = (.*?) and z2 = (.*?)\. Find the (real|imaginary) part of the (product|quotient)", text
    ):
        z1, z2 = cx(m.group(1)), cx(m.group(2))
        w = value_of(z1 * z2 if m.group(4) == "product" else z1 / z2)
        expected = w.real if m.group(3) == "real" else w.imag
    elif m := re.match(r"Let z = (.*?)\. Find the (real|imaginary) part of z([²³⁴⁵⁶⁷⁸]), as an exact value\.", text):
        w = value_of(cx(m.group(1)) ** _SUPER[m.group(3)])
        expected = w.real if m.group(2) == "real" else w.imag
    elif m := re.match(r"Evaluate (.*?), giving (?:a real number|an exact real value)\.", text):
        w = value_of(cx(m.group(1)))
        assert abs(w.imag) < 1e-12, text
        expected = w.real
    else:
        raise AssertionError(text)
    assert math.isclose(key_of(q), expected, rel_tol=1e-9, abs_tol=1e-9), (text, q.answer_text, expected)


def run_tier(tier, seeds=range(200)):
    for q in questions(SK, tier, seeds):
        check_key(q)


def test_easy():
    assert_single_answer_rules(SK, "easy", SEEDS)
    run_tier("easy")


def test_medium():
    assert_single_answer_rules(SK, "medium", SEEDS)
    run_tier("medium")


def test_hard():
    assert_single_answer_rules(SK, "hard", SEEDS)
    run_tier("hard")


def test_tiers_use_different_operations():
    text = {t: " ".join(q.text for q in questions(SK, t, range(200))) for t in ("easy", "medium", "hard")}
    assert "cos(" in text["easy"] and "e^(" in text["easy"] and "Evaluate" in text["easy"]
    assert "product" not in text["easy"] and "quotient" not in text["easy"]
    assert "product" in text["medium"] and "quotient" in text["medium"]
    assert re.search(r"z[⁴⁵⁶⁷⁸]", text["hard"]) and "/i" in text["hard"] and "√" in text["hard"]
    assert "z1" not in text["hard"]


def test_every_answer_is_a_real_exact_value_typed_as_an_expression():
    for tier in ("easy", "medium", "hard"):
        for q in questions(SK, tier, range(100)):
            assert q.answer_kind == "expression"
            assert "i" not in q.answer_text.replace("sin", ""), q.answer_text
            key_of(q)


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcde", marks=[1, 1, 2, 2, 2])
    topic = sources(SK)["exam"]
    for seed in range(120):
        q = make(topic, seed=seed, difficulty="exam")
        z = cx(re.match(r"Let z = (.*?)\.", q.text).group(1))
        n = _SUPER[re.search(r"\(a \+ bi\)([²³])", q.parts[2].prompt).group(1)]
        zz, zn = value_of(z), value_of(z**n)
        want = [zz.real, zz.imag, zn.real, zn.imag, abs(zn)]
        got = [key_of(p) for p in q.parts]
        for g, w in zip(got, want, strict=True):
            assert math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-9), (q.text, got, want)
        # the follow-through formulas reproduce the keys from the learner's own (a) and (b)
        a, b = got[0], got[1]
        expand = {2: (a * a - b * b, 2 * a * b), 3: (a**3 - 3 * a * b * b, 3 * a * a * b - b**3)}[n]
        assert math.isclose(got[2], expand[0], abs_tol=1e-9) and math.isclose(got[3], expand[1], abs_tol=1e-9)
        assert math.isclose(got[4], math.hypot(got[2], got[3]), rel_tol=1e-9)


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_the_new_skill_does_not_ask_what_the_old_one_asks():
    """ucv1-complex-exponential-form answers θ/π; no tier here asks for an argument."""
    for tier in ("easy", "medium", "hard"):
        for q in questions(SK, tier, range(100)):
            assert "argument" not in q.text and "θ/π" not in q.text


def test_deterministic_varied_and_clean():
    for tier, distinct in (("easy", 25), ("medium", 25), ("hard", 25)):
        assert_deterministic_and_varied("sc_complex_euler", tier, distinct=distinct)
        assert all("**" not in q.text for q in questions(SK, tier, range(30)))
    assert_deterministic_and_varied("sc_complex_euler_exam", "exam", distinct=25)
