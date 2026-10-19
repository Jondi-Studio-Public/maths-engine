"""Tests the polynomial-roots-sum-product skill: keys re-derived from numerical roots, grading, and the exam chain."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import r3i_support as sup

SKILL = "polynomial-roots-sum-product"
TOPIC = "r3i_poly_roots"
EXAM = "r3i_exam_poly_roots"
X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _poly(text):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    return sympy.Poly(parse_expr(text, local_dict={"x": X}, transformations=_TRANSFORMS), X)


def _roots(poly):
    return [complex(r) for r in poly.nroots(n=20, maxsteps=1000)]


def _value(ask, roots):
    names = dict(zip("αβγ", ("ra", "rb", "rc")))
    text = ask.replace("²", "**2")
    for greek, name in names.items():
        text = text.replace(greek, f" {name} ")
    value = parse_expr(
        text, local_dict={n: sympy.Symbol(n) for n in names.values()}, transformations=_TRANSFORMS, evaluate=False
    )
    symbols = {s.name: s for s in value.free_symbols}
    return complex(value.subs({symbols[n]: roots[i] for i, n in enumerate(names.values()) if n in symbols}))


def _same(value, key):
    return abs(value - float(Fraction(key))) < 1e-8 * max(1.0, abs(value))


def _derive_easy_medium(text):
    poly_text = re.search(r"roots of (?:the equation )?(.+?) = 0", text).group(1)
    ask = re.search(r"exact value of (.+)\.$", text).group(1)
    return _value(ask, _roots(_poly(poly_text)))


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        [
            "senior-solve-simple-polynomial-equations",
            "senior-solve-quadratic-equations-using-the-quadratic-formula-and",
        ],
    )


def test_easy_keys_match_numerical_roots():
    for seed in range(150):
        q = sup.tier_question(TOPIC, "easy", seed)
        assert _same(_derive_easy_medium(q.text), q.answer_text), (seed, q.text, q.answer_text)


def test_medium_keys_match_numerical_roots():
    for seed in range(150):
        q = sup.tier_question(TOPIC, "medium", seed)
        if "Find the value of k" in q.text:
            poly_text, target = re.search(
                r"roots of (.+?) = 0 are α, β and γ, and α² \+ β² \+ γ² = (-?\d+)", q.text
            ).groups()
            k = int(q.answer_text)
            roots = _roots(_poly(poly_text.replace("kx", f"({k})*x")))
            assert abs(sum(r * r for r in roots) - int(target)) < 1e-8, (seed, q.text, q.answer_text)
        else:
            assert _same(_derive_easy_medium(q.text), q.answer_text), (seed, q.text, q.answer_text)


def test_hard_keys_match_numerical_roots():
    seen = set()
    for seed in range(200):
        q = sup.tier_question(TOPIC, "hard", seed)
        if q.text.startswith("The cubic"):
            poly_text = re.search(r"The cubic (.+?) = 0", q.text).group(1)
            roots = sorted(r.real for r in _roots(_poly(poly_text)))
            if "arithmetic" in q.text:
                seen.add("ap")
                assert (
                    abs((roots[2] - roots[1]) - int(q.answer_text)) < 1e-7
                    and abs((roots[1] - roots[0]) - int(q.answer_text)) < 1e-7
                )
            else:
                seen.add("gp")
                assert (
                    abs(roots[2] / roots[1] - int(q.answer_text)) < 1e-7
                    and abs(roots[1] / roots[0] - int(q.answer_text)) < 1e-7
                )
            continue
        seen.add("transform")
        poly_text, first, second = re.search(
            r"roots of (.+?) = 0\. Find the quadratic with integer coefficients and a leading coefficient of 1 whose roots are (.+?) and (.+?)\.$",
            q.text,
        ).groups()
        r1, r2 = _roots(_poly(poly_text))
        new = [_value(first, [r1]), _value(second.replace("β", "α"), [r2])]
        expected = _poly(q.answer_text).all_coeffs()
        assert abs(expected[1] + (new[0] + new[1])) < 1e-7 and abs(expected[2] - new[0] * new[1]) < 1e-7, (
            seed,
            q.text,
            q.answer_text,
        )
    assert seen == {"ap", "gp", "transform"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=12, top=0.2)


def test_tiers_differ():
    easy = [sup.tier_question(TOPIC, "easy", s).text for s in range(60)]
    medium = [sup.tier_question(TOPIC, "medium", s).text for s in range(60)]
    hard = [sup.tier_question(TOPIC, "hard", s).text for s in range(60)]
    assert all("α and β" in t or "roots of" in t for t in easy) and not any("γ" in t for t in easy)
    assert all("γ" in t for t in medium)
    assert any("Find the quadratic" in t for t in hard) and any("progression" in t for t in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_hard_transformed_roots_are_built_from_sum_and_product():
    """The roots are irrational, so a learner cannot just find them: the discriminant is not a square."""
    for seed in range(60):
        q = sup.tier_question(TOPIC, "hard", seed)
        if "Find the quadratic" not in q.text:
            continue
        poly = _poly(re.search(r"roots of (.+?) = 0", q.text).group(1)).all_coeffs()
        disc = poly[1] ** 2 - 4 * poly[2]
        assert disc > 0 and int(disc**0.5) ** 2 != disc


def test_exam_shape_and_grading():
    for seed in range(40):
        q = sup.exam_question(EXAM, seed)
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [1, 1, 2, 3]
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
        assert {p.answer_kind for p in q.parts} <= {"number", "fraction"}
        assert all(p.sensible_range is not None for p in q.parts)
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=60, top=0.05)


def test_exam_keys_match_numerical_roots():
    for seed in range(150):
        q = sup.exam_question(EXAM, seed)
        poly = _poly(re.search(r"roots of (.+?) = 0", q.text).group(1))
        r = _roots(poly)
        keys = [
            sum(r),
            r[0] * r[1] + r[1] * r[2] + r[2] * r[0],
            sum(v**2 for v in r),
            sum(v**3 for v in r),
        ]
        assert all(_same(k, p.answer_text) for k, p in zip(keys, q.parts)), (
            seed,
            q.text,
            [p.answer_text for p in q.parts],
        )


def test_exam_follow_through_from_wrong_early_parts():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: Fraction(p.answer_text) for p in q.parts}
        for da, db in ((1, 0), (0, 1), (2, 3), (3, 2), (1, 2)):
            a, b = canon["a"] + da, canon["b"] + db
            c = a * a - 2 * b
            replies = {k: str(v) for k, v in canon.items()} | {"a": str(a), "b": str(b), "c": str(c)}
            result = grade(sk, q, replies)
            if result.parts["c"].marks == q.parts[2].marks:
                break
        else:
            raise AssertionError(f"no usable wrong reply for seed {seed}")
