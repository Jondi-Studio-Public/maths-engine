"""Tests `uc2-gram-schmidt` (topic `s2b3_gram_schmidt`) and its exam chain `exd_x_s2b3_gram_schmidt`.

The oracle reads the vectors (or polynomials, or weights) back out of the question TEXT and re-derives the key with
SymPy, so it shares no code with the generator's own exact-fraction route.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2b3_gram_schmidt_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.s2b3_support import (
    answer_vector,
    check_exam_grading,
    check_single_tier,
    check_wiring,
    int_vectors,
    skill,
    stem,
)

TOPIC = "s2b3_gram_schmidt"
EXAM = "exd_x_s2b3_gram_schmidt"
SK = skill("uc2-gram-schmidt")
TIERS = ("easy", "medium", "hard")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
X = sympy.symbols("x")


def _num(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.Integer(value)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _orthogonalise(vectors, inner):
    """SymPy Gram-Schmidt for an arbitrary inner product, not normalised."""
    out = []
    for v in vectors:
        u = v
        for w in out:
            u = u - inner(v, w) / inner(w, w) * w
        out.append(u)
    return out


def _dot_oracle(text):
    vs = [sympy.Matrix(v) for v in int_vectors(text)]
    return list(_orthogonalise(vs, lambda a, b: a.dot(b))[-1])


def _poly(text):
    expr = text.replace("²", "**2").replace("−", "-")
    return sympy.Poly(parse_expr(expr, transformations=_TRANSFORMS, local_dict={"x": X}), X)


def _poly_oracle(text):
    lo = int(re.search(r"from (-?\d+) to 1", text).group(1))
    polys = [
        sympy.expand(_poly(m).as_expr()) for m in re.findall(r"p\d\(x\) = ([^,.]*?)(?= and p\d| are |,|\. |$)", text)
    ]
    if len(polys) < 2:
        polys = [
            sympy.expand(_poly(m).as_expr())
            for m in re.findall(r"p\d\(x\) = (.*?)(?:, p\d|, and | and p\d|\. |\s+are )", text + ". ")
        ]
    inner = lambda a, b: sympy.integrate(sympy.expand(a * b), (X, lo, 1))  # noqa: E731
    last = _orthogonalise(polys, inner)[-1]
    coeffs = sympy.Poly(sympy.expand(last), X).all_coeffs()[::-1]
    return coeffs + [0] * (3 - len(coeffs))


def _weights(text):
    form = re.search(r"= ((?:\d*x1y1)(?: \+ \d*x2y2)(?: \+ \d*x3y3))", text).group(1)
    return [int(m or 1) for m in re.findall(r"(\d*)x\dy\d", form)]


# -- wiring ---------------------------------------------------------------------------------------------------


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["ucv1-gram-schmidt-2d", "vector-dot-product"])


# -- the three tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_the_second_orthogonal_vector():
    dims = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        v1, v2 = [sympy.Matrix(v) for v in int_vectors(q.text)]
        u2 = v2 - (v2.dot(v1) / v1.dot(v1)) * v1
        assert answer_vector(q) == list(u2), (seed, q.text)
        assert u2.dot(v1) == 0
        dims[len(v1)] += 1
    assert set(dims) == {3, 4}


def test_medium_key_is_the_third_orthogonal_vector():
    for seed in range(120):
        q = _q("medium", seed)
        assert answer_vector(q) == _dot_oracle(q.text), (seed, q.text)
        vs = [sympy.Matrix(v) for v in int_vectors(q.text)]
        assert len(vs) == 3
        # a genuinely different answer from "project onto v1 and v2": the vectors are not already orthogonal
        assert vs[1].dot(vs[0]) != 0 or vs[2].dot(vs[0]) != 0


def test_hard_covers_polynomials_and_weights():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        if "∫" in q.text:
            assert answer_vector(q) == _poly_oracle(q.text), (seed, q.text)
            seen["poly"] += 1
        else:
            w = _weights(q.text)
            vs = [sympy.Matrix(v) for v in int_vectors(q.text)]
            inner = lambda a, b: sum(wi * ai * bi for wi, ai, bi in zip(w, a, b))  # noqa: E731
            assert answer_vector(q) == list(_orthogonalise(vs, inner)[-1]), (seed, q.text)
            seen["weighted"] += 1
    assert seen["poly"] >= 50 and seen["weighted"] >= 30, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=80)


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(40)]
    medium = [_q("medium", s).text for s in range(40)]
    hard = [_q("hard", s).text for s in range(40)]
    assert all(t.count("v1 =") == 1 and "v3" not in t for t in easy)
    assert all("v3 =" in t for t in medium)
    assert all(("∫" in t or "inner product" in t or "⟨x, y⟩" in t) for t in hard)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, s).answer_text for s in range(80)]
        assert len(set(answers)) >= 60, (tier, Counter(answers).most_common(3))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 3]
        assert [p.answer_kind for p in q.parts] == ["fraction", "matrix", "fraction", "matrix"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]


def test_exam_keys_match_sympy():
    for seed in range(120):
        q = _exam(seed)
        v1, v2, v3 = [sympy.Matrix(v) for v in int_vectors(stem(q))]
        us = _orthogonalise([v1, v2, v3], lambda a, b: a.dot(b))
        coeff = v2.dot(v1) / v1.dot(v1)
        assert _num(q.parts[0].answer) == coeff
        assert answer_vector(q.parts[1]) == list(us[1])
        assert _num(q.parts[2].answer) == us[1].dot(us[1])
        assert answer_vector(q.parts[3]) == list(us[2])


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=40)


def test_exam_follow_through_from_a_wrong_coefficient():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        v1, v2, _v3 = [sympy.Matrix(v) for v in int_vectors(stem(q))]
        canon = {p.label: p.answer_text for p in q.parts}
        wrong = sympy.Rational(v2.dot(v1), v1.dot(v1)) / 2
        carried = v2.dot(v2) - wrong**2 * v1.dot(v1)
        if wrong == 0 or carried <= 0 or carried == q.parts[2].answer:
            continue
        result = grade(SK, q, {**canon, "a": str(wrong), "c": str(carried)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 20


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 190
