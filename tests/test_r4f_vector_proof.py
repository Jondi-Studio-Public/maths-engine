"""Tests the skill `vector-proof-geometry`.

The oracle never uses the generator's pair arithmetic. It reads the figure back out of the rendered stem and puts it on
coordinates: a = (3, 1) and b = (-1, 2) for the vector answers (a typed answer in a and b is checked by evaluating it
at those two vectors), and a = (r, 0), b = (3s/5, 4s/5) for the lengths |a| = r, |b| = s of a dot product. The scalar
in the diagonals question is found by solving the two coefficient equations with SymPy.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests.r4f_support import (
    canonical,
    check_answer_space_single,
    check_exam_clean,
    check_exam_oracle,
    check_exam_shape,
    check_single_clean,
    check_tiers_differ,
    check_wired,
    exam_questions,
    parse,
    skill,
)

SKILL_ID = "vector-proof-geometry"
TOPIC = "r4f_vector_proof"
EXAM = "exd_x_r4f_vector_proof"
A_SYM, B_SYM = sympy.symbols("a b")
A_VEC, B_VEC = (Fraction(3), Fraction(1)), (Fraction(-1), Fraction(2))


def comb(p, q):
    return (p * A_VEC[0] + q * B_VEC[0], p * A_VEC[1] + q * B_VEC[1])


def sub(u, v):
    return (u[0] - v[0], u[1] - v[1])


def add(u, v):
    return (u[0] + v[0], u[1] + v[1])


def half(u):
    return (u[0] / 2, u[1] / 2)


def scale(c, u):
    return (c * u[0], c * u[1])


def basis(v):
    """(p, q) with v = p a + q b, by solving the 2 x 2 system (determinant 7)."""
    det = A_VEC[0] * B_VEC[1] - A_VEC[1] * B_VEC[0]
    p = (v[0] * B_VEC[1] - v[1] * B_VEC[0]) / det
    q = (A_VEC[0] * v[1] - A_VEC[1] * v[0]) / det
    assert comb(p, q) == v
    return p, q


def as_expr(v):
    p, q = basis(v)
    return sympy.Rational(p.numerator, p.denominator) * A_SYM + sympy.Rational(q.numerator, q.denominator) * B_SYM


def k_of(text: str) -> Fraction:
    """The k in 'vector CB = ka' (3a/2, a/2, 2a), or 1 for a parallelogram."""
    m = re.search(r"vector \w\w = ([^.,]+?)\.", text.split("with vector")[-1]) if False else None
    m = re.search(r"vector [A-Z][A-Z] = ([0-9]*a(?:/[0-9]+)?|[0-9]*a/[0-9]+)\.", text)
    if "parallelogram" in text.split(".")[0]:
        return Fraction(1)
    expr = parse(m.group(1), a=A_SYM)
    return Fraction(int(expr.coeff(A_SYM).p), int(expr.coeff(A_SYM).q))


def trapezium_names(text: str):
    m = re.match(r"([A-Z])([A-Z])([A-Z])([A-Z]) is a (?:trapezium|parallelogram)", text)
    return m.groups()


def lam_of(k: Fraction) -> Fraction:
    lam, mu = sympy.symbols("lam mu")
    sol = sympy.solve([lam * sympy.Rational(k.numerator, k.denominator) - (1 - mu), lam - mu], [lam, mu])
    return Fraction(int(sol[lam].p), int(sol[lam].q))


def expected_vector(text: str):
    """The expected vector answer (as an expression in a, b) of an easy or hard vector question."""
    if text.startswith("O is the centre of a circle"):
        target = re.search(r"Express the vector (\w\w)", text).group(1)
        pts = {"A": A_VEC, "B": scale(-1, A_VEC), "P": B_VEC}
        return as_expr(sub(pts[target[1]], pts[target[0]]))
    if "is a parallelogram with vector" in text and "Express the vector" in text:
        v0, v1, v2, v3 = trapezium_names(text)
        pts = {v0: (Fraction(0), Fraction(0)), v1: A_VEC, v3: B_VEC}
        pts[v2] = add(A_VEC, B_VEC)
        if m := re.search(r"The point (\w) is the midpoint of (\w)(\w)\.", text):
            pts[m.group(1)] = half(add(pts[m.group(2)], pts[m.group(3)]))
        elif m := re.search(r"The point (\w) is on (\w)(\w) with \w\w : \w\w = (\d+) : (\d+)\.", text):
            t = Fraction(int(m.group(4)), int(m.group(4)) + int(m.group(5)))
            pts[m.group(1)] = add(pts[m.group(2)], scale(t, sub(pts[m.group(3)], pts[m.group(2)])))
        s, e = re.search(r"Express the vector (\w)(\w) in terms", text).groups()
        return as_expr(sub(pts[e], pts[s]))
    if "The point M is the midpoint of" in text:  # midline of a trapezium
        o, a_, b_, c_ = trapezium_names(text)
        k = k_of(text)
        pts = {o: (Fraction(0), Fraction(0)), a_: A_VEC, c_: B_VEC}
        pts[b_] = add(B_VEC, scale(k, A_VEC))
        m_pt, n_pt = half(pts[c_]), half(add(pts[a_], pts[b_]))
        return as_expr(sub(n_pt, m_pt))
    m = re.search(r"In triangle (\w)(\w)(\w), vector \w\w = a and vector \w\w = b\. (.*) Express the vector XY", text)
    assert m, text
    ratio = re.search(r"(\d+) : (\d+)", m.group(4))
    t = Fraction(int(ratio.group(1)), int(ratio.group(1)) + int(ratio.group(2))) if ratio else Fraction(1, 2)
    return as_expr(sub(scale(t, B_VEC), scale(t, A_VEC)))


def dot_expected(text: str) -> Fraction:
    r, s = (int(x) for x in re.search(r"\|a\| = (\d+)[^|]*\|b\| = (\d+)", text).groups())
    a = (Fraction(r), Fraction(0))
    b = (Fraction(3 * s, 5), Fraction(4 * s, 5))
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1]  # noqa: E731
    neg_a = scale(-1, a)
    if text.startswith("O is the centre"):
        return dot(sub(b, a), sub(b, neg_a))
    if "diagonals" in text:
        return dot(add(a, b), sub(b, a))
    return dot(half(add(a, b)), sub(b, a))


def test_wired():
    check_wired(SKILL_ID, TOPIC, EXAM)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_single_tiers_grade_cleanly(tier):
    check_single_clean(SKILL_ID, TOPIC, tier, n=60)


def test_tiers_differ_in_shape():
    check_tiers_differ(TOPIC)


@pytest.mark.parametrize(("tier", "floor"), [("easy", 20), ("medium", 8), ("hard", 25)])
def test_answer_space(tier, floor):
    check_answer_space_single(TOPIC, tier, n=300, floor=floor)


def test_easy_key_rederived_on_coordinates():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="easy")
        got = parse(q.answer_text, a=A_SYM, b=B_SYM)
        assert sympy.simplify(got - expected_vector(q.text)) == 0, (seed, q.text, q.answer_text)


def test_easy_uses_midpoints_and_the_semicircle():
    texts = [make(TOPIC, seed=s, difficulty="easy").text for s in range(80)]
    assert any("midpoint" in t for t in texts) and any("centre of a circle" in t for t in texts)
    assert any(" : " in t for t in texts)


def test_medium_key_is_lambda_from_the_coefficient_equations():
    ks = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="medium")
        k = k_of(q.text)
        ks.add(k)
        assert Fraction(q.answer_text) == lam_of(k) == 1 / (k + 1), q.text
    assert Fraction(1) in ks and len(ks) >= 6


def test_hard_keys():
    kinds = set()
    for seed in range(240):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "dot product" in q.text:
            kinds.add("dot")
            assert Fraction(q.answer_text) == dot_expected(q.text), q.text
        else:
            kinds.add("vector")
            got = parse(q.answer_text, a=A_SYM, b=B_SYM)
            assert sympy.simplify(got - expected_vector(q.text)) == 0, (q.text, q.answer_text)
    assert kinds == {"dot", "vector"}


def test_dot_questions_cover_three_theorems_and_unequal_lengths():
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "dot product" in q.text:
            r, s = (int(x) for x in re.search(r"\|a\| = (\d+)[^|]*\|b\| = (\d+)", q.text).groups())
            assert r != s
            seen.add("semicircle" if q.text.startswith("O is") else "rhombus" if "diagonals" in q.text else "chord")
    assert seen == {"semicircle", "rhombus", "chord"}


def exam_oracle(q):
    text, prompts = q.stem, {p.label: p.prompt for p in q.parts}
    if "trapezium" in text:
        o, a_, b_, c_ = trapezium_names(text)
        k = k_of(text)
        s, e = re.search(r"Find the vector (\w)(\w) in terms", prompts["a"]).groups()
        pts = {o: (Fraction(0), Fraction(0)), a_: A_VEC, c_: B_VEC, b_: add(B_VEC, scale(k, A_VEC))}
        lam = lam_of(k)
        return {
            "a": as_expr(sub(pts[e], pts[s])),
            "b": as_expr(pts[b_]),
            "c": lam,
            "d": k * lam,
        }
    if text.startswith("O is the centre"):
        r, s = (int(x) for x in re.search(r"Given \|a\| = (\d+) and \|b\| = (\d+)", prompts["c"]).groups())
        full = f"O is the centre of a circle and AB is a diameter, with vector OA = a and vector OB = -a, where |a| = {r}. The point P has vector OP = b with |b| = {s}. Find the dot product of the vectors AP and BP."
        dot = dot_expected(full)
        return {
            "a": as_expr(sub(B_VEC, A_VEC)),
            "b": as_expr(add(B_VEC, A_VEC)),
            "c": dot,
            "d": -dot,
        }
    ratio = re.search(r"(\d+) : (\d+)", text)
    m, n = int(ratio.group(1)), int(ratio.group(2))
    t = Fraction(m, m + n)
    s0, e0 = re.search(r"Find the vector (\w)(\w) in terms", prompts["a"]).groups()
    names = re.search(r"In triangle (\w)(\w)(\w)", text).groups()
    pts = {names[0]: (Fraction(0), Fraction(0)), names[1]: A_VEC, names[2]: B_VEC}
    forward = e0 == names[2]
    xy = sub(scale(t, B_VEC), scale(t, A_VEC))
    length = int(re.search(r"is (\d+) cm", prompts["d"]).group(1))
    return {
        "a": as_expr(sub(pts[e0], pts[s0])),
        "b": as_expr(xy if forward else scale(-1, xy)),
        "c": t,
        "d": t * length,
    }


def test_exam_clean_and_graded():
    check_exam_clean(SKILL_ID, EXAM, n=60)


def test_exam_key_rederived_on_coordinates():
    check_exam_oracle(EXAM, exam_oracle, n=240)


def test_exam_has_all_three_shapes():
    kinds = set()
    for q in exam_questions(EXAM, 60):
        kinds.add("trapezium" if "trapezium" in q.stem else "semicircle" if q.stem.startswith("O is") else "division")
    assert kinds == {"trapezium", "semicircle", "division"}


def test_exam_answer_space():
    check_exam_shape(EXAM, floor_distinct=40, n=300, cap=0.12)


def test_exam_parts_kinds():
    for q in exam_questions(EXAM, 40):
        kinds = [p.answer_kind for p in q.parts]
        assert kinds[:2] == ["expression", "expression"]
        assert kinds[2] in ("fraction", "number") and kinds[3] in ("fraction", "number")
        assert q.parts[3].follow_through is not None


def test_follow_through_from_a_wrong_lambda():
    """A wrong λ carried correctly into the coefficient of a is marked as follow-through."""
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 160):
        if "trapezium" not in q.stem:
            continue
        k = k_of(q.stem)
        wrong = Fraction(q.parts[2].distractors[0])
        if not 0 < wrong < 1:
            continue
        got = grade(sk, q, canonical(q) | {"c": str(wrong), "d": str(k * wrong)})
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
        done += 1
    assert done >= 15


def test_follow_through_from_a_wrong_dot_product():
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 160):
        if not q.stem.startswith("O is"):
            continue
        wrong = int(q.parts[2].distractors[0])
        if wrong == 0:
            continue
        got = grade(sk, q, canonical(q) | {"c": str(wrong), "d": str(-wrong)})
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
        done += 1
    assert done >= 5


def test_follow_through_from_a_wrong_ratio():
    sk = skill(SKILL_ID)
    done = 0
    for q in exam_questions(EXAM, 160):
        if not q.stem.startswith("In triangle"):
            continue
        length = int(re.search(r"is (\d+) cm", q.parts[3].prompt).group(1))
        wrong = Fraction(q.parts[2].distractors[0])
        if not 0 < wrong < 1 or (wrong * length).denominator != 1:
            continue
        got = grade(sk, q, canonical(q) | {"c": str(wrong), "d": str(wrong * length)})
        assert got.parts["c"].marks == 0
        assert got.parts["d"].marks == got.parts["d"].max_marks and got.parts["d"].follow_through, q.text
        done += 1
    assert done >= 5
