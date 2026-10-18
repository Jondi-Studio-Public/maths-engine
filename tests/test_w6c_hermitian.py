"""Tests `uc2-hermitian-unitary-matrices` (topic `w6c_hermitian`) and its exam chain `exd_x_w6c_hermitian`.

The oracle reads the complex matrix back out of the question TEXT (entries such as `2 - 3i`) and re-derives each key
with SymPy: the conjugate transpose `Matrix.H`, `eigenvals()`, `det()`, and the condition U*U = I.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

import mathgen.generators.w6c_hermitian_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests import w6c_support as support

TOPIC = "w6c_hermitian"
EXAM = "exd_x_w6c_hermitian"
SKILL = support.skill_named("uc2-hermitian-unitary-matrices")
_BRACKETS = re.compile(r"\[\[.*?\]\]")
K = sympy.symbols("k", positive=True)
X = sympy.symbols("x", real=True)


def _expr(text):
    """A matrix literal such as `[[2, 3 - i], [3 + i, 5]]` as a SymPy Matrix (i is the imaginary unit)."""
    text = re.sub(r"(\d+)i", r"\1*I", text)
    text = re.sub(r"(?<![\w*])i", "I", text)
    text = text.replace("ki", "k*I").replace("k*I", "k*I")
    return sympy.Matrix(sympy.sympify(text, locals={"k": K, "x": X}))


def _matrix(text):
    return _expr(_BRACKETS.search(text).group(0))


def _value(text):
    f = Fraction(text)
    return sympy.Rational(f.numerator, f.denominator)


def test_skill_wiring():
    support.check_wiring(SKILL, TOPIC, EXAM, ["two-by-two-eigenvalues"])


def test_each_tier_key_accepted_and_every_distractor_rejected():
    for tier in support.TIERS:
        support.check_tier(SKILL, TOPIC, tier)


def test_easy_keys_come_from_the_conjugate_transpose():
    kinds = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="easy")
        a = _matrix(q.text)
        g = (a.H * a).expand()
        key = _value(q.answer_text)
        if q.text.startswith(("Find tr", "A* denotes")):
            assert key == sympy.re(g.trace()) == sum(abs(v) ** 2 for v in a), (seed, q.text)
            kinds.add("trace")
        else:
            i, j = (int(n) for n in re.search(r"\((\d), (\d)\) entry", q.text).groups())
            entry = g[i - 1, j - 1]
            part = sympy.re(entry) if "real part" in q.text else sympy.im(entry)
            assert key == part, (seed, q.text)
            kinds.add("re" if "real part" in q.text else "im")
    assert kinds == {"trace", "re", "im"}


def test_medium_keys_are_hermitian_eigenvalues():
    kinds = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="medium")
        h = _matrix(q.text)
        assert h == h.H
        low, high = sorted(h.eigenvals())
        key = _value(q.answer_text)
        if "difference" in q.text or "apart" in q.text:
            assert key == high - low
            kinds.add("gap")
        elif "larger" in q.text or "largest" in q.text:
            assert key == high
            kinds.add("hi")
        else:
            assert key == low
            kinds.add("lo")
    assert kinds == {"hi", "lo", "gap"}


def test_hard_keys():
    kinds = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        key = _value(q.answer_text)
        if "unitary" in q.text and "Find N" in q.text or "positive real N" in q.text:
            m = _matrix(q.text)
            u = m / key
            assert (u.H * u).expand() == sympy.eye(2)
            assert sympy.expand((m.H * m)[0, 0]) == key**2
            kinds.add("N")
        elif "unitary" in q.text:
            n = int(re.search(r"\(1/(\d+)\)", q.text).group(1))
            m = _matrix(q.text)
            good = (m.subs(X, key) / n).applyfunc(sympy.expand)
            assert (good.H * good).expand() == sympy.eye(2), (seed, q.text)
            bad = (m.subs(X, -key) / n).applyfunc(sympy.expand)
            assert (bad.H * bad).expand() != sympy.eye(2)
            kinds.add("x")
        else:
            lam = int(re.search(r"(-?\d+) (?:is one of its eigenvalues|is an eigenvalue)", q.text).group(1)) if "is one of" in q.text else int(re.search(r"Given that (-?\d+) is an eigenvalue", q.text).group(1))
            h = _matrix(q.text)
            assert sympy.solve(sympy.expand((h - lam * sympy.eye(2)).det()), K) == [key]
            kinds.add("k")
    assert kinds == {"N", "x", "k"}


def test_tier_shapes_and_answer_space():
    support.check_tier_shapes_differ(TOPIC)
    for tier in ("easy", "medium", "hard"):
        support.check_answer_space(TOPIC, tier, distinct=6, share=0.4)


def test_exam_shape():
    q = make(EXAM, seed=1, difficulty="exam")
    assert [p.answer_kind for p in q.parts] == ["number"] * 4
    assert [p.marks for p in q.parts] == [2, 3, 1, 2]
    assert q.parts[2].follow_through.endswith("- part_b")
    assert q.parts[3].follow_through == "part_b**2 + part_c**2"


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        h = _matrix(q.stem)
        assert h == h.H
        low, high = sorted(h.eigenvals())
        assert _value(q.parts[0].answer_text) == sympy.im((h.H * h).expand()[0, 1])
        assert _value(q.parts[1].answer_text) == low
        assert _value(q.parts[2].answer_text) == high
        assert _value(q.parts[3].answer_text) == sympy.re((h.H * h).expand().trace())


def test_exam_grading_and_follow_through():
    support.check_exam_grading(SKILL, EXAM, [2, 3, 1, 2])
    support.check_follow_through(SKILL, EXAM)


def test_exam_answer_space():
    answers = {make(EXAM, seed=s, difficulty="exam").answer_text for s in range(200)}
    assert len(answers) >= 60
