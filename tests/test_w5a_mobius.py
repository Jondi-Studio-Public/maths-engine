"""``uc2-mobius-transformations``: answers grade right, distractors wrong, each answer is recomputed from the stem."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    Z,
    answer_value,
    balanced_after,
    check_follow_through,
    check_grading,
    check_text,
    close,
    coverage,
    numeric,
    parse,
    questions,
    tier_topic,
)

NAME = "mobius"
ALL = ("easy", "medium", "hard", "exam")
PROBES = (
    sympy.Rational(3, 7) + sympy.I / 5,
    -sympy.Rational(11, 4) + sympy.Rational(5, 3) * sympy.I,
    sympy.Rational(5, 2) - sympy.Rational(2, 9) * sympy.I,
)


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    assert len({q.text for q in qs}) >= 12
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))


# -- the oracle ----------------------------------------------------------------------------------------------------------


def _transformation(text: str):
    """The T(z) written out in the stem (the one that is not the equation T(z) = z or T(z) = w)."""
    for m in re.finditer(r"T\(z\) = ", text):
        expr = balanced_after(text[m.start() :], "T(z) = ")
        if expr in ("z",) or not expr:
            continue
        value = parse(expr)
        if value.has(Z):
            return value
    return None


def _fixed_points(t):
    num, den = sympy.fraction(sympy.together(t - Z))
    return [sympy.simplify(r) for r in sympy.roots(sympy.Poly(sympy.expand(num), Z))]


def _pick(roots, text: str):
    if len(roots) == 1 or "only one" in text or "repeated" in text or "exactly one" in text:
        return roots[0]
    if "positive imaginary" in text:
        return max(roots, key=lambda r: float(sympy.im(sympy.N(r))))
    return max(roots, key=lambda r: float(sympy.re(sympy.N(r))))


def _agree(f, g) -> bool:
    return all(abs(complex(sympy.N((f - g).subs(Z, p), 20))) < 1e-9 for p in PROBES)


def _pairs(text: str):
    return [
        (parse(a), parse(b))
        for a, b in re.findall(r"T\(([^()]+(?:\([^()]*\))?)\) = ([^,.]+?)(?=,| and |\.|\?|$)", text)
        if a != "z"
    ]


def _through(pairs):
    """The Möbius map a z + b / (c z + d) through three (z, w) pairs, solved as a linear system."""
    a, b, c, d = sympy.symbols("a b c d")
    eqs = []
    for zz, ww in pairs:
        if zz == sympy.oo:
            eqs.append(a - ww * c if ww != sympy.oo else c)
        elif ww == sympy.oo:
            eqs.append(c * zz + d)
        else:
            eqs.append(a * zz + b - ww * (c * zz + d))
    sol = sympy.solve(eqs + [d - 1], [a, b, c, d], dict=True)
    if not sol:
        sol = sympy.solve(eqs + [c - 1], [a, b, c, d], dict=True)
    s = sol[0]
    return (s[a] * Z + s[b]) / (s[c] * Z + s[d])


def _expected(q):
    text = q.text
    t = _transformation(text)
    if "point at infinity is mapped to" in text:
        return sympy.limit(t, Z, sympy.oo)
    if t is not None and ("fixed point" in text or re.search(r"T\(z\) = z(?![/\w(])", text)):
        return _pick(_fixed_points(t), text)
    if t is not None and re.search(r"Evaluate T\(", text):
        p = parse(re.search(r"Evaluate T\((.+?)\) for", text).group(1))
        return t.subs(Z, p)
    if t is not None and re.search(r"image of the point z = (.+?) under T", text):
        return t.subs(Z, parse(re.search(r"image of the point z = (.+?) under T", text).group(1)))
    if t is not None and re.search(r"maps z = (.+?) to a point", text):
        return t.subs(Z, parse(re.search(r"maps z = (.+?) to a point", text).group(1)))
    if t is not None and re.search(r"maps to w = |maps to w|T maps to w = (.+?)\.", text):
        w = parse(re.search(r"T maps to w = (.+?)\.", text).group(1))
        return sympy.solve(sympy.Eq(t, w), Z)[0]
    if t is not None and re.search(r"sends a point z to w = (.+?)\.", text):
        w = parse(re.search(r"sends a point z to w = (.+?)\.", text).group(1))
        return sympy.solve(sympy.Eq(t, w), Z)[0]
    if t is not None and re.search(r"Solve T\(z\) = (.+?) for", text):
        w = parse(re.search(r"Solve T\(z\) = (.+?) for", text).group(1))
        return sympy.solve(sympy.Eq(t, w), Z)[0]
    if t is not None and ("T(T(z))" in text or "(T∘T)(z)" in text):
        return sympy.simplify(t.subs(Z, t))
    return None


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_is_recomputed_from_the_stem(tier):
    checked = 0
    qs = questions(NAME, tier, ORACLE_SEEDS)
    for q in qs:
        want = _expected(q)
        if want is not None:
            if want.has(Z):  # a function: compare at probe points
                assert _agree(parse(q.answer_text), want), (q.text, q.answer_text, want)
            else:
                assert close(answer_value(q), numeric(want), 1e-8), (q.text, q.answer_text, want)
            checked += 1
            continue
        pairs = _pairs(q.text)
        if len(pairs) == 3 and re.search(r"single fraction", q.text):
            assert _agree(parse(q.answer_text), _through(pairs)), (q.text, q.answer_text)
            checked += 1
        elif len(pairs) == 3:
            fourth = re.search(r"(?:Find|determine) T\((.+?)\)\.|image of z = (.+?)\?", q.text)
            point = parse(fourth.group(1) or fourth.group(2))
            want = _through(pairs).subs(Z, point)
            assert close(answer_value(q), numeric(want), 1e-8), (q.text, q.answer_text, want)
            checked += 1
    coverage(checked, len(qs), 0.9)


def test_exam_parts_are_recomputed_from_the_stem():
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        t = parse(balanced_after(q.text, "T(z) = "))
        first = parse(re.search(r"Find T\((.+?)\)\.", q.parts[0].prompt).group(1))
        a = t.subs(Z, first)
        b = t.subs(Z, a)
        total = sum(_fixed_points(t))
        c = t.subs(Z, total)
        for part, want in zip(q.parts, (a, b, total, c)):
            assert close(numeric(parse(part.answer_text)), numeric(want), 1e-8), (
                q.text,
                part.prompt,
                part.answer_text,
                want,
            )
