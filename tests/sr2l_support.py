"""Shared helpers for the `test_sr2l_*` files: independent readings of a rendered stem, and the checks every skill runs.

Nothing here imports a generator module. Each test re-derives the answer from the RENDERED stem with SymPy (expanding
the printed relation into a polynomial in x and y, or solving the printed inequality), then compares it with what the
question keys and what the grader accepts.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import _tuplify, grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

x, y = sympy.symbols("x y", real=True)
_T = standard_transformations + (implicit_multiplication_application,)
_LOCAL = {"x": x, "y": y, "I": sympy.I}


@cache
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


@cache
def sample(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def questions(topic: str, tier: str, n: int):
    return [sample(topic, tier, s) for s in range(n)]


def source_topics(skill_id: str) -> dict[str, str]:
    return {s.difficulty: s.topic for s in skill(skill_id).sources}


def check_wired(skill_id: str, topics: dict[str, str]) -> None:
    assert source_topics(skill_id) == topics


# -- reading maths text -------------------------------------------------------------------------------------------


def expr(text: str, **local):
    """A learner-style expression (``2x - 3``, ``(-1/2)x + 19/4``, ``3 - 2i``) as an exact SymPy value."""
    s = text.replace("−", "-").replace("²", "**2").replace("^", "**")
    s = re.sub(r"(?<![A-Za-z])i(?=z)", "I*", s)
    s = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", s)
    s = re.sub(r"(\d)i\b", r"\1*I", s)
    return parse_expr(s, local_dict={**_LOCAL, **local}, transformations=_T)


def rational(text: str):
    return sympy.Rational(str(expr(text)))


def pair(text: str) -> tuple:
    """``(3, -5/2)`` as a tuple of exact rationals."""
    a, b = text.strip()[1:-1].split(",")
    return rational(a), rational(b)


def number_list(text: str) -> list:
    return sorted(rational(p) for p in text.split(","))


def line_form(text: str) -> tuple:
    """A line ``y = mx + c`` or ``x = k`` as the normalised coefficients (a, b, c) of ax + by + c = 0."""
    left, right = (expr(p) for p in text.split("="))
    return normal_line(sympy.expand(left - right))


def normal_line(poly) -> tuple:
    a, b, c = (
        sympy.Rational(poly.coeff(x, 1)),
        sympy.Rational(poly.coeff(y, 1)),
        sympy.Rational(poly.subs({x: 0, y: 0})),
    )
    lead = a if a != 0 else b
    return a / lead, b / lead, c / lead


def mod_squared(u):
    """|u|^2 for u an expression in the real symbols x and y."""
    return sympy.expand(sympy.re(u) ** 2 + sympy.im(u) ** 2)


def _bars(text: str) -> str:
    """Rewrite ``|expr|`` as ``M(expr)``; the squared forms ``|expr|^2`` come through as ``M(expr)**2``."""
    return re.sub(r"\|([^|]+)\|", r"M(\1)", text)


def relation_polynomial(text: str):
    """The polynomial P(x, y) with P = 0 equivalent to the printed relation ``|u| = k|v|`` / ``|u|^2 = k|v|^2`` / ``P = 0``.

    A modulus relation is squared, so a circle or a line comes out as a polynomial of degree 2 or 1.
    """
    s = text.replace("−", "-")
    left, right = s.split("=")
    zz = x + sympy.I * y
    if "|" not in s:
        left, right = s.replace("z̄", "W").replace("z", "V").split("=")
        sub = {"V": zz, "W": x - sympy.I * y}
        return sympy.expand(expr(left, **sub) - expr(right, **sub))

    squared = "²" in s

    def side(part: str):
        t = _bars(part.replace("²", "**2")).replace("z̄", "W").replace("z", "(zz)")
        e = parse_expr(
            re.sub(r"(\d)i\b", r"\1*I", re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", t)),
            local_dict={
                **_LOCAL,
                "zz": zz,
                "W": x - sympy.I * y,
                "M": lambda u: sympy.sqrt(mod_squared(sympy.expand(u))),
            },
            transformations=_T,
        )
        return sympy.expand(e if squared else e**2)

    return sympy.expand(side(left) - side(right))


def circle_of(poly) -> tuple:
    """(centre, radius squared) of a circle polynomial a(x^2 + y^2) + dx + ey + f = 0."""
    a = sympy.Rational(poly.coeff(x, 2))
    assert a != 0 and sympy.Rational(poly.coeff(y, 2)) == a, poly
    d, e, f = (
        sympy.Rational(poly.coeff(x, 1).subs(y, 0)),
        sympy.Rational(poly.coeff(y, 1).subs(x, 0)),
        poly.subs({x: 0, y: 0}),
    )
    cx, cy = -d / (2 * a), -e / (2 * a)
    return (sympy.nsimplify(cx), sympy.nsimplify(cy)), sympy.nsimplify(cx**2 + cy**2 - f / a)


# -- the generic checks -------------------------------------------------------------------------------------------


def check_graded_single(skill_id: str, topic: str, tier: str, n: int = 25) -> None:
    """The key is accepted, and every listed distractor is marked wrong."""
    sk = skill(skill_id)
    for q in questions(topic, tier, n):
        assert grade(sk, q, q.answer_text).correct, (q.text, q.answer_text)
        assert 2 <= len(q.distractors) <= 3, (q.text, q.distractors)
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (q.text, d)
            assert q.option_notes.get(d) in CATALOGUE, (d, q.option_notes.get(d))


def check_graded_exam(skill_id: str, topic: str, n: int = 25) -> None:
    """The composed key earns full marks; each distractor, in its own part, earns that part nothing."""
    sk = skill(skill_id)
    for q in questions(topic, "exam", n):
        full = grade(sk, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks, (q.text, q.answer_text)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert 2 <= len(p.distractors) <= 3, (q.text, p.label, p.distractors)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def check_structure(topic: str, kinds: list[str], marks: list[int], follow: list[bool], n: int = 40) -> None:
    for q in questions(topic, "exam", n):
        assert [p.label for p in q.parts] == [chr(97 + i) for i in range(len(kinds))]
        assert [p.answer_kind for p in q.parts] == kinds
        assert [p.marks for p in q.parts] == marks
        assert [p.follow_through is not None for p in q.parts] == follow


def check_clean(topic: str, tiers: list[str], n: int = 40) -> None:
    for tier in tiers:
        for seed in range(n):
            q = sample(topic, tier, seed)
            assert find_text_defects(q) == [], (tier, seed, q.text, find_text_defects(q))
            assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()


def check_tiers_differ(skill_id: str, topics: dict[str, str], n: int = 60) -> None:
    """Each practice tier asks something in a stem shape the tier below it never uses."""
    shapes = {}
    for tier in ("easy", "medium", "hard"):
        src = next(s for s in skill(skill_id).sources if s.difficulty == tier)
        shapes[tier] = {
            prompt_shape(make(src.topic, seed=s, difficulty=tier, **_tuplify(src.params))) for s in range(n)
        }
    assert shapes["medium"] - shapes["easy"], "medium adds no stem shape beyond easy"
    assert shapes["hard"] - shapes["medium"], "hard adds no stem shape beyond medium"


def check_answer_space(topic: str, tier: str, n: int = 200, minimum: int = 12, ceiling: float = 0.2) -> None:
    keys = Counter(sample(topic, tier, s).answer_text for s in range(n))
    assert len(keys) >= minimum, (tier, len(keys))
    assert max(keys.values()) / n <= ceiling, keys.most_common(3)


_END = re.compile(r"\. |, which| lie on| form a| is a circle|\.$")


def relation_in(text: str) -> str:
    """The printed relation in a stem: from the first ``|`` (or ``x²``) up to the end of the sentence or the clause."""
    start = min(m.start() for m in (re.search(r"\|", text), re.search(r"\d*x²", text), re.search(r"z z̄", text)) if m)
    while start > 0 and text[start - 1].isdigit():  # a coefficient written straight before the bar: 3|z - 1|
        start -= 1
    eq = text.index("=", start)
    end = _END.search(text, eq)
    return text[start : end.start() if end else len(text)].strip()


def stem_of(q) -> str:
    """The shared stem of a multi-part question, or the whole text of a single-part one."""
    return q.stem or q.text


def figure_circles(fig) -> list[tuple]:
    """(cx, cy, r) of each drawn circle, read from the extremes of the sampled curve alone."""
    out = []
    for curve in fig["curves"]:
        pts = [p for piece in curve["pieces"] for p in piece]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy, r = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(xs) - min(xs)) / 2
        assert abs(r - (max(ys) - min(ys)) / 2) < 0.01
        out.append(tuple(sympy.Rational(round(v)) for v in (cx, cy, r)))
        assert all(abs(v - round(v)) < 0.01 for v in (cx, cy, r)), (cx, cy, r)
    return out


def complex_parts(text: str) -> tuple:
    v = sympy.expand(expr(text))
    return sympy.re(v), sympy.im(v)


def bisector_line(a: tuple, b: tuple) -> tuple:
    """The perpendicular bisector of the points a and b as normalised (A, B, C)."""
    return normal_line(sympy.expand((x - a[0]) ** 2 + (y - a[1]) ** 2 - (x - b[0]) ** 2 - (y - b[1]) ** 2))


def number_close(text: str, target, tol: float) -> bool:
    return abs(float(rational(text)) - float(target)) <= tol
