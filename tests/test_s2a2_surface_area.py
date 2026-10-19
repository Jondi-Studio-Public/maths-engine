"""Tests `uc2-surface-area-parametric`: the topic `s2a2_surface_area` (easy, medium, hard) and the exam chain
`exd_x_s2a2_surface_area`.

The oracle reads only the question text: it parses r(u, v) and the region out of the stem, differentiates r with SymPy
and integrates the length of r_u x r_v numerically by Gauss-Legendre quadrature, then compares that number with the
stored exact answer.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_surface_area_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-surface-area-parametric"
TOPIC = "s2a2_surface_area"
EXAM_TOPIC = "exd_x_s2a2_surface_area"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
u, v = sympy.symbols("u v")
TIERS = ("easy", "medium", "hard")
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(40)


def read(text: str):
    s = text.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("π", "pi").replace("√", "sqrt")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"u": u, "v": v})


def split_top(s: str):
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def surface_in(text: str):
    match = re.search(r"r\(u, v\) = ⟨(.*?)⟩", text)
    return sympy.Matrix([read(c) for c in split_top(match.group(1))])


def region_in(text: str):
    """(u0, u1, v0, v1 or a function of u) from the stem."""
    tri = re.search(r"u \+ v ≤ (\d+)", text)
    if tri:
        k = int(tri.group(1))
        return 0.0, float(k), 0.0, lambda uu: k - uu
    tri = re.search(r"0 ≤ v ≤ (\d+) - u", text)
    if tri:
        k = int(tri.group(1))
        return 0.0, float(k), 0.0, lambda uu: k - uu
    match = re.search(r"(\S+) ≤ u ≤ (\S+) and (\S+) ≤ v ≤ ([^\s,]+)", text)
    vals = [float(read(g.rstrip("."))) for g in match.groups()]
    return vals[0], vals[1], vals[2], vals[3]


def numeric_area(text):
    r = surface_in(text)
    ru, rv = r.diff(u), r.diff(v)
    cross = ru.cross(rv)
    length = sympy.lambdify((u, v), sympy.sqrt(sum(c * c for c in cross)), "math")
    u0, u1, v0, v1 = region_in(text)
    total = 0.0
    for tu, wu in zip(_NODES, _WEIGHTS):
        uu = (u1 + u0) / 2 + (u1 - u0) / 2 * tu
        hi = v1(uu) if callable(v1) else v1
        lo = v0
        inner = sum(wv * length(uu, (hi + lo) / 2 + (hi - lo) / 2 * tv) for tv, wv in zip(_NODES, _WEIGHTS))
        total += wu * (u1 - u0) / 2 * inner * (hi - lo) / 2
    return total


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_numerical_integral(tier):
    for seed in range(40):
        q = _q(tier, seed)
        exact = float(sympy.N(read(q.answer_text)))
        assert exact == pytest.approx(numeric_area(q.text), rel=1e-8), (seed, q.text, q.answer_text)


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM_TOPIC, "exam", ["exam"])
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert len(q.option_notes) == len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_a_reply_in_another_spelling_is_accepted():
    for seed in range(30):
        q = _q("hard", seed)
        alt = str(sympy.expand(read(q.answer_text))).replace("**", "^")
        assert grade(_SKILL, q, alt).correct, (seed, alt, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 20
    assert max(answers.count(a) for a in set(answers)) <= 10


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s).text for s in range(30)]
    assert all("sin" not in t and "cos" not in t and "π" not in t for t in easy)
    medium = [_q("medium", s).text for s in range(40)]
    assert all("cos(v)" in t for t in medium)
    assert any("u cos(v)" in t for t in medium) and any("⟩" in t and "u cos(v)" not in t for t in medium)
    hard = [_q("hard", s).text for s in range(100)]
    kinds = {
        "sphere": sum("sin(u) cos(v)" in t or "cos(v) sin(u)" in t for t in hard),
        "paraboloid": sum(re.search(r"u² ⟩|u²⟩", t) is not None for t in hard),
        "torus": sum(
            "cos(u)" in t and "+" in t and "cos(v)" in t and "sin(u)" not in t.split("⟨")[1].split(",")[0] for t in hard
        ),
        "triangle": sum("u + v ≤" in t or "- u" in t for t in hard),
    }
    assert all(n >= 8 for n in kinds.values()), kinds


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    r = surface_in(q.stem)
    cross = r.diff(u).cross(r.diff(v))
    length = sympy.lambdify((u, v), sympy.sqrt(sum(c * c for c in cross)), "math")
    c = length(1.7, 0.3) / 1.7
    assert length(2.9, 1.1) / 2.9 == pytest.approx(c, rel=1e-12)  # the length really is a constant times u
    price = int(re.search(r"\$(\d+) per", q.parts[2].prompt).group(1))
    area = numeric_area(q.stem)
    u0, u1, v0, v1 = region_in(q.stem)
    return dict(c=c, area=area, price=price, U=u1, vlen=v1 - v0)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "number"]
        assert [p.marks for p in q.parts] == [2, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert float(sympy.N(read(parts["a"]))) == pytest.approx(f["c"], rel=1e-9), (seed, q.stem)
        assert float(sympy.N(read(parts["b"]))) == pytest.approx(f["area"], rel=1e-8)
        cost = Decimal(repr(float(f["area"] * f["price"]))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert abs(Decimal(parts["c"]) - cost) <= Decimal("0.01"), (seed, parts["c"], cost)
        assert re.fullmatch(r"\d+\.\d\d", parts["c"])


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(30):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_constant_into_the_area():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        u1 = int(round(f["U"]))
        vlen = sympy.nsimplify(f["vlen"] / np.pi) * sympy.pi
        for wrong in q.parts[0].distractors:
            w = read(wrong)
            if w in (0, 1) or float(w) == pytest.approx(f["c"]):
                continue
            wb = w * sympy.Rational(u1 * u1, 2) * vlen
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": sympy.sstr(wb).replace("**", "^")})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 60


def test_follow_through_from_a_wrong_area_into_the_cost():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[1].distractors:
            wb = read(wrong)
            cost = Decimal(repr(float(sympy.N(wb)) * f["price"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if abs(float(wb) - f["area"]) < 1e-9:
                continue
            result = grade(_SKILL, q, {**canon, "b": wrong, "c": str(cost)})
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong, cost)
            checked += 1
    assert checked >= 60


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 120
    assert _exam(7).to_json() == _exam(7).to_json()
