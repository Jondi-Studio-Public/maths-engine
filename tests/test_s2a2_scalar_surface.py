"""Tests `uc2-scalar-surface-integral`: the topic `s2a2_scalar_surface` (easy, medium, hard) and the exam chain
`exd_x_s2a2_scalar_surface`.

The oracle reads only the question text: it parses r(u, v), the region and g out of the stem, differentiates r with SymPy
and integrates g(r(u, v)) |r_u x r_v| numerically by Gauss-Legendre quadrature, then compares that number with the
stored exact answer.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_scalar_surface_integral_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-scalar-surface-integral"
TOPIC = "s2a2_scalar_surface"
EXAM_TOPIC = "exd_x_s2a2_scalar_surface"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
u, v, x, y, z = sympy.symbols("u v x y z")
TIERS = ("easy", "medium", "hard")
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(40)


def read(text: str):
    s = text.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("π", "pi").replace("√", "sqrt")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"u": u, "v": v, "x": x, "y": y, "z": z})


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
    tri = re.search(r"u \+ v ≤ (\d+)", text) or re.search(r"0 ≤ v ≤ (\d+) - u", text)
    if tri:
        k = int(tri.group(1))
        return 0.0, float(k), 0.0, lambda uu: k - uu
    match = re.search(r"(\S+) ≤ u ≤ (\S+) and (\S+) ≤ v ≤ ([^\s,]+)", text)
    vals = [float(read(g.rstrip("."))) for g in match.groups()]
    return vals[0], vals[1], vals[2], vals[3]


def g_in(text: str):
    match = re.search(r"g\(x, y, z\) = (.*?) (?:over|\(in kilograms)", text)
    return read(match.group(1))


def integral(text, g=None):
    """The surface integral of g over the surface in `text` (the area when g is None), by quadrature."""
    r = surface_in(text)
    cross = r.diff(u).cross(r.diff(v))
    length = sympy.lambdify((u, v), sympy.sqrt(sum(c * c for c in cross)), "math")
    gf = sympy.lambdify((u, v), 1 if g is None else g.subs(dict(zip((x, y, z), r))), "math")
    u0, u1, v0, v1 = region_in(text)
    total = 0.0
    for tu, wu in zip(_NODES, _WEIGHTS):
        uu = (u1 + u0) / 2 + (u1 - u0) / 2 * tu
        hi = v1(uu) if callable(v1) else v1
        inner = sum(
            wv * length(uu, vv) * gf(uu, vv)
            for tv, wv in zip(_NODES, _WEIGHTS)
            for vv in [(hi + v0) / 2 + (hi - v0) / 2 * tv]
        )
        total += wu * (u1 - u0) / 2 * inner * (hi - v0) / 2
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
        assert exact == pytest.approx(integral(q.text, g_in(q.text)), rel=1e-8, abs=1e-8), (seed, q.text, q.answer_text)
        assert exact != 0


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
            assert text != "0"


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
    assert all("sin" not in t and "cos" not in t for t in easy)
    medium = [_q("medium", s).text for s in range(40)]
    assert all("cos(v)" in t for t in medium)
    assert any("mass" in t for t in medium)
    hard = [_q("hard", s).text for s in range(100)]
    kinds = {
        "sphere": sum("sin(u) cos(v)" in t or "cos(v) sin(u)" in t for t in hard),
        "triangle": sum("u + v ≤" in t or "- u" in t for t in hard),
        "cone": sum("u cos(v)" in t for t in hard),
    }
    assert all(n >= 15 for n in kinds.values()), kinds


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


def test_a_sheet_density_is_never_negative_over_the_surface():
    checked = 0
    for tier in TIERS:
        for seed in range(60):
            q = _q(tier, seed)
            if "in kilograms" not in q.text:
                continue
            r = surface_in(q.text)
            fn = sympy.lambdify((u, v), g_in(q.text).subs(dict(zip((x, y, z), r))), "math")
            u0, u1, v0, v1 = region_in(q.text)
            for i in range(9):
                for j in range(9):
                    assert fn(u0 + (u1 - u0) * i / 8, v0 + (v1 - v0) * j / 8) >= -1e-9, (seed, q.text)
            checked += 1
    assert checked >= 15


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    r = surface_in(q.stem)
    a = r[2].subs(u, 0)
    u0, u1, v0, v1 = region_in(q.stem)
    alpha = sympy.nsimplify(u1 / np.pi) * sympy.pi
    k = int(re.search(r"(\d+)z kilograms", q.stem).group(1))
    mass = integral(q.stem, k * z)
    area = integral(q.stem)
    cross = r.diff(u).cross(r.diff(v))
    length = sympy.lambdify((u, v), sympy.sqrt(sum(c * c for c in cross)), "math")
    c = length(0.7, 0.4) / np.sin(0.7)
    assert length(1.1, 2.0) / np.sin(1.1) == pytest.approx(c, rel=1e-12)
    return dict(a=a, alpha=alpha, k=k, mass=mass, area=area, c=c, mean=mass / area)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert [p.marks for p in q.parts] == [2, 3, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert float(sympy.N(read(parts["a"]))) == pytest.approx(f["c"], rel=1e-9), (seed, q.stem)
        assert float(sympy.N(read(parts["b"]))) == pytest.approx(f["mass"], rel=1e-8)
        assert float(sympy.N(read(parts["c"]))) == pytest.approx(f["area"], rel=1e-8)
        assert float(sympy.N(read(parts["d"]))) == pytest.approx(f["mean"], rel=1e-8)


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(30):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert text != "0"
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_constant_into_the_mass_and_the_area():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        a, alpha, k = f["a"], f["alpha"], f["k"]
        for wrong in q.parts[0].distractors:
            w = read(wrong)
            if w in (0, 1) or float(w) == pytest.approx(f["c"]):
                continue
            mass = w * a * k * sympy.pi * sympy.sin(alpha) ** 2
            area = w * 2 * sympy.pi * (1 - sympy.cos(alpha))
            reply = {
                **canon,
                "a": wrong,
                "b": sympy.sstr(mass).replace("**", "^"),
                "c": sympy.sstr(area).replace("**", "^"),
            }
            result = grade(_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, wrong, "b")
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong, "c")
            checked += 1
    assert checked >= 60


def test_follow_through_from_a_wrong_mass_into_the_average():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        area = read(canon["c"])
        for wrong in q.parts[1].distractors:
            wm = read(wrong)
            mean = sympy.simplify(wm / area)
            if float(mean) == pytest.approx(f["mean"]) or mean == 0:
                continue
            reply = {**canon, "b": wrong, "d": sympy.sstr(mean).replace("**", "^")}
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, wrong, mean)
            checked += 1
    assert checked >= 60


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 60
    assert _exam(7).to_json() == _exam(7).to_json()
