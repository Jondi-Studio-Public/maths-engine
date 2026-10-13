"""Tests `ucv1-triple-integral`: the topic `triple_integral` (easy, medium, hard) and the exam chain
`exd_x_uc_triple_integral`.

The oracle reads only the question text. It parses the integrand, the box, the plane, the inequalities, the density
and the coordinate asked for, and re-derives every key with SymPy's `integrate` over explicit limits it writes for
itself (never the generator's polynomial algebra, its Dirichlet formula or its Gauss-Legendre check).
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.ucv1_triple_integral_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "ucv1-triple-integral"
TOPIC = "triple_integral"
EXAM_TOPIC = "exd_x_uc_triple_integral"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, z = sympy.symbols("x y z")
SYMBOLS = {"x": x, "y": y, "z": z}
TIERS = ("easy", "medium", "hard")


def _read(text: str):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=dict(SYMBOLS))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _tetra_limits(plane: str):
    """(a, b, c) from 'px + qy + rz = d'."""
    lhs, d = plane.split(" = ")
    expr = _read(lhs)
    p, q, r = (expr.coeff(v) for v in (x, y, z))
    return sympy.Rational(int(d)) / p, sympy.Rational(int(d)) / q, sympy.Rational(int(d)) / r


def _over_tetra(f, a, b, c):
    return sympy.integrate(
        sympy.integrate(sympy.integrate(f, (z, 0, c * (1 - x / a - y / b))), (y, 0, b * (1 - x / a))), (x, 0, a)
    )


def _oracle(tier, q):
    text = q.text
    if tier == "easy":
        f = _read(re.search(r"∭ (.+?) dV over the box", text).group(1))
        lims = [(int(lo), int(hi)) for lo, hi in re.findall(r"(-?\d+) ≤ [xyz] ≤ (-?\d+)", text)]
        return sympy.integrate(f, (x, *lims[0]), (y, *lims[1]), (z, *lims[2]))
    if "plane" in text:
        plane = re.search(r"plane (.+? = \d+)", text).group(1)
        a, b, c = _tetra_limits(plane)
        integrand = re.search(r"∭ (.+?) dV", text)
        f = _read(integrand.group(1)) if integrand else sympy.Integer(1)
        return _over_tetra(f, a, b, c)
    chain = re.search(r"0 ≤ ([xyz]) ≤ ([xyz]) ≤ ([xyz]) ≤ (\d+)", text)
    if chain:
        u, v, w, c = chain.groups()
        f = _read(re.search(r"∭ (.+?) dV", text).group(1))
        su, sv, sw = SYMBOLS[u], SYMBOLS[v], SYMBOLS[w]
        return sympy.integrate(sympy.integrate(sympy.integrate(f, (su, 0, sv)), (sv, 0, sw)), (sw, 0, int(c)))
    if "iterated integral" in text:
        c = int(re.search(r"x = 0 to (\d+)", text).group(1))
        f = _read(re.search(r"z = 0 to y (.+?) dz dy dx", text).group(1))
        return sympy.integrate(sympy.integrate(sympy.integrate(f, (z, 0, y)), (y, 0, x)), (x, 0, c))
    region = re.search(r"0 ≤ x ≤ (\d+), 0 ≤ y ≤ (\d+), 0 ≤ z ≤ ([^.]+?)\.$", text)
    a, b, top = int(region.group(1)), int(region.group(2)), _read(region.group(3))
    integrand = re.search(r"∭ (.+?) dV", text)
    f = _read(integrand.group(1)) if integrand else sympy.Integer(1)
    return sympy.integrate(sympy.integrate(sympy.integrate(f, (z, 0, top)), (y, 0, b)), (x, 0, a))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages, last.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_integration(tier):
    for seed in range(50):
        q = _q(tier, seed)
        expected = _oracle(tier, q)
        assert sympy.Rational(q.answer_text) == expected, (seed, q.text, q.answer_text, expected)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert len(q.distractor_values) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


def test_a_decimal_or_unsimplified_fraction_is_accepted_as_the_same_value():
    q = _q("medium", 3)
    assert grade(_SKILL, q, q.answer_text).correct
    value = sympy.Rational(q.answer_text)
    assert grade(_SKILL, q, f"{value.p * 2}/{value.q * 2}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 8


def test_each_tier_has_the_shapes_the_design_asks_for():
    easy = [_q("easy", s).text for s in range(40)]
    assert all("over the box" in t for t in easy)
    medium = [_q("medium", s).text for s in range(60)]
    assert sum("plane" in t for t in medium) >= 15 and sum("0 ≤ z ≤" in t for t in medium) >= 15
    hard = [_q("hard", s).text for s in range(90)]
    assert sum("plane" in t for t in hard) >= 15
    assert sum(re.search(r"0 ≤ [xyz] ≤ [xyz] ≤ [xyz] ≤", t) is not None for t in hard) >= 15
    assert sum("order of integration changed" in t for t in hard) >= 10


def test_hard_integrands_are_not_separable():
    for seed in range(60):
        text = _q("hard", seed).text
        match = re.search(r"∭ (.+?) dV", text) or re.search(r"z = 0 to y (.+?) dz dy dx", text)
        f = _read(match.group(1))
        mixed = any(
            sympy.degree(term, x) * sympy.degree(term, y)
            + sympy.degree(term, y) * sympy.degree(term, z)
            + sympy.degree(term, x) * sympy.degree(term, z)
            > 0
            for term in sympy.Add.make_args(sympy.expand(f))
        )
        sum_of_variables = len(f.free_symbols) >= 2
        assert mixed or sum_of_variables, (seed, text)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q) -> dict:
    stem = q.stem
    density = re.search(r"ρ = (\d+) \+ (\d*)([xyz])", stem)
    k, p, w = int(density.group(1)), int(density.group(2) or 1), SYMBOLS[density.group(3)]
    u = SYMBOLS[re.search(r"Find the ([xyz])-coordinate", q.parts[2].prompt).group(1)]
    if "plane" in stem:
        a, b, c = _tetra_limits(re.search(r"plane (.+? = \d+)", stem).group(1))

        def integrate(f):
            return _over_tetra(f, a, b, c)

        extent = {x: a, y: b, z: c}
    else:
        a, b = (int(v) for v in re.search(r"0 ≤ x ≤ (\d+), 0 ≤ y ≤ (\d+), 0 ≤ z ≤ x \+ y", stem).groups())

        def integrate(f):
            return sympy.integrate(sympy.integrate(sympy.integrate(f, (z, 0, x + y)), (y, 0, b)), (x, 0, a))

        extent = {x: a, y: b, z: a + b}
    rho = k + p * w
    V, Iw, M, Mu = integrate(1), integrate(w), integrate(rho), integrate(u * rho)
    return dict(V=V, Iw=Iw, M=M, Mu=Mu, ubar=Mu / M, avg=M / V, k=k, p=p, extent=extent[u])


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 3, 1]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(35):
        q = _exam(seed)
        f = _facts(q)
        keys = [sympy.Rational(p.answer_text) for p in q.parts]
        assert keys == [f["V"], f["M"], f["ubar"], f["avg"]], (seed, q.text, keys, f)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(40):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 9
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_volume_into_the_mass():
    checked = 0
    for seed in range(35):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[0].distractors:
            wv = sympy.Rational(wrong)
            wm = f["k"] * wv + f["p"] * f["Iw"]
            if wm == f["M"] or wm.q > 400:
                continue
            reply = {**canon, "a": wrong, "b": str(wm)}
            result = grade(_SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 40


def test_follow_through_from_a_wrong_mass_into_the_centre_of_mass_and_average_density():
    checked = 0
    for seed in range(50):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[1].distractors:
            wm = sympy.Rational(wrong)
            wu = f["Mu"] / wm
            wd = wm / f["V"]
            if wu == f["ubar"] or wu.q > 400 or wd.q > 400 or not (0 < wu < f["extent"]):
                continue
            if wd in (f["V"], wm) or wu in (f["V"], wm):
                continue  # the engine gives no follow-through for a reply that merely repeats an earlier one
            if str(wu) in q.parts[2].distractors or str(wd) in q.parts[3].distractors:
                continue  # a follow-through value that happens to be a listed wrong answer is marked as that error
            reply = {**canon, "b": wrong, "c": str(wu), "d": str(wd)}
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (seed, wrong, result)
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 25


def test_exam_covers_both_solids_and_all_coordinates():
    seen = set()
    for seed in range(120):
        q = _exam(seed)
        seen.add(("plane" in q.stem, re.search(r"Find the ([xyz])-coordinate", q.parts[2].prompt).group(1)))
    assert {s[0] for s in seen} == {True, False} and {s[1] for s in seen} == {"x", "y", "z"}


def test_exam_answer_space():
    answers = [_exam(seed).answer_text for seed in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_exam_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
