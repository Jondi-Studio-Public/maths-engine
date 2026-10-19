"""Tests `uc2-conservative-field-potential`: the topic `s2a2_conservative3d` (easy, medium, hard) and the exam chain
`exd_x_s2a2_conservative3d`.

The oracle reads only the question text. It parses the field, the points or the curve back out of the stem and
re-derives every key with SymPy: `solve` on the curl for the constants, a step-by-step integration for the potential,
and `integrate` of F . r' along the path for the work (never the generator's potential difference).
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_conservative3d_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-conservative-field-potential"
TOPIC = "s2a2_conservative3d"
EXAM_TOPIC = "exd_x_s2a2_conservative3d"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, z, k, m, t = sympy.symbols("x y z k m t")
TIERS = ("easy", "medium", "hard")


def read(text: str):
    s = text.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("π", "pi")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x, "y": y, "z": z, "k": k, "m": m, "t": t})


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


def vector_after(text: str, marker: str):
    start = text.index(marker) + len(marker)
    end = text.index("⟩", start)
    return tuple(read(c) for c in split_top(text[start:end]))


def field_in(text: str):
    return vector_after(text, "(x, y, z) = ⟨")


def curl(F):
    P, Q, R = F
    return (
        sympy.diff(R, y) - sympy.diff(Q, z),
        sympy.diff(P, z) - sympy.diff(R, x),
        sympy.diff(Q, x) - sympy.diff(P, y),
    )


def solve_constants(F):
    unknowns = [s for s in (k, m) if any(c.has(s) for c in F)]
    eqs = [coeff for c in curl(F) for coeff in sympy.Poly(sympy.expand(c), x, y, z).coeffs() if c != 0]
    sol = sympy.solve(eqs, unknowns, dict=True)
    assert len(sol) == 1, (F, sol)
    return sol[0]


def potential(F):
    P, Q, R = F
    phi = sympy.integrate(P, x)
    phi += sympy.integrate(sympy.expand(Q - sympy.diff(phi, y)), y)
    phi += sympy.integrate(sympy.expand(R - sympy.diff(phi, z)), z)
    return sympy.expand(phi)


def work_along(F, r, lo, hi):
    integrand = sum(c.subs({x: r[0], y: r[1], z: r[2]}, simultaneous=True) * sympy.diff(ri, t) for c, ri in zip(F, r))
    return sympy.simplify(sympy.integrate(sympy.expand(integrand), (t, lo, hi)))


def points_in(text: str):
    return [tuple(int(c) for c in m_) for m_ in re.findall(r"\((-?\d+), (-?\d+), (-?\d+)\)", text)]


def curve_in(text: str):
    r = vector_after(text, "r(t) = ⟨")
    lo, hi = re.search(r"(0) ≤ t ≤ (1|π/2|π)", text).groups()
    return r, 0, read(hi)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    text = q.text
    F = field_in(text)
    if "r(t) = " in text:
        r, lo, hi = curve_in(text)
        assert all(c == 0 for c in curl(F)), "a curve question must have a conservative field"
        return work_along(F, r, lo, hi)
    if "scalar potential" in text:
        return potential(F)
    sol = solve_constants(F)
    if "work" not in text:
        return sol[k]
    G = tuple(c.subs(sol) for c in F)
    assert all(sympy.expand(c) == 0 for c in curl(G))
    A, B = points_in(text.split("moving")[1])
    straight = tuple(a_ + (b_ - a_) * t for a_, b_ in zip(A, B))
    return work_along(G, straight, 0, 1)


def _same(answer_text: str, expected) -> bool:
    return sympy.simplify(read(answer_text) - expected) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(25):
        q = _q(tier, seed)
        assert _same(q.answer_text, oracle(q)), (seed, q.text, q.answer_text, oracle(q))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM_TOPIC, "exam", ["exam"])
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(25):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(25):
        q = _q(tier, seed)
        assert len(q.option_notes) == len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 25
    assert max(answers.count(a) for a in set(answers)) <= 8


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s) for s in range(30)]
    assert all(" k" in q.text and "work" not in q.text and "potential" not in q.text for q in easy)
    medium = [_q("medium", s) for s in range(30)]
    assert all("scalar potential" in q.text and " k" not in q.text for q in medium)
    hard = [_q("hard", s).text for s in range(60)]
    curves = [t_ for t_ in hard if "r(t) = " in t_]
    constants = [t_ for t_ in hard if "constants k and m" in t_]
    assert len(curves) >= 15 and len(constants) >= 15 and len(curves) + len(constants) == 60


def test_hard_curve_fields_are_conservative_but_not_labelled_so():
    for seed in range(60):
        q = _q("hard", seed)
        if "r(t) = " in q.text:
            assert "conservative" not in q.text
            assert all(sympy.expand(c) == 0 for c in curl(field_in(q.text)))


def test_easy_constant_is_not_visible_in_the_field():
    """k is not equal to any coefficient printed in the field (it takes the curl condition to find it)."""
    for seed in range(40):
        q = _q("easy", seed)
        F = field_in(q.text)
        printed = {abs(int(c)) for comp in F for c in sympy.Poly(sympy.expand(comp.subs(k, 0)), x, y, z).coeffs()}
        assert int(q.answer_text) not in printed or abs(int(q.answer_text)) in {1}, (seed, q.text)


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    F = field_in(q.stem)
    sol = solve_constants(F)
    G = tuple(c.subs(sol) for c in F)
    phi = potential(G)
    A, B = points_in(q.parts[2].prompt)
    r, lo, hi = curve_in(q.parts[3].prompt)
    straight = tuple(a_ + (b_ - a_) * t for a_, b_ in zip(A, B))
    return dict(
        F=F,
        k=sol[k],
        G=G,
        phi=phi,
        A=A,
        B=B,
        r=r,
        lo=lo,
        hi=hi,
        work_ab=work_along(G, straight, 0, 1),
        work_c=work_along(G, r, lo, hi),
        mono=sympy.expand(F[0]).coeff(k),
    )


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(30):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "expression", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(30):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert sympy.Rational(parts["a"]) == f["k"], (seed, q.stem)
        assert sympy.expand(read(parts["b"]) - f["phi"]) == 0
        assert sympy.Rational(parts["c"]) == f["work_ab"]
        assert sympy.Rational(parts["d"]) == f["work_c"]


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(20):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_k_into_the_potential():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        piece = sympy.integrate(f["mono"], x)
        for wrong in q.parts[0].distractors:
            wk = sympy.Rational(wrong)
            if wk in (0, f["k"]):
                continue
            wphi = sympy.expand(f["phi"] + (wk - f["k"]) * piece)
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(wphi).replace("**", "^")})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_potential_into_both_work_values():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        end_c, start_c = tuple(c.subs(t, f["hi"]) for c in f["r"]), tuple(c.subs(t, 0) for c in f["r"])
        for wrong in q.parts[1].distractors:
            w = read(wrong)
            at = lambda p: w.subs(dict(zip((x, y, z), p)))  # noqa: E731
            wab, wc = at(f["B"]) - at(f["A"]), at(end_c) - at(start_c)
            if wab in (0, f["work_ab"]) or wc in (0, f["work_c"]) or wab.q > 12 or wc.q > 12:
                continue
            result = grade(_SKILL, q, {**canon, "b": wrong, "c": str(wab), "d": str(wc)})
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong)
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 40


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(120)]
    assert len(set(answers)) >= 100
    assert _exam(7).to_json() == _exam(7).to_json()
