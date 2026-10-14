"""Tests `uc2-divergence-of-field`: the topic `s2a2_divergence` (easy, medium, hard) and the exam chain
`exd_x_s2a2_divergence`.

The oracle reads only the question text. It parses the field out of the stem and re-derives every key with SymPy:
`diff` for the divergence, `solve` for the solenoidal constants, direct substitution for the point values.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_divergence_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-divergence-of-field"
TOPIC = "s2a2_divergence"
EXAM_TOPIC = "exd_x_s2a2_divergence"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, z, k, m = sympy.symbols("x y z k m")
TIERS = ("easy", "medium", "hard")


def read(text: str):
    s = re.sub(r"e\^\(", "exp(", text)
    s = re.sub(r"e\^([a-z])", r"exp(\1)", s)
    s = s.replace("²", "**2").replace("³", "**3").replace("^", "**").replace("ln(", "log(")
    s = s.replace("π", "pi")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x, "y": y, "z": z, "k": k, "m": m})


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


def field_in(text: str):
    match = re.search(r"[A-Za-z]\(x, y, z\) = ⟨(.*?)⟩", text)
    return tuple(read(c) for c in split_top(match.group(1)))


def point_in(text: str):
    match = re.search(r"point (?:P)?\((-?\d+), (-?\d+), (-?\d+)\)", text)
    return tuple(int(c) for c in match.groups()) if match else None


def divergence(F):
    return sum(sympy.diff(c, v) for c, v in zip(F, (x, y, z)))


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    text = q.text
    if "(x² + y² + z²)" in text and "⟨x, y, z⟩" in text:
        n = {"(x² + y² + z²)⟨": 1, "(x² + y² + z²)²⟨": 2}
        for pattern, value in n.items():
            if pattern in text:
                power = value
                break
        else:
            power = int(re.search(r"\^\((-\d+)\)", text).group(1))
        pt = point_in(text)
        F = tuple(c * (x**2 + y**2 + z**2) ** power for c in (x, y, z))
        return divergence(F).subs(dict(zip((x, y, z), pt)))
    F = field_in(text)
    if "solenoidal" in text:
        d = sympy.expand(divergence(F))
        unknowns = [s for s in (k, m) if d.has(s)]
        sol = sympy.solve(sympy.Poly(d, x, y, z).coeffs(), unknowns, dict=True)[0]
        return tuple(sol[s] for s in unknowns) if len(unknowns) > 1 else sol[unknowns[0]]
    d = divergence(F)
    pt = point_in(text)
    return d.subs(dict(zip((x, y, z), pt))) if pt else d


def _same(answer_text: str, expected) -> bool:
    if isinstance(expected, tuple):
        parts = split_top(answer_text.strip()[1:-1])
        return all(sympy.simplify(read(p) - e) == 0 for p, e in zip(parts, expected)) and len(parts) == len(expected)
    return sympy.simplify(read(answer_text) - expected) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert _same(q.answer_text, oracle(q)), (seed, q.text, q.answer_text, oracle(q))


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


def test_expression_answers_are_accepted_in_other_spellings():
    for seed in range(30):
        q = _q("easy", seed)
        expanded = str(sympy.expand(read(q.answer_text))).replace("**", "^")
        assert grade(_SKILL, q, expanded).correct
        assert grade(_SKILL, q, q.answer_text.replace("*", "")).correct or "sin" in q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 25
    assert max(answers.count(a) for a in set(answers)) <= 8


def test_tiers_escalate_in_reasoning_not_range():
    easy = [_q("easy", s) for s in range(30)]
    assert all(not any(f in q.text for f in ("sin", "cos", "e^")) for q in easy)
    assert all(not re.search(r"point|solenoidal", q.text) for q in easy)
    medium = [_q("medium", s) for s in range(30)]
    assert all(any(f in q.text for f in ("sin", "cos", "e^")) for q in medium)
    hard = [_q("hard", s).text for s in range(90)]
    kinds = {
        "solenoidal": sum("solenoidal" in t for t in hard),
        "radial": sum("(x² + y² + z²)" in t for t in hard),
        "exp": sum("exact value" in t for t in hard),
    }
    assert all(v >= 10 for v in kinds.values()), kinds
    assert sum(kinds.values()) == 90


def test_solenoidal_fields_have_a_divergence_free_remainder():
    for seed in range(60):
        q = _q("hard", seed)
        if "solenoidal" not in q.text:
            continue
        F = field_in(q.text)
        d = sympy.expand(divergence(F))
        unknowns = [s for s in (k, m) if d.has(s)]
        assert unknowns
        # with the constants at their values the field is divergence free, and it is not for other values
        sol = dict(zip(unknowns, oracle(q) if isinstance(oracle(q), tuple) else (oracle(q),)))
        assert sympy.expand(d.subs(sol)) == 0
        assert sympy.expand(d.subs({unknowns[0]: sol[unknowns[0]] + 1})) != 0


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    F = field_in(q.stem)
    pt = point_in(q.parts[1].prompt)
    d = sympy.expand(divergence(F))
    at = dict(zip((x, y, z), pt))
    b = d.subs(at)
    g = x**2 + y**2 + z**2
    return dict(F=F, pt=pt, d=d, b=b, k=-b, full=divergence(tuple(g * c for c in F)).subs(at))


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction", "fraction"]
        assert [p.marks for p in q.parts] == [3, 1, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert sympy.simplify(read(parts["a"]) - f["d"]) == 0, (seed, q.stem)
        assert sympy.Rational(parts["b"]) == f["b"]
        assert sympy.Rational(parts["c"]) == f["k"]
        assert sympy.Rational(parts["d"]) == f["full"]
        # the cancelling constant really cancels the divergence at the point
        F = f["F"]
        new = (F[0], F[1], F[2] + f["k"] * z)
        assert divergence(new).subs(dict(zip((x, y, z), f["pt"]))) == 0


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(30):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_divergence_expression():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for wrong in q.parts[0].distractors:
            try:
                wd = read(wrong)
            except Exception:  # noqa: BLE001  (the vector distractor is not an expression)
                continue
            if not isinstance(wd, sympy.Expr):
                continue
            wb = wd.subs(dict(zip((x, y, z), f["pt"])))
            if wb in (0, f["b"]) or wb.q != 1:
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(wb)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong, wb)
            checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_value_into_k_and_the_product_rule():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        g = x**2 + y**2 + z**2
        g0 = g.subs(dict(zip((x, y, z), f["pt"])))
        c0 = f["full"] - g0 * f["b"]
        for wrong in q.parts[1].distractors:
            wb = sympy.Rational(wrong)
            if wb in (0, f["b"]):
                continue
            reply = {**canon, "b": wrong, "c": str(-wb), "d": str(g0 * wb + c0)}
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong)
            assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 60


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(200)]
    assert len(set(answers)) >= 150
    assert _exam(7).to_json() == _exam(7).to_json()
