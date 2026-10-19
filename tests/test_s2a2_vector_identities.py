"""Tests `uc2-vector-identities`: the topic `s2a2_vector_identities` (easy, medium, hard) and the exam chain
`exd_x_s2a2_vector_identities`.

The oracle reads only the question text. It parses the scalar and vector fields out of the stem and evaluates the
asked quantity by expanding the product and differentiating directly with SymPy, never by the identity the generator
used.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.s2a2_vector_identities_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

SKILL_ID = "uc2-vector-identities"
TOPIC = "s2a2_vector_identities"
EXAM_TOPIC = "exd_x_s2a2_vector_identities"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
x, y, z = sympy.symbols("x y z")
XYZ = (x, y, z)
TIERS = ("easy", "medium", "hard")


def read(text: str):
    s = text.replace("²", "**2").replace("³", "**3").replace("⁴", "**4").replace("^", "**")
    return parse_expr(s, transformations=_TRANSFORMS, local_dict={"x": x, "y": y, "z": z})


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


def scalar(text: str, name: str):
    match = re.search(
        name + r"\(x, y, z\) = (.+?)(?= and [A-Za-z]\(x, y, z\)|\. Find| \(degrees| degrees|\. The)", text
    )
    return read(match.group(1))


def vector(text: str, name: str):
    match = re.search(name + r"\(x, y, z\) = ⟨(.*?)⟩", text)
    return tuple(read(c) for c in split_top(match.group(1)))


def point(text: str):
    match = re.search(r"(?:point|is) \(?(-?\d+), (-?\d+), (-?\d+)\)", text)
    return dict(zip(XYZ, (int(c) for c in match.groups())))


def grad(f):
    return tuple(sympy.diff(f, v) for v in XYZ)


def div(F):
    return sum(sympy.diff(c, v) for c, v in zip(F, XYZ))


def curl(F):
    P, Q, R = F
    return (
        sympy.diff(R, y) - sympy.diff(Q, z),
        sympy.diff(P, z) - sympy.diff(R, x),
        sympy.diff(Q, x) - sympy.diff(P, y),
    )


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def lap(f):
    return sum(sympy.diff(f, v, 2) for v in XYZ)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def oracle(q):
    text = q.text
    at = point(text)
    if "∇ × (f F)" in text:
        f, F = scalar(text, "f"), vector(text, "F")
        return tuple(sympy.simplify(c.subs(at)) for c in curl(tuple(f * c for c in F)))
    if "∇ · (f F)" in text:
        f, F = scalar(text, "f"), vector(text, "F")
        return sympy.simplify(div(tuple(f * c for c in F)).subs(at))
    if "∇ · (F × G)" in text:
        return sympy.simplify(div(cross(vector(text, "F"), vector(text, "G"))).subs(at))
    if "∇²(f g)" in text:
        return sympy.simplify(lap(scalar(text, "f") * scalar(text, "g")).subs(at))
    if "∇ · (∇f × G)" in text:
        return sympy.simplify(div(cross(grad(scalar(text, "f")), vector(text, "G"))).subs(at))
    if "∇²f" in text:
        return sympy.simplify(lap(scalar(text, "f")).subs(at))
    if "f ∇g - g ∇f" in text:
        f, g = scalar(text, "f"), scalar(text, "g")
        return sympy.simplify(div(tuple(f * a - g * b for a, b in zip(grad(g), grad(f)))).subs(at))
    if "∇ × (∇ × F)" in text:
        return tuple(sympy.simplify(c.subs(at)) for c in curl(curl(vector(text, "F"))))
    raise AssertionError(text)


def _same(answer_text: str, expected) -> bool:
    if isinstance(expected, tuple):
        parts = split_top(answer_text.strip()[1:-1])
        return len(parts) == len(expected) and all(sympy.simplify(read(p) - e) == 0 for p, e in zip(parts, expected))
    return sympy.simplify(read(answer_text) - expected) == 0


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_independent_sympy_derivation(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert _same(q.answer_text, oracle(q)), (seed, q.text, q.answer_text, oracle(q))


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM_TOPIC, "exam", ["exam"])
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_and_distinct(tier):
    for seed in range(30):
        q = _q(tier, seed)
        assert len(q.option_notes) == len(q.distractors) >= 3
        assert len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(60)]
    assert len(set(answers)) >= 30
    assert max(answers.count(a) for a in set(answers)) <= 6


def test_tiers_differ_in_shape_and_every_variant_appears():
    easy = [_q("easy", s).text for s in range(40)]
    assert any("∇ × (f F)" in t for t in easy) and any("∇ · (f F)" in t for t in easy)
    medium = [_q("medium", s).text for s in range(60)]
    assert all(any(op in t for op in ("∇ · (F × G)", "∇²(f g)", "∇ · (∇f × G)")) for t in medium)
    assert {op for t in medium for op in ("∇ · (F × G)", "∇²(f g)", "∇ · (∇f × G)") if op in t} == {
        "∇ · (F × G)",
        "∇²(f g)",
        "∇ · (∇f × G)",
    }
    hard = [_q("hard", s).text for s in range(60)]
    assert (
        any("∇²f" in t for t in hard)
        and any("f ∇g - g ∇f" in t for t in hard)
        and any("∇ × (∇ × F)" in t for t in hard)
    )


def test_hard_laplacian_of_a_power_is_n_times_n_plus_one():
    seen = 0
    for seed in range(120):
        q = _q("hard", seed)
        if "∇²f" not in q.text:
            continue
        f = scalar(q.text, "f")
        exponent = sympy.Poly(f.args[0] if f.is_Pow else f, x, y, z) if False else f.as_base_exp()[1]
        n = 2 * exponent
        r = sympy.sqrt(sum(v * v for v in point(q.text).values()))
        assert sympy.simplify(read(q.answer_text) - n * (n + 1) * r ** (n - 2)) == 0, (seed, q.text)
        seen += 1
    assert seen >= 8


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _facts(q):
    f, F = scalar(q.stem, "f"), vector(q.stem, "v")
    at = point(q.stem)
    a = div(F).subs(at)
    b = sum(c.subs(at) * g.subs(at) for c, g in zip(F, grad(f)))
    return dict(f=f, F=F, at=at, a=a, b=b, f0=f.subs(at), c=div(tuple(f * c for c in F)).subs(at))


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "fraction", "mcq"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert find_text_defects(q) == []


def test_exam_keys_match_independent_derivation():
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        parts = _canonical(q)
        assert sympy.Rational(parts["a"]) == f["a"], (seed, q.stem)
        assert sympy.Rational(parts["b"]) == f["b"]
        assert sympy.Rational(parts["c"]) == f["c"]


def test_exam_mcq_is_a_true_identity_and_its_options_are_not():
    texts = set()
    for seed in range(60):
        q = _exam(seed)
        mcq = q.parts[3]
        assert len(mcq.distractors) == 4
        assert mcq.answer_text not in mcq.distractors
        assert len(set(mcq.distractors)) == 4
        texts.add(mcq.answer_text)
    assert len(texts) == 4  # four different identities are asked across seeds


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(25):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_into_the_product_rule_from_wrong_a_or_b():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        f = _facts(q)
        canon = _canonical(q)
        for label, index in (("a", 0), ("b", 1)):
            for wrong in q.parts[index].distractors:
                w = sympy.Rational(wrong)
                if w in (0, f[label]):
                    continue
                a_, b_ = (w, f["b"]) if label == "a" else (f["a"], w)
                c = f["f0"] * a_ + b_
                if c == f["c"]:
                    continue
                result = grade(_SKILL, q, {**canon, label: wrong, "c": str(c)})
                assert result.parts[label].marks == 0
                assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, label, wrong)
                checked += 1
    assert checked >= 50


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(100)]
    assert len(set(answers)) >= 60
    assert _exam(7).to_json() == _exam(7).to_json()
