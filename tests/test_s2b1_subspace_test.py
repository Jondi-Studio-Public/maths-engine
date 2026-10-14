"""Tests `uc2-subspace-test` (topic `s2b1_subspace_test`) and its exam chain `exd_x_s2b1_subspace`.

The oracle reads every set back out of the question TEXT and decides, by theory rather than by the generator's
grid test, whether it is a subspace: homogeneous linear conditions are, anything with a non-zero constant, an
inequality, an integrality or a non-linear condition is not, and unions, intersections and complements are decided
by containment of the planes and the line involved.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication, parse_expr, standard_transformations

import mathgen.generators.s2b1_subspace_test_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "s2b1_subspace_test"
EXAM = "exd_x_s2b1_subspace"
SKILL_ID = "uc2-subspace-test"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_K = sympy.Symbol("k")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _parse(text):
    """A polynomial in k written as 'k² - 6k + 8'."""
    return parse_expr(
        text.replace("²", "**2"),
        local_dict={"k": _K},
        transformations=standard_transformations + (implicit_multiplication,),
    )


def _coeffs(expr):
    """Coefficients of x, y, z in a text such as '-2x + y - 3z'."""
    out = []
    for name in "xyz":
        m = re.search(r"([+-]?)\s*(\d*)" + name, expr)
        out.append(0 if not m else (-1 if m.group(1) == "-" else 1) * (int(m.group(2)) if m.group(2) else 1))
    return out


_FIXED_NO = (
    "x ≥",
    "integers",
    "xz = y²",
    "xy =",
    "x² + y² + z²",
    "|x| = |z|",
    "invertible",
    "A² = A",
    "at least",
    "non-negative integer",
    "every coefficient of p is an integer",
    "degree exactly",
    "p(x) ≥",
    "det A",
)
_FIXED_YES = ("upper triangular", "Aᵀ = A}", "p(-x) = p(x)", "divisible by")


def _rhs_zero(body):
    return re.search(r"= (-?\d+)\s*$", body) is not None and int(re.search(r"= (-?\d+)\s*$", body).group(1)) == 0


def _is_subspace(item):
    """Decide one listed set from its text alone (not the unions of planes, handled by `_conceptual`)."""
    body = item.strip()
    if "x = " in body and (" or " in body):
        return False
    if any(token in body for token in _FIXED_NO):
        return False
    if "Aᵀ = A or" in body:
        return False
    if any(token in body for token in _FIXED_YES):
        return True
    if "AB = BA" in body or re.search(r"p\(-?\d+\) = p\(-?\d+\)", body):
        return True
    if re.search(r"p\(-?\d+\) p\(-?\d+\)", body):
        return False
    if "A(" in body and "ᵀ = (" in body:
        lhs, rhs = body.split("ᵀ = ", 1)
        vec = re.findall(r"-?\d+", rhs)
        return all(int(v) == 0 for v in vec)
    if " and " in body and body.startswith("{(x, y, z)"):
        return all(int(m) == 0 for m in re.findall(r"= (-?\d+)(?= and|\})", body)) if "x =" not in body else True
    if "x = " in body and " and " in body:
        return True
    if "≥" in body:
        return False
    return _rhs_zero(body.rstrip("}")) if "=" in body else False


def _items(text):
    return re.findall(r"\((?:i|ii|iii|iv)\) (.+?)(?=; \((?:i|ii|iii|iv)\)|\. |\?| is a subset)", text)


def _conceptual(text, items):
    def plane(letter):
        m = re.search(letter + r" = \{\(x, y, z\) : (.+?)\}", text)
        return _coeffs(m.group(1).split(" = ")[0])

    n1, n2 = plane("U"), plane("W")
    v = sympy.Matrix([int(c) for c in re.search(r"L = span\{\(([^)]*)\)\}", text).group(1).split(",")])
    in_u = sympy.Matrix([n1]) * v == sympy.Matrix([0])
    in_w = sympy.Matrix([n2]) * v == sympy.Matrix([0])
    both_dir = sympy.Matrix([n1, n2]).nullspace()[0]
    verdicts = []
    for it in items:
        if it == "U ∩ W" or it == "W ∩ L":
            verdicts.append(True)
        elif it == "U ∪ W":
            verdicts.append(False)
        elif it == "U ∪ L":
            verdicts.append(in_u)
        elif it == "W ∪ L":
            verdicts.append(in_w)
        elif it == "U ∩ (W ∪ L)":
            # (U ∩ W) ∪ (U ∩ L): a union of a line and (the line L or {0}).
            parallel = sympy.Matrix.hstack(both_dir, v).rank() == 1
            verdicts.append(bool(in_u is False or in_u and parallel))
        else:
            verdicts.append(False)  # complements, with or without 0
    return verdicts


def _oracle_count(q):
    text = q.text
    items = _items(text)
    if "L = span" in text:
        return sum(_conceptual(text, items))
    return sum(_is_subspace(it) for it in items)


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert "university-year-2" in _SKILL.tags


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_key_matches_independent_count(tier):
    for seed in range(200):
        q = _q(tier, seed)
        items = _items(q.text)
        assert len(items) in (3, 4), (seed, q.text)
        assert int(q.answer) == _oracle_count(q), (seed, q.text)


def _hard_root(q):
    text = q.text
    m = re.search(r"A(\([^)]*\)ᵀ) = \(k ([+-]) (\d+), k² - (\d+)\)ᵀ", text)
    if m:
        r = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
        return -r  # k + r = 0 or k - r = 0
    rhs = re.search(r"= (k² [^}]*?)\}", text)
    poly = _parse(rhs.group(1))
    roots = sorted(sympy.solve(poly, _K))
    return roots[-1]


def test_hard_key_matches_independent_derivation():
    seen = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        if "L = span" in q.text:
            seen["conceptual"] += 1
            assert int(q.answer) == _oracle_count(q), (seed, q.text)
        else:
            seen["parameter"] += 1
            assert int(q.answer) == _hard_root(q), (seed, q.text)
    assert min(seen.values()) >= 20 and len(seen) == 2


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(50):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_differ_in_shape():
    easy = Counter("ℝ³" in _q("easy", s).text for s in range(60))
    assert easy[True] == 60
    medium = Counter(
        "P₂" if "P₂" in _q("medium", s).text else "M₂" if "M₂" in _q("medium", s).text else "R3" for s in range(90)
    )
    assert len(medium) >= 2 and min(medium.values()) >= 10
    hard = Counter("k" in _q("hard", s).text for s in range(90))
    assert hard[True] >= 30 and hard[False] >= 10


def test_answer_variety():
    for tier in TIERS:
        texts = {_q(tier, s).text for s in range(80)}
        assert len(texts) >= 40, (tier, len(texts))
        assert len({_q(tier, s).answer_text for s in range(80)}) >= 3


# -- exam -------------------------------------------------------------------------------------------------------------


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        text = q.text
        body = re.search(r"satisfying (.+?)\. \(a\)", text).group(1)
        eqs = body.split(" and ")
        first_lhs, first_rhs = eqs[0].split(" = ")
        rows = [_coeffs(e.split(" = ")[0]) for e in eqs]
        c = _parse(first_rhs)
        roots = sorted(sympy.solve(c, _K))
        assert len(roots) == 2
        m = int(re.search(r"Take k = (-?\d+)", text).group(1))
        assert c.subs(_K, m) != 0  # the zero vector is missing at the stated k
        assert q.parts[0].answer_text == "The zero vector is not in S_k"
        assert int(q.parts[1].answer) == roots[-1]
        assert int(q.parts[2].answer) == 3 - sympy.Matrix(rows).rank()
        assert int(q.parts[3].answer) == roots[0]


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["mcq", "number", "number", "number"]
        assert q.parts[3].follow_through is not None and "part_b" in q.parts[3].follow_through
        assert find_text_defects(q) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(60):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 9
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_larger_root():
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        total = int(q.parts[1].answer) + int(q.parts[3].answer)
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (int(q.parts[1].answer) + 2, int(q.parts[1].answer) - 3):
            reply = {**canon, "b": str(wrong), "d": str(total - wrong)}
            # The grader withholds follow-through when it would merely repeat the earlier reply, or hit a distractor.
            if reply["d"] in q.parts[3].distractors or total - wrong in (0, wrong):
                continue
            result = grade(_SKILL, q, reply)
            assert result.parts["b"].marks == 0
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 40


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 30
