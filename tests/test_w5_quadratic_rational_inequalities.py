"""Tests `quadratic-inequalities` and `rational-inequalities` (topics `w5_quadratic_inequality`,
`w5_rational_inequality` and their exam chains).

The oracle reads the inequality back out of the question TEXT and solves it with SymPy's own univariate inequality
solver, a different route from the generator's sign-diagram construction.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import mathgen.generators.w5_quadratic_inequalities_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from mathgen.grading.inequality import read_solution_set
from tests import w5_support as sup

QUAD, QUAD_EXAM = "w5_quadratic_inequality", "exd_x_w5_quadratic_inequality"
RATL, RATL_EXAM = "w5_rational_inequality", "exd_x_w5_rational_inequality"
QSKILL, RSKILL = sup.skill_of("quadratic-inequalities"), sup.skill_of("rational-inequalities")

X, T, N = sympy.symbols("x t n")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_REL = re.compile(r"^(?:Solve the inequality |Solve |Find all values for which )(.*?) (≤|≥|<|>) (-?\d+)\. ")
_MODEL = re.compile(r"[hP] = (.+?)\. For which values of ([tn]) is [hP] (more|less) than (\d+)")
_OPS = {"≤": "<=", "≥": ">=", "<": "<", ">": ">"}


def _expr(text: str, var=X):
    return parse_expr(text.replace("sqrt", "sqrt"), local_dict={"x": X, "t": T, "n": N}, transformations=_TRANSFORMS)


def _solve(lhs, op: str, rhs, var=X, domain=sympy.S.Reals):
    rel = sympy.Rel(lhs, rhs, _OPS[op])
    return sympy.solve_univariate_inequality(rel, var, relational=False, domain=domain)


def _same(a, b) -> bool:
    return a == b or a.symmetric_difference(b) == sympy.S.EmptySet


def _oracle(q):
    model = _MODEL.search(q.text)
    if model:
        expr, var, word, level = model.groups()
        v = {"t": T, "n": N}[var]
        return _solve(_expr(expr), ">" if word == "more" else "<", int(level), v, sympy.Interval(0, sympy.oo)), v
    m = _REL.match(q.text)
    assert m, q.text
    lhs, op, rhs = m.groups()
    return _solve(_expr(lhs), op, int(rhs)), X


def _check_oracle(topic, tier, seeds=40):
    kinds = set()
    for seed in range(seeds):
        q = sup.single(topic, tier, seed)
        want, _ = _oracle(q)
        got, _ = read_solution_set(q.answer_text)
        assert _same(want, got), (tier, seed, q.text, q.answer_text, want)
        kinds.add(type(want).__name__)
    return kinds


def test_wiring():
    for skill, topic, exam in ((QSKILL, QUAD, QUAD_EXAM), (RSKILL, RATL, RATL_EXAM)):
        sup.check_wiring(skill, topic, exam, "inequality")
        steps = [skill.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
        assert steps == sorted(steps) and len(set(steps)) == 4
        assert skill.sources[-1].params == {"tier": "exam"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_quadratic_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(QSKILL, QUAD, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_rational_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(RSKILL, RATL, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_quadratic_keys_match_sympy(tier):
    _check_oracle(QUAD, tier)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_rational_keys_match_sympy(tier):
    _check_oracle(RATL, tier)


def test_quadratic_hard_covers_every_shape():
    shapes = set()
    for seed in range(120):
        q = sup.single(QUAD, "hard", seed)
        if "metres" in q.text or "dollars" in q.text:
            shapes.add("model")
        elif "sqrt" in q.text and "Give exact bounds" in q.text:
            shapes.add("surd")
        elif q.answer_text in ("all real numbers", "no solution"):
            shapes.add("no-roots")
        else:
            shapes.add("repeated")
    assert shapes == {"model", "surd", "no-roots", "repeated"}


def test_hard_tiers_add_shapes_the_easy_tier_never_shows():
    easy = {q for q in (sup.single(QUAD, "easy", s).answer_text for s in range(80))}
    assert not easy & {"all real numbers", "no solution"}
    assert all(re.fullmatch(r"-?\d+", b) or b == "x" for a in easy for b in re.findall(r"-?[\d/]+|x", a))
    rational_hard = [sup.single(RATL, "hard", s).text for s in range(60)]
    assert any("^2" in t for t in rational_hard) and any(")(" in t for t in rational_hard)


def test_unions_are_accepted_in_any_spelling():
    q = next(q for q in (sup.single(QUAD, "easy", s) for s in range(40)) if " or " in q.answer_text)
    lo, hi = re.findall(r"-?\d+", q.answer_text)[:2]
    strict = "<" in q.answer_text.split(" or ")[0]
    open_, close = ("(", ")") if strict else ("(", "]")
    spellings = [
        q.answer_text,
        q.answer_text.replace(" or ", " ∪ "),
        f"x < {lo}, x > {hi}".replace("<", "<" if strict else "≤").replace(">", ">" if strict else "≥"),
    ]
    for reply in spellings:
        assert grade(QSKILL, q, reply).correct, reply
    assert not grade(QSKILL, q, f"x > {lo}").correct
    assert not grade(QSKILL, q, f"{lo} < x < {hi}").correct
    del open_, close


@pytest.mark.parametrize("tier", sup.TIERS)
def test_answers_are_unguessable(tier):
    for topic in (QUAD, RATL):
        assert len(sup.answer_spread(topic, tier, 60)) >= 12, (topic, tier)


def test_pools_and_shapes():
    sup.check_pools(QUAD)
    sup.check_pools(RATL)


# -- exam chains ----------------------------------------------------------------------------------------------------


def test_quadratic_exam_chain():
    sup.check_exam_chain_kinds(
        QSKILL,
        QUAD_EXAM,
        [1, 2, 1, 1],
        [{"number", "fraction"}, {"inequality"}, {"number", "fraction"}, {"number", "fraction"}],
    )
    for seed in range(60):
        q = sup.exam(QUAD_EXAM, seed)
        f = _expr(re.match(r"Let f\(x\) = (.*?), which", q.text).group(1))
        roots = sorted(sympy.solve(f, X))
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == Fraction(str(sum(roots) / 2))
        op = re.search(r"Solve f\(x\) (\S+) 0", b.prompt).group(1)
        assert _same(_solve(f, op, 0), read_solution_set(b.answer_text)[0])
        op_c = re.search(r"satisfy f\(x\) (\S+) 0", c.prompt).group(1)
        assert int(c.answer_text) == sum(
            1 for k in range(-40, 41) if sympy.Rel(f.subs(X, k), 0, _OPS[op_c]) == sympy.true
        )
        vertex = f.subs(X, sum(roots) / 2)
        assert Fraction(d.answer_text) == Fraction(str(vertex))
        assert ("least" in d.prompt) == (sympy.Poly(f, X).LC() > 0)


def test_quadratic_exam_follows_through_from_the_turning_point():
    """A wrong turning point carries into the count (c) and the extreme value (d): the learner's own later answers,
    worked from their own turning point, still earn the marks."""
    checked = {"c": 0, "d": 0}
    for label in checked:
        for seed in range(30):
            q = sup.exam(QUAD_EXAM, seed)
            canon = {p.label: p.answer_text for p in q.parts}
            f = _expr(re.match(r"Let f\(x\) = (.*?), which", q.text).group(1))
            p_root = min(sympy.solve(f, X))
            lead = sympy.Poly(f, X).LC()
            v = Fraction(canon["a"]) + 2
            if v == 0:  # a zero earlier reply earns no follow-through, by the marking rule
                v += 1
            if label == "c":
                strict = "≥" not in q.parts[2].prompt and "≤" not in q.parts[2].prompt
                mine = 2 * (v - Fraction(str(p_root))) + (-1 if strict else 1)
            else:
                mine = -Fraction(str(lead)) * (v - Fraction(str(p_root))) ** 2
            result = grade(QSKILL, q, {**canon, "a": str(v), label: str(mine)})
            part = next(p for p in q.parts if p.label == label)
            assert result.parts[label].marks == part.marks, (seed, label, v, mine)
            assert result.parts[label].follow_through is True
            checked[label] += 1
    assert min(checked.values()) >= 25


def test_rational_exam_chain():
    sup.check_exam_chain_kinds(
        RSKILL, RATL_EXAM, [1, 2, 3, 2], [{"number"}, {"inequality"}, {"inequality"}, {"number"}]
    )
    for seed in range(60):
        q = sup.exam(RATL_EXAM, seed)
        g = _expr(re.match(r"Let g\(x\) = (.*?)\. \(a\)", q.text).group(1))
        a, b, c, d = q.parts
        assert int(a.answer_text) == sympy.solve(sympy.denom(sympy.together(g)), X)[0]
        op_b = re.search(r"Solve g\(x\) (\S+) 0", b.prompt).group(1)
        assert _same(_solve(g, op_b, 0), read_solution_set(b.answer_text)[0])
        op_c, k = re.search(r"Solve g\(x\) (\S+) (-?\d+)\.", c.prompt).groups()
        assert _same(_solve(g, op_c, int(k)), read_solution_set(c.answer_text)[0])
        want = sum(
            1
            for n in range(-10, 11)
            if n != int(a.answer_text) and sympy.Rel(g.subs(X, n), int(k), _OPS[op_c]) == sympy.true
        )
        assert int(d.answer_text) == want
