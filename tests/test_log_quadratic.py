"""Recipe subject `log_quadratic` (mathgen/generators/recipe_subject_log_quadratic.py): ln of a quadratic."""

import math
import random

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication, parse_expr, standard_transformations

from mathgen import core
from mathgen.grading import grade
from mathgen.recipes import RecipeError, conformance
from mathgen.generators import recipe_asks_generic as m
from mathgen.generators.recipe_subject_log_quadratic import LogQuadratic

TOPIC = "recipe_methods_ln_quadratic_1"
SEEDS = range(120)


def _ctx():
    from mathgen.recipes import AskContext

    return AskContext(label="a", index=0)


def test_recipe_conforms():
    assert conformance("methods_ln_quadratic_1") == []


def test_exam_q7_numbers():
    """VCE Methods 2023 Exam 1 Q7: f(x) = ln(3 + 2x - x^2): domain -1 < x < 3, maximum at x = 1, f = 0 at 1 ± √3."""
    s = LogQuadratic(r1=-1, m=2, k=1, ascending=True)
    assert s.stem() == "Let f(x) = ln(3 + 2x - x^2)."
    rng = random.Random(0)
    a = m.implied_domain(s, rng, _ctx())
    b = m.stationary_point(s, rng, _ctx())
    c = m.solve_equal_constant(s, rng, _ctx())
    assert a["answer_text"] == "-1 < x < 3"
    assert b["answer"] == 1
    assert c["answer_text"] == "x = 1 + √3" and "f(x) = 0" in c["prompt"]


@pytest.mark.parametrize("seed", SEEDS)
def test_answers_against_an_independent_oracle(seed):
    q = core.make(TOPIC, seed=seed)
    xs = sp.Symbol("x")
    inner = q.stem[q.stem.index("ln(") + 3 : -2].replace("^", "**")
    poly = sp.Poly(parse_expr(inner, transformations=standard_transformations + (implicit_multiplication,)), xs)
    coefficients = poly.all_coeffs()
    roots = sorted(sp.roots(poly))
    r1, r2 = roots
    assert coefficients[0] == -1
    assert q.parts[0].answer_text == f"{r1} < x < {r2}"
    assert q.parts[1].answer == (r1 + r2) // 2
    # numeric: q is positive only inside, and f is largest at the claimed x
    pts = [r1 + (r2 - r1) * i / 200 for i in range(1, 200)]
    best = max(pts, key=lambda p: math.log(float(poly.eval(p))))
    assert abs(best - q.parts[1].answer) <= (r2 - r1) / 200 + 1e-9
    # the solutions satisfy f(x) = rhs
    rhs = 0 if "f(x) = 0" in q.parts[2].prompt else math.log(int(q.parts[2].prompt.split("ln(")[1].split(")")[0]))
    h = q.parts[1].answer
    d = int(q.parts[2].answer_text.split("√")[1])
    for x in (h - math.sqrt(d), h + math.sqrt(d)):
        assert math.log(float(poly.eval(x))) == pytest.approx(rhs, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_canonical_answers_grade_correct_and_distractors_do_not(seed):
    q = core.make(TOPIC, seed=seed)
    result = grade("number", q, q.answer_text)
    assert result.correct, {k: v.correct for k, v in result.parts.items()}
    for part in q.parts:
        for text in part.distractors:
            assert not grade(part.answer_kind, part, text).correct, (part.label, text)


def test_fewer_than_three_prompt_shapes_per_part_are_avoided():
    prompts = {i: set() for i in range(3)}
    for seed in range(60):
        q = core.make(TOPIC, seed=seed)
        for i, p in enumerate(q.parts):
            prompts[i].add(p.prompt)
    assert all(len(v) >= 3 for v in prompts.values())


def test_technology_active_true_is_refused_clearly():
    with pytest.raises(RecipeError, match="exact only"):
        core.make(TOPIC, seed=1, technology_active=True)


@pytest.mark.parametrize("seed", range(60))
def test_declared_properties_are_true_of_the_drawn_subject(seed):
    """The recipe vocabulary is closed and honest: each property the subject declares holds for the draw."""
    s = LogQuadratic.draw(random.Random(seed), False)
    x = sp.Symbol("x", real=True)
    argument = s.expr.args[0]
    assert {
        "quadratic_argument",
        "restricted_domain",
        "has_stationary_point",
        "has_surd_roots",
    } <= LogQuadratic.properties
    assert LogQuadratic.families == {"logarithmic"}
    assert sp.Poly(argument, x).degree() == 2  # quadratic_argument
    assert sp.N(argument.subs(x, s.r1 - 1)) < 0  # restricted_domain: ln undefined outside the roots
    assert sp.solve(sp.diff(s.expr, x), x)  # has_stationary_point
    assert all(not sol.is_rational for sol in s.solutions())  # has_surd_roots


@pytest.mark.parametrize("seed", range(150))
def test_a_shifted_or_wrong_solution_is_rejected(seed):
    """Only the larger solution is accepted: a shifted p, the smaller root, a bare ± form and a bare √d are not."""
    q = core.make(TOPIC, seed=seed)
    part = q.parts[2]
    h, d = q.parts[1].answer, part.answer_text.split("√")[1]
    assert grade("expression", part, part.answer_text).correct
    assert grade("expression", part, f"{h}+sqrt({d})").correct
    bad = [
        f"x = {h + 1} + √{d}",
        f"x = {h - 1} + √{d}",
        f"x = {h} - √{d}",
        f"x = {h} ± √{d}",
        f"x = ± √{d}",
    ]
    if h != 0:
        bad += [f"x = √{d}", f"x = {-h} + √{d}"]
    for reply in bad:
        assert not grade("expression", part, reply).correct, reply
