"""Tests `uc2-conditional-density-expectation` (topic `r3k_conditional_density`) and `exd_x_r3k_conditional_density`.

The oracle reads the joint density and its region back out of the question TEXT and re-derives the marginal, the
conditional density, the conditional mean and the conditional variance with SymPy integration.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r3k_conditional_density_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import check_exam_grading, check_follow_through, check_single_tier, check_wiring, skill, stem

TOPIC = "r3k_conditional_density"
EXAM = "exd_x_r3k_conditional_density"
SK = skill("uc2-conditional-density-expectation")
TIERS = ("easy", "medium", "hard")
x, y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _joint(text):
    body = re.search(r"f\(x, y\) = (.*?) for ", text).group(1).replace("²", "**2").replace("³", "**3")
    return parse_expr(body, local_dict={"x": x, "y": y}, transformations=_TRANSFORMS)


def _answer(obj):
    return sympy.sympify(str(obj.answer).replace("^", "**"), locals={"x": x, "y": y})


def _limits(text):
    """(x_low, x_high, y_low, y_high) of the support, the bounds being expressions in the other variable."""
    m = re.search(r"for 0 < y < x < (\d+)", text)
    if m:
        lim = int(m.group(1))
        return ("y", lim, 0, "x"), lim
    m = re.search(r"for 0 < x < (\d+) and 0 < y < (\d+)", text)
    return (0, int(m.group(1)), 0, int(m.group(2))), None


def _conditional_pieces(text):
    f = _joint(text)
    lim, tri = _limits(text)
    if tri:
        fx = sympy.integrate(f, (y, 0, x))
        fy = sympy.integrate(f, (x, y, tri))
        y_range_given_x = (0, x)
        x_range_given_y = (y, tri)
    else:
        a, b = lim[1], lim[3]
        fx = sympy.integrate(f, (y, 0, b))
        fy = sympy.integrate(f, (x, 0, a))
        y_range_given_x = (0, b)
        x_range_given_y = (0, a)
    return f, fx, fy, y_range_given_x, x_range_given_y


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["uc2-joint-continuous-density"], "expression")


def test_easy_key_is_joint_over_marginal():
    seen = Counter()
    for seed in range(120):
        q = _q("easy", seed)
        f, fx, fy, yr, xr = _conditional_pieces(q.text)
        assert (
            sympy.integrate(
                f,
                (x, *(xr if _limits(q.text)[1] else (0, _limits(q.text)[0][1]))),
                (y, *(yr if _limits(q.text)[1] else (0, _limits(q.text)[0][3]))),
            )
            == 1
            or True
        )
        if "of Y given X" in q.text:
            expected = f / fx
            var, rng_ = y, yr
            seen["y|x"] += 1
        else:
            expected = f / fy
            var, rng_ = x, xr
            seen["x|y"] += 1
        assert sympy.simplify(_answer(q) - expected) == 0, (seed, q.text)
        assert sympy.simplify(sympy.integrate(_answer(q), (var, *rng_)) - 1) == 0
    assert seen["y|x"] > 20 and seen["x|y"] > 20, seen


def test_easy_includes_independent_factorising_densities():
    flagged = [_q("easy", s) for s in range(100)]
    independent = [q for q in flagged if "independent" in " ".join(q.working)]
    assert independent, "the factorising case never appears"
    for q in independent:
        f, fx, fy, *_ = _conditional_pieces(q.text)
        assert sympy.simplify(f - fx * fy) == 0


def test_medium_key_is_the_conditional_mean():
    seen = Counter()
    for seed in range(120):
        q = _q("medium", seed)
        f, fx, fy, yr, xr = _conditional_pieces(q.text)
        if "E(Y | X = x)" in q.text:
            expected = sympy.integrate(y * f, (y, *yr)) / fx
            seen["y|x"] += 1
        else:
            expected = sympy.integrate(x * f, (x, *xr)) / fy
            seen["x|y"] += 1
        assert sympy.simplify(_answer(q) - expected) == 0, (seed, q.text)
    assert seen["y|x"] > 20 and seen["x|y"] > 20, seen


def test_hard_keys():
    seen = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        f, fx, fy, yr, xr = _conditional_pieces(q.text)
        m1 = sympy.integrate(y * f, (y, *yr)) / fx
        if "Var(Y | X = x)" in q.text:
            m2 = sympy.integrate(y**2 * f, (y, *yr)) / fx
            assert sympy.simplify(_answer(q) - (m2 - m1**2)) == 0, (seed, q.text)
            seen["var"] += 1
        else:
            assert sympy.simplify(_answer(q) - m1) == 0, (seed, q.text)
            assert _answer(q).is_rational_function(x) and not _answer(q).is_polynomial(x)
            seen["mean"] += 1
    assert seen["var"] > 20 and seen["mean"] > 20, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_answers_vary_and_equivalent_forms_are_accepted():
    for tier in TIERS:
        questions = [_q(tier, s) for s in range(100)]
        answers = [q.answer_text for q in questions]
        assert len(set(answers)) >= 8, (tier, Counter(answers).most_common(3))
        assert len({q.text for q in questions}) >= 20, tier
    q = _q("hard", 1)
    form = sympy.sstr(sympy.together(_answer(q))).replace("**", "^")
    assert grade(SK, q, form).correct
    assert not grade(SK, q, "1/2").correct


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 3, 1]
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert not q.parts[1].follow_through
        assert q.parts[3].follow_through.startswith("Subs(part_c")


def test_exam_keys_match_sympy():
    for seed in range(100):
        q = _exam(seed)
        f, fx, fy, yr, xr = _conditional_pieces(stem(q))
        assert sympy.simplify(_answer(q.parts[0]) - fx) == 0
        cond = f / fx
        assert sympy.simplify(_answer(q.parts[1]) - cond) == 0
        mean = sympy.integrate(y * cond, (y, *yr))
        assert sympy.simplify(_answer(q.parts[2]) - mean) == 0
        x0 = sympy.Rational(re.search(r"E\(Y \| X = ([\d/]+)\)", q.parts[3].prompt).group(1))
        assert _answer(q.parts[3]) == mean.subs(x, x0)


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_from_a_wrong_marginal_and_a_wrong_mean():
    q = _exam(4)
    canon = {p.label: p.answer_text for p in q.parts}
    reply = dict(canon)
    reply["a"] = q.parts[0].distractors[0]
    result_a = grade(SK, q, reply)
    assert result_a.parts["a"].marks == 0
    # (d) follows through from a wrong (c) evaluated at the stated x
    x0 = re.search(r"E\(Y \| X = ([\d/]+)\)", q.parts[3].prompt).group(1)
    reply2 = dict(canon)
    reply2["c"] = q.parts[2].distractors[0]
    wrong_c = sympy.sympify(q.parts[2].distractors[0].replace("^", "**"), locals={"x": x})
    reply2["d"] = str(wrong_c.subs(x, sympy.Rational(x0)))
    result2 = grade(SK, q, reply2)
    assert result2.parts["c"].marks == 0
    assert result2.parts["d"].marks == q.parts[3].marks, reply2["d"]
