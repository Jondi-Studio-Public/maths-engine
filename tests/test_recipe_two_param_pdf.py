"""The `two_param_pdf` recipe subject: a density with two unknown constants, answered exactly.

Every ask is re-derived here from the rendered stem and prompts by a different route from the generator's: the
generator solves area = 1 for c and then E(X) = given for d; these tests solve the two equations together as a
system and keep the solution with c, d > 0 and f >= 0 on [0, d].
"""

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import core, recipes
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE

RECIPE = "methods_two_param_pdf_1"
TOPIC = "recipe_" + RECIPE
SEEDS = range(150)
x, c, d = sympy.symbols("x k w", positive=True)  # c and d below are the unknowns k and w


def _q(seed):
    return core.make(TOPIC, seed=seed)


def _frac(text):
    return Fraction(text)


def _read(q):
    """The density, the stated mean and the threshold, read back from the question text."""
    stem = q.stem
    if "kx +" in stem:
        k = sympy.Rational(re.search(r"kx \+ (\d+(?:/\d+)?)", stem).group(1))
        density = c * x + k
        family = "linear"
    else:
        n = int(re.search(r"kx\^(\d)", stem).group(1)) if "kx^" in stem else 1
        density = c * x**n
        family = "power"
    mean = sympy.Rational(re.search(r"E\(X\) = (\d+(?:/\d+)?)", q.parts[1].prompt).group(1))
    m = sympy.Rational(re.search(r"P\(X > (\d+(?:/\d+)?)\)", q.parts[3].prompt).group(1))
    return family, density, mean, m


def _solve(density, mean):
    """Every (c, d) with area 1 and E(X) = mean, by solving the pair of equations together."""
    area = sympy.integrate(density, (x, 0, d)) - 1
    first = sympy.integrate(x * density, (x, 0, d)) - mean
    found = []
    for sol in sympy.solve([area, first], [c, d], dict=True):
        cv, dv = sol[c], sol[d]
        if cv > 0 and dv > 0 and all(density.subs({c: cv}).subs(x, dv * sympy.Rational(i, 8)) >= 0 for i in range(9)):
            found.append((cv, dv))
    return found


def test_recipe_is_registered_with_the_stated_marks():
    assert TOPIC in dict((t, lvl) for lvl, t in core.topics())
    q = _q(1)
    assert [p.marks for p in q.parts] == [2, 3, 1, 2]
    assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction", "fraction"]
    assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_conformance_is_clean():
    assert recipes.conformance(RECIPE, seeds=300) == []


def test_technology_form_is_refused():
    with pytest.raises(ValueError, match="exact answers only"):
        core.make(TOPIC, seed=1, technology_active=True)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_part_matches_an_independent_derivation(seed):
    q = _q(seed)
    family, density, mean, m = _read(q)
    solutions = _solve(density, mean)
    assert len(solutions) == 1, (q.text, solutions)  # exactly one admissible root
    cv, dv = solutions[0]
    a, b, cc, pp = q.parts
    # (a): the stated expression for c satisfies area = 1 identically in d
    rel = sympy.sympify(a.answer.replace("^", "**"), locals={"w": d})
    assert sympy.simplify(sympy.integrate(density.subs(c, rel), (x, 0, d)) - 1) == 0
    assert _frac(b.answer_text) == Fraction(int(dv.p), int(dv.q))
    assert _frac(cc.answer_text) == Fraction(int(cv.p), int(cv.q))
    tail = 1 - sympy.integrate(density.subs(c, cv), (x, 0, m))
    assert _frac(pp.answer_text) == Fraction(int(tail.p), int(tail.q))
    assert 0 < tail < 1 and m < dv


def test_linear_family_second_root_is_rejected_and_offered_as_a_distractor():
    seen = 0
    for seed in range(300):
        q = _q(seed)
        family, density, mean, _ = _read(q)
        if family != "linear":
            continue
        k = density.subs({c: 0, x: 0})
        # E(X) = 2d/3 - k d^2/6, i.e. k d^2 - 4 d + 6 E = 0; roots (2 +- sqrt(4 - 6 k E)) / k
        root = sympy.sqrt(4 - 6 * k * mean)
        low, high = (2 - root) / k, (2 + root) / k
        assert Fraction(str(low)) == _frac(q.parts[1].answer_text) if low.is_Rational else True
        assert 2 * (1 - k * high) / high**2 < 0  # the other root gives a negative c
        assert (
            q.parts[1].option_notes.get(str(high.p) if high.q == 1 else f"{high.p}/{high.q}")
            == "tpp-extraneous-root-kept"
        )
        seen += 1
    assert seen > 50


def test_both_families_and_all_powers_occur():
    shapes = {re.sub(r"\d+/\d+", "b", re.search(r"f\(x\) = ([^f]*?) for", _q(s).stem).group(1)) for s in range(200)}
    assert {"kx", "kx^2", "kx^3", "kx + b"} <= shapes


@pytest.mark.parametrize("seed", range(60))
def test_canonical_answer_earns_full_marks_and_every_distractor_none(seed):
    q = _q(seed)
    result = grade("expression", q, q.answer_text)
    assert result.correct and result.marks == result.max_marks == 8
    for part in q.parts:
        assert len(part.distractors) >= 2
        for text in part.distractors:
            assert not grade(part.answer_kind, part, text).correct, (part.label, text)
            assert part.option_notes[text] in CATALOGUE


def test_parts_do_not_leak_later_answers():
    for seed in range(100):
        q = _q(seed)
        seen = q.stem
        for part in q.parts[1:]:
            seen += " " + q.parts[q.parts.index(part) - 1].prompt
            assert (
                not re.search(rf"(?<![\w.]){re.escape(part.answer_text)}(?![\w.])", seen) or len(part.answer_text) < 3
            )


def _reply(q, **override):
    base = {p.label: p.answer_text for p in q.parts}
    base.update(override)
    return base


def test_follow_through_from_a_wrong_relation_credits_c_and_the_probability():
    checked = 0
    for seed in range(120):
        q = _q(seed)
        a, b, cc, pp = q.parts
        wrong = a.distractors[0]
        dv = sympy.Rational(b.answer_text)
        c_wrong = sympy.sympify(wrong.replace("^", "**"), locals={"w": d}).subs(d, dv)
        family, density, mean, m = _read(q)
        p_wrong = 1 - sympy.integrate(density.subs(c, c_wrong), (x, 0, m))
        if not (c_wrong > 0 and 0 < p_wrong < 1):
            continue
        result = grade("expression", q, _reply(q, a=wrong, c=str(c_wrong), d=str(p_wrong)))
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 3
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 2
        checked += 1
    assert checked >= 20


def test_follow_through_from_a_wrong_d_credits_c():
    checked = 0
    for seed in range(120):
        q = _q(seed)
        a, b, cc, pp = q.parts
        rel = sympy.sympify(a.answer.replace("^", "**"), locals={"w": d})
        wrong_d = sympy.Rational(b.distractors[0])
        c_wrong = rel.subs(d, wrong_d)
        if c_wrong <= 0:
            continue  # a negative c is no sensible carry
        result = grade("expression", q, _reply(q, b=str(wrong_d), c=str(c_wrong)))
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        checked += 1
    assert checked >= 20


def test_a_wrong_c_is_not_rescued_when_the_earlier_parts_were_right():
    q = _q(3)
    wrong_c = q.parts[2].distractors[0]
    result = grade("expression", q, _reply(q, c=wrong_c))
    assert result.parts["c"].marks == 0
