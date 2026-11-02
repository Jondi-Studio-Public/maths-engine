"""The `exact_binomial` recipe subject: R ~ Bin(n, p) with exact fraction and surd answers.

Each ask is recomputed here from the rendered stem and prompts with SymPy's own binomial distribution, a route that
shares no code with the generator's `math.comb` arithmetic.
"""

import re
from fractions import Fraction

import pytest
import sympy
from sympy.stats import Binomial, P, density, std

from mathgen import core, recipes
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE

RECIPE = "methods_exact_binomial_1"
TOPIC = "recipe_" + RECIPE


def _q(seed):
    return core.make(TOPIC, seed=seed)


def _dist(q):
    n = int(re.search(r"n = (\d+)", q.stem).group(1))
    p = sympy.Rational(re.search(r"p = (\d+/\d+)", q.stem).group(1))
    return n, p, Binomial("R", n, p)


def test_recipe_is_registered_with_the_stated_marks():
    q = _q(1)
    assert [p.marks for p in q.parts] == [1, 2, 3]
    assert [p.answer_kind for p in q.parts] == ["fraction", "expression", "fraction"]
    assert [p.answer_form for p in q.parts] == [None, "simplified_surd", None]
    assert all(p.follow_through is None for p in q.parts)


def test_conformance_is_clean():
    assert recipes.conformance(RECIPE, seeds=300) == []


def test_technology_form_is_refused():
    with pytest.raises(ValueError, match="exact answers only"):
        core.make(TOPIC, seed=1, technology_active=True)


@pytest.mark.parametrize("seed", range(150))
def test_every_part_matches_sympy_stats(seed):
    q = _q(seed)
    n, p, rv = _dist(q)
    assert 2 <= n <= 5 and p.q <= 6
    a, b, c = q.parts
    k = int(re.search(r"R = (\d+)", a.prompt).group(1))
    assert Fraction(a.answer_text) == Fraction(str(density(rv)(k)))
    sd = std(rv)
    assert sympy.simplify(sympy.sympify(b.answer_text) - sd) == 0
    assert not sd.is_Rational  # always a genuine surd
    hi, lo = (int(v) for v in re.search(r"R ≥ (\d+) \| R ≥ (\d+)", c.prompt).groups())
    cond = P(rv >= hi, rv >= lo)
    assert Fraction(c.answer_text) == Fraction(str(sympy.nsimplify(cond)))
    assert 0 < Fraction(c.answer_text) < 1


@pytest.mark.parametrize("seed", range(60))
def test_canonical_earns_full_marks_and_every_distractor_none(seed):
    q = _q(seed)
    result = grade("fraction", q, q.answer_text)
    assert result.correct and result.marks == result.max_marks == 6
    for part in q.parts:
        assert len(part.distractors) >= 2
        for text in part.distractors:
            assert not grade(part.answer_kind, part, text).correct, (part.label, text)
            assert part.option_notes[text] in CATALOGUE


def test_the_unsimplified_surd_is_refused():
    for seed in range(60):
        part = _q(seed).parts[1]
        value = sympy.sympify(part.answer_text)
        radicand = sympy.nsimplify(value**2)
        if radicand.q != 1 and (radicand.p * radicand.q**2) != radicand.p:
            reply = f"sqrt({radicand.p * 4 * radicand.q**2})/{2 * radicand.q}"  # sqrt(4m)/(2q) with an unsimplified radicand
            assert grade("expression", part, reply).correct is False
            return
    pytest.fail("no question exercised the unsimplified surd")


def test_conditional_part_is_the_ratio_of_tails_not_the_numerator_alone():
    for seed in range(40):
        q = _q(seed)
        c = q.parts[2]
        assert c.option_notes[c.distractors[0]] in CATALOGUE
        assert Fraction(c.distractors[0]) != Fraction(c.answer_text)


def test_mean_and_variance_slips_are_tagged_for_the_standard_deviation():
    tags = {t for s in range(120) for t in _q(s).parts[1].option_notes.values()}
    assert tags <= {"dd-sd-gives-the-variance", "dd-sd-gives-the-mean", "dd-sd-root-of-the-mean"}
    assert "dd-sd-gives-the-variance" in tags
