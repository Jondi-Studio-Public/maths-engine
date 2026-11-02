"""The even_odd recipe subject: conformance, plus independent oracles for each ask over many seeds."""

import re

import pytest
from sympy import Symbol, cos, pi, simplify, sin, sympify

from mathgen import core
from mathgen.generators import _even_odd_maths as m
from mathgen.grading import grade
from mathgen.recipes import conformance

RECIPE = "methods_even_odd_1"
TOPIC = "recipe_" + RECIPE
SEEDS = range(120)


X = Symbol("x", real=True)


def parse(t):
    """Read the stem's plain-text function back into SymPy, independently of the generator's own objects."""
    t = t.translate(str.maketrans("²³⁴⁵", "2345"))
    t = re.sub(r"x([2-5])", r"x**\1", t)
    t = re.sub(r"(\d)(x|sin|cos)", r"\1*\2", t)
    t = re.sub(r"^-(sin|cos|x)", r"-1*\1", t)
    return sympify(t, locals={"x": X, "sin": sin, "cos": cos})


def test_recipe_conforms():
    assert conformance(RECIPE) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_every_part_has_three_distractors_and_the_key_grades(seed):
    q = core.make(TOPIC, seed=seed)
    assert [p.label for p in q.parts] == list("abcde")
    for p in q.parts:
        assert len(p.distractors) == 3
        assert p.answer_text not in [str(d) for d in p.distractors]
    assert grade("expression", q, {p.label: p.answer_text for p in q.parts}).correct


@pytest.mark.parametrize("seed", SEEDS)
def test_parity_answers_match_an_independent_numeric_oracle(seed):
    q = core.make(TOPIC, seed=seed)
    p_txt, q_txt = re.search(r"p\(x\) = (.+?) and q\(x\) = (.+?)\. Let", q.stem).groups()
    pe, qe = parse(p_txt), parse(q_txt)
    word = {m.EVEN: "Even", m.ODD: "Odd", m.NEITHER: "Neither even nor odd"}
    assert q.parts[0].answer_text == word[m.numeric_parity(pe + qe)]
    assert q.parts[1].answer_text == word[m.numeric_parity(pe * qe)]


@pytest.mark.parametrize("seed", SEEDS)
def test_find_c_and_evaluation_against_symbolic_check(seed):
    q = core.make(TOPIC, seed=seed)
    a_txt, trig_name, b_txt = re.search(r"h\(x\) = (.+?) \+ (sin|cos)\((?:(\d)x|x) \+ c\)", q.stem).groups()
    a = parse(a_txt)
    b = int(b_txt) if b_txt else 1
    trig = sin if trig_name == "sin" else cos
    target = m.EVEN if "h is even" in q.parts[3].prompt else m.ODD
    fam = q.parts[3].answer_text
    c0 = pi / 2 if fam.startswith("pi/2") else 0
    for k in (-1, 0, 2):
        assert m.numeric_parity(a + trig(b * X + c0 + k * pi)) == target
    for off in (pi / 3, pi / 4):
        assert m.numeric_parity(a + trig(b * X + c0 + off)) == m.NEITHER
    # the family is the one for that parity, not just some family
    assert m.numeric_parity(a) == target
    j_txt = re.search(r"find h\((\d?)π\)", q.parts[4].prompt).group(1)
    j = int(j_txt) if j_txt else 1
    c_small = pi if c0 == 0 else pi / 2
    value = a.subs(X, j * pi) + trig(b * j * pi + c_small)
    assert simplify(sympify(q.parts[4].answer_text.replace("^", "**"), locals={"pi": pi}) - value) == 0
