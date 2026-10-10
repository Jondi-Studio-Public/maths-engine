"""Independent oracle for `rearrange_powers_roots`, rearranging formulas with powers and roots for a stated
variable.

The topic started with 10 (formula, target) cases and now has **7**: three of them (`E = mc²` for `c`,
`P = I²R` for `I`, `y = a(x-h)²+k` for `x`) were found to have a real grading-gap defect -- their canonical
answer's natural alternative form is rejected by the real grader (see the generator module's docstring for the
full writeup) -- and were removed. This oracle no longer knows those three formulas at all.

Parses the formula and target variable out of `question.text` -- the same
"Make X the subject of the formula ..." shape the existing `rearrange` topic
uses -- and re-derives the rearrangement with `sympy.solve` called directly
here (never imported from `mathgen.generators.p8b_rearrange_powers_build`),
then asserts the parsed `question.answer_text` is symbolically identical to
the oracle's own derivation.

Using `sympy.solve` here too is allowed: the independence requirement is
"does not import the generator's module/helpers", not "may not use the same
library".

Symbol-assumption note: every oracle symbol below is declared
`positive=True`, including the SUVAT letters (`s`, `u`, `a`) that are signed
in general (a SUVAT problem can have negative displacement/velocity/
acceleration). This is a deliberate simplification for STRING-matching
purposes, not a physical claim: the same literal letter (e.g. "a") is reused
across two unrelated formulas (Pythagoras's leg and SUVAT's acceleration),
and `_LOCALS` can only bind one Python object per letter for `parse_expr` to
use, so every formula that shares a letter must agree on ONE assumption for
it. For every shared letter remaining here (SUVAT's `s`/`u`/`a`), the letter
appears only LINEARLY in the answer (never under a root or absolute value),
so the choice of assumption cannot change the algebraic form `solve()`
returns -- verified empirically below in comments -- so using
`positive=True` uniformly is safe for the equality checks this file
performs, even though it is not the most physically honest assumption for a
SUVAT quantity.
"""

import re

import pytest
from sympy import Eq, Rational, pi, solve, sqrt, symbols, simplify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

SEEDS = range(60)

T, l, g, A, r = symbols("T l g A r", positive=True)
V = symbols("V", positive=True)
a, b, c = symbols("a b c", positive=True)
s, u, t = symbols("s u t", positive=True)

# `answer_text` uses learner-facing notation ("T^2g/(4pi^2)": a caret for the power, a coefficient
# run directly against the following symbol/paren with no operator at all). `_to_ascii` is kept as a
# harmless no-op normalisation step (nothing left for it to substitute) so this oracle still works
# unchanged if a display alias is re-introduced later. Parse with `convert_xor` +
# `implicit_multiplication` against a `locals=` map naming the oracle's own `positive=True` symbols
# above -- without it, `sympify`'s fresh, assumption-free "T"/"g"/... would not equal these (same
# string, different sympy object) and `simplify(a - b)` could never cancel them.
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_LOCALS = {
    "T": T,
    "l": l,
    "g": g,
    "A": A,
    "r": r,
    "pi": pi,
    "V": V,
    "a": a,
    "b": b,
    "c": c,
    "s": s,
    "u": u,
    "t": t,
}


def _to_ascii(text: str) -> str:
    return text.replace("√(", "sqrt(").replace("π", "pi")


def _parse_answer(text: str):
    ascii_text = _to_ascii(text).replace("^", "**")
    return parse_expr(ascii_text, local_dict=_LOCALS, transformations=_TRANSFORMATIONS)


# Every distinct formula text this topic can render, and how to reconstruct
# its equation independently. `l`/`g` share one formula text (two targets);
# `a`(suvat)/`u` share another; `r` is a target of two UNRELATED formulas
# (the circle and the sphere), so the oracle's own equation lookup is keyed
# on formula text first, exactly as `_parse_question_text` already extracts
# it from the rendered question -- never on target letter alone.
def _oracle_expr(formula_text: str, target_name: str):
    """Re-derive the rearrangement for one of the five known formula texts,
    independently of the generator's own symbol table or case list."""
    if formula_text == "A = πr²":
        eq = Eq(A, pi * r**2)
        target = {"r": r}[target_name]
    elif formula_text == "T = 2π√(l/g)":
        eq = Eq(T, 2 * pi * sqrt(l / g))
        target = {"l": l, "g": g}[target_name]
    elif formula_text == "V = (4/3)πr³":
        eq = Eq(V, Rational(4, 3) * pi * r**3)
        target = {"r": r}[target_name]
    elif formula_text == "c = √(a² + b²)":
        eq = Eq(c, sqrt(a**2 + b**2))
        target = {"a": a}[target_name]
    elif formula_text == "s = ut + ½at²":
        eq = Eq(s, u * t + Rational(1, 2) * a * t**2)
        target = {"a": a, "u": u}[target_name]
    else:
        raise AssertionError(f"unrecognised formula text: {formula_text!r}")
    solutions = solve(eq, target)
    if len(solutions) == 1:
        return solutions[0]
    raise AssertionError(f"unexpected solution count for {formula_text}/{target_name}: {solutions}")


_TEXT_RE = re.compile(r"^Make (?P<target>\w+) the subject of the formula (?P<formula>.+?)(?: \(.+\))?$")


def _parse_question_text(text: str):
    m = _TEXT_RE.match(text)
    assert m, f"could not parse question text: {text!r}"
    return m.group("target"), m.group("formula")


def test_oracle_matches_generator_answer_for_every_seed():
    checked = 0
    seen_pairs = set()
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        target_name, formula_text = _parse_question_text(q.text)
        seen_pairs.add((formula_text, target_name))

        oracle_expr = _oracle_expr(formula_text, target_name)
        parsed_answer = _parse_answer(q.answer_text)

        assert simplify(oracle_expr - parsed_answer) == 0, (q.text, q.answer_text, oracle_expr)
        checked += 1

    assert checked == len(SEEDS)
    # All seven (formula, target) cases must appear across 60 seeds -- the
    # generator's stated pool size (see
    # test_p8b_rearrange_powers_build.py's test_pool_size_is_at_least_seven).
    # Tracked as (formula, target) PAIRS, not target letters alone, because
    # several letters now name physically different quantities across
    # formulas ("r" is both the circle's and the sphere's radius; "a" is
    # both Pythagoras's leg and SUVAT's acceleration).
    assert seen_pairs == {
        ("A = πr²", "r"),
        ("T = 2π√(l/g)", "l"),
        ("T = 2π√(l/g)", "g"),
        ("V = (4/3)πr³", "r"),
        ("c = √(a² + b²)", "a"),
        ("s = ut + ½at²", "a"),
        ("s = ut + ½at²", "u"),
    }


def test_oracle_r_answer_is_positive_for_sample_positive_a_values():
    """The circle's `r` case's whole point is the physical positive-root constraint, stated in the question text.
    Confirm the parsed answer numerically substitutes to a positive value across several sample A draws,
    matching an oracle solve constrained the same way (r declared positive)."""
    checked = False
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        target_name, formula_text = _parse_question_text(q.text)
        if formula_text != "A = πr²":
            continue
        checked = True
        oracle_expr = _oracle_expr(formula_text, target_name)
        parsed_answer = _parse_answer(q.answer_text)
        for a_value in (1, 4, 9, 50, 137):
            oracle_value = float(oracle_expr.subs(A, a_value))
            answer_value = float(parsed_answer.subs(A, a_value))
            assert oracle_value > 0
            assert answer_value > 0
            assert oracle_value == pytest.approx(answer_value)
        break
    assert checked, "no seed in range drew the circle r-target case"


def test_oracle_sphere_r_answer_is_positive_for_sample_positive_v_values():
    """'s odd-power case: same discipline as the circle's `r` case above, but for the sphere's volume formula
    (target `r`, free symbol `V`), confirming the two formulas that share a target letter are not confused
    with each other."""
    checked = False
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        target_name, formula_text = _parse_question_text(q.text)
        if formula_text != "V = (4/3)πr³":
            continue
        checked = True
        oracle_expr = _oracle_expr(formula_text, target_name)
        parsed_answer = _parse_answer(q.answer_text)
        for v_value in (1, 4, 9, 50, 137):
            oracle_value = float(oracle_expr.subs(V, v_value))
            answer_value = float(parsed_answer.subs(V, v_value))
            assert oracle_value > 0
            assert answer_value > 0
            assert oracle_value == pytest.approx(answer_value)
        break
    assert checked, "no seed in range drew the sphere r-target case"
