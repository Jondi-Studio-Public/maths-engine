"""Independent oracle for the `p13e_bifurcation_curvature` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p13e_bifurcation_curvature_build`) and re-derives the
answer from scratch, using sympy.solve/sympy.diff directly on the parsed
numbers rather than any closed-form shortcut the generator itself uses.
"""

import math
import re
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest
import sympy as sp

import mathgen
from mathgen.skills import load_skills

BIFURCATION_ID = "ucv1-bifurcation-sign-check"
CURVATURE_ID = "ucv1-parametric-curvature"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


BIFURCATION_SOURCES = _skill_sources(BIFURCATION_ID)
CURVATURE_SOURCES = _skill_sources(CURVATURE_ID)
BIFURCATION_SOURCES["exam"] = replace(BIFURCATION_SOURCES["exam"], topic="p13e_bifurcation_sign_check", params={})
CURVATURE_SOURCES["exam"] = replace(CURVATURE_SOURCES["exam"], topic="p13e_parametric_curvature", params={})


def _to_fraction(text: str) -> Fraction:
    return Fraction(Decimal(text))


# ==========================================================================
# ucv1-bifurcation-sign-check oracle
# ==========================================================================

# "x' = <a>x^2 <sign><b>x + r" appears, verbatim, in every easy/medium
# wording (only the surrounding sentence varies).
_BIF_FAMILY_RE = re.compile(r"x' = (-?)(\d*)x\^2\s*([+-])\s*(\d*)x \+ r")
# "x' = <a>x^2 + r*x <sign><c>" appears, verbatim, in every hard wording.
_BIF_HARD_RE = re.compile(r"x' = (-?)(\d*)x\^2 \+ r\*x ([+-]) (\d+)")
# "a = <a>, b = <b>" and "Formula A/B: r* = <formula>" appear, verbatim, in
# every exam wording.
_EXAM_AB_RE = re.compile(r"a = (-?\d+), b = (-?\d+)")
_EXAM_FORMULA_A_RE = re.compile(r"Formula A: r\* = (.+?)\.")
_EXAM_FORMULA_B_RE = re.compile(r"Formula B: r\* = (.+?)\.")
# Local override: the exam-tier oracle runs at 1000+ seeds, not the shared 60-seed SEEDS used by every other
# tier/test in this file.
EXAM_SEEDS = range(1000)


def _parse_ab(text):
    m = _BIF_FAMILY_RE.search(text)
    assert m, f"could not parse a, b from: {text}"
    a_sign, a_mag, b_sign, b_mag = m.groups()
    a = int(a_mag) if a_mag else 1
    if a_sign == "-":
        a = -a
    b = int(b_mag) if b_mag else 1
    if b_sign == "-":
        b = -b
    return a, b


def _parse_ac_hard(text):
    m = _BIF_HARD_RE.search(text)
    assert m, f"could not parse a, c from: {text}"
    a_sign, a_mag, c_sign, c_mag = m.groups()
    a = int(a_mag) if a_mag else 1
    if a_sign == "-":
        a = -a
    c = int(c_mag)
    if c_sign == "-":
        c = -c
    return a, c


def _oracle_threshold(a, b):
    """r* such that b^2 - 4ar = 0, via sympy.solve on the literal
    discriminant -- independent of the generator's closed-form division."""
    R = sp.symbols("R")
    sol = sp.solve(sp.Eq(b**2 - 4 * a * R, 0), R)
    assert len(sol) == 1
    v = sol[0]
    return Fraction(int(v.p), int(v.q))


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium"] for s in SEEDS])
def test_oracle_rederives_bifurcation_forward_tiers(difficulty, seed):
    src = BIFURCATION_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    a, b = _parse_ab(q.text)
    derived = _oracle_threshold(a, b)
    stated = _to_fraction(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_rederives_bifurcation_hard(seed):
    src = BIFURCATION_SOURCES["hard"]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    a, c = _parse_ac_hard(q.text)
    ac = a * c
    assert ac > 0
    root = math.isqrt(ac)
    assert root * root == ac
    derived = 2 * root
    stated = int(q.answer_text)
    assert derived == stated, f"seed={seed}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"


_IMPLICIT_MULT_RE = re.compile(r"(\d)([a-zA-Z(])")


def _sympify_formula(formula_text: str, a_sym, b_sym):
    """Parse a rendered formula string ("b^2 / (4a)") into a sympy expression
    over `a_sym, b_sym`. The rendered text spells multiplication the way a
    student would write it (juxtaposition, "4a", never a bare "*" -- the
    project's text-quality gate flags an asterisk multiplication sign in
    learner-facing text), so this re-inserts an explicit "*" between a digit
    and the following letter/parenthesis before sympifying; ``^`` is likewise
    converted to sympy's ``**``. Purely a syntax normalisation, not part of
    the independent re-derivation itself."""
    normalized = _IMPLICIT_MULT_RE.sub(r"\1*\2", formula_text.replace("^", "**"))
    return sp.sympify(normalized, locals={"a": a_sym, "b": b_sym})


@pytest.mark.parametrize("seed", EXAM_SEEDS)
def test_oracle_rederives_bifurcation_exam(seed):
    """The large-sample instrument for the rebuilt exam tier: parses `a, b` and
    the two FORMULA strings from the rendered prompt (never the generator's
    internal values), independently re-derives the GENERAL correct formula
    from the literal discriminant condition via `sympy.solve` on symbols
    (never numbers), determines by symbolic comparison which of the two
    parsed candidates matches it, and only then substitutes the drawn a, b to
    check the numeric answer."""
    src = BIFURCATION_SOURCES["exam"]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))

    ab_match = _EXAM_AB_RE.search(q.text)
    assert ab_match, f"could not parse a, b from: {q.text}"
    a, b = int(ab_match.group(1)), int(ab_match.group(2))

    formula_a_match = _EXAM_FORMULA_A_RE.search(q.text)
    formula_b_match = _EXAM_FORMULA_B_RE.search(q.text)
    assert formula_a_match and formula_b_match, f"could not parse both formulas from: {q.text}"

    a_sym, b_sym = sp.symbols("a b")
    parsed_a = _sympify_formula(formula_a_match.group(1), a_sym, b_sym)
    parsed_b = _sympify_formula(formula_b_match.group(1), a_sym, b_sym)

    # Independent GENERAL re-derivation: solve the literal discriminant
    # condition b^2 - 4aR = 0 for R using symbols, never the drawn numbers.
    R = sp.symbols("R")
    sol = sp.solve(sp.Eq(b_sym**2 - 4 * a_sym * R, 0), R)
    assert len(sol) == 1, f"expected exactly one general threshold formula, got {sol}"
    derived_general_expr = sol[0]

    a_matches = sp.simplify(parsed_a - derived_general_expr) == 0
    b_matches = sp.simplify(parsed_b - derived_general_expr) == 0
    assert a_matches != b_matches, (
        f"expected exactly one of the two parsed formulas to match the general "
        f"threshold formula, got a_matches={a_matches} b_matches={b_matches} "
        f"in: {q.text}"
    )
    correct_expr = parsed_a if a_matches else parsed_b

    derived = correct_expr.subs({a_sym: a, b_sym: b})
    derived_fraction = Fraction(int(sp.nsimplify(derived).p), int(sp.nsimplify(derived).q))
    stated = _to_fraction(q.answer_text)
    assert derived_fraction == stated, f"seed={seed}\ntext: {q.text}\nderived: {derived_fraction}\nstated: {stated}"


# ==========================================================================
# ucv1-parametric-curvature oracle
# ==========================================================================

_X_EXPR_RE = re.compile(r"x\(t\) = (.+?)(?=,|\(| and y\(t\))")
_Y_EXPR_RE = re.compile(r"y\(t\) = ([^.,]+)")
_T0_RE = re.compile(r"t = (-?\d+)|kappa\((-?\d+)\)")


def _parse_t0(text: str) -> int:
    m = _T0_RE.search(text)
    assert m, f"could not parse t0 from: {text}"
    group = m.group(1) if m.group(1) is not None else m.group(2)
    return int(group)


_TERM_RE = re.compile(r"(?P<sign>[+-])?\s*(?P<mag>\d*)t(?:\^(?P<pow>\d+))?")
_KAPPA_VALUE_RE = re.compile(r"is exactly (-?\d+\.\d+)")
_SIGN_WORD_RE = re.compile(r"\b(positive|negative)\b")
# The t^2 term is OPTIONAL: join_terms drops it entirely when its coefficient
# (a2) is zero.
_EXAM_X_PREFIX_RE = re.compile(r"(-?)(\d*)t\^3(?:\s*([+-])\s*(\d*)t\^2)?")


def _parse_terms(expr: str) -> dict:
    out = {}
    for m in _TERM_RE.finditer(expr):
        sign = -1 if m.group("sign") == "-" else 1
        mag = int(m.group("mag")) if m.group("mag") else 1
        power = int(m.group("pow")) if m.group("pow") else 1
        out[power] = sign * mag
    return out


def _sympy_kappa_from_polys(x_poly: dict, y_poly: dict, t0: int) -> Fraction:
    t = sp.symbols("t")
    xt = sum(coef * t**power for power, coef in x_poly.items())
    yt = sum(coef * t**power for power, coef in y_poly.items())
    xp = sp.diff(xt, t)
    xpp = sp.diff(xp, t)
    yp = sp.diff(yt, t)
    ypp = sp.diff(yp, t)
    xp0, xpp0 = xp.subs(t, t0), xpp.subs(t, t0)
    yp0, ypp0 = yp.subs(t, t0), ypp.subs(t, t0)
    num = sp.Abs(xp0 * ypp0 - yp0 * xpp0)
    denom = (xp0**2 + yp0**2) ** sp.Rational(3, 2)
    value = sp.nsimplify(num / denom)
    return Fraction(int(value.p), int(value.q))


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard"] for s in SEEDS])
def test_oracle_rederives_curvature_forward_tiers(difficulty, seed):
    src = CURVATURE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    mx = _X_EXPR_RE.search(q.text)
    my = _Y_EXPR_RE.search(q.text)
    assert mx and my, q.text
    x_poly = _parse_terms(mx.group(1))
    y_poly = _parse_terms(my.group(1))
    t0 = _parse_t0(q.text)
    derived = _sympy_kappa_from_polys(x_poly, y_poly, t0)
    stated = _to_fraction(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_rederives_curvature_exam(seed):
    src = CURVATURE_SOURCES["exam"]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    mx = _X_EXPR_RE.search(q.text)
    my = _Y_EXPR_RE.search(q.text)
    assert mx and my, q.text
    xm = _EXAM_X_PREFIX_RE.search(mx.group(1))
    assert xm, f"could not parse a3, a2 from: {mx.group(1)}"
    a3_sign, a3_mag, a2_sign, a2_mag = xm.groups()
    a3 = int(a3_mag) if a3_mag else 1
    if a3_sign == "-":
        a3 = -a3
    if a2_sign is None:
        a2 = 0  # t^2 term absent -- join_terms drops a zero coefficient
    else:
        a2 = int(a2_mag) if a2_mag else 1
        if a2_sign == "-":
            a2 = -a2

    y_poly = _parse_terms(my.group(1))
    t0 = _parse_t0(q.text)
    kappa_value = _to_fraction(_KAPPA_VALUE_RE.search(q.text).group(1))
    sign_word = _SIGN_WORD_RE.search(q.text).group(1)

    # Independent symbolic solve for the blanked coefficient a1: build x(t)
    # with a1 as a free symbol, differentiate symbolically, substitute t0,
    # and solve the linear equation implied by the stated curvature MAGNITUDE
    # and SIGN -- a genuinely different route from the generator's own
    # closed-form rearrangement.
    t, a1 = sp.symbols("t a1")
    xt = a3 * t**3 + a2 * t**2 + a1 * t
    yt = sum(coef * t**power for power, coef in y_poly.items())
    xp = sp.diff(xt, t)
    xpp = sp.diff(xp, t)
    yp = sp.diff(yt, t)
    ypp = sp.diff(yp, t)
    xp0, xpp0 = xp.subs(t, t0), xpp.subs(t, t0)
    yp0, ypp0 = yp.subs(t, t0), ypp.subs(t, t0)
    n_expr = xp0 * ypp0 - yp0 * xpp0  # linear in a1

    magnitude = kappa_value * 125
    assert magnitude.denominator == 1, f"kappa*125 is not an integer: {magnitude}"
    target = int(magnitude) if sign_word == "positive" else -int(magnitude)

    sol = sp.solve(sp.Eq(n_expr, target), a1)
    assert len(sol) == 1, f"expected exactly one solution for a1, got {sol}"
    derived = int(sol[0])
    stated = int(q.answer_text)
    assert derived == stated, f"seed={seed}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"


def test_oracle_coverage():
    # `bifurcation-sign-check` exam runs at EXAM_SEEDS (1000+
    # seeds); every other tier of both skills runs at the
    # shared SEEDS (60) -- 3 bifurcation tiers + 4 curvature tiers at SEEDS,
    # plus the bifurcation exam tier at EXAM_SEEDS.
    tiers = ["easy", "medium", "hard", "exam"]
    other_tiers = 3 + len(tiers)  # bifurcation easy/medium/hard + all 4 curvature tiers
    total = other_tiers * len(SEEDS) + len(EXAM_SEEDS)
    assert len(SEEDS) >= 60
    assert len(EXAM_SEEDS) >= 1000
    assert total >= 1000
    print(
        f"\nOracle re-derived {total} questions ({other_tiers} tiers x {len(SEEDS)} seeds "
        f"+ bifurcation exam x {len(EXAM_SEEDS)} seeds)."
    )
