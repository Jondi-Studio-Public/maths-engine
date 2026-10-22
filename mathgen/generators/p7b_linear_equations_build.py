"""Generates `solve_linear_equation_fraction_both_sides` and `identify_and_solve_linear_equation` questions on
solving linear equations.

The linear-equations boundary pair.

``senior-solve-linear-equations`` (ACMMM005) and ``senior-identify-and-solve-linear-equations`` (ACMGM038) are
0.75-similar by a duplicate-candidate similarity check (same name vocabulary by design). The distinction
between them is the deliverable, not just the generators:

- ``senior-solve-linear-equations`` is a 100% algebraic SOLVE. Every question
  it produces is a genuine, already-known-linear equation; the generator
  never emits a non-linear equation, and it never asks the learner to judge
  linearity.
- ``senior-identify-and-solve-linear-equations`` is a CLASSIFICATION. Every
  question presents a one-variable equation that is sometimes linear and
  sometimes not, and the learner must decide which before (optionally)
  solving. The two skills never overlap in what they ask the learner to do.

- ``solve_linear_equation_fraction_both_sides`` (senior-solve-linear-equations,
  ACMMM005, Mathematical Methods) -- scope:
  "Solve a linear equation in one variable whose coefficients include
  algebraic fractions, with the unknown appearing on both sides." Concretely:
  every question has (a) at least one coefficient written as an algebraic
  fraction whose numerator contains the variable, and (b) the unknown
  appearing in more than one term of the equation. This is genuinely new
  content: none of ``skills/equations/one-step-equations.yaml``,
  ``two-step-equations.yaml``, ``equations-with-brackets.yaml`` or
  ``variables-on-both-sides.yaml`` names a fraction-coefficient case in its
  own description. ``response_family: scalar_expression``,
  ``answer: fraction`` (canonical is the bare fraction text of the solved
  ``x``, e.g. ``"7/4"``, ``"-3"`` -- no ``"x = "`` label, matching
  ``p3a_number_equations_build.py::unit_fractions_combine_whole``'s
  precedent; the question text itself says "Solve for x" so the bare
  fraction is unambiguous).

  Construction, every tier: pick the surface coefficients as small integers,
  then derive the solution ``x0`` by hand ``Fraction`` arithmetic from
  clearing the denominator(s) -- a route that never touches ``sympy`` -- and
  verify independently with ``sympy.solve`` before returning
  (``check_equivalent``). This is the same guarantee "construct so the
  algebra is guaranteed solvable, verify with an independent route",
  reordered: backward-from-``x0`` needs solving a second
  intermediate equation for one of the surface coefficients at medium/hard
  (two and three fraction terms), which is more error-prone to hand-derive
  than simply picking the coefficients directly and computing ``x0`` once by
  clearing denominators -- the ``sympy.solve`` cross-check is still a
  genuinely independent second route, since it never sees the hand-Fraction
arithmetic. Disclosed.

  - ``easy``: one algebraic-fraction term, unknown on both sides, one other
    plain term: ``(x + a)/p = x + b``, ``p`` in ``{2, 3, 4}``. ``x0 = (p*b -
    a) / (1 - p)``; non-integer whenever ``(1 - p)`` does not divide
    ``(p*b - a)`` evenly (always true at ``p=2``, so the non-integer share
    comes entirely from ``p in {3, 4}``, which is a large fraction of the
    seeds since ``p`` is drawn uniformly from the three).
  - ``medium``: two algebraic-fraction terms with different denominators,
    unknown across both: ``(x + a)/p - (b*x + c)/q = d`` (the exact shape
    from the scope above). ``x0 = (p*q*d - q*a + p*c) / (q - p*b)``. The
    escalation is a genuine second fraction term with its own denominator and
    its own ``x``-coefficient, not bigger numbers.
  - ``hard``: three algebraic-fraction terms, denominators throughout:
    ``(2x - a)/3 + (x + b)/5 = (x - c)/2 + d``. Clearing to the fixed LCD 30
    gives ``11*x = 10*a - 6*b - 15*c + 30*d``, so ``x0`` has denominator 11
    whenever 11 does not divide the numerator -- non-integer on the large
    majority of seeds. The escalation is collecting across three fraction
    terms and a wider common denominator, not a larger ``x0`` range.

  Misconceptions (each a genuinely different ``Fraction`` value from ``x0``,
  never a "does not simplify" distractor, checked with ``check_distinct``
  every seed unconditionally):

  - ``wrong-side-collection`` (every tier): collected the unknown terms on
    the wrong side when isolating ``x``, flipping the sign of the whole
    coefficient -- the resulting value is exactly ``-x0``. Distinct from
    ``x0`` whenever ``x0 != 0``, enforced by construction (resampled).
  - ``partial-clear-denominator`` (every tier, tier-appropriate
    manifestation): cleared the denominator on only part of the equation.
    At easy this is "distributed the cleared multiplier onto the unknown
    term but not the constant term of the other side" (forgot to multiply
    ``b`` by ``p``). At medium/hard this is literally "cross-multiplied only
    one of the two/three fraction terms and left another with its own
    denominator" -- modelled by substituting the *un-cleared* denominator
    value ``1`` for the term that was skipped in the correct closed-form
    formula.
  - ``sign-error-clearing-denominator`` (medium/hard only -- needs a
    subtraction between two fraction-term numerators, which does not exist
    at easy's single-fraction-term shape, disclosed here rather than forced):
    distributed a negative sign incorrectly across a numerator while
    combining fraction terms over a common denominator, e.g. computed
    ``-p*(b*x + c)`` as ``-p*b*x + p*c`` instead of ``-p*b*x - p*c`` (the
    sign of the ``c`` term is left unflipped).

- ``identify_and_solve_linear_equation`` (senior-identify-and-solve-linear-
  equations, ACMGM038, General Mathematics) -- scope:
  "Classify a given one-variable equation as linear or not
  linear." ``response_family: scalar_expression`` (not the default ``constrained_
  structured``, because the deliverable is a two-way classification, an
  mcq-shaped value with one correct label), ``answer: mcq``. Every question
  presents a one-variable equation, sometimes genuinely linear and sometimes
  not, worded identically regardless of branch so a learner cannot tell in
  advance from the phrasing alone. The four MCQ option shapes: the correct
  answer (``"Linear; x = N"`` or ``"Not linear"``), a wrong-classification
  option (the opposite label), a right-classification-wrong-value option
  (only when the correct label is "Linear"), and -- hard tier only -- a
  surface-pattern-misclassification option (the label a learner would pick
  by pattern-matching the equation's shape before expanding/simplifying).

  Independent check, every branch/tier: build the equation as ``lhs``/``rhs``
  sympy expressions and classify by ``sympy.Poly(expand(lhs - rhs), x)`` --
  catching ``PolynomialError`` for a genuine reciprocal-in-the-unknown shape
  (the unknown in a denominator is never a polynomial expression at all) and
  reading ``.degree()`` otherwise (1 -> linear, verified further against
  ``sympy.solve``; anything else -> not linear). This never reuses the
  construction route.

  - ``easy``: ``a*x + b = c`` (linear) or ``x^2 + a = b`` (obvious squared
    term -- spotting it is immediate, no trap).
  - ``medium``: ``(x + a)/p = c`` (linear -- unknown in the numerator of an
    algebraic fraction) contrasted with ``a/(x + d) = c`` (NOT linear --
    unknown in the denominator, a reciprocal). The reciprocal branch's own
    misconception is ``solved-so-assumed-linear`` (folded into
    ``surface-pattern-misclassification`` below): a learner can correctly
    solve ``a/(x+d)=c`` for a single value of ``x`` and wrongly concludes
    that makes it linear -- solvability is not linearity.
  - ``hard``: ``(x + a)(x - b) - x^2 = c`` (the ``x^2`` terms cancel --
    collapses to genuinely linear, degree exactly 1 since ``a != b`` is
    enforced) contrasted with the minimal-pair sibling ``(x + a)(x - b) +
    x^2 = c`` (the ``x^2`` terms add, staying genuinely degree 2). Same
    bracket shape, opposite sign on the appended ``x^2`` term -- exactly the
    "looks the same, only simplifying tells you which" pair.
    Every hard-tier seed of *both* branches carries a
    ``surface-pattern-misclassification`` distractor (the "ignore the x^2 /
    assume it cancels like the sibling" trap), asserted in the test file
    over the full seed sweep, not merely "possible on some seed."

  Misconceptions (mcq option ids, shared across tiers where the underlying
  error generalises):

  - ``wrong-classification``: stated the opposite label from the true one.
  - ``right-classification-wrong-value``: correctly said "Linear" but with a
    wrong solved value (sign error or a dropped/duplicated term). Only
    applicable when the true label is "Linear" (the "Not linear" label
    carries no value to get wrong).
  - ``surface-pattern-misclassification``: classified from the equation's
    surface shape (a solvable-looking reciprocal, or a bracket pair that
    looks like its sibling) without actually expanding/simplifying or
    checking where the unknown sits.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Eq, Poly, PolynomialError, Rational, solve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    rational_text,
)

x = symbols("x")


# ============================================================================
# SKILL 1: senior-solve-linear-equations
# ============================================================================


def _frac_text(fr: Fraction) -> str:
    return rational_text(Rational(fr.numerator, fr.denominator))


def _sleq_easy(rng):
    # (x + a)/p = x + b -- one fraction term, unknown on both sides.
    while True:
        p = rng.choice([2, 3, 4])
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        b = rng.choice([v for v in range(-6, 7) if v != 0])
        x0 = Fraction(p * b - a, 1 - p)
        if x0 == 0:
            continue
        break

    lhs = (x + a) / p
    rhs = x + b
    solved = solve(Eq(lhs, rhs), x)
    check(len(solved) == 1, f"expected one solution, got {solved}")
    check_equivalent(
        solved[0], Rational(x0.numerator, x0.denominator), "hand-derived x0 must match sympy.solve's independent route"
    )

    wrong_side = Fraction(p * b - a, p - 1)  # = -x0: flipped collection sign
    partial_clear = Fraction(b - a, 1 - p)  # forgot to multiply b by p
    check_distinct(
        Rational(x0.numerator, x0.denominator),
        Rational(wrong_side.numerator, wrong_side.denominator),
        Rational(partial_clear.numerator, partial_clear.denominator),
        msg="every misconception distractor must differ from x0 by value",
    )

    answer_text = _frac_text(x0)
    text = f"Solve for x: ({join_terms([(1, 'x'), (a, '')])})/{p} = {join_terms([(1, 'x'), (b, '')])}"
    distractors = pick_distractors(answer_text, [_frac_text(wrong_side), _frac_text(partial_clear)])
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": x0.numerator, "q": x0.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Clear the denominator: x {('+' if a >= 0 else '-')} {abs(a)} = {p}(x {('+' if b >= 0 else '-')} {abs(b)})",
            f"x = {answer_text}",
        ],
        distractors=distractors,
    )


def _sleq_medium(rng):
    # (x + a)/p - (b*x + c)/q = d -- two fraction terms, different denominators.
    while True:
        p, q = rng.sample([2, 3, 4, 5], 2)
        b = rng.choice([v for v in range(-4, 5) if v != 0])
        if q - p * b == 0:
            continue
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        c = rng.choice([v for v in range(-9, 10) if v != 0])
        d = rng.choice([v for v in range(-6, 7) if v != 0])
        numerator = p * q * d - q * a + p * c
        denom = q - p * b
        if numerator == 0:
            continue
        x0 = Fraction(numerator, denom)

        wrong_side = Fraction(numerator, p * b - q)  # = -x0
        partial_clear = Fraction(p * d - a + p * c, 1 - p * b)  # substitute q=1 (skipped clearing q)
        sign_error = Fraction(p * q * d - q * a - p * c, denom)  # -p*(b*x+c) mis-distributed

        if x0 in (wrong_side, partial_clear, sign_error):
            continue
        break

    lhs = (x + a) / p - (b * x + c) / q
    rhs = d
    solved = solve(Eq(lhs, rhs), x)
    check(len(solved) == 1, f"expected one solution, got {solved}")
    check_equivalent(
        solved[0], Rational(x0.numerator, x0.denominator), "hand-derived x0 must match sympy.solve's independent route"
    )

    x0_r = Rational(x0.numerator, x0.denominator)
    check_distinct(
        x0_r,
        Rational(wrong_side.numerator, wrong_side.denominator),
        Rational(partial_clear.numerator, partial_clear.denominator),
        Rational(sign_error.numerator, sign_error.denominator),
        msg="every misconception distractor must differ from x0 by value",
    )

    answer_text = _frac_text(x0)
    numer2 = join_terms([(b, "x"), (c, "")])
    text = f"Solve for x: ({join_terms([(1, 'x'), (a, '')])})/{p} - ({numer2})/{q} = {d}"
    distractors = pick_distractors(
        answer_text, [_frac_text(wrong_side), _frac_text(partial_clear), _frac_text(sign_error)]
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": x0.numerator, "q": x0.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Clear both denominators (common denominator {p * q}), then collect x.", f"x = {answer_text}"],
        distractors=distractors,
    )


def _sleq_hard(rng):
    # (2x - a)/3 + (x + b)/5 = (x - c)/2 + d -- three fraction terms, LCD 30.
    while True:
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        c = rng.randint(1, 9)
        d = rng.choice([v for v in range(-9, 10) if v != 0])
        numerator = 10 * a - 6 * b - 15 * c + 30 * d
        if numerator == 0:
            continue
        x0 = Fraction(numerator, 11)

        wrong_side = Fraction(numerator, -11)  # = -x0
        partial_clear = Fraction(30 * d - 6 * b + 10 * a - c, 25)  # left the (x-c)/2 term uncleared
        sign_error = Fraction(10 * a - 6 * b + 15 * c + 30 * d, 11)  # -15c mis-distributed to +15c

        if x0 in (wrong_side, partial_clear, sign_error):
            continue
        break

    lhs = (2 * x - a) / 3 + (x + b) / 5
    rhs = (x - c) / 2 + d
    solved = solve(Eq(lhs, rhs), x)
    check(len(solved) == 1, f"expected one solution, got {solved}")
    check_equivalent(
        solved[0], Rational(x0.numerator, x0.denominator), "hand-derived x0 must match sympy.solve's independent route"
    )

    x0_r = Rational(x0.numerator, x0.denominator)
    check_distinct(
        x0_r,
        Rational(wrong_side.numerator, wrong_side.denominator),
        Rational(partial_clear.numerator, partial_clear.denominator),
        Rational(sign_error.numerator, sign_error.denominator),
        msg="every misconception distractor must differ from x0 by value",
    )

    answer_text = _frac_text(x0)
    d_sign = "+" if d >= 0 else "-"
    text = (
        f"Solve for x: ({join_terms([(2, 'x'), (-a, '')])})/3 + ({join_terms([(1, 'x'), (b, '')])})/5 "
        f"= ({join_terms([(1, 'x'), (-c, '')])})/2 {d_sign} {abs(d)}"
    )
    distractors = pick_distractors(
        answer_text, [_frac_text(wrong_side), _frac_text(partial_clear), _frac_text(sign_error)]
    )
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": x0.numerator, "q": x0.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=["Clear all three denominators over the LCD 30, then collect x.", f"x = {answer_text}"],
        distractors=distractors,
    )


@generator(
    "solve_linear_equation_fraction_both_sides",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def solve_linear_equation_fraction_both_sides(rng, tier="easy"):
    """Solve a linear equation in one variable whose coefficients include
    algebraic fractions, with the unknown appearing on both sides."""
    if tier == "easy":
        return _sleq_easy(rng)
    if tier == "medium":
        return _sleq_medium(rng)
    if tier == "hard":
        return _sleq_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ============================================================================
# SKILL 2: senior-identify-and-solve-linear-equations
# ============================================================================


def _classify(lhs, rhs):
    """Independent classification route: build lhs - rhs as a sympy
    expression and read it through sympy.Poly, never reusing whatever
    construction believed the classification to be. Returns
    ("linear", solved_value) or ("not_linear", None)."""
    expr = (lhs - rhs).expand()
    try:
        poly = Poly(expr, x)
    except PolynomialError:
        return "not_linear", None
    deg = poly.degree()
    if deg == 1:
        solved = solve(Eq(lhs, rhs), x)
        check(len(solved) == 1, f"expected exactly one solution for a linear equation, got {solved}")
        return "linear", solved[0]
    return "not_linear", None


def _undo_text(k: int) -> str:
    """The step that removes the constant k from one side: "Subtract 4" or "Add 5"."""
    return f"Subtract {k}" if k > 0 else f"Add {-k}"


def _iasle_easy(rng):
    stem = "Is this equation linear in x? If it is linear, also give the value of x that solves it."
    if rng.choice([True, False]):
        # Linear: a*x + b = c.
        a = rng.randint(2, 9)
        x0 = rng.choice([v for v in range(-9, 10) if v != 0])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        c = a * x0 + b
        lhs, rhs = a * x + b, c
        label, solved = _classify(lhs, rhs)
        check(label == "linear", f"expected linear, independent check said {label}")
        check_equivalent(solved, x0, "independent solve must match construction's x0")

        text = f"{join_terms([(a, 'x'), (b, '')])} = {c}. {stem}"
        answer_text = f"Linear; x = {x0}"
        wrong_value_1 = -x0
        wrong_value_2 = c + b  # forgot to subtract b (added instead) before dividing by a
        check_distinct(x0, wrong_value_1, msg="wrong-classification-value distractor must differ from x0")
        candidates = [
            "Not linear",
            f"Linear; x = {wrong_value_1}",
            f"Linear; x = {wrong_value_2}",
            f"Linear; x = {c - b}",  # forgot to divide by a
            f"Linear; x = {_frac_text(Fraction(c, a))}",  # forgot to subtract b
        ]
        steps = [
            f"{_undo_text(b)} from both sides: {a}x = {c - b}.",
            f"Divide both sides by {a}: x = {x0}.",
            "The highest power of x is 1, so the equation is linear.",
        ]
    else:
        # Not linear: x^2 + a = b (obvious squared term, no trap intended).
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        b = rng.choice([v for v in range(-9, 10) if v != 0])
        lhs, rhs = x**2 + a, b
        label, _ = _classify(lhs, rhs)
        check(label == "not_linear", f"expected not_linear, independent check said {label}")

        text = f"{join_terms([(1, 'x^2'), (a, '')])} = {b}. {stem}"
        answer_text = "Not linear"
        naive_1, naive_2, naive_3 = b - a, a - b, b + a
        candidates = [
            f"Linear; x = {naive_1}",
            f"Linear; x = {naive_2}",
            f"Linear; x = {naive_3}",
            f"Linear; x = {b}",  # left the constant a out
            f"Linear; x = {-b}",
        ]
        steps = ["The equation has an x^2 term, so the highest power of x is 2 and it is not linear."]

    distractors = pick_distractors(answer_text, candidates)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=steps,
        distractors=distractors,
    )


def _iasle_medium(rng):
    stem = "Is this equation linear in x? If it is linear, also give the value of x that solves it."
    if rng.choice([True, False]):
        # Linear: (x + a)/p = c -- unknown in the numerator of a fraction.
        p = rng.choice([2, 3, 4, 5])
        c = rng.choice([v for v in range(-9, 10) if v != 0])
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        x0 = p * c - a
        lhs, rhs = (x + a) / p, c
        label, solved = _classify(lhs, rhs)
        check(label == "linear", f"expected linear, independent check said {label}")
        check_equivalent(solved, x0, "independent solve must match construction's x0")

        text = f"({join_terms([(1, 'x'), (a, '')])})/{p} = {c}. {stem}"
        answer_text = f"Linear; x = {x0}"
        wrong_value_1 = c - a  # forgot to multiply c by p
        wrong_value_2 = p * c + a  # sign error on a
        check_distinct(
            x0, wrong_value_1, wrong_value_2, msg="wrong-classification-value distractors must differ from x0"
        )
        candidates = [
            "Not linear",
            f"Linear; x = {wrong_value_1}",
            f"Linear; x = {wrong_value_2}",
            f"Linear; x = {p * c}",  # forgot to take a away
            f"Linear; x = {_frac_text(Fraction(c, p) - a)}",  # divided by p instead of multiplying
        ]
        steps = [
            f"Multiply both sides by {p}: {join_terms([(1, 'x'), (a, '')])} = {p * c}.",
            f"{_undo_text(a)} from both sides: x = {x0}.",
            "The highest power of x is 1, so the equation is linear.",
        ]
    else:
        # Not linear: a/(x + d) = c -- unknown in the denominator (reciprocal).
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        d = rng.choice([v for v in range(-6, 7) if v != 0])
        c = rng.choice([v for v in range(-6, 7) if v != 0])
        lhs, rhs = a / (x + d), c
        label, _ = _classify(lhs, rhs)
        check(label == "not_linear", f"expected not_linear, independent check said {label}")

        real_solution = Fraction(a, c) - d  # correctly solved a=c*(x+d) -- but still not linear
        bad_cross_multiply = Fraction(c, a) - d  # inverted the cross-multiplication
        check(
            real_solution != bad_cross_multiply or a * a == c * c,
            "surface-pattern distractor values should typically differ",
        )
        text = f"{a}/({join_terms([(1, 'x'), (d, '')])}) = {c}. {stem}"
        answer_text = "Not linear"
        candidates = [
            f"Linear; x = {_frac_text(real_solution)}",
            f"Linear; x = {_frac_text(bad_cross_multiply)}",
            f"Linear; x = {_frac_text(Fraction(a, c) + d)}",  # sign error on d
            f"Linear; x = {a * c - d}",  # multiplied a by c instead of dividing
            f"Linear; x = {_frac_text(-real_solution)}",
            f"Linear; x = {c - d}",  # treated the equation as x + d = c
            f"Linear; x = {a - d}",  # treated the equation as x + d = a
            f"Linear; x = {-d}",  # set the denominator to zero
            f"Linear; x = {d}",
        ]
        steps = ["The unknown x is in the denominator of a fraction, so the equation is not linear in x."]

    distractors = pick_distractors(answer_text, candidates)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=steps,
        distractors=distractors,
    )


def _iasle_hard(rng):
    stem = "Is this equation linear in x? If it is linear, also give the value of x that solves it."
    while True:
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        b = rng.randint(1, 9)
        c = rng.choice([v for v in range(-9, 10) if v != 0])
        if a == b:
            continue
        if c + a * b == 0:
            continue  # keeps x0 != 0 in the "collapses to linear" branch, so -x0 != x0
        break
    numer1, numer2 = join_terms([(1, "x"), (a, "")]), join_terms([(1, "x"), (-b, "")])

    if rng.choice([True, False]):
        # (x+a)(x-b) - x^2 = c -- the x^2 terms cancel: collapses to genuinely linear.
        lhs, rhs = (x + a) * (x - b) - x**2, c
        label, solved = _classify(lhs, rhs)
        check(label == "linear", f"expected linear (x^2 terms should cancel), independent check said {label}")
        x0 = Fraction(c + a * b, a - b)
        check_equivalent(solved, Rational(x0.numerator, x0.denominator), "independent solve must match hand-derived x0")

        text = f"({numer1})({numer2}) - x^2 = {c}. {stem}"
        answer_text = f"Linear; x = {_frac_text(x0)}"
        wrong_side = Fraction(c + a * b, b - a)  # = -x0
        sign_error = Fraction(c - a * b, a - b)  # sign error on the a*b term
        check_distinct(
            Rational(x0.numerator, x0.denominator),
            Rational(wrong_side.numerator, wrong_side.denominator),
            Rational(sign_error.numerator, sign_error.denominator),
            msg="wrong-classification-value distractors must differ from x0",
        )
        candidates = [
            "Not linear",
            f"Linear; x = {_frac_text(wrong_side)}",
            f"Linear; x = {_frac_text(sign_error)}",
            f"Linear; x = {_frac_text(Fraction(c, a - b))}",  # left the a*b term out
            f"Linear; x = {_frac_text(-sign_error)}",
        ]
        expanded = join_terms([(1, "x^2"), (a - b, "x"), (-a * b, ""), (-1, "x^2")])
        steps = [
            f"Expand: {expanded} = {c}.",
            f"The x^2 terms cancel, leaving {join_terms([(a - b, 'x'), (-a * b, '')])} = {c}, so the equation is linear.",
            f"Solve: x = {_frac_text(x0)}.",
        ]
    else:
        # (x+a)(x-b) + x^2 = c -- the x^2 terms add: stays genuinely quadratic.
        lhs, rhs = (x + a) * (x - b) + x**2, c
        label, _ = _classify(lhs, rhs)
        check(label == "not_linear", f"expected not_linear (x^2 should survive), independent check said {label}")

        text = f"({numer1})({numer2}) + x^2 = {c}. {stem}"
        answer_text = "Not linear"
        # The trap: pretend the x^2 cancels the same way the sibling shape does.
        ignored_x2 = Fraction(c + a * b, a - b)
        sign_variant = Fraction(c - a * b, a - b)
        candidates = [
            f"Linear; x = {_frac_text(ignored_x2)}",
            f"Linear; x = {_frac_text(sign_variant)}",
            f"Linear; x = {_frac_text(Fraction(c, a - b))}",  # left the a*b term out as well
            f"Linear; x = {_frac_text(-ignored_x2)}",
        ]
        expanded = join_terms([(1, "x^2"), (a - b, "x"), (-a * b, ""), (1, "x^2")])
        steps = [
            f"Expand: {expanded} = {c}.",
            "The x^2 terms add to 2x^2, so an x^2 term remains and the equation is not linear.",
        ]

    distractors = pick_distractors(answer_text, candidates)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=steps,
        distractors=distractors,
    )


@generator(
    "identify_and_solve_linear_equation",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def identify_and_solve_linear_equation(rng, tier="easy"):
    """Classify a given one-variable equation as linear or not linear."""
    if tier == "easy":
        return _iasle_easy(rng)
    if tier == "medium":
        return _iasle_medium(rng)
    if tier == "hard":
        return _iasle_hard(rng)
    raise ValueError(f"unknown tier {tier!r}")
