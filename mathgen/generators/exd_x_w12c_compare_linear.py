"""Exam chain for comparing linear relationships: one given as a graph, the other as a table or a rule.

Relationship A is a straight line drawn on a grid (through lattice points); relationship B is a table of (x, y) pairs
or a worded rule. The two meet at a whole-number x. Set the way a Year 8 exam sets it:

(a) the rule for A, read from the graph                              expression, 2 marks
(b) the gradient of B, from its table or rule                        fraction,   1 mark
(c) the difference between the y-values of A and B at a far x        number,     2 marks, follow-through (a), (b)
(d) the x at which A and B have the same y-value                     number,     2 marks, follow-through (a), (b)

## The algebra (checked on every question)

A = (mA, cA) is read back from the figure's two through points. B = (mB, cB) is fitted to the table (every pair must
lie on one line). The meeting point is found with SymPy's ``solve`` and must equal the constructed one. The values at
the far x are evaluated from both rules. The follow-through for (c) and (d) reads the learner's own rule for A
(its value at 0 and at 1) and their own gradient for B; the intercept of B is always visible in the stem (the pair at
x = 0, or the number added or subtracted in the rule).

## Distractors

* (a): the gradient and intercept swapped, the intercept's sign reversed, the gradient's sign reversed, the gradient
  upside down.
* (b): the rise between two table rows without the run, the intercept, the sign reversed, output over input.
* (c): the gradients compared alone, the starting values compared alone, one value instead of the difference, the two
  values added.
* (d): the y-value given for x, the intercepts subtracted with no division, the sign reversed.
"""

from __future__ import annotations

from fractions import Fraction as F

import sympy

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase, rational_text
from mathgen.generators._exd_x_w12c_kit import redraw, expression_part, fraction_part, int_part
from mathgen.generators.fg1_line_reading_build import draw_line, eq_text, pairs_text, pick_pair
from mathgen.generators.pv1a_line_figure_build import line_figure, read_line

_X = sympy.Symbol("x")
_ALT = "a straight line on a coordinate grid"

_OPEN = (
    "Relationship A is shown in the graph.",
    "The graph shows relationship A, a straight line.",
    "Relationship A is the straight line drawn on the grid.",
)
_B_TABLE = (
    "Relationship B is linear and has these (x, y) pairs: {pairs}.",
    "Relationship B is a straight-line relationship. Its (x, y) pairs are {pairs}.",
    "The (x, y) pairs for relationship B, which is linear, are {pairs}.",
)
_B_RULE = (
    "Relationship B is found by multiplying x by {m} and then {verb} {c}.",
    "For relationship B, y is found by multiplying x by {m} and then {verb} {c}.",
    "Relationship B follows this rule: multiply x by {m}, then {imperative} {c}.",
)
_TABLE_XS = ((0, 1, 2, 3), (0, 2, 4, 6), (0, 3, 6, 9))


def _rhs(m, c) -> str:
    """The right-hand side of ``y = mx + c`` the way a student writes it ("-x/2 + 3")."""
    return eq_text(m, c)[4:]


def _build(rng):
    """Rule for A from its graph, gradient of B, the gap at a far x, and where the two relationships meet."""
    for _ in attempts(3000, "a graph and a second relationship that meet at a whole-number x"):
        win, M, C, pts = draw_line(
            rng,
            slopes=[F(s) for s in (-3, -2, -1, 1, 2, 3)] + [F(1, 2), F(-1, 2), F(3, 2), F(1, 3), F(2, 3)],
            c_range=(-4, 4),
            y_axis=True,
            c_visible=True,
            c_nonzero=True,
        )
        mA, cA = F(M), F(C)
        mB = rng.choice((-3, -2, 2, 3, 4))
        if F(mB) == mA:
            continue
        xi = mA.denominator * rng.choice([k for k in range(-6, 11) if k not in (0,)])
        yi = mA * xi + cA
        cB = yi - mB * xi
        if cB.denominator != 1 or abs(cB) > 12 or cB in (0, cA) or abs(xi) > 12:
            continue
        cB = int(cB)
        far = [v for v in (6, 8, 9, 10, 12, -6, -8, -10, -12) if v % mA.denominator == 0 and v != xi]
        if not far:
            continue
        X = rng.choice(far)
        break
    pa, pb = pick_pair(rng, pts)
    fig = line_figure(win, [(pa, pb)], _ALT, labels=["A"])
    check(read_line(fig) == (mA, cA), "the drawn line is not relationship A")

    # Relationship B, as a table or as a worded rule, always showing its value at x = 0.
    as_table = rng.random() < 0.55
    if as_table:
        xs = rng.choice(_TABLE_XS)
        rows = [(x, mB * x + cB) for x in xs]
        step = xs[1] - xs[0]
        b_text = phrase(rng, _B_TABLE, pairs=pairs_text(rows))
    else:
        step = 1
        verb, imperative = ("adding", "add") if cB > 0 else ("subtracting", "subtract")
        b_text = phrase(rng, _B_RULE, m=mB, verb=verb, imperative=imperative, c=abs(cB))
    stem = f"{phrase(rng, _OPEN)} {b_text}"

    # Second route to every key: solve A = B with SymPy and evaluate both rules.
    a_expr = mA.numerator * _X / mA.denominator + int(cA)
    b_expr = mB * _X + cB
    solved = sympy.solve(sympy.Eq(a_expr, b_expr), _X)
    check(solved == [xi], "the meeting point is not the constructed one")
    gap = abs(a_expr.subs(_X, X) - b_expr.subs(_X, X))
    check(gap == abs((mA * X + cA) - (mB * X + cB)) and gap.is_integer, "the gap at the far x is not a whole number")
    gap = int(gap)
    sign = 1 if a_expr.subs(_X, X) > b_expr.subs(_X, X) else -1  # (c) follows through with this sign, not abs()

    a_text = _rhs(mA, cA)
    mA_text = rational_text(mA)
    part_a = expression_part(
        "Find the rule for relationship A. Give the expression that y equals.",
        a_text,
        2,
        [
            f"The line crosses the y-axis at {int(cA)}, so the intercept is {int(cA)}.",
            f"Moving along the line, y changes by {mA_text} for each 1 across, so the gradient is {mA_text}.",
            f"The rule is y = {a_text}.",
        ],
        [
            (_rhs(F(int(cA)), mA), "gradient-and-intercept-swapped"),
            (_rhs(mA, -cA), "intercept-read-with-wrong-sign"),
            (_rhs(-mA, cA), "gradient-sign-from-point-order"),
            (_rhs(1 / mA, cA), "inverted-gradient"),
        ],
    )

    last_x, last_y = rows[-1] if as_table else (None, None)
    b_cands = [
        (F(cB), "gradient-and-intercept-swapped"),
        (F(-mB), "gradient-sign-from-point-order"),
    ]
    if as_table:
        b_cands.insert(0, (F(mB * step), "gradient-rise-only"))
        b_cands.append((F(last_y, last_x), "output-over-input-as-rate"))
        b_working = [
            f"Between consecutive rows x goes up by {step} and y changes by {mB * step}.",
            f"The gradient is {mB * step} / {step} = {mB}.",
        ]
    else:
        b_cands.append((F(1, mB), "inverted-gradient"))
        b_working = [f"y is x multiplied by {mB}, then changed by {cB}, so the gradient is {mB}."]
    part_b = fraction_part("Find the gradient of relationship B.", F(mB), 1, b_working, b_cands)

    a_far, b_far = mA * X + cA, mB * X + cB
    part_c = int_part(
        f"Find the difference between the y-values of relationship A and relationship B when x = {X}. "
        "Give a positive number.",
        gap,
        2,
        [
            f"A: y = {rational_text(mA)} x {X} {'+' if cA > 0 else '-'} {abs(int(cA))} = {rational_text(a_far)}.",
            f"B: y = {mB} x {X} {'+' if cB > 0 else '-'} {abs(cB)} = {b_far}.",
            f"The difference is {gap}.",
        ],
        [
            (abs((mA - mB) * X), "compares-gradients-only"),
            (abs(cA - cB), "compares-starting-values-only"),
            (abs(b_far), "difference-not-taken"),
            (abs(a_far + b_far), "adds-instead-of-subtracts"),
        ],
        lo=0,
        hi=200,
        follow_through=f"{sign}*(Subs(part_a,(x,),({X},)) - (part_b*({X}) + ({cB})))",
    )
    part_d = int_part(
        "At what value of x do relationship A and relationship B have the same y-value?",
        xi,
        2,
        [
            f"Set the rules equal: {a_text} = {_rhs(F(mB), F(cB))}.",
            f"Collect the x terms and the numbers: x = {xi}.",
            f"Check: both give y = {rational_text(yi)} at x = {xi}.",
        ],
        [
            (yi, "swapped-coordinates"),
            (cB - cA, "forgot-to-divide-by-gradient-difference"),
            (-xi, "sign-reversed-result"),
        ],
        lo=-60,
        hi=60,
        follow_through=(
            f"(({cB}) - Subs(part_a,(x,),(0,))) / (Subs(part_a,(x,),(1,)) - Subs(part_a,(x,),(0,)) - part_b)"
        ),
    )
    return dict(text=stem, figure=fig, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_w12c_compare_linear", level=12, presets={"exam": {"tier": "exam"}})
def exd_x_w12c_compare_linear(rng, tier: str = "exam"):
    """Rule for A from its graph, gradient of B, the gap at a far x, and where the two relationships meet."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return redraw(rng, _build)
