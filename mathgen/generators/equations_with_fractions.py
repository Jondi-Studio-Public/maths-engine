"""Linear equations with fractions (an exam-style item).

Every one of the thirteen existing ``equations/`` skills renders integer
coefficients only -- nothing asks anything like ``x/3 + x/4 = 7``. This
module is the missing "clear the denominator" skill: one topic,
``linear_equation_fraction``, with a ``kind`` switch across four genuinely
different reasoning shapes (not just bigger numbers):

- ``single``: one fractional term, ``x/4 = 6`` or ``2x/5 = 8``.
- ``two_terms``: two fractional terms with different denominators,
  ``x/3 + x/4 = 7`` -- the "second denominator" escalation.
- ``binomial``: a fraction with a binomial numerator, ``(x + 2)/3 = 5`` or
  ``(2x - 1)/4 = 3`` -- the "bracketed numerator" escalation.
- ``both_sides``: fractions on both sides, ``(x + 1)/2 = (x - 3)/5`` --
  unknowns on both sides.

Every kind is constructed backwards from an integer solution ``x0`` chosen
first (never solved forwards), so every question has an exact integer
answer -- ``answer: number`` on the skill, matching the sibling equations
skills (``two-step-equations``, ``equations-with-brackets``,
``variables-on-both-sides``), never a rounded decimal standing in for a
fractional root. Every solve is re-checked with an independent
``sympy.solve`` call on the *fractional* equation (never the pre-cleared
integer restatement), and every misconception distractor is checked
distinct from ``x0`` with ``check_distinct`` unconditionally, every seed.
"""

from __future__ import annotations

from sympy import Eq, Rational, solve, symbols

from ..core import check, check_distinct, generator, join_terms, pick_distractors, rational_text

x = symbols("x")


def _fraction_single(
    rng, den_choices=(2, 3, 4, 5, 6), coef_choices=(1, 1, 1, 2, 2, 3), sol_hi=10, allow_negative=False
):
    # a*x/d = c. Backwards: pick the denominator and the *quotient* x0/d = m
    # first, so x0 = m*d divides cleanly and c = a*m falls out exactly.
    d = rng.choice(den_choices)
    a = rng.choice(coef_choices)
    m = rng.randint(1, sol_hi)
    if allow_negative and rng.random() < 0.4:
        m = -m
    x0 = m * d
    c = a * m

    coef_str = "" if a == 1 else str(a)
    text = f"{coef_str}x/{d} = {c}"
    check(solve(Eq(Rational(a, d) * x, c), x) == [x0], "single-term fraction equation did not solve to x0")

    # Misconceptions: forgot to multiply by d (solved a*x = c directly);
    # forgot to divide by a after clearing the denominator; divided by d
    # instead of multiplying it out.
    cands = [Rational(c, a), Rational(c, a * d)]
    if a != 1:
        cands.append(c * d)
    check_distinct(x0, *cands, msg="a fraction-single distractor equals x0")

    answer_text = f"x = {x0}"
    cand_texts = [f"x = {rational_text(v)}" for v in cands]
    return dict(
        text=f"Solve {text}",
        latex=f"plain:Solve {text}",
        answer=x0,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Multiply both sides by {d}: {coef_str}x = {c * d}",
            f"x = {c * d}/{a} = {x0}" if a != 1 else f"x = {x0}",
        ],
        distractors=pick_distractors(answer_text, cand_texts),
    )


def _fraction_two_terms(rng, den_choices=(2, 3, 4, 5), k_hi=2, allow_minus=True):
    # x/p (+/-) x/q = c, p != q. Backwards: x0 = k*p*q makes both terms land
    # on an integer exactly (x0/p = k*q, x0/q = k*p), the way multiplying
    # through by a common denominator pq is meant to.
    p, q = rng.sample(list(den_choices), 2)
    k = rng.randint(1, k_hi)
    if allow_minus and rng.random() < 0.4 and p != q:
        # Order p < q so the subtraction lands on a positive constant.
        p, q = min(p, q), max(p, q)
        op = "-"
        c = k * (q - p)
    else:
        op = "+"
        c = k * (p + q)
    x0 = k * p * q

    lhs = Rational(1, p) * x + (Rational(1, q) * x if op == "+" else -Rational(1, q) * x)
    check(solve(Eq(lhs, c), x) == [x0], "two-term fraction equation did not solve to x0")

    denom_combo = (p + q) if op == "+" else (q - p)
    only_lhs_by_lcd = Rational(c, denom_combo)  # multiplied the LHS by the LCD but not the RHS
    added_denominators = c * denom_combo  # treated x/p (+/-) x/q as x/(p (+/-) q)
    dropped_second_term = c * p  # solved as if only the x/p term were there
    check_distinct(
        x0, only_lhs_by_lcd, added_denominators, dropped_second_term, msg="a fraction-two-terms distractor equals x0"
    )

    text = f"x/{p} {op} x/{q} = {c}"
    answer_text = f"x = {x0}"
    cand_texts = [f"x = {rational_text(v)}" for v in (only_lhs_by_lcd, added_denominators, dropped_second_term)]
    return dict(
        text=f"Solve {text}",
        latex=f"plain:Solve {text}",
        answer=x0,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Multiply every term by the common denominator {p * q}.", f"x = {x0}"],
        distractors=pick_distractors(answer_text, cand_texts),
    )


def _fraction_binomial(
    rng, den_choices=(2, 3, 4, 5, 6), coef_choices=(1, 1, 1, 2, 2, 3), h_hi=9, x0_hi=9, allow_negative_x0=True
):
    # (a*x + h)/d = c. Backwards: pick x0, a, h freely, then find a
    # denominator that divides the numerator a*x0 + h cleanly.
    for _ in range(1000):
        a = rng.choice(coef_choices)
        h = rng.choice([v for v in range(-h_hi, h_hi + 1) if v != 0])
        x0 = rng.randint(-x0_hi, x0_hi) if allow_negative_x0 else rng.randint(1, x0_hi)
        numerator_val = a * x0 + h
        candidate_d = [d for d in den_choices if numerator_val % d == 0 and numerator_val // d != 0]
        if candidate_d:
            d = rng.choice(candidate_d)
            c = numerator_val // d
            break
    else:
        raise RuntimeError("could not construct a binomial-numerator fraction equation")

    numerator_text = join_terms([(a, "x"), (h, "")])
    text = f"({numerator_text})/{d} = {c}"
    check(
        solve(Eq((Rational(a, d) * x + Rational(h, d)), c), x) == [x0],
        "binomial-numerator fraction equation did not solve to x0",
    )

    # Misconceptions: sign error isolating h after clearing the denominator;
    # dropped h entirely when clearing the denominator; split the fraction as
    # x/d + h = c (forgot h itself is also divided by d).
    sign_error_on_h = Rational(c * d + h, a)
    dropped_h = Rational(c * d, a)
    split_fraction = Rational((c - h) * d, a)
    check_distinct(x0, sign_error_on_h, dropped_h, split_fraction, msg="a fraction-binomial distractor equals x0")

    answer_text = f"x = {x0}"
    cand_texts = [f"x = {rational_text(v)}" for v in (sign_error_on_h, dropped_h, split_fraction)]
    return dict(
        text=f"Solve {text}",
        latex=f"plain:Solve {text}",
        answer=x0,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Multiply both sides by {d}: {numerator_text} = {c * d}", f"x = {x0}"],
        distractors=pick_distractors(answer_text, cand_texts),
    )


def _fraction_both_sides(rng, d1_choices=(2, 3), t_choices=(2, 3), h_hi=6, sol_hi=9, allow_negative=True):
    # (x + h1)/d1 = (x + h2)/d2, d1 != d2. Backwards: fix d2 = d1*t so d1 | d2, pick x0 and h1
    # freely, and derive the h2 that makes the two sides equal.
    for _ in range(1000):
        d1 = rng.choice(d1_choices)
        t = rng.choice(t_choices)
        d2 = d1 * t
        x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
        h1 = rng.choice([v for v in range(-h_hi, h_hi + 1) if v != 0])
        h2 = (t - 1) * x0 + t * h1
        if h2 == h1 or abs(h2) > 20:
            continue
        # Misconceptions: cross-multiplied but only multiplied one numerator
        # by the other side's denominator; multiplied each numerator by the
        # WRONG denominator (swapped which denominator cross-multiplies which
        # side); split each fraction as x/d + h = ... and forgot h is also
        # divided by d. Resample rather than accept a seed where one happens
        # to coincide with x0 (a "does not simplify" trap would slip past a
        # single-value tolerance check but not this one).
        multiplied_one_side_only = Rational(h2 - d2 * h1, d2 - 1)
        swapped_cross_multiply = Rational(d2 * h2 - d1 * h1, d1 - d2)
        split_fraction = Rational((h2 - h1) * d1 * d2, d2 - d1)
        if x0 in (multiplied_one_side_only, swapped_cross_multiply, split_fraction):
            continue
        break
    else:
        raise RuntimeError("could not construct a both-sides fraction equation")

    numerator1 = join_terms([(1, "x"), (h1, "")])
    numerator2 = join_terms([(1, "x"), (h2, "")])
    text = f"({numerator1})/{d1} = ({numerator2})/{d2}"
    check(
        solve(Eq(Rational(1, d1) * (x + h1), Rational(1, d2) * (x + h2)), x) == [x0],
        "both-sides fraction equation did not solve to x0",
    )
    check_distinct(
        x0,
        multiplied_one_side_only,
        swapped_cross_multiply,
        split_fraction,
        msg="a fraction-both-sides distractor equals x0",
    )

    answer_text = f"x = {x0}"
    cand_texts = [f"x = {rational_text(v)}" for v in (multiplied_one_side_only, swapped_cross_multiply, split_fraction)]
    return dict(
        text=f"Solve {text}",
        latex=f"plain:Solve {text}",
        answer=x0,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Cross-multiply: {d2}({numerator1}) = {d1}({numerator2})", f"x = {x0}"],
        distractors=pick_distractors(answer_text, cand_texts),
    )


@generator("linear_equation_fraction", level=7)
def linear_equation_fraction(rng, kind="single", **params):
    if kind == "single":
        return _fraction_single(rng, **params)
    if kind == "two_terms":
        return _fraction_two_terms(rng, **params)
    if kind == "binomial":
        return _fraction_binomial(rng, **params)
    if kind == "both_sides":
        return _fraction_both_sides(rng, **params)
    raise ValueError(f"unknown kind {kind!r}")
