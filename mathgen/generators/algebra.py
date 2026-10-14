"""Levels 7-10: linear equations, simultaneous equations, quadratics, functions."""

import math

from sympy import symbols, expand, solve, linsolve, Eq, latex, Rational, FiniteSet

from ..answers import Distractor, exact_value
from ..core import (
    draw_until,
    generator,
    check,
    pick_distractors,
    rational_text,
    xy_solution,
    join_terms,
    fmt_signed,
    paren_negative,
    GRADER_TOLERANCE,
)

x, y = symbols("x y")


def _linear_equation_brackets(rng, coef_hi, sol_hi, b_hi, allow_negative):
    # a(x + h) = c: distribute, then solve as usual.
    x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    h = rng.randint(1, min(b_hi, 12))
    if rng.random() < 0.5:
        h = -h
    c = a * (x0 + h)
    ah = a * h
    sign = "+" if h >= 0 else "-"
    sign_ah = "+" if ah >= 0 else "-"
    text = f"{a}(x {sign} {abs(h)}) = {c}"
    check(solve(Eq(a * (x + h), c), x) == [x0])
    # Misconceptions: distributed onto x only (forgot the h term, so a*x = c);
    # ignored the brackets entirely (a*x + h = c); sign error on h when isolating x.
    cands = [
        f"x = {rational_text(Rational(c, a))}",
        f"x = {rational_text(Rational(c - h, a))}",
        f"x = {rational_text(Rational(c + ah, a))}" if ah else None,
    ]
    cands = [c_ for c_ in cands if c_ is not None]
    return dict(
        text=f"Solve {text}",
        latex=rf"\text{{Solve }} {a}(x {sign} {abs(h)}) = {c}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[
            f"Expand: {a}x {sign_ah} {abs(ah)} = {c}",
            f"x = ({c} {'-' if ah >= 0 else '+'} {abs(ah)}) / {a} = {x0}",
        ],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_brackets_leading(rng, coef_hi, sol_hi, b_hi, allow_negative):
    # d (+/-) a(x + h) = c: a leading constant term OUTSIDE the bracket, on
    # the same side, before the distribution step. `a` may now be drawn negative as well as
    # positive (the paste's original ask), but the genuine reasoning
    # escalation is the leading constant `d`: order of operations now
    # requires handling `d` before/while distributing `a` across the
    # bracket -- a real token-sequence change (an extra leading number and
    # operator no lower tier's shape has) regardless of what happens to any
    # one digit's own sign.
    x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    if rng.random() < 0.5:
        a = -a
    h = rng.randint(1, min(b_hi, 12))
    if rng.random() < 0.5:
        h = -h
    d = rng.randint(1, b_hi)
    c = d + a * (x0 + h)
    ah = a * h
    sign_h = "+" if h >= 0 else "-"
    sign_a = "+" if a >= 0 else "-"
    sign_ah = "+" if ah >= 0 else "-"
    text = f"{d} {sign_a} {abs(a)}(x {sign_h} {abs(h)}) = {c}"
    check(solve(Eq(d + a * (x + h), c), x) == [x0])
    # Misconceptions specific to the leading constant (the existing two --
    # partial distribution, sign error on the bracket -- are the same slip
    # as candidate 2 below once a leading term is in play):
    #  1. moved the leading constant to the other side with the WRONG sign
    #     (solved a(x+h) = c + d instead of c - d);
    #  2. distributed `a` onto x but not onto h (solved d + a*x + h = c);
    #  3. forgot the leading constant entirely (solved as if it read
    #     a(x+h) = c).
    # All three are provably distinct from x0 by construction: writing
    # x0 = (c-d)/a - h, cand1 - x0 = 2d/a (nonzero, d >= 1); cand2 - x0 =
    # h*(a-1)/a (nonzero, h != 0 and a >= 2 in magnitude); cand3 - x0 = d/a
    # (nonzero, d >= 1) -- so no filtering is needed before handing these to
    # `pick_distractors` (its own dedup still drops anything that happens to
    # collide with the answer's rendered text/value).
    cand1 = rational_text(Rational(c + d, a) - h)
    cand2 = rational_text(Rational(c - d - h, a))
    cand3 = rational_text(Rational(c, a) - h)
    cands = [f"x = {cand1}", f"x = {cand2}", f"x = {cand3}"]
    return dict(
        text=f"Solve {text}",
        latex=rf"\text{{Solve }} {d} {sign_a} {abs(a)}(x {sign_h} {abs(h)}) = {c}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[
            f"{d} {sign_a} {abs(a)}x {sign_ah} {abs(ah)} = {c}",
            f"x = ({c} {'-' if ah >= 0 else '+'} {abs(ah)} - {d}) / {a} = {x0}",
        ],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_double_brackets(rng, coef_hi, sol_hi, b_hi, allow_negative):
    # a1(x + h1) = a2(x + h2): brackets on BOTH sides, each side its own multiplier, at least one
    # drawn negative. A fuller form would also carry a coefficient on the variable inside each
    # bracket ("3b", "2b"); this generator keeps that inner coefficient at 1 (a1(x
    # + h1), not a1(m1*x + h1)) -- still the same structural escalation
    # (brackets on both sides, each side its own multiplier, one of them negative), a
    # deliberate simplification.
    x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
    a_hi = min(coef_hi, 9)
    h_hi = min(b_hi, 12)
    a1 = a2 = h1 = h2 = None
    for _ in range(500):
        m1 = rng.randint(2, a_hi)
        m2 = rng.randint(2, a_hi)
        sign1 = rng.choice([1, -1])
        sign2 = -1 if sign1 == 1 else rng.choice([1, -1])
        a1_try, a2_try = sign1 * m1, sign2 * m2
        if a1_try == a2_try:
            continue
        h1_candidates = [v for v in range(-h_hi, h_hi + 1) if v != 0]
        rng.shuffle(h1_candidates)
        for h1_try in h1_candidates:
            V = a1_try * (x0 + h1_try)
            if V % a2_try:
                continue
            h2_try = V // a2_try - x0
            if h2_try == 0 or h2_try == h1_try or abs(h2_try) > h_hi:
                continue
            a1, a2, h1, h2 = a1_try, a2_try, h1_try, h2_try
            break
        if a1 is not None:
            break
    if a1 is None:
        raise RuntimeError("_linear_equation_double_brackets: could not construct valid params")

    sign_h1 = "+" if h1 >= 0 else "-"
    sign_h2 = "+" if h2 >= 0 else "-"
    lhs = f"{a1}(x {sign_h1} {abs(h1)})"
    rhs = f"{a2}(x {sign_h2} {abs(h2)})"
    check(solve(Eq(a1 * (x + h1), a2 * (x + h2)), x) == [x0])
    b1, b2 = a1 * h1, a2 * h2
    # Once both brackets are expanded the equation is exactly
    # `_linear_equation_both_sides`'s own shape (a1*x + b1 = a2*x + b2), so
    # its three misconceptions carry over unchanged, plus one bracket-
    # specific slip: distributed the multiplier onto x but not onto h on
    # either side (solved a1*x + h1 = a2*x + h2).
    cands = []
    if a1 + a2:
        cands.append(f"x = {rational_text(Rational(b2 - b1, a1 + a2))}")
    cands.append(f"x = {rational_text(Rational(b2, a1 - a2))}")
    cands.append(f"x = {rational_text(Rational(b1 - b2, a1 - a2))}")
    cands.append(f"x = {rational_text(Rational(h2 - h1, a1 - a2))}")
    return dict(
        text=f"Solve {lhs} = {rhs}",
        latex=rf"\text{{Solve }} {lhs} = {rhs}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[
            f"Expand: {fmt_signed(a1, 'x')} {'+' if b1 >= 0 else '-'} {abs(b1)} = "
            f"{fmt_signed(a2, 'x')} {'+' if b2 >= 0 else '-'} {abs(b2)}",
            f"x = ({b2} - {paren_negative(b1)}) / ({a1} - {paren_negative(a2)}) = {x0}",
        ],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_both_sides(rng, coef_hi, sol_hi, allow_negative):
    # a1*x + b1 = a2*x + b2, a1 != a2: variables on both sides.
    x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
    a1 = rng.randint(2, coef_hi)
    a2 = rng.randint(1, coef_hi)
    while a2 == a1:
        a2 = rng.randint(1, coef_hi)
    b1 = rng.randint(-20, 20)
    b2 = (a1 - a2) * x0 + b1
    sign1 = "+" if b1 >= 0 else "-"
    sign2 = "+" if b2 >= 0 else "-"
    a1x, a2x = fmt_signed(a1, "x"), fmt_signed(a2, "x")
    lhs = f"{a1x} {sign1} {abs(b1)}"
    rhs = f"{a2x} {sign2} {abs(b2)}"
    check(solve(Eq(a1 * x + b1, a2 * x + b2), x) == [x0])
    # Misconceptions: added the x-terms instead of subtracting; moved only the
    # constant and left both x-terms in; subtracted in the wrong direction.
    cands = [
        f"x = {rational_text(Rational(b2 - b1, a1 + a2))}",
        f"x = {rational_text(Rational(b2, a1 - a2))}",
        f"x = {rational_text(Rational(b1 - b2, a1 - a2))}",
    ]
    return dict(
        text=f"Solve {lhs} = {rhs}",
        latex=rf"\text{{Solve }} {lhs} = {rhs}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[f"({a1} - {a2})x = {b2} - ({b1}) = {b2 - b1}", f"x = {b2 - b1}/{a1 - a2} = {x0}"],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_division_form(rng, coef_hi, sol_hi, allow_negative):
    # x/a = c: the opposite undo-operation from every other `kind` in this function -- all of those
    # are solved by DIVIDING the coefficient away; this one is solved by MULTIPLYING both sides by
    # `a`. Backwards construction, same as `standard`: choose x0 and a first, then c falls out as
    # x0/a -- c will often be a non-integer rational, formatted with `rational_text` exactly like
    # the brackets/both-sides helpers above. x0 == 0 excluded: it collapses c to 0 too, which
    # collides all three misconception candidates below with the answer simultaneously (not a
    # probabilistic near-miss like the other helpers' occasional single collision -- x0 == 0 forces
    # c/a == c == a*c.p == 0 == x0 every time), leaving zero distractors. Redraw rather than
    # special-case afterwards.
    if allow_negative:
        x0 = rng.randint(-sol_hi, sol_hi)
        while x0 == 0:
            x0 = rng.randint(-sol_hi, sol_hi)
    else:
        x0 = rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    c = Rational(x0, a)
    text = f"x/{a} = {rational_text(c)}"
    # Independent symbolic check: solve x/a = c from scratch (not a
    # restatement of the c = Rational(x0, a) construction arithmetic).
    check(solve(Eq(x / a, c), x) == [x0])
    # Misconceptions, each proved distinct from x0 by construction:
    #  1. divided by a instead of multiplying -- the single most likely slip
    #     given every other `kind` here trains "divide to solve". Value is
    #     c/a = x0/a**2; equal to x0 only when x0 == 0 (a >= 2 always), the
    #     one case pick_distractors's own value-dedup drops for us.
    #  2. reported c itself as the answer -- skipped the operation entirely.
    #     Value is c = x0/a; equal to x0 only when x0 == 0, same dedup case.
    #  3. multiplied only the numerator of c by a, forgetting c's own
    #     denominator -- "multiplied only the one side that already had a
    #     fraction" (c's rendered fraction), never touching the divide-by-q
    #     that the fraction bar still owes. With c reduced to p/q (sympy's
    #     own c.p/c.q), this candidate is a*p; a*p - x0 = a*p - a*c =
    #     a*p*(1 - 1/q) = a*p*(q-1)/q, zero iff q == 1 (c renders as a whole
    #     number for that seed) or p == 0 (x0 == 0) -- exactly the case that collides, and
    #     it is fine to let pick_distractors
    #     drop; distinct whenever c is a genuine non-integer fraction.
    cands = [
        f"x = {rational_text(c / a)}",
        f"x = {rational_text(c)}",
        f"x = {a * c.p}",
    ]
    return dict(
        text=f"Solve {text}",
        latex=rf"\text{{Solve }} {text}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[f"Multiply both sides by {a}: x = {a} × ({rational_text(c)})", f"x = {x0}"],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_division_two_step(rng, coef_hi, sol_hi, b_hi, allow_negative):
    # x/a + b = c (or x/a - b = c when b is drawn negative): a genuine
    # two-step equation whose FINAL solving operation is the opposite of
    # every other two-step shape in this function -- the standard kind's
    # "a*x + b = c" is solved by subtracting b, then DIVIDING by a; this one
    # 863, `two-step-equations`'s range-only hard tier). Backwards
    # construction, like `division_form`: choose x0, a, b first, then
    # p = x/a AFTER subtracting b falls out as Rational(x0, a) exactly, and
    # c = p + b. x0 == 0 excluded when allow_negative: like division_form,
    # it collapses the "divided instead of multiplied" misconception
    # (p/a == x0 == 0) into the answer.
    if allow_negative:
        x0 = rng.randint(-sol_hi, sol_hi)
        while x0 == 0:
            x0 = rng.randint(-sol_hi, sol_hi)
    else:
        x0 = rng.randint(1, sol_hi)
    a = rng.randint(2, coef_hi)
    b = rng.randint(1, b_hi)
    if rng.random() < 0.5:
        b = -b
    p = Rational(x0, a)  # x/a after subtracting b -- exact by construction
    c = p + b
    sign = "+" if b >= 0 else "-"
    text = f"x/{a} {sign} {abs(b)} = {rational_text(c)}"
    # Independent symbolic check: solve x/a + b = c from scratch (not a
    # restatement of the c = p + b construction arithmetic).
    check(solve(Eq(x / a + b, c), x) == [x0])
    # Misconceptions, each proved distinct from x0 by construction:
    #  1. divided by a instead of multiplying at the final step -- habituated
    #     from every other `kind` in this function, which all solve by
    #     dividing. Value is p/a = x0/a**2; equal to x0 only when x0 == 0
    #     (a >= 2 always) -- excluded above.
    #  2. added b instead of subtracting it (wrong first-step sign): solved
    #     x/a = c + b, then multiplied by a. Value is a*(c + b) =
    #     a*(p + 2b) = x0 + 2*a*b; equal to x0 only if a*b == 0, impossible
    #     (a >= 2, b != 0 by construction).
    #  3. undid the two steps in the wrong order: multiplied by a BEFORE
    #     subtracting b (computed a*c - b instead of a*(c - b)). Value is
    #     a*c - b = a*(p + b) - b = x0 + b*(a - 1); equal to x0 only if
    #     b*(a - 1) == 0, impossible (a >= 2, b != 0).
    cands = [
        f"x = {rational_text(p / a)}",
        f"x = {rational_text(a * (c + b))}",
        f"x = {rational_text(a * c - b)}",
    ]
    return dict(
        text=f"Solve {text}",
        latex=rf"\text{{Solve }} {text}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[
            f"x/{a} = {rational_text(c)} {'-' if b >= 0 else '+'} {abs(b)} = {rational_text(p)}",
            f"x = {a} × ({rational_text(p)}) = {x0}",
        ],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


def _linear_equation_both_sides_bracket(rng, coef_hi, sol_hi, b_hi, allow_negative):
    # a1(x + h) = a2*x + b2: ONE side (the left) carries a bracket that must be distributed BEFORE
    # the two sides' variable terms can be collected -- combines "distribute" with "variables on
    # both sides" in one question, a genuine extra step no existing `both_sides` question requires.
    # Backwards construction, same style as `_linear_equation_double_brackets`: choose x0, a1 (the
    # bracket's coefficient), h (the constant inside the bracket) and a2 (the right side's
    # x-coefficient, a2 != a1 so the equation is solvable) first, then b2 falls out as a1*(x0+h) -
    # a2*x0 -- no bounded-attempts search needed, unlike `double_brackets` (that helper's two
    # independent brackets can collide on `a1 == a2` or `h2` out of range; here only `a1 != a2`
    # needs avoiding, and b2 is unconstrained so it always exists). x0 == 0 excluded when
    # allow_negative: the "collected on the wrong side" misconception below is exactly -x0, which
    # collides with x0 only when x0 == 0.
    if allow_negative:
        x0 = rng.randint(-sol_hi, sol_hi)
        while x0 == 0:
            x0 = rng.randint(-sol_hi, sol_hi)
    else:
        x0 = rng.randint(1, sol_hi)
    a1 = rng.randint(2, coef_hi)
    a2 = rng.randint(1, coef_hi)
    while a2 == a1:
        a2 = rng.randint(1, coef_hi)
    h = rng.randint(1, min(b_hi, 12))
    if rng.random() < 0.5:
        h = -h
    b2 = a1 * (x0 + h) - a2 * x0
    a1h = a1 * h
    sign_h = "+" if h >= 0 else "-"
    sign_b2 = "+" if b2 >= 0 else "-"
    sign_a1h = "+" if a1h >= 0 else "-"
    a2x = fmt_signed(a2, "x")
    lhs = f"{a1}(x {sign_h} {abs(h)})"
    rhs = f"{a2x} {sign_b2} {abs(b2)}"
    # Independent symbolic check: solve a1(x + h) = a2*x + b2 from scratch
    # (not a restatement of the b2 = a1*(x0+h) - a2*x0 construction
    # arithmetic).
    check(solve(Eq(a1 * (x + h), a2 * x + b2), x) == [x0])
    # Misconceptions, each proved distinct from x0 by construction:
    #  1. distributed a1 onto x but not onto h (solved a1*x + h = a2*x + b2).
    #     Value is (b2 - h)/(a1 - a2); minus x0 = h*(a1 - 1)/(a1 - a2),
    #     nonzero since h != 0 and a1 >= 2.
    #  2. collected the variable terms on the wrong side: subtracted a2*x
    #     from the LEFT (moving the bracket's own x-term to the right)
    #     instead of subtracting a1*x from the right (the correct move,
    #     since the bracket's x-term stays put once expanded) -- a
    #     sign-direction slip specific to "both sides", producing the
    #     equation with (a1 - a2) negated. Value is (a1*h - b2)/(a1 - a2) =
    #     -x0; equal to x0 only when x0 == 0, excluded above.
    #  3. forgot the bracket entirely and solved a1*x = a2*x + b2 (dropped
    #     h). Value is b2/(a1 - a2); minus x0 = a1*h/(a1 - a2), nonzero since
    #     a1 >= 2 and h != 0.
    cands = [
        f"x = {rational_text(Rational(b2 - h, a1 - a2))}",
        f"x = {rational_text(Rational(a1 * h - b2, a1 - a2))}",
        f"x = {rational_text(Rational(b2, a1 - a2))}",
    ]
    return dict(
        text=f"Solve {lhs} = {rhs}",
        latex=rf"\text{{Solve }} {lhs} = {rhs}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        working=[
            f"Expand: {a1}x {sign_a1h} {abs(a1h)} = {rhs}",
            f"x = ({b2} {'-' if a1h >= 0 else '+'} {abs(a1h)}) / ({a1} - {a2}) = {x0}",
        ],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


@generator(
    "linear_equation",
    level=7,
    presets={
        "easy": dict(coef_hi=5, sol_hi=9, b_hi=10),
        "medium": dict(coef_hi=9, sol_hi=12, b_hi=20),
        "hard": dict(coef_hi=12, sol_hi=30, b_hi=50),
        # Registered so `make("linear_equation", difficulty="exam")` does not raise. The real
        # exam-tier SHAPE comes entirely from `equations-with-brackets.yaml`'s own explicit `params:
        # {kind: double_brackets}`, not from this preset's own coef_hi/sol_hi/b_hi.
        "exam": dict(coef_hi=9, sol_hi=12, b_hi=15),
    },
)
def linear_equation(
    rng,
    coef_hi=9,
    sol_hi=12,
    b_hi=20,
    a_fixed=None,
    b_zero=False,
    allow_negative=False,
    rational_b=False,
    kind="standard",
):
    if kind == "brackets":
        return _linear_equation_brackets(rng, coef_hi, sol_hi, b_hi, allow_negative)
    if kind == "both_sides":
        return _linear_equation_both_sides(rng, coef_hi, sol_hi, allow_negative)
    if kind == "brackets_leading":
        return _linear_equation_brackets_leading(rng, coef_hi, sol_hi, b_hi, allow_negative)
    if kind == "double_brackets":
        return _linear_equation_double_brackets(rng, coef_hi, sol_hi, b_hi, allow_negative)
    if kind == "division_form":
        return _linear_equation_division_form(rng, coef_hi, sol_hi, allow_negative)
    if kind == "division_two_step":
        return _linear_equation_division_two_step(rng, coef_hi, sol_hi, b_hi, allow_negative)
    if kind == "both_sides_bracket":
        return _linear_equation_both_sides_bracket(rng, coef_hi, sol_hi, b_hi, allow_negative)
    # kind == "standard" (default). Backwards: choose the solution first.
    x0 = rng.randint(-sol_hi, sol_hi) if allow_negative else rng.randint(1, sol_hi)
    a = a_fixed if a_fixed is not None else rng.randint(2, coef_hi)
    if b_zero:
        b = 0
    elif rational_b:
        den = rng.choice([2, 3, 4, 5])
        num = rng.randint(1, den - 1)
        b = Rational(num, den)
        if rng.random() < 0.5:
            b = -b
    else:
        b = rng.randint(1, b_hi)
        if rng.random() < 0.5:
            b = -b
    c = a * x0 + b
    coef_str = "" if a == 1 else str(a)
    sign = "+" if b >= 0 else "-"
    text = f"{coef_str}x = {c}" if b == 0 else f"{coef_str}x {sign} {abs(b)} = {c}"
    check(solve(Eq(a * x + b, c), x) == [x0])
    # Misconceptions: sign error moving b; forgot to divide by a; ignored b.
    slips = [
        ((c + b) * Rational(1, a), "sign-error-moving-term"),
        (Rational(c - b), "forgot-to-divide-by-coefficient"),
        (c * Rational(1, a), "ignored-constant-term"),
    ]
    cands = [Distractor(f"x = {rational_text(v)}", tag, exact_value(v)) for v, tag in slips]
    return dict(
        text=f"Solve {text}",
        latex=rf"\text{{Solve }} {text}",
        answer=x0,
        answer_text=f"x = {x0}",
        answer_latex=f"x = {x0}",
        answer_kind="number",
        working=[f"{coef_str}x = {c} {'-' if b >= 0 else '+'} {abs(b)} = {c - b}", f"x = {c - b}/{a} = {x0}"],
        distractors=pick_distractors(f"x = {x0}", cands),
    )


@generator(
    "simultaneous",
    level=8,
    presets={
        "easy": dict(sol_hi=5, coef_hi=3),
        "medium": dict(sol_hi=8, coef_hi=5),
        "hard": dict(sol_hi=12, coef_hi=9),
    },
)
def simultaneous(rng, sol_hi=8, coef_hi=5):
    # Backwards: choose (x0, y0), then coefficients with nonzero determinant.
    if coef_hi < 1:
        raise ValueError("simultaneous needs coef_hi >= 1: every coefficient must be nonzero")
    x0, y0 = rng.randint(-sol_hi, sol_hi), rng.randint(-sol_hi, sol_hi)
    a1, b1, a2, b2 = draw_until(
        rng,
        lambda r: (
            r.randint(-coef_hi, coef_hi),
            r.randint(-coef_hi, coef_hi),
            r.randint(-coef_hi, coef_hi),
            r.randint(-coef_hi, coef_hi),
        ),
        lambda c: c[0] * c[3] - c[2] * c[1] != 0 and all(c),
        what="four nonzero coefficients with a nonzero determinant",
    )
    c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0

    def side(a, b):
        return join_terms([(a, "x"), (b, "y")])

    eqs = [Eq(a1 * x + b1 * y, c1), Eq(a2 * x + b2 * y, c2)]
    check(linsolve(eqs, x, y) == FiniteSet((x0, y0)))
    e1, e2 = side(a1, b1) + f" = {c1}", side(a2, b2) + f" = {c2}"
    return dict(
        text=f"Solve the simultaneous equations: {e1} and {e2}",
        latex=rf"\begin{{cases}} {latex(eqs[0])} \ {latex(eqs[1])} \end{{cases}}",
        **xy_solution(x0, y0),
    )


def _roots_text(roots):
    return "x = " + (str(roots[0]) if len(roots) == 1 else f"{roots[0]} or x = {roots[1]}")


def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _quadratic_formula(rng, root_hi):
    # Backwards-ish: pick a,b,c so the discriminant is positive but not a
    # perfect square (roots genuinely need the formula, not factoring).
    while True:
        a_ = rng.randint(1, 3)
        b_ = rng.randint(-root_hi, root_hi)
        c_ = rng.randint(-root_hi, root_hi)
        D = b_ * b_ - 4 * a_ * c_
        if D > 0 and not _is_perfect_square(D):
            break
    expr = a_ * x**2 + b_ * x + c_
    sqrtD = math.sqrt(D)
    r_lo, r_hi = sorted([(-b_ - sqrtD) / (2 * a_), (-b_ + sqrtD) / (2 * a_)])
    ans = [round(r_lo, 2), round(r_hi, 2)]
    # Independent check: solve() agrees numerically.
    exact_roots = sorted(float(r) for r in solve(Eq(expr, 0), x))
    check(all(abs(ex - ap) < 0.01 for ex, ap in zip(exact_roots, ans)))
    # Every displayed root is formatted to exactly 2 d.p. -- the question
    # states "(2 d.p.)" and the grader derives its tolerance from the digits
    # AS WRITTEN (`mathgen/grading/typed.py`'s `_decimal_places`), so a stray "2.7"
    # (from a bare `round()`, which drops a trailing zero) would grade at the
    # wrong, looser tolerance.
    ans_text = f"x = {ans[0]:.2f} or x = {ans[1]:.2f}"
    answer_latex = rf"x = {ans[0]:.2f} \text{{ or }} x = {ans[1]:.2f}"
    # Misconceptions: sign error using +b instead of -b; forgot to double a in
    # the denominator; discriminant sign error (b^2 + 4ac).
    sign_b = sorted([(b_ - sqrtD) / (2 * a_), (b_ + sqrtD) / (2 * a_)])
    no_double_a = sorted([(-b_ - sqrtD) / a_, (-b_ + sqrtD) / a_])
    cands = [f"x = {sign_b[0]:.2f} or x = {sign_b[1]:.2f}", f"x = {no_double_a[0]:.2f} or x = {no_double_a[1]:.2f}"]
    Dwrong = b_ * b_ + 4 * a_ * c_
    if Dwrong >= 0:
        sw = math.sqrt(Dwrong)
        wrong_disc = sorted([(-b_ - sw) / (2 * a_), (-b_ + sw) / (2 * a_)])
        cands.append(f"x = {wrong_disc[0]:.2f} or x = {wrong_disc[1]:.2f}")
    return dict(
        text=f"Solve {expr} = 0 using the quadratic formula (2 d.p.)",
        latex=rf"\text{{Solve }} {latex(expr)} = 0 \ \text{{using the quadratic formula (2 d.p.)}}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=answer_latex,
        working=[f"x = (-({b_}) ± √({D})) / (2×{a_})"],
        # tolerance=GRADER_TOLERANCE: this skill's `answer: list` is graded by
        # `mathgen/grading/typed.py`'s `grade_list`, whose comparison tolerance is the
        # fixed `_LIST_TOLERANCE` (0.051), never `_decimal_places`-derived --
        # that derivation is `grade_number`'s own mechanism, for `answer:
        # number` skills, not this one. A distractor within 0.051 of a root
        # is a distractor the real grader would accept, so it must be dropped
        # here at that same tolerance (matching `zero_product_property`'s own
        # `pick_distractors(..., tolerance=GRADER_TOLERANCE)` for the same
        # `answer: list` shape).
        distractors=pick_distractors(ans_text, cands, tolerance=GRADER_TOLERANCE),
    )


def _quadratic_formula_rearrange(rng, root_hi):
    # Same discriminant-based construction as `_quadratic_formula`, but
    # displayed UNREARRANGED as x(ax + b) = -c -- the learner must expand
    # and move every term to one side before the quadratic formula even
    # applies, a genuinely different first step from `_quadratic_formula`'s
    # equation, which is already in standard form.
    while True:
        a_ = rng.randint(1, 3)
        b_ = rng.randint(-root_hi, root_hi)
        c_ = rng.randint(-root_hi, root_hi)
        D = b_ * b_ - 4 * a_ * c_
        if D > 0 and not _is_perfect_square(D):
            break
    expr = a_ * x**2 + b_ * x + c_
    sqrtD = math.sqrt(D)
    r_lo, r_hi = sorted([(-b_ - sqrtD) / (2 * a_), (-b_ + sqrtD) / (2 * a_)])
    ans = [round(r_lo, 2), round(r_hi, 2)]
    exact_roots = sorted(float(r) for r in solve(Eq(expr, 0), x))
    check(all(abs(ex - ap) < 0.01 for ex, ap in zip(exact_roots, ans)))
    ans_text = f"x = {ans[0]:.2f} or x = {ans[1]:.2f}"
    answer_latex = rf"x = {ans[0]:.2f} \text{{ or }} x = {ans[1]:.2f}"
    b_sign = "+" if b_ >= 0 else "-"
    lhs = f"x({fmt_signed(a_, 'x')} {b_sign} {abs(b_)})"
    rhs = -c_
    sign_b = sorted([(b_ - sqrtD) / (2 * a_), (b_ + sqrtD) / (2 * a_)])
    no_double_a = sorted([(-b_ - sqrtD) / a_, (-b_ + sqrtD) / a_])
    cands = [f"x = {sign_b[0]:.2f} or x = {sign_b[1]:.2f}", f"x = {no_double_a[0]:.2f} or x = {no_double_a[1]:.2f}"]
    Dwrong = b_ * b_ + 4 * a_ * c_
    if Dwrong >= 0:
        sw = math.sqrt(Dwrong)
        wrong_disc = sorted([(-b_ - sw) / (2 * a_), (-b_ + sw) / (2 * a_)])
        cands.append(f"x = {wrong_disc[0]:.2f} or x = {wrong_disc[1]:.2f}")
    return dict(
        text=f"Solve {lhs} = {rhs} using the quadratic formula (2 d.p.)",
        latex=rf"\text{{Solve }} {a_}x^2 {b_sign} {abs(b_)}x = {rhs} \ \text{{using the quadratic formula (2 d.p.)}}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=answer_latex,
        working=[f"Expand: {a_}x^2 {b_sign} {abs(b_)}x = {rhs} → {expr} = 0", f"x = (-({b_}) ± √({D})) / (2×{a_})"],
        distractors=pick_distractors(ans_text, cands, tolerance=GRADER_TOLERANCE),
    )


def _quadratic_vertex(rng, root_hi):
    # Backwards: choose the vertex (h, k) and a leading coefficient first.
    h = rng.randint(-root_hi, root_hi)
    k = rng.randint(-root_hi, root_hi)
    a_ = rng.randint(1, 3) * rng.choice([1, -1])
    expr = expand(a_ * (x - h) ** 2 + k)
    # Independent check: complete the square via x = -b/2a.
    b_coef = expr.coeff(x, 1)
    a_coef = expr.coeff(x, 2)
    vx = -b_coef / (2 * a_coef)
    vy = expr.subs(x, vx)
    check(vx == h and vy == k)
    ans_text = f"({h}, {k})"
    # Misconceptions: forgot to negate h (used +b/2a); sign error on k only;
    # both signs flipped.
    cands = [f"({-h}, {k})", f"({h}, {-k})", f"({-h}, {-k})"]
    # Fallbacks for a vertex on an axis or the origin, where the sign slips
    # above collapse onto the answer: off-by-one coordinates.
    cands += [f"({h}, {k + 1})", f"({h + 1}, {k})", f"({h}, {k - 1})"]
    return dict(
        text=f"Find the turning point (vertex) of y = {expr}",
        latex=rf"\text{{Vertex of }} y = {latex(expr)}",
        answer={"x": h, "y": k},
        answer_text=ans_text,
        answer_latex=ans_text,
        working=[f"x = -b/(2a) = {vx}", f"y = f({vx}) = {vy}"],
        distractors=pick_distractors(ans_text, cands),
    )


def _roots_text_rational(roots):
    # Like `_roots_text` but formats each root with `rational_text` -- the
    # non-monic "divided b only" misconception below can land on a root with
    # denominator 2, which `_roots_text`'s bare `str()` was never asked to
    # handle (the monic `factor`/`formula`/`vertex` branches never call this).
    strs = [rational_text(r) for r in roots]
    return "x = " + (strs[0] if len(strs) == 1 else f"{strs[0]} or x = {strs[1]}")


def _nonmonic_expr(a, r1, r2):
    # The expanded a*(x - r1)*(x - r2) -- kept as its own step (rather than
    # inlined into `_quadratic_factor_nonmonic`) so a test can monkeypatch it
    # to a corrupted value and prove the in-body independent check below
    # actually fires, the same discipline `exd_compose_example._part_engineering`
    # uses for its own firing test.
    return expand(a * (x - r1) * (x - r2))


def _quadratic_factor_nonmonic(rng, root_hi):
    # Target: non-monic a*x^2 + b*x + c = 0, a in 2..6, both roots integer. Backwards, same
    # discipline as the monic `factor` branch: choose integer roots first, then an integer leading
    # coefficient -- a*(x - r1)*(x - r2) expanded always has integer middle/constant coefficients
    # since a is just an integer multiplier of the monic expansion.
    r1 = rng.randint(-root_hi, root_hi)
    r2 = rng.randint(-root_hi, root_hi)
    a = rng.randint(2, 6)
    expr = _nonmonic_expr(a, r1, r2)
    roots = sorted(set([r1, r2]))
    # Independent check: sympy.solve on the expanded equation, never trusting
    # the backwards construction that built `expr`.
    check(sorted(solve(Eq(expr, 0), x)) == roots)
    # Text is built by hand with `join_terms` (caret "x^2", no "*") rather
    # than interpolating sympy's own `str(expr)` ("2*x**2 - 4*x"), which
    # trips `find_text_defects`'s double_star/star_operator patterns -- the
    # monic `factor`/`formula`/`vertex` branches above already carry that
    # legacy rendering (out of scope: EXISTING branches are untouched), but
    # this NEW branch has no such inheritance to preserve, so it uses the
    # `_build`-module convention (`t2_factorise_quadratic_build.py`'s own
    # `join_terms([(a_coef, "x^2"), ...])`) instead. `b_coef`/`c_coef` are
    # `expr`'s own coefficients, recomputed directly rather than parsed back
    # out of `expr` -- b = -a(r1+r2), c = a*r1*r2, from expanding
    # a*(x-r1)*(x-r2) by hand.
    b_coef = -a * (r1 + r2)
    c_coef = a * r1 * r2
    lhs_text = join_terms([(a, "x^2"), (b_coef, "x"), (c_coef, "")])
    ans_text = _roots_text(roots)
    # Roughly half the seeds flip the equation to "0 = ..." rather than
    # "... = 0" (so both written forms occur) --
    # a genuine token-order structural difference, not decoration.
    flipped = rng.random() < 0.5
    if flipped:
        text = f"0 = {lhs_text}"
        latex_text = rf"0 = {latex(expr)}"
    else:
        text = f"Solve {lhs_text} = 0"
        latex_text = rf"\text{{Solve }} {latex(expr)} = 0"
    # Misconceptions: sign-flip on both roots, sign-flip on only one root
    # (mirrors the monic `factor` branch's two candidates), plus one slip
    # specific to non-monic: a learner who ignores the leading coefficient
    # `a` entirely and "solves" x^2 + b*x + c = 0 using the SAME (undivided)
    # b and c the non-monic equation displays, instead of dividing the whole
    # equation through by `a` first. Offered only when that wrong quadratic
    # has a real, rational pair of roots (perfect-square discriminant) --
    # same conditional discipline as `_quadratic_formula`'s own `Dwrong`
    # candidate above.
    cands = [_roots_text(sorted(set([-r1, -r2])))]
    ignores_a_b = b_coef
    ignores_a_c = c_coef
    Dw = ignores_a_b * ignores_a_b - 4 * ignores_a_c
    if Dw >= 0 and _is_perfect_square(Dw):
        sqrt_dw = math.isqrt(Dw)
        wrong_roots = sorted(
            set(
                [
                    Rational(-ignores_a_b - sqrt_dw, 2),
                    Rational(-ignores_a_b + sqrt_dw, 2),
                ]
            )
        )
        cands.append(_roots_text_rational(wrong_roots))
    cands.append(_roots_text(sorted(set([-r1, r2]))))
    cands.append(_roots_text(sorted(set([r1, -r2]))))
    # Fallbacks for roots the sign flips cannot move (r1 = r2 = 0, r1 = -r2):
    # both roots shifted by one.
    cands += [_roots_text(sorted(set([r1 + 1, r2 + 1]))), _roots_text(sorted(set([r1 - 1, r2 - 1])))]
    return dict(
        text=text,
        latex=latex_text,
        answer=roots,
        answer_text=ans_text,
        answer_latex="x = " + (str(roots[0]) if len(roots) == 1 else rf"{roots[0]},\; x = {roots[1]}"),
        working=[f"Factorise: {a}(x {'-' if r1 >= 0 else '+'} {abs(r1)})(x {'-' if r2 >= 0 else '+'} {abs(r2)}) = 0"],
        distractors=pick_distractors(ans_text, cands),
    )


@generator(
    "quadratic",
    level=9,
    presets={
        "easy": dict(root_hi=5),
        "medium": dict(root_hi=9),
        "hard": dict(root_hi=15),
        # Registered so `make("quadratic", difficulty="exam")` does not raise (`mathgen.core.make`
        # requires any `difficulty=` to be a known preset whenever the topic has ANY presets at all,
        # like the same fix for `differential_equation` in calculus.py) -- the real exam-tier
        # SHAPE comes entirely from `solve-quadratics-factorable.yaml`'s own explicit `params:
        # {kind: factor_nonmonic}`, not from this preset's own root_hi, exactly like
        # "easy"/"medium"/"hard" above are never what makes any OTHER skill's sources look the way
        # they do either.
        "exam": dict(root_hi=9),
    },
)
def quadratic(rng, root_hi=9, kind="factor"):
    if kind == "formula":
        return _quadratic_formula(rng, root_hi)
    if kind == "formula_rearrange":
        return _quadratic_formula_rearrange(rng, root_hi)
    if kind == "vertex":
        return _quadratic_vertex(rng, root_hi)
    if kind == "factor_nonmonic":
        return _quadratic_factor_nonmonic(rng, root_hi)
    # kind == "factor" (default). Backwards: choose integer roots, expand.
    r1 = rng.randint(-root_hi, root_hi)
    r2 = rng.randint(-root_hi, root_hi)
    expr = expand((x - r1) * (x - r2))
    roots = sorted(set([r1, r2]))
    check(sorted(solve(Eq(expr, 0), x)) == roots)
    ans_text = _roots_text(roots)
    # Misconceptions: sign flip when factorising (both roots, or just one).
    slips = [
        ([-r1, -r2], "sign-flip-both-roots"),
        ([-r1, r2], "sign-flip-one-root"),
        ([r1, -r2], "sign-flip-one-root"),
    ]
    # x^2 - k^2 = 0 is the one the sign flips cannot move: the classic slip there is the positive root alone.
    if r1 == -r2 != 0:
        slips.append(([abs(r1)], "missed-negative-root"))
    # Fallbacks for a double root at 0: both roots shifted by one, which stands for no particular error.
    slips += [([r1 + 1, r2 + 1], None), ([r1 - 1, r2 - 1], None)]
    cands = [Distractor(_roots_text(sorted(set(v))), tag, sorted(set(v))) for v, tag in slips]
    return dict(
        text=f"Solve {expr} = 0",
        latex=rf"\text{{Solve }} {latex(expr)} = 0",
        answer=roots,
        answer_text=ans_text,
        answer_kind="list",
        answer_latex="x = " + (str(roots[0]) if len(roots) == 1 else rf"{roots[0]},\; x = {roots[1]}"),
        working=[f"Factorise: (x {'-' if r1 >= 0 else '+'} {abs(r1)})(x {'-' if r2 >= 0 else '+'} {abs(r2)}) = 0"],
        distractors=pick_distractors(ans_text, cands),
    )


@generator(
    "function_evaluation",
    level=10,
    presets={
        "easy": dict(coef_hi=3, arg_hi=4),
        "medium": dict(coef_hi=5, arg_hi=6),
        "hard": dict(coef_hi=9, arg_hi=9),
        "linear": dict(a_lo=0),
    },
)
def function_evaluation(rng, coef_hi=5, arg_hi=6, a_lo=1):
    a = rng.randint(a_lo, coef_hi)
    b = rng.randint(-coef_hi, coef_hi)
    c = rng.randint(-9, 9)
    n = rng.randint(-arg_hi, arg_hi)
    f = a * x**2 + b * x + c
    ans = a * n**2 + b * n + c
    check(f.subs(x, n) == ans)
    # Misconceptions. The sign slips are only errors a learner makes with a negative input: (-n)^2 read as -(n^2),
    # and b(-n) read as bn. With any input, squaring can be done as doubling, and c can be dropped.
    slips = []
    if n < 0:
        slips.append((-a * n**2 + b * n + c, "sign-on-power"))
    slips += [(a * 2 * n + b * n + c, "squared-as-doubled"), (a * n**2 + b * n, "dropped-constant-term")]
    if n < 0:
        slips.append((a * n**2 - b * n + c, "sign-error-linear-term"))
    # Fallbacks for a degenerate draw (a = 0, b = 0 or n = 0) where the slips
    # above all equal the answer: off by one, which stands for no particular error.
    slips += [(ans + 1, None), (ans - 1, None)]
    cands = [Distractor(f"f({n}) = {v}", tag, int(v)) for v, tag in slips]
    return dict(
        text=f"f(x) = {f}, find f({n})",
        latex=rf"f(x) = {latex(f)},\ \text{{find }} f({n})",
        answer=ans,
        answer_text=f"f({n}) = {ans}",
        answer_latex=f"f({n}) = {ans}",
        answer_kind="number",
        working=[
            f"f({n}) = {a}({n})^2 {'+' if b >= 0 else '-'} {abs(b)}({n}) {'+' if c >= 0 else '-'} {abs(c)} = {ans}"
        ],
        distractors=pick_distractors(f"f({n}) = {ans}", cands),
    )
