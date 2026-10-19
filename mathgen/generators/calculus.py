"""Levels 14-18: differentiation, integration, limits, ODEs, partial derivatives."""

from sympy import (
    symbols,
    Function,
    diff,
    integrate,
    limit,
    latex,
    expand,
    Eq,
    sin,
    cos,
    oo,
    Rational,
    dsolve,
    checkodesol,
    solve,
    Piecewise,
    simplify,
    together,
    exp,
)

from ..answers import Distractor
from ..core import (
    generator,
    check,
    pick_distractors,
    rational_text,
    fmt_signed,
    check_equivalent,
    check_distinct,
    latex_answer,
    phrase,
    join_terms,
)

x, y = symbols("x y")


def _rand_terms(rng, terms=3, coef_hi=7, min_pow=1, max_pow=5):
    """Random polynomial as an explicit [(coef, power), ...] term list so
    misconception distractors can be computed exactly."""
    powers = rng.sample(range(min_pow, max_pow + 1), k=min(terms, max_pow - min_pow + 1))
    return [(rng.randint(1, coef_hi) * rng.choice([1, -1]), p) for p in sorted(powers, reverse=True)]


def _poly(ts):
    return sum(c * x**p for c, p in ts)


@generator(
    "differentiation",
    level=14,
    presets={
        "easy": dict(kinds=("polynomial",), max_pow=3),
        "medium": dict(kinds=("polynomial", "chain"), max_pow=5),
        "hard": dict(kinds=("polynomial", "chain", "trig"), max_pow=6),
    },
)
def differentiation(rng, kinds=("polynomial", "chain", "trig"), max_pow=5):
    kind = rng.choice(list(kinds))
    if kind == "polynomial":
        ts = _rand_terms(rng, terms=3, max_pow=max_pow)
        f = _poly(ts) + rng.randint(-9, 9)
        ans = diff(f, x)
        # Independent check: integrating the derivative recovers f up to a constant.
        check(expand(integrate(ans, x) - f).is_number)
        # Misconceptions: forgot to decrement the power; forgot to multiply by it;
        # integrated instead of differentiating.
        forgot_dec = sum(c * p * x**p for c, p in ts)
        forgot_mult = sum(c * x ** (p - 1) for c, p in ts)
        integrated = expand(integrate(f, x))
        return dict(
            text=f"Differentiate {f} with respect to x",
            latex=rf"\frac{{d}}{{dx}}\left({latex(f)}\right)",
            answer=str(ans),
            answer_text=str(ans),
            answer_latex=latex(ans),
            distractors=pick_distractors(str(ans), [str(forgot_dec), str(forgot_mult), str(integrated)]),
        )
    if kind == "chain":
        # d/dx (ax + b)^n = n*a*(ax + b)^(n-1)
        n = rng.randint(2, 5)
        a = rng.randint(2, 5)
        b = rng.randint(-9, 9) or 1
        f = (a * x + b) ** n
        ans = n * a * (a * x + b) ** (n - 1)
        check(expand(diff(f, x)) == expand(ans))
        # Misconceptions: forgot the inner derivative; differentiated inner only.
        cands = [n * (a * x + b) ** (n - 1), a * (a * x + b) ** n]
        b_sign = "+" if b >= 0 else "-"
        return dict(
            text=f"Differentiate ({a}*x {b_sign} {abs(b)})**{n} with respect to x",
            latex=rf"\frac{{d}}{{dx}}\left({latex(f)}\right)",
            answer=str(ans),
            answer_text=str(ans),
            answer_latex=latex(ans),
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
        )
    if kind == "chain_quadratic":
        # d/dx (x^2 + c)^n = n*(x^2 + c)^(n-1) * 2x -- the inner function is
        # QUADRATIC, so its own derivative (2x) is itself a power-rule step,
        # not a constant. Genuinely different from the "chain" branch above,
        # whose inner function ax + b always has a constant derivative.
        n = rng.randint(2, 4)
        c_ = rng.randint(-9, 9) or 1
        inner = x**2 + c_
        f = inner**n
        ans = expand(n * inner ** (n - 1) * 2 * x)
        check(expand(diff(f, x)) == ans)
        # Misconceptions: forgot the inner derivative entirely (treated the
        # inner function as if its derivative were 1); used the LINEAR
        # chain-rule pattern and forgot to differentiate the x^2 term (used
        # a constant "inner derivative" of 1 instead of 2x); differentiated
        # only the inner function and stopped there.
        forgot_inner = n * inner ** (n - 1)
        used_linear_pattern = n * inner ** (n - 1) * x
        inner_only = 2 * x
        cands = [forgot_inner, used_linear_pattern, inner_only]
        check_distinct(ans, *cands)
        c_sign = "+" if c_ >= 0 else "-"
        return dict(
            text=f"Differentiate (x**2 {c_sign} {abs(c_)})**{n} with respect to x",
            latex=rf"\frac{{d}}{{dx}}\left({latex(f)}\right)",
            answer=str(ans),
            answer_text=str(ans),
            answer_latex=latex(ans),
            working=[
                f"inner = x^2 {c_sign} {abs(c_)}, d(inner)/dx = 2x",
                f"n·(inner)^(n-1)·2x = {n}·({inner})^{n - 1}·2x = {ans}",
            ],
            distractors=pick_distractors(str(ans), [str(c) for c in cands]),
        )
    if kind == "product_rule":
        # f = (a1*x + b1)(a2*x^2 + b2*x + c2); check by expand-then-diff (a
        # route independent of the u'v + uv' formula used to build `ans`).
        a1, b1 = rng.randint(1, 5), rng.randint(-9, 9)
        a2, b2, c2 = rng.randint(1, 5), rng.randint(-9, 9), rng.randint(-9, 9)
        u, v = a1 * x + b1, a2 * x**2 + b2 * x + c2
        up, vp = diff(u, x), diff(v, x)
        ans = expand(up * v + u * vp)
        check(expand(diff(expand(u * v), x)) == ans)
        # Misconceptions: multiplied the derivatives instead of the product
        # rule; differentiated only one factor and left the other as-is.
        cands = [expand(up * vp), expand(up * v), expand(u * vp)]
        return dict(
            text=f"Differentiate ({u})({v}) with respect to x",
            latex=rf"\frac{{d}}{{dx}}\left(({latex(u)})({latex(v)})\right)",
            answer=str(ans),
            answer_text=str(ans),
            answer_latex=latex(ans),
            working=[f"u = {u}, v = {v}", f"u'v + uv' = ({up})({v}) + ({u})({vp}) = {ans}"],
            distractors=pick_distractors(str(ans), [str(c_) for c_ in cands]),
        )
    if kind == "quotient_rule":
        # f = (a1*x + b1) / (a2*x + b2); check via simplify(diff(f) - formula).
        # Re-draw if u and v are scalar multiples of each other (a1*b2 ==
        # a2*b1): then u/v is a constant and the "quotient rule" question
        # degenerates to a derivative of 0, which teaches nothing.
        while True:
            a1, b1 = rng.randint(1, 6), rng.randint(-9, 9)
            a2, b2 = rng.randint(1, 6), rng.randint(-9, 9) or 1
            if a1 * b2 - a2 * b1 != 0:
                break
        u, v = a1 * x + b1, a2 * x + b2
        up, vp = diff(u, x), diff(v, x)
        ans = (up * v - u * vp) / v**2
        check_equivalent(diff(u / v, x), ans)
        # Misconceptions: numerator order swapped (uv' - u'v); forgot to
        # square the denominator; used the product rule instead.
        swapped = (u * vp - up * v) / v**2
        no_square = (up * v - u * vp) / v
        product_instead = expand(up * vp)
        cands = [str(swapped), str(no_square), str(product_instead)]
        return dict(
            text=f"Differentiate ({u}) / ({v}) with respect to x",
            latex=rf"\frac{{d}}{{dx}}\left(\frac{{{latex(u)}}}{{{latex(v)}}}\right)",
            answer=str(together(ans)),
            answer_text=str(together(ans)),
            answer_latex=latex(together(ans)),
            working=[f"u = {u}, v = {v}", f"(u'v - uv') / v² = (({up})({v}) - ({u})({vp})) / ({v})²"],
            distractors=pick_distractors(str(together(ans)), cands),
        )
    if kind == "second_derivative":
        ts = _rand_terms(rng, terms=3, max_pow=max_pow, min_pow=2)
        f = _poly(ts) + rng.randint(-9, 9)
        sym_second = diff(f, x, 2)
        # Independent check: apply the power rule twice by hand, term by term.
        manual_second = expand(sum(c * p * (p - 1) * x ** (p - 2) for c, p in ts))
        check(expand(sym_second - manual_second) == 0)
        # Misconceptions: differentiated only once; applied the power rule
        # once more without decrementing again (forgot the second drop).
        first = diff(f, x)
        forgot_second_drop = sum(c * p * x ** (p - 1) for c, p in ts)
        return dict(
            text=f"Find the second derivative of {f} with respect to x",
            latex=rf"\frac{{d^2}}{{dx^2}}\left({latex(f)}\right)",
            answer=str(sym_second),
            answer_text=str(sym_second),
            answer_latex=latex(sym_second),
            working=[f"f'(x) = {first}", f"f''(x) = {sym_second}"],
            distractors=pick_distractors(str(sym_second), [str(first), str(forgot_second_drop)]),
        )
    if kind == "stationary_points":
        # Backwards: choose the stationary x-values, build f' from them, then
        # f as its antiderivative.
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        while r2 == r1:
            r2 = rng.randint(-6, 6)
        k = rng.randint(1, 3)
        fprime = expand(k * (x - r1) * (x - r2))
        f = integrate(fprime, x)
        roots = sorted([r1, r2])
        check(sorted(solve(Eq(diff(f, x), 0), x)) == roots)
        ans_text = f"x = {roots[0]} and x = {roots[1]}"
        # Misconceptions: sign-flipped roots; solved f(x) = 0 instead of
        # f'(x) = 0; found only one stationary point.
        cands = [f"x = {-roots[0]} and x = {-roots[1]}", f"x = {roots[0]}"]
        try:
            wrong_roots = sorted(r for r in solve(Eq(f, 0), x) if r.is_real)
            if len(wrong_roots) == 2:
                cands.insert(1, f"x = {wrong_roots[0]} and x = {wrong_roots[1]}")
        except Exception:
            pass
        return dict(
            text=f"Find the x-coordinates of the stationary points of f(x) = {f}",
            latex=rf"\text{{Stationary points of }} f(x) = {latex(f)}",
            answer=roots,
            answer_text=ans_text,
            answer_latex=rf"x = {roots[0]} \text{{ and }} x = {roots[1]}",
            working=[f"f'(x) = {fprime}", f"f'(x) = 0 ⇒ x = {roots[0]} or x = {roots[1]}"],
            # unordered: the answer is a SET of stationary points and grade_list
            # accepts them in any order, so the sign-flip candidate was the
            # right answer rewritten whenever the roots were symmetric.
            distractors=pick_distractors(ans_text, cands, unordered=True),
        )
    # trig: d/dx (a sin(bx) + c cos(dx))
    a, c = rng.randint(1, 5), rng.randint(1, 5)
    b, d = rng.randint(1, 4), rng.randint(1, 4)
    f = a * sin(b * x) + c * cos(d * x)
    ans = a * b * cos(b * x) - c * d * sin(d * x)
    check(expand(diff(f, x) - ans) == 0)
    # Misconceptions: dropped the inner-derivative factors; sign error on the
    # derivative of cos.
    cands = [a * cos(b * x) - c * sin(d * x), a * b * cos(b * x) + c * d * sin(d * x)]
    return dict(
        text=f"Differentiate {f} with respect to x",
        latex=rf"\frac{{d}}{{dx}}\left({latex(f)}\right)",
        answer=str(ans),
        answer_text=str(ans),
        answer_latex=latex(ans),
        distractors=pick_distractors(str(ans), [str(c_) for c_ in cands]),
    )


@generator(
    "integration",
    level=15,
    presets={
        "easy": dict(kinds=("polynomial",), max_pow=3),
        "medium": dict(kinds=("polynomial",), max_pow=5),
        "hard": dict(kinds=("polynomial", "trig"), max_pow=6),
    },
)
def integration(rng, kinds=("polynomial", "trig"), max_pow=5):
    kind = rng.choice(list(kinds))
    if kind == "polynomial":
        # Backwards: choose the antiderivative F first; the question is F'.
        ts = _rand_terms(rng, terms=3, min_pow=2, max_pow=max_pow)
        F = _poly(ts)
        f = expand(diff(F, x))
        check(expand(integrate(f, x) - F).is_number)
        # Misconceptions: raised the power without dividing (exactly x*f for a
        # polynomial); differentiated instead of integrating.
        cands = [f"{expand(x * f)} + C", f"{expand(diff(f, x))} + C"]
        return dict(
            text=f"Find ∫ ({f}) dx",
            latex=rf"\int \left({latex(f)}\right)\, dx",
            answer=str(F),
            answer_text=f"{F} + C",
            answer_latex=latex(F) + " + C",
            distractors=pick_distractors(f"{F} + C", cands),
        )
    if kind == "rewrite_condition":
        # A term that must first be REWRITTEN as a power (c/x^n = c*x^-n, or
        # c*sqrt(x) = c*x^(1/2)) before the reverse power rule applies, and a
        # stated condition F(1) = k fixes the constant of integration -- so
        # the answer is a fully-determined particular antiderivative, never
        # "+ C". Genuinely different from the "polynomial" branch above (no
        # rewriting step, indefinite answer).
        style = rng.choice(["neg_power", "root"])
        c_ = rng.randint(2, 9)
        if style == "neg_power":
            n = rng.randint(2, 4)
            f = c_ * x ** (-n)
            f_display = f"{c_}/x^{n}"
        else:
            n = 1  # unused for "root", kept so wrong_exponent below has one shape
            f = c_ * x ** Rational(1, 2)
            f_display = f"{c_}*sqrt(x)"
        F_indef = integrate(f, x)
        k_val = rng.randint(-9, 9)
        C = k_val - F_indef.subs(x, 1)
        while C == 0:
            k_val = rng.randint(-9, 9)
            C = k_val - F_indef.subs(x, 1)
        F = expand(F_indef + C)
        check(expand(diff(F, x) - f) == 0)
        check_equivalent(F.subs(x, 1), k_val)
        # Misconceptions: ignored the condition entirely (C = 0, an
        # indefinite-style answer with no constant added); raised the power
        # without dividing by the new exponent (kept x's ORIGINAL exponent
        # instead of rewriting it first); used the wrong sign/fraction on the
        # rewritten exponent (treated c/x^n as c*x^n, or c*sqrt(x) as c*x^-1/2).
        F_zero_c = F_indef
        raised_no_divide = expand(x * f) + C
        if style == "neg_power":
            wrong_exponent = integrate(c_ * x**n, x) + C
        else:
            wrong_exponent = integrate(c_ * x ** Rational(-1, 2), x) + C
        cands = [F_zero_c, raised_no_divide, wrong_exponent]
        check_distinct(F, *cands)
        return dict(
            text=f"Find F(x) if F'(x) = {f_display} and F(1) = {k_val}",
            latex=rf"F'(x) = {latex(f)},\ F(1) = {k_val}.\ \text{{Find }} F(x)",
            answer=str(F),
            answer_text=str(F),
            answer_latex=latex(F),
            working=[f"∫({f_display}) dx = {F_indef} + C", f"F(1) = {k_val} ⇒ C = {C}", f"F(x) = {F}"],
            distractors=pick_distractors(str(F), [str(c) for c in cands]),
        )
    if kind == "definite":
        ts = _rand_terms(rng, terms=2, min_pow=1, max_pow=max_pow)
        f = _poly(ts)
        a_, b_ = sorted(rng.sample(range(-4, 5), 2))

        # Independent check: evaluate the term-by-term antiderivative formula
        # at the bounds by hand, rather than relying on integrate() twice.
        def _F(val):
            return sum(Rational(c, p + 1) * val ** (p + 1) for c, p in ts)

        manual = _F(b_) - _F(a_)
        ans = integrate(f, (x, a_, b_))
        check(ans == manual)
        ans_out = int(ans) if ans.q == 1 else {"p": int(ans.p), "q": int(ans.q)}
        ans_text = rational_text(ans)
        # Misconceptions: forgot to subtract F(a) (only evaluated at b);
        # swapped the bounds (sign flip); evaluated at a instead of b.
        only_b = _F(b_)
        swapped = _F(a_) - _F(b_)
        only_a = _F(a_)
        cands = [rational_text(only_b), rational_text(swapped), rational_text(only_a)]
        return dict(
            text=f"Evaluate the definite integral of ({f}) dx from x = {a_} to x = {b_}",
            latex=rf"\int_{{{a_}}}^{{{b_}}} \left({latex(f)}\right)\, dx",
            answer=ans_out,
            answer_text=ans_text,
            answer_latex=ans_text,
            working=[f"F(x) = {_poly([(Rational(c, p + 1), p + 1) for c, p in ts])}", f"F({b_}) - F({a_}) = {ans_text}"],
            distractors=pick_distractors(ans_text, cands),
        )
    if kind == "substitution":
        # Backwards: F = k*(c*x + d)^(n+1)/(n+1); the question is F' in
        # factored (substitution-friendly) form k*c*(c*x + d)^n.
        n = rng.randint(2, 4)
        c_ = rng.randint(2, 5)
        d_ = rng.randint(-9, 9) or 1
        k = rng.randint(1, 4)
        inner = c_ * x + d_
        F = k * inner ** (n + 1) / (n + 1)
        f = k * c_ * inner**n
        check(expand(diff(F, x) - f) == 0)
        # Misconceptions: forgot to divide by the inner derivative c (treated
        # the coefficient as if it were just k); differentiated instead of
        # integrating; dropped the overall coefficient entirely.
        forgot_c_divide = k * c_ * inner ** (n + 1) / (n + 1)
        differentiated = diff(f, x)
        dropped_coef = inner ** (n + 1) / (n + 1)
        cands = [f"{forgot_c_divide} + C", f"{differentiated} + C", f"{dropped_coef} + C"]
        k_dot = "" if k == 1 else f"{k}·"
        return dict(
            text=f"Find ∫ ({f}) dx using the substitution u = {inner}",
            latex=rf"\int \left({latex(f)}\right)\, dx,\ u = {latex(inner)}",
            answer=str(F),
            answer_text=f"{F} + C",
            answer_latex=latex(F) + " + C",
            working=[f"u = {inner}, du = {c_} dx", f"∫ {k_dot}u^{n} du = {k_dot}u^{n + 1}/{n + 1} = {F}"],
            distractors=pick_distractors(f"{F} + C", cands),
        )
    if kind == "nonlinear":
        # u-substitution with a NON-LINEAR inner function u = x^2 + d: the
        # integrand is k*x*(x^2+d)^n, and du/dx = 2x -- so the leading x
        # factor is (up to the constant 2) exactly du, not a coincidence to
        # ignore. Genuinely different from "substitution" above, whose inner
        # function is always LINEAR (c*x + d).
        n = rng.randint(2, 4)
        d_ = rng.randint(1, 6)
        k_ = rng.randint(1, 4)
        u_expr = x**2 + d_
        f = k_ * x * u_expr**n
        F = k_ * u_expr ** (n + 1) / (2 * (n + 1))
        check(expand(diff(F, x) - f) == 0)
        check_equivalent(diff(F, x), f)
        # Misconceptions: forgot to divide by the du/dx factor (2) entirely
        # (u^(n+1)/(n+1) instead of u^(n+1)/(2(n+1)) -- note this is now the
        # SAME value `2*F`, since F itself is built in u-form, so it is
        # listed only once); kept the power at n instead of raising it to
        # n+1 (an off-by-one on the reverse power rule, dividing by 2n
        # instead of 2(n+1)).
        forgot_half = k_ * u_expr ** (n + 1) / (n + 1)
        wrong_power = k_ * u_expr**n / (2 * n)
        cands = [forgot_half, wrong_power]
        check_distinct(F, *cands)
        return dict(
            text=f"Find ∫ ({f}) dx using the substitution u = {u_expr}",
            latex=rf"\int \left({latex(f)}\right)\, dx,\ u = {latex(u_expr)}",
            answer=str(F),
            answer_text=f"{F} + C",
            answer_latex=latex(F) + " + C",
            working=[f"u = {u_expr}, du = 2x dx", f"∫ k·x·u^{n} dx = (k/2)∫ u^{n} du = {F}"],
            distractors=pick_distractors(f"{F} + C", [f"{c} + C" for c in cands]),
        )
    if kind == "phase":
        # A phase-shifted argument k*sin(ax+b) / k*cos(ax+b) with b != 0, so
        # the antiderivative also carries the phase -- genuinely different
        # from the trig fallback below (whose argument is always bx, b=0).
        a_ = rng.randint(2, 5)
        b_ = rng.randint(1, 9) * rng.choice([1, -1])
        fn = rng.choice([sin, cos])
        k_ = rng.randint(1, 5)
        inner = a_ * x + b_
        f = k_ * fn(inner)
        F = integrate(f, x)
        check(expand(diff(F, x) - f) == 0)
        # Misconceptions: dropped the phase entirely (integrated k*fn(ax)
        # instead, as if b were 0); forgot to divide by the inner
        # coefficient a (treated a as 1); wrong sign on the sin/cos family.
        dropped_phase = integrate(k_ * fn(a_ * x), x)
        forgot_a = -k_ * cos(inner) if fn == sin else k_ * sin(inner)
        wrong_sign = k_ * cos(inner) if fn == sin else -k_ * sin(inner)
        cands = [dropped_phase, forgot_a, wrong_sign]
        check_distinct(F, *cands)
        b_sign = "+" if b_ >= 0 else "-"
        fn_name = "sin" if fn == sin else "cos"
        return dict(
            text=f"Find ∫ {k_}*{fn_name}({a_}*x {b_sign} {abs(b_)}) dx",
            latex=rf"\int {k_}{latex(fn(inner))}\, dx",
            answer=str(F),
            answer_text=f"{F} + C",
            answer_latex=latex(F) + " + C",
            distractors=pick_distractors(f"{F} + C", [f"{c} + C" for c in cands]),
        )
    # Backwards trig: F = a*sin(bx) or a*cos(bx); question is F'.
    a = rng.randint(1, 5)
    b = rng.randint(2, 5)
    F = rng.choice([a * sin(b * x), a * cos(b * x)])
    f = diff(F, x)
    check(expand(integrate(f, x) - F) == 0)
    # Misconceptions: forgot to divide by b; wrong sign on the sin/cos family.
    cands = [f"{b * F} + C", f"{-F} + C"]
    return dict(
        text=f"Find ∫ ({f}) dx",
        latex=rf"\int \left({latex(f)}\right)\, dx",
        answer=str(F),
        answer_text=f"{F} + C",
        answer_latex=latex(F) + " + C",
        distractors=pick_distractors(f"{F} + C", cands),
    )


@generator(
    "limits",
    level=16,
    presets={
        "easy": dict(kinds=("removable",)),
        "medium": dict(kinds=("removable", "infinity")),
        "hard": dict(kinds=("removable", "infinity")),
    },
)
def limits(rng, kinds=("removable", "infinity")):
    kind = rng.choice(list(kinds))
    if kind == "removable":
        # Backwards: build a removable singularity at x = a with known limit.
        a = rng.randint(-5, 5)
        b = rng.randint(-5, 5)
        while b == a:
            b = rng.randint(-5, 5)
        num = expand((x - a) * (x - b))
        den = x - a
        ans = a - b
        check(limit(num / den, x, a) == ans)
        # Misconceptions: substituted and stopped at 0/0 ("undefined"); saw the
        # zero numerator and answered 0; sign error when cancelling.
        cands = [
            Distractor("undefined", "stopped-at-undefined"),
            Distractor("0", "answered-zero", 0),
            Distractor(str(b - a), "sign-error-cancelling", b - a),
        ]
        return dict(
            text=f"Find the limit as x → {a} of ({num}) / ({den})",
            latex=rf"\lim_{{x \to {a}}} \frac{{{latex(num)}}}{{{latex(den)}}}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            answer_kind="number",
            working=[f"({num})/({den}) = x - ({b}) for x ≠ {a}", f"limit = {a} - ({b}) = {ans}"],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "quad_over_quad":
        # A QUADRATIC over a quadratic (both factor with the same removable
        # root x=a), not quadratic-over-LINEAR like "removable" above -- the
        # learner must cancel the common factor and is left with a genuine
        # RATIO of two remaining linear factors, not a bare difference.
        while True:
            a = rng.randint(-5, 5)
            c = rng.randint(-5, 5)
            if c == a:
                continue
            r = rng.choice([v for v in range(-4, 5) if v not in (0, 1)])
            b = a + r * (c - a)
            if b not in (a, c):
                break
        num = expand((x - a) * (x - b))
        den = expand((x - a) * (x - c))
        ans = r
        check(limit(num / den, x, a) == ans)
        # Misconceptions: substituted directly and stopped at 0/0
        # ("undefined"); assumed the singularity means no limit exists;
        # inverted the factor pairing (used (c-a)/(b-a) instead of
        # (b-a)/(c-a)).
        cands = [Distractor("undefined", "stopped-at-undefined"), Distractor("does not exist", "no-limit-at-hole")]
        inverted = Rational(c - a, b - a)
        if inverted != ans and inverted.q == 1 and int(inverted) != ans:
            cands.append(Distractor(str(int(inverted)), "inverted-ratio", int(inverted)))
        check_distinct(ans, *(d.value for d in cands if d.value is not None))
        return dict(
            text=f"Find the limit as x → {a} of ({num}) / ({den})",
            latex=rf"\lim_{{x \to {a}}} \frac{{{latex(num)}}}{{{latex(den)}}}",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            answer_kind="number",
            working=[
                f"({num})/({den}) = (x - {b}) / (x - {c}) for x ≠ {a}",
                f"limit = ({a} - {b}) / ({a} - {c}) = {ans}",
            ],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "one_sided":
        # Piecewise jump discontinuity at x = a_pt; ask for one specific side.
        a_pt = rng.randint(-5, 5)
        p = rng.randint(-5, 5)
        q = rng.randint(-5, 5)
        while q == p:
            q = rng.randint(-5, 5)
        side = rng.choice(["+", "-"])
        f = Piecewise((x + p, x < a_pt), (x + q, x >= a_pt))
        ans = a_pt + (q if side == "+" else p)
        # Independent check: numerically nudge x towards a_pt from the
        # requested side and substitute into the Piecewise directly, rather
        # than trusting SymPy's symbolic Piecewise-limit machinery.
        nudge = a_pt + Rational(1, 1000) if side == "+" else a_pt - Rational(1, 1000)
        check(abs(f.subs(x, nudge) - ans) < Rational(1, 100))
        dir_word = "right" if side == "+" else "left"
        arrow = f"{a_pt}^+" if side == "+" else f"{a_pt}^-"
        other_side = a_pt + (p if side == "+" else q)
        # Misconceptions: read the limit from the wrong side; assumed the
        # limit doesn't exist because of the jump; substituted a_pt directly
        # into whichever piece happens to be listed first.
        cands = [
            Distractor(str(other_side), "ignored-side", other_side),
            Distractor("does not exist", "no-one-sided-limit-at-jump"),
            Distractor(str(a_pt + p), "used-first-listed-piece", a_pt + p),
        ]

        def _signed(v):
            return f"+ {v}" if v >= 0 else f"- {abs(v)}"

        near_offset = q if side == "+" else p
        return dict(
            text=(
                f"f(x) = x {_signed(p)} for x < {a_pt}, f(x) = x {_signed(q)} for x ≥ {a_pt}. Find lim x→{arrow} f(x)"
            ),
            latex=rf"\lim_{{x \to {arrow}}} f(x)",
            answer=ans,
            answer_text=str(ans),
            answer_latex=str(ans),
            answer_kind="number",
            working=[f"Approaching from the {dir_word}, f(x) = x {_signed(near_offset)} → {ans}"],
            distractors=pick_distractors(str(ans), cands),
        )
    if kind == "one_sided_quad":
        # One piece is QUADRATIC (x^2 + p*x + q for x < a_pt), the other
        # stays linear (x + r for x >= a_pt) -- a genuinely different
        # evaluation step from "one_sided" above, whose pieces are both
        # linear.
        while True:
            a_pt = rng.randint(-4, 4)
            p = rng.randint(-5, 5)
            q = rng.randint(-5, 5)
            r = rng.randint(-5, 5)
            if a_pt + r == a_pt**2 + p * a_pt + q:
                continue
            side = rng.choice(["+", "-"])
            quad_val = a_pt**2 + p * a_pt + q
            lin_val = a_pt + r
            ans = lin_val if side == "+" else quad_val
            other = quad_val if side == "+" else lin_val
            # Independent check: nudge x toward a_pt from the requested side and
            # evaluate whichever piece applies there directly, rather than
            # trusting the algebra above.
            nudge = a_pt + Rational(1, 1000) if side == "+" else a_pt - Rational(1, 1000)
            piece_val = (nudge + r) if side == "+" else (nudge**2 + p * nudge + q)
            if not (abs(piece_val - ans) < Rational(1, 50)):
                continue
            dir_word = "right" if side == "+" else "left"
            arrow = f"{a_pt}^+" if side == "+" else f"{a_pt}^-"
            quad_wrong_sign = a_pt**2 - p * a_pt + q
            cands = [
                Distractor(str(other), "ignored-side", other),
                Distractor("does not exist", "no-one-sided-limit-at-jump"),
                Distractor(str(quad_wrong_sign), "substitution-sign-error", quad_wrong_sign),
            ]
            try:
                check_distinct(ans, other, quad_wrong_sign)
            except Exception:
                continue
            # `join_terms` (mathgen/core.py, already used by other generators for exactly this)
            # drops zero terms and renders a unit coefficient bare, the way a student would write
            # it.
            quad_piece = join_terms([(1, "x^2"), (p, "x"), (q, "")])
            lin_piece = join_terms([(1, "x"), (r, "")])
            return dict(
                text=(
                    f"f(x) = {quad_piece} for x < {a_pt}, f(x) = {lin_piece} for x ≥ {a_pt}. Find lim x→{arrow} f(x)"
                ),
                latex=rf"\lim_{{x \to {arrow}}} f(x)",
                answer=ans,
                answer_text=str(ans),
                answer_latex=str(ans),
                answer_kind="number",
                working=[f"Approaching from the {dir_word}, evaluate the piece that applies there → {ans}"],
                distractors=pick_distractors(str(ans), cands),
            )
    if kind == "degree_comparison":
        # Compare polynomial degrees at infinity without computing a ratio.
        a_, b_ = rng.randint(1, 9) * rng.choice([1, -1]), rng.randint(1, 9) * rng.choice([1, -1])
        case = rng.choice(["zero", "infinite"])
        if case == "zero":
            num_deg = rng.randint(0, 2)
            den_deg = num_deg + rng.randint(1, 2)
            num, den = a_ * x**num_deg, b_ * x**den_deg
            check(limit(num / den, x, oo) == 0)
            ans, ans_text = 0, "0"
            ratio = Rational(a_, b_)
            cands = [
                Distractor("∞" if a_ * b_ > 0 else "-∞", "assumed-infinite"),
                Distractor(rational_text(ratio), "ignored-degree-mismatch", str(ratio)),
                Distractor("does not exist", "no-limit-at-infinity"),
            ]
            reason = "deg(numerator) < deg(denominator) ⇒ limit is 0"
        else:
            den_deg = rng.randint(0, 2)
            num_deg = den_deg + rng.randint(1, 2)
            num, den = a_ * x**num_deg, b_ * x**den_deg
            sign = "∞" if a_ * b_ > 0 else "-∞"
            check(limit(num / den, x, oo) == (oo if sign == "∞" else -oo))
            ans, ans_text = sign, sign
            ratio = Rational(a_, b_)
            cands = [
                Distractor("0", "inverted-degree-comparison", "0"),
                Distractor(rational_text(ratio), "ignored-degree-mismatch", str(ratio)),
                Distractor("-∞" if sign == "∞" else "∞", "wrong-infinite-sign"),
            ]
            reason = f"deg(numerator) > deg(denominator) ⇒ limit is {sign}"
        return dict(
            text=f"Find the limit as x → ∞ of ({num}) / ({den})",
            latex=rf"\lim_{{x \to \infty}} \frac{{{latex(num)}}}{{{latex(den)}}}",
            answer=ans,
            answer_text=ans_text,
            answer_latex=ans_text,
            answer_kind="expression",
            working=[reason],
            distractors=pick_distractors(ans_text, cands),
        )
    # Backwards at infinity: the answer is the ratio of leading coefficients.
    a, b = rng.randint(1, 9) * rng.choice([1, -1]), rng.randint(1, 9)
    c, d = rng.randint(-9, 9), rng.randint(-9, 9)
    e, f_ = rng.randint(-9, 9), rng.randint(-9, 9)
    num = a * x**2 + c * x + d
    den = b * x**2 + e * x + f_
    ans = Rational(a, b)
    check(limit(num / den, x, oo) == ans)
    ans_text = rational_text(ans)
    # Misconceptions: "x → ∞ so the limit is ∞"; "denominator grows so 0";
    # inverted the leading-coefficient ratio.
    inverted = Rational(b, a)
    cands = [
        Distractor("∞", "assumed-infinite"),
        Distractor("0", "denominator-grows-so-zero", {"p": 0, "q": 1}),
        Distractor(rational_text(inverted), "inverted-ratio", {"p": int(inverted.p), "q": int(inverted.q)}),
    ]
    return dict(
        text=f"Find the limit as x → ∞ of ({num}) / ({den})",
        latex=rf"\lim_{{x \to \infty}} \frac{{{latex(num)}}}{{{latex(den)}}}",
        answer={"p": int(ans.p), "q": int(ans.q)},
        answer_text=ans_text,
        answer_latex=latex(ans),
        answer_kind="fraction",
        working=[f"Divide top and bottom by x²: limit = {a}/{b} = {ans_text}"],
        distractors=pick_distractors(ans_text, cands),
    )


# Word-problem contexts for the "exponential" branch's word_problem/exam tiers, deliberately
# SIGN-NEUTRAL ("changes at a rate proportional to", never "grows"/"decays") since the drawn k can
# be either sign -- a "decay constant" that turned out positive would be a wording bug, not a maths
# one. The three contexts are chosen for varied nouns/verbs, not varied maths, so
# `mathgen.text_quality.prompt_shape` (masks digits only, not this vocabulary -- none of it is in
# `mathgen.slots`) sees three genuinely different sentence shapes. `{k}`/`{x0}`/`{y0}`
# (word_problem) or `{k}`/`{x0}`/`{m}` (exam) are filled by `phrase`.
_ODE_WORD_PROBLEM_CONTEXTS = (
    "A population's size, y, changes at a rate proportional to its current "
    "size, with constant of proportionality {k} (negative means y is "
    "decreasing). At x = {x0}, y = {y0}. Find y(x).",
    "The mass, y, of a chemical sample changes at a rate proportional to "
    "the mass present, with constant of proportionality {k} (negative "
    "means y is decreasing). At x = {x0}, y = {y0}. Find y(x).",
    "An investment's value, y, changes continuously at a rate proportional "
    "to its current value, with continuous rate {k} (negative means y is "
    "decreasing). At x = {x0}, y = {y0}. Find y(x).",
)

# Same three contexts, escalated once more for the exam tier: the stated
# condition is the RATE dy/dx at x0 (m = k*y0 by construction), not the
# value of y -- the learner must invert through the ODE itself
# (y0 = m/k, always exact since m is built from y0, never rounded) before
# the word_problem tier's own solve-for-C step even starts.
_ODE_EXAM_CONTEXTS = (
    "A population's size, y, changes at a rate proportional to its current "
    "size, with constant of proportionality {k} (negative means y is "
    "decreasing). At x = {x0}, the rate of change of y is {m}. Find y(x).",
    "The mass, y, of a chemical sample changes at a rate proportional to "
    "the mass present, with constant of proportionality {k} (negative "
    "means y is decreasing). At x = {x0}, dy/dx = {m}. Find y(x).",
    "An investment's value, y, changes continuously at a rate proportional "
    "to its current value, with continuous rate {k} (negative means y is "
    "decreasing). At x = {x0}, the value is changing at a rate of {m} per "
    "unit time. Find y(x).",
)


def _ode_exponential(rng, ic_mode):
    """dy/dx = ky, PARTICULAR solution from a stated initial condition --
    a rebuild of the deferred general-solution shape.

    A particular solution has NO free constant: `answer` and `answer_text`
    are the exact same fully-determined sympy expression, so ordinary
    `grade_expression`/`symbolically_equal` grades it with no special-casing
    (`mathgen/grading/typed.py`'s `_expected_expression` docstring names this
    generator's old general-solution shape as the reason that trick existed
    at all). Both use `exp(...)`, never `e^(...)`: `_parse_expression`
    treats a bare lowercase `e` as an undefined Symbol, not Euler's number
    (checked directly against `mathgen/grading/typed.py::_parse_expression` while
    building this), so a canonical spelled with `e^` would silently grade a
    learner's algebraically-identical `exp(...)` answer wrong. Same
    convention `p5e_product_quotient_chain_build.py`'s `_typed` already
    uses: the native `str(expr)` with `**` -> `^` for display, `exp` kept
    verbatim.

    `ic_mode` selects the tier's reasoning shape -- genuine escalation each
    step, never a wider number range:
      - "zero": y(0) = y0 stated directly -- fix C by inspection (e^0 = 1).
      - "nonzero": y(x0) = y0 for a stated NONZERO x0 -- an actual
        algebraic step to solve for C.
      - "word_problem": a context the learner must translate into
        dy/dx = ky themselves (the equation is never written down), on top
        of the nonzero-x0 step above.
      - "exam": the word_problem shape, but the stated condition is the
        RATE dy/dx at x0, not the value of y -- see
        `_ODE_EXAM_CONTEXTS` above for why that is always exact.

    Verified two independent ways, every mode: `checkodesol` (does the
    constructed expression solve the ODE at all -- true for ANY constant)
    and a direct substitution of x0 into it (does it hit the stated y0 --
    the fact that pins down WHICH constant). Together, existence and
    uniqueness for a first-order linear ODE make those two facts the
    complete proof, the same two-independent-facts discipline
    `check_equivalent`/`check_distinct` exist for elsewhere in this module
    -- deliberately not a second `dsolve(..., ics=...)` call: that would be
    a second symbolic solve for no extra verification power, since
    uniqueness already makes "solves the ODE" + "hits y0 at x0" complete.
    """
    fy = Function("y")
    k = rng.choice([v for v in range(-5, 6) if v != 0])
    y0 = rng.choice([v for v in range(-9, 10) if v != 0])
    if ic_mode == "zero":
        x0 = 0
    elif ic_mode == "nonzero":
        x0 = rng.choice([v for v in range(-6, 7) if v != 0])
    else:  # "word_problem", "exam"
        x0 = rng.randint(1, 6)
    m = k * y0  # only used by "exam", but always exact and cheap to build

    ans = y0 * exp(k * (x - x0))
    ode = Eq(fy(x).diff(x), k * fy(x))
    ok, _ = checkodesol(ode, Eq(fy(x), ans))
    check(ok, "particular solution does not solve the ODE")
    check_equivalent(ans.subs(x, x0), y0, "particular solution misses its own initial condition")

    if ic_mode == "exam":
        # Used the stated RATE as if it were the stated VALUE of y.
        cand_ic = m * exp(k * (x - x0)) if k != 1 else None
    else:
        # Used y0 as C without solving for it against a nonzero x0 (vacuous,
        # and equal to the answer, when x0 == 0 -- guarded out).
        cand_ic = y0 * exp(k * x) if x0 != 0 else None
    cand_dropped_k = y0 * exp(x - x0) if k != 1 else None
    cand_added = y0 + exp(k * (x - x0))
    cand_exprs = [c for c in (cand_ic, cand_dropped_k, cand_added) if c is not None]
    check_distinct(ans, *cand_exprs)

    answer_text = f"y = {str(ans).replace('**', '^')}"
    cand_strs = [f"y = {str(c).replace('**', '^')}" for c in cand_exprs]

    kxy = fmt_signed(k, "y")
    if ic_mode == "zero":
        text = phrase(
            rng,
            (
                "Solve dy/dx = {kxy}, given y(0) = {y0}. Find y(x).",
                "The quantity y satisfies dy/dx = {kxy}. Given y(0) = {y0}, find y(x).",
            ),
            kxy=kxy,
            y0=y0,
        )
        result_latex = rf"\frac{{dy}}{{dx}} = {kxy},\ y(0) = {y0}"
    elif ic_mode == "nonzero":
        text = phrase(
            rng,
            (
                "Solve dy/dx = {kxy}, given y({x0}) = {y0}. Find y(x).",
                "The quantity y satisfies dy/dx = {kxy}. Given y({x0}) = {y0}, find y(x).",
            ),
            kxy=kxy,
            x0=x0,
            y0=y0,
        )
        result_latex = rf"\frac{{dy}}{{dx}} = {kxy},\ y({x0}) = {y0}"
    elif ic_mode == "word_problem":
        text = phrase(rng, _ODE_WORD_PROBLEM_CONTEXTS, k=k, x0=x0, y0=y0)
        result_latex = None  # no equation is shown -- the free auto-latex fallback fits prose fine
    else:  # "exam"
        text = phrase(rng, _ODE_EXAM_CONTEXTS, k=k, x0=x0, m=m)
        result_latex = None

    result = dict(
        text=text,
        answer=str(ans),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=pick_distractors(answer_text, cand_strs),
    )
    if result_latex is not None:
        result["latex"] = result_latex
    return result


@generator(
    "differential_equation",
    level=17,
    presets={
        "easy": dict(kinds=("separable_poly",)),
        "medium": dict(kinds=("exponential", "separable_poly")),
        "hard": dict(kinds=("exponential", "separable_poly")),
        # Registered so `make(..., difficulty="exam")` does not raise (`mathgen.core.make` requires
        # any `difficulty=` to be a known preset whenever the topic has ANY presets at all) -- the
        # real exam-tier SHAPE comes entirely from `solve-ode-exponential.yaml`'s own explicit
        # `params: {ic_mode: exam}`, not from this preset's own kinds, exactly like "medium"/"hard"
        # above are never what makes that skill's own sources look the way they do either.
        "exam": dict(kinds=("exponential", "separable_poly")),
    },
)
def differential_equation(rng, kinds=("exponential", "separable_poly"), ic_mode="zero"):
    fy = Function("y")
    kind = rng.choice(list(kinds))
    if kind == "exponential":
        return _ode_exponential(rng, ic_mode)
    # dy/dx = polynomial in x: direct integration.
    p = _poly(_rand_terms(rng, terms=2, max_pow=3))
    ode = Eq(fy(x).diff(x), p)
    sol = dsolve(ode, fy(x))
    ok, _ = checkodesol(ode, sol)
    check(ok)
    F = integrate(p, x)
    # Misconceptions: raised powers without dividing; differentiated instead;
    # forgot the constant of integration.
    cands = [f"y = {expand(x * p)} + C", f"y = {expand(diff(p, x))} + C", f"y = {F}"]
    return dict(
        text=f"Solve dy/dx = {p}",
        latex=rf"\frac{{dy}}{{dx}} = {latex(p)}",
        answer=str(F),
        answer_text=f"y = {F} + C",
        answer_latex=rf"y = {latex(F)} + C",
        distractors=pick_distractors(f"y = {F} + C", cands),
    )


@generator(
    "partial_derivatives",
    level=18,
    presets={
        "easy": dict(coef_hi=3, pow_hi=2),
        "medium": dict(coef_hi=6, pow_hi=3),
        "hard": dict(coef_hi=9, pow_hi=4),
    },
)
def partial_derivatives(rng, coef_hi=6, pow_hi=3, kinds=("dx", "dy", "grad")):
    # f(x, y) with mixed terms; ask for one partial or the gradient.
    a, b, c = (rng.randint(1, coef_hi) * rng.choice([1, -1]) for _ in range(3))
    px, py = rng.randint(1, pow_hi), rng.randint(1, pow_hi)
    f = a * x ** (px + 1) + b * x**px * y**py + c * y ** (py + 1)
    kind = rng.choice(list(kinds))
    fx, fy_ = diff(f, x), diff(f, y)
    # Independent check: mixed partials commute.
    check(diff(fx, y) == diff(fy_, x))
    if kind == "dx":
        # Misconceptions: differentiated w.r.t. the wrong variable;
        # differentiated w.r.t. both and summed; took the mixed partial.
        cands = [str(fy_), str(fx + fy_), str(diff(fx, y))]
        return dict(
            text=f"f(x, y) = {f}. Find ∂f/∂x",
            latex=rf"f(x,y) = {latex(f)}.\ \text{{Find }} \frac{{\partial f}}{{\partial x}}",
            answer=str(fx),
            answer_text=str(fx),
            answer_latex=latex(fx),
            distractors=pick_distractors(str(fx), cands),
        )
    if kind == "dy":
        cands = [str(fx), str(fx + fy_), str(diff(fy_, x))]
        return dict(
            text=f"f(x, y) = {f}. Find ∂f/∂y",
            latex=rf"f(x,y) = {latex(f)}.\ \text{{Find }} \frac{{\partial f}}{{\partial y}}",
            answer=str(fy_),
            answer_text=str(fy_),
            answer_latex=latex(fy_),
            distractors=pick_distractors(str(fy_), cands),
        )
    # Misconceptions: swapped the components; used the same variable for both.
    cands = [f"∇f = ({fy_}, {fx})", f"∇f = ({fx}, {fx})", f"∇f = ({fy_}, {fy_})"]
    return dict(
        text=f"f(x, y) = {f}. Find ∇f",
        latex=rf"f(x,y) = {latex(f)}.\ \text{{Find }} \nabla f",
        answer=[str(fx), str(fy_)],
        answer_text=f"∇f = ({fx}, {fy_})",
        answer_latex=rf"\nabla f = \left({latex(fx)},\ {latex(fy_)}\right)",
        distractors=pick_distractors(f"∇f = ({fx}, {fy_})", cands),
    )
