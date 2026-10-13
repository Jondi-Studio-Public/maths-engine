"""Generates the five calculus "establish and use the formula" proof topics:

- ``proof_derivative_exp``: d/dx(e^x) = e^x and its uses.
- ``proof_derivative_ln``: d/dx(ln x) = 1/x and its uses.
- ``proof_integral_exp``: the integral of e^x is e^x + c and its uses.
- ``proof_integral_recip_pos``: the integral of 1/x is ln x + c for x > 0 and its uses.
- ``proof_integral_recip_neg``: the same formula as ln|x| + c on x < 0, with negative limits.

Each question names what the learner may assume (a limit, a derivative, the chain rule, the Fundamental
Theorem of Calculus) and asks for a complete argument. The argument is marked against a rubric, so `answer` is
a marking dict (see `_free_proof_kit`), not a value.

Every claim a question asks the learner to prove is confirmed here with SymPy before the question is returned:
a derivative claim by differentiating, an indefinite integral claim by differentiating the stated antiderivative,
and a definite integral claim by comparing the exact integral, all inside the stated domain (x is declared
positive, and claims on x < 0 are checked on the substitution x = -t).
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from ..core import check, check_equivalent, generator
from ._free_proof_kit import Fm, Prompt, TAILS, attempts, finish, marking

x = sp.Symbol("x", positive=True)
t = sp.Symbol("t", positive=True)  # for claims on x < 0, written with x = -t
VARS = ("x", "t", "u")


def _tail(rng) -> str:
    return rng.choice(TAILS)


def _q(n: int | Fraction) -> sp.Rational:
    f = Fraction(n)
    return sp.Rational(f.numerator, f.denominator)


# ---------------------------------------------------------------------------
# Formatting helpers: every formula is held as LaTeX and as plain text.
# ---------------------------------------------------------------------------
def _num(n: int) -> str:
    return str(n)


def _scale(c: int, body: Fm) -> Fm:
    """c·body, with 1 and -1 written as nothing and a minus sign."""
    if c == 1:
        return body
    gap = " " if body.plain.startswith("ln") else ""
    if c == -1:
        return Fm("-" + gap + body.tex, "-" + gap + body.plain)
    return Fm(str(c) + gap + body.tex, str(c) + gap + body.plain)


def _aff(a: int, b: int, var: str = "x") -> Fm:
    """a·var + b as a formula, leading term first."""
    lead = var if a == 1 else "-" + var if a == -1 else f"{a}{var}"
    if b == 0:
        return Fm(lead)
    return Fm(f"{lead} {'+' if b > 0 else '-'} {abs(b)}")


def _e_pow(arg: Fm | str | int) -> Fm:
    """e raised to a formula; a bare 1 gives e, a bare 0 gives 1."""
    if isinstance(arg, int):
        if arg == 0:
            return Fm("1")
        if arg == 1:
            return Fm("e")
        return Fm("e^{" + str(arg) + "}", f"e^({arg})" if arg < 0 else f"e^{arg}")
    arg = Fm.of(arg)
    plain = arg.plain if len(arg.plain) == 1 else f"({arg.plain})"
    return Fm("e^{" + arg.tex + "}", "e^" + plain)


def _ln(arg: Fm | str) -> Fm:
    arg = Fm.of(arg)
    if len(arg.plain) == 1:
        return Fm("\\ln " + arg.tex, "ln " + arg.plain)
    return Fm("\\ln\\left(" + arg.tex + "\\right)", "ln(" + arg.plain + ")")


def _ddx(f: Fm, var: str = "x") -> Fm:
    return Fm(f"\\frac{{d}}{{d{var}}}\\left(" + f.tex + "\\right)", f"d/d{var}(" + f.plain + ")")


def _frac(num: Fm | str, den: Fm | str) -> Fm:
    n, d = Fm.of(num), Fm.of(den)
    return Fm(
        "\\frac{" + n.tex + "}{" + d.tex + "}",
        f"({n.plain})/({d.plain})" if len(d.plain) > 1 or len(n.plain) > 1 else f"{n.plain}/{d.plain}",
    )


def _has_top_level_sum(plain: str) -> bool:
    depth = 0
    for i, ch in enumerate(plain):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-" and 0 < i < len(plain) - 1 and plain[i - 1] == " " and plain[i + 1] == " ":
            return True
    return False


def _int(f: Fm, lo: Fm | str | int | None = None, hi: Fm | str | int | None = None, var: str = "x") -> Fm:
    """The integral of f, bracketing an integrand that is a sum or difference."""
    if _has_top_level_sum(f.plain):
        f = Fm("\\left(" + f.tex + "\\right)", "(" + f.plain + ")")
    if lo is None:
        return Fm("\\int " + f.tex + f"\\, d{var}", f"∫ {f.plain} d{var}")
    lo, hi = Fm.of(str(lo) if isinstance(lo, int) else lo), Fm.of(str(hi) if isinstance(hi, int) else hi)
    return Fm(
        "\\int_{" + lo.tex + "}^{" + hi.tex + "} " + f.tex + f"\\, d{var}",
        f"∫ from {lo.plain} to {hi.plain} of {f.plain} d{var}",
    )


def _eq_prime(s: int, p: int) -> str:
    """y'' - s y' + p y = 0 with signs written out."""

    def term(c: int, body: str) -> str:
        if c == 0:
            return ""
        sign = "+" if c > 0 else "-"
        mag = abs(c)
        return f" {sign} {'' if mag == 1 else mag}{body}"

    return "y''" + term(-s, "y'") + term(p, "y") + " = 0"


E_LIMIT = Fm("\\lim_{h \\to 0} \\frac{e^{h}-1}{h} = 1", "lim(h→0) (e^h - 1)/h = 1")
D_EXP = _ddx(_e_pow("x")) + " = " + _e_pow("x")
D_LN = _ddx(Fm("\\ln x", "ln x")) + " = " + Fm("\\frac{1}{x}", "1/x")


def _signed_term(c: int, body: str) -> str:
    """' + 3body' / ' - body' for building sums, with the sign made explicit."""
    return f" {'+' if c > 0 else '-'} {'' if abs(c) == 1 else abs(c)}{body}"


# ===========================================================================
# d/dx (e^x) = e^x
# ===========================================================================
def _dexp_first_principles(rng) -> dict:
    v = rng.choice(VARS)
    claim = _ddx(_e_pow(v), v) + " = " + _e_pow(v)
    p = Prompt()
    p.t("Given that ").m(E_LIMIT).t(", prove from first principles that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"By definition, d/d{v}(e^{v}) = lim(h→0) (e^({v}+h) - e^{v})/h.",
        f"e^({v}+h) = e^{v} × e^h, so the quotient is e^{v}(e^h - 1)/h.",
        f"e^{v} does not depend on h, so the limit is e^{v} × lim(h→0) (e^h - 1)/h.",
        f"The given limit is 1, so the derivative is e^{v} × 1 = e^{v}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses e^({v}+h) = e^{v} e^h and takes e^{v} out of the limit to reach e^{v} × 1"),
            ("logical_completeness", "starts from the limit definition of the derivative and ends at e^x"),
            ("definitions_assumptions", f"states that e^{v} is constant as h varies, so it can move outside the limit"),
            ("communication", "writes the limit with its variable at every line"),
        ],
        traps=[
            ("assumes_the_result", "does it quote the derivative of e^x instead of deriving it from the definition?"),
            ("drops_the_limit_notation", "does it write the quotient with no limit sign, so the lines are not equal?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_gradient_at_point(rng) -> dict:
    a = rng.choice((-2, -1, 0, 1, 2, 3))
    ea = _e_pow(a)
    p = Prompt()
    p.t("Given that ").m(E_LIMIT).t(f", prove from first principles that the gradient of the curve ").m(
        Fm("y = e^{x}", "y = e^x")
    ).t(f" at the point where x = {a} is ").m(ea).t(". ").t(_tail(rng))
    eap = ea.plain
    ref = [
        f"The gradient at x = {a} is lim(h→0) (e^({a}+h) - {eap})/h.",
        f"e^({a}+h) = {eap} × e^h, so the quotient is {eap}(e^h - 1)/h.",
        f"{eap} is a constant, so the limit is {eap} × lim(h→0) (e^h - 1)/h.",
        f"The given limit is 1, so the gradient is {eap}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {eap} × 1 using the given limit"),
            ("logical_completeness", "starts from the difference quotient at the point and takes h to 0"),
            ("definitions_assumptions", f"treats {eap} as a constant"),
            ("communication", "keeps the limit notation until the limit is evaluated"),
        ],
        traps=[
            (
                "differentiates_without_justification",
                "does it differentiate e^x by rule instead of from first principles?",
            ),
            ("evaluates_at_zero_early", "does it set h = 0 in the quotient before simplifying, giving 0/0?"),
        ],
    )
    return finish(p, f"gradient = {eap}", mark)


def _tangent_eq(a: int) -> Fm:
    """The tangent to y = e^x at x = a, in its simplest factored form."""
    if a == 0:
        return Fm("y = x + 1")
    if a == 1:
        return Fm("y = ex", "y = ex")
    return Fm(
        "y = e^{" + str(a) + "}\\left(" + _aff(1, 1 - a).tex + "\\right)",
        f"y = {_e_pow(a).plain}({_aff(1, 1 - a).plain})",
    )


def _dexp_tangent(rng) -> dict:
    a = rng.choice((0, 1, 2, -1, -2, 3))
    xx = sp.Symbol("x")
    tangent = sp.exp(a) * (xx - a + 1)
    check_equivalent(tangent.subs(xx, a), sp.exp(a))
    check_equivalent(sp.diff(tangent, xx), sp.exp(a))
    claim = _tangent_eq(a)
    ea = _e_pow(a).plain
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", prove that the tangent to the curve ").m(Fm("y = e^{x}", "y = e^x")).t(
        f" at the point where x = {a} has equation "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"At x = {a}, y = {ea}, so the point of contact is ({a}, {ea}).",
        f"The gradient is the value of dy/dx = e^x at x = {a}, which is {ea}.",
        f"The tangent is y - {ea} = {ea}(x - {a}).",
        f"Add {ea} to both sides and factor: y = {ea}(x - {a} + 1)"
        + (f" = {claim.plain[4:]}." if a not in (0, 1) else f", which is {claim.plain}."),
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the point ({a}, {ea}) and gradient {ea} to reach the stated equation"),
            ("logical_completeness", "finds both the point and the gradient before writing the tangent"),
            ("definitions_assumptions", "says the gradient of the tangent is the derivative at the point of contact"),
            ("communication", "shows the rearrangement to the stated form"),
        ],
        traps=[
            ("gradient_at_wrong_place", f"is the gradient evaluated at x = {a}, not at x = 0 or x = 1?"),
            ("forgets_the_y_coordinate", f"does it use y = {ea} at the point of contact and not y = {a}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _k_fm(k: Fraction) -> Fm:
    """k as a coefficient: a whole number, or a bracketed half-integer."""
    if k.denominator == 1:
        return Fm(str(k.numerator))
    sign = "-" if k < 0 else ""
    return Fm(f"{sign}\\frac{{{abs(k.numerator)}}}{{{k.denominator}}}", f"{sign}{abs(k.numerator)}/{k.denominator}")


def _kx(k: Fraction) -> Fm:
    """k·x as an exponent."""
    if k.denominator == 1:
        return _aff(k.numerator, 0)
    sign = "-" if k < 0 else ""
    n = abs(k.numerator)
    num = "x" if n == 1 else f"{n}x"
    return Fm(f"{sign}\\frac{{{num}}}{{{k.denominator}}}", f"{sign}{num}/{k.denominator}")


def _dexp_chain_k(rng) -> dict:
    k = Fraction(rng.choice((2, 3, 4, 5, -1, -2, -3, Fraction(1, 2), Fraction(-1, 2), Fraction(3, 2))))
    arg = _kx(k)
    kk = sp.Rational(k.numerator, k.denominator)
    check_equivalent(sp.diff(sp.exp(kk * x), x), kk * sp.exp(kk * x))
    coefficient = _k_fm(k)
    if k.denominator != 1 or k.numerator in (1, -1):
        lead = coefficient if k.denominator != 1 else (Fm("") if k.numerator == 1 else Fm("-"))
    else:
        lead = coefficient
    claim = _ddx(_e_pow(arg)) + " = " + lead + _e_pow(arg)
    kp = coefficient.plain
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and using the chain rule, prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"Let u = {arg.plain}, so y = e^u.",
        f"Then dy/du = e^u (the given derivative) and du/dx = {kp}.",
        f"The chain rule gives dy/dx = dy/du × du/dx = e^u × {kp}.",
        f"Substitute back u = {arg.plain}: dy/dx = {kp} e^({arg.plain}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains {kp} e^({arg.plain}) from dy/du × du/dx"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", "defines u and writes both derivatives before combining them"),
            ("communication", "substitutes back so the answer is in terms of x"),
        ],
        traps=[
            ("forgets_the_inner_derivative", "does it stop at e^(kx) and omit the factor k?"),
            ("divides_instead_of_multiplies", "does it divide by k (the integral's rule) instead of multiplying?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_chain_affine(rng) -> dict:
    a = rng.choice((2, 3, 5, -1, -2, -4))
    b = rng.choice((1, 2, 3, -1, -2, -3))
    arg = _aff(a, b)
    check_equivalent(sp.diff(sp.exp(a * x + b), x), a * sp.exp(a * x + b))
    claim = _ddx(_e_pow(arg)) + " = " + _scale(a, _e_pow(arg))
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and using the chain rule, prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"Let u = {arg.plain}, so y = e^u.",
        f"dy/du = e^u and du/dx = {a}.",
        f"By the chain rule dy/dx = e^u × {a}.",
        f"Substitute back: dy/dx = {a} e^({arg.plain}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets du/dx = {a} (the constant {b} differentiates to 0) and multiplies"),
            ("logical_completeness", "names the inner function and applies the chain rule"),
            ("definitions_assumptions", "says the constant term has derivative 0"),
            ("communication", "substitutes back so the answer is in x"),
        ],
        traps=[
            ("keeps_the_constant", f"does the derivative of {arg.plain} wrongly include {b}?"),
            ("forgets_the_inner_derivative", f"does it omit the factor {a}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_ode(rng) -> dict:
    amp = rng.choice((2, 3, 4, 5, 7))
    k = rng.choice((2, 3, 4, -1, -2, -3, 5))
    arg = _aff(k, 0)
    y = _scale(amp, _e_pow(arg))
    check_equivalent(sp.diff(amp * sp.exp(k * x), x), k * amp * sp.exp(k * x))
    kc = {1: "", -1: "-"}.get(k, str(k))
    claim = Fm("\\frac{dy}{dx} = " + kc + "y", f"dy/dx = {kc}y")
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", show that ").m(Fm("y = " + y.tex, "y = " + y.plain)).t(" satisfies ").m(claim).t(
        ". "
    ).t(_tail(rng))
    ref = [
        f"y = {y.plain}, so by the chain rule dy/dx = {amp} × ({k}) × e^({arg.plain}).",
        f"That is dy/dx = {amp * k} e^({arg.plain}).",
        f"But {amp * k} e^({arg.plain}) = {k} × ({amp} e^({arg.plain})) = {kc}y.",
        f"So dy/dx = {kc}y.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates to {amp * k}e^({arg.plain}) and recognises it as {kc}y"),
            (
                "logical_completeness",
                "differentiates and then compares with the right-hand side for all x, not one value",
            ),
            ("definitions_assumptions", "keeps the constant multiple and the chain-rule factor"),
            ("communication", f"ends by stating dy/dx = {kc}y"),
        ],
        traps=[
            ("checks_one_value", "does it check the equation at a single x instead of for all x?"),
            ("loses_the_amplitude", f"does it lose the factor {amp} when differentiating?"),
        ],
    )
    return finish(p, f"y = {y.plain} satisfies {claim.plain}", mark)


def _poly_term(c: int, n: int) -> Fm:
    """c·x^n written as a term (n >= 1), with unit coefficients dropped."""
    body_tex = "x" if n == 1 else f"x^{{{n}}}"
    body_plain = "x" if n == 1 else f"x^{n}"
    lead = "" if c == 1 else "-" if c == -1 else str(c)
    return Fm(lead + body_tex, lead + body_plain)


def _dexp_power_exponent(rng) -> dict:
    c = rng.choice((1, 2, 3, -1))
    n = rng.choice((2, 3))
    d = rng.choice((0, 1, -1, 2))
    g = _poly_term(c, n) + (f" {'+' if d > 0 else '-'} {abs(d)}" if d else "")
    dg = _poly_term(c * n, n - 1)
    check_equivalent(sp.diff(sp.exp(c * x**n + d), x), c * n * x ** (n - 1) * sp.exp(c * x**n + d))
    claim = _ddx(_e_pow(g)) + " = " + dg + _e_pow(g)
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and using the chain rule, prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"Let u = {g.plain}, so y = e^u.",
        f"dy/du = e^u and du/dx = {dg.plain}.",
        f"By the chain rule dy/dx = e^u × {dg.plain}.",
        f"Substitute back: dy/dx = {dg.plain} e^({g.plain}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates the exponent to {dg.plain} and multiplies it by e^({g.plain})"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", "differentiates each term of the exponent, a constant giving 0"),
            ("communication", "substitutes back so the answer is in x"),
        ],
        traps=[
            ("differentiates_the_exponent_only", "does it differentiate the exponent and drop the factor e^u?"),
            ("power_rule_slip", f"is the derivative of the power term {dg.plain}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_first_principles_k(rng) -> dict:
    k = rng.choice((2, 3, 4, 5, -1, -2, -3))
    arg = _aff(k, 0)
    check_equivalent(sp.diff(sp.exp(k * x), x), k * sp.exp(k * x))
    claim = _ddx(_e_pow(arg)) + " = " + _scale(k, _e_pow(arg))
    p = Prompt()
    p.t("Given that ").m(E_LIMIT).t(", prove from first principles that ").m(claim).t(". ").t(_tail(rng))
    kx = arg.plain
    ref = [
        f"By definition, d/dx(e^({kx})) = lim(h→0) (e^({k}(x+h)) - e^({kx}))/h.",
        f"e^({k}(x+h)) = e^({kx}) × e^({k}h), so the quotient is e^({kx}) × (e^({k}h) - 1)/h.",
        f"Let u = {k}h. As h → 0, u → 0, and (e^({k}h) - 1)/h = {k} × (e^u - 1)/u.",
        f"By the given limit, (e^u - 1)/u → 1, so (e^({k}h) - 1)/h → {k}.",
        f"Therefore d/dx(e^({kx})) = {k} e^({kx}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"changes variable to u = {k}h and gets the limit {k}"),
            (
                "logical_completeness",
                "uses the difference quotient, factorises e^(kx), and applies the given limit after the substitution",
            ),
            ("definitions_assumptions", f"notes that {k} is not 0, so u → 0 as h → 0"),
            ("communication", "writes each limit with its variable"),
        ],
        traps=[
            (
                "uses_the_chain_rule_instead",
                "does it use the chain rule, which the question does not give, instead of first principles?",
            ),
            (
                "limit_not_matched",
                "does it apply the given limit to (e^(kh) - 1)/h directly, without the substitution u = kh?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_two_exponentials(rng) -> dict:
    for _ in attempts(200, "two exponents with a non-zero sum"):
        a, b = rng.sample((1, 2, 3, -1, -2, -3, 4), 2)
        if a + b != 0:
            break
    s, pr = a + b, a * b
    ya, yb = _e_pow(_aff(a, 0)), _e_pow(_aff(b, 0))
    y = ya + " + " + yb
    yy = sp.exp(a * x) + sp.exp(b * x)
    check_equivalent(sp.diff(yy, x, 2) - s * sp.diff(yy, x) + pr * yy, 0)
    eq = _eq_prime(s, pr)
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", show that ").m(Fm("y = " + y.tex, "y = " + y.plain)).t(" satisfies ").m(
        Fm(eq, eq)
    ).t(", where y' and y'' are the first and second derivatives with respect to x. ").t(_tail(rng))
    ref = [
        f"y' = {a}e^({a}x) + ({b})e^({b}x), by the chain rule.",
        f"y'' = {a * a}e^({a}x) + {b * b}e^({b}x).",
        f"In y'' - ({s})y' + ({pr})y the coefficient of e^({a}x) is {a * a} - ({s})({a}) + {pr} = {a * a - s * a + pr}.",
        f"The coefficient of e^({b}x) is {b * b} - ({s})({b}) + {pr} = {b * b - s * b + pr}.",
        "Both coefficients are 0, so the left-hand side is 0 for all x.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds y' and y'' correctly with the chain-rule factors"),
            ("logical_completeness", "collects the coefficient of each exponential and shows each is 0"),
            ("definitions_assumptions", "treats the two exponentials as separate terms with separate coefficients"),
            ("communication", "shows the arithmetic of each coefficient"),
        ],
        traps=[
            ("coefficient_slip", "are the coefficients combined correctly for each exponential separately?"),
            ("checks_a_point", "does it substitute one value of x instead of showing the identity holds for all x?"),
        ],
    )
    return finish(p, eq, mark)


def _dexp_stationary(rng) -> dict:
    c = rng.choice((0, 1, 2, 3, -1, -2))
    xs = -(c + 1)
    yy = (x + c) * sp.exp(x)
    check_equivalent(sp.diff(yy, x), (x + c + 1) * sp.exp(x))
    check(sp.solve(sp.Eq(sp.diff(yy, x), 0), x) in ([xs], []) or xs <= 0, "stationary point")
    factor = _aff(1, c)
    y = Fm("\\left(" + factor.tex + "\\right) e^{x}", f"({factor.plain})e^x") if c else Fm("x e^{x}", "x e^x")
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", prove that the curve ").m(Fm("y = " + y.tex, "y = " + y.plain)).t(
        f" has exactly one stationary point, at x = {xs}. "
    ).t(_tail(rng))
    cplus = c + 1
    ref = [
        f"Let u = {factor.plain} and v = e^x: u' = 1 and v' = e^x (given).",
        f"Product rule: dy/dx = u'v + uv' = e^x + ({factor.plain})e^x = (x + {cplus})e^x.",
        f"Stationary points have dy/dx = 0, so (x + {cplus})e^x = 0.",
        f"e^x > 0 for every x, so it is never 0; the factor x + {cplus} must be 0.",
        f"So x = {xs} is the only stationary point.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates by the product rule to (x + {cplus})e^x"),
            (
                "logical_completeness",
                "justifies that e^x is never zero, so the only solution comes from the linear factor",
            ),
            ("definitions_assumptions", "states that stationary means dy/dx = 0"),
            ("communication", "concludes with the single value of x"),
        ],
        traps=[
            ("cancels_the_exponential", "does it divide by e^x without saying why that is allowed?"),
            ("product_rule_slip", "does the derivative drop the e^x from the first term or the u'v part?"),
        ],
    )
    return finish(p, f"the only stationary point is at x = {xs}", mark)


def _dexp_inverse(rng) -> dict:
    v = rng.choice(("x", "t"))
    w = "y"
    ev = _e_pow(v)
    claim = _ddx(ev, v) + " = " + ev
    p = Prompt()
    p.t("Given that ").m(_ddx(_lnx(v), v) + " = " + Fm("\\frac{1}{" + v + "}", f"1/{v}")).t(
        f" for {v} > 0, and using the chain rule, prove that "
    ).m(claim).t(". Start by writing ").m(Fm(f"y = e^{{{v}}}", f"y = e^{v}")).t(" in the form ").m(
        Fm(f"{v} = \\ln y", f"{v} = ln y")
    ).t(". ").t(_tail(rng))
    ref = [
        f"Let y = e^{v}. Then y > 0 and {v} = ln y.",
        f"Differentiate both sides with respect to {v}. The left side gives 1.",
        f"By the chain rule d/d{v}(ln y) = (1/y) × dy/d{v}.",
        f"So 1 = (1/y) dy/d{v}, which gives dy/d{v} = y.",
        f"Since y = e^{v}, d/d{v}(e^{v}) = e^{v}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates {v} = ln y implicitly and solves for dy/d{v}"),
            ("logical_completeness", f"applies the chain rule to ln y with y a function of {v}"),
            ("definitions_assumptions", f"notes that y = e^{v} is positive, so ln y exists"),
            ("communication", f"returns to y = e^{v} at the end"),
        ],
        traps=[
            ("circular_use", f"does it use d/d{v}(e^{v}) = e^{v} to differentiate, which is what is being proved?"),
            ("misses_dy_dx", f"does it differentiate ln y as 1/y without the factor dy/d{v}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dexp_tangent_intercept(rng) -> dict:
    a = rng.choice((1, 2, 3, 4, 5, -1, -2, -3))
    xx = sp.Symbol("x")
    tangent = sp.exp(a) * (xx - a + 1)
    check(sp.solve(sp.Eq(tangent, 0), xx) == [a - 1], "x-intercept")
    ea = _e_pow(a).plain
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", prove that the tangent to the curve ").m(Fm("y = e^{x}", "y = e^x")).t(
        f", drawn at the point where x = {a}, crosses the x-axis at x = {a - 1}. "
    ).t(_tail(rng))
    ref = [
        f"At x = {a}: y = {ea}, so the point is ({a}, {ea}).",
        f"The gradient is the derivative e^x at x = {a}, which is {ea}.",
        f"The tangent is y - {ea} = {ea}(x - {a}).",
        f"It crosses the x-axis where y = 0: -{ea} = {ea}(x - {a}).",
        f"Divide by {ea} (which is not 0): -1 = x - {a}, so x = {a - 1}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"sets y = 0 in the tangent and solves for x = {a - 1}"),
            ("logical_completeness", "finds the tangent first, then its intercept"),
            ("definitions_assumptions", f"says it is dividing by {ea}, which is never 0"),
            ("communication", "states the intercept as a value of x"),
        ],
        traps=[
            ("wrong_gradient", f"is the gradient {ea}, not e or 1?"),
            ("intercept_on_wrong_axis", "does it find where x = 0 (the y-axis) instead of y = 0?"),
        ],
    )
    return finish(p, f"the tangent crosses the x-axis at x = {a - 1}", mark)


def _dexp_damped(rng) -> dict:
    a = rng.choice((1, 2, -1, -2))
    b = rng.choice((1, 2, 3))
    use_cos = rng.random() < 0.5
    fn = sp.cos if use_cos else sp.sin
    name = "cos" if use_cos else "sin"
    s, pr = 2 * a, a * a + b * b
    yy = sp.exp(a * x) * fn(b * x)
    check_equivalent(sp.diff(yy, x, 2) - s * sp.diff(yy, x) + pr * yy, 0)
    ex = _e_pow(_aff(a, 0))
    bx = _aff(b, 0)
    y = ex + Fm(f"\\{name}\\left(" + bx.tex + "\\right)", f" {name}({bx.plain})")
    eq = _eq_prime(s, pr)
    ep = ex.plain
    if use_cos:
        d1 = f"y' = {ep}({a} cos({bx.plain}) - {b} sin({bx.plain}))"
        d2 = f"y'' = {ep}({a * a - b * b} cos({bx.plain}) - {2 * a * b} sin({bx.plain}))"
        cos_coef = (a * a - b * b) - s * a + pr
        sin_coef = -2 * a * b + s * b
        coef_line = f"cos terms: {a * a - b * b} - ({s})({a}) + {pr} = {cos_coef}; sin terms: {-2 * a * b} - ({s})({-b}) = {sin_coef}."
    else:
        d1 = f"y' = {ep}({a} sin({bx.plain}) + {b} cos({bx.plain}))"
        d2 = f"y'' = {ep}({a * a - b * b} sin({bx.plain}) + {2 * a * b} cos({bx.plain}))"
        sin_coef = (a * a - b * b) - s * a + pr
        cos_coef = 2 * a * b - s * b
        coef_line = f"sin terms: {a * a - b * b} - ({s})({a}) + {pr} = {sin_coef}; cos terms: {2 * a * b} - ({s})({b}) = {cos_coef}."
    check(sin_coef == 0 and cos_coef == 0, "damped oscillation coefficients cancel")
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", prove that ").m(Fm("y = " + y.tex, "y = " + y.plain)).t(" satisfies ").m(
        Fm(eq, eq)
    ).t(", where y' and y'' are the first and second derivatives with respect to x. ").t(_tail(rng))
    ref = [
        f"Product rule with u = {ep} (u' = {a}{ep}) and v = {name}({bx.plain}):",
        d1 + ".",
        "Differentiate again with the product rule: " + d2 + ".",
        f"In y'' - ({s})y' + {pr}y, the coefficients are: {coef_line}",
        f"Both are 0, so {eq}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds y' and y'' by the product and chain rules with correct coefficients"),
            (
                "logical_completeness",
                "substitutes into the left-hand side and shows both the sine and cosine parts cancel",
            ),
            ("definitions_assumptions", "keeps y' and y'' distinct and writes them out before combining them"),
            ("communication", "shows the grouping of like terms"),
        ],
        traps=[
            ("product_rule_sign", "is the sign of the derivative of the trigonometric factor right?"),
            ("checks_a_point", "does it test one value of x instead of showing the identity for all x?"),
        ],
    )
    return finish(p, eq, mark)


DEXP_FAMILIES = {
    "easy": (_dexp_first_principles, _dexp_gradient_at_point, _dexp_tangent),
    "medium": (_dexp_chain_k, _dexp_chain_affine, _dexp_ode, _dexp_power_exponent),
    "hard": (_dexp_first_principles_k, _dexp_two_exponentials, _dexp_stationary),
    "exam": (_dexp_inverse, _dexp_tangent_intercept, _dexp_damped),
}


# ===========================================================================
# d/dx (ln x) = 1/x
# ===========================================================================
def _lnx(var: str = "x") -> Fm:
    return Fm("\\ln " + var, "ln " + var)


def _dln_from_exp(rng) -> dict:
    v = rng.choice(("x", "t"))
    w = "y" if v != "y" else "z"
    claim = _ddx(_lnx(v), v) + " = " + Fm("\\frac{1}{" + v + "}", f"1/{v}")
    p = Prompt()
    p.t("Given that ").m(_ddx(_e_pow(w), w) + " = " + _e_pow(w)).t(f", prove that ").m(claim).t(
        f" for {v} > 0, by writing {w} = "
    ).m(_lnx(v)).t(f" in the form {v} = ").m(_e_pow(w)).t(". ").t(_tail(rng))
    ref = [
        f"Let {w} = ln {v}. Then {v} = e^{w}.",
        f"Differentiate both sides with respect to {v}. The left side gives 1.",
        f"By the chain rule d/d{v}(e^{w}) = e^{w} × d{w}/d{v}.",
        f"So 1 = e^{w} d{w}/d{v}. Since e^{w} = {v}, 1 = {v} d{w}/d{v}.",
        f"Therefore d{w}/d{v} = 1/{v}, that is d/d{v}(ln {v}) = 1/{v}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates {v} = e^{w} implicitly and solves for d{w}/d{v}"),
            ("logical_completeness", f"replaces e^{w} by {v} before solving"),
            ("definitions_assumptions", f"notes {v} > 0 so that ln {v} exists and e^{w} = {v}"),
            ("communication", f"ends with d/d{v}(ln {v}) = 1/{v}"),
        ],
        traps=[
            ("circular_use", "does it quote the derivative of ln x, which is being proved?"),
            ("misses_the_chain_factor", f"does it differentiate e^{w} as e^{w} without the factor d{w}/d{v}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_tangent(rng) -> dict:
    a = rng.choice((1, 2, 3, 4, 5, 6))
    tangent = x / a + sp.log(a) - 1
    check_equivalent(tangent.subs(x, a), sp.log(a))
    check_equivalent(sp.diff(tangent, x), sp.diff(sp.log(x), x).subs(x, a))
    if a == 1:
        claim = Fm("y = x - 1")
    else:
        claim = Fm(f"y = \\frac{{x}}{{{a}}} + \\ln {a} - 1", f"y = x/{a} + ln {a} - 1")
    ln_a = "0" if a == 1 else f"ln {a}"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that the tangent to the curve ").m(Fm("y = \\ln x", "y = ln x")).t(
        f" at the point where x = {a} has equation "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"At x = {a}, y = ln {a}, so the point of contact is ({a}, {ln_a}).",
        f"The gradient is the value of dy/dx = 1/x at x = {a}, which is 1/{a}.",
        f"The tangent is y - {ln_a} = (1/{a})(x - {a}).",
        f"Expand: y = x/{a} - 1 + {ln_a}, that is " + claim.plain + ".",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the point ({a}, {ln_a}) and gradient 1/{a} to reach the stated equation"),
            ("logical_completeness", "finds both the point and the gradient before writing the tangent"),
            ("definitions_assumptions", "says the tangent's gradient is the derivative at the point of contact"),
            ("communication", "shows the expansion to the stated form"),
        ],
        traps=[
            ("gradient_is_ln_a", f"does it use ln {a} as the gradient instead of 1/{a}?"),
            ("point_slip", f"does it use the point ({a}, {a}) instead of ({a}, ln {a})?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_kx(rng) -> dict:
    k = rng.choice((2, 3, 4, 5, 6, 7))
    check_equivalent(sp.diff(sp.log(k * x), x), 1 / x)
    arg = Fm(f"{k}x")
    claim = _ddx(_ln(arg)) + " = " + Fm("\\frac{1}{x}", "1/x")
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0 and using the law ").m(
        Fm("\\ln(ab) = \\ln a + \\ln b", "ln(ab) = ln a + ln b")
    ).t(", prove that ").m(claim).t(" for x > 0. ").t(_tail(rng))
    ref = [
        f"By the log law, ln({k}x) = ln {k} + ln x for x > 0.",
        f"ln {k} is a constant, so its derivative is 0.",
        f"Therefore d/dx(ln({k}x)) = 0 + d/dx(ln x) = 1/x.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"splits ln({k}x) and differentiates, getting 1/x"),
            ("logical_completeness", "uses both the log law and the given derivative"),
            ("definitions_assumptions", f"says ln {k} is a constant and differentiates to 0"),
            ("communication", "ends with the stated result"),
        ],
        traps=[
            ("chain_rule_slip", f"does it get {k}/({k}x) and not simplify, or lose the factor?"),
            ("differentiates_the_constant", f"does it treat ln {k} as if it depended on x?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_affine(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, -1, -2))
    b = rng.choice((1, 2, 3, 5, 6))
    arg = _aff(a, b)
    # evaluate at points where ax + b > 0
    xs = [sp.Rational(-b, a) + sp.Rational(d, 10) * (1 if a > 0 else -1) for d in (3, 7, 14)]
    u = sp.Symbol("u", real=True)
    for pt in xs:
        lhs = sp.diff(sp.log(a * u + b), u).subs(u, pt)
        rhs = (a / (a * u + b)).subs(u, pt)
        check(abs(sp.N(lhs - rhs)) < 1e-12 and a * pt + b > 0, "d/dx ln(ax+b)")
    claim = _ddx(_ln(arg)) + " = " + Fm("\\frac{" + str(a) + "}{" + arg.tex + "}", f"{a}/({arg.plain})")
    p = Prompt()
    p.t("Given that ").m(Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")).t(
        " for u > 0 and using the chain rule, prove that "
    ).m(claim).t(" for ").m(Fm(arg.tex + " > 0", arg.plain + " > 0")).t(". ").t(_tail(rng))
    ref = [
        f"Let u = {arg.plain}, so y = ln u, and u > 0 on the stated domain.",
        f"dy/du = 1/u (given) and du/dx = {a}.",
        f"The chain rule gives dy/dx = (1/u) × {a} = {a}/u.",
        f"Substitute back u = {arg.plain}: dy/dx = {a}/({arg.plain}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains {a}/({arg.plain}) from (1/u) × {a}"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", "notes that u > 0 on the domain, so the given derivative applies"),
            ("communication", "substitutes back so the answer is in x"),
        ],
        traps=[
            ("forgets_the_inner_derivative", f"does it stop at 1/({arg.plain}) and omit the factor {a}?"),
            ("ignores_the_domain", "does it say nothing about u being positive?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_power(rng) -> dict:
    n = rng.choice((2, 3, 4, 5, -1, -2, -3))
    check_equivalent(sp.diff(sp.log(x**n), x), n / x)
    arg = Fm(f"x^{{{n}}}", f"x^({n})" if n < 0 else f"x^{n}")
    claim = _ddx(_ln(arg)) + " = " + Fm("\\frac{" + str(n) + "}{x}", f"{n}/x")
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0 and using the law ").m(
        Fm("\\ln\\left(a^{n}\\right) = n \\ln a", "ln(a^n) = n ln a")
    ).t(", prove that ").m(claim).t(" for x > 0. ").t(_tail(rng))
    ref = [
        f"By the log law, ln(x^({n})) = {n} ln x for x > 0.",
        f"{n} is a constant, so d/dx({n} ln x) = {n} × d/dx(ln x).",
        f"By the given derivative, this is {n} × (1/x) = {n}/x.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {n}/x from {n} × (1/x)"),
            ("logical_completeness", "uses both the log law and the given derivative"),
            ("definitions_assumptions", "states x > 0 so the log law and the derivative apply"),
            ("communication", "ends with the stated result"),
        ],
        traps=[
            (
                "power_rule_confusion",
                "does it differentiate x^n by the power rule and put it over x^n, instead of using the log law?",
            ),
            ("drops_the_exponent", f"does it give 1/x and lose the factor {n}?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_quadratic(rng) -> dict:
    c = rng.choice((1, 2, 3, 4, 5))
    arg = Fm(f"x^{{2}} + {c}", f"x^2 + {c}")
    check_equivalent(sp.diff(sp.log(x**2 + c), x), 2 * x / (x**2 + c))
    claim = _ddx(_ln(arg)) + " = " + Fm("\\frac{2x}{x^{2} + " + str(c) + "}", f"2x/(x^2 + {c})")
    p = Prompt()
    p.t("Given that ").m(Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")).t(
        " for u > 0 and using the chain rule, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"Let u = x^2 + {c}, so y = ln u. Since x^2 + {c} ≥ {c} > 0, u is positive for every x.",
        "dy/du = 1/u (given) and du/dx = 2x.",
        "The chain rule gives dy/dx = (1/u) × 2x = 2x/u.",
        f"Substitute back u = x^2 + {c}: dy/dx = 2x/(x^2 + {c}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains 2x/(x^2 + {c}) from (1/u) × 2x"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", f"shows u = x^2 + {c} > 0 for all x, so ln u is defined"),
            ("communication", "substitutes back so the answer is in x"),
        ],
        traps=[
            ("forgets_the_inner_derivative", f"does it stop at 1/(x^2 + {c}) and omit the factor 2x?"),
            ("splits_the_log", "does it write ln(x^2 + c) = ln x^2 + ln c, which is false?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_antiderivative(rng) -> dict:
    n = rng.choice((2, 3, 4, 5))
    f = (x**n / n) * sp.log(x) - x**n / (n * n)
    check_equivalent(sp.diff(f, x), x ** (n - 1) * sp.log(x))
    f_fm = Fm(
        f"\\frac{{x^{{{n}}}}}{{{n}}} \\ln x - \\frac{{x^{{{n}}}}}{{{n * n}}}", f"(x^{n}/{n}) ln x - x^{n}/{n * n}"
    )
    rhs = Fm(f"x^{{{n - 1}}} \\ln x" if n > 2 else "x \\ln x", f"x^{n - 1} ln x" if n > 2 else "x ln x")
    claim = _ddx(f_fm) + " = " + rhs
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that ").m(claim).t(" for x > 0. ").t(_tail(rng))
    ref = [
        f"Product rule on (x^{n}/{n}) ln x: d/dx = x^{n - 1} ln x + (x^{n}/{n}) × (1/x).",
        f"(x^{n}/{n}) × (1/x) = x^{n - 1}/{n}.",
        f"d/dx(x^{n}/{n * n}) = {n}x^{n - 1}/{n * n} = x^{n - 1}/{n}.",
        f"So the derivative is x^{n - 1} ln x + x^{n - 1}/{n} - x^{n - 1}/{n} = x^{n - 1} ln x.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "applies the product rule and uses d/dx(ln x) = 1/x"),
            ("logical_completeness", "differentiates both terms and shows the extra terms cancel"),
            ("definitions_assumptions", "states x > 0 so that ln x exists"),
            ("communication", "shows the cancellation line by line"),
        ],
        traps=[
            (
                "product_rule_as_product_of_derivatives",
                "does it multiply the two derivatives instead of using u'v + uv'?",
            ),
            ("misses_the_cancellation", "does it leave the x^(n-1)/n terms without showing they cancel?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_log_base(rng) -> dict:
    a = rng.choice((2, 3, 5, 7, 10))
    expected = 1 / (x * sp.log(a))
    check_equivalent(sp.diff(sp.log(x) / sp.log(a), x), expected)
    check_equivalent(sp.log(x, a), sp.log(x) / sp.log(a))
    claim = Fm(
        f"\\frac{{d}}{{dx}}\\left(\\log_{{{a}}} x\\right) = \\frac{{1}}{{x \\ln {a}}}",
        f"d/dx(log_{a} x) = 1/(x ln {a})",
    )
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0 and using the change of base formula ").m(
        Fm("\\log_{a} x = \\frac{\\ln x}{\\ln a}", "log_a x = ln x / ln a")
    ).t(", prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"By the change of base formula, log_{a} x = ln x / ln {a}.",
        f"ln {a} is a constant (it does not depend on x), so d/dx(log_{a} x) = (1/ln {a}) × d/dx(ln x).",
        f"By the given derivative, this is (1/ln {a}) × (1/x) = 1/(x ln {a}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches 1/(x ln {a}) from (1/ln {a}) × (1/x)"),
            ("logical_completeness", "uses both the change of base formula and the given derivative"),
            ("definitions_assumptions", f"treats ln {a} as a constant"),
            ("communication", "ends with the stated result"),
        ],
        traps=[
            ("differentiates_the_base", f"does it differentiate ln {a} as if it depended on x?"),
            ("wrong_denominator", f"does it write 1/(x log_{a} {a}) or 1/({a}x) in place of 1/(x ln {a})?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_ratio(rng) -> dict:
    pp = rng.choice((1, 2, 3, 4, 5))
    f = sp.log((x + pp) / (x - pp))
    # check for x > p, shifting x
    for off in (sp.Rational(1, 2), sp.Rational(7, 3), sp.Rational(15, 4)):
        pt = pp + off
        d = sp.diff(sp.log(x + pp) - sp.log(x - pp), x).subs(x, pt)
        check(abs(sp.N(d + 2 * pp / (pt**2 - pp**2))) < 1e-12, "d/dx ln((x+p)/(x-p))")
    _ = f
    arg = Fm(f"\\frac{{x + {pp}}}{{x - {pp}}}", f"(x + {pp})/(x - {pp})")
    claim = _ddx(_ln(arg)) + " = " + Fm(f"-\\frac{{{2 * pp}}}{{x^{{2}} - {pp * pp}}}", f"-{2 * pp}/(x^2 - {pp * pp})")
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, the chain rule and the law ").m(
        Fm("\\ln\\left(\\frac{a}{b}\\right) = \\ln a - \\ln b", "ln(a/b) = ln a - ln b")
    ).t(", prove that ").m(claim).t(f" for x > {pp}. ").t(_tail(rng))
    ref = [
        f"For x > {pp}, both x + {pp} and x - {pp} are positive, so ln((x + {pp})/(x - {pp})) = ln(x + {pp}) - ln(x - {pp}).",
        f"By the chain rule (inner derivative 1 in each case), d/dx(ln(x + {pp})) = 1/(x + {pp}) and d/dx(ln(x - {pp})) = 1/(x - {pp}).",
        f"So the derivative is 1/(x + {pp}) - 1/(x - {pp}) = ((x - {pp}) - (x + {pp}))/((x + {pp})(x - {pp})).",
        f"The numerator is -{2 * pp} and the denominator is x^2 - {pp * pp}, so the derivative is -{2 * pp}/(x^2 - {pp * pp}).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "combines the two fractions to reach -2p/(x² - p²)"),
            ("logical_completeness", "splits the logarithm, differentiates each part, then combines"),
            ("definitions_assumptions", f"states x > {pp} so both logarithms are defined"),
            ("communication", "shows the common-denominator step"),
        ],
        traps=[
            (
                "quotient_rule_on_the_inside",
                "does it differentiate the fraction inside the log and forget to divide by it?",
            ),
            ("sign_slip", "does the subtraction of the two fractions give the right sign?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_first_principles(rng) -> dict:
    limit = Fm("\\lim_{u \\to 0} (1 + u)^{1/u} = e", "lim(u→0) (1 + u)^(1/u) = e")
    if rng.random() < 0.5:
        v = rng.choice(("x", "t"))
        claim = _ddx(_lnx(v), v) + " = " + Fm("\\frac{1}{" + v + "}", f"1/{v}")
        at = f" for {v} > 0"
        point = v
        res = f"1/{v}"
    else:
        a = rng.choice((2, 3, 4, 5, 6))
        claim = Fm(f"\\text{{gradient}} = \\frac{{1}}{{{a}}}", f"gradient = 1/{a}")
        at = ""
        point = str(a)
        res = f"1/{a}"
    p = Prompt()
    if point.isdigit():
        p.t("Given that ").m(limit).t(" and that ").m(Fm("\\ln", "ln")).t(
            f" is continuous, prove from first principles that the gradient of the curve "
        ).m(Fm("y = \\ln x", "y = ln x")).t(f" at the point where x = {point} is 1/{point}. ")
    else:
        p.t("Given that ").m(limit).t(" and that ").m(Fm("\\ln", "ln")).t(
            " is continuous, prove from first principles that "
        ).m(claim).t(at + ". ")
    p.t(_tail(rng))
    s = point
    ref = [
        f"By definition, the derivative at {s} is lim(h→0) (ln({s}+h) - ln {s})/h.",
        f"By the log law, ln({s}+h) - ln {s} = ln(1 + h/{s}), so the quotient is (1/h) ln(1 + h/{s}).",
        f"Let u = h/{s}. As h → 0, u → 0, and 1/h = 1/({s}u), so the quotient is (1/{s}) × (1/u) ln(1 + u) = (1/{s}) ln((1 + u)^(1/u)).",
        "Since ln is continuous and (1 + u)^(1/u) → e, ln((1 + u)^(1/u)) → ln e = 1.",
        f"So the limit is (1/{s}) × 1 = {res}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reduces the quotient to (1/{s}) ln((1 + u)^(1/u)) and evaluates the limit as {res}"),
            ("logical_completeness", "uses the log law, the substitution u = h/x, and the given limit in order"),
            ("definitions_assumptions", "says why the limit can pass inside ln: ln is continuous"),
            ("communication", "writes each limit with its variable"),
        ],
        traps=[
            ("assumes_the_result", "does it quote d/dx(ln x) = 1/x?"),
            (
                "substitution_without_change_of_limit",
                "does it substitute u = h/x but still take h → 0 in the final limit?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


def _dln_tangent_intercept(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, 6, 7))
    tangent = x / a + sp.log(a) - 1
    check_equivalent(tangent.subs(x, 0), sp.log(a) - 1)
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that the tangent to the curve ").m(Fm("y = \\ln x", "y = ln x")).t(
        f", drawn at the point where x = {a}, crosses the y-axis at the point (0, ln {a} - 1). "
    ).t(_tail(rng))
    ref = [
        f"At x = {a}, y = ln {a}, so the point of contact is ({a}, ln {a}).",
        f"The gradient is the value of 1/x at x = {a}, which is 1/{a}.",
        f"The tangent is y - ln {a} = (1/{a})(x - {a}).",
        f"It crosses the y-axis where x = 0: y - ln {a} = (1/{a})(-{a}) = -1.",
        f"So y = ln {a} - 1, and the crossing point is (0, ln {a} - 1).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"puts x = 0 into the tangent and obtains y = ln {a} - 1"),
            ("logical_completeness", "finds the tangent first, then its y-intercept"),
            (
                "definitions_assumptions",
                "says the y-axis is where x = 0, even though x = 0 is outside the domain of the curve itself",
            ),
            ("communication", "states the coordinates of the crossing point"),
        ],
        traps=[
            ("evaluates_the_curve_at_zero", "does it put x = 0 into y = ln x (undefined) instead of into the tangent?"),
            ("wrong_gradient", f"is the gradient 1/{a}, not {a} or ln {a}?"),
        ],
    )
    return finish(p, f"the tangent crosses the y-axis at (0, ln {a} - 1)", mark)


def _dln_minimum(rng) -> dict:
    k = rng.choice((1, 2, 3, 4, 5))
    f = x - k * sp.log(x)
    check_equivalent(sp.diff(f, x), 1 - k / x)
    check(sp.solve(sp.Eq(sp.diff(f, x), 0), x) == [k], "stationary point at x = k")
    check(sp.diff(f, x, 2).subs(x, k) > 0, "minimum")
    value = "1" if k == 1 else f"{k} - {k} ln {k}"
    curve = Fm(f"y = x - {'' if k == 1 else k}\\ln x", f"y = x - {'' if k == 1 else str(k) + ' '}ln x")
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that the curve ").m(curve).t(
        f" (x > 0) has exactly one stationary point, that this is a minimum, and that the minimum value of y is {value if k == 1 else value}. "
    ).t(_tail(rng))
    kx = "1" if k == 1 else str(k)
    ref = [
        f"dy/dx = 1 - {kx}/x, using the given derivative of ln x.",
        f"Stationary points have dy/dx = 0: 1 = {kx}/x, so x = {k}. This is the only solution, and it is in the domain x > 0.",
        f"d²y/dx² = {kx}/x², which is positive at x = {k}, so the stationary point is a minimum.",
        f"At x = {k}: y = {k} - {kx} ln {k} = {value}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds dy/dx = 1 - {kx}/x, solves x = {k}, and evaluates y = {value}"),
            ("logical_completeness", "covers uniqueness, the nature of the point and the value, not only the first"),
            ("definitions_assumptions", "keeps to the domain x > 0"),
            ("communication", "states the second-derivative test and its sign"),
        ],
        traps=[
            ("differentiates_ln_as_ln", f"does it differentiate {kx} ln x as {kx}/ln x or similar?"),
            ("no_nature_test", "does it find the stationary point but not show it is a minimum?"),
        ],
    )
    return finish(p, f"stationary point at x = {k} with minimum value {value}", mark)


def _dln_asinh(rng) -> dict:
    c = rng.choice((1, 2, 3, 4, 5))
    g = x + sp.sqrt(x**2 + c)
    check_equivalent(sp.diff(sp.log(g), x), 1 / sp.sqrt(x**2 + c))
    arg = Fm(f"x + \\sqrt{{x^{{2}} + {c}}}", f"x + √(x^2 + {c})")
    claim = _ddx(_ln(arg)) + " = " + Fm(f"\\frac{{1}}{{\\sqrt{{x^{{2}} + {c}}}}}", f"1/√(x^2 + {c})")
    p = Prompt()
    p.t("Given that ").m(Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")).t(
        " for u > 0 and using the chain rule, prove that "
    ).m(claim).t(" for every real x. ").t(_tail(rng))
    s = f"√(x^2 + {c})"
    ref = [
        f"Let u = x + {s}. Since {s} > |x| for every x, u > 0, so ln u is defined.",
        f"du/dx = 1 + x/{s} = ({s} + x)/{s} = u/{s}.",
        f"By the chain rule dy/dx = (1/u) × du/dx = (1/u) × u/{s}.",
        f"The factor u cancels (it is not 0), leaving dy/dx = 1/{s}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds du/dx and simplifies it to u divided by the square root"),
            ("logical_completeness", "applies the chain rule, then cancels u"),
            ("definitions_assumptions", "shows u > 0 for all real x, so the given derivative applies"),
            ("communication", "writes u explicitly where it cancels"),
        ],
        traps=[
            (
                "wrong_inner_derivative",
                "is the derivative of the square root x/√(x² + c) with its factor 1/2 cancelled against the 2x?",
            ),
            (
                "cancels_without_reason",
                "does it cancel the sum x + √(x² + c) without showing it matches the numerator?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


DLN_FAMILIES = {
    "easy": (_dln_from_exp, _dln_tangent),
    "medium": (_dln_kx, _dln_affine, _dln_power, _dln_quadratic),
    "hard": (_dln_antiderivative, _dln_log_base, _dln_ratio),
    "exam": (_dln_first_principles, _dln_tangent_intercept, _dln_minimum, _dln_asinh),
}


# ===========================================================================
# integral of e^x
# ===========================================================================
def _iexp_antiderivative(rng) -> dict:
    v = rng.choice(("x", "t"))
    claim = _int(_e_pow(v), var=v) + " = " + _e_pow(v) + " + c"
    p = Prompt()
    p.t("Given that ").m(_ddx(_e_pow(v), v) + " = " + _e_pow(v)).t(", prove that ").m(claim).t(
        f", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"An antiderivative of f({v}) is a function F with F'({v}) = f({v}); the indefinite integral is F({v}) + c.",
        f"Differentiate e^{v} + c: d/d{v}(e^{v} + c) = e^{v} + 0 = e^{v}.",
        f"So e^{v} + c has derivative e^{v}, and every antiderivative of e^{v} differs from e^{v} by a constant.",
        f"Therefore the integral of e^{v} with respect to {v} is e^{v} + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates e^{v} + c to e^{v}"),
            ("logical_completeness", "links the derivative back to the definition of an antiderivative"),
            (
                "definitions_assumptions",
                "explains why the constant c is needed (any two antiderivatives differ by a constant)",
            ),
            ("communication", "states the conclusion as an integral"),
        ],
        traps=[
            ("integrates_by_rule_without_proof", "does it just write the answer with no differentiation check?"),
            ("forgets_the_constant", "does the conclusion omit + c?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_definite_basic(rng) -> dict:
    a = rng.choice((-2, -1, 0, 1))
    b = rng.choice([v for v in (1, 2, 3, 4) if v > a])
    value = sp.exp(b) - sp.exp(a)
    s = sp.Symbol("s")
    check_equivalent(value, sp.integrate(sp.exp(s), (s, a, b)))
    rhs = _e_pow(b) + " - " + _e_pow(a)
    claim = _int(_e_pow("x"), a, b) + " = " + rhs
    p = Prompt()
    p.t("Given that ").m(Fm("\\int e^{x}\\, dx = e^{x} + c", "∫ e^x dx = e^x + c")).t(
        " and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ea, eb = _e_pow(a).plain, _e_pow(b).plain
    ref = [
        "By the Fundamental Theorem of Calculus, the definite integral is F(upper) - F(lower) for any antiderivative F.",
        "Take F(x) = e^x, an antiderivative of e^x (the constant c is not needed for a definite integral).",
        f"The integral is F({b}) - F({a}) = {eb} - {ea}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates F({b}) - F({a}) = {eb} - {ea}"),
            ("logical_completeness", "names the theorem and the antiderivative, then substitutes both limits"),
            ("definitions_assumptions", "says the constant of integration cancels"),
            ("communication", "writes the substitution as F(upper) - F(lower)"),
        ],
        traps=[
            ("limits_reversed", "does it compute F(lower) - F(upper)?"),
            ("e_to_zero_slip", "does it treat e^0 as 0?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_k(rng) -> dict:
    k = Fraction(rng.choice((2, 3, 4, 5, -1, -2, -3, Fraction(1, 2), Fraction(-1, 2))))
    kk = sp.Rational(k.numerator, k.denominator)
    check_equivalent(sp.diff(sp.exp(kk * x) / kk, x), sp.exp(kk * x))
    arg = _kx(k)
    inv = _k_fm(1 / k)
    lead = Fm("\\left(" + inv.tex + "\\right)", "(" + inv.plain + ")") if k.denominator == 1 else inv
    if k.denominator == 1 and abs(k) == 1:
        lead = Fm("-") if k < 0 else Fm("")
    claim = _int(_e_pow(arg)) + " = " + lead + _e_pow(arg) + " + c"
    kp = _k_fm(k).plain
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and the chain rule, prove by differentiating the right-hand side that ").m(claim).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Let F(x) = ({inv.plain}) e^({arg.plain}). Differentiate with the chain rule.",
        f"d/dx(e^({arg.plain})) = {kp} e^({arg.plain}) (the given derivative times the inner derivative {kp}).",
        f"So F'(x) = ({inv.plain}) × {kp} × e^({arg.plain}) = e^({arg.plain}).",
        f"Hence F is an antiderivative of e^({arg.plain}), and the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "differentiates the proposed antiderivative and obtains e^(kx) exactly"),
            ("logical_completeness", "applies the chain rule and then links F' = e^(kx) to the integral"),
            ("definitions_assumptions", "notes k is not 0 so that 1/k exists"),
            ("communication", "ends with the integral including + c"),
        ],
        traps=[
            ("multiplies_by_k", "does the integral have the factor k instead of 1/k?"),
            ("forgets_the_constant", "does the conclusion omit + c?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_definite_k(rng) -> dict:
    k = rng.choice((2, 3, 4, -1, -2, -3))
    b = rng.choice((1, 2, 3))
    value = (sp.exp(k * b) - 1) / k
    s = sp.Symbol("s")
    check_equivalent(value, sp.integrate(sp.exp(k * s), (s, 0, b)))
    arg = _aff(k, 0)
    rhs = Fm(f"\\frac{{e^{{{k * b}}} - 1}}{{{k}}}", f"(e^({k * b}) - 1)/({k})" if k < 0 else f"(e^{k * b} - 1)/{k}")
    claim = _int(_e_pow(arg), 0, b) + " = " + rhs
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and the Fundamental Theorem of Calculus, prove that ").m(claim).t(". ").t(
        _tail(rng)
    )
    ref = [
        f"Let F(x) = e^({arg.plain})/{k}. By the chain rule F'(x) = ({k}/{k}) e^({arg.plain}) = e^({arg.plain}), so F is an antiderivative.",
        "By the Fundamental Theorem of Calculus the integral is F(upper) - F(lower).",
        f"F({b}) = e^({k * b})/{k} and F(0) = e^0/{k} = 1/{k}.",
        f"So the integral is (e^({k * b}) - 1)/{k}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds F(b) - F(0) correctly, using e^0 = 1"),
            ("logical_completeness", "proves F is an antiderivative before applying the theorem"),
            ("definitions_assumptions", "names the Fundamental Theorem of Calculus"),
            ("communication", "shows the substitution of both limits"),
        ],
        traps=[
            ("forgets_the_lower_limit", "does it forget to subtract F(0) = 1/k?"),
            ("wrong_factor", f"is the factor 1/{k} applied to the whole difference?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_linear(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, -2, -3))
    b = rng.choice((1, 2, 3, 4, -1, -2, -3))
    check_equivalent(sp.diff(a * sp.exp(x) + b * x, x), a * sp.exp(x) + b)
    integrand = _scale(a, _e_pow("x")) + f" {'+' if b > 0 else '-'} {abs(b)}"
    antiderivative = _scale(a, _e_pow("x")) + f" {'+' if b > 0 else '-'} " + ("" if abs(b) == 1 else str(abs(b))) + "x"
    claim = _int(integrand) + " = " + antiderivative + " + c"
    p = Prompt()
    p.t("Given that ").m(Fm("\\int e^{x}\\, dx = e^{x} + c", "∫ e^x dx = e^x + c")).t(", prove that ").m(claim).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Differentiate the right-hand side: d/dx({a}e^x {'+' if b > 0 else '-'} {abs(b)}x + c).",
        f"The derivative of {a}e^x is {a}e^x (given), the derivative of {b}x is {b}, and the derivative of c is 0.",
        f"So the derivative is {a}e^x {'+' if b > 0 else '-'} {abs(b)}, which is the integrand.",
        "Hence the right-hand side is an antiderivative of the integrand, and the integral is that expression + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "differentiates each term of the proposed antiderivative correctly"),
            ("logical_completeness", "compares the derivative with the integrand and concludes"),
            ("definitions_assumptions", "uses that the derivative of a constant is 0"),
            ("communication", "treats the two terms separately"),
        ],
        traps=[
            (
                "integrates_the_constant_as_a_constant",
                "does it integrate the constant term as if it stayed a constant instead of becoming a multiple of x?",
            ),
            ("scales_the_exponential", "does it change the coefficient of e^x when integrating?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_ln_limit(rng) -> dict:
    k = rng.choice((2, 3, 4, 5, 6, 7, 8, 9))
    s = sp.Symbol("s")
    check_equivalent(sp.integrate(sp.exp(s), (s, 0, sp.log(k))), k - 1)
    claim = _int(_e_pow("x"), 0, Fm(f"\\ln {k}", f"ln {k}")) + f" = {k - 1}"
    p = Prompt()
    p.t("Given that ").m(Fm("\\int e^{x}\\, dx = e^{x} + c", "∫ e^x dx = e^x + c")).t(" and that ").m(
        Fm(f"e^{{\\ln {k}}} = {k}", f"e^(ln {k}) = {k}")
    ).t(", prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        "By the Fundamental Theorem of Calculus the integral is F(upper) - F(lower) with F(x) = e^x.",
        f"F(ln {k}) = e^(ln {k}) = {k}.",
        "F(0) = e^0 = 1.",
        f"So the integral is {k} - 1 = {k - 1}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates e^(ln {k}) = {k} and e^0 = 1 and subtracts"),
            ("logical_completeness", "states the theorem and both substitutions"),
            ("definitions_assumptions", "names F(x) = e^x as the antiderivative"),
            ("communication", "shows the working to the final number"),
        ],
        traps=[
            ("e_to_ln_slip", f"does it leave e^(ln {k}) unevaluated or simplify it wrongly?"),
            ("e_zero_slip", "does it take e^0 as 0?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_affine(rng) -> dict:
    a = rng.choice((2, 3, 4, -1, -2, -3))
    b = rng.choice((1, 2, -1, -3))
    check_equivalent(sp.diff(sp.exp(a * x + b) / a, x), sp.exp(a * x + b))
    arg = _aff(a, b)
    claim = (
        _int(_e_pow(arg))
        + " = "
        + Fm(f"\\frac{{1}}{{{a}}}" if a > 0 else f"-\\frac{{1}}{{{-a}}}", f"(1/{a})" if a > 0 else f"(-1/{-a})")
        + _e_pow(arg)
        + " + c"
    )
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and the chain rule, prove by differentiating the right-hand side that ").m(claim).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Let F(x) = (1/{a}) e^({arg.plain}).",
        f"By the chain rule d/dx(e^({arg.plain})) = {a} e^({arg.plain}).",
        f"So F'(x) = (1/{a}) × {a} × e^({arg.plain}) = e^({arg.plain}).",
        "Hence F is an antiderivative of the integrand, and the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets F'(x) = e^({arg.plain}) after the factors {a} and 1/{a} cancel"),
            ("logical_completeness", "applies the chain rule, then links F' to the integral"),
            (
                "definitions_assumptions",
                f"notes {a} ≠ 0 so 1/{a} exists, and that the constant {b} does not change the derivative factor",
            ),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("divides_by_the_constant_term", f"does it divide by {b} or use the constant term in the factor?"),
            ("forgets_the_constant", "does the conclusion omit + c?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_x_exp(rng) -> dict:
    k = rng.choice((1, 2, 3, -1, -2))
    anti = (k * x - 1) / (k * k) * sp.exp(k * x)
    check_equivalent(sp.diff(anti, x), x * sp.exp(k * x))
    ex = _e_pow(_aff(k, 0))
    integrand = Fm("x") + ex
    if k == 1:
        coeff = Fm("\\left(x - 1\\right)", "(x - 1)")
        coeff_plain = "(x - 1)"
    elif k == -1:
        coeff = Fm("-\\left(x + 1\\right)", "-(x + 1)")
        coeff_plain = "-(x + 1)"
    else:
        top = _aff(k, -1)
        coeff = Fm(f"\\frac{{{top.tex}}}{{{k * k}}}", f"({top.plain})/{k * k}")
        coeff = Fm("\\left(" + coeff.tex + "\\right)", "(" + coeff.plain + ")")
        coeff_plain = coeff.plain
    claim = _int(integrand) + " = " + coeff + ex + " + c"
    inv_k = Fraction(1, k)
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(", prove by differentiating the right-hand side that ").m(claim).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    kx = _aff(k, 0).plain
    ref = [
        f"Let F(x) = {coeff_plain} e^({kx}). Use the product rule with u = {coeff_plain} and v = e^({kx}).",
        f"u' = {inv_k} (the derivative of the linear factor) and v' = {k}e^({kx}) (chain rule and the given derivative).",
        f"F'(x) = u'v + uv' = e^({kx}) × [{inv_k} + {k} × {coeff_plain}].",
        f"The bracket simplifies to x: the constant parts {inv_k} and -{inv_k} cancel and the x term is x.",
        f"So F'(x) = x e^({kx}), and F is an antiderivative of the integrand; the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "applies the product rule and the chain rule factor k correctly"),
            ("logical_completeness", "simplifies the bracket to x and concludes F' equals the integrand"),
            ("definitions_assumptions", "notes the constant c has derivative 0"),
            ("communication", "shows the combining of like terms"),
        ],
        traps=[
            ("product_rule_slip", "does it forget the u'v term, or the chain-rule factor in v'?"),
            ("tests_a_point_only", "does it check one value of x instead of the identity for all x?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_square(rng) -> dict:
    a = rng.choice((1, 2, 3, -1, -2))
    anti = sp.exp(2 * x) / 2 + 2 * a * sp.exp(x) + a * a * x
    check_equivalent(sp.diff(anti, x), (sp.exp(x) + a) ** 2)
    base = _e_pow("x") + f" {'+' if a > 0 else '-'} {abs(a)}"
    integrand = Fm("\\left(" + base.tex + "\\right)^{2}", "(" + base.plain + ")^2")
    rhs = (
        Fm("\\frac{1}{2}e^{2x}", "(1/2)e^(2x)")
        + f" {'+' if a > 0 else '-'} {abs(2 * a)}"
        + _e_pow("x")
        + f" + {a * a if a * a != 1 else ''}x"
    )
    claim = _int(integrand) + " = " + rhs + " + c"
    p = Prompt()
    p.t("Given that ").m(Fm("\\int e^{x}\\, dx = e^{x} + c", "∫ e^x dx = e^x + c")).t(" and ").m(
        Fm("\\int e^{2x}\\, dx = \\frac{1}{2}e^{2x} + c", "∫ e^(2x) dx = (1/2)e^(2x) + c")
    ).t(", prove that ").m(claim).t(", where c is a constant. ").t(_tail(rng))
    ref = [
        f"Expand the square: (e^x {'+' if a > 0 else '-'} {abs(a)})^2 = e^(2x) {'+' if a > 0 else '-'} {abs(2 * a)}e^x + {a * a}, using (e^x)^2 = e^(2x).",
        "Integrate term by term (the integral of a sum is the sum of the integrals).",
        f"∫ e^(2x) dx = (1/2)e^(2x), ∫ {abs(2 * a)}e^x dx = {abs(2 * a)}e^x and ∫ {a * a} dx = {a * a}x.",
        "Add the three results and one constant of integration c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "expands the square correctly, with e^(2x) and the middle term"),
            ("logical_completeness", "integrates all three terms, including the constant term"),
            ("definitions_assumptions", "uses term-by-term integration and one constant c"),
            ("communication", "shows the expansion before integrating"),
        ],
        traps=[
            (
                "integrates_the_square_of_the_integral",
                "does it square e^x + a and then use the integral of the base, instead of expanding first?",
            ),
            ("misses_the_half", "does it integrate e^(2x) as e^(2x) or 2e^(2x)?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_fraction(rng) -> dict:
    k = rng.choice((1, 2, 3, 5))
    check_equivalent(sp.diff(sp.log(k + sp.exp(x)), x), sp.exp(x) / (k + sp.exp(x)))
    denominator = f"{k} + e^{{x}}"
    integrand = Fm(f"\\frac{{e^{{x}}}}{{{denominator}}}", f"e^x/({k} + e^x)")
    claim = _int(integrand) + " = " + Fm(f"\\ln\\left({denominator}\\right)", f"ln({k} + e^x)") + " + c"
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and ").m(
        Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")
    ).t(" for u > 0, prove by differentiating the right-hand side that ").m(claim).t(", where c is a constant. ").t(
        _tail(rng)
    )
    ref = [
        f"Let F(x) = ln({k} + e^x). Since e^x > 0, {k} + e^x > 0 for every x, so F is defined.",
        f"Let u = {k} + e^x, so F = ln u, with dF/du = 1/u and du/dx = e^x (given derivative of e^x).",
        f"By the chain rule F'(x) = (1/u) × e^x = e^x/({k} + e^x).",
        "So F is an antiderivative of the integrand, and the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets F'(x) = e^x/({k} + e^x) from the chain rule"),
            ("logical_completeness", "differentiates and compares with the integrand, then concludes"),
            ("definitions_assumptions", f"notes {k} + e^x > 0 for all x, so ln of it exists"),
            ("communication", "ends with + c"),
        ],
        traps=[
            (
                "splits_the_fraction",
                "does it split e^x/(k + e^x) into separate integrals as if the denominator were a sum of terms?",
            ),
            ("drops_the_inner_derivative", "does it differentiate ln u as 1/u and forget e^x?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_x_pow(rng) -> dict:
    n = rng.choice((2, 3, 4))
    check_equivalent(sp.diff(sp.exp(x**n) / n, x), x ** (n - 1) * sp.exp(x**n))
    power = "x" if n == 1 else f"x^{{{n}}}"
    integrand = Fm(
        (f"x^{{{n - 1}}}" if n > 2 else "x") + f"e^{{{power}}}", (f"x^{n - 1}" if n > 2 else "x") + f"e^(x^{n})"
    )
    claim = _int(integrand) + " = " + Fm(f"\\frac{{1}}{{{n}}}e^{{{power}}}", f"(1/{n})e^(x^{n})") + " + c"
    p = Prompt()
    p.t("Given that ").m(D_EXP).t(" and the chain rule, prove by differentiating the right-hand side that ").m(claim).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Let F(x) = (1/{n}) e^(x^{n}), and u = x^{n}.",
        f"By the chain rule d/dx(e^u) = e^u × du/dx = e^(x^{n}) × {n}x^{n - 1}.",
        f"So F'(x) = (1/{n}) × {n}x^{n - 1} e^(x^{n}) = x^{n - 1} e^(x^{n}).",
        "Hence F is an antiderivative of the integrand, and the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains x^{n - 1}e^(x^{n}) after the factor {n} cancels the 1/{n}"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", "keeps the constant c in the conclusion"),
            ("communication", "shows the cancellation of the constant factor"),
        ],
        traps=[
            ("integrates_the_exponent", "does it integrate the exponent x^n and use e^(x^(n+1)/(n+1))?"),
            ("misses_the_chain_factor", f"does it forget the factor {n}x^({n - 1}) when checking by differentiation?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _iexp_area_between(rng) -> dict:
    a = rng.choice((1, 2, 3))
    s = sp.Symbol("s")
    value = sp.exp(a) + sp.exp(-a) - 2
    check_equivalent(sp.integrate(sp.exp(s) - sp.exp(-s), (s, 0, a)), value)
    rhs = _e_pow(a) + " + " + _e_pow(-a) + " - 2"
    p = Prompt()
    p.t("The curves ").m(Fm("y = e^{x}", "y = e^x")).t(" and ").m(Fm("y = e^{-x}", "y = e^(-x)")).t(
        f" meet at x = 0, and e^x ≥ e^(-x) for x ≥ 0. Given that "
    ).m(Fm("\\int e^{x}\\, dx = e^{x} + c", "∫ e^x dx = e^x + c")).t(" and ").m(
        Fm("\\int e^{-x}\\, dx = -e^{-x} + c", "∫ e^(-x) dx = -e^(-x) + c")
    ).t(f", prove that the area enclosed by the two curves and the line x = {a} is ").m(rhs).t(". ").t(_tail(rng))
    ref = [
        f"For 0 ≤ x ≤ {a}, e^x ≥ e^(-x), so the area is the integral of (e^x - e^(-x)) from 0 to {a}.",
        "An antiderivative is F(x) = e^x + e^(-x), using the two given integrals.",
        f"F({a}) = {_e_pow(a).plain} + {_e_pow(-a).plain} and F(0) = e^0 + e^0 = 2.",
        f"So the area is F({a}) - F(0) = {_e_pow(a).plain} + {_e_pow(-a).plain} - 2.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "finds F(x) = e^x + e^(-x) and evaluates F(a) - F(0)"),
            ("logical_completeness", "sets up the area as the integral of the upper curve minus the lower curve"),
            ("definitions_assumptions", "states which curve is above on the interval, and where the curves meet"),
            ("communication", "shows the values of F at both limits"),
        ],
        traps=[
            ("sign_of_the_second_term", "does the antiderivative of -e^(-x) come out as +e^(-x)?"),
            ("area_as_sum", "does it add the two integrals instead of subtracting the lower curve?"),
        ],
    )
    return finish(p, f"area = {rhs.plain}", mark)


IEXP_FAMILIES = {
    "easy": (_iexp_antiderivative, _iexp_definite_basic),
    "medium": (_iexp_k, _iexp_definite_k, _iexp_linear, _iexp_ln_limit),
    "hard": (_iexp_affine, _iexp_x_exp, _iexp_square),
    "exam": (_iexp_fraction, _iexp_x_pow, _iexp_area_between),
}


# ===========================================================================
# integral of 1/x, x > 0
# ===========================================================================
INV_X = Fm("\\frac{1}{x}", "1/x")


def _irp_antiderivative(rng) -> dict:
    v = rng.choice(("x", "t"))
    inv = Fm("\\frac{1}{" + v + "}", f"1/{v}")
    claim = _int(inv, var=v) + " = " + _lnx(v) + " + c"
    p = Prompt()
    p.t("Given that ").m(_ddx(_lnx(v), v) + " = " + inv).t(f" for {v} > 0, prove that ").m(claim).t(
        f" for {v} > 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Differentiate ln {v} + c for {v} > 0: d/d{v}(ln {v} + c) = 1/{v} + 0 = 1/{v}.",
        f"So ln {v} + c is an antiderivative of 1/{v} on {v} > 0.",
        f"Any two antiderivatives differ by a constant, so every antiderivative of 1/{v} has this form.",
        f"Therefore the integral of 1/{v} with respect to {v} is ln {v} + c for {v} > 0.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates ln {v} + c to 1/{v}"),
            ("logical_completeness", "links the derivative to the definition of an antiderivative"),
            ("definitions_assumptions", f"states the domain {v} > 0 where ln {v} is defined"),
            ("communication", "ends with the stated integral including + c"),
        ],
        traps=[
            ("power_rule_on_the_integral", "does it try the power rule, which would give x^0/0?"),
            ("forgets_the_domain", "does it say nothing about x being positive?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_one_to_a(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, 6, 7, 8, 9))
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, 1, a)), sp.log(a))
    claim = _int(INV_X, 1, a) + " = " + Fm(f"\\ln {a}", f"ln {a}")
    p = Prompt()
    p.t("Given that ").m(Fm("\\int \\frac{1}{x}\\, dx = \\ln x + c", "∫ 1/x dx = ln x + c")).t(
        " for x > 0 and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        "Take the antiderivative F(x) = ln x, which is defined on the whole interval [1, a] because x > 0 there.",
        "By the Fundamental Theorem of Calculus the integral is F(upper) - F(lower).",
        f"It equals ln {a} - ln 1.",
        f"Since ln 1 = 0, the integral is ln {a}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates ln {a} - ln 1 = ln {a}"),
            ("logical_completeness", "names the theorem, the antiderivative and both substitutions"),
            ("definitions_assumptions", "notes the interval lies in x > 0, where ln x and 1/x are defined"),
            ("communication", "shows ln 1 = 0 explicitly"),
        ],
        traps=[
            ("ln_one_slip", "does it treat ln 1 as 1?"),
            ("limits_reversed", "does it compute ln 1 - ln a?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_ratio(rng) -> dict:
    a = rng.choice((1, 2, 3))
    m = rng.choice((2, 3, 4, 5))
    b = a * m
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, a, b)), sp.log(m))
    claim = _int(INV_X, a, b) + " = " + Fm(f"\\ln {m}", f"ln {m}")
    p = Prompt()
    p.t("Given that ").m(Fm("\\int \\frac{1}{x}\\, dx = \\ln x + c", "∫ 1/x dx = ln x + c")).t(
        " for x > 0, the Fundamental Theorem of Calculus and the law "
    ).m(Fm("\\ln a - \\ln b = \\ln\\left(\\frac{a}{b}\\right)", "ln a - ln b = ln(a/b)")).t(", prove that ").m(claim).t(
        ". "
    ).t(_tail(rng))
    ref = [
        "The antiderivative F(x) = ln x is defined on [" + f"{a}, {b}] because x > 0 there.",
        f"By the Fundamental Theorem of Calculus the integral is ln {b} - ln {a}.",
        f"By the log law, ln {b} - ln {a} = ln({b}/{a}).",
        f"{b}/{a} = {m}, so the integral is ln {m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reduces ln {b} - ln {a} to ln {m}"),
            ("logical_completeness", "uses the theorem and the log law in order"),
            ("definitions_assumptions", "notes the interval lies in x > 0"),
            ("communication", f"shows {b}/{a} = {m}"),
        ],
        traps=[
            ("subtracts_the_arguments", f"does it write ln({b} - {a}) instead of ln({b}/{a})?"),
            ("limits_reversed", "does it compute ln a - ln b?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_const_mult(rng) -> dict:
    k = rng.choice((2, 3, 4, 5, 6, -1, -2, -3))
    check_equivalent(sp.diff(k * sp.log(x), x), k / x)
    claim = _int(Fm("\\frac{" + str(k) + "}{x}", f"{k}/x")) + " = " + _scale(k, _lnx("x")) + " + c"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove by differentiating the right-hand side that ").m(claim).t(
        " for x > 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Differentiate {k} ln x + c for x > 0.",
        f"The constant factor {k} stays, so d/dx({k} ln x) = {k} × (1/x) = {k}/x, using the given derivative.",
        "The derivative of the constant c is 0.",
        f"So the derivative is {k}/x, the integrand, and the integral is {k} ln x + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"differentiates {k} ln x to {k}/x"),
            ("logical_completeness", "uses the constant-multiple rule and the given derivative, then concludes"),
            ("definitions_assumptions", "states x > 0"),
            ("communication", "ends with + c"),
        ],
        traps=[
            (
                "moves_the_constant_into_the_log",
                f"does it write ln(x^{k}) without using the log law, or treat {k} ln x as ln({k}x)?",
            ),
            ("forgets_the_constant", "does the conclusion omit + c?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_scaled(rng) -> dict:
    a = rng.choice((2, 3, 4, 5))
    check_equivalent(sp.diff(sp.log(x) / a, x), 1 / (a * x))
    integrand = Fm(f"\\frac{{1}}{{{a}x}}", f"1/({a}x)")
    claim = _int(integrand) + " = " + Fm(f"\\frac{{1}}{{{a}}}\\ln x", f"(1/{a}) ln x") + " + c"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove by differentiating the right-hand side that ").m(claim).t(
        " for x > 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Differentiate (1/{a}) ln x + c for x > 0.",
        f"The constant factor 1/{a} stays: d/dx((1/{a}) ln x) = (1/{a}) × (1/x).",
        f"(1/{a}) × (1/x) = 1/({a}x), the integrand.",
        f"So the integral is (1/{a}) ln x + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches 1/({a}x) from (1/{a}) × (1/x)"),
            ("logical_completeness", "uses the constant-multiple rule and the given derivative"),
            ("definitions_assumptions", "states x > 0"),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("ln_of_ax", f"does it integrate to ln({a}x) and not show that this differs only by a constant?"),
            ("factor_inverted", f"does it write {a} ln x instead of (1/{a}) ln x?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_poly_over_x(rng) -> dict:
    k = rng.choice((1, 2, 3, 4, 5))
    check_equivalent(sp.diff(x + k * sp.log(x), x), (x + k) / x)
    integrand = Fm(f"\\frac{{x + {k}}}{{x}}", f"(x + {k})/x")
    claim = (
        _int(integrand)
        + " = "
        + Fm("x + " + ("" if k == 1 else str(k) + " ") + "\\ln x", "x + " + ("" if k == 1 else str(k) + " ") + "ln x")
        + " + c"
    )
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that ").m(claim).t(" for x > 0, where c is a constant. ").t(
        _tail(rng)
    )
    ref = [
        f"Split the fraction: (x + {k})/x = 1 + {k}/x for x > 0.",
        f"The integral of 1 is x, and the integral of {k}/x is {k} ln x for x > 0 (differentiating {k} ln x gives {k}/x).",
        f"So the integral is x + {k} ln x + c.",
        f"Check: d/dx(x + {k} ln x + c) = 1 + {k}/x = (x + {k})/x.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"splits to 1 + {k}/x and integrates each part"),
            ("logical_completeness", "justifies the integral of k/x by differentiation or by the given derivative"),
            ("definitions_assumptions", "states x > 0"),
            ("communication", "ends with + c and a check"),
        ],
        traps=[
            (
                "integrates_numerator_and_denominator_separately",
                "does it integrate the numerator and the denominator separately and divide?",
            ),
            ("forgets_the_constant", "does the conclusion omit + c?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_e_power(rng) -> dict:
    n = rng.choice((2, 3, 4, 5, 6))
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, 1, sp.exp(n))), n)
    claim = _int(INV_X, 1, _e_pow(n)) + f" = {n}"
    p = Prompt()
    p.t("Given that ").m(Fm("\\int \\frac{1}{x}\\, dx = \\ln x + c", "∫ 1/x dx = ln x + c")).t(
        " for x > 0, the Fundamental Theorem of Calculus and "
    ).m(Fm(f"\\ln\\left(e^{{{n}}}\\right) = {n}", f"ln(e^{n}) = {n}")).t(", prove that ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"The antiderivative F(x) = ln x is defined on [1, e^{n}] because x > 0 there.",
        f"By the Fundamental Theorem of Calculus the integral is ln(e^{n}) - ln 1.",
        f"ln(e^{n}) = {n} (given) and ln 1 = 0.",
        f"So the integral is {n}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates ln(e^{n}) - ln 1 = {n}"),
            ("logical_completeness", "names the theorem and both limits"),
            ("definitions_assumptions", "notes the interval lies in x > 0"),
            ("communication", "shows ln 1 = 0"),
        ],
        traps=[
            ("e_to_n_slip", f"does it write ln(e^{n}) = e^{n} or similar?"),
            ("limits_reversed", "does it subtract in the wrong order?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_affine(rng) -> dict:
    a = rng.choice((2, 3, 4, 5))
    b = rng.choice((1, 2, 3, -1, -2))
    u = sp.Symbol("u", real=True)
    for d in (sp.Rational(1, 3), sp.Rational(7, 4), sp.Rational(5, 1)):
        pt = sp.Rational(-b, a) + d
        diff_val = sp.diff(sp.log(a * u + b) / a, u).subs(u, pt)
        check(abs(sp.N(diff_val - 1 / (a * pt + b))) < 1e-12 and a * pt + b > 0, "antiderivative of 1/(ax+b)")
    arg = _aff(a, b)
    integrand = Fm("\\frac{1}{" + arg.tex + "}", f"1/({arg.plain})")
    claim = (
        _int(integrand)
        + " = "
        + Fm(f"\\frac{{1}}{{{a}}}\\ln\\left({arg.tex}\\right)", f"(1/{a}) ln({arg.plain})")
        + " + c"
    )
    p = Prompt()
    p.t("Given that ").m(Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")).t(
        " for u > 0 and the chain rule, prove by differentiating the right-hand side that "
    ).m(claim).t(" on any interval where ").m(Fm(arg.tex + " > 0", arg.plain + " > 0")).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Let F(x) = (1/{a}) ln({arg.plain}) and u = {arg.plain}, which is positive on the interval.",
        f"By the chain rule d/dx(ln u) = (1/u) × du/dx = {a}/({arg.plain}).",
        f"So F'(x) = (1/{a}) × {a}/({arg.plain}) = 1/({arg.plain}).",
        "Hence F is an antiderivative of the integrand, and the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets F'(x) = 1/({arg.plain}) after the factors {a} and 1/{a} cancel"),
            ("logical_completeness", "applies the chain rule with a stated inner function"),
            ("definitions_assumptions", "states that u > 0 on the interval, so the given derivative applies"),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("drops_the_inner_factor", f"does it forget the factor {a} when differentiating ln({arg.plain})?"),
            ("log_of_the_constant", f"does it involve ln {abs(b)} or ln {a} where none belongs?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_scaling(rng) -> dict:
    a = rng.choice((2, 3, 4, 5))
    b = rng.choice((2, 3, 4, 5))
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, a, a * b)), sp.integrate(1 / s, (s, 1, b)))
    claim = _int(INV_X, a, a * b) + " = " + _int(INV_X, 1, b)
    p = Prompt()
    p.t("Without using the logarithm function, prove that ").m(claim).t(" by using the substitution ").m(
        Fm(f"u = \\frac{{x}}{{{a}}}", f"u = x/{a}")
    ).t(". ").t(_tail(rng))
    ref = [
        f"Let u = x/{a}. Then x = {a}u and dx = {a} du.",
        f"When x = {a}, u = 1. When x = {a * b}, u = {b}.",
        f"The integrand 1/x dx becomes (1/({a}u)) × {a} du = (1/u) du.",
        f"So the integral from {a} to {a * b} of 1/x dx equals the integral from 1 to {b} of 1/u du.",
        "The letter used for the variable of integration does not change the value, so this is the integral from 1 to "
        + f"{b} of 1/x dx.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"transforms the integrand and gets the new limits 1 and {b}"),
            ("logical_completeness", "changes the limits, the integrand and the differential together"),
            ("definitions_assumptions", "notes the variable of integration is a dummy variable"),
            ("communication", "shows each substitution"),
        ],
        traps=[
            ("keeps_the_old_limits", f"does it leave the limits as {a} and {a * b} after substituting?"),
            ("forgets_dx", f"does it drop the factor {a} from dx = {a} du?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_ln_power(rng) -> dict:
    n = rng.choice((1, 2, 3, 4))
    check_equivalent(sp.diff(sp.log(x) ** (n + 1) / (n + 1), x), sp.log(x) ** n / x)
    num = Fm("\\left(\\ln x\\right)" + (f"^{{{n}}}" if n > 1 else ""), "(ln x)" + (f"^{n}" if n > 1 else ""))
    integrand = Fm("\\frac{" + num.tex + "}{x}", f"{num.plain}/x")
    rhs = Fm(f"\\frac{{\\left(\\ln x\\right)^{{{n + 1}}}}}{{{n + 1}}}", f"(ln x)^{n + 1}/{n + 1}")
    claim = _int(integrand) + " = " + rhs + " + c"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove by differentiating the right-hand side that ").m(claim).t(
        " for x > 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Let F(x) = (ln x)^{n + 1}/{n + 1} and u = ln x, so F = u^{n + 1}/{n + 1}.",
        f"By the chain rule F'(x) = u^{n} × du/dx = (ln x)^{n} × (1/x), using the given derivative.",
        f"So F'(x) = (ln x)^{n}/x, the integrand, for x > 0.",
        "Hence the integral is F(x) + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains (ln x)^{n}/x from the chain rule"),
            ("logical_completeness", "uses the given derivative of ln x as the inner derivative"),
            ("definitions_assumptions", "states x > 0"),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("drops_the_inner_derivative", "does it differentiate the power and omit the factor 1/x?"),
            ("integrates_numerator_alone", "does it integrate (ln x)^n and divide by x?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_quadratic_over_x(rng) -> dict:
    a = rng.choice((2, 4, 6, 1, 3))
    b = rng.choice((1, 2, 3, 5))
    c = rng.choice((1, 2, 3, 4))
    anti = sp.Rational(a, 2) * x**2 + b * x + c * sp.log(x)
    check_equivalent(sp.diff(anti, x), (a * x**2 + b * x + c) / x)
    lead = "" if a == 1 else str(a)
    bx = "x" if b == 1 else f"{b}x"
    top = Fm(f"{lead}x^{{2}} + {bx} + {c}", f"{lead}x^2 + {bx} + {c}")
    integrand = Fm("\\frac{" + top.tex + "}{x}", f"({top.plain})/x")
    half = Fraction(a, 2)
    if half.denominator == 1:
        h = "" if half == 1 else str(half.numerator)
        lead_term = Fm(f"{h}x^{{2}}", f"{h}x^2")
    else:
        lead_term = Fm(
            f"\\frac{{{half.numerator}x^{{2}}}}{{{half.denominator}}}", f"{half.numerator}x^2/{half.denominator}"
        )
    log_term = Fm(("" if c == 1 else f"{c} ") + "\\ln x", ("" if c == 1 else f"{c} ") + "ln x")
    rhs = lead_term + f" + {bx} + " + log_term
    claim = _int(integrand) + " = " + rhs + " + C"
    half_text = f"{half}" if half.denominator == 1 else f"({half})"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, prove that ").m(claim).t(
        " for x > 0, where C is a constant of integration. "
    ).t(_tail(rng))
    ref = [
        f"Split the fraction: ({top.plain})/x = {lead}x + {b} + {c}/x for x > 0.",
        f"Integrate term by term: the integral of {lead}x is {half_text}x^2 (power rule), of {b} is {bx}, and of {c}/x is {c} ln x for x > 0 (differentiating {c} ln x gives {c}/x by the given derivative).",
        f"Add the results and one constant: {half_text}x^2 + {bx} + {c} ln x + C.",
        f"Check: the derivative of the result is {lead}x + {b} + {c}/x, which equals the integrand.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "splits the fraction into three terms and integrates each correctly"),
            ("logical_completeness", "justifies the 1/x term by the given derivative, not by the power rule"),
            ("definitions_assumptions", "states x > 0 and uses one constant of integration"),
            ("communication", "shows the split before integrating"),
        ],
        traps=[
            ("power_rule_on_one_over_x", "does it integrate c/x by the power rule, which would divide by 0?"),
            ("divides_term_by_term_wrongly", "does it integrate the numerator and divide by x instead of splitting?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_ln_ln(rng) -> dict:
    k = rng.choice((1, 2, 3))
    check_equivalent(sp.diff(k * sp.log(sp.log(x)), x), k / (x * sp.log(x)))
    integrand = Fm(f"\\frac{{{k if k != 1 else 1}}}{{x \\ln x}}", f"{k}/(x ln x)")
    rhs = Fm(("" if k == 1 else str(k)) + "\\ln\\left(\\ln x\\right)", ("" if k == 1 else str(k) + " ") + "ln(ln x)")
    claim = _int(integrand) + " = " + rhs + " + c"
    p = Prompt()
    p.t("Given that ").m(D_LN).t(" for x > 0, and that ").m(
        Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")
    ).t(" for u > 0, prove by differentiating the right-hand side that ").m(claim).t(
        " for x > 1, where c is a constant. "
    ).t(_tail(rng))
    kk = "" if k == 1 else f"{k} × "
    ref = [
        "For x > 1, u = ln x > 0, so ln(ln x) is defined.",
        "Let y = ln u. By the chain rule dy/dx = (1/u) × du/dx = (1/ln x) × (1/x), using the given derivative of ln x.",
        f"So d/dx({'' if k == 1 else str(k) + ' '}ln(ln x)) = {kk}1/(x ln x) = {k}/(x ln x).",
        "Hence the right-hand side is an antiderivative of the integrand, and the integral is that expression + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "gets 1/(x ln x) from the chain rule applied twice (outer ln, inner ln x)"),
            ("logical_completeness", "uses both given derivatives, for ln u and for ln x"),
            ("definitions_assumptions", "shows x > 1 so that ln x > 0 and ln(ln x) exists"),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("misses_the_second_factor", "does it differentiate ln(ln x) as 1/ln x and omit the 1/x?"),
            ("ignores_the_domain", "does it fail to say why ln(ln x) needs x > 1?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irp_log_law_integral(rng) -> dict:
    a, b = rng.choice(((2, 3), (2, 5), (3, 4), (3, 5), (2, 7), (4, 5), (3, 7), (5, 6)))
    s = sp.Symbol("s", positive=True)
    check_equivalent(
        sp.integrate(1 / s, (s, 1, a * b)), sp.integrate(1 / s, (s, 1, a)) + sp.integrate(1 / s, (s, 1, b))
    )
    claim = _int(INV_X, 1, a * b) + " = " + _int(INV_X, 1, a) + " + " + _int(INV_X, 1, b)
    p = Prompt()
    p.t("Without using the logarithm function, prove that ").m(claim).t(
        f". You may split an integral at an interior point and substitute "
    ).m(Fm(f"u = \\frac{{x}}{{{a}}}", f"u = x/{a}")).t(". ").t(_tail(rng))
    ref = [
        f"Split at x = {a}: the integral from 1 to {a * b} equals the integral from 1 to {a} plus the integral from {a} to {a * b}.",
        f"For the second part, let u = x/{a}, so x = {a}u and dx = {a} du.",
        f"The limits x = {a} and x = {a * b} become u = 1 and u = {b}, and (1/x) dx becomes (1/({a}u)) × {a} du = (1/u) du.",
        f"So the integral from {a} to {a * b} of 1/x dx equals the integral from 1 to {b} of 1/u du.",
        f"Therefore the integral from 1 to {a * b} equals the integral from 1 to {a} plus the integral from 1 to {b}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"splits at {a}, substitutes u = x/{a}, and identifies the second integral with the one from 1 to {b}",
            ),
            ("logical_completeness", "handles the limits, integrand and differential in the substitution"),
            ("definitions_assumptions", "does not use ln or its log laws, as the question requires"),
            ("communication", "shows the split before the substitution"),
        ],
        traps=[
            (
                "uses_the_log_law",
                "does it write ln(ab) = ln a + ln b, which is what the integrals are meant to establish?",
            ),
            (
                "wrong_split_point",
                f"does it split at a point other than {a}, so the substitution no longer gives limits 1 and {b}?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


IRP_FAMILIES = {
    "easy": (_irp_antiderivative, _irp_one_to_a, _irp_ratio),
    "medium": (_irp_const_mult, _irp_scaled, _irp_poly_over_x, _irp_e_power),
    "hard": (_irp_affine, _irp_scaling, _irp_ln_power),
    "exam": (_irp_quadratic_over_x, _irp_ln_ln, _irp_log_law_integral),
}


# ===========================================================================
# integral of 1/x on x < 0 (ln|x|)
# ===========================================================================
D_LN_U = Fm("\\frac{d}{du}\\left(\\ln u\\right) = \\frac{1}{u}", "d/du(ln u) = 1/u")
INT_RECIP_ABS = Fm("\\int \\frac{1}{x}\\, dx = \\ln|x| + c", "∫ 1/x dx = ln|x| + c")


def _irn_deriv_neg(rng) -> dict:
    v = rng.choice(("x", "t"))
    inv = Fm("\\frac{1}{" + v + "}", f"1/{v}")
    if rng.random() < 0.5:
        form = Fm("\\ln(-" + v + ")", f"ln(-{v})")
        how = f"ln(-{v})"
    else:
        form = Fm("\\ln|" + v + "|", f"ln|{v}|")
        how = f"ln|{v}|"
    claim = _ddx(form, v) + " = " + inv
    p = Prompt()
    p.t("Given that ").m(D_LN_U).t(" for u > 0 and the chain rule, prove that ").m(claim).t(f" for {v} < 0. ")
    if how.startswith("ln|"):
        p.t(f"Recall that |{v}| = -{v} when {v} < 0. ")
    p.t(_tail(rng))
    ref = [
        f"For {v} < 0, |{v}| = -{v} > 0, so {how} = ln(-{v}) is defined."
        if how.startswith("ln|")
        else f"For {v} < 0, -{v} > 0, so ln(-{v}) is defined.",
        f"Let u = -{v}, so y = ln u with u > 0. Then dy/du = 1/u (given) and du/d{v} = -1.",
        f"The chain rule gives dy/d{v} = (1/u) × (-1) = -1/u.",
        f"Substitute back u = -{v}: dy/d{v} = -1/(-{v}) = 1/{v}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains -1/(-{v}) = 1/{v} from (1/u) × (-1)"),
            ("logical_completeness", "applies the chain rule with inner function u = -x"),
            ("definitions_assumptions", f"states that -{v} > 0 when {v} < 0, so the given derivative can be used"),
            ("communication", "substitutes back so the answer is in the original variable"),
        ],
        traps=[
            (
                "applies_given_derivative_to_negative",
                f"does it use d/du(ln u) = 1/u with u = {v} < 0, where ln u is not defined?",
            ),
            ("drops_the_sign", "does it lose the factor -1 from du/dx?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_tangent(rng) -> dict:
    a = rng.choice((1, 2, 3, 4))
    xx = sp.Symbol("xx", negative=True)
    tangent = -xx / a + sp.log(a) - 1
    check_equivalent(tangent.subs(xx, -a), sp.log(a))
    check_equivalent(sp.diff(tangent, xx), (1 / xx).subs(xx, -a))
    ln_a = "ln " + str(a) if a != 1 else "0"
    claim = Fm("y = -x - 1") if a == 1 else Fm(f"y = -\\frac{{x}}{{{a}}} + \\ln {a} - 1", f"y = -x/{a} + ln {a} - 1")
    p = Prompt()
    p.t("Given that ").m(Fm("\\frac{d}{dx}\\left(\\ln|x|\\right) = \\frac{1}{x}", "d/dx(ln|x|) = 1/x")).t(
        " for x < 0, prove that the tangent to the curve "
    ).m(Fm("y = \\ln|x|", "y = ln|x|")).t(f" at the point where x = -{a} has equation ").m(claim).t(". ").t(_tail(rng))
    ref = [
        f"At x = -{a}, y = ln|-{a}| = ln {a}, so the point of contact is (-{a}, {ln_a}).",
        f"The gradient is the value of dy/dx = 1/x at x = -{a}, which is -1/{a}.",
        f"The tangent is y - {ln_a} = (-1/{a})(x + {a}).",
        f"Expand: y = -x/{a} - 1 + {ln_a}, that is {claim.plain}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"uses the point (-{a}, {ln_a}) and gradient -1/{a} to reach the stated equation"),
            ("logical_completeness", "finds both the point and the gradient before writing the tangent"),
            ("definitions_assumptions", f"evaluates ln|x| as ln {a}, since |-{a}| = {a}"),
            ("communication", "shows the expansion to the stated form"),
        ],
        traps=[
            ("positive_gradient", f"does it use 1/{a} instead of -1/{a} as the gradient?"),
            ("ln_of_a_negative", f"does it write ln(-{a}) for the y-coordinate?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_antiderivative(rng) -> dict:
    claim = _int(INV_X) + " = " + Fm("\\ln|x| + c", "ln|x| + c")
    p = Prompt()
    p.t("Given that ").m(D_LN_U).t(" for u > 0 and the chain rule, prove that ").m(claim).t(
        " on any interval where x < 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        "For x < 0, |x| = -x, so ln|x| + c = ln(-x) + c, and -x > 0.",
        "Let u = -x, so y = ln u with u > 0: dy/du = 1/u (given) and du/dx = -1.",
        "By the chain rule d/dx(ln(-x)) = (1/u)(-1) = -1/(-x) = 1/x.",
        "The derivative of the constant c is 0, so d/dx(ln|x| + c) = 1/x for x < 0.",
        "Hence ln|x| + c is an antiderivative of 1/x on x < 0, and the integral is ln|x| + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "differentiates ln(-x) to 1/x for x < 0"),
            ("logical_completeness", "links the derivative to the definition of an antiderivative and includes + c"),
            ("definitions_assumptions", "uses |x| = -x for x < 0 and notes -x > 0"),
            ("communication", "ends with the stated integral"),
        ],
        traps=[
            ("ln_x_for_negative_x", "does it write ln x for x < 0, which is not defined?"),
            ("forgets_the_absolute_value", "does the conclusion drop the modulus signs?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_definite(rng) -> dict:
    pp = rng.choice((1, 2, 3, 4))
    m = rng.choice((2, 3, 4, 5, 6))
    q = pp * m
    s = sp.Symbol("s", positive=True)
    # integral over [-q, -p] of 1/x: substitute x = -s, dx = -ds: = integral_{s=q}^{p} (1/(-s))(-ds) = integral_q^p ds/s = -ln(q/p)
    check_equivalent(sp.integrate(1 / s, (s, q, pp)), -sp.log(m))
    claim = _int(INV_X, -q, -pp) + " = " + Fm(f"-\\ln {m}", f"-ln {m}")
    p = Prompt()
    p.t("Given that ").m(INT_RECIP_ABS).t(
        " on any interval where x < 0, and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"The interval [-{q}, -{pp}] lies in x < 0, so F(x) = ln|x| is an antiderivative of 1/x on it.",
        "By the Fundamental Theorem of Calculus the integral is F(upper) - F(lower).",
        f"F(-{pp}) = ln|-{pp}| = ln {pp} and F(-{q}) = ln|-{q}| = ln {q}.",
        f"So the integral is ln {pp} - ln {q} = ln({pp}/{q}) = ln(1/{m}) = -ln {m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates ln {pp} - ln {q} = -ln {m}"),
            ("logical_completeness", "names the theorem, the antiderivative with modulus, and both substitutions"),
            ("definitions_assumptions", "notes the whole interval lies in x < 0, so the modulus form applies"),
            ("communication", "shows ln|-a| = ln a explicitly"),
        ],
        traps=[
            ("ln_of_a_negative", f"does it write ln(-{pp}) or ln(-{q}), which are not defined?"),
            (
                "sign_of_the_result",
                "does it give +ln m, forgetting that the upper limit is the smaller in absolute value?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_const(rng) -> dict:
    k = rng.choice((2, 3, 4, 5, -1, -2, -3))
    check_equivalent(
        sp.diff(k * sp.log(-sp.Symbol("w", negative=True)), sp.Symbol("w", negative=True)),
        k / sp.Symbol("w", negative=True),
    )
    claim = _int(Fm("\\frac{" + str(k) + "}{x}", f"{k}/x")) + " = " + _scale(k, Fm("\\ln|x|", "ln|x|")) + " + c"
    p = Prompt()
    p.t("Given that ").m(INT_RECIP_ABS).t(" on any interval where x < 0, prove that ").m(claim).t(
        " on any interval where x < 0, where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"Write {k}/x = {k} × (1/x). A constant factor can be taken outside an integral.",
        f"So the integral is {k} × ∫ (1/x) dx = {k} × (ln|x| + c1) for x < 0, using the given result.",
        f"{k} c1 is itself an arbitrary constant, so the integral is {k} ln|x| + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {k} ln|x| + c"),
            ("logical_completeness", "takes out the constant factor and applies the given result"),
            ("definitions_assumptions", "explains that a constant times a constant is another constant"),
            ("communication", "keeps the modulus signs"),
        ],
        traps=[
            ("drops_the_modulus", "does the answer say ln x instead of ln|x|?"),
            ("multiplies_the_constant_wrongly", "does it add k to the constant instead of multiplying?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_affine(rng) -> dict:
    a = rng.choice((2, 3, 4, 5, -2, -3))
    b = rng.choice((1, 2, 3, -1, -2, -4))
    u = sp.Symbol("u", real=True)
    for d in (sp.Rational(1, 3), sp.Rational(7, 4), sp.Rational(5, 1)):
        pt = sp.Rational(-b, a) - d * (1 if a > 0 else -1)  # a*pt + b = -a*d*sign -> negative
        val = a * pt + b
        diff_val = sp.diff(sp.log(-(a * u + b)) / a, u).subs(u, pt)
        check(val < 0 and abs(sp.N(diff_val - 1 / val)) < 1e-12, "antiderivative of 1/(ax+b) where ax+b<0")
    arg = _aff(a, b)
    integrand = Fm("\\frac{1}{" + arg.tex + "}", f"1/({arg.plain})")
    claim = (
        _int(integrand)
        + " = "
        + Fm(
            f"\\frac{{1}}{{{a}}}\\ln\\left|{arg.tex}\\right|"
            if a > 0
            else f"-\\frac{{1}}{{{-a}}}\\ln\\left|{arg.tex}\\right|",
            f"(1/{a}) ln|{arg.plain}|",
        )
        + " + c"
    )
    p = Prompt()
    p.t("Given that ").m(D_LN_U).t(
        " for u > 0 and the chain rule, prove by differentiating the right-hand side that "
    ).m(claim).t(" on any interval where ").m(Fm(arg.tex + " < 0", arg.plain + " < 0")).t(
        ", where c is a constant. "
    ).t(_tail(rng))
    ref = [
        f"On the interval, {arg.plain} < 0, so |{arg.plain}| = -({arg.plain}) > 0.",
        f"Let u = -({arg.plain}), so ln|{arg.plain}| = ln u with u > 0, and du/dx = {-a}.",
        f"By the chain rule d/dx(ln u) = (1/u) × ({-a}) = {-a}/(-({arg.plain})) = {a}/({arg.plain}).",
        f"So d/dx((1/{a}) ln|{arg.plain}|) = (1/{a}) × {a}/({arg.plain}) = 1/({arg.plain}).",
        "Hence the right-hand side is an antiderivative of the integrand, and the integral is that expression + c.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets 1/({arg.plain}) after the factors {a} and 1/{a} cancel"),
            ("logical_completeness", "uses the modulus form through u = -(ax + b), then the chain rule"),
            (
                "definitions_assumptions",
                "states that ax + b < 0 on the interval, so the argument of the log is positive",
            ),
            ("communication", "ends with + c"),
        ],
        traps=[
            ("ln_of_a_negative", "does it apply ln directly to a negative expression?"),
            ("sign_of_the_inner_derivative", "does it lose the minus sign from u = -(ax + b)?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_symmetry(rng) -> dict:
    a = rng.choice((1, 2, 3))
    b = a + rng.choice((1, 2, 3, 4))
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, b, a)), -sp.integrate(1 / s, (s, a, b)))
    claim = _int(INV_X, -b, -a) + " = -" + _int(INV_X, a, b)
    p = Prompt()
    p.t("Without using the logarithm function, prove that ").m(claim).t(" by using the substitution ").m(
        Fm("u = -x", "u = -x")
    ).t(". ").t(_tail(rng))
    ref = [
        "Let u = -x. Then x = -u and dx = -du.",
        f"When x = -{b}, u = {b}. When x = -{a}, u = {a}.",
        "The integrand (1/x) dx becomes (1/(-u)) × (-du) = (1/u) du.",
        f"So the integral from -{b} to -{a} of 1/x dx equals the integral from {b} to {a} of 1/u du.",
        f"Swapping the limits changes the sign: that is -(the integral from {a} to {b} of 1/u du), which is -(the integral from {a} to {b} of 1/x dx).",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "gets the new limits and the integrand 1/u correctly"),
            (
                "logical_completeness",
                "changes the limits, the integrand and the differential together, then reverses the limits",
            ),
            ("definitions_assumptions", "notes the variable of integration is a dummy variable"),
            ("communication", "shows each substitution"),
        ],
        traps=[
            ("forgets_the_sign_of_dx", "does it replace dx by du instead of -du?"),
            ("does_not_swap_limits", "does it forget that the limits come out in the reverse order?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_definite_scaled(rng) -> dict:
    k = rng.choice((2, 3, 5, -1, -2))
    pp = rng.choice((1, 2, 3))
    m = rng.choice((2, 3, 4, 5))
    q = pp * m
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(k / s, (s, q, pp)), -k * sp.log(m))
    c = -k
    value = f"ln {m}" if c == 1 else f"-ln {m}" if c == -1 else f"{c} ln {m}"
    claim = _int(Fm("\\frac{" + str(k) + "}{x}", f"{k}/x"), -q, -pp) + " = " + Fm(value.replace("ln", "\\ln"), value)
    p = Prompt()
    p.t("Given that ").m(INT_RECIP_ABS).t(
        " on any interval where x < 0, and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"The interval [-{q}, -{pp}] lies in x < 0, so F(x) = {k} ln|x| is an antiderivative of {k}/x on it.",
        f"By the Fundamental Theorem of Calculus the integral is F(-{pp}) - F(-{q}).",
        f"F(-{pp}) = {k} ln {pp} and F(-{q}) = {k} ln {q}.",
        f"So the integral is {k}(ln {pp} - ln {q}) = {k} ln({pp}/{q}) = {k} ln(1/{m}) = {value}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"evaluates {k}(ln {pp} - ln {q}) as {value}"),
            ("logical_completeness", "names the theorem, the antiderivative with modulus, and both substitutions"),
            ("definitions_assumptions", "notes the interval lies in x < 0, so the modulus form applies"),
            ("communication", "shows the log law step ln a - ln b = ln(a/b)"),
        ],
        traps=[
            ("ln_of_a_negative", "does it write ln of a negative number?"),
            ("sign_slip", f"is the sign of {value} right given the limits are negative and in increasing order?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_poly_over_x(rng) -> dict:
    a = rng.choice((2, 4, 1, 3))
    b = rng.choice((1, 2, 3, -1, -2))
    c = rng.choice((1, 2, 3))
    pp = rng.choice((1, 2))
    m = rng.choice((2, 3, 4))
    q = pp * m
    xx = sp.Symbol("xx", negative=True)
    exact = sp.integrate((a * xx**2 + b * xx + c) / xx, (xx, -q, -pp))
    rational_part = sp.Rational(a, 2) * (pp**2 - q**2) + b * (q - pp)
    check(abs(sp.N(exact - (rational_part - c * sp.log(m)))) < 1e-9, "definite integral over negative limits")
    lead = "" if a == 1 else str(a)
    bterm = f" + {b}x" if b > 0 else f" - {-b}x"
    if abs(b) == 1:
        bterm = " + x" if b > 0 else " - x"
    top = Fm(f"{lead}x^{{2}}{bterm} + {c}", f"{lead}x^2{bterm} + {c}")
    integrand = Fm("\\frac{" + top.tex + "}{x}", f"({top.plain})/x")
    r = Fraction(int(rational_part.p), int(rational_part.q))
    log_text = ("" if c == 1 else f"{c} ") + f"ln {m}"
    log_tex = ("" if c == 1 else f"{c} ") + f"\\ln {m}"
    if r == 0:
        value = Fm("-" + log_tex, "-" + log_text)
    else:
        r_tex = (
            f"{r.numerator}"
            if r.denominator == 1
            else f"{'-' if r < 0 else ''}\\frac{{{abs(r.numerator)}}}{{{r.denominator}}}"
        )
        r_plain = f"{r.numerator}" if r.denominator == 1 else f"{r.numerator}/{r.denominator}"
        value = Fm(f"{r_tex} - {log_tex}", f"{r_plain} - {log_text}")
    claim = _int(integrand, -q, -pp) + " = " + value
    p = Prompt()
    p.t("Given that ").m(INT_RECIP_ABS).t(
        " on any interval where x < 0, and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))

    cc = "" if c == 1 else f"{c} "

    def poly_at(v: int) -> sp.Rational:
        return sp.Rational(a, 2) * v**2 + b * v

    f_upper, f_lower = poly_at(-pp), poly_at(-q)
    ref = [
        f"Split the integrand for x < 0: ({top.plain})/x = {lead}x {'+' if b > 0 else '-'} {abs(b)} + {c}/x.",
        f"An antiderivative on x < 0 is F(x) = {sp.Rational(a, 2)}x^2 {'+' if b > 0 else '-'} {abs(b)}x + {cc}ln|x|, using the given result for the 1/x term.",
        f"F(-{pp}) = {f_upper} + {cc}ln {pp} and F(-{q}) = {f_lower} + {cc}ln {q}.",
        f"Subtracting, the polynomial part is {f_upper} - ({f_lower}) = {rational_part} and the log part is {cc.replace(' ', '') + '(' if c != 1 else '('}ln {pp} - ln {q}) = -{cc}ln {m}.",
        f"So the integral is {value.plain}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "integrates the three terms and evaluates F at both negative limits"),
            ("logical_completeness", "uses the modulus form for the 1/x term and subtracts F(lower) from F(upper)"),
            ("definitions_assumptions", "notes the interval lies in x < 0"),
            ("communication", "separates the polynomial part from the logarithm part"),
        ],
        traps=[
            ("ln_of_a_negative", "does it write ln of a negative number?"),
            (
                "sign_of_the_polynomial_part",
                "are the squares and the products with negative limits handled with the right signs?",
            ),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_cancel(rng) -> dict:
    a = rng.choice((1, 2, 3))
    b = a + rng.choice((1, 2, 3, 4))
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, b, a)) + sp.integrate(1 / s, (s, a, b)), 0)
    claim = _int(INV_X, -b, -a) + " + " + _int(INV_X, a, b) + " = 0"
    p = Prompt()
    p.t("Given that ").m(INT_RECIP_ABS).t(
        " on any interval that does not contain 0, and the Fundamental Theorem of Calculus, prove that "
    ).m(claim).t(". ").t(_tail(rng))
    ref = [
        f"Both intervals [-{b}, -{a}] and [{a}, {b}] avoid 0, so F(x) = ln|x| is an antiderivative of 1/x on each.",
        f"The first integral is F(-{a}) - F(-{b}) = ln {a} - ln {b}.",
        f"The second integral is F({b}) - F({a}) = ln {b} - ln {a}.",
        "The sum is (ln a - ln b) + (ln b - ln a) = 0.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "evaluates both integrals and shows they are negatives of each other"),
            ("logical_completeness", "treats the two intervals separately, each avoiding 0"),
            ("definitions_assumptions", "states that neither interval contains 0, where 1/x is undefined"),
            ("communication", "shows the cancellation"),
        ],
        traps=[
            ("treats_as_one_integral", f"does it integrate once from -{b} to {b} through 0?"),
            ("ln_of_a_negative", "does it write ln of a negative number?"),
        ],
    )
    return finish(p, claim.plain, mark)


def _irn_student_error(rng) -> dict:
    pp = rng.choice((1, 2, 3))
    m = rng.choice((2, 3, 4, 5))
    q = pp * m
    s = sp.Symbol("s", positive=True)
    check_equivalent(sp.integrate(1 / s, (s, q, pp)), -sp.log(m))
    p = Prompt()
    p.t(f"A student works out ").m(_int(INV_X, -q, -pp)).t(
        f" by writing ln(-{pp}) - ln(-{q}) and says the answer is undefined. "
    ).t("Given that ").m(INT_RECIP_ABS).t(
        " on any interval where x < 0, explain what is wrong with the student's working, and prove that the integral equals "
    ).m(Fm(f"-\\ln {m}", f"-ln {m}")).t(". ").t(_tail(rng))
    ref = [
        f"The student used ln x as the antiderivative. But ln(-{pp}) and ln(-{q}) are logarithms of negative numbers, which are not defined, so that antiderivative cannot be used on the interval [-{q}, -{pp}].",
        "On x < 0 the correct antiderivative of 1/x is ln|x| (given), which is defined there.",
        f"By the Fundamental Theorem of Calculus the integral is ln|-{pp}| - ln|-{q}| = ln {pp} - ln {q}.",
        f"ln {pp} - ln {q} = ln({pp}/{q}) = ln(1/{m}) = -ln {m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains -ln {m} from the modulus antiderivative"),
            ("logical_completeness", "does both parts: identifies the error and proves the value"),
            (
                "definitions_assumptions",
                "says why ln of a negative number is undefined and that the interval lies in x < 0",
            ),
            ("communication", "contrasts the student's antiderivative with the correct one"),
        ],
        traps=[
            (
                "blames_the_integral",
                "does it conclude the integral does not exist, rather than that the antiderivative was wrong?",
            ),
            ("explains_only", "does it explain the error but not prove the value?"),
        ],
    )
    return finish(p, f"the integral is -ln {m}", mark)


IRN_FAMILIES = {
    "easy": (_irn_deriv_neg, _irn_tangent),
    "medium": (_irn_antiderivative, _irn_definite, _irn_const),
    "hard": (_irn_affine, _irn_symmetry, _irn_definite_scaled),
    "exam": (_irn_poly_over_x, _irn_cancel, _irn_student_error),
}

_PRESETS = {t: dict(variant=t) for t in ("easy", "medium", "hard", "exam")}


def _pick(families: dict, variant: str, rng) -> dict:
    if variant not in families:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(families[variant])(rng)


@generator("proof_derivative_exp", level=14, presets=_PRESETS)
def proof_derivative_exp(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-frac-d-dx`. Rubric-assessed (`answer: rubric`): every question
    returned carries a per-item marking dict (`Question.answer`), never a plain value."""
    return _pick(DEXP_FAMILIES, variant, rng)


@generator("proof_derivative_ln", level=14, presets=_PRESETS)
def proof_derivative_ln(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-frac-d-dx-acmmm161`. Rubric-assessed (`answer: rubric`)."""
    return _pick(DLN_FAMILIES, variant, rng)


@generator("proof_integral_exp", level=14, presets=_PRESETS)
def proof_integral_exp(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-int-e-xdx`. Rubric-assessed (`answer: rubric`)."""
    return _pick(IEXP_FAMILIES, variant, rng)


@generator("proof_integral_recip_pos", level=14, presets=_PRESETS)
def proof_integral_recip_pos(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-int-frac1xdx-ln`. Rubric-assessed (`answer: rubric`)."""
    return _pick(IRP_FAMILIES, variant, rng)


@generator("proof_integral_recip_neg", level=14, presets=_PRESETS)
def proof_integral_recip_neg(rng, variant="easy"):
    """`senior-establish-and-use-the-formula-int-frac1xdx-ln-acmsm118`. Rubric-assessed (`answer: rubric`)."""
    return _pick(IRN_FAMILIES, variant, rng)
