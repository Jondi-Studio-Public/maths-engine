"""Generates `s2c1_exact_equation` questions (easy, medium, hard) and the exam chain `exd_x_s2c1_exact_equation`.

``uc2-exact-equation`` (``ordinary-differential-equations``, answer ``expression``): first-order equations
``M dx + N dy = 0``, exact or made exact by an integrating factor.

* easy: an exact equation with a polynomial potential; find the potential F(x, y) with F(0, 0) = 0.
* medium: either find the value of k that makes the equation exact (the answer is a fraction), or solve an initial
  value problem for an exact equation containing an exponential or trigonometric cross term and state the constant C
  in F(x, y) = C.
* hard: a non-exact equation with an integrating factor that depends on x alone or on y alone (a power, or an
  exponential); find that factor with the constant of integration taken as 0.
* exam (``exd_x_s2c1_exact_equation``): (a) the quotient (M_y - N_x)/N or (N_x - M_y)/M, (b) the integrating factor,
  (c) the potential F of the exact equation it produces, (d) the constant C for a stated initial condition (follows
  through from (c)).

Every equation is built from a potential, and the claims are re-checked by differentiating: dF/dx = M and dF/dy = N
(or the same for mu M and mu N), M_y = N_x for the exact equations and M_y != N_x for the others, and the quotient
that selects the integrating factor must depend on one variable only (the other candidate must fail to).
"""

from __future__ import annotations

import functools

import sympy

from ..core import attempts, check, generator, rational_text
from ..answers import exact_value
from ._s2c1_kit import (
    expr_options,
    fraction_options,
    second_route,
    machine,
    number_options,
    plain,
    prompt_part,
    say_marked,
)

x, y, k = sympy.symbols("x y k")
_R = sympy.Rational
_POINTS2 = [(_R(7, 10), _R(11, 10)), (_R(13, 10), _R(9, 10)), (_R(19, 10), _R(17, 10))]


@functools.lru_cache(maxsize=4096)
def _int_term(term, var):
    """The integral of one term from a small table: a monomial in `var` (power rule), sin or cos of `var`, e^(r var),
    or a monomial times e^(r var) (repeated parts). Anything else is a defect in the caller, not a case to solve."""
    coeff, rest = term.as_independent(var)
    if rest == 1:
        return term * var
    base, exponent = rest.as_base_exp()
    if base == var and exponent.is_number and exponent != -1:
        return coeff * var ** (exponent + 1) / (exponent + 1)
    if base == var and exponent == -1:
        return coeff * sympy.log(var)
    if rest.func is sympy.sin and rest.args[0] == var:
        return -coeff * sympy.cos(var)
    if rest.func is sympy.cos and rest.args[0] == var:
        return coeff * sympy.sin(var)
    if rest.func is sympy.exp:
        r = sympy.diff(rest.args[0], var)
        if r.is_number and r != 0 and sympy.diff(r, var) == 0:
            return coeff * rest / r
    # a monomial times e^(r var): integrate by parts n times
    factors = sympy.Mul.make_args(rest)
    exps = [f for f in factors if f.func is sympy.exp]
    mono = sympy.Mul(*[f for f in factors if f.func is not sympy.exp])
    if len(exps) == 1 and mono.is_polynomial(var) and mono.free_symbols <= {var}:
        r = sympy.diff(exps[0].args[0], var)
        if r.is_number and r != 0:
            poly = sympy.Poly(mono, var)
            total, d = 0, mono
            for n in range(poly.degree() + 1):
                total += (-1) ** n * d / r ** (n + 1)
                d = sympy.diff(d, var)
            return coeff * total * exps[0]
    raise ValueError(f"no table entry for the integral of {term} with respect to {var}")


def _int(expr, var):
    """Termwise antiderivative with constant of integration 0."""
    return sympy.Add(*(_int_term(t, var) for t in sympy.Add.make_args(sympy.expand(sympy.sympify(expr)))))


def _simp(expr):
    """A cheap simplification for rational functions of x, y and exponentials (SymPy's full simplifier is slow)."""
    expanded = sympy.expand(sympy.sympify(expr))
    return expanded if expanded.is_polynomial(x, y) else sympy.cancel(expanded)


@functools.lru_cache(maxsize=4096)
def _compile2(expr):
    return sympy.lambdify((x, y), expr, "math")


def _eval2(expr, px, py) -> float:
    e = sympy.sympify(expr)
    try:
        return float(_compile2(e)(float(px), float(py)))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return float(e.xreplace({x: sympy.Float(float(px), 17), y: sympy.Float(float(py), 17)}))


def _agree2(left, right) -> bool:
    for px, py in _POINTS2:
        a, b = _eval2(left, px, py), _eval2(right, px, py)
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            return False
    return True


def _nonzero(rng, lo: int, hi: int) -> int:
    for _ in attempts(100, "a non-zero integer"):
        v = rng.randint(lo, hi)
        if v:
            return v
    raise AssertionError("unreachable")


def _poly_potential(rng, cross_only: bool = False):
    """A polynomial F with F(0, 0) = 0, at least one cross term x^a y^b (a, b >= 1) and a term in each variable."""
    cross_pool = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]
    pure_pool = [(2, 0), (3, 0), (0, 2), (0, 3), (1, 0), (0, 1), (4, 0), (0, 4)]
    for _ in attempts(500, "a polynomial potential"):
        cross = rng.sample(cross_pool, rng.choice([1, 1, 2]))
        pure = rng.sample(pure_pool, rng.choice([1, 2]))
        terms = {mon: _nonzero(rng, -4, 4) for mon in cross + ([] if cross_only else pure)}
        F = sum(c * x**a * y**b for (a, b), c in terms.items())
        M, N = sympy.diff(F, x), sympy.diff(F, y)
        if M == 0 or N == 0 or not M.has(y) or not N.has(x):
            continue
        if not (M.has(x) or N.has(y)):
            continue
        return F
    raise AssertionError("unreachable")


def _term_with(expr, differential: str, first: bool) -> str:
    """`expr` times a differential, in the way it is written inside a sum: ``(a + b) dx``, ``- 2x dy``, ``+ dy``."""
    expr = sympy.sympify(expr)
    if isinstance(expr, sympy.Add):
        body = f"({plain(expr)}) {differential}"
        return body if first else f"+ {body}"
    sign = "-" if expr.could_extract_minus_sign() else "+"
    mag = -expr if sign == "-" else expr
    body = differential if mag == 1 else f"{plain(mag)} {differential}"
    if first:
        return body if sign == "+" else f"-{body}"
    return f"{sign} {body}"


def _differential(M, N) -> str:
    return f"{_term_with(M, 'dx', True)} {_term_with(N, 'dy', False)} = 0"


def _check_exact(F, M, N):
    check(_agree2(sympy.diff(F, x), M) and _agree2(sympy.diff(F, y), N), "dF/dx, dF/dy are not M, N")
    check(_agree2(sympy.diff(M, y), sympy.diff(N, x)), "M_y != N_x for an exact equation")


# ---------------------------------------------------------------------------------------------------------------
# easy: the potential of an exact equation


def _potential_wrongs(F, M, N):
    A = _int(M, x)
    Bn = _int(N, y)
    g = F - A
    h = F - Bn
    wrongs = [
        ("exact-dropped-function-of-y", A),
        ("exact-both-integrals-added", A + Bn),
        ("exact-integrated-wrong-variable", _int(M, y)),
        ("exact-sign-on-g", A - g),
        ("exact-dropped-function-of-x", Bn),
        ("exact-sign-on-g", Bn - h),
    ]
    return wrongs


def _easy(rng):
    F = _poly_potential(rng)
    M, N = sympy.diff(F, x), sympy.diff(F, y)
    _check_exact(F, M, N)
    template = rng.choice(
        [
            "The differential equation «{eq}» is exact. Find its potential function «F(x, y)», with «F(0, 0) = 0», "
            "so that the solutions are the curves «F(x, y) = C».",
            "Solve the exact equation «{eq}»: find the function «F(x, y)» with «F(0, 0) = 0» such that the solutions "
            "are given implicitly by «F(x, y) = C».",
        ]
    )
    stem, stem_latex = say_marked(template.format(eq=_differential(M, N)))
    dist = _options2(F, _potential_wrongs(F, M, N))
    check(len(dist) >= 2, "too few exact-equation distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(F),
        answer_text=plain(F),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"Since M_y = N_x the equation is exact. Integrating M with respect to x gives {plain(sympy.expand(_int(M, x)))} "
            f"plus a function of y; matching F_y = N fixes it, so F = {plain(F)}."
        ],
    )


def _options2(answer, wrongs, n: int = 4):
    """Distractors for an answer in two variables (the kit's `expr_options` probes one variable)."""
    from ..answers import Distractor
    from ..core import pick_distractors

    kept = []
    for tag, w in wrongs:
        w = sympy.expand(sympy.sympify(w))
        if _agree2(w, answer) or any(_agree2(w, other) for _, other in kept):
            continue
        kept.append((tag, w))
    return pick_distractors(plain(answer), [Distractor(plain(w), tag, machine(w)) for tag, w in kept], n)


# ---------------------------------------------------------------------------------------------------------------
# medium: the value of k, or an initial value problem


def _noise(rng):
    """A cross term c*y*cos(x) or c*x*e^y (exact by itself) and a pure term, as (potential, M, N)."""
    c = _nonzero(rng, -3, 3)
    cross = rng.choice([y * sympy.cos(x), x * sympy.exp(y), y * sympy.sin(x), sympy.exp(x) * y])
    pure = rng.choice([x**2, y**2, x**3, y**3]) * _nonzero(rng, -3, 3)
    F = c * cross + pure
    return F


def _medium_k(rng):
    for _ in attempts(200, "a value of k that is not trivial"):
        a, b = rng.choice([(1, 2), (2, 2), (1, 3), (2, 3), (3, 2)])
        n = _nonzero(rng, -5, 5)
        in_m = rng.random() < 0.5
        base = _noise(rng)
        if in_m:  # M has k x^a y^b, N has n x^(a+1) y^(b-1)
            M_term, N_term = k * x**a * y**b, n * x ** (a + 1) * y ** (b - 1)
            answer = _R(n * (a + 1), b)
        else:  # N has k x^(a+1) y^(b-1), M has n x^a y^b
            M_term, N_term = n * x**a * y**b, k * x ** (a + 1) * y ** (b - 1)
            answer = _R(n * b, a + 1)
        M = sympy.diff(base, x) + M_term
        N = sympy.diff(base, y) + N_term
        if answer == 0 or abs(answer) > 30:
            continue
        # independent check of the claim: with k = answer the equation is exact, with any other value it is not
        exact = (M.subs(k, answer), N.subs(k, answer))
        check(_agree2(sympy.diff(exact[0], y), sympy.diff(exact[1], x)), "equation is not exact at the answer")
        other = (M.subs(k, answer + 1), N.subs(k, answer + 1))
        check(not _agree2(sympy.diff(other[0], y), sympy.diff(other[1], x)), "equation is exact at another k")
        terms = [(M, N)]
        # the learner's two natural slips
        wrongs = [
            ("exact-test-wrong-partials", _solve_k(sympy.diff(M, x), sympy.diff(N, y))),
            ("exact-derivative-power-dropped", _R(n, 1) if in_m else _R(n * b, 1)),
            ("exact-sign-of-k", -answer),
            ("exact-k-inverted", 1 / answer),
        ]
        dist = expr_options(answer, wrongs, 4)
        if len(dist) < 2:
            continue
        template = rng.choice(
            [
                "Find the value of «k» for which the differential equation «{eq}» is exact.",
                "The differential equation «{eq}» is exact for exactly one value of «k». Find it.",
            ]
        )
        stem, stem_latex = say_marked(template.format(eq=_differential(M, N)))
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(answer),
            answer_text=rational_text(answer),
            answer_kind="expression",
            distractors=dist,
            working=[f"Exactness needs M_y = N_x. Comparing the cross terms gives k = {rational_text(answer)}."],
        )
    raise AssertionError("unreachable")


def _solve_k(left, right):
    expr = sympy.expand(sympy.sympify(left) - sympy.sympify(right))  # linear in k
    slope = sympy.diff(expr, k)
    if slope == 0 or slope.has(x, y):
        return None
    value = sympy.cancel(-expr.subs(k, 0) / slope)
    return None if value.has(x, y) else value


def _medium_ivp(rng):
    for _ in attempts(300, "an exact initial value problem"):
        F = _poly_potential(rng) + sympy.Integer(_nonzero(rng, -3, 3)) * rng.choice(
            [sympy.exp(x) * y, y * sympy.cos(x), x * sympy.exp(y), sympy.sin(x) * y + x * y**2 * 0]
        )
        M, N = sympy.diff(F, x), sympy.diff(F, y)
        _check_exact(F, M, N)
        x0, y0 = rng.choice([1, 2]), rng.choice([1, 2, 3])
        if F.has(sympy.sin) or F.has(sympy.cos):
            x0 = rng.choice([1, 2])
        C = _simp(F.subs({x: x0, y: y0}))
        if C == 0 or C.has(sympy.sin, sympy.cos):
            continue
        # distractors: slips in finding F, then the same evaluation
        wrongs = []
        for tag, w in _potential_wrongs(F, M, N):
            val = _simp(sympy.sympify(w).subs({x: x0, y: y0}))
            wrongs.append((tag, val))
        wrongs.append(("exact-ic-sign-on-constant", -C))
        wrongs.append(("exact-ic-swapped-point", _simp(F.subs({x: y0, y: x0}))))
        dist = _constant_options(C, wrongs)
        if len(dist) < 2:
            continue
        stem, stem_latex = say_marked(
            f"Solve the initial value problem «{_differential(M, N)}» with «y({x0}) = {y0}». Write the solution as "
            f"«F(x, y) = C» where «F(0, 0) = 0», and give «C»."
        )
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(C),
            answer_text=plain(C),
            answer_kind="expression",
            distractors=dist,
            working=[f"F = {plain(F)}, so C = F({x0}, {y0}) = {plain(C)}."],
        )
    raise AssertionError("unreachable")


def _constant_options(answer, wrongs, n: int = 4):
    from ..answers import Distractor
    from ..core import pick_distractors

    out = []
    seen = [sympy.sympify(answer)]
    for tag, w in wrongs:
        w = _simp(sympy.sympify(w))
        if w.free_symbols or any(abs(float((w - s).evalf())) < 1e-9 for s in seen):
            continue
        seen.append(w)
        out.append(Distractor(plain(w), tag, machine(w)))
    return pick_distractors(plain(answer), out, n)


# ---------------------------------------------------------------------------------------------------------------
# hard and exam: integrating factors


def _random_poly(rng, need: str):
    """A polynomial in x and y with no constant term whose derivative in `need` is not zero."""
    for _ in attempts(200, "a polynomial H"):
        terms = rng.sample([x, y, x * y, x**2, y**2], rng.choice([2, 2, 3]))
        H = sum(_nonzero(rng, -3, 3) * t for t in terms)
        if sympy.diff(H, need) != 0:
            return H
    raise AssertionError("unreachable")


def _factor_case(rng, accept=None):
    """(M, N, mu, F, ratio, variable): the non-exact equation M dx + N dy = 0, its integrating factor mu and the
    potential F of mu M dx + mu N dy with F(0, 0) = 0."""
    for _ in attempts(100, "an integrating-factor equation with a single answer"):
        case = _factor_draw(rng, accept)
        if case is not None:
            return case
    raise AssertionError("unreachable")


def _factor_draw(rng, accept=None):
    """A non-exact M dx + N dy = 0 built backwards from its integrating factor mu and potential F.

    With mu = x^n and F = x^(n+1) H(x, y): M = (n+1) H + x H_x and N = x H_y, so (M_y - N_x)/N = n/x. The other
    three families (e^(rx) G(x, y), y^n H, e^(ry) G) follow the same way, giving the ratio n/x, r, n/y or r."""
    kind = rng.choice(["xpow", "xexp", "ypow", "yexp"])
    n = rng.choice([1, 2, 3])
    r = rng.choice([-2, -1, 1, 2])
    if kind == "xpow":
        H = _random_poly(rng, y)
        M, N = (n + 1) * H + x * sympy.diff(H, x), x * sympy.diff(H, y)
        mu, F, var, ratio = x**n, x ** (n + 1) * H, x, _R(n) / x
        other_num = -n * sympy.diff(H, y)
    elif kind == "xexp":
        G = _random_poly(rng, y)
        M, N = r * G + sympy.diff(G, x), sympy.diff(G, y)
        mu, F, var, ratio = sympy.exp(r * x), sympy.exp(r * x) * G, x, sympy.Integer(r)
        other_num = -r * sympy.diff(G, y)
    elif kind == "ypow":
        H = _random_poly(rng, x)
        M, N = y * sympy.diff(H, x), (n + 1) * H + y * sympy.diff(H, y)
        mu, F, var, ratio = y**n, y ** (n + 1) * H, y, _R(n) / y
        other_num = -n * sympy.diff(H, x)
    else:
        G = _random_poly(rng, x)
        M, N = sympy.diff(G, x), r * G + sympy.diff(G, y)
        mu, F, var, ratio = sympy.exp(r * y), sympy.exp(r * y) * G, y, sympy.Integer(r)
        other_num = -r * sympy.diff(G, x)
    if accept is not None and not accept(F, mu, var):
        return None  # cheap test first: the caller's initial condition would not give a whole-number constant
    M, N = sympy.expand(M), sympy.expand(N)
    if M == 0 or N == 0 or other_num == 0:
        return None
    if second_route():
        # independent checks: M dx + N dy is not exact; mu makes it exact; F is its potential
        check(not _agree2(sympy.diff(M, y), sympy.diff(N, x)), "the equation is already exact")
        check(_agree2(sympy.diff(mu * M, y), sympy.diff(mu * N, x)), "mu does not make the equation exact")
        check(_agree2(sympy.diff(F, x), mu * M) and _agree2(sympy.diff(F, y), mu * N), "F is not the potential")
        quotient = (sympy.diff(M, y) - sympy.diff(N, x)) / N if var == x else (sympy.diff(N, x) - sympy.diff(M, y)) / M
        check(_agree2(quotient, ratio), "the quotient is not the closed form")
    # A second integrating factor of the other kind would make the answer ambiguous: the other quotient
    # other_num / (M or N) must still depend on `var`. Compared numerically at two values of `var`.
    den = M if var == x else N
    if var == x:
        pairs = ((_R(7, 10), _R(11, 10)), (_R(13, 10), _R(11, 10)))
    else:
        pairs = ((_R(11, 10), _R(7, 10)), (_R(11, 10), _R(13, 10)))
    try:
        v1, v2 = (_eval2(other_num, *q) / _eval2(den, *q) for q in pairs)
    except ZeroDivisionError:
        return None
    if abs(v1 - v2) <= 1e-9 * max(1.0, abs(v1), abs(v2)):
        return None
    return M, N, mu, sympy.expand(F), ratio, var


def _factor_wrongs(mu, ratio, var):
    integral = _int(ratio, var)
    return [
        ("exact-factor-sign-reversed", _simp(1 / mu)),
        ("exact-factor-not-exponentiated", integral),
        ("exact-factor-ratio-only", ratio),
        ("exact-factor-wrong-variable", mu.subs(var, x if var == y else y)),
    ]


def _hard(rng):
    for _ in attempts(100, "an integrating factor question"):
        M, N, mu, F, ratio, var = _factor_case(rng)
        dist = _options2(mu, _factor_wrongs(mu, ratio, var))
        if len(dist) < 2:
            continue
        stem, stem_latex = say_marked(
            rng.choice(
                [
                    "The equation «{eq}» is not exact, but it has an integrating factor that depends on «x» alone or "
                    "on «y» alone. Find it, taking the constant of integration to be 0.",
                    "Find an integrating factor for «{eq}» that is a function of one variable only, taking the "
                    "constant of integration to be 0.",
                ]
            ).format(eq=_differential(M, N))
        )
        return dict(
            text=stem,
            latex=stem_latex,
            answer=machine(mu),
            answer_text=plain(mu),
            answer_kind="expression",
            distractors=dist,
            working=[
                f"The quotient {'(M_y - N_x)/N' if var == x else '(N_x - M_y)/M'} is {plain(ratio)}, a function of "
                f"{var} only, so the integrating factor is {plain(mu)}."
            ],
        )
    raise AssertionError("unreachable")


@generator(
    "s2c1_exact_equation",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_exact_equation(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium_k(rng) if rng.random() < 0.5 else _medium_ivp(rng)
    return _hard(rng)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_exact_equation", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_exact_equation(rng, tier="exam"):
    """Exam tier: an integrating factor, the potential it produces and an initial condition, in four parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam exact-equation question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    chosen = {}

    def accept(F, mu, var):
        # the initial condition: on an axis when the factor is an exponential, so that C stays a whole number
        if mu.has(sympy.exp):
            x0, y0 = (0, rng.choice([1, 2, 3])) if var == x else (rng.choice([1, 2, 3]), 0)
        else:
            x0, y0 = rng.choice([1, 2]), rng.choice([1, 2, 3])
        C = sympy.nsimplify(sympy.expand(F).subs({x: x0, y: y0}))
        if not C.is_Integer or C == 0 or abs(C) > 400:
            return False
        chosen.update(x0=x0, y0=y0, C=C)
        return True

    M, N, mu, F, ratio, var = _factor_case(rng, accept)
    x0, y0, C = chosen["x0"], chosen["y0"], chosen["C"]
    quotient = "(M_y - N_x)/N" if var == x else "(N_x - M_y)/M"
    num_name = "M_y - N_x" if var == x else "N_x - M_y"

    # (a) the quotient
    numerator = sympy.diff(M, y) - sympy.diff(N, x) if var == x else sympy.diff(N, x) - sympy.diff(M, y)
    wrong_a = [
        ("exact-factor-sign-reversed", -ratio),
        ("exact-factor-wrong-denominator", numerator / (M if var == x else N)),
        ("exact-quotient-numerator-only", numerator),
        ("exact-quotient-wrong-partials", (sympy.diff(M, x) - sympy.diff(N, y)) / (N if var == x else M)),
    ]
    a_dist = _options2(ratio, [(t, _simp(w)) for t, w in wrong_a])
    # (b) the factor
    b_dist = _options2(mu, _factor_wrongs(mu, ratio, var))
    # (c) the potential of the exact equation
    xm, xn = sympy.expand(mu * M), sympy.expand(mu * N)
    Ac = _int(xm, x)
    Bc = _int(xn, y)
    wrong_c = [
        ("exact-dropped-function-of-y", Ac),
        ("exact-both-integrals-added", Ac + Bc),
        (
            "exact-factor-not-applied",
            _int(M, x) + (_int(N - sympy.diff(_int(M, x), y), y)),
        ),
        ("exact-dropped-function-of-x", Bc),
    ]
    c_dist = _options2(F, [(t, sympy.expand(w)) for t, w in wrong_c])
    # (d) the constant
    wrong_d = []
    for d in c_dist:
        wrong_d.append((d.misconception, float(sympy.sympify(d.value).subs({x: x0, y: y0}).evalf())))
    wrong_d.append(("exact-ic-swapped-point", float(F.subs({x: y0, y: x0}))))
    d_dist = number_options(float(C), 0, wrong_d, 4, gap=1)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem, stem_latex = say_marked(
        f"The differential equation «{_differential(M, N)}» is not exact, but it has an integrating factor that is a "
        f"function of «{var}» alone."
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part(f"Find the quotient «{quotient}»."),
                answer_kind="expression",
                answer=machine(ratio),
                answer_text=plain(ratio),
                marks=2,
                working=[f"{num_name} divided by {'N' if var == x else 'M'} is {plain(ratio)}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    f"Hence find the integrating factor «μ({var})», taking the constant of integration to be 0."
                ),
                answer_kind="expression",
                answer=machine(mu),
                answer_text=plain(mu),
                marks=2,
                working=[f"μ = exp of the integral of {plain(ratio)} with respect to {var}, which is {plain(mu)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part(
                    "Multiply the equation by «μ» and find the potential function «F(x, y)» of the exact equation "
                    "that results, with «F(0, 0) = 0»."
                ),
                answer_kind="expression",
                answer=machine(F),
                answer_text=plain(F),
                marks=3,
                working=[f"F = {plain(F)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part(
                    "The solution curve passes through «({x0}, {y0})». Find the constant «C» in «F(x, y) = C».",
                    x0=x0,
                    y0=y0,
                ),
                answer_kind="number",
                answer=int(C),
                answer_text=str(int(C)),
                marks=1,
                follow_through=f"Subs(part_c, (x, y, e), ({x0}, {y0}, E))",
                working=[f"C = F({x0}, {y0}) = {int(C)}."],
                distractors=d_dist,
            ),
        ],
    )
