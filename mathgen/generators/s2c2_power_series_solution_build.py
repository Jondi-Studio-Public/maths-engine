"""Generates `s2c2_power_series_solution` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_power_series_solution`.

``uc2-power-series-solution`` (``ordinary-differential-equations``, answer ``fraction``): the power series solution
y = a_0 + a_1 t + a_2 t^2 + ... (t = x - x0) of P(x) y'' + Q(x) y' + R(x) y = 0 about an ordinary point, found from
the recurrence for the coefficients, with y(x0) and y'(x0) given.

* easy: P = 1, Q = p x, R = q about x = 0; the coefficient of x^4 or x^5.
* medium: a leading coefficient that is not 1, P = 1 + c x^2, with Q = p x and R = q about x = 0; the coefficient
  of x^4 or x^5.
* hard: P = 1 with Q and R linear in x, expanded about x0 = 1, 2 or -1, so the coefficients must be re-expanded in
  powers of (x - x0) and the recurrence has three terms; the coefficient of (x - x0)^3 or (x - x0)^4.
* exam (``exd_x_s2c2_power_series_solution``): y'' + (p0 + p1 x) y' + (q0 + q1 x) y = 0 about x = 0 in four parts:
  (a) a_2, (b) a_3 (follows through from (a)), (c) a_4 (follows through from (a) and (b)), (d) the polynomial
  a_0 + ... + a_4 x^4 at a stated x to 3 decimal places (follows through from (a), (b) and (c)).

The coefficients come from the recurrence obtained by matching powers of t. SymPy then substitutes the truncated
series into the differential equation and checks that every coefficient the truncation determines vanishes.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text, rounds_unambiguously
from ._s2_options import keep_answer_shape
from ._ucv1_texsay import say_marked

x, t = sympy.symbols("x t")
_SUP = {2: "²", 3: "³"}


def _coef(poly, j):
    return poly[j] if j < len(poly) else Fraction(0)


def solve_series(p, q, r, a0, a1, upto, *, sign=1, drop_second=False, drop_deriv=False, swap=False):
    """a_0 ... a_upto for P y'' + Q y' + R y = 0 (P, Q, R as lists of coefficients in t), with optional slips.

    The recurrence comes from the coefficient of t^n: the sum over j of
    P_j (n-j+2)(n-j+1) a_(n-j+2) + Q_j (n-j+1) a_(n-j+1) + R_j a_(n-j) = 0, solved for a_(n+2). The slips are
    the ones learners make: a sign error when moving terms across (`sign`), a lost factor in y'' (`drop_second`) or
    in y' (`drop_deriv`), and y(x0) and y'(x0) the wrong way round (`swap`)."""
    a = [Fraction(a1), Fraction(a0)] if swap else [Fraction(a0), Fraction(a1)]

    def at(k):
        return a[k] if 0 <= k < len(a) else Fraction(0)

    for n in range(0, upto - 1):
        total = Fraction(0)
        for j in range(1, len(p)):
            k = n - j + 2
            if k >= 2:
                total += _coef(p, j) * (k if drop_second else k * (k - 1)) * at(k)
        for j in range(0, len(q)):
            k = n - j + 1
            if k >= 1:
                total += _coef(q, j) * (1 if drop_deriv else k) * at(k)
        for j in range(0, len(r)):
            k = n - j
            if k >= 0:
                total += _coef(r, j) * at(k)
        lead = _coef(p, 0) * ((n + 2) if drop_second else (n + 2) * (n + 1))
        a.append(sign * -total / lead)
    return a


def _poly_text(coeffs, var="x"):
    terms = []
    for k, c in enumerate(coeffs):
        if c:
            name = "" if k == 0 else (var if k == 1 else f"{var}{_SUP[k]}")
            terms.append((int(c), name))
    return join_terms(terms)


def _coefficient_text(coeffs, deriv):
    """The text of ``(poly) deriv`` as a term of the equation, and whether it needs a leading sign."""
    nonzero = [(k, c) for k, c in enumerate(coeffs) if c]
    if len(nonzero) == 1:
        k, c = nonzero[0]
        mono = "" if k == 0 else ("x" if k == 1 else f"x{_SUP[k]}")
        body = f"{mono} {deriv}" if mono else deriv
        if abs(c) == 1:
            return c < 0, body
        return c < 0, f"{abs(int(c))}{body}"
    return False, f"({_poly_text(coeffs)}){deriv}"


def ode_text(p, q, r):
    pieces = [_coefficient_text(p, "y''"), _coefficient_text(q, "y'"), _coefficient_text(r, "y")]
    out = []
    for i, (neg, body) in enumerate(pieces):
        if i == 0:
            out.append(f"-{body}" if neg else body)
        else:
            out.append(f"{'-' if neg else '+'} {body}")
    return " ".join(out) + " = 0"


def _verify(p, q, r, a0, a1, a, shift):
    """Substitute the truncated series into the equation with SymPy: the coefficients up to t^(len(a) - 3) must vanish."""
    poly_t = lambda coeffs: sum(sympy.Rational(c.numerator, c.denominator) * t**k for k, c in enumerate(coeffs))  # noqa: E731
    series = sum(sympy.Rational(c.numerator, c.denominator) * t**k for k, c in enumerate(a))
    residual = sympy.expand(
        poly_t(p) * sympy.diff(series, t, 2) + poly_t(q) * sympy.diff(series, t) + poly_t(r) * series
    )
    poly = sympy.Poly(residual, t)
    for k in range(0, len(a) - 2):
        check(poly.coeff_monomial(t**k) == 0, f"the series does not satisfy the equation at t^{k}")
    check(a[0] == a0 and a[1] == a1, "initial values are not matched")


def _shifted(coeffs_x, x0):
    """Coefficients (in x) of a polynomial re-expanded in t = x - x0, by repeated synthetic division (the hand route)."""
    work = [Fraction(c) for c in coeffs_x]
    out = []
    for _ in range(len(work)):
        # Horner evaluation at x0 gives the constant term; the quotient carries on
        acc, quotient = Fraction(0), []
        for c in reversed(work):
            acc = acc * x0 + c
            quotient.append(acc)
        out.append(quotient[-1])
        work = list(reversed(quotient[:-1]))
        if not work:
            break
    return out


def _check_shift(coeffs_x, x0, shifted):
    expr = sum(sympy.Rational(c.numerator, c.denominator) * x**k for k, c in enumerate(coeffs_x))
    back = sympy.expand(expr.subs(x, t + sympy.Rational(x0)))
    mine = sum(sympy.Rational(c.numerator, c.denominator) * t**k for k, c in enumerate(shifted))
    check(sympy.expand(back - mine) == 0, "the re-expansion about x0 is wrong")


def _sup_power(n, base="x"):
    return f"{base}^{n}"


def _ic_text(x0, a0, a1):
    return f"y({x0}) = {a0}", f"y'({x0}) = {a1}"


def _draw(rng, tier):
    for _ in attempts(4000, "a power series problem with moderate coefficients"):
        a0, a1 = rng.randint(-4, 4), rng.randint(-4, 4)
        if a0 == 0 or a1 == 0:
            continue
        x0 = 0
        if tier == "easy":
            p_x, q_x, r_x = (
                [Fraction(1)],
                [Fraction(0), Fraction(rng.choice([-3, -2, -1, 1, 2, 3]))],
                [Fraction(rng.choice([-3, -2, -1, 1, 2, 3, 4]))],
            )
            n = rng.choice([4, 5])
        elif tier == "medium":
            p_x = [Fraction(1), Fraction(0), Fraction(rng.choice([1, 2, 3]))]
            q_x = [Fraction(0), Fraction(rng.choice([-2, -1, 1, 2, 3]))]
            r_x = [Fraction(rng.choice([-3, -2, -1, 1, 2, 3, 4]))]
            n = rng.choice([4, 5])
        else:
            x0 = rng.choice([1, 2, -1])
            p_x = [Fraction(1)]
            q_x = [Fraction(rng.choice([-2, -1, 0, 1, 2])), Fraction(rng.choice([-2, -1, 1, 2]))]
            r_x = [Fraction(rng.choice([-2, -1, 0, 1, 2])), Fraction(rng.choice([-2, -1, 1, 2]))]
            n = rng.choice([3, 4])
        if tier == "hard":
            p, q, r = _shifted(p_x, x0), _shifted(q_x, x0), _shifted(r_x, x0)
            for orig, sh in ((p_x, p), (q_x, q), (r_x, r)):
                _check_shift(orig, x0, sh)
        else:
            p, q, r = p_x, q_x, r_x
        a = solve_series(p, q, r, a0, a1, n)
        value = a[n]
        if value == 0 or value.denominator > 300 or abs(value) > 600:
            continue
        slips = _slips(p, q, r, a0, a1, n, a, p_x, q_x, r_x, x0)
        distinct = {v for _, v in slips if v != value}
        if len(distinct) < 3:
            continue
        return x0, (p_x, q_x, r_x), (p, q, r), (a0, a1), n, a, slips
    raise AssertionError("unreachable")


def _slips(p, q, r, a0, a1, n, a, p_x, q_x, r_x, x0):
    out = [
        ("ode2-series-second-derivative-factor-dropped", solve_series(p, q, r, a0, a1, n, drop_second=True)[n]),
        ("ode2-series-derivative-factor-dropped", solve_series(p, q, r, a0, a1, n, drop_deriv=True)[n]),
        ("ode2-series-sign-slip", solve_series(p, q, r, a0, a1, n, sign=-1)[n]),
        ("ode2-series-initial-values-swapped", solve_series(p, q, r, a0, a1, n, swap=True)[n]),
        ("ode2-series-wrong-coefficient-index", a[n - 1]),
    ]
    if x0 != 0:
        out.insert(
            0,
            (
                "ode2-series-expansion-point-ignored",
                solve_series(p_x, q_x, r_x, a0, a1, n)[n],
            ),
        )
    return out


_STEMS = (
    "The initial value problem «{ode}», «{ic1}», «{ic2}», has a power series solution about «x = {x0}», "
    "«y = a₀ + a₁({v}) + a₂({v})² + a₃({v})³ + …». Find the coefficient «a{sub}».",
)


def _subscript(n):
    return f"_{n}"


@generator(
    "s2c2_power_series_solution",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2c2_power_series_solution(rng, tier="easy"):
    x0, (p_x, q_x, r_x), (p, q, r), (a0, a1), n, a, slips = _draw(rng, tier)
    _verify(p, q, r, Fraction(a0), Fraction(a1), solve_series(p, q, r, a0, a1, n + 2), x0)
    value = a[n]
    variable = "x" if x0 == 0 else (f"x - {x0}" if x0 > 0 else f"x + {-x0}")
    ic1, ic2 = _ic_text(x0, a0, a1)
    template = (
        "The initial value problem «{ode}», «{ic1}», «{ic2}», has a power series solution "
        "«y = a_0 + a_1 {v1} + a_2 {v2} + a_3 {v3} + …» about «x = {x0}». Find the coefficient «a{sub}»."
    )
    if x0 == 0:
        v1, v2, v3 = "x", "x²", "x³"
    else:
        v1, v2, v3 = f"({variable})", f"({variable})²", f"({variable})³"
    text, latex = say_marked(
        template.format(ode=ode_text(p_x, q_x, r_x), ic1=ic1, ic2=ic2, v1=v1, v2=v2, v3=v3, x0=x0, sub=_subscript(n))
    )
    answer_text = rational_text(value)
    scored = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in slips if v != value]
    distractors = pick_distractors(answer_text, scored, 4)
    check(len(distractors) >= 3, "too few series distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=answer_text,
        distractors=distractors,
        working=[
            "Matching the coefficient of each power gives a recurrence; with "
            f"a_0 = {a0} and a_1 = {a1} it gives a{_subscript(n)} = {answer_text}."
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


def _exam_draw(rng):
    for _ in attempts(4000, "an exam power series problem"):
        a0, a1 = rng.randint(-4, 4), rng.randint(-4, 4)
        if a0 == 0 or a1 == 0:
            continue
        q0, q1 = rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])
        r0, r1 = rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2])
        a = solve_series([Fraction(1)], [Fraction(q0), Fraction(q1)], [Fraction(r0), Fraction(r1)], a0, a1, 4)
        if any(v == 0 or v.denominator > 60 for v in a[2:]):
            continue
        x_eval = rng.choice([Fraction(1, 2), Fraction(1), Fraction(-1, 2)])
        value = sum(float(a[k]) * float(x_eval) ** k for k in range(5))
        if not (0.02 <= abs(value) <= 400 and rounds_unambiguously(value, 3, 0.1)):
            continue
        return (q0, q1, r0, r1, a0, a1), a, x_eval, value
    raise AssertionError("unreachable")


@generator("exd_x_s2c2_power_series_solution", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_power_series_solution(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    (q0, q1, r0, r1, a0, a1), a, x_eval, value = _exam_draw(rng)
    p = [Fraction(1)]
    q = [Fraction(q0), Fraction(q1)]
    r = [Fraction(r0), Fraction(r1)]
    _verify(p, q, r, Fraction(a0), Fraction(a1), solve_series(p, q, r, a0, a1, 6), 0)
    slip_sets = {
        "sign": dict(sign=-1),
        "second": dict(drop_second=True),
        "deriv": dict(drop_deriv=True),
        "swap": dict(swap=True),
    }
    tags = {
        "sign": "ode2-series-sign-slip",
        "second": "ode2-series-second-derivative-factor-dropped",
        "deriv": "ode2-series-derivative-factor-dropped",
        "swap": "ode2-series-initial-values-swapped",
    }
    seqs = {name: solve_series(p, q, r, a0, a1, 4, **flags) for name, flags in slip_sets.items()}

    def dists(k):
        cands = [(tags[name], seq[k]) for name, seq in seqs.items() if seq[k] != a[k]]
        cands.append(("ode2-series-wrong-coefficient-index", a[k - 1]))
        return pick_distractors(
            rational_text(a[k]), [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in cands if v != a[k]], 4
        )

    a_dist, b_dist, c_dist = dists(2), dists(3), dists(4)
    xf = float(x_eval)
    d_text = f"{value:.3f}"
    d_cands = []
    for name, seq in seqs.items():
        v = sum(float(seq[k]) * xf**k for k in range(5))
        if abs(v) < 1e6 and abs(round(v, 3) - round(value, 3)) > 0.006:
            d_cands.append(Distractor(f"{v:.3f}", tags[name], round(v, 3)))
    # leaving out the x^4 term of the partial sum
    short = sum(float(a[k]) * xf**k for k in range(4))
    if abs(round(short, 3) - round(value, 3)) > 0.006:
        d_cands.append(Distractor(f"{short:.3f}", "ode2-series-term-left-out-of-sum", round(short, 3)))
    d_dist = pick_distractors(d_text, d_cands, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    ode = ode_text(p, q, r)
    stem, stem_latex = say_marked(
        f"Consider the initial value problem «{ode}», «y(0) = {a0}», «y'(0) = {a1}». "
        "Its solution has a power series «y = a_0 + a_1 x + a_2 x² + a_3 x³ + a_4 x⁴ + …»."
    )
    x_text = rational_text(x_eval)
    x_ft = x_text

    def prompt(template, **slots):
        text, tex = say_marked(template.format(**slots))
        return dict(prompt=text, prompt_latex=tex)

    follow_b = f"-(2*({q0})*part_a + (({q1}) + ({r0}))*({a1}) + ({r1})*({a0}))/6"
    follow_c = f"-(3*({q0})*part_b + (2*({q1}) + ({r0}))*part_a + ({r1})*({a1}))/12"
    follow_d = f"({a0}) + ({a1})*({x_ft}) + part_a*({x_ft})**2 + part_b*({x_ft})**3 + part_c*({x_ft})**4"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Find «a_2»."),
                answer_kind="fraction",
                answer=exact_value(a[2]),
                answer_text=rational_text(a[2]),
                marks=2,
                working=[
                    f"Matching the constant terms: 2a_2 + ({q0})a_1 + ({r0})a_0 = 0, so a_2 = {rational_text(a[2])}."
                ],
                distractors=a_dist,
            ),
            dict(
                **prompt("Find «a_3»."),
                answer_kind="fraction",
                answer=exact_value(a[3]),
                answer_text=rational_text(a[3]),
                marks=2,
                follow_through=follow_b,
                working=[f"Matching the coefficients of x gives a_3 = {rational_text(a[3])}."],
                distractors=b_dist,
            ),
            dict(
                **prompt("Find «a_4»."),
                answer_kind="fraction",
                answer=exact_value(a[4]),
                answer_text=rational_text(a[4]),
                marks=2,
                follow_through=follow_c,
                working=[f"Matching the coefficients of x² gives a_4 = {rational_text(a[4])}."],
                distractors=c_dist,
            ),
            dict(
                **prompt("Use the terms up to «x⁴» to estimate «y({x})», correct to 3 decimal places.", x=x_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=d_text,
                marks=1,
                follow_through=follow_d,
                working=[f"a_0 + a_1({x_text}) + … + a_4({x_text})^4 = {d_text}."],
                distractors=d_dist,
                sensible_range=(-1000, 1000),
            ),
        ],
    )
