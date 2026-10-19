"""Generates `s2c1_second_order_ivp` questions (easy, medium, hard) and the exam chain `exd_x_s2c1_second_order_ivp`.

``uc2-second-order-ivp`` (``ordinary-differential-equations``, answer ``expression``): the complete solution of
``y'' + b y' + c y = g(x)`` with ``y(0)`` and ``y'(0)`` given. The stem shows the equation only; the learner finds the
characteristic roots, the particular solution and the two constants.

* easy: two distinct real roots with a constant, linear or ``K e^(px)`` forcing term.
* medium: a repeated root (exponential, linear or constant forcing), complex roots (exponential or linear forcing), or
  distinct roots with ``K cos(wx)`` or ``K sin(wx)`` forcing.
* hard: forcing that matches the homogeneous solution (a simple root, a double root, or ``cos(wx)`` when b = 0 and
  c = w²) so the particular solution needs a factor x or x², or damped complex roots with sinusoidal forcing.
* exam (``exd_x_s2c1_second_order_ivp``): (a) the particular solution, (b) the constant c1, (c) the constant c2
  (follows through from (b)), (d) y at a stated x to 3 decimal places (follows through from (a), (b) and (c)).

The particular solution is derived by formula (never by guessing and fitting), then every claimed solution is checked
by substituting it back into the equation numerically at three points and against both initial values.
"""

from __future__ import annotations

import sympy

from ..answers import exact_value
from ..core import VerificationError, attempts, check, generator, join_terms, rational_text, rounds_unambiguously
from ._s2c1_kit import (
    X,
    expr_options,
    fraction_options,
    machine,
    number_options,
    plain,
    prompt_part,
    say_marked,
    small_fraction,
    solve_constants,
    solves_ode,
)

x = X
_R = sympy.Rational


def _basis(kind, p, q):
    if kind == "distinct":
        return sympy.exp(p * x), sympy.exp(q * x)
    if kind == "repeated":
        return sympy.exp(p * x), x * sympy.exp(p * x)
    return sympy.exp(p * x) * sympy.cos(q * x), sympy.exp(p * x) * sympy.sin(q * x)


def _coefficients(kind, p, q):
    """(b, c) of y'' + b y' + c y = 0 for the given roots."""
    if kind == "distinct":
        return -(p + q), p * q
    if kind == "repeated":
        return -2 * p, p * p
    return -2 * p, p * p + q * q


def _draw_roots(rng, kind):
    if kind == "distinct":
        p, q = sorted(rng.sample([-4, -3, -2, -1, 1, 2, 3, 4], 2), reverse=True)
        return sympy.Integer(p), sympy.Integer(q)
    if kind == "repeated":
        return sympy.Integer(rng.choice([-3, -2, -1, 1, 2, 3])), None
    return sympy.Integer(rng.choice([-2, -1, 0, 1, 2])), sympy.Integer(rng.choice([1, 2, 3]))


def _is_root(kind, p, q, value) -> bool:
    return value == p or (kind == "distinct" and value == q)


def _forcing(rng, kind, p, q, b, c, mode):
    """(g, y_p, wrong_yps) for the requested forcing mode; wrong_yps are (misconception id, wrong y_p)."""
    K = rng.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12])
    if mode == "const":
        if c == 0:
            return None
        return sympy.Integer(K), _R(K, c), [("ode-particular-coefficient-slip", sympy.Integer(K))]
    if mode == "linear":
        if c == 0:
            return None
        L = rng.choice([-5, -3, -2, -1, 0, 1, 2, 3, 5])
        yp = _R(K, c) * x + (_R(L, c) - _R(K * b, c * c))
        wrong = [
            ("ode-particular-coefficient-slip", _R(K, c) * x + _R(L, c)),
            ("ode-particular-derivative-dropped", _R(K, c) * x),
        ]
        return K * x + L, yp, wrong
    if mode == "exp":
        e = rng.choice([-2, -1, 1, 2, 3])
        d = e * e + b * e + c
        if d == 0:
            return None
        yp = _R(K, d) * sympy.exp(e * x)
        wrong = [
            ("ode-particular-coefficient-slip", sympy.Integer(K) * sympy.exp(e * x)),
            (
                "ode-particular-coefficient-slip",
                _R(K, e * e + c - b * e if e * e + c - b * e else 1) * sympy.exp(e * x),
            ),
            ("ode-particular-coefficient-slip", _R(K, e * e + c) * sympy.exp(e * x) if e * e + c else None),
        ]
        return K * sympy.exp(e * x), yp, wrong
    if mode in ("cos", "sin"):
        w = rng.choice([1, 2, 3])
        delta = (c - w * w) ** 2 + (b * w) ** 2
        if delta == 0:
            return None
        if mode == "cos":
            A, B = _R(K * (c - w * w), delta), _R(K * b * w, delta)
            g = K * sympy.cos(w * x)
        else:
            A, B = _R(-K * b * w, delta), _R(K * (c - w * w), delta)
            g = K * sympy.sin(w * x)
        yp = A * sympy.cos(w * x) + B * sympy.sin(w * x)
        wrong = [
            (
                "ode-particular-coefficient-slip",
                _R(K, c - w * w if c != w * w else 1) * (sympy.cos(w * x) if mode == "cos" else sympy.sin(w * x)),
            ),
            ("ode-particular-coefficient-slip", A * sympy.cos(w * x) - B * sympy.sin(w * x)),
        ]
        return g, yp, wrong
    if mode == "exp-res":  # g = K e^(px) with p a simple root
        e = p if kind == "distinct" else None
        if e is None:
            return None
        slope = 2 * e + b
        if slope == 0:
            return None
        yp = _R(K, slope) * x * sympy.exp(e * x)
        wrong = [
            ("ode-resonance-coefficient-slip", sympy.Integer(K) * x * sympy.exp(e * x)),
            ("ode-resonance-coefficient-slip", _R(K, 2 * e) * x * sympy.exp(e * x) if e else None),
            ("ode-resonance-factor-x-dropped", _R(K, slope) * sympy.exp(e * x)),
        ]
        return K * sympy.exp(e * x), yp, wrong
    if mode == "exp-dres":  # g = K e^(px), p the double root
        if kind != "repeated":
            return None
        yp = _R(K, 2) * x**2 * sympy.exp(p * x)
        wrong = [
            ("ode-resonance-coefficient-slip", sympy.Integer(K) * x**2 * sympy.exp(p * x)),
            ("ode-resonance-factor-x-dropped", _R(K, 2) * x * sympy.exp(p * x)),
            ("ode-resonance-coefficient-slip", sympy.Integer(K) * x * sympy.exp(p * x)),
        ]
        return K * sympy.exp(p * x), yp, wrong
    if mode == "trig-res":  # b = 0, c = w²
        if kind != "complex" or p != 0:
            return None
        w = q
        if rng.random() < 0.5:
            g, yp = K * sympy.cos(w * x), _R(K, 2 * w) * x * sympy.sin(w * x)
        else:
            g, yp = K * sympy.sin(w * x), -_R(K, 2 * w) * x * sympy.cos(w * x)
        wrong = [
            ("ode-resonance-coefficient-slip", yp * 2 * w / (w if w != 1 else 1) if w != 1 else yp * 2),
            ("ode-resonance-factor-x-dropped", yp / x),
            ("ode-resonance-coefficient-slip", -yp),
        ]
        return g, yp, wrong
    raise AssertionError(f"unknown forcing mode {mode}")


def _assemble(kind, p, q, yp, y0, v0, *, ic="full", roots_sign=False):
    """y = c1 f1 + c2 f2 + yp with constants from the initial conditions; `ic` names a slip in fitting them."""
    if roots_sign:
        p, q = -p, (-q if kind == "distinct" else q)
    f1, f2 = _basis(kind, p, q)
    yp_use = yp
    if ic == "hom_only":
        c1, c2 = solve_constants(f1, f2, 0, 0, y0, v0)
    elif ic == "no_particular_slope":
        c1, c2 = solve_constants(f1, f2, yp.subs(x, 0) + 0 * x, 0, y0, v0)
    else:
        c1, c2 = solve_constants(f1, f2, yp, 0, y0, v0)
    return sympy.expand(c1 * f1 + c2 * f2 + yp_use), (c1, c2)


def _draw(rng, tier):
    plan = {
        "easy": [("distinct", m) for m in ("const", "linear", "exp", "exp")],
        "medium": [
            ("repeated", "exp"),
            ("repeated", "linear"),
            ("repeated", "const"),
            ("complex", "exp"),
            ("complex", "linear"),
            ("distinct", "cos"),
            ("distinct", "sin"),
        ],
        "hard": [
            ("distinct", "exp-res"),
            ("repeated", "exp-dres"),
            ("complex", "trig-res"),
            ("complex", "cos"),
            ("complex", "sin"),
        ],
    }[tier]
    for _ in attempts(1000, "a second-order initial value problem with clean constants"):
        kind, mode = rng.choice(plan)
        p, q = _draw_roots(rng, kind)
        if tier == "hard" and mode == "trig-res":
            p, q = sympy.Integer(0), sympy.Integer(rng.choice([1, 2, 3]))
        if tier == "hard" and kind == "complex" and mode in ("cos", "sin"):
            p = sympy.Integer(rng.choice([-2, -1]))  # damped, so the sinusoidal forcing is not resonant
        b, c = _coefficients(kind, p, q)
        got = _forcing(rng, kind, p, q, b, c, mode)
        if got is None:
            continue
        g, yp, wrongs = got
        y0, v0 = rng.randint(-4, 5), rng.randint(-6, 6)
        if y0 == 0 and v0 == 0:
            continue
        try:
            sol, (c1, c2) = _assemble(kind, p, q, yp, y0, v0)
        except VerificationError:
            continue
        if c1 == 0 or c2 == 0 or not (small_fraction(c1, 6, 24) and small_fraction(c2, 6, 24)):
            continue
        pts = (_R(-1, 10), _R(1, 2), _R(1))
        check(solves_ode(sol, (1, b, c), g, x, pts), "the solution does not satisfy the equation")
        check(sol.subs(x, 0) == y0 and sympy.diff(sol, x).subs(x, 0) == v0, "initial conditions are not met")
        return dict(
            kind=kind, mode=mode, p=p, q=q, b=b, c=c, g=g, yp=yp, wrongs=wrongs, y0=y0, v0=v0, sol=sol, c1=c1, c2=c2
        )
    raise AssertionError("unreachable")


def _wrong_solutions(d):
    kind, p, q, yp, y0, v0 = d["kind"], d["p"], d["q"], d["yp"], d["y0"], d["v0"]
    out = []

    def add(tag, fn):
        try:
            out.append((tag, fn()))
        except VerificationError:
            pass

    add("ode-ic-applied-to-homogeneous-only", lambda: _assemble(kind, p, q, yp, y0, v0, ic="hom_only")[0])
    add("ode-ic-derivative-particular-forgotten", lambda: _assemble_slope_slip(kind, p, q, yp, y0, v0))
    add("ode-homogeneous-root-sign", lambda: _assemble(kind, p, q, yp, y0, v0, roots_sign=True)[0])
    for tag, wrong_yp in d["wrongs"]:
        if wrong_yp is not None:
            add(tag, lambda wrong_yp=wrong_yp: _assemble(kind, p, q, sympy.sympify(wrong_yp), y0, v0)[0])
    add("ode-particular-sign-error", lambda: _assemble(kind, p, q, -yp, y0, v0)[0])
    if kind == "complex":
        add("ode-complex-real-part-dropped", lambda: _assemble_no_exp(p, q, yp, y0, v0))
    return out


def _assemble_slope_slip(kind, p, q, yp, y0, v0):
    """Constants fitted to y(0) - yp(0) and y'(0) (the slope of the particular solution forgotten)."""
    f1, f2 = _basis(kind, p, q)
    c1, c2 = solve_constants(f1, f2, yp.subs(x, 0), 0, y0, v0)
    return sympy.expand(c1 * f1 + c2 * f2 + yp)


def _assemble_no_exp(p, q, yp, y0, v0):
    """Complex roots written as cos(qx), sin(qx) with the factor e^(px) dropped."""
    if p == 0:
        raise VerificationError("no real part to drop")
    f1, f2 = sympy.cos(q * x), sympy.sin(q * x)
    c1, c2 = solve_constants(f1, f2, yp, 0, y0, v0)
    return sympy.expand(c1 * f1 + c2 * f2 + yp)


def _options(answer, wrongs, n=4):
    return expr_options(answer, wrongs, n)


def _ode_text(d) -> str:
    lhs = join_terms([(1, "y''"), (d["b"], "y'"), (d["c"], "y")])
    return f"{lhs} = {plain(d['g'])}"


@generator(
    "s2c1_second_order_ivp",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_second_order_ivp(rng, tier="easy"):
    d = _draw(rng, tier)
    stem, stem_latex = say_marked(
        rng.choice(
            [
                "Solve the initial value problem «{ode}, y(0) = {y0}, y'(0) = {v0}».",
                "Find «y(x)» given «{ode}» with «y(0) = {y0}» and «y'(0) = {v0}».",
            ]
        ).format(ode=_ode_text(d), y0=d["y0"], v0=d["v0"])
    )
    dist = _options(d["sol"], _wrong_solutions(d))
    check(len(dist) >= 2, "too few second-order distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(d["sol"]),
        answer_text=plain(d["sol"]),
        answer_kind="expression",
        distractors=dist,
        working=[
            f"The complementary function has the characteristic roots of {join_terms([(1, 'r²'), (d['b'], 'r'), (d['c'], '')])}.",
            f"A particular solution is {plain(d['yp'])}; the initial conditions fix the constants, giving y = {plain(d['sol'])}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c1_second_order_ivp", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_second_order_ivp(rng, tier="exam"):
    """Exam tier: a non-homogeneous second-order initial value problem in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam second-order question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    d = _draw(rng, rng.choice(["easy", "medium", "medium"]))
    kind, p, q, yp, y0, v0, sol, c1, c2 = (
        d["kind"],
        d["p"],
        d["q"],
        d["yp"],
        d["y0"],
        d["v0"],
        d["sol"],
        d["c1"],
        d["c2"],
    )
    f1, f2 = _basis(kind, p, q)
    x1 = rng.choice([_R(1, 2), _R(1), _R(3, 2)])
    value = float(sol.subs(x, x1).evalf(30))
    if not (0.01 <= abs(value) <= 500 and rounds_unambiguously(value, 3, 0.1)):
        raise _Redraw

    wrongs_sol = _wrong_solutions(d)
    # (a) the particular solution
    a_dist = expr_options(
        yp, [(t, w) for t, w in d["wrongs"] if w is not None] + [("ode-particular-sign-error", -yp)], 4
    )
    # (b), (c) the constants of the wrong fits
    b_wrongs, c_wrongs = [], []
    for tag, fn in (
        ("ode-ic-applied-to-homogeneous-only", lambda: solve_constants(f1, f2, 0, 0, y0, v0)),
        ("ode-ic-derivative-particular-forgotten", lambda: solve_constants(f1, f2, yp.subs(x, 0), 0, y0, v0)),
        ("ode-particular-sign-error", lambda: solve_constants(f1, f2, -yp, 0, y0, v0)),
    ):
        try:
            k1, k2 = fn()
        except VerificationError:
            continue
        b_wrongs.append((tag, k1))
        c_wrongs.append((tag, k2))
    b_dist = fraction_options(c1, b_wrongs, 4)
    c_dist = fraction_options(c2, c_wrongs, 4)
    d_wrongs = []
    for tag, w in wrongs_sol:
        try:
            d_wrongs.append((tag, float(sympy.sympify(w).subs(x, x1).evalf(30))))
        except (TypeError, ValueError):
            continue
    d_dist = number_options(value, 3, d_wrongs, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    if kind == "distinct":
        form = "«y = c1 e^(r1 x) + c2 e^(r2 x) + y_p», where «r1 > r2» are the roots of the characteristic equation"
    elif kind == "repeated":
        form = "«y = (c1 + c2 x) e^(rx) + y_p», where «r» is the repeated root of the characteristic equation"
    else:
        form = "«y = c1 e^(αx) cos(βx) + c2 e^(αx) sin(βx) + y_p», where «α ± βi» are the roots of the characteristic equation and «β > 0»"
    x1_text = f"{float(x1):g}"
    x1_ft = str(x1.p) if x1.q == 1 else f"{x1.p}/{x1.q}"
    yp1 = sympy.diff(yp, x).subs(x, 0)
    f1s, f2s = sympy.diff(f1, x).subs(x, 0), sympy.diff(f2, x).subs(x, 0)
    ft_c = f"({machine(sympy.nsimplify(v0 - yp1))})/({machine(f2s)}) - part_b*({machine(f1s)})/({machine(f2s)})"
    ft_d = (
        f"Subs(part_a, (x, e), ({x1_ft}, E)) + part_b*({machine(f1.subs(x, x1))}) + part_c*({machine(f2.subs(x, x1))})"
    )
    stem, stem_latex = say_marked(f"Consider the initial value problem «{_ode_text(d)}, y(0) = {y0}, y'(0) = {v0}».")
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part(
                    "Find the particular solution «y_p(x)» given by undetermined coefficients, with no terms from the "
                    "complementary function."
                ),
                answer_kind="expression",
                answer=machine(yp),
                answer_text=plain(yp),
                marks=3,
                working=[f"y_p = {plain(yp)}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(f"Write the general solution as {form}. Use the initial conditions to find «c1»."),
                answer_kind="fraction",
                answer=exact_value(c1),
                answer_text=rational_text(c1),
                marks=2,
                working=[f"c1 = {rational_text(c1)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Find «c2»."),
                answer_kind="fraction",
                answer=exact_value(c2),
                answer_text=rational_text(c2),
                marks=2,
                **({"follow_through": ft_c} if f1s != 0 else {}),
                working=[f"c2 = {rational_text(c2)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part("Find «y({x1})», correct to 3 decimal places.", x1=x1_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=f"{value:.3f}",
                marks=1,
                follow_through=ft_d,
                working=[f"y({x1_text}) = {value:.3f}."],
                distractors=d_dist,
            ),
        ],
    )
