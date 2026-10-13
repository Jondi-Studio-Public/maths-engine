"""Generates `fy1ms_chain_rule` questions and the exam chain `exd_x_fy1ms_chain_rule`.

Skill: `ucv1-multivariable-chain-rule` (multivariable-and-vector-calculus). The chain rule for z = f(x, y) when x and
y depend on one parameter t (dz/dt along a path) or on two (partial derivatives after a change of variables).

Tiers (topic `fy1ms_chain_rule`; every answer is a whole number):

* easy   -- dz/dt at a given t along a straight-line path x = at + b, y = ct + d, for a quadratic f;
* medium -- one of: dz/dt along a curved (quadratic or cubic) path; or the partial derivative dz/du after a linear
            change of variables x = au + bv, y = cu + dv;
* hard   -- one of: a partial derivative dz/du or dz/dv after a curved change of variables (x = uv, y = u + v^2 and
            similar, so the inner derivatives are not constants); or the second derivative d^2z/dt^2 along a straight
            line, which needs f_xx, f_xy and f_yy and the cross term 2 f_xy x' y'.

Exam (topic `exd_x_fy1ms_chain_rule`): a metal plate with temperature T(x, y) and a sensor moving along a path.
(a) T_x at the sensor's position at t0 (1 mark), (b) T_y there (1 mark), (c) dT/dt at t0 (2 marks, follow-through
from (a) and (b)), (d) the approximate change in the reading over a short time dt (2 marks, follow-through from (c)).

Every key is computed by the chain rule from the partial derivatives and checked a second way: substitute the path
into f first, then differentiate with respect to the parameter(s) (SymPy). Distractors are the usual slips: the
inner derivatives left out, only one of the two terms, the partials swapped, the partials evaluated at the parameter
instead of at the point, the cross term of the second derivative dropped or not doubled.
"""

from __future__ import annotations

import functools
import random

import sympy
from sympy import symbols

from ..core import attempts, check, generator
from ._fy1ms_common import Fm, build_until, num_dist, number_part, number_question, plain_expr, say

x, y, t, u, v = symbols("x y t u v")


@functools.cache
def _diff(expr, *wrt):
    """sympy.diff, cached: the same few surfaces and paths are differentiated on every seed."""
    return sympy.diff(expr, *wrt)


_MONOMIALS = [x**2, x * y, y**2, x, y, x**3, x**2 * y, x * y**2, y**3]
_COEFFS = (-3, -2, -1, 1, 2, 3)


def _poly(rng: random.Random, *, cubic: bool) -> sympy.Expr:
    pool = _MONOMIALS if cubic else _MONOMIALS[:5]
    for _ in attempts(500, "a polynomial in x and y with both variables"):
        chosen = rng.sample(pool, rng.choice((3, 4)))
        f = sum((rng.choice(_COEFFS) * m for m in chosen), sympy.Integer(0))
        if sympy.degree(f, x) >= 1 and sympy.degree(f, y) >= 1 and f.has(x) and f.has(y):
            if (cubic and sympy.Poly(f, x, y).total_degree() >= 3) or (
                not cubic and sympy.Poly(f, x, y).total_degree() == 2
            ):
                return f
    raise AssertionError("unreachable")  # pragma: no cover


def _lin(rng: random.Random, var) -> sympy.Expr:
    return rng.choice((-3, -2, -1, 1, 2, 3)) * var + rng.choice((-3, -2, -1, 0, 1, 2, 3))


def _partials(f, at: dict):
    return _diff(f, x).subs(at), _diff(f, y).subs(at)


def _val(expr) -> int:
    expr = sympy.nsimplify(expr)
    check(expr.is_Integer, f"{expr} is not a whole number")
    return int(expr)


def _along_path(f, xt, yt, t0):
    """dz/dt at t0 by the chain rule (primary route) and by substituting the path first (second route)."""
    pt = {x: xt.subs(t, t0), y: yt.subs(t, t0)}
    fx, fy = _partials(f, pt)
    dxdt, dydt = _diff(xt, t).subs(t, t0), _diff(yt, t).subs(t, t0)
    chain = fx * dxdt + fy * dydt
    direct = _diff(f.subs({x: xt, y: yt}), t).subs(t, t0)
    check(chain == direct, "chain rule disagrees with substitute-then-differentiate")
    return pt, fx, fy, dxdt, dydt, int(chain)


def _tex_path(label: str, e) -> Fm:
    return Fm(f"{label} = {plain_expr(e)}", f"{label} = {sympy.latex(e)}")


def _cands_dzdt(f, xt, yt, t0, pt, fx, fy, dxdt, dydt):
    fx_t, fy_t = _partials(f, {x: t0, y: t0})
    return [
        num_dist(fx + fy, 0, "chain-rule-inner-derivatives-omitted"),
        num_dist(fx * dxdt, 0, "chain-rule-one-term-only"),
        num_dist(fy * dxdt + fx * dydt, 0, "chain-rule-partials-swapped"),
        num_dist(fx_t * dxdt + fy_t * dydt, 0, "chain-rule-partials-evaluated-at-parameter"),
    ]


# ---------------------------------------------------------------------------
# dz/dt along a path


def _draw_path(rng: random.Random, *, straight: bool):
    for _ in attempts(2000, "a usable path"):
        cubic = not straight and rng.random() < 0.4
        f = _poly(rng, cubic=False) if straight else _poly(rng, cubic=cubic)
        if straight:
            xt, yt = _lin(rng, t), _lin(rng, t)
        else:
            xt = rng.choice((-2, -1, 1, 2)) * t**2 + rng.choice((-2, -1, 0, 1, 2)) * t + rng.choice((-1, 0, 1, 2))
            yt = rng.choice((-2, -1, 1, 2)) * t ** rng.choice((1, 2, 3)) + rng.choice((-2, -1, 0, 1, 2))
        t0 = rng.choice((-2, -1, 1, 2, 3))
        pt, fx, fy, dxdt, dydt, answer = _along_path(f, xt, yt, t0)
        if answer == 0 or abs(answer) > 2500 or fx == 0 or fy == 0 or dxdt == 0 or dydt == 0:
            continue
        if fx + fy == answer or fx * dxdt == answer:
            continue
        return f, xt, yt, t0, pt, fx, fy, dxdt, dydt, answer
    raise AssertionError("unreachable")  # pragma: no cover


def _dzdt_question(rng: random.Random, *, straight: bool) -> dict:
    f, xt, yt, t0, pt, fx, fy, dxdt, dydt, answer = _draw_path(rng, straight=straight)
    cands = _cands_dzdt(f, xt, yt, t0, pt, fx, fy, dxdt, dydt)
    text, latex = say(
        "Let ",
        Fm("z = " + plain_expr(f), "z = " + sympy.latex(f)),
        ", where ",
        _tex_path("x", xt),
        " and ",
        _tex_path("y", yt),
        ". Find ",
        Fm("dz/dt", r"\frac{dz}{dt}"),
        " when ",
        Fm(f"t = {t0}"),
        ".",
    )
    return number_question(
        text,
        latex,
        answer,
        0,
        cands,
        working=[
            f"At t = {t0}: x = {pt[x]}, y = {pt[y]}, dx/dt = {dxdt}, dy/dt = {dydt}.",
            f"∂z/∂x = {fx} and ∂z/∂y = {fy} there, so dz/dt = {fx}({dxdt}) + {fy}({dydt}) = {answer}.".replace(
                "+ -", "- "
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Partial derivatives after a change of variables

_CURVED_MAPS = [
    (lambda: u * v, lambda: u + v**2),
    (lambda: u**2 - v**2, lambda: 2 * u * v),
    (lambda: u**2 + v, lambda: u - v**2),
    (lambda: u * v, lambda: u**2 - v),
    (lambda: u + v**2, lambda: u * v + 1),
]


def _draw_change(rng: random.Random, *, linear: bool):
    for _ in attempts(3000, "a usable change of variables"):
        f = _poly(rng, cubic=not linear and rng.random() < 0.5)
        if linear:
            a, b, c, d = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(4))
            if a * d - b * c == 0:
                continue
            xs, ys = a * u + b * v, c * u + d * v
        else:
            mx, my = rng.choice(_CURVED_MAPS)
            xs, ys = mx(), my()
        u0, v0 = rng.choice((-2, -1, 1, 2)), rng.choice((-2, -1, 1, 2))
        pt = {x: xs.subs({u: u0, v: v0}), y: ys.subs({u: u0, v: v0})}
        fx, fy = _partials(f, pt)
        wrt = rng.choice((u, v))
        dxs, dys = _diff(xs, wrt).subs({u: u0, v: v0}), _diff(ys, wrt).subs({u: u0, v: v0})
        answer = int(fx * dxs + fy * dys)
        direct = _diff(f.subs({x: xs, y: ys}), wrt).subs({u: u0, v: v0})
        check(answer == direct, "chain rule disagrees with substitute-then-differentiate")
        other = v if wrt == u else u
        dxo, dyo = _diff(xs, other).subs({u: u0, v: v0}), _diff(ys, other).subs({u: u0, v: v0})
        if answer == 0 or abs(answer) > 2500 or fx == 0 or fy == 0 or dxs == 0 or dys == 0:
            continue
        if len({answer, int(fx + fy), int(fx * dxs), int(fy * dxs + fx * dys), int(fx * dxo + fy * dyo)}) < 5:
            continue
        return f, xs, ys, wrt, u0, v0, pt, fx, fy, dxs, dys, dxo, dyo, answer
    raise AssertionError("unreachable")  # pragma: no cover


def _change_question(rng: random.Random, *, linear: bool) -> dict:
    f, xs, ys, wrt, u0, v0, pt, fx, fy, dxs, dys, dxo, dyo, answer = _draw_change(rng, linear=linear)
    fx_u, fy_u = _partials(f, {x: u0, y: v0})
    cands = [
        num_dist(int(fx + fy), 0, "chain-rule-inner-derivatives-omitted"),
        num_dist(int(fx * dxs), 0, "chain-rule-one-term-only"),
        num_dist(int(fy * dxs + fx * dys), 0, "chain-rule-partials-swapped"),
        num_dist(int(fx * dxo + fy * dyo), 0, "chain-rule-wrong-variable-differentiated"),
        num_dist(int(fx_u * dxs + fy_u * dys), 0, "chain-rule-partials-evaluated-at-parameter"),
    ]
    name = "u" if wrt == u else "v"
    text, latex = say(
        "Let ",
        Fm("z = " + plain_expr(f), "z = " + sympy.latex(f)),
        ", where ",
        _tex_path("x", xs),
        " and ",
        _tex_path("y", ys),
        ". Find ",
        Fm(f"∂z/∂{name}", rf"\frac{{\partial z}}{{\partial {name}}}"),
        " when ",
        Fm(f"u = {u0}"),
        " and ",
        Fm(f"v = {v0}"),
        ".",
    )
    return number_question(
        text,
        latex,
        answer,
        0,
        cands,
        working=[
            f"At u = {u0}, v = {v0}: x = {pt[x]}, y = {pt[y]}, ∂x/∂{name} = {dxs}, ∂y/∂{name} = {dys}.",
            f"∂z/∂x = {fx} and ∂z/∂y = {fy} there, so ∂z/∂{name} = {fx}({dxs}) + {fy}({dys}) = {answer}.".replace(
                "+ -", "- "
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Second derivative along a straight line


def _second_question(rng: random.Random) -> dict:
    for _ in attempts(3000, "a usable second-derivative question"):
        f = _poly(rng, cubic=rng.random() < 0.6)
        xt, yt = _lin(rng, t), _lin(rng, t)
        a, c = _diff(xt, t), _diff(yt, t)
        t0 = rng.choice((-2, -1, 1, 2))
        pt = {x: xt.subs(t, t0), y: yt.subs(t, t0)}
        fxx, fxy, fyy = (_diff(f, *vars_).subs(pt) for vars_ in ((x, x), (x, y), (y, y)))
        answer = int(fxx * a**2 + 2 * fxy * a * c + fyy * c**2)
        direct = _diff(f.subs({x: xt, y: yt}), t, 2).subs(t, t0)
        check(answer == direct, "second derivative disagrees with substitute-then-differentiate")
        first = int(_diff(f, x).subs(pt) * a + _diff(f, y).subs(pt) * c)
        wrong = {
            "second-derivative-cross-term-dropped": int(fxx * a**2 + fyy * c**2),
            "second-derivative-cross-term-not-doubled": int(fxx * a**2 + fxy * a * c + fyy * c**2),
            "second-derivative-inner-derivatives-not-squared": int(fxx * a + 2 * fxy * a * c + fyy * c),
            "second-derivative-stopped-at-first": first,
        }
        if fxy == 0 or answer == 0 or abs(answer) > 2500 or answer in wrong.values() or len(set(wrong.values())) < 4:
            continue
        break
    cands = [num_dist(val, 0, tag) for tag, val in wrong.items()]
    text, latex = say(
        "Let ",
        Fm("z = " + plain_expr(f), "z = " + sympy.latex(f)),
        ", where ",
        _tex_path("x", xt),
        " and ",
        _tex_path("y", yt),
        ". Find ",
        Fm("d²z/dt²", r"\frac{d^2 z}{dt^2}"),
        " when ",
        Fm(f"t = {t0}"),
        ".",
    )
    return number_question(
        text,
        latex,
        answer,
        0,
        cands,
        working=[
            f"At t = {t0}: f_xx = {fxx}, f_xy = {fxy}, f_yy = {fyy}, dx/dt = {a}, dy/dt = {c}.",
            f"d²z/dt² = f_xx(dx/dt)² + 2f_xy(dx/dt)(dy/dt) + f_yy(dy/dt)² = {answer} (the path is straight, so x'' = y'' = 0).",
        ],
    )


@generator(
    "fy1ms_chain_rule",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_chain_rule(rng: random.Random, tier: str = "easy") -> dict:
    """Chain rule: dz/dt on a straight path (easy), a curved path or linear change of variables (medium), a curved change of variables or d2z/dt2 (hard)."""
    if tier == "easy":
        return build_until(rng, lambda r: _dzdt_question(r, straight=True), "an easy chain-rule question")
    if tier == "medium":
        return build_until(
            rng,
            lambda r: _dzdt_question(r, straight=False) if r.random() < 0.5 else _change_question(r, linear=True),
            "a medium chain-rule question",
        )
    if tier == "hard":
        return build_until(
            rng,
            lambda r: _change_question(r, linear=False) if r.random() < 0.55 else _second_question(r),
            "a hard chain-rule question",
        )
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain: a temperature field and a moving sensor


@generator("exd_x_fy1ms_chain_rule", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_chain_rule(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: T_x, T_y at the sensor, dT/dt, the change over a short time (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a chain-rule exam question")


def _exam_chain(rng: random.Random) -> dict:
    for _ in attempts(3000, "a usable sensor path"):
        f = _poly(rng, cubic=rng.random() < 0.5)
        xt = rng.choice((1, 2, 3)) * t + rng.choice((0, 1, 2, 3))
        if rng.random() < 0.5:
            xt = xt + rng.choice((1, 2)) * t**2 if rng.random() < 0.5 else xt
        yt = rng.choice((-2, -1, 1, 2)) * t + rng.choice((0, 1, 2, 3, 4))
        t0 = rng.choice((1, 2, 3))
        pt, fx, fy, dxdt, dydt, answer = _along_path(f, xt, yt, t0)
        if answer == 0 or abs(answer) > 900 or fx == 0 or fy == 0 or dxdt == 0 or dydt == 0 or fx == fy:
            continue
        if min(pt[x], pt[y]) <= 0 or len({answer, fx + fy, fx * dxdt, fy * dxdt + fx * dydt}) < 4:
            continue
        break
    dt = rng.choice((1, 2, 5)) * sympy.Rational(1, 10)
    dp = 1 if dt.q == 10 else 2
    est = answer * dt
    check(est.q in (1, 2, 5, 10), "estimate is not a finite decimal")
    fx_t, fy_t = _partials(f, {x: t0, y: t0})
    a_cands = [
        num_dist(fy, 0, "chain-rule-partials-swapped"),
        num_dist(fx_t, 0, "chain-rule-partials-evaluated-at-parameter"),
        num_dist(fx * dxdt, 0, "partial-multiplied-by-path-derivative"),
        num_dist(-fx, 0, "partial-sign-slip"),
        num_dist(_diff(f, x).subs({x: pt[y], y: pt[x]}), 0, "chain-rule-point-coordinates-swapped"),
    ]
    b_cands = [
        num_dist(fx, 0, "chain-rule-partials-swapped"),
        num_dist(fy_t, 0, "chain-rule-partials-evaluated-at-parameter"),
        num_dist(fy * dydt, 0, "partial-multiplied-by-path-derivative"),
        num_dist(-fy, 0, "partial-sign-slip"),
        num_dist(_diff(f, y).subs({x: pt[y], y: pt[x]}), 0, "chain-rule-point-coordinates-swapped"),
    ]
    c_cands = [
        num_dist(fx + fy, 0, "chain-rule-inner-derivatives-omitted"),
        num_dist(fx * dxdt, 0, "chain-rule-one-term-only"),
        num_dist(fy * dxdt + fx * dydt, 0, "chain-rule-partials-swapped"),
        num_dist(fx_t * dxdt + fy_t * dydt, 0, "chain-rule-partials-evaluated-at-parameter"),
    ]
    d_cands = [
        num_dist(answer / dt, dp, "rate-divided-by-interval"),
        num_dist(answer * t0, dp, "rate-times-time-not-interval"),
        num_dist(answer + dt, dp, "rate-plus-interval"),
        num_dist(answer * (t0 + dt), dp, "rate-times-final-time"),
    ]
    stem, stem_latex = say(
        "A thin metal plate has temperature ",
        Fm("T(x, y) = " + plain_expr(f), "T(x, y) = " + sympy.latex(f)),
        " degrees Celsius at the point (x, y), where x and y are in centimetres. A sensor moves across the plate so that at time t seconds it is at ",
        _tex_path("x", xt),
        ", ",
        _tex_path("y", yt),
        ".",
    )
    dt_text = f"{float(dt):g}"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            number_part(
                f"Find ∂T/∂x at the sensor's position when t = {t0}.",
                fx,
                0,
                a_cands,
                marks=1,
                prompt_latex=rf"\text{{Find }}\frac{{\partial T}}{{\partial x}}\text{{ at the sensor's position when t = {t0}.}}",
                working=[f"At t = {t0} the sensor is at ({pt[x]}, {pt[y]}), where ∂T/∂x = {fx}."],
            ),
            number_part(
                "Find ∂T/∂y at the same position.",
                fy,
                0,
                b_cands,
                marks=1,
                prompt_latex=r"\text{Find }\frac{\partial T}{\partial y}\text{ at the same position.}",
                working=[f"∂T/∂y = {fy} at ({pt[x]}, {pt[y]})."],
            ),
            number_part(
                f"Hence find dT/dt, the rate at which the sensor's reading is changing, when t = {t0}.",
                answer,
                0,
                c_cands,
                marks=2,
                prompt_latex=rf"\text{{Hence find }}\frac{{dT}}{{dt}}\text{{, the rate at which the sensor's reading is changing, when t = {t0}.}}",
                follow_through=f"part_a*({dxdt}) + part_b*({dydt})",
                working=[
                    f"dx/dt = {dxdt} and dy/dt = {dydt} at t = {t0}, so dT/dt = {fx}({dxdt}) + {fy}({dydt}) = {answer}.".replace(
                        "+ -", "- "
                    )
                ],
            ),
            number_part(
                f"Use your answer to estimate how much the sensor's reading changes in the {dt_text} seconds after t = {t0}. Give your answer correct to {dp} decimal place{'s' if dp != 1 else ''}.",
                est,
                dp,
                d_cands,
                marks=2,
                follow_through=f"part_c*({dt.p}/{dt.q})",
                working=[f"Change ≈ (dT/dt)(Δt) = {answer} × {dt_text} = {float(est):.{dp}f} degrees."],
            ),
        ],
    )
