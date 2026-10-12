"""Generates `cp1_cartesian_path_proof` items: show that a path given by a vector equation lies on a stated curve.

Topic ``cp1_cartesian_path_proof`` (open-response proof, ``level=14``). Each item gives r(t) = x(t) i + y(t) j, a
Cartesian equation, and asks the learner to replace x and y by their expressions in t on one side and simplify to the
other side, as a chain of equal expressions: the parameter t disappears and the Cartesian equation is what is left.

Tiers: circles and ellipses centred at the origin; ellipses with a centre and the same curves cleared of fractions;
hyperbolas built from sec and tan, and a double-angle path that is a parabola; rational parametrisations, a hyperbola
from t + 1/t, and shifted curves with the denominators cleared.

The generator never runs a full CAS chain check. Every reference chain is confirmed numerically at fixed irregular
points of t, and every stated Cartesian equation numerically along the path, with the `math` module only.
"""

from __future__ import annotations

import functools

import math

import sympy as sp

from ..core import check, generator
from . import _chain_proof_kit as kit
from ._chain_proof_kit import M, T

_SECOND_POINTS = (0.41, 0.83, 1.27, 1.71, 2.19)


def _sg(expr: str, c: int) -> str:
    """`expr + c` or `expr - |c|` written the way a person writes it."""
    return f"{expr} + {c}" if c >= 0 else f"{expr} - {-c}"


def _on_path(key: str, xs: str, ys: str, lhs: str, rhs: str, symbols_kind: str = "real") -> None:
    """The stated equation holds along the path: lhs(x(t), y(t)) == rhs(x(t), y(t)) at five values of t."""
    x, y, t = sp.symbols("x y t")
    lhs_e, rhs_e = sp.sympify(lhs), sp.sympify(rhs)
    xe, ye = sp.sympify(xs), sp.sympify(ys)
    for tv in _SECOND_POINTS:
        subs = {x: xe.subs(t, tv), y: ye.subs(t, tv)}
        a, b = float(lhs_e.subs(subs)), float(rhs_e.subs(subs))
        check(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f"{key}: the stated equation fails on the path at t = {tv}"
        )


def _fam(key, tier, xs, ys, claim_lhs, claim_rhs, ref, *, domain="", symbols_kind="real", jump_ok=False):
    _on_path(key, xs, ys, claim_lhs, claim_rhs)
    lead = [
        T("A path has vector equation r(t) = ("),
        M(xs),
        T(") i + ("),
        M(ys),
        T(") j. Show that its points satisfy "),
        M(claim_lhs),
        T(" = "),
        M(claim_rhs),
        T(". Replace x and y by their expressions in t, then simplify so that t disappears."),
    ]
    return dict(
        key=key,
        tier=tier,
        kind="identity",
        symbols={"t": symbols_kind},
        ref=ref,
        lead=lead,
        domain=domain,
        jump_ok=jump_ok,
        given_step="states the identity relating the trigonometric terms (for example sin^2 + cos^2 = 1) that removes t",
    )


def _circle(r):
    r2 = r * r
    ref = [
        f"({r}*cos(t))**2 + ({r}*sin(t))**2",
        f"{r2}*cos(t)**2 + {r2}*sin(t)**2",
        f"{r2}*(cos(t)**2 + sin(t)**2)",
        f"{r2}",
    ]
    return _fam(f"circle_{r}", "easy", f"{r}*cos(t)", f"{r}*sin(t)", "x**2 + y**2", f"{r2}", ref)


def _ellipse_origin(a, b):
    ref = [
        f"({a}*cos(t))**2/{a * a} + ({b}*sin(t))**2/{b * b}",
        "cos(t)**2 + sin(t)**2",
        "1",
    ]
    return _fam(
        f"ellipse_origin_{a}_{b}", "easy", f"{a}*cos(t)", f"{b}*sin(t)", f"x**2/{a * a} + y**2/{b * b}", "1", ref
    )


def _ellipse_centre(a, b, h, k):
    xs, ys = _sg(f"{a}*cos(t)", h), _sg(f"{b}*sin(t)", k)
    # (x - h) with x written out: (a cos t + h) - h
    xm = _sg(xs, -h)
    ym = _sg(ys, -k)
    xh = f"(x - {h})" if h > 0 else f"(x + {-h})"
    yk = f"(y - {k})" if k > 0 else f"(y + {-k})"
    ref = [
        f"({xm})**2/{a * a} + ({ym})**2/{b * b}",
        f"({a}*cos(t))**2/{a * a} + ({b}*sin(t))**2/{b * b}",
        "cos(t)**2 + sin(t)**2",
        "1",
    ]
    return _fam(
        f"ellipse_centre_{a}_{b}_{h}_{k}",
        "medium",
        xs,
        ys,
        f"{xh}**2/{a * a} + {yk}**2/{b * b}",
        "1",
        ref,
        jump_ok=True,
    )


def _ellipse_cleared(a, b):
    a2, b2 = a * a, b * b
    ref = [
        f"{b2}*({a}*cos(t))**2 + {a2}*({b}*sin(t))**2",
        f"{a2 * b2}*cos(t)**2 + {a2 * b2}*sin(t)**2",
        f"{a2 * b2}*(cos(t)**2 + sin(t)**2)",
        f"{a2 * b2}",
    ]
    return _fam(
        f"ellipse_cleared_{a}_{b}",
        "medium",
        f"{a}*cos(t)",
        f"{b}*sin(t)",
        f"{b2}*x**2 + {a2}*y**2",
        f"{a2 * b2}",
        ref,
    )


def _hyperbola(a, b, h, k):
    xs, ys = _sg(f"{a}*sec(t)", h), _sg(f"{b}*tan(t)", k)
    xm, ym = _sg(xs, -h), _sg(ys, -k)
    xh = f"(x - {h})" if h > 0 else f"(x + {-h})"
    yk = f"(y - {k})" if k > 0 else f"(y + {-k})"
    ref = [
        f"({xm})**2/{a * a} - ({ym})**2/{b * b}",
        f"({a}*sec(t))**2/{a * a} - ({b}*tan(t))**2/{b * b}",
        "sec(t)**2 - tan(t)**2",
        "1/cos(t)**2 - sin(t)**2/cos(t)**2",
        "(1 - sin(t)**2)/cos(t)**2",
        "1",
    ]
    return _fam(
        f"hyperbola_{a}_{b}_{h}_{k}",
        "hard",
        xs,
        ys,
        f"{xh}**2/{a * a} - {yk}**2/{b * b}",
        "1",
        ref,
        domain="Take t with cos(t) not equal to 0.",
        jump_ok=True,
    )


def _hyperbola_cleared(a, b):
    a2, b2 = a * a, b * b
    c = a2 * b2
    ref = [
        f"{b2}*({a}*sec(t))**2 - {a2}*({b}*tan(t))**2",
        f"{c}*sec(t)**2 - {c}*tan(t)**2",
        f"{c}*(sec(t)**2 - tan(t)**2)",
        f"{c}*(1/cos(t)**2 - sin(t)**2/cos(t)**2)",
        f"{c}*(1 - sin(t)**2)/cos(t)**2",
        f"{c}",
    ]
    return _fam(
        f"hyperbola_cleared_{a}_{b}",
        "hard",
        f"{a}*sec(t)",
        f"{b}*tan(t)",
        f"{b2}*x**2 - {a2}*y**2",
        f"{c}",
        ref,
        domain="Take t with cos(t) not equal to 0.",
        jump_ok=True,
    )


def _double_angle(a, b):
    coeff = sp.Rational(2 * b, a * a)
    cstr = f"{coeff.p}/{coeff.q}" if coeff.q != 1 else f"{coeff.p}"
    ref = [
        f"{b} - ({cstr})*({a}*sin(t))**2",
        f"{b} - {2 * b}*sin(t)**2",
        f"{b}*(1 - 2*sin(t)**2)",
        f"{b}*cos(2*t)",
    ]
    fam = _fam(
        f"double_angle_{a}_{b}",
        "hard",
        f"{a}*sin(t)",
        f"{b}*cos(2*t)",
        "y",
        f"{b} - ({cstr})*x**2",
        ref,
        jump_ok=True,
    )
    # The task is to reach y from the right-hand side, so the chain runs the other way round.
    fam["lead"][-1] = T(
        ". Replace x by its expression in t on the right-hand side and simplify until you reach the expression for y."
    )
    return fam


def _rational_circle(r):
    r2 = r * r
    ref = [
        f"({r}*(1 - t**2)/(1 + t**2))**2 + ({2 * r}*t/(1 + t**2))**2",
        f"{r2}*((1 - t**2)**2 + 4*t**2)/(1 + t**2)**2",
        f"{r2}*(1 + t**2)**2/(1 + t**2)**2",
        f"{r2}",
    ]
    return _fam(
        f"rational_circle_{r}",
        "exam",
        f"{r}*(1 - t**2)/(1 + t**2)",
        f"{2 * r}*t/(1 + t**2)",
        "x**2 + y**2",
        f"{r2}",
        ref,
        domain="Here t is any real number.",
        jump_ok=True,
    )


def _reciprocal_hyperbola(a, b):
    ref = [
        f"({a}*(t + 1/t))**2/{a * a} - ({b}*(t - 1/t))**2/{b * b}",
        "(t + 1/t)**2 - (t - 1/t)**2",
        "t**2 + 2 + 1/t**2 - (t**2 - 2 + 1/t**2)",
        "4",
    ]
    return _fam(
        f"reciprocal_hyperbola_{a}_{b}",
        "exam",
        f"{a}*(t + 1/t)",
        f"{b}*(t - 1/t)",
        f"x**2/{a * a} - y**2/{b * b}",
        "4",
        ref,
        domain="Take t to be a non-zero real number.",
        symbols_kind="nonzero",
        jump_ok=True,
    )


def _sine_cosine_sum(a):
    ref = [
        f"({a}*(sin(t) + cos(t)))**2 + ({a}*(sin(t) - cos(t)))**2",
        f"{a * a}*((sin(t) + cos(t))**2 + (sin(t) - cos(t))**2)",
        f"{a * a}*(2*sin(t)**2 + 2*cos(t)**2)",
        f"{2 * a * a}*(sin(t)**2 + cos(t)**2)",
        f"{2 * a * a}",
    ]
    return _fam(
        f"sine_cosine_sum_{a}",
        "exam",
        f"{a}*(sin(t) + cos(t))",
        f"{a}*(sin(t) - cos(t))",
        "x**2 + y**2",
        f"{2 * a * a}",
        ref,
        jump_ok=True,
    )


def _shifted_cleared(a, b, h, k):
    a2, b2 = a * a, b * b
    xs, ys = _sg(f"{a}*cos(t)", h), _sg(f"{b}*sin(t)", k)
    xm, ym = _sg(xs, -h), _sg(ys, -k)
    xh = f"(x - {h})" if h > 0 else f"(x + {-h})"
    yk = f"(y - {k})" if k > 0 else f"(y + {-k})"
    ref = [
        f"{b2}*({xm})**2 + {a2}*({ym})**2",
        f"{b2}*({a}*cos(t))**2 + {a2}*({b}*sin(t))**2",
        f"{a2 * b2}*cos(t)**2 + {a2 * b2}*sin(t)**2",
        f"{a2 * b2}*(cos(t)**2 + sin(t)**2)",
        f"{a2 * b2}",
    ]
    return _fam(
        f"shifted_cleared_{a}_{b}_{h}_{k}",
        "exam",
        xs,
        ys,
        f"{b2}*{xh}**2 + {a2}*{yk}**2",
        f"{a2 * b2}",
        ref,
        jump_ok=True,
    )


def _build_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": [], "exam": []}
    pools["easy"] += [_circle(r) for r in (2, 3, 4, 5, 6, 7)]
    pools["easy"] += [_ellipse_origin(a, b) for a, b in ((2, 3), (3, 2), (4, 3), (5, 2), (3, 5), (4, 5), (2, 5))]
    pools["medium"] += [
        _ellipse_centre(a, b, h, k)
        for a, b, h, k in (
            (3, 2, 1, -4),
            (2, 3, -2, 1),
            (4, 3, 2, 2),
            (5, 2, -1, 3),
            (3, 4, 3, -2),
            (2, 5, -3, -1),
            (4, 2, 1, 1),
            (3, 3, -2, 4),
        )
    ]
    pools["medium"] += [_ellipse_cleared(a, b) for a, b in ((3, 2), (2, 3), (4, 3), (5, 2), (3, 5), (4, 5))]
    pools["hard"] += [
        _hyperbola(a, b, h, k)
        for a, b, h, k in (
            (3, 2, 1, -4),
            (2, 3, -2, 1),
            (4, 3, 2, 2),
            (5, 2, -1, 3),
            (3, 4, 3, -2),
            (2, 5, -3, -1),
        )
    ]
    pools["hard"] += [_hyperbola_cleared(a, b) for a, b in ((3, 2), (2, 3), (4, 3), (5, 2), (3, 5))]
    pools["hard"] += [_double_angle(a, b) for a, b in ((2, 3), (4, 3), (3, 2), (2, 5), (4, 5), (3, 4))]
    pools["exam"] += [_rational_circle(r) for r in (1, 2, 3, 4, 5)]
    pools["exam"] += [_reciprocal_hyperbola(a, b) for a, b in ((1, 1), (2, 3), (3, 2), (2, 5), (4, 3), (3, 5))]
    pools["exam"] += [_sine_cosine_sum(a) for a in (2, 3, 4, 5, 6)]
    pools["exam"] += [
        _shifted_cleared(a, b, h, k)
        for a, b, h, k in ((3, 2, 2, -1), (2, 3, -1, 2), (4, 3, 1, 3), (5, 2, -3, -2), (3, 4, -2, 1), (2, 5, 3, 1))
    ]
    return pools


@functools.lru_cache(maxsize=None)
def pools() -> dict[str, list[dict]]:
    """The families, built on first use (building them runs SymPy, which the package import should not pay for)."""
    return _build_pools()


@generator(
    "cp1_cartesian_path_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def cp1_cartesian_path_proof(rng, variant="easy"):
    """`senior-derive-the-cartesian-equation-of-a-path-given`. Every question carries a marking dict
    (`Question.answer`): the working as a chain of equal expressions, the reference chain, the key steps and the traps."""
    if variant not in pools():
        raise ValueError(f"unknown variant: {variant}")
    return kit.prove_item(rng, kit.pick(rng, pools()[variant]))
