"""Generates `s2a3_stokes_circulation` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_stokes_circulation`.

``uc2-stokes-theorem-circulation`` (``multivariable-and-vector-calculus``, answer ``expression``): the circulation of a
field round a space curve C that bounds a surface, with the direction of travel stated and no surface supplied for
the curve: the learner picks one and applies Stokes' theorem, ∮ F · dr = ∬ curl F · n dS.

* easy: the triangle whose vertices are three given points of a plane z = d - p x - q y, counterclockwise from above.
* medium: the curve formed on a cylinder x^2 + y^2 = a^2 by an oblique plane z = p x + q y + c, counterclockwise from
  above; the surface is the elliptical piece of the plane, a graph over the disc.
* hard: the same with a cylinder whose axis is off-centre, or the rim of the paraboloid z = a^2 - x^2 - y^2; clockwise
  as often as counterclockwise, and a field of degree 3.
* exam (``exd_x_s2a3_stokes_circulation``): (a) the integrand (curl F) · (-f_x, -f_y, 1) on the surface as a function of
  x and y, (b) the circulation counterclockwise, (c) the work when the loop is reversed and the force tripled (follows
  through from (b)), (d) the circulation of F + ∇φ round the same loop, which is the same number.

Every circulation is worked out two ways: as the line integral of F along the parametrised curve, and as the flux of curl
F through the surface written as a graph (the cross product of its tangent vectors, polar coordinates where needed).
Stokes' theorem says they agree, and the generator will not build a question unless they do. Both are exact and checked
by quadrature.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from . import _s2a3_vc as vc
from ._s2a3_vc import C, S, x, y, z


def _curl(F):
    P, Q, R = F
    return [
        sp.expand(sp.diff(R, y) - sp.diff(Q, z)),
        sp.expand(sp.diff(P, z) - sp.diff(R, x)),
        sp.expand(sp.diff(Q, x) - sp.diff(P, y)),
    ]


def _field(rng, max_degree: int):
    for _ in attempts(300, "a field with a real curl that depends on z"):
        comps = [vc.rand_poly(rng, (x, y, z), max_degree, rng.choice([2, 2, 3]), (-3, 3)) for _ in range(3)]
        curl = _curl(comps)
        if all(c != 0 for c in comps) and any(c != 0 for c in curl) and any(c.has(z) for c in comps):
            return comps
    raise AssertionError("unreachable")


def _surface_integrand(curl, f, variant: str = "right"):
    """The function of (x, y) whose integral over the base is the upward flux of curl F through z = f(x, y)."""
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    cx, cy, cz = (c.xreplace({z: f}) for c in curl)
    if variant == "right":
        return sp.expand(-cx * fx - cy * fy + cz)
    if variant == "signs":
        return sp.expand(cx * fx + cy * fy + cz)
    if variant == "flat":
        return sp.expand(cz)
    if variant == "z-not-replaced":
        c0 = [c.xreplace({z: 0}) for c in curl]
        return sp.expand(-c0[0] * fx - c0[1] * fy + c0[2])
    if variant == "curl-sign":
        return sp.expand(-cx * fx + cy * fy + cz)
    raise ValueError(variant)


def _stokes(F, f, region, boundary):
    """(circulation counterclockwise from above, integrand g(x, y)), by the surface and by the boundary: they must agree."""
    curl = _curl(F)
    g = _surface_integrand(curl, f)
    by_surface = vc.region_integral(g, region)
    if vc.CHECKS:
        by_graph = vc.flux(curl, vc.graph_over(region, f), (0, 0, 1))
        by_boundary = sum((vc.line(F, c) for c in boundary), sp.Integer(0))
        check(
            sp.expand(by_surface - by_graph) == 0, "the surface integral of curl F disagrees with its parametrised flux"
        )
        check(
            sp.expand(by_surface - by_boundary) == 0,
            f"Stokes' theorem fails: surface {by_surface}, boundary {by_boundary}",
        )
    return by_surface, g, curl


# ---------------------------------------------------------------------------------------------------------------
# the curves


def _pt3(p) -> str:
    return f"({p[0]}, {p[1]}, {p[2]})"


def _plane_text(f) -> str:
    return "z = " + vc.poly_text(f, (x, y))


def _shape(rng, tier):
    if tier == "easy":
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        d, p, q = rng.randint(1, 4), rng.choice([-1, 1, 2]), rng.choice([-1, 1, 2])
        f = d - p * x - q * y
        pts = [(0, 0, d), (a, 0, d - p * a), (0, b, d - q * b)]
        loop = [vc.segment(pts[i], pts[(i + 1) % 3]) for i in range(3)]
        text = f"the triangle with vertices «{', '.join(_pt3(q_) for q_ in pts)}»"
        return dict(f=f, region=vc.tri_region(a, b, "corner"), boundary=loop, curve=text, polar=False)
    hard = tier == "hard"
    if hard and rng.random() < 0.4:
        a = rng.randint(1, 2)
        f = a * a - x**2 - y**2
        rim = vc.circle_curve((0, 0), a, height=0)
        text = f"the circle «x² + y² = {a * a}» in the plane «z = 0», which is the rim of the paraboloid «z = {a * a} - x² - y²»"
        return dict(f=f, region=vc.polar_region(a), boundary=[rim], curve=text, polar=True)
    a = rng.randint(1, 3 if not hard else 2)
    h, k = (rng.choice([-1, 1]), rng.choice([0, 1, -1])) if hard else (0, 0)
    p, q, c = rng.choice([-2, -1, 1, 2]), rng.choice([-1, 0, 1, 2]), rng.randint(0, 3)
    f = c + p * x + q * y
    rim_f = f.xreplace({x: h + a * C, y: k + a * S})
    rim = vc.Curve(sp.Matrix([h + a * C, k + a * S, rim_f]), vc.Dom((vc.t,), ((0, 4),), {vc.t: (C, S)}))
    cyl = (
        "x² + y²"
        if (h, k) == (0, 0)
        else f"{'x' if h == 0 else (f'(x - {h})' if h > 0 else f'(x + {-h})')}² + {'y' if k == 0 else (f'(y - {k})' if k > 0 else f'(y + {-k})')}²"
    )
    text = f"the curve formed on the cylinder «{cyl} = {a * a}» by the plane «{_plane_text(f)}»"
    return dict(f=f, region=vc.polar_region(a, centre=(h, k)), boundary=[rim], curve=text, polar=True)


def _candidates(F, shape, sign, answer, g):
    f, region = shape["f"], shape["region"]
    curl = _curl(F)
    out = [("stokes-wrong-direction", -answer)]
    for tag, variant in (
        ("stokes-flat-surface", "flat"),
        ("stokes-normal-signs", "signs"),
        ("stokes-z-not-replaced", "z-not-replaced"),
        ("stokes-curl-sign-slip", "curl-sign"),
    ):
        out.append(
            (
                tag,
                lambda variant=variant: (
                    sign * vc.region_integral(_surface_integrand(curl, f, variant), region, verify=False)
                ),
            )
        )
    out.append(("stokes-flux-of-field", lambda: sign * vc.flux(F, vc.graph_over(region, f), (0, 0, 1), verify=False)))
    if shape["polar"]:
        sub = dict(zip((x, y), region.r[:2]))
        out.append(
            ("stokes-polar-no-r", lambda: sign * vc.integrate(sp.expand(g.xreplace(sub)), region.dom, verify=False))
        )
    return out


_STEMS = {
    "easy": [
        "Let C be {curve}, traversed counterclockwise when viewed from above. Evaluate «∮C F · dr» for «F(x, y, z) = {F}».",
        "A particle goes once round {curve}, in the order given and counterclockwise as seen from above, under the force «F(x, y, z) = {F}». Find the work done.",
    ],
    "medium": [
        "Let C be {curve}, oriented counterclockwise when viewed from above. Evaluate the circulation «∮C F · dr» of «F(x, y, z) = {F}».",
        "A bead slides once round {curve}, counterclockwise as seen from above, under the force «F(x, y, z) = {F}». Find the work done.",
    ],
}
_STEMS["hard"] = [
    "Let C be {curve}, traversed {way} when viewed from above. Evaluate «∮C F · dr» for «F(x, y, z) = {F}».",
    "Find the circulation of «F(x, y, z) = {F}» round {curve}, traversed {way} as seen from above.",
]


def _draw(rng, tier):
    F = _field(rng, 2 if tier != "hard" else 3)
    shape = _shape(rng, tier)
    way = "counterclockwise" if tier != "hard" else rng.choice(["counterclockwise", "clockwise"])
    sign = 1 if way == "counterclockwise" else -1
    ccw, g, _ = _stokes(F, shape["f"], shape["region"], shape["boundary"])
    answer = sign * ccw
    if answer == 0 or abs(answer) > 4000:
        return None
    template = rng.choice(_STEMS[tier]).replace("{curve}", shape["curve"])
    text = vc.stem(template, F=vc.mvec(F), way=way)
    working = [
        "Choose a surface with boundary C (the part of the plane or paraboloid inside C). Stokes: the counterclockwise circulation is the flux of curl F through it with the upward normal, "
        "which is the integral of (curl F) · (-f_x, -f_y, 1) over the base region."
        + (" A clockwise traversal changes the sign." if sign < 0 else "")
    ]
    return dict(**text, answer=answer, candidates=_candidates(F, shape, sign, answer, g), working=working)


@generator(
    "s2a3_stokes_circulation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a3_stokes_circulation(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A wire loop is {curve}, with lengths in metres. A charged bead slides round it under the force «F(x, y, z) = {F}», in newtons.",
    "A cable is laid along {curve} (lengths in metres). A winch pulls a cart round it under the force «F(x, y, z) = {F}», in newtons.",
]
_A = [
    "Take the surface to be the piece of the plane inside the loop, as a graph «z = f(x, y)», with the upward normal. Write the work in one counterclockwise circuit as «∬D g(x, y) dA» over its base D, and find «g(x, y)».",
    "By Stokes' theorem the work in one counterclockwise circuit is «∬D g(x, y) dA», where D is the base of the plane piece inside the loop and «g = curl F · (-f_x, -f_y, 1)». Find «g(x, y)».",
]
_B = [
    "Find the work done by F in one counterclockwise circuit, in joules.",
    "Evaluate the work done by the force in one counterclockwise circuit, in joules.",
]
_C = [
    "The cart is sent round the loop clockwise as seen from above, with the force tripled to «3F». Find the work done in one circuit, in joules.",
    "Now the loop is run clockwise as seen from above and the force is tripled to «3F». Find the work done in one circuit, in joules.",
]
_D = [
    "A conservative force «∇φ», with «φ(x, y, z) = {phi}», is added, so the force is «F + ∇φ». Find the work done in one counterclockwise circuit, in joules.",
    "The force is changed to «F + ∇φ», where «φ(x, y, z) = {phi}». Find the work done in one counterclockwise circuit, in joules.",
]


def _exam_once(rng):
    F = _field(rng, 2)
    shape = _shape(rng, "medium")
    ccw, g, curl = _stokes(F, shape["f"], shape["region"], shape["boundary"])
    if ccw == 0 or abs(ccw) > 4000:
        raise vc.Redraw
    phi = vc.rand_poly(rng, (x, y, z), 3, 2, (1, 3))
    G = [sp.expand(F[0] + sp.diff(phi, x)), sp.expand(F[1] + sp.diff(phi, y)), sp.expand(F[2] + sp.diff(phi, z))]
    if vc.CHECKS:
        check(
            sp.expand(sum(vc.line(G, c) for c in shape["boundary"]) - ccw) == 0,
            "adding a gradient changed the circulation",
        )
    f, region = shape["f"], shape["region"]
    wrong = {v: _surface_integrand(curl, f, v) for v in ("signs", "flat", "z-not-replaced", "curl-sign")}
    wrong_vals = {v: lambda e=e: vc.region_integral(e, region, verify=False) for v, e in wrong.items()}
    tags = {
        "signs": "stokes-normal-signs",
        "flat": "stokes-flat-surface",
        "z-not-replaced": "stokes-z-not-replaced",
        "curl-sign": "stokes-curl-sign-slip",
    }

    def line_gradient():
        grad = [sp.diff(phi, x), sp.diff(phi, y), sp.diff(phi, z)]
        return sum((vc.line(grad, c, verify=False) for c in shape["boundary"][:1]), sp.Integer(0))

    def flux_field():
        return vc.flux(F, vc.graph_over(region, f), (0, 0, 1), verify=False)

    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_A)),
            g,
            [(tags[v], e) for v, e in wrong.items()],
            3,
            ["Substitute z = f(x, y) into curl F and dot with (-f_x, -f_y, 1)."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_B)),
            ccw,
            [(tags[v], val) for v, val in wrong_vals.items()]
            + [("stokes-wrong-direction", -ccw), ("stokes-flux-of-field", flux_field)],
            3,
            ["Integrate g over the base region D, using polar coordinates."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_C)),
            -3 * ccw,
            [
                ("stokes-orientation-ignored", 3 * ccw),
                ("stokes-scale-ignored", -ccw),
                ("stokes-scale-squared", -9 * ccw),
            ],
            2,
            ["Clockwise changes the sign; tripling the force triples the work: -3 times (b)."],
            follow_through="-3*part_b",
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_D), phi=vc.poly_text(phi, (x, y, z))),
            ccw,
            [
                ("stokes-gradient-not-ignored", lambda: ccw + line_gradient()),
                ("stokes-conservative-all-zero", sp.Integer(0)),
                ("stokes-wrong-direction", -ccw),
                ("stokes-gradient-doubled", 2 * ccw),
            ],
            2,
            ["curl of a gradient is 0, so the circulation is unchanged: the same as (b)."],
            follow_through="part_b",
        ),
    ]
    template = rng.choice(_EXAM_STEMS).replace("{curve}", shape["curve"])
    return dict(**vc.stem(template, F=vc.mvec(F)), parts=parts)


@generator("exd_x_s2a3_stokes_circulation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_stokes_circulation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
