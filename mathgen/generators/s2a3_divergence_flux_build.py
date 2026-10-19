"""Generates `s2a3_divergence_flux` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_divergence_flux`.

``uc2-divergence-theorem-flux`` (``multivariable-and-vector-calculus``, answer ``expression``): the outward flux of a
field through a closed surface, found as the integral of div F over the solid it encloses.

* easy: the whole surface of a box, whose six faces would each need a separate integral.
* medium: a sphere, or a closed cylinder (side and two ends). The volume integral is done in spherical or cylindrical
  coordinates.
* hard: a closed cone with its lid, a paraboloid bowl with its lid, or only the curved surface of a hemisphere, where
  the learner has to close the surface with the flat base and take the base flux away again.
* exam (``exd_x_s2a3_divergence_flux``): a dome, a cone or a bowl: (a) div F, (b) the total outward flux through the
  closed surface (flat piece included), (c) the flux through the flat piece alone, (d) the flux through the curved
  piece, which follows through as (b) - (c).

Every flux is worked out two ways: as the integral of div F over the solid, and as the sum of the surface fluxes of the
pieces of the boundary, each parametrised with an outward normal. The divergence theorem says they agree, and the
generator will not build a question unless they do. Both routes are exact and also checked by quadrature.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from ._s2_options import keep_answer_shape
from . import _s2a3_vc as vc
from ._s2a3_vc import C, S, u, v, w, x, y, z


def _div(F):
    return sp.expand(sum(sp.diff(c, g) for c, g in zip(F, (x, y, z))))


def _field(rng, max_degree: int):
    for _ in attempts(300, "a field with a non-zero divergence"):
        comps = [vc.rand_poly(rng, (x, y, z), max_degree, rng.choice([1, 2]), (-3, 3)) for _ in range(3)]
        if all(c != 0 for c in comps) and _div(comps) != 0:
            return comps
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# solids and their boundaries: name -> (solid, [(surface piece, point inside)], text, curved-piece index or None)


def _box(rng):
    ext = [rng.randint(1, 3) for _ in range(3)]
    lo = [rng.choice([0, 0, -1]) for _ in range(3)]
    solid = vc.Solid(sp.Matrix([u + lo[0], v + lo[1], w + lo[2]]), vc.Dom((u, v, w), tuple((0, e) for e in ext), {}))
    inside = tuple(lo[i] + ext[i] / 2 for i in range(3))
    pieces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for level in (lo[axis], lo[axis] + ext[axis]):
            comps = [None] * 3
            comps[axis] = sp.Integer(level)
            comps[others[0]], comps[others[1]] = u + lo[others[0]], v + lo[others[1]]
            dom = vc.Dom((u, v), ((0, ext[others[0]]), (0, ext[others[1]])), {})
            pieces.append(vc.Surf(sp.Matrix(comps), dom))
    desc = "the surface of the box «" + ", ".join(f"{lo[i]} ≤ {'xyz'[i]} ≤ {lo[i] + ext[i]}" for i in range(3)) + "»"
    return solid, pieces, inside, desc


def _ball(rng):
    a = rng.randint(1, 3)
    return vc.ball_solid(a), [vc.sphere_piece(a)], (0, 0, 0), f"the sphere «x² + y² + z² = {a * a}»"


def _cylinder(rng):
    a, h = rng.randint(1, 3), rng.randint(1, 3)
    return (
        vc.cylinder_solid(a, h),
        [vc.cylinder_side(a, h), vc.disc_cap(a, h), vc.disc_cap(a, 0)],
        (0, 0, h / 2),
        f"the closed surface made of the cylinder «x² + y² = {a * a}» for «0 ≤ z ≤ {h}» together with its circular top and base",
    )


def _cone(rng):
    h = rng.randint(1, 3)
    return (
        vc.cone_solid(h),
        [vc.cone_side(h), vc.disc_cap(h, h)],
        (0, 0, 0.75 * h),
        f"the closed surface made of the cone «z = √(x² + y²)» for «0 ≤ z ≤ {h}» together with the disc «x² + y² ≤ {h * h}» in the plane «z = {h}»",
    )


def _bowl(rng):
    r = rng.randint(1, 2)
    h = r * r
    return (
        vc.bowl_solid(r, h),
        [vc.bowl_side(r), vc.disc_cap(r, h)],
        (0, 0, 0.9 * h),
        f"the closed surface made of the paraboloid «z = x² + y²» for «0 ≤ z ≤ {h}» together with the disc «x² + y² ≤ {h}» in the plane «z = {h}»",
    )


def _dome(rng):
    a = rng.randint(1, 3)
    return (
        vc.ball_solid(a, (0, 4), (0, 1)),
        [vc.sphere_piece(a, (0, 4), (0, 1)), vc.disc_cap(a, 0)],
        (0, 0, a / 2),
        f"the upper hemisphere «x² + y² + z² = {a * a}, z ≥ 0» together with its flat base, the disc «x² + y² ≤ {a * a}» in the plane «z = 0»",
    )


def _both_ways(F, solid, pieces, inside, need=(1,)):
    """(the integral of div F over the solid, the outward flux through each piece). With the cross-checks on, every
    piece is parametrised and the divergence theorem is checked; otherwise only the pieces in `need` are worked out
    (the others are None) and the rest is found by subtraction."""
    by_volume = vc.solid_integral(_div(F), solid)
    if not vc.CHECKS:
        return by_volume, [vc.flux(F, p, vc.outward_from(inside)) if i in need else None for i, p in enumerate(pieces)]
    each = [vc.flux(F, piece, vc.outward_from(inside)) for piece in pieces]
    by_surface = sum(each, sp.Integer(0))
    check(sp.expand(by_volume - by_surface) == 0, f"divergence theorem fails: {by_volume} against {by_surface}")
    return by_volume, each


def _flat_at_wrong_height(F, flat_piece):
    """The flux through the flat piece with the field evaluated at the wrong height (its mirror image in height)."""
    radius = flat_piece.dom.ranges[0][1]
    level = flat_piece.r[2]
    sigma = 1 if level != 0 else -1
    wrong = 0 if level != 0 else radius
    return vc.flux(F, vc.disc_cap(radius, wrong), (0, 0, sigma), verify=False)


def _wrong_integrals(F, solid):
    """The volume integrals a learner gets from common slips with the divergence."""
    P, Q, R = F
    px, qy, rz = sp.diff(P, x), sp.diff(Q, y), sp.diff(R, z)
    out = [
        ("div-sum-of-components", P + Q + R),
        ("div-mixed-partials", sp.diff(P, y) + sp.diff(Q, z) + sp.diff(R, x)),
        ("div-dropped-term", px + qy),
        ("div-dropped-term", qy + rz),
    ]
    return [(tag, lambda g=g: vc.solid_integral(g, solid, verify=False)) for tag, g in out]


def _jacobian_slips(F, solid):
    """Spherical or cylindrical coordinates with part of the volume element left out."""
    g = _div(F)
    jac = sp.expand(solid.jacobian())
    sub = dict(zip((x, y, z), solid.r))
    integrand = sp.expand(g.xreplace(sub))
    out = []
    if vc.SP in jac.free_symbols and vc.u in jac.free_symbols:
        out.append(("div-spherical-no-rho-squared", sp.cancel(jac / vc.u**2)))
        out.append(("div-spherical-no-sin", sp.cancel(jac / vc.SP)))
    elif vc.u in jac.free_symbols:
        out.append(("div-cylindrical-no-r", sp.cancel(jac / vc.u)))
    results = []
    sign = 1 if vc._sub_mid(jac, solid.dom) > 0 else -1
    for tag, j in out:
        results.append((tag, lambda j=j: vc.integrate(sign * integrand * j, solid.dom, verify=False)))
    return results


_EASY = [
    "Find the outward flux of «F(x, y, z) = {F}» through {surface}.",
    "A fluid has velocity «F(x, y, z) = {F}». Find the net flux out of {surface}.",
    "Evaluate «∬S F · dS» over {surface}, oriented outward, for «F(x, y, z) = {F}».",
]
_HARD_CLOSED = [
    "Let S be {surface}, oriented outward. Find the flux of «F(x, y, z) = {F}» through S.",
    "Find the total outward flux of «F(x, y, z) = {F}» through {surface}.",
]
_HARD_OPEN = [
    "Let S be the curved surface of the upper hemisphere «x² + y² + z² = {a2}, z ≥ 0», with the normal pointing away from the origin. Find the flux of «F(x, y, z) = {F}» through S.",
    "A gas has velocity «F(x, y, z) = {F}». Find the flux out of the origin through the dome «x² + y² + z² = {a2}, z ≥ 0» (the curved surface only).",
]


def _draw(rng, tier):
    if tier == "easy":
        solid, pieces, inside, desc = _box(rng)
        F = _field(rng, 3)
    elif tier == "medium":
        solid, pieces, inside, desc = rng.choice([_ball, _cylinder])(rng)
        F = _field(rng, 3)
    else:
        kind = rng.choice(["cone", "bowl", "dome", "dome"])
        solid, pieces, inside, desc = {"cone": _cone, "bowl": _bowl, "dome": _dome}[kind](rng)
        F = _field(rng, 3)
    total, each = _both_ways(F, solid, pieces, inside)
    if tier == "hard" and kind == "dome":
        base = each[1]
        answer = total - base  # the curved surface only; with the cross-checks on it was also parametrised directly
        check(each[0] is None or sp.expand(each[0] - answer) == 0, "the curved surface disagrees with total minus base")
        cands = [
            ("div-open-surface", total),
            ("div-base-added", total + base),
            ("div-inward", -answer),
            ("div-flat-wrong-height", lambda: total - _flat_at_wrong_height(F, pieces[1])),
        ]
        radius = pieces[1].dom.ranges[0][1]
        template = rng.choice(_HARD_OPEN).replace("{a2}", str(radius * radius))
        text = vc.stem(template, F=vc.mvec(F))
        working = [
            "The hemisphere alone is not closed. Close it with the base disc; the total outward flux is the integral of div F, and the base flux (normal -k) is subtracted."
        ]
    else:
        answer = total
        cands = _wrong_integrals(F, solid) + _jacobian_slips(F, solid) + [("div-inward", -answer)]
        template = rng.choice(_EASY if tier != "hard" else _HARD_CLOSED).replace("{surface}", desc)
        text = vc.stem(template, F=vc.mvec(F))
        working = ["The surface is closed, so the outward flux is the integral of div F over the solid it encloses."]
    if answer == 0 or abs(answer) > 6000:
        return None
    return dict(**text, answer=answer, candidates=cands, working=working)


@generator(
    "s2a3_divergence_flux",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2a3_divergence_flux(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A sealed storage vessel has as its boundary {surface}; lengths are in metres. Gas moves with velocity «F(x, y, z) = {F}», in metres per second.",
    "A tent frame is modelled by {surface}, with lengths in metres. Air flows with velocity «F(x, y, z) = {F}» in metres per second.",
]
_A = [
    "Find the divergence «div F» of the velocity field.",
    "Find «∇ · F» for the velocity field.",
]
_B = [
    "Use the divergence theorem to find the total outward flux of F through the closed surface, in cubic metres per second.",
    "Find the total outward flux of F through the closed surface, in cubic metres per second, using the divergence theorem.",
]
_C = [
    "Find the outward flux of F through the flat {flat} alone, in cubic metres per second.",
    "Find the flux of F through the flat {flat} alone, with its outward normal, in cubic metres per second.",
]
_D = [
    "Hence find the outward flux of F through the curved surface, in cubic metres per second.",
    "Use your answers to find the outward flux through the curved surface only, in cubic metres per second.",
]


def _exam_once(rng):
    kind = rng.choice(["cone", "bowl", "dome"])
    solid, pieces, inside, desc = {"cone": _cone, "bowl": _bowl, "dome": _dome}[kind](rng)
    F = _field(rng, 3)
    total, each = _both_ways(F, solid, pieces, inside)
    flat = each[1]
    curved = total - flat
    check(each[0] is None or sp.expand(each[0] - curved) == 0, "the curved surface disagrees with total minus base")
    if 0 in (total, curved, flat) or abs(total) > 6000:
        raise vc.Redraw
    g = _div(F)
    flat_name = "base" if kind == "dome" else "lid"
    wrong_g = _wrong_integrals(F, solid)
    slips = _jacobian_slips(F, solid)
    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_A)),
            g,
            [
                ("div-sum-of-components", sp.expand(F[0] + F[1] + F[2])),
                ("div-mixed-partials", sp.expand(sp.diff(F[0], y) + sp.diff(F[1], z) + sp.diff(F[2], x))),
                ("div-dropped-term", sp.expand(sp.diff(F[0], x) + sp.diff(F[1], y))),
                ("div-dropped-term", sp.expand(sp.diff(F[1], y) + sp.diff(F[2], z))),
            ],
            2,
            ["div F = P_x + Q_y + R_z."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_B)),
            total,
            wrong_g + slips + [("div-inward", -total)],
            3,
            ["The flux is the triple integral of div F over the solid; use cylindrical or spherical coordinates."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_C).replace("{flat}", flat_name)),
            flat,
            [
                ("div-base-sign", -flat),
                ("div-flat-wrong-height", lambda: _flat_at_wrong_height(F, pieces[1])),
                ("div-total-for-flat", total),
            ],
            2,
            [
                f"On the {flat_name} the outward normal is {'+' if kind != 'dome' else '-'}k, so the flux is the integral of ±R over the disc."
            ],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_D)),
            curved,
            [
                ("div-open-surface", total),
                ("div-base-added", total + flat),
                ("div-inward", -curved),
                ("div-total-reversed", flat - total),
            ],
            2,
            ["Total = curved + flat, so curved = (b) - (c)."],
            follow_through="part_b - part_c",
        ),
    ]
    template = rng.choice(_EXAM_STEMS).replace("{surface}", desc)
    return dict(**vc.stem(template, F=vc.mvec(F)), parts=parts)


@generator("exd_x_s2a3_divergence_flux", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_divergence_flux(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
