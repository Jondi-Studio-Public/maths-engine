"""Generates `s2a3_flux_pieces` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_flux_pieces`.

``uc2-flux-through-closed-surface-pieces`` (``multivariable-and-vector-calculus``, answer ``expression``): the outward
flux of a field through the pieces of a closed surface, found by parametrising each piece.

* easy: one face of a box. The learner picks the component of the field that crosses the face, evaluates it on the
  face and fixes the sign of the outward normal (the face at the low end of an axis has normal -e).
* medium: the curved side of a cylinder x^2 + y^2 = a^2, or one of its circular ends, with the outward normal.
* hard: a piece of a sphere (a hemisphere, or an eighth), or the total outward flux through the whole closed cylinder,
  which is the sum over three pieces whose normals point three different ways.
* exam (``exd_x_s2a3_flux_pieces``): a closed cylinder: (a) the flux through the curved side, (b) through the top,
  (c) through the base, (d) the total outward flux, which is the sum of the three.

Every piece is parametrised, its normal is the cross product of the tangent vectors with the sign fixed by pointing
away from a point inside the solid, and the integral is done exactly (Wallis formulas) and by quadrature. A piece of
a sphere is checked a third way, by the divergence theorem on the solid it caps less the flat pieces that close it;
the total for a closed cylinder is checked against the divergence theorem too.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from . import _s2a3_vc as vc
from ._s2a3_vc import C, S, v, w, x, y, z

# ---------------------------------------------------------------------------------------------------------------
# fields and shapes


def _field(rng, max_degree: int, comps_terms=(1, 2), z_in_R: bool = False):
    for _ in attempts(200, "a field with three non-zero components"):
        comps = [vc.rand_poly(rng, (x, y, z), max_degree, rng.choice(comps_terms), (-3, 3)) for _ in range(3)]
        if z_in_R:  # terms whose integrals over the side and the discs survive, so no piece is 0 and top, base differ
            comps[0] = vc.rand_poly(rng, (x, y, z), max_degree, 2, (-3, 3), need=(rng.choice([x, x * z, x**3]),))
            comps[1] = vc.rand_poly(rng, (x, y, z), max_degree, 2, (-3, 3), need=(rng.choice([y, y * z, y**3]),))
            comps[2] = vc.rand_poly(
                rng,
                (x, y, z),
                max_degree,
                2,
                (-3, 3),
                need=(rng.choice([z, z**2, x**2 * z, y**2 * z]), rng.choice([x**2, y**2, sp.Integer(1)])),
            )
        if all(c != 0 for c in comps):
            return comps
    raise AssertionError("unreachable")


def _div(F):
    return sum(sp.diff(c, g) for c, g in zip(F, (x, y, z)))


def _pc(piece, F, inside):
    """Outward flux through a parametrised piece (normal pointing away from `inside`)."""
    return vc.flux(F, piece, vc.outward_from(inside))


# ---------------------------------------------------------------------------------------------------------------
# easy: a face of a box


def _easy(rng):
    ext = [rng.randint(1, 3) for _ in range(3)]
    F = _field(rng, 3)
    axis = rng.randrange(3)
    high = rng.random() < 0.5
    level = ext[axis] if high else 0
    sign = 1 if high else -1
    inside = tuple(e / 2 for e in ext)
    face = vc.box_face(axis, level, ext)
    answer = _pc(face, F, inside)
    # the same by the formula: sign * the integral of the normal component over the face
    others = [i for i in range(3) if i != axis]
    dom = face.dom

    def over_face(comp, at):
        g = F[comp].xreplace({(x, y, z)[axis]: at, (x, y, z)[others[0]]: vc.u, (x, y, z)[others[1]]: vc.v})
        return vc.integrate(g, dom, verify=False)

    check(sp.expand(sign * over_face(axis, level) - answer) == 0, "face flux disagrees with the component formula")
    other_level = 0 if high else ext[axis]
    cands = [
        ("flux-piece-inward-normal", -answer),
        ("flux-piece-opposite-face", sign * over_face(axis, other_level)),
    ]
    for j in range(3):
        if j != axis:
            cands.append(("flux-piece-wrong-component", sign * over_face(j, level)))
    names = "xyz"
    plane = f"{names[axis]} = {level}"
    box = f"«0 ≤ x ≤ {ext[0]}, 0 ≤ y ≤ {ext[1]}, 0 ≤ z ≤ {ext[2]}»"
    template = rng.choice(_EASY).replace("{box}", box)
    text = vc.stem(template, F=vc.mvec(F), plane=plane)
    working = [
        f"On the face {plane} the outward normal is {'+' if high else '-'}{'ijk'[axis]}, so the flux is "
        f"{'+' if high else '-'} the integral of {'PQR'[axis]}({plane}) over the face."
    ]
    return dict(**text, answer=answer, candidates=cands, working=working)


_EASY = [
    "Let S be the face of the box {box} that lies in the plane «{plane}». Find the flux of «F(x, y, z) = {F}» out of the box through S.",
    "The box {box} is in a flow with velocity «F(x, y, z) = {F}». Find the outward flux through the face of the box on the plane «{plane}».",
    "Find the outward flux of «F(x, y, z) = {F}» through the face «{plane}» of the box {box}.",
]

# ---------------------------------------------------------------------------------------------------------------
# medium: the side or an end of a cylinder


def _medium(rng):
    a, h = rng.randint(1, 3), rng.randint(1, 3)
    F = _field(rng, 2)
    inside = (0, 0, h / 2)
    if rng.random() < 0.5:
        side = vc.cylinder_side(a, h)
        answer = _pc(side, F, inside)
        cands = [
            ("flux-piece-inward-normal", -answer),
            ("flux-piece-forgot-radius-factor", answer / a),
            ("flux-piece-unnormalised-normal", answer * a),
            (
                "flux-piece-wrong-component",
                vc.integrate(F[0].xreplace({x: a * C, y: a * S, z: w}) * C * a, side.dom, False),
            ),
        ]
        template = rng.choice(_SIDE)
        text = vc.stem(template, F=vc.mvec(F), a2=a * a, h=h)
        working = [
            "Parametrise by (a cos θ, a sin θ, z). The outward normal is a(cos θ, sin θ, 0) dθ dz, so only P and Q cross the side."
        ]
        return dict(**text, answer=answer, candidates=cands, working=working)
    top = rng.random() < 0.5
    level = h if top else 0
    cap = vc.disc_cap(a, level)
    answer = _pc(cap, F, inside)
    sign = 1 if top else -1
    R_here = F[2].xreplace({z: level})
    plain = vc.integrate(R_here.xreplace({x: vc.u * C, y: vc.u * S}) * vc.u, cap.dom, False)
    check(sp.expand(sign * plain - answer) == 0, "cap flux disagrees with the formula")
    cands = [
        ("flux-piece-inward-normal", -answer),
        ("flux-piece-forgot-polar-r", sign * vc.integrate(R_here.xreplace({x: vc.u * C, y: vc.u * S}), cap.dom, False)),
        (
            "flux-piece-cap-at-wrong-height",
            sign * vc.integrate(F[2].xreplace({z: h - level, x: vc.u * C, y: vc.u * S}) * vc.u, cap.dom, False),
        ),
        (
            "flux-piece-wrong-component",
            sign * vc.integrate(F[0].xreplace({z: level, x: vc.u * C, y: vc.u * S}) * vc.u, cap.dom, False),
        ),
    ]
    where = "top" if top else "base"
    template = rng.choice(_CAP)
    text = vc.stem(template, F=vc.mvec(F), a2=a * a, h=h, where=where, level=level)
    working = [
        f"On the {where} the outward normal is {'+' if top else '-'}k, so the flux is {'+' if top else '-'} the integral of R(x, y, {level}) over the disc, in polar coordinates."
    ]
    return dict(**text, answer=answer, candidates=cands, working=working)


_SIDE = [
    "Let S be the curved side of the cylinder «x² + y² = {a2}, 0 ≤ z ≤ {h}», oriented away from the axis. Find the flux of «F(x, y, z) = {F}» through S.",
    "Water moves with velocity «F(x, y, z) = {F}». Find the flux out through the curved side of the cylinder «x² + y² = {a2}, 0 ≤ z ≤ {h}».",
    "Evaluate «∬S F · dS» for «F(x, y, z) = {F}», where S is the side of the cylinder «x² + y² = {a2}», «0 ≤ z ≤ {h}», with the normal pointing away from the axis.",
]
_CAP = [
    "A closed cylinder is «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}». Find the outward flux of «F(x, y, z) = {F}» through its {where}, the disc in the plane «z = {level}».",
    "Let S be the {where} of the solid cylinder «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}», with the outward normal. Evaluate «∬S F · dS» for «F(x, y, z) = {F}».",
]

# ---------------------------------------------------------------------------------------------------------------
# hard: a piece of a sphere, or the whole closed cylinder

_PIECES = {
    "upper hemisphere": ((0, 4), (0, 1), "the upper hemisphere «x² + y² + z² = {a2}, z ≥ 0»"),
    "front hemisphere": ((3, 5), (0, 2), "the hemisphere «x² + y² + z² = {a2}, x ≥ 0»"),
    "octant": ((0, 1), (0, 1), "the part of the sphere «x² + y² + z² = {a2}» in the first octant «x, y, z ≥ 0»"),
}


def _sphere_cands(F, piece, answer, a):
    sign = vc.orient(piece, vc.outward_from((0, 0, 0)))
    on_surface = [c.xreplace(dict(zip((x, y, z), piece.r))) for c in F]
    unit_normal = piece.normal().applyfunc(lambda e: sp.cancel(e / vc.SP))  # leaves out the factor sin(phi)
    no_sin = vc.integrate(sign * sum(f * n for f, n in zip(on_surface, unit_normal)), piece.dom, False)
    return [
        ("flux-piece-inward-normal", -answer),
        ("flux-piece-sphere-no-sin", no_sin),
        ("flux-piece-sphere-radius-power", answer / a),
        ("flux-piece-sphere-radius-power", answer * a),
    ]


def _closing_discs(name, a):
    """The flat pieces that close a sphere piece, each with the direction pointing out of the closed solid."""
    disc = lambda comps, q, ref: (vc.Surf(sp.Matrix(comps), vc.Dom((vc.u, v), ((0, a), q), {v: (C, S)})), ref)  # noqa: E731
    uu = vc.u
    if name == "upper hemisphere":
        return [disc([uu * C, uu * S, 0], (0, 4), (0, 0, -1))]
    if name == "front hemisphere":
        return [disc([0, uu * C, uu * S], (0, 4), (-1, 0, 0))]
    return [
        disc([uu * C, uu * S, 0], (0, 1), (0, 0, -1)),
        disc([uu * C, 0, uu * S], (0, 1), (0, -1, 0)),
        disc([0, uu * C, uu * S], (0, 1), (-1, 0, 0)),
    ]


def _sphere_piece_flux(F, name, a):
    """The outward flux through a sphere piece, by parametrising it and, as a check, by the divergence theorem on the
    solid it caps, taking off the flat pieces that close it."""
    theta_q, phi_q, _ = _PIECES[name]
    piece = vc.sphere_piece(a, theta_q, phi_q)
    answer = _pc(piece, F, (0, 0, 0))
    solid = vc.ball_solid(a, theta_q, phi_q)
    closed = vc.solid_integral(_div(F), solid) - sum(
        (vc.flux(F, d, ref) for d, ref in _closing_discs(name, a)), sp.Integer(0)
    )
    check(sp.expand(answer - closed) == 0, f"sphere piece {name}: surface {answer} but divergence theorem {closed}")
    return piece, answer


def _hard(rng):
    a, h = rng.randint(1, 3), rng.randint(1, 3)
    if rng.random() < 0.5:
        F = _field(rng, 2)
        name = rng.choice(list(_PIECES))
        desc = _PIECES[name][2]
        piece, answer = _sphere_piece_flux(F, name, a)
        cands = _sphere_cands(F, piece, answer, a)
        template = rng.choice(_SPHERE).replace("{piece}", desc)
        text = vc.stem(template, F=vc.mvec(F), a2=a * a)
        working = [
            "Parametrise by (a sin φ cos θ, a sin φ sin θ, a cos φ); the outward normal is a² sin φ times the unit radial vector."
        ]
        return dict(**text, answer=answer, candidates=cands, working=working)
    F = _field(rng, 2, z_in_R=True)
    pieces = _cylinder_pieces(F, a, h)
    answer = sum(pieces.values(), sp.Integer(0))
    if pieces["top"] + pieces["base"] == 0:
        return None
    check(
        sp.expand(answer - vc.solid_integral(_div(F), vc.cylinder_solid(a, h))) == 0,
        "the sum over the pieces disagrees with the divergence theorem",
    )
    cands = [
        ("flux-piece-total-missing-end", answer - pieces["top"]),
        ("flux-piece-total-missing-end", answer - pieces["base"]),
        ("flux-piece-base-sign", answer + 2 * pieces["base"]),
        ("flux-piece-total-missing-side", pieces["top"] + pieces["base"]),
        ("flux-piece-inward-normal", -answer),
    ]
    text = vc.stem(rng.choice(_CLOSED), F=vc.mvec(F), a2=a * a, h=h)
    working = ["Add the outward fluxes through the side (normal away from the axis), the top (+k) and the base (-k)."]
    return dict(**text, answer=answer, candidates=cands, working=working)


def _cylinder_pieces(F, a, h):
    inside = (0, 0, h / 2)
    return {
        "side": _pc(vc.cylinder_side(a, h), F, inside),
        "top": _pc(vc.disc_cap(a, h), F, inside),
        "base": _pc(vc.disc_cap(a, 0), F, inside),
    }


_SPHERE = [
    "Let S be {piece}, oriented by the normal pointing away from the origin. Find the flux of «F(x, y, z) = {F}» through S.",
    "A gas has velocity «F(x, y, z) = {F}». Find the flux out of the origin through {piece}.",
]
_CLOSED = [
    "A closed surface S is the boundary of the solid cylinder «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}»: the curved side together with the top and the base. Find the total outward flux of «F(x, y, z) = {F}» through S.",
    "Find the net flux of «F(x, y, z) = {F}» out of the solid cylinder «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}», through its side, top and base together.",
]


@generator(
    "s2a3_flux_pieces",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a3_flux_pieces(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    draw = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]

    def checked(rng_, tier_):
        d = draw(rng_)
        return d if d is not None and d["answer"] != 0 and abs(d["answer"]) < 5000 else None

    return vc.single_question(rng, checked, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A closed tank is the cylinder «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}», with lengths in metres. Water flows with velocity «F(x, y, z) = {F}», in metres per second.",
    "A grain silo is modelled by the closed cylinder «x² + y² ≤ {a2}, 0 ≤ z ≤ {h}» (lengths in metres). Air moves with velocity «F(x, y, z) = {F}» in metres per second.",
]
_PARTS = [
    "Find the outward flux of F through the curved side of the tank, in cubic metres per second.",
    "Find the outward flux of F through the circular top, in cubic metres per second.",
    "Find the outward flux of F through the circular base, in cubic metres per second.",
    "Hence find the total outward flux of F through the whole surface of the tank, in cubic metres per second.",
]


def _exam_once(rng):
    a, h = rng.randint(1, 3), rng.randint(1, 3)
    F = _field(rng, 2, z_in_R=True)
    pieces = _cylinder_pieces(F, a, h)
    side, top, base = pieces["side"], pieces["top"], pieces["base"]
    total = side + top + base
    check(
        sp.expand(total - vc.solid_integral(_div(F), vc.cylinder_solid(a, h))) == 0,
        "the three fluxes do not add to the divergence integral",
    )
    if 0 in (side, top, base, total) or top + base == 0 or abs(total) > 4000:
        raise vc.Redraw
    side_surf, cap = vc.cylinder_side(a, h), vc.disc_cap(a, h)
    sub_side = {x: a * C, y: a * S, z: w}
    parts = [
        vc.part_dict(
            vc.prompt(_PARTS[0]),
            side,
            [
                ("flux-piece-inward-normal", -side),
                ("flux-piece-forgot-radius-factor", side / a),
                ("flux-piece-unnormalised-normal", side * a),
                ("flux-piece-wrong-component", vc.integrate(F[0].xreplace(sub_side) * C * a, side_surf.dom, False)),
            ],
            3,
            ["Parametrise by (a cos θ, a sin θ, z); the outward normal is a(cos θ, sin θ, 0) dθ dz."],
        ),
        vc.part_dict(
            vc.prompt(_PARTS[1]),
            top,
            [
                ("flux-piece-inward-normal", -top),
                (
                    "flux-piece-forgot-polar-r",
                    vc.integrate(F[2].xreplace({z: h, x: vc.u * C, y: vc.u * S}), cap.dom, False),
                ),
                (
                    "flux-piece-cap-at-wrong-height",
                    vc.integrate(F[2].xreplace({z: 0, x: vc.u * C, y: vc.u * S}) * vc.u, cap.dom, False),
                ),
            ],
            2,
            [f"On the top z = {h} the normal is +k, so the flux is the integral of R(x, y, {h}) over the disc."],
        ),
        vc.part_dict(
            vc.prompt(_PARTS[2]),
            base,
            [
                ("flux-piece-base-sign", -base),
                (
                    "flux-piece-forgot-polar-r",
                    -vc.integrate(F[2].xreplace({z: 0, x: vc.u * C, y: vc.u * S}), cap.dom, False),
                ),
                (
                    "flux-piece-cap-at-wrong-height",
                    -vc.integrate(F[2].xreplace({z: h, x: vc.u * C, y: vc.u * S}) * vc.u, cap.dom, False),
                ),
            ],
            2,
            [
                "On the base z = 0 the outward normal is -k, so the flux is minus the integral of R(x, y, 0) over the disc."
            ],
        ),
        vc.part_dict(
            vc.prompt(_PARTS[3]),
            total,
            [
                ("flux-piece-total-missing-end", side + top),
                ("flux-piece-total-missing-end", side + base),
                ("flux-piece-base-sign", side + top - base),
                ("flux-piece-total-missing-side", top + base),
            ],
            2,
            ["Add the three outward fluxes."],
            follow_through="part_a + part_b + part_c",
        ),
    ]
    return dict(**vc.stem(rng.choice(_EXAM_STEMS), F=vc.mvec(F), a2=a * a, h=h), parts=parts)


@generator("exd_x_s2a3_flux_pieces", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_flux_pieces(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
