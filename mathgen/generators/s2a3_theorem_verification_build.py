"""Generates `s2a3_theorem_verification` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_theorem_verification`.

``uc2-integral-theorem-verification`` (``multivariable-and-vector-calculus``, answer ``expression``): checking an
integral theorem on an example by working out both of its sides. A question gives a field and a curve or surface and
asks for the number on which the two sides agree; the learner must be able to do each side and trust the theorem.

* easy: Green's theorem for a right triangle: the line integral along the three sides against the double integral.
* medium: the divergence theorem for a sphere or a closed cylinder, or Stokes' theorem for a triangle in space.
* hard: Green's theorem for a disc, half-disc or parabola region, the divergence theorem for a cone or bowl with its lid,
  or Stokes' theorem for the curve in which a cylinder meets a plane.
* exam (``exd_x_s2a3_theorem_verification``): a triangle (Green), a triangle in space (Stokes) or a closed cylinder
  (divergence), with the boundary in three pieces: (a) the value of the theorem side, (b) and (c) the direct integrals
  over two of the pieces, (d) the third piece, which follows through as (a) - (b) - (c): the value it must have if the
  theorem is to hold. The generator works out the third piece directly as well and checks that it is that value.

Every value is exact, and each side is computed independently (the area, volume or surface integral of the theorem's
integrand, and the direct integral over each piece of the boundary). The generator will not build a question unless
the two sides agree.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from . import _s2a3_vc as vc
from . import s2a3_divergence_flux_build as dv
from . import s2a3_green_circulation_build as gc
from . import s2a3_stokes_circulation_build as st
from ._s2a3_vc import x, y, z

# ---------------------------------------------------------------------------------------------------------------
# scenarios. Each is a dict: kind, F (M text), total (the common value), g (the integrand of the theorem side),
# pieces (direct values of the pieces of the boundary), texts (how each piece is named), desc (the curve or surface),
# wrong_total ((tag, value) slips of the theorem side), working.


def _later(fn, *args):
    """A piece's direct integral, worked out when first asked for."""
    cache = []

    def value():
        if not cache:
            cache.append(fn(*args))
        return cache[0]

    return value


def _agree(scenario):
    """With the cross-checks on, the direct integrals over the pieces must add up to the theorem side."""
    if vc.CHECKS:
        direct = sum((p() for p in scenario["pieces"]), sp.Integer(0))
        check(
            sp.expand(direct - scenario["total"]) == 0, f"the pieces give {direct} but the theorem {scenario['total']}"
        )
    return scenario


def _pt(p) -> str:
    return "(" + ", ".join(str(c) for c in p) + ")"


def _green_triangle(rng):
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    if rng.random() < 0.5:
        pts, region = [(0, 0), (a, 0), (0, b)], vc.tri_region(a, b, "corner")
    else:
        pts, region = [(0, 0), (a, 0), (a, b)], vc.tri_region(a, b, "under")
    loop = pts + [pts[0]]
    shape = dict(region=region, pieces=[vc.segment(p, q) for p, q in zip(loop, loop[1:])])
    P, Q = gc._field(rng, 2)
    total, curl, _ = gc._circulation(P, Q, shape)
    texts = [f"the edge from «{_pt(p)}» to «{_pt(q)}»" for p, q in zip(loop, loop[1:])]
    return dict(
        kind="green",
        F=vc.mvec([P, Q], "ij"),
        total=total,
        g=curl,
        pieces=[_later(vc.line, [P, Q], piece) for piece in shape["pieces"]],
        texts=texts,
        desc=f"the triangle with vertices «{', '.join(_pt(p) for p in pts)}»",
        wrong_total=gc._candidates(P, Q, dict(shape, curve="", pieces=shape["pieces"][:1]), total, 1, curl)[1:5],
        working=["Green: the counterclockwise circulation is the integral of Q_x - P_y over the triangle."],
    )


def _green_hard(rng):
    P, Q = gc._field(rng, 3)
    shape = gc._shape(rng, "hard")
    total, curl, _ = gc._circulation(P, Q, shape)
    return dict(
        kind="green",
        F=vc.mvec([P, Q], "ij"),
        total=total,
        g=curl,
        pieces=[_later(vc.line, [P, Q], piece) for piece in shape["pieces"]],
        texts=[],
        desc=shape["curve"],
        wrong_total=gc._candidates(P, Q, shape, total, 1, curl),
        working=["Green: the counterclockwise circulation is the integral of Q_x - P_y over the region inside C."],
    )


def _div(rng, builder, texts=()):
    solid, pieces, inside, desc = builder(rng)
    F = dv._field(rng, 3)
    total = vc.solid_integral(dv._div(F), solid)
    return dict(
        kind="divergence",
        F=vc.mvec(F),
        total=total,
        g=dv._div(F),
        pieces=[_later(vc.flux, F, piece, vc.outward_from(inside)) for piece in pieces],
        texts=list(texts),
        desc=desc,
        wrong_total=dv._wrong_integrals(F, solid) + dv._jacobian_slips(F, solid) + [("div-inward", -total)],
        working=["The surface is closed, so the outward flux is the integral of div F over the solid it encloses."],
    )


def _div_cylinder(rng):
    return _div(
        rng,
        dv._cylinder,
        ["the circular top, with its outward normal", "the circular base, with its outward normal", "the curved side"],
    )


def _stokes_triangle(rng):
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    d, p, q = rng.randint(1, 4), rng.choice([-1, 1, 2]), rng.choice([-1, 1, 2])
    f = d - p * x - q * y
    pts = [(0, 0, d), (a, 0, d - p * a), (0, b, d - q * b)]
    loop = [vc.segment(pts[i], pts[(i + 1) % 3]) for i in range(3)]
    F = st._field(rng, 2)
    region = vc.tri_region(a, b, "corner")
    total, g, curl = st._stokes(F, f, region, loop)
    shape = dict(f=f, region=region, polar=False)
    return dict(
        kind="stokes",
        F=vc.mvec(F),
        total=total,
        g=g,
        pieces=[_later(vc.line, F, c) for c in loop],
        texts=[f"the edge from «{_pt(pts[i])}» to «{_pt(pts[(i + 1) % 3])}»" for i in range(3)],
        desc=f"the triangle with vertices «{', '.join(_pt(p) for p in pts)}»",
        wrong_total=st._candidates(F, shape, 1, total, g)[1:],
        working=["Stokes: the circulation is the flux of curl F through the triangle, upward normal."],
    )


def _stokes_curve(rng, tier):
    F = st._field(rng, 2 if tier == "medium" else 3)
    shape = st._shape(rng, tier)
    total, g, curl = st._stokes(F, shape["f"], shape["region"], shape["boundary"])
    return dict(
        kind="stokes",
        F=vc.mvec(F),
        total=total,
        g=g,
        pieces=[_later(vc.line, F, c) for c in shape["boundary"]],
        texts=[],
        desc=shape["curve"],
        wrong_total=st._candidates(F, shape, 1, total, g)[1:],
        working=["Stokes: the circulation is the flux of curl F through a surface that C bounds."],
    )


# ---------------------------------------------------------------------------------------------------------------
# single-answer questions

_STEMS = {
    "green": [
        "Verify Green's theorem for «F(x, y) = {F}» and the boundary C of {desc}, traversed counterclockwise: work out the line integral «∮C F · dr» and the double integral of «Q_x - P_y» over the region inside, and give the value on which they agree.",
        "A particle goes once round {desc}, counterclockwise, under the force «F(x, y) = {F}». Check Green's theorem by finding the work both as a line integral and as a double integral, and state the common value.",
    ],
    "divergence": [
        "Verify the divergence theorem for «F(x, y, z) = {F}» and {desc}, outward: work out the flux as a surface integral and the integral of «div F» over the solid, and give the value on which they agree.",
        "Check the divergence theorem for the velocity field «F(x, y, z) = {F}» and {desc}: find the outward flux through the surface and the triple integral of «∇ · F», and state the common value.",
    ],
    "stokes": [
        "Verify Stokes' theorem for «F(x, y, z) = {F}» and C, {desc}, counterclockwise as seen from above: work out «∮C F · dr» and the flux of «curl F» through a surface that C bounds, and give the value on which they agree.",
        "Check Stokes' theorem for «F(x, y, z) = {F}» round {desc}, counterclockwise as seen from above: find the circulation as a line integral and as a surface integral, and state the common value.",
    ],
}


def _candidates(s):
    total, pieces = s["total"], s["pieces"]
    out = [("verify-wrong-direction", -total)]
    if len(pieces) > 1:
        out.append(("verify-piece-missing", lambda: total - pieces[-1]()))
        out.append(("verify-piece-reversed", lambda: total - 2 * pieces[0]()))
    out += s["wrong_total"]
    if len(pieces) > 1:
        out += [("verify-piece-missing", lambda p=p: total - p()) for p in pieces]
        out += [("verify-piece-reversed", lambda p=p: total - 2 * p()) for p in pieces]
    out.append(("verify-sides-added", 2 * total))
    return out


def _draw(rng, tier):
    if tier == "easy":
        s = _green_triangle(rng)
    elif tier == "medium":
        s = rng.choice(
            [_div_cylinder, lambda r: _div(r, dv._ball), _stokes_triangle, lambda r: _stokes_curve(r, "medium")]
        )(rng)
    else:
        s = rng.choice(
            [_green_hard, lambda r: _div(r, rng.choice([dv._cone, dv._bowl])), lambda r: _stokes_curve(r, "hard")]
        )(rng)
    total = s["total"]
    if total == 0 or abs(total) > 4000:
        return None
    _agree(s)
    template = rng.choice(_STEMS[s["kind"]]).replace("{desc}", s["desc"])
    text = vc.stem(template, F=s["F"])
    return dict(**text, answer=total, candidates=_candidates(s), working=s["working"])


@generator(
    "s2a3_theorem_verification",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a3_theorem_verification(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = {
    "green": "A particle goes once round {desc}, counterclockwise, under the force «F(x, y) = {F}», in newtons, with lengths in metres. You will check Green's theorem for this loop.",
    "divergence": "A sealed vessel has as its boundary {desc}; lengths are in metres. Gas moves with velocity «F(x, y, z) = {F}», in metres per second. You will check the divergence theorem for this vessel.",
    "stokes": "A cable runs round {desc}, with lengths in metres, and a cart is pulled round it counterclockwise as seen from above under the force «F(x, y, z) = {F}», in newtons. You will check Stokes' theorem for this loop.",
}
_THEOREM_SIDE = {
    "green": [
        "Use Green's theorem to find the work in one lap, in joules, as the double integral of «Q_x - P_y» over the triangle.",
        "Find the work in one lap, in joules, using Green's theorem.",
    ],
    "divergence": [
        "Use the divergence theorem to find the total outward flux, in cubic metres per second.",
        "Find the total outward flux in cubic metres per second, using the divergence theorem.",
    ],
    "stokes": [
        "Use Stokes' theorem to find the work in one lap, in joules, as the flux of «curl F» through the triangle with the upward normal.",
        "Find the work in one lap, in joules, using Stokes' theorem.",
    ],
}
_PIECE = {
    "green": [
        "Now find the work done along {piece}, directly as a line integral, in joules.",
        "Find the line integral of F along {piece}, in joules, directly from a parametrisation.",
    ],
    "divergence": [
        "Now find the outward flux through {piece}, directly as a surface integral, in cubic metres per second.",
        "Find the flux of F through {piece}, directly from a parametrisation, in cubic metres per second.",
    ],
}
_PIECE["stokes"] = _PIECE["green"]
_LAST = {
    "green": "Green's theorem says the work along the three edges adds up to your answer to (a). Hence find the work along {piece}, in joules.",
    "divergence": "The divergence theorem says the fluxes through the three pieces add up to your answer to (a). Hence find the outward flux through {piece}, in cubic metres per second.",
    "stokes": "Stokes' theorem says the work along the three edges adds up to your answer to (a). Hence find the work along {piece}, in joules.",
}


def _exam_once(rng):
    kind = rng.choice(["green", "divergence", "stokes"])
    s = {"green": _green_triangle, "divergence": _div_cylinder, "stokes": _stokes_triangle}[kind](rng)
    total, pieces, texts = s["total"], s["pieces"], s["texts"]
    if kind == "divergence":
        pieces = [pieces[1], pieces[2], pieces[0]]  # top, base, side
    check(len(pieces) == 3 and len(texts) == 3, "the exam needs three pieces")
    if abs(total) > 4000:
        raise vc.Redraw
    p1, p2 = pieces[0](), pieces[1]()
    p3 = total - p1 - p2  # the third piece is what the theorem forces; the cross-checks work it out directly too
    check(not vc.CHECKS or sp.expand(pieces[2]() - p3) == 0, "the three pieces do not add up to the theorem side")
    if 0 in (total, p1, p2, p3) or abs(total) > 4000:
        raise vc.Redraw

    def prompt_piece(t, i):
        return vc.prompt(rng.choice(_PIECE[kind]).replace("{piece}", t[i]))

    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_THEOREM_SIDE[kind])),
            total,
            [(tag, val) for tag, val in s["wrong_total"]] + [("verify-wrong-direction", -total)],
            3,
            s["working"],
        ),
        vc.part_dict(
            prompt_piece(texts, 0),
            p1,
            [
                ("verify-piece-reversed", -p1),
                ("verify-piece-is-whole", total),
                ("verify-piece-share", total / 3),
                ("verify-piece-swapped", p2),
            ],
            2,
            ["Parametrise the piece, substitute into the field and integrate."],
        ),
        vc.part_dict(
            prompt_piece(texts, 1),
            p2,
            [
                ("verify-piece-reversed", -p2),
                ("verify-piece-is-whole", total),
                ("verify-piece-share", total / 3),
                ("verify-piece-swapped", p1),
            ],
            2,
            ["Parametrise the piece, substitute into the field and integrate."],
        ),
        vc.part_dict(
            vc.prompt(_LAST[kind].replace("{piece}", texts[2])),
            p3,
            [
                ("verify-one-piece-subtracted", total - p1),
                ("verify-one-piece-subtracted", total - p2),
                ("verify-pieces-for-remaining", p1 + p2),
                ("verify-remaining-sign", p1 + p2 - total),
                ("verify-total-added", total + p1 + p2),
            ],
            2,
            ["The three pieces add up to the theorem side: the third is (a) - (b) - (c)."],
            follow_through="part_a - part_b - part_c",
        ),
    ]
    template = _EXAM_STEMS[kind].replace("{desc}", s["desc"])
    return dict(**vc.stem(template, F=s["F"]), parts=parts)


@generator("exd_x_s2a3_theorem_verification", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_theorem_verification(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
