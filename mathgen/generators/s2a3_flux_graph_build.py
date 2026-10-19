"""Generates `s2a3_flux_graph` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_flux_graph`.

``uc2-flux-through-graph-surface`` (``multivariable-and-vector-calculus``, answer ``expression``): the flux of a
vector field through a surface z = f(x, y), with the direction of the normal stated.

* easy: a plane z = p x + q y + c over a rectangle, upward normal. The field depends on z, so z must be replaced by
  the plane's height before anything is integrated.
* medium: a curved graph (z = c + a x^2 + b y^2, a saddle, a ridge) over a rectangle or a right triangle; the flux is
  sometimes taken upward and sometimes downward.
* hard: the same kind of graph over a disc, half-disc or quarter-disc, so the integral is done in polar coordinates;
  upward or downward normal; the answer carries pi.
* exam (``exd_x_s2a3_flux_graph``): fluid crossing a membrane z = f(x, y): (a) the integrand g(x, y) the flux reduces
  to, (b) the flux up through the membrane, (c) the total outward flux from the solid beneath it, by the divergence theorem, (d) the flux through the vertical walls of that solid, from (b), (c) and a stated base flux.

Each answer is worked out twice. The first route reduces the flux to the double integral of -P f_x - Q f_y + R over
the base region. The second parametrises the surface, takes the cross product of the tangent vectors with SymPy and
integrates that, exactly and by quadrature. For part (d) the walls are also integrated face by face (rectangle) or on the cylinder (disc), and compared with the
divergence-theorem value.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from ._s2_options import keep_answer_shape
from . import _s2a3_vc as vc
from ._s2a3_vc import u, w, x, y, z

# ---------------------------------------------------------------------------------------------------------------
# the field, the graph and the base region


def _field(rng, depth: int):
    """P, Q, R of total degree at most 2 (3 terms in R at most) that really depend on z."""
    for _ in attempts(200, "a field that depends on z"):
        comps = [vc.rand_poly(rng, (x, y, z), 2, rng.choice([1, 2, 2]), (-3, 3)) for _ in range(3)]
        if all(c != 0 for c in comps) and any(c.has(z) for c in comps):
            return comps
    raise AssertionError("unreachable")


def _graph(rng, kind: str):
    """f(x, y) with integer coefficients. kind: plane, curved or bowl."""
    c0 = rng.randint(1, 5)
    if kind == "plane":
        p, q = rng.choice([-2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2, 3])
        return c0 + p * x + q * y
    if kind == "curved":
        shapes = [
            lambda a, b: c0 + a * x**2 + b * y**2,
            lambda a, b: c0 + a * x * y + b * x,
            lambda a, b: c0 + a * x**2 + b * y,
            lambda a, b: c0 + a * y**2 + b * x * y,
        ]
        return rng.choice(shapes)(rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2]))
    shapes = [
        lambda a, b: c0 + a * x**2 + b * y**2,
        lambda a, b: c0 + a * x**2 + b * y**2 + rng.choice([1, 2]) * x,
        lambda a, b: c0 + a * x**2 + b * y**2 + rng.choice([-2, -1, 1, 2]) * x * y,
    ]
    return rng.choice(shapes)(rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2]))


def _region(rng, kind: str):
    """(region, description). kind: rect, tri or disc."""
    if kind == "rect":
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        return vc.rect_region(0, a, 0, b), f"the rectangle «0 ≤ x ≤ {a}, 0 ≤ y ≤ {b}»"
    if kind == "tri":
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        if rng.random() < 0.5:
            return vc.tri_region(a, b, "corner"), f"the triangle with vertices «(0, 0), ({a}, 0), (0, {b})»"
        return vc.tri_region(a, b, "under"), f"the triangle with vertices «(0, 0), ({a}, 0), ({a}, {b})»"
    radius = rng.randint(1, 3)
    shape = rng.choice(["full", "upper", "right", "quarter"])
    spec = {
        "full": ((0, 4), f"the disc «x² + y² ≤ {radius**2}»"),
        "upper": ((0, 2), f"the half-disc «x² + y² ≤ {radius**2}, y ≥ 0»"),
        "right": ((3, 5), f"the half-disc «x² + y² ≤ {radius**2}, x ≥ 0»"),
        "quarter": ((0, 1), f"the quarter-disc «x² + y² ≤ {radius**2}, x ≥ 0, y ≥ 0»"),
    }[shape]
    return vc.polar_region(radius, *spec[0]), spec[1]


# ---------------------------------------------------------------------------------------------------------------
# the flux, by the formula and by parametrisation


def _integrand(F, f, variant: str = "right"):
    """The function of (x, y) whose integral over the base region is the upward flux."""
    P, Q, R = F
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    sub = {z: f}
    Ps, Qs, Rs = (c.xreplace(sub) for c in (P, Q, R))
    if variant == "right":
        return sp.expand(-Ps * fx - Qs * fy + Rs)
    if variant == "signs":
        return sp.expand(Ps * fx + Qs * fy + Rs)
    if variant == "z-not-replaced":
        return sp.expand(-P.subs(z, 0) * fx - Q.subs(z, 0) * fy + R.subs(z, 0))
    if variant == "no-R":
        return sp.expand(-Ps * fx - Qs * fy)
    if variant == "swapped":
        return sp.expand(-Ps * fy - Qs * fx + Rs)
    raise ValueError(variant)


def _upward_flux(F, f, region):
    g = _integrand(F, f)
    value = vc.region_integral(g, region, "flux as a double integral")
    if vc.CHECKS:
        second = vc.flux(F, vc.graph_over(region, f), (0, 0, 1), "flux by parametrisation")
        check(sp.simplify(value - second) == 0, f"the two routes to the flux disagree: {value} vs {second}")
    return value, g


def _candidates(F, f, region, value, sign, polar: bool):
    out = []
    for tag, variant in (
        ("flux-graph-normal-signs", "signs"),
        ("flux-graph-z-not-replaced", "z-not-replaced"),
        ("flux-graph-dropped-r", "no-R"),
        ("flux-graph-components-swapped", "swapped"),
    ):
        out.append(
            (tag, lambda variant=variant: sign * vc.region_integral(_integrand(F, f, variant), region, verify=False))
        )
    out.append(("flux-graph-wrong-direction", -sign * value))
    if polar:  # the Jacobian r of polar coordinates left out
        g = _integrand(F, f).subs({x: region.r[0], y: region.r[1]}, simultaneous=True)
        out.append(("flux-graph-polar-no-r", lambda: sign * vc.integrate(g, region.dom, verify=False)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# single questions

_EASY = [
    "Let S be the part of the plane «z = {f}» above {region}, with the normal pointing upward. Find the flux of «F(x, y, z) = {F}» through S.",
    "The surface S is the graph «z = {f}» over {region}, oriented by its upward normal. Evaluate «∬S F · dS» for «F(x, y, z) = {F}».",
    "Find the flux of the field «F(x, y, z) = {F}» upward through the plane section «z = {f}» that lies over {region}.",
]
_MEDIUM = [
    "Let S be the surface «z = {f}» over {region}, with the normal pointing {way}. Find the flux of «F(x, y, z) = {F}» through S.",
    "S is the graph of «z = {f}» above {region}, oriented with the {way} normal. Evaluate «∬S F · dS» for «F(x, y, z) = {F}».",
    "A fluid has velocity «F(x, y, z) = {F}». Find the flux through the surface «z = {f}» above {region}, taken with the {way} normal.",
]
_HARD = _MEDIUM


def _draw(rng, tier):
    F = _field(rng, 0)
    if tier == "easy":
        f, (region, region_text), way = _graph(rng, "plane"), _region(rng, "rect"), "upward"
    elif tier == "medium":
        f = _graph(rng, "curved")
        (region, region_text) = _region(rng, rng.choice(["rect", "tri"]))
        way = rng.choice(["upward", "upward", "downward"])
    else:
        f = _graph(rng, "bowl")
        (region, region_text) = _region(rng, "disc")
        way = rng.choice(["upward", "downward"])
    sign = 1 if way == "upward" else -1
    value, g = _upward_flux(F, f, region)
    answer = sign * value
    if answer == 0 or abs(answer) > 4000:
        return None
    Ftxt = vc.mvec(F)
    template = rng.choice({"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}[tier])
    template = template.replace("{region}", region_text)
    text = vc.stem(template, f=vc.poly_text(f, (x, y)), F=Ftxt, way=way)
    working = [
        "With z = f(x, y), the upward normal is (-f_x, -f_y, 1) dA, so the flux is the integral of -P f_x - Q f_y + R "
        "over the base region, with z replaced by f.",
        f"That integrand is {vc.poly_text(g, (x, y))}"
        + ("; take the sign for a downward normal." if sign < 0 else "."),
    ]
    return dict(
        **text,
        answer=answer,
        candidates=_candidates(F, f, region, value, sign, tier == "hard"),
        working=working,
    )


@generator(
    "s2a3_flux_graph",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2a3_flux_graph(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "Water flows through a thin membrane stretched over {region}. The membrane is the graph «z = {f}», and the water has velocity «F(x, y, z) = {F}», in metres per second, with lengths in metres.",
    "A sail is modelled by the surface «z = {f}» over {region}. The wind has velocity «F(x, y, z) = {F}» (metres per second, lengths in metres).",
    "Cooling air passes through a vent panel, the graph «z = {f}» over {region}, with air velocity «F(x, y, z) = {F}» in metres per second (lengths in metres).",
]
_A = [
    "The upward flux through the surface is a double integral «∬D g(x, y) dA» over the base region D. Find «g(x, y)».",
    "Write the flux through the surface in the upward direction as «∬D g(x, y) dA», where D is the base region. Find «g(x, y)».",
]
_B = [
    "Find the flux of F upward through the surface, in cubic metres per second.",
    "Evaluate the flux of F through the surface in the upward direction, in cubic metres per second.",
]
_C = [
    "The surface, the flat base D in the plane «z = 0» and the vertical walls between them bound a solid. Use the divergence theorem to find the total flux of F out of this solid, in cubic metres per second.",
    "Let W be the solid between the plane «z = 0» and the surface, above D. Find the total outward flux of F through the boundary of W, in cubic metres per second, using the divergence theorem.",
]
_D = [
    "The flux of F through the base D, with the downward normal, is «{down}». Hence find the flux of F outward through the vertical walls of the solid, in cubic metres per second.",
    "The outward flux through the base D is «{down}» cubic metres per second. Hence find the outward flux through the vertical walls, in cubic metres per second.",
]


def _walls_direct(F, f, region, kind, extent):
    """Outward flux through the vertical walls, face by face (a rectangle) or on the cylinder (a disc)."""
    total = sp.Integer(0)
    if kind == "rect":
        a, b = extent
        faces = [
            (sp.Matrix([0, u, w * f.subs({x: 0, y: u})]), ((0, b), (0, 1)), (-1, 0, 0)),
            (sp.Matrix([a, u, w * f.subs({x: a, y: u})]), ((0, b), (0, 1)), (1, 0, 0)),
            (sp.Matrix([u, 0, w * f.subs({x: u, y: 0})]), ((0, a), (0, 1)), (0, -1, 0)),
            (sp.Matrix([u, b, w * f.subs({x: u, y: b})]), ((0, a), (0, 1)), (0, 1, 0)),
        ]
        for r, ranges, ref in faces:
            total += vc.flux(F, vc.Surf(r, vc.Dom((u, w), ranges, {})), ref, "wall flux")
        return total
    R = extent
    rim = f.subs({x: R * vc.C, y: R * vc.S})
    surf = vc.Surf(sp.Matrix([R * vc.C, R * vc.S, w * rim]), vc.Dom((vc.v, w), ((0, 4), (0, 1)), {vc.v: (vc.C, vc.S)}))
    return vc.flux(F, surf, lambda p: (p[0], p[1], 0), "wall flux")


def _exam_once(rng):
    for _ in attempts(200, "a surface that stays above the plane z = 0"):
        F = _field(rng, 0)
        if F[2].subs(z, 0) == 0:
            continue
        f = _graph(rng, rng.choice(["plane", "bowl"]))
        kind = "disc" if (f.has(x**2) or f.has(y**2)) and rng.random() < 0.6 else "rect"
        if kind == "rect":
            a, b = rng.randint(1, 3), rng.randint(1, 3)
            region, region_text, extent = (
                vc.rect_region(0, a, 0, b),
                f"the rectangle «0 ≤ x ≤ {a}, 0 ≤ y ≤ {b}»",
                (a, b),
            )
        else:
            R = rng.randint(1, 3)
            region, region_text, extent = vc.polar_region(R), f"the disc «x² + y² ≤ {R**2}»", R
        height = sp.lambdify((x, y), f, "math")
        if min(height(a_, b_) for a_, b_ in _samples(region)) > 0.5:
            break
    value, g = _upward_flux(F, f, region)
    R0 = sp.expand(F[2].subs(z, 0))
    base = vc.region_integral(R0, region, "flux through the base")
    if base == 0:
        raise vc.Redraw
    div = sum(sp.diff(c, v) for c, v in zip(F, (x, y, z)))
    anti = sum(c * z ** (k[0] + 1) / (k[0] + 1) for k, c in sp.Poly(sp.expand(div), z).terms())
    inner = anti.xreplace({z: f})
    total = vc.region_integral(sp.expand(inner), region, "divergence integral")
    walls = total - value + base
    if vc.CHECKS:
        direct = _walls_direct(F, f, region, kind, extent)
        check(
            sp.simplify(walls - direct) == 0,
            f"the wall flux disagrees with the divergence theorem: {walls} vs {direct} ({kind}, {f}, {F})",
        )
    if 0 in (value, total, walls) or abs(total) > 6000:
        raise vc.Redraw
    wrong = lambda v: _integrand(F, f, v)  # noqa: E731
    ints = {
        v: (lambda v=v: vc.region_integral(wrong(v), region, verify=False))
        for v in ("signs", "z-not-replaced", "no-R", "swapped")
    }
    top = lambda: vc.region_integral(sp.expand(div.subs(z, f)), region, verify=False)  # noqa: E731
    flat = lambda: vc.region_integral(sp.expand(f * div.subs(z, 0)), region, verify=False)  # noqa: E731
    down = -base
    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_A)),
            g,
            [
                ("flux-graph-normal-signs", wrong("signs")),
                ("flux-graph-z-not-replaced", wrong("z-not-replaced")),
                ("flux-graph-dropped-r", wrong("no-R")),
                ("flux-graph-components-swapped", wrong("swapped")),
            ],
            2,
            ["With z = f(x, y), dS = (-f_x, -f_y, 1) dA, so g = -P f_x - Q f_y + R with z replaced by f."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_B)),
            value,
            [
                ("flux-graph-normal-signs", ints["signs"]),
                ("flux-graph-z-not-replaced", ints["z-not-replaced"]),
                ("flux-graph-dropped-r", ints["no-R"]),
                ("flux-graph-components-swapped", ints["swapped"]),
                ("flux-graph-wrong-direction", -value),
            ],
            3,
            ["Integrate g(x, y) over the base region."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_C)),
            total,
            [
                ("flux-graph-div-at-top", top),
                ("flux-graph-div-constant-height", flat),
                ("flux-graph-wrong-direction", -total),
                ("flux-graph-bottom-ignored", value),
            ],
            3,
            ["The outward flux is the triple integral of div F; integrate in z from 0 to f(x, y), then over D."],
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_D), down=vc.val_text(down)),
            walls,
            [
                ("flux-graph-base-sign", total - value - base),
                ("flux-graph-bottom-ignored", total - value),
                ("flux-graph-walls-reversed", value - base - total),
                ("flux-graph-wrong-direction", -walls),
            ],
            2,
            [
                "Total = top + base + walls, so walls = total - top - base (outward through the base is the downward flux)."
            ],
            follow_through=f"part_c - part_b - ({sp.sstr(down)})",
        ),
    ]
    template = rng.choice(_EXAM_STEMS).replace("{region}", region_text)
    return dict(**vc.stem(template, f=vc.poly_text(f, (x, y)), F=vc.mvec(F)), parts=parts)


_SAMPLES: dict = {}


def _samples(region):
    """A few points inside the base region, from its own parameters (kept: the regions are shared shapes)."""
    if id(region) not in _SAMPLES:
        _SAMPLES[id(region)] = (region, _sample_points(region))
    return _SAMPLES[id(region)][1]


def _sample_points(region):
    out = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            pt = {}
            for var, frac in zip(region.dom.vars, (p, q)):
                lo, hi = region.dom.ranges[region.dom.vars.index(var)]
                if var in region.dom.angles:
                    c, s = region.dom.angles[var]
                    a = float(sp.pi) * (lo + (hi - lo) * frac) / 2
                    pt[c], pt[s] = sp.cos(a), sp.sin(a)
                else:
                    pt[var] = lo + (hi - lo) * frac
            out.append((float(region.r[0].subs(pt)), float(region.r[1].subs(pt))))
    return out


@generator("exd_x_s2a3_flux_graph", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_flux_graph(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
