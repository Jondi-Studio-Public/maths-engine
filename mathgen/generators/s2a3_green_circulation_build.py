"""Generates `s2a3_green_circulation` questions (easy, medium, hard) and the exam chain `exd_x_s2a3_green_circulation`.

``uc2-greens-theorem-circulation`` (``multivariable-and-vector-calculus``, answer ``expression``): the circulation
∮ F · dr of a plane field round a closed curve, with the direction of travel stated.

* easy: a rectangle, counterclockwise. Q_x - P_y is a linear function, so the double integral is short.
* medium: a right triangle, counterclockwise or clockwise, and a field whose curl is quadratic.
* hard: a disc (possibly not centred on the origin), a half-disc whose boundary is an arc and a diameter, or the
  region between a parabola and a chord; the integral is done in polar coordinates or over curved limits, and the
  answer carries pi or is a fraction with a large denominator.
* exam (``exd_x_s2a3_green_circulation``): a force field round a loop: (a) the integrand Q_x - P_y, (b) the work in one
  counterclockwise lap, (c) the mean value of Q_x - P_y over the region (follows through from (b) and the area),
  (d) the work when the loop is run clockwise and the force doubled (follows through from (b)).

Every circulation is worked out two ways: as the double integral of Q_x - P_y over the region, and as the sum of the
line integrals along the pieces of the boundary, each parametrised. Green's theorem says they agree, and the
generator will not build a question unless they do. Both are done exactly and by quadrature.
"""

from __future__ import annotations

import sympy as sp

from ..core import attempts, check, generator
from . import _s2a3_vc as vc
from ._s2a3_vc import C, S, x, y

# ---------------------------------------------------------------------------------------------------------------
# regions: (area element region, boundary pieces counterclockwise, description of the curve, area)


def _poly_in_xy(rng, terms: int, max_degree: int):
    return vc.rand_poly(rng, (x, y), max_degree, terms, (-3, 3))


def _field(rng, max_degree: int):
    for _ in attempts(300, "a field whose curl is not constant"):
        P = _poly_in_xy(rng, rng.choice([2, 3]), max_degree)
        Q = _poly_in_xy(rng, rng.choice([2, 3]), max_degree)
        curl = sp.expand(sp.diff(Q, x) - sp.diff(P, y))
        if P != 0 and Q != 0 and curl != 0 and sp.Poly(curl, x, y).total_degree() >= 1:
            return P, Q
    raise AssertionError("unreachable")


def _shape(rng, tier: str):
    if tier == "easy":
        x0, y0 = rng.choice([0, 0, 1, -1]), rng.choice([0, 0, 1, -1])
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        x1, y1 = x0 + a, y0 + b
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return dict(
            region=vc.rect_region(x0, x1, y0, y1),
            pieces=[vc.segment(p, q) for p, q in zip(pts, pts[1:])],
            curve=f"the rectangle with vertices «({x0}, {y0}), ({x1}, {y0}), ({x1}, {y1}), ({x0}, {y1})»",
            area=sp.Integer(a * b),
        )
    if tier == "medium":
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        if rng.random() < 0.5:
            pts = [(0, 0), (a, 0), (0, b), (0, 0)]
            region = vc.tri_region(a, b, "corner")
            names = f"(0, 0), ({a}, 0), (0, {b})"
        else:
            pts = [(0, 0), (a, 0), (a, b), (0, 0)]
            region = vc.tri_region(a, b, "under")
            names = f"(0, 0), ({a}, 0), ({a}, {b})"
        return dict(
            region=region,
            pieces=[vc.segment(p, q) for p, q in zip(pts, pts[1:])],
            curve=f"the triangle with vertices «{names}»",
            area=sp.Rational(a * b, 2),
        )
    kind = rng.choice(["disc", "offset", "half", "parabola"])
    R = rng.randint(1, 3)
    if kind == "disc":
        return dict(
            region=vc.polar_region(R),
            pieces=[vc.circle_curve((0, 0), R)],
            curve=f"the circle «x² + y² = {R * R}»",
            area=sp.pi * R * R,
        )
    if kind == "offset":
        h, k = rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 0, 1, 2])
        hx = f"(x - {h})" if h > 0 else f"(x + {-h})"
        ky = "y²" if k == 0 else (f"(y - {k})²" if k > 0 else f"(y + {-k})²")
        return dict(
            region=vc.polar_region(R, centre=(h, k)),
            pieces=[vc.circle_curve((h, k), R)],
            curve=f"the circle «{hx}² + {ky} = {R * R}»",
            area=sp.pi * R * R,
        )
    if kind == "half":
        return dict(
            region=vc.polar_region(R, 0, 2),
            pieces=[vc.segment((-R, 0), (R, 0)), vc.circle_curve((0, 0), R, quarters=(0, 2))],
            curve=f"the closed curve made of the diameter from «(-{R}, 0)» to «({R}, 0)» and the upper half of the circle «x² + y² = {R * R}»",
            area=sp.pi * R * R / 2,
        )
    m = rng.randint(1, 3)
    dom = vc.Dom((vc.u, vc.v), ((0, 1), (0, 1)), {})
    xx = m * vc.u
    region = vc.Surf(sp.Matrix([xx, xx**2 + (m * xx - xx**2) * vc.v, 0]), dom)
    parabola = vc.Curve(sp.Matrix([m * vc.t, m**2 * vc.t**2]), vc.Dom((vc.t,), ((0, 1),), {}))
    return dict(
        region=region,
        pieces=[parabola, vc.segment((m, m * m), (0, 0))],
        curve=f"the closed curve made of the parabola «y = x²» from «(0, 0)» to «({m}, {m * m})» and the line «y = {'' if m == 1 else m}x» back to the origin",
        area=sp.Rational(m**3, 6),
    )


def _circulation(P, Q, shape):
    """(circulation counterclockwise, the curl integrand, the line integral along each piece or None). With the
    cross-checks on, the boundary is integrated piece by piece and must agree with the double integral."""
    curl = sp.expand(sp.diff(Q, x) - sp.diff(P, y))
    by_area = vc.region_integral(curl, shape["region"])
    if not vc.CHECKS:
        return by_area, curl, None
    edges = [vc.line([P, Q], piece) for piece in shape["pieces"]]
    by_boundary = sum(edges, sp.Integer(0))
    check(sp.expand(by_area - by_boundary) == 0, f"Green's theorem fails: {by_area} against {by_boundary}")
    return by_area, curl, edges


def _candidates(P, Q, shape, signed, sign, circ):
    region = shape["region"]
    out = [
        ("green-wrong-direction", -signed),
        (
            "green-divergence-for-curl",
            lambda: sign * vc.region_integral(sp.diff(Q, y) - sp.diff(P, x), region, verify=False),
        ),
        ("green-dropped-py", lambda: sign * vc.region_integral(sp.diff(Q, x), region, verify=False)),
        ("green-dropped-qx", lambda: sign * vc.region_integral(-sp.diff(P, y), region, verify=False)),
        ("green-not-differentiated", lambda: sign * vc.region_integral(Q - P, region, verify=False)),
    ]
    if region.dom.angles:  # polar: the factor r left out of dA
        g = sp.expand(circ.xreplace(dict(zip((x, y), region.r[:2]))))
        out.append(("green-polar-no-r", lambda: sign * vc.integrate(g, region.dom, verify=False)))
    if len(shape["pieces"]) == 2:  # half-disc or parabola: only the curved piece counted
        out.append(
            (
                "green-open-boundary",
                sign
                * vc.line(
                    [P, Q], shape["pieces"][-1] if "diameter" in shape["curve"] else shape["pieces"][0], verify=False
                ),
            )
        )
    return out


_STEMS = {
    "easy": [
        "Evaluate the circulation «∮C F · dr» of «F(x, y) = {F}» once round the boundary C of {curve}, traversed counterclockwise.",
        "Find the work done by the force «F(x, y) = {F}» on a particle that goes once round {curve}, counterclockwise.",
        "Let C be {curve}, traversed counterclockwise. Evaluate «∮C F · dr» for «F(x, y) = {F}».",
    ],
}
_STEMS["medium"] = [
    "Evaluate «∮C F · dr» for «F(x, y) = {F}», where C is {curve}, traversed {way}.",
    "Find the circulation of «F(x, y) = {F}» round {curve}, traversed {way}.",
    "A particle goes once round {curve}, {way}, under the force «F(x, y) = {F}». Find the work done by the force.",
]
_STEMS["hard"] = _STEMS["medium"]


def _draw(rng, tier):
    P, Q = _field(rng, 2 if tier != "hard" else 3)
    shape = _shape(rng, tier)
    way = "counterclockwise" if tier == "easy" else rng.choice(["counterclockwise", "counterclockwise", "clockwise"])
    sign = 1 if way == "counterclockwise" else -1
    ccw, curl, edges = _circulation(P, Q, shape)
    answer = sign * ccw
    if answer == 0 or abs(answer) > 3000:
        return None
    template = rng.choice(_STEMS[tier]).replace("{curve}", shape["curve"])
    text = vc.stem(template, F=vc.mvec([P, Q], "ij"), way=way)
    working = [
        "Green's theorem: the counterclockwise circulation is the integral of Q_x - P_y over the region inside C"
        + ("; clockwise changes the sign." if sign < 0 else "."),
        f"Q_x - P_y = {vc.poly_text(curl, (x, y))}.",
    ]
    return dict(**text, answer=answer, candidates=_candidates(P, Q, shape, answer, sign, curl), working=working)


@generator(
    "s2a3_green_circulation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a3_green_circulation(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return vc.single_question(rng, _draw, tier)


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A particle goes round {curve} under the force field «F(x, y) = {F}», with force in newtons and lengths in metres.",
    "A robot arm drags a tool once round {curve}. The force on the tool is «F(x, y) = {F}», in newtons, with lengths in metres.",
]
_A = [
    "Write the work done in one counterclockwise lap as «∬D g(x, y) dA» over the region D inside the loop. Find «g(x, y)».",
    "By Green's theorem the work in one counterclockwise lap is «∬D g(x, y) dA». Find «g(x, y)».",
]
_B = [
    "Find the work done by F in one counterclockwise lap, in joules.",
    "Evaluate the work done by the force in one counterclockwise lap, in joules.",
]
_C = [
    "The region D inside the loop has area «{area}» m². Hence find the mean value of «g(x, y)» over D.",
    "D has area «{area}» m². Use your answer to (b) to find the mean value of «g(x, y)» over D.",
]
_D = [
    "The tool is sent round the loop clockwise instead, with the force doubled to «2F». Find the work done in one lap, in joules.",
    "Now the loop is run clockwise and the force is doubled to «2F». Find the work done in one lap, in joules.",
]


def _exam_once(rng):
    tier = rng.choice(["medium", "hard"])
    P, Q = _field(rng, 2)
    shape = _shape(rng, tier)
    ccw, curl, edges = _circulation(P, Q, shape)
    area = shape["area"]
    mean = ccw / area
    if 0 in (ccw, mean) or area == 1 or abs(ccw) > 3000:
        raise vc.Redraw
    region = shape["region"]
    wrong_curls = {
        "green-divergence-for-curl": sp.expand(sp.diff(Q, y) - sp.diff(P, x)),
        "green-dropped-py": sp.expand(sp.diff(Q, x)),
        "green-dropped-qx": sp.expand(-sp.diff(P, y)),
        "green-not-differentiated": sp.expand(Q - P),
        "green-wrong-direction": sp.expand(-curl),
    }
    cand_b = [(tag, lambda g=g: vc.region_integral(g, region, verify=False)) for tag, g in wrong_curls.items()]
    cand_c = [
        ("green-mean-no-area", ccw),
        ("green-mean-times-area", ccw * area),
        ("green-divergence-for-curl", lambda: cand_b[0][1]() / area),
        ("green-wrong-direction", -mean),
    ]
    cand_d = [
        ("green-orientation-ignored", 2 * ccw),
        ("green-scale-ignored", -ccw),
        ("green-scale-squared", -4 * ccw),
        ("green-orientation-ignored", ccw),
    ]
    area_text = vc.val_text(area)
    parts = [
        vc.part_dict(
            vc.prompt(rng.choice(_A)),
            curl,
            [(tag, g) for tag, g in wrong_curls.items()],
            2,
            ["With P and Q the components of F, g = Q_x - P_y."],
        ),
        vc.part_dict(vc.prompt(rng.choice(_B)), ccw, cand_b, 3, ["Integrate g over the region inside the loop."]),
        vc.part_dict(
            vc.prompt(rng.choice(_C), area=area_text),
            mean,
            cand_c,
            2,
            ["Mean value = (integral of g over D) / (area of D), using the answer to (b)."],
            follow_through=f"part_b / ({sp.sstr(area)})",
        ),
        vc.part_dict(
            vc.prompt(rng.choice(_D)),
            -2 * ccw,
            cand_d,
            2,
            ["Clockwise changes the sign and doubling the force doubles the work: -2 times (b)."],
            follow_through="-2*part_b",
        ),
    ]
    template = rng.choice(_EXAM_STEMS).replace("{curve}", shape["curve"])
    return dict(**vc.stem(template, F=vc.mvec([P, Q], "ij")), parts=parts)


@generator("exd_x_s2a3_green_circulation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a3_green_circulation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return vc.exam_question(rng, _exam_once)
