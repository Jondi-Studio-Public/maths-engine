"""Exam-tier arc, sector and segment questions with exact answers.

The practice tiers apply the arc-length or sector-area formula once to values
that are given directly. These questions need a formula to be combined with
another idea; the answers are exact (in terms of pi, and surds where they
occur):

- ``perimeter_from_area``: a sector's area and angle are given; the radius must
  be recovered before the perimeter (two radii plus the arc) can be found.
- ``segment_area``: the area of the segment cut off by a chord, i.e. sector
  area minus triangle area, using the exact sine of a standard angle.
- ``cone_base_radius``: a sector is rolled into a cone; the arc becomes the
  circumference of the base, so the base radius is found.
"""

from __future__ import annotations

import sympy as sp

from ..core import check, generator, latex_answer, phrase, pick_distractors

_ANGLES = {  # angle as a multiple of pi -> exact sine
    sp.Rational(1, 6): sp.Rational(1, 2),
    sp.Rational(1, 3): sp.sqrt(3) / 2,
    sp.Rational(1, 2): sp.Integer(1),
    sp.Rational(2, 3): sp.sqrt(3) / 2,
    sp.Rational(5, 6): sp.Rational(1, 2),
}

_PERIMETER = (
    "A sector of a circle has an angle of {theta} radians at the centre and an area of {area} cm^2. Find the exact perimeter of the sector, in cm.",
    "The area of a sector is {area} cm^2 and its angle at the centre is {theta} radians. Find the exact perimeter of the sector, including both straight edges, in cm.",
)
_SEGMENT = (
    "A chord cuts a circle of radius {r} cm so that it subtends an angle of {theta} radians at the centre. Find the exact area of the minor segment between the chord and the arc, in cm^2.",
    "In a circle of radius {r} cm, a chord subtends an angle of {theta} radians at the centre. Find the exact area, in cm^2, of the region between the chord and the minor arc.",
)
_CONE = (
    "A sector of a circle with radius {R} cm and angle {theta} radians is bent round to form the curved surface of a cone, the two straight edges meeting. Find the radius of the base of the cone, in cm.",
    "A paper sector has radius {R} cm and an angle of {theta} radians at the centre. Its straight edges are joined to make an open cone. Find the radius of the circular base, in cm.",
)


def _pi_text(coeff) -> str:
    """Text of coeff*pi in the spelling the question uses, e.g. 7/3 -> '7pi/3'."""
    coeff = sp.Rational(coeff)
    head = "pi" if abs(coeff.p) == 1 else f"{abs(coeff.p)}pi"
    sign = "-" if coeff.p < 0 else ""
    return f"{sign}{head}" if coeff.q == 1 else f"{sign}{head}/{coeff.q}"


def _text(expr) -> str:
    """Canonical text of ``pi_part*pi + rest``: the pi term first, then the rest."""
    e = sp.expand(expr)
    pi_part = e.coeff(sp.pi)
    rest = sp.expand(e - pi_part * sp.pi)
    out = str(sp.expand(pi_part * sp.pi)) if pi_part != 0 else ""
    if rest != 0:
        rest_text = str(rest)
        if not out:
            return rest_text
        out = f"{out} - {rest_text[1:]}" if rest_text.startswith("-") else f"{out} + {rest_text}"
    return out or "0"


def _perimeter(rng):
    r = rng.randint(3, 12)
    c = rng.choice([sp.Rational(n, d) for n, d in ((1, 3), (1, 2), (2, 3), (3, 4), (5, 6), (1, 4), (3, 2))])
    area = r * r * c / 2  # the coefficient of pi in the stated area
    key = 2 * r + r * c * sp.pi
    cands = [
        r * c * sp.pi,  # the arc length alone
        r + r * c * sp.pi,  # only one radius added
        2 * r * r + r * r * c * sp.pi,  # radius left squared: forgot the square root
        2 * r + area * sp.pi,  # the sector area used where the arc length belongs
    ]
    text = phrase(rng, _PERIMETER, theta=_pi_text(c), area=_pi_text(area))
    return text, key, cands, f"Area = (1/2) r^2 theta gives r = {r}; perimeter = 2r + r theta."


def _segment(rng):
    r = rng.randint(2, 12)
    c = rng.choice(list(_ANGLES))
    sin_t = _ANGLES[c]
    sector = sp.Rational(1, 2) * r * r * c * sp.pi
    triangle = sp.Rational(1, 2) * r * r * sin_t
    key = sector - triangle
    cands = [
        sector,  # forgets to remove the triangle
        sector + triangle,  # adds the triangle
        sector - 2 * triangle,  # drops the one half from the triangle area
        sp.Rational(1, 2) * r * r * c - triangle,  # theta used without its factor of pi
    ]
    text = phrase(rng, _SEGMENT, r=r, theta=_pi_text(c))
    return text, key, cands, "Segment = sector - triangle = (1/2) r^2 (theta - sin theta)."


def _cone(rng):
    for _ in range(100):
        c = rng.choice([sp.Rational(n, d) for n, d in ((1, 2), (2, 3), (3, 2), (4, 3), (5, 3), (5, 6), (3, 4))])
        big_r = rng.randint(6, 24)
        key = sp.Rational(big_r) * c / 2  # 2 pi r = R theta with theta = c pi
        if key.q > 4:
            continue
        cands = [
            2 * key,  # arc taken as pi r rather than 2 pi r
            key / 2,  # the sector-area half applied to the arc
            big_r * c * sp.pi,  # the arc length itself
            big_r - key,
        ]
        text = phrase(rng, _CONE, R=big_r, theta=_pi_text(c))
        return text, key, cands, "Arc length R theta equals the base circumference 2 pi r."
    check(False, "exd_x_f4_arcs: no cone draw")


_BUILDERS = (_perimeter, _segment, _cone)


@generator("exd_x_f4_arcs", level=12, presets={"exam": dict(tier="exam")})
def exd_x_f4_arcs(rng, tier: str = "exam"):
    text, key, cands, step = rng.choice(_BUILDERS)(rng)
    answer_text = _text(key)
    shown = [_text(cnd) for cnd in cands if sp.simplify(cnd - key) != 0]
    distractors = pick_distractors(answer_text, shown)
    check(len(distractors) >= 2, "exd_x_f4_arcs: too few distractors")
    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[step, f"Answer = {answer_text}."],
        distractors=distractors,
    )
