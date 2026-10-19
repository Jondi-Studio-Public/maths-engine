"""Argument loci Arg(z - a) = theta and wedges theta1 <= Arg(z - a) <= theta2: `sr2l_arg_ray` (easy, medium, hard) and
`sr2l_arg_ray_exam`.

Skill: locus-argument-ray-wedge. A ray or wedge is not a full line, but every boundary lies on one, and a line equation
is what the single-equation `pair` path can mark. The ray itself (an open end at a, never the point a) is drawn in the
medium tier and described in words elsewhere.

* easy: Arg(z - a) = theta with theta a multiple of pi/4 and a with signs to resolve (Arg(z + 3 - 2i)). Give the line
  that contains the ray.
* medium: the ray is drawn from an open dot on an Argand diagram, in a direction other than a multiple of pi/4 (slopes
  such as 2, 1/2 and -3), and the line has to be read from the lattice points on it.
* hard: a wedge theta1 <= Arg(z - a) <= theta2 with a at a general point and angles negative or obtuse. Give the line
  that contains the boundary ray with the greater (or smaller) argument.
* exam: a wedge of angle between lines through a. Both boundary lines, the length it cuts from a horizontal or vertical
  line, and the area of the triangle that cut forms with a (follow-through).

Every key is re-derived by taking a point on the claimed ray, measuring its argument about a with `cmath.phase`, and
checking that it equals the stated angle and lies on the claimed line.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

from mathgen import figures
from mathgen.core import attempts, check, generator, phrase

from . import _sr2l_kit as kit

_FORM = "Give the equation in the form y = mx + c, or x = k if the line is vertical."
_DIR = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), -1: (1, -1), -2: (0, -1), -3: (-1, -1)}
_ALT_RAY = "An Argand diagram on a square grid showing a ray that starts at an open dot."


def ang_tex(k: int) -> str:
    """k multiples of pi/4 in tex: 0, \\frac{\\pi}{4}, -\\frac{3\\pi}{4}, \\pi."""
    if k == 0:
        return "0"
    if k == 4:
        return r"\pi"
    sign = "-" if k < 0 else ""
    n = abs(k)
    if n == 2:
        return rf"{sign}\frac{{\pi}}{{2}}"
    return rf"{sign}\frac{{{'' if n == 1 else n}\pi}}{{4}}"


def arg_tex(a, k: int) -> str:
    return rf"\operatorname{{Arg}}({kit.z_minus(*a)}) = {ang_tex(k)}"


def _ray_line(a, d):
    m, c = kit.line_through(a, d)
    return m, c


def _verify_ray(a, k: int, line) -> None:
    """A point on the ray at angle k*pi/4 from a has that argument about a and lies on the claimed line."""
    m, c = line
    for t in (1, 2, 5):
        d = _DIR[k]
        z = complex(a[0] + t * d[0], a[1] + t * d[1])
        check(
            abs(cmath.phase(z - complex(*a)) - k * math.pi / 4) < 1e-9
            or (k == 4 and abs(abs(cmath.phase(z - complex(*a))) - math.pi) < 1e-9),
            "argument disagrees",
        )
        if m is None:
            check(abs(z.real - float(c)) < 1e-9, "the ray is not on the claimed vertical line")
        else:
            check(abs(z.imag - (float(m) * z.real + float(c))) < 1e-9, "the ray is not on the claimed line")


def _wrong_lines(a, d, line):
    """Wrong lines for a ray from a in direction d, as (m, c, tag)."""
    out = []

    def add(point, direction, tag):
        wm, wc = kit.line_through(point, direction)
        out.append((wm, wc, tag))

    add((-a[0], -a[1]), d, "sr2l-centre-sign-flipped")
    add(a, (d[0], -d[1]), "sr2l-argument-sign-flipped")
    add((a[1], a[0]), (d[1], d[0]), "sr2l-real-imaginary-swapped")
    add((0, 0), d, "sr2l-vertex-ignored")
    add((-a[0], -a[1]), (d[0], -d[1]), "sr2l-centre-sign-flipped")
    add(a, (-d[1], d[0]), "sr2l-perpendicular-line-for-ray")
    return [(wm, wc, tag) for wm, wc, tag in out if (wm, wc) != line]


def _key(line):
    m, c = line
    return (None if m is None else Fraction(m), Fraction(c))


def _line_fields(m, c):
    return kit.vertical_fields(c) if m is None else kit.line_fields(m, c)


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng):
    k = rng.choice((-3, -2, -1, 0, 1, 2, 3, 4)) if rng.random() < 0.85 else rng.choice((1, 3, -1, -3))
    a = (rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v]))
    d = _DIR[k]
    line = _ray_line(a, d)
    _verify_ray(a, k, line)
    fields = _line_fields(*line)
    stems = (
        "The locus of the complex number z is the ray {rel}. Find the Cartesian equation of the line that contains the ray, "
        "where z = x + iy. " + _FORM,
        "A complex number z satisfies {rel}. Find the equation of the line containing this ray, with z = x + iy. "
        + _FORM,
        "The points z with {rel} form a ray. Write the Cartesian equation of the line on which the ray lies, where z = x + iy. "
        + _FORM,
    )
    return dict(
        **kit.prompt(phrase(rng, stems, rel=kit.maths(arg_tex(a, k)))),
        working=[
            f"The ray starts at a = {kit.cplx(*a)} and makes an angle {kit.plain(ang_tex(k))} with the positive real direction.",
            f"The line through ({a[0]}, {a[1]}) in that direction is {fields['answer_text']}.",
        ],
        distractors=kit.tagged_lines(_key(line), _wrong_lines(a, d, line)),
        **fields,
    )


# -- medium ----------------------------------------------------------------------------------------------------------

_SLOPED = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, -2),
    (2, -1),
    (1, -3),
    (3, -1),
    (2, -3),
    (3, -2),
    (-1, 2),
    (-2, 1),
)


def _exit_point(a, d, lo=-8, hi=8):
    """Where the ray from a along d leaves the square [lo, hi]^2."""
    ts = []
    for coord, step in zip(a, d):
        if step > 0:
            ts.append(Fraction(hi - coord, step))
        elif step < 0:
            ts.append(Fraction(lo - coord, step))
    t = min(ts)
    return a[0] + t * d[0], a[1] + t * d[1]


def _medium(rng):
    for _ in attempts(500, "a ray with lattice points to read"):
        d = rng.choice(_SLOPED)
        a = (rng.randint(-5, 5), rng.randint(-5, 5))
        end = _exit_point(a, d)
        if a[0] != 0 and a[1] != 0 and (end[0] - a[0]) ** 2 + (end[1] - a[1]) ** 2 > 40:
            break
    line = _ray_line(a, d)
    fig = kit.plane(
        lo=-8,
        hi=8,
        alt=_ALT_RAY,
        segments=[{"from": [a[0], a[1]], "to": [float(end[0]), float(end[1])], "open_start": True}],
    )
    fields = _line_fields(*line)
    check(
        _read_figure(type("Q", (), {"figure": fig})) == fields["answer_text"],
        "the drawn ray does not give the keyed line",
    )
    stems = (
        r"The diagram shows the locus of z, where z satisfies \(\operatorname{Arg}(z - a) = \theta\) for a complex number a "
        "and an angle θ. Find the Cartesian equation of the line that contains the ray, where z = x + iy. " + _FORM,
        r"The ray drawn on the Argand diagram is the locus \(\operatorname{Arg}(z - a) = \theta\), with the open dot at a. "
        "Find the Cartesian equation of the line containing the ray, where z = x + iy. " + _FORM,
        r"The ray in the diagram has the equation \(\operatorname{Arg}(z - a) = \theta\), where a is marked by an open dot. "
        "Write the Cartesian equation of the line that the ray lies on, with z = x + iy. " + _FORM,
    )
    return dict(
        **kit.prompt(rng.choice(stems)),
        figure=fig,
        working=[
            f"The open dot is the point a = {kit.cplx(*a)}; the ray goes {abs(d[1])} {'up' if d[1] > 0 else 'down'} for every {abs(d[0])} {'right' if d[0] > 0 else 'left'}.",
            f"So the line has gradient {kit.rt(Fraction(d[1], d[0]))} and passes through ({a[0]}, {a[1]}): {fields['answer_text']}.",
        ],
        distractors=kit.tagged_lines(_key(line), _wrong_lines(a, d, line)),
        **fields,
    )


def _read_figure(question) -> str:
    """Re-derive the line from the drawn ray alone: its start and a second point on it."""
    seg = question.figure["segments"][0]
    p, q = seg["from"], seg["to"]
    fx = lambda v: Fraction(v).limit_denominator(1000)  # noqa: E731 - the drawn end is a float on the frame
    m, c = kit.line_through((fx(p[0]), fx(p[1])), (fx(q[0]) - fx(p[0]), fx(q[1]) - fx(p[1])))
    return kit.vertical_text(c) if m is None else kit.line_text(m, c)


# -- hard ------------------------------------------------------------------------------------------------------------


def _hard(rng):
    for _ in attempts(500, "a wedge with distinct boundaries"):
        k1 = rng.randint(-3, 3)
        k2 = k1 + rng.randint(1, 3)
        if k2 <= 4 and (k1, k2) != (-3, 4):
            break
    a = (rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v]))
    greater = rng.random() < 0.5
    kb, ko = (k2, k1) if greater else (k1, k2)
    d, other = _DIR[kb], _DIR[ko]
    line = _ray_line(a, d)
    _verify_ray(a, kb, line)
    fields = _line_fields(*line)
    which = "greater" if greater else "smaller"
    rel = rf"{ang_tex(k1)} \le \operatorname{{Arg}}({kit.z_minus(*a)}) \le {ang_tex(k2)}"
    stem = (
        rf"The set S of complex numbers z satisfies \({rel}\). The boundary of S is two rays from a point. "
        f"Find the Cartesian equation of the line that contains the boundary ray with the {which} argument, "
        "where z = x + iy. " + _FORM
    )
    wrong = _wrong_lines(a, d, line)
    om, oc = kit.line_through(a, other)
    cands = [(om, oc, "sr2l-wrong-boundary-ray")] + wrong
    return dict(
        **kit.prompt(stem),
        working=[
            f"The wedge has vertex {kit.cplx(*a)} and its boundary rays have arguments {kit.plain(ang_tex(k1))} and {kit.plain(ang_tex(k2))}.",
            f"The boundary with the {which} argument has argument {kit.plain(ang_tex(kb))}; its line is {fields['answer_text']}.",
        ],
        distractors=kit.tagged_lines(_key(line), cands),
        **fields,
    )


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("sr2l_arg_ray", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_arg_ray(rng, tier: str = "easy"):
    """The line that contains a ray Arg(z - a) = theta or a wedge boundary."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, _TIERS[tier])


figures.register_reader("sr2l_arg_ray", _read_figure)


# -- exam ------------------------------------------------------------------------------------------------------------

_FAMILIES = {
    "right": [(-1, 0), (0, 1), (-1, 1)],
    "up": [(1, 2), (2, 3), (1, 3)],
    "down": [(-3, -2), (-2, -1), (-3, -1)],
}


def _cut(a, k, family, t):
    """Where the boundary ray at angle k*pi/4 from a meets the cutting line, a distance t from a along the wedge."""
    dx, dy = _DIR[k]
    if family == "right":
        s = Fraction(t, dx)
    elif family == "up":
        s = Fraction(t, dy)
    else:
        s = Fraction(t, -dy)
    return a[0] + s * dx, a[1] + s * dy


@generator("sr2l_arg_ray_exam", level=18, presets={"exam": {}})
def sr2l_arg_ray_exam(rng):
    return kit.redraw(rng, _exam)


def _exam(rng):
    """A wedge: its two boundary lines, the chord it cuts from a horizontal or vertical line, and the triangle's area."""
    family = rng.choice(list(_FAMILIES))
    k1, k2 = rng.choice(_FAMILIES[family])
    a = (rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-5, 6) if v]))
    t = rng.randint(2, 6)
    rel = rf"{ang_tex(k1)} \le \operatorname{{Arg}}({kit.z_minus(*a)}) \le {ang_tex(k2)}"
    p1, p2 = _cut(a, k1, family, t), _cut(a, k2, family, t)
    if family == "right":
        cut_line, chord = f"x = {a[0] + t}", abs(p2[1] - p1[1])
    elif family == "up":
        cut_line, chord = f"y = {a[1] + t}", abs(p2[0] - p1[0])
    else:
        cut_line, chord = f"y = {a[1] - t}", abs(p2[0] - p1[0])
    area = chord * t / 2
    for k in (k1, k2):  # both boundary points lie on the cutting line, and on the ray at the stated argument
        px, py = _cut(a, k, family, t)
        check(
            abs(cmath.phase(complex(px - a[0], py - a[1])) - k * math.pi / 4) < 1e-9,
            "a cut point is not on its boundary ray",
        )
    d1, d2 = _DIR[k1], _DIR[k2]
    m2, c2 = _ray_line(a, d2)
    m1, c1 = _ray_line(a, d1)
    part_a = kit.line_part(
        "Find the Cartesian equation of the line that contains the boundary ray with the greater argument.",
        m2,
        c2,
        2,
        [
            f"The ray with argument {kit.plain(ang_tex(k2))} leaves the vertex {kit.cplx(*a)} in the direction ({d2[0]}, {d2[1]})."
        ],
        _wrong_lines(a, d2, (m2, c2)) + [(m1, c1, "sr2l-wrong-boundary-ray")],
    )
    part_b = kit.line_part(
        "Find the Cartesian equation of the line that contains the boundary ray with the smaller argument.",
        m1,
        c1,
        2,
        [
            f"The ray with argument {kit.plain(ang_tex(k1))} leaves the vertex {kit.cplx(*a)} in the direction ({d1[0]}, {d1[1]})."
        ],
        _wrong_lines(a, d1, (m1, c1)) + [(m2, c2, "sr2l-wrong-boundary-ray")],
    )
    part_c = kit.number_part(
        f"Each boundary ray crosses the line {cut_line}. Find the distance between the two points where they cross it.",
        chord,
        0,
        2,
        [
            f"The boundary rays meet {cut_line} at ({kit.rt(p1[0])}, {kit.rt(p1[1])}) and ({kit.rt(p2[0])}, {kit.rt(p2[1])}); the distance between them is {kit.rt(chord)}."
        ],
        [
            (t, "sr2l-chord-half-only"),
            (chord + t, "sr2l-chord-and-height-added"),
            (2 * chord, "sr2l-chord-doubled"),
            (chord * t, "sr2l-triangle-half-omitted"),
        ],
        sensible=(0, None),
    )
    part_d = kit.number_part(
        "The vertex of the wedge and the two points in part (c) form a triangle. Find its area.",
        area,
        1,
        2,
        [
            f"The triangle has base {kit.rt(chord)} (from part (c)) and height {t}, the distance from the vertex to {cut_line}: area = 1/2 x base x height."
        ],
        [
            (chord * t, "sr2l-triangle-half-omitted"),
            (chord / 2, "sr2l-base-for-area"),
            (chord + t, "sr2l-chord-and-height-added"),
            (t * t / 2, "sr2l-chord-half-only"),
        ],
        follow=f"part_c*{t}/2",
        sensible=(0, None),
    )
    stem = rf"The set S of complex numbers z = x + iy satisfies \({rel}\). The boundary of S is two rays from the point a = {kit.cplx(*a)}."
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])
