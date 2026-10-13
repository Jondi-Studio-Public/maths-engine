"""The perpendicular-bisector locus |z - a| = |z - b|: `sr2l_bisector` (easy, medium, hard) and `sr2l_bisector_exam`.

Skill: locus-perpendicular-bisector. The answer to every practice question is the Cartesian equation of the locus
as a single line equation, graded by the existing single-equation `pair` path: y = mx + c, or x = k when the
line is vertical.

* easy: a and b are a whole-number distance apart in a simple direction (across, up, or on a diagonal), written with
  signs to resolve (|z + 3| = |z - 5i|).
* medium: the two points are plotted on an Argand diagram, in any direction, and the line is found from their midpoint
  and the gradient perpendicular to the line joining them.
* hard: the condition has to be rewritten first: |z - a| = |z-bar - b|, where the conjugate reflects b; or
  |kz - A| = |kz - B|, where the points are A/k and B/k; or the two points are the roots of a quadratic given by its sum
  and product.
* exam: the locus, where it meets the real axis, the area it cuts off with the axes (follow-through), and the shortest
  distance from the origin to it (follow-through from the area and the intercept).

Every key is re-derived by evaluating the printed condition at points on the claimed line, where the two sides are
equal, and at points just off it, where they are not.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen import figures
from mathgen.core import attempts, check, generator, join_terms, phrase

from . import _sr2l_kit as kit

_FORM = "Give the equation in the form y = mx + c, or x = k if the line is vertical."
_ALT_AB = "An Argand diagram on a square grid with two labelled points A and B joined by a segment."


def bisector(a, b):
    """(m, c) of the perpendicular bisector of a and b, or (None, k) for the vertical line x = k."""
    mid = (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))
    return kit.line_through(mid, (-(b[1] - a[1]), b[0] - a[0]))


def _verify(relation: str, line) -> None:
    """The printed relation holds along the claimed line and fails just off it."""
    m, c = line
    for x in (-3, -1, 0.5, 2, 4.5):
        z = complex(float(c), x) if m is None else complex(x, float(m) * x + float(c))
        scale = 1 + abs(z) ** 2
        check(kit.residual(relation, z) < 1e-9 * scale, f"{relation!r} fails on the claimed line")
        for off in (0.4, -0.4):
            check(
                kit.residual(relation, z + off) > 1e-6 or kit.residual(relation, z + 1j * off) > 1e-6,
                "also holds off the line",
            )


def _wrong_lines(a, b, m, c):
    """Candidate wrong lines for the bisector of a and b, as (m, c, tag)."""
    mid = (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))
    d = (b[0] - a[0], b[1] - a[1])
    perp = (-d[1], d[0])
    out = []

    def add(point, direction, tag):
        wm, wc = kit.line_through(point, direction)
        out.append((wm, wc, tag))

    add(mid, d, "sr2l-chord-line-for-bisector")
    add(a, perp, "sr2l-bisector-through-endpoint")
    add((-mid[0], -mid[1]), perp, "sr2l-centre-sign-flipped")
    add(mid, (perp[1], perp[0]), "sr2l-gradient-sign-wrong")
    add((mid[1], mid[0]), (perp[1], perp[0]), "sr2l-real-imaginary-swapped")
    add(b, perp, "sr2l-bisector-through-endpoint")
    return [(wm, wc, tag) for wm, wc, tag in out if (wm, wc) != (m, c)]


def _line_question(rng, stem_tex, a, b, working, *, figure=None, extra_wrong=(), relation=None):
    m, c = bisector(a, b)
    fields = kit.vertical_fields(c) if m is None else kit.line_fields(m, c)
    cands = list(extra_wrong) + _wrong_lines(a, b, m, c)
    out = dict(
        **kit.prompt(stem_tex),
        working=working + [f"The locus is the line {fields['answer_text']}."],
        distractors=kit.tagged_lines((None if m is None else Fraction(m), Fraction(c)), cands),
        **fields,
    )
    if figure is not None:
        out["figure"] = figure
    if relation is not None:
        _verify(relation, (m, c))
    return out


def _rel(a, b):
    return rf"{kit.mod(kit.z_minus(*a))} = {kit.mod(kit.z_minus(*b))}"


# -- easy ------------------------------------------------------------------------------------------------------------

_DIRS_EASY = ((1, 0), (0, 1), (1, 1), (1, -1))


def _easy(rng):
    for _ in attempts(500, "two points with a simple bisector"):
        d = rng.choice(_DIRS_EASY)
        k = rng.choice((2, 4, 6))
        a = (rng.randint(-5, 5), rng.randint(-5, 5))
        sign = rng.choice((1, -1))
        b = (a[0] + sign * k * d[0], a[1] + sign * k * d[1])
        if a != b and max(map(abs, (*a, *b))) <= 8 and a != (0, 0) and b != (0, 0):
            break
    rel = _rel(a, b)
    stems = (
        "The locus of the complex number z is given by {rel}. Find its Cartesian equation, where z = x + iy. " + _FORM,
        "A complex number z satisfies {rel}. Write the locus of z as a Cartesian equation, with z = x + iy. " + _FORM,
        "The points z with {rel} lie on a line. Find the equation of that line, where z = x + iy. " + _FORM,
    )
    return _line_question(
        rng,
        phrase(rng, stems, rel=kit.maths(rel)),
        a,
        b,
        [
            f"|z - a| = |z - b| says z is equally far from a = {kit.cplx(*a)} and b = {kit.cplx(*b)}.",
            "The locus is the perpendicular bisector of the segment joining them: through the midpoint, perpendicular to it.",
        ],
        relation=kit.plain(rel),
    )


# -- medium ----------------------------------------------------------------------------------------------------------

_DIRS_MEDIUM = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, -2), (2, -1), (1, -3), (3, -1), (2, -3), (3, -2))


def _plot(a, b):
    return kit.plane(
        lo=-9,
        hi=9,
        alt=_ALT_AB,
        points=[{"at": [a[0], a[1]], "label": "A"}, {"at": [b[0], b[1]], "label": "B"}],
        segments=[{"from": [a[0], a[1]], "to": [b[0], b[1]]}],
    )


def _medium(rng):
    for _ in attempts(500, "two plotted points in a general direction"):
        d = rng.choice(_DIRS_MEDIUM)
        k = rng.choice((1, 2))
        a = (rng.randint(-6, 6), rng.randint(-6, 6))
        sign = rng.choice((1, -1))
        b = (a[0] + sign * k * d[0], a[1] + sign * k * d[1])
        if max(map(abs, (*a, *b))) <= 8:
            break
    fig = _plot(a, b)
    a_s, b_s = kit.cplx(*a), kit.cplx(*b)
    stems = (
        "The points A and B on the Argand diagram represent the complex numbers a and b. A complex number z satisfies "
        r"\(\lvert z - a\rvert = \lvert z - b\rvert\). Find the Cartesian equation of the locus of z, where z = x + iy. "
        + _FORM,
        "A and B are plotted on the Argand diagram and represent the complex numbers a and b. The locus of z is "
        r"\(\lvert z - a\rvert = \lvert z - b\rvert\). Write it as a Cartesian equation, with z = x + iy. " + _FORM,
        "The diagram shows two points A and B, representing the complex numbers a and b. The points z for which "
        r"\(\lvert z - a\rvert = \lvert z - b\rvert\) lie on a line. Find the equation of that line, where z = x + iy. "
        + _FORM,
    )
    out = _line_question(
        rng,
        phrase(rng, stems),
        a,
        b,
        [f"Reading the grid, a = {a_s} and b = {b_s}.", "The locus is the perpendicular bisector of AB."],
        figure=fig,
    )
    check(
        _read_figure(type("Q", (), {"figure": fig})) == out["answer_text"],
        "the plotted points do not give the keyed line",
    )
    return out


def _read_figure(question) -> str:
    """Re-derive the keyed line from the two plotted points alone."""
    pts = {p["label"]: p["at"] for p in question.figure["points"]}
    m, c = bisector(tuple(map(int, pts["A"])), tuple(map(int, pts["B"])))
    return kit.vertical_text(c) if m is None else kit.line_text(m, c)


# -- hard ------------------------------------------------------------------------------------------------------------


def _pick_pair(rng, span=5, conj=False):
    """Two points in a general direction (so the bisector is a slanted line with modest fractions), neither on an axis."""
    for _ in attempts(2000, "two points in a general direction"):
        d = rng.choice(_DIRS_MEDIUM)
        k = rng.choice((1, 2))
        a = (rng.randint(-span, span), rng.randint(-span, span))
        sign = rng.choice((1, -1))
        b = (a[0] + sign * k * d[0], a[1] + sign * k * d[1])
        other = (b[0], -b[1]) if conj else b
        if min(abs(a[1]), abs(b[1])) > 0 and a[0] != b[0] and max(map(abs, (*a, *b))) <= 8 and a != other:
            return a, b
    raise AssertionError("unreachable")  # pragma: no cover


def _hard_conjugate(rng):
    a, b = _pick_pair(rng, conj=True)
    # |z - a| = |z-bar - b| = |z - conj(b)|: the bisector of a and conj(b)
    target = (b[0], -b[1])
    tex = rf"{kit.mod(kit.z_minus(*a))} = {kit.mod(kit.z_minus(*b, var=chr(92) + 'bar{z}'))}"
    plain = kit.plain(tex)
    extra = [(*bisector(a, b), "sr2l-conjugate-ignored")]
    stem = (
        rf"The complex number z satisfies \({tex}\). Find the Cartesian equation of the locus of z, where z = x + iy. "
        + _FORM
    )
    out = _line_question(
        rng,
        stem,
        a,
        target,
        [
            f"|z-bar - b| = |z - conj(b)|, because a complex number and its conjugate have the same modulus. Here conj(b) = {kit.cplx(*target)}.",
            f"So z is equidistant from {kit.cplx(*a)} and {kit.cplx(*target)}.",
        ],
        extra_wrong=extra,
        relation=plain,
    )
    return out


def _hard_scaled(rng):
    a, b = _pick_pair(rng, span=4)
    k = rng.choice((2, 3))
    big_a, big_b = (k * a[0], k * a[1]), (k * b[0], k * b[1])
    left = join_terms([(k, "z"), (-big_a[0], ""), (-big_a[1], "i")])
    right = join_terms([(k, "z"), (-big_b[0], ""), (-big_b[1], "i")])
    tex = rf"{kit.mod(left)} = {kit.mod(right)}"
    extra = [(*bisector(big_a, big_b), "sr2l-scale-factor-not-divided")]
    stem = (
        rf"The complex number z satisfies \({tex}\). Find the Cartesian equation of the locus of z, where z = x + iy. "
        + _FORM
    )
    return _line_question(
        rng,
        stem,
        a,
        b,
        [
            f"Divide each side by {k}: |z - ({kit.cplx(*a)})| = |z - ({kit.cplx(*b)})|.",
            "So z is equidistant from those two points, and the locus is their perpendicular bisector.",
        ],
        extra_wrong=extra,
        relation=kit.plain(tex),
    )


def _coefficient(v, lead: bool) -> str:
    """The coefficient v of z (or the constant) with its sign: ' - 5', ' + 2i', ' - (7 - 2i)'."""
    re_, im = v
    if re_ == 0 and im == 0:
        return ""
    if im == 0 or re_ == 0:
        n = re_ if im == 0 else im
        unit = "" if im == 0 else "i"
        mag = "" if abs(n) == 1 and unit else str(abs(n))
        return f" {'+' if n > 0 else '-'} {mag}{unit}"
    return f" + ({kit.cplx(re_, im)})"


def _quadratic(s, p) -> str:
    """z^2 - Sz + P = 0 with the signs resolved: z^2 - 5z + (20 - 12i) = 0, z^2 + 3iz - 4 = 0."""
    neg = (-s[0], -s[1])
    return f"z^2{_coefficient(neg, True)}z{_coefficient(p, False)} = 0"


def _hard_roots(rng):
    a, b = _pick_pair(rng, span=4)
    s = (a[0] + b[0], a[1] + b[1])
    p = complex(*a) * complex(*b)
    p = (int(p.real), int(p.imag))
    quad = _quadratic(s, p)
    # the printed quadratic is verified at both roots before it is used
    for root in (a, b):
        zr = complex(*root)
        check(abs(zr * zr - complex(*s) * zr + complex(*p)) < 1e-9, "a stated root does not satisfy the quadratic")
    stem = (
        rf"The complex numbers a and b are the roots of \({quad}\). A complex number z satisfies "
        r"\(\lvert z - a\rvert = \lvert z - b\rvert\). Find the Cartesian equation of the locus of z, where z = x + iy. "
        + _FORM
    )
    half = (Fraction(s[0], 2), Fraction(s[1], 2))
    extra = [(*kit.line_through(half, (-s[1], s[0])), "sr2l-root-sum-as-point")] if s != (0, 0) else []
    out = _line_question(
        rng,
        stem,
        a,
        b,
        [
            f"Solve the quadratic: the roots are a = {kit.cplx(*a)} and b = {kit.cplx(*b)} (their sum is {kit.cplx(*s)} and their product is {kit.cplx(*p)}).",
            "The locus is the perpendicular bisector of the segment joining the roots.",
        ],
        extra_wrong=extra,
    )
    return out


def _hard(rng):
    return rng.choice((_hard_conjugate, _hard_scaled, _hard_roots))(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("sr2l_bisector", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_bisector(rng, tier: str = "easy"):
    """The perpendicular-bisector locus as a line equation."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, _TIERS[tier])


figures.register_reader("sr2l_bisector", _read_figure)


# -- exam ------------------------------------------------------------------------------------------------------------


@generator("sr2l_bisector_exam", level=18, presets={"exam": {}})
def sr2l_bisector_exam(rng):
    return kit.redraw(rng, _exam)


def _exam(rng):
    """The locus, its x-intercept, the area it cuts off with the axes, and its distance from the origin."""
    for _ in attempts(5000, "a bisector with a negative gradient and a positive intercept"):
        d = rng.choice(_DIRS_MEDIUM)
        k = rng.choice((1, 2))
        a = (rng.randint(-5, 5), rng.randint(-5, 5))
        sign = rng.choice((1, -1))
        b = (a[0] + sign * k * d[0], a[1] + sign * k * d[1])
        m, c = bisector(a, b)
        if (
            m is not None
            and m < 0
            and c > 0
            and (c / -m * 100).denominator == 1
            and max(map(abs, (*a, *b))) <= 8
            and a != (0, 0)
            and b != (0, 0)
        ):
            break
    x0 = -c / m
    area = x0 * c / 2
    hyp2 = x0 * x0 + c * c
    dist = c / math.sqrt(1 + float(m) ** 2)
    check(
        abs(float(2 * area) / math.sqrt(float(hyp2)) - dist) < 1e-9,
        "area-over-hypotenuse and point-to-line distance disagree",
    )
    rel = _rel(a, b)
    kit_rel = kit.plain(rel)
    _verify(kit_rel, (m, c))
    stem = rf"The complex number z = x + iy satisfies \({rel}\)."
    part_a = kit.line_part(
        "Find the Cartesian equation of the locus of z, in the form y = mx + c.",
        m,
        c,
        2,
        [
            f"z is equidistant from {kit.cplx(*a)} and {kit.cplx(*b)}, so the locus is their perpendicular bisector.",
            f"The line is {kit.line_text(m, c)}.",
        ],
        [(wm, wc, tag) for wm, wc, tag in _wrong_lines(a, b, m, c)],
    )
    part_b = kit.number_part(
        "Find the value of x where the locus meets the real axis, correct to 2 decimal places.",
        x0,
        2,
        1,
        [f"Put y = 0: 0 = {kit.rt(m)}x + {kit.rt(c)}, so x = {kit.number_text(x0, 2)}."],
        [
            (c, "sr2l-intercept-for-root"),
            (-x0, "sr2l-sign-lost-solving"),
            (c / m, "sr2l-sign-lost-solving"),
            (m * c, "sr2l-intercept-for-root"),
        ],
        sensible=(0, None),
    )
    part_c = kit.number_part(
        "The locus and the two axes enclose a triangle. Find its area, correct to 2 decimal places.",
        area,
        2,
        2,
        [
            f"The triangle has base {kit.number_text(x0, 2)} on the real axis and height {kit.rt(c)} on the imaginary axis: area = 1/2 x base x height."
        ],
        [
            (x0 * c, "sr2l-triangle-half-omitted"),
            (x0 + c, "sr2l-base-and-height-added"),
            (c * c / 2, "sr2l-wrong-triangle-side"),
            (x0 * x0 / 2, "sr2l-wrong-triangle-side"),
        ],
        follow=f"part_b*{kit.rt(c)}/2",
        sensible=(0, None),
    )
    part_d = kit.number_part(
        "Find the least value of |z| for z on the locus, correct to 2 decimal places.",
        dist,
        2,
        2,
        [
            "The nearest point of the line to the origin is the foot of the perpendicular.",
            f"Hypotenuse sqrt({kit.number_text(x0, 2)}^2 + {kit.rt(c)}^2); the height to it is 2 x area / hypotenuse.",
        ],
        [
            (c, "sr2l-intercept-for-distance"),
            (x0, "sr2l-intercept-for-distance"),
            (float(hyp2) ** 0.5, "sr2l-hypotenuse-for-distance"),
            (float(area) / float(hyp2) ** 0.5, "sr2l-factor-two-omitted"),
        ],
        follow=f"2*part_c/sqrt(part_b**2 + {kit.rt(c * c)})",
        sensible=(0, None),
    )
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])
