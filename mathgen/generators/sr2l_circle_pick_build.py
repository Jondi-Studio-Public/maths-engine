"""Circles and discs in the Argand plane, read from a diagram: `sr2l_circle_pick` (easy, medium, hard) and
`sr2l_circle_pick_exam`.

Skill: locus-circle-disc-pick-sketch. Every question is multiple choice and every tier is a different decision. A figure
cannot be shaded or dashed, so the diagram carries what a drawing can (circles and points on a grid) and the stem
carries the rest in words.

* easy: a circle is drawn; choose the equation |z - w| = r of the locus. The wrong options are the three classic
  misreadings: the centre's signs flipped, the real and imaginary parts swapped, and the diameter taken as the radius.
* medium: four labelled points are plotted and a disc inequality (<, <=, > or >=) is stated. Decide which points
  satisfy it. A point sits exactly on the circle, so the strict and non-strict forms differ.
* hard: two concentric circles are drawn, and the stem names which of the two boundaries belongs to the region
  between them. Choose the double inequality r1 <= |z - w| < r2 (or its variants) for that region.
* exam: a circle through a given point. The radius, the greatest and least |z| over the disc, and which of four
  complex numbers lies inside it.

Each figure is read back from the drawn data alone by `_read_figure` (registered for the easy and hard tiers), and
medium's key set is recomputed from the plotted points.
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.core import attempts, check, generator, latex_answer, phrase

from . import _sr2l_kit as kit

_ALT_CIRCLE = "An Argand diagram on a square grid: a circle, with the real axis across and the imaginary axis up."
_ALT_RINGS = (
    "An Argand diagram on a square grid: two concentric circles, with the real axis across and the imaginary axis up."
)
_ALT_POINTS = "An Argand diagram on a square grid with four labelled points A, B, C and D."

_EASY_STEMS = (
    "The circle drawn in the Argand diagram is the locus of the complex number z. Which equation describes it?",
    "The diagram shows the locus of z as a circle in the complex plane. Which equation is satisfied by the points of "
    "the circle?",
    "A circle is drawn on the Argand diagram. Which of these equations has exactly that circle as its locus?",
)


# -- easy ------------------------------------------------------------------------------------------------------------


def _circle_figure(p, q, r):
    return kit.plane(lo=-8, hi=8, alt=_ALT_CIRCLE, curves=[kit.circle_curve(p, q, r)], points=[{"at": [p, q]}])


def _read_circle(fig):
    """Centre and radius read back from a drawn circle."""
    pts = fig["curves"][0]["pieces"][0][:-1]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    r = math.hypot(pts[0][0] - cx, pts[0][1] - cy)
    return round(cx), round(cy), round(r)


def _easy(rng):
    for _ in attempts(500, "a circle with distinguishable misreadings"):
        p, q = rng.randint(-4, 4), rng.randint(-4, 4)
        r = rng.randint(2, 4)
        if (p, q) != (0, 0) and p != q and p != -q and p != 0 and q != 0:
            break
    fig = _circle_figure(p, q, r)
    cx, cy, rr = _read_circle(fig)
    check((cx, cy, rr) == (p, q, r), "the figure does not show the keyed circle")
    key = kit.circle_plain(p, q, r)
    wrong = [
        (kit.circle_plain(-p, -q, r), "sr2l-centre-sign-flipped"),
        (kit.circle_plain(q, p, r), "sr2l-real-imaginary-swapped"),
        (kit.circle_plain(p, q, 2 * r), "sr2l-diameter-for-radius"),
        (kit.circle_plain(-q, -p, r), "sr2l-real-imaginary-swapped"),
    ]
    fields = kit.mcq_fields(key, wrong, key_latex=kit.circle_rel(p, q, r))
    return dict(
        **kit.prompt(phrase(rng, _EASY_STEMS)),
        figure=fig,
        working=[
            f"The centre is at ({p}, {q}), which is the complex number {kit.cplx(p, q)}, and the radius is {r}.",
            f"So |z - ({kit.cplx(p, q)})| = {r}, which is {key}.",
        ],
        **fields,
    )


# -- medium ----------------------------------------------------------------------------------------------------------

_OPS = {
    "<": lambda d2, r2: d2 < r2,
    "≤": lambda d2, r2: d2 <= r2,
    ">": lambda d2, r2: d2 > r2,
    "≥": lambda d2, r2: d2 >= r2,
}
_TEX_OP = {"<": "<", "≤": r"\le", ">": ">", "≥": r"\ge"}
_FLIP_BOUNDARY = {"<": "≤", "≤": "<", ">": "≥", "≥": ">"}
_COMPLEMENT = {"<": "≥", "≤": ">", ">": "≤", "≥": "<"}
_LABELS = "ABCD"


def _names(subset):
    names = [_LABELS[i] for i in sorted(subset)]
    if not names:
        return "None of the points"
    if len(names) == 4:
        return "All four points"
    if len(names) == 1:
        return f"{names[0]} only"
    return f"{', '.join(names[:-1])} and {names[-1]} only"


def _d2(pt, c):
    return (pt[0] - c[0]) ** 2 + (pt[1] - c[1]) ** 2


_MEDIUM_STEMS = (
    "The points A, B, C and D are plotted on the Argand diagram. Which of them satisfy {rel}?",
    "Four points are plotted in the complex plane. Which of the points A, B, C and D satisfy {rel}?",
    "The complex numbers represented by A, B, C and D are plotted. Which of the points satisfy {rel}?",
)


def _medium(rng):
    for _ in attempts(4000, "four points around a circle with distinct wrong sets"):
        c = (rng.randint(-2, 2), rng.randint(-2, 2))
        r = rng.choice((5, 5, 10))
        op = rng.choice(list(_OPS))
        v = rng.choice([w for w in kit.PYTH if math.hypot(*w) == r])
        on = (c[0] + v[0], c[1] + v[1])
        pts = [on]
        for _k in range(3):
            pts.append((rng.randint(-10, 10), rng.randint(-10, 10)))
        if len(set(pts)) < 4 or any(abs(x) > 10 or abs(y) > 10 for x, y in pts):
            continue
        d2s = [_d2(pt, c) for pt in pts]
        r2 = r * r
        if not (any(d < r2 for d in d2s) and any(d > r2 for d in d2s)):
            continue
        near_edge = [d for d in d2s if abs(math.sqrt(d) - r) < 0.45 and d != r2]
        if near_edge:  # a point a hair off the circle cannot be told from the circle by eye
            continue
        rng.shuffle(pts)
        d2s = [_d2(pt, c) for pt in pts]
        key = {i for i, d in enumerate(d2s) if _OPS[op](d, r2)}
        flipped = {i for i, pt in enumerate(pts) if _OPS[op](_d2(pt, (-c[0], -c[1])), r2)}
        compl = {i for i in range(4) if i not in key}
        boundary = {i for i, d in enumerate(d2s) if _OPS[_FLIP_BOUNDARY[op]](d, r2)}
        squared = {i for i, d in enumerate(d2s) if _OPS[op](d, r2 * r2)}
        cand = [
            (compl, "sr2l-inside-outside-reversed"),
            (flipped, "sr2l-centre-sign-flipped"),
            (boundary, "sr2l-boundary-included-or-excluded"),
            (squared, "sr2l-radius-squared"),
        ]
        wrong = [(_names(s), tag) for s, tag in cand if s != key]
        if 1 <= len(key) <= 3 and len({t for t, _ in wrong}) >= 3 and _names(key) not in {t for t, _ in wrong}:
            break
    rel = kit.circle_rel(c[0], c[1], r, _TEX_OP[op])
    fig = kit.plane(
        lo=-10,
        hi=10,
        alt=_ALT_POINTS,
        points=[{"at": [pt[0], pt[1]], "label": _LABELS[i]} for i, pt in enumerate(pts)],
    )
    shown = figures.check(fig)["points"]
    read = {i for i, p in enumerate(shown) if _OPS[op](_d2(p["at"], c), r2)}
    check(read == key, "the plotted points do not give the keyed set")
    fields = kit.mcq_fields(_names(key), wrong, key_latex=latex_answer(_names(key), plain=True))
    inside = [_LABELS[i] for i in range(4) if d2s[i] < r2]
    return dict(
        **kit.prompt(phrase(rng, _MEDIUM_STEMS, rel=kit.maths(rel))),
        figure=fig,
        working=[
            f"The centre is ({c[0]}, {c[1]}) and the radius is {r}. A point satisfies the relation when its distance "
            f"from the centre compares with {r} as stated.",
            "Distances squared from the centre: "
            + ", ".join(f"{_LABELS[i]} {d2s[i]}" for i in range(4))
            + f" (compare with {r2}).",
            f"Inside the circle: {', '.join(inside) or 'none'}; on it: "
            f"{', '.join(_LABELS[i] for i in range(4) if d2s[i] == r2) or 'none'}.",
        ],
        **fields,
    )


# -- hard ------------------------------------------------------------------------------------------------------------

_BOUNDS = {
    ("≤", "<"): "including the inner circle but not the outer circle",
    ("<", "≤"): "including the outer circle but not the inner circle",
    ("≤", "≤"): "including both circles",
    ("<", "<"): "not including either circle",
}
_HARD_STEMS = (
    "The diagram shows two concentric circles. Which inequality describes the points of the region between them, {words}?",
    "Two concentric circles are drawn in the complex plane. Which statement describes z lying between the circles, "
    "{words}?",
    "Which condition on z gives the region between the two circles in the diagram, {words}?",
)


def _band(p, q, r1, o1, r2, o2):
    mid = kit.mod(kit.z_minus(p, q))
    return rf"{r1} {_TEX_OP[o1]} {mid} {_TEX_OP[o2]} {r2}"


def _rings_figure(p, q, r1, r2):
    return kit.plane(
        lo=-9,
        hi=9,
        alt=_ALT_RINGS,
        curves=[kit.circle_curve(p, q, r1), kit.circle_curve(p, q, r2)],
        points=[{"at": [p, q]}],
    )


def _read_rings(fig):
    (cx, cy, r1), (_, _, r2) = (_read_circle({"curves": [c]}) for c in fig["curves"])
    return cx, cy, min(r1, r2), max(r1, r2)


def _hard(rng):
    for _ in attempts(500, "two concentric circles that fit the plane"):
        p, q = rng.randint(-3, 3), rng.randint(-3, 3)
        r1 = rng.randint(1, 3)
        r2 = r1 + rng.randint(1, 3)
        if p != 0 and q != 0 and p != q and p != -q and max(abs(p), abs(q)) + r2 <= 8:
            break
    o1, o2 = rng.choice(list(_BOUNDS))
    fig = _rings_figure(p, q, r1, r2)
    check(_read_rings(fig) == (p, q, r1, r2), "the figure does not show the keyed circles")

    def flip(o):
        return _FLIP_BOUNDARY[o]

    key = kit.plain(_band(p, q, r1, o1, r2, o2))
    wrong = [
        (kit.plain(_band(p, q, r1, flip(o1), r2, flip(o2))), "sr2l-boundary-included-or-excluded"),
        (kit.plain(_band(-p, -q, r1, o1, r2, o2)), "sr2l-centre-sign-flipped"),
        (
            kit.plain(rf"{kit.mod(kit.z_minus(p, q))} {_TEX_OP[o2]} {r2}"),
            "sr2l-inner-circle-ignored",
        ),
        (kit.plain(_band(p, q, r1, o1, r2 - r1, o2)), "sr2l-difference-of-radii"),
    ]
    fields = kit.mcq_fields(key, wrong, key_latex=_band(p, q, r1, o1, r2, o2))
    return dict(
        **kit.prompt(phrase(rng, _HARD_STEMS, words=_BOUNDS[(o1, o2)])),
        figure=fig,
        working=[
            f"Both circles have centre ({p}, {q}), the complex number {kit.cplx(p, q)}; their radii are {r1} and {r2}.",
            f"Points between the circles have a distance from the centre between {r1} and {r2}; {_BOUNDS[(o1, o2)]}.",
            f"So {key}.",
        ],
        **fields,
    )


_REL_RE = re.compile(r"\|([^|]+)\|\s*(<|≤|>|≥)\s*(\d+)")


def _centre_of(inner: str):
    """The centre (x, y) of |inner| from an expression such as `z + 2 - 3i`: evaluate it at z = 0 and negate."""
    expr = re.sub(r"(\d+)i", r"\1j", inner)
    expr = re.sub(r"(?<![\w])i(?!\w)", "1j", expr).replace("z", "0")
    check(re.fullmatch(r"[\d\s+\-j.]+", expr) is not None, f"unreadable expression {inner!r}")
    w = -eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - digits, signs and j only
    return round(w.real), round(w.imag)


def _read_figure(question) -> str:
    """Re-derive the keyed option from the drawn figure (and, for what a drawing cannot show, the stem's words)."""
    fig = question.figure
    if fig["points"] and len(fig["points"]) == 4 and not fig["curves"]:
        inner, op, radius = _REL_RE.search(question.text).groups()
        c, r = _centre_of(inner), int(radius)
        shown = {pt["label"]: pt["at"] for pt in fig["points"]}
        return _names({i for i, name in enumerate(_LABELS) if _OPS[op](_d2(shown[name], c), r * r)})
    if len(fig["curves"]) == 2:
        p, q, r1, r2 = _read_rings(fig)
        ops = next(k for k, words in _BOUNDS.items() if words in question.text)
        return kit.plain(_band(p, q, r1, ops[0], r2, ops[1]))
    p, q, r = _read_circle(fig)
    return kit.circle_plain(p, q, r)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("sr2l_circle_pick", level=18, presets={t: {"tier": t} for t in _TIERS})
def sr2l_circle_pick(rng, tier: str = "easy"):
    """Read a circle, a set of points or a pair of concentric circles from a diagram and choose the relation."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return kit.redraw(rng, _TIERS[tier])


# -- exam ------------------------------------------------------------------------------------------------------------

_EXAM_STEM = (
    r"A circle in the complex plane has centre {c} and passes through the point {w}. Let D be the disc "
    r"\(\lvert z - c\rvert \le r\) bounded by the circle, where \(c = {cc}\) and r is the radius."
)


def _exam_numbers(rng):
    for _ in attempts(2000, "a circle through a point with whole-number lengths"):
        cv = rng.choice([v for v in kit.PYTH if math.hypot(*v) >= 10])
        dv = rng.choice([v for v in kit.PYTH if math.hypot(*v) < math.hypot(*cv)])
        return cv, dv
    raise AssertionError("unreachable")  # pragma: no cover


@generator("sr2l_circle_pick_exam", level=18, presets={"exam": {}})
def sr2l_circle_pick_exam(rng):
    return kit.redraw(rng, _exam)


def _exam(rng):
    """A circle through a point: radius, greatest and least |z| over the disc, then which number lies inside it."""
    cv, dv = _exam_numbers(rng)
    c = cv
    w = (c[0] + dv[0], c[1] + dv[1])
    r = int(math.hypot(*dv))
    mc = int(math.hypot(*c))
    check(mc > r, "the origin must lie outside the disc")
    stem = _EXAM_STEM.format(c=kit.cplx(*c), w=kit.cplx(*w), cc=kit.cplx(*c))
    dx, dy = dv
    part_a = kit.number_part(
        r"Find the radius r of the circle.",
        r,
        0,
        1,
        [f"r = |w - c| = |{kit.cplx(dx, dy)}| = sqrt({dx * dx} + {dy * dy}) = {r}."],
        [
            (abs(dx) + abs(dy), "sr2l-components-added"),
            (r * r, "sr2l-radius-squared"),
            (max(abs(dx), abs(dy)), "sr2l-larger-component-only"),
            (math.hypot(*w), "sr2l-modulus-of-point-not-distance"),
        ],
        sensible=(0, None),
    )
    part_b = kit.number_part(
        r"Find the greatest value of \(\lvert z\rvert\) for z in D.",
        mc + r,
        0,
        2,
        [f"The point of D furthest from the origin lies on the line through 0 and c: |c| + r = {mc} + {r} = {mc + r}."],
        [
            (mc - r, "sr2l-greatest-and-least-confused"),
            (r, "sr2l-radius-for-modulus"),
            (mc, "sr2l-centre-modulus-only"),
            (mc + 2 * r, "sr2l-diameter-for-radius"),
        ],
        follow=f"{mc} + part_a",
        sensible=(0, None),
    )
    part_c = kit.number_part(
        r"Find the least value of \(\lvert z\rvert\) for z in D.",
        mc - r,
        0,
        2,
        [f"The origin lies outside D, so the nearest point is |c| - r = {mc} - {r} = {mc - r}."],
        [
            (mc + r, "sr2l-greatest-and-least-confused"),
            (r - mc, "sr2l-subtracted-the-wrong-way"),
            (mc, "sr2l-centre-modulus-only"),
            (r, "sr2l-radius-for-modulus"),
        ],
        follow=f"{mc} - part_a",
        sensible=(0, None),
    )
    # which number lies strictly inside the open disc |z - c| < r
    lim = mc + r + 2
    inside, on, mirror, origin_disc = [], [], [], []
    for x in range(-lim, lim + 1):
        for y in range(-lim, lim + 1):
            if y == 0:
                continue  # a bare whole number would be read as an option's position, not its text
            d2 = (x - c[0]) ** 2 + (y - c[1]) ** 2
            if d2 < r * r and (x, y) != c:
                inside.append((x, y))
            elif d2 == r * r:
                on.append((x, y))
            elif (x + c[0]) ** 2 + (y + c[1]) ** 2 < r * r:
                mirror.append((x, y))
            elif x * x + y * y < r * r:
                origin_disc.append((x, y))
    check(inside and on and mirror and origin_disc, "no candidates for the membership question")
    picks = {
        "key": rng.choice(inside),
        "on": rng.choice(on),
        "mirror": rng.choice(mirror),
        "origin": rng.choice(origin_disc),
    }
    key = kit.cplx(*picks["key"])
    fields = kit.mcq_fields(
        key,
        [
            (kit.cplx(*picks["on"]), "sr2l-boundary-included-or-excluded"),
            (kit.cplx(*picks["mirror"]), "sr2l-centre-sign-flipped"),
            (kit.cplx(*picks["origin"]), "sr2l-modulus-compared-with-radius"),
        ],
    )
    part_d = dict(
        **kit.part_prompt(rf"Which of these complex numbers satisfies \(\lvert z - c\rvert < {r}\)?"),
        marks=1,
        working=[
            f"The distance from {key} to c is less than {r}; the others are on the circle, or measured from the wrong centre."
        ],
        **fields,
    )
    return dict(**kit.prompt(stem), parts=[part_a, part_b, part_c, part_d])


figures.register_reader("sr2l_circle_pick", _read_figure)
