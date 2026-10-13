"""``fr2a_box_coordinates`` and ``exd_x_fr2a_box_chain``: position and distance in a drawn box.

Skill ``f10-three-d-coordinates-position-distance`` (AC9M8SP03). A box is drawn in oblique view with its corners named
``O`` and ``P`` to ``V``. Corner ``O`` is the origin, the x-axis runs along ``OP``, the y-axis along ``OR`` and the z-axis
along ``OS``; three edges carry their lengths. The answers are labelled pairs (three values, ``x = _, y = _, z = _``) or a
number. This is not the displacement skill: the learner has to build each corner's coordinates from the lengths and the
position of the origin.

Tiers:

- easy:   the coordinates of a named corner (``Q``, ``T``, ``U`` or ``V``), the three lengths all different.
- medium: the midpoint of the line joining two named corners (an edge, a face diagonal or a space diagonal). Lengths are
          even so every coordinate of the midpoint is whole.
- hard:   the box sits at a stated point of a survey grid, not at the origin, and the question asks the straight-line
          distance between two named corners (a face diagonal or a space diagonal). The distance is whole by construction,
          and the position of the box cancels, which the learner has to see or work through.

``exd_x_fr2a_box_chain`` is the exam tier: the coordinates of the far upper corner, the midpoint of the space diagonal
from the origin to it, the length of the floor diagonal to 1 decimal place, and the space diagonal (follow-through from the
floor diagonal).

Everything is derived from the corner coordinates by `math.dist`, and again from the lengths by Pythagoras.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.figures import register_reader

from . import _fr2a_kit as kit
from .exd_x_sc3_support import number_part, redraw, tagged

_SMALL = ("cubby house", "chicken coop", "tent frame", "dog kennel", "garden storage box")
_MEDIUM = ("storage shed", "greenhouse", "shipping container", "water tank", "grain bin", "workshop")
_LARGE = ("warehouse", "aircraft hangar", "indoor sports hall", "cold store", "packing shed")


def _thing(rng, dims) -> str:
    """A structure that is plausible at this size (the longest edge in metres)."""
    longest = max(dims)
    return rng.choice(_SMALL if longest <= 6 else _MEDIUM if longest <= 12 else _LARGE)


def _corner(name, dims, base=(0, 0, 0)):
    v = kit.vertex(name)
    return tuple(base[i] + v[i] * dims[i] for i in range(3))


def _xyz(p):
    return kit.labelled_pair(x=p[0], y=p[1], z=p[2])


def _pair3(tag, p):
    text, value = _xyz(p)
    return Distractor(text, tag, value)


def _setting(rng, dims, base=None):
    thing = _thing(rng, dims)
    head = (
        f"A {thing} is modelled as a box, as shown. Distances are in metres."
        if base is None
        else f"A {thing} is modelled as a box, as shown. Distances are in metres, and the box is on a survey grid."
    )
    origin = (
        "Corner O is the origin, with the x-axis along OP, the y-axis along OR and the z-axis along OS (upwards)."
        if base is None
        else f"Corner O is at the point {_triple(base)} on the grid; the x-axis runs along OP, the y-axis along OR and "
        "the z-axis along OS (upwards)."
    )
    return f"{head} {origin}"


def _triple(p) -> str:
    return "(" + ", ".join(str(int(v)) for v in p) + ")"


def _fig(dims, base=None):
    return kit.box_figure(
        dims,
        alt="A box drawn in perspective with its corners named O and P to V and three edge lengths labelled.",
    )[0]


def _finish_pair(text, answer, cands, working, fig):
    ans_text, ans = _xyz(answer)
    check(not kit.leaks(fig, ans_text), "the figure shows the answer")
    dis = pick_distractors(ans_text, [c for c in cands if c.value != ans], 3, tolerance=GRADER_TOLERANCE)
    check(len(dis) >= 3, f"only {len(dis)} distractors for {ans_text}")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        answer_kind="pair",
        working=working,
        distractors=dis,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ easy


def _easy(rng):
    a, b, c = rng.sample(range(2, 10), 3)
    dims = (a, b, c)
    name = rng.choice("QTUV")
    p = _corner(name, dims)
    fig = _fig(dims)
    v = kit.vertex(name)
    check(p == tuple(d * f for d, f in zip(dims, v)), "corner differs between the two routes")
    cands = [
        _pair3("swapped-coordinates", (p[1], p[0], p[2])),
        _pair3("swapped-coordinates", (p[0], p[2], p[1])),
        _pair3("fr2a-box-corner-all-lengths", dims),
    ]
    cands += [
        _pair3("fr2a-box-length-left-out", tuple(0 if k == i else x for k, x in enumerate(p))) for i in range(3) if p[i]
    ]
    text = f"{_setting(rng, dims)} Write the coordinates of {name} as x = _, y = _, z = _."
    along = [
        f"{length} m along {axis}" if f else f"0 along {axis}" for length, axis, f in zip(dims, ("x", "y", "z"), v)
    ]
    working = [
        f"OP = {a} m is along x, OR = {b} m is along y and OS = {c} m is up (z).",
        f"{name} is {along[0]}, {along[1]} and {along[2]}, so {name} = {_triple(p)}.",
    ]
    return _finish_pair(text, p, cands, working, fig)


# ------------------------------------------------------------------------------------------ medium


def _medium(rng):
    for _ in attempts(500, "even lengths all different"):
        a, b, c = rng.sample(range(2, 17, 2), 3)
        break
    dims = (a, b, c)
    names = [n for n in "PQRSTUV"]
    for _ in attempts(500, "two corners joined by an edge, a face diagonal or a space diagonal"):
        u, w = rng.sample(names + ["O"], 2)
        if "O" in (u, w):
            continue
        distinct = sum(x != y for x, y in zip(kit.vertex(u), kit.vertex(w)))
        if distinct >= 1:
            break
    pu, pw = _corner(u, dims), _corner(w, dims)
    mid = tuple((x + y) // 2 for x, y in zip(pu, pw))
    check(all((x + y) % 2 == 0 for x, y in zip(pu, pw)), "the midpoint is not whole")
    check(
        tuple(Fraction(x + y, 2) for x, y in zip(pu, pw)) == tuple(Fraction(m) for m in mid), "midpoint routes differ"
    )
    fig = _fig(dims)
    cands = [
        _pair3("fr2a-midpoint-not-halved", tuple(x + y for x, y in zip(pu, pw))),
        _pair3("fr2a-midpoint-difference-halved", tuple(abs(y - x) // 2 for x, y in zip(pu, pw))),
        _pair3("fr2a-midpoint-is-an-endpoint", pu),
        _pair3("fr2a-midpoint-is-an-endpoint", pw),
        _pair3("fr2a-midpoint-one-coordinate-halved", (mid[0], pu[1] + pw[1], pu[2] + pw[2])),
        _pair3("swapped-coordinates", (mid[1], mid[0], mid[2])),
    ]
    text = phrase(
        rng,
        (
            f"{_setting(rng, dims)} Find the midpoint of the line joining {u} and {w}. Write it as x = _, y = _, z = _.",
            f"{_setting(rng, dims)} M is the midpoint of the line joining {u} and {w}. Write the coordinates of M as "
            "x = _, y = _, z = _.",
        ),
    )
    working = [
        f"{u} = {_triple(pu)} and {w} = {_triple(pw)}.",
        "Add the coordinates of the two ends and halve each: "
        + ", ".join(f"({x} + {y}) ÷ 2 = {m}" for x, y, m in zip(pu, pw, mid))
        + ".",
    ]
    return _finish_pair(text, mid, cands, working, fig)


# ------------------------------------------------------------------------------------------ hard

_SPACE = [("O", "U"), ("P", "V"), ("Q", "S"), ("R", "T")]


def _double_triples(limit=30):
    """(a, b, c, h, d): a^2 + b^2 = h^2 and h^2 + c^2 = d^2, all whole, d at most ``limit``."""
    out = []
    for a in range(3, limit):
        for b in range(a, limit):
            h = math.isqrt(a * a + b * b)
            if h * h != a * a + b * b:
                continue
            for c in range(2, limit):
                d = math.isqrt(h * h + c * c)
                if d * d == h * h + c * c and d <= limit:
                    out.append((a, b, c, h, d))
    return tuple(out)


_DOUBLES = _double_triples()


def _hard(rng):
    base = (rng.randint(5, 40), rng.randint(5, 40), rng.randint(0, 6))
    a, b, c, h, d = rng.choice(_DOUBLES)
    if rng.random() < 0.5:
        a, b = b, a
    dims = (a, b, c)
    u, w = rng.choice(_SPACE)
    pu, pw = _corner(u, dims, base), _corner(w, dims, base)
    flat = math.hypot(pu[0] - pw[0], pu[1] - pw[1])
    gap = math.dist(pu, pw)
    check(flat == h and gap == d, "distances differ between the coordinate route and Pythagoras")
    fig = _fig(dims)
    ans_text, ans = kit.labelled_pair(floor=h, straight=d)
    check(not kit.leaks(fig, ans_text), "the figure shows the answer")

    def pair(tag, f, s_):
        text, value = kit.labelled_pair(floor=f, straight=s_)
        return Distractor(text, tag, value)

    cands = [
        pair("adds-instead-of-using-pythagoras", a + b, a + b + c),
        pair("adds-instead-of-using-pythagoras", h, h + c),
        pair("fr2a-distance-one-dimension-left-out", h, h),
        pair("forgets-the-square-root", h * h, d * d),
        pair("fr2a-height-not-squared", h, h * h + c),
        pair("fr2a-distances-swapped", d, h),
        pair("fr2a-floor-distance-from-one-edge", max(a, b), d),
    ]
    dis = pick_distractors(ans_text, [x for x in cands if x.value != ans], 3, tolerance=GRADER_TOLERANCE)
    check(len(dis) >= 3, "too few distractors")
    text = (
        f"{_setting(rng, dims, base)} Corners {u} and {w} are at opposite ends of a diagonal through the box. "
        f"How far apart are {u} and {w} horizontally, and in a straight line, in metres? "
        "Answer as floor = _, straight = _."
    )
    dx, dy, dz = (abs(x - y) for x, y in zip(pu, pw))
    working = [
        f"{u} = {_triple(pu)} and {w} = {_triple(pw)}.",
        f"The differences in x, y and z are {dx}, {dy} and {dz} (where the box sits on the grid does not matter).",
        f"Horizontally: √({dx}² + {dy}²) = √{dx * dx + dy * dy} = {h} m.",
        f"Straight line: √({h}² + {dz}²) = √{h * h + dz * dz} = {d} m.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        answer_kind="pair",
        working=working,
        distractors=dis,
        figure=fig,
    )


@generator(
    "fr2a_box_coordinates",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_box_coordinates(rng, tier="easy"):
    """Coordinates, midpoints and distances for corners of a drawn box."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return redraw(
        {"easy": _easy, "medium": _medium, "hard": _hard}[tier],
        rng,
        "a box question with enough distinct wrong answers",
    )


# ------------------------------------------------------------------------------------------ the figure reader


def _read_box(question):
    """Re-derive the answer from the lengths drawn on the box's edges (and the corner names the stem asks about)."""
    fig, text = question.figure, question.text
    length = {}
    for seg in fig["segments"]:
        m = re.fullmatch(r"(\d+) m", seg.get("label", ""))
        if m:
            length[frozenset((seg["from"], seg["to"]))] = int(m[1])
    dims = (length[frozenset("OP")], length[frozenset("OR")], length[frozenset("OS")])
    base = (0, 0, 0)
    mb = re.search(r"Corner O is at the point \((\d+), (\d+), (\d+)\)", text)
    if mb:
        base = tuple(int(v) for v in mb.groups())
    if "floor = _" in text:
        u, w = re.search(r"Corners ([A-V]) and ([A-V])", text).groups()
        pu, pw = _corner(u, dims, base), _corner(w, dims, base)
        return kit.labelled_pair(
            floor=round(math.hypot(pu[0] - pw[0], pu[1] - pw[1])), straight=round(math.dist(pu, pw))
        )[0]
    if "midpoint" in text:
        u, w = re.search(r"joining ([A-V]) and ([A-V])", text).groups()
        pu, pw = _corner(u, dims, base), _corner(w, dims, base)
        return _xyz(tuple((x + y) // 2 for x, y in zip(pu, pw)))[0]
    name = re.search(r"coordinates of ([A-V]) as", text)[1]
    return _xyz(_corner(name, dims, base))[0]


register_reader("fr2a_box_coordinates", _read_box)


# ------------------------------------------------------------------------------------------ the exam chain


@generator("exd_x_fr2a_box_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_fr2a_box_chain(rng, tier="exam"):
    """Exam tier: far upper corner, midpoint of the space diagonal, floor diagonal, space diagonal."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_fr2a_box_chain, rng, "a chain with enough distinct wrong answers")


def _fr2a_box_chain(rng):
    for _ in attempts(500, "a box whose chain follows through within one decimal place"):
        a, b, c = rng.sample(range(2, 17, 2), 3)
        floor = math.hypot(a, b)
        space = math.sqrt(a * a + b * b + c * c)
        floor_r = round(floor + 1e-12, 1)
        space_r = round(space + 1e-12, 1)
        if abs(math.sqrt(floor_r**2 + c * c) - space_r) <= 0.1 and floor != int(floor):
            break
    dims = (a, b, c)
    u = _corner("U", dims)
    mid = tuple(x // 2 for x in u)
    check(math.dist((0, 0, 0), u) == space and math.dist((0, 0, 0), _corner("Q", dims)) == floor, "lengths disagree")
    fig = _fig(dims)
    thing = _thing(rng, dims)
    stem = (
        f"A {thing} is modelled as a box, as shown. Corner O is the origin, with the x-axis along OP, the y-axis along OR "
        "and the z-axis along OS (upwards). Distances are in metres."
    )
    a_text, a_val = _xyz(u)
    part_a = dict(
        prompt="Write the coordinates of U, the corner opposite O, as x = _, y = _, z = _.",
        answer_kind="pair",
        answer=a_val,
        answer_text=a_text,
        answer_latex=latex_answer(a_text, plain=True),
        marks=2,
        working=[f"U is {a} m along x, {b} m along y and {c} m up, so U = {_triple(u)}."],
        distractors=pick_distractors(
            a_text,
            [
                c_
                for c_ in [
                    _pair3("swapped-coordinates", (b, a, c)),
                    _pair3("swapped-coordinates", (a, c, b)),
                    _pair3("fr2a-box-length-left-out", (a, b, 0)),
                    _pair3("fr2a-box-lengths-added", (a + b, c, 0)),
                ]
                if c_.value != a_val
            ],
            3,
            tolerance=GRADER_TOLERANCE,
        ),
    )
    m_text, m_val = _xyz(mid)
    part_b = dict(
        prompt="M is the midpoint of the line joining O and U. Write the coordinates of M as x = _, y = _, z = _.",
        answer_kind="pair",
        answer=m_val,
        answer_text=m_text,
        answer_latex=latex_answer(m_text, plain=True),
        marks=2,
        working=[f"Halve each coordinate of U {_triple(u)}: M = {_triple(mid)}."],
        distractors=pick_distractors(
            m_text,
            [
                c_
                for c_ in [
                    _pair3("fr2a-midpoint-not-halved", u),
                    _pair3("fr2a-midpoint-is-an-endpoint", (0, 0, 0)),
                    _pair3("fr2a-midpoint-one-coordinate-halved", (mid[0], u[1], u[2])),
                    _pair3("swapped-coordinates", (mid[1], mid[0], mid[2])),
                ]
                if c_.value != m_val
            ],
            3,
            tolerance=GRADER_TOLERANCE,
        ),
    )
    fl = Fraction(floor_r).limit_denominator(10)
    f_cands = [
        (Fraction(a + b), "adds-instead-of-using-pythagoras"),
        (Fraction(a * a + b * b), "forgets-the-square-root"),
        (Fraction(round(math.hypot(a, c) + 1e-12, 1)).limit_denominator(10), "fr2a-wrong-face-diagonal"),
        (Fraction(max(a, b)), "fr2a-longest-edge-for-distance"),
    ]
    part_c = number_part(
        "Find the length of the diagonal OQ across the floor, correct to 1 decimal place.",
        fl,
        1,
        2,
        [f"OQ = √({a}² + {b}²) = √{a * a + b * b} = {floor_r:.1f} m."],
        tagged(fl, f_cands, 1, lo=0),
        sensible_range=(0, None),
    )
    sp = Fraction(space_r).limit_denominator(10)
    s_cands = [
        (Fraction(round(a + b + c, 1)), "adds-instead-of-using-pythagoras"),
        (Fraction(round(math.sqrt(floor_r**2 + c), 1)).limit_denominator(10), "fr2a-height-not-squared"),
        (Fraction(round(floor_r + c, 1)).limit_denominator(10), "adds-instead-of-using-pythagoras"),
        (Fraction(round(math.hypot(floor_r, a), 1)).limit_denominator(10), "fr2a-wrong-face-diagonal"),
    ]
    part_d = number_part(
        "Find the straight-line distance from O to U, correct to 1 decimal place.",
        sp,
        1,
        2,
        [f"OU = √(OQ² + {c}²) = √({floor_r:.1f}² + {c}²) = {space_r:.1f} m (or √({a}² + {b}² + {c}²))."],
        tagged(sp, kit.far_from(sp, s_cands, Fraction(1, 10)), 1, lo=0),
        follow_through=f"sqrt(part_c**2+{c}**2)",
        sensible_range=(0, None),
    )
    parts = [part_a, part_b, part_c, part_d]
    kit.require_distractors(parts)
    return dict(text=stem, figure=fig, parts=parts)
