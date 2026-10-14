"""``fr2a_transform_image`` and ``exd_x_fr2a_transform_chain``: the image of a drawn triangle after a stated move.

Skill ``f10-transform-image-coordinates`` (AC9M5SP03, AC9M7SP03). A triangle ABC is drawn on a coordinate grid. The
stem names a move, or two moves in order, and the learner writes the coordinates of the image of one named corner
as a labelled pair, ``x = 3, y = -2``. The image is not drawn, only the grid it lands on, so the corners of ABC
must be read from the picture (the stem never lists them).

Tiers:

- easy:   one translation (both ways at once), or a reflection in the x-axis or the y-axis.
- medium: a reflection in a line ``x = k`` or ``y = k`` away from the axes (the line is drawn), a half turn about the
          origin or about another point, or a quarter turn clockwise or anticlockwise about the origin or a point.
- hard:   two moves in order (a reflection or turn first, then a translation or another reflection). The order
          matters: the image under the moves taken the other way round is always a different point.

``exd_x_fr2a_transform_chain`` is the exam tier: a right-angled triangle is reflected and then translated; the
learner gives the first image of A, the final image of A, the length of the longest side of the final triangle and
the perimeter of the final triangle (follow-through from the longest side).

Every image is found by coordinate arithmetic and again by complex-number arithmetic (``_fr2a_kit.image_of``).
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.figures import register_reader

from . import _fr2a_kit as kit
from .exd_x_sc3_support import redraw

_NAMES = ("A", "B", "C")
_LIMIT = 11  # every corner and image stays within this many units of the origin


# ------------------------------------------------------------------------------------------ drawing a triangle


def _triangle(rng, spread: int):
    """Three corners, scalene, not in a line, so no move leaves the picture looking the same."""
    for _ in attempts(500, "a scalene triangle"):
        pts = [(rng.randint(-spread, spread), rng.randint(-spread, spread)) for _ in range(3)]
        if len(set(pts)) < 3:
            continue
        (ax, ay), (bx, by), (cx, cy) = pts
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) == 0:
            continue
        sides = {(ax - bx) ** 2 + (ay - by) ** 2, (bx - cx) ** 2 + (by - cy) ** 2, (cx - ax) ** 2 + (cy - ay) ** 2}
        if len(sides) == 3:
            return pts
    raise AssertionError("unreachable")


def _on_one_side(pts, step) -> bool:
    """A mirror line must miss the triangle (all corners strictly on one side)."""
    if step[0] == "RV":
        side = {(p[0] > step[1]) - (p[0] < step[1]) for p in pts}
    elif step[0] == "RH":
        side = {(p[1] > step[1]) - (p[1] < step[1]) for p in pts}
    else:
        return True
    return len(side) == 1 and 0 not in side


# ------------------------------------------------------------------------------------------ choosing the moves


def _easy_step(rng):
    kind = rng.choice(("T", "T", "RX", "RY"))
    if kind == "T":
        dx = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        dy = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        return ("T", dx, dy)
    return (kind,)


def _centre(rng, allow_origin=True):
    if allow_origin and rng.random() < 0.4:
        return (0, 0)
    return (rng.randint(-4, 4), rng.randint(-4, 4))


def _medium_step(rng):
    kind = rng.choice(("RV", "RH", "H", "Q", "Q"))
    if kind in ("RV", "RH"):
        return (kind, rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
    cx, cy = _centre(rng)
    if kind == "H":
        return ("H", cx, cy)
    return ("Q", cx, cy, rng.choice(("clockwise", "anticlockwise")))


def _hard_steps(rng):
    first = rng.choice(
        [("RX",), ("RY",), ("RV", rng.choice([-3, -2, 2, 3])), ("RH", rng.choice([-3, -2, 2, 3])), ("H",), ("Q",)]
    )
    if first[0] == "H":
        cx, cy = _centre(rng)
        first = ("H", cx, cy)
    elif first[0] == "Q":
        cx, cy = _centre(rng)
        first = ("Q", cx, cy, rng.choice(("clockwise", "anticlockwise")))
    if rng.random() < 0.6:
        second = ("T", rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]), rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]))
    else:
        second = rng.choice([("RX",), ("RY",), ("RV", rng.choice([-3, -1, 1, 3])), ("RH", rng.choice([-3, -1, 1, 3]))])
    return [first, second]


# ------------------------------------------------------------------------------------------ distractors


def _slips(steps, p, answer):
    """(point, misconception id) for each specific slip, in priority order."""
    x, y = p
    out = []
    if len(steps) == 2:
        first, second = steps
        out.append((kit.image_of([second, first], p), "fr2a-steps-in-wrong-order"))
        out.append((kit.apply_step(first, p), "fr2a-second-step-left-out"))
        out.append((kit.apply_step(second, p), "fr2a-first-step-left-out"))
        mid = kit.apply_step(first, p)
        for pt, tag in _slips([second], mid, kit.apply_step(second, mid)):
            out.append((pt, tag))
        return out
    step = steps[0]
    kind = step[0]
    if kind == "T":
        dx, dy = step[1], step[2]
        out += [
            ((x - dx, y - dy), "wrong-direction"),
            ((x + dy, y + dx), "fr2a-translation-components-swapped"),
            ((x + dx, y), "fr2a-translation-component-dropped"),
            ((x, y + dy), "fr2a-translation-component-dropped"),
        ]
    elif kind in ("RX", "RY"):
        other = ("RY",) if kind == "RX" else ("RX",)
        out += [
            (kit.apply_step(other, p), "wrong-axis-reflected"),
            ((-x, -y), "fr2a-reflection-as-half-turn"),
        ]
    elif kind == "RV":
        k = step[1]
        out += [
            ((-x, y), "fr2a-line-reflected-as-axis"),
            ((k, y), "fr2a-image-on-mirror-line"),
            (kit.apply_step(("RH", k), p), "wrong-axis-reflected"),
            ((x - 2 * k, y), "fr2a-mirror-offset-sign"),
        ]
    elif kind == "RH":
        k = step[1]
        out += [
            ((x, -y), "fr2a-line-reflected-as-axis"),
            ((x, k), "fr2a-image-on-mirror-line"),
            (kit.apply_step(("RV", k), p), "wrong-axis-reflected"),
            ((x, y - 2 * k), "fr2a-mirror-offset-sign"),
        ]
    elif kind == "H":
        cx, cy = step[1], step[2]
        if (cx, cy) != (0, 0):
            out.append(((-x, -y), "fr2a-rotation-about-origin-not-centre"))
        out += [
            ((2 * cx - x, y), "fr2a-half-turn-one-coordinate"),
            ((x, 2 * cy - y), "fr2a-half-turn-one-coordinate"),
        ]
    elif kind == "Q":
        cx, cy, way = step[1], step[2], step[3]
        flipped = "anticlockwise" if way == "clockwise" else "clockwise"
        out.append((kit.apply_step(("Q", cx, cy, flipped), p), "fr2a-turn-direction-reversed"))
        out.append(((answer[1], answer[0]), "swapped-x-and-y"))
        out.append((kit.apply_step(("H", cx, cy), p), "fr2a-quarter-turn-as-half-turn"))
        if (cx, cy) != (0, 0):
            out.append((kit.apply_step(("Q", 0, 0, way), p), "fr2a-rotation-about-origin-not-centre"))
    if kind != "Q":
        out.append(((answer[1], answer[0]), "swapped-x-and-y"))
    return out


def _pair_distractors(steps, p, answer):
    cands = [
        kit.pair_distractor(tag, x=pt[0], y=pt[1]) for pt, tag in _slips(steps, p, answer) if tuple(pt) != tuple(answer)
    ]
    ans_text, _ = kit.labelled_pair(x=answer[0], y=answer[1])
    return pick_distractors(ans_text, cands, 3, tolerance=GRADER_TOLERANCE)


# ------------------------------------------------------------------------------------------ working and text


def _working(steps, p, name):
    lines = [f"Read {name} from the grid: {kit.coord_text(*p)}."]
    q = p
    for s in steps:
        nxt = kit.apply_step(s, q)
        kind = s[0]
        if kind == "T":
            how = f"add {s[1]} to x and {s[2]} to y"
        elif kind == "RX":
            how = "the x-axis is the mirror, so y changes sign"
        elif kind == "RY":
            how = "the y-axis is the mirror, so x changes sign"
        elif kind == "RV":
            how = f"the point and its image are the same distance from x = {s[1]}, so x becomes 2 × {s[1]} − x"
        elif kind == "RH":
            how = f"the point and its image are the same distance from y = {s[1]}, so y becomes 2 × {s[1]} − y"
        elif kind == "H":
            how = f"the centre {kit.coord_text(s[1], s[2])} is the midpoint of the point and its image"
        else:
            how = f"turn the vector from the centre {kit.coord_text(s[1], s[2])} through 90° {s[3]}"
        lines.append(f"{kit.step_phrase(s).capitalize()}: {how}, giving {kit.coord_text(*nxt)}.")
        q = nxt
    return lines


def _stem(rng, steps, final_name):
    if len(steps) == 1:
        body = phrase(
            rng,
            (
                "Triangle ABC is drawn on the grid. It is {move}.",
                "The grid shows triangle ABC. The triangle is {move}.",
                "Triangle ABC is shown on the coordinate grid. It is {move}.",
            ),
            move=kit.step_phrase(steps[0]),
        )
    else:
        body = phrase(
            rng,
            (
                "Triangle ABC is drawn on the grid. It is {a}, and then the image is {b}.",
                "The grid shows triangle ABC. The triangle is {a}. The image is then {b}.",
            ),
            a=kit.step_phrase(steps[0]),
            b=kit.step_phrase(steps[1]),
        )
    return body, f"Write the coordinates of {final_name} as x = _, y = _."


def _final_name(name, steps):
    return name + "'" * len(steps)


# ------------------------------------------------------------------------------------------ the generator


@generator(
    "fr2a_transform_image",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_transform_image(rng, tier="easy"):
    """The image of a named corner of a drawn triangle after one move, or two moves in order."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    for _ in attempts(3000, "a triangle, a move and a picture that fit"):
        steps = {
            "easy": lambda: [_easy_step(rng)],
            "medium": lambda: [_medium_step(rng)],
            "hard": lambda: _hard_steps(rng),
        }[tier]()
        pts = _triangle(rng, 5 if tier != "easy" else 6)
        if any(not _on_one_side(pts, s) for s in steps) and tier != "hard":
            continue
        if tier == "hard" and not _on_one_side(pts, steps[0]):
            continue
        which = rng.randrange(3)
        p = pts[which]
        imgs = [kit.image_of(steps, q) for q in pts]
        every = pts + imgs
        if any(abs(c) > _LIMIT for q in every for c in q):
            continue
        answer = imgs[which]
        if answer == p:
            continue
        if len(steps) == 2 and kit.image_of(steps[::-1], p) == answer:
            continue
        distractors = _pair_distractors(steps, p, answer)
        if len(distractors) < 3:
            continue
        mirror = [s for s in steps if s[0] in ("RV", "RH")]
        lines = []
        for s in mirror:
            if s[0] == "RV":
                lines.append(((s[1], -_LIMIT), (s[1], _LIMIT), f"x = {s[1]}"))
            else:
                lines.append(((-_LIMIT, s[1]), (_LIMIT, s[1]), f"y = {s[1]}"))
        # the grid has to cover the image, but the mirror lines are drawn far past it: keep them inside the window
        fig = kit.plane_figure(
            [(pts, list(_NAMES))],
            alt="A triangle ABC drawn on a coordinate grid.",
            lines=[((a[0], a[1]), (b[0], b[1]), lab) for a, b, lab in lines],
            hull=imgs + [(0, 0)],
        )
        name = _NAMES[which]
        final = _final_name(name, steps)
        ans_text, ans = kit.labelled_pair(x=answer[0], y=answer[1])
        if kit.leaks(fig, ans_text, final):
            continue
        break
    body, ask = _stem(rng, steps, final)
    text = f"{body} {ask}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        answer_kind="pair",
        working=_working(steps, p, name),
        distractors=distractors,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ the figure reader


def _read_transform_figure(question):
    """Re-derive the answer from the drawn triangle (corners recovered from the figure) and the move named in the
    stem, by complex-number arithmetic."""
    import re

    fig = question.figure
    corners = {pt["label"]: tuple(int(c) for c in pt["at"]) for pt in fig["points"] if pt.get("label") in _NAMES}
    name = re.search(r"coordinates of ([ABC])'+", question.text)[1]
    steps = kit.steps_in(question.text)
    q = corners[name]
    z = complex(*q)
    for s in steps:
        q = kit.apply_step(s, q)
        z = complex(*kit.apply_step_complex(s, (z.real, z.imag)))
    check(tuple(q) == (round(z.real), round(z.imag)), "reader routes disagree")
    return kit.labelled_pair(x=q[0], y=q[1])[0]


register_reader("fr2a_transform_image", _read_transform_figure)


# ------------------------------------------------------------------------------------------ the exam chain

_CONTEXT = (
    "A designer plots the plan of a triangular garden bed on a coordinate grid.",
    "A map of a triangular paddock is plotted on a coordinate grid.",
    "A sail is drawn as triangle ABC on a coordinate grid.",
    "A triangular tile is drawn on a coordinate grid.",
)


@generator("exd_x_fr2a_transform_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_fr2a_transform_chain(rng, tier="exam"):
    """Exam tier: first image, final image, longest side of the image and its perimeter."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_fr2a_transform_chain, rng, "a chain with enough distinct wrong answers")


def _fr2a_transform_chain(rng):
    for _ in attempts(3000, "a right-angled triangle, two moves and a picture that fit"):
        leg1, leg2, hyp = rng.choice(kit.LEG_TRIPLES)
        if rng.random() < 0.5:
            leg1, leg2 = leg2, leg1
        bx, by = rng.randint(-4, 4), rng.randint(-4, 4)
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        a = (bx + sx * leg1, by)
        b = (bx, by)
        c = (bx, by + sy * leg2)
        pts = [a, b, c]
        first = rng.choice([("RX",), ("RY",), ("RV", rng.choice([-3, -2, 2, 3])), ("RH", rng.choice([-3, -2, 2, 3]))])
        if not _on_one_side(pts, first):
            continue
        second = ("T", rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]), rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]))
        mid = [kit.image_of([first], q) for q in pts]
        fin = [kit.image_of([first, second], q) for q in pts]
        if any(abs(v) > _LIMIT for q in pts + mid + fin for v in q):
            continue
        if kit.image_of([second, first], a) == fin[0]:
            continue
        break
    # lengths of the final triangle, from its own corners
    d = lambda p, q: math.dist(p, q)  # noqa: E731
    check(
        round(d(fin[0], fin[2])) == hyp and round(d(mid[0], mid[2])) == hyp, "the longest side changed under the moves"
    )
    mirror = [first] if first[0] in ("RV", "RH") else []
    lines = [
        ((s[1], -_LIMIT), (s[1], _LIMIT), f"x = {s[1]}")
        if s[0] == "RV"
        else ((-_LIMIT, s[1]), (_LIMIT, s[1]), f"y = {s[1]}")
        for s in mirror
    ]
    fig = kit.plane_figure(
        [(pts, list(_NAMES))],
        alt="A right-angled triangle ABC drawn on a coordinate grid.",
        lines=lines,
        hull=mid + fin,
    )
    stem = (
        f"{rng.choice(_CONTEXT)} The corners are A{kit.coord_text(*a)}, B{kit.coord_text(*b)} and "
        f"C{kit.coord_text(*c)}, where each unit on the grid is 1 metre. The triangle is {kit.step_phrase(first)} to give "
        f"A'B'C', and then A'B'C' is {kit.step_phrase(second)} to give A''B''C''."
    )

    def pair_part(prompt, point, marks, cands, work):
        text, value = kit.labelled_pair(x=point[0], y=point[1])
        return dict(
            prompt=prompt,
            answer_kind="pair",
            answer=value,
            answer_text=text,
            answer_latex=latex_answer(text, plain=True),
            marks=marks,
            working=work,
            distractors=pick_distractors(text, cands, 3, tolerance=GRADER_TOLERANCE),
        )

    def cands_for(steps, p, answer):
        return [
            kit.pair_distractor(tag, x=pt[0], y=pt[1])
            for pt, tag in _slips(steps, p, answer)
            if tuple(pt) != tuple(answer)
        ]

    part_a = pair_part(
        "Write the coordinates of A' as x = _, y = _.",
        mid[0],
        2,
        cands_for([first], a, mid[0]),
        _working([first], a, "A")[1:],
    )
    part_b = pair_part(
        "Write the coordinates of A'' as x = _, y = _.",
        fin[0],
        2,
        cands_for([second], mid[0], fin[0])
        + [
            kit.pair_distractor("fr2a-steps-in-wrong-order", x=v[0], y=v[1])
            for v in [kit.image_of([second, first], a)]
            if tuple(v) != tuple(fin[0])
        ]
        + [kit.pair_distractor("fr2a-first-step-left-out", x=v[0], y=v[1]) for v in [kit.apply_step(second, a)]],
        _working([second], mid[0], "A'")[1:],
    )
    part_c, part_d = kit.side_and_perimeter_parts(
        leg1,
        leg2,
        hyp,
        "A''B''C''",
        extra_cands=[
            (Fraction(abs(fin[0][0] - mid[2][0]) + abs(fin[0][1] - mid[2][1])), "fr2a-image-compared-with-wrong-point")
        ],
    )
    parts = [part_a, part_b, part_c, part_d]
    kit.require_distractors(parts)
    return dict(text=stem, figure=fig, parts=parts)
