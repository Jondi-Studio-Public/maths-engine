"""``fr2a_describe_transform`` and ``exd_x_fr2a_describe_chain``: describe the move from a triangle and its image.

Skill ``f10-describe-transformation-from-points`` (AC9M5SP03, AC9M7SP03). Both triangle ABC and its image are drawn on
a coordinate grid with every corner named; nothing is listed in the stem, so the coordinates are read from the picture.
The learner describes the move as a labelled pair.

Tiers:

- easy:   a translation. The answer is ``right = _, up = _`` (negative for a move left or down).
- medium: a reflection in a vertical or horizontal line (the answer is the mirror line's value ``k`` and the distance
          ``d`` from A to it), or a half turn (the answer is the centre of the turn, ``x = _, y = _``).
- hard:   two moves, only the final image drawn: a reflection in an axis or a half turn about the origin, and a
          translation, in either order. The learner finds the translation, which means undoing the other move first.

``exd_x_fr2a_describe_chain`` is the exam tier on one right-angled triangle and its image: (a) the kind of move (a
choice of six), (b) its details as a labelled pair, (c) the longest side of the image, (d) its perimeter (follow-through
from (c)).

Every pair is found from the corner coordinates and again by applying the described move to every corner.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
)
from mathgen.figures import register_reader

from . import _fr2a_kit as kit
from .exd_x_sc3_support import redraw
from .fr2a_transform_image_build import _LIMIT, _on_one_side, _triangle

_NAMES = ("A", "B", "C")


def _images(steps, pts):
    return [kit.image_of(steps, p) for p in pts]


def _figure(pts, imgs, primes, alt):
    return kit.plane_figure(
        [(pts, list(_NAMES)), (imgs, [n + "'" * primes for n in _NAMES])],
        alt=alt,
    )


def _pairs_text(**values):
    return kit.labelled_pair(**values)


def _pair_cands(items):
    """Tagged pair distractors from ``(dict of values, tag)`` items."""
    return [kit.pair_distractor(tag, **values) for values, tag in items]


def _clean(cands, answer):
    return [c for c in cands if c.value != answer]


@generator(
    "fr2a_describe_transform",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_describe_transform(rng, tier="easy"):
    """Describe the move that takes a drawn triangle to its drawn image."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return redraw(lambda r: _describe(r, tier), rng, "a picture with enough distinct wrong answers")


def _describe(rng, tier):
    for _ in attempts(3000, "a triangle, a move and a picture that fit"):
        pts = _triangle(rng, 5)
        if tier == "easy":
            kind = "T"
            t = (
                rng.choice([-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7]),
                rng.choice([-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7]),
            )
            steps = [("T", t[0], t[1])]
        elif tier == "medium":
            kind = rng.choice(("RV", "RH", "H"))
            if kind == "H":
                steps = [("H", rng.randint(-4, 4), rng.randint(-4, 4))]
            else:
                steps = [(kind, rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))]
                if not _on_one_side(pts, steps[0]):
                    continue
        else:
            order = rng.choice(("move-first", "turn-first"))
            other = rng.choice([("RX",), ("RY",), ("H", 0, 0)])
            t = (
                rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]),
                rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]),
            )
            steps = [other, ("T", *t)] if order == "move-first" else [("T", *t), other]
            kind = "TT"
        imgs = _images(steps, pts)
        if any(abs(c) > _LIMIT for q in pts + imgs for c in q):
            continue
        if len(set(pts) & set(imgs)) or imgs == pts:
            continue
        break

    primes = len(steps)
    fig = _figure(pts, imgs, primes, "Triangle ABC and its image drawn on a coordinate grid, every corner named.")
    a, a1 = pts[0], imgs[0]
    if kind == "T":
        dx, dy = a1[0] - a[0], a1[1] - a[1]
        check((dx, dy) == (t[0], t[1]), "translation read off A disagrees with the stated one")
        answer = dict(right=dx, up=dy)
        cands = _pair_cands(
            [
                (dict(right=dy, up=dx), "fr2a-translation-components-swapped"),
                (dict(right=-dx, up=-dy), "wrong-direction"),
                (dict(right=abs(dx), up=abs(dy)), "fr2a-vector-sign-dropped"),
                (dict(right=a1[0] - pts[1][0], up=a1[1] - pts[1][1]), "fr2a-vector-from-different-corners"),
                (dict(right=a1[0] + a[0], up=a1[1] + a[1]), "fr2a-vector-sum-not-difference"),
                (dict(right=dx, up=-dy), "fr2a-vector-component-sign"),
            ]
        )
        text = phrase(
            rng,
            (
                "Triangle ABC is translated to give A'B'C', as shown on the grid. Give the translation as "
                "right = _, up = _ (use a negative number for a move to the left or down).",
                "Triangle ABC has slid across the grid to the position A'B'C'. How far has it moved? Answer as "
                "right = _, up = _, with a negative number for a move left or down.",
                "The grid shows triangle ABC and its image A'B'C' after a translation. Describe the translation "
                "as right = _, up = _ (negative for left or down).",
            ),
        )
        verify = [("T", dx, dy)]
        working = [
            f"A is at {kit.coord_text(*a)} and A' is at {kit.coord_text(*a1)}.",
            f"Right: {a1[0]} − {paren_negative(a[0])} = {dx}. Up: {a1[1]} − {paren_negative(a[1])} = {dy}.",
        ]
    elif kind in ("RV", "RH"):
        k = steps[0][1]
        axis = 0 if kind == "RV" else 1
        d = abs(a[axis] - k)
        check(a1[axis] == 2 * k - a[axis] and (a1[axis] + a[axis]) == 2 * k, "mirror line is not midway")
        answer = dict(k=k, dist=d)
        cands = _pair_cands(
            [
                (dict(k=k, dist=2 * d), "distance-to-axis-not-to-image"),
                (dict(k=a[axis] + a1[axis], dist=d), "fr2a-mirror-line-not-halved"),
                (dict(k=d, dist=k), "swapped-coordinates"),
                (
                    dict(k=abs(a1[axis] - a[axis]) // 2 if k >= 0 else -(abs(a1[axis] - a[axis]) // 2), dist=d),
                    "fr2a-mirror-line-from-gap",
                ),
                (dict(k=-k, dist=d), "fr2a-mirror-line-sign"),
            ]
        )
        which = "vertical" if kind == "RV" else "horizontal"
        var = "x" if kind == "RV" else "y"
        text = (
            f"Triangle ABC is reflected in a {which} line to give A'B'C', as shown on the grid. The mirror line is "
            f"{var} = k, and corner A is a distance dist from it, in grid units. Find k and dist, and give them as k = _, dist = _."
        )
        verify = [(kind, k)]
        working = [
            f"A is at {kit.coord_text(*a)} and A' is at {kit.coord_text(*a1)}.",
            f"The mirror line is halfway between a point and its image: {var} = ({a[axis]} + {paren_negative(a1[axis])}) ÷ 2 = {k}.",
            f"A is |{a[axis]} − {paren_negative(k)}| = {kit.units(d)} from the line.",
        ]
    elif kind == "H":
        cx, cy = steps[0][1], steps[0][2]
        answer = dict(x=cx, y=cy)
        cands = _pair_cands(
            [
                (dict(x=a[0] + a1[0], y=a[1] + a1[1]), "fr2a-centre-not-halved"),
                (dict(x=(a1[0] - a[0]) // 2, y=(a1[1] - a[1]) // 2), "fr2a-centre-from-difference"),
                (dict(x=0, y=0), "fr2a-centre-assumed-origin"),
                (dict(x=cy, y=cx), "swapped-x-and-y"),
                (dict(x=cx, y=a[1]), "fr2a-centre-one-coordinate"),
                (dict(x=a[0], y=cy), "fr2a-centre-one-coordinate"),
            ]
        )
        text = (
            "Triangle ABC is rotated 180° about a point to give A'B'C', as shown on the grid. Find the centre of "
            "the turn and give it as x = _, y = _."
        )
        verify = [("H", cx, cy)]
        working = [
            f"A is at {kit.coord_text(*a)} and A' is at {kit.coord_text(*a1)}.",
            "The centre of a half turn is the midpoint of any point and its image: "
            f"x = ({a[0]} + {paren_negative(a1[0])}) ÷ 2 = {cx}, "
            f"y = ({a[1]} + {paren_negative(a1[1])}) ÷ 2 = {cy}.",
        ]
    else:  # two moves, translation unknown
        other = steps[0] if steps[0][0] != "T" else steps[1]
        t = steps[1][1:] if steps[0][0] != "T" else steps[0][1:]
        first_is_turn = steps[0][0] != "T"
        if first_is_turn:
            before = kit.apply_step(other, a)
            right, up = a1[0] - before[0], a1[1] - before[1]
            ignore = (a1[0] - a[0], a1[1] - a[1])
            wrong_order = (kit.apply_step(other, a1)[0] - a[0], kit.apply_step(other, a1)[1] - a[1])
        else:
            right, up = kit.apply_step(other, a1)[0] - a[0], kit.apply_step(other, a1)[1] - a[1]
            ignore = (a1[0] - a[0], a1[1] - a[1])
            wrong_order = (a1[0] - kit.apply_step(other, a)[0], a1[1] - kit.apply_step(other, a)[1])
        check((right, up) == tuple(t), "translation found by undoing the other move disagrees with the stated one")
        check(ignore != (right, up) and wrong_order != (right, up), "a wrong method gives the right translation")
        answer = dict(right=right, up=up)
        cands = _pair_cands(
            [
                (dict(right=ignore[0], up=ignore[1]), "fr2a-other-move-ignored"),
                (dict(right=wrong_order[0], up=wrong_order[1]), "fr2a-steps-in-wrong-order"),
                (dict(right=-right, up=-up), "wrong-direction"),
                (dict(right=up, up=right), "fr2a-translation-components-swapped"),
                (dict(right=-ignore[0], up=ignore[1]), "fr2a-other-move-ignored"),
            ]
        )
        move_name = {
            "RX": "reflected in the x-axis",
            "RY": "reflected in the y-axis",
            "H": "rotated 180° about the origin",
        }[other[0]]
        move_noun = "rotation" if other[0] == "H" else "reflection"
        if first_is_turn:
            seq = f"{move_name}, and then the image is translated"
        else:
            seq = f"translated, and then the image is {move_name}"
        text = (
            f"Triangle ABC is {seq}, to give A''B''C'' (the first image is not drawn). "
            "Give the translation as right = _, up = _ (use a negative number for a move to the left or down)."
        )
        verify = steps
        working = [
            f"A is at {kit.coord_text(*a)} and A'' is at {kit.coord_text(*a1)}.",
            (
                f"The {move_noun} comes first, so A' is {kit.coord_text(*before)}. The translation takes A' to A'': "
                f"right {right}, up {up}."
                if first_is_turn
                else f"The translation comes first, then the {move_noun}. Undo the {move_noun} on A'' "
                f"to get {kit.coord_text(*kit.apply_step(other, a1))}, the point the translation reached from A: right {right}, up {up}."
            ),
        ]
    # second route: apply the described move to every corner and compare
    check(_images(verify, pts) == imgs, "the described move does not take every corner to its image")
    ans_text, ans = _pairs_text(**answer)
    check(not kit.leaks(fig, ans_text), "the figure shows the answer")
    distractors = pick_distractors(ans_text, _clean(cands, ans), 3, tolerance=GRADER_TOLERANCE)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        answer_kind="pair",
        working=working,
        distractors=distractors,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ the figure reader


def _corners(fig, suffix):
    return {
        pt["label"][0]: tuple(int(c) for c in pt["at"])
        for pt in fig["points"]
        if re.fullmatch(rf"[ABC]{suffix}", pt.get("label", ""))
    }


def _read_describe(question):
    """Re-derive the pair from the two drawn triangles alone (the stem only says which kind of answer is wanted)."""
    fig, text = question.figure, question.text
    a = _corners(fig, "")["A"]
    a1 = (_corners(fig, "'") or _corners(fig, "''"))["A"]
    if "right = _" in text and "first image is not drawn" not in text:
        return kit.labelled_pair(right=a1[0] - a[0], up=a1[1] - a[1])[0]
    if "first image is not drawn" in text:
        other = kit.steps_in(text)[0]
        turn_at = min(i for i in (text.find("reflected in"), text.find("rotated")) if i >= 0)
        first_turn = turn_at < text.index("translated")
        if first_turn:
            before = kit.apply_step(other, a)
            return kit.labelled_pair(right=a1[0] - before[0], up=a1[1] - before[1])[0]
        un = kit.apply_step(other, a1)
        return kit.labelled_pair(right=un[0] - a[0], up=un[1] - a[1])[0]
    if "k = _" in text:
        axis = 0 if "vertical" in text else 1
        k = (a[axis] + a1[axis]) // 2
        return kit.labelled_pair(k=k, dist=abs(a[axis] - k))[0]
    return kit.labelled_pair(x=(a[0] + a1[0]) // 2, y=(a[1] + a1[1]) // 2)[0]


register_reader("fr2a_describe_transform", _read_describe)


# ------------------------------------------------------------------------------------------ the exam chain

_OPTIONS = (
    "a translation",
    "a reflection in a vertical line",
    "a reflection in a horizontal line",
    "a rotation of 180° (a half turn)",
    "a rotation of 90° clockwise",
    "a rotation of 90° anticlockwise",
)
#: which wrong option stands for which slip, by the true kind
_NAMING_SLIPS = {
    "T": {
        1: "fr2a-slide-named-as-reflection",
        2: "fr2a-slide-named-as-reflection",
        3: "fr2a-slide-named-as-turn",
        4: "fr2a-slide-named-as-turn",
        5: "fr2a-slide-named-as-turn",
    },
    "RV": {
        0: "fr2a-reflection-named-as-slide",
        2: "fr2a-mirror-line-orientation",
        3: "fr2a-reflection-named-as-turn",
        4: "fr2a-reflection-named-as-turn",
        5: "fr2a-reflection-named-as-turn",
    },
    "H": {
        0: "fr2a-turn-named-as-slide",
        1: "fr2a-turn-named-as-reflection",
        2: "fr2a-turn-named-as-reflection",
        4: "fr2a-turn-size-or-direction",
        5: "fr2a-turn-size-or-direction",
    },
}
_CONTEXT = (
    "A school logo is drawn on a coordinate grid, then copied to a new position.",
    "A garden bed in the shape of a triangle is plotted on a coordinate grid and then a copy is plotted.",
    "A triangular sail pattern is drawn on a coordinate grid and then moved.",
    "A triangular tile pattern is drawn on a coordinate grid and then copied.",
)


@generator("exd_x_fr2a_describe_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_fr2a_describe_chain(rng, tier="exam"):
    """Exam tier: name the move, describe it, find the longest side of the image, then its perimeter."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_fr2a_describe_chain, rng, "a chain with enough distinct wrong answers")


def _fr2a_describe_chain(rng):
    for _ in attempts(3000, "a right-angled triangle, a move and a picture that fit"):
        leg1, leg2, hyp = rng.choice(kit.LEG_TRIPLES)
        if rng.random() < 0.5:
            leg1, leg2 = leg2, leg1
        bx, by = rng.randint(-3, 3), rng.randint(-3, 3)
        sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
        pts = [(bx + sx * leg1, by), (bx, by), (bx, by + sy * leg2)]
        kind = rng.choice(("T", "RV", "H"))
        if kind == "T":
            step = ("T", rng.choice([-6, -5, -4, -3, 3, 4, 5, 6]), rng.choice([-5, -4, -3, -2, 2, 3, 4, 5]))
        elif kind == "RV":
            step = ("RV", rng.choice([-4, -3, -2, 2, 3, 4]))
        else:
            step = ("H", rng.randint(-3, 3), rng.randint(-3, 3))
        if not _on_one_side(pts, step):
            continue
        imgs = _images([step], pts)
        if any(abs(c) > _LIMIT for q in pts + imgs for c in q):
            continue
        break
    a, a1 = pts[0], imgs[0]
    fig = _figure(pts, imgs, 1, "A right-angled triangle ABC and its image A'B'C' drawn on a coordinate grid.")
    stem = (
        f"{rng.choice(_CONTEXT)} Triangle ABC has a right angle at B, and its copy is A'B'C'. Each unit on the grid is "
        "1 metre."
    )
    # (a) the kind of move
    key_index = {"T": 0, "RV": 1, "H": 3}[kind]
    slips = _NAMING_SLIPS[kind]
    options = [Distractor(opt, slips.get(i), opt) for i, opt in enumerate(_OPTIONS) if i != key_index]
    part_a = dict(
        prompt="Which one of these describes the move that takes triangle ABC to triangle A'B'C'?",
        answer_kind="mcq",
        answer=_OPTIONS[key_index],
        answer_text=_OPTIONS[key_index],
        answer_latex=latex_answer(_OPTIONS[key_index], plain=True),
        marks=1,
        working=[
            "Compare A and A' (and the other corners). "
            + {
                "T": "Each corner has moved the same distance in the same direction, and the triangle faces the same way: a translation.",
                "RV": "The triangle faces the opposite way and each corner is the same distance from a vertical line on the other side of it: a reflection in a vertical line.",
                "H": "The triangle is turned upside down, and every corner and its image are the same distance from one point: a half turn.",
            }[kind]
        ],
        distractors=options[:5],
    )
    # (b) the details
    if kind == "T":
        dx, dy = step[1], step[2]
        answer = dict(right=dx, up=dy)
        cands = [
            (dict(right=dy, up=dx), "fr2a-translation-components-swapped"),
            (dict(right=-dx, up=-dy), "wrong-direction"),
            (dict(right=abs(dx), up=abs(dy)), "fr2a-vector-sign-dropped"),
            (dict(right=a1[0] - pts[1][0], up=a1[1] - pts[1][1]), "fr2a-vector-from-different-corners"),
        ]
        work_b = [f"A{kit.coord_text(*a)} to A'{kit.coord_text(*a1)}: right {dx}, up {dy}."]
    elif kind == "RV":
        k = step[1]
        d = abs(a[0] - k)
        answer = dict(k=k, dist=d)
        cands = [
            (dict(k=k, dist=2 * d), "distance-to-axis-not-to-image"),
            (dict(k=a[0] + a1[0], dist=d), "fr2a-mirror-line-not-halved"),
            (dict(k=d, dist=k), "swapped-coordinates"),
            (dict(k=-k, dist=d), "fr2a-mirror-line-sign"),
        ]
        work_b = [
            f"The mirror line is halfway between A and A': x = ({a[0]} + {paren_negative(a1[0])}) ÷ 2 = {k}.",
            f"A is {kit.units(d)} from it.",
        ]
    else:
        cx, cy = step[1], step[2]
        answer = dict(x=cx, y=cy)
        cands = [
            (dict(x=a[0] + a1[0], y=a[1] + a1[1]), "fr2a-centre-not-halved"),
            (dict(x=(a1[0] - a[0]) // 2, y=(a1[1] - a[1]) // 2), "fr2a-centre-from-difference"),
            (dict(x=0, y=0), "fr2a-centre-assumed-origin"),
            (dict(x=cy, y=cx), "swapped-x-and-y"),
        ]
        work_b = [f"The centre is the midpoint of A{kit.coord_text(*a)} and A'{kit.coord_text(*a1)}: ({cx}, {cy})."]
    ans_text, ans = _pairs_text(**answer)
    cands_d = [c for c in _pair_cands(cands) if c.value != ans]
    part_b = dict(
        prompt=(
            "Give the details of the move. For a translation write right = _, up = _ (negative for left or down). "
            "For a reflection write the mirror line as k = _ (its x or y value) and the distance dist from A to it as "
            "dist = _. For a half turn write the centre as x = _, y = _."
        ),
        answer_kind="pair",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        marks=2,
        working=work_b,
        distractors=pick_distractors(ans_text, cands_d, 3, tolerance=GRADER_TOLERANCE),
    )
    part_c, part_d = kit.side_and_perimeter_parts(leg1, leg2, hyp, "A'B'C'")
    parts = [part_a, part_b, part_c, part_d]
    kit.require_distractors(parts)
    return dict(text=stem, figure=fig, parts=parts)
