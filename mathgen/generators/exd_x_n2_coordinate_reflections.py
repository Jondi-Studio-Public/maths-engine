"""Exam tier for "read and plot coordinates".

The practice tiers read one point or complete an axis-parallel rectangle. Here a point is reflected in one axis
and then the image is reflected in the other, and the four points together form a rectangle whose size must be
found. The learner has to read the starting point (given as coordinates or described by its distances from the
axes), carry out two reflections in the stated order, and turn the coordinates into side lengths and an area.

(a) the coordinates of the second image R (2 marks, a pair);
(b) the length of the first side PQ (1 mark);
(c) the length of the second side QR (1 mark);
(d) the area of the rectangle PQRS (2 marks, follow-through from (b) and (c)).

P is drawn in any quadrant with |x| different from |y|, so the rectangle is never a square and the two side
lengths cannot be confused with each other by luck.
"""

from __future__ import annotations

from mathgen.answers import Distractor
from mathgen.core import check, draw_until, generator, latex_answer, phrase, pick_distractors

_NUMBER_STEMS = (
    "P is the point ({x}, {y}). P is reflected in the {first}-axis to give Q, and then Q is reflected in the "
    "{second}-axis to give R. The point S completes the rectangle PQRS.",
    "On a Cartesian plane, the point P has coordinates ({x}, {y}). Its reflection in the {first}-axis is Q, and "
    "the reflection of Q in the {second}-axis is R. PQRS is a rectangle.",
)
_WORD_STEMS = (
    "P is {h} {hdir} of the y-axis and {v} {vdir} the x-axis. P is reflected in the {first}-axis to "
    "give Q, and then Q is reflected in the {second}-axis to give R. The point S completes the rectangle PQRS.",
    "A point P on the Cartesian plane lies {v} {vdir} the x-axis and {h} {hdir} of the y-axis. Q is "
    "the reflection of P in the {first}-axis and R is the reflection of Q in the {second}-axis. PQRS is a "
    "rectangle.",
)


def _units(n: int) -> str:
    return f"{n} unit" if n == 1 else f"{n} units"


def _pair(x: int, y: int) -> str:
    return f"({x}, {y})"


def _draw(rng):
    x = rng.choice([v for v in range(-9, 10) if v])
    y = rng.choice([v for v in range(-9, 10) if v])
    return x, y, rng.random() < 0.5, rng.random() < 0.5


@generator("exd_x_n2_coordinate_reflections", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_coordinate_reflections(rng, tier="exam"):
    """Two reflections, then the rectangle they make; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    x, y, x_first, as_words = draw_until(
        rng, _draw, lambda d: abs(d[0]) != abs(d[1]), what="a point with different coordinate sizes"
    )
    first, second = ("x", "y") if x_first else ("y", "x")
    q = (x, -y) if x_first else (-x, y)
    r = (-x, -y)
    pq = 2 * abs(y) if x_first else 2 * abs(x)
    qr = 2 * abs(x) if x_first else 2 * abs(y)
    check(q[0] == r[0] or q[1] == r[1], "Q and R must share a coordinate")
    check(pq * qr == 4 * abs(x) * abs(y), "area does not match the rectangle's dimensions")
    if as_words:
        stem = phrase(
            rng,
            _WORD_STEMS,
            h=_units(abs(x)),
            hdir="right" if x > 0 else "left",
            v=_units(abs(y)),
            vdir="above" if y > 0 else "below",
            first=first,
            second=second,
        )
    else:
        stem = phrase(rng, _NUMBER_STEMS, x=x, y=y, first=first, second=second)

    key = _pair(*r)
    p_cands = [
        Distractor(_pair(*q), "single-reflection-only", {"x": q[0], "y": q[1]}),
        Distractor(_pair(x, y), "wrong-axis-reflected", {"x": x, "y": y}),
        Distractor(_pair(r[1], r[0]), "swapped-x-and-y", {"x": r[1], "y": r[0]}),
    ]
    near_pq, near_qr = (abs(y), abs(x)) if x_first else (abs(x), abs(y))

    def lengths(own, half, other, other_half):
        return [
            Distractor(str(half), "distance-to-axis-not-to-image", half),
            Distractor(str(other), "other-side-of-rectangle", other),
            Distractor(str(other_half), "distance-to-axis-not-to-image", other_half),
        ]

    def num_part(prompt, value, marks, work, cands, ft=None):
        picked = pick_distractors(str(value), cands, 3)
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=latex_answer(str(value)),
            marks=marks,
            sensible_range=(0, None),
            working=work,
            distractors=picked,
        )
        if ft:
            out["follow_through"] = ft
        return out

    area = pq * qr
    parts = [
        dict(
            prompt="Write the coordinates of R.",
            answer_kind="pair",
            answer={"x": r[0], "y": r[1]},
            answer_text=key,
            answer_latex=key,
            marks=2,
            working=[
                f"Reflecting P in the {first}-axis gives Q{_pair(*q)}.",
                f"Reflecting Q in the {second}-axis gives R{key}.",
            ],
            distractors=pick_distractors(key, p_cands, 3),
        ),
        num_part(
            "Find the length of PQ, in units.",
            pq,
            1,
            [f"P{_pair(x, y)} and Q{_pair(*q)} differ by {pq} in one coordinate, so PQ = {pq}."],
            lengths(pq, near_pq, qr, near_qr),
        ),
        num_part(
            "Find the length of QR, in units.",
            qr,
            1,
            [f"Q{_pair(*q)} and R{key} differ by {qr} in one coordinate, so QR = {qr}."],
            lengths(qr, near_qr, pq, near_pq),
        ),
        num_part(
            "Find the area of the rectangle PQRS, in square units.",
            area,
            2,
            [f"Area = PQ × QR = {pq} × {qr} = {area}."],
            [
                Distractor(str(2 * (pq + qr)), "perimeter-for-area", 2 * (pq + qr)),
                Distractor(str(pq + qr), "sides-added-for-area", pq + qr),
                Distractor(str(near_pq * near_qr), "distance-to-axis-not-to-image", near_pq * near_qr),
            ],
            ft="part_b*part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
