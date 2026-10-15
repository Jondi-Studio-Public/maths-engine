"""Exam-tier applied Pythagoras questions.

The practice tiers name one right triangle in the story and ask for one side.
These questions need the learner to find the right triangle(s) themselves, and
two of the three need more than one application of the theorem:

- ``ladder_slide``: a ladder slips down a wall; how far does its foot move?
  (two right triangles sharing a hypotenuse)
- ``pole_wire``: a wire joins the tops of two poles of different heights.
  (the vertical leg is a difference of heights, not a given length)
- ``rhombus``: a rhombus is given a side and one diagonal; find the other.
  (the diagonals bisect at right angles, so the triangle is half of each)

Every question is built backwards from integer Pythagorean triples, so each
answer is a whole number of metres or centimetres.
"""

from __future__ import annotations

import math
from functools import cache

from ..core import check, generator, phrase, pick_distractors


@cache
def _pythag():
    triples = [(a, b, c) for c in range(5, 101) for a in range(3, c) for b in range(a, c) if a * a + b * b == c * c]

    # hypotenuse -> every (leg, leg) pair with that hypotenuse; ladders need at least two.
    by_hyp: dict[int, list[tuple[int, int]]] = {}
    for a, b, c in triples:
        by_hyp.setdefault(c, []).append((a, b))
    shared = sorted(h for h, pairs in by_hyp.items() if len(pairs) >= 2 and h <= 65)
    return triples, by_hyp, shared


_LADDER = (
    "A {h} cm ladder leans against a vertical wall with its foot {x1} cm from the wall. The top of the ladder slips {s} cm down the wall. How far does the foot of the ladder move away from the wall, in cm?",
    "The foot of a {h} cm ladder stands {x1} cm from a vertical wall. The ladder slides so that its top is {s} cm lower than before. By how many centimetres has the foot moved out?",
    "A ladder {h} cm long rests against a wall, its foot {x1} cm out from the wall. It slips until the top has dropped {s} cm. Find the distance the foot has moved along the ground, in cm.",
)
_POLES = (
    "Two vertical poles stand on level ground {d} m apart. One is {h1} m tall and the other is {h2} m tall. A straight wire joins their tops. How long is the wire, in metres?",
    "A straight cable runs from the top of a {h1} m pole to the top of a {h2} m pole. The poles are vertical and {d} m apart on flat ground. Find the length of the cable in metres.",
)
_RHOMBUS = (
    "A rhombus has sides of length {s} cm and one diagonal of length {g} cm. Find the length of the other diagonal, in cm.",
    "Each side of a rhombus is {s} cm, and one of its diagonals measures {g} cm. What is the length of the other diagonal in cm?",
    "The diagonals of a rhombus cross at right angles. Its sides are {s} cm long and one diagonal is {g} cm. Find the other diagonal in cm.",
)


def _ladder(rng):
    h = rng.choice(_pythag()[2])
    (xa, ya), (xb, yb) = rng.sample(_pythag()[1][h], 2)
    # order so the first position has the foot nearer the wall (top higher) and the second has it further out
    (x1, y1), (x2, y2) = sorted([(xa, ya), (xb, yb)], key=lambda p: p[0])
    check(x1 < x2 and y1 > y2 and x1 * x1 + y1 * y1 == h * h and x2 * x2 + y2 * y2 == h * h)
    # work in tens of centimetres so the ladder is a sensible length
    h, x1, y1, x2, y2 = (10 * v for v in (h, x1, y1, x2, y2))
    s = y1 - y2
    key = x2 - x1
    cands = [s, x2, y2, y1]
    cands = [c for c in cands if c != key]
    return phrase(rng, _LADDER, h=h, x1=x1, s=s), key, cands


def _poles(rng):
    for _ in range(100):
        dh, d, w = rng.choice(_pythag()[0])
        if rng.random() < 0.5:
            dh, d = d, dh
        k = rng.randint(1, 3)
        dh, d, w = dh * k, d * k, w * k
        if w > 60:
            continue
        h2 = rng.randint(2, 15)
        h1 = h2 + dh
        key = w
        cands = [
            d + dh,  # adds the two legs
            dh,  # reports the difference in heights
            d,  # reports the ground distance
            h1,  # reports the taller pole's height
        ]
        return phrase(rng, _POLES, d=d, h1=h1, h2=h2), key, [c for c in cands if c != key]
    check(False, "exd_x_f4_pythag: no pole draw")


def _rhombus(rng):
    for _ in range(100):
        a, b, s = rng.choice(_pythag()[0])
        k = rng.randint(1, 3)
        a, b, s = a * k, b * k, s * k
        if s > 85:
            continue
        if rng.random() < 0.5:
            a, b = b, a
        # half-diagonals a and b; the given diagonal is 2a
        g = 2 * a
        key = 2 * b
        if g == key:
            continue
        cands = [
            b,  # stops at the half-diagonal
            g,  # assumes the diagonals are equal
            s,  # reads the side as the other diagonal
            a + b,  # adds the two half-diagonals
        ]
        return phrase(rng, _RHOMBUS, s=s, g=g), key, [c for c in cands if c != key]
    check(False, "exd_x_f4_pythag: no rhombus draw")


_BUILDERS = (_ladder, _poles, _rhombus)
_WORKING = {
    _ladder: "Find the height of the ladder top before and after the slip, then the foot distance after, and subtract.",
    _poles: "The vertical leg is the difference of the pole heights; apply Pythagoras with the distance between the poles.",
    _rhombus: "The diagonals bisect at right angles: half the known diagonal and the side give the other half-diagonal; then double it.",
}


@generator("exd_x_f4_pythag", level=9, presets={"exam": dict(tier="exam")})
def exd_x_f4_pythag(rng, tier: str = "exam"):
    builder = rng.choice(_BUILDERS)
    text, key, cands = builder(rng)
    distractors = pick_distractors(str(key), [_fmt(c) for c in cands])
    check(len(distractors) >= 2, "exd_x_f4_pythag: too few distractors")
    return dict(
        text=text,
        latex=text,
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        working=[_WORKING[builder]],
        distractors=distractors,
    )


def _fmt(value) -> str:
    return str(int(value))
