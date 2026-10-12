"""Exam-tier source for `senior-apply-euler-s-formula-v-f-e-2`: from a face-length list to how many edges can still
be added.

The stem describes a connected plane graph by its faces: the outer face has L sides and the other faces are
triangles, quadrilaterals and pentagons (every edge borders two different faces). The parts use Euler's formula and
then the edge bound for a simple planar graph:

(a) the number of edges e                                     number, 1 mark
(b) the number of vertices v                                  number, 2 marks, 2 + part_a - f
(c) the most edges a simple connected planar graph on v vertices can have, 3v - 6      number, 1 mark, 3 part_b - 6
(d) how many more edges could be added                       number, 2 marks, part_c - part_a

## Construction

The counts of faces are drawn first; the sum of the face lengths is kept even, so e is a whole number. Because every
face has at least three sides, the spare edges in (d) equal the sum over faces of (length - 3), which the generator
checks against 3v - 6 - e as an independent route.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._exd_x_r3_distinct import distinct_answers
from .exd_x_r3i_support import number_part

_FACE_WORDS = (("triangle", 3), ("quadrilateral", 4), ("pentagon", 5))


def _faces_text(counts: list[int]) -> str:
    """ "20 are triangles, 8 are quadrilaterals and 1 is a pentagon": the verb and noun agree with each count."""
    named = [(n, w) for n, (w, _) in zip(counts, _FACE_WORDS) if n]
    clauses = [f"{n} is a {w}" if n == 1 else f"{n} are {w}s" for n, w in named]
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]


@generator("exd_x_r3i_euler_edge_room", level=17, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3i_euler_edge_room(rng, tier: str = "exam") -> dict:
    """Edges and vertices from face lengths, the edge bound for simple planar graphs, and the spare edges."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "face counts with an even side total"):
        outer = rng.randint(3, 9)
        t, q, pn = rng.randint(4, 24), rng.randint(0, 8), rng.randint(0, 5)
        if not (t and (q or pn)):
            continue
        total = outer + 3 * t + 4 * q + 5 * pn
        if total % 2:
            continue
        break
    faces = 1 + t + q + pn
    e = total // 2
    v = 2 - faces + e
    most = 3 * v - 6
    spare = most - e
    check(v >= 3 and spare >= 0, "vertex count or edge bound out of range")
    check(spare == (outer - 3) + 0 * t + q + 2 * pn, "spare edges disagree with the sum of (length - 3)")
    check(e == (outer + 3 * t + 4 * q + 5 * pn) / 2, "edge count disagrees with the face lengths")

    counts = [t, q, pn]
    stem = phrase(
        rng,
        (
            "A connected planar graph is drawn in the plane with no edges crossing, and every edge borders two "
            "different faces. The outer face has {L} sides; of the other faces, {rest}.",
            "A connected plane graph has the property that each edge lies on the boundary of exactly two different "
            "faces. Its outer face has {L} sides and, of its other faces, {rest}.",
            "A road network divides the plane into regions, and every edge borders two different regions. The "
            "outer, unbounded region has {L} edges around it; of the remaining regions, {rest}.",
        ),
        L=outer,
        rest=_faces_text(counts),
    )
    inner_sum = total - outer
    part_a = number_part(
        "Find the number of edges.",
        Fraction(e),
        0,
        1,
        [f"Sum of the face lengths = {total}. Each edge borders two faces, so e = {total}/2 = {e}."],
        [
            ("face-sum-as-edges", Fraction(total)),
            ("outer-face-left-out", Fraction(inner_sum, 2) if inner_sum % 2 == 0 else None),
            ("face-sum-as-edges", Fraction(inner_sum)),
            ("off-by-one-count", Fraction(e + 1)),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Hence find the number of vertices.",
        Fraction(v),
        0,
        2,
        [f"f = {faces}. From v + f - e = 2: v = 2 + {e} - {faces} = {v}."],
        [
            ("dropped-constant-term", Fraction(e - faces)),
            ("sign-slip-in-formula", Fraction(e + faces - 2)),
            ("outer-face-left-out", Fraction(v + 1)),
            ("face-sum-as-edges", Fraction(2 + total - faces)),
        ],
        follow_through=f"2 + part_a - {faces}",
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Find the largest number of edges that any simple connected planar graph with this many vertices can have.",
        Fraction(most),
        0,
        1,
        [f"At most 3v - 6 = 3 × {v} - 6 = {most}."],
        [
            ("dropped-constant-term", Fraction(3 * v)),
            ("wrong-constant-in-formula", Fraction(3 * v - 3)),
            ("wrong-constant-in-formula", Fraction(2 * v - 4)),
            ("sign-slip-in-formula", Fraction(3 * v + 6)),
        ],
        follow_through="3*part_b - 6",
        sensible_range=(0, None),
    )
    part_d = number_part(
        "How many more edges could be added to the graph while keeping it simple, connected and planar?",
        Fraction(spare),
        0,
        2,
        [f"{most} - {e} = {spare}."],
        [
            ("added-instead-of-subtracted", Fraction(most + e)),
            ("spare-edges-reported-as-maximum", Fraction(most)),
            ("sign-of-result-reversed", Fraction(e - most)),
            ("off-by-one-count", Fraction(spare + 1)),
        ],
        follow_through="part_c - part_a",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
