"""Exam source for `pythagoras-truss-chain` (`skills/trigonometry/pythagoras-truss-chain.yaml`), topic
`symmetric_truss`.

The skill's existing three sources (`t3_pythagoras_truss_chain_build.py`,
`pythagoras_truss_chain`) all track one computed length through a CHAIN of
right triangles peeling off each other's found leg. This source is a
genuinely different truss shape: a SYMMETRIC truss built from a half-span, an
apex height and a panel count, where the target is one of three sibling
quantities read off the SAME pair of similar triangles -- the top chord (a
direct Pythagorean hypotenuse of the whole half-truss), the apex web at an
interior panel point (a NEW hypotenuse, from that panel point straight up to
the apex -- not the chord), or the perpendicular web at that panel point (a
similar-triangles ratio, not a second Pythagoras application). None of these
three shapes is "track a length through a chain" -- the learner instead reads
one of three quantities off one fixed pair of nested right triangles, so this
is the genuine escalation over the chain skill's own hard tier (target_mode
="random" there still only ever asks for a leg or the closing hypotenuse of
SOME triangle in a chain -- never a similar-triangles ratio).

Construction (half-span A, apex height H, top chord L, for a primitive
Pythagorean triple (a, b, c) and integer scale t):

    A = a*c*t, H = b*c*t, L = c*c*t   (A^2 + H^2 = L^2 exactly, by algebra)

For a panel count m with m | A, and an interior panel index j in 1..m-1:

    x = A*j/m (integer, guaranteed by m | A), d = A - x

    apex web  = sqrt(d^2 + H^2)   -- integer only for SOME (triple, t, m, j)
    perp web  = x*H/L             -- integer only for SOME (triple, t, m, j)
    (the perpendicular from the panel point onto the sloped chord: the small
    triangle with legs (perp web, something) and hypotenuse x is similar to
    the big triangle with legs A, H and hypotenuse L, so perp web / x = H / L)

Both integrality conditions are checked with exact integer arithmetic
(`math.isqrt` for the apex web, plain divisibility for the perpendicular web)
and every combination that clears at least one of them is enumerated ONCE, at
IMPORT TIME, into `_TABLE` -- a dict keyed by which target the row supports,
built with plain nested loops over triple/orientation/t/m/j (2947 rows over 6
primitive triples x 2 leg orientations x t in 1..6 x m in 2..12 x every
interior j's own widened bank). `symmetric_truss()` itself is then
a single `rng.choice()` per call -- O(1), never a runtime rejection-sampling
loop: the two simultaneous integrality conditions are sparse enough (the
apex-web-only slice is 59 of 2947 rows) that a bounded retry here risks
`check_generator_speed.py`'s hard-fail or an unlucky seed that never resolves,
unlike this skill's own sibling `_draw_chain`, whose constraint space is
dense enough that bounded rejection sampling is the right tool there.

Two hand-verified instances:

1. (a,b,c)=(4,3,5), t=1 -> A=20, H=15, L=25, m=10. At j=6: x=12, d=8, apex
   web = sqrt(8^2+15^2) = sqrt(289) = 17. At j=5: x=10, perpendicular web =
   10*15/25 = 6.
2. (a,b,c)=(4,3,5), t=5 -> A=100, H=75, L=125, m=5. At j=3: x=60, d=40, apex
   web = sqrt(40^2+75^2) = sqrt(7225) = 85; the SAME point's perpendicular
   web = 60*75/125 = 36.

Independent check inside the generator (`_verify_truss_geometry`): re-derives
the chord, the apex web and the perpendicular web from the raw
(A, H, L, m, j) tuple via a DIFFERENT computation route than the table build
used -- a fresh `math.isqrt` on `A^2+H^2` and on `d^2+H^2`, and a fresh
`fractions.Fraction(x*H, L)` exactness test (the table build used plain `%`
divisibility) for the perpendicular web -- and `check()`s the result against
whatever the table already asserts for that row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..core import check, check_distinct, generator, pick_distractors

# The same primitive-triple bank `geometry.py`'s TRIPLES uses (kept as this
# module's own literal copy rather than an import, so this new topic does not
# create a dependency on another generator module).
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]

_T_RANGE = range(1, 7)  # scale
_M_RANGE = range(2, 13)  # panel count


@dataclass(frozen=True)
class _Entry:
    A: int
    H: int
    L: int
    m: int
    j: int
    x: int
    d: int
    target: str  # "chord" | "apex_web" | "perp_web"
    answer: int
    candidates: tuple[int, int, int]
    misconceptions: tuple[str, str, str]


def _all_distinct(answer: int, candidates: tuple[int, int, int]) -> bool:
    values = [answer, *candidates]
    return len(set(values)) == len(values)


def _build_table() -> dict[str, list[_Entry]]:
    """Every (triple, orientation, t, m, j) combination that clears at least
    one of the two integrality conditions, split by which target(s) it
    supports. Plain nested loops, exact integer arithmetic throughout -- no
    float, no retry."""
    table: dict[str, list[_Entry]] = {"chord": [], "apex_web": [], "perp_web": []}
    for p, q, c in _TRIPLES:
        for a, b in ((p, q), (q, p)):  # both leg orientations: wide-flat / tall-narrow
            for t in _T_RANGE:
                A, H, L = a * c * t, b * c * t, c * c * t
                for m in _M_RANGE:
                    if A % m:
                        continue
                    for j in range(1, m):
                        x = A * j // m
                        d = A - x

                        apex_diff = d * d + H * H
                        apex_root = math.isqrt(apex_diff)
                        apex_ok = apex_root * apex_root == apex_diff
                        apex_web = apex_root if apex_ok else None

                        perp_num = x * H
                        perp_ok = perp_num % L == 0
                        perp_web = perp_num // L if perp_ok else None

                        if not (apex_ok or perp_ok):
                            continue

                        # target = chord: always computable (A, H always
                        # integers by construction); the "other web" third
                        # distractor always exists since apex_ok or perp_ok.
                        chord_third = apex_web if apex_ok else perp_web
                        chord_candidates = (A, A + H, chord_third)
                        if _all_distinct(L, chord_candidates):
                            table["chord"].append(
                                _Entry(
                                    A,
                                    H,
                                    L,
                                    m,
                                    j,
                                    x,
                                    d,
                                    "chord",
                                    L,
                                    chord_candidates,
                                    ("stopped-at-intermediate", "added-instead-of-pythagoras", "wrong-stage-value"),
                                )
                            )

                        if apex_ok:
                            apex_third = perp_web if perp_ok else x
                            apex_candidates = (L, d + H, apex_third)
                            if _all_distinct(apex_web, apex_candidates):
                                table["apex_web"].append(
                                    _Entry(
                                        A,
                                        H,
                                        L,
                                        m,
                                        j,
                                        x,
                                        d,
                                        "apex_web",
                                        apex_web,
                                        apex_candidates,
                                        ("stopped-at-intermediate", "added-instead-of-pythagoras", "wrong-stage-value"),
                                    )
                                )

                        if perp_ok:
                            # H (the full apex height), not d, is the fallback
                            # third distractor when apex_ok is False: d
                            # collides with x whenever j == m/2 (m even), the
                            # one systematic collision this table build found
                            # by construction, so it is never used here.
                            perp_third = apex_web if apex_ok else H
                            perp_candidates = (L, perp_third, x)
                            if _all_distinct(perp_web, perp_candidates):
                                table["perp_web"].append(
                                    _Entry(
                                        A,
                                        H,
                                        L,
                                        m,
                                        j,
                                        x,
                                        d,
                                        "perp_web",
                                        perp_web,
                                        perp_candidates,
                                        ("stopped-at-intermediate", "wrong-stage-value", "wrong-stage-value"),
                                    )
                                )
    return table


_TABLE = _build_table()
_TARGETS = ("chord", "apex_web", "perp_web")


def _verify_truss_geometry(A: int, H: int, L: int, m: int, j: int) -> tuple[int, int | None, int | None]:
    """Re-derive the chord, the apex web and the perpendicular web at panel
    point j from the raw (A, H, L, m, j) tuple, via a DIFFERENT computation
    route than `_build_table` used: a fresh `math.isqrt` on the chord
    identity and on d^2+H^2 (the table build also used `math.isqrt` for the
    apex web, but this call is independent of the one that built the table
    row being checked), and a fresh `Fraction(x*H, L)` exactness test for the
    perpendicular web (the table build used plain `%` divisibility)."""
    check(m >= 2 and 1 <= j < m, "symmetric_truss: panel index out of range")
    check(A % m == 0, "symmetric_truss: panel count does not evenly divide the half-span")
    x = A * j // m
    d = A - x

    chord_sq = A * A + H * H
    chord_root = math.isqrt(chord_sq)
    check(chord_root * chord_root == chord_sq and chord_root == L, "symmetric_truss: chord does not re-derive exactly")

    apex_diff = d * d + H * H
    apex_root = math.isqrt(apex_diff)
    apex_web = apex_root if apex_root * apex_root == apex_diff else None

    perp_frac = Fraction(x * H, L)
    perp_web = int(perp_frac) if perp_frac.denominator == 1 else None

    return chord_root, apex_web, perp_web


_TARGET_LABEL = {
    "chord": "the top chord (the sloped member from the support up to the apex)",
    "apex_web": "the apex web at panel point {j} (the member running from that panel point straight up to the apex)",
    "perp_web": "the perpendicular web at panel point {j} (the member running from that panel point perpendicular to the top chord)",
}


def _render_text(entry: _Entry) -> str:
    base = (
        f"A symmetric truss has a half-span of {entry.A} m, an apex height of {entry.H} m, "
        f"and is divided into {entry.m} equal panels along each sloped half. Panel point "
        f"{entry.j} (out of {entry.m}, counting from the support) lies {entry.x} m from the "
        f"support, measured along the base."
    )
    label = _TARGET_LABEL[entry.target].format(j=entry.j)
    return f"{base} Find the length, in m, of {label}."


def _render_working(entry: _Entry) -> list[str]:
    working = [f"Chord: A^2 + H^2 = {entry.A}^2 + {entry.H}^2 = {entry.L}^2, so the top chord = {entry.L} m."]
    if entry.target == "apex_web":
        working.append(
            f"d = A - x = {entry.A} - {entry.x} = {entry.d} m. Apex web = sqrt(d^2 + H^2) = "
            f"sqrt({entry.d}^2 + {entry.H}^2) = {entry.answer} m."
        )
    elif entry.target == "perp_web":
        working.append(
            f"By similar triangles, perpendicular web / x = H / L, so perpendicular web = "
            f"{entry.x} x {entry.H} / {entry.L} = {entry.answer} m."
        )
    working.append(f"Answer: {entry.answer} m.")
    return working


@generator("symmetric_truss", level=11)
def symmetric_truss(rng):
    """A symmetric truss: half-span A, apex height H, m equal panels, target one of {top chord, apex web at an
    interior panel point, perpendicular web at that panel point} -- the same pair of nested/similar right
    triangles, three different quantities read off it, varied per draw via `rng.choice` over the target first
    and then over the precomputed table of every (triple, t, m, j) combination that makes the chosen target an
    exact integer -- built once at import time, never by runtime retry."""
    target = rng.choice(_TARGETS)
    entry = rng.choice(_TABLE[target])

    chord, apex_web, perp_web = _verify_truss_geometry(entry.A, entry.H, entry.L, entry.m, entry.j)
    check(chord == entry.L, "symmetric_truss: chord disagrees with fresh re-derivation")
    if entry.target == "apex_web":
        check(apex_web == entry.answer, "symmetric_truss: apex web disagrees with fresh re-derivation")
    elif entry.target == "perp_web":
        check(perp_web == entry.answer, "symmetric_truss: perpendicular web disagrees with fresh re-derivation")

    answer_val = entry.answer
    answer_text = str(answer_val)
    check_distinct(answer_val, *entry.candidates, msg="symmetric_truss: a distractor equals the answer")
    distractors = pick_distractors(answer_text, [str(c) for c in entry.candidates])

    text = _render_text(entry)
    working = _render_working(entry)

    return {
        "text": text,
        "latex": text,
        "answer": answer_val,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
