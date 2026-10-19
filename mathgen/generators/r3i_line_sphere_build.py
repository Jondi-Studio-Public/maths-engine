"""Generates `r3i_line_sphere` questions (easy, medium, hard) and the exam chain `r3i_exam_line_sphere`.

Skill ``line-sphere-and-plane-intersection`` (``vectors``, answer ``number``): the distance from a point to a plane,
where a line meets a plane, and where a line meets a sphere. Every answer is a number (a whole number or an exact
fraction), so a point is asked for one coordinate or one value of the parameter λ at a time.

* easy: the distance from a point to a plane ``ax + by + cz = d``; the normal has a whole-number length (3, 7, 9, 11, 5).
* medium: a line ``r = p + λv`` meets a plane: the value of λ, a coordinate of the point of intersection, or (for a line
  through two points A and B) a coordinate of where line AB meets the plane.
* hard: a line meets a sphere ``(x - a)² + (y - b)² + (z - c)² = R²`` in two points: the distance between the points, the
  larger value of λ, or the value of λ at the point of the line nearest the centre.
* exam (``r3i_exam_line_sphere``): the line and sphere substituted together give ``Aλ² + 2Bλ + C = 0``; (a) A, (b) B,
  (c) the smaller λ (follows through from A and B), (d) the distance between the two points (follows through from A, B).

The independent check builds the actual points: the foot of the perpendicular for the distance, the intersection point
for the line and plane, and both intersection points for the sphere (each is shown to lie on the line and on the
surface) and measures the distances between them with ``math.dist``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3i_kit import cross, dot, num_cands, num_fields, pick, retry_short, signed_text, vtext

# Direction and normal vectors whose length is a whole number: (a, b, c) and the length.
_NICE = [
    (1, 2, 2, 3),
    (2, 3, 6, 7),
    (2, 6, 9, 11),
    (1, 4, 8, 9),
    (4, 4, 7, 9),
    (3, 4, 0, 5),
    (0, 3, 4, 5),
    (2, 1, 2, 3),
    (6, 3, 2, 7),
]


def _signed_vector(rng):
    """A vector with a whole-number length: a random choice from `_NICE`, permuted and signed. Returns (v, length)."""
    a, b, c, n = rng.choice(_NICE)
    parts = [a, b, c]
    rng.shuffle(parts)
    return tuple(p * rng.choice([1, -1]) for p in parts), n


def _term(coef: int, var: str, first: bool) -> str:
    if coef == 0:
        return ""
    sign = "-" if coef < 0 else "+"
    mag = abs(coef)
    body = var if mag == 1 else f"{mag}{var}"
    return (("-" if coef < 0 else "") + body) if first else f" {sign} {body}"


def _plane_text(n, d: int) -> str:
    out, first = "", True
    for coef, var in zip(n, "xyz"):
        piece = _term(coef, var, first)
        if piece:
            out += piece
            first = False
    return f"{out} = {d}"


def _sq(var: str, c: int) -> str:
    return f"{var}²" if c == 0 else f"({var} - {c})²" if c > 0 else f"({var} + {-c})²"


def _sphere_text(c, r2: int) -> str:
    return " + ".join(_sq(v, k) for v, k in zip("xyz", c)) + f" = {r2}"


def _line_text(p, v) -> str:
    return f"r = {vtext(p)} + λ{vtext(v)}"


def _at(p, v, lam):
    return tuple(Fraction(a) + Fraction(lam) * b for a, b in zip(p, v))


def _dist(u, w) -> float:
    return math.dist([float(x) for x in u], [float(x) for x in w])


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a point off the plane"):
        n, length = _signed_vector(rng)
        pt = tuple(rng.randint(-5, 5) for _ in range(3))
        d = rng.randint(-12, 12)
        gap = dot(n, pt) - d
        if gap != 0:
            break
    answer = Fraction(abs(gap), length)
    foot = tuple(Fraction(p) - Fraction(gap, length * length) * c for p, c in zip(pt, n))
    check(
        dot(n, foot) == d and abs(_dist(pt, foot) - float(answer)) < 1e-9,
        "distance differs from the foot of the perpendicular",
    )
    fields = num_fields(answer)
    cands = num_cands(
        [
            (Fraction(gap, length), "r3i-linesphere-absolute-value-forgotten") if gap < 0 else (None, ""),
            (Fraction(abs(gap)), "r3i-linesphere-normal-length-forgotten"),
            (Fraction(abs(gap), length * length), "r3i-linesphere-length-squared-used"),
            (Fraction(abs(gap), sum(abs(c) for c in n)), "r3i-linesphere-components-added-for-length"),
            (Fraction(abs(dot(n, pt) + d), length), "r3i-linesphere-plane-constant-sign"),
            (Fraction(abs(d), length), "r3i-linesphere-point-ignored"),
        ],
        answer,
    )
    return dict(
        text=f"Find the distance from the point {vtext(pt)} to the plane {_plane_text(n, d)}.",
        working=[
            f"The normal is {vtext(n)}, with length {length}.",
            f"Distance = |{_plane_text(n, d).split(' = ')[0]} evaluated at the point - ({d})| ÷ {length} = |{dot(n, pt)} - ({d})| ÷ {length} = {fields['answer_text']}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _line_plane(rng):
    for _ in attempts(500, "a line that meets the plane at a whole-number parameter"):
        n = tuple(rng.randint(-3, 3) for _ in range(3))
        p = tuple(rng.randint(-5, 5) for _ in range(3))
        v = tuple(rng.randint(-3, 3) for _ in range(3))
        lam = rng.choice([x for x in range(-4, 5) if x != 0])
        if (
            any(n)
            and any(v)
            and dot(n, v) != 0
            and abs(dot(n, v)) <= 9
            and dot(n, p) != 0
            and sum(c != 0 for c in n) >= 2
        ):
            break
    hit = _at(p, v, lam)
    d = int(dot(n, hit))
    return n, p, v, lam, hit, d


def _medium_lambda(rng):
    n, p, v, lam, hit, d = _line_plane(rng)
    check(dot(n, hit) == d, "intersection point is off the plane")
    fields = num_fields(lam)
    num, den = d - dot(n, p), dot(n, v)
    cands = num_cands(
        [
            (-lam, "r3i-linesphere-plane-constant-sign"),
            (Fraction(num), "r3i-linesphere-direction-dot-forgotten"),
            (Fraction(d, den), "r3i-linesphere-point-term-forgotten"),
            (Fraction(dot(n, p) - d, den) if dot(n, p) != d else None, "r3i-linesphere-moved-term-sign"),
            (Fraction(den, num) if num else None, "r3i-linesphere-quotient-inverted"),
            (Fraction(num, dot(n, p)) if dot(n, p) else None, "r3i-linesphere-divided-by-point-term"),
        ],
        lam,
    )
    return dict(
        text=(
            f"The line {_line_text(p, v)} meets the plane {_plane_text(n, d)}. Find the value of λ at the point of intersection."
        ),
        working=[
            f"Substitute the line into the plane: {dot(n, p)} + λ({dot(n, v)}) = {d}.",
            f"So λ = {num}/{den} = {fields['answer_text']}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _medium_coordinate(rng):
    n, p, v, lam, hit, d = _line_plane(rng)
    which = rng.choice([i for i in range(3) if v[i] != 0])
    name = "xyz"[which]
    value = hit[which]
    fields = num_fields(value)
    cands = num_cands(
        [
            (p[which] - lam * v[which], "r3i-linesphere-plane-constant-sign"),
            (Fraction(lam * v[which]), "r3i-linesphere-point-term-forgotten"),
            (Fraction(p[which] + lam), "r3i-linesphere-direction-dot-forgotten"),
            (Fraction(lam), "r3i-linesphere-parameter-given-for-coordinate"),
            (Fraction(p[which]) + (lam + 1) * v[which], "r3i-linesphere-moved-term-sign"),
        ],
        value,
    )
    return dict(
        text=(
            f"The line {_line_text(p, v)} meets the plane {_plane_text(n, d)}. Find the {name}-coordinate of the point of intersection."
        ),
        working=[
            f"Substitute the line into the plane: {dot(n, p)} + λ({dot(n, v)}) = {d}, so λ = {lam}.",
            f"The {name}-coordinate is {p[which]} {'+' if lam >= 0 else '-'} {abs(lam)}({v[which]}) = {fields['answer_text']}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _medium_two_points(rng):
    n, p, v, lam, hit, d = _line_plane(rng)
    a_pt = p
    b_pt = tuple(a + b for a, b in zip(p, v))
    which = rng.choice([i for i in range(3) if v[i] != 0])
    name = "xyz"[which]
    value = hit[which]
    check(abs(sum(n[i] * float(hit[i]) for i in range(3)) - d) < 1e-9, "point off the plane")
    fields = num_fields(value)
    summed = tuple(a + b for a, b in zip(a_pt, b_pt))
    wrong_lam = Fraction(d - dot(n, a_pt), dot(n, summed)) if dot(n, summed) else None
    cands = num_cands(
        [
            (Fraction(a_pt[which] - lam * v[which]), "r3i-linesphere-direction-reversed"),
            (
                None if wrong_lam is None else Fraction(a_pt[which]) + wrong_lam * summed[which],
                "r3i-linesphere-direction-added-not-subtracted",
            ),
            (Fraction(b_pt[which]) + lam * v[which], "r3i-linesphere-wrong-point-used"),
            (Fraction(lam * v[which]), "r3i-linesphere-point-term-forgotten"),
            (Fraction(lam), "r3i-linesphere-parameter-given-for-coordinate"),
        ],
        value,
    )
    return dict(
        text=(
            f"The line through A{vtext(a_pt)} and B{vtext(b_pt)} meets the plane {_plane_text(n, d)}. "
            f"Find the {name}-coordinate of the point of intersection."
        ),
        working=[
            f"The direction is B - A = {vtext(v)}, so the line is {_line_text(a_pt, v)}.",
            f"Substituting into the plane: {dot(n, a_pt)} + λ({dot(n, v)}) = {d}, so λ = {lam}.",
            f"The {name}-coordinate is {a_pt[which]} {'+' if lam >= 0 else '-'} {abs(lam)}({v[which]}) = {fields['answer_text']}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _medium(rng):
    roll = rng.random()
    return _medium_lambda(rng) if roll < 0.35 else _medium_coordinate(rng) if roll < 0.7 else _medium_two_points(rng)


# -- hard and exam: a line and a sphere ---------------------------------------------------------------------------


def _line_sphere(rng, *, exam: bool = False):
    """A line p + λv (|v| whole) meeting a sphere at the parameters m ± h. Returns the pieces."""
    for _ in attempts(2000, "a line and sphere with small whole-number data"):
        v, length = _signed_vector(rng)
        u = tuple(rng.randint(-2, 2) for _ in range(3))
        w = cross(v, u)
        vlen = dot(w, w)
        m, h = rng.randint(-3, 3), rng.randint(1, 3)
        foot = tuple(rng.randint(-4, 4) for _ in range(3))
        if 0 < vlen <= 50 and (m != 0 or not exam) and length * length * h * h + vlen <= 400:
            break
    p = tuple(f - m * c for f, c in zip(foot, v))
    centre = tuple(f + x for f, x in zip(foot, w))
    r2 = int(vlen + h * h * length * length)
    # independent check: both points are on the line and the sphere, and their distance is 2h|v|
    lam1, lam2 = m - h, m + h
    p1, p2 = _at(p, v, lam1), _at(p, v, lam2)
    for pt in (p1, p2):
        check(sum((a - b) ** 2 for a, b in zip(pt, centre)) == r2, "an intersection point is off the sphere")
    check(abs(_dist(p1, p2) - 2 * h * length) < 1e-9, "chord length differs from the distance between the points")
    check(sum((a - b) ** 2 for a, b in zip(_at(p, v, m), centre)) < r2, "the line should pass inside the sphere")
    return p, v, length, centre, r2, m, h, lam1, lam2


def _hard_chord(rng):
    p, v, length, c, r2, m, h, lam1, lam2 = _line_sphere(rng)
    answer = 2 * h * length
    fields = num_fields(answer)
    cands = num_cands(
        [
            (h * length, "r3i-linesphere-half-chord-given"),
            (2 * h, "r3i-linesphere-direction-length-forgotten"),
            (2 * h * length * length, "r3i-linesphere-length-squared-used"),
            (lam2, "r3i-linesphere-parameter-given-for-length"),
            (m + h * length, "r3i-linesphere-midpoint-added-to-length"),
        ],
        answer,
    )
    return dict(
        text=(
            f"The line {_line_text(p, v)} meets the sphere {_sphere_text(c, r2)} at two points. "
            "Find the distance between these two points."
        ),
        working=[
            f"Substitute the line into the sphere: {length * length}λ²{signed_text(2 * int(dot(v, tuple(a - b for a, b in zip(p, c)))))}λ{signed_text(int(sum((a - b) ** 2 for a, b in zip(p, c))) - r2)} = 0, with solutions λ = {lam1} and λ = {lam2}.",
            f"The points are {lam2 - lam1} parameter units apart and |{vtext(v)}| = {length}, so the distance is {lam2 - lam1} × {length} = {answer}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _hard_larger(rng):
    p, v, length, c, r2, m, h, lam1, lam2 = _line_sphere(rng)
    fields = num_fields(lam2)
    cands = num_cands(
        [
            (lam1, "r3i-linesphere-smaller-solution-given"),
            (m, "r3i-linesphere-midpoint-given"),
            (h, "r3i-linesphere-half-gap-given"),
            (-lam2, "r3i-linesphere-moved-term-sign"),
            (m + h * length, "r3i-linesphere-midpoint-added-to-length"),
        ],
        lam2,
    )
    return dict(
        text=(
            f"The line {_line_text(p, v)} meets the sphere {_sphere_text(c, r2)} at two points. "
            "Find the larger of the two values of λ at these points."
        ),
        working=[
            f"Substitute the line into the sphere: {length * length}λ²{signed_text(2 * int(dot(v, tuple(a - b for a, b in zip(p, c)))))}λ{signed_text(int(sum((a - b) ** 2 for a, b in zip(p, c))) - r2)} = 0.",
            f"The solutions are λ = {lam1} and λ = {lam2}, so the larger is {lam2}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _hard_nearest(rng):
    p, v, length, c, r2, m, h, lam1, lam2 = _line_sphere(rng)
    fields = num_fields(m)
    pc = tuple(a - b for a, b in zip(c, p))
    cands = num_cands(
        [
            (-m, "r3i-linesphere-moved-term-sign"),
            (Fraction(int(dot(pc, v))), "r3i-linesphere-length-squared-forgotten"),
            (Fraction(int(dot(pc, v)), length), "r3i-linesphere-length-not-squared"),
            (m + 1, "r3i-linesphere-nearby-value-given"),
            (lam2, "r3i-linesphere-intersection-given-for-nearest"),
        ],
        m,
    )
    return dict(
        text=(
            f"The line {_line_text(p, v)} passes through the sphere {_sphere_text(c, r2)}. Find the value of λ at the point of "
            "the line that is closest to the centre of the sphere."
        ),
        working=[
            f"The vector from {vtext(p)} to the centre {vtext(c)} is {vtext(pc)}.",
            f"λ = ({vtext(pc)} · {vtext(v)}) ÷ |{vtext(v)}|² = {int(dot(pc, v))} ÷ {length * length} = {m}.",
        ],
        distractors=pick(fields, cands),
        **fields,
    )


def _hard(rng):
    roll = rng.random()
    return _hard_chord(rng) if roll < 0.4 else _hard_larger(rng) if roll < 0.7 else _hard_nearest(rng)


@generator(
    "r3i_line_sphere",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_line_sphere(rng, tier="easy"):
    """Point-plane distance, line-plane and line-sphere intersections; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3i_exam_line_sphere", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_line_sphere(rng, tier="exam"):
    """Exam chain: substitute a line into a sphere, then use the quadratic in λ."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p, v, length, c, r2, m, h, lam1, lam2 = _line_sphere(rng, exam=True)
    pc = tuple(a - b for a, b in zip(p, c))
    big_a, big_b = length * length, int(dot(v, pc))
    big_c = int(dot(pc, pc)) - r2
    check(big_b * big_b - big_a * big_c == (big_a * h) ** 2, "the discriminant differs from (A h)²")
    check(big_c == big_a * (m * m - h * h), "C differs from A(m² - h²)")
    chord = 2 * h * length
    f_a, f_b, f_c, f_d = num_fields(big_a), num_fields(big_b), num_fields(lam1), num_fields(chord)
    disc_root = big_a * h
    return dict(
        text=(
            f"The line {_line_text(p, v)} meets the sphere {_sphere_text(c, r2)} at two points. "
            "Substituting the line into the equation of the sphere gives Aλ² + 2Bλ + C = 0."
        ),
        parts=[
            dict(
                prompt="Find A.",
                marks=1,
                sensible_range=(0, 500),
                working=[f"A = |{vtext(v)}|² = {big_a}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (length, "r3i-linesphere-length-squared-forgotten"),
                            (sum(abs(x) for x in v), "r3i-linesphere-components-added-for-length"),
                            (sum(x for x in v), "r3i-linesphere-components-added-for-length"),
                            (big_a + 1, "r3i-linesphere-squares-miscounted"),
                        ],
                        big_a,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Find B.",
                marks=2,
                sensible_range=(-500, 500),
                working=[
                    f"B = {vtext(v)} · {vtext(pc)} = {big_b}, where {vtext(pc)} is the vector from the centre to {vtext(p)}."
                ],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (-big_b, "r3i-linesphere-moved-term-sign"),
                            (2 * big_b, "r3i-linesphere-two-included-in-b"),
                            (big_b + length, "r3i-linesphere-length-added"),
                            (int(dot(v, p)), "r3i-linesphere-centre-forgotten"),
                        ],
                        big_b,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt=f"Given that C = {big_c}, find the smaller solution of Aλ² + 2Bλ + C = 0.",
                marks=2,
                follow_through=f"(-part_b - sqrt(part_b**2 - part_a*({big_c})))/part_a",
                sensible_range=(-100, 100),
                working=[
                    f"λ = (-B ± √(B² - AC)) ÷ A = (-({big_b}) ± {disc_root}) ÷ {big_a}, so the smaller solution is {lam1}."
                ],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (lam2, "r3i-linesphere-smaller-solution-given"),
                            (m, "r3i-linesphere-midpoint-given"),
                            (Fraction(-big_b - disc_root, 1), "r3i-linesphere-division-by-a-forgotten"),
                            (-lam1, "r3i-linesphere-moved-term-sign"),
                        ],
                        lam1,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Find the distance between the two points where the line meets the sphere.",
                marks=2,
                follow_through=f"2*sqrt(part_b**2 - part_a*({big_c}))/sqrt(part_a)",
                sensible_range=(0, 200),
                working=[
                    f"The solutions differ by {lam2 - lam1}, and the distance is {lam2 - lam1} × |{vtext(v)}| = {lam2 - lam1} × {length} = {chord}."
                ],
                distractors=pick(
                    f_d,
                    num_cands(
                        [
                            (h * length, "r3i-linesphere-half-chord-given"),
                            (2 * h, "r3i-linesphere-direction-length-forgotten"),
                            (2 * h * length * length, "r3i-linesphere-length-squared-used"),
                            (lam2 - lam1 + length, "r3i-linesphere-length-added"),
                        ],
                        chord,
                    ),
                ),
                **f_d,
            ),
        ],
    )
