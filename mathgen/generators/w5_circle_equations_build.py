"""Circle equations in algebraic form, (x - h)^2 + (y - k)^2 = r^2: five skills, each with easy, medium, hard and exam tiers.

* `w5_circle_forms` (`circle-equation-centre-radius-form`, answer `ordered_list`): read the centre and radius from the
  equation, write the equation from a centre and radius (typed as the numbers a, b, c of (x + a)^2 + (y + b)^2 = c, so
  the sign flip and the squaring are both tested), then from a translation, a concentric circle through a point, the
  ends of a diameter, a circle through the origin and two axis points, and a circle touching both axes.
* `w5_circle_complete_square` (`circle-complete-the-square`, answer `ordered_list`): find the centre and radius from
  x^2 + y^2 + Dx + Ey + F = 0 by completing the square (whole, then half-integer centres, then a common factor or terms
  on both sides), and expand a given circle back to D, E, F.
* `w5_circle_point_position` (`circle-point-position`, answer `mcq`): which of five points lies inside, on or outside a
  circle given in centre-radius or expanded form.
* `w5_circle_unknown` (`circle-find-unknown-constant`, answer `number`): an unknown coordinate of a point on the circle,
  or the constant F of x^2 + y^2 + Dx + Ey + F = 0 for a given radius, the greatest whole F that still gives a circle,
  or tangency to an axis.
* `w5_circle_tangent` (`circle-tangent-at-a-point`, answer `number`): the gradient of the radius, the gradient and
  the intercepts of the tangent at a point of the circle.

Exam chains are `exd_x_w5_circle_forms`, `exd_x_w5_circle_complete_square`, `exd_x_w5_circle_point_position`,
`exd_x_w5_circle_unknown` and `exd_x_w5_circle_tangent`.

Every answer is built from a chosen centre and radius, then re-derived a second way: the expanded equation is
evaluated at the canonical centre plus and minus the radius along each axis, and every point used is tested by exact
integer arithmetic on its squared distance from the centre.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from ._w5_kit import (
    TooFewDistractors,
    enough,
    mcq_cands,
    mcq_fields,
    number_cands,
    number_fields,
    ordered_cands,
    ordered_fields,
    part,
    retry_draw,
    rt,
    spread,
)

# Pythagorean offsets (u, v) with u^2 + v^2 = r^2, r up to 25, and their radius.
_TRIPLES = (
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (8, 15, 17),
    (9, 12, 15),
    (12, 16, 20),
    (7, 24, 25),
    (15, 20, 25),
)


def _signed_offset(rng, triple):
    u, v, r = triple
    if rng.random() < 0.5:
        u, v = v, u
    return u * rng.choice((1, -1)), v * rng.choice((1, -1)), r


def _centre(rng, span=6):
    return rng.randint(-span, span), rng.randint(-span, span)


def _sq(var: str, shift: int) -> str:
    """(x - 3)^2, (y + 2)^2 or just x^2 when the shift is 0."""
    if shift == 0:
        return f"{var}^2"
    return f"({var} {'-' if shift > 0 else '+'} {rt(abs(shift))})^2"


def _standard(h: int, k: int, rr) -> str:
    return f"{_sq('x', h)} + {_sq('y', k)} = {rt(rr)}"


def _expanded(h: int, k: int, rr) -> str:
    D, E, F = -2 * h, -2 * k, h * h + k * k - rr
    left = join_terms([(1, "x^2"), (1, "y^2"), (D, "x"), (E, "y"), (F, "")])
    return f"{left} = 0"


def _point(p) -> str:
    return f"({p[0]}, {p[1]})"


def _on_circle(h, k, rr, x, y) -> bool:
    return (x - h) ** 2 + (y - k) ** 2 == rr


def _verify_expanded(h, k, rr) -> None:
    """The expanded equation is zero at the four axis points of the circle (second route to the same centre/radius)."""
    D, E, F = -2 * h, -2 * k, h * h + k * k - rr
    r = math.isqrt(int(rr)) if Fr(rr).denominator == 1 else None
    if r is not None and r * r == rr:
        for x, y in ((h + r, k), (h - r, k), (h, k + r), (h, k - r)):
            check(x * x + y * y + D * x + E * y + F == 0, "the expanded equation is not zero on the circle")


_ENTER_HKR = "Enter the x-coordinate of the centre, the y-coordinate of the centre and the radius, in that order."
_ENTER_ABC = "Write the equation as (x + a)^2 + (y + b)^2 = c and enter a, b and c, in that order."
_ENTER_HKRR = (
    "Enter the x-coordinate of the centre, the y-coordinate of the centre and the radius squared, in that order."
)


# -- circle forms (read and write the centre-radius equation) --------------------------------------------------------


def _forms_read(rng):
    h, k = rng.choice(((0, 0), _centre(rng), _centre(rng), (0, rng.randint(-6, 6)), (rng.randint(-6, 6), 0)))
    r = rng.randint(2, 12)
    fields = ordered_fields([h, k, r])
    cands = ordered_cands(
        fields,
        [
            ([-h, -k, r] if (h or k) else None, "w5c-centre-sign-as-written"),
            ([k, h, r] if h != k else None, "w5c-coordinates-swapped"),
            ([h, k, r * r], "w5c-radius-is-r-squared"),
            ([-h, k, r] if h else [h, -k, r], "w5c-centre-sign-as-written"),
            ([h, k, r + 1], "w5c-radius-off-by-one"),
        ],
    )
    enough(cands)
    stem = phrase(
        rng,
        ("A circle has the equation {e}. {a}", "The equation of a circle is {e}. {a}", "Consider the circle {e}. {a}"),
        e=_standard(h, k, r * r),
        a=_ENTER_HKR,
    )
    work = [
        f"Compare with (x - h)^2 + (y - k)^2 = r^2: the centre is ({h}, {k}).",
        f"The number on the right is r^2 = {r * r}, so the radius is {r}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _forms_write(rng):
    h, k = rng.choice((_centre(rng), _centre(rng), (0, rng.randint(-6, 6)), (rng.randint(-6, 6), 0)))
    r = rng.randint(2, 12)
    fields = ordered_fields([-h, -k, r * r])
    cands = ordered_cands(
        fields,
        [
            ([h, k, r * r] if (h or k) else None, "w5c-centre-sign-unchanged"),
            ([-h, -k, r], "w5c-writes-radius-not-squared"),
            ([h, k, r], "w5c-centre-sign-unchanged"),
            ([-h, -k, 2 * r], "w5c-doubles-radius-not-squares"),
            ([-k, -h, r * r] if h != k else None, "w5c-coordinates-swapped"),
        ],
    )
    enough(cands)
    stem = f"A circle has centre ({h}, {k}) and radius {r}. {_ENTER_ABC}"
    work = [
        f"(x - {paren_negative(h)})^2 + (y - {paren_negative(k)})^2 = {r}^2, so a = {-h}, b = {-k} and c = {r * r}."
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _forms_medium(rng):
    kind = rng.choice(("translate", "concentric", "diameter_unit"))
    if kind == "translate":
        r = rng.randint(2, 10)
        dx, dy = rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v])
        horizontal = f"{abs(dx)} {'right' if dx > 0 else 'left'}"
        vertical = f"{abs(dy)} {'up' if dy > 0 else 'down'}"
        fields = ordered_fields([-dx, -dy, r * r])
        cands = ordered_cands(
            fields,
            [
                ([dx, dy, r * r], "w5c-centre-sign-unchanged"),
                ([-dx, -dy, r], "w5c-writes-radius-not-squared"),
                ([-dy, -dx, r * r], "w5c-coordinates-swapped"),
                ([dx, -dy, r * r], "w5c-centre-sign-unchanged"),
            ],
        )
        stem = (
            f"The circle x^2 + y^2 = {r * r} is moved {horizontal} and {vertical}. Its new equation is "
            f"(x + a)^2 + (y + b)^2 = c. Enter a, b and c, in that order."
        )
        work = [
            f"The centre moves from (0, 0) to ({dx}, {dy}); the radius stays {r}.",
            f"So a = {-dx}, b = {-dy}, c = {r * r}.",
        ]
    elif kind == "concentric":
        h, k = _centre(rng)
        for _ in attempts(200, "a point off the centre"):
            px, py = rng.randint(-9, 9), rng.randint(-9, 9)
            if (px, py) != (h, k):
                break
        rr = (px - h) ** 2 + (py - k) ** 2
        fields = ordered_fields([-h, -k, rr])
        cands = ordered_cands(
            fields,
            [
                ([h, k, rr], "w5c-centre-sign-unchanged"),
                ([-h, -k, math.isqrt(rr) if math.isqrt(rr) ** 2 == rr else None], "w5c-writes-radius-not-squared"),
                ([-h, -k, abs(px - h) + abs(py - k)], "w5c-distance-by-adding-offsets"),
                ([-h, -k, (px + h) ** 2 + (py + k) ** 2], "w5c-uses-centre-sign-in-distance"),
            ],
        )
        stem = (
            f"A circle has centre ({h}, {k}) and passes through the point ({px}, {py}). "
            f"Its equation is (x + a)^2 + (y + b)^2 = c. Enter a, b and c, in that order."
        )
        work = [
            f"r^2 = ({px} - {paren_negative(h)})^2 + ({py} - {paren_negative(k)})^2 = {rr}.",
            f"So a = {-h}, b = {-k}, c = {rr}.",
        ]
    else:
        h, k = _centre(rng)
        u, v, r = _signed_offset(rng, rng.choice(_TRIPLES[:3]))
        a1, b1 = (h - u, k - v), (h + u, k + v)
        fields = ordered_fields([-h, -k, r * r])
        cands = ordered_cands(
            fields,
            [
                ([-(a1[0] + b1[0]), -(a1[1] + b1[1]), r * r], "w5c-centre-is-sum-not-midpoint"),
                ([-h, -k, 4 * r * r], "w5c-diameter-squared-for-radius"),
                ([-h, -k, r], "w5c-writes-radius-not-squared"),
                ([h, k, r * r], "w5c-centre-sign-unchanged"),
            ],
        )
        stem = (
            f"A circle has a diameter with ends A({a1[0]}, {a1[1]}) and B({b1[0]}, {b1[1]}). "
            f"Its equation is (x + a)^2 + (y + b)^2 = c. Enter a, b and c, in that order."
        )
        work = [f"The centre is the midpoint ({h}, {k}); the radius is half of AB, which is {r}.", f"So c = {r * r}."]
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


def _forms_hard(rng):
    kind = rng.choice(("origin_axes", "touches_axes", "origin_axes", "touches_axes", "tangent_axis"))
    if kind == "origin_axes":
        a, b = rng.choice((2, 4, 6, 8, 10, 12)), rng.choice((2, 4, 6, 8, 10, 14, 16))
        sa, sb = rng.choice((1, -1)), rng.choice((1, -1))
        a, b = a * sa, b * sb
        h, k, rr = Fr(a, 2), Fr(b, 2), Fr(a * a + b * b, 4)
        fields = ordered_fields([-h, -k, rr])
        cands = ordered_cands(
            fields,
            [
                ([-a, -b, a * a + b * b], "w5c-endpoint-not-midpoint"),
                ([h, k, rr], "w5c-centre-sign-unchanged"),
                ([-h, -k, Fr(a * a + b * b)], "w5c-diameter-squared-for-radius"),
                ([-h, -k, Fr(abs(a) + abs(b), 2) ** 2], "w5c-distance-by-adding-offsets"),
            ],
        )
        stem = (
            f"A circle passes through O(0, 0), A({a}, 0) and B(0, {b}). Its equation is "
            f"(x + p)^2 + (y + q)^2 = c. Enter p, q and c, in that order."
        )
        work = [
            "The angle at O is a right angle, so AB is a diameter.",
            f"The centre is the midpoint of AB, ({rt(h)}, {rt(k)}), and c is the squared distance from the centre to O.",
        ]
    elif kind == "touches_axes":
        r = rng.randint(2, 12)
        sx, sy = rng.choice((1, -1)), rng.choice((1, -1))
        quadrant = {(1, 1): "first", (-1, 1): "second", (-1, -1): "third", (1, -1): "fourth"}[(sx, sy)]
        h, k = sx * r, sy * r
        fields = ordered_fields([-h, -k, r * r])
        cands = ordered_cands(
            fields,
            [
                ([h, k, r * r], "w5c-centre-sign-unchanged"),
                ([-r, -r, r * r] if (sx, sy) != (1, 1) else None, "w5c-ignores-quadrant"),
                ([-h, -k, r], "w5c-writes-radius-not-squared"),
                ([-h, -k, 2 * r], "w5c-doubles-radius-not-squares"),
            ],
        )
        stem = (
            f"A circle of radius {r} lies in the {quadrant} quadrant and touches both axes. Its equation is "
            f"(x + a)^2 + (y + b)^2 = c. Enter a, b and c, in that order."
        )
        work = [
            f"Touching both axes puts the centre {r} units from each, in the {quadrant} quadrant: ({h}, {k}).",
            f"So a = {-h}, b = {-k}, c = {r * r}.",
        ]
    else:
        h, k = (
            rng.choice([v for v in range(-8, 9) if abs(v) >= 2]),
            rng.choice([v for v in range(-8, 9) if abs(v) >= 2]),
        )
        axis = rng.choice(("x-axis", "y-axis"))
        r = abs(k) if axis == "x-axis" else abs(h)
        fields = ordered_fields([-h, -k, r * r])
        wrong_axis = abs(h) if axis == "x-axis" else abs(k)
        cands = ordered_cands(
            fields,
            [
                ([-h, -k, wrong_axis**2] if wrong_axis != r else None, "w5c-tangent-wrong-axis"),
                ([h, k, r * r], "w5c-centre-sign-unchanged"),
                ([-h, -k, r], "w5c-writes-radius-not-squared"),
                ([-h, -k, h * h + k * k], "w5c-uses-distance-from-origin"),
            ],
        )
        stem = (
            f"A circle has centre ({h}, {k}) and touches the {axis}. Its equation is "
            f"(x + a)^2 + (y + b)^2 = c. Enter a, b and c, in that order."
        )
        work = [
            f"Touching the {axis} means the radius equals the distance from the centre to that axis, {r}.",
            f"So c = {r * r}, a = {-h}, b = {-k}.",
        ]
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    "w5_circle_forms",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_circle_forms(rng, tier="easy"):
    """Read and write the centre-radius equation of a circle; the tiers are described in the module docstring."""
    pool = {"easy": (_forms_read, _forms_write), "medium": (_forms_medium,), "hard": (_forms_hard,)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


def _forms_exam(rng):
    cx, cy, d0 = rng.choice(
        ((3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10), (5, 12, 13), (12, 5, 13), (0, 5, 5), (5, 0, 5), (9, 12, 15))
    )
    h, k = cx * rng.choice((1, -1)), cy * rng.choice((1, -1))
    u, v, r = _signed_offset(rng, rng.choice(_TRIPLES[:5]))
    a1, b1 = (h - u, k - v), (h + u, k + v)
    check(math.hypot(b1[0] - a1[0], b1[1] - a1[1]) == 2 * r, "diameter length differs")
    check(math.hypot(h, k) == d0, "distance to the origin differs")
    if d0 == r:
        raise TooFewDistractors("the circle passes through the origin")
    centre = ordered_fields([h, k])
    centre_d = ordered_cands(
        centre,
        [
            ([a1[0] + b1[0], a1[1] + b1[1]], "w5c-centre-is-sum-not-midpoint"),
            ([b1[0] - a1[0], b1[1] - a1[1]], "w5c-centre-is-difference"),
            ([k, h] if h != k else None, "w5c-coordinates-swapped"),
            ([a1[0], a1[1]], "w5c-endpoint-not-midpoint"),
        ],
    )
    rad = number_fields(r)
    rad_d = number_cands(
        r,
        [
            (2 * r, "w5c-diameter-for-radius"),
            (r * r, "w5c-radius-is-r-squared"),
            (abs(u) + abs(v), "w5c-distance-by-adding-offsets"),
            (r + 1, "w5c-radius-off-by-one"),
        ],
    )
    abc = ordered_fields([-h, -k, r * r])
    abc_d = ordered_cands(
        abc,
        [
            ([h, k, r * r], "w5c-centre-sign-unchanged"),
            ([-h, -k, r], "w5c-writes-radius-not-squared"),
            ([-h, -k, 4 * r * r], "w5c-diameter-squared-for-radius"),
            ([-k, -h, r * r] if h != k else None, "w5c-coordinates-swapped"),
        ],
    )
    gap = abs(d0 - r)
    gap_f = number_fields(gap, fraction=True)
    gap_d = number_cands(
        gap,
        [
            (d0 + r, "w5c-adds-radius-for-gap"),
            (d0, "w5c-distance-to-centre-not-circle"),
            (r, "w5c-radius-for-gap"),
            (abs(d0 * d0 - r * r), "w5c-uses-squared-distances"),
        ],
    )
    for d in (centre_d, rad_d, abc_d, gap_d):
        enough(d)
    sub = lambda f: {kk: vv for kk, vv in f.items() if kk != "answer_kind"}  # noqa: E731
    return dict(
        text=phrase(
            rng,
            (
                "A circle has a diameter with ends A({ax}, {ay}) and B({bx}, {by}).",
                "The segment AB, where A is ({ax}, {ay}) and B is ({bx}, {by}), is a diameter of a circle.",
                "A and B are opposite ends of a diameter of a circle: A({ax}, {ay}) and B({bx}, {by}).",
            ),
            ax=a1[0],
            ay=a1[1],
            bx=b1[0],
            by=b1[1],
        ),
        parts=[
            part(
                "Find the coordinates of the centre. Enter the x-coordinate then the y-coordinate.",
                "ordered_list",
                sub(centre),
                1,
                [f"The centre is the midpoint of AB: ({h}, {k})."],
                centre_d,
            ),
            part(
                "Find the radius of the circle.",
                rad["answer_kind"],
                sub(rad),
                1,
                [f"AB = {2 * r}, so the radius is {r}."],
                rad_d,
                sensible_range=(0, None),
            ),
            part(
                "Write the equation as (x + a)^2 + (y + b)^2 = c and enter a, b and c, in that order.",
                "ordered_list",
                sub(abc),
                2,
                [
                    f"(x - {paren_negative(h)})^2 + (y - {paren_negative(k)})^2 = {r}^2 gives a = {-h}, b = {-k}, c = {r * r}."
                ],
                abc_d,
            ),
            part(
                "What is the shortest distance from the origin O to the circle?",
                gap_f["answer_kind"],
                sub(gap_f),
                2,
                [f"The centre is {d0} from O and the radius is {r}, so the shortest distance is |{d0} - {r}| = {gap}."],
                gap_d,
                follow_through=f"Abs({d0} - part_b)",
                sensible_range=(0, None),
            ),
        ],
    )


@generator("exd_x_w5_circle_forms", level=21, presets={"exam": dict(tier="exam")})
def exd_x_w5_circle_forms(rng, tier="exam"):
    """Exam chain: centre and radius from the ends of a diameter, the equation, and the shortest distance from O."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _forms_exam)


# -- complete the square ----------------------------------------------------------------------------------------------


def _square_sum(texts: list, values: list) -> str:
    """``h^2 + k^2`` as text, leaving out a square that is zero (no ``+ 0``)."""
    kept = [str(t) for t, v in zip(texts, values) if v != 0]
    return " + ".join(kept) if kept else "0"


def _sub(fields: dict) -> dict:
    """The answer fields of a part: everything but the kind, which the part carries on its own."""
    return {key: val for key, val in fields.items() if key != "answer_kind"}


def _squares_slips(h, k, r, F):
    """Wrong (h, k, r) triples for the standard completing-the-square slips."""
    out = [
        ([-h, -k, r] if (h or k) else None, "w5c-centre-is-half-coefficient-with-its-sign"),
        ([2 * h, 2 * k, r] if (h or k) else None, "w5c-centre-not-halved"),
        ([h, k, r * r], "w5c-radius-is-r-squared"),
        ([k, h, r] if h != k else None, "w5c-coordinates-swapped"),
    ]
    root = math.isqrt(-F) if -F > 0 else None
    if root is not None and root * root == -F and root != r:
        out.append(([h, k, root], "w5c-forgets-to-add-completed-squares"))
    return out


def _cs_easy(rng):
    for _ in attempts(100, "a centre away from the origin"):
        h, k = _centre(rng)
        if (h, k) != (0, 0):
            break
    r = rng.randint(2, 10)
    rr = r * r
    F = h * h + k * k - rr
    _verify_expanded(h, k, rr)
    fields = ordered_fields([h, k, r])
    cands = ordered_cands(fields, _squares_slips(h, k, r, F))
    enough(cands)
    stem = f"A circle has the equation {_expanded(h, k, rr)}. Complete the square. {_ENTER_HKR}"
    work = [
        f"Complete the square in x and in y: {_standard(h, k, rr)}.",
        f"The centre is ({h}, {k}) and the radius is {r}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _cs_medium(rng):
    kind = rng.choice(("halves", "expand", "constant_right"))
    if kind == "halves":
        for _ in attempts(200, "a half-integer centre"):
            h2, k2 = rng.randint(-9, 9), rng.randint(-9, 9)
            if (h2 % 2 or k2 % 2) and (h2, k2) != (0, 0):
                break
        h, k = Fr(h2, 2), Fr(k2, 2)
        D, E = -h2, -k2
        for _ in attempts(200, "a real circle"):
            F = rng.randint(-12, 6)
            rr = h * h + k * k - F
            if rr > 0:
                break
        fields = ordered_fields([h, k, rr])
        cands = ordered_cands(
            fields,
            [
                ([-h, -k, rr], "w5c-centre-is-half-coefficient-with-its-sign"),
                ([D, E, rr], "w5c-centre-not-halved"),
                ([h, k, -F], "w5c-forgets-to-add-completed-squares"),
                ([h, k, h * h + k * k + F], "w5c-completed-squares-subtracted-twice"),
            ],
        )
        left = join_terms([(1, "x^2"), (1, "y^2"), (D, "x"), (E, "y"), (F, "")])
        stem = f"A circle has the equation {left} = 0. Complete the square. {_ENTER_HKRR} Write fractions such as 3/2."
        work = [
            (
                f"({join_terms([(1, 'x^2'), (D, 'x')])}) gives {_sq('x', h)} - {rt(h * h)}; do the same for y."
                if D
                else "x^2 is already a square; complete the square for y in the same way."
            ),
            f"r^2 = {_square_sum([rt(h * h), rt(k * k)], [h * h, k * k])} - ({F}) = {rt(rr)}.",
        ]
    elif kind == "expand":
        h, k = _centre(rng)
        r = rng.randint(2, 9)
        D, E, F = -2 * h, -2 * k, h * h + k * k - r * r
        fields = ordered_fields([D, E, F])
        cands = ordered_cands(
            fields,
            [
                ([2 * h, 2 * k, F], "w5c-expands-with-wrong-signs"),
                ([-h, -k, F], "w5c-linear-term-not-doubled"),
                ([D, E, h * h + k * k + r * r], "w5c-radius-squared-added-not-subtracted"),
                ([D, E, r * r - h * h - k * k], "w5c-constant-sign-reversed"),
            ],
        )
        stem = (
            f"The circle {_standard(h, k, r * r)} is written as x^2 + y^2 + Dx + Ey + F = 0. "
            f"Enter D, E and F, in that order."
        )
        work = [
            f"Expand: {join_terms([(1, 'x^2'), (D, 'x'), (h * h, '')])} + {join_terms([(1, 'y^2'), (E, 'y'), (k * k, '')])} = {r * r}.",
            f"So D = {D}, E = {E}, F = {F}.",
        ]
    else:
        for _ in attempts(100, "a centre away from the origin"):
            h, k = _centre(rng)
            if (h, k) != (0, 0):
                break
        r = rng.randint(2, 10)
        D, E, F = -2 * h, -2 * k, h * h + k * k - r * r
        fields = ordered_fields([h, k, r])
        cands = ordered_cands(fields, _squares_slips(h, k, r, F))
        left = join_terms([(1, "x^2"), (1, "y^2"), (D, "x"), (E, "y")])
        stem = f"A circle has the equation {left} = {-F}. Complete the square. {_ENTER_HKR}"
        work = [
            f"Add {h * h} and {k * k} to both sides: right side {-F} + {h * h + k * k} = {r * r}.",
            f"The radius is {r}.",
        ]
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


def _cs_hard(rng):
    kind = rng.choice(("factor", "both_sides", "factor", "both_sides_factor"))
    h, k = _centre(rng, 5)
    if (h, k) == (0, 0):
        h = 2
    rr = rng.randint(2, 40)
    F = h * h + k * k - rr
    fields = ordered_fields([h, k, rr])
    cands = ordered_cands(
        fields,
        [
            ([-h, -k, rr], "w5c-centre-is-half-coefficient-with-its-sign"),
            ([h, k, -F], "w5c-forgets-to-add-completed-squares"),
            ([h, k, rr * rr] if rr > 1 else None, "w5c-squares-the-radius-squared"),
            ([2 * h, 2 * k, rr], "w5c-centre-not-halved"),
        ],
    )
    enough(cands)
    D, E = -2 * h, -2 * k
    if kind in ("factor", "both_sides_factor"):
        m = rng.choice((2, 3, 4))
        left = join_terms([(m, "x^2"), (m, "y^2"), (m * D, "x"), (m * E, "y"), (m * F, "")])
        cands = ordered_cands(
            fields,
            [
                ([h, k, rr * m], "w5c-ignores-the-common-factor"),
                ([m * h, m * k, rr], "w5c-ignores-the-common-factor"),
                ([-h, -k, rr], "w5c-centre-is-half-coefficient-with-its-sign"),
                ([h, k, -F], "w5c-forgets-to-add-completed-squares"),
            ],
        )
        enough(cands)
        stem = f"A circle has the equation {left} = 0. {_ENTER_HKRR}"
        work = [
            f"Divide every term by {m}: {join_terms([(1, 'x^2'), (1, 'y^2'), (D, 'x'), (E, 'y'), (F, '')])} = 0, then complete the square."
        ]
        if kind == "both_sides_factor":
            stem = (
                f"A circle has the equation {join_terms([(m, 'x^2'), (m, 'y^2')])} = "
                f"{join_terms([(-m * D, 'x'), (-m * E, 'y'), (-m * F, '')])}. {_ENTER_HKRR}"
            )
            work = ["Bring every term to the left side, divide by the common factor, then complete the square."]
    else:
        right = join_terms([(-D, "x"), (-E, "y"), (-F, "")])
        stem = f"A circle has the equation x^2 + y^2 = {right}. {_ENTER_HKRR}"
        work = [
            "Bring the x and y terms to the left side: the signs of their coefficients change.",
            "Then complete the square.",
        ]
    work.append(f"The centre is ({h}, {k}) and r^2 = {rr}.")
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    "w5_circle_complete_square",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_circle_complete_square(rng, tier="easy"):
    """Centre and radius from the expanded equation of a circle; the tiers are described in the module docstring."""
    pool = {"easy": (_cs_easy,), "medium": (_cs_medium,), "hard": (_cs_hard,)}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


_CHORD = ((3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10), (5, 12, 13), (12, 5, 13), (9, 12, 15), (12, 9, 15), (8, 15, 17))


def _cs_exam(rng):
    ak, w, r = rng.choice(_CHORD)  # |k|, half the chord on the x-axis, and the radius
    k = ak * rng.choice((1, -1))
    h = rng.randint(-8, 8)
    check(k * k + w * w == r * r, "triple differs")
    D, E, F = -2 * h, -2 * k, h * h + k * k - r * r
    stem_eq = join_terms([(1, "x^2"), (1, "y^2"), (D, "x"), (E, "y"), (F, "")])
    centre = ordered_fields([h, k])
    centre_d = ordered_cands(
        centre,
        [
            ([-h, -k], "w5c-centre-is-half-coefficient-with-its-sign"),
            ([D, E], "w5c-centre-not-halved"),
            ([k, h] if h != k else None, "w5c-coordinates-swapped"),
            ([h, -k], "w5c-centre-is-half-coefficient-with-its-sign"),
        ],
    )
    rad = number_fields(r)
    rad_d = number_cands(
        r,
        [
            (r * r, "w5c-radius-is-r-squared"),
            (-F, "w5c-forgets-to-add-completed-squares"),
            (r + ak, "w5c-adds-the-centre-offset"),
            (w, "w5c-chord-half-for-radius"),
        ],
    )
    from ._w5_kit import list_cands, list_fields

    on_line = list_fields([k - r, k + r])
    on_line_d = list_cands(
        on_line,
        [
            ([-k - r, -k + r], "w5c-centre-is-half-coefficient-with-its-sign"),
            ([h - r, h + r], "w5c-x-for-y"),
            ([-r, r], "w5c-ignores-centre-offset"),
            ([k + r], "w5c-one-intersection-only"),
        ],
    )
    chord = number_fields(2 * w)
    chord_d = number_cands(
        2 * w,
        [
            (w, "w5c-half-chord-for-chord"),
            (2 * r, "w5c-diameter-for-chord"),
            (2 * ak, "w5c-uses-centre-height-for-chord"),
            (2 * (r - ak), "w5c-chord-by-subtracting"),
        ],
    )
    for d in (centre_d, rad_d, on_line_d, chord_d):
        enough(d)
    return dict(
        text=f"A circle has the equation {stem_eq} = 0.",
        parts=[
            part(
                "Find the centre of the circle. Enter the x-coordinate then the y-coordinate.",
                "ordered_list",
                _sub(centre),
                2,
                ["Complete the square in x and in y."],
                centre_d,
            ),
            part(
                "Find the radius of the circle.",
                rad["answer_kind"],
                _sub(rad),
                2,
                [f"r^2 = {_square_sum([h * h, k * k], [h * h, k * k])} - ({F}) = {r * r}."],
                rad_d,
                sensible_range=(0, None),
            ),
            part(
                f"Find the y-coordinates of the points where the circle meets the vertical line x = {h}.",
                "list",
                _sub(on_line),
                1,
                [f"On x = {h} the points are directly above and below the centre, a distance r away."],
                on_line_d,
            ),
            part(
                "The circle meets the x-axis at two points. How long is the chord between them?",
                chord["answer_kind"],
                _sub(chord),
                2,
                [
                    f"Half the chord is the square root of r^2 - k^2 = {r * r} - {k * k} = {w * w}, so the chord is {2 * w}."
                ],
                chord_d,
                follow_through=f"2*sqrt(part_b**2 - ({k})**2)",
                sensible_range=(0, None),
            ),
        ],
    )


@generator("exd_x_w5_circle_complete_square", level=21, presets={"exam": dict(tier="exam")})
def exd_x_w5_circle_complete_square(rng, tier="exam"):
    """Exam chain: centre, radius, the points on a vertical line and the chord cut off on the x-axis."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _cs_exam)


# -- point inside, on or outside --------------------------------------------------------------------------------------


def _side(d2: int, rr: int) -> str:
    return "inside" if d2 < rr else "on" if d2 == rr else "outside"


def _wrong_rules(target: str, h: int, k: int, rr: int):
    """(tag, predicate on (x, y)): a point a learner using this slip would call `target`."""

    def dist2(x, y, cx=h, cy=k):
        return (x - cx) ** 2 + (y - cy) ** 2

    rules = []
    if (h, k) != (0, 0):
        rules.append(("w5c-centre-sign-flipped", lambda x, y: _side(dist2(x, y, -h, -k), rr) == target))
        rules.append(("w5c-ignores-centre", lambda x, y: _side(x * x + y * y, rr) == target))
    rules.append(
        ("w5c-compares-squared-distance-with-radius", lambda x, y: _side(dist2(x, y), int(math.isqrt(rr))) == target)
    )
    rules.append(
        (
            "w5c-adds-offsets-without-squaring",
            lambda x, y: _side((abs(x - h) + abs(y - k)) ** 2, rr) == target,
        )
    )
    return rules


def _point_options(rng, h, k, rr, target, n_wrong=4):
    """The correct point of side `target` and `n_wrong` points of other sides, each tagged with a slip that picks it."""
    r = math.isqrt(rr)
    box = [(x, y) for x in range(h - r - 3, h + r + 4) for y in range(k - r - 3, k + r + 4)]
    by_side = {"inside": [], "on": [], "outside": []}
    for x, y in box:
        by_side[_side((x - h) ** 2 + (y - k) ** 2, rr)].append((x, y))
    if target == "on":
        pool = [p for p in by_side["on"] if p != (h + r, k)] or by_side["on"]
    elif target == "inside":
        pool = [p for p in by_side["inside"] if p != (h, k)]
    else:
        pool = [p for p in by_side["outside"] if (p[0] - h) ** 2 + (p[1] - k) ** 2 < 4 * rr]
    correct = rng.choice(pool)
    wrong = []
    used = {correct}
    rules = _wrong_rules(target, h, k, rr)
    for tag, rule in rules:
        cands = [
            p for p in box if p not in used and _side((p[0] - h) ** 2 + (p[1] - k) ** 2, rr) != target and rule(*p)
        ]
        if cands:
            p = rng.choice(cands)
            wrong.append((p, tag))
            used.add(p)
    # fill with points just across the boundary, which are the honest near misses
    boundary = [
        p
        for p in box
        if p not in used
        and _side((p[0] - h) ** 2 + (p[1] - k) ** 2, rr) != target
        and abs((p[0] - h) ** 2 + (p[1] - k) ** 2 - rr) <= max(3, r)
    ]
    rng.shuffle(boundary)
    for p in boundary:
        if len(wrong) >= n_wrong:
            break
        tag = "w5c-near-miss-treated-as-on" if target == "on" else "w5c-boundary-misjudged"
        wrong.append((p, tag))
        used.add(p)
    wrong = wrong[:n_wrong]
    if len(wrong) < 3:
        raise TooFewDistractors("not enough points")
    return correct, wrong


def _position_question(rng, expanded: bool, centred: bool):
    for _ in attempts(100, "a circle"):
        r = rng.randint(3, 10)
        h, k = (0, 0) if not centred else _centre(rng, 5)
        if centred and (h, k) == (0, 0):
            continue
        break
    rr = r * r
    target = rng.choice(("inside", "on", "outside", "inside", "outside"))
    correct, wrong = _point_options(rng, h, k, rr, target)
    check(_side((correct[0] - h) ** 2 + (correct[1] - k) ** 2, rr) == target, "the key is on the wrong side")
    fields = mcq_fields(_point(correct))
    cands = mcq_cands(fields["answer_text"], [(_point(p), tag) for p, tag in wrong])
    enough(cands, 3)
    options = [fields["answer_text"], *[c.text for c in cands]]
    spread(options, fields["answer_text"])
    eq = _expanded(h, k, rr) if expanded else _standard(h, k, rr)
    stem = phrase(
        rng,
        (
            "Which of these points lies {w} the circle {e}?",
            "Exactly one of the following points lies {w} the circle {e}. Which one?",
        ),
        w=target,
        e=eq,
    )
    work = [
        f"The centre is ({h}, {k}) and r^2 = {rr}.",
        f"For each point work out (x - {paren_negative(h)})^2 + (y - {paren_negative(k)})^2 and compare it with {rr}: less means inside, equal means on the circle, more means outside.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _pos_easy(rng):
    return _position_question(rng, expanded=False, centred=False)


def _pos_medium(rng):
    return _position_question(rng, expanded=False, centred=True)


def _pos_hard(rng):
    return _position_question(rng, expanded=True, centred=True)


@generator(
    "w5_circle_point_position",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_circle_point_position(rng, tier="easy"):
    """Which point lies inside, on or outside a circle; the tiers are described in the module docstring."""
    build = {"easy": _pos_easy, "medium": _pos_medium, "hard": _pos_hard}.get(tier)
    if build is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, build)


def _pos_exam(rng):
    cx, cy, d = rng.choice(
        ((3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10), (5, 12, 13), (12, 5, 13), (9, 12, 15), (0, 5, 5), (5, 0, 5))
    )
    h, k = _centre(rng, 4)
    px, py = h + cx * rng.choice((1, -1)), k + cy * rng.choice((1, -1))
    r = rng.choice([v for v in range(3, 12) if v != d])
    rr = r * r
    check((px - h) ** 2 + (py - k) ** 2 == d * d, "P is not a whole distance from the centre")
    eq = _standard(h, k, rr)
    dist = number_fields(d)
    dist_d = number_cands(
        d,
        [
            (d * d, "w5c-forgets-square-root"),
            (abs(px - h) + abs(py - k), "w5c-distance-by-adding-offsets"),
            (abs(px + h) + abs(py + k) if (h or k) else None, "w5c-uses-centre-sign-in-distance"),
            (d + r, "w5c-adds-radius-for-gap"),
        ],
    )
    side = rng.choice(("inside", "outside"))
    correct, wrong = _point_options(rng, h, k, rr, side)
    mc = mcq_fields(_point(correct))
    mc_d = mcq_cands(mc["answer_text"], [(_point(p), tag) for p, tag in wrong])
    enough(mc_d, 3)
    least = number_fields(abs(d - r), fraction=True)
    least_d = number_cands(
        abs(d - r),
        [
            (d + r, "w5c-adds-radius-for-gap"),
            (d, "w5c-distance-to-centre-not-circle"),
            (abs(d * d - rr), "w5c-uses-squared-distances"),
            (r, "w5c-radius-for-gap"),
        ],
    )
    far = number_fields(d + r, fraction=True)
    far_d = number_cands(
        d + r,
        [
            (abs(d - r), "w5c-subtracts-radius-for-farthest"),
            (d, "w5c-distance-to-centre-not-circle"),
            (d + rr, "w5c-adds-squared-radius"),
            (2 * (d + r), "w5c-doubles-the-distance"),
        ],
    )
    for dd in (dist_d, least_d, far_d):
        enough(dd)
    return dict(
        text=f"A circle has the equation {eq}. The point P is ({px}, {py}).",
        parts=[
            part(
                "How far is P from the centre of the circle?",
                dist["answer_kind"],
                _sub(dist),
                1,
                [f"({px} - {paren_negative(h)})^2 + ({py} - {paren_negative(k)})^2 = {d * d}, so the distance is {d}."],
                dist_d,
                sensible_range=(0, None),
            ),
            part(
                f"Which of these points lies {side} the circle?",
                "mcq",
                _sub(mc),
                2,
                [f"Compare (x - {paren_negative(h)})^2 + (y - {paren_negative(k)})^2 with {rr} for each point."],
                mc_d,
            ),
            part(
                "Find the shortest distance from P to the circle.",
                least["answer_kind"],
                _sub(least),
                2,
                [f"P is {d} from the centre and the radius is {r}: |{d} - {r}| = {abs(d - r)}."],
                least_d,
                follow_through=f"Abs(part_a - {r})",
                sensible_range=(0, None),
            ),
            part(
                "Find the greatest distance from P to a point on the circle.",
                far["answer_kind"],
                _sub(far),
                1,
                [f"The farthest point is on the far side of the centre: {d} + {r} = {d + r}."],
                far_d,
                follow_through=f"part_a + {r}",
                sensible_range=(0, None),
            ),
        ],
    )


@generator("exd_x_w5_circle_point_position", level=21, presets={"exam": dict(tier="exam")})
def exd_x_w5_circle_point_position(rng, tier="exam"):
    """Exam chain: the distance from P to the centre, a point on one side of the circle, and the nearest and farthest
    distances from P to the circle."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _pos_exam)


# -- an unknown coordinate or constant ----------------------------------------------------------------------------------


def _expanded_with_f(h, k, unknown: str = "F") -> str:
    return join_terms([(1, "x^2"), (1, "y^2"), (-2 * h, "x"), (-2 * k, "y"), (1, unknown)]) + " = 0"


def _unknown_easy(rng):
    h, k = _centre(rng)
    u, v, r = _signed_offset(rng, rng.choice(_TRIPLES[:6]))
    solve_y = rng.random() < 0.5
    above = rng.random() < 0.5
    if solve_y:
        # the point is (h + u, t); t = k + v or k - v, chosen by "above"/"below" the centre
        t = k + (abs(v) if above else -abs(v))
        other = k + (-abs(v) if above else abs(v))
        px, py, unknown = h + u, "t", t
        word = "above" if above else "below"
        wrong = [
            (other, "w5c-takes-the-other-point"),
            (abs(v) if above else -abs(v), "w5c-forgets-the-centre-offset"),
            (-k + (abs(v) if above else -abs(v)), "w5c-centre-sign-as-written"),
            (k + r if above else k - r, "w5c-radius-for-offset"),
        ]
        check(_on_circle(h, k, r * r, px, t), "the point is not on the circle")
    else:
        t = h + (abs(u) if above else -abs(u))
        other = h + (-abs(u) if above else abs(u))
        px, py, unknown = "t", k + v, t
        word = "to the right of" if above else "to the left of"
        wrong = [
            (other, "w5c-takes-the-other-point"),
            (abs(u) if above else -abs(u), "w5c-forgets-the-centre-offset"),
            (-h + (abs(u) if above else -abs(u)), "w5c-centre-sign-as-written"),
            (h + r if above else h - r, "w5c-radius-for-offset"),
        ]
        check(_on_circle(h, k, r * r, t, py), "the point is not on the circle")
    fields = number_fields(unknown)
    cands = number_cands(unknown, wrong)
    enough(cands)
    stem = f"The point ({px}, {py}) lies on the circle {_standard(h, k, r * r)} and is {word} the centre. Find t."
    work = [
        f"Substitute the known coordinate: the other squared term must equal {r * r} - {(abs(u) if solve_y else abs(v)) ** 2}.",
        f"That gives t - {paren_negative(k if solve_y else h)} = ±{abs(v) if solve_y else abs(u)}; the point is {word} the centre, so t = {unknown}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _unknown_medium(rng):
    kind = rng.choice(("radius", "radius", "through_point"))
    h, k = _centre(rng)
    if (h, k) == (0, 0):
        h = 3
    if kind == "radius":
        r = rng.randint(2, 10)
        F = h * h + k * k - r * r
        fields = number_fields(F)
        cands = number_cands(
            F,
            [
                (r * r - h * h - k * k, "w5c-constant-sign-reversed"),
                (h * h + k * k + r * r, "w5c-radius-squared-added-not-subtracted"),
                (-r * r, "w5c-constant-is-minus-r-squared"),
                (h * h + k * k - r, "w5c-writes-radius-not-squared"),
            ],
        )
        stem = f"The equation {_expanded_with_f(h, k)} describes a circle of radius {r}. Find F."
        work = [
            f"Complete the square: r^2 = {_square_sum([h * h, k * k], [h * h, k * k])} - F = {r * r}.",
            f"So F = {F}.",
        ]
    else:
        for _ in attempts(100, "a point off the centre"):
            px, py = rng.randint(-8, 8), rng.randint(-8, 8)
            if (px, py) != (h, k):
                break
        F = -(px * px + py * py - 2 * h * px - 2 * k * py)
        fields = number_fields(F)
        cands = number_cands(
            F,
            [
                (px * px + py * py - 2 * h * px - 2 * k * py, "w5c-constant-sign-reversed"),
                (-(px * px + py * py + 2 * h * px + 2 * k * py), "w5c-expands-with-wrong-signs"),
                (-(px * px + py * py), "w5c-ignores-the-linear-terms"),
                (h * h + k * k - ((px - h) ** 2 + (py - k) ** 2), "w5c-confuses-f-with-centre-form"),
            ],
        )
        stem = f"The circle {_expanded_with_f(h, k)} passes through the point ({px}, {py}). Find F."
        work = [f"Substitute x = {px}, y = {py} and solve for F: F = {F}."]
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


def _unknown_hard(rng):
    kind = rng.choice(("greatest", "tangent_x", "tangent_y", "greatest", "tangent_x"))
    for _ in attempts(100, "a centre off both axes"):
        h, k = _centre(rng, 9)
        if h and k:
            break
    base = h * h + k * k
    if kind == "greatest":
        F = base - 1
        cands = number_cands(
            F,
            [
                (base, "w5c-allows-zero-radius"),
                (base + 1, "w5c-wrong-direction-for-bound"),
                (-base + 1, "w5c-constant-sign-reversed"),
                (base - 2, "w5c-off-by-one-at-the-bound"),
            ],
        )
        stem = (
            f"The equation {_expanded_with_f(h, k)} describes a circle. Find the greatest whole number that F can be."
        )
        work = [
            f"A circle needs r^2 = {base} - F > 0, so F < {base}.",
            f"The greatest whole number below {base} is {F}.",
        ]
    elif kind == "tangent_x":
        F = h * h
        cands = number_cands(
            F,
            [
                (k * k, "w5c-tangent-wrong-axis"),
                (base, "w5c-allows-zero-radius"),
                (-h * h, "w5c-constant-sign-reversed"),
                (h * h + k * k - abs(k), "w5c-writes-radius-not-squared"),
            ],
        )
        stem = f"The circle {_expanded_with_f(h, k)} touches the x-axis. Find F."
        work = [f"Touching the x-axis means r = |k| = {abs(k)}, so r^2 = {k * k} = {base} - F.", f"F = {F}."]
    else:
        F = k * k
        cands = number_cands(
            F,
            [
                (h * h, "w5c-tangent-wrong-axis"),
                (base, "w5c-allows-zero-radius"),
                (-k * k, "w5c-constant-sign-reversed"),
                (h * h + k * k - abs(h), "w5c-writes-radius-not-squared"),
            ],
        )
        stem = f"The circle {_expanded_with_f(h, k)} touches the y-axis. Find F."
        work = [f"Touching the y-axis means r = |h| = {abs(h)}, so r^2 = {h * h} = {base} - F.", f"F = {F}."]
    fields = number_fields(F)
    enough(cands)
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    "w5_circle_unknown",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_circle_unknown(rng, tier="easy"):
    """An unknown coordinate or constant in a circle equation; the tiers are described in the module docstring."""
    build = {"easy": _unknown_easy, "medium": _unknown_medium, "hard": _unknown_hard}.get(tier)
    if build is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, build)


def _unknown_exam(rng):
    for _ in attempts(100, "a centre off both axes"):
        h, k = _centre(rng, 7)
        if h and k:
            break
    r = rng.randint(2, 9)
    base = h * h + k * k
    stem_eq = _expanded_with_f(h, k)
    a = number_fields(h)
    a_d = number_cands(
        h,
        [
            (-h, "w5c-centre-is-half-coefficient-with-its-sign"),
            (-2 * h, "w5c-centre-not-halved"),
            (k, "w5c-coordinates-swapped"),
        ],
    )
    greatest = base - 1
    b = number_fields(greatest, fraction=True)
    b_d = number_cands(
        greatest,
        [
            (base, "w5c-allows-zero-radius"),
            (base + 1, "w5c-wrong-direction-for-bound"),
            (greatest - 1, "w5c-off-by-one-at-the-bound"),
        ],
    )
    F_r = base - r * r
    c = number_fields(F_r)
    c_d = number_cands(
        F_r,
        [
            (r * r - base, "w5c-constant-sign-reversed"),
            (base + r * r, "w5c-radius-squared-added-not-subtracted"),
            (-r * r, "w5c-constant-is-minus-r-squared"),
        ],
    )
    F_t = h * h
    d = number_fields(F_t, fraction=True)
    d_d = number_cands(
        F_t,
        [(k * k, "w5c-tangent-wrong-axis"), (base, "w5c-allows-zero-radius"), (-h * h, "w5c-constant-sign-reversed")],
    )
    for dd in (a_d, b_d, c_d, d_d):
        enough(dd)
    return dict(
        text=f"The equation {stem_eq} describes a circle for suitable values of F.",
        parts=[
            part(
                "Find the x-coordinate of the centre.",
                a["answer_kind"],
                _sub(a),
                1,
                ["Halve the coefficient of x and change its sign."],
                a_d,
            ),
            part(
                "Find the greatest whole number that F can be.",
                b["answer_kind"],
                _sub(b),
                2,
                [f"r^2 = {base} - F must be positive, so F < {base}; the greatest whole number is {greatest}."],
                b_d,
                follow_through=f"part_a**2 + ({k})**2 - 1",
            ),
            part(
                f"Find F when the circle has radius {r}.",
                c["answer_kind"],
                _sub(c),
                2,
                [f"{base} - F = {r * r}, so F = {F_r}."],
                c_d,
            ),
            part(
                "Find F when the circle touches the x-axis.",
                d["answer_kind"],
                _sub(d),
                2,
                [f"r = |k| = {abs(k)}, so {base} - F = {k * k} and F = {F_t}."],
                d_d,
                follow_through="part_a**2",
            ),
        ],
    )


@generator("exd_x_w5_circle_unknown", level=21, presets={"exam": dict(tier="exam")})
def exd_x_w5_circle_unknown(rng, tier="exam"):
    """Exam chain: the centre's x-coordinate, the greatest whole F, F for a given radius and F for tangency."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _unknown_exam)


# -- the tangent at a point ---------------------------------------------------------------------------------------------


def _tangent_draw(rng):
    """A centre, an offset (u, v) with both parts non-zero, the squared radius u^2 + v^2 and the point P on the circle."""
    h, k = _centre(rng, 5)
    u, v = rng.choice([n for n in range(-9, 10) if n]), rng.choice([n for n in range(-9, 10) if n])
    rr = u * u + v * v
    return h, k, u, v, rr, h + u, k + v


def _tangent_stem(h, k, rr, px, py):
    return f"The point P({px}, {py}) lies on the circle {_standard(h, k, rr)}."


def _tangent_easy(rng):
    h, k, u, v, rr, px, py = _tangent_draw(rng)
    check(_on_circle(h, k, rr, px, py), "P is not on the circle")
    g = Fr(v, u)
    fields = number_fields(g)
    cands = number_cands(
        g,
        [
            (Fr(u, v), "w5c-gradient-inverted"),
            (-g, "w5c-gradient-sign-flipped"),
            (Fr(py, px) if px else None, "w5c-gradient-from-origin"),
            (Fr(v + k, u + h) if u + h else None, "w5c-gradient-with-centre-sign-flipped"),
        ],
    )
    enough(cands)
    stem = f"{_tangent_stem(h, k, rr, px, py)} Find the gradient of the radius from the centre to P."
    work = [
        f"The centre is ({h}, {k}). Gradient = ({py} - {paren_negative(k)})/({px} - {paren_negative(h)}) = {rt(g)}."
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _tangent_medium(rng):
    h, k, u, v, rr, px, py = _tangent_draw(rng)
    g = Fr(v, u)
    m = -Fr(u, v)
    check(m * g == -1, "tangent is not perpendicular to the radius")
    fields = number_fields(m)
    cands = number_cands(
        m,
        [
            (g, "w5c-tangent-parallel-to-radius"),
            (Fr(u, v), "w5c-gradient-inverted"),
            (-g, "w5c-gradient-sign-flipped"),
        ],
    )
    enough(cands)
    stem = f"{_tangent_stem(h, k, rr, px, py)} Find the gradient of the tangent to the circle at P."
    work = [
        f"The radius to P has gradient {rt(g)}.",
        f"The tangent is perpendicular to it, so its gradient is the negative reciprocal, {rt(m)}.",
    ]
    return dict(text=stem, working=work, distractors=cands, **fields)


def _tangent_hard(rng):
    h, k, u, v, rr, px, py = _tangent_draw(rng)
    m = -Fr(u, v)
    which = rng.choice(("y_intercept", "x_intercept", "y_intercept", "x_intercept"))
    c = py - m * px
    x0 = px - Fr(py, 1) / m
    # second route: the tangent is u(x - h) + v(y - k) = r^2, found from the dot product with the radius
    check(u * (px - h) + v * (py - k) == rr, "tangent line equation differs")
    if which == "y_intercept":
        value = c
        wrong = [
            (py, "w5c-intercept-is-the-y-coordinate"),
            (py - (-1 / m) * px, "w5c-intercept-with-radius-gradient"),
            (py + m * px, "w5c-intercept-sign-error"),
            (-c, "w5c-intercept-sign-error"),
        ]
        target = "where the tangent meets the y-axis. Find the y-coordinate of that point"
        work = [
            f"The tangent is y - {paren_negative(py)} = {rt(m)}(x - {paren_negative(px)}).",
            f"Put x = 0: y = {rt(value)}.",
        ]
    else:
        value = x0
        wrong = [
            (px, "w5c-intercept-is-the-x-coordinate"),
            (px + Fr(py, 1) / m, "w5c-intercept-sign-error"),
            (px - Fr(py, 1) * (-1 / m), "w5c-intercept-with-radius-gradient"),
            (-x0, "w5c-intercept-sign-error"),
        ]
        target = "where the tangent meets the x-axis. Find the x-coordinate of that point"
        work = [
            f"The tangent is y - {paren_negative(py)} = {rt(m)}(x - {paren_negative(px)}).",
            f"Put y = 0: x = {rt(value)}.",
        ]
    fields = number_fields(value)
    cands = number_cands(value, wrong)
    enough(cands)
    stem = f"{_tangent_stem(h, k, rr, px, py)} The tangent to the circle at P meets the axes; consider {target}."
    return dict(text=stem, working=work, distractors=cands, **fields)


@generator(
    "w5_circle_tangent",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5_circle_tangent(rng, tier="easy"):
    """The radius and the tangent at a point of a circle; the tiers are described in the module docstring."""
    build = {"easy": _tangent_easy, "medium": _tangent_medium, "hard": _tangent_hard}.get(tier)
    if build is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, build)


def _tangent_exam(rng):
    h, k, u, v, rr, px, py = _tangent_draw(rng)
    g, m = Fr(v, u), -Fr(u, v)
    c = py - m * px
    x0 = px - Fr(py, 1) / m
    ga = number_fields(g)
    ga_d = number_cands(
        g,
        [
            (Fr(u, v), "w5c-gradient-inverted"),
            (-g, "w5c-gradient-sign-flipped"),
            (Fr(v + k, u + h) if u + h else None, "w5c-gradient-with-centre-sign-flipped"),
        ],
    )
    gb = number_fields(m, fraction=True)
    gb_d = number_cands(
        m,
        [(g, "w5c-tangent-parallel-to-radius"), (Fr(u, v), "w5c-gradient-inverted"), (-g, "w5c-gradient-sign-flipped")],
    )
    cc = number_fields(c, fraction=True)
    cc_d = number_cands(
        c,
        [
            (py, "w5c-intercept-is-the-y-coordinate"),
            (py + m * px, "w5c-intercept-sign-error"),
            (-c, "w5c-intercept-sign-error"),
            (py - (-1 / m) * px, "w5c-intercept-with-radius-gradient"),
        ],
    )
    dd = number_fields(x0, fraction=True)
    dd_d = number_cands(
        x0,
        [
            (px, "w5c-intercept-is-the-x-coordinate"),
            (px + Fr(py, 1) / m, "w5c-intercept-sign-error"),
            (-x0, "w5c-intercept-sign-error"),
            (px - Fr(py, 1) * (-1 / m), "w5c-intercept-with-radius-gradient"),
        ],
    )
    for d in (ga_d, gb_d, cc_d, dd_d):
        enough(d)
    return dict(
        text=f"{_tangent_stem(h, k, rr, px, py)} The tangent to the circle at P meets the axes at A on the y-axis and B on the x-axis.",
        parts=[
            part(
                "Find the gradient of the radius from the centre to P.",
                ga["answer_kind"],
                _sub(ga),
                1,
                [f"Gradient = ({py} - {paren_negative(k)})/({px} - {paren_negative(h)}) = {rt(g)}."],
                ga_d,
            ),
            part(
                "Hence find the gradient of the tangent at P.",
                gb["answer_kind"],
                _sub(gb),
                1,
                [f"The tangent is perpendicular to the radius: use the negative reciprocal, {rt(m)}."],
                gb_d,
                follow_through="-1/part_a",
            ),
            part(
                "Find the y-coordinate of A.",
                cc["answer_kind"],
                _sub(cc),
                2,
                [f"y - {paren_negative(py)} = {rt(m)}(x - {paren_negative(px)}); put x = 0 to get y = {rt(c)}."],
                cc_d,
                follow_through=f"{py} - part_b*({px})",
            ),
            part(
                "Find the x-coordinate of B.",
                dd["answer_kind"],
                _sub(dd),
                2,
                [f"Put y = 0 in y - {paren_negative(py)} = {rt(m)}(x - {paren_negative(px)}), so x = {rt(x0)}."],
                dd_d,
                follow_through=f"{px} - {py}/part_b",
            ),
        ],
    )


@generator("exd_x_w5_circle_tangent", level=21, presets={"exam": dict(tier="exam")})
def exd_x_w5_circle_tangent(rng, tier="exam"):
    """Exam chain: gradient of the radius, gradient of the tangent, and its two intercepts (with follow-through)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _tangent_exam)
