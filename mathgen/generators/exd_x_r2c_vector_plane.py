"""Exam chain for the Cartesian equation of a plane through three points.

A plane passes through ``A``, ``B`` and ``C`` and is written ``a x + b y + z = d``. Set the way a senior exam sets it:

(a) the constant a                                                  fraction, 2 marks
(b) the constant b                                                  fraction, 1 mark
(c) the constant d, using a and b                                   fraction, 2 marks, follow-through (a), (b)
(d) the missing coordinate k of a point P(x1, y1, k) on the plane   fraction, 2 marks, follow-through (a), (b), (c)

## Construction

``AB`` and ``AC`` are drawn with small whole-number components and the normal is ``n = AB x AC``. The z-component
of the normal is kept non-zero, and at most 6 in size, so the plane can be written with a coefficient of 1 on z:
``a = nx/nz``, ``b = ny/nz``, ``d = (n . A)/nz``. Those are generally fractions. A, B and C are each checked
against the printed equation, and ``k`` is checked by putting P back into it.

## Distractors

* (a), (b): the sign of the normal component wrong, the component left undivided, the x and y components swapped,
  the normal taken as the sum of the edge vectors.
* (c): the constant undivided, the z term subtracted, the z term forgotten, the sign of d wrong.
* (d): the x and y terms added instead of subtracted, the y term forgotten, d given as k.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import check, generator, latex_answer, phrase, pick_distractors

_STEMS = (
    "The plane Π passes through the points A({a}), B({b}) and C({c}). Π has Cartesian equation "
    "ax + by + z = d, where a, b and d are constants.",
    "A plane contains the points A({a}), B({b}) and C({c}). Its Cartesian equation can be written as "
    "ax + by + z = d for constants a, b and d.",
    "Consider the plane through A({a}), B({b}) and C({c}). It has an equation of the form ax + by + z = d, "
    "where a, b and d are constants.",
)


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _part(prompt, value, marks, working, wrong, follow=None):
    value = Fraction(value)
    text = _text(value)
    cands = [Distractor(_text(Fraction(v)), tag, exact_value(Fraction(v))) for v, tag in wrong if v is not None]
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, cands),
    )
    if follow:
        part["follow_through"] = follow
    return part


def _cross(u, w):
    return (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])


def _ratio(num, den):
    return Fraction(num, den) if den else None


@generator("exd_x_r2c_vector_plane", level=12, presets={"exam": {"tier": "exam"}})
def exd_x_r2c_vector_plane(rng, tier: str = "exam"):
    """A plane through three points, written ax + by + z = d: a, b, d, and a missing coordinate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(1000):  # bounded rejection loop
        point_a = (
            rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]),
            rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]),
            rng.randint(-5, 5),
        )
        u = tuple(rng.randint(-4, 4) for _ in range(3))
        w = tuple(rng.randint(-4, 4) for _ in range(3))
        n = _cross(u, w)
        if 1 <= abs(n[2]) <= 6 and n[0] != 0 and n[1] != 0 and len({n[0], n[1], n[2]}) == 3:
            a_val, b_val = Fraction(n[0], n[2]), Fraction(n[1], n[2])
            if a_val.denominator != 1 or b_val.denominator != 1 or rng.random() < 0.35:
                break
    else:  # pragma: no cover
        check(False, "no usable draw")
    point_b = tuple(point_a[i] + u[i] for i in range(3))
    point_c = tuple(point_a[i] + w[i] for i in range(3))
    d_val = Fraction(sum(n[i] * point_a[i] for i in range(3)), n[2])
    x1, y1 = rng.choice([v for v in range(-6, 7) if v]), rng.choice([v for v in range(-6, 7) if v])
    k_val = d_val - a_val * x1 - b_val * y1

    # Independent check: each of A, B, C satisfies a x + b y + z = d, and P with the key k does too.
    for point in (point_a, point_b, point_c):
        check(a_val * point[0] + b_val * point[1] + point[2] == d_val, "a point is off the printed plane")
    check(a_val * x1 + b_val * y1 + k_val == d_val, "P is off the plane")

    def pt(point):
        return ", ".join(str(v) for v in point)

    stem = phrase(rng, _STEMS, a=pt(point_a), b=pt(point_b), c=pt(point_c))

    # Wrong normals: the sum of the edge vectors in place of the cross product.
    summed = tuple(u[i] + w[i] for i in range(3))
    a_sum, b_sum = _ratio(summed[0], summed[2]), _ratio(summed[1], summed[2])
    part_a = _part(
        "Find the value of a.",
        a_val,
        2,
        [
            f"AB = ({pt(u)}) and AC = ({pt(w)}), so a normal is AB × AC = ({pt(n)}).",
            f"Dividing by the z-component: a = {_text(a_val)}.",
        ],
        [
            (-a_val, "sign-reversed-result"),
            (Fraction(n[0]), "forgot-to-divide-by-coefficient"),
            (b_val, "gave-the-other-coefficient"),
            (a_sum, "edge-sum-used-as-normal"),
        ],
    )
    part_b = _part(
        "Find the value of b.",
        b_val,
        1,
        [f"b = {n[1]}/{n[2]} = {_text(b_val)}."],
        [
            (-b_val, "sign-reversed-result"),
            (Fraction(n[1]), "forgot-to-divide-by-coefficient"),
            (a_val, "gave-the-other-coefficient"),
            (b_sum, "edge-sum-used-as-normal"),
        ],
    )
    part_c = _part(
        "Find the value of d.",
        d_val,
        2,
        [f"d = a·{point_a[0]} + b·{point_a[1]} + {point_a[2]} = {_text(d_val)}."],
        [
            (Fraction(sum(n[i] * point_a[i] for i in range(3))), "forgot-to-divide-by-coefficient"),
            (a_val * point_a[0] + b_val * point_a[1] - point_a[2], "sign-error-moving-term"),
            (a_val * point_a[0] + b_val * point_a[1], "dropped-constant-term"),
            (-d_val, "sign-error-moving-term"),
            (a_val * point_a[0] - b_val * point_a[1] + point_a[2], "sign-error-moving-term"),
            (-a_val * point_a[0] + b_val * point_a[1] + point_a[2], "sign-error-moving-term"),
        ],
        follow=f"part_a*({point_a[0]}) + part_b*({point_a[1]}) + ({point_a[2]})",
    )
    part_d = _part(
        f"The point P({x1}, {y1}, k) lies on the plane. Find the value of k.",
        k_val,
        2,
        [f"k = d - a·({x1}) - b·({y1}) = {_text(k_val)}."],
        [
            (d_val + a_val * x1 + b_val * y1, "sign-error-moving-term"),
            (d_val - a_val * x1, "left-out-a-term-when-solving-for-constant"),
            (d_val - b_val * y1, "left-out-a-term-when-solving-for-constant"),
            (d_val, "gave-d-for-k"),
            (-k_val, "sign-error-moving-term"),
        ],
        follow=f"part_c - part_a*({x1}) - part_b*({y1})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
