"""Exam-tier questions for four complex-number skills that practice at a single step:

- ``exd_x_complex_point``: locate a point in the plane after a geometric operation (the fourth vertex of a
  parallelogram given as complex numbers, or a quarter-turn about a point that is not the origin).
- ``exd_x_complex_i_power``: evaluate a power of ``1 + i`` or ``1 - i`` (pairing the binomial square with the
  four-cycle of powers of ``i``), or a signed sum of large and negative powers of ``i``.
- ``exd_x_complex_nth_root``: using the equal spacing of the cube roots of a complex number to find a missing
  root, or to pick the root lying in a stated quadrant.
- ``exd_x_complex_equating``: equate real and imaginary parts of a non-linear equation (a square, or a number
  times its conjugate), where a stated sign condition selects a single solution.

Every question is built backwards from the chosen answer, so each key is exact; each key is also re-derived
by a second route (Python's ``cmath`` complex arithmetic) before it is returned.
"""

from __future__ import annotations

import cmath
import math
import random

from ..core import check, generator, join_terms, phrase, pick_distractors
from .p5b_complex_plane_build import ComplexPair

_PRESETS = {"exam": dict(tier="exam")}


def _require_exam(tier: str) -> None:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}: this topic only has an exam tier")


def _ctext(re_part: int, im_part: int) -> str:
    return join_terms([(re_part, ""), (im_part, "i")])


# -- point in the plane ------------------------------------------------------------


def _xy(x: int, y: int) -> ComplexPair:
    return ComplexPair("x", float(x), "y", float(y))


def _point_question(rng: random.Random) -> dict:
    shape = rng.choice(("parallelogram", "rotation"))
    while True:
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        b = (rng.randint(-8, 8), rng.randint(-8, 8))
        c = (rng.randint(-8, 8), rng.randint(-8, 8))
        if len({a, b, c}) < 3:
            continue
        if shape == "parallelogram":
            # Reject degenerate (collinear) triples: the three given points must form a real parallelogram.
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0:
                continue
            d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
            wrong = [
                (a[0] + b[0] - c[0], a[1] + b[1] - c[1]),  # the vertex opposite a different corner
                (b[0] + c[0] - a[0], b[1] + c[1] - a[1]),
                (a[0] + c[0], a[1] + c[1]),  # sum of the two neighbours, no subtraction
                (d[1], d[0]),
            ]
            expr = (
                f"The points A, B and C represent the complex numbers {_ctext(*a)}, {_ctext(*b)} and "
                f"{_ctext(*c)}. ABCD is a parallelogram with its vertices in that order."
            )
            ask = "State the Cartesian coordinates (x, y) of the point D."
            working = ["D = A + C - B, because the diagonals of a parallelogram bisect each other."]
            z_d = complex(*a) + complex(*c) - complex(*b)
        else:
            direction = rng.choice((1, -1))
            factor = 1j * direction
            centre = b
            z0 = complex(*a)
            zc = complex(*centre)
            image = zc + factor * (z0 - zc)
            d = (int(round(image.real)), int(round(image.imag)))
            about_origin = factor * z0
            opposite = zc + (-factor) * (z0 - zc)
            wrong = [
                (int(round(about_origin.real)), int(round(about_origin.imag))),
                (int(round(opposite.real)), int(round(opposite.imag))),
                (a[1], a[0]),
                (centre[0] + (a[1] - centre[1]), centre[1] + (a[0] - centre[0])),
            ]
            turn = "anticlockwise" if direction == 1 else "clockwise"
            expr = (
                f"The point P represents the complex number {_ctext(*a)}. P is rotated {turn} through "
                f"a quarter turn about the point C, which represents {_ctext(*centre)}."
            )
            ask = "State the Cartesian coordinates (x, y) of the image of P."
            working = [
                f"Image = c + ({'i' if direction == 1 else '-i'})(p - c), since a quarter turn multiplies by ±i."
            ]
            z_d = image
        if any(isinstance(v, float) for v in d):
            continue
        if abs(z_d - complex(*d)) > 1e-9:
            continue
        if len({d, *wrong}) == 5:
            break
    answer = _xy(*d)
    distractors = pick_distractors(answer.answer_text(), [_xy(*w).answer_text() for w in wrong])
    check(len(distractors) == 3, "expected three distinct distractors")
    text = f"{expr} {ask}"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer.answer_dict(),
        answer_text=answer.answer_text(),
        answer_latex=rf"({d[0]}, {d[1]})",
        working=working,
        distractors=distractors,
    )


@generator("exd_x_complex_point", level=12, presets=_PRESETS)
def exd_x_complex_point(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a point located by a geometric operation on complex numbers."""
    _require_exam(tier)
    return _point_question(rng)


# -- powers of i ---------------------------------------------------------------------

_CYCLE = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _ipow(n: int, shift: int = 0, absolute: bool = False) -> tuple[int, int]:
    m = abs(n) if absolute else n
    return _CYCLE[(m + shift) % 4]


def _i_power_question(rng: random.Random) -> dict:
    shape = rng.choice(("binomial", "binomial", "sum"))
    if shape == "binomial":
        while True:
            sign = rng.choice((1, -1))
            m = rng.randint(2, 7)
            n = 2 * m
            # (1 + i)^2 = 2i and (1 - i)^2 = -2i, so (1 +- i)^(2m) = 2^m * (+-1)^m * i^m
            mag = 2**m
            base = _CYCLE[m % 4]
            factor = sign**m
            key = (mag * factor * base[0], mag * factor * base[1])
            wrong_cycle = _CYCLE[(m + 1) % 4]
            candidates = [
                (mag * factor * wrong_cycle[0], mag * factor * wrong_cycle[1]),  # off by one in the i-cycle
                (-key[0], -key[1]),  # sign slip on the whole result
                (m * factor * base[0], m * factor * base[1]),  # m used as the multiplier instead of 2^m
                (key[1], key[0]),  # real and imaginary parts swapped
            ]
            if len({key, *candidates}) == 5:
                break
        z = complex(1, sign) ** n
        check(abs(z - complex(*key)) < 1e-6, "(1 +- i)^n must match the constructed value")
        expr_text = f"(1 {'+' if sign == 1 else '-'} i)^{n}"
        patterns = (
            "Evaluate {e}. Give your answer in the form a + bi.",
            "Simplify {e}, giving your answer in the form a + bi.",
            "Write {e} in the form a + bi.",
        )
        text = phrase(rng, patterns, e=expr_text)
        working = [
            "(1 + i)^2 = 2i and (1 - i)^2 = -2i, so the power is a power of 2i or -2i.",
            f"The result is {_ctext(*key)}.",
        ]
    else:
        while True:
            exps = [rng.randint(20, 120), rng.randint(-60, -5), rng.randint(20, 120)]
            signs = [1, rng.choice((1, -1)), -1]
            key = (0, 0)
            for s, e in zip(signs, exps):
                v = _ipow(e)
                key = (key[0] + s * v[0], key[1] + s * v[1])
            shifted = (0, 0)
            absolute = (0, 0)
            minus = (0, 0)
            for s, e in zip(signs, exps):
                v = _ipow(e, 1)
                shifted = (shifted[0] + s * v[0], shifted[1] + s * v[1])
                v = _ipow(e, absolute=True)
                absolute = (absolute[0] + s * v[0], absolute[1] + s * v[1])
                v = _ipow(e, -1)
                minus = (minus[0] + s * v[0], minus[1] + s * v[1])
            candidates = [shifted, absolute, minus, (key[1], key[0])]
            if len({key, *candidates}) == 5 and key != (0, 0):
                break
        parts = []
        for idx, (s, e) in enumerate(zip(signs, exps)):
            term = f"i^{e}" if e >= 0 else f"i^({e})"
            parts.append(term if idx == 0 and s == 1 else (f"- {term}" if s == -1 else f"+ {term}"))
        expr_text = " ".join(parts)
        z = sum(s * 1j**e if e >= 0 else s * (1 / 1j) ** (-e) for s, e in zip(signs, exps))
        check(abs(z - complex(*key)) < 1e-6, "the sum of powers of i must match complex arithmetic")
        patterns = (
            "Evaluate {e}. Give your answer in the form a + bi.",
            "Simplify {e}, giving your answer in the form a + bi.",
        )
        text = phrase(rng, patterns, e=expr_text)
        working = ["Reduce each exponent modulo 4 (a negative exponent included), then add."]
    answer_text = _ctext(*key)
    distractors = pick_distractors(answer_text, [_ctext(*c) for c in candidates])
    check(len(distractors) == 3, "expected three distinct distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator("exd_x_complex_i_power", level=12, presets=_PRESETS)
def exd_x_complex_i_power(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a power of 1 +- i, or a signed sum of large and negative powers of i."""
    _require_exam(tier)
    return _i_power_question(rng)


# -- n-th roots ----------------------------------------------------------------------

_BASES = (2, 3, 4)


def _polar(r: int, theta: int) -> ComplexPair:
    return ComplexPair("r", float(r), "theta", float(theta))


def _root_question(rng: random.Random) -> dict:
    shape = rng.choice(("missing_root", "quadrant_root"))
    base = rng.choice(_BASES)
    if shape == "missing_root":
        t0 = rng.randrange(0, 120, 10)
        angles = [t0, t0 + 120, t0 + 240]
        missing = rng.choice(angles)
        given = [a for a in angles if a != missing]
        rng.shuffle(given)
        r_w = base**3
        # the third root completes the set of three roots of one number: cube each and compare
        cubes = {cmath.rect(r_w, math.radians(3 * a)) for a in angles}
        check(
            all(abs(c - cmath.rect(r_w, math.radians(3 * given[0]))) < 1e-6 for c in cubes),
            "the three angles must be the cube roots of a single number",
        )
        key = (base, missing)
        wrong = [
            (base, (max(given) + 120)),  # next step round with no reduction
            (base, (min(given) - 120)),
            (missing, base),
            (base**3, missing),
        ]
        text = (
            f"Two of the cube roots of a complex number w are {base}cis({given[0]}°) and "
            f"{base}cis({given[1]}°), where each root is written r cis(θ°) with 0 ≤ θ < 360. "
            f"Find the third cube root of w in polar form (r, θ), 0 ≤ θ < 360."
        )
        working = ["The three cube roots are equally spaced, 120° apart, on a circle about the origin."]
    else:
        while True:
            theta_w = rng.randrange(15, 360, 15)
            t0 = theta_w // 3
            roots = [t0, t0 + 120, t0 + 240]
            quads = {}
            for qn in range(4):
                inside = [a for a in roots if 90 * qn < a < 90 * (qn + 1)]
                if len(inside) == 1:
                    quads[qn] = inside[0]
            if quads:
                break
        qn = rng.choice(sorted(quads))
        root_angle = quads[qn]
        r_w = base**3
        check(
            abs(cmath.rect(base, math.radians(root_angle)) ** 3 - cmath.rect(r_w, math.radians(theta_w))) < 1e-6,
            "the chosen root cubed must equal w",
        )
        key = (base, root_angle)
        others = [a for a in roots if a != root_angle]
        wrong = [(base, t0), (base, others[0]), (base, others[1]), (root_angle, base)]
        names = ["first", "second", "third", "fourth"]
        text = (
            f"Let w = {r_w}cis({theta_w}°). One of the cube roots of w lies strictly inside the "
            f"{names[qn]} quadrant of the complex plane. Find that root in polar form (r, θ), "
            f"with 0 ≤ θ < 360."
        )
        working = [
            "Cube roots: r^(1/3) cis((θ + 360k)/3). Compute all three angles and choose the one in the stated quadrant."
        ]
    answer = _polar(*key)
    distractors = pick_distractors(answer.answer_text(), [_polar(*w).answer_text() for w in wrong])
    check(len(distractors) == 3, "expected three distinct distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer.answer_dict(),
        answer_text=answer.answer_text(),
        answer_latex=rf"r = {key[0]},\; \theta = {key[1]}°",
        working=working,
        distractors=distractors,
    )


@generator("exd_x_complex_nth_root", level=12, presets=_PRESETS)
def exd_x_complex_nth_root(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the missing cube root, or the cube root in a stated quadrant."""
    _require_exam(tier)
    return _root_question(rng)


# -- equating real and imaginary parts of a non-linear equation -------------------------


def _xy_answer(x: int, y: int) -> tuple[str, dict, str]:
    return f"x = {x}, y = {y}", {"x": x, "y": y}, rf"x = {x},\; y = {y}"


def _equating_question(rng: random.Random) -> dict:
    shape = rng.choice(("square", "square_imaginary", "conj_plus_multiple"))
    x0 = rng.randint(1, 6)
    y0 = rng.choice([v for v in range(-6, 7) if v != 0])
    if shape == "square":
        # z^2 has exactly two roots, z0 and -z0, and exactly one has x > 0.
        w = (x0 * x0 - y0 * y0, 2 * x0 * y0)
        text = f"Let z = x + yi, where x and y are real and x > 0. If z^2 = {_ctext(*w)}, find x and y."
        wrong = [(-x0, -y0), (y0, x0), (x0, -y0), (-x0, y0)]
        z0 = complex(x0, y0)
        check(abs(z0 * z0 - complex(*w)) < 1e-9, "z^2 must equal w")
        check(abs(cmath.sqrt(complex(*w)) - z0) < 1e-9 or abs(cmath.sqrt(complex(*w)) + z0) < 1e-9, "root check")
        working = ["z^2 = (x^2 - y^2) + 2xyi. Equate real and imaginary parts, then use the sign condition."]
    elif shape == "square_imaginary":
        # Here the sign condition is on y: draw y0 > 0 and x0 of either sign.
        x0, y0 = y0, abs(x0)
        w = (x0 * x0 - y0 * y0, 2 * x0 * y0)
        text = f"Let z = x + yi, where x and y are real and y > 0. If z^2 = {_ctext(*w)}, find x and y."
        wrong = [(-x0, -y0), (y0, x0), (x0, -y0), (-x0, y0)]
        z0 = complex(x0, y0)
        check(abs(z0 * z0 - complex(*w)) < 1e-9, "z^2 must equal w")
        check(abs(cmath.sqrt(complex(*w)) - z0) < 1e-9 or abs(cmath.sqrt(complex(*w)) + z0) < 1e-9, "root check")
        working = ["z^2 = (x^2 - y^2) + 2xyi. Equate real and imaginary parts, then use the sign condition."]
    else:
        # z * conj(z) + k z = (x^2 + y^2 + k x) + k y i. The imaginary part gives y, then x solves
        # x^2 + k x + (y^2 - r) = 0, whose roots sum to -k < 0, so exactly one of them is positive.
        k = rng.choice((2, 3, 4, 5))
        w = (x0 * x0 + y0 * y0 + k * x0, k * y0)
        text = (
            f"Let z = x + yi, where x and y are real and x > 0. If z multiplied by its conjugate, plus "
            f"{k} times z, equals {_ctext(*w)}, find x and y."
        )
        wrong = [(-x0 - k, y0), (y0, x0), (x0, -y0), (x0, y0 + 1)]
        z0 = complex(x0, y0)
        check(abs(z0 * z0.conjugate() + k * z0 - complex(*w)) < 1e-9, "z*conj(z) + k*z must equal w")
        check(-k - x0 < 0 and x0 > 0, "the other root of the quadratic in x must be negative")
        working = [
            "z times its conjugate is x^2 + y^2. The imaginary parts give y; the real parts give a quadratic "
            "in x, and the condition x > 0 keeps one root."
        ]
    answer_text, answer, answer_latex = _xy_answer(x0, y0)
    cands = [f"x = {cx}, y = {cy}" for cx, cy in wrong]
    distractors = pick_distractors(answer_text, cands)
    check(len(distractors) == 3, "expected three distinct distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_latex,
        working=working,
        distractors=distractors,
    )


@generator("exd_x_complex_equating", level=12, presets=_PRESETS)
def exd_x_complex_equating(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: equate parts of a non-linear equation in z = x + yi, with a sign condition on x or y."""
    _require_exam(tier)
    return _equating_question(rng)
