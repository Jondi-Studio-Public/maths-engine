"""Generates `salg_polar_multiply_divide` questions (easy, medium, hard) and the exam chain
`salg_exam_polar_multiply_divide`.

Skill ``senior-define-and-use-multiplication-and-division-of-complex-numbers-in-polar-form`` (``senior-mathematics``,
answer ``pair``): multiply and divide complex numbers written as ``r cis θ`` with whole-number moduli and arguments in
degrees, by multiplying or dividing the moduli and adding or subtracting the arguments. Powers (De Moivre's theorem)
are the sibling skill ``senior-use-de-moivre-s-theorem-for-integral-powers``.

* easy: ``zw`` for ``z = r1 cis θ1`` and ``w = r2 cis θ2`` with θ1 + θ2 below 360°.
* medium: ``z/w`` with r1 a multiple of r2 and θ1 greater than θ2, so the modulus quotient is a whole number and the
  argument is a plain difference.
* hard: either the argument leaves 0° to 360° (``zw`` with θ1 + θ2 of 360° or more, or ``z/w`` with θ1 below θ2) and
  has to be brought back into range, or three numbers are combined as ``zw/u``.
* exam (``salg_exam_polar_multiply_divide``): (a) the modulus of ``zw``; (b) its argument, reduced into 0° to 360°;
  (c) the modulus of ``zw/u``, using (a); (d) the argument of ``zw/u``, using (b).

Each result is computed from the rules and confirmed by multiplying (or dividing) the Cartesian numbers with Python's
``cmath`` and reading the modulus and argument back.
"""

from __future__ import annotations

import cmath
import math

from ..core import check, draw_until, generator, phrase, pick_distractors
from ._salg_kit import int_dist, polar_dist, polar_pair

_ANGLES = list(range(10, 180, 10))
_RULE_CIS = "cis"
_PAIR_ASK = "Give the result as r cis θ by writing the pair (r, θ), with θ in degrees and 0 ≤ θ < 360."


def _cis(r: int, theta: int) -> complex:
    return cmath.rect(r, math.radians(theta))


def _confirm(value: complex, r: int, theta: int) -> None:
    """The modulus and argument of the Cartesian result are the pair the rules gave."""
    check(abs(abs(value) - r) < 1e-9, f"modulus {abs(value)} differs from {r}")
    arg = math.degrees(cmath.phase(value)) % 360
    check(min(abs(arg - theta), 360 - abs(arg - theta)) < 1e-7, f"argument {arg} differs from {theta}")


def _norm(angle: int) -> int:
    return angle % 360


def _label(name: str, r: int, theta: int) -> str:
    return f"{name} = {r} cis {theta}°"


def _pair_question(rng, stems, slots, r, theta, cands, working):
    check(len({(a, b) for a, b, _t in cands} - {(r, theta)}) >= 2, "too few distinct wrong pairs")
    answer = polar_pair(r, theta)
    dist = pick_distractors(answer.answer_text(), [polar_dist(a, b, t) for a, b, t in cands if (a, b) != (r, theta)], 3)
    check(len(dist) >= 2, "too few polar distractors")
    return dict(
        text=phrase(rng, stems, **slots) + " " + _PAIR_ASK,
        answer=answer.answer_dict(),
        answer_text=answer.answer_text(),
        answer_latex=rf"r = {r}, \theta = {theta}°",
        answer_kind="pair",
        working=working,
        distractors=dist,
    )


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    r1, r2 = draw_until(
        rng,
        lambda r: (r.randint(2, 6), r.randint(2, 6)),
        lambda p: p[0] + p[1] != p[0] * p[1],
        what="two moduli whose sum is not their product",
    )
    t1, t2 = draw_until(
        rng,
        lambda r: (r.choice(_ANGLES), r.choice(_ANGLES)),
        lambda p: p[0] != p[1] and p[0] + p[1] < 360 and p[1] != 2 * p[0] and p[0] != 2 * p[1],
        what="two arguments with a sum below 360",
    )
    r, theta = r1 * r2, t1 + t2
    _confirm(_cis(r1, t1) * _cis(r2, t2), r, theta)
    cands = [
        (r1 + r2, theta, "polar-moduli-added"),
        (r, abs(t1 - t2), "polar-arguments-subtracted-on-multiplying"),
        (r, t1, "polar-argument-of-first-only"),
        (r1 + r2, abs(t1 - t2), "polar-moduli-added"),
    ]
    return _pair_question(
        rng,
        (
            "Let {z} and {w}. Find zw.",
            "Given {z} and {w}, find zw.",
            "Two complex numbers are {z} and {w}. Find zw.",
        ),
        dict(z=_label("z", r1, t1), w=_label("w", r2, t2)),
        r,
        theta,
        cands,
        [f"Multiply the moduli: {r1} × {r2} = {r}.", f"Add the arguments: {t1}° + {t2}° = {theta}°."],
    )


# -- medium ---------------------------------------------------------------------------------------------------------


def _medium(rng):
    r2, k = draw_until(
        rng,
        lambda r: (r.randint(2, 6), r.randint(2, 7)),
        lambda p: p[0] * (p[1] - 1) != p[1],
        what="a modulus and ratio whose difference is not the ratio",
    )
    r1 = r2 * k
    t1, t2 = draw_until(
        rng,
        lambda r: (r.choice(_ANGLES), r.choice(_ANGLES)),
        lambda p: p[0] > p[1] and p[0] != 2 * p[1],
        what="two arguments with the first larger",
    )
    r, theta = k, t1 - t2
    _confirm(_cis(r1, t1) / _cis(r2, t2), r, theta)
    cands = [
        (r1 - r2, theta, "polar-moduli-subtracted"),
        (r, t1 + t2, "polar-division-arguments-added"),
        (r, _norm(t2 - t1), "polar-arguments-subtracted-in-wrong-order"),
        (r1 * r2, theta, "polar-division-as-multiplication"),
    ]
    return _pair_question(
        rng,
        (
            "Let {z} and {w}. Find z/w.",
            "Given {z} and {w}, find z/w.",
            "Two complex numbers are {z} and {w}. Find z/w.",
        ),
        dict(z=_label("z", r1, t1), w=_label("w", r2, t2)),
        r,
        theta,
        cands,
        [f"Divide the moduli: {r1} ÷ {r2} = {r}.", f"Subtract the arguments: {t1}° - {t2}° = {theta}°."],
    )


# -- hard -----------------------------------------------------------------------------------------------------------


def _hard_wrap(rng):
    if rng.random() < 0.5:
        r1, r2 = rng.randint(2, 6), rng.randint(2, 6)
        t1, t2 = draw_until(
            rng,
            lambda r: (r.choice(range(110, 360, 10)), r.choice(range(110, 360, 10))),
            lambda p: p[0] + p[1] >= 360,
            what="arguments with a sum of 360 or more",
        )
        r, raw = r1 * r2, t1 + t2
        theta = _norm(raw)
        _confirm(_cis(r1, t1) * _cis(r2, t2), r, theta)
        cands = [
            (r, raw, "polar-angle-not-reduced"),
            (r1 + r2, theta, "polar-moduli-added"),
            (r, _norm(raw - 180), "polar-angle-reduced-by-half-turn"),
            (r, abs(t1 - t2), "polar-arguments-subtracted-on-multiplying"),
        ]
        return _pair_question(
            rng,
            ("Let {z} and {w}. Find zw.", "Given {z} and {w}, find zw."),
            dict(z=_label("z", r1, t1), w=_label("w", r2, t2)),
            r,
            theta,
            cands,
            [
                f"Multiply the moduli: {r1} × {r2} = {r}.",
                f"Add the arguments: {t1}° + {t2}° = {raw}°, which is {raw - theta}° more than a full turn, so theta = {theta}°.",
            ],
        )
    r2, k = rng.randint(2, 5), rng.randint(2, 5)
    r1 = r2 * k
    t1, t2 = draw_until(
        rng,
        lambda r: (r.choice(range(10, 250, 10)), r.choice(range(40, 360, 10))),
        lambda p: p[0] < p[1],
        what="two arguments with the first smaller",
    )
    r, raw = k, t1 - t2
    theta = _norm(raw)
    _confirm(_cis(r1, t1) / _cis(r2, t2), r, theta)
    cands = [
        (r, raw, "polar-angle-not-reduced"),
        (r, abs(raw), "polar-arguments-subtracted-in-wrong-order"),
        (r, _norm(t1 + t2), "polar-division-arguments-added"),
        (r1 - r2, theta, "polar-moduli-subtracted"),
    ]
    return _pair_question(
        rng,
        ("Let {z} and {w}. Find z/w.", "Given {z} and {w}, find z/w."),
        dict(z=_label("z", r1, t1), w=_label("w", r2, t2)),
        r,
        theta,
        cands,
        [
            f"Divide the moduli: {r1} ÷ {r2} = {r}.",
            f"Subtract the arguments: {t1}° - {t2}° = {raw}°, which is negative, so add 360° to get theta = {theta}°.",
        ],
    )


def _combined_numbers(rng):
    """Moduli r1, r2, r3 with r3 a proper divisor of r1 r2 that is neither r1 nor r2."""

    def draw(r):
        return r.choice([2, 3, 4, 6]), r.choice([2, 3, 4, 5, 6])

    r1, r2 = draw_until(
        rng,
        draw,
        lambda p: any(d not in p for d in range(2, p[0] * p[1]) if (p[0] * p[1]) % d == 0),
        what="two moduli whose product has a third divisor",
    )
    divisors = [d for d in range(2, r1 * r2) if (r1 * r2) % d == 0 and d not in (r1, r2)]
    return r1, r2, rng.choice(divisors)


def _hard_combined(rng):
    r1, r2, r3 = _combined_numbers(rng)
    t1, t2, t3 = draw_until(
        rng,
        lambda r: (r.choice(range(20, 360, 10)), r.choice(range(20, 360, 10)), r.choice(range(20, 360, 10))),
        lambda p: len(set(p)) == 3 and p[0] + p[1] - p[2] not in range(0, 360),
        what="three arguments whose combination leaves 0 to 360",
    )
    r, raw = r1 * r2 // r3, t1 + t2 - t3
    theta = _norm(raw)
    _confirm(_cis(r1, t1) * _cis(r2, t2) / _cis(r3, t3), r, theta)
    cands = [
        (r1 * r2 * r3, _norm(t1 + t2 + t3), "polar-division-as-multiplication"),
        (r, _norm(t1 + t2 + t3), "polar-division-arguments-added"),
        (r, raw, "polar-angle-not-reduced"),
        (r1 * r2 - r3, theta, "polar-moduli-subtracted"),
        (r, _norm(t1 - t2 - t3), "polar-argument-signs-mixed"),
    ]
    return _pair_question(
        rng,
        ("Let {z}, {w} and {u}. Find zw/u.", "Given {z}, {w} and {u}, find zw/u."),
        dict(z=_label("z", r1, t1), w=_label("w", r2, t2), u=_label("u", r3, t3)),
        r,
        theta,
        cands,
        [
            f"Multiply and divide the moduli: {r1} × {r2} ÷ {r3} = {r}.",
            f"Add the arguments of z and w and subtract that of u: {t1}° + {t2}° - {t3}° = {raw}°.",
            f"Bring this into range: theta = {theta}°.",
        ],
    )


@generator(
    "salg_polar_multiply_divide",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def salg_polar_multiply_divide(rng, tier="easy"):
    """Multiply and divide complex numbers in polar form; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_wrap, _hard_wrap, _hard_combined])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


_EXAM_STEMS = (
    "Let {z}, {w} and {u}.",
    "Three complex numbers are {z}, {w} and {u}.",
    "Given {z}, {w} and {u}, consider the product zw and the quotient zw/u.",
)


@generator("salg_exam_polar_multiply_divide", level=12, presets={"exam": dict(tier="exam")})
def salg_exam_polar_multiply_divide(rng, tier="exam"):
    """Exam chain: |zw|, Arg zw, then |zw/u| and Arg zw/u by following through."""
    check(tier == "exam", f"unknown tier {tier!r}")
    r1, r2, r3 = _combined_numbers(rng)

    def draw(r):
        return (r.choice(range(140, 360, 10)), r.choice(range(140, 360, 10)), r.choice(range(20, 300, 10)))

    t1, t2, t3 = draw_until(
        rng,
        draw,
        lambda p: p[0] + p[1] >= 360 and len(set(p)) == 3 and (p[0] + p[1] - 360) > p[2],
        what="arguments whose sum needs reducing and exceeds the third",
    )
    prod_r, raw = r1 * r2, t1 + t2
    prod_theta = raw - 360
    quot_r, quot_theta = prod_r // r3, prod_theta - t3
    _confirm(_cis(r1, t1) * _cis(r2, t2), prod_r, prod_theta)
    _confirm(_cis(r1, t1) * _cis(r2, t2) / _cis(r3, t3), quot_r, quot_theta)
    check(0 < quot_theta < 360 and quot_r == prod_r / r3, "the chain's last parts are inconsistent")

    def number_part(prompt, key, marks, cands, working, ft=None, bounds=None):
        part = dict(
            prompt=prompt,
            answer_kind="number",
            answer=key,
            answer_text=str(key),
            answer_latex=str(key),
            marks=marks,
            sensible_range=bounds,
            working=working,
            distractors=pick_distractors(str(key), [int_dist(v, t) for v, t in cands if v != key and v > 0], 3),
        )
        check(len(part["distractors"]) >= 2, f"too few distractors for {prompt!r}")
        if ft:
            part["follow_through"] = ft
        return part

    return dict(
        text=phrase(
            rng,
            _EXAM_STEMS,
            z=_label("z", r1, t1),
            w=_label("w", r2, t2),
            u=_label("u", r3, t3),
        ),
        parts=[
            number_part(
                "Find the modulus of zw.",
                prod_r,
                1,
                [
                    (r1 + r2, "polar-moduli-added"),
                    (r1, "polar-modulus-of-first-only"),
                    (r2, "polar-modulus-of-first-only"),
                ],
                [f"|zw| = |z||w| = {r1} × {r2} = {prod_r}."],
                bounds=(0, None),
            ),
            number_part(
                "Find the argument of zw, in degrees, with 0 ≤ arg < 360.",
                prod_theta,
                2,
                [
                    (raw, "polar-angle-not-reduced"),
                    (abs(t1 - t2), "polar-arguments-subtracted-on-multiplying"),
                    ((raw - 180) % 360, "polar-angle-reduced-by-half-turn"),
                    (t1, "polar-argument-of-first-only"),
                ],
                [
                    f"Add the arguments: {t1}° + {t2}° = {raw}°.",
                    f"This is more than a full turn, so the argument is {raw}° - 360° = {prod_theta}°.",
                ],
                bounds=(0, 360),
            ),
            number_part(
                f"Hence find the modulus of zw/u.",
                quot_r,
                1,
                [
                    (prod_r * r3, "polar-division-as-multiplication"),
                    (prod_r - r3, "polar-moduli-subtracted"),
                    (prod_r + r3, "polar-moduli-added"),
                ],
                [f"|zw/u| = |zw| ÷ |u| = {prod_r} ÷ {r3} = {quot_r}."],
                ft=f"part_a/{r3}",
                bounds=(0, None),
            ),
            number_part(
                "Hence find the argument of zw/u, in degrees, with 0 ≤ arg < 360.",
                quot_theta,
                2,
                [
                    (prod_theta + t3, "polar-division-arguments-added"),
                    (t3 - prod_theta, "polar-arguments-subtracted-in-wrong-order"),
                    (prod_theta, "polar-argument-of-first-only"),
                ],
                [f"Subtract the argument of u: {prod_theta}° - {t3}° = {quot_theta}°."],
                ft=f"part_b - {t3}",
                bounds=(0, 360),
            ),
        ],
    )
