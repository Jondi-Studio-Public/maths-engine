"""Exam-tier chain for `senior-recognise-and-use-the-additivity-and-linearity-of`: definite integrals from given values.

The question gives integrals of ``f`` over two adjacent intervals (one of them stated with its limits reversed on
half the draws) and of ``g`` over the whole interval ``[a, c]``. No function is ever integrated. The chain asks for

(a) the integral of ``f`` over ``[a, c]``,
(b) the integral of ``p f(x) + q g(x) + r`` over ``[a, c]``, which also integrates a constant,
(c) the constant ``k`` that makes the integral of ``f(x) + k g(x)`` over ``[a, c]`` a stated value, and
(d) the mean value of ``f`` over ``[a, c]``.

Everything is an exact integer except (d), a whole number divided by the length of the interval; the length is
always 2, 4, 5 or 10 so (d) terminates within two decimal places.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms

from .exd_x_r2i_support import number_part

_LENGTHS = [2, 4, 5, 10]


@generator("exd_x_r2i_integral_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_integral_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: combine stated integrals of f and g, including a constant, then solve for k and find a mean value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "stated integrals that give a non-trivial chain"):
        length = rng.choice(_LENGTHS)
        a = rng.randint(-3, 3)
        c = a + length
        b = rng.randint(a + 1, c - 1)
        i1 = rng.randint(-9, 14)
        i2 = rng.randint(-9, 14)
        j = rng.choice([x for x in range(-9, 10) if abs(x) >= 2])
        p = rng.choice([2, 3, 4, -2, -3, 5])
        q = rng.choice([1, 2, 3, -1, -2, -3])
        r = rng.choice([x for x in range(-6, 7) if x != 0])
        k = rng.choice([x for x in range(-5, 6) if x not in (0, 1)])
        total_f = i1 + i2
        if total_f == 0 or abs(total_f) < 3 or total_f % length == 0:
            continue
        if i1 == 0 or i2 == 0 or i1 == i2:
            continue
        break
    reversed_limits = rng.random() < 0.5
    stated_second = -i2 if reversed_limits else i2
    value_b = p * total_f + q * j + r * length
    target = total_f + k * j
    mean = Fraction(total_f, length)
    check(value_b - r * length == p * total_f + q * j, "linearity disagrees")
    check(Fraction(target - total_f, j) == k, "k does not reproduce the stated integral")

    second_limits = (c, b) if reversed_limits else (b, c)
    stem = (
        f"For two functions f and g it is known that the integral of f(x) from x = {a} to x = {b} is {i1}, "
        f"the integral of f(x) from x = {second_limits[0]} to x = {second_limits[1]} is {stated_second}, "
        f"and the integral of g(x) from x = {a} to x = {c} is {j}."
    )
    combo = join_terms([(p, "f(x)"), (q, "g(x)"), (r, "")])
    plain_second = i1 - stated_second if reversed_limits else i1 + stated_second
    check(plain_second == total_f, "the integral of f over the whole interval disagrees")

    wrong_a = i1 + stated_second if reversed_limits else i1 - stated_second
    parts = [
        number_part(
            f"Find the integral of f(x) from x = {a} to x = {c}.",
            Fraction(total_f),
            0,
            1,
            [
                f"The second integral is taken from {second_limits[0]} to {second_limits[1]}, so "
                + (f"reversing the limits gives {i2}." if reversed_limits else f"it is already from {b} to {c}."),
                f"Integral from {a} to {c} = {join_terms([(i1, ''), (i2, '')])} = {total_f}.",
            ],
            [
                (wrong_a, "ignored-reversed-limits" if reversed_limits else "subtracted-instead-of-added"),
                (i1, "gave-a-part-not-the-whole"),
                (i2, "gave-a-part-not-the-whole"),
            ],
        ),
        number_part(
            f"Find the integral of {combo} from x = {a} to x = {c}.",
            Fraction(value_b),
            0,
            2,
            [
                f"{join_terms([(1 if p > 0 else -1, f'{abs(p)} × {total_f}'), (1 if q > 0 else -1, f'{abs(q)} × {j}'), (1 if r > 0 else -1, f'{abs(r)} × ({c} − {a})')])} = {value_b}.",
            ],
            [
                (total_f + j + r * length, "dropped-scalar-multiplier"),
                (p * total_f + q * j + r, "constant-integrated-as-value"),
                (p * total_f + q * j, "constant-integrated-as-value"),
            ],
            follow_through=f"({p})*part_a + ({q})*({j}) + ({r})*{length}",
        ),
        number_part(
            f"Find the constant k such that the integral of f(x) + k g(x) from x = {a} to x = {c} is {target}. "
            "Give k as a whole number or a fraction.",
            Fraction(k),
            None,
            2,
            [f"{total_f} + k × {j} = {target}.", f"k = ({target} − {total_f}) ÷ {j} = {k}."],
            [
                (Fraction(target + total_f, j), "sign-error-moving-term"),
                (Fraction(target, j), "left-out-a-value"),
                (Fraction(total_f - target, j), "reversed-difference"),
            ],
            follow_through=f"({target} - part_a)/({j})",
        ),
        number_part(
            f"Find the mean value of f(x) over the interval from x = {a} to x = {c}. Give your answer to 2 decimal places.",
            mean,
            2,
            1,
            [f"Mean value = (1 ÷ {length}) × {total_f} = {float(mean):.2f}."],
            [
                (Fraction(total_f), "divided-by-wrong-amount"),
                (Fraction(total_f, b - a), "divided-by-wrong-amount"),
                (Fraction(-total_f, length), "answer-sign-reversed"),
            ],
            follow_through=f"part_a/{length}",
        ),
    ]
    return dict(text=stem, parts=parts)
