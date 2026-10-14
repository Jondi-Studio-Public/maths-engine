"""Exam tier for the long-term behaviour of a first-order linear recurrence: a three-part chain through the fixed point.

u(n+1) = a u(n) + b with a > 0, a != 1, and a stated u(0). The behaviour is decided by the fixed point F = b/(1 - a)
and by which side of it u(0) lies on, because u(n) - F = (u(0) - F) a^n. The chain follows that argument:

(a) the fixed point F, the value that the recurrence leaves unchanged;
(b) the signed difference u(0) - F, whose sign (with a) says whether the sequence rises, falls or settles;
(c) the term u(N) for a stated N, to 2 decimal places, from F + (u(0) - F) a^N.

Parts (b) and (c) follow through from the learner's own earlier answers. Both a > 1 (a balance with a withdrawal) and
0 < a < 1 (a dose that decays) are drawn, so the learner meets rising, falling and settling sequences.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import number_part

# factor -> the step F must be a multiple of so that b is a whole number
_FACTORS = {
    "1.05": 20,
    "1.1": 10,
    "1.2": 5,
    "1.25": 4,
    "1.5": 2,
    "0.9": 10,
    "0.8": 5,
    "0.75": 4,
    "0.6": 5,
    "0.5": 2,
}
_MARGIN = Fraction(1, 50)


def _clear(value: Fraction) -> bool:
    return abs((value * 100) % 1 - Fraction(1, 2)) > _MARGIN


@generator("exd_x_r3h_recurrence_steady_state", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_recurrence_steady_state(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a recurrence whose terms stay positive and clear of rounding"):
        a_text = rng.choice(list(_FACTORS))
        a = Fraction(a_text)
        fixed = _FACTORS[a_text] * rng.randint(1, 15)
        b = fixed * (1 - a)
        if b.denominator != 1:
            continue
        b = int(b)
        dev = rng.randint(-(fixed - 1), fixed)
        n = rng.randint(3, 6)
        if dev == 0 or dev * a**n + fixed < 1:
            continue
        start = fixed + dev
        term = fixed + dev * a**n
        if _clear(term):
            break
    check(start * a + b - fixed == a * (start - fixed), "u(1) - F = a (u(0) - F)")
    walk = Fraction(start)
    for _ in range(n):
        walk = a * walk + b
    check(walk == term, "iterating the recurrence agrees with the closed form")

    if a > 1:
        # Only a few per cent a year is a believable savings rate; faster growth is a bacterial culture.
        subject = rng.choice(("balance", "culture")) if a_text == "1.05" else "culture"
        if subject == "balance":
            rule = f"A(n+1) = {a_text} A(n) - {-b}"
            lead = (
                f"The balance A(n), in dollars, of a savings account n years after it is opened satisfies "
                f"{rule}, with A(0) = {start}."
            )
        else:
            rule = f"C(n+1) = {a_text} C(n) - {-b}"
            lead = (
                f"The number C(n), in thousands, of bacteria in a culture n hours after it is started satisfies "
                f"{rule}, with C(0) = {start}."
            )
        name = "A" if subject == "balance" else "C"
    else:
        rule = f"M(n+1) = {a_text} M(n) + {b}"
        lead = (
            f"The amount M(n), in milligrams, of a medicine in a patient's blood after n daily doses satisfies "
            f"{rule}, with M(0) = {start}."
        )
        name = "M"
    money = a > 1 and name == "A"
    term_text = f"to 2 decimal places{' (nearest cent)' if money else ''}"
    fixed_f, dev_f = Fraction(fixed), Fraction(dev)
    a_frac = f"({a.numerator}/{a.denominator})"
    return dict(
        text=lead,
        parts=[
            dict(
                **number_part(
                    f"Find the value of {name} for which {name}(n+1) = {name}(n).",
                    fixed_f,
                    0,
                    2,
                    [f"Set u = {a_text}u + ({b}) and solve: u = {b}/(1 - {a_text}) = {fixed}."],
                    [
                        (-fixed, "sign-of-result-reversed"),
                        (Fraction(b) / a, "divides-by-retained-fraction"),
                        (start, "mistakes-recorded-value-for-longterm"),
                        (b, "reports-added-amount"),
                    ],
                    sensible_range=None,
                )
            ),
            dict(
                **number_part(
                    f"Find {name}(0) minus your answer to (a).",
                    dev_f,
                    0,
                    1,
                    [f"{start} - {fixed} = {dev}."],
                    [
                        (-dev, "sign-of-result-reversed"),
                        (start, "gave-start-value-for-deviation"),
                        (dev_f * a, "deviation-multiplied-by-factor-once"),
                        (start + fixed, "added-instead-of-subtracted"),
                    ],
                    follow_through=f"{start} - part_a",
                )
            ),
            dict(
                **number_part(
                    f"Find {name}({n}), {term_text}.",
                    term,
                    2,
                    3,
                    [f"{name}(n) - {fixed} = ({dev})({a_text})^n, so {name}({n}) = {fixed} + ({dev})({a_text})^{n}."],
                    [
                        (fixed_f + dev_f * a * n, "linear-growth-instead-of-geometric"),
                        (Fraction(start) * a**n, "dropped-constant-term"),
                        (fixed_f, "reported-fixed-point-for-term"),
                        (fixed_f + dev_f * a ** (n - 1), "off-by-one-count"),
                    ],
                    follow_through=f"part_a + part_b*{a_frac}**{n}",
                )
            ),
        ],
    )
