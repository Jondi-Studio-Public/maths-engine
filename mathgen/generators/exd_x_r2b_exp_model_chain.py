"""Generates `exd_x_r2b_exp_model_chain` questions: a multi-part exam chain on an exponential growth or decay model.

Skill: senior-use-exponential-functions-and-their-derivatives-to-solve (exam tier).

A quantity is modelled by `N(t) = A e^(kt)` with `A` stated and `k` not. The stem gives the rate at which N is
changing at t = 0, `N'(0) = kA`, and the parts run as a senior exam sets them:

(a) find k from the initial rate (negative for a decay)           number, 2 places, 2 marks
(b) the rate of change at the moment N reaches a stated value     number, 2 places, 1 mark,  k N
(c) the time at which N reaches that value                        number, 2 places, 2 marks, ln(M / A) / k
(d) the average rate of change of N from t = 0 to that time       number, 2 places, 2 marks, (M - A) / T

## Construction: choose k and the target first

`A` is a multiple of 100, `k` has two decimal places and either sign, and the target `M` is a whole-number multiple
or fraction of `A`, so `kA` and `kM` are exact two-place decimals. The time `T = ln(M / A) / k` must be sensible
(between 3 and 60) and (c) and (d) are drawn only clear of a rounding tie; the route through the rounded `T` must
give the same last digit for (d) as the key.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational, log

from ..core import check, generator, phrase, round_half_up
from .exd_x_r2b_support import Redraw, clear_of_boundary, draw, number_part, real

_A = (100, 200, 400, 500)
_K = tuple(Fraction(n, 100) for n in range(3, 16))
_GROW = (Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(5))
_DECAY = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 5), Fraction(2, 5))
_GROWTH_CONTEXTS = (
    ("The number of bacteria N in a laboratory culture", "bacteria", "hour", "hours"),
    ("The number of followers N of a community page", "followers", "week", "weeks"),
    ("The value N, in dollars, of an investment", "dollars", "year", "years"),
)
_DECAY_CONTEXTS = (
    ("The mass N, in milligrams, of a medicine in a patient's bloodstream", "milligrams", "hour", "hours"),
    ("The mass N, in grams, of a radioactive sample", "grams", "day", "days"),
    ("The number N of litres of water left in a leaking tank", "litres", "minute", "minutes"),
)
_PATTERNS = (
    "{subject}, t {plural} after it is first measured, is modelled by N(t) = {A}e^(kt), where k is a constant. "
    "At t = 0, N is {verb} at a rate of {r0} {unit} per {singular}.",
    "{subject} is modelled by N(t) = {A}e^(kt), where t is the time in {plural} after it is first measured and "
    "k is a constant. When t = 0, N is {verb} at {r0} {unit} per {singular}.",
)


def _dec(value: Fraction) -> str:
    for places in range(0, 5):
        scaled = value * 10**places
        if scaled.denominator == 1:
            return f"{scaled.numerator / 10**places:.{places}f}"
    raise Redraw("not a terminating decimal")


def _build(rng):
    A = rng.choice(_A)
    growing = rng.random() < 0.5
    k = rng.choice(_K) * (1 if growing else -1)
    ratio = rng.choice(_GROW if growing else _DECAY)
    M = A * ratio
    if M.denominator != 1:
        raise Redraw("target not a whole number")
    M = int(M)
    r0, rM = k * A, k * M
    ratio_s, k_s = Rational(M, A), Rational(k.numerator, k.denominator)
    T = real(log(ratio_s) / k_s)
    if not 3 <= T <= 60:
        raise Redraw("time not sensible")
    avg = Fraction(M - A) / T
    if not (clear_of_boundary(T, 2) and clear_of_boundary(avg, 2)):
        raise Redraw("a key too close to a rounding tie")
    Tr = round_half_up(T, 2)
    if round_half_up(Fraction(M - A) / Tr, 2) != round_half_up(avg, 2):
        raise Redraw("rounding T first changes the last digit of the average rate")
    # An independent route: N(T) = A e^(kT) must return the target.
    check(abs(A * math.exp(float(k) * float(T)) - M) < 1e-6, "N(T) is not the target")

    subject, unit, singular, plural = rng.choice(_GROWTH_CONTEXTS if growing else _DECAY_CONTEXTS)
    stem = phrase(
        rng,
        _PATTERNS,
        subject=subject,
        A=A,
        plural=plural,
        singular=singular,
        unit=unit,
        verb="increasing" if growing else "decreasing",
        r0=_dec(abs(r0)),
    )

    # (a) k = r0 / A, negative for a decay
    a_wrong = [
        (Fraction(A) / r0, "inverted-ratio"),
        (r0, "forgets-the-multiplier"),  # N'(0) read as k itself
        (-k, "sign-reversed-result"),  # decreasing read as a positive rate constant, or the reverse
    ]
    part_a = number_part(
        "Find the value of k.",
        k,
        2,
        2,
        ["N'(t) = kAe^(kt), so N'(0) = kA.", f"k = {_dec(r0)} / {A} = {k}."],
        a_wrong,
    )

    # (b) N'(t) = k N at the moment N = M
    b_wrong = [
        (Fraction(M), "value-instead-of-rate"),
        (r0, "used-initial-rate"),  # the rate at t = 0
        (Fraction(M) / k, "inverted-ratio"),
        (k, "exponential-derivative-factor-dropped"),
    ]
    part_b = number_part(
        f"Find the rate of change of N, in {unit} per {singular}, at the moment N = {M}.",
        rM,
        2,
        1,
        [f"N'(t) = kN, so N' = {k} × {M} = {rM}."],
        b_wrong,
        follow_through=f"part_a*{M}",
    )

    # (c) the time N reaches M
    c_wrong = [
        (real(log(M) / k_s), "initial-value-ignored"),  # A dropped from M / A
        (Fraction(M, A) / k, "forgot-to-take-logarithm"),
        (real(log(ratio_s, 10) / k_s), "log-base-confused"),
        (real(log(ratio_s) * k_s), "multiplied-instead-of-divided"),
    ]
    part_c = number_part(
        f"Find the time at which N = {M}, in {plural}.",
        T,
        2,
        2,
        [f"{A}e^({k}t) = {M}, so e^({k}t) = {_dec(Fraction(M, A))}.", f"t = ln({_dec(Fraction(M, A))}) / {k} = {Tr}."],
        c_wrong,
        follow_through=f"log(Rational({ratio_s.p}, {ratio_s.q}))/part_a",
        sensible_range=(0, None),
    )

    # (d) the average rate of change over [0, T]
    d_wrong = [
        (Fraction(M) / T, "initial-value-ignored"),  # the final value over the time, not the change
        (rM, "instantaneous-for-average"),
        (Fraction(M - A) * T, "multiplied-instead-of-divided"),
    ]
    part_d = number_part(
        f"Find the average rate of change of N, in {unit} per {singular}, from t = 0 to this time.",
        avg,
        2,
        2,
        [f"Average rate = (N(T) - N(0)) / T = ({M} - {A}) / {Tr} = {round_half_up(avg, 2)}."],
        d_wrong,
        follow_through=f"({M} - {A})/part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_exp_model_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_exp_model_chain(rng, tier: str = "exam"):
    """An exponential model with an unknown rate constant: k, the rate at a stated size, the time, an average rate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
