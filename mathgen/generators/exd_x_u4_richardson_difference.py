"""Exam-tier source for `centered-difference-polynomial`: two centred differences and Richardson extrapolation.

The practice tiers ask for one centred difference of a stated cubic at a stated step. The exam question models the
position of a cyclist with a cubic in time and asks for a velocity estimate that is better than either of two centred
differences. The learner must

1. compute the centred difference with step `h`,
2. compute it again with step `h/2`, and
3. combine the two so the error term proportional to `h^2` cancels, `(4 D(h/2) - D(h)) / 3`.

For a cubic the centred difference is exactly `f'(x) + a h^2` (the cubic coefficient is `a`), so the extrapolated value
is exactly the derivative. That identity is the independent route: the key for (c) is checked against the derivative
computed by SymPy, and the key for (a) and (b) against the same identity.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational, Symbol, diff

from mathgen.core import check, check_equivalent, generator, join_terms, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_T = Symbol("t")
_SETTINGS = [
    "The distance s metres of a cyclist from the start line, t seconds after the start, is modelled by s(t) = {poly}.",
    "The height s metres of a drone above the ground, t seconds after take-off, is modelled by s(t) = {poly}.",
    "The distance s metres of a tram from its depot, t seconds after it leaves, is modelled by s(t) = {poly}.",
]


def _value(coefs: tuple[int, int, int, int], t: Fraction) -> Fraction:
    a, b, c, d = coefs
    return a * t**3 + b * t**2 + c * t + d


@generator("exd_x_u4_richardson_difference", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_richardson_difference(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: centred differences at h and h/2, then the extrapolated velocity."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        a = rng.randint(1, 3)
        b, c, d = rng.randint(-5, 5), rng.randint(-6, 6), rng.randint(-5, 5)
        t0 = rng.randint(1, 4)
        k = rng.randint(2, 5)
        coefs = (a, b, c, d)
        h = Fraction(1, k)

        def centred(step: Fraction) -> Fraction:
            return (_value(coefs, t0 + step) - _value(coefs, t0 - step)) / (2 * step)

        d1, d2 = centred(h), centred(h / 2)
        richardson = (4 * d2 - d1) / 3
        exact = Fraction(3 * a * t0**2 + 2 * b * t0 + c)
        keys = (d1, d2, richardson)
        if {t0, k, a, b, c, d} & set(keys) or len(set(keys)) < 3:
            continue
        if not survives_nsimplify(*keys):
            continue
        break
    else:
        check(False, "no usable cubic in 1000 tries")

    # Independent route: for a cubic the centred difference is f'(t0) + a h^2 exactly, and the extrapolation is f'(t0).
    s_expr = a * _T**3 + b * _T**2 + c * _T + d
    derivative = diff(s_expr, _T).subs(_T, t0)
    check(d1 == exact + a * h**2 and d2 == exact + a * (h / 2) ** 2, "centred differences disagree with f' + a h^2")
    check_equivalent(derivative, Rational(richardson.numerator, richardson.denominator), "extrapolation is not f'(t0)")

    poly = join_terms([(a, "t³"), (b, "t²"), (c, "t"), (d, "")])
    stem = (
        f"{rng.choice(_SETTINGS).format(poly=poly)} The velocity at t = {t0} is estimated by the centred difference "
        f"D(h) = [s({t0} + h) − s({t0} − h)] / (2h). The error in D(h) is proportional to h², so two estimates can "
        f"be combined to cancel it. Take h = 1/{k} s. Give exact values as fractions or integers."
    )
    forward = (_value(coefs, t0 + h) - _value(coefs, t0)) / h
    forward_half = (_value(coefs, t0 + h / 2) - _value(coefs, t0)) / (h / 2)
    a_candidates = [
        (forward, "difference-forward-not-centred"),
        (2 * d1, "difference-quotient-wrong-width"),
        (exact, "difference-exact-not-estimate"),
    ]
    b_candidates = [
        (d1, None),
        (forward_half, "difference-forward-not-centred"),
        (2 * d2, "difference-quotient-wrong-width"),
        (exact, "difference-exact-not-estimate"),
    ]
    c_candidates = [
        ((d1 + d2) / 2, "richardson-plain-average"),
        (2 * d2 - d1, "richardson-wrong-weights"),
        ((4 * d1 - d2) / 3, "richardson-wrong-weights"),
        (d2, None),
    ]
    d1t, d2t, rt = (rational_text(v) for v in keys)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find D(h) with h = 1/{k}.",
                answer_kind="fraction",
                answer=fraction_payload(d1)[0],
                answer_text=d1t,
                marks=2,
                working=[
                    f"s({t0} + 1/{k}) = {rational_text(_value(coefs, t0 + h))} and "
                    f"s({t0} − 1/{k}) = {rational_text(_value(coefs, t0 - h))}.",
                    f"D(h) = [difference] / (2/{k}) = {d1t}.",
                ],
                distractors=fraction_distractors(d1, a_candidates),
            ),
            dict(
                prompt=f"Find D(h/2) with h/2 = 1/{2 * k}.",
                answer_kind="fraction",
                answer=fraction_payload(d2)[0],
                answer_text=d2t,
                marks=2,
                working=[f"D(h/2) = [s({t0} + 1/{2 * k}) − s({t0} − 1/{2 * k})] / (2/{2 * k}) = {d2t}."],
                distractors=fraction_distractors(d2, b_candidates),
            ),
            dict(
                prompt="Using your two estimates, combine them to cancel the error term proportional to h² and "
                "find the improved velocity estimate.",
                answer_kind="fraction",
                answer=fraction_payload(richardson)[0],
                answer_text=rt,
                marks=2,
                follow_through="(4*part_b - part_a)/3",
                working=[
                    "If D(h) = v + Ch² then D(h/2) = v + Ch²/4, so v = (4D(h/2) − D(h))/3.",
                    f"v = (4({d2t}) − ({d1t}))/3 = {rt} m/s.",
                ],
                distractors=fraction_distractors(richardson, c_candidates),
            ),
        ],
    )
