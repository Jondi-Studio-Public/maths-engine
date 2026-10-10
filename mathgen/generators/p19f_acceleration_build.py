"""Acceleration as the second derivative of position with respect to time.

Skill: senior-recognise-acceleration-as-the-second-derivative-of-position
(ACMMM109, topic ``p19f_acceleration_second_derivative``, answer: number).

Given a particle's position function s(t) and a stated time t, compute the
particle's acceleration at that time as the second derivative of position
with respect to time -- always evaluated numerically in an explicit
motion/position context, never a bare symbolic second derivative
(that boundary is ``second-derivative``, and this generator never asks
"differentiate"/"find f''(x)").

Design:

- easy: s(t) = B*t^2 + C*t + D (a quadratic position -- constant
  acceleration = 2B). The learner must still recognise a constant is a
  valid *instantaneous* value at the stated t.
- medium: s(t) = A*t^3 + B*t^2 + C*t + D, A != 0 -- acceleration = 6At + 2B
  genuinely depends on t now.
- hard: s(t) = A*t^4 + B*t^3 + C*t^2 + D*t + E, A != 0 -- acceleration =
  12At^2 + 6Bt + 2C, two power-rule applications needed.
- exam: same quartic-position complexity as hard, but the stated time is
  reached via a one-step relation ("k seconds after the particle passes
  t = p seconds", i.e. evaluate at p + k) -- a new decision (compute the
  evaluation time itself before differentiating) not present at any lower
  tier.

Every coefficient and the stated time are integers, so every second
derivative -- a polynomial with integer coefficients evaluated at an
integer time -- is an exact integer answer by construction; no rounding
policy is needed anywhere in this generator.

**Independent check.** The hand closed-form second derivative (built
term-by-term from the stated coefficients) is checked with
`check_equivalent` against `sympy.diff(s_sym, t, 2).subs(t, point)` -- a
second, symbolic route to the same value, never the same formula restated.

**Misconceptions**, every one `check_distinct`ed against the true
acceleration on every draw (redrawing the coefficients, deterministically
under the seeded `rng`, on the rare collision rather than soft-dropping):
reports the velocity `s'(point)` instead of the acceleration; reports the
position `s(point)`; drops the extra power-rule coefficient the *second*
differentiation contributes (halves every coefficient of the true second
derivative, as if only one power-rule reduction had been applied); and a
plain sign error (negates the true acceleration).
"""

from __future__ import annotations

from sympy import diff, symbols

from ..core import (
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
)

t = symbols("t")

# (noun phrase for the particle/object, capitalised noun phrase, position symbol). Symbol kept fixed
# as "s"; only the scenario noun varies for text variety.
_SCENARIOS = (
    "a particle moving along a straight line",
    "a small robot moving along a straight track",
    "a lift (elevator) moving up and down a shaft",
    "a toy car moving along a straight rail",
)


def _time_word(n: int) -> str:
    return "second" if abs(n) == 1 else "seconds"


def _draw_easy(rng):
    while True:
        B = rng.choice([v for v in range(-9, 10) if v != 0])
        C = rng.randint(-9, 9)
        D = rng.randint(-9, 9)
        p = rng.randint(1, 6)
        accel = 2 * B
        velocity_val = 2 * B * p + C
        position_val = B * p * p + C * p + D
        dropped_coeff = B  # forgot the coefficient-2 the power rule contributes
        sign_flip = -accel
        candidates = [velocity_val, position_val, dropped_coeff, sign_flip]
        if all(c != accel for c in candidates):
            return dict(B=B, C=C, D=D, p=p, point=p, accel=accel, candidates=candidates)


def _draw_medium(rng):
    while True:
        A = rng.choice([v for v in range(-9, 10) if v != 0])
        B = rng.randint(-9, 9)
        C = rng.randint(-9, 9)
        D = rng.randint(-9, 9)
        p = rng.randint(1, 6)
        accel = 6 * A * p + 2 * B
        velocity_val = 3 * A * p * p + 2 * B * p + C
        position_val = A * p**3 + B * p * p + C * p + D
        dropped_coeff = 3 * A * p + B  # halved: forgot the extra factor of 2
        sign_flip = -accel
        candidates = [velocity_val, position_val, dropped_coeff, sign_flip]
        # Same realistic bound as the quartic tiers.
        if abs(accel) <= 50 and all(c != accel for c in candidates):
            return dict(A=A, B=B, C=C, D=D, p=p, point=p, accel=accel, candidates=candidates)


def _draw_quartic(rng, *, exam: bool):
    while True:
        A = rng.choice([v for v in range(-6, 7) if v != 0])
        B = rng.randint(-9, 9)
        C = rng.randint(-9, 9)
        D = rng.randint(-9, 9)
        E = rng.randint(-9, 9)
        p = rng.randint(1, 6)
        k = rng.randint(2, 5) if exam else 0
        point = p + k
        accel = 12 * A * point * point + 6 * B * point + 2 * C
        velocity_val = 4 * A * point**3 + 3 * B * point * point + 2 * C * point + D
        position_val = A * point**4 + B * point**3 + C * point * point + D * point + E
        dropped_coeff = 6 * A * point * point + 3 * B * point + C  # halved coefficients
        sign_flip = -accel
        candidates = [velocity_val, position_val, dropped_coeff, sign_flip]
        if abs(accel) <= 50 and all(c != accel for c in candidates):
            return dict(A=A, B=B, C=C, D=D, E=E, p=p, k=k, point=point, accel=accel, candidates=candidates)


@generator(
    "p19f_acceleration_second_derivative",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19f_acceleration_second_derivative(rng, tier="easy"):
    noun = rng.choice(_SCENARIOS)
    noun_cap = noun[0].upper() + noun[1:]

    if tier == "easy":
        draw = _draw_easy(rng)
        s_expr = join_terms([(draw["B"], "t^2"), (draw["C"], "t"), (draw["D"], "")])
        s_sym = draw["B"] * t**2 + draw["C"] * t + draw["D"]
    elif tier == "medium":
        draw = _draw_medium(rng)
        s_expr = join_terms([(draw["A"], "t^3"), (draw["B"], "t^2"), (draw["C"], "t"), (draw["D"], "")])
        s_sym = draw["A"] * t**3 + draw["B"] * t**2 + draw["C"] * t + draw["D"]
    elif tier == "hard":
        draw = _draw_quartic(rng, exam=False)
        s_expr = join_terms(
            [(draw["A"], "t^4"), (draw["B"], "t^3"), (draw["C"], "t^2"), (draw["D"], "t"), (draw["E"], "")]
        )
        s_sym = draw["A"] * t**4 + draw["B"] * t**3 + draw["C"] * t**2 + draw["D"] * t + draw["E"]
    elif tier == "exam":
        draw = _draw_quartic(rng, exam=True)
        s_expr = join_terms(
            [(draw["A"], "t^4"), (draw["B"], "t^3"), (draw["C"], "t^2"), (draw["D"], "t"), (draw["E"], "")]
        )
        s_sym = draw["A"] * t**4 + draw["B"] * t**3 + draw["C"] * t**2 + draw["D"] * t + draw["E"]
    else:
        raise ValueError(f"unknown tier {tier!r}")

    accel = draw["accel"]
    point = draw["point"]

    # Independent check: the hand closed-form second derivative against
    # sympy's own symbolic second derivative, evaluated at `point`.
    check_equivalent(
        diff(s_sym, t, 2).subs(t, point), accel, f"{tier}: hand second derivative disagrees with sympy at t={point}"
    )

    check_distinct(accel, *draw["candidates"], msg=f"{tier}: a misconception distractor equals the true acceleration")

    p = draw["p"]
    if tier == "exam":
        k = draw["k"]
        text = phrase(
            rng,
            (
                "The position of {noun} at time t {timeunit_generic} is given by "
                "s(t) = {expr}, in metres. Find its acceleration {k} {kunit} after it "
                "passes t = {p} {punit}, in metres per second squared.",
                "At time t {timeunit_generic}, {noun} has position s(t) = {expr}, in "
                "metres. Find the acceleration {k} {kunit} after it passes t = {p} "
                "{punit}, in metres per second squared.",
                "A motion sensor records the position of {noun} as s(t) = {expr}, in "
                "metres, where t is the time in {timeunit_generic}. Find its "
                "acceleration {k} {kunit} after it passes t = {p} {punit}, in metres "
                "per second squared.",
            ),
            noun=noun,
            noun_cap=noun_cap,
            expr=s_expr,
            p=p,
            punit=_time_word(p),
            k=k,
            kunit=_time_word(k),
            timeunit_generic="seconds",
        )
    else:
        text = phrase(
            rng,
            (
                "The position of {noun} at time t {timeunit_generic} is given by "
                "s(t) = {expr}, in metres. Find its acceleration at t = {p} {punit}, "
                "in metres per second squared.",
                "At time t {timeunit_generic}, {noun} has position s(t) = {expr}, in "
                "metres. Find the acceleration at t = {p} {punit}, in metres per "
                "second squared.",
                "A motion sensor records the position of {noun} as s(t) = {expr}, in "
                "metres, where t is the time in {timeunit_generic}. Determine its "
                "acceleration at t = {p} {punit}, in metres per second squared.",
            ),
            noun=noun,
            noun_cap=noun_cap,
            expr=s_expr,
            p=p,
            punit=_time_word(p),
            timeunit_generic="seconds",
        )

    answer_text = str(accel)
    distractors = pick_distractors(answer_text, [str(c) for c in draw["candidates"]])

    return dict(
        text=text,
        latex=text,
        answer=accel,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            "Differentiate s(t) once to get the velocity s'(t), then differentiate "
            "again to get the acceleration s''(t).",
            f"s''(t) evaluated at t = {point}: {answer_text} metres per second squared.",
        ],
        distractors=distractors,
    )
