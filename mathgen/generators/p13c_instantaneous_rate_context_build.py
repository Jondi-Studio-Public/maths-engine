"""Generates `p13c_instantaneous_rate_context` questions (ACMMM092): find an instantaneous rate of change in a
stated context.

Kernel: given a real-world quantity Q modelled as a function of time t, with a
stated polynomial formula and units, find the INSTANTANEOUS rate of change of
Q -- a derivative VALUE -- at a stated instant, with units carried through
into the answer prompt.

**The ``rate-of-change`` boundary** (``skills/linear-relations/rate-of-change.yaml``):
that skill computes an AVERAGE rate ``(y2-y1)/(x2-x1)`` from two stated points
or a table -- no calculus, no derivative, no formula to differentiate. Every
question here always states a polynomial FORMULA for Q(t) and requires
differentiating it; the average-rate computation over an interval never
appears here as anything but a named, checked-distinct WRONG answer (the
first misconception below).

**The ``senior-find-the-slope-of-a-tangent-and-the`` boundary**
(``skills/differentiation/senior-find-the-slope-of-a-tangent-and-the.yaml``):
that skill is an abstract curve ``y = f(x)`` with no context and no units, and
its medium tier is the mirror-image inverse ("has a tangent with slope k,
find x"). Every question here states a NAMED practical quantity (height,
volume, distance, population, ...) with UNITS in both the prompt and the
answer's implied unit -- never a bare ``y = f(x)``.

**The ACMMM103 boundary** (``senior-use-trigonometric-functions-and-their-
derivatives-to-solve``, ``mathgen/generators/p12c_trig_apply_implicit_diff_build.py``):
that skill's model is SINUSOIDAL (``H(t) = a*sin(b*t) + c``). Every model here
is POLYNOMIAL (quadratic or cubic) -- the two skills draw from disjoint
function families even though both compute a derivative value in context.

**The ``senior-calculate-total-change-by-integrating-instantaneous-or-
marginal`` boundary**: that skill integrates a given rate function to find
total change. This generator never integrates; every tier only
differentiates a stated Q(t) and evaluates or solves for t.

**Design decision: every tier's answer is an exact integer, never a rounded
decimal.** The hard tier's inverse problem ("at what time is Q changing at
rate k?") is built by choosing an integer ``t_target`` FIRST and computing
``k = Q'(t_target)`` from it, so the unique solution recovered by solving
``Q'(t) = k`` is always exactly ``t_target`` -- a genuinely different decision
from evaluating (solving a linear equation for t), with no rounding-boundary
risk at all. This is a deliberate simplification over ACMMM103's own hard
tier (which needed 2-d.p. rounding because its inverse involved arccos); a
polynomial model's derivative is always linear in the unknown, so there is no
structural need to introduce non-exact answers here.

**Difficulty ladder** (see also each tier's own docstring below):
- easy: Q(t) = a*t^2 + b*t + c (quadratic, one of 3 named contexts). Evaluate
  the derivative at a single stated t0. Single evaluation, no further
  decision.
- medium: Q(t) = a*t^3 + b*t^2 + c*t + d (cubic, one of 2 named contexts).
  Derivative is quadratic; more power-rule terms to track through the
  substitution -- a genuinely bigger computation, not yet a different kind of
  decision.
- hard: inverse problem on the SAME quadratic family as easy -- state a
  target rate k and ask at what time t the rate equals k (solve a linear
  equation for t). A genuinely different decision (solving, not evaluating).
- exam: instantaneous vs average, on the cubic family -- state Q(t), a
  stated interval [t1, t2], and ask for the instantaneous rate AT a specific
  instant (t1, t2, or their midpoint, varied) while the interval and its
  endpoints sit in the prompt as a deliberate pull toward the average-rate
  misconception (offered as a distractor, checked distinct).

**Why the exam tier uses the CUBIC family, not the quadratic one (a finding, not an assumption).** For a
quadratic Q(t) = a*t^2 + b*t + c, the average rate over [t1, t2] is IDENTICALLY EQUAL to the instantaneous
rate at the interval's own midpoint: (Q(t2)-Q(t1))/(t2-t1) = a*(t1+t2) + b = Q'((t1+t2)/2). That is an
algebraic identity, not a coincidence, so an exam tier built on the quadratic family with target = midpoint
would make the average-rate distractor collide with the correct answer on EVERY seed that draws that branch --
exactly the misconception distractor the exam tier exists to test, silently disappearing. The cubic family has
no such identity (the mean-value-theorem point for a cubic secant is not the midpoint in general), so the
average-rate distractor stays genuinely distinct there.
"""

from __future__ import annotations

from sympy import diff, solve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
)

t = symbols("t")

# Unit words that lose their trailing "s" when the stated count's magnitude
# is exactly 1 -- "1 second", never "1 seconds" (`_singular_count_plural`,
# `mathgen/text_quality.py`).
_UNIT_SINGULAR = {
    "metres": "metre",
    "litres": "litre",
    "seconds": "second",
    "minutes": "minute",
    "hours": "hour",
    "thousand cells": "thousand cell",
}


def _qty(value: int, unit: str) -> str:
    """ "<value> <unit>", singularising `unit` when `abs(value) == 1`."""
    word = _UNIT_SINGULAR.get(unit, unit) if abs(value) == 1 else unit
    return f"{value} {word}"


def _bare_rate_unit(unit: str, time_unit: str) -> str:
    """ "<unit>/<time-singular>" for a rate unit with no number immediately
    in front of it (e.g. "..., in litres/second."). The numerator stays
    plural (there is no stated count for `_singular_count_plural` to catch);
    the denominator is always singular by convention ("litres/second", never
    "litres/seconds")."""
    time_word = _UNIT_SINGULAR.get(time_unit, time_unit)
    return f"{unit}/{time_word}"


def _rate_unit(numerator_value: int, unit: str, time_unit: str) -> str:
    """ "<unit-or-singular>/<time-singular>" for a rate unit that DOES follow
    a stated numeral (e.g. "equals 1 litre/minute") -- the numerator
    singularises only when that count's own magnitude is exactly 1, matching
    ``p12c_trig_apply_implicit_diff_build.py``'s ``_rate_unit_for``. Use
    `_bare_rate_unit` instead when no number precedes the unit."""
    num_word = _UNIT_SINGULAR.get(unit, unit) if abs(numerator_value) == 1 else unit
    time_word = _UNIT_SINGULAR.get(time_unit, time_unit)
    return f"{num_word}/{time_word}"


# Quadratic contexts: (noun, symbol, unit, time_unit, a-range, b-range,
# c-range, t-range). `a` never includes 0 (a genuine quadratic, not a
# disguised linear model). Each context keeps its own plausible magnitude --
# shipped a 659 mm bolt next to a 134 cm adult man.
# `t_hard` is a wider integer window for the hard tier's `t_target` draw
# only (never used by easy) -- easy's own `t` range is kept tight for
# prompt realism (a thrown ball is only airborne a few seconds), but that
# same narrow range made the hard tier's ANSWER (the solved t itself, drawn
# from this same small integer set) concentrate at 21.5% on one value over
# 200 seeds (the pool-variety check), just under the 25% bar but thin (8 distinct
# answers). Hard reuses easy's (a, b, c) COEFFICIENT family per the
# own design, not its narrow t-window.
# **Second variety fix.** Tried widening all three windows in
# place first (keeping them stacked on the same low integers) -- reasoned
# through the arithmetic rather than shipping it, because it does not fix
# the OVERLAP: every value in the shared low range stays reachable from
# all three contexts' independent uniform draws, so small integers keep
# absorbing 3x the probability mass of values only one context can reach.
# Also tried shifting the tank's window up to (9, 22) alongside the ball
# and car -- that measured 22/40 distinct with 0 flags, but a second look
# at the RENDERED text caught a plausibility defect the pool-variety and
# distinctness checks cannot see: the tank's rate `2*a*t + b` (`a` in
# 1..2, `b` in -40..-15) only stays negative -- i.e. the tank is actually
# draining -- for `t < |b|/(2a)`, between 3.75 and 20 depending on the
# draw; shifting the window to (9, 22) put ~80% of tank draws (measured:
# 10/11 rendered seeds) PAST that point, asking for "the instantaneous
# rate of change of V" on a tank whose modelled volume is by then
# INCREASING (e.g. one rendered seed: V(t) = 2t^2 - 15t + 284, asked at
# t=15, where V(15) = 509 L is already above the starting 284 L -- a
# "draining tank" that has filled back up). That is exactly the kind of
# warns about, so the fix here is DROPPING THE TANK from the hard tier's
# eligible contexts entirely (restricting
# which contexts are hard-eligible) rather than disclosing and shipping
# the mismatch. The tank still exists for easy/medium/exam (unaffected --
# this generator's `_draw_quad` is untouched; only the HARD branch below
# filters by `hard_eligible`).
# The two hard-eligible contexts are both monotone across their whole
# hard-tier window, so there is no equivalent sign-flip risk left to
# check: the ball's rate `2*a*t + b` is negative everywhere in `t_hard`
# for every allowed `a`, `b` (`b` <= 20 and `a` <= -2` means the rate
# is already negative from `t=1`), and the car's rate `2*a*t + b` is
# positive everywhere for every allowed `a > 0`, `b > 0` -- an
# accelerating car's speed never reverses sign in this model.
# - ball (1-9s, `hard_eligible=True`): a ball's own flight time already
#   reaches up to about 10 s at this context's largest launch speed
#   (`b=20`, `a=-2`: roots of -2t^2+20t+c=0 land near t=10), so 1-9 stays
#   inside the model's own plausible ceiling without stretching it.
# - tank (`hard_eligible=False`): excluded from the hard tier per the
#   plausibility finding above; its `t` range for easy/medium/exam is
#   untouched.
# - car (10-34s, `hard_eligible=True`): shifted up and widened past the
#   ball's band (no overlap needed now there are only two hard-eligible
#   contexts) -- "a car accelerating from rest" has no comparable hard
#   ceiling the way a ball's flight time does, so an on-ramp merge or a
#   full 0-100 km/h run comfortably spans this window. This DOES push the
#   stated rate substantially higher than before (worst case `k` was
#   ~136 m/s at the old `t_hard` ceiling of 16 with `a=4, b=8`; the new
#   ceiling of 34 reaches `k=280` m/s, roughly double) -- an abstraction
#   stretch on an already-abstract "accelerating from rest" model, but
#   the SIGN and DIRECTION stay correct throughout (still speeding up,
#   never slowing or reversing), unlike the tank's problem above.
_QUAD_CONTEXTS = (
    dict(
        noun="the height above the ground of a ball thrown into the air",
        sym="h",
        unit="metres",
        time_unit="seconds",
        a=(-5, -2),
        b=(10, 20),
        c=(0, 2),
        t=(1, 3),
        t_hard=(1, 9),
        hard_eligible=True,
    ),
    dict(
        noun="the volume of water remaining in a draining tank",
        sym="V",
        unit="litres",
        time_unit="minutes",
        a=(1, 2),
        b=(-40, -15),
        c=(200, 400),
        t=(1, 5),
        t_hard=None,
        hard_eligible=False,
    ),
    dict(
        noun="the distance travelled by a car accelerating from rest",
        sym="d",
        unit="metres",
        time_unit="seconds",
        a=(1, 4),
        b=(2, 8),
        c=(0, 0),
        t=(2, 8),
        t_hard=(10, 34),
        hard_eligible=True,
    ),
)

#: Hard tier only ever draws from the hard-eligible subset above (see the
#: docstring block on `_QUAD_CONTEXTS` for why the tank is excluded).
_HARD_QUAD_CONTEXTS = tuple(c for c in _QUAD_CONTEXTS if c["hard_eligible"])

# Cubic contexts: same shape plus a `d`-range (the constant term).
_CUBIC_CONTEXTS = (
    dict(
        noun="the volume of air inside an inflating weather balloon",
        sym="V",
        unit="litres",
        time_unit="seconds",
        a=(1, 2),
        b=(-4, 4),
        c=(3, 10),
        d=(20, 60),
        t=(1, 6),
    ),
    dict(
        noun="the size of a growing bacteria colony",
        sym="P",
        unit="thousand cells",
        time_unit="hours",
        a=(1, 3),
        b=(-3, 3),
        c=(2, 8),
        d=(10, 40),
        t=(1, 6),
    ),
)


def _draw_quad(rng):
    ctx = rng.choice(_QUAD_CONTEXTS)
    a = rng.randint(*ctx["a"])
    b = rng.randint(*ctx["b"])
    c = rng.randint(*ctx["c"])
    return ctx, a, b, c


def _draw_quad_hard(rng):
    """Like `_draw_quad`, but draws only from the hard-eligible subset (see
    the docstring on `_QUAD_CONTEXTS` -- the tank is excluded because its
    modelled rate changes sign inside a wide `t_hard` window, which would
    ask "the instantaneous rate of change" of a "draining" tank that is, by
    the drawn time, actually refilling)."""
    ctx = rng.choice(_HARD_QUAD_CONTEXTS)
    a = rng.randint(*ctx["a"])
    b = rng.randint(*ctx["b"])
    c = rng.randint(*ctx["c"])
    return ctx, a, b, c


def _draw_cubic(rng):
    ctx = rng.choice(_CUBIC_CONTEXTS)
    a = rng.randint(*ctx["a"])
    b = rng.randint(*ctx["b"])
    c = rng.randint(*ctx["c"])
    d = rng.randint(*ctx["d"])
    return ctx, a, b, c, d


def _add_distinct(pool: list, answer: int, *candidates: int) -> None:
    """Append each candidate only if it is genuinely a different VALUE from
    `answer`. Some collisions here are expected structural coincidences (an
    easy-tier ``dropped_coeff`` value always equals ``avg_rate`` -- see the
    easy-tier docstring), not construction bugs, so silently dropping a
    colliding candidate is correct here, matching
    ``p12c_trig_apply_implicit_diff_build.py``'s own ``_add_distinct``."""
    for candidate in candidates:
        if candidate != answer and candidate not in pool:
            pool.append(candidate)


def _sentence(noun: str) -> str:
    return f"{noun[0].upper()}{noun[1:]}"


@generator(
    "p13c_instantaneous_rate_context",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13c_instantaneous_rate_context(rng, tier="easy"):
    """See module docstring for the full ladder and the exam/quadratic
    boundary finding.

    easy:   Q(t) = a*t^2 + b*t + c. Find dQ/dt at a stated t0.
    medium: Q(t) = a*t^3 + b*t^2 + c*t + d. Find dQ/dt at a stated t0.
    hard:   inverse problem on the quadratic family -- find t such that
            dQ/dt = k, for a k constructed so the unique solution is exact.
    exam:   cubic family, stated interval [t1, t2]; find the instantaneous
            rate AT t1, t2 or their midpoint (varied), with the interval and
            its average rate present as a deliberate distractor pull.
    """
    if tier == "easy":
        ctx, a, b, c = _draw_quad(rng)
        sym, unit, time_unit = ctx["sym"], ctx["unit"], ctx["time_unit"]
        t0 = rng.randint(*ctx["t"])

        Q = a * t**2 + b * t + c
        dQ = diff(Q, t)
        rate = 2 * a * t0 + b
        check_equivalent(dQ.subs(t, t0), rate, f"easy dQ/dt mismatch: a={a}, b={b}, t0={t0}")

        # Misconception 1: average rate over [0, t0] instead of the
        # instantaneous rate -- (Q(t0)-Q(0))/t0 = a*t0 + b.
        avg_rate = a * t0 + b
        # Misconception 2: evaluates Q(t0) (the VALUE) instead of Q'(t0)
        # (its RATE).
        value_instead = a * t0**2 + b * t0 + c
        # Misconception 3: drops the power-rule coefficient -- differentiates
        # a*t^2 as a*t (coefficient left un-doubled), giving a*t0 + b at t0.
        # NOTE: for a quadratic this is algebraically IDENTICAL to
        # `avg_rate` above (the mean-value-theorem point of a quadratic on
        # [0, t0] is always its own midpoint, t0/2, and Q'(t0/2) = a*t0 + b),
        # so the two candidates always collide here; `_add_distinct` keeps only one.
        dropped_coeff = a * t0 + b
        # Misconception 4: sign error on the derivative.
        sign_flip = -rate

        cands = []
        _add_distinct(cands, rate, avg_rate, value_instead, dropped_coeff, sign_flip)
        answer_value = rate

        text = (
            f"{_sentence(ctx['noun'])} is modelled by {sym}(t) = {join_terms([(a, 't^2'), (b, 't'), (c, '')])}, "
            f"where t is the time in {time_unit} since t = 0 ({sym} in {unit}). Find the instantaneous rate "
            f"of change of {sym} with respect to t at t = {_qty(t0, time_unit)}, in {_bare_rate_unit(unit, time_unit)}."
        )

    elif tier == "medium":
        ctx, a, b, c, d = _draw_cubic(rng)
        sym, unit, time_unit = ctx["sym"], ctx["unit"], ctx["time_unit"]
        t0 = rng.randint(*ctx["t"])

        Q = a * t**3 + b * t**2 + c * t + d
        dQ = diff(Q, t)
        rate = 3 * a * t0**2 + 2 * b * t0 + c
        check_equivalent(dQ.subs(t, t0), rate, f"medium dQ/dt mismatch: a={a}, b={b}, c={c}, t0={t0}")

        # Misconception 1: average rate over [0, t0] -- (Q(t0)-Q(0))/t0.
        avg_rate = a * t0**2 + b * t0 + c
        # Misconception 2: evaluates Q(t0) instead of Q'(t0).
        value_instead = a * t0**3 + b * t0**2 + c * t0 + d
        # Misconception 3: drops the power-rule coefficient on the cubic
        # term -- differentiates a*t^3 as a*t^2 (un-tripled), keeping the
        # t^2 term's own doubling correct.
        dropped_coeff = a * t0**2 + 2 * b * t0 + c
        # Misconception 4: sign error on the derivative.
        sign_flip = -rate

        cands = []
        _add_distinct(cands, rate, avg_rate, value_instead, dropped_coeff, sign_flip)
        answer_value = rate

        text = (
            f"{_sentence(ctx['noun'])} is modelled by {sym}(t) = "
            f"{join_terms([(a, 't^3'), (b, 't^2'), (c, 't'), (d, '')])}, where t is the time in {time_unit} "
            f"since t = 0 ({sym} in {unit}). Find the instantaneous rate of change of {sym} with respect to "
            f"t at t = {_qty(t0, time_unit)}, in {_bare_rate_unit(unit, time_unit)}."
        )

    elif tier == "hard":
        ctx, a, b, c = _draw_quad_hard(rng)
        sym, unit, time_unit = ctx["sym"], ctx["unit"], ctx["time_unit"]
        t_target = rng.randint(*ctx["t_hard"])

        Q = a * t**2 + b * t + c
        dQ = diff(Q, t)  # 2*a*t + b
        k = 2 * a * t_target + b
        # Independent check: solve dQ/dt = k for t via sympy.solve, a route
        # independent of however `t_target`/`k` were constructed above.
        roots = solve(dQ - k, t)
        check(len(roots) == 1, f"hard: expected exactly one root, got {roots}")
        check_equivalent(roots[0], t_target, f"hard solved-t mismatch: a={a}, b={b}, k={k}")

        # Misconception 1 (dropped-power-rule-coefficient): solves
        # a*t + b = k (un-doubled derivative) for t -- since k = 2*a*t_target
        # + b, this always gives t = 2*t_target exactly.
        dropped_coeff_t = 2 * t_target
        # Misconception 2 (solves-value-equation-not-rate-equation, per the
        # ACMMM103 naming precedent): forgets to divide by 2*a when
        # isolating t, i.e. computes k - b directly -- since k - b =
        # 2*a*t_target, this is 2*a*t_target.
        forgot_divide_t = 2 * a * t_target
        # Misconception 3 (sign-error-on-solved-t): reports the negative of
        # the correct time.
        sign_error_t = -t_target

        cands = []
        _add_distinct(cands, t_target, dropped_coeff_t, forgot_divide_t, sign_error_t)

        rate_phrase = f"{k} {_rate_unit(k, unit, time_unit)}"
        text = (
            f"{_sentence(ctx['noun'])} is modelled by {sym}(t) = {join_terms([(a, 't^2'), (b, 't'), (c, '')])}, "
            f"where t is the time in {time_unit} since t = 0 ({sym} in {unit}). Find the time t (t >= 0, in "
            f"{time_unit}) at which the instantaneous rate of change of {sym} equals {rate_phrase}."
        )
        answer_value = t_target

    else:  # exam
        ctx, a, b, c, d = _draw_cubic(rng)
        sym, unit, time_unit = ctx["sym"], ctx["unit"], ctx["time_unit"]
        lo, hi = ctx["t"]
        t1 = rng.randint(lo, hi - 2)
        max_k = (hi - t1) // 2
        gap_k = rng.randint(1, max_k)
        t2 = t1 + 2 * gap_k  # even gap -> (t1+t2) always even -> exact midpoint
        target_kind = rng.choice(("t1", "t2", "mid"))
        target = {"t1": t1, "t2": t2, "mid": (t1 + t2) // 2}[target_kind]

        Q = a * t**3 + b * t**2 + c * t + d
        dQ = diff(Q, t)
        rate = 3 * a * target**2 + 2 * b * target + c
        check_equivalent(dQ.subs(t, target), rate, f"exam dQ/dt mismatch: target={target}")

        # Independent check: the secant slope (Q(t2)-Q(t1))/(t2-t1), computed
        # via sympy substitution, a route independent of the closed-form
        # `avg_rate` expression below.
        Q1, Q2 = Q.subs(t, t1), Q.subs(t, t2)
        avg_rate_oracle = (Q2 - Q1) / (t2 - t1)
        avg_rate = a * (t1**2 + t1 * t2 + t2**2) + b * (t1 + t2) + c
        check_equivalent(avg_rate_oracle, avg_rate, f"exam avg-rate mismatch: t1={t1}, t2={t2}")

        # Misconception 1: average rate over the stated interval instead of
        # the instantaneous rate at the stated instant -- the reason this
        # tier exists. NOT guaranteed distinct by an algebraic identity for
        # a cubic (unlike the quadratic family -- see module docstring), so
        # this is verified empirically over many seeds, not assumed.
        # Misconception 2: evaluates Q(target) instead of Q'(target).
        value_instead = a * target**3 + b * target**2 + c * target + d
        # Misconception 3: sign error on the derivative.
        sign_flip = -rate

        cands = []
        _add_distinct(cands, rate, avg_rate, value_instead, sign_flip)

        text = (
            f"{_sentence(ctx['noun'])} is modelled by {sym}(t) = "
            f"{join_terms([(a, 't^3'), (b, 't^2'), (c, 't'), (d, '')])}, where t is the time in {time_unit} "
            f"since t = 0 ({sym} in {unit}). Over the interval from t = {t1} to t = {_qty(t2, time_unit)}, find "
            f"the INSTANTANEOUS rate of change of {sym} at t = {_qty(target, time_unit)}, in "
            f"{_bare_rate_unit(unit, time_unit)} (not the average rate over the interval)."
        )
        answer_value = rate

    # `cands` was already filtered by `_add_distinct` above, but re-assert
    # the invariant unconditionally here too -- the discipline this plan's
    # convention requires ("check every candidate, including the ones you
    # are sure about") -- rather than trust the filter silently.
    check_distinct(answer_value, *cands, msg="a distractor equals the answer")

    answer_text = str(answer_value)
    distractors = pick_distractors(answer_text, [str(v) for v in cands])

    return dict(
        text=text,
        latex=text,
        answer=int(answer_value),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )
