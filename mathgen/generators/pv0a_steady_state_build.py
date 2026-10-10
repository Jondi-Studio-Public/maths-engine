"""Generates steady-state questions for first-order linear recurrence relations (ACMGM077).

The skill's name: "Use first-order linear recurrence relations to model and analyse
(numerically or graphically only) practical problems; for example, investigating the growth
of a trout population in a lake ... or the amount owing on a reducing balance loan after each
payment is made." The scope drops both worked examples and keeps only: given
a first-order linear recurrence u(n+1) = a*u(n) + b with 0 < a < 1 modelling a stated
NON-FINANCIAL quantity, find by iterating the long-term (steady-state) value the sequence
settles to, to a stated precision. This module's kernel answers that directly: every context
below is a fixed fraction lost each period plus a fixed amount added (never a loan, never
interest), and every tier's answer is the fixed point L = b/(1 - a), never the balance after n
steps -- that shape belongs to the shipping ``senior-use-technology-and-a-recurrence-relation-
to-model`` (a loan) and is explicitly excluded.

Tiers escalate by which DECISION the learner must make, not by bigger numbers:
- easy: the recurrence is given with its numbers; find L directly from the closed form.
- medium: the model is described in words (a percentage lost, then an amount added); the
  learner must first form a and b before finding L.
- hard: the model's u(n) is stated at ONE point in the cycle (immediately after the addition,
  or immediately before it); the learner must find the long-term value at the OTHER point,
  a*L or L, not the point the recurrence itself was given at.
- exam: neither a nor b nor a formula is stated; three consecutive recorded values u(0), u(1),
  u(2) are given, and the learner must recover a and b from their consecutive differences
  before finding L. The kernel's "given a first-order linear recurrence" is honoured because
  the FORM u(n+1) = a*u(n) + b is stated explicitly; only its constants are unknown.

Every value is built by drawing the long-term value L over a wide range first, rounding the
added amount b to a clean whole number, and recomputing L exactly from that clean b -- so the
displayed b, u(0), u(1) are always whole numbers and only the FINAL answer L may show the
context's stated decimal place.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from mathgen.core import (
    check,
    check_pairwise_distinct,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_ITERATE_PERIODS = 400
_ITERATE_TOL = Fraction(1, 10**9)


@dataclass(frozen=True)
class _Ctx:
    subject: str  # "the amount of a drug in a patient's bloodstream"
    subject_cap: str  # sentence-initial capitalised form
    short: str  # "the amount of drug"
    short_point: str  # the full "u(n) is ___" noun phrase -- for a measurement context
    # ("the amount of drug in mg") this folds {short} and {unit} together naturally; for a COUNT
    # context whose short noun already IS the unit ("the fish population"/"fish", "the membership"/
    # "members") this is "the number of fish"/"the number of members" instead, so the unit is never
    # stated twice.
    unit: str  # "mg"
    time: str  # "day"
    loss_clause_tpl: str  # "{p}% of the drug in the bloodstream is eliminated"
    add_clause_tpl: str  # "a {b} mg dose is then taken"
    event: str  # "dose" / "restocking" / "inflow" / "top-up" / "intake of new members"
    d: int  # stated precision: 0 = nearest whole number, 1 = 1 decimal place
    l_lo: int
    l_hi: int


_CONTEXTS = (
    _Ctx(
        subject="the amount of a drug in a patient's bloodstream",
        subject_cap="The amount of a drug in a patient's bloodstream",
        short="the amount of drug",
        short_point="the amount of drug in mg",
        unit="mg",
        time="day",
        loss_clause_tpl="{p}% of the drug in the bloodstream is eliminated",
        add_clause_tpl="a {b} mg dose is then taken",
        event="dose",
        d=1,
        l_lo=20,
        l_hi=150,
    ),
    _Ctx(
        subject="the number of fish in a lake",
        subject_cap="The number of fish in a lake",
        short="the fish population",
        short_point="the number of fish",
        unit="fish",
        time="year",
        loss_clause_tpl="{p}% of the fish in the lake die or are caught",
        add_clause_tpl="{b} fish are then restocked",
        event="restocking",
        d=0,
        l_lo=60,
        l_hi=400,
    ),
    _Ctx(
        subject="the mass of a pollutant in a lake",
        subject_cap="The mass of a pollutant in a lake",
        short="the mass of pollutant",
        short_point="the mass of pollutant in kg",
        unit="kg",
        time="week",
        loss_clause_tpl="{p}% of the pollutant in the lake breaks down",
        add_clause_tpl="{b} kg of pollutant then flows in from a factory upstream",
        event="inflow",
        d=1,
        l_lo=30,
        l_hi=300,
    ),
    _Ctx(
        subject="the mass of chlorine dissolved in a swimming pool",
        subject_cap="The mass of chlorine dissolved in a swimming pool",
        short="the mass of chlorine",
        short_point="the mass of chlorine in g",
        unit="g",
        time="day",
        loss_clause_tpl="{p}% of the chlorine in the pool breaks down in sunlight",
        add_clause_tpl="the pool is then topped up with {b} g of chlorine",
        event="top-up",
        d=1,
        l_lo=5,
        l_hi=40,
    ),
    _Ctx(
        subject="the membership of a social club",
        subject_cap="The membership of a social club",
        short="the membership",
        short_point="the number of members",
        unit="members",
        time="month",
        loss_clause_tpl="{p}% of the club's members leave",
        add_clause_tpl="{b} new members then join",
        event="intake of new members",
        d=0,
        l_lo=50,
        l_hi=400,
    ),
)


def _prec_phrase(d: int) -> str:
    return "to the nearest whole number" if d == 0 else "to 1 decimal place"


def _fmt_a(a: Fraction) -> str:
    return stated_precision_string(a, 1)


def _fmt(v, d: int) -> str:
    return stated_precision_string(round_half_up(Fraction(v), d), d)


def _fmt_recorded(v) -> str:
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    return stated_precision_string(v, 1)


def _iterate(a: Fraction, b: Fraction, start: Fraction, periods: int) -> Fraction:
    u = start
    for _ in range(periods):
        u = a * u + b
    return u


def _verify_steady_state(a: Fraction, b: Fraction, L: Fraction) -> None:
    """Independent route: iterate the recurrence from an arbitrary starting value for many
    periods and confirm it converges to L -- a genuinely different computation from the closed
    form b/(1 - a) construction used to derive L, since 0 < a < 1 guarantees geometric
    convergence of a^n toward 0."""
    iterated = _iterate(a, b, Fraction(0), _ITERATE_PERIODS)
    check(abs(iterated - L) < _ITERATE_TOL, "iterated recurrence does not converge to the claimed steady-state value")


def _draw_core(rng, ctx: _Ctx):
    """Draw (a, b, L_exact, L_rounded, u0). Construct to target: draw a candidate long-term
    value first over a wide range, round the implied added amount to a clean whole number, then
    recompute L exactly from that clean b."""
    for _ in range(300):
        k = rng.randint(3, 9)
        a = Fraction(k, 10)
        l_target = rng.randint(ctx.l_lo, ctx.l_hi)
        b = round_half_up(Fraction(l_target) * (1 - a), 0)
        if b < 1:
            continue
        l_exact = b / (1 - a)
        if not (ctx.l_lo * Fraction(1, 2) <= l_exact <= ctx.l_hi * Fraction(3, 2)):
            continue
        l_float = float(l_exact)
        if not rounds_unambiguously(l_float, ctx.d):
            continue
        u0 = rng.randint(max(1, int(l_float // 2)), int(l_float * 2) + 20)
        if abs(u0 - l_float) < max(3, 0.15 * l_float):
            continue
        l_rounded = round_half_up(l_exact, ctx.d)
        _verify_steady_state(a, b, l_exact)
        return a, int(b), l_exact, l_rounded, u0
    check(False, "could not draw a plausible steady-state model within the retry budget")


def _common_candidates(a: Fraction, b: int, u0: int) -> list[Fraction]:
    u1 = a * u0 + b
    u2 = a * u1 + b
    return [Fraction(b) / a, Fraction(b), u1, u2]


def _finalise(text: str, answer_value: Fraction, cand_vals: list[Fraction], d: int, working: str) -> dict:
    # Drop any candidate equal to the answer (the same real-world coincidence risk
    # `p16c_finance_time_build._drop_collisions_money` guards against -- e.g. u(1) genuinely
    # equalling L on some rare seed), AND any candidate that coincidentally equals an EARLIER
    # candidate's rounded value (a different misconception formula landing on the same wrong
    # number by chance, not a systematic construction bug). Both are legitimate coincidences to
    # drop before the pairwise-distinct invariant check below, which still fires on a genuine
    # bug that collapses every surviving candidate together.
    answer_text = _fmt(answer_value, d)
    answer_rounded = round_half_up(Fraction(answer_value), d)
    seen = {answer_rounded}
    kept: list[Fraction] = []
    for v in cand_vals:
        rv = round_half_up(Fraction(v), d)
        if rv in seen or rv <= 0:
            # A non-positive candidate (e.g. the exam tier's reciprocal-ratio recovery can yield a
            # negative L on some draws) is never a plausible quantity for any of these contexts
            # and is identifiable by shape alone -- drop it rather than offer
            # it as a distractor.
            continue
        seen.add(rv)
        kept.append(v)
    rounded_kept = [round_half_up(Fraction(v), d) for v in kept]
    check_pairwise_distinct(answer_rounded, *rounded_kept, msg="steady-state distractor collides")
    check(len(kept) >= 1, "no distractor candidate survived construction")
    candidates = [_fmt(v, d) for v in kept]
    distractors = pick_distractors(answer_text, candidates, n=3, tolerance=stated_precision_tolerance(d))
    return dict(
        text=text,
        answer=numeric_answer(answer_rounded),
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=[working],
    )


# =============================================================================
# easy -- the recurrence is stated with its numbers
# =============================================================================

_EASY_TEMPLATES = (
    "Each {time}, {subject} follows the recurrence u(n+1) = {a}u(n) + {b}, where u(n) is "
    "{short_point} recorded at the end of the {time}. Given u(0) = {u0}, find the long-term value "
    "{short} settles to, {prec}.",
    "A model for {subject} is u(n+1) = {a}u(n) + {b}, where u(n) is {short_point} at the end of "
    "each {time}. If u(0) = {u0}, what value does {short} settle to in the long run, {prec}?",
    "The recurrence u(n+1) = {a}u(n) + {b} describes {subject}, with u(n) representing "
    "{short_point} each {time} and u(0) = {u0}. Find the steady long-term value {short} "
    "approaches, {prec}.",
)


def _easy(rng, ctx: _Ctx) -> dict:
    a, b, l_exact, l_rounded, u0 = _draw_core(rng, ctx)
    template = rng.choice(_EASY_TEMPLATES)
    text = template.format(
        subject=ctx.subject,
        short=ctx.short,
        short_point=ctx.short_point,
        unit=ctx.unit,
        time=ctx.time,
        a=_fmt_a(a),
        b=b,
        u0=u0,
        prec=_prec_phrase(ctx.d),
    )
    cand_vals = _common_candidates(a, b, u0)
    working = f"The steady-state value satisfies L = {b} / (1 - {_fmt_a(a)}) = {_fmt(l_exact, ctx.d)}."
    return _finalise(text, l_exact, cand_vals, ctx.d, working)


# =============================================================================
# medium -- the model is described in words; the learner forms a and b
# =============================================================================

_MEDIUM_TEMPLATES = (
    "Each {time}, {loss_clause}, and {add_clause}. If {short} starts at {u0} {unit}, find the "
    "long-term value it settles to, {prec}.",
    "Every {time}, {loss_clause}; {add_clause}. Starting from {u0} {unit}, what value does "
    "{short} settle to in the long run, {prec}?",
    "{loss_clause_cap} each {time}; {add_clause}. Given a starting value of {u0} {unit}, find the "
    "steady long-term value {short} approaches, {prec}.",
)


def _medium(rng, ctx: _Ctx) -> dict:
    a, b, l_exact, l_rounded, u0 = _draw_core(rng, ctx)
    k = a * 10
    p = int(100 - k * 10)
    loss_clause = ctx.loss_clause_tpl.format(p=p)
    add_clause = ctx.add_clause_tpl.format(b=b)
    template = rng.choice(_MEDIUM_TEMPLATES)
    text = template.format(
        time=ctx.time,
        loss_clause=loss_clause,
        add_clause=add_clause,
        short=ctx.short,
        unit=ctx.unit,
        u0=u0,
        prec=_prec_phrase(ctx.d),
        loss_clause_cap=loss_clause[0].upper() + loss_clause[1:],
    )
    cand_vals = _common_candidates(a, b, u0)
    working = (
        f"{p}% lost each {ctx.time} means a = {_fmt_a(a)}; with b = {b}, "
        f"L = {b} / (1 - {_fmt_a(a)}) = {_fmt(l_exact, ctx.d)}."
    )
    return _finalise(text, l_exact, cand_vals, ctx.d, working)


# =============================================================================
# hard -- the OTHER point of the cycle is asked
# =============================================================================

_HARD_AFTER_TEMPLATES = (
    "Each {time}, {subject} follows the recurrence u(n+1) = {a}u(n) + {b}, where u(n) is {short} "
    "measured immediately after the {event}. Given u(0) = {u0}, find the long-term value of {short} "
    "measured immediately before the next {event}, {prec}.",
    "A model for {subject} is u(n+1) = {a}u(n) + {b}, where u(n) is {short} recorded immediately "
    "after the {event} each {time}. If u(0) = {u0}, what value does {short} settle to when measured "
    "immediately before the next {event}, {prec}?",
    "The recurrence u(n+1) = {a}u(n) + {b} models {subject}, with u(n) taken immediately after the "
    "{event} and u(0) = {u0}. Find the steady value of {short} measured immediately before the next "
    "{event}, {prec}.",
)

# "Dose then eliminate" order (u(n) is PRE-event): each period, {b} is added FIRST and THEN the loss
# fraction a is applied to the whole total -- so the recurrence for THIS point is u(n+1) = a(u(n) +
# b), never the same u(n+1) = a u(n) + b the "after" templates use.
_HARD_BEFORE_TEMPLATES = (
    "Each {time}, {subject} follows the recurrence u(n+1) = {a}(u(n) + {b}), where u(n) is {short} "
    "measured immediately before the {event}. Given u(0) = {u0}, find the long-term value of "
    "{short} measured immediately after the next {event}, {prec}.",
    "A model for {subject} is u(n+1) = {a}(u(n) + {b}), where u(n) is {short} recorded immediately "
    "before the {event} each {time}. If u(0) = {u0}, what value does {short} settle to when measured "
    "immediately after the next {event}, {prec}?",
    "The recurrence u(n+1) = {a}(u(n) + {b}) models {subject}, with u(n) taken immediately before "
    "the {event} and u(0) = {u0}. Find the steady value of {short} measured immediately after the "
    "next {event}, {prec}.",
)


def _hard(rng, ctx: _Ctx) -> dict:
    # Both directions share the SAME two cycle values {L, a*L} (L = b/(1-a), the post-event fixed
    # point; a*L, the pre-event fixed point) -- direction only picks which recurrence FORM is
    # shown (a u(n) + b for a post-event u(n), or a(u(n) + b) for a pre-event u(n)) and therefore
    # which of the two values is the "given" one embedded in that formula's own fixed point, and
    # which is "asked". `_draw_core` guards L itself against a rounding boundary; a*L is a
    # DIFFERENT value that can independently land on its own boundary, so redraw until whichever
    # value this direction asks for also rounds unambiguously.
    for _ in range(50):
        a, b, l_exact, l_rounded, u0 = _draw_core(rng, ctx)
        direction = rng.choice(("after", "before"))
        given_value = l_exact if direction == "after" else a * l_exact
        answer_value = a * l_exact if direction == "after" else l_exact
        if rounds_unambiguously(float(answer_value), ctx.d):
            break
    else:
        check(False, "could not draw a hard-tier answer clear of a rounding boundary")

    if direction == "after":
        template = rng.choice(_HARD_AFTER_TEMPLATES)
        u1 = a * u0 + b
        u2 = a * u1 + b
    else:
        template = rng.choice(_HARD_BEFORE_TEMPLATES)
        u1 = a * (u0 + b)
        u2 = a * (u1 + b)
    text = template.format(
        subject=ctx.subject,
        short=ctx.short,
        time=ctx.time,
        event=ctx.event,
        a=_fmt_a(a),
        b=b,
        u0=u0,
        prec=_prec_phrase(ctx.d),
    )
    cand_vals = [Fraction(b) / a, Fraction(b), u1, u2, given_value]
    working = (
        f"u(n) itself settles to {_fmt(given_value, ctx.d)}; the OTHER point in the cycle "
        f"is {_fmt(answer_value, ctx.d)}."
    )
    return _finalise(text, answer_value, cand_vals, ctx.d, working)


# =============================================================================
# exam -- only three recorded consecutive values are given
# =============================================================================

_EXAM_TEMPLATES = (
    "{subject_cap} follows a recurrence of the form u(n+1) = a u(n) + b for constants a and b, "
    "where u(n) is {short_point} at the end of {time} n. Three consecutive recorded values "
    "are u(0) = {u0}, u(1) = {u1} and u(2) = {u2}. Find the long-term value {short} settles to, "
    "{prec}.",
    "A recurrence relation of the form u(n+1) = a u(n) + b models {subject}, with u(0) = {u0}, "
    "u(1) = {u1} and u(2) = {u2} recorded on three consecutive {time}s. Determine the constants a "
    "and b, then find the long-term steady value, {prec}.",
    "{subject_cap} is known to satisfy u(n+1) = a u(n) + b for some constants a and b. Given the "
    "recorded values u(0) = {u0}, u(1) = {u1}, u(2) = {u2}, find the value {short} settles to in "
    "the long term, {prec}.",
)


def _draw_exam(rng, ctx: _Ctx):
    for _ in range(300):
        k = rng.randint(3, 9)
        a = Fraction(k, 10)
        if ctx.d == 0:
            m = rng.randint(1, 4)
            b0 = rng.randint(1, 6)
            u0 = 100 * m
            b = 10 * b0
            u1 = 10 * (k * m + b0)
            u2 = k * k * m + b0 * (k + 10)
        else:
            m = rng.randint(2, 20)
            b0 = rng.randint(1, 20)
            u0 = 10 * m
            b = b0
            u1 = k * m + b0
            u2 = Fraction(k, 10) * u1 + b0
        if Fraction(u1) == Fraction(u0) or Fraction(u2) == Fraction(u1):
            continue  # degenerate draw: consecutive differences would divide by zero
        l_exact = Fraction(b) / (1 - a)
        if not (ctx.l_lo * Fraction(1, 2) <= l_exact <= ctx.l_hi * Fraction(3, 2)):
            continue
        if not rounds_unambiguously(float(l_exact), ctx.d):
            continue
        a_rec = (Fraction(u2) - Fraction(u1)) / (Fraction(u1) - Fraction(u0))
        check(a_rec == a, "exam construction: recovered rate does not match the drawn rate")
        b_rec = Fraction(u1) - a_rec * Fraction(u0)
        check(b_rec == Fraction(b), "exam construction: recovered added amount does not match")
        _verify_steady_state(a, Fraction(b), l_exact)
        return a, Fraction(b), u0, u1, u2, l_exact
    check(False, "could not draw a plausible exam-tier model within the retry budget")


def _exam(rng, ctx: _Ctx) -> dict:
    a, b, u0, u1, u2, l_exact = _draw_exam(rng, ctx)
    template = rng.choice(_EXAM_TEMPLATES)
    text = template.format(
        subject=ctx.subject,
        subject_cap=ctx.subject_cap,
        short=ctx.short,
        short_point=ctx.short_point,
        unit=ctx.unit,
        time=ctx.time,
        u0=_fmt_recorded(u0),
        u1=_fmt_recorded(u1),
        u2=_fmt_recorded(u2),
        prec=_prec_phrase(ctx.d),
    )

    cand_vals: list[Fraction] = [Fraction(u1), Fraction(u2)]
    denom = Fraction(u2) - Fraction(u1)
    if denom != 0:
        a_wrong = (Fraction(u1) - Fraction(u0)) / denom
        if a_wrong != 1:
            b_wrong = Fraction(u1) - a_wrong * Fraction(u0)
            cand_vals.append(b_wrong / (1 - a_wrong))
    if a != 1:
        b_wrong2 = Fraction(u1) - Fraction(u0)
        cand_vals.append(b_wrong2 / (1 - a))

    working = (
        f"Consecutive differences give a = (u2 - u1)/(u1 - u0) = {_fmt_a(a)} and "
        f"b = u1 - a u0 = {int(b) if b.denominator == 1 else float(b)}; "
        f"L = b / (1 - a) = {_fmt(l_exact, ctx.d)}."
    )
    return _finalise(text, l_exact, cand_vals, ctx.d, working)


_PRESETS = {
    "easy": {"variant": "easy"},
    "medium": {"variant": "medium"},
    "hard": {"variant": "hard"},
    "exam": {"variant": "exam"},
}


@generator("pv0a_recurrence_steady_state", level=13, presets=_PRESETS)
def recurrence_steady_state(rng, variant="easy"):
    ctx = rng.choice(_CONTEXTS)
    if variant == "easy":
        return _easy(rng, ctx)
    if variant == "medium":
        return _medium(rng, ctx)
    if variant == "hard":
        return _hard(rng, ctx)
    if variant == "exam":
        return _exam(rng, ctx)
    raise ValueError(f"unknown variant: {variant!r}")
