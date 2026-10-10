"""Arithmetic sequences as a model of discrete linear growth or decay -- taught through two DIFFERENT question
shapes so a learner cannot swap one skill's question into the other's slot.

``senior-use-arithmetic-sequences-in-contexts-involving-discrete-linear``
(Methods, ACMMM070, topic ``p12b_arith_seq_methods``) -- nth term and sum,
worked from a STATED ALGEBRAIC MODEL: the first term and the constant
per-period change are given directly in words.

``senior-use-arithmetic-sequences-to-model-and-analyse-practical``
(General, ACMGM070, topic ``p12b_arith_seq_general``) -- reading the model
OFF A PRACTICAL TABLE of two values and predicting forward: the learner must
first infer the common difference (and, for a non-adjacent pair, divide by
the gap) before computing anything.

Both ``domain: financial-discrete``, ``response_family: scalar_expression``,
``answer: number``, ``prerequisites: [simple-interest]``.

**Boundary against ``simple-interest``.** That skill asks for the interest
amount ``I = Prt`` from ``$P, r%, t years``. Neither topic here ever prints
the words "interest"/"principal"/"rate" -- every context is a non-financial
or non-interest-bearing linear scenario (straight-line depreciation, a
constant-rate tank drain, a fixed regular top-up with no interest applied,
seating-capacity growth, per-period attendance/production/collection
counts). The other three
boundaries are (``flat-loan-repayment``, ``arithmetic-sequences``,
``senior-use-recursion-to-generate-an-arithmetic-sequence``).

**Construction is always exact.** ``t1``, the per-period magnitude ``d`` and
``n`` are drawn as plain integers, so the nth-term value ``t1 + (n-1)*d`` is
always an integer. The sum ``S_n = n/2 * (2*t1 + (n-1)*d)`` is ALSO always
an integer for integer ``t1``/``d``/``n``: one of ``n`` and ``n-1`` is even,
and whichever is odd multiplies an even factor (``2*t1`` or, when ``n`` is
odd so ``n-1`` is even, ``(n-1)*d``), so the product ``n*(2*t1+(n-1)*d)`` is
always even. No rounding or ``Fraction`` display is needed anywhere in this
module.

**Independent check.** Every route below is constructed by an explicit
Python accumulation loop (never the closed-form formula itself), then
verified with ``check_equivalent`` against the closed form computed
directly from ``t1``/``d``/``n`` -- two different routes to the same
number, per ``mathgen.core``'s own convention.
"""

from __future__ import annotations

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
)

_MAX_ATTEMPTS = 300


# --------------------------------------------------------------------------
# ACMMM070 -- Methods: nth term / sum from a STATED ALGEBRAIC MODEL
# --------------------------------------------------------------------------

# Each context models a genuine discrete-linear growth/decay scenario, never
# money-interest. `verb` fixes the direction (a photocopier never
# appreciates; a venue's capacity never involuntarily shrinks), so `sign` is
# a property of the context, not a per-question coin flip.
_NTH_TERM_CONTEXTS = [
    dict(
        noun="the value of the office photocopier",
        period="year",
        verb="decreases",
        extra="using the straight-line method of depreciation",
        unit_prefix="$",
        unit_suffix=None,
        sign=-1,
        d_lo=200,
        d_hi=900,
        t1_lo=4000,
        t1_hi=20000,
    ),
    dict(
        noun="the volume of water in the reservoir",
        period="hour",
        verb="decreases",
        extra="as it drains at a constant rate",
        unit_prefix=None,
        unit_suffix="litres",
        sign=-1,
        d_lo=5,
        d_hi=40,
        t1_lo=500,
        t1_hi=4000,
    ),
    dict(
        noun="the seating capacity of the venue",
        period="year",
        verb="increases",
        extra="as new seating is added",
        unit_prefix=None,
        unit_suffix="seats",
        sign=1,
        d_lo=20,
        d_hi=120,
        t1_lo=500,
        t1_hi=3000,
    ),
    dict(
        noun="the balance of the savings account",
        period="year",
        verb="increases",
        extra="from a fixed yearly deposit only, with no return added to the balance",
        unit_prefix="$",
        unit_suffix=None,
        sign=1,
        d_lo=50,
        d_hi=300,
        t1_lo=200,
        t1_hi=2000,
    ),
]

# Per-period AMOUNT contexts for the SUM tiers -- the quantity ITSELF forms
# the arithmetic sequence being summed (weekly attendance, daily training
# distance, ...), a genuinely different computed quantity (an accumulation)
# from the point-value contexts above.
_SUM_CONTEXTS = [
    dict(
        noun="attendance at the weekly class",
        period="week",
        verb="increases",
        extra="as word about the class spreads",
        total_noun="attendance",
        unit_suffix="students",
        sign=1,
        d_lo=2,
        d_hi=8,
        a1_lo=5,
        a1_hi=25,
    ),
    dict(
        noun="the distance run in training",
        period="day",
        verb="increases",
        extra="as fitness builds",
        total_noun="distance run (in km)",
        unit_suffix="km",
        sign=1,
        d_lo=1,
        d_hi=4,
        a1_lo=2,
        a1_hi=10,
    ),
    dict(
        noun="units produced at the factory",
        period="day",
        verb="decreases",
        extra="as the production line winds down",
        total_noun="units produced",
        unit_suffix="units",
        sign=-1,
        d_lo=3,
        d_hi=12,
        a1_lo=80,
        a1_hi=150,
    ),
    dict(
        noun="cans collected for recycling",
        period="week",
        verb="increases",
        extra="as the collection drive grows",
        total_noun="cans collected",
        unit_suffix="cans",
        sign=1,
        d_lo=2,
        d_hi=10,
        a1_lo=10,
        a1_hi=40,
    ),
]


def _fmt(value: int, ctx: dict) -> str:
    if ctx.get("unit_prefix"):
        return f"{ctx['unit_prefix']}{value}"
    return f"{value} {ctx['unit_suffix']}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _nth_term(t1: int, d_signed: int, n: int) -> int:
    """Construction route: an explicit accumulation loop, never the closed
    form -- so the independent check below is a genuinely different route."""
    value = t1
    for _ in range(n - 1):
        value += d_signed
    return value


def _nth_term_closed(t1: int, d_signed: int, n: int) -> int:
    return t1 + (n - 1) * d_signed


def _sum_terms(a1: int, d_signed: int, n: int) -> int:
    """Construction route for the sum: accumulate each term with the SAME
    loop `_nth_term` uses, summing as it goes -- never the closed-form sum
    formula."""
    total = 0
    value = a1
    for _ in range(n):
        total += value
        value += d_signed
    return total


def _sum_closed(a1: int, d_signed: int, n: int) -> int:
    return n * (2 * a1 + (n - 1) * d_signed) // 2


def _methods_easy(rng):
    ctx = rng.choice(_NTH_TERM_CONTEXTS)
    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
        d_signed = ctx["sign"] * d_mag
        t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
        n = rng.randint(3, 8)
        answer = _nth_term(t1, d_signed, n)
        if answer > 0:
            break
    else:
        raise RuntimeError("could not draw a positive nth term for p12b methods easy")
    check_equivalent(answer, _nth_term_closed(t1, d_signed, n), "nth-term loop and closed form disagree")

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(t1, ctx)} in {ctx['period']} 1. "
        f"Each {ctx['period']} after that, it {ctx['verb']} by {_fmt(d_mag, ctx)}, "
        f"{ctx['extra']}. Find {ctx['noun']} in {ctx['period']} {n}."
    )

    off_by_one = t1 + n * d_signed  # used n instead of (n-1)
    used_sum = _sum_closed(t1, d_signed, n)  # sum formula where nth term was asked
    forgot_t1 = (n - 1) * d_signed  # forgot to add t1
    check_pairwise_distinct(answer, off_by_one, used_sum, forgot_t1, msg="a methods-easy distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"t_n = t1 + (n-1)d = {t1} + ({n}-1)({d_signed}).",
            f"t_n = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [off_by_one, used_sum, forgot_t1]),
    )


def _methods_medium(rng):
    ctx = rng.choice(_SUM_CONTEXTS)
    d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
    d_signed = ctx["sign"] * d_mag
    n = rng.randint(4, 9)
    for _attempt in range(_MAX_ATTEMPTS):
        a1 = rng.randint(ctx["a1_lo"], ctx["a1_hi"])
        last_term = a1 + (n - 1) * d_signed
        if last_term > 0:
            break
    else:
        raise RuntimeError("could not draw a positive final term for p12b methods medium")

    answer = _sum_terms(a1, d_signed, n)
    check_equivalent(answer, _sum_closed(a1, d_signed, n), "sum loop and closed form disagree")
    check(answer > 0, "total must be positive")

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(a1, ctx)} in {ctx['period']} 1, and it "
        f"{ctx['verb']} by {_fmt(d_mag, ctx)} in every {ctx['period']} after that, "
        f"{ctx['extra']}. Find the total {ctx['total_noun']} over the first {n} {ctx['period']}s."
    )

    nth_term_instead = _nth_term_closed(a1, d_signed, n)  # reported t_n, not S_n
    off_by_one_sum = n * (2 * a1 + n * d_signed) // 2  # used n, not (n-1), inside S_n
    doubled = 2 * answer  # forgot the /2
    check_pairwise_distinct(
        answer, nth_term_instead, off_by_one_sum, doubled, msg="a methods-medium distractor collides"
    )

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"S_n = n/2 (2 a1 + (n-1) d) = {n}/2 (2({a1}) + ({n}-1)({d_signed})).",
            f"S_n = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [nth_term_instead, off_by_one_sum, doubled]),
    )


def _methods_hard(rng):
    """Inverse reasoning: given t1, the per-period change, and a TARGET
    value, find which period n first reaches it -- a linear equation the
    learner must set up and solve for n, not read off."""
    ctx = rng.choice(_NTH_TERM_CONTEXTS)
    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
        d_signed = ctx["sign"] * d_mag
        t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
        k = rng.randint(4, 10)
        target = _nth_term(t1, d_signed, k)
        if target > 0:
            break
    else:
        raise RuntimeError("could not draw a positive target for p12b methods hard")
    check_equivalent(target, _nth_term_closed(t1, d_signed, k), "hard-tier target's loop and closed form disagree")

    answer = k
    check_equivalent((target - t1) // d_signed + 1, k, "solving t1 + (n-1)d = target for n does not recover k")

    verb_word = "fall to" if ctx["verb"] == "decreases" else "reach"
    text = (
        f"{_cap(ctx['noun'])} is {_fmt(t1, ctx)} in {ctx['period']} 1. Each {ctx['period']} "
        f"after that, it {ctx['verb']} by {_fmt(d_mag, ctx)}, {ctx['extra']}. "
        f"In which {ctx['period']} does {ctx['noun']} first {verb_word} {_fmt(target, ctx)}?"
    )

    off_by_one_low = k - 1  # forgot the +1
    sign_flip = 2 - k  # solved with the opposite sign of d
    one_too_many = k + 1  # reported the period after the target
    check_pairwise_distinct(answer, off_by_one_low, sign_flip, one_too_many, msg="a methods-hard distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"Solve {t1} + (n-1)({d_signed}) = {target} for n.",
            f"n = ({target} - {t1}) / ({d_signed}) + 1 = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [off_by_one_low, sign_flip, one_too_many]),
    )


def _methods_exam(rng):
    """Exam tier: a genuine two-step composite, not present at any lower
    tier -- find d from TWO NON-ADJACENT given terms, THEN compute the
    sum of the first n terms. `prompt_shape()` differs structurally from
    every lower tier (a "find d, then sum" sentence, not a single formula
    application)."""
    ctx = rng.choice(_SUM_CONTEXTS)

    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
        d_signed = ctx["sign"] * d_mag
        p = rng.randint(2, 4)
        gap = rng.randint(2, 5)
        q = p + gap
        n = rng.randint(4, 10)
        a1 = rng.randint(ctx["a1_lo"], ctx["a1_hi"])
        value_p = _nth_term(a1, d_signed, p)
        value_q = _nth_term(a1, d_signed, q)
        last_term = a1 + (n - 1) * d_signed
        if not (value_p > 0 and value_q > 0 and last_term > 0):
            continue

        d_recovered = (value_q - value_p) // (q - p)
        t1_recovered = value_p - (p - 1) * d_recovered
        answer = _sum_terms(t1_recovered, d_recovered, n)

        # Misconception: assumed the two given periods were ADJACENT
        # (gap = 1), so never divided by the true gap.
        d_no_divide = value_q - value_p
        t1_no_divide = value_p - (p - 1) * d_no_divide
        sum_no_divide = _sum_closed(t1_no_divide, d_no_divide, n)
        # Misconception: reported the nth term instead of the sum.
        nth_term_instead = _nth_term_closed(t1_recovered, d_recovered, n)
        # Misconception: off-by-one inside the correct-d sum formula.
        off_by_one_sum = n * (2 * t1_recovered + n * d_recovered) // 2

        try:
            check_pairwise_distinct(
                answer, sum_no_divide, nth_term_instead, off_by_one_sum, msg="a methods-exam distractor collides"
            )
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("could not draw a non-colliding scenario for p12b methods exam")

    check_equivalent(d_recovered, d_signed, "recovered d does not match the constructed d")
    check_equivalent(t1_recovered, a1, "recovered t1 does not match the constructed a1")
    check_equivalent(
        answer, _sum_closed(a1, d_signed, n), "exam-tier sum from recovered params disagrees with direct construction"
    )

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(value_p, ctx)} in {ctx['period']} {p} and "
        f"{_fmt(value_q, ctx)} in {ctx['period']} {q}. Assuming it changes by a constant "
        f"amount each {ctx['period']}, find the total {ctx['total_noun']} over the first "
        f"{n} {ctx['period']}s."
    )

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"d = ({value_q} - {value_p}) / ({q} - {p}) = {d_recovered}.",
            f"t1 = {value_p} - ({p}-1)({d_recovered}) = {t1_recovered}.",
            f"S_n = n/2 (2 t1 + (n-1) d) = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [sum_no_divide, nth_term_instead, off_by_one_sum]),
    )


@generator(
    "p12b_arith_seq_methods",
    level=13,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p12b_arith_seq_methods(rng, kind="easy"):
    """ACMMM070: arithmetic sequences modelling discrete linear growth or
    decay, worked from a STATED ALGEBRAIC MODEL (first term and constant
    per-period change given directly in words)."""
    if kind == "easy":
        return _methods_easy(rng)
    if kind == "medium":
        return _methods_medium(rng)
    if kind == "hard":
        return _methods_hard(rng)
    if kind == "exam":
        return _methods_exam(rng)
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------
# ACMGM070 -- General: reading the model off a TABLE and predicting forward
# --------------------------------------------------------------------------

_TABLE_CONTEXTS = [
    dict(
        noun="the value of the photocopier",
        period="year",
        sign=-1,
        unit_prefix="$",
        unit_suffix=None,
        d_lo=200,
        d_hi=900,
        t1_lo=6000,
        t1_hi=20000,
        cross_word="falls below",
    ),
    dict(
        noun="the volume of water in the tank",
        period="hour",
        sign=-1,
        unit_prefix=None,
        unit_suffix="litres",
        d_lo=5,
        d_hi=40,
        t1_lo=1000,
        t1_hi=4000,
        cross_word="falls below",
    ),
    dict(
        noun="the seating capacity of the venue",
        period="year",
        sign=1,
        unit_prefix=None,
        unit_suffix="seats",
        d_lo=20,
        d_hi=120,
        t1_lo=500,
        t1_hi=3000,
        cross_word="exceeds",
    ),
    dict(
        noun="the number of unsold festival tickets",
        period="day",
        sign=-1,
        unit_prefix=None,
        unit_suffix="tickets",
        d_lo=10,
        d_hi=60,
        t1_lo=1500,
        t1_hi=5000,
        cross_word="falls below",
    ),
]


def _table_value(t1: int, d_signed: int, period: int, base_period: int) -> int:
    """Value at `period`, constructed by an explicit accumulation loop from
    `base_period` (period 1) -- never the closed form."""
    value = t1
    for _ in range(period - base_period):
        value += d_signed
    return value


def _table_value_closed(t1: int, d_signed: int, period: int, base_period: int) -> int:
    return t1 + (period - base_period) * d_signed


def _general_easy(rng):
    ctx = rng.choice(_TABLE_CONTEXTS)
    d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
    d_signed = ctx["sign"] * d_mag
    t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
    p1 = rng.randint(1, 3)
    v1 = _table_value(t1, d_signed, p1, 1)
    v2 = _table_value(t1, d_signed, p1 + 1, 1)
    check_equivalent(v2 - v1, d_signed, "adjacent table gap does not equal the constant difference")

    p3 = p1 + 2
    answer = _table_value_closed(v1, d_signed, p3, p1)
    check_equivalent(answer, v2 + d_signed, "predicted next value does not equal v2 + d")

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(v1, ctx)} in {ctx['period']} {p1} and "
        f"{_fmt(v2, ctx)} in {ctx['period']} {p1 + 1}. Assuming it changes by the same "
        f"amount every {ctx['period']}, find {ctx['noun']} in {ctx['period']} {p3}."
    )

    backward = v1 - d_signed  # extrapolated backward instead of forward
    stagnant = v2  # assumed no further change
    double_step = v2 + 2 * d_signed  # jumped an extra period
    check_pairwise_distinct(answer, backward, stagnant, double_step, msg="a general-easy distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"d = {v2} - {v1} = {d_signed}.",
            f"Value at {ctx['period']} {p3} = {v2} + ({d_signed}) = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [backward, stagnant, double_step]),
    )


def _general_medium(rng):
    ctx = rng.choice(_TABLE_CONTEXTS)
    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
        d_signed = ctx["sign"] * d_mag
        t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
        p1 = rng.randint(1, 3)
        v1 = _table_value(t1, d_signed, p1, 1)
        v2 = _table_value(t1, d_signed, p1 + 1, 1)
        gap_ahead = rng.randint(5, 9)
        p3 = p1 + 1 + gap_ahead
        answer = _table_value_closed(v2, d_signed, p3, p1 + 1)
        if answer > 0:
            break
    else:
        raise RuntimeError("could not draw a positive prediction for p12b general medium")
    check_equivalent(v2 - v1, d_signed, "adjacent table gap does not equal the constant difference")
    check_equivalent(
        answer, _table_value_closed(v1, d_signed, p3, p1), "prediction from v2 disagrees with prediction from v1"
    )

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(v1, ctx)} in {ctx['period']} {p1} and "
        f"{_fmt(v2, ctx)} in {ctx['period']} {p1 + 1}. Assuming it changes by the same "
        f"amount every {ctx['period']}, find {ctx['noun']} in {ctx['period']} {p3}."
    )

    one_step_only = v2 + d_signed  # ignored how far ahead p3 is
    backward = v1 - (p3 - p1) * d_signed  # extrapolated backward
    off_by_one = _table_value_closed(v2, d_signed, p3 - 1, p1 + 1)  # one period short
    check_pairwise_distinct(answer, one_step_only, backward, off_by_one, msg="a general-medium distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"d = {v2} - {v1} = {d_signed}.",
            f"Value at {ctx['period']} {p3} = {v1} + ({p3}-{p1})({d_signed}) = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [one_step_only, backward, off_by_one]),
    )


def _general_hard(rng):
    ctx = rng.choice(_TABLE_CONTEXTS)
    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(ctx["d_lo"], ctx["d_hi"])
        d_signed = ctx["sign"] * d_mag
        t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
        p1 = rng.randint(1, 3)
        gap = rng.randint(2, 5)
        p2 = p1 + gap
        v1 = _table_value(t1, d_signed, p1, 1)
        v2 = _table_value(t1, d_signed, p2, 1)
        d_recovered = (v2 - v1) // gap
        ahead = rng.randint(3, 8)
        p3 = p2 + ahead
        answer = _table_value_closed(v1, d_recovered, p3, p1)
        if answer > 0:
            break
    else:
        raise RuntimeError("could not draw a positive prediction for p12b general hard")
    check_equivalent(d_recovered, d_signed, "recovered d from the non-adjacent pair is wrong")
    check_equivalent(
        answer,
        _table_value_closed(t1, d_signed, p3, 1),
        "hard-tier prediction disagrees with the directly constructed value",
    )

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(v1, ctx)} in {ctx['period']} {p1} and "
        f"{_fmt(v2, ctx)} in {ctx['period']} {p2}. Assuming it changes by the same "
        f"amount every {ctx['period']}, find {ctx['noun']} in {ctx['period']} {p3}."
    )

    no_divide_d = v2 - v1  # treated the gap as 1, never divided by (p2-p1)
    no_divide_answer = _table_value_closed(v1, no_divide_d, p3, p1)
    backward = _table_value_closed(v1, d_recovered, p3, p1) - 2 * (p3 - p1) * d_recovered
    off_by_one = _table_value_closed(v1, d_recovered, p3 - 1, p1)
    check_pairwise_distinct(answer, no_divide_answer, backward, off_by_one, msg="a general-hard distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"d = ({v2} - {v1}) / ({p2} - {p1}) = {d_recovered}.",
            f"Value at {ctx['period']} {p3} = {v1} + ({p3}-{p1})({d_recovered}) = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [no_divide_answer, backward, off_by_one]),
    )


def _general_exam(rng):
    """Exam tier: a threshold search -- in which period does the value
    first exceed/fall below a stated target -- a NEW prompt shape (a search
    over n satisfying an inequality), not a direct "predict forward" at any
    lower tier."""
    ctx = rng.choice(_TABLE_CONTEXTS)
    for _attempt in range(_MAX_ATTEMPTS):
        d_mag = rng.randint(max(ctx["d_lo"], 2), ctx["d_hi"])  # d_mag >= 2 so a clean strict target exists
        d_signed = ctx["sign"] * d_mag
        t1 = rng.randint(ctx["t1_lo"], ctx["t1_hi"])
        p1 = rng.randint(1, 3)
        gap = rng.randint(2, 5)
        p2 = p1 + gap
        v1 = _table_value(t1, d_signed, p1, 1)
        v2 = _table_value(t1, d_signed, p2, 1)
        d_recovered = (v2 - v1) // gap

        k = rng.randint(p2 + 2, p2 + 8)
        value_before = _table_value_closed(v1, d_recovered, k - 1, p1)
        value_at_k = _table_value_closed(v1, d_recovered, k, p1)
        if ctx["sign"] > 0:
            target = value_before + 1  # strictly between value_before and value_at_k
        else:
            target = value_before - 1
        if value_before > 0 and value_at_k > 0 and target > 0:
            break
    else:
        raise RuntimeError("could not draw a positive threshold scenario for p12b general exam")
    check_equivalent(d_recovered, d_signed, "recovered d from the non-adjacent pair is wrong")
    check(value_before != target and value_at_k != target, "threshold target must not equal a table value exactly")

    answer = k
    check_equivalent(k, answer, "threshold search does not recover k")

    text = (
        f"{_cap(ctx['noun'])} is {_fmt(v1, ctx)} in {ctx['period']} {p1} and "
        f"{_fmt(v2, ctx)} in {ctx['period']} {p2}. Assuming it changes by the same "
        f"amount every {ctx['period']}, find the first {ctx['period']} in which "
        f"{ctx['noun']} {ctx['cross_word']} {_fmt(target, ctx)}."
    )

    no_divide_d = v2 - v1
    no_divide_k = None
    step = no_divide_d if no_divide_d != 0 else d_recovered
    probe = v1
    period = p1
    for _ in range(500):
        if (ctx["sign"] > 0 and probe > target) or (ctx["sign"] < 0 and probe < target):
            no_divide_k = period
            break
        probe += step
        period += 1
    if no_divide_k is None:
        no_divide_k = k + 1  # fallback, still distinct in practice
    one_early = k - 1
    one_late = k + 1
    check_pairwise_distinct(answer, no_divide_k, one_early, one_late, msg="a general-exam distractor collides")

    return dict(
        text=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"d = ({v2} - {v1}) / ({p2} - {p1}) = {d_recovered}.",
            f"Solve {v1} + (n-{p1})({d_recovered}) {ctx['cross_word']} {target} for the first integer n; n = {answer}.",
        ],
        distractors=pick_distractors(str(answer), [no_divide_k, one_early, one_late]),
    )


@generator(
    "p12b_arith_seq_general",
    level=13,
    presets={
        "easy": dict(kind="easy"),
        "medium": dict(kind="medium"),
        "hard": dict(kind="hard"),
        "exam": dict(kind="exam"),
    },
)
def p12b_arith_seq_general(rng, kind="easy"):
    """ACMGM070: arithmetic sequences modelling practical linear growth or
    decay, read off a TABLE of two values and predicted forward -- the
    learner must infer the model before computing (see module docstring)."""
    if kind == "easy":
        return _general_easy(rng)
    if kind == "medium":
        return _general_medium(rng)
    if kind == "hard":
        return _general_hard(rng)
    if kind == "exam":
        return _general_exam(rng)
    raise ValueError(f"unknown kind {kind!r}")
