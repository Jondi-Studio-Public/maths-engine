"""Growth and decay modelled by a recurrence: `r3j_recurrence` (easy, medium, hard) and the exam chain
`exd_x_r3j_recurrence_savings`.

``compound-recurrence-model`` (``financial-discrete``, answers ``number`` and, in the exam chain, ``expression``). The
model is V(n+1) = r V(n), or V(n+1) = r V(n) + D or r V(n) - D, with the change per step given as a percentage and
the extra amount described in words. Loans and annuities are covered by their own skills; the contexts here are
savings and funds, a drug in the body, chlorine in a pool and water in a tank.

* easy: the value after a few steps of V(n+1) = r V(n), the rate given as a percentage growth or loss.
* medium: the value after a few steps when an amount is added or taken out at the end of every step.
* hard: the first step at which the value passes a target; or the amount to add or remove each step that holds the
  value constant; or the total change that is not due to the amount added.
* exam (``exd_x_r3j_recurrence_savings``): write the rule as an expression, the balance after three years, the
  interest earned over those years and the first year the balance passes a target.

Every value is built twice: by running the recurrence step by step, and from the closed form
V(n) = r^n V(0) + D (r^n - 1) / (r - 1), which never shares an intermediate with the step-by-step route.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import PEOPLE, dollars, number_part, number_text, plain, single_number

# kind, label for V, unit text (after the number), start range, step noun, "%-rate wording"
_CONTEXTS = [
    dict(
        key="savings",
        money=True,
        v0=(2000, 12000, 500),
        rates=[2, 3, 3.5, 4, 4.5, 5, 6],
        grow=True,
        step="year",
        add=True,
        d=(200, 1200, 100),
        intro="{name} opens a savings account with {v0}. The account earns interest at {p}% each year, added at the end of the year.",
        change="At the end of each year, after the interest is added, {name} deposits {d}.",
        remove="At the end of each year, after the interest is added, {name} withdraws {d}.",
        ask="the balance of the account",
        unit="",
    ),
    dict(
        key="fund",
        money=True,
        v0=(20000, 90000, 5000),
        rates=[3, 4, 5, 6, 7],
        grow=True,
        step="year",
        add=False,
        d=(1000, 6000, 500),
        intro="A community fund starts with {v0}. It earns {p}% a year, added at the end of the year.",
        change="At the end of each year, after the interest is added, {d} is paid into the fund.",
        remove="At the end of each year, after the interest is added, {d} is paid out as grants.",
        ask="the balance of the fund",
        unit="",
    ),
    dict(
        key="drug",
        money=False,
        v0=(200, 800, 50),
        rates=[10, 15, 20, 25, 30],
        grow=False,
        step="hour",
        add=True,
        d=(20, 80, 10),
        intro="A patient has {v0} of a medicine in the bloodstream. Each hour the body removes {p}% of what is there.",
        change="Straight after that, a further {d} is given by a drip each hour.",
        remove="Straight after that, a further {d} is removed by a filter each hour.",
        ask="the amount of medicine",
        unit=" mg",
    ),
    dict(
        key="chlorine",
        money=False,
        v0=(20, 60, 5),
        rates=[10, 12, 15, 20, 25],
        grow=False,
        step="day",
        add=True,
        d=(4, 12, 1),
        intro="A swimming pool holds {v0} of chlorine. Each day {p}% of the chlorine in the pool is lost to sunlight.",
        change="At the end of each day, after that loss, {d} of chlorine is added.",
        remove="At the end of each day, after that loss, {d} of chlorine is drawn off for testing.",
        ask="the amount of chlorine in the pool",
        unit=" g",
    ),
    dict(
        key="tank",
        money=False,
        v0=(800, 3000, 100),
        rates=[2, 3, 4, 5],
        grow=False,
        step="week",
        add=True,
        d=(40, 200, 10),
        intro="A rainwater tank holds {v0}. Each week {p}% of the water in the tank is lost to evaporation and leaks.",
        change="At the end of each week, after that loss, {d} of rain run in.",
        remove="At the end of each week, after that loss, {d} are drawn off for the garden.",
        ask="the volume of water in the tank",
        unit=" litres",
    ),
]


def _amount(ctx, value, places=None):
    if ctx["money"]:
        return dollars(value, places)
    text = number_text(value, places if places is not None else (0 if Fraction(value).denominator == 1 else 2))
    return text + ctx["unit"]


def _model(rng, *, with_d, sign=None, wide_d=False):
    """A random context and its numbers: (ctx, name, v0, p (Fraction percent), r, d (signed Fraction))."""
    ctx = rng.choice(_CONTEXTS)
    name = rng.choice(PEOPLE)
    lo, hi, step = ctx["v0"]
    v0 = Fraction(rng.randrange(lo, hi + 1, step))
    p = Fraction(str(rng.choice(ctx["rates"])))
    r = 1 + p / 100 if ctx["grow"] else 1 - p / 100
    d = Fraction(0)
    if with_d:
        dl, dh, ds = ctx["d"]
        d = Fraction(rng.randrange(dl, dh + 1, ds))
        add = ctx["add"] if sign is None else sign
        if sign is None and rng.random() < 0.35:
            add = not add
        d = d if add else -d
    return ctx, name, v0, p, r, d


def _run(v0, r, d, n):
    v = Fraction(v0)
    for _ in range(n):
        v = r * v + d
    return v


def _closed(v0, r, d, n):
    if r == 1:
        return v0 + n * d
    return r**n * v0 + d * (r**n - 1) / (r - 1)


def _value(v0, r, d, n):
    step_by_step = _run(v0, r, d, n)
    check(step_by_step == _closed(v0, r, d, n), "recurrence and closed form disagree")
    return step_by_step


def _stays_positive(v0, r, d, n):
    v = Fraction(v0)
    for _ in range(n):
        v = r * v + d
        if v <= v0 / 20:
            return False
    return True


def _stem(ctx, name, v0, p, d):
    parts = [ctx["intro"].format(name=name, v0=_amount(ctx, v0), p=plain(p))]
    if d:
        template = ctx["change"] if d > 0 else ctx["remove"]
        parts.append(template.format(name=name, d=_amount(ctx, abs(d))))
    return " ".join(parts)


def _slips_value(ctx, v0, p, r, d, n):
    return [(v, tag) for v, tag in _raw_slips(v0, p, r, d, n) if v > 0]


def _raw_slips(v0, p, r, d, n):
    if d:
        return [
            (_value(v0, r, 0, n), "r3j-recurrence-added-amount-left-out"),
            (_value(v0, r, 0, n) + n * d, "r3j-recurrence-added-amount-not-compounded"),
            (_run(v0, r, d, n - 1), "r3j-recurrence-one-step-short"),
            (_run(v0, r, d, n + 1), "r3j-recurrence-one-step-too-many"),
            (_run_amount_first(v0, r, d, n), "r3j-recurrence-amount-before-change"),
            (_run(v0, r, -d, n), "r3j-recurrence-sign-of-amount-reversed"),
        ]
    return [
        (v0 * (1 + n * (r - 1)), "r3j-recurrence-simple-instead-of-compound"),
        (_run(v0, r, 0, n - 1), "r3j-recurrence-one-step-short"),
        (_run(v0, r, 0, n + 1), "r3j-recurrence-one-step-too-many"),
        (_run(v0, 1 + (r - 1) * 10, 0, n), "r3j-recurrence-percent-as-tenth"),
        (v0 * (2 - r) ** n, "r3j-recurrence-growth-and-decay-swapped"),
    ]


def _run_amount_first(v0, r, d, n):
    v = Fraction(v0)
    for _ in range(n):
        v = r * (v + d)
    return v


def _easy(rng):
    ctx, name, v0, p, r, d = _model(rng, with_d=False)
    n = rng.choice([3, 4, 5])
    key = _value(v0, r, d, n)
    text = f"{_stem(ctx, name, v0, p, d)} What is {ctx['ask']} after {n} {ctx['step']}s, to 2 decimal places?"
    slips = _slips_value(ctx, v0, p, r, d, n)
    working = [f"Each {ctx['step']} the amount is multiplied by {plain(r)}: V(n+1) = {plain(r)} V(n)."] + [
        f"V({i}) = {number_text(_run(v0, r, d, i), 2)}." for i in range(1, n + 1)
    ]
    return single_number(text, key, 2, working, slips)


def _medium(rng):
    for _ in attempts(500, "a model whose value stays positive"):
        ctx, name, v0, p, r, d = _model(rng, with_d=True)
        n = rng.choice([3, 4, 5])
        if _stays_positive(v0, r, d, n + 1):
            break
    key = _value(v0, r, d, n)
    text = f"{_stem(ctx, name, v0, p, d)} What is {ctx['ask']} after {n} {ctx['step']}s, to 2 decimal places?"
    working = [
        f"V(n+1) = {plain(r)} V(n) {'+' if d > 0 else '-'} {plain(abs(d))}, with V(0) = {plain(v0)}.",
    ] + [f"V({i}) = {number_text(_run(v0, r, d, i), 2)}." for i in range(1, n + 1)]
    return single_number(text, key, 2, working, _slips_value(ctx, v0, p, r, d, n))


def _first_passing(v0, r, d, target, cap=60):
    v, n = Fraction(v0), 0
    rising = target > v0
    for _ in range(cap):
        if (rising and v >= target) or (not rising and v <= target):
            return n
        v = r * v + d
        n += 1
    return None


def _hard_target(rng):
    for _ in attempts(2000, "a target first passed after 5 to 14 steps"):
        ctx, name, v0, p, r, d = _model(rng, with_d=True)
        if r > 1 and d > 0:
            rising = True
        elif r < 1 and d < 0:
            rising = False
        else:
            continue
        n = rng.randint(5, 14)
        value, before = _value(v0, r, d, n), _value(v0, r, d, n - 1)
        if abs(value - before) < v0 / 50:
            continue
        mid = (value + before) / 2
        target = Fraction(round(mid / 50) * 50) if mid > 1000 else Fraction(round(mid))
        if (rising and not before < target <= value) or (not rising and not value <= target < before):
            continue
        if _first_passing(v0, r, d, target) == n:
            break
    goal = f"first reach at least {_amount(ctx, target)}" if rising else f"first fall to {_amount(ctx, target)} or less"
    text = f"{_stem(ctx, name, v0, p, d)} After how many {ctx['step']}s will {ctx['ask']} {goal}?"
    slips = [
        (n - 1, "r3j-recurrence-one-step-short"),
        (n + 1, "r3j-recurrence-one-step-too-many"),
        (n + 3, "r3j-recurrence-constant-change-assumed"),
        (n * 2, "r3j-recurrence-doubled-count"),
    ]
    working = [f"V({i}) = {number_text(_run(v0, r, d, i), 2)}." for i in range(1, n + 1)]
    working.append(
        f"V({n - 1}) = {number_text(before, 2)} has not passed {_amount(ctx, target)}; V({n}) has, so n = {n}."
    )
    return single_number(text, Fraction(n), 0, working, slips, keep=3)


def _hard_steady(rng):
    ctx, name, v0, p, r, d = _model(rng, with_d=False)
    needed = (1 - r) * v0  # added each step to hold the value at V(0); negative means taken out
    check(r * v0 + needed == v0, "the steady amount does not hold the value")
    action = "taken out" if needed < 0 else "added"
    timing = "after the growth" if ctx["grow"] else "after the loss"
    text = (
        f"{_stem(ctx, name, v0, p, Fraction(0))} At the end of each {ctx['step']}, {timing}, an amount is {action} so "
        f"that {ctx['ask']} stays exactly {_amount(ctx, v0)}. How much is {action} each {ctx['step']}? "
        "Give your answer to 2 decimal places."
    )
    key = abs(needed)
    slips = [
        (v0 * p / 100 * 2, "r3j-recurrence-double-change"),
        (v0 * r, "r3j-recurrence-new-value-given"),
        (v0 / r - v0 if v0 / r - v0 > 0 else v0 - v0 / r, "r3j-recurrence-divided-by-multiplier"),
        (v0 * p / 10, "r3j-recurrence-percent-as-tenth"),
    ]
    working = [
        f"Hold the value at {plain(v0)}: {plain(r)} x {plain(v0)} {'-' if needed < 0 else '+'} D = {plain(v0)}.",
        f"D = {number_text(key, 2)}.",
    ]
    return single_number(text, key, 2, working, slips)


def _hard_total(rng):
    ctx, name, v0, p, r, d = _model(rng, with_d=True, sign=True)
    # growth contexts with amounts added: the part of the change that the growth earned
    for _ in attempts(500, "a growth context"):
        if r > 1 and d > 0:
            break
        ctx, name, v0, p, r, d = _model(rng, with_d=True, sign=True)
    n = rng.choice([3, 4, 5, 6])
    final = _value(v0, r, d, n)
    key = final - v0 - n * d
    text = (
        f"{_stem(ctx, name, v0, p, d)} How much of the change in {ctx['ask']} over {n} {ctx['step']}s is due to the "
        f"interest alone, not to the {'deposits' if ctx['key'] == 'savings' else 'payments'}? Give your answer to 2 decimal places."
    )
    slips = [
        (final - v0, "r3j-recurrence-total-change-given"),
        (v0 * (r - 1) * n, "r3j-recurrence-interest-on-start-only"),
        (final - n * d, "r3j-recurrence-start-value-left-in"),
        (final - v0 - (n - 1) * d, "r3j-recurrence-one-amount-short"),
    ]
    working = [
        f"V({n}) = {number_text(final, 2)}.",
        f"Total added = {n} x {plain(d)} = {plain(n * d)}.",
        f"Interest alone = {number_text(final, 2)} - {plain(v0)} - {plain(n * d)} = {number_text(key, 2)}.",
    ]
    return single_number(text, key, 2, working, slips)


@generator(
    "r3j_recurrence",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_recurrence(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_target, _hard_steady, _hard_total])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------

V = sympy.Symbol("V")


def _rule_text(r, d):
    mult = plain(r)
    sign = "+" if d > 0 else "-"
    return f"{mult}V {sign} {plain(abs(d))}"


@generator("exd_x_r3j_recurrence_savings", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_recurrence_savings(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    name = rng.choice(PEOPLE)
    for _ in attempts(500, "a savings target reached in 5 to 12 years"):
        v0 = Fraction(rng.randrange(2000, 9001, 500))
        p = Fraction(str(rng.choice([3, 3.5, 4, 4.5, 5, 6])))
        r = 1 + p / 100
        d = Fraction(rng.randrange(300, 1501, 100))
        n_target = rng.randint(6, 12)
        hi, lo = _value(v0, r, d, n_target), _value(v0, r, d, n_target - 1)
        target = Fraction(round(((hi + lo) / 2) / 500) * 500)
        if lo < target <= hi and _first_passing(v0, r, d, target) == n_target:
            break
    v3 = _value(v0, r, d, 3)
    interest = v3 - v0 - 3 * d
    rule = r * V + d
    text = (
        f"{name} opens a savings account with {dollars(v0)}. The account earns interest at {plain(p)}% a year, added at "
        f"the end of each year, and straight after that {name} deposits {dollars(d)}. Let V be the balance at the "
        "end of one year, in dollars, after the deposit."
    )
    parts = [
        dict(
            prompt="Write an expression, in V, for the balance at the end of the next year.",
            answer_kind="expression",
            answer=str(rule),
            answer_text=_rule_text(r, d),
            answer_latex=latex_answer(rule),
            marks=2,
            working=[
                f"Multiply by 1 + {plain(p)}/100 = {plain(r)} for the interest, then add {plain(d)}: {_rule_text(r, d)}."
            ],
            distractors=[
                Distractor(f"{plain(p / 100)}V + {plain(d)}", "r3j-recurrence-rate-not-multiplier"),
                Distractor(f"{plain(r)}(V + {plain(d)})", "r3j-recurrence-amount-before-change"),
                Distractor(f"V + {number_text(p, 1)} + {plain(d)}", "r3j-recurrence-percent-added-as-dollars"),
            ],
        ),
        number_part(
            "Find the balance at the end of the third year, to the nearest cent.",
            v3,
            2,
            2,
            [
                f"V(1) = {number_text(_run(v0, r, d, 1), 2)}, V(2) = {number_text(_run(v0, r, d, 2), 2)}, V(3) = {number_text(v3, 2)}."
            ],
            _slips_value(None, v0, p, r, d, 3),
        ),
        number_part(
            "How much of the balance at the end of the third year is interest earned?",
            interest,
            2,
            2,
            [f"Interest = {number_text(v3, 2)} - {plain(v0)} - 3 x {plain(d)} = {number_text(interest, 2)}."],
            [
                (v3 - v0, "r3j-recurrence-total-change-given"),
                (v3 - 3 * d, "r3j-recurrence-start-value-left-in"),
                (v0 * (r - 1) * 3, "r3j-recurrence-interest-on-start-only"),
                (v3 - v0 - 2 * d, "r3j-recurrence-one-amount-short"),
            ],
            follow_through=f"part_b - {plain(v0)} - 3*{plain(d)}",
        ),
        number_part(
            f"After how many years will the balance first reach {dollars(target)}?",
            Fraction(n_target),
            0,
            2,
            [
                f"V({n_target - 1}) = {number_text(lo, 2)} is below {dollars(target)}; V({n_target}) = {number_text(hi, 2)} is not."
            ],
            [
                (n_target - 1, "r3j-recurrence-one-step-short"),
                (n_target + 1, "r3j-recurrence-one-step-too-many"),
                (n_target + 3, "r3j-recurrence-constant-change-assumed"),
            ],
            sensible_range=(1, 40),
        ),
    ]
    return dict(text=text, parts=parts)
