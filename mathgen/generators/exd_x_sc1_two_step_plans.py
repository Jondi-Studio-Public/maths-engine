"""Exam source of `two-step-equations`: two charging plans compared through one shared quantity.

The practice tiers give a two-step equation to solve. Here the equations are built from a story and the second one
needs the answer to the first:

    (a) plan A charges a fixed fee f plus r per unit and the bill is T: find the number of units n   (3 marks)
    (b) plan B charges a fixed fee g plus an unknown price s per unit; for the same n units its bill
        is U: find s                                                                                    (3 marks, follow-through)

Both are two-step equations (a x + b = c): undo the fixed fee, then divide by the number the other quantity
multiplies. Every answer is a whole number, drawn first.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rounded_decimal_string

# Each theme: plausible ranges for the count, the per-unit price and the fixed fee, and the wording of both plans
_THEMES = [
    dict(
        n=(4, 26),
        rate=(5, 20),
        fee=(20, 80),
        a="Gym A charges a joining fee of ${f} plus ${r} for each week of membership.",
        a_ask="A member has paid ${T} in total.",
        b="Gym B charges a joining fee of ${g} plus the same amount for every week.",
        b_ask="A member who joins Gym B for the same number of weeks pays ${U} in total.",
        a_find="Write an equation and solve it to find how many weeks the member has been with Gym A.",
        b_find="Use your answer to (a). Write an equation and solve it to find Gym B's weekly charge, in dollars.",
    ),
    dict(
        n=(3, 25),
        rate=(2, 5),
        fee=(4, 12),
        a="Taxi company A charges a flag-fall of ${f} plus ${r} for each kilometre travelled.",
        a_ask="A trip across Perth costs ${T}.",
        b="Taxi company B charges a flag-fall of ${g} plus the same amount for every kilometre.",
        b_ask="The same trip with company B costs ${U}.",
        a_find="Write an equation and solve it to find how many kilometres the trip was.",
        b_find="Use your answer to (a). Write an equation and solve it to find company B's charge per kilometre, in dollars.",
    ),
    dict(
        n=(2, 8),
        rate=(15, 45),
        fee=(20, 60),
        a="A hire shop charges a delivery fee of ${f} plus ${r} for each hour a jumping castle is hired.",
        a_ask="A birthday party hires one for a total of ${T}.",
        b="A second shop charges a delivery fee of ${g} plus the same amount for every hour.",
        b_ask="Hiring from the second shop for the same number of hours costs ${U}.",
        a_find="Write an equation and solve it to find how many hours the castle was hired for.",
        b_find="Use your answer to (a). Write an equation and solve it to find the second shop's hourly charge, in dollars.",
    ),
    dict(
        n=(4, 20),
        rate=(30, 60),
        fee=(20, 80),
        a="A tutoring centre charges an enrolment fee of ${f} plus ${r} for each session.",
        a_ask="A student has paid ${T} in total.",
        b="A second centre charges an enrolment fee of ${g} plus the same amount for every session.",
        b_ask="A student who attends the same number of sessions there pays ${U} in total.",
        a_find="Write an equation and solve it to find how many sessions the student has attended.",
        b_find="Use your answer to (a). Write an equation and solve it to find the second centre's charge per session, in dollars.",
    ),
    dict(
        n=(2, 30),
        rate=(1, 6),
        fee=(5, 40),
        a="Mobile plan A has a set-up fee of ${f} plus ${r} for each gigabyte of data used.",
        a_ask="One customer's monthly bill comes to ${T}.",
        b="Mobile plan B has a set-up fee of ${g} plus the same amount for every gigabyte.",
        b_ask="With the same data use, a customer on plan B is billed ${U}.",
        a_find="Write an equation and solve it to find how many gigabytes the customer used.",
        b_find="Use your answer to (a). Write an equation and solve it to find plan B's price per gigabyte, in dollars.",
    ),
]


def _dist(value, tag):
    """A wrong answer written the way a learner with a calculator would: to at most 2 decimal places."""
    text = rounded_decimal_string(Fraction(value), 2)
    return Distractor(text, tag, Fraction(text))


def _nice(value: Fraction) -> bool:
    return 0 < value <= 2000


def _part(prompt, key, marks, working, candidates, follow_through=None):
    cands = [c for c in candidates if _nice(Fraction(c.value))]
    dists = pick_distractors(str(key), cands, 3)
    check(len(dists) >= 2, "too few distractors")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=str(key),
        answer_latex=latex_answer(str(key)),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=dists,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_sc1_two_step_plans", level=8, presets={"exam": dict(tier="exam")})
def exd_x_sc1_two_step_plans(rng, tier="exam"):
    theme = rng.choice(_THEMES)
    for _ in attempts(1000, "two plans with whole-number answers"):
        n = rng.randint(*theme["n"])
        r = rng.randint(*theme["rate"])
        s = rng.randint(*theme["rate"])
        f = rng.randint(*theme["fee"])
        g = rng.randint(*theme["fee"])
        if s == r or f == g:
            continue
        T, U = f + r * n, g + s * n
        if len({n, r, s, f, g, T, U}) < 7:
            continue
        break
    check((T - f) % r == 0 and (T - f) // r == n and (U - g) % n == 0 and (U - g) // n == s, "roots disagree")

    a_part = _part(
        theme["a_find"],
        n,
        3,
        [f"{r}n + {f} = {T}.", f"{r}n = {T} - {f} = {T - f}.", f"n = {T - f} ÷ {r} = {n}."],
        [
            _dist(Fraction(T, r), "ignored-constant-term"),
            _dist(Fraction(T + f, r), "sign-error-moving-term"),
            _dist(T - f, "forgot-to-divide-by-coefficient"),
            _dist(Fraction(T, r) - f, "divided-before-subtracting"),
            _dist((T - f) * r, "wrong-inverse-operation"),
        ],
    )
    b_part = _part(
        theme["b_find"],
        s,
        3,
        [f"{n}s + {g} = {U}.", f"{n}s = {U} - {g} = {U - g}.", f"s = {U - g} ÷ {n} = {s}."],
        [
            _dist(Fraction(U, n), "ignored-constant-term"),
            _dist(Fraction(U + g, n), "sign-error-moving-term"),
            _dist(U - g, "forgot-to-divide-by-coefficient"),
            _dist(Fraction(U, n) - g, "divided-before-subtracting"),
            _dist((U - g) * n, "wrong-inverse-operation"),
        ],
        follow_through=f"({U} - {g})/part_a",
    )
    stem = f"{theme['a'].format(f=f, r=r)} {theme['a_ask'].format(T=T)} {theme['b'].format(g=g)} {theme['b_ask'].format(U=U)}"
    return dict(text=stem, parts=[a_part, b_part])
