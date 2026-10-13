"""Exam-tier source for `f10-factor-multiple-divisibility-algorithm`: a flagging rule with overlapping divisors.

The practice tiers trace a rule whose two divisors are coprime, so the numbers divisible by both are exactly the
multiples of their product. An exam question chooses divisors that share a factor, so the overlap is the
multiples of their lowest common multiple, and the rule says "a multiple of d1 but not of d2" or "a multiple of
exactly one of them". Part (a) counts the flagged numbers; part (b) turns the count into the chance that a
number picked at random from the range is flagged, following through from (a).

The count is built from closed forms and re-checked by running the rule over the whole range, a different route.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from sympy import Rational

from ..core import check, draw_until, generator, latex_answer, pick_distractors, rational_text

_ACTORS = ["A computer program", "A school timetable program", "A raffle machine", "A sorting program"]
_THINGS = ["ticket", "locker", "seat", "entry"]


@generator("exd_x_n3_divisibility_count_chance", level=8, presets={"exam": dict(tier="exam")})
def exd_x_n3_divisibility_count_chance(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: count the numbers a rule flags (divisors share a factor), then find the chance of a flagged one."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        g = r.randint(2, 4)
        a, b = r.sample(range(2, 7), 2)
        return g * a, g * b, r.randint(150, 999), r.choice(["but_not", "exactly_one"]), r.randrange(len(_ACTORS))

    def accept(t):
        d1, d2, rr, rule, _ = t
        lcm = math.lcm(d1, d2)
        return math.gcd(d1, d2) > 1 and d1 % d2 != 0 and d2 % d1 != 0 and lcm < rr // 2

    d1, d2, R, rule, actor = draw_until(rng, draw, accept, what="divisors that share a factor")
    lcm = math.lcm(d1, d2)
    n1, n2, nl = R // d1, R // d2, R // lcm
    if rule == "but_not":
        count = n1 - nl
        wrong = [n1, n1 - R // (d1 * d2), n1 + n2 - nl, nl, n1 - n2]
        condition = f"a multiple of {d1} but not a multiple of {d2}"
    else:
        count = n1 + n2 - 2 * nl
        wrong = [n1 + n2 - nl, n1 + n2, nl, n1 + n2 - 2 * (R // (d1 * d2))]
        condition = f"a multiple of {d1} or a multiple of {d2}, but not both"
    flagged = [
        n
        for n in range(1, R + 1)
        if ((n % d1 == 0) != (n % d2 == 0) if rule == "exactly_one" else (n % d1 == 0 and n % d2 != 0))
    ]
    check(len(flagged) == count, "closed form disagrees with the rule run over the range")

    a_text = str(count)
    a_dist = pick_distractors(a_text, [str(w) for w in wrong if w > 0], 3)
    prob = Fraction(count, R)
    b_text = rational_text(Rational(prob.numerator, prob.denominator))
    b_cands = []
    for w in [int(t) for t in a_dist] + [R - count]:
        f = Fraction(w, R)
        if f != prob and f < 1:
            b_cands.append(rational_text(Rational(f.numerator, f.denominator)))
    b_cands.append(rational_text(Rational(Fraction(count, R + 1).numerator, Fraction(count, R + 1).denominator)))
    b_dist = pick_distractors(b_text, b_cands, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    thing = rng.choice(_THINGS)
    stem = (
        f"{rng.choice(_ACTORS)} checks every {thing} number from 1 to {R} inclusive. "
        f"It flags a number when the number is {condition}."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many numbers does it flag?",
                answer_kind="number",
                answer=count,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"Multiples of {d1}: {n1}. Multiples of {d2}: {n2}. Multiples of both are the multiples of "
                    f"lcm({d1}, {d2}) = {lcm}: {nl}.",
                    f"Flagged: {count}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"One {thing} number from 1 to {R} is chosen at random. Find the probability that it is flagged, as a fraction in simplest form.",
                answer_kind="fraction",
                answer={"p": prob.numerator, "q": prob.denominator},
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, 1),
                follow_through=f"part_a/{R}",
                working=[f"{count}/{R} = {b_text}."],
                distractors=b_dist,
            ),
        ],
    )
