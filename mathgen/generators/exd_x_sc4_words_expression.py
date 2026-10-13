"""Exam tier for "translate words into expressions": a priced allowance, then a discount, then a value.

The practice tiers turn one phrase into one expression. Here a price has a fixed charge that covers an
allowance and a rate for every unit beyond it, so the learner must write ``fee + rate (n - allowance)`` and
expand it. A percentage discount on the whole bill is then written as a new expression using the first, and
the discounted expression is evaluated at a stated quantity. Parts (b) and (c) carry ``follow_through``: part
(b) from the learner's expression in (a), part (c) from the learner's expression in (b).

Built backwards from the fee, the rate, the allowance and the discount. The fee and rate are chosen so every
coefficient of the discounted expression is a whole number of dollars.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import (
    check,
    check_distinct,
    draw_until,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    symbolically_equal,
)
from ._exd_x_sc4_kit import num_part, txt

# (setup sentence, request noun for part (a), variable, unit word, allowance unit word, rate range, discount sentence)
_CONTEXTS = (
    dict(
        setup="A company hires out a concrete mixer. The delivery fee of ${F} includes the first {t} days of hire, "
        "and each extra day costs ${r}.",
        ask="Write an expression for the total cost, in dollars, of hiring the mixer for n days, where n > {t}.",
        var="n",
        rates=(20, 60),
        fees=(60, 200),
        discount="A loyalty discount of {k}% is taken off the total cost.",
        eval="Find the discounted cost, in dollars, of hiring the mixer for {v0} days.",
        span=(1, 10),
    ),
    dict(
        setup="A mobile plan costs ${F} a month and includes {t} GB of data. Each extra GB costs ${r}.",
        ask="Write an expression for the monthly bill, in dollars, when g GB are used, where g > {t}.",
        var="g",
        rates=(5, 15),
        fees=(40, 120),
        discount="A bundle discount of {k}% is taken off the whole bill.",
        eval="Find the discounted bill, in dollars, for a month in which {v0} GB are used.",
        span=(1, 12),
    ),
    dict(
        setup="A community hall costs ${F} to hire, which includes the first {t} hours. Each extra hour costs ${r}.",
        ask="Write an expression for the hire cost, in dollars, for h hours, where h > {t}.",
        var="h",
        rates=(20, 60),
        fees=(80, 240),
        discount="A discount of {k}% is taken off the hire cost for community groups.",
        eval="Find the discounted hire cost, in dollars, for an event lasting {v0} hours.",
        span=(1, 8),
    ),
)

_DISCOUNTS = (10, 20, 25, 50)


def _linear_text(expr, var: str) -> str:
    """A linear expression written ``an + b``, constant last."""
    poly = sympy.Poly(sympy.expand(expr), sympy.Symbol(var))
    coefs = poly.all_coeffs()
    check(len(coefs) == 2 and all(c.is_Integer for c in coefs), "expected a linear expression with whole coefficients")
    return join_terms([(int(coefs[0]), var), (int(coefs[1]), "")])


@generator("exd_x_sc4_words_expression", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc4_words_expression(rng: random.Random, tier: str = "exam") -> dict:
    """Write a priced-allowance expression, discount it, then evaluate it. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)
    var = ctx["var"]
    x = sympy.Symbol(var)

    def draw(r):
        return (
            r.randint(*ctx["fees"]),
            r.randint(*ctx["rates"]),
            r.randint(2, 4),
            r.choice(_DISCOUNTS),
        )

    def fine(values):
        fee, rate, allowance, k = values
        keep = 100 - k
        return fee % 5 == 0 and (fee * keep) % 100 == 0 and (rate * keep) % 100 == 0 and fee > allowance * rate

    fee, rate, allowance, k = draw_until(rng, draw, fine, what="a fee, rate and discount on whole dollars")
    v0 = allowance + rng.randint(*ctx["span"])
    keep = Fraction(100 - k, 100)

    base = fee + rate * (x - allowance)
    discounted = sympy.Rational(100 - k, 100) * base
    base_text = _linear_text(base, var)
    disc_text = _linear_text(discounted, var)
    key_value = discounted.subs(x, v0)
    check(key_value.is_Integer, "the value must be a whole number of dollars")

    # (a) the allowance is ignored, subtracted without the rate, the fee left out, the fee multiplied by the extra units.
    a_wrong = [
        ("ignored-included-allowance", fee + rate * x),
        ("allowance-not-multiplied-by-rate", fee + rate * x - allowance),
        ("left-out-fixed-fee", rate * (x - allowance)),
        ("fee-multiplied-by-extra-units", (fee + rate) * (x - allowance)),
    ]
    # (b) the discount kept instead of taken off, taken off as dollars, or taken off only the per-unit part.
    d = sympy.Rational(k, 100)
    b_wrong = [
        ("kept-discount-not-balance", d * base),
        ("percent-as-absolute", base - k),
        ("discount-on-variable-part-only", fee + (1 - d) * rate * (x - allowance)),
    ]
    # A candidate can coincide with the key for particular numbers (a 50% discount kept is a 50% discount taken).
    a_wrong = [(mid, e) for mid, e in a_wrong if not symbolically_equal(base, e)]
    b_wrong = [(mid, e) for mid, e in b_wrong if not symbolically_equal(discounted, e)]
    for _, e in a_wrong:
        check_distinct(base, e, msg="an (a) distractor equals the key")
    for _, e in b_wrong:
        check_distinct(discounted, e, msg="a (b) distractor equals the key")

    def tagged(key_text, wrong):
        options = []
        for mid, e in wrong:
            e = sympy.expand(e)
            if not e.is_polynomial(x) or sympy.Poly(e, x).degree() != 1:
                continue
            if not all(c.is_Integer for c in sympy.Poly(e, x).all_coeffs()):
                continue
            options.append(Distractor(_linear_text(e, var), mid, str(e)))
        return pick_distractors(key_text, options, 3)

    a_dist = tagged(base_text, a_wrong)
    b_dist = tagged(disc_text, b_wrong)
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    c_cands = []
    for mid, e in [*b_wrong, ("ignored-discount", base)]:
        value = sympy.expand(e).subs(x, v0)
        c_cands.append((Fraction(int(value.p), int(value.q)), mid))
    c_part = num_part(
        ctx["eval"].format(v0=v0),
        Fraction(int(key_value)),
        0,
        1,
        [f"{disc_text} with {var} = {v0} gives ${txt(Fraction(int(key_value)), 0)}."],
        c_cands,
        follow_through=f"Subs(part_b, ({var},), ({v0},))",
    )
    c_part["marks"] = 1
    c_part.pop("sensible_range", None)

    stem = ctx["setup"].format(F=fee, r=rate, t=allowance)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=ctx["ask"].format(t=allowance) + " Expand your expression.",
                answer_kind="expression",
                answer=str(sympy.expand(base)),
                answer_text=base_text,
                answer_latex=latex_answer(sympy.expand(base)),
                marks=2,
                working=[
                    f"The first {allowance} units are covered by the fee, so {allowance} are not charged extra.",
                    f"Cost = {fee} + {rate}({var} - {allowance}) = {base_text}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=ctx["discount"].format(k=k) + " Hence write an expression for the discounted cost, in dollars. "
                "Expand your expression.",
                answer_kind="expression",
                answer=str(sympy.expand(discounted)),
                answer_text=disc_text,
                answer_latex=latex_answer(sympy.expand(discounted)),
                marks=2,
                working=[f"Paying {100 - k}% of the cost: {keep} × ({base_text}) = {disc_text}."],
                distractors=b_dist,
                follow_through=f"{keep}*part_a",
            ),
            c_part,
        ],
    )
