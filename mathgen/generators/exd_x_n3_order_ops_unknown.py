"""Exam-tier source for `order-of-operations`: evaluate a mixed expression, then solve for an unknown in a second one.

The practice tiers evaluate one expression. An end-of-year paper asks the learner to carry the value on:
part (a) evaluates an expression with a bracketed difference, a power, multiplication and division (so the order
matters at every step), and part (b) asks for the whole number `n` that makes a second bracketed expression
equal to that value. Part (b) follows through from (a), so a wrong (a) that is used correctly still earns (b).

Built backwards: the integers are drawn so every division is exact and `n` is a positive whole number. The
printed expression is re-evaluated with SymPy, a different route from the arithmetic that built it.
"""

from __future__ import annotations

import random

import sympy

from ..core import GRADER_TOLERANCE, check, draw_until, generator, latex_answer, pick_distractors

_MINUS = "−"


def _value(text: str) -> sympy.Rational:
    """Evaluate a printed expression independently of the code that built it."""
    clean = text.replace("×", "*").replace("÷", "/").replace(_MINUS, "-").replace("²", "**2")
    value = sympy.sympify(clean)
    check(value.is_Rational, "expression did not evaluate to a rational")
    return value


def _draw_expression(rng: random.Random):
    """Return (printed expression, value, wrong-order candidates as integers or None)."""
    shape = rng.randrange(3)
    if shape == 2:
        a = rng.randint(2, 9)
        b = rng.randint(2, 12)
        c = rng.randint(2, 9)
        d = rng.randint(1, 20)
        text = f"{a} \u00d7 ({b} + {c}\u00b2) {_MINUS} {d}"
        value = a * (b + c * c) - d
        wrong = [a * b + c * c - d, a * (b + c * 2) - d, a * (b + c * c - d)]
    elif shape == 1:
        c = rng.randint(6, 16)
        d = rng.randint(1, c - 2)
        e = rng.choice([2, 3, 4, 5, 6])
        b = e * rng.randint(1, 5)
        a = rng.randint(2, 30)
        text = f"{a} + {b} × ({c} {_MINUS} {d})² ÷ {e}"
        w = c - d
        value = a + b * w * w // e
        wrong = [
            (a + b) * w * w // e if ((a + b) * w * w) % e == 0 else None,  # left to right on + and x
            b * w * 2 // e + a if (b * w * 2) % e == 0 else None,  # bracket "squared" as doubled
            (a + b * w * w) // e if (a + b * w * w) % e == 0 else None,  # divides the whole sum
        ]
    else:
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        c = rng.randint(2, 9)
        e = rng.choice([2, 3, 4, 5, 6])
        d = e * rng.randint(1, 3)
        text = f"({a} + {b}) × {c} {_MINUS} {d}² ÷ {e}"
        value = (a + b) * c - d * d // e
        wrong = [
            ((a + b) * c - d * d) // e if ((a + b) * c - d * d) % e == 0 else None,  # divides the whole difference
            a + b * c - d * d // e,  # ignores the brackets
            (a + b) * c - d * 2 // e if (d * 2) % e == 0 else None,  # power read as doubling
        ]
    return text, value, [w for w in wrong if w is not None]


@generator("exd_x_n3_order_ops_unknown", level=6, presets={"exam": dict(tier="exam")})
def exd_x_n3_order_ops_unknown(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: evaluate E, then find the whole number n that makes a second expression equal to E."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        text, value, wrong = _draw_expression(r)
        p = r.randint(2, 9)
        q = r.randint(1, 9)
        rr = (-value) % p + p * r.randint(0, 3)
        return text, value, wrong, p, q, rr

    def accept(t):
        text, value, wrong, p, q, rr = t
        n = (value + rr) // p - q
        distinct = {w for w in wrong if w != value}
        return 3 <= value <= 400 and n >= 1 and rr >= 1 and len(distinct) >= 2

    text, value, wrong, p, q, rr = draw_until(rng, draw, accept, what="an expression and an unknown")
    n = (value + rr) // p - q
    check(_value(text) == value, "printed expression disagrees with its value")
    check(p * (n + q) - rr == value, "n does not satisfy the second expression")

    a_text = str(value)
    a_dist = pick_distractors(a_text, [str(w) for w in wrong], 3, tolerance=GRADER_TOLERANCE)
    b_cands = [
        (value - rr) // p - q if (value - rr) % p == 0 else None,  # moves the subtracted term with the wrong sign
        (value + rr) // p + q if (value + rr) % p == 0 else None,  # adds q instead of subtracting it
        value + rr - q,  # forgets to divide by the multiplier
        (value + rr - q) // p if (value + rr - q) % p == 0 else None,  # subtracts q before dividing
    ]
    b_dist = pick_distractors(
        str(n), [str(c) for c in b_cands if c is not None and c >= 0], 3, tolerance=GRADER_TOLERANCE
    )
    check(len(a_dist) >= 2 and len(b_dist) >= 2, "too few distractors")

    stem = rng.choice(["Let E = {t}.", "Consider the expression E = {t}.", "A student writes E = {t}."]).format(t=text)
    second = f"{p} × (n + {q}) {_MINUS} {rr}"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Work out the value of E.",
                answer_kind="number",
                answer=int(value),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=["Brackets first, then the power, then multiplication and division left to right, then +/-."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the whole number n for which {second} has the same value as E.",
                answer_kind="number",
                answer=int(n),
                answer_text=str(n),
                answer_latex=latex_answer(str(n)),
                marks=3,
                follow_through=f"(part_a + {rr})/{p} - {q}",
                working=[
                    f"{p} × (n + {q}) {_MINUS} {rr} = {value}, so {p} × (n + {q}) = {value + rr}.",
                    f"n + {q} = {(value + rr) // p}, so n = {n}.",
                ],
                distractors=b_dist,
            ),
        ],
    )
