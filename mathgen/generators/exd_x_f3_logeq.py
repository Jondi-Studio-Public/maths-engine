"""Exam tier for solving equations involving indices using logarithms.

The practice tiers solve ``a^x = b`` and its coefficient and linear-exponent
variants: take one logarithm and divide. The exam tier asks the learner to
choose the method:

* ``two_bases`` -- unlike bases on each side, ``p^(x + a) = q^(x + b)``; the
  unknown appears in both exponents and has to be collected before it can
  be isolated.
* ``hidden_quadratic`` -- ``p^(2x) + c * p^x - m = 0``; factorise as a
  quadratic in ``p^x``, discard the impossible factor (``p^x`` is never
  negative), then take a logarithm.

Every answer is typed to 3 decimal places and is rejected as ambiguous
unless it clears a rounding boundary by a safe margin.
"""

from __future__ import annotations

import math
import random
import re

from ..core import (
    check,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
    rounds_unambiguously,
    stated_precision_tolerance,
)

_PRECISION = "Give your answer to 3 decimal places."
_BASES = [2, 3, 5, 6, 7]


def _well_separated(x: float, wrong: list[float]) -> bool:
    """At least three wrong values, each clear of the key and of each other by more than the grader allows."""
    vals = sorted({round(w, 3) for w in wrong if math.isfinite(w) and abs(w - x) > 0.01})
    return (
        len(vals) == len({round(w, 3) for w in wrong})
        and len(vals) >= 3
        and all(b - a > 0.01 for a, b in zip(vals, vals[1:]))
    )


def _is_power_of(n: int, base: int) -> bool:
    while n % base == 0:
        n //= base
    return n == 1


def _two_bases(rng: random.Random):
    while True:
        p, q = rng.sample([2, 3, 5, 7], 2)
        a = rng.randint(1, 6)
        b = rng.randint(-3, 5)
        if a == b:
            continue
        lp, lq = math.log(p), math.log(q)
        x = (b * lq - a * lp) / (lp - lq)
        if abs(x) > 40 or abs(x) < 0.05 or not rounds_unambiguously(x, 3):
            continue
        if not _well_separated(x, [(b - a) / (lp - lq), -x, (b * lq - a * lp) / (lp + lq), (b - a) * lq / (lp - lq)]):
            continue
        break
    lhs = f"{p}^(x + {a})"
    rhs = f"{q}^(x + {b})" if b > 0 else (f"{q}^(x - {-b})" if b < 0 else f"{q}^x")
    text = f"Solve {lhs} = {rhs} for x."
    # An independent route: both sides as exponentials must agree at the key.
    key = round(x, 3)
    check(abs((key + a) * lp - (key + b) * lq) < 0.01 * (lp + lq), "sides disagree at the key")
    wrong = [
        (b - a) / (lp - lq),  # takes the log of the bases only, leaving the shifts unlogged
        -x,  # sign slip when moving the x terms across
        (b * lq - a * lp) / (lp + lq),  # adds instead of subtracts the logs of the bases
        (b - a) * lq / (lp - lq),
    ]
    # join_terms writes "2ln 5"; a space before ln reads as written maths.
    numerator = re.sub(r"(\d)ln", r"\1 ln", join_terms([(b, f"ln {q}"), (-a, f"ln {p}")]))
    working = [
        f"Take logs: ({join_terms([(1, 'x'), (a, '')])}) ln {p} = ({join_terms([(1, 'x'), (b, '')])}) ln {q}",
        f"x(ln {p} - ln {q}) = {numerator}",
        f"x = ({numerator}) / (ln {p} - ln {q}) = {x:.3f}",
    ]
    return x, text, wrong, working


def _hidden_quadratic(rng: random.Random):
    while True:
        p = rng.choice([2, 3, 5])
        u = rng.randint(3, 40)
        v = rng.randint(2, 12)
        if _is_power_of(u, p) or _is_power_of(v, p) or u == v:
            continue
        x = math.log(u) / math.log(p)
        if not rounds_unambiguously(x, 3):
            continue
        if not _well_separated(x, [math.log(v) / math.log(p), x / 2, math.log(abs(v - u)) / math.log(p)]):
            continue
        break
    mid = v - u
    # p^(2x) + mid * p^x - u*v = 0  with p^x = u the only admissible root.
    terms = f"{p}^(2x)"
    terms += f" + {mid}({p}^x)" if mid > 0 else (f" - {-mid}({p}^x)" if mid < 0 else "")
    text = f"Solve {terms} - {u * v} = 0 for x."
    check(u * u + mid * u - u * v == 0, "u is not a root of the quadratic in p^x")
    wrong = [
        math.log(v) / math.log(p),  # keeps the inadmissible factor
        math.log(u) / (2 * math.log(p)),  # halves, as if the exponent were 2 on the whole of p^x
        math.log(abs(mid)) / math.log(p),  # takes the middle coefficient as the value of p^x
    ]
    working = [
        f"Let y = {p}^x: y^2{f' + {mid}y' if mid > 0 else (f' - {-mid}y' if mid < 0 else '')} - {u * v} = 0, so (y - {u})(y + {v}) = 0",
        f"y = {p}^x is positive, so y = {u} (y = -{v} is impossible)",
        f"x = ln {u} / ln {p} = {x:.3f}",
    ]
    return x, text, wrong, working


_FORMS = {"two_bases": _two_bases, "hidden_quadratic": _hidden_quadratic}


@generator("exd_x_f3_logeq", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f3_logeq(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """Solve an index equation that needs a method choice, to 3 decimal places."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    name = form or rng.choice(sorted(_FORMS))
    x, text, wrong, working = _FORMS[name](rng)
    answer_text = f"{x:.3f}"
    cands = [f"{w:.3f}" for w in wrong if math.isfinite(w)]
    distractors = pick_distractors(answer_text, cands, n=3, tolerance=stated_precision_tolerance(3) * 4)
    check(len(distractors) >= 2, "too few principled distractors")
    return dict(
        text=f"{text} {_PRECISION}",
        answer=round(x, 3),
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )
