"""Exam tier for stating the recursive definition of an arithmetic sequence.

The practice tiers show consecutive terms and ask the learner to read off the
first term and the common difference. The exam tier never shows the first
window: the learner has to recover ``t1`` and ``d`` from information about
the sequence, then state the recursive rule. Three question forms:

* ``two_terms`` -- two non-adjacent terms ``t_p`` and ``t_q`` (``q - p >= 2``).
* ``term_sum`` -- a sum of two terms and one further term, a 2 x 2 linear
  system in ``t1`` and ``d``.
* ``parametric`` -- the first three terms are given as expressions in an
  unknown ``k``; the learner finds ``k`` from the constant-difference
  property first, then reads off ``t1`` and ``d``.

Every question is built backwards from a chosen ``(t1, d)`` so the key is
exact. The key is the labelled pair ``first_term = a, d = b``.
"""

from __future__ import annotations

import random

from ..core import (
    GRADER_TOLERANCE,
    check,
    generator,
    join_terms,
    paren_negative,
    pick_distractors,
)

_ASK = "State its recursive definition: the first term, and the value of d in t(n+1) = t(n) + d."


def _term(t1: int, d: int, n: int) -> int:
    return t1 + (n - 1) * d


def _pair_text(a: int, d: int) -> str:
    return f"first_term = {a}, d = {d}"


def _two_terms(rng: random.Random):
    d = rng.choice([v for v in range(-9, 10) if v not in (0, 1, -1)])
    t1 = rng.choice([v for v in range(-25, 26) if v != 0])
    p = rng.randint(2, 4)
    q = p + rng.randint(2, 4)
    tp, tq = _term(t1, d, p), _term(t1, d, q)
    text = f"An arithmetic sequence has t{p} = {tp} and t{q} = {tq}."
    # An independent route: two-point difference, then back-step.
    check((tq - tp) % (q - p) == 0 and (tq - tp) // (q - p) == d, "two-point difference disagrees")
    check(tp - (p - 1) * d == t1, "back-step disagrees")
    whole = tq - tp  # the full gap, not divided by q - p
    wrong = [
        (t1, whole) if whole != d else None,  # whole gap taken as d, no division by q - p
        (tp - p * d, d),  # back-steps p places instead of p - 1
        (tp, d),  # reports the given term as the first term
        (t1 + d, d),
    ]
    working = [
        f"t{q} - t{p} = {q - p}d, so d = ({tq} - {paren_negative(tp)}) / {q - p} = {d}",
        f"t1 = t{p} - {p - 1}d = {tp} - {p - 1}({d}) = {t1}",
    ]
    return t1, d, text, wrong, working


def _term_sum(rng: random.Random):
    while True:
        d = rng.choice([v for v in range(-8, 9) if v != 0])
        t1 = rng.choice([v for v in range(-20, 21) if v != 0])
        p, q = sorted(rng.sample(range(2, 8), 2))
        r = rng.randint(2, 9)
        # sum equation: 2*t1 + (p+q-2)*d ; single term: t1 + (r-1)*d
        det = 2 * (r - 1) - (p + q - 2)
        if det == 0 or r in (p, q):
            continue
        break
    s = _term(t1, d, p) + _term(t1, d, q)
    tr = _term(t1, d, r)
    text = f"In an arithmetic sequence, t{p} + t{q} = {s} and t{r} = {tr}."
    # Independent solve by Cramer's rule on 2a + (p+q-2)d = s ; a + (r-1)d = tr.
    a_num = s * (r - 1) - (p + q - 2) * tr
    d_num = 2 * tr - s
    check(a_num % det == 0 and d_num % det == 0, "system solution is not integral")
    check((a_num // det, d_num // det) == (t1, d), "Cramer's rule disagrees with construction")
    wrong = [
        (t1 + d, d),  # reports t2 as the first term
        (t1, -d),  # sign slip on d
        (t1 - d, d),  # reports the term before the first
    ]
    working = [
        f"t{p} + t{q} = 2t1 + {p + q - 2}d = {s} and t{r} = t1 + {r - 1}d = {tr}",
        f"Eliminate t1: 2({tr}) - ({s}) = {det}d, so d = {d}",
        f"t1 = {tr} - {r - 1}({d}) = {t1}",
    ]
    return t1, d, text, wrong, working


def _parametric(rng: random.Random):
    while True:
        k = rng.choice([v for v in range(-6, 7) if v != 0])
        d = rng.choice([v for v in range(-7, 8) if v != 0])
        t1 = rng.choice([v for v in range(-15, 16) if v != 0])
        c = [rng.randint(1, 4) for _ in range(3)]
        coef = c[0] - 2 * c[1] + c[2]
        if coef == 0:
            continue
        vals = [_term(t1, d, n) for n in (1, 2, 3)]
        consts = [v - ci * k for v, ci in zip(vals, c)]
        if max(abs(e) for e in consts) > 12:
            continue
        break
    exprs = [join_terms([(ci, "k"), (e, "")]) for ci, e in zip(c, consts)]
    text = (
        f"The first three terms of an arithmetic sequence are {exprs[0]}, {exprs[1]} and {exprs[2]}, "
        "where k is a constant."
    )
    # Independent: the constant-difference condition 2*t2 = t1 + t3 gives k directly.
    lhs = 2 * c[1] - c[0] - c[2]
    rhs = consts[0] + consts[2] - 2 * consts[1]
    check(lhs != 0 and rhs % lhs == 0 and rhs // lhs == k, "constant-difference equation disagrees on k")
    wrong_vals = []
    for kw in (-k, k + 1):
        seq = [ci * kw + e for ci, e in zip(c, consts)]
        wrong_vals.append((seq[0], seq[1] - seq[0]))
    wrong = [
        wrong_vals[0],  # sign error when solving for k
        wrong_vals[1],  # off-by-one in k
        (k, d),  # reports k as the first term
        (t1, -d),
    ]
    working = [
        f"Equal differences: 2({exprs[1]}) = ({exprs[0]}) + ({exprs[2]}), so k = {k}",
        f"The terms are {vals[0]}, {vals[1]}, {vals[2]}",
        f"d = {vals[1]} - {paren_negative(vals[0])} = {d}",
    ]
    return t1, d, text, wrong, working


_FORMS = {"two_terms": _two_terms, "term_sum": _term_sum, "parametric": _parametric}


@generator(
    "exd_x_f3_recdef",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f3_recdef(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """State (first term, d) for an arithmetic sequence described indirectly."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    name = form or rng.choice(sorted(_FORMS))
    t1, d, text, wrong, working = _FORMS[name](rng)
    text = f"{text} {_ASK}"
    check(t1 != 0, "t1 must be non-zero")
    answer_text = _pair_text(t1, d)
    pairs = [(t1, d)] + [w for w in wrong if w is not None]
    # Drop candidates that equal the key or each other before they reach the learner.
    cands = [_pair_text(a, b) for a, b in pairs[1:]]
    distractors = pick_distractors(answer_text, cands, n=3, tolerance=GRADER_TOLERANCE)
    check(len(distractors) >= 2, "too few principled distractors")
    return dict(
        text=text,
        answer={"first_term": t1, "d": d},
        answer_text=answer_text,
        answer_latex=rf"\text{{first term}} = {t1},\; d = {d}",
        working=working,
        distractors=distractors,
    )
