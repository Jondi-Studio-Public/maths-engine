"""Exam-tier source for `common-difference`: find an arithmetic sequence's common difference when it is not
given by two neighbouring numbers.

Every question is built BACKWARDS from a chosen first term and common difference, so the key is exact:

  expressions -- three linear expressions in an unknown `x` are said to be consecutive terms. The learner
                 must set up `2 * middle = first + third`, solve for `x`, and only then take a difference.
  partial sums -- the sums of the first `p` and the first `q` terms are given; two unknowns (first term and
                 difference) come out of a pair of simultaneous equations.
  term relations -- the sum of two terms and the value of a third are given; again two unknowns, and the
                 learner chooses how to eliminate the first term.

All three need a method chosen by the learner. None is a bigger range of the hard tier's "two terms, divide by
the number of steps" question.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, check_distinct, generator, join_terms, pick_distractors, rational_text


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _signed(rng: random.Random, lo: int, hi: int) -> int:
    """A non-zero integer in `[-hi, -lo] + [lo, hi]`."""
    return rng.choice([-1, 1]) * rng.randint(lo, hi)


def _term(a1: int, d: int, n: int) -> int:
    return a1 + (n - 1) * d


def _finish(answer: int, text: str, working: list[str], cands: list) -> dict:
    answer_text = str(answer)
    texts = [rational_text(Fraction(c)) for c in cands]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, texts, tolerance=GRADER_TOLERANCE),
    )


def _expressions(rng: random.Random) -> dict:
    """Three linear expressions in x that are consecutive terms for exactly one value of x."""
    for _ in range(200):
        x0 = _signed(rng, 1, 8)
        d = _signed(rng, 2, 9)
        a1 = rng.randint(-12, 12)
        coefs = [_signed(rng, 1, 4) for _ in range(3)]
        if coefs[0] - 2 * coefs[1] + coefs[2] == 0:
            continue  # the equation in x would be degenerate
        values = [a1, a1 + d, a1 + 2 * d]
        consts = [v - c * x0 for v, c in zip(values, coefs)]
        if max(abs(k) for k in consts) > 45 or d == x0:
            continue
        break
    else:
        check(False, "no usable expression triple")
    exprs = [join_terms([(c, "x"), (k, "")]) for c, k in zip(coefs, consts)]
    text = (
        f"The expressions {exprs[0]}, {exprs[1]} and {exprs[2]} are three consecutive terms of an arithmetic "
        "sequence, for one value of x. Find the common difference."
    )
    # Independent check: solve 2 * middle = first + third for x, then subtract.
    num = consts[0] + consts[2] - 2 * consts[1]
    den = 2 * coefs[1] - coefs[0] - coefs[2]
    check(den != 0 and Fraction(num, den) == x0, "expression triple does not solve to the chosen x")
    check(coefs[1] * x0 + consts[1] - (coefs[0] * x0 + consts[0]) == d)
    working = [
        f"Consecutive terms: 2({exprs[1]}) = ({exprs[0]}) + ({exprs[2]}), so x = {x0}.",
        f"The terms are {values[0]}, {values[1]}, {values[2]}; the common difference is {d}.",
    ]
    cands = [-d, x0, values[0], 2 * d]
    return _finish(d, text, working, cands)


def _partial_sums(rng: random.Random) -> dict:
    a1 = rng.randint(-9, 9)
    d = _signed(rng, 1, 8)
    p, q = sorted(rng.sample(range(3, 10), 2))
    s_p = p * a1 + p * (p - 1) // 2 * d
    s_q = q * a1 + q * (q - 1) // 2 * d
    text = (
        f"An arithmetic sequence has a sum of {s_p} for its first {p} terms and a sum of {s_q} for its first "
        f"{q} terms. Find the common difference."
    )
    # Independent check: the two sums, solved as simultaneous equations by elimination.
    #   S_n = n * a1 + n(n-1)/2 * d
    det = Fraction(p * q * (q - p), 2)
    solved_d = Fraction(p * s_q - q * s_p, 1) / det
    check(solved_d == d, "partial-sum system does not return the chosen common difference")
    working = [
        f"{p}a + {p * (p - 1) // 2}d = {s_p} and {q}a + {q * (q - 1) // 2}d = {s_q}.",
        f"Eliminate a: d = {d}.",
    ]
    mean_gap = (Fraction(s_q, q) - Fraction(s_p, p)) / (q - p)
    cands = [a1, -d, mean_gap, Fraction(s_q - s_p, q - p)]
    return _finish(d, text, working, cands)


def _term_relation(rng: random.Random) -> dict:
    for _ in range(200):
        a1 = rng.randint(-9, 9)
        d = _signed(rng, 1, 8)
        p, q = sorted(rng.sample(range(2, 10), 2))
        r = rng.randint(2, 9)
        if p + q - 2 * r == 0:
            continue
        break
    else:
        check(False, "no usable term relation")
    sum_pq = _term(a1, d, p) + _term(a1, d, q)
    t_r = _term(a1, d, r)
    text = (
        f"In an arithmetic sequence, the sum of the {_ordinal(p)} and {_ordinal(q)} terms is {sum_pq} and the "
        f"{_ordinal(r)} term is {t_r}. Find the common difference."
    )
    # Independent check: eliminate the first term between the two relations.
    #   t_p + t_q = 2a + (p+q-2)d ;  t_r = a + (r-1)d
    solved_d = Fraction(sum_pq - 2 * t_r, p + q - 2 * r)
    check(solved_d == d, "term relation does not return the chosen common difference")
    working = [
        f"2a + {p + q - 2}d = {sum_pq} and a + {r - 1}d = {t_r}.",
        f"Eliminate a: {p + q - 2 * r}d = {sum_pq - 2 * t_r}, so d = {d}.",
    ]
    cands = [a1, -d, t_r, sum_pq - 2 * t_r]
    return _finish(d, text, working, cands)


_BUILDERS = {"expressions": _expressions, "partial_sums": _partial_sums, "term_relation": _term_relation}


@generator(
    "exd_x_f0_common_difference",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_f0_common_difference(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "this topic only builds the exam tier")
    kind = rng.choice(sorted(_BUILDERS))
    out = _BUILDERS[kind](rng)
    check_distinct(out["answer"], *_numeric(out["distractors"]))
    return out


def _numeric(texts: list[str]) -> list[Fraction]:
    return [Fraction(t) for t in texts]
