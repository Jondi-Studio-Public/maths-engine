"""Exam source for `senior-use-implication-converse-equivalence-negation-contrapositive`: when does each form fail?

The practice tiers ask the learner to pick the converse, test an equivalence or read a negation from a list. Here the
learner works with two statements P and Q about the integers 1 to N and must count the integers on which each
logical form is false, using the fact that "if P then Q" fails exactly when P is true and Q is false:

  (a) how many integers make "if P then Q" false;
  (b) how many make the converse, "if Q then P", false;
  (c) hence how many make "P if and only if Q" false (the two failing sets share no integer, so the counts add;
      follow-through from (a) and (b)).

Each statement is drawn from five kinds: "n is divisible by k", "n is a multiple of k", "n is greater than t",
"n is less than t" and "n is a perfect square". A draw is kept only when "both true" and each failing set hold at
least two integers, and the two failing counts differ.

Every key is counted by brute force over 1..N.
"""

from __future__ import annotations

import math
import random

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer


def _part(prompt, answer, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(answer),
        answer_text=str(int(answer)),
        answer_latex=latex_answer(str(int(answer))),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _three(candidates, key):
    seen, out = {key}, []
    fillers = [Distractor(str(key + d), None, key + d) for d in (1, -1, 2, -2, 3) if key + d >= 1]
    for c in [*candidates, *fillers]:
        if c.value in seen:
            continue
        seen.add(c.value)
        out.append(c)
        if len(out) == 3:
            break
    check(len(out) == 3, "fewer than three distractors")
    return out


def _statement(rng: random.Random, n_max: int):
    """One statement about n and its predicate: a divisor, a multiple, a lower or upper bound, or a perfect square."""
    kind = rng.choice(["divisible", "multiple", "greater", "less", "square"])
    if kind == "divisible":
        k = rng.randint(2, 12)
        return f"n is divisible by {k}", lambda n: n % k == 0
    if kind == "multiple":
        k = rng.randint(2, 9)
        return f"n is a multiple of {k}", lambda n: n % k == 0
    if kind == "greater":
        t = rng.randrange(n_max // 5, n_max * 3 // 5)
        return f"n is greater than {t}", lambda n: n > t
    if kind == "less":
        t = rng.randrange(n_max * 2 // 5, n_max * 4 // 5)
        return f"n is less than {t}", lambda n: n < t
    return "n is a perfect square", lambda n: math.isqrt(n) ** 2 == n


_STEMS = [
    "For each integer n from 1 to {top}, let P be the statement \u201c{p}\u201d and Q the statement \u201c{q}\u201d.",
    "Consider the integers n from 1 to {top}. Statement P is \u201c{p}\u201d and statement Q is \u201c{q}\u201d.",
]
_PART_A = [
    "For how many of these integers is the statement \u201cif P then Q\u201d false?",
    "How many of these integers make \u201cif P then Q\u201d false?",
]
_PART_B = [
    "For how many of these integers is the converse, \u201cif Q then P\u201d, false?",
    "How many of these integers make the converse, \u201cif Q then P\u201d, false?",
]


@generator("exd_x_sc7_implication_counts", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc7_implication_counts(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "statements whose three failure counts are positive and distinct"):
        n_max = rng.randrange(60, 301, 10)
        (p_text, p), (q_text, q) = _statement(rng, n_max), _statement(rng, n_max)
        if p_text == q_text:
            continue
        universe = range(1, n_max + 1)
        fail_pq = sum(1 for n in universe if p(n) and not q(n))
        fail_qp = sum(1 for n in universe if q(n) and not p(n))
        both = sum(1 for n in universe if p(n) and q(n))
        p_true = sum(1 for n in universe if p(n))
        q_true = sum(1 for n in universe if q(n))
        fail_iff = sum(1 for n in universe if p(n) != q(n))
        if min(fail_pq, fail_qp, both) >= 2 and fail_pq != fail_qp:
            break
    check(fail_iff == fail_pq + fail_qp, "the two failing sets overlap")
    stem = rng.choice(_STEMS).format(top=n_max, p=p_text, q=q_text)
    a_cands = [
        Distractor(str(p_true), "ignores-the-conclusion", p_true),
        Distractor(str(fail_qp), "swaps-hypothesis-and-conclusion", fail_qp),
        Distractor(str(both), "treats-implication-as-and", both),
        Distractor(str(n_max - fail_pq), "counts-true-cases-not-false", n_max - fail_pq),
    ]
    b_cands = [
        Distractor(str(q_true), "ignores-the-conclusion", q_true),
        Distractor(str(fail_pq), "swaps-hypothesis-and-conclusion", fail_pq),
        Distractor(str(both), "treats-implication-as-and", both),
        Distractor(str(n_max - fail_qp), "counts-true-cases-not-false", n_max - fail_qp),
    ]
    c_cands = [
        Distractor(str(fail_pq), "checks-only-one-direction", fail_pq),
        Distractor(str(n_max - fail_iff), "counts-true-cases-not-false", n_max - fail_iff),
        Distractor(str(fail_iff + both), "treats-both-true-as-failing", fail_iff + both),
        Distractor(str(fail_qp), "checks-only-one-direction", fail_qp),
        Distractor(str(fail_iff + 1), None, fail_iff + 1),
        Distractor(str(fail_iff - 1), None, fail_iff - 1),
    ]
    parts = [
        _part(
            "For how many of these integers is the statement “if P then Q” false?",
            fail_pq,
            2,
            _three(a_cands, fail_pq),
            [
                "“If P then Q” is false exactly when P is true and Q is false.",
                f"Counting those integers from 1 to {n_max} gives {fail_pq}.",
            ],
        ),
        _part(
            "For how many of these integers is the converse, “if Q then P”, false?",
            fail_qp,
            2,
            _three(b_cands, fail_qp),
            [
                "The converse is false exactly when Q is true and P is false.",
                f"Counting those integers gives {fail_qp}.",
            ],
        ),
        _part(
            "Hence find how many of these integers make “P if and only if Q” false.",
            fail_iff,
            1,
            _three(c_cands, fail_iff),
            [
                "“P if and only if Q” fails when either direction fails, and no integer fails both directions.",
                f"{fail_pq} + {fail_qp} = {fail_iff}.",
            ],
            follow_through="part_a + part_b",
        ),
    ]
    return dict(text=stem, parts=parts)
