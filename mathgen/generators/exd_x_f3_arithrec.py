"""Exam tier for generating an arithmetic sequence by recursion.

The practice tiers state the first term and the common difference, or (hard)
two terms from which both follow by a back-step. The exam tier gives only
conditions that must be solved together before any term can be generated:

* ``term_sum`` -- a sum of two terms and one further term, a 2 x 2 linear
  system in ``t1`` and ``d``.
* ``sum_first`` -- the sum of the first ``m`` terms and one further term.
* ``parametric`` -- the first three terms are expressions in an unknown ``k``;
  the constant-difference property gives ``k`` first.

The answer is always the four terms after ``t1``, in order.
"""

from __future__ import annotations

import random

from ..core import check, generator, pick_distractors
from .exd_x_f3_recdef import _parametric, _term, _term_sum

_K = 4
_ASK = f"Write the next {_K} terms after t1, in order."


def _sum_first(rng: random.Random):
    while True:
        d = rng.choice([v for v in range(-8, 9) if v != 0])
        t1 = rng.choice([v for v in range(-20, 21) if v != 0])
        m = rng.randint(3, 6)
        r = rng.randint(2, 9)
        tri = m * (m - 1) // 2
        det = m * (r - 1) - tri
        if det == 0:
            continue
        break
    s = m * t1 + tri * d
    tr = _term(t1, d, r)
    text = f"The sum of the first {m} terms of an arithmetic sequence is {s}, and t{r} = {tr}."
    # Independent: Cramer's rule on  m*a + tri*d = s ;  a + (r-1)*d = tr.
    d_num = m * tr - s
    a_num = s * (r - 1) - tri * tr
    check(d_num % det == 0 and a_num % det == 0, "system solution is not integral")
    check((a_num // det, d_num // det) == (t1, d), "Cramer's rule disagrees with construction")
    wrong = [(t1 + d, d)]
    working = [
        f"S{m} = {m}t1 + {tri}d = {s} and t{r} = t1 + {r - 1}d = {tr}",
        f"Eliminate t1: {m}({tr}) - {s} = {det}d, so d = {d}",
        f"t1 = {tr} - {r - 1}({d}) = {t1}",
    ]
    return t1, d, text, wrong, working


_FORMS = {"term_sum": _term_sum, "sum_first": _sum_first, "parametric": _parametric}


@generator("exd_x_f3_arithrec", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f3_arithrec(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """Generate terms of an arithmetic sequence described only by conditions to be solved."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    t1, d, text, _wrong_pairs, working = _FORMS[form or rng.choice(sorted(_FORMS))](rng)
    correct = [_term(t1, d, n) for n in range(2, _K + 2)]
    wrong = [
        [t1] + correct[:-1],  # includes t1
        correct[::-1],  # reversed
        [t1 - n * d for n in range(1, _K + 1)],  # sign slip on d
        [_term(t1, d, n) for n in range(3, _K + 3)],  # starts one place late
    ]
    answer_text = ", ".join(map(str, correct))
    distractors = pick_distractors(answer_text, [", ".join(map(str, w)) for w in wrong], n=3)
    check(len(distractors) == 3, "too few principled distractors")
    working = working + [f"The next {_K} terms: {answer_text}"]
    return dict(
        text=f"{text} {_ASK}",
        answer=correct,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )
