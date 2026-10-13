"""Exam tier for generating a geometric sequence by recursion.

The practice tiers state the first term and the ratio and ask for the next
terms. The exam tier states neither:

* ``two_terms`` -- two non-adjacent terms ``t_p`` and ``t_q`` plus the sign of
  the ratio (an even gap alone leaves the sign open); the learner extracts
  ``r`` as a ``(q - p)``-th root, back-steps to ``t1`` and then generates.
* ``parametric`` -- the first three terms are ``k``, ``k + a`` and ``k + b``
  with ``k`` unknown; equating the two ratios gives ``k`` before anything can
  be generated.

The answer is always the four terms after ``t1``, in order.
"""

from __future__ import annotations

import random

from ..core import check, generator, join_terms, pick_distractors

_K = 4  # terms asked for


def _terms_after_first(t1: int, r: int) -> list[int]:
    out, cur = [], t1
    for _ in range(_K):
        cur *= r
        out.append(cur)
    return out


def _fmt(values) -> str:
    return ", ".join(str(v) for v in values)


def _two_terms(rng: random.Random):
    while True:
        r = rng.choice([-4, -3, -2, 2, 3, 4])
        t1 = rng.choice([v for v in range(-9, 10) if v != 0])
        p = rng.randint(2, 3)
        gap = rng.randint(2, 3)
        q = p + gap
        tp, tq = t1 * r ** (p - 1), t1 * r ** (q - 1)
        if abs(tq) > 4000 or abs(t1 * r**_K) > 5000:
            continue
        break
    sign = "negative" if r < 0 else "positive"
    text = (
        f"A geometric sequence has t{p} = {tp} and t{q} = {tq}, and its common ratio r is {sign} "
        f"(t(n+1) = r × t(n)). Write the next {_K} terms after t1, in order."
    )
    # Independent: find r by trial among the integers of the stated sign, then check both given terms.
    cands = [c for c in range(-20, 21) if c and (c < 0) == (r < 0) and tp * c**gap == tq]
    check(cands == [r], "ratio is not unique or disagrees")
    check(tp % (r ** (p - 1)) == 0 and tp // r ** (p - 1) == t1, "back-step disagrees")
    whole = tq // tp  # the full ratio between the two given terms, not rooted
    correct = _terms_after_first(t1, r)
    wrong = [
        [t1] + correct[:-1],  # includes t1
        correct[::-1],
        [t1 + r * n for n in range(1, _K + 1)],  # adds r instead of multiplying
        _terms_after_first(t1, whole) if abs(whole) ** _K * abs(t1) < 10**6 else None,  # ratio not rooted
    ]
    working = [
        f"t{q} = t{p} × r^{gap}, so r^{gap} = {tq}/{tp} = {r**gap}, and r is {sign}, so r = {r}",
        f"t1 = t{p} / r^{p - 1} = {t1}",
        f"The next {_K} terms: {_fmt(correct)}",
    ]
    return correct, text, wrong, working


def _parametric(rng: random.Random):
    while True:
        r = rng.choice([-3, -2, 2, 3, 4])
        k = rng.choice([v for v in range(-6, 7) if v != 0])
        a, b = (r - 1) * k, (r * r - 1) * k
        if abs(k * r**_K) > 5000 or a == 0 or b == 2 * a:
            continue
        break
    terms = [join_terms([(1, "k"), (c, "")]) for c in (0, a, b)]
    terms[0] = "k"
    text = (
        f"The first three terms of a geometric sequence are {terms[0]}, {terms[1]} and {terms[2]}, where k is a "
        f"non-zero constant (t(n+1) = r × t(n)). Write the next {_K} terms after t1, in order."
    )
    # Independent: (k + a)^2 = k (k + b)  =>  k = a^2 / (b - 2a).
    check(a * a % (b - 2 * a) == 0 and a * a // (b - 2 * a) == k, "equal-ratio equation disagrees on k")
    correct = _terms_after_first(k, r)
    wrong = [
        [k] + correct[:-1],
        correct[::-1],
        [k + r * n for n in range(1, _K + 1)],
        _terms_after_first(-k, r),  # sign slip when solving for k
        _terms_after_first(k, r + 1),  # off-by-one in the ratio
    ]
    ka, kb = join_terms([(1, "k"), (a, "")]), join_terms([(1, "k"), (b, "")])
    working = [
        f"Equal ratios: ({ka})/k = ({kb})/({ka}), so ({ka})² = k({kb}), giving k = {k}",
        f"The terms are {k}, {k * r}, {k * r * r}, so r = {r}",
        f"The next {_K} terms: {_fmt(correct)}",
    ]
    return correct, text, wrong, working


_FORMS = {"two_terms": _two_terms, "parametric": _parametric}


@generator("exd_x_f3_georec", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f3_georec(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """Generate the next terms of a geometric sequence that is described only indirectly."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    correct, text, wrong, working = _FORMS[form or rng.choice(sorted(_FORMS))](rng)
    answer_text = _fmt(correct)
    cands = [_fmt(w) for w in wrong if w is not None]
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) == 3, "too few principled distractors")
    return dict(text=text, answer=correct, answer_text=answer_text, working=working, distractors=distractors)
