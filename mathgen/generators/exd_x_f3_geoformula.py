"""Exam tier for the geometric nth-term formula t_n = r^(n-1) * t1.

The practice tiers give ``t1`` and either ``r``, ``n`` or a target term, and
substitute once. The exam tier gives two terms that are NOT the first, so the
learner must find ``r`` from ``t_q / t_p = r^(q-p)`` and ``t1`` from a
back-step before the formula can be used:

* ``find_term`` -- find another term ``t_m`` of the sequence.
* ``find_position`` -- find which term number equals a stated value.

The sign of the ratio is stated, because an even gap alone leaves it open.
Every number is an exact integer.
"""

from __future__ import annotations

import random

from ..core import check, generator, pick_distractors


def _setup(rng: random.Random):
    while True:
        r = rng.choice([-4, -3, -2, 2, 3, 4])
        t1 = rng.choice([v for v in range(-6, 7) if v != 0])
        p = rng.randint(2, 3)
        q = p + rng.randint(2, 3)
        m = rng.randint(1, 8)
        if m in (p, q):
            continue
        if max(abs(t1 * r ** (n - 1)) for n in (p, q, m)) > 20000:
            continue
        return r, t1, p, q, m


def _distinct_wrong(key: int, wrong: list) -> int:
    return len({w for w in wrong if w is not None and w != key})


def _find_term(rng: random.Random):
    while True:
        r, t1, p, q, m = _setup(rng)
        tp, tq = t1 * r ** (p - 1), t1 * r ** (q - 1)
        key = t1 * r ** (m - 1)
        wrong = [
            t1 * r**m,  # exponent n instead of n - 1
            r ** (m - 1),  # drops t1
            abs(key),  # drops the sign
            tp * (tq // tp) ** (m - p) if m >= p else None,  # uses t_q / t_p unrooted as r
        ]
        if _distinct_wrong(key, wrong) >= 3:
            break
    sign = "negative" if r < 0 else "positive"
    text = (
        f"In a geometric sequence with {sign} common ratio, t{p} = {tp} and t{q} = {tq}. "
        f"Use t_n = r^(n-1) × t1 to find t{m}."
    )
    # Independent: from t_p, scale by r^(m-p) directly (exact when m >= p, a division otherwise).
    via_tp = tp * r ** (m - p) if m >= p else tp // r ** (p - m)
    check(via_tp == key, "scaling from t_p disagrees with the closed form")
    gap = q - p
    working = [
        f"t{q}/t{p} = r^{gap} = {r**gap}, and r is {sign}, so r = {r}",
        f"t1 = t{p} / r^{p - 1} = {t1}",
        f"t{m} = {t1} × ({r})^{m - 1} = {key}",
    ]
    return key, text, wrong, working


def _find_position(rng: random.Random):
    r, t1, p, q, m = _setup(rng)
    tp, tq = t1 * r ** (p - 1), t1 * r ** (q - 1)
    sign = "negative" if r < 0 else "positive"
    target = t1 * r ** (m - 1)
    text = (
        f"In a geometric sequence with {sign} common ratio, t{p} = {tp} and t{q} = {tq}. "
        f"Which term of the sequence equals {target}?"
    )
    hits = [n for n in range(1, 40) if t1 * r ** (n - 1) == target]
    check(hits == [m], "target position is not unique")
    wrong = [m - 1, m + 1, m - p]  # exponent read as the position; one past; counted from the given term
    working = [
        f"r^{q - p} = {tq}/{tp} = {r ** (q - p)}, and r is {sign}, so r = {r}; t1 = {t1}",
        f"{t1} × ({r})^(n-1) = {target} gives (n - 1) = {m - 1}",
        f"n = {m}",
    ]
    return m, text, wrong, working


_FORMS = {"find_term": _find_term, "find_position": _find_position}


@generator("exd_x_f3_geoformula", level=11, presets={"exam": dict(tier="exam")})
def exd_x_f3_geoformula(rng: random.Random, tier: str = "exam", form: str | None = None) -> dict:
    """A term or position of a geometric sequence described by two non-initial terms."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    key, text, wrong, working = _FORMS[form or rng.choice(sorted(_FORMS))](rng)
    answer_text = str(key)
    cands = [str(w) for w in wrong if w is not None and w >= -(10**7)]
    distractors = pick_distractors(answer_text, cands, n=3)
    check(len(distractors) >= 2, "too few principled distractors")
    return dict(text=text, answer=key, answer_text=answer_text, working=working, distractors=distractors)
