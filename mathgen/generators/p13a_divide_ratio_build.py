"""Generates `p13a_divide_ratio` questions for `divide-quantity-in-ratio` (ACMEM069): divide a supplied quantity
into two shares in a supplied whole-number ratio.

    values = extract_values(_strip_comma_groups(given)) if given else []
    correct = len(values) == len(expected_values) and all(
        v == e or abs(float(v) - float(e)) <= _PAIR_TOLERANCE
        for v, e in zip(values, expected_values)
    )

That is a POSITIONAL, order-sensitive comparison (`zip(values, expected_values)`
in the order given) -- a swapped-order pair is a genuinely different (and
wrong) answer under this grader, never accepted either way. This is why the
"swap ratio term" misconception below is free to reuse the exact reversed pair
at every tier except hard (where the worked example calls for a
different construction): a reversed pair is already rejected by the real
grader, so it is a valid, exercisable distractor here (contrast the labelled
``x = ..., y = ...`` shape `_labelled_pair_verdict` exists for, where order is
irrelevant and a reversed pair would NOT be a valid distractor).

Four tiers, each a genuinely different DECISION about which quantity is
unknown -- never a bigger number standing in for a harder question:

- easy: total and the ratio p:q are both given directly. Compute both shares.
- medium: the DIFFERENCE between the two shares and the ratio are given
  instead of the total. The difference maps to |p-q| parts, not the ratio's
  sum -- a genuinely different divisor from easy's p+q.
- hard: ONE share (the first, `share_a`) and the ratio are given -- find the
  other share, reconstructing the per-part unit from a single known part
  (`k = share_a / p`) rather than from a stated total or difference. The
  implied total is shown only as a working-step self-check, never part of
  the graded pair (the graded pair still echoes the given `share_a` as its
  first component, to keep the fixed-order convention identical across every
  tier).
- exam: the ratio itself is given in NON-integer form (a pair of one-decimal-
  place decimals, or a pair of unit fractions) and the total is given -- the
  learner must first reduce the stated ratio to a whole-number ratio (clear
  the decimal places or common-denominator the fractions) before dividing.
  This is a genuine reasoning escalation over hard: an extra symbolic
  clearing step before the same total-based division easy already teaches,
  not merely bigger numbers.

Construct-to-target throughout: `p`, `q` (2-9, coprime, `p != q`) and an
integer multiplier `k` are drawn FIRST, so `share_a = k*p`, `share_b = k*q`
and every stated quantity (total, difference, one share, or the reduced
ratio + total) is computed from them -- never solve-then-round. Every tier's
independent check re-derives the two shares from the STATED quantity using
that tier's own division formula and then confirms the result via
CROSS-MULTIPLICATION (`share_a * q == share_b * p`, plus the stated-quantity
identity), a genuinely different algebraic route from the direct `k*p`/`k*q`
construction it is checked against.
"""

from __future__ import annotations

import math

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    check_equivalent,
    phrase,
    pick_distractors,
    generator,
)

_P_LO, _P_HI = 2, 9
_Q_LO, _Q_HI = 2, 9
_K_LO, _K_HI = 2, 9
_RETRY_LIMIT = 100

_EASY_PATTERNS = (
    "Divide {total} in the ratio {p}:{q}.",
    "Split {total} into two shares in the ratio {p}:{q}.",
    "Share {total} between two people in the ratio {p}:{q}.",
)
_MEDIUM_PATTERNS = (
    "Two shares are in the ratio {p}:{q}. Their difference is {diff}. Find both shares.",
    "The difference between two shares in the ratio {p}:{q} is {diff}. Find both shares.",
    "In the ratio {p}:{q}, the difference between the two shares is {diff}. Find both shares.",
)
_HARD_PATTERNS = (
    "Two shares are divided in the ratio {p}:{q}. The first share is {share_a}. Find both shares.",
    "A quantity is divided in the ratio {p}:{q}. The share matching the first term is {share_a}. Find both shares.",
    "In the ratio {p}:{q}, the share for the first term is {share_a}. Find both shares.",
)
_EXAM_DECIMAL_PATTERNS = (
    "Divide {total} in the ratio {p_disp}:{q_disp}.",
    "Share {total} in the ratio {p_disp}:{q_disp}.",
)
_EXAM_FRACTION_PATTERNS = (
    "Divide {total} in the ratio {p_disp}:{q_disp}.",
    "Share {total} in the ratio {p_disp}:{q_disp}.",
)


def _draw_pq(rng):
    """(p, q): distinct, coprime, both in [2, 9]. Coprime keeps the stated
    ratio from trivially reducing further in a way that would confuse the
    tier design (a ratio the question ALREADY states in lowest terms)."""
    for _ in range(_RETRY_LIMIT):
        p = rng.randint(_P_LO, _P_HI)
        q = rng.randint(_Q_LO, _Q_HI)
        if p != q and math.gcd(p, q) == 1:
            return p, q
    raise RuntimeError("could not draw a distinct coprime (p, q)")


def _cross_multiplication_check(share_a: int, share_b: int, p: int, q: int) -> None:
    """The one independent check shared by every tier: a genuinely different
    algebraic route from `k*p`/`k*q` construction -- the ratio IDENTITY
    `share_a/p == share_b/q`, cross-multiplied to stay in integers."""
    check(share_a * q == share_b * p, "cross-multiplication ratio identity failed")


# --- easy: total + ratio given -----------------------------------------


def _build_easy(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq(rng)
        k = rng.randint(_K_LO, _K_HI)
        total = k * (p + q)
        if total % 2 != 0:
            continue  # m1 (equal split) needs an exact integer half
        share_a, share_b = k * p, k * q

        # Independent derivation FROM the stated total (a different route
        # from the k*p/k*q construction above): divide by total parts.
        check(total % (p + q) == 0, "total is not an exact multiple of p + q")
        derived_a = total * p // (p + q)
        derived_b = total * q // (p + q)
        check_equivalent(
            (derived_a, derived_b), (share_a, share_b), "easy: division-from-total disagrees with construction"
        )
        _cross_multiplication_check(share_a, share_b, p, q)
        check(share_a + share_b == total, "shares do not sum to the stated total")

        m1 = (total // 2, total // 2)  # ignores the ratio entirely
        m2 = (share_b, share_a)  # swapped ratio-term roles
        u = total // (p + q + 1)
        m3 = (u * p, u * q)  # off-by-one denominator

        answer_tuple = (share_a, share_b)
        try:
            check_distinct(answer_tuple, m1, m2, m3, msg="an easy-tier distractor equals the answer")
            check_pairwise_distinct(m1, m2, m3, msg="two easy-tier distractors coincide")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor easy question")

    text = phrase(rng, _EASY_PATTERNS, total=total, p=p, q=q)
    ans_text = f"({share_a}, {share_b})"
    working = [
        f"Total parts: {p} + {q} = {p + q}.",
        f"Value of one part: {total} / {p + q} = {total // (p + q)}.",
        f"First share: {p} x {total // (p + q)} = {share_a}. Second share: {q} x {total // (p + q)} = {share_b}.",
    ]
    candidates = [f"({m1[0]}, {m1[1]})", f"({m2[0]}, {m2[1]})", f"({m3[0]}, {m3[1]})"]
    return dict(
        text=text,
        answer={"share_p": share_a, "share_q": share_b},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


# --- medium: difference + ratio given -----------------------------------


def _build_medium(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq(rng)
        k = rng.randint(_K_LO, _K_HI)
        dp = abs(p - q)
        diff = k * dp
        total = k * (p + q)
        if total % 2 != 0:
            continue
        share_a, share_b = k * p, k * q

        check(diff % dp == 0, "difference is not an exact multiple of |p - q|")
        k_solved = diff // dp
        derived_a, derived_b = k_solved * p, k_solved * q
        check_equivalent(
            (derived_a, derived_b), (share_a, share_b), "medium: division-from-difference disagrees with construction"
        )
        _cross_multiplication_check(share_a, share_b, p, q)
        check(abs(share_a - share_b) == diff, "shares do not produce the stated difference")

        m1 = (total // 2, total // 2)  # ignores the ratio entirely
        m2 = (share_b, share_a)  # swapped ratio-term roles
        k_wrong = diff // (p + q)  # off-by-one denominator: p+q, not |p-q|
        m3 = (k_wrong * p, k_wrong * q)

        answer_tuple = (share_a, share_b)
        try:
            check_distinct(answer_tuple, m1, m2, m3, msg="a medium-tier distractor equals the answer")
            check_pairwise_distinct(m1, m2, m3, msg="two medium-tier distractors coincide")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor medium question")

    text = phrase(rng, _MEDIUM_PATTERNS, p=p, q=q, diff=diff)
    ans_text = f"({share_a}, {share_b})"
    working = [
        f"Difference in parts: |{p} - {q}| = {dp}.",
        f"Value of one part: {diff} / {dp} = {k_solved}.",
        f"First share: {p} x {k_solved} = {share_a}. Second share: {q} x {k_solved} = {share_b}.",
    ]
    candidates = [f"({m1[0]}, {m1[1]})", f"({m2[0]}, {m2[1]})", f"({m3[0]}, {m3[1]})"]
    return dict(
        text=text,
        answer={"share_p": share_a, "share_q": share_b},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


# --- hard: one share + ratio given ---------------------------------------


def _build_hard(rng):
    for _ in range(_RETRY_LIMIT):
        p, q = _draw_pq(rng)
        k = rng.randint(_K_LO, _K_HI)
        total = k * (p + q)
        if total % 2 != 0:
            continue
        share_a, share_b = k * p, k * q

        check(share_a % p == 0, "share_a is not an exact multiple of p")
        k_solved = share_a // p
        derived_b = k_solved * q
        check_equivalent(derived_b, share_b, "hard: division-from-one-share disagrees with construction")
        _cross_multiplication_check(share_a, share_b, p, q)
        total_reconstructed = share_a + share_b  # shown only as a working-step check

        m1 = (share_a, share_a)  # solves share_b = k*p instead of k*q
        m2 = (total // 2, total // 2)  # ignores the ratio entirely
        k_wrong = share_a // (p + 1)  # off-by-one denominator on p
        m3 = (share_a, k_wrong * q)

        answer_tuple = (share_a, share_b)
        try:
            check_distinct(answer_tuple, m1, m2, m3, msg="a hard-tier distractor equals the answer")
            check_pairwise_distinct(m1, m2, m3, msg="two hard-tier distractors coincide")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor hard question")

    text = phrase(rng, _HARD_PATTERNS, p=p, q=q, share_a=share_a)
    ans_text = f"({share_a}, {share_b})"
    working = [
        f"One part: {share_a} / {p} = {k_solved}.",
        f"Second share: {q} x {k_solved} = {share_b}.",
        f"Check (implied total, not the graded answer): {share_a} + {share_b} = {total_reconstructed}.",
    ]
    candidates = [f"({m1[0]}, {m1[1]})", f"({m2[0]}, {m2[1]})", f"({m3[0]}, {m3[1]})"]
    return dict(
        text=text,
        answer={"share_p": share_a, "share_q": share_b},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


# --- exam: total given, ratio stated in non-integer form -----------------


def _draw_decimal_ratio(rng):
    """A one-decimal-place ratio (e.g. "0.4:0.6") that reduces to a
    whole-number ratio by multiplying both sides by 10; the digits ARE the
    reduced whole-number ratio, so `p_whole`/`q_whole` are read straight off
    `da`/`db`."""
    for _ in range(_RETRY_LIMIT):
        da = rng.randint(1, 9)
        db = rng.randint(1, 9)
        if da != db:
            return da, db, f"0.{da}", f"0.{db}"
    raise RuntimeError("could not draw a distinct decimal ratio")


def _draw_fraction_ratio(rng):
    """A unit-fraction ratio (e.g. "1/2:1/3") that reduces to a whole-number
    ratio by clearing to the common denominator L = lcm(d1, d2):
    `p_whole = L/d1`, `q_whole = L/d2`. Distinct denominators guarantee
    `p_whole != q_whole` (`L/d1 == L/d2` would force `d1 == d2`)."""
    denominators = [2, 3, 4, 5, 6]
    for _ in range(_RETRY_LIMIT):
        d1 = rng.choice(denominators)
        d2 = rng.choice(denominators)
        if d1 == d2:
            continue
        lcm = math.lcm(d1, d2)
        p_whole, q_whole = lcm // d1, lcm // d2
        return p_whole, q_whole, f"1/{d1}", f"1/{d2}"
    raise RuntimeError("could not draw a distinct fraction ratio")


def _build_exam(rng):
    for _ in range(_RETRY_LIMIT):
        form = rng.choice(["decimal", "fraction"])
        if form == "decimal":
            p_whole, q_whole, p_disp, q_disp = _draw_decimal_ratio(rng)
            reduce_line = f"Clear the decimals (multiply both sides by 10): {p_disp}:{q_disp} = {p_whole}:{q_whole}."
        else:
            p_whole, q_whole, p_disp, q_disp = _draw_fraction_ratio(rng)
            reduce_line = f"Clear the fractions to a common denominator: {p_disp}:{q_disp} = {p_whole}:{q_whole}."
        k = rng.randint(_K_LO, _K_HI)
        total = k * (p_whole + q_whole)
        if total % 2 != 0:
            continue
        share_a, share_b = k * p_whole, k * q_whole

        check(total % (p_whole + q_whole) == 0, "total is not an exact multiple of the reduced parts")
        derived_a = total * p_whole // (p_whole + q_whole)
        derived_b = total * q_whole // (p_whole + q_whole)
        check_equivalent(
            (derived_a, derived_b), (share_a, share_b), "exam: division-from-total disagrees with construction"
        )
        _cross_multiplication_check(share_a, share_b, p_whole, q_whole)
        check(share_a + share_b == total, "shares do not sum to the stated total")

        m1 = (total // 2, total // 2)  # ignores the ratio entirely
        m2 = (share_b, share_a)  # swapped ratio-term roles
        u = total // (p_whole + q_whole + 1)
        m3 = (u * p_whole, u * q_whole)  # off-by-one denominator

        answer_tuple = (share_a, share_b)
        try:
            check_distinct(answer_tuple, m1, m2, m3, msg="an exam-tier distractor equals the answer")
            check_pairwise_distinct(m1, m2, m3, msg="two exam-tier distractors coincide")
        except Exception:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor exam question")

    patterns = _EXAM_DECIMAL_PATTERNS if form == "decimal" else _EXAM_FRACTION_PATTERNS
    text = phrase(rng, patterns, total=total, p_disp=p_disp, q_disp=q_disp)
    ans_text = f"({share_a}, {share_b})"
    working = [
        reduce_line,
        f"Total parts: {p_whole} + {q_whole} = {p_whole + q_whole}.",
        f"Value of one part: {total} / {p_whole + q_whole} = {total // (p_whole + q_whole)}.",
        f"First share: {p_whole} x {total // (p_whole + q_whole)} = {share_a}. "
        f"Second share: {q_whole} x {total // (p_whole + q_whole)} = {share_b}.",
    ]
    candidates = [f"({m1[0]}, {m1[1]})", f"({m2[0]}, {m2[1]})", f"({m3[0]}, {m3[1]})"]
    return dict(
        text=text,
        answer={"share_p": share_a, "share_q": share_b},
        answer_text=ans_text,
        working=working,
        distractors=pick_distractors(ans_text, candidates),
    )


_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    "p13a_divide_ratio",
    level=8,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13a_divide_ratio(rng, tier="easy"):
    """Divide a supplied quantity into two shares in a supplied whole-number
    ratio. `tier` selects one of the four genuinely different decisions (see
    module docstring): which quantity is unknown, and (for `exam`) whether
    the stated ratio itself must first be reduced from decimal/fraction
    form. Every tier returns the same `answer: pair` shape, an unlabelled,
    positionally-ordered `(first-term share, second-term share)`."""
    builder = _BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_BUILDERS)}")
    return builder(rng)
