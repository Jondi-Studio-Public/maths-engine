"""Estimation-methods cluster, both senior ``acara-senior-v8-4``, domain ``arithmetic``, ``answer: number``,
graded EXACTLY (an integer canonical, no decimal point).

Two generators:

- ``leading_digit_estimate`` (``senior-use-leading-digit-approximation-to-
  obtain-estimates-of``, ACMEM004) -- round EACH operand to its OWN leading
  digit (1 significant figure), using standard round-half-up, then compute.
  The method is always "leading-digit rounding" -- never named any other
  way, never a different target place. ``+``/``-``/``x`` only; division is
out of scope.

- ``apply_approximation_strategy`` (``senior-apply-approximation-strategies-
  for-calculations``, ACMEM010) -- round EACH operand to an EXPLICITLY
  STATED, NAMED strategy that is a DIFFERENT operation from leading-digit
  rounding: ``front_end`` (truncate to the leading digit's place) or
  ``round_to_place`` (round to a STATED place, ten or hundred). ``+``/``-``
  only; no multiplication, no "compatible numbers" (that strategy has no
  single prescribed procedure, so it cannot be graded to one exact value).

Both topics share one property, enforced by construction, not by hope: for
EVERY seed at EVERY tier, ``apply_approximation_strategy``'s own answer must
differ from what ``leading_digit_estimate``'s own rounding rule would give
for the SAME operands -- otherwise ACMEM010 degenerates into "ACMEM004 in
disguise".

Overlap prevention, by construction:

- ``front_end`` operands: every operand's digit immediately after the
  leading digit is forced into ``{5, 6, 7, 8, 9}`` via ``second_min=5/6``
  (ACMEM004's own easy tier, whose operands are 2-digit, forces ``{6,...,9}``
  specifically, since a literal ``5`` there is already the exact leading-
  digit tie and excluded by ``_draw_n_digit``'s own tie guard; ACMEM010's
  ``front_end`` operands are >= 3 digits, where the exact tie is a DIFFERENT,
  much rarer value -- second digit exactly 5 AND every digit after it exactly
  0 -- so ``{5,...,9}`` is achievable there and a literal second digit of 5
does appear, e.g. ``158``, ``555``, ``453``). Either
  way, every operand with a second digit >= 5 rounds UP under leading-digit
  rounding by exactly one unit of its own leading place, so truncating down
  (``front_end``) always differs from it.
- ``round_to_place`` operands: the stated place (``ten``/``hundred``) is
  always strictly finer than the operand's own leading-digit place --
  ``ten`` only for operands with >= 3 digits, ``hundred`` only for operands
  with >= 4 digits.
- Per-operand distinctness does not, by itself, give ANSWER-level
  distinctness (a rounding delta on one operand can cancel an opposite delta
  on another when they are combined). Every ``apply_approximation_strategy``
  tier therefore redraws, at the whole-expression level, whenever the
  constructed answer happens to equal what leading-digit rounding of the
  SAME operands would give (``_aas_easy``'s ``check_distinct(answer,
  ld_est)`` and ``_aas_medium``/``_aas_hard``'s ``check_pairwise_distinct(answer,
  ld_est, ...)``). In practice this guard never actually trips for a
  ``front_end`` tier: every operand there is forced to round UP by exactly
  one unit of its own (shared) leading place, so at easy tier (2 terms,
  always addition) the total delta is trivially ``2 * place_size > 0``, and
  at hard tier (3 terms, coefficient pattern ``(+1, +/-1, +/-1)``) the total
  delta ``place_size * (1 +/- 1 +/- 1)`` can never sum to zero since
  ``1 +/- 1 +/- 1`` is always odd -- proved, not merely observed, in the
The guard is load-bearing for ``round_to_place``
  tiers, where no such parity argument holds.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_distinct,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 200
_ANSWER_CAP = 10_000  # every constructed answer must be strictly less than this


# =============================================================================
# Shared helpers -- leading-digit place/rounding/truncation, place-named
# rounding, tie exclusion, and a redraw-capable digit-count sampler.
# =============================================================================


def _leading_place(n: int) -> int:
    """The place-value size of ``n``'s own leading (first) digit, via
    repeated integer division -- e.g. ``_leading_place(45) == 10``,
    ``_leading_place(512) == 100``, ``_leading_place(4238) == 1000``.
    Construction's own route; the in-generator checks below derive the same
    quantity a DIFFERENT way (``10 ** (len(str(n)) - 1)``'s own
    suggested check formula) so a bug in one route is not silently mirrored
    in the other."""
    d = abs(n)
    place = 1
    while d >= 10:
        d //= 10
        place *= 10
    return place


def _is_place_tie(n: int, place: int) -> bool:
    """Whether ``n`` sits exactly on a rounding tie for the target ``place``
    (the digit at that place is exactly half the place's own unit, e.g. 45
    for ``place=10``, 450/3500 for ``place=100``/``1000``)."""
    return place > 1 and (n % place) == place // 2


def _round_to_leading_digit(n: int) -> int:
    """Round ``n`` to its OWN leading-digit place, ties away from zero --
    ACMEM004's one and only method."""
    place = _leading_place(n)
    return int(round_half_up(Fraction(n, place), 0) * place)


def _front_end_truncate(n: int) -> int:
    """Truncate ``n`` DOWN to its own leading-digit place -- never round,
    regardless of the next digit. ACMEM010's ``front_end`` strategy."""
    place = _leading_place(n)
    return (n // place) * place


_STATED_PLACE_SIZE = {"ten": 10, "hundred": 100}


def _round_to_stated_place(n: int, place_name: str) -> int:
    """Round ``n`` to an EXPLICITLY STATED place (``"ten"``/``"hundred"``),
    independent of ``n``'s own leading-digit place. ACMEM010's
    ``round_to_place`` strategy."""
    size = _STATED_PLACE_SIZE[place_name]
    return int(round_half_up(Fraction(n, size), 0) * size)


def _round_to_hundred(n: int) -> int:
    """``_round_to_stated_place(n, "hundred")`` as a plain unary callable,
    for use where a ``transform`` variable must be assigned rather than
    called inline (the hard-tier ``round_to_place`` branch)."""
    return _round_to_stated_place(n, "hundred")


def _draw_n_digit(
    rng, digits: int, *, second_min: int | None = None, leading_max: int | None = None, tie_place: int | None = None
) -> int:
    """Draw a random ``digits``-digit positive integer, redrawing away from
    a rounding tie rather than resolving it in code.

    ``second_min`` forces the digit immediately after the leading digit to be at least that value (the
    overlap-prevention forcing for ``front_end`` operands). ``leading_max`` caps the leading digit.
    ``tie_place`` is the place size to exclude a tie at; defaults to the number's own leading place
    (ACMEM004's tie rule) -- pass an explicit place (10 or 100) to exclude a tie at a STATED place instead
    (ACMEM010's ``round_to_place`` operands).
    """
    lead_hi = leading_max if leading_max is not None else 9
    while True:
        parts = [rng.randint(1, lead_hi)]
        if digits >= 2:
            parts.append(rng.randint(second_min, 9) if second_min is not None else rng.randint(0, 9))
        for _ in range(digits - 2):
            parts.append(rng.randint(0, 9))
        n = int("".join(str(d) for d in parts))
        place = tie_place if tie_place is not None else _leading_place(n)
        if not _is_place_tie(n, place):
            return n


# --- Independent per-operand checks (in-generator's own literal
# formula: place size from digit count via `10 ** (len(str(n)) - 1)`) -------


def _verify_leading_round(n: int, expected: int):
    place = 10 ** (len(str(abs(n))) - 1)
    check_equivalent(
        round_half_up(Fraction(n, place), 0) * place, expected, f"leading-digit rounding of {n} must equal {expected}"
    )


def _verify_front_end(n: int, expected: int):
    place = 10 ** (len(str(abs(n))) - 1)
    check_equivalent((n // place) * place, expected, f"front-end truncation of {n} must equal {expected}")


def _verify_round_to_place(n: int, place_name: str, expected: int):
    size = {"ten": 10, "hundred": 100}[place_name]
    check_equivalent(
        round_half_up(Fraction(n, size), 0) * size,
        expected,
        f"rounding {n} to the nearest {place_name} must equal {expected}",
    )


# =============================================================================
# Skill 1: leading_digit_estimate (ACMEM004)
# =============================================================================


def _ld_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = _draw_n_digit(rng, 2, leading_max=8, second_min=6)  # always rounds UP
        b = _draw_n_digit(rng, 2, leading_max=8)
        op = rng.choice(["+", "x"])
        ra, rb = _round_to_leading_digit(a), _round_to_leading_digit(b)
        answer = ra + rb if op == "+" else ra * rb
        if answer >= _ANSWER_CAP:
            continue
        trunc_val = (
            _front_end_truncate(a) + _front_end_truncate(b)
            if op == "+"
            else _front_end_truncate(a) * _front_end_truncate(b)
        )
        try:
            check_distinct(answer, trunc_val)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("leading_digit_estimate: no collision-free easy draw found")

    _verify_leading_round(a, ra)
    _verify_leading_round(b, rb)
    check_equivalent(ra + rb if op == "+" else ra * rb, answer, "easy answer must equal the rounded expression")

    op_symbol = "+" if op == "+" else "×"
    text = f"Round each number to its own leading digit, then estimate {a} {op_symbol} {b}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(trunc_val)])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{a} rounds to {ra}; {b} rounds to {rb}; {ra} {op_symbol} {rb} = {answer}."],
        distractors=distractors,
    )


def _ld_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        pair_kind = rng.choice(("2_3", "3_4"))
        if pair_kind == "2_3":
            small, big = _draw_n_digit(rng, 2), _draw_n_digit(rng, 3)
        else:
            small, big = _draw_n_digit(rng, 3), _draw_n_digit(rng, 4, leading_max=7)
        op = rng.choice(["+", "-"])
        rs, rbig = _round_to_leading_digit(small), _round_to_leading_digit(big)
        answer = rbig + rs if op == "+" else rbig - rs
        if answer < 0 or answer >= _ANSWER_CAP:
            continue
        same_fixed = (
            _round_to_stated_place(big, "ten") + _round_to_stated_place(small, "ten")
            if op == "+"
            else _round_to_stated_place(big, "ten") - _round_to_stated_place(small, "ten")
        )
        trunc_val = (
            _front_end_truncate(big) + _front_end_truncate(small)
            if op == "+"
            else _front_end_truncate(big) - _front_end_truncate(small)
        )
        try:
            check_pairwise_distinct(answer, same_fixed, trunc_val)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("leading_digit_estimate: no collision-free medium draw found")

    _verify_leading_round(small, rs)
    _verify_leading_round(big, rbig)
    check_equivalent(rbig + rs if op == "+" else rbig - rs, answer, "medium answer must equal the rounded expression")

    text = f"Round each number to its own leading digit, then estimate {big} {op} {small}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(same_fixed), str(trunc_val)])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{big} rounds to {rbig}; {small} rounds to {rs}; {rbig} {op} {rs} = {answer}."],
        distractors=distractors,
    )


def _ld_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = _draw_n_digit(rng, 2, leading_max=8)
        b = _draw_n_digit(rng, 2, leading_max=8)
        c = _draw_n_digit(rng, 3)
        ra, rb, rc = _round_to_leading_digit(a), _round_to_leading_digit(b), _round_to_leading_digit(c)
        answer = rc + ra * rb
        if answer >= _ANSWER_CAP:
            continue
        ooo_wrong = (rc + ra) * rb  # left-to-right instead of precedence
        same_fixed = _round_to_stated_place(c, "ten") + ra * rb  # a, b already round the same at "ten"
        trunc_val = _front_end_truncate(c) + _front_end_truncate(a) * _front_end_truncate(b)
        try:
            check_pairwise_distinct(answer, ooo_wrong, same_fixed, trunc_val)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("leading_digit_estimate: no collision-free hard draw found")

    _verify_leading_round(a, ra)
    _verify_leading_round(b, rb)
    _verify_leading_round(c, rc)
    check_equivalent(rc + ra * rb, answer, "hard answer must equal the rounded expression evaluated with precedence")

    text = f"Round each number to its own leading digit, then estimate {c} + {a} × {b}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(ooo_wrong), str(same_fixed), str(trunc_val)])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{c} rounds to {rc}; {a} rounds to {ra}; {b} rounds to {rb}; "
            f"{rc} + {ra} × {rb} = {rc} + {ra * rb} = {answer}."
        ],
        distractors=distractors,
    )


@generator(
    "leading_digit_estimate",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def leading_digit_estimate(rng, tier="hard"):
    """Round EACH operand to its OWN leading digit, then compute. easy: two
    2-digit operands, one operation (+/x). medium: two operands of DIFFERENT
    digit counts, +/-, so each operand's own leading-digit place genuinely
    differs. hard: three operands, two operations (a x b + c shape),
    exercising order-of-operations on top of per-operand rounding.

    Bare-call default is "hard" (not "easy") -- matching the
    precedent -- because "easy" structurally offers only one distractor
    (the "same fixed place" misconception collides with the correct answer
    whenever both operands are 2-digit, since their own leading-digit place
    already IS the nearest ten), which would fail
    the usual 85%-of-seeds->=2-distractors bar on a bare, no-difficulty call. Every
    skill source passes ``difficulty`` explicitly, so this default only
    matters for that bare call.
    """
    if tier == "easy":
        return _ld_easy(rng)
    if tier == "medium":
        return _ld_medium(rng)
    if tier == "hard":
        return _ld_hard(rng)
    raise ValueError(f"unknown tier: {tier!r}")


# =============================================================================
# Skill 2: apply_approximation_strategy (ACMEM010)
# =============================================================================


def _ld_estimate_two(a: int, b: int, op: str) -> int:
    """What ``leading_digit_estimate``'s own rounding rule would give for
    the SAME two operands and operator -- used by the in-generator
    overlap-prevention redraw guard below. Deliberately reuses
    ``_round_to_leading_digit`` (construction's own route) rather than
    re-implementing the rounding rule a second time: this function's job is
    only to guarantee this generator's OWN answer never equals ACMEM004's
    -- a self-consistency guard the generator can act on (redraw) before it
    ever ships a question. The genuinely independent re-implementation, the
    one that would catch a bug shared with ``_round_to_leading_digit``
    itself, lives in the test file
    (``test_apply_approximation_strategy_always_differs_from_leading_digit_rounding``'s
    own ``_independent_leading_digit_round``) and in the oracle, neither of
    which imports anything from this module."""
    ra = _round_to_leading_digit(a)
    rb = _round_to_leading_digit(b)
    return ra + rb if op == "+" else ra - rb


def _ld_estimate_three(a: int, b: int, c: int, op1: str, op2: str) -> int:
    ra, rb, rc = _round_to_leading_digit(a), _round_to_leading_digit(b), _round_to_leading_digit(c)
    total = ra
    total = total + rb if op1 == "+" else total - rb
    total = total + rc if op2 == "+" else total - rc
    return total


def _aas_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = _draw_n_digit(rng, 3, leading_max=4, second_min=5)
        b = _draw_n_digit(rng, 3, leading_max=4, second_min=5)
        fa, fb = _front_end_truncate(a), _front_end_truncate(b)
        answer = fa + fb
        if answer >= _ANSWER_CAP:
            continue
        ld_est = _ld_estimate_two(a, b, "+")
        try:
            check_distinct(answer, ld_est)
        except VerificationError:
            continue
        check(ld_est != answer, "overlap-prevention guard must hold")
        break
    else:
        raise ValueError("apply_approximation_strategy: no collision-free easy draw found")

    _verify_front_end(a, fa)
    _verify_front_end(b, fb)
    check_equivalent(fa + fb, answer, "easy answer must equal the truncated sum")

    text = f"Use front-end estimation to estimate {a} + {b}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(ld_est)])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{a} truncates to {fa}; {b} truncates to {fb}; {fa} + {fb} = {answer}."],
        distractors=distractors,
    )


def _aas_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a = _draw_n_digit(rng, 3, tie_place=10)
        b = _draw_n_digit(rng, 3, tie_place=10)
        if a == b:
            continue
        op = rng.choice(["+", "-"])
        if op == "-":
            hi, lo = (a, b) if a > b else (b, a)
        else:
            hi, lo = a, b
        rhi, rlo = _round_to_stated_place(hi, "ten"), _round_to_stated_place(lo, "ten")
        answer = rhi + rlo if op == "+" else rhi - rlo
        if answer < 0 or answer >= _ANSWER_CAP:
            continue
        ld_est = _ld_estimate_two(hi, lo, op)
        front_end_est = (
            _front_end_truncate(hi) + _front_end_truncate(lo)
            if op == "+"
            else _front_end_truncate(hi) - _front_end_truncate(lo)
        )
        try:
            check_pairwise_distinct(answer, ld_est, front_end_est)
        except VerificationError:
            continue
        if ld_est == answer:
            continue
        break
    else:
        raise ValueError("apply_approximation_strategy: no collision-free medium draw found")

    _verify_round_to_place(hi, "ten", rhi)
    _verify_round_to_place(lo, "ten", rlo)
    check_equivalent(
        rhi + rlo if op == "+" else rhi - rlo, answer, "medium answer must equal the ten-rounded expression"
    )

    text = f"Round each number to the nearest ten, then estimate {hi} {op} {lo}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(front_end_est), str(ld_est)])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{hi} rounds to {rhi}; {lo} rounds to {rlo}; {rhi} {op} {rlo} = {answer}."],
        distractors=distractors,
    )


def _aas_hard(rng):
    strategy = rng.choice(("front_end", "round_to_place"))
    for _ in range(_RESAMPLE_ATTEMPTS):
        if strategy == "front_end":
            a = _draw_n_digit(rng, 3, second_min=5)
            b = _draw_n_digit(rng, 3, second_min=5)
            c = _draw_n_digit(rng, 3, second_min=5)
            transform = _front_end_truncate
            other_transform = _round_to_leading_digit  # ACMEM004's own method
        else:
            a = _draw_n_digit(rng, 4, leading_max=2, tie_place=100)
            b = _draw_n_digit(rng, 4, leading_max=2, tie_place=100)
            c = _draw_n_digit(rng, 4, leading_max=2, tie_place=100)
            transform = _round_to_hundred
            other_transform = _front_end_truncate

        op1 = rng.choice(["+", "-"])
        op2 = rng.choice(["+", "-"])
        ta, tb, tc = transform(a), transform(b), transform(c)
        running = ta
        running = running + tb if op1 == "+" else running - tb
        if running < 0 or running >= _ANSWER_CAP:
            continue
        answer = running + tc if op2 == "+" else running - tc
        if answer < 0 or answer >= _ANSWER_CAP:
            continue

        ld_est = _ld_estimate_three(a, b, c, op1, op2)
        # "switches strategy partway": correct transform on a, b; the OTHER
        # transform on c only.
        oc = other_transform(c)
        switch_running = ta
        switch_running = switch_running + tb if op1 == "+" else switch_running - tb
        switch_est = switch_running + oc if op2 == "+" else switch_running - oc

        if strategy == "front_end":
            # `other_transform` (leading-digit ROUNDING) is ACMEM004's own method, and for front_end
            # operands that is structurally the SAME formula as `ld_est` above (both round every
            # operand to its own leading digit) -- offering it a second time would always collide
            # with `ld_est`, not merely sometimes, so it is not offered as a separate candidate
            # here.
            candidates = [ld_est, switch_est]
        else:
            oa, ob = other_transform(a), other_transform(b)
            other_running = oa
            other_running = other_running + ob if op1 == "+" else other_running - ob
            other_est = other_running + oc if op2 == "+" else other_running - oc
            candidates = [ld_est, other_est, switch_est]

        if any(v < 0 for v in candidates):
            continue  # every distractor must share the answer's plain-positive-integer shape
        try:
            check_pairwise_distinct(answer, *candidates)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("apply_approximation_strategy: no collision-free hard draw found")

    if strategy == "front_end":
        _verify_front_end(a, ta)
        _verify_front_end(b, tb)
        _verify_front_end(c, tc)
        strategy_text = "Use front-end estimation to estimate"
    else:
        _verify_round_to_place(a, "hundred", ta)
        _verify_round_to_place(b, "hundred", tb)
        _verify_round_to_place(c, "hundred", tc)
        strategy_text = "Round each number to the nearest hundred, then estimate"

    check_equivalent(
        (ta + tb if op1 == "+" else ta - tb) + tc if op2 == "+" else (ta + tb if op1 == "+" else ta - tb) - tc,
        answer,
        "hard answer must equal the transformed expression",
    )

    text = f"{strategy_text} {a} {op1} {b} {op2} {c}."
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(v) for v in candidates])

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{a} -> {ta}; {b} -> {tb}; {c} -> {tc}; {ta} {op1} {tb} {op2} {tc} = {answer}."],
        distractors=distractors,
    )


@generator(
    "apply_approximation_strategy",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def apply_approximation_strategy(rng, tier="hard"):
    """Round EACH operand to an EXPLICITLY NAMED strategy -- ``front_end``
    (truncate to the leading digit's place) or ``round_to_place`` (round to
    a STATED place, ten or hundred) -- then compute. easy: front_end, two
    3-digit operands (second digit forced 5-9), addition only. medium:
    round_to_place="ten", two 3-digit operands, +/-. hard: three operands,
    two operations, ONE named strategy applied CONSISTENTLY (front_end on
    three 3-digit operands, or round_to_place="hundred" on three 4-digit
    operands).

    Bare-call default is "hard", for the same reason as
    ``leading_digit_estimate``'s own default: "easy" structurally offers
    only one distractor value (the "other strategy"/"ACMEM004 method"
    misconceptions collapse to the same formula when no place is stated).
    """
    if tier == "easy":
        return _aas_easy(rng)
    if tier == "medium":
        return _aas_medium(rng)
    if tier == "hard":
        return _aas_hard(rng)
    raise ValueError(f"unknown tier: {tier!r}")
