"""Generates `p10c_population_compounding` questions: compound growth or decay applied to a population rather
than to money.

Compounding applied OUTSIDE money.

``senior-consider-similar-problems-involving-compounding-for-example-population``
(ACMEM170, topic ``p10c_population_compounding``, ``answer: number``).

**BOUNDARY -- the whole point of this skill.** ACMEM170's own descriptor
text is "for example, population growth"; its curriculum topic label ("Loans
and compound interest") is misleading -- the descriptor itself asks for compounding
applied to something that is NOT money. This module must never render a `$`
sign or the word "interest"/"loan"/"invest[ment]" anywhere a generated
question, working line or distractor can reach -- that shape already belongs
to ``skills/financial-discrete/compound-interest.yaml``
(``mathgen.generators.curriculum_expansion.financial_interest``, which
explicitly prints ``$...`` and "invested ... compounded annually"). Every
tier below is a concrete, non-financial, whole-number physical quantity
(organisms, a population count) with no currency symbol, rate-of-interest
word, loan or investment anywhere in scope.

**Exact-arithmetic construction (no rounding, ever).** The per-period rate
is ``g%`` drawn from ``{10, 20, 25, 50}``, so ``k = 100 // g`` is a small
integer (``{10, 5, 4, 2}``) and the rate is exactly ``1/k``: growth
multiplies by ``Fraction(k + 1, k)``, decay by ``Fraction(k - 1, k)``. The
starting quantity is chosen as an exact multiple of ``k ** n`` inside its
tier's range, which by the following divisibility argument makes EVERY
intermediate compounding step -- not just the final one -- land on a whole
number: if ``P0 = m * k ** n``, then after ``i <= n`` compounding steps the
value is ``m * k ** (n - i) * (k +/- 1) ** i``, an exact integer for every
``i`` from ``0`` to ``n`` because ``gcd(k, k +/- 1) == 1`` (consecutive
integers share no factor), so no factor of ``k`` is ever needed from the
``(k +/- 1)`` part -- the whole ``k ** i`` in the denominator is supplied by
the ``k ** n`` already inside ``P0``.

Each tier's construction route and independent-check route are deliberately
different (see each function's docstring); ``_forward_steps`` is the loop
route (asserted whole at every intermediate step via ``check``, not merely
assumed), ``_closed_form`` is the single-power route used only to verify.

This is a real property of the formulas, not a bug hidden by the resample.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    latex_answer,
    pick_distractors,
)

# g% -> k = 100 // g, so the rate is exactly 1/k. {10, 20, 25, 50} -> {10, 5, 4, 2}.
_RATES = (10, 20, 25, 50)
_MAX_ATTEMPTS = 400

# How many multiples of k**n the MEDIUM tier's p0 draw offers per rate, held equal across g=10/20/25
# (g=50 never survives medium's decline filter regardless).
_MEDIUM_MULTIPLES_PER_RATE = 28


def _k_for(g: int) -> int:
    return 100 // g


def _forward_steps(p0: int, k: int, *, up: bool, n: int) -> list[Fraction]:
    """Construction route for easy/medium: a step LOOP multiplying by the
    growth (``up=True``) or decay (``up=False``) multiplier ``n`` times,
    returning every intermediate value (not just the last) so the caller can
    assert each one is an exact whole number directly."""
    mult = Fraction(k + 1, k) if up else Fraction(k - 1, k)
    values = []
    value = Fraction(p0)
    for _ in range(n):
        value *= mult
        values.append(value)
    return values


def _closed_form(p0: int, k: int, *, up: bool, n: int) -> Fraction:
    """Independent-check route for easy/medium: the single closed-form power,
    never touching the step loop above."""
    mult = Fraction(k + 1, k) if up else Fraction(k - 1, k)
    return Fraction(p0) * mult**n


def _multiple_in_range(rng, k: int, power: int, lo: int, hi: int) -> int | None:
    """A random multiple of ``k ** power`` inside ``[lo, hi]``, or ``None``
    if the range holds none -- a genuine search, never a silent clamp."""
    step = k**power
    m_lo = -(-lo // step)  # ceil(lo / step)
    m_hi = hi // step
    if m_lo > m_hi:
        return None
    return rng.randint(m_lo, m_hi) * step


# --- easy: population growth, forward, n = 2 --------------------------------


def _easy(rng):
    n = 2
    for _attempt in range(_MAX_ATTEMPTS):
        g = rng.choice(_RATES)
        k = _k_for(g)
        p0 = _multiple_in_range(rng, k, n, 500, 5000)
        if p0 is None:
            continue
        steps = _forward_steps(p0, k, up=True, n=n)
        check(
            all(v.denominator == 1 for v in steps),
            f"easy: non-integer intermediate step: g={g}, p0={p0}, steps={steps}",
        )
        answer = int(steps[-1])

        simple_growth = p0 + n * (p0 * g // 100)
        applied_once = p0 * (k + 1) // k
        candidates = (simple_growth, applied_once)
        if any(c <= 0 for c in candidates):
            continue
        if any(c == answer for c in candidates):
            continue
        break
    else:
        raise RuntimeError("p10c_population_compounding/easy: no valid draw found")

    check_equivalent(_closed_form(p0, k, up=True, n=n), answer, f"easy independent check failed: g={g}, p0={p0}, n={n}")
    check_distinct(answer, simple_growth, applied_once, msg=f"easy distractor collides with answer: g={g}, p0={p0}")

    text = f"A population of {p0} grows by {g}% each year. Find the population after {n} years."
    answer_text = str(answer)
    working = [
        f"Multiply by {k + 1}/{k} each year: {p0} -> {int(steps[0])} -> {answer}.",
    ]
    distractors = pick_distractors(answer_text, [str(simple_growth), str(applied_once)])
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=working,
        distractors=distractors,
    )


# --- medium: bacteria colony decay, forward, n = 3 ---------------------------


def _medium(rng):
    n = 3
    for _attempt in range(_MAX_ATTEMPTS):
        g = rng.choice(_RATES)
        k = _k_for(g)
        # `lo, hi` used to be the fixed (200, 2000) pair below, which reaches wildly different
        # multiple-COUNTS per rate because `step = k ** n` scales with `k` while the range does not:
        # g=10 (k=10, step=1000) reaches only 2 multiples, g=20 (k=5, step=125) reaches 15, g=25
        # (k=4, step=64) reaches 28. g=10 is drawn ~1/3 of the time (g=50 never survives the decline
        # filter below) but split over only 2 outcomes, so each gets ~1/6 of ALL draws -- the
        # measured 17%/16% concentration on 729/1458. Fix: hold the multiple-COUNT constant across
        # rates instead of the value range, by computing a per-`k` `hi` from
        # `_MEDIUM_MULTIPLES_PER_RATE`, set to 28 (g=25's own pre-fix count, the widest of the
        # three) so this only WIDENS the narrow rates (g=10 up to hi=28000, g=20 up to hi=3625) and
        # never narrows g=25 itself.
        step = k**n
        lo = 200
        m_lo = -(-lo // step)  # ceil(lo / step), matching _multiple_in_range's internals
        hi = (m_lo + _MEDIUM_MULTIPLES_PER_RATE - 1) * step
        p0 = _multiple_in_range(rng, k, n, lo, hi)
        if p0 is None:
            continue
        steps = _forward_steps(p0, k, up=False, n=n)
        check(
            all(v.denominator == 1 for v in steps),
            f"medium: non-integer intermediate step: g={g}, p0={p0}, steps={steps}",
        )
        answer = int(steps[-1])
        if answer <= 0:
            continue

        # simple-growth-not-compound (decay analogue): linear decline instead of compounding.
        simple_decline = p0 - n * (p0 * g // 100)
        applied_once = p0 * (k - 1) // k
        wrong_direction = int(_closed_form(p0, k, up=True, n=n))
        candidates = (simple_decline, applied_once, wrong_direction)
        if any(c <= 0 for c in candidates):
            continue
        if any(c == answer for c in candidates):
            continue
        break
    else:
        raise RuntimeError("p10c_population_compounding/medium: no valid draw found")

    check_equivalent(
        _closed_form(p0, k, up=False, n=n), answer, f"medium independent check failed: g={g}, p0={p0}, n={n}"
    )
    check_distinct(
        answer,
        simple_decline,
        applied_once,
        wrong_direction,
        msg=f"medium distractor collides with answer: g={g}, p0={p0}",
    )

    text = (
        f"A bacteria colony of {p0} organisms decreases by {g}% each hour. "
        f"Find the number of organisms after {n} hours."
    )
    answer_text = str(answer)
    working = [
        f"Multiply by {k - 1}/{k} each hour: {p0} -> {int(steps[0])} -> {int(steps[1])} -> {answer}.",
    ]
    distractors = pick_distractors(answer_text, [str(simple_decline), str(applied_once), str(wrong_direction)])
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=working,
        distractors=distractors,
    )


# --- hard: population growth, INVERSE ---------------------------------------


def _hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        n = rng.choice((2, 3))
        g = rng.choice(_RATES)
        k = _k_for(g)
        p0 = _multiple_in_range(rng, k, n, 500, 5000)
        if p0 is None:
            continue
        steps = _forward_steps(p0, k, up=True, n=n)
        check(
            all(v.denominator == 1 for v in steps),
            f"hard: non-integer intermediate step: g={g}, p0={p0}, n={n}, steps={steps}",
        )
        a_value = int(steps[-1])
        answer = p0  # the unknown the question asks for

        # un-simple-growth inversion (the hard-tier analogue of
        # simple-growth-not-compound): subtracts a flat g% of A, n times,
        # instead of dividing by the compounding factor. Computed EXACTLY
        # as a Fraction (A * (100 - n*g) / 100), not with the more obvious
        # literal "//" -- unlike easy/medium, A is not generally a multiple
        # of 100/g here, so a floor-division misconception value is only an
        # approximation of what a learner following that method would get
        # (a real defect found on review: for A=8415, g=50, n=2, the
        # literal `A - n*(A*g//100)` gives 1, but a learner computing "50%
        # of 8415, twice" by hand gets 4207.5 each time and 8415-2*4207.5=0
        # -- the exact value below, floored only for DISPLAY, matches that).
        # The positivity condition is then just `100 - n*g > 0`, which is
        # false whenever `g=50, n>=2` -- so g=50 is excluded from hard
        # entirely by this filter, same as it is from medium.
        simple_unwind_exact = Fraction(a_value) * Fraction(100 - n * g, 100)
        raw_multiplied = Fraction(a_value) * (Fraction(k + 1, k) ** n)
        exact_candidates = (simple_unwind_exact, raw_multiplied)
        if any(c <= 0 for c in exact_candidates):
            continue
        if any(c == answer for c in exact_candidates):
            continue
        simple_unwind = simple_unwind_exact.numerator // simple_unwind_exact.denominator
        multiplied_wrong = raw_multiplied.numerator // raw_multiplied.denominator  # floor
        if simple_unwind <= 0 or multiplied_wrong <= 0:
            continue
        if simple_unwind == answer or multiplied_wrong == answer:
            continue
        break
    else:
        raise RuntimeError("p10c_population_compounding/hard: no valid draw found")

    # Independent check: derive P0 FRESH by dividing A by multiplier**n --
    # never re-uses the seed's own p0 except to compare the result against
    # it. This is what a learner given only A, g and n must also do.
    mult = Fraction(k + 1, k)
    derived_p0 = Fraction(a_value) / (mult**n)
    check(derived_p0.denominator == 1, f"hard: derived P0 is not whole: A={a_value}, g={g}, n={n}")
    check_equivalent(derived_p0, answer, f"hard independent check failed: A={a_value}, g={g}, n={n}")
    # multiplied_wrong is A's own forward-compounded value again: strictly
    # larger than A itself (multiplier > 1), and A > P0 (growth), so
    # multiplied_wrong > A > P0 = answer algebraically, not merely observed.
    check(
        multiplied_wrong > a_value > answer,
        f"hard: multiplied_wrong ({multiplied_wrong}) not > A ({a_value}) > P0 ({answer})",
    )
    check_distinct(
        answer, simple_unwind, multiplied_wrong, msg=f"hard distractor collides with answer: A={a_value}, g={g}, n={n}"
    )

    text = f"A population grew by {g}% each year for {n} years and is now {a_value}. What was the original population?"
    answer_text = str(answer)
    working = [
        f"Divide by ({k + 1}/{k}) {n} times: {a_value} -> {answer}.",
    ]
    distractors = pick_distractors(answer_text, [str(simple_unwind), str(multiplied_wrong)])
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=working,
        distractors=distractors,
    )


@generator(
    "p10c_population_compounding",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10c_population_compounding(rng, tier="easy"):
    """Consider similar problems involving compounding, applied OUTSIDE
    money -- population growth (easy, forward), a bacteria colony's die-off
    (medium, decay, forward), and the original-population inverse (hard).
    See the module docstring for the exact-arithmetic construction and the
    non-financial-wording boundary."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    raise KeyError(f"p10c_population_compounding has no tier {tier!r}")
