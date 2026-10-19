"""Two generators for two new graded operations in the ``equations`` and ``primary-number`` domains, both
framework ``acara-f10-v9``.

- ``multiplication_division_inverse_unknown`` (AC9M5A01 and AC9M5A02 -- one
  skill, two target areas -- prerequisites ``multiplication-facts`` and
  ``whole-number-division-facts``) -- given a numerical multiplication or
  division equation with one missing value, use the inverse
  multiplication-division relationship to find the missing value.

  Every question presents one of five equation shapes (chosen per seed):
  ``a x ? = c`` (missing second factor), ``? x b = c`` (missing first
  factor), ``c / a = ?`` (missing quotient), ``c / ? = a`` (missing divisor),
The question text always names "the inverse
  multiplication-division relationship" explicitly as the method. Canonical
  answer is the single missing integer (exact-match tolerance).

  Three tiers, escalating genuine reasoning steps:

  - ``easy``: ``a, b`` both in ``2..10`` (single-digit multiplication-table
    facts), only the two multiplication-missing-factor shapes (``a x ? = c``
    and ``? x b = c``).
  - ``medium``: ``a, b`` both in ``2..12``, all four single-inverse shapes
    including both division shapes (introducing the division-equation
    reading, a new equation *shape* to parse, not just bigger numbers).
  - ``hard``: ``a`` in ``2..12``, ``b`` in ``11..25`` (products up to low
    300s, past instant table recall), the same four single-inverse shapes
    PLUS ``three_factor_product``, shape randomised independently
of magnitude.
    four-shape tuple medium does, and the (dead-code) tuple-identity range
    dispatch routed both to the medium range -- so the hard tier was
    range-only over medium. The range dispatch is now keyed by the SET of
    shapes requested (``_ranges_for_shapes``), additive and proved
    byte-identical for the two pre-existing combinations by a
    render-baseline replay; the skill's hard source now requests the fifth
    shape, so it reaches this range genuinely, and ``three_factor_product``
    itself adds a second reasoning step (combine the two known factors, THEN
    invert) no single-inverse-lookup shape requires.

  Independent check: recomputes the missing value **from the equation's own
  presented components** (the two/three shown numbers) using the inverse
  operation appropriate to the shown shape (never a reuse of the variable
  construction stored the answer in). For example, for ``a x ? = c``,
  recomputes ``c // a`` and asserts ``c % a == 0`` and the quotient equals
  the missing value; for ``a x b x ? = c``, recomputes ``c // (a * b)``.

  Misconceptions (confirmed distinct from correct answer by construction-time
  verification on every seed):

  - ``same-operation-not-inverse``: applies the *shown* operation to the two
    given numbers instead of its inverse (e.g. for ``a x ? = c``, computes
    ``a * c`` instead of ``c / a``). **Applicable to multiplication shapes
    only** (division shapes cannot apply this misconception, since dividing
    the two given numbers in either order is always distinct from the correct
    division-shape answer); for division shapes, this misconception is dropped
    entirely, never generated.
  - ``wrong-role-swap``: for a division shape, divides the two given numbers
    in the wrong order (e.g. computes ``a / c`` instead of ``c / a``).
    **Applicable to division shapes only.**
  - ``adjacent-fact-slip``: reports the answer for a neighbouring, off-by-one
    fact (e.g. ``a x (b+1)`` or ``a x (b-1)``'s result, or for a
    missing-factor shape, ``b + 1`` or ``b - 1`` directly, or for
    ``three_factor_product``, ``missing + 1``/``missing - 1`` directly).
    Applicable to all five shapes; clamped to stay a valid positive integer.
  - ``single-factor-divide``: when
    both factors on one side are known, divides by only ONE of them instead
    of their product (``c // a`` instead of ``c // (a * b)``). Always
    distinct from the answer by construction: ``c // a == b * missing``,
    which cannot equal ``missing`` since ``b >= 11`` in the hard range.
    ``three_factor_product`` also offers a second candidate, multiplying all
    three visible numbers together instead of dividing (``a * b * c``),
    always distinct since ``a * b * c == (a * b) ** 2 * missing`` and
    ``(a * b) ** 2 >= 4``.

- ``unit_fractions_combine_whole`` (AC9M3N02 -- one target area, primary-number
  domain -- prerequisites ``f10-fractions-half-quarter-eighth`` and
  ``fraction-arithmetic``) -- represent listed unit fractions and their
  multiples and combine same-denominator fractions to complete a whole. Contract:
  "Represent listed unit fractions and their multiples and combine
  same-denominator fractions to complete a whole."

  **Boundary against the prerequisite skill ``f10-fractions-half-quarter-eighth``**
that skill's entire operation is *repeated halving*
  of a fraction (forward or inverse), producing only values in
  ``{1, 1/2, 1/4, 1/8}``; this skill never halves anything -- it *multiplies*
  a unit fraction by an integer count (representing "n copies of 1/d") and
  *subtracts from a whole* (completing 1 given how many same-denominator parts
  are already used) -- a structurally different operation, even where a
  denominator value (``2`` or ``4``) happens to overlap. Every denominator
  this generator uses is drawn from the curriculum text's own explicit set
  ``{2, 3, 4, 5, 10}``.

  Two question shapes (beyond difficulty):

  - ``multiple``: "What fraction is ``k`` copies of ``1/d``?" -> answer
    ``Fraction(k, d)`` (never itself reducible to ``1`` or exceeding ``1`` --
    keep ``k < d``, this represents parts *of* a single whole).
  - ``complete_the_whole``: "``d`` equal parts make one whole. ``n`` of the
    ``d`` parts are already used (each part is ``1/d``). What fraction is still
    needed to complete the whole?" -> answer ``Fraction(d - n, d)``.

  Three tiers, escalating genuine reasoning:

  - ``easy``: ``multiple`` shape only, ``d`` in ``{2, 3, 4, 5}``, ``k`` in
    ``1..(d-1)``.
  - ``medium``: ``complete_the_whole`` shape only (a genuinely different
    operation from easy), ``d`` in ``{2, 3, 4, 5, 10}``, ``n`` in ``1..(d-1)``.
  - ``hard``: combine two same-denominator fractions already used, then find
    what completes the whole: "``d`` equal parts make a whole. ``n1`` parts
    are used, then ``n2`` more parts are used. What fraction is still needed?"
    -> answer ``Fraction(d - n1 - n2, d)`` with ``n1 + n2 < d`` enforced (a
    genuine extra reasoning step).

  Independent check: for ``multiple``, construction computes ``Fraction(k, d)``
  directly by division; the check recomputes via **repeated addition**
  (``sum(Fraction(1, d) for _ in range(k))``), a genuinely different route.
  For ``complete_the_whole`` (both single-term and two-term hard), construction
  computes ``1 - Fraction(n, d)`` directly; the check recomputes via the
  complementary route ``Fraction(d - n, d)`` built from the integer part-count,
  not via subtraction from ``1``.

  Misconceptions (confirmed distinct and exercisable):

  - ``denominator-inflation``: for ``multiple``, incorrectly scales the
    denominator by ``k`` too, e.g. ``Fraction(k, d * k)`` (which always
    reduces to ``1/d``, a different value from the correct ``k/d`` whenever
    ``k != 1``). **Applicable to easy tier only** (where ``multiple`` is
    generated). **At easy tier with ``d=2``, the only legal value is ``k=1``,
    forcing a 100% collision rate with the correct answer.** This
    defect is handled conservatively: ``d=2`` is excluded from easy (measured
    collision rate: 0% over sweep seeds), leaving ``d in {3, 4, 5}`` for easy,
    which has ``k >= 2``, so denominator-inflation always reduces to
``1/d != k/d`` (correct answer).
  - ``reports-amount-used``: for ``complete_the_whole``, reports ``n/d`` (or
    ``(n1+n2)/d`` for hard's two-term case) instead of the remaining amount.
    Collides exactly when the used and remaining amounts are equal
    (``n == d - n``, or ``n1+n2 == d - n1 - n2`` for hard). Measured
  - ``off-by-one-part``: reports one part more or fewer than the correct
    amount, e.g. ``Fraction(d - n - 1, d)`` or ``Fraction(d - n + 1, d)``
    (clamped to stay a valid non-negative fraction at most 1).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors


# -- multiplication_division_inverse_unknown ----------------------------------

_EASY_RANGE = range(2, 11)  # 2..10
_MEDIUM_RANGE = range(2, 13)  # 2..12
_HARD_A_RANGE = range(2, 13)  # 2..12
_HARD_B_RANGE = range(11, 26)  # 11..25

# Per-shape-set range lookup, replacing the former tuple-identity dispatch
# (`if shape_choices == (...)`). That dispatch was fragile: it recognised
# exactly the easy combination and exactly the medium/hard four-shape
# combination (which happened to be the SAME tuple, since the hard preset
# historically passed the same four shapes as medium), so its final `else`
# branch -- the one computing `_HARD_A_RANGE`/`_HARD_B_RANGE` -- was dead
# code in practice; hard silently drew from the medium range. Keyed by the
# SET of shapes requested (order-independent) so this is purely additive:
# both combinations below still map to the exact ranges they always did
# (proved by a before/after byte-identity replay), and any OTHER combination -- including one that adds the new
# `three_factor_product` shape -- falls through to `_DEFAULT_RANGE`,
# preserving the prior `else` branch's fallback behaviour.
_RANGE_BY_SHAPE_SET: dict[frozenset, tuple[range, range]] = {
    frozenset(("a_times_missing_equals_c", "missing_times_b_equals_c")): (_EASY_RANGE, _EASY_RANGE),
    frozenset(
        (
            "a_times_missing_equals_c",
            "missing_times_b_equals_c",
            "c_div_a_equals_missing",
            "c_div_missing_equals_a",
        )
    ): (_MEDIUM_RANGE, _MEDIUM_RANGE),
}
_DEFAULT_RANGE = (_HARD_A_RANGE, _HARD_B_RANGE)


def _ranges_for_shapes(shape_choices) -> tuple[range, range]:
    """Number-range lookup for a `shape_choices` combination -- see
    `_RANGE_BY_SHAPE_SET`'s comment for why this replaces tuple-identity
    comparison."""
    return _RANGE_BY_SHAPE_SET.get(frozenset(shape_choices), _DEFAULT_RANGE)


def _verify_multiplication_division_inverse(shape: str, a: int, b: int, c: int, missing: int) -> None:
    """Independent check: recomputes the missing value from the two
    presented numbers using the inverse operation appropriate to the shown
    shape, without reusing construction's stored answer."""
    if shape == "a_times_missing_equals_c":
        check(c % a == 0, f"c={c} is not divisible by a={a}")
        computed = c // a
    elif shape == "missing_times_b_equals_c":
        check(c % b == 0, f"c={c} is not divisible by b={b}")
        computed = c // b
    elif shape == "c_div_a_equals_missing":
        check(c % a == 0, f"c={c} is not divisible by a={a}")
        computed = c // a
    elif shape == "c_div_missing_equals_a":
        check(c % a == 0, f"c={c} is not divisible by a={a}")
        computed = c // a
    elif shape == "three_factor_product":
        # Genuine second step: combine the two KNOWN factors (a*b) before
        # inverting against c, never reusing construction's stored `missing`.
        check(c % (a * b) == 0, f"c={c} is not divisible by a*b={a * b}")
        computed = c // (a * b)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    check(
        computed == missing,
        f"shape={shape}, a={a}, b={b}, c={c}: recomputed {computed}, expected {missing}",
    )


@generator(
    "multiplication_division_inverse_unknown",
    level=3,
    presets={
        "easy": dict(shape_choices=("a_times_missing_equals_c", "missing_times_b_equals_c")),
        "medium": dict(
            shape_choices=(
                "a_times_missing_equals_c",
                "missing_times_b_equals_c",
                "c_div_a_equals_missing",
                "c_div_missing_equals_a",
            )
        ),
        "hard": dict(
            shape_choices=(
                "a_times_missing_equals_c",
                "missing_times_b_equals_c",
                "c_div_a_equals_missing",
                "c_div_missing_equals_a",
            )
        ),
    },
)
def multiplication_division_inverse_unknown(rng, shape_choices=None):
    """Given a multiplication or division equation with one missing value,
    use the inverse multiplication-division relationship to find the missing
    value."""
    if shape_choices is None:
        shape_choices = (
            "a_times_missing_equals_c",
            "missing_times_b_equals_c",
            "c_div_a_equals_missing",
            "c_div_missing_equals_a",
        )

    a_range, b_range = _ranges_for_shapes(shape_choices)

    a = rng.choice(list(a_range))
    b = rng.choice(list(b_range))
    c = a * b

    shape = rng.choice(shape_choices)

    # Build question text and determine missing value
    if shape == "a_times_missing_equals_c":
        missing = b
        text = f"Solve: {a} × ? = {c}. Use the inverse multiplication-division relationship."
        # Misconceptions for multiplication shapes
        same_op = a * c  # same operation (multiply) instead of inverse
        adjacent = b + 1 if rng.choice([True, False]) else max(1, b - 1)
        candidates = [same_op, adjacent]

    elif shape == "missing_times_b_equals_c":
        missing = a
        text = f"Solve: ? × {b} = {c}. Use the inverse multiplication-division relationship."
        same_op = b * c
        adjacent = a + 1 if rng.choice([True, False]) else max(1, a - 1)
        candidates = [same_op, adjacent]

    elif shape == "c_div_a_equals_missing":
        missing = b
        text = f"Solve: {c} ÷ {a} = ?. Use the inverse multiplication-division relationship."
        # Division shapes: wrong-role-swap and adjacent-fact-slip only
        wrong_role = a // c if c != 0 else 1  # floor division, but clamp to be displayable
        if wrong_role == 0:
            wrong_role = 1  # avoid 0
        adjacent = b + 1 if rng.choice([True, False]) else max(1, b - 1)
        # For division, express wrong-role as a decimal or integer
        candidates = [str(wrong_role), str(adjacent)]

    elif shape == "c_div_missing_equals_a":
        missing = b
        text = f"Solve: {c} ÷ ? = {a}. Use the inverse multiplication-division relationship."
        wrong_role = a // c if c != 0 else 1
        if wrong_role == 0:
            wrong_role = 1
        adjacent = b + 1 if rng.choice([True, False]) else max(1, b - 1)
        candidates = [str(wrong_role), str(adjacent)]

    elif shape == "three_factor_product":
        # Genuine second reasoning step: two KNOWN factors on one side
        # (a, b) must be combined before inverting against c -- neither
        # single-inverse-lookup shape above ever requires this. Construction
        # runs backwards from the answer, like every other shape here: pick
        # a, b, missing, then c = a * b * missing.
        missing = rng.choice(list(a_range))
        c = a * b * missing
        text = f"Solve: {a} × {b} × ? = {c}. Use the inverse multiplication-division relationship."
        # single-factor-divide: divides by only ONE of the two known
        # factors instead of their product (c // a == b * missing, which is
        # b * missing != missing since b >= 2 and missing >= 2).
        single_factor_divide = c // a
        # multiplies all three visible numbers together instead of
        # dividing -- always distinct here since a*b*c == (a*b)**2 * missing
        # and (a*b)**2 >= 4 > 1 for every legal a, b.
        multiply_all_three = a * b * c
        adjacent = missing + 1 if rng.choice([True, False]) else max(1, missing - 1)
        candidates = [single_factor_divide, multiply_all_three, adjacent]

    else:
        raise ValueError(f"unknown shape {shape!r}")

    # Verify independent check fires
    _verify_multiplication_division_inverse(shape, a, b, c, missing)

    # Verify distractors are distinct from answer
    if isinstance(candidates[0], str):
        # Division shapes: candidates are already strings
        answer_text = str(missing)
        for cand in candidates:
            check(
                cand != answer_text,
                f"distractor {cand} equals answer {answer_text}",
            )
    else:
        # Multiplication shapes: candidates are integers
        for cand in candidates:
            check(
                cand != missing,
                f"distractor {cand} equals answer {missing}",
            )

    answer_text = str(missing)
    distractors = pick_distractors(answer_text, [str(c) if isinstance(c, int) else c for c in candidates])

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=missing,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"The missing value is {missing}."],
        distractors=distractors,
    )


# -- unit_fractions_combine_whole ---------------------------------------------

# Easy: exclude d=2 because k=1 forces denominator-inflation collision
_EASY_DENOMS = {3, 4, 5}
_MEDIUM_DENOMS = {2, 3, 4, 5, 10}
_EASY_DENOMS_WIDE = (3, 4, 5, 6, 8, 10)  # for wide=True; the original easy set is {3, 4, 5}
_HARD_DENOMS = {3, 4, 5, 10}  # Exclude 2; d >= 3 ensures n1+n2 < d is feasible


def _verify_fraction_multiple(k: int, d: int, answer: Fraction) -> None:
    """Independent check for multiple shape: construction computes
    Fraction(k, d) directly; the check recomputes via repeated addition."""
    expected = sum(Fraction(1, d) for _ in range(k))
    check(
        answer == expected,
        f"k copies of 1/{d} should give {expected}, got {answer}",
    )


def _verify_fraction_complete_whole(d: int, n_total: int, answer: Fraction) -> None:
    """Independent check for complete_the_whole shape: construction
    computes 1 - Fraction(n_total, d); the check recomputes via
    Fraction(d - n_total, d)."""
    expected = Fraction(d - n_total, d)
    check(
        answer == expected,
        f"completing whole with {n_total}/{d} used should give {expected}, got {answer}",
    )


@generator(
    "unit_fractions_combine_whole",
    level=3,
    presets={
        "easy": dict(shape="multiple"),
        "medium": dict(shape="complete_the_whole"),
        "hard": dict(shape="complete_two_parts"),
    },
)
def unit_fractions_combine_whole(rng, shape="multiple", wide=False):
    """Represent and combine unit fractions to complete a whole. ``wide=True`` lets the easy shape use
    sixths, eighths and tenths too (the default keeps the original draws)."""
    if shape == "multiple":
        d = rng.choice(list(_EASY_DENOMS_WIDE if wide else _EASY_DENOMS))
        # k in 2..(d-1) -- exclude k=1 to avoid denominator-inflation collision
        # Since d >= 3 for easy, we have d-1 >= 2, so range is valid
        k = rng.randint(2, d - 1)
        answer = Fraction(k, d)
        _verify_fraction_multiple(k, d, answer)

        text = f"What fraction is {k} copies of 1/{d}?"
        raw_working, raw_fraction = f"{k} copies of 1/{d} is {k} × 1/{d} = {k}/{d}", f"{k}/{d}"
        # Misconceptions
        denom_inflation = Fraction(k, d * k)  # reduces to 1/d if k != 1
        # off-by-one-part: ±1 from numerator, clamped to stay in [0, d]
        off_by_one_plus = Fraction(min(k + 1, d), d)
        off_by_one_minus = Fraction(max(0, k - 1), d)
        candidates = [denom_inflation, off_by_one_plus, off_by_one_minus]

    elif shape == "complete_the_whole":
        d = rng.choice(list(_MEDIUM_DENOMS))
        n = rng.randint(1, d - 1)
        answer = Fraction(d - n, d)
        _verify_fraction_complete_whole(d, n, answer)

        raw_fraction = f"{d - n}/{d}"
        raw_working = (
            f"{d} - {n} = {d - n} {'part is' if d - n == 1 else 'parts are'} still needed, "
            f"so the fraction is {raw_fraction}"
        )
        text = f"{d} equal parts make one whole. {n} of the {d} parts {'is' if n == 1 else 'are'} already used. What fraction is still needed to complete the whole?"
        # Misconceptions
        reports_used = Fraction(n, d)
        # off-by-one-part: remaining ± 1 part, clamped to stay in [0, d]
        remaining = d - n
        off_by_one_plus = Fraction(min(remaining + 1, d), d)
        off_by_one_minus = Fraction(max(0, remaining - 1), d)
        candidates = [reports_used, off_by_one_plus, off_by_one_minus]

    elif shape == "complete_two_parts":
        d = rng.choice(list(_HARD_DENOMS))
        # n1, n2 in 1..(d-1) with n1 + n2 < d
        max_sum = d - 1
        n1 = rng.randint(1, max_sum - 1)
        n2 = rng.randint(1, max_sum - n1)
        n_total = n1 + n2
        answer = Fraction(d - n_total, d)
        _verify_fraction_complete_whole(d, n_total, answer)

        raw_fraction = f"{d - n_total}/{d}"
        raw_working = (
            f"{n1} + {n2} = {n_total} parts are used, so {d} - {n_total} = {d - n_total} "
            f"{'part is' if d - n_total == 1 else 'parts are'} still needed, which is {raw_fraction}"
        )
        text = f"{d} equal parts make a whole. {n1} {'part is' if n1 == 1 else 'parts are'} used, then {n2} more {'part is' if n2 == 1 else 'parts are'} used. What fraction is still needed to complete the whole?"
        # Misconceptions
        reports_used = Fraction(n_total, d)
        # off-by-one-part: remaining ± 1 part, clamped to stay in [0, d]
        remaining = d - n_total
        off_by_one_plus = Fraction(min(remaining + 1, d), d)
        off_by_one_minus = Fraction(max(0, remaining - 1), d)
        candidates = [reports_used, off_by_one_plus, off_by_one_minus]

    else:
        raise ValueError(f"unknown shape {shape!r}")

    # Format answer as fraction text
    answer_text = f"{answer.numerator}/{answer.denominator}"

    working_text = raw_working + (f" = {answer_text}." if raw_fraction != answer_text else ".")

    # Convert candidates to text; pick_distractors will handle deduplication
    distractor_texts = [f"{c.numerator}/{c.denominator}" for c in candidates]
    distractors = pick_distractors(answer_text, distractor_texts)

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[working_text],
        distractors=distractors,
    )
