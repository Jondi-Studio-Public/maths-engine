"""Early-foundations pair C: two more generators for two genuinely new graded operations, both framework
``acara-f10-v9``. **These two IDs finish the ``early-foundations`` domain** -- every skill in this
domain has now either shipped an active skill or remains explicitly gate-blocked.

- ``order_to_120`` (AC9M1N01, prerequisites ``f10-number-represent-order-to-20``,
  ``place-value-to-100``) -- order a stated set of distinct whole numbers,
  ascending, by numeral. The graded operation comes from the descriptor wording --
  "represent, name and order whole numbers to at least 120 using materials,
  numerals, a number line or chart" -- narrowed to the one operation that
  needs no visual construction: ordering a stated set of whole numbers by
  numeral (the removal note: "numerals" is one of AC9M1N01's own listed
  representation modes; the
  materials/number-line/chart modes are dropped as separate, undeveloped
  representation options). This skill's own prerequisite
  ``f10-number-represent-order-to-20``
  already covers the 0-20 range, including a forced zero-inclusion trap at
  its own hard tier -- so the sampled population here begins at 10 (two
  digits), the same "already covered by the smaller prerequisite" reasoning
  ``order_to_1000`` used to justify starting its own population at 10
  rather than 0. The genuinely new content is the range escalation from 20
  up to 120 and the two-digit/three-digit boundary itself (never sampling a
  four-digit number, since the contract's own "at least 120" ceiling is
  bounded above by the descriptor's own "three-digit place-value regrouping"
  exclusion -- this skill's numbers stop at 120, not 999). Three traps, not
  span alone: ``easy`` uses a single length pool (two-digit only) with wide
  forced gaps, a leading-digit read; ``medium`` forces a genuine
  teen-reversal pair (e.g. 13 vs 31 -- two numbers whose digits are
  literally swapped, a real, documented early-years place-value
  misconception distinct from the close-pair trap ``order_to_twenty``/
  ``order_to_1000`` already used, chosen explicitly over a plain close-pair
  repeat); ``hard`` forces at least one number >=100 alongside at least one
  number <100 on every single seed (never left to chance, the exact gap
  mixing two-digit (10-99) and three-digit (100-120) values so digit *count*
  must be compared first.

- ``quantify_compare_to_20`` (AC9MFN03, prerequisites ``compare-quantities-to-10``,
  ``count-to-10``, ``f10-number-represent-order-to-20``) -- given two named
  collections, determine which has more, which has fewer, or whether they
  are equal. Only the numeric comparison
  of the two collections' cardinalities is graded -- "communicate a reason
  for the comparison" is dropped (explaining a reason is an open-response
  capability, not a deterministic one), and
  ``visual_construction`` is not needed because AC9MFN03's own curriculum text
  ("quantify and compare collections to at least 20 using counting and
  explain or demonstrate reasoning") never mentions materials or a diagram
  at all, only "using counting". ``answer: mcq``, exactly three fixed semantic options across every
  question (never a fourth): "the first collection has more", "the second
  collection has more", "they are equal" -- matching the existing
  prerequisite skill ``compare-quantities-to-10``'s own mcq/scalar_expression
  shape ("Decide which group has more objects, or whether they are equal"),
  extended in range (10 -> 20) and reasoning demand. Three real difficulty
  axes: ``easy`` states both counts directly with a wide gap (>=5, a
  glance-read); ``medium`` states both counts directly with a narrow gap
  (1-2), with a genuine forced-equal branch on a real fraction of seeds
  (the "are they equal" outcome actually exercised, not just theoretically
  reachable); ``hard`` presents at least one collection as two named
  sub-groups that must be combined before comparing -- the contract's own
  literal "count or partition" alternative, a categorically harder
  reasoning demand (an addition step before the comparison can even be
  attempted), not an invented axis.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors, reversed_and_adjacent_swap

# -- order_to_120 --------------------------------------------------------

_TWO_DIGIT_RANGE = (10, 99)
_THREE_DIGIT_RANGE = (100, 120)  # the descriptor's own "at least 120" ceiling
_LENGTH_RANGES = {2: _TWO_DIGIT_RANGE, 3: _THREE_DIGIT_RANGE}


def _min_gap(numbers: list[int]) -> int:
    ordered = sorted(numbers)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def _teen_reversal_pair(rng) -> tuple[int, int]:
    """A teen number T (12-19) and its digit-reversed two-digit counterpart
    R (21-91): T = 10 + d, R = d*10 + 1 for d in 2-9. d=1 is excluded (T=R=11,
    no genuine reversal); d=0 is excluded from the teen range entirely
    (10-19 requires d>=0 but reversal of 10 is 1, single-digit, not a valid
    two-digit counterpart). A learner who reads a teen numeral's digits in
    the wrong order (mistaking "tens digit, ones digit" for "ones digit,
    tens digit") confuses T for R or vice versa -- a real, documented
    early-years place-value misconception, distinct from a plain close-pair
    trap."""
    d = rng.randint(2, 9)
    t = 10 + d
    r = d * 10 + 1
    return t, r


def _sample_order_to_120_numbers(
    rng,
    count: int,
    *,
    digit_lengths: tuple[int, ...],
    min_gap: int | None,
    forced: tuple[int, ...] = (),
    max_attempts: int = 8000,
) -> list[int]:
    """Rejection-sample `count` distinct whole numbers from the requested
    digit-length pools (10-99 and/or 100-120). `forced`, if supplied, must
    already lie within a single requested digit length's pool -- every
    sampled set includes exactly those numbers plus enough additional
    distinct numbers from the same pool to reach `count`."""
    if count < 2:
        raise ValueError("count must be at least 2")
    pool: list[int] = []
    for length in digit_lengths:
        lo, hi = _LENGTH_RANGES[length]
        pool.extend(range(lo, hi + 1))
    if count > len(pool):
        raise ValueError(f"cannot sample {count} distinct numbers from a pool of only {len(pool)}")
    if forced:
        if len(digit_lengths) != 1:
            raise ValueError("forced numbers are only supported with a single digit-length pool")
        if any(n not in pool for n in forced):
            raise ValueError(f"forced numbers {forced!r} are outside the requested pool")
        if len(forced) > count:
            raise ValueError("more forced numbers than the requested count")

    for _ in range(max_attempts):
        if len(digit_lengths) > 1:
            # Hard tier: force at least one number from each requested
            # digit length to actually appear -- a naive independent choice
            # per item has a real (if small) chance of landing on the same
            # length every time, silently degenerating into the single-
            # length case.
            chosen_lengths = [rng.choice(digit_lengths) for _ in range(count)]
            if len(set(chosen_lengths)) < len(digit_lengths):
                missing = [length for length in digit_lengths if length not in chosen_lengths]
                for i, length in enumerate(missing):
                    chosen_lengths[i] = length
            numbers: list[int] = []
            seen: set[int] = set()
            ok = True
            for length in chosen_lengths:
                lo, hi = _LENGTH_RANGES[length]
                attempt = 0
                n = rng.randint(lo, hi)
                while n in seen:
                    n = rng.randint(lo, hi)
                    attempt += 1
                    if attempt > 500:
                        ok = False
                        break
                if not ok:
                    break
                seen.add(n)
                numbers.append(n)
            if not ok:
                continue
        else:
            remaining_pool = [n for n in pool if n not in forced]
            rest = rng.sample(remaining_pool, count - len(forced))
            numbers = list(forced) + rest

        if min_gap is not None and _min_gap(numbers) < min_gap:
            continue
        return numbers

    raise RuntimeError(
        "could not sample numbers satisfying the requested difficulty shape "
        f"(count={count}, digit_lengths={digit_lengths}, min_gap={min_gap}, "
        f"forced={forced}) in {max_attempts} attempts"
    )


def _verify_ascending_order(sampled: list[int], ordered: list[int]) -> None:
    """Independent check, same all-pairs discipline as ``order_to_twenty``'s
    own ``_verify_ascending_order``/``order_to_1000``'s own
    ``_verify_order_to_1000``: verify every pair (i, j) with i < j -- not
    just adjacent pairs, and never by re-running the `sorted(...)` call that
    built `ordered` -- via direct integer comparison of the two numbers'
    actual values (the population never exceeds 120, so no digit-count/
    string-comparison machinery is needed -- plain integer `<` is exact)."""
    check(sorted(sampled) == sorted(ordered), "ordered list is not a permutation of the sampled numbers")
    check(len(ordered) == len(set(ordered)), "sampled numbers must be distinct")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            check(int(ordered[i]) < int(ordered[j]), f"{ordered[i]} does not precede {ordered[j]}")


@generator(
    "order_to_120",
    level=1,
    presets={
        "easy": dict(count=(3, 4), digit_lengths=(2,), min_gap=20, force_teen_reversal=False),
        "medium": dict(count=4, digit_lengths=(2,), min_gap=None, force_teen_reversal=True),
        "hard": dict(count=5, digit_lengths=(2, 3), min_gap=None, force_teen_reversal=False),
    },
)
def order_to_120(rng, count=4, digit_lengths=(2,), min_gap=None, force_teen_reversal=False):
    """Order a stated set of distinct whole numbers up to 120 (numerals
    only), ascending. See the module docstring for the difficulty
    rationale."""
    if isinstance(count, (tuple, list)):
        if not count:
            raise ValueError("count sequence must not be empty")
        count = rng.choice(tuple(count))
    if count < 2:
        raise ValueError("count must be at least 2")
    digit_lengths = tuple(digit_lengths)
    unknown = [length for length in digit_lengths if length not in _LENGTH_RANGES]
    if not digit_lengths or unknown:
        raise ValueError(f"digit_lengths must be a non-empty subset of {sorted(_LENGTH_RANGES)}; got {digit_lengths!r}")
    if force_teen_reversal and digit_lengths != (2,):
        raise ValueError("force_teen_reversal is only defined for the two-digit-only pool")

    teen_pair: tuple[int, int] | None = None
    forced: tuple[int, ...] = ()
    if force_teen_reversal:
        if count < 2:
            raise ValueError("force_teen_reversal needs count >= 2")
        teen_pair = _teen_reversal_pair(rng)
        forced = teen_pair

    numbers = _sample_order_to_120_numbers(
        rng,
        count,
        digit_lengths=digit_lengths,
        min_gap=min_gap,
        forced=forced,
    )

    mixed_lengths = len(digit_lengths) > 1
    if mixed_lengths:
        # Never left to chance: the sampling loop above already forces this, but assert it directly
        # here too.
        check(
            any(n >= 100 for n in numbers) and any(n < 100 for n in numbers),
            "hard tier must mix a two-digit and a three-digit number",
        )

    ordered = sorted(numbers)
    _verify_ascending_order(numbers, ordered)

    display_order = list(numbers)
    rng.shuffle(display_order)
    display_text = ", ".join(str(n) for n in display_order)
    answer_text = ", ".join(str(n) for n in ordered)

    if mixed_lengths:
        working = [
            "If two numbers have a different number of digits, the one with more digits is always larger.",
            "Within numbers that have the same number of digits, compare digit by digit from the left.",
            f"In order: {answer_text}.",
        ]
    elif force_teen_reversal:
        t, r = teen_pair
        working = [
            f"Read each number's digits in the order they are written -- {t} and {r} look similar but "
            "are not the same value.",
            "Compare the numbers by value, smallest first.",
            f"In order: {answer_text}.",
        ]
    else:
        working = [
            "Compare the numbers by value, smallest first.",
            f"In order: {answer_text}.",
        ]

    # Misconception distractors:
    #  - reversed-order: largest to smallest instead of smallest to largest.
    #  - adjacent-swap: correctly ordered overall, but the two middle values
    #    are swapped, as if two neighbouring values did not need comparing.
    #  - teen-reversal-confusion (medium, force_teen_reversal only): swaps
    #    the positions of the forced teen/reversal pair in the ordered list,
    #    as if the learner misread one numeral's digits as the other's.
    #  - leading-digit-only (hard/mixed-length only): sorted by each
    #    numeral's leading digit alone, ignoring overall digit count -- e.g.
    #    wrongly ranking "99" above "105" because "9" > "1".
    reversed_order, adjacent_swap = reversed_and_adjacent_swap(ordered)
    candidates = [
        ", ".join(str(n) for n in reversed_order),
        ", ".join(str(n) for n in adjacent_swap),
    ]
    if force_teen_reversal:
        t, r = teen_pair
        idx_t, idx_r = ordered.index(t), ordered.index(r)
        teen_reversal_confusion = list(ordered)
        teen_reversal_confusion[idx_t], teen_reversal_confusion[idx_r] = (
            teen_reversal_confusion[idx_r],
            teen_reversal_confusion[idx_t],
        )
        candidates.append(", ".join(str(n) for n in teen_reversal_confusion))
    if mixed_lengths:
        leading_digit_only = sorted(display_order, key=lambda n: str(n)[0])
        candidates.append(", ".join(str(n) for n in leading_digit_only))

    return dict(
        text=f"Arrange these numbers from smallest to largest: {display_text}.",
        latex=rf"\text{{Order: }} {display_text}",
        answer=list(ordered),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# -- quantify_compare_to_20 -----------------------------------------------

_OPTION_FIRST_MORE = "the first collection has more"
_OPTION_SECOND_MORE = "the second collection has more"
_OPTION_EQUAL = "they are equal"
_OPTIONS = (_OPTION_FIRST_MORE, _OPTION_SECOND_MORE, _OPTION_EQUAL)


def _label_for(total_a: int, total_b: int) -> str:
    if total_a > total_b:
        return _OPTION_FIRST_MORE
    if total_b > total_a:
        return _OPTION_SECOND_MORE
    return _OPTION_EQUAL


def _sum_via_manual_loop(parts: list[int]) -> int:
    """Independent accumulation route, distinct from `sum(...)` (whichever
    route construction itself used) -- used only by the independent check,
    never by construction."""
    total = 0
    for part in parts:
        total = total + part
    return total


def _verify_quantify_compare(parts_a: list[int], parts_b: list[int], total_a: int, total_b: int, label: str) -> None:
    """Independent check: recompute each collection's true total via a
    manual accumulation loop (never `sum(...)`, whichever route construction
    used), then re-derive the more/fewer/equal label via direct integer
    comparison of the two recomputed totals -- never re-reading the label
    construction already decided."""
    recomputed_a = _sum_via_manual_loop(parts_a)
    recomputed_b = _sum_via_manual_loop(parts_b)
    check(recomputed_a == total_a, f"collection A total mismatch: {recomputed_a} != {total_a}")
    check(recomputed_b == total_b, f"collection B total mismatch: {recomputed_b} != {total_b}")
    check(_label_for(recomputed_a, recomputed_b) == label, "comparison label does not match recomputed totals")


def _sample_direct_totals(
    rng,
    hi: int,
    *,
    min_gap: int | None = None,
    max_gap: int | None = None,
    max_attempts: int = 2000,
) -> tuple[int, int]:
    for _ in range(max_attempts):
        total_a = rng.randint(1, hi)
        total_b = rng.randint(1, hi)
        gap = abs(total_a - total_b)
        if min_gap is not None and gap < min_gap:
            continue
        if max_gap is not None and (gap == 0 or gap > max_gap):
            continue
        return total_a, total_b
    raise RuntimeError(
        f"could not sample two direct totals within 1-{hi} satisfying "
        f"min_gap={min_gap}, max_gap={max_gap} in {max_attempts} attempts"
    )


def _partition_into_two(rng, total: int) -> tuple[int, int]:
    if total < 2:
        raise ValueError("cannot partition a total below 2 into two positive sub-groups")
    first = rng.randint(1, total - 1)
    second = total - first
    return first, second


@generator(
    "quantify_compare_to_20",
    level=1,
    presets={
        "easy": dict(mode="direct", hi=20, min_gap=5, equal_fraction=0.0),
        "medium": dict(mode="direct", hi=20, max_gap=2, equal_fraction=0.3),
        "hard": dict(mode="partition", hi=20, equal_fraction=0.0),
    },
)
def quantify_compare_to_20(rng, mode="direct", hi=20, min_gap=None, max_gap=None, equal_fraction=0.0):
    """Given two named collections (counts, or a "count OR partition"
    framing at hard), determine which has more, which has fewer, or whether
    they are equal. See the module docstring for the difficulty rationale."""
    if hi < 2:
        raise ValueError("hi must allow at least two distinguishable counts (>=2)")
    if mode not in ("direct", "partition"):
        raise ValueError(f"unknown mode {mode!r}; expected 'direct' or 'partition'")
    if not (0.0 <= equal_fraction <= 1.0):
        raise ValueError(f"equal_fraction must be within 0.0-1.0; got {equal_fraction!r}")
    if mode == "partition" and hi < 2:
        raise ValueError("partition mode needs hi >= 2 (a total must be splittable into two positive parts)")

    force_equal = equal_fraction > 0 and rng.random() < equal_fraction

    if mode == "direct":
        if force_equal:
            total_a = rng.randint(1, hi)
            total_b = total_a
        else:
            total_a, total_b = _sample_direct_totals(rng, hi, min_gap=min_gap, max_gap=max_gap)
        parts_a = [total_a]
        parts_b = [total_b]
        partitioned_side = None
    else:
        total_a = rng.randint(2, hi)
        total_b = rng.randint(2, hi)
        if force_equal:
            total_b = total_a
        partitioned_side = rng.choice(("a", "b"))
        if partitioned_side == "a":
            parts_a = list(_partition_into_two(rng, total_a))
            parts_b = [total_b]
        else:
            parts_a = [total_a]
            parts_b = list(_partition_into_two(rng, total_b))

    label = _label_for(total_a, total_b)
    _verify_quantify_compare(parts_a, parts_b, total_a, total_b, label)

    def _describe(parts: list[int]) -> str:
        if len(parts) == 1:
            noun = "object" if parts[0] == 1 else "objects"
            return f"{parts[0]} {noun}"
        return f"a group of {parts[0]} and a group of {parts[1]}"

    text = f"Group A has {_describe(parts_a)}. Group B has {_describe(parts_b)}. Which is true?"

    working = []
    if partitioned_side == "a":
        working.append(f"Group A: combine the two groups first, {parts_a[0]} + {parts_a[1]} = {total_a}.")
    elif partitioned_side == "b":
        working.append(f"Group B: combine the two groups first, {parts_b[0]} + {parts_b[1]} = {total_b}.")
    working.append(f"Compare the totals: Group A has {total_a}, Group B has {total_b}.")
    working.append(f"{label[0].upper()}{label[1:]}.")

    # Misconceptions (mcq -- exactly 3 fixed semantic options total, so
    # each distractor is one of the two options other than the correct one):
    #  - swapped-comparison: states the *other* collection has more than the
    #    true comparison -- exercisable whenever the correct answer is
    #    directional (total_a != total_b), i.e. every easy question, most
    #    medium questions and most hard questions.
    #  - forgets-to-combine (partition mode only): a learner who reads only
    #    the first-named sub-group of a partitioned collection and compares
    #    that partial count alone against the other collection's total,
    #    instead of combining both sub-groups first. Its resulting *label*
    #    is still necessarily one of the same three fixed option strings
    #    (the answer space has only three possible values), so it never
    #    introduces a new distractor string beyond the two already offered
    #    -- but it is only a genuinely *distinct wrong* answer from the
    #    truth on the subset of hard seeds where the partial-count label
    #    differs from the true label (see the generator test file for the
    #    measured, asserted-nonzero fraction).
    candidates = [option for option in _OPTIONS if option != label]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=label,
        answer_text=label,
        answer_latex=latex_answer(label, plain=True),
        working=working,
        distractors=pick_distractors(label, candidates, n=2),
    )
