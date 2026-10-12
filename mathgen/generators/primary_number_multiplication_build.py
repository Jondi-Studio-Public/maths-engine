"""Primary-number pilot pair B: two more generators for two
genuinely new graded operations in the ``primary-number`` domain, both
framework ``acara-f10-v9``.

- ``multidigit_multiplication`` (AC9M5N06, prerequisites
  ``multiplication-facts``, ``place-value-to-100``) -- solve ``a * b`` for a
  multi-digit ``a``/``b``. The graded scope is exactly solving a multi-digit
  multiplication problem -- choosing an efficient written, mental or digital
  strategy and justifying that choice are both out of scope (they would need
  ``technology_use`` and ``communication`` capability gates). Difficulty is driven
  by genuine procedural-complexity growth in the standard written algorithm
  (more partial products to combine, more carry/regrouping steps to track),
  not merely a bigger final number:
    - ``easy``: 2-digit x 1-digit -- a single partial product, at most one
      carry (``max_total_carries=1``).
    - ``medium``: 2-digit x 2-digit -- two partial products (multiply by the
      ones digit, multiply by the tens digit and shift) that must be
      combined; this doubling of partial products is itself the added
      procedural step, independent of how many individual carries occur.
    - ``hard``: 3-digit x 2-digit, with regrouping **guaranteed** in *both*
      partial-product computations (``min_carry_per_partial=1``) -- more
      carries to track than either easier tier, and combined with the
      medium tier's own two-partial-product-combination step.
  Independent check: place-value decomposition of ``a`` (digit-by-digit,
  ``a = sum(digit_i * 10**i)``), verified via ``sum(digit_i * 10**i * b)`` --
  a genuinely different computation route from the single ``a*b`` evaluation
  used to construct the answer, the same "construct one way, verify via an
  independently-derived computation" discipline as
  ``foundations.order_of_operations``. Decomposed *by ``a``*, not ``b``:
  decomposing ``b`` (``b =
  tens*10 + ones``, checked via ``a*tens*10 + a*ones``) degenerates into a
  restatement of the same ``a*b`` evaluation whenever ``b`` is a single
  digit (``tens=0``, so ``a*tens*10 + a*ones`` collapses algebraically to
  ``a*ones = a*b`` exactly) -- true at the easy tier, where ``b`` is always
  1-digit. Decomposing ``a`` instead is non-degenerate at every tier, since
  ``a`` is always sampled as a genuine multi-digit number (>= 2 digits,
  enforced by ``a_range``'s own ``lo >= 10`` guard).

- ``multiply_divide_one_digit_arrays`` (AC9M2N05 and AC9M3N04 -- one skill,
  two target areas -- prerequisites ``multiplication-facts``,
  ``equal-groups``, ``whole-number-division-facts``) -- given a one-digit
  multiplication or division situation framed via equal groups or arrays,
  compute the result. The graded deliverable is the computed number. The
  situation is drawn with the ``counters`` figure kind (equal groups or an
  array): at ``easy`` the picture is the whole situation and the learner
  reads the group count and group size from it; at ``medium`` it is the
  empty groups to share into (the total is stated); at ``hard`` it is one
  example group (the total is stated). Division never produces a remainder:
  every question is built from a known ``groups * each = total`` triple, so
  it always divides exactly (verified explicitly, not just assumed by
  construction). Difficulty is driven by situational framing and which
  quantity is unknown -- the skill caps every operand at one digit
  throughout, so operand magnitude is deliberately held constant across all
  three tiers and is not used as a difficulty axis:
    - ``easy`` (``variant="find_total"``): groups and each-group size both
      stated, find the total (multiplication).
    - ``medium`` (``variant="find_each"``): total and number of groups
      stated, find each-group size -- partitive division.
    - ``hard`` (``variant="find_groups"``): total and each-group size
      stated, find the number of groups -- quotitive division, genuinely the
      harder of the two division framings for young learners (requires
      seeing "how many times does this fit" rather than "share this out").
  Independent check: for multiplication, verify via repeated addition (sum
  ``each`` added to itself ``groups`` times) -- a route distinct from
  ``groups * each``. For division, verify by multiplying the found unknown
  back against the known factor and asserting it reconstructs the stated
  total -- a distinct check from whichever division operator built the
  answer.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import check, generator, pick_distractors
from ..figure_kinds.early import counters, counters_total
from ..slots import NAMES

# -- multidigit_multiplication -----------------------------------------------


def _digit_multiply_carries(a: int, digit: int) -> int:
    """Simulate the standard right-to-left single-digit multiplication step
    of the written long-multiplication algorithm: multiply ``a`` by a single
    ``digit`` (0-9), tracking how many column-to-column carries occur. This
    is the real "regrouping" artefact of the written algorithm, used only to
    *shape* the difficulty-tier sampling below -- never to construct the
    graded answer itself."""
    if not (0 <= digit <= 9):
        raise ValueError(f"digit must be a single digit 0-9, got {digit!r}")
    carry = 0
    carries = 0
    for ch in reversed(str(a)):
        product = int(ch) * digit + carry
        carry = product // 10
        if carry:
            carries += 1
    return carries


def _no_carry_multiply(a: int, b: int) -> int:
    """Misconception ``dropped-carry``: computes each digit-by-digit partial
    product but drops the carry into the next column (every digit product
    truncated to its own units digit, mod 10, in isolation) -- the standard
    "forgot to regroup" written-algorithm error, not a generic numeric
    offset."""
    total = 0
    for i, bd_char in enumerate(reversed(str(b))):
        bd = int(bd_char)
        partial = 0
        for place, ad_char in enumerate(reversed(str(a))):
            digit_product_no_carry = (int(ad_char) * bd) % 10
            partial += digit_product_no_carry * (10**place)
        total += partial * (10**i)
    return total


def _transpose_first_two_digits(n: int) -> int | None:
    """Misconception ``transposed-digits``: swap two digits of ``n``,
    preferring the leading pair, falling back to the trailing pair if the
    leading two digits are identical (a same-digit swap is not a distinct
    value) or would introduce a leading zero. Returns ``None`` only when no
    distinct, same-digit-count transposition exists (e.g. every digit of
    ``n`` is identical)."""
    s = str(n)
    if len(s) < 2:
        return None
    if s[0] != s[1]:
        swapped = s[1] + s[0] + s[2:]
        if swapped[0] != "0":
            return int(swapped)
    if len(s) >= 3 and s[-1] != s[-2]:
        swapped = s[:-2] + s[-1] + s[-2]
        if swapped[0] != "0":
            return int(swapped)
    return None


def _verify_multidigit_multiplication(a: int, b: int, answer: int) -> None:
    """Independent check: construction computes ``answer = a * b`` directly.
    This check never re-runs that same evaluation -- it decomposes ``a`` by
    place value, digit by digit (``a = sum(digit_i * 10**i)``), and
    independently computes ``sum(digit_i * 10**i * b)``, asserting it equals
    the constructed answer. Decomposing ``a`` (not ``b``) is deliberate: ``a``
    is always multi-digit at every difficulty tier (enforced by the
    generator's own ``lo_a >= 10`` guard), so this route is genuinely
    non-degenerate throughout -- decomposing ``b`` instead would collapse
    into a restatement of the same ``a*b`` evaluation whenever ``b`` is a
    single digit, as it always is at the easy tier. A construction bug in
    the direct ``a*b`` evaluation would not silently reappear here, since
    this is a genuinely different arithmetic route."""
    digits = [int(ch) for ch in reversed(str(a))]
    check(sum(d * (10**i) for i, d in enumerate(digits)) == a, f"digit decomposition does not reconstruct a={a}")
    decomposed = sum(d * (10**i) * b for i, d in enumerate(digits))
    check(decomposed == answer, f"place-value decomposition of a gives {decomposed}, expected {answer}")


@generator(
    "multidigit_multiplication",
    level=5,
    presets={
        "easy": dict(a_range=(10, 99), b_range=(2, 9), max_total_carries=1, min_carry_per_partial=0),
        "medium": dict(a_range=(10, 99), b_range=(10, 99), max_total_carries=None, min_carry_per_partial=0),
        "hard": dict(a_range=(100, 999), b_range=(10, 99), max_total_carries=None, min_carry_per_partial=1),
    },
)
def multidigit_multiplication(
    rng,
    a_range=(10, 99),
    b_range=(10, 99),
    max_total_carries=None,
    min_carry_per_partial=0,
):
    """Solve ``a * b`` for a multi-digit ``a``/``b``. See the module
    docstring for the difficulty rationale."""
    lo_a, hi_a = a_range
    lo_b, hi_b = b_range
    if lo_a > hi_a or lo_b > hi_b:
        raise ValueError("a_range/b_range must each have lo <= hi")
    if lo_a < 10:
        raise ValueError("a_range must sample a multi-digit (>= 2 digit) multiplicand")
    if lo_b < 1:
        raise ValueError("b_range must sample a positive multiplier")
    if min_carry_per_partial > 0 and hi_b < 10:
        raise ValueError("min_carry_per_partial > 0 requires a multi-digit b (two partial products to force)")

    max_attempts = 8000
    for _ in range(max_attempts):
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        ones = b % 10
        tens = b // 10
        ones_carries = _digit_multiply_carries(a, ones)
        tens_carries = _digit_multiply_carries(a, tens) if tens else 0
        total_carries = ones_carries + tens_carries

        if max_total_carries is not None and total_carries > max_total_carries:
            continue
        if min_carry_per_partial > 0:
            if ones_carries < min_carry_per_partial:
                continue
            if tens == 0 or tens_carries < min_carry_per_partial:
                continue

        answer = a * b
        # Misconceptions:
        #  - dropped-carry: digit-by-digit multiplication with no carrying
        #    between place-value columns.
        #  - place-value-shift-error (only when b has a tens digit):
        #    multiplies by the tens digit but forgets to shift that partial
        #    product left by one place (off by a factor of 10 on that
        #    partial product).
        #  - transposed-digits: two digits of a transposed before
        #    multiplying.
        candidates = [str(_no_carry_multiply(a, b))]
        if tens:
            shifted_wrong = a * ones + a * tens  # forgot the x10 shift on the tens-digit partial product
            candidates.append(str(shifted_wrong))
        transposed_a = _transpose_first_two_digits(a)
        if transposed_a is not None:
            candidates.append(str(transposed_a * b))
        distractors = pick_distractors(str(answer), candidates)
        if not distractors:
            # Every candidate happened to collide with the real answer for
            # this specific a/b (e.g. a round number like 10 leaves both the
            # dropped-carry and transposed-digits routes undefined or
            # coincidentally correct) -- resample rather than ship a
            # question with zero usable distractors.
            continue
        break
    else:
        raise RuntimeError(
            "could not sample a/b satisfying the requested carry shape with at least one usable distractor "
            f"(a_range={a_range}, b_range={b_range}, max_total_carries={max_total_carries}, "
            f"min_carry_per_partial={min_carry_per_partial}) in {max_attempts} attempts"
        )

    _verify_multidigit_multiplication(a, b, answer)

    if tens:
        ones_partial = a * ones
        tens_partial = a * tens * 10
        working = [
            f"{a} × {ones} = {ones_partial}.",
            f"{a} × {tens * 10} = {tens_partial}.",
            f"Add the partial products: {ones_partial} + {tens_partial} = {answer}.",
        ]
    else:
        working = [f"{a} × {b} = {answer}."]

    return dict(
        text=f"Work out {a} × {b}.",
        latex=rf"{a} \times {b} = \square",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


# -- multiply_divide_one_digit_arrays ----------------------------------------

_VARIANTS = ("find_total", "find_each", "find_groups")

_TEXT_TOTAL_GROUPS = (
    "{n} puts counters in equal groups. How many counters altogether?",
    "{n} sees equal groups of counters. How many counters in all?",
    "{n} is counting equal groups of counters. How many counters are there altogether?",
)
_TEXT_TOTAL_ARRAY = (
    "{n} lines up counters in rows. How many counters altogether?",
    "{n} makes an array of counters. How many counters in all?",
    "{n} looks at an array. How many counters are there altogether?",
)
_TEXT_EACH = (
    "{t} counters are shared equally into these groups. How many counters are in each group?",
    "Share {t} counters equally between these groups. How many counters does each group get?",
    "{t} counters go into these equal groups. How many counters are in each group?",
)
_TEXT_GROUPS = (
    "{t} counters are put into equal groups like this one. How many equal groups are there?",
    "Every group has this many counters. {t} counters make equal groups. How many equal groups?",
    "{t} counters are split into groups the same size as this one. How many groups are there?",
)


def _verify_multiplication_repeated_addition(each: int, groups: int, total: int) -> None:
    """Independent check for the ``find_total`` (multiplication) variant:
    construction computes ``total = groups * each`` directly. This check
    never re-runs that multiplication -- it sums ``each`` added to itself
    ``groups`` times via a plain accumulating loop, a route distinct from
    the multiplication operator that built ``total``."""
    running = 0
    for _ in range(groups):
        running += each
    check(running == total, f"repeated addition of {each}, {groups} times, gives {running}, expected {total}")


def _verify_division_reconstructs(unknown: int, known_factor: int, total: int) -> None:
    """Independent check for both division variants (``find_each``,
    ``find_groups``): multiplies the found unknown back against the known
    factor and asserts it reconstructs the stated total -- a distinct check
    from whichever division/floor-division operator built the unknown."""
    reconstructed = unknown * known_factor
    check(reconstructed == total, f"{unknown} × {known_factor} = {reconstructed}, expected {total}")


def _verify_exact_division(total: int, divisor: int) -> int:
    """Division here must never produce a
    remainder. Every question is built from a known ``groups * each =
    total`` triple, so this always holds by construction -- verified
    explicitly here rather than only assumed."""
    quotient, remainder = divmod(total, divisor)
    check(remainder == 0, f"{total} ÷ {divisor} leaves a remainder of {remainder}; not a valid equal-groups fact")
    return quotient


@generator(
    "multiply_divide_one_digit_arrays",
    level=2,
    presets={
        "easy": dict(variant="find_total", groups_range=(2, 9), each_range=(2, 9)),
        "medium": dict(variant="find_each", groups_range=(2, 9), each_range=(2, 9)),
        "hard": dict(variant="find_groups", groups_range=(2, 9), each_range=(2, 9)),
    },
)
def multiply_divide_one_digit_arrays(rng, variant="find_total", groups_range=(2, 9), each_range=(2, 9)):
    """Given a one-digit multiplication or division situation framed via
    equal groups or arrays, compute the result. See the module docstring for
    the difficulty rationale."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    lo_g, hi_g = groups_range
    lo_e, hi_e = each_range
    if lo_g > hi_g or lo_e > hi_e:
        raise ValueError("groups_range/each_range must each have lo <= hi")
    if lo_g < 2 or lo_e < 2:
        raise ValueError("groups_range/each_range must start at 2 or above (a genuine multi-item grouping)")

    while True:
        groups = rng.randint(lo_g, hi_g)
        each = rng.randint(lo_e, hi_e)
        if variant == "find_total" and groups == 2 and each == 2:
            # groups + each (the wrong-operation distractor) coincides with
            # groups * each only at groups=each=2 -- resample rather than
            # ship a collided distractor for this one edge case.
            continue
        break

    total = groups * each
    quotient_check = _verify_exact_division(total, groups)
    check(quotient_check == each, f"total {total} ÷ groups {groups} gives {quotient_check}, expected each={each}")
    quotient_check2 = _verify_exact_division(total, each)
    check(quotient_check2 == groups, f"total {total} ÷ each {each} gives {quotient_check2}, expected groups={groups}")

    if variant == "find_total":
        answer = total
        _verify_multiplication_repeated_addition(each, groups, total)
        addition_line = " + ".join([str(each)] * groups)
        if rng.random() < 0.5:
            fig = counters(layout="groups", alt="equal groups of counters", groups=[each] * groups)
            text = rng.choice(_TEXT_TOTAL_GROUPS).format(n=rng.choice(NAMES))
        else:
            fig = counters(layout="array", alt="counters in an array of rows and columns", rows=groups, cols=each)
            text = rng.choice(_TEXT_TOTAL_ARRAY).format(n=rng.choice(NAMES))
        working = [
            f"{groups} groups of {each}.",
            f"Repeated addition: {addition_line} = {answer}.",
        ]
        wrong_operation = groups + each  # adds instead of multiplies
        confusion_candidate = None  # not applicable: only one unknown quantity in a find-total question
    elif variant == "find_each":
        answer = each
        _verify_division_reconstructs(answer, groups, total)
        fig = counters(layout="groups", alt="empty groups to share counters into", groups=[0] * groups)
        text = rng.choice(_TEXT_EACH).format(t=total)
        working = [
            f"{total} shared equally into {groups} groups.",
            f"{groups} × {answer} = {total}, so each group has {answer}.",
        ]
        wrong_operation = total * groups  # multiplies instead of divides
        confusion_candidate = groups  # reports the stated number-of-groups as if it answered each-group-size
    else:  # find_groups
        answer = groups
        _verify_division_reconstructs(answer, each, total)
        fig = counters(layout="groups", alt="one group of counters", groups=[each])
        text = rng.choice(_TEXT_GROUPS).format(t=total)
        working = [
            f"{total} split into groups of {each}.",
            f"{answer} × {each} = {total}, so there are {answer} groups.",
        ]
        wrong_operation = total * each  # multiplies instead of divides
        confusion_candidate = each  # reports the stated each-group-size as if it answered number-of-groups

    off_by_one = answer + 1 if answer > 0 else 1

    # Misconceptions:
    #  - wrong-operation: applies the wrong operation for the situation
    #    (adds instead of multiplies at find_total; multiplies instead of
    #    divides at find_each/find_groups).
    #  - partitive-quotitive-confusion (find_each/find_groups only): reports
    #    the *other* stated quantity, confusing which of "each-group size"
    #    vs "number of groups" is being asked for.
    #  - off-by-one on the group count or each-group size.
    candidates = [str(wrong_operation)]
    if confusion_candidate is not None:
        candidates.append(str(confusion_candidate))
    candidates.append(str(off_by_one))

    figures.assert_no_leak(fig, [str(answer)])
    check(_read_arrays(fig, text) == str(answer), "the picture does not show the keyed answer")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=pick_distractors(str(answer), candidates),
        figure=fig,
    )


def _read_arrays(fig: dict, text: str) -> str:
    """Re-derive the answer from the picture (and the total the prompt states, for the two division tiers)."""
    stated = re.search(r"\d+", text)
    if stated is None:  # nothing stated: the multiplication tier, the whole picture is the total
        return str(counters_total(fig))
    total = int(stated.group())
    rings = fig["groups"]
    if not any(rings):  # empty rings to share into: the number of rings is the divisor
        return str(total // len(rings))
    check(len(rings) == 1, "the grouping question shows one example group")
    return str(total // rings[0])


figures.register_reader("multiply_divide_one_digit_arrays", lambda q: _read_arrays(q.figure, q.text))
