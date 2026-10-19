"""Generates `multidigit_add_sub_regroup` questions: add or subtract two- and three-digit numbers by partitioning,
rearranging or regrouping, without a calculator.

Scope: whole numbers only. No decimals and no word problems.

## What makes each tier different

The numbers are drawn so that the regrouping the learner has to do is controlled, not left to chance: every
operand pair is checked by `_carries` / `_borrows`, which read the digits of the pair the way a column
algorithm would and report which columns carry or borrow.

- easy: two-digit numbers with exactly one regroup. An addition carries from the ones to the tens
  (`47 + 38`); a subtraction borrows once, from the tens to the ones (`62 - 38`). DECISION: carry out one
  regroup.
- medium: three-digit numbers with exactly two regroups in a row (ones and tens). An addition carries twice
  (`468 + 275`, or `468 + 75`); a subtraction borrows twice, with no zero in the tens place of the number being
  subtracted from (`543 - 278`). DECISION: the regroup in the tens column depends on the one made in the ones
  column, so the learner has to carry the first result into the second step.
- hard: the cases that go wrong in practice, with the prompt also asking for a check by the inverse operation. An addition that carries in all three columns and reaches the
  thousands (`678 + 459`); a subtraction that has to borrow through a zero (`503 - 276`, `400 - 167`).
  DECISION: borrowing through a zero means the tens place cannot lend until the hundreds place has lent to it,
  which easy and medium never need, and an addition whose answer has a new place.

## Independent check for every draw

The key is worked out two ways with `check_equivalent`: by the arithmetic of the whole numbers, and by place
value (partitioning into hundreds, tens and ones for an addition; counting up from the smaller number to the
larger for a subtraction). The carry or borrow pattern of the drawn pair is checked against the tier's
specification. `check_distinct` runs on every distractor candidate.

The distractors are the errors of the column method, each simulated column by column:
a carry that is never added or is added twice, the smaller digit subtracted from the larger in each column, a
borrow whose lender is not reduced, and the key plus or minus one.

The oracle parses the two numbers and the operation from `question.text`, works out the answer by a different
route (digit by digit with its own carry loop) and never imports this module.
"""

from __future__ import annotations

from ..core import check, check_distinct, check_equivalent, generator, phrase
from ._option_rank import pick_ranked_distractors

TOPIC = "multidigit_add_sub_regroup"
_TIERS = ("easy", "medium", "hard")

# Each tier words its prompt around the method it asks for, so the tiers differ in what the learner is told to
# do as well as in the numbers: easy partitions into tens and ones, medium works column by column with place value,
# hard also asks for a check by the inverse operation.
_ADD = {
    "easy": (
        "Work out {a} + {b} without a calculator.",
        "What is {a} + {b}? Partition the numbers to help.",
        "Find the sum of {a} and {b}.",
        "Add {a} and {b} without a calculator.",
    ),
    "medium": (
        "Partition both numbers into hundreds, tens and ones to work out {a} + {b}.",
        "Work out {a} + {b} by adding the hundreds, tens and ones separately.",
        "Use place value to find {a} + {b}.",
    ),
    "hard": (
        "Work out {a} + {b} without a calculator, then check your answer by subtracting.",
        "Find {a} + {b}, and check your answer by subtracting.",
        "Add {a} and {b}, then check your answer by subtracting.",
    ),
}
_SUB = {
    "easy": (
        "Work out {a} - {b} without a calculator.",
        "What is {a} - {b}? Regroup if you need to.",
        "Subtract {b} from {a} without a calculator.",
    ),
    "medium": (
        "Regroup where you need to and work out {a} - {b}.",
        "Work out {a} - {b} by regrouping a ten or a hundred when a column is too small.",
        "Use place value to work out {a} - {b}.",
    ),
    "hard": (
        "Work out {a} - {b} without a calculator, then check your answer by adding.",
        "Find {a} - {b}, and check your answer by adding.",
        "Subtract {b} from {a}, then check your answer by adding.",
    ),
}
# easy only: the larger number may be written first or second
_SUB_DIFFERENCE = ("Find the difference between {a} and {b}.",)


def _digits(n: int) -> list[int]:
    """Ones, tens, hundreds and thousands digits."""
    return [(n // 10**i) % 10 for i in range(4)]


def _carries(a: int, b: int) -> list[bool]:
    """For ones, tens and hundreds: does that column carry into the next one when `a + b` is added by columns?"""
    da, db = _digits(a), _digits(b)
    carry, out = 0, []
    for i in range(3):
        total = da[i] + db[i] + carry
        carry = 1 if total >= 10 else 0
        out.append(bool(carry))
    return out


def _borrows(a: int, b: int) -> list[bool]:
    """For ones, tens and hundreds: does that column borrow from the next one when `b` is taken from `a` by
    columns?"""
    da, db = _digits(a), _digits(b)
    borrow, out = 0, []
    for i in range(3):
        diff = da[i] - db[i] - borrow
        borrow = 1 if diff < 0 else 0
        out.append(bool(borrow))
    return out


def _sum_by_place_value(a: int, b: int) -> int:
    da, db = _digits(a), _digits(b)
    return sum((da[i] + db[i]) * 10**i for i in range(4))


def _difference_by_counting_up(a: int, b: int) -> int:
    """How far from `b` up to `a`: to the next ten, then hundreds, then the rest."""
    steps = 0
    here = b
    for unit in (10, 100):
        nxt = (here // unit + 1) * unit
        if here % unit != 0 and nxt <= a:
            steps += nxt - here
            here = nxt
    return steps + (a - here)


# -- error simulations -----------------------------------------------------------------------------------------


def _no_carry_sum(a: int, b: int) -> int:
    da, db = _digits(a), _digits(b)
    return sum(((da[i] + db[i]) % 10) * 10**i for i in range(4))


def _smaller_from_larger(a: int, b: int) -> int:
    da, db = _digits(a), _digits(b)
    return sum(abs(da[i] - db[i]) * 10**i for i in range(4))


def _borrow_not_paid(a: int, b: int) -> int:
    """Borrow ten in a column that is too small but never reduce the column it was borrowed from."""
    da, db = _digits(a), _digits(b)
    total = 0
    for i in range(4):
        top = da[i] + 10 if da[i] < db[i] and i < 3 else da[i]
        total += (top - db[i]) * 10**i
    return total


def _reverse_digits(n: int) -> int:
    return int(str(n)[::-1])


# -- drawing -------------------------------------------------------------------------------------------------


def _draw_pair(rng, accept, lo_a, hi_a, lo_b, hi_b, tries: int = 4000) -> tuple[int, int]:
    for _ in range(tries):
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        if accept(a, b):
            return a, b
    check(False, f"{TOPIC}: could not draw a pair satisfying the tier's regrouping")


def _easy_add(rng):
    return "+", _draw_pair(rng, lambda a, b: _carries(a, b) == [True, False, False] and a + b < 100, 11, 89, 11, 89)


def _easy_sub(rng):
    return "-", _draw_pair(
        rng, lambda a, b: _borrows(a, b) == [True, False, False] and a - b >= 10 and b >= 11, 21, 99, 11, 89
    )


def _medium_add(rng):
    def ok(a, b):
        return _carries(a, b) == [True, True, False] and a + b < 1000

    b_hi = rng.choice((99, 899))
    return "+", _draw_pair(rng, ok, 100, 899, 11, b_hi)


def _medium_sub(rng):
    def ok(a, b):
        return _borrows(a, b) == [True, True, False] and _digits(a)[1] != 0 and a - b >= 20

    b_hi = rng.choice((99, 899))
    return "-", _draw_pair(rng, ok, 201, 999, 11, b_hi)


def _hard_add(rng):
    return "+", _draw_pair(
        rng, lambda a, b: _carries(a, b) == [True, True, True] and (a % 100) != 0 and (b % 100) != 0, 200, 999, 200, 999
    )


def _hard_sub_through_zero(rng):
    def ok(a, b):
        return _borrows(a, b) == [True, True, False] and _digits(a)[1] == 0 and a - b >= 20

    b_hi = rng.choice((99, 899))
    return "-", _draw_pair(rng, ok, 201, 909, 11, b_hi)


def _hard_sub_round(rng):
    """A round minuend such as 400, so the borrow has to pass through two zeros."""
    for _ in range(4000):
        a = 100 * rng.randint(2, 9)
        b = rng.randint(11, a - 20)
        if b % 10 != 0:
            return "-", (a, b)
    check(False, f"{TOPIC}: could not draw a round minuend")


_DRAWS = {
    "easy": (_easy_add, _easy_sub),
    "medium": (_medium_add, _medium_sub),
    "hard": (_hard_add, _hard_sub_through_zero, _hard_sub_round),
}

# What each tier's regrouping must look like, re-checked on the drawn pair.
_EXPECTED_CARRIES = {"easy": [True, False, False], "medium": [True, True, False], "hard": [True, True, True]}
_EXPECTED_BORROWS = {"easy": [True, False, False], "medium": [True, True, False], "hard": [True, True, False]}


def _candidates_add(a: int, b: int, key: int) -> list[int]:
    carry_in = [i for i, c in enumerate(_carries(a, b)) if c]  # a carry is made OUT of these columns
    out = [_no_carry_sum(a, b)]
    for col in carry_in:
        out.append(key - 10 ** (col + 1))  # that carry was never added to the next column
        out.append(key + 10 ** (col + 1))  # that carry was added twice
    out += [key - 1, key + 1, _reverse_digits(key)]
    return out


def _candidates_sub(a: int, b: int, key: int) -> list[int]:
    borrow_out = [i for i, c in enumerate(_borrows(a, b)) if c]
    out = [_smaller_from_larger(a, b), _borrow_not_paid(a, b)]
    for col in borrow_out:
        out.append(key + 10 ** (col + 1))  # the lender was not reduced
        out.append(key - 10 ** (col + 1))  # the lender was reduced twice
    out += [key - 1, key + 1]
    return out


def _build(rng, tier: str = "easy") -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    op, (a, b) = rng.choice(_DRAWS[tier])(rng)

    if op == "+":
        key = a + b
        check_equivalent(key, _sum_by_place_value(a, b), f"{TOPIC} {tier}: place-value sum disagrees for {a} + {b}")
        check(_carries(a, b) == _EXPECTED_CARRIES[tier], f"{TOPIC} {tier}: {a} + {b} has the wrong carry pattern")
        candidates = _candidates_add(a, b, key)
        working = [f"Add column by column, regrouping when a column makes 10 or more: {a} + {b} = {key}."]
        shown_a, shown_b = (a, b) if rng.random() < 0.5 else (b, a)
        text = phrase(rng, _ADD[tier], a=shown_a, b=shown_b)
    else:
        key = a - b
        check_equivalent(key, _difference_by_counting_up(a, b), f"{TOPIC} {tier}: counting up disagrees for {a} - {b}")
        check_equivalent(b + key, a, f"{TOPIC} {tier}: {b} + {key} should make {a}")
        check(_borrows(a, b) == _EXPECTED_BORROWS[tier], f"{TOPIC} {tier}: {a} - {b} has the wrong borrow pattern")
        candidates = _candidates_sub(a, b, key)
        working = [
            f"Subtract column by column, regrouping when a digit is too small: {a} - {b} = {key}.",
            f"Check: {b} + {key} = {a}.",
        ]
        templates = _SUB[tier] + (_SUB_DIFFERENCE if tier == "easy" else ())
        pattern = rng.choice(templates)
        if pattern in _SUB_DIFFERENCE:
            shown_a, shown_b = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            shown_a, shown_b = a, b
        text = pattern.format(a=shown_a, b=shown_b)

    chosen = pick_ranked_distractors(rng, key, candidates, lo=1, hi=1999, fillers=(2, 3, 5))
    check_distinct(key, *chosen, msg=f"{TOPIC} {tier}: a distractor equals the key")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        distractors=[str(c) for c in chosen],
        working=working,
    )


@generator(
    TOPIC,
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def multidigit_add_sub_regroup(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
