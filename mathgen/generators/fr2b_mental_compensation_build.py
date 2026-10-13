"""Generates `fr2b_mental_compensation` questions for `f10-mental-compensation-partition` (AC9M3A02). `answer: number`.

Scope: adding and subtracting in your head by compensation (change a number to a nearby round one, then put back the
difference) and by partitioning (work with the tens, then the ones). Every question has two parts: (a) the
intermediate value that the strategy gives, and (b) the answer to the original calculation, which follows through
from (a) by the adjustment that the strategy needs.

Ladder (reasoning escalation, not size):
  easy:   addition. A number one or two below a multiple of ten is rounded up, or a two-digit number is added tens
          first then ones. The adjustment is taken off (compensation) or the ones are added on (partitioning).
  medium: subtraction. The number taken away is rounded up, so the adjustment is added back; or tens are taken first
          and the ones are larger than the ones of the starting number, so the subtraction has to bridge a ten.
  hard:   three-digit numbers, and both numbers rounded at once: two adjustments to combine, and in a subtraction
          they point the same way even though one number is rounded down and the other up.

Each key is the calculation done directly, and the follow-through expression is checked against it.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._fr2b_kit import build, int_dist, tagged

_TIERS = ("easy", "medium", "hard")


def _dists(answer, options):
    """Tagged integer distractors from (value, misconception) pairs, positives only."""
    return tagged(str(answer), [int_dist(v, m) for v, m in options if v > 0], need=3)


def _part(prompt, answer, options, *, follow=None, marks=1):
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        marks=marks,
        distractors=_dists(answer, options),
    )
    if follow:
        out["follow_through"] = follow
    return out


def _compensate_one(rng, sub, unit, low, high):
    """Round one number up by 1..3 (or 1..2) to a multiple of `unit`, then undo it."""
    for _ in attempts(500, "a near-multiple"):
        e = rng.randint(1, 2 if unit == 10 else 3)
        r = rng.randint(2, 6) * unit
        b = r - e
        a = rng.randint(low, high)
        if a % unit == 0 or b < 2 * unit:
            continue
        if sub and a - b <= 0:
            continue
        if not sub and a + b > (99 if unit == 10 else 999):
            continue
        break
    op = "-" if sub else "+"
    exact = a - b if sub else a + b
    inter = a - r if sub else a + r
    check(exact == (inter + e if sub else inter - e), "the adjustment does not give the exact answer")
    check(inter == (a - r if sub else a + r), "the intermediate value is wrong")
    stem = f"To work out {a} {op} {b} in your head, round {b} up to {r} first."
    p_a = _part(
        f"What is {a} {op} {r}?",
        inter,
        [
            (inter + unit, "fr2b-mental-place-slip"),
            (inter - unit, "fr2b-mental-place-slip"),
            (inter + 1, "fr2b-mental-place-slip"),
            ((a - r // unit) if sub else (a + r // unit), "fr2b-mental-tens-treated-as-ones"),
        ],
    )
    if sub:
        p_b = _part(
            f"Use your answer to work out {a} - {b}.",
            exact,
            [
                (inter - e, "fr2b-compensation-adjusts-the-wrong-way"),
                (inter, "fr2b-forgets-to-compensate"),
                (inter + 2 * e, "fr2b-compensation-adjusts-twice"),
                (exact + unit, "fr2b-mental-place-slip"),
                (exact - unit, "fr2b-mental-place-slip"),
            ],
            follow=f"part_a + {e}",
        )
    else:
        p_b = _part(
            f"Use your answer to work out {a} + {b}.",
            exact,
            [
                (inter + e, "fr2b-compensation-adjusts-the-wrong-way"),
                (inter, "fr2b-forgets-to-compensate"),
                (inter - 2 * e, "fr2b-compensation-adjusts-twice"),
                (exact + unit, "fr2b-mental-place-slip"),
                (exact - unit, "fr2b-mental-place-slip"),
            ],
            follow=f"part_a - {e}",
        )
    working = [
        f"{b} is {e} less than {r}, so {a} {op} {r} is {e} {'too much taken away' if sub else 'too much added'}.",
        f"{a} {op} {r} = {inter}, then {inter} {'+' if sub else '-'} {e} = {exact}.",
    ]
    return dict(text=stem, parts=[p_a, p_b], working=working)


def _partition(rng, sub):
    for _ in attempts(500, "a partition draw"):
        a = rng.randint(31, 99) if sub else rng.randint(21, 59)
        t, u = rng.randint(2, 5), rng.randint(4, 9)
        b = 10 * t + u
        if a == b or (sub and (a - b <= 0 or u <= a % 10)):
            continue
        if not sub and a + b > 99:
            continue
        break
    op = "-" if sub else "+"
    inter = a - 10 * t if sub else a + 10 * t
    exact = a - b if sub else a + b
    check(exact == (inter - u if sub else inter + u), "partition does not give the exact answer")
    stem = f"To work out {a} {op} {b} in your head, {'take away' if sub else 'add'} the tens of {b} first and then the ones."
    p_a = _part(
        f"What is {a} {op} {10 * t}?",
        inter,
        [
            (inter + 10, "fr2b-mental-place-slip"),
            (inter - 10, "fr2b-mental-place-slip"),
            ((a - t) if sub else (a + t), "fr2b-mental-tens-treated-as-ones"),
            (inter + (1 if sub else -1), "fr2b-mental-place-slip"),
        ],
    )
    if sub:
        smaller_from_larger = 10 * (inter // 10) + (u - inter % 10)
        options = [
            (smaller_from_larger, "fr2b-mental-subtracts-smaller-ones-from-larger"),
            (inter + u, "fr2b-mental-adds-the-ones-when-subtracting"),
            (inter - u + 10, "fr2b-mental-carry-slip"),
            (inter - u - 10, "fr2b-mental-carry-slip"),
            (exact + 1, "fr2b-mental-place-slip"),
        ]
        follow = f"part_a - {u}"
    else:
        options = [
            (inter + 10 * u, "fr2b-mental-ones-treated-as-tens"),
            (inter - u, "fr2b-mental-subtracts-the-ones-when-adding"),
            (inter + u + 10, "fr2b-mental-carry-slip"),
            (inter, "fr2b-mental-ones-left-out"),
            (exact - 10, "fr2b-mental-carry-slip"),
        ]
        follow = f"part_a + {u}"
    p_b = _part(f"Use your answer to work out {a} {op} {b}.", exact, options, follow=follow)
    working = [f"{a} {op} {10 * t} = {inter}.", f"{inter} {op} {u} = {exact}."]
    return dict(text=stem, parts=[p_a, p_b], working=working)


def _easy(rng):
    return _compensate_one(rng, False, 10, 21, 78) if rng.random() < 0.5 else _partition(rng, False)


def _medium(rng):
    return _compensate_one(rng, True, 10, 61, 99) if rng.random() < 0.5 else _partition(rng, True)


def _both(rng, sub):
    for _ in attempts(500, "two near-hundreds"):
        e1, e2 = rng.randint(1, 3), rng.randint(1, 3)
        x, y = rng.randint(2, 8), rng.randint(2, 8)
        if sub:
            a, b = 100 * x + e1, 100 * y - e2
            if a - b <= 0 or x - y < 1:
                continue
            inter, exact, adj = 100 * x - 100 * y, a - b, e1 + e2
        else:
            a, b = 100 * x - e1, 100 * y - e2
            inter, exact, adj = 100 * x + 100 * y, a + b, -(e1 + e2)
        break
    check(exact == inter + adj, "the two adjustments do not give the exact answer")
    op = "-" if sub else "+"
    stem = f"To work out {a} {op} {b} in your head, round both numbers to the nearest hundred first."
    p_a = _part(
        f"What do you get when you {'subtract' if sub else 'add'} the two rounded numbers?",
        inter,
        [
            (inter + 100, "fr2b-mental-place-slip"),
            (inter - 100, "fr2b-mental-place-slip"),
            (inter + 10, "fr2b-mental-place-slip"),
            (100 * (x + y) if sub else abs(100 * x - 100 * y), "fr2b-mental-uses-the-wrong-operation"),
        ],
    )
    if sub:
        options = [
            (inter - e1 - e2, "fr2b-compensation-adjusts-the-wrong-way"),
            (inter + e1, "fr2b-compensation-adjusts-only-one-number"),
            (inter + e2, "fr2b-compensation-adjusts-only-one-number"),
            (inter, "fr2b-forgets-to-compensate"),
            (inter + e1 - e2, "fr2b-compensation-adjusts-the-wrong-way"),
            (exact + 100, "fr2b-mental-place-slip"),
        ]
        follow = f"part_a + {e1 + e2}"
    else:
        options = [
            (inter + e1 + e2, "fr2b-compensation-adjusts-the-wrong-way"),
            (inter - e1, "fr2b-compensation-adjusts-only-one-number"),
            (inter - e2, "fr2b-compensation-adjusts-only-one-number"),
            (inter, "fr2b-forgets-to-compensate"),
            (exact + 100, "fr2b-mental-place-slip"),
        ]
        follow = f"part_a - {e1 + e2}"
    p_b = _part(f"Use your answer to work out {a} {op} {b}.", exact, options, follow=follow)
    if sub:
        working = [
            f"{a} rounds down to {100 * x}, which is {e1} too small. {b} rounds up to {100 * y}, which is {e2} too big.",
            f"Both changes make the answer smaller than it should be, so add {e1} + {e2} = {e1 + e2}: {inter} + {e1 + e2} = {exact}.",
        ]
    else:
        working = [
            f"{a} and {b} are each a little less than their rounded numbers, by {e1} and {e2}.",
            f"{inter} - {e1} - {e2} = {exact}.",
        ]
    return dict(text=stem, parts=[p_a, p_b], working=working)


def _hard(rng):
    pick = rng.random()
    if pick < 0.2:
        return _compensate_one(rng, False, 100, 135, 789)
    return _both(rng, pick < 0.6)


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_mental_compensation", level=3, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_mental_compensation(rng, tier="easy"):
    """Mental addition and subtraction by compensation or partitioning, in two parts; tiers in the docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
