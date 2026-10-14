"""Generates `uc2-generating-function-partition-counting` questions (easy, medium, hard) and the exam chain
`exd_x_r4j_generating_function_partition_counting`.

``uc2-generating-function-partition-counting`` (``discrete-mathematics``, answer ``number``): counting the ways to write
a total as an unordered sum, which is a coefficient of a product of generating-function factors. Nothing in the stem names
a generating function; the learner sets up the product (or counts directly) and reads off the coefficient.

* easy: partitions of n into parts of three allowed sizes, any number of each: [x^n] 1/((1 - x^a)(1 - x^b)(1 - x^c)).
* medium: partitions of n into different parts, each from 1 to m: [x^n] (1 + x)(1 + x^2)...(1 + x^m).
* hard: change-making with Australian coins and notes where two denominations are in limited supply: unlimited factors
  1/(1 - x^d) for the plentiful ones and (1 + x^d + ... + x^(cd)) for the limited ones.
* exam (``exd_x_r4j_generating_function_partition_counting``): making a total from 1, 3, 4 and 7 point items with at most two 1-point items. (a) the ways with no 1-point item, (b) with exactly one, (c) with exactly two, (d) the total, which follows through
  from (a) to (c).

Every count is computed twice: from the product of truncated integer series, and by direct recursion over how many of
each part to use. The tests add a third route (SymPy's series expansion).
"""

from __future__ import annotations

from math import gcd

from ..core import attempts, check, generator
from ._r4j_kit import keep_distinct, say

_TAGS = {
    "required": "partition-part-size-required",
    "ordered": "partition-order-counted",
    "omitted": "partition-part-size-omitted",
    "index": "gf-coefficient-index-off-by-one",
    "repeats": "partition-repeats-allowed",
    "range": "partition-part-range-off-by-one",
    "ignored": "partition-limit-ignored",
    "cap": "partition-cap-off-by-one",
    "exactly": "partition-at-most-as-exactly",
    "cases": "partition-cases-not-added",
}


# -- the count, two ways ---------------------------------------------------------------------------------------------


def _series(factors, n):
    """The coefficients up to x^n of a product of factors. A factor is ``(size, cap)``: ``cap`` None for 1/(1 - x^size),
    or a whole number for 1 + x^size + ... + x^(cap * size)."""
    out = [1] + [0] * n
    for size, cap in factors:
        if cap is None:
            for i in range(size, n + 1):
                out[i] += out[i - size]
        else:
            new = [0] * (n + 1)
            for i, c in enumerate(out):
                if c:
                    for j in range(cap + 1):
                        if i + j * size <= n:
                            new[i + j * size] += c
            out = new
    return out


def _direct(factors, n):
    """The same count by recursion on how many of each size to use."""

    def go(i, left):
        if i == len(factors):
            return 1 if left == 0 else 0
        size, cap = factors[i]
        top = left // size if cap is None else min(cap, left // size)
        return sum(go(i + 1, left - j * size) for j in range(top + 1))

    return go(0, n)


def _count(factors, n):
    if n < 0:
        return 0
    value = _series(factors, n)[n]
    check(value == _direct(factors, n), "series and direct count disagree")
    return value


def _compositions(sizes, n):
    """Ordered sums of the allowed sizes that total n."""
    c = [1] + [0] * n
    for i in range(1, n + 1):
        c[i] = sum(c[i - s] for s in sizes if i >= s)
    return c[n]


def _ordered_distinct(m, n):
    """Ordered lists of different integers from 1 to m with total n."""
    from math import factorial

    total = 0
    for mask in range(1, 1 << m):
        parts = [i + 1 for i in range(m) if mask >> i & 1]
        if sum(parts) == n:
            total += factorial(len(parts))
    return total


def _cands(value, items):
    return keep_distinct(value, [(v, tag, str(v)) for v, tag in items if v is not None and v > 0])


def _out(text, latex, value, distractors, working):
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        working=working,
        distractors=distractors,
    )


def _sizes_text(sizes):
    return ", ".join(str(s) for s in sizes[:-1]) + f" or {sizes[-1]}"


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(2000, "a three-size partition count that is neither tiny nor huge"):
        sizes = sorted(rng.sample(range(2, 10), 3))
        if gcd(gcd(sizes[0], sizes[1]), sizes[2]) != 1:
            continue
        n = rng.randint(18, 30)
        factors = [(s, None) for s in sizes]
        value = _count(factors, n)
        if 5 <= value <= 16 and _count(factors[:2], n) != value:
            break
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"In how many ways can {n} be written as a sum in which every term is {_sizes_text(sizes)}? "
            "Sums that differ only in the order of their terms count as the same way."
        )
    elif form == 1:
        text, latex = say(f"How many partitions of {n} have every part equal to {_sizes_text(sizes)}?")
    else:
        text, latex = say(
            f"A bag holds plenty of {sizes[0]}-point, {sizes[1]}-point and {sizes[2]}-point tokens. In how many different "
            f"ways can tokens be chosen to total exactly {n} points? Only how many of each kind are chosen matters."
        )
    items = [
        (_count(factors, n - sum(sizes)), _TAGS["required"]),
        (_compositions(sizes, n), _TAGS["ordered"]),
        (_count(factors[:2], n), _TAGS["omitted"]),
        (_count(factors, n - 1), _TAGS["index"]),
        (_count(factors, n + 1), _TAGS["index"]),
    ]
    a, b, c = sizes
    return _out(
        text,
        latex,
        value,
        _cands(value, items),
        [
            f"Each size can be used any number of times, so the generating function is 1/((1 − x^{a})(1 − x^{b})(1 − x^{c})).",
            f"Its coefficient of x^{n} is the number of solutions of {a}i + {b}j + {c}k = {n} in non-negative integers.",
            f"Counting them gives {value}.",
        ],
    )


# -- medium --------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(2000, "a distinct-parts count that is neither tiny nor huge"):
        m = rng.randint(6, 8)
        n = rng.randint(m + 2, m * (m + 1) // 2 - 4)
        factors = [(s, 1) for s in range(1, m + 1)]
        value = _count(factors, n)
        if 5 <= value <= 16:
            break
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"In how many ways can {n} be written as a sum of different whole numbers, each from 1 to {m}? "
            "The order of the terms does not matter."
        )
    elif form == 1:
        text, latex = say(f"How many partitions of {n} have distinct parts, none larger than {m}?")
    else:
        text, latex = say(
            f"{m} cards are numbered 1 to {m}. In how many ways can some of the cards be chosen so that their numbers "
            f"add to {n}?"
        )
    items = [
        (_count([(s, None) for s in range(1, m + 1)], n), _TAGS["repeats"]),
        (_ordered_distinct(m, n), _TAGS["ordered"]),
        (_count([(s, 1) for s in range(1, m)], n), _TAGS["range"]),
        (_count([(s, 1) for s in range(1, m + 2)], n), _TAGS["range"]),
        (_count(factors, n + 1), _TAGS["index"]),
    ]
    return _out(
        text,
        latex,
        value,
        _cands(value, items),
        [
            f"Each of 1, 2, …, {m} is either used once or not at all, giving the factor (1 + x^j) for each j.",
            f"The generating function is (1 + x)(1 + x²)…(1 + x^{m}); we need its coefficient of x^{n}.",
            f"Listing the subsets of {{1, …, {m}}} with total {n} gives {value}.",
        ],
    )


# -- hard ----------------------------------------------------------------------------------------------------------

_NAMES = {
    1: ("one-dollar coins", "one-dollar coin"),
    2: ("two-dollar coins", "two-dollar coin"),
    5: ("five-dollar notes", "five-dollar note"),
    10: ("ten-dollar notes", "ten-dollar note"),
}


def _hard(rng):
    for _ in attempts(4000, "a limited-supply change count of moderate size"):
        limited = sorted(rng.sample([1, 2, 5, 10], 2))
        caps = {d: rng.randint(2, 4) for d in limited}
        n = rng.randint(17, 34)
        factors = [(d, caps.get(d)) for d in (1, 2, 5, 10)]
        value = _count(factors, n)
        if 8 <= value <= 60 and value != _count([(d, None) for d in (1, 2, 5, 10)], n):
            break
    plenty = [d for d in (1, 2, 5, 10) if d not in caps]
    have = " and ".join(f"only {caps[d]} of the {_NAMES[d][0]}" for d in limited)
    lots = " and ".join(_NAMES[d][0] for d in plenty)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"A kiosk must give exactly ${n} in change using one-dollar coins, two-dollar coins, five-dollar notes and ten-dollar notes. "
            f"It has {have}, but plenty of {lots}. How many different combinations of coins and notes make exactly ${n}?"
        )
    elif form == 1:
        text, latex = say(
            f"The school canteen till holds {have}, and plenty of {lots}. It holds no other money. "
            f"In how many different ways can the till pay out exactly ${n}?"
        )
    else:
        text, latex = say(
            f"Count the ways to make ${n} from one-dollar coins, two-dollar coins, five-dollar notes and ten-dollar notes when the supply is "
            f"{have}, with plenty of {lots}. Two ways are the same if they use the same number of each."
        )
    unlimited = [(d, None) for d in (1, 2, 5, 10)]
    items = [
        (_count(unlimited, n), _TAGS["ignored"]),
        (_count([(d, caps[d] + 1 if d in caps else None) for d in (1, 2, 5, 10)], n), _TAGS["cap"]),
        (_count([(d, caps[d] - 1 if d in caps else None) for d in (1, 2, 5, 10)], n), _TAGS["cap"]),
        (_count([f for f in factors if f[0] != 10], n), _TAGS["omitted"]),
        (_count(factors, n - 1), _TAGS["index"]),
        (
            _count([(d, None) if d not in caps else (d, 1) for d in (1, 2, 5, 10)], n - sum(d * caps[d] for d in caps))
            if n >= sum(d * caps[d] for d in caps)
            else None,
            _TAGS["exactly"],
        ),
    ]
    parts_text = []
    for d in (1, 2, 5, 10):
        if d in caps:
            parts_text.append(f"(1 + x^{d} + … + x^{caps[d] * d})")
        else:
            parts_text.append(f"1/(1 − x^{d})")
    return _out(
        text,
        latex,
        value,
        _cands(value, items),
        [
            "Each denomination contributes a factor in x^(its value): 1/(1 − x^d) when supply is plentiful, "
            "and 1 + x^d + … + x^(cd) when only c are available.",
            f"The generating function is {' · '.join(parts_text)}.",
            f"Its coefficient of x^{n} is {value}.",
        ],
    )


@generator(
    "r4j_generating_function_partition_counting",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_generating_function_partition_counting(rng, tier="easy"):
    """Counting unordered sums by a generating-function coefficient; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator("exd_x_r4j_generating_function_partition_counting", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_generating_function_partition_counting(rng, tier="exam"):
    """Exam chain: ways to pay with no, one and two 1-point items, then the total."""
    check(tier == "exam", f"unknown tier {tier!r}")
    base = [(3, None), (4, None), (7, None)]
    for _ in attempts(2000, "an amount whose three counts are distinct"):
        n = rng.randint(22, 70)
        a, b, c = _count(base, n), _count(base, n - 1), _count(base, n - 2)
        if len({a, b, c}) == 3 and min(a, b, c) >= 4 and n - 2 >= 14:
            break
    total = a + b + c
    check(total == _count([(1, 2), *base], n), "the case split disagrees with the full generating function")
    scene = rng.randrange(3)
    noun = ("1-credit token", "1-point play", "1-point stamp")[scene]
    if scene == 0:
        stem, stem_tex = say(
            f"Leah needs exactly {n} credits on an arcade card. She tops up with tokens worth 1, 3, 4 and 7 credits, "
            "but the machine takes at most two 1-credit tokens per top-up."
        )
    elif scene == 1:
        stem, stem_tex = say(
            f"A game awards {n} points to the winner. Scores come only from 1-point, 3-point, 4-point and 7-point plays, "
            "and a team may make at most two 1-point plays."
        )
    else:
        stem, stem_tex = say(
            f"Tickets for a school fair are made up from stamps worth 1, 3, 4 and 7 points. A ticket must total exactly "
            f"{n} points and may carry at most two 1-point stamps."
        )

    def part(prompt, value, marks, items, working, follow=None):
        d = dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            marks=marks,
            working=working,
            distractors=_cands(value, items),
            sensible_range=(0, None),
        )
        if follow:
            d["follow_through"] = follow
        return d

    parts = [
        part(
            f"How many ways are there if no {noun} is used?",
            a,
            2,
            [
                (b, _TAGS["index"]),
                (_count(base[:2], n), _TAGS["omitted"]),
                (_count(base, n - 14), _TAGS["required"]),
                (_count(base, n + 1), _TAGS["index"]),
            ],
            [
                f"Count the solutions of 3i + 4j + 7k = {n}: the coefficient of x^{n} in 1/((1 − x³)(1 − x⁴)(1 − x⁷)) is {a}."
            ],
        ),
        part(
            f"How many ways are there if exactly one {noun} is used?",
            b,
            2,
            [
                (a, _TAGS["index"]),
                (c, _TAGS["index"]),
                (_count(base[:2], n - 1), _TAGS["omitted"]),
                (_count(base, n - 2 - 1), _TAGS["index"]),
            ],
            [f"The other items make {n - 1} points: the coefficient of x^{n - 1} is {b}."],
        ),
        part(
            f"How many ways are there if exactly two {noun}s are used?",
            c,
            2,
            [
                (b, _TAGS["index"]),
                (a, _TAGS["index"]),
                (_count(base[:2], n - 2), _TAGS["omitted"]),
                (_count(base, n - 4), _TAGS["index"]),
            ],
            [f"The other items make {n - 2} points: the coefficient of x^{n - 2} is {c}."],
        ),
        part(
            "Hence find the total number of ways.",
            total,
            2,
            [
                (a + b, _TAGS["cases"]),
                (a + b + c + _count(base, n - 3), _TAGS["cap"]),
                (max(a, b, c), _TAGS["cases"]),
                (_count([(1, None), *base], n), _TAGS["ignored"]),
            ],
            [f"The cases are disjoint, so add them: {a} + {b} + {c} = {total}."],
            follow="part_a + part_b + part_c",
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)
