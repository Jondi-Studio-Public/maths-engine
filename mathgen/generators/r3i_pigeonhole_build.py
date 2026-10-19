"""Generates `r3i_pigeonhole` questions (easy, medium, hard) and the exam chain `r3i_exam_pigeonhole`.

Skill ``pigeonhole-and-binomial-identities`` (``discrete-mathematics``, answer ``expression``): the pigeonhole
principle (the least number in some box, the number of items that forces a repeat), and simplifying binomial and
factorial expressions with Pascal's identity. A count is typed as a whole number, an identity as an expression in n.

* easy: ``n`` objects into ``k`` boxes (the least that must share a box), or the smallest group that forces two in the
  same category.
* medium: how many to draw to be sure of ``m`` of one kind (with or without colours that run short), a factorial ratio
  ``(n + s)!/(n - t)!``, or ``C(n + 1, 2) - C(n, 2)``.
* hard: how many numbers from ``{1, ..., N}`` force two consecutive numbers, or two that add to ``N + 1``; and Pascal's
  identity as an expression, ``C(n, 2) + C(n + 1, 2)``, ``C(n + 1, 3) - C(n, 3)``, ``C(n, 2) + C(n, 3)``, or the value of
  ``C(a, b) + C(a, b + 1)``.
* exam (``r3i_exam_pigeonhole``): students in k houses; (a) the least number that some house must have, (b) the group size
  that forces one more in a house (follows through from (a)), (c) ``C(n + 1, 2) - C(n, 2)``, (d) the number of new pairs
  formed when a student joins a group of that size (follows through from (b)).

Counting answers are checked by exhaustive search (every way of filling the boxes, every subset of ``{1..N}``); the
binomial expressions are checked against ``math.comb`` at several values of n.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._r3i_kit import (
    expr_dist,
    expr_fields,
    expr_num_cands,
    expr_num_fields,
    nice,
    num_cands,
    num_fields,
    pick_text,
    retry_short,
)

N = sympy.Symbol("n")

_THINGS = [
    ("students", "classes", "a class"),
    ("letters", "pigeonholes", "a pigeonhole"),
    ("parcels", "shelves", "a shelf"),
    ("coins", "pockets", "a pocket"),
    ("books", "boxes", "a box"),
]
_CATEGORIES = [
    ("birth month", 12, "people"),
    ("day of the week of their birth", 7, "people"),
    ("suit", 4, "cards"),
    ("score", 6, "rolls of a die"),
    ("last digit", 10, "phone numbers"),
    ("remainder when divided by 5", 5, "integers"),
    ("remainder when divided by 8", 8, "integers"),
    ("remainder when divided by 3", 3, "integers"),
]


def _cands(key, pairs):
    """Wrong expressions that are different functions of n from the key, as tagged typed answers."""
    out, seen = [], {nice(key)}
    for expr, tag in pairs:
        if expr is None:
            continue
        expr = sympy.expand(expr)
        text = nice(expr)
        if sympy.expand(expr - key) == 0 or text in seen:
            continue
        seen.add(text)
        out.append(expr_dist(text, tag))
    return out


def _binom(top: sympy.Expr, k: int) -> sympy.Expr:
    return sympy.expand(sympy.expand_func(sympy.binomial(top, k)))


def _verify_identity(text_value, key, what: str) -> None:
    """`text_value(n)` (computed with math.comb or factorials) equals the key at several n."""
    fn = sympy.lambdify(N, key, "math")
    for n in (4, 5, 6, 9, 13):
        check(abs(fn(n) - text_value(n)) < 1e-9, f"{what}: the key fails at n = {n}")


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy_boxes(rng):
    things, holders, _ = rng.choice(_THINGS)
    for _ in attempts(200, "more objects than boxes"):
        k, n = rng.randint(3, 9), rng.randint(10, 60)
        if n > k and n % k != 0 or rng.random() < 0.15 and n > k:
            break
    answer = next(m for m in range(1, n + 1) if k * m >= n)  # the least capacity per box that can hold everything
    check(answer == -(-n // k), "ceiling differs from the capacity search")
    fields = expr_num_fields(answer)
    cands = expr_num_cands(
        [
            (n // k, "r3i-pigeonhole-rounded-down"),
            (answer + 1, "r3i-pigeonhole-extra-one-added"),
            (k, "r3i-pigeonhole-box-count-given"),
            (n - k, "r3i-pigeonhole-objects-minus-boxes"),
            (Fraction(n, k), "r3i-pigeonhole-quotient-not-rounded") if n % k else (None, ""),
        ],
        answer,
    )
    return dict(
        text=f"{n} {things} are placed in {k} {holders}. What is the greatest number m such that some one of the {holders} must hold at least m of them?",
        working=[f"{n} ÷ {k} = {n / k:.2f}, which rounds up to {answer}: some one has at least {answer}."],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _easy_repeat(rng):
    category, k, items = rng.choice(_CATEGORIES)
    answer = k + 1
    fields = expr_num_fields(answer)
    cands = expr_num_cands(
        [
            (k, "r3i-pigeonhole-one-more-forgotten"),
            (2 * k, "r3i-pigeonhole-pair-count-doubled"),
            (k - 1, "r3i-pigeonhole-one-fewer"),
            (2 * k + 1, "r3i-pigeonhole-pair-count-doubled"),
        ],
        answer,
    )
    return dict(
        text=f"How many {items} are needed to be certain that two of them have the same {category}?",
        working=[
            f"There are {k} possible values. With {k} {items} they could all differ, so one more, {answer}, is needed."
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _easy(rng):
    return _easy_boxes(rng) if rng.random() < 0.6 else _easy_repeat(rng)


# -- medium -------------------------------------------------------------------------------------------------------


def _brute_safe(caps, m: int) -> int:
    """The most items that can be held with at most m - 1 of each kind and at most `caps[i]` of kind i (exhaustive)."""
    return max(sum(c) for c in itertools.product(*[range(min(cap, m - 1) + 1) for cap in caps]))


def _medium_same_kind(rng):
    category, k, items = rng.choice(_CATEGORIES)
    m = rng.randint(3, 8)
    answer = k * (m - 1) + 1
    fields = expr_num_fields(answer)
    cands = expr_num_cands(
        [
            (k * m, "r3i-pigeonhole-times-m-not-m-minus-one"),
            (k * (m - 1), "r3i-pigeonhole-one-more-forgotten"),
            (k * m + 1, "r3i-pigeonhole-times-m-not-m-minus-one"),
            (k + m - 1, "r3i-pigeonhole-added-not-multiplied"),
            (m * (k - 1) + 1, "r3i-pigeonhole-categories-and-count-swapped"),
        ],
        answer,
    )
    return dict(
        text=f"How many {items} are needed to be certain that {m} of them have the same {category}?",
        working=[
            f"The worst case has {m - 1} of each of the {k} values, which is {k * (m - 1)} {items}.",
            f"One more forces {m} of one value: {answer}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium_colours(rng):
    for _ in attempts(500, "colours of which at least one runs short"):
        k = rng.randint(3, 4)
        m = rng.randint(3, 5)
        counts = [rng.randint(1, 9) for _ in range(k)]
        safe = sum(min(c, m - 1) for c in counts)
        if any(c < m - 1 for c in counts) and safe + 1 <= sum(counts) and k * (m - 1) + 1 != safe + 1:
            break
    check(_brute_safe(counts, m) == safe, "the exhaustive search differs from the sum of capped counts")
    answer = safe + 1
    fields = expr_num_fields(answer)
    names = ["red", "blue", "green", "black"][:k]
    cands = expr_num_cands(
        [
            (k * (m - 1) + 1, "r3i-pigeonhole-short-colour-ignored"),
            (safe, "r3i-pigeonhole-one-more-forgotten"),
            (sum(counts) - 1, "r3i-pigeonhole-total-less-one"),
            (sum(counts), "r3i-pigeonhole-whole-bag-taken"),
            (answer + 1, "r3i-pigeonhole-extra-one-added"),
        ],
        answer,
    )
    listing = ", ".join(f"{c} {name}" for c, name in zip(counts, names))
    return dict(
        text=(
            f"A bag holds {listing} counters. Counters are drawn one at a time without looking. "
            f"What is the fewest that must be drawn to be certain of {m} of the same colour?"
        ),
        working=[
            f"The most that can be drawn with fewer than {m} of every colour is "
            + " + ".join(f"min({c}, {m - 1})" for c in counts)
            + f" = {safe}.",
            f"The next draw forces {m} of one colour: {answer}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium_factorial(rng):
    for _ in attempts(100, "a ratio of at least two factors"):
        s, t = rng.choice([0, 1, 2]), rng.choice([1, 2, 3])
        if s + t >= 2:
            break
    factors = [N + j for j in range(-t + 1, s + 1)]
    key = sympy.expand(sympy.prod(factors))
    _verify_identity(lambda n: math.factorial(n + s) // math.factorial(n - t), key, "factorial ratio")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (sympy.prod(factors[1:]), "r3i-pigeonhole-factor-dropped"),
            (sympy.prod(factors[:-1]), "r3i-pigeonhole-factor-dropped"),
            (sympy.prod(factors) * (N - t), "r3i-pigeonhole-extra-factor-kept"),
            (sympy.prod(factors) / (N - t + 1), "r3i-pigeonhole-factor-dropped"),
            (sympy.prod([N + j for j in range(-t, s + 1)]), "r3i-pigeonhole-extra-factor-kept"),
            ((N + s) / (N - t), "r3i-pigeonhole-factorials-cancelled-as-numbers"),
            (sympy.prod([N + j for j in range(-t + 2, s + 1)]), "r3i-pigeonhole-factor-dropped"),
        ],
    )
    top = "n" if s == 0 else f"n + {s}"
    bottom = "n" if t == 0 else f"n - {t}"
    return dict(
        text=f"Simplify ({top})! ÷ ({bottom})! as an expression in n.",
        working=[f"({top})! ÷ ({bottom})! = " + " × ".join(f"({nice(f)})" for f in factors) + f" = {nice(key)}."],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium_difference(rng):
    for _ in attempts(100, "a difference of two binomials with a shift"):
        up, down = rng.choice([0, 1, 2, 3]), rng.choice([0, 1, 2])
        if up + down >= 1:
            break
    top = N + up if up else N
    low = N - down if down else N
    key = sympy.expand(_binom(top, 2) - _binom(low, 2))
    _verify_identity(lambda n: math.comb(n + up, 2) - math.comb(n - down, 2), key, "difference")
    fields = expr_fields(key)
    cands = _cands(
        key,
        [
            (key + 1, "r3i-pigeonhole-pascal-index-off-by-one"),
            (key - 1, "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(top, 2), "r3i-pigeonhole-one-term-only"),
            (_binom(top, 2) + _binom(low, 2), "r3i-pigeonhole-difference-taken-as-sum"),
            (
                sympy.expand(key * (N + 1) / N) if up + down == 1 else sympy.Integer(up + down),
                "r3i-pigeonhole-difference-taken-as-gap",
            ),
            (_binom(top - low, 2) if top != low else None, "r3i-pigeonhole-difference-taken-as-gap"),
        ],
    )
    left = f"C(n + {up}, 2)" if up else "C(n, 2)"
    right = f"C(n - {down}, 2)" if down else "C(n, 2)"
    return dict(
        text=f"Simplify {left} - {right} as an expression in n.",
        working=[
            f"{left} = ({nice(top)})({nice(top - 1)})/2 and {right} = ({nice(low)})({nice(low - 1)})/2.",
            f"The difference is {nice(key)}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _medium(rng):
    roll = rng.random()
    return (
        _medium_same_kind(rng)
        if roll < 0.28
        else _medium_colours(rng)
        if roll < 0.56
        else _medium_factorial(rng)
        if roll < 0.82
        else _medium_difference(rng)
    )


# -- hard ---------------------------------------------------------------------------------------------------------


def _brute_force_hard(size: int, kind: str) -> int:
    """The fewest numbers from {1..size} that force the property, by exhaustive search over subsets: one more than the
    largest subset without it."""
    best = 0
    for mask in range(1 << size):
        if kind == "consecutive":
            ok = mask & (mask >> 1) == 0
        else:
            ok = all(not ((mask >> (i - 1)) & 1 and (mask >> (size - i)) & 1) for i in range(1, size + 1))
        if ok:
            best = max(best, bin(mask).count("1"))
    return best + 1


def _hard_integers(rng):
    consecutive = rng.random() < 0.55
    size = rng.randint(6, 13) if consecutive else rng.choice([6, 8, 10, 12])
    answer = (-(-size // 2) + 1) if consecutive else size // 2 + 1
    check(
        _brute_force_hard(size, "consecutive" if consecutive else "sum") == answer,
        "exhaustive search differs from the pairing count",
    )
    fields = expr_num_fields(answer)
    cands = expr_num_cands(
        [
            (answer - 1, "r3i-pigeonhole-one-more-forgotten"),
            (answer + 1, "r3i-pigeonhole-extra-one-added"),
            (size // 2 + 2, "r3i-pigeonhole-extra-one-added"),
            (size - 1, "r3i-pigeonhole-pairs-not-formed"),
            (size, "r3i-pigeonhole-pairs-not-formed"),
            (size // 2, "r3i-pigeonhole-one-more-forgotten"),
        ],
        answer,
    )
    if consecutive:
        text = f"Numbers are chosen from the whole numbers 1 to {size}. What is the fewest numbers that must be chosen to be certain that two of them are consecutive integers?"
        working = [
            f"Without two consecutive integers you can take every other number, {-(-size // 2)} of them.",
            f"So {answer} must be chosen to force a consecutive pair.",
        ]
    else:
        text = f"Numbers are chosen from the whole numbers 1 to {size}. What is the fewest numbers that must be chosen to be certain that two different ones add up to {size + 1}?"
        working = [
            f"Pair the numbers as (1, {size}), (2, {size - 1}), ... : there are {size // 2} pairs, each adding to {size + 1}.",
            f"Choosing {size // 2} numbers can take one from every pair, so {answer} are needed.",
        ]
    return dict(text=text, working=working, distractors=pick_text(fields, cands), **fields)


def _hard_identity(rng):
    shape = rng.choice(["sum-square", "difference-3", "pascal-23", "numeric"])
    if shape == "numeric":
        a = rng.randint(8, 16)
        b = rng.randint(2, a - 3)
        answer = math.comb(a, b) + math.comb(a, b + 1)
        check(answer == math.comb(a + 1, b + 1), "Pascal's identity")
        fields = expr_num_fields(answer)
        cands = expr_num_cands(
            [
                (math.comb(a + 1, b), "r3i-pigeonhole-pascal-index-off-by-one"),
                (math.comb(a + 1, b + 2), "r3i-pigeonhole-pascal-index-off-by-one"),
                (math.comb(a, b) * math.comb(a, b + 1), "r3i-pigeonhole-sum-taken-as-product"),
                (math.comb(a + 2, b + 1), "r3i-pigeonhole-pascal-index-off-by-one"),
                (math.comb(a, b + 1), "r3i-pigeonhole-one-term-only"),
            ],
            answer,
        )
        return dict(
            text=f"Use Pascal's identity to find the value of C({a}, {b}) + C({a}, {b + 1}).",
            working=[f"C({a}, {b}) + C({a}, {b + 1}) = C({a + 1}, {b + 1}) = {answer}."],
            distractors=pick_text(fields, cands),
            **fields,
        )
    if shape == "sum-square":
        text, key = "Simplify C(n, 2) + C(n + 1, 2) as an expression in n.", N**2
        _verify_identity(lambda n: math.comb(n, 2) + math.comb(n + 1, 2), key, shape)
        wrong = [
            (N**2 / 2, "r3i-pigeonhole-one-term-only"),
            (N * (N + 1) / 2, "r3i-pigeonhole-one-term-only"),
            (N**2 - N, "r3i-pigeonhole-pascal-index-off-by-one"),
            (N**2 + N, "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(N + 2, 2), "r3i-pigeonhole-pascal-index-off-by-one"),
        ]
        working = ["C(n, 2) = n(n - 1)/2 and C(n + 1, 2) = (n + 1)n/2.", "Their sum is n(n - 1)/2 + n(n + 1)/2 = n²."]
    elif shape == "difference-3":
        text, key = "Simplify C(n + 1, 3) - C(n, 3) as an expression in n.", _binom(N, 2)
        _verify_identity(lambda n: math.comb(n + 1, 3) - math.comb(n, 3), key, shape)
        wrong = [
            (_binom(N + 1, 2), "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(N, 3), "r3i-pigeonhole-difference-not-pascal"),
            (_binom(N - 1, 2), "r3i-pigeonhole-pascal-index-off-by-one"),
            (N, "r3i-pigeonhole-difference-taken-as-lower-index"),
            (N * (N - 1), "r3i-pigeonhole-halving-forgotten"),
        ]
        working = [
            "By Pascal's identity C(n + 1, 3) = C(n, 3) + C(n, 2).",
            f"So the difference is C(n, 2) = {nice(key)}.",
        ]
    else:
        text, key = "Simplify C(n, 2) + C(n, 3) as an expression in n.", _binom(N + 1, 3)
        _verify_identity(lambda n: math.comb(n, 2) + math.comb(n, 3), key, shape)
        wrong = [
            (_binom(N + 1, 2), "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(N, 3), "r3i-pigeonhole-one-term-only"),
            (_binom(N + 1, 4), "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(N + 2, 3), "r3i-pigeonhole-pascal-index-off-by-one"),
            (_binom(N, 2) * _binom(N, 3), "r3i-pigeonhole-sum-taken-as-product"),
        ]
        working = [
            "By Pascal's identity C(n, 2) + C(n, 3) = C(n + 1, 3).",
            f"C(n + 1, 3) = (n + 1)n(n - 1)/6 = {nice(key)}.",
        ]
    key = sympy.expand(key)
    fields = expr_fields(key)
    return dict(text=text, working=working, distractors=pick_text(fields, _cands(key, wrong)), **fields)


def _hard(rng):
    return _hard_integers(rng) if rng.random() < 0.45 else _hard_identity(rng)


@generator(
    "r3i_pigeonhole",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_pigeonhole(rng, tier="easy"):
    """The pigeonhole principle and binomial identities; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3i_exam_pigeonhole", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_pigeonhole(rng, tier="exam"):
    """Exam chain: students in houses, then the pairs formed when one more joins."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a group that does not divide evenly into the houses"):
        k, total = rng.randint(5, 12), rng.randint(15, 80)
        if total % k != 0 and total > k:
            break
    a = next(m for m in range(1, total + 1) if k * m >= total)
    b = k * a + 1
    pairs_before, pairs_after = math.comb(b, 2), math.comb(b + 1, 2)
    check(pairs_after - pairs_before == b, "new pairs differ from the group size")
    check(_brute_safe([2, 2, 2], 3) == 6, "exhaustive search sanity check")
    key = N
    _verify_identity(lambda n: math.comb(n + 1, 2) - math.comb(n, 2), key, "exam difference")
    f_a, f_b, f_d = num_fields(a), num_fields(b), num_fields(b)
    f_c = expr_fields(key)
    return dict(
        text=phrase(
            rng,
            [
                "There are {total} students in a year group. Each student belongs to exactly one of {k} houses.",
                "There are {total} students in Year 9 at a school, and every one of them is placed in exactly one of "
                "{k} houses.",
                "There are {total} students in a year group, and each is a member of exactly one of {k} houses.",
            ],
            total=total,
            k=k,
        ),
        parts=[
            dict(
                prompt="What is the greatest number m such that some house must have at least m of these students?",
                marks=1,
                sensible_range=(1, 20),
                working=[f"{total} ÷ {k} = {total / k:.2f}, which rounds up to {a}."],
                distractors=pick_text(
                    f_a,
                    num_cands(
                        [
                            (total // k, "r3i-pigeonhole-rounded-down"),
                            (a + 1, "r3i-pigeonhole-extra-one-added"),
                            (k, "r3i-pigeonhole-box-count-given"),
                            (total - k, "r3i-pigeonhole-objects-minus-boxes"),
                        ],
                        a,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="How many students would a year group need, to be certain that some house has at least one more student than your answer to part (a)?",
                marks=2,
                follow_through=f"{k}*part_a+1",
                sensible_range=(1, 200),
                working=[
                    f"The worst case has {a} students in each of the {k} houses, {k * a}. One more forces {a + 1}: {b}."
                ],
                distractors=pick_text(
                    f_b,
                    num_cands(
                        [
                            (k * (a + 1), "r3i-pigeonhole-times-m-not-m-minus-one"),
                            (k * a, "r3i-pigeonhole-one-more-forgotten"),
                            (k * (a + 1) + 1, "r3i-pigeonhole-times-m-not-m-minus-one"),
                            (a + k, "r3i-pigeonhole-added-not-multiplied"),
                        ],
                        b,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="When one more student joins a group of n students, the number of pairs of students rises from C(n, 2) to C(n + 1, 2). Simplify C(n + 1, 2) - C(n, 2).",
                marks=2,
                working=["C(n + 1, 2) = n(n + 1)/2 and C(n, 2) = n(n - 1)/2, so the difference is n."],
                distractors=pick_text(
                    f_c,
                    _cands(
                        key,
                        [
                            (N + 1, "r3i-pigeonhole-pascal-index-off-by-one"),
                            (N - 1, "r3i-pigeonhole-pascal-index-off-by-one"),
                            (N * (N - 1) / 2, "r3i-pigeonhole-difference-not-pascal"),
                            (sympy.Integer(1), "r3i-pigeonhole-difference-taken-as-one"),
                        ],
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Hence find the number of new pairs of students formed when one more student joins a group of the size you found in part (b).",
                marks=1,
                follow_through="part_b",
                sensible_range=(1, 200),
                working=[f"The number of new pairs is n = {b}."],
                distractors=pick_text(
                    f_d,
                    num_cands(
                        [
                            (b + 1, "r3i-pigeonhole-pascal-index-off-by-one"),
                            (b - 1, "r3i-pigeonhole-pascal-index-off-by-one"),
                            (pairs_after, "r3i-pigeonhole-difference-not-pascal"),
                            (pairs_before, "r3i-pigeonhole-difference-not-pascal"),
                        ],
                        b,
                    ),
                ),
                **f_d,
            ),
        ],
    )
