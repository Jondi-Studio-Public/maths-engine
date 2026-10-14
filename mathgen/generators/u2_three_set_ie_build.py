"""Generates `u2_three_set_ie` questions and the exam chain `exd_x_u2_three_set_ie`.

``ucv1-three-set-inclusion-exclusion-derangements`` (``discrete-mathematics``, answer ``number``): inclusion-exclusion
for three sets, including the case where one region's count is missing, and its use for derangements and for
permutations with forbidden positions.

* easy: the size of the triple intersection from the three set sizes, the three pairwise intersections and the union.
* medium: from a total and the sizes of the sets, the pairwise intersections and the triple intersection, the number
  in none of the sets or in exactly two of them; or the number of derangements of 4, 5 or 6 objects.
* hard: the integers up to N divisible by none of three given numbers (least common multiples, not products); the
  permutations with exactly k fixed points; the permutations of n objects in which j named objects avoid their own
  positions.
* exam: a year-level subject survey with a stated number in none of the three subjects. (a) the triple intersection,
  (b) exactly one subject, (c) exactly two subjects, (d) the derangements of the students in exactly two subjects
  (follows through from (c)).

Every count is re-derived by listing: the survey from the eight region counts the question was built from, the
divisibility count by testing every integer, the permutation counts by listing all permutations.
"""

from __future__ import annotations

import itertools
from math import comb, factorial, lcm

from ..core import attempts, check, generator
from ._u2_kit import int_d, keep, number_part, number_q, redraws

TOPIC = "u2_three_set_ie"
EXAM_TOPIC = "exd_x_u2_three_set_ie"


def derangements(n):
    """D(n): permutations of n objects with no object in its own place."""
    return sum((-1) ** k * comb(n, k) * factorial(n - k) for k in range(n + 1))


def _brute_derangements(n):
    return sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))


class Survey:
    """Eight region counts of a three-set Venn diagram, and every figure the questions quote."""

    def __init__(self, a, b, c, ab, ac, bc, abc, none):
        self.a, self.b, self.c = a, b, c  # only-regions
        self.ab, self.ac, self.bc = ab, ac, bc  # exactly-two regions
        self.abc, self.none = abc, none
        self.size_a = a + ab + ac + abc
        self.size_b = b + ab + bc + abc
        self.size_c = c + ac + bc + abc
        self.pair_ab, self.pair_ac, self.pair_bc = ab + abc, ac + abc, bc + abc
        self.union = a + b + c + ab + ac + bc + abc
        self.total = self.union + none
        self.exactly_one = a + b + c
        self.exactly_two = ab + ac + bc
        # Independent route: the formula and the regions agree.
        s = self.size_a + self.size_b + self.size_c
        p = self.pair_ab + self.pair_ac + self.pair_bc
        check(self.union == s - p + abc, "inclusion-exclusion differs from the region sum")

    @property
    def singles(self):
        return self.size_a + self.size_b + self.size_c

    @property
    def pairs(self):
        return self.pair_ab + self.pair_ac + self.pair_bc


def _survey(rng, two_range=(3, 14), low_triple=2):
    for _ in attempts(500, "a three-set survey"):
        only = [rng.randint(4, 22) for _ in range(3)]
        pair = [rng.randint(0, 6) for _ in range(3)]
        if not two_range[0] <= sum(pair) <= two_range[1] or 0 in pair:
            continue
        triple = rng.randint(low_triple, 6)
        none = rng.randint(3, 15)
        return Survey(*only, *pair, triple, none)
    raise AssertionError  # pragma: no cover


_CONTEXTS = (
    ("students at a Hobart college", "take", ("Mathematical Methods", "Physics", "Chemistry")),
    ("members of a Perth surf club", "train in", ("surfing", "paddling", "rescue skills")),
    ("households in a Darwin suburb", "own", ("a bicycle", "a kayak", "a caravan")),
    ("staff at an Adelaide hospital", "work", ("night shifts", "weekend shifts", "public holidays")),
)


def _names(rng):
    return rng.choice(_CONTEXTS)


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    s = _survey(rng)
    who, verb, (x, y, z) = _names(rng)
    t = s.abc
    cands = [
        int_d(s.pairs - s.union, "inex3-forgot-singles"),
        int_d(s.singles - s.union, "inex3-sign-reversed"),
        int_d(-t, "inex3-sign-reversed"),
        int_d(s.union - s.singles - s.pairs, "inex3-pairs-added"),
    ]
    text = (
        f"All {s.union} {who} {verb} at least one of {x}, {y} and {z}. "
        f"{s.size_a} {verb} {x}, {s.size_b} {verb} {y} and {s.size_c} {verb} {z}. "
        f"{s.pair_ab} {verb} both {x} and {y}, {s.pair_ac} {verb} both {x} and {z}, and "
        f"{s.pair_bc} {verb} both {y} and {z}. How many {verb} all three?"
    )
    working = [
        f"|A ∪ B ∪ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∩ B ∩ C|, so {s.union} = {s.singles} - {s.pairs} + |A ∩ B ∩ C|.",
        f"|A ∩ B ∩ C| = {s.union} - {s.singles} + {s.pairs} = {t}.",
    ]
    return number_q(text, t, working, keep(str(t), cands, 4))


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_survey(rng):
    s = _survey(rng)
    who, verb, (x, y, z) = _names(rng)
    ask_none = rng.random() < 0.5
    stem = (
        f"A survey of {s.total} {who} asked about {x}, {y} and {z}. "
        f"{s.size_a} {verb} {x}, {s.size_b} {verb} {y} and {s.size_c} {verb} {z}. "
        f"{s.pair_ab} {verb} both {x} and {y}, {s.pair_ac} {verb} both {x} and {z}, and "
        f"{s.pair_bc} {verb} both {y} and {z}. {s.abc} {verb} all three. "
    )
    if ask_none:
        value = s.none
        text = stem + f"How many {verb} none of the three?"
        cands = [
            int_d(s.total - (s.singles - s.pairs), "inex3-forgot-triple"),
            int_d(s.total - (s.singles - s.pairs + 2 * s.abc), "inex3-sign-reversed"),
            int_d(s.total - s.singles, "inex3-forgot-overlaps"),
            int_d(s.union, "inex3-union-not-complement"),
        ]
        working = [
            f"|A ∪ B ∪ C| = {s.singles} - {s.pairs} + {s.abc} = {s.union}.",
            f"{s.total} - {s.union} = {value} are involved in none of the three.",
        ]
    else:
        value = s.exactly_two
        text = stem + f"How many {verb} exactly two of the three?"
        cands = [
            int_d(s.pairs, "inex3-exactly-two-keeps-triple"),
            int_d(s.pairs - s.abc, "inex3-exactly-two-keeps-triple"),
            int_d(s.pairs - 2 * s.abc, "inex3-exactly-two-keeps-triple"),
            int_d(s.pairs - 3 * s.abc + s.abc, "inex3-exactly-two-keeps-triple"),
        ]
        working = [
            f"Each pairwise count includes the {s.abc} in all three, so exactly two of a pair is that count minus {s.abc}.",
            f"The three pairs give {s.pairs} - 3 × {s.abc} = {value}.",
        ]
    # Independent route: the regions the survey was built from.
    check(value == (s.none if ask_none else s.ab + s.ac + s.bc), "region count differs")
    return number_q(text, value, working, keep(str(value), cands, 4))


def _medium_derangement(rng):
    n = rng.choice([4, 5, 6])
    value = derangements(n)
    check(value == _brute_derangements(n), "brute-force derangements differ")
    partial = [sum((-1) ** k * comb(n, k) * factorial(n - k) for k in range(upto + 1)) for upto in range(1, n)]
    cands = [
        int_d(partial[1], "derange-truncated-sum"),
        int_d(factorial(n - 1), "derange-first-not-fixed-only"),
        int_d(partial[2] if len(partial) > 2 else partial[0], "derange-truncated-sum"),
        int_d(factorial(n) - factorial(n - 1), "derange-first-not-fixed-only"),
    ]
    items = rng.choice(["letters", "tickets", "parcels"])
    text = (
        f"{n} {items}, each addressed to a different person, are put one into each of {n} envelopes labelled with "
        f"those {n} names, in a random order. In how many of the {factorial(n)} arrangements does no {items[:-1]} "
        "go into the envelope bearing its own name?"
    )
    terms = " ".join((("+" if k % 2 == 0 else "-") + f" {comb(n, k)}·{n - k}!") if k else f"{n}!" for k in range(n + 1))
    working = [f"By inclusion-exclusion over the {n} places, D_{n} = {terms} = {value}."]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _medium(rng):
    return rng.choice([_medium_survey, _medium_survey, _medium_derangement])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard_divisibility(rng):
    p, q, r = rng.choice(
        [(2, 3, 5), (2, 3, 7), (3, 4, 5), (4, 6, 9), (6, 10, 15), (2, 5, 7), (4, 6, 10), (6, 8, 9), (3, 5, 7)]
    )
    n = rng.choice([100, 120, 150, 180, 200, 240, 300])
    f = lambda d: n // d  # noqa: E731
    value = n - f(p) - f(q) - f(r) + f(lcm(p, q)) + f(lcm(p, r)) + f(lcm(q, r)) - f(lcm(p, q, r))
    check(value == sum(1 for k in range(1, n + 1) if k % p and k % q and k % r), "direct count differs")
    prod = lambda *d: n // (d[0] * (d[1] if len(d) > 1 else 1) * (d[2] if len(d) > 2 else 1))  # noqa: E731
    wrong_products = n - f(p) - f(q) - f(r) + prod(p, q) + prod(p, r) + prod(q, r) - prod(p, q, r)
    cands = [
        int_d(n - f(p) - f(q) - f(r) + f(lcm(p, q)) + f(lcm(p, r)) + f(lcm(q, r)), "inex3-forgot-triple"),
        int_d(wrong_products, "inex3-product-not-lcm"),
        int_d(n - f(p) - f(q) - f(r), "inex3-forgot-overlaps"),
        int_d(n - value, "inex3-union-not-complement"),
    ]
    text = f"How many integers from 1 to {n} inclusive are divisible by none of {p}, {q} and {r}?"
    working = [
        f"Count the multiples of {p}, {q}, {r}: {f(p)}, {f(q)}, {f(r)}. Pairs (by least common multiple {lcm(p, q)}, {lcm(p, r)}, {lcm(q, r)}): {f(lcm(p, q))}, {f(lcm(p, r))}, {f(lcm(q, r))}. All three (lcm {lcm(p, q, r)}): {f(lcm(p, q, r))}.",
        f"{n} - {f(p) + f(q) + f(r)} + {f(lcm(p, q)) + f(lcm(p, r)) + f(lcm(q, r))} - {f(lcm(p, q, r))} = {value}.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _hard_fixed_points(rng):
    n = rng.choice([5, 6, 6, 7])
    k = rng.choice([1, 2, 3])
    value = comb(n, k) * derangements(n - k)
    if n <= 6:
        check(
            value == sum(1 for p in itertools.permutations(range(n)) if sum(p[i] == i for i in range(n)) == k),
            "listing differs",
        )
    cands = [
        int_d(comb(n, k) * factorial(n - k), "derange-at-least-k-fixed"),
        int_d(derangements(n - k), "derange-choose-fixed-forgotten"),
        int_d(comb(n, k) * derangements(n), "derange-wrong-remaining"),
        int_d(derangements(n) * k, "derange-wrong-remaining"),
    ]
    text = (
        f"{n} friends each bring a different umbrella to a Melbourne cafe and the umbrellas are returned at random, "
        f"one to each person. In how many of the {factorial(n)} ways do exactly {k} friend{'s' if k > 1 else ''} get "
        "their own umbrella back?"
    )
    working = [
        f"Choose which {k} friend{'s' if k > 1 else ''} keep their own umbrella: C({n}, {k}) = {comb(n, k)} ways.",
        f"The other {n - k} must all get a wrong one: D_{n - k} = {derangements(n - k)}. So {comb(n, k)} × {derangements(n - k)} = {value}.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _hard_forbidden(rng):
    n = rng.choice([5, 6])
    j = rng.choice([2, 3, 4])
    value = sum((-1) ** i * comb(j, i) * factorial(n - i) for i in range(j + 1))
    check(
        value == sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(j))),
        "listing differs",
    )
    cands = [
        int_d(factorial(n) - j * factorial(n - 1), "surj-count-no-add-back"),
        int_d(factorial(n - j) * derangements(j), "derange-wrong-remaining"),
        int_d(sum((-1) ** i * comb(j, i) * factorial(n - i) for i in range(min(j, 2))), "derange-truncated-sum"),
        int_d(factorial(n) - factorial(n - j), "inex3-forgot-overlaps"),
    ]
    names = ["Asha", "Ben", "Chloe", "Dev"][:j]
    seats = ", ".join(f"seat {i + 1}" for i in range(j))
    text = (
        f"{n} people are to be seated in a row of {n} numbered seats. {', '.join(names[:-1])} and {names[-1]} each "
        f"refuse to take their own numbered seat ({names[0]} seat 1, {names[1]} seat 2"
        + (f", {names[2]} seat 3" if j > 2 else "")
        + (f", {names[3]} seat 4" if j > 3 else "")
        + f"); the others have no preference. How many seatings are possible?"
    )
    del seats
    working = [
        f"Count all {n}! seatings and remove those where at least one of the {j} sits in the forbidden seat, by inclusion-exclusion.",
        f"{' '.join((('+' if i % 2 == 0 else '-') + f' {comb(j, i)}·{n - i}!') if i else f'{n}!' for i in range(j + 1))} = {value}.",
    ]
    return number_q(text, value, working, keep(str(value), cands, 4))


def _hard(rng):
    return rng.choice([_hard_divisibility, _hard_fixed_points, _hard_forbidden, _hard_divisibility])(rng)


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_three_set_ie(rng, tier="easy"):
    """Three-set inclusion-exclusion, derangements and forbidden positions; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_three_set_ie(rng, tier="exam"):
    """A three-subject survey with a stated 'none': triple overlap, exactly one, exactly two, then a derangement."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a survey whose exactly-two count is 3 to 6"):
        s = _survey(rng, two_range=(3, 6))
        if len({s.ab, s.ac, s.bc}) >= 2:
            break
    who, (x, y, z) = rng.choice(
        [
            ("students at a Hobart college", ("Mathematical Methods", "Physics", "Chemistry")),
            ("students at a Perth senior high school", ("Specialist Mathematics", "Biology", "Economics")),
            ("first-year students at a Brisbane university", ("Calculus", "Statistics", "Programming")),
        ]
    )
    k = s.exactly_two
    stem = rng.choice(
        [
            f"A survey of {s.total} {who} found that {s.none} take none of {x}, {y} and {z}. "
            f"{s.size_a} take {x}, {s.size_b} take {y} and {s.size_c} take {z}. "
            f"{s.pair_ab} take both {x} and {y}, {s.pair_ac} take both {x} and {z}, and {s.pair_bc} take both {y} and {z}.",
            f"Of {s.total} {who} who were surveyed, {s.none} take none of {x}, {y} and {z}. "
            f"The numbers taking each subject are {s.size_a} for {x}, {s.size_b} for {y} and {s.size_c} for {z}. "
            f"Taking both {x} and {y} are {s.pair_ab}; both {x} and {z}, {s.pair_ac}; both {y} and {z}, {s.pair_bc}.",
            f"{s.total} {who} were asked which of {x}, {y} and {z} they take. {s.none} take none of them. "
            f"The counts are: {x} {s.size_a}, {y} {s.size_b}, {z} {s.size_c}, {x} and {y} together {s.pair_ab}, "
            f"{x} and {z} together {s.pair_ac}, {y} and {z} together {s.pair_bc}.",
        ]
    )
    t = s.abc
    a_cands = [
        int_d(-t, "inex3-sign-reversed"),
        int_d(s.union - s.singles - s.pairs, "inex3-pairs-added"),
        int_d(s.total - s.singles + s.pairs, "inex3-union-not-complement"),
    ]
    b_cands = [
        int_d(s.singles - s.pairs, "inex3-exactly-one-keeps-overlaps"),
        int_d(s.singles - 2 * s.pairs + t, "inex3-exactly-one-keeps-overlaps"),
        int_d(s.singles - 2 * s.pairs + 2 * t, "inex3-exactly-one-keeps-overlaps"),
    ]
    c_cands = [
        int_d(s.pairs, "inex3-exactly-two-keeps-triple"),
        int_d(s.pairs - t, "inex3-exactly-two-keeps-triple"),
        int_d(s.pairs - 2 * t, "inex3-exactly-two-keeps-triple"),
    ]
    d_val = derangements(k)
    check(d_val == _brute_derangements(k), "brute-force derangements differ")
    d_cands = [
        int_d(factorial(k - 1), "derange-first-not-fixed-only"),
        int_d(factorial(k) - 1, "derange-first-not-fixed-only"),
        int_d(derangements(k - 1), "derange-wrong-remaining"),
        int_d(factorial(k), "inj-count-all-functions"),
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                f"How many take all three of {x}, {y} and {z}?",
                t,
                3,
                [
                    f"{s.total} - {s.none} = {s.union} take at least one subject.",
                    f"{s.union} = {s.singles} - {s.pairs} + t gives t = {t}.",
                ],
                keep(str(t), a_cands),
            ),
            number_part(
                "How many take exactly one of the three subjects?",
                s.exactly_one,
                2,
                [f"{s.singles} - 2 × {s.pairs} + 3 × {t} = {s.exactly_one}."],
                keep(str(s.exactly_one), b_cands),
                follow_through=f"{s.singles} - 2*{s.pairs} + 3*part_a",
            ),
            number_part(
                "How many take exactly two of the three subjects?",
                k,
                2,
                [f"{s.pairs} - 3 × {t} = {k}."],
                keep(str(k), c_cands),
                follow_through=f"{s.pairs} - 3*part_a",
            ),
            number_part(
                "Each student who takes exactly two of the subjects hands in a worksheet, and the worksheets are "
                "returned one to each of these students. In how many ways can they be returned so that no student "
                "gets their own?",
                d_val,
                3,
                [f"There are {k} such students, so the count is the number of derangements D_{k} = {d_val}."],
                keep(str(d_val), d_cands),
                follow_through="floor(factorial(part_c)/E + 1/2)",
            ),
        ],
    )
