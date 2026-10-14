"""Generates `uc2-partial-order-hasse` questions (easy, medium, hard) and the exam chain `exd_x_r4j_partial_order_hasse`.

``uc2-partial-order-hasse`` (``discrete-mathematics``, answer ``set``): reading a finite poset whose elements are some of
the divisors of a number, ordered by divisibility. The poset is given as a list of integers; no diagram is drawn, so
the learner builds the Hasse diagram's structure from the divisibility relation.

* easy: the minimal elements, or the maximal elements, of the set under divisibility.
* medium: the elements a given element covers (or is covered by), with the cover relation defined in the stem; the set
  has gaps, so a cover in the divisor lattice need not be a cover in the set.
* hard: the elements that lie on at least one chain of greatest length.
* exam (``exd_x_r4j_partial_order_hasse``): (a) the maximal elements, (b) the size of a longest chain, (c) the size of a
  largest antichain, (d) the fewest chains that cover the set, which by Dilworth's theorem follows through from (c).

Every set is computed by a direct definition, and the chain and antichain sizes are each computed twice (a longest-chain
dynamic programme against a search over chains; a search over subsets against Dilworth's theorem through a matching in
the tests).
"""

from __future__ import annotations

import re

from ..answers import Distractor
from ..core import attempts, check, generator
from ._r4j_kit import keep_distinct
from ._r4j_kit import say as _say

_TAGS = {
    "swapped": "hasse-minimal-maximal-swapped",
    "least": "hasse-least-confused-with-minimal",
    "greatest": "hasse-greatest-confused-with-maximal",
    "prime": "hasse-minimal-taken-as-prime",
    "below": "hasse-cover-vs-below",
    "direction": "hasse-cover-direction-swapped",
    "gaps": "hasse-cover-ignores-missing-elements",
    "itself": "hasse-includes-element-itself",
    "all": "hasse-all-elements-on-chain",
    "ends": "hasse-endpoints-only",
    "single": "hasse-single-chain",
    "edges": "hasse-chain-length-edges",
    "chain_anti": "hasse-chain-antichain-swapped",
    "anti_min": "hasse-antichain-is-minimal-elements",
    "anti_max": "hasse-antichain-is-maximal-elements",
    "anti_comp": "hasse-antichain-complement-of-chain",
    "mirsky": "hasse-dilworth-mirsky-swapped",
    "each": "hasse-chain-cover-one-each",
}

_BASES = (60, 72, 90, 120, 180, 210)


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _lt(a, b):
    return a != b and b % a == 0


# -- the structure, by definition -----------------------------------------------------------------------------------


def minimal(s):
    return sorted(x for x in s if not any(_lt(y, x) for y in s))


def maximal(s):
    return sorted(x for x in s if not any(_lt(x, y) for y in s))


def lower_covers(s, x):
    below = [y for y in s if _lt(y, x)]
    return sorted(y for y in below if not any(_lt(y, z) for z in below))


def upper_covers(s, x):
    above = [y for y in s if _lt(x, y)]
    return sorted(y for y in above if not any(_lt(z, y) for z in above))


def _longest_ending(s):
    best = {}
    for x in sorted(s):
        best[x] = 1 + max((best[y] for y in best if _lt(y, x)), default=0)
    return best


def longest_chain_size(s):
    return max(_longest_ending(s).values())


def chains_of_size(s, size):
    """Every chain (sorted tuple) with exactly `size` elements, by depth-first search."""
    out = []

    def grow(chain):
        if len(chain) == size:
            out.append(tuple(chain))
            return
        for y in sorted(s):
            if _lt(chain[-1], y):
                grow([*chain, y])

    for x in sorted(s):
        grow([x])
    return out


def longest_chain_elements(s):
    """The elements on at least one longest chain, by the heights from below and above."""
    down = _longest_ending(s)
    up = {}
    for x in sorted(s, reverse=True):
        up[x] = 1 + max((up[y] for y in up if _lt(x, y)), default=0)
    top = max(down.values())
    return sorted(x for x in s if down[x] + up[x] - 1 == top)


def largest_antichain_size(s):
    items = sorted(s)
    n = len(items)
    comparable = [0] * n
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i != j and (_lt(a, b) or _lt(b, a)):
                comparable[i] |= 1 << j
    best = 0
    # extend an antichain item by item: a subset is an antichain when no member is comparable to another
    stack = [(0, 0, 0)]  # (next index, members mask, size)
    while stack:
        i, mask, size = stack.pop()
        best = max(best, size)
        for j in range(i, n):
            if not comparable[j] & mask:
                stack.append((j + 1, mask | 1 << j, size + 1))
    return best


def _random_set(rng, size):
    base = rng.choice(_BASES)
    divs = _divisors(base)
    s = sorted(rng.sample(divs, size))
    return base, s


def _set_text(values):
    return "{" + ", ".join(str(v) for v in values) + "}"


def _set_distractors(answer, items):
    return keep_distinct(
        answer, [(sorted(v), tag, _set_text(sorted(v))) for v, tag in items if v], key=lambda v: tuple(sorted(v))
    )


def _single(text, latex, answer, distractors, working):
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="set",
        answer=list(answer),
        answer_text=_set_text(answer),
        working=working,
        distractors=distractors,
    )


_SET_LITERAL = re.compile(r"((?:S = )?\{[^{}]*\})")


def say(*parts):
    """`_r4j_kit.say` with every set literal in the prose, such as `S = {2, 3, 6}`, typeset as maths."""
    out = []
    for part in parts:
        if not isinstance(part, str):
            out.append(part)
            continue
        for piece in _SET_LITERAL.split(part):
            if _SET_LITERAL.fullmatch(piece):
                out.append((piece, piece.replace("{", "\\{").replace("}", "\\}")))
            elif piece:
                out.append(piece)
    return _say(*out)


_ORDER = "ordered by divisibility (a ≤ b means a divides b)"


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(2000, "a set whose minimal or maximal elements are not the obvious ones"):
        base, s = _random_set(rng, rng.randint(8, 10))
        want_min = rng.random() < 0.5
        answer = minimal(s) if want_min else maximal(s)
        primes = [x for x in s if len(_divisors(x)) == 2]
        other = maximal(s) if want_min else minimal(s)
        if len(answer) >= 2 and answer != other and (want_min and answer != primes or not want_min):
            break
    word = "minimal" if want_min else "maximal"
    st = _set_text(s)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(f"The set S = {st} is {_ORDER}. Find the set of {word} elements of S.")
    elif form == 1:
        text, latex = say(
            f"Consider the poset (S, ≤) where S = {st} and a ≤ b means a divides b. Which elements of S are {word}?"
        )
    else:
        text, latex = say(
            f"In the divisibility order on S = {st}, list every {word} element. Give your answer as a set."
        )
    if want_min:
        items = [(other, _TAGS["swapped"]), ([min(s)], _TAGS["least"]), (primes, _TAGS["prime"])]
        rule = "no other element of S divides it"
    else:
        items = [
            (other, _TAGS["swapped"]),
            ([max(s)], _TAGS["greatest"]),
            ([x for x in s if x not in answer][-2:] + answer[:1], None),
        ]
        rule = "it divides no other element of S"
    items = [(v, tag) for v, tag in items if tag]
    working = [
        f"An element is {word} when {rule}.",
        f"Checking each element of S against the others gives {_set_text(answer)}.",
        "There can be several minimal (or maximal) elements; a least or greatest element is a different, stronger idea.",
    ]
    return _single(text, latex, answer, _set_distractors(answer, items), working)


# -- medium --------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(4000, "an element with several covers that differ from the simple answers"):
        base, s = _random_set(rng, rng.randint(9, 11))
        x = rng.choice(s)
        lower = rng.random() < 0.5
        covers = lower_covers(s, x) if lower else upper_covers(s, x)
        allbelow = [y for y in s if (_lt(y, x) if lower else _lt(x, y))]
        opposite = upper_covers(s, x) if lower else lower_covers(s, x)
        lattice = [
            y
            for y in allbelow
            if (x // y if lower else y // x) in (2, 3, 5, 7) and len(_divisors(x // y if lower else y // x)) == 2
        ]
        if len(covers) >= 2 and covers != allbelow and covers != opposite and lattice != covers:
            break
    verb = "covers" if lower else "is covered by"
    st = _set_text(s)
    defn = f"An element {'x covers y' if lower else 'y covers x'} when {'y' if lower else 'x'} divides {'x' if lower else 'y'}, the two are different, and no other element of S lies strictly between them under divisibility."
    form = rng.randrange(3)
    which = "Which elements of S does " + str(x) + " cover?" if lower else "Which elements of S cover " + str(x) + "?"
    if form == 0:
        text, latex = say(f"The set S = {st} is {_ORDER}. {defn} {which}")
    elif form == 1:
        text, latex = say(
            f"{defn} In the divisibility poset on S = {st}, find the set of elements that {x} {verb}."
            if lower
            else f"{defn} In the divisibility poset on S = {st}, find the set of elements that cover {x}."
        )
    else:
        text, latex = say(
            f"The Hasse diagram of S = {st} under divisibility joins each element to the elements that cover it. {defn} {which}"
        )
    items = [
        (allbelow, _TAGS["below"]),
        (opposite, _TAGS["direction"]),
        (lattice, _TAGS["gaps"]),
        ([x, *covers], _TAGS["itself"]),
    ]
    working = [
        f"The elements of S {'below' if lower else 'above'} {x} (under divisibility) are {_set_text(allbelow)}.",
        f"Keep those with no other element of S between them and {x}: {_set_text(covers)}.",
        "A gap in S can make an element a cover even when it is not a prime factor away.",
    ]
    return _single(text, latex, covers, _set_distractors(covers, items), working)


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard(rng):
    for _ in attempts(6000, "a set with several longest chains whose union is a proper part of S"):
        base, s = _random_set(rng, rng.randint(10, 12))
        union = longest_chain_elements(s)
        size = longest_chain_size(s)
        chains = chains_of_size(s, size)
        first = list(chains[0])
        ends = sorted(set(minimal(s)) | set(maximal(s)))
        if len(chains) >= 2 and len(union) > size and union != sorted(s) and union != ends and len(union) < len(s) - 1:
            break
    st = _set_text(s)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"The set S = {st} is {_ORDER}. A chain is a set of elements any two of which are comparable. "
            "Find the set of all elements of S that belong to at least one chain of greatest possible size."
        )
    elif form == 1:
        text, latex = say(
            f"Under divisibility, S = {st} is a poset. Call a chain longest if no chain in S has more elements. "
            "Which elements of S lie on some longest chain?"
        )
    else:
        text, latex = say(
            f"In the divisibility poset on S = {st}, find the longest chains (two elements are in a chain when one divides the other). "
            "Give the set of elements that appear in at least one of them."
        )
    items = [
        (sorted(s), _TAGS["all"]),
        (ends, _TAGS["ends"]),
        (first, _TAGS["single"]),
    ]
    working = [
        f"The longest chains have {size} elements; they are " + "; ".join(_chain_text(c) for c in chains) + ".",
        f"The elements appearing in at least one of them are {_set_text(union)}.",
        "Every element lies on some maximal chain, but only some lie on a longest one.",
    ]
    return _single(text, latex, union, _set_distractors(union, items), working)


def _chain_text(chain):
    return " | ".join(str(x) for x in chain)


@generator(
    "r4j_partial_order_hasse",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_partial_order_hasse(rng, tier="easy"):
    """Reading a divisibility poset; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator("exd_x_r4j_partial_order_hasse", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_partial_order_hasse(rng, tier="exam"):
    """Exam chain: maximal elements, longest chain, largest antichain, then the fewest covering chains."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(6000, "a set whose chain and antichain sizes are not the obvious ones"):
        base, s = _random_set(rng, rng.randint(10, 12))
        top = maximal(s)
        low = minimal(s)
        chain = longest_chain_size(s)
        anti = largest_antichain_size(s)
        check(
            chain == len(chains_of_size(s, chain)[0]) and not chains_of_size(s, chain + 1),
            "longest chain disagrees with search",
        )
        if (
            len(top) >= 2
            and 3 <= chain <= 5
            and 3 <= anti <= 6
            and anti not in (len(top), len(low))
            and chain != anti
            and chain != len(s) - anti
        ):
            break
    st = _set_text(s)
    form = rng.randrange(3)
    if form == 0:
        stem, stem_tex = say(f"The set S = {st} is {_ORDER}.")
    elif form == 1:
        stem, stem_tex = say(f"Let (S, ≤) be the poset on S = {st} in which a ≤ b means a divides b.")
    else:
        stem, stem_tex = say(f"A poset has elements {st}, with a ≤ b when a divides b.")

    def part(prompt, kind, value, text, marks, items, working, follow=None):
        d = dict(
            prompt=prompt,
            answer_kind=kind,
            answer=value,
            answer_text=text,
            marks=marks,
            working=working,
            distractors=items,
        )
        if follow:
            d["follow_through"] = follow
        if kind == "number":
            d["sensible_range"] = (0, None)
        return d

    def nums(answer, pairs):
        return keep_distinct(answer, [(x, tag, str(x)) for x, tag in pairs if x > 0])

    parts = [
        part(
            "Find the set of maximal elements of S.",
            "set",
            top,
            _set_text(top),
            2,
            _set_distractors(
                top,
                [
                    (low, _TAGS["swapped"]),
                    ([max(s)], _TAGS["greatest"]),
                    (sorted(set(top) | {max(low)}), _TAGS["swapped"]),
                ],
            ),
            [f"An element is maximal when it divides nothing else in S: {_set_text(top)}."],
        ),
        part(
            "What is the greatest number of elements in a chain in S?",
            "number",
            chain,
            str(chain),
            2,
            nums(
                chain,
                [
                    (chain - 1, _TAGS["edges"]),
                    (anti, _TAGS["chain_anti"]),
                    (chain + 1, _TAGS["edges"]),
                    (len(top), _TAGS["anti_max"]),
                ],
            ),
            [f"A longest chain is {_chain_text(chains_of_size(s, chain)[0])}, and no chain has {chain + 1} elements."],
        ),
        part(
            "What is the greatest number of elements in an antichain in S (a set in which no element divides another)?",
            "number",
            anti,
            str(anti),
            2,
            nums(
                anti,
                [
                    (len(low), _TAGS["anti_min"]),
                    (len(top), _TAGS["anti_max"]),
                    (len(s) - chain, _TAGS["anti_comp"]),
                    (chain, _TAGS["chain_anti"]),
                ],
            ),
            [
                f"A largest antichain has {anti} elements, for example {_set_text(_an_antichain(s, anti))}; no set of {anti + 1} works."
            ],
        ),
        part(
            "Hence find the fewest chains needed to cover every element of S.",
            "number",
            anti,
            str(anti),
            2,
            nums(
                anti,
                [
                    (chain, _TAGS["mirsky"]),
                    (len(s), _TAGS["each"]),
                    (len(low), _TAGS["anti_min"]),
                    (anti + 1, _TAGS["each"]),
                ],
            ),
            [f"Dilworth's theorem: the fewest chains that cover a finite poset equals its largest antichain, {anti}."],
            follow="part_c",
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)


def _an_antichain(s, size):
    items = sorted(s)

    def grow(start, chosen):
        if len(chosen) == size:
            return chosen
        for i in range(start, len(items)):
            if all(not _lt(items[i], c) and not _lt(c, items[i]) for c in chosen):
                got = grow(i + 1, [*chosen, items[i]])
                if got:
                    return got
        return None

    got = grow(0, [])
    check(got is not None, "no antichain of the stated size")
    return got
