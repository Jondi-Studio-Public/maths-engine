"""Exam-tier counting questions for the multiplication principle, permutations and combinations skills.

Each practice tier of these skills asks for a single formula applied to bare numbers: multiply the option
counts, evaluate ``nPr`` or evaluate ``nCr``. The exam tier changes the KIND of question:

``exd_x_h2_multiplication_principle``
    A choice that cannot be made slot by slot without care, so the learner must pick a method.
    * ``code``: digit codes with no repeats, a non-zero first digit and a restriction on the last digit.
      Because the last digit may or may not be the digit 0, the count of first-digit options depends on which
      last digit was used, so the choices have to be split into cases.
    * ``alternate``: boys and girls in a line with no two of the same kind adjacent (a pattern count times the
      two internal orders).
    * ``at_least``: strings with repetition allowed that contain at least one special symbol, counted as all
      strings minus the strings that avoid every special symbol.

``exd_x_h2_permutations``
    The count is given and the learner solves for the size of the group. Either ``nPr = N`` with ``r`` stated,
    or the ordered and unordered selection counts are both stated, so the learner must first recover ``r``
    from ``nPr / nCr = r!`` and only then solve for ``n``.

``exd_x_h2_combinations``
    Selections under a constraint that needs a case split or a complement: at least ``j`` of one kind, two
    named people who will not serve together, or triangles from points of which some are collinear.

Every answer is an integer, built forward from the drawn values and re-counted by brute-force enumeration
before the question is returned. Every distractor is the result of a named, plausible mistake.
"""

from __future__ import annotations

import random
from itertools import combinations as _combos
from itertools import permutations as _perms
from itertools import product as _product
from math import comb, factorial, perm

from ..core import check, generator, phrase, pick_distractors


def _finish(text: str, answer: int, candidates: list[int], working: list[str]) -> dict:
    """Pack an integer answer, dropping non-positive and duplicate wrong values."""
    cands = [str(c) for c in candidates if isinstance(c, int) and c > 0]
    distractors = pick_distractors(str(answer), cands)
    check(len(distractors) >= 1, "no usable distractor")
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# multiplication principle
# ---------------------------------------------------------------------------

_LAST_DIGIT_RULES = (
    ("even", (0, 2, 4, 6, 8)),
    ("a multiple of 5", (0, 5)),
    ("a prime number", (2, 3, 5, 7)),
)

_CODE_NOUNS = (("PIN", "PINs"), ("locker code", "locker codes"), ("security code", "security codes"))


def _code_count(k: int, last_digits: tuple[int, ...]) -> int:
    """Case split on the last digit: a non-zero last digit removes one of the nine non-zero first digits."""
    total = 0
    for last in last_digits:
        first_options = 9 - (1 if last != 0 else 0)
        total += first_options * perm(8, k - 2)
    return total


def _code_question(rng: random.Random) -> dict:
    k = rng.choice((3, 4))
    desc, last_digits = rng.choice(_LAST_DIGIT_RULES)
    noun, nouns = rng.choice(_CODE_NOUNS)
    answer = _code_count(k, last_digits)
    # Independent route: enumerate every arrangement of k different digits.
    brute = sum(1 for digits in _perms(range(10), k) if digits[0] != 0 and digits[-1] in last_digits)
    check(brute == answer, "code count disagrees with enumeration")
    text = phrase(
        rng,
        (
            "A {noun} is made from {k} different digits chosen from 0 to 9. The first digit cannot be 0 and the last "
            "digit must be {desc}. How many different {nouns} are possible?",
            "A {k}-digit {noun} uses the digits 0 to 9 with no digit repeated. It cannot start with 0, and its last "
            "digit is {desc}. How many {nouns} can be made?",
        ),
        noun=noun,
        nouns=nouns,
        k=k,
        desc=desc,
    )
    n_last = len(last_digits)
    cands = [
        n_last * 9 * perm(8, k - 2),  # first-digit options not reduced by the last digit
        n_last * 9 * 10 ** (k - 2),  # digits allowed to repeat
        n_last * perm(9, k - 1),  # the non-zero first digit rule ignored
    ]
    working = [
        f"Split by the last digit. A last digit of 0 leaves 9 first digits; any other allowed last digit leaves 8. "
        f"The {k - 2} middle place(s) are then filled from the 8 unused digits in {perm(8, k - 2)} ways each. "
        f"Total = {answer}."
    ]
    return _finish(text, answer, cands, working)


def _alternate_question(rng: random.Random) -> dict:
    m = rng.choice((2, 3, 4))
    extra = rng.choice((0, 1))
    g = m + extra
    if extra:
        answer = factorial(g) * factorial(m)
    else:
        answer = 2 * factorial(m) * factorial(g)
    # Independent route: count the alternating boy/girl patterns, then arrange each gender within its places.
    patterns = sum(
        1
        for pattern in _product("BG", repeat=m + g)
        if pattern.count("B") == m and all(a != b for a, b in zip(pattern, pattern[1:]))
    )
    check(patterns * factorial(m) * factorial(g) == answer, "alternating count disagrees with enumeration")
    text = phrase(
        rng,
        (
            "{m} different boys and {g} different girls stand in a line so that no two boys are next to each other and "
            "no two girls are next to each other. In how many ways can they stand?",
            "A queue is formed from {m} boys and {g} girls, all different people, so that boys and girls alternate "
            "along the line. How many different queues are possible?",
        ),
        m=m,
        g=g,
    )
    cands = [
        factorial(m + g),  # no alternation rule
        factorial(m) * factorial(g) * (1 if not extra else 2),  # wrong number of starting patterns
        factorial(m) + factorial(g),  # adds instead of multiplies
        factorial(g) * (2 if not extra else 1),  # only the girls arranged
    ]
    if extra:
        working = [
            f"With one more girl than boys the line must start and end with a girl, so there is only 1 pattern. "
            f"Girls: {g}! ways, boys: {m}! ways. Total = {answer}."
        ]
    else:
        working = [
            f"The line can start with a boy or with a girl, so there are 2 patterns. "
            f"Each pattern has {m}! orders for the boys and {g}! for the girls. Total = {answer}."
        ]
    return _finish(text, answer, cands, working)


_AT_LEAST_CONTEXTS = (
    # (alphabet size, special symbols, lengths, wording of the alphabet, noun phrase for a special symbol)
    (10, 1, (3, 4), "the digits 0 to 9", "digit 7"),
    (10, 5, (3, 4), "the digits 0 to 9", "even digit"),
    (10, 2, (3, 4), "the digits 0 to 9", "digit that is a 3 or an 8"),
    (26, 5, (3,), "the 26 letters of the alphabet", "vowel (A, E, I, O or U)"),
    (26, 2, (3,), "the 26 letters of the alphabet", "letter that is an X or a Z"),
)


def _at_least_question(rng: random.Random) -> dict:
    size, special, lengths, alphabet, special_desc = rng.choice(_AT_LEAST_CONTEXTS)
    k = rng.choice(lengths)
    answer = size**k - (size - special) ** k
    # Independent route: enumerate every string and test it directly (special symbols are indices below `special`).
    brute = sum(1 for word in _product(range(size), repeat=k) if any(c < special for c in word))
    check(brute == answer, "at-least count disagrees with enumeration")
    text = phrase(
        rng,
        (
            "A code of length {k} is formed from {alphabet}, and symbols may be repeated. How many codes contain "
            "at least one {special_desc}?",
            "How many strings of length {k} can be written using {alphabet}, with repetition allowed, so that the "
            "string contains at least one {special_desc}?",
        ),
        k=k,
        alphabet=alphabet,
        special_desc=special_desc,
    )
    cands = [
        size**k,  # no condition
        special * size ** (k - 1),  # choose the special place, let the rest be anything (double counts)
        (size - special) ** k,  # strings with none of the special symbols
        size**k - special**k,  # complement taken over the wrong set
    ]
    working = [
        f"Count the complement. All strings: {size}^{k} = {size**k}. Strings with no special symbol: "
        f"{size - special}^{k} = {(size - special) ** k}. Difference = {answer}."
    ]
    return _finish(text, answer, cands, working)


@generator(
    "exd_x_h2_multiplication_principle",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_multiplication_principle(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier multiplication-principle counting: case-split code, alternating line, or at-least-one."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_code_question, _alternate_question, _at_least_question))(rng)


# ---------------------------------------------------------------------------
# permutations: solve for the group size
# ---------------------------------------------------------------------------

_ROLES = {
    2: "a captain and a vice-captain",
    3: "a president, a secretary and a treasurer",
    4: "a chair, a deputy chair, a secretary and a treasurer",
}


def _solve_n_for_perm(r: int, value: int) -> int:
    """Smallest group size n >= r with nPr = value (nPr is strictly increasing in n)."""
    n = r
    while perm(n, r) < value:
        n += 1
    check(perm(n, r) == value, "no integer n gives that permutation count")
    return n


def _perm_given_count(rng: random.Random) -> dict:
    r = rng.choice((2, 3, 3, 4))
    n = rng.randint(r + 3, 14 if r < 4 else 11)
    value = perm(n, r)
    check(_solve_n_for_perm(r, value) == n, "n is not recovered from the count")
    text = phrase(
        rng,
        (
            "A club fills the roles of {roles}, each role going to a different member. This can be done in {value} "
            "different ways. How many members does the club have?",
            "The roles of {roles} are given to different people chosen from a group, and there are {value} possible "
            "outcomes. How many people are in the group?",
        ),
        roles=_ROLES[r],
        value=value,
    )
    cands = [
        round(value ** (1 / r)),  # treats the count as n to the power r
        n - r + 1,  # the smallest factor of the product
        n + 1,  # starts the product one place too high
        n - 1,
    ]
    working = [
        f"nP{r} is a product of {r} consecutive descending whole numbers starting at n, and it equals {value}. "
        f"It grows with n, and n = {n} gives {value}, so n = {n}."
    ]
    return _finish(text, n, cands, working)


def _perm_and_comb(rng: random.Random) -> dict:
    r = rng.choice((2, 3, 4))
    n = rng.randint(r + 3, 13)
    p, c = perm(n, r), comb(n, r)
    check(p % c == 0 and p // c == factorial(r), "ordered / unordered must be r factorial")
    text = phrase(
        rng,
        (
            "Some people are chosen from a group to form a team. If the order of selection matters there are {p} "
            "possible outcomes, and if the order does not matter there are {c}. How many people are in the group?",
            "From a group, some people are selected. There are {p} ways to make the selection when the order counts "
            "and {c} ways when it does not. How many people are in the group?",
        ),
        p=p,
        c=c,
    )
    cands = [
        r,  # the size of the selection, not the group
        p // c,  # r factorial, the ratio itself
        n + 1,
        n - 1,
    ]
    working = [
        f"nPr / nCr = r!, so r! = {p} / {c} = {p // c}, giving r = {r}. Then nP{r} = {p} gives n = {n}.",
    ]
    return _finish(text, n, cands, working)


@generator(
    "exd_x_h2_permutations",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_permutations(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier permutations: recover the group size from a permutation count (and a combination count)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_perm_given_count, _perm_and_comb))(rng)


# ---------------------------------------------------------------------------
# combinations under a constraint
# ---------------------------------------------------------------------------


def _at_least_of_one_kind(rng: random.Random) -> dict:
    men = rng.randint(3, 7)
    women = rng.randint(3, 7)
    k = rng.randint(3, 5)
    j = rng.choice((1, 2))
    answer = sum(comb(women, w) * comb(men, k - w) for w in range(j, min(women, k) + 1) if k - w <= men)
    # Independent route: enumerate every committee (people 0..women-1 are the women).
    brute = sum(1 for team in _combos(range(women + men), k) if sum(1 for p in team if p < women) >= j)
    check(brute == answer, "committee count disagrees with enumeration")
    check(answer > 0, "empty count")
    at_least = "at least one woman" if j == 1 else "at least two women"
    text = phrase(
        rng,
        (
            "A committee of {k} is chosen from {men} men and {women} women. How many committees include {at_least}?",
            "From {men} men and {women} women, {k} people are selected for a panel. In how many ways can the panel "
            "have {at_least}?",
        ),
        k=k,
        men=men,
        women=women,
        at_least=at_least,
    )
    cands = [
        comb(men + women, k),  # no condition
        comb(women, j) * comb(men, k - j) if k - j <= men else 0,  # exactly j women only
        comb(men + women, k) - comb(women, j) if j == 1 else comb(men + women, k) - comb(men, k),  # wrong complement
        comb(women, j) * comb(men + women - j, k - j),  # pick j women first, then anyone (double counts)
    ]
    working = [
        f"Add the cases with {j} to {min(women, k)} women, or subtract the committees with too few women "
        f"from all {comb(men + women, k)} committees. Total = {answer}."
    ]
    return _finish(text, answer, cands, working)


_PAIR_NAMES = (("Alex", "Bree"), ("Chen", "Dara"), ("Eli", "Fatima"), ("Gus", "Hana"))


def _two_refuse_together(rng: random.Random) -> dict:
    n = rng.randint(7, 12)
    k = rng.randint(3, 5)
    first, second = rng.choice(_PAIR_NAMES)
    answer = comb(n, k) - comb(n - 2, k - 2)
    # Independent route: enumerate every team (people 0 and 1 are the pair) and reject those holding both.
    brute = sum(1 for team in _combos(range(n), k) if not (0 in team and 1 in team))
    check(brute == answer, "team count disagrees with enumeration")
    text = phrase(
        rng,
        (
            "A team of {k} is chosen from {n} people. {first} and {second} are among the {n} but will not play on the "
            "same team. How many teams are possible?",
            "{n} people are available to form a group of {k}. {first} and {second} refuse to be in the group together "
            "(either may be in it alone). How many groups can be formed?",
        ),
        n=n,
        k=k,
        first=first,
        second=second,
    )
    cands = [
        comb(n, k),  # no restriction
        comb(n - 2, k - 2),  # the teams that hold both
        comb(n, k) - comb(n - 2, k),  # subtracts the teams that hold neither
        comb(n - 2, k),  # teams that hold neither of them
    ]
    working = [
        f"All teams: C({n},{k}) = {comb(n, k)}. Teams containing both: C({n - 2},{k - 2}) = {comb(n - 2, k - 2)}."
    ]
    return _finish(text, answer, cands, working)


def _triangles_from_points(rng: random.Random) -> dict:
    n = rng.randint(7, 12)
    line = rng.randint(3, 5)
    answer = comb(n, 3) - comb(line, 3)
    # Independent route: enumerate triples; points 0..line-1 are the collinear ones.
    brute = sum(1 for tri in _combos(range(n), 3) if not all(p < line for p in tri))
    check(brute == answer, "triangle count disagrees with enumeration")
    text = phrase(
        rng,
        (
            "There are {n} points in a plane. Exactly {line} of them lie on one straight line and no other three "
            "points are collinear. How many triangles have their vertices at three of the points?",
            "{n} points are marked on a page, {line} of which are in a straight line; no other three are in a line. "
            "How many different triangles can be drawn using three of the points as vertices?",
        ),
        n=n,
        line=line,
    )
    cands = [
        comb(n, 3),  # ignores the collinear points
        comb(n, 3) - line,  # removes one triple per collinear point
        comb(n, 3) - comb(line, 2),
        comb(n, 3) - 1,  # removes a single degenerate triple
    ]
    working = [
        f"All triples: C({n},3) = {comb(n, 3)}. Collinear triples give no triangle: C({line},3) = {comb(line, 3)}."
    ]
    return _finish(text, answer, cands, working)


@generator(
    "exd_x_h2_combinations",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_combinations(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier combinations: a constraint that needs cases or a complement."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_at_least_of_one_kind, _two_refuse_together, _triangles_from_points))(rng)
