"""Generates `sr2b_counting_restrictions` questions (easy, medium, hard) and the exam chain
`sr2b_exam_counting_restrictions`.

Skill ``counting-with-restrictions`` (``senior-mathematics``, answer ``number``): counting arrangements and selections
when there is a restriction. The plain n-factorial, nPr and nCr skills are prerequisites.

* easy: the number of arrangements of all the letters of a place name with repeated letters, ``n!`` divided by the
  factorial of each repeat count.
* medium: n different people or objects in a row, with two of them next to each other, two of them not next to each
  other, or one of them at an end.
* hard: either n different people around a circular table (no fixed seat), with no restriction, two next to each other, or
  two not next to each other; or a committee chosen from men and women with "at least" or "at most" some number of
  women, or with one person who must or must not be on it.
* exam (``sr2b_exam_counting_restrictions``): the letters of a place name with a repeated letter, or the digits of a
  locker code with a repeated digit: (a) all
  arrangements, (b) those with the two copies of one letter together, (c) the probability a random arrangement has them
  together (follows through from (a) and (b)), (d) the number with them apart (follows through from (a) and (b)).

Every count is also found by a second route: brute-force enumeration for the small cases, or a count by choosing places
for each repeated letter in turn for the long words.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, factorial

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, rational_text
from ._sr2b_kit import int_dist, pick, retry_short

_NAMES = (
    "BANANA",
    "CANBERRA",
    "ADELAIDE",
    "MELBOURNE",
    "BALLARAT",
    "GEELONG",
    "TOOWOOMBA",
    "BUNDABERG",
    "PARRAMATTA",
    "WAGGA",
    "TOOWONG",
    "BUNBURY",
    "KALGOORLIE",
    "WOLLONGONG",
    "LAUNCESTON",
    "MANDURAH",
    "ARMIDALE",
    "COOMA",
    "NOOSA",
    "BUNDANOON",
    "NARRABRI",
    "TAMWORTH",
    "BELLINGEN",
    "MITTAGONG",
    "KATOOMBA",
    "BRISBANE",
    "DUBBO",
    "EMERALD",
    "MOREE",
    "BLACKALL",
    "MOLONG",
    "ARARAT",
)


def arrangements(word: str) -> int:
    n = len(word)
    out = factorial(n)
    for c in Counter(word).values():
        out //= factorial(c)
    return out


#: Place names with a repeated letter whose arrangements stay readable (at most seven digits, at most ten letters).
_WORDS = tuple(w for w in dict.fromkeys(_NAMES) if len(set(w)) < len(w) and len(w) <= 10 and arrangements(w) <= 10**7)


def arrangements_by_places(word: str) -> int:
    """The same count by choosing positions for each distinct letter in turn."""
    left, total = len(word), 1
    for c in Counter(word).values():
        total *= comb(left, c)
        left -= c
    return total


def _brute_arrangements(word: str) -> int:
    return len(set(permutations(word)))


def _together(word: str, letter: str) -> int:
    """Arrangements with all copies of `letter` in one block."""
    rest = word.replace(letter, "") + letter  # the block counts as one letter, written once
    return arrangements(rest)


def _together_by_places(word: str, letter: str) -> int:
    """Count by sliding the block of copies into one of the n - c + 1 places among the other letters' arrangements."""
    return (len(word) - word.count(letter) + 1) * arrangements(word.replace(letter, ""))


# -- easy -----------------------------------------------------------------------------------------------------


def _easy(rng):
    word = rng.choice(_WORDS)
    n = len(word)
    counts = sorted((c for c in Counter(word).values() if c > 1), reverse=True)
    ans = arrangements(word)
    check(ans == arrangements_by_places(word), "arrangement count differs from the places route")
    if n <= 7:
        check(ans == _brute_arrangements(word), "arrangement count differs from enumeration")
    cands = [
        int_dist(factorial(n), "sr2b-repeats-ignored"),
        int_dist(factorial(n) // factorial(counts[0]), "sr2b-one-repeat-only"),
        int_dist(factorial(n) // factorial(sum(counts)), "sr2b-repeat-counts-added"),
        int_dist(factorial(n) // _prod(counts), "sr2b-divided-by-counts-not-factorials"),
        int_dist(factorial(n - 1), "sr2b-letter-fixed"),
    ]
    stem = phrase(
        rng,
        (
            "How many different arrangements are there of all the letters of the word {w}?",
            "How many different ways can all the letters of {w} be arranged in a row?",
            "The letters of the town name {w} are rearranged in every possible order. How many different arrangements are there?",
        ),
        w=word,
    )
    division = " × ".join(f"{c}!" for c in counts)
    return dict(
        text=stem,
        answer_kind="number",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        working=[
            f"{word} has {n} letters, and the letters repeated are {', '.join(f'{l} ({c} times)' for l, c in Counter(word).items() if c > 1)}.",
            f"Arrangements = {n}! ÷ ({division}) = {ans}.",
        ],
        distractors=pick(str(ans), cands),
    )


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= v
    return out


# -- medium ---------------------------------------------------------------------------------------------------

_ROW_CONTEXTS = (
    ("people", "stand in a row for a photograph", ("Priya", "Tom", "Alisha")),
    ("books", "are placed on a shelf", ("the atlas", "the dictionary", "the almanac")),
    (
        "flags",
        "are flown from a line of poles",
        ("the Australian flag", "the Torres Strait flag", "the Aboriginal flag"),
    ),
    ("runners", "line up at the start of a relay", ("Mia", "Jack", "Zara")),
)


def _row_count(n: int, kind: str) -> int:
    return {
        "together": 2 * factorial(n - 1),
        "apart": factorial(n) - 2 * factorial(n - 1),
        "end": 2 * factorial(n - 1),
        "ends": 2 * factorial(n - 2),
        "together3": 6 * factorial(n - 2),
    }[kind]


def _row_brute(n: int, kind: str) -> int:
    count = 0
    for perm in permutations(range(n)):
        i, j, k = perm.index(0), perm.index(1), perm.index(2)
        if kind == "together":
            count += abs(i - j) == 1
        elif kind == "apart":
            count += abs(i - j) != 1
        elif kind == "end":
            count += i in (0, n - 1)
        elif kind == "ends":
            count += {i, j} == {0, n - 1}
        else:
            count += max(i, j, k) - min(i, j, k) == 2
    return count


def _medium(rng):
    n = rng.randint(5, 7)
    kind = rng.choice(("together", "apart", "end", "ends", "together3"))
    noun, setting, (a, b, c) = rng.choice(_ROW_CONTEXTS)
    ans = _row_count(n, kind)
    check(ans == _row_brute(n, kind), "row count differs from enumeration")
    f, g, h = factorial(n), factorial(n - 1), factorial(n - 2)
    if kind == "together":
        cands = [
            int_dist(g, "sr2b-internal-order-missed"),
            int_dist(f // 2, "sr2b-together-treated-as-half"),
            int_dist(f - 2 * g, "sr2b-apart-count-given"),
            int_dist(2 * h, "sr2b-block-and-rest-miscounted"),
        ]
        ask = f"{a} and {b} must be next to each other"
        working = [
            f"Treat {a} and {b} as one block: the block and the other {n - 2} give {n - 1} objects to arrange, in {n - 1}! = {g} ways.",
            f"The two inside the block can swap: {g} × 2 = {ans}.",
        ]
    elif kind == "apart":
        cands = [
            int_dist(2 * g, "sr2b-together-count-given"),
            int_dist(f - g, "sr2b-internal-order-missed"),
            int_dist(f - 2 * h, "sr2b-block-and-rest-miscounted"),
            int_dist(f // 2, "sr2b-together-treated-as-half"),
        ]
        ask = f"{a} and {b} must not be next to each other"
        working = [
            f"Total arrangements = {n}! = {f}.",
            f"Arrangements with {a} and {b} together = 2 × {n - 1}! = {2 * g}.",
            f"Not together = {f} - {2 * g} = {ans}.",
        ]
    elif kind == "end":
        cands = [
            int_dist(g, "sr2b-one-end-only"),
            int_dist(2 * h, "sr2b-block-and-rest-miscounted"),
            int_dist(n * g, "sr2b-restriction-ignored"),
            int_dist(2 * g * (n - 1), "sr2b-restriction-ignored"),
        ]
        ask = f"{a} must be at one of the two ends"
        working = [
            f"Choose the end for {a}: 2 ways.",
            f"Arrange the other {n - 1} in the remaining places: {n - 1}! = {g}. Total 2 × {g} = {ans}.",
        ]
    elif kind == "ends":
        cands = [
            int_dist(h, "sr2b-end-order-missed"),
            int_dist(2 * g, "sr2b-one-end-only"),
            int_dist(n * (n - 1) * h, "sr2b-restriction-ignored"),
            int_dist(g, "sr2b-one-end-only"),
        ]
        ask = f"{a} and {b} must be at the two ends, one at each end"
        working = [
            f"{a} and {b} can take the two end places in 2 ways.",
            f"The other {n - 2} fill the middle in {n - 2}! = {h} ways. Total 2 × {h} = {ans}.",
        ]
    else:
        cands = [
            int_dist(h, "sr2b-internal-order-missed"),
            int_dist(2 * h, "sr2b-internal-order-missed"),
            int_dist(g, "sr2b-block-and-rest-miscounted"),
            int_dist(6 * g, "sr2b-block-and-rest-miscounted"),
        ]
        ask = f"{a}, {b} and {c} must be next to each other, in any order among themselves"
        working = [
            f"Treat the three as one block: the block and the other {n - 3} give {n - 2} objects to arrange, in {n - 2}! = {h} ways.",
            f"The three inside the block can be ordered in 3! = 6 ways: {h} × 6 = {ans}.",
        ]
    stem = phrase(
        rng,
        (
            "{n} different {noun} {setting}. In how many ways can they be arranged if {ask}?",
            "In how many ways can {n} different {noun} be arranged when they {setting}, if {ask}?",
        ),
        n=n,
        noun=noun,
        setting=setting.replace("are placed", "are placed").replace("stand", "stand"),
        ask=ask,
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        working=working,
        distractors=pick(str(ans), [c for c in cands if int(c.text) > 0]),
    )


# -- hard -----------------------------------------------------------------------------------------------------


def _circle_count(n: int, kind: str) -> int:
    if kind == "free":
        return factorial(n - 1)
    if kind == "together":
        return 2 * factorial(n - 2)
    return factorial(n - 1) - 2 * factorial(n - 2)


def _circle_brute(n: int, kind: str) -> int:
    """Seat person 0 first (rotations are the same arrangement), then count the orders of the rest."""
    count = 0
    for rest in permutations(range(1, n)):
        adjacent = _adjacent((0, *rest), 1, 2, n)
        count += 1 if kind == "free" else (adjacent if kind == "together" else not adjacent)
    return count


def _adjacent(seats, a, b, n) -> bool:
    i, j = seats.index(a), seats.index(b)
    return (i - j) % n in (1, n - 1)


def _hard_circle(rng):
    n = rng.randint(5, 7)
    kind = rng.choice(("free", "together", "apart"))
    ans = _circle_count(n, kind)
    check(ans == _circle_brute(n, kind), "circular count differs from enumeration")
    f, g = factorial(n), factorial(n - 1)
    h = factorial(n - 2)
    a, b = rng.choice((("Ravi", "Sam"), ("Imogen", "Kai"), ("Ngaire", "Dale")))
    setting_base, setting_finite = rng.choice(
        (
            ("sit around a circular table", "sit around a circular table"),
            ("be seated at a round table", "are seated at a round table"),
        )
    )
    if kind == "free":
        cands = [
            int_dist(f, "sr2b-rotations-counted-as-different"),
            int_dist(f // 2, "sr2b-reflection-halved"),
            int_dist(factorial(n - 2), "sr2b-two-people-fixed"),
            int_dist(g * 2, "sr2b-rotations-counted-as-different"),
        ]
        ask = ""
        working = [
            f"Rotating everyone round the table gives the same arrangement, so fix one person and arrange the other {n - 1}: ({n} - 1)! = {ans}."
        ]
    elif kind == "together":
        cands = [
            int_dist(2 * g, "sr2b-rotations-counted-as-different"),
            int_dist(h, "sr2b-internal-order-missed"),
            int_dist(g, "sr2b-restriction-ignored"),
            int_dist(g - 2 * h, "sr2b-apart-count-given"),
        ]
        ask = f", with {a} and {b} next to each other"
        working = [
            f"Treat {a} and {b} as one block: {n - 1} objects round the table, arranged in ({n - 1} - 1)! = {h} ways.",
            f"The two inside the block can swap: {h} × 2 = {ans}.",
        ]
    else:
        cands = [
            int_dist(2 * h, "sr2b-together-count-given"),
            int_dist(g - h, "sr2b-internal-order-missed"),
            int_dist(g, "sr2b-restriction-ignored"),
            int_dist(f - 2 * g, "sr2b-rotations-counted-as-different"),
        ]
        ask = f", with {a} and {b} not next to each other"
        working = [
            f"All arrangements round the table: {n - 1}! = {g}.",
            f"With {a} and {b} together: 2 × {n - 2}! = {2 * h}.",
            f"Not together: {g} - {2 * h} = {ans}.",
        ]
    stem = phrase(
        rng,
        (
            "In how many different ways can {n} people {setting_base}{ask}? Arrangements that are rotations of each other count as the same.",
            "{n} different people {setting_finite}{ask}. How many different seating arrangements are there, if rotations are not counted as different?",
        ),
        n=n,
        setting_base=setting_base,
        setting_finite=setting_finite,
        ask=ask,
    )
    return dict(
        text=stem,
        answer_kind="number",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        working=working,
        distractors=pick(str(ans), [c for c in cands if int(c.text) > 0]),
    )


def _committee_count(m: int, w: int, k: int, kind: str, j: int) -> int:
    if kind == "at_least":
        return sum(comb(w, i) * comb(m, k - i) for i in range(j, min(w, k) + 1))
    if kind == "at_most":
        return sum(comb(w, i) * comb(m, k - i) for i in range(0, min(j, w, k) + 1))
    return comb(m + w - 1, k - 1)  # one named person must be on it


def _committee_brute(m: int, w: int, k: int, kind: str, j: int) -> int:
    people = [("m", i) for i in range(m)] + [("w", i) for i in range(w)]
    count = 0
    for group in combinations(range(m + w), k):
        women = sum(1 for g in group if g >= m)
        if kind == "at_least":
            count += women >= j
        elif kind == "at_most":
            count += women <= j
        else:
            count += 0 in group
    return count


def _hard_committee(rng):
    m, w = rng.randint(4, 7), rng.randint(3, 6)
    k = rng.randint(3, min(5, m + w - 2))
    kind = rng.choice(("at_least", "at_least", "at_most", "named"))
    j = rng.randint(1, 2) if kind != "named" else 0
    j = min(j, w, k - 1)
    ans = _committee_count(m, w, k, kind, j)
    check(ans == _committee_brute(m, w, k, kind, j), "committee count differs from enumeration")
    total = comb(m + w, k)
    if kind == "at_least":
        compl = sum(comb(w, i) * comb(m, k - i) for i in range(0, j))
        cands = [
            int_dist(comb(w, j) * comb(m + w - j, k - j), "sr2b-at-least-double-counted"),
            int_dist(comb(w, j) * comb(m, k - j), "sr2b-exactly-for-at-least"),
            int_dist(compl, "sr2b-complement-given"),
        ]
        if j > 1:
            cands.append(int_dist(total - comb(m, k), "sr2b-complement-taken-wrongly"))
        ask = f"at least {j} {'woman' if j == 1 else 'women'}"
        working = [
            f"Total committees = C({m + w}, {k}) = {total}.",
            f"Those with fewer than {j} women = {compl}.",
            f"At least {j}: {total} - {compl} = {ans}.",
        ]
    elif kind == "at_most":
        cands = [
            int_dist(comb(w, j) * comb(m, k - j), "sr2b-exactly-for-at-most"),
            int_dist(comb(m, k), "sr2b-case-left-out"),
            int_dist(total - ans, "sr2b-complement-given"),
            int_dist(sum(comb(w, i) * comb(m, k - i) for i in range(1, min(j, w) + 1)), "sr2b-case-left-out"),
        ]
        ask = f"at most {j} {'woman' if j == 1 else 'women'}"
        working = [
            "Add the cases 0, 1, ... women: "
            + " + ".join(f"C({w}, {i}) × C({m}, {k - i})" for i in range(0, j + 1))
            + f" = {ans}."
        ]
    else:
        cands = [
            int_dist(total, "sr2b-restriction-ignored"),
            int_dist(comb(m + w - 1, k), "sr2b-named-person-left-off"),
            int_dist(k * comb(m + w - 1, k - 1), "sr2b-named-person-arranged"),
            int_dist(comb(m + w - 1, k - 1) + comb(m + w - 1, k), "sr2b-case-left-out"),
        ]
        ask = "Meera on it"
        working = [
            f"Meera is already chosen, so choose the other {k - 1} members from the remaining {m + w - 1} people: C({m + w - 1}, {k - 1}) = {ans}."
        ]
    if kind == "named":
        stem = f"A committee of {k} people is to be chosen from {m} men and {w} women. Meera is one of the women. How many different committees can be chosen if Meera must be on the committee?"
    else:
        stem = phrase(
            rng,
            (
                "A committee of {k} people is to be chosen from {m} men and {w} women. How many different committees contain {ask}?",
                "From {m} men and {w} women, a committee of {k} is chosen. In how many ways can it be chosen so that it has {ask}?",
            ),
            k=k,
            m=m,
            w=w,
            ask=ask,
        )
    return dict(
        text=stem,
        answer_kind="number",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        working=working,
        distractors=pick(str(ans), [c for c in cands if int(c.text) > 0]),
    )


@generator(
    "sr2b_counting_restrictions",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_counting_restrictions(rng, tier="easy"):
    """A restricted count; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_circle, _hard_committee])(rng)


# -- exam -------------------------------------------------------------------------------------------------------

#: Extra town names with at least one letter that occurs exactly twice, so the exam chain is not limited to the
#: names the practice tiers use. Names are kept to ten letters or fewer so every count stays small enough to check.
_EXAM_EXTRA_NAMES = (
    "ALLORA",
    "GLENELG",
    "BELLBIRD",
    "ABERDARE",
    "BALLINA",
    "TERRIGAL",
    "MURRAY",
    "KILLARA",
    "BALMAIN",
    "ROSEBERRY",
    "WILLUNGA",
    "GOONDIWINDI",
    "TUMUT",
    "NARRANDERA",
    "MARRICKVILLE",
    "BOOROOMA",
    "COBRAM",
    "FOOTSCRAY",
    "MACCLESFIELD",
    "KINGAROY",
    "BERRIMA",
    "CARRATHOOL",
)


def _exam_choices():
    """Every (word, letter) pair where `letter` occurs two or three times in a town name of at most ten letters."""
    words = dict.fromkeys(_WORDS + tuple(w for w in _EXAM_EXTRA_NAMES if len(w) <= 10))
    out = []
    for w in words:
        if len(w) > 10 or arrangements(w) > 10**7:
            continue
        for letter, c in Counter(w).items():
            if c in (2, 3):
                out.append((w, letter))
    return tuple(sorted(out))


#: Each pair is one (word, letter) scenario; the words run from five to ten letters so the answers differ widely.
_EXAM_CHOICES = _exam_choices()


def _digit_patterns():
    """Repeat patterns for a digit code: how often each distinct digit occurs, with at least one digit used two or three times."""
    out = []

    def split(left, biggest, parts):
        if left == 0:
            if (2 in parts or 3 in parts) and len(parts) >= 3:
                out.append(tuple(parts))
            return
        for k in range(min(left, biggest), 0, -1):
            split(left - k, k, [*parts, k])

    for n in range(5, 11):
        split(n, 4, [])
    return tuple(out)


#: The second scenario family: all the digits of a locker code, each used as often as it is listed. The answers depend
#: only on how many digits there are and how often each repeats, so these patterns widen the set of answer tuples.
_DIGIT_PATTERNS = _digit_patterns()


def _draw_exam_scenario(rng):
    """A (text, letter, unit) triple: the string to arrange, the repeated symbol kept together, and its unit name."""
    if rng.random() < 0.5:
        word, letter = rng.choice(_EXAM_CHOICES)
        return word, letter, "letter"
    pattern = rng.choice(_DIGIT_PATTERNS)
    digits = rng.sample("0123456789", len(pattern))
    pairs = [d for d, k in zip(digits, pattern, strict=True) if k in (2, 3)]
    code = "".join(sorted(d * k for d, k in zip(digits, pattern, strict=True)))
    return "".join(sorted(code)), rng.choice(pairs), "digit"


@generator("sr2b_exam_counting_restrictions", level=13, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_counting_restrictions(rng, tier="exam"):
    """Exam chain: all arrangements, a repeated letter together, the probability, and the number apart."""
    check(tier == "exam", f"unknown tier {tier!r}")
    word, letter, unit = _draw_exam_scenario(rng)
    copies = word.count(letter)
    check(copies in (2, 3), "the exam chain keeps two or three copies of a symbol together")
    many = {2: "two", 3: "three"}[copies]
    n = len(word)
    total = arrangements(word)
    near = _together(word, letter)
    apart = total - near
    prob = Fraction(near, total)
    check(total == arrangements_by_places(word), "total differs from the places route")
    check(near == _together_by_places(word, letter), "together differs from the places route")
    check(apart > 0 and 0 < prob < 1, "degenerate chain")
    if n <= 7:
        check(total == _brute_arrangements(word), "total differs from enumeration")
        check(
            near == sum(1 for p in set(permutations(word)) if letter * copies in "".join(p)),
            "together differs from enumeration",
        )
    counts = sorted((c for c in Counter(word).values() if c > 1), reverse=True)
    a_d = [
        int_dist(factorial(n), "sr2b-repeats-ignored"),
        int_dist(factorial(n) // factorial(counts[0]), "sr2b-one-repeat-only"),
        int_dist(factorial(n) // _prod(counts), "sr2b-divided-by-counts-not-factorials"),
    ]
    rest = arrangements(word.replace(letter, ""))
    b_d = [
        int_dist(factorial(copies) * near, "sr2b-internal-order-counted-for-identical"),
        int_dist(total // 2, "sr2b-together-treated-as-half"),
        int_dist(factorial(n - 1), "sr2b-repeats-ignored"),
        int_dist(rest, "sr2b-pair-position-fixed"),
        int_dist(rest * n, "sr2b-pair-places-miscounted"),
    ]
    c_d = [
        _frac_dist(Fraction(apart, total), "sr2b-apart-probability-given"),
        _frac_dist(Fraction(near, apart), "sr2b-odds-for-probability"),
        _frac_dist(Fraction(near, total + near), "sr2b-probability-over-wrong-total"),
        _frac_dist(Fraction(1, 2), "sr2b-equally-likely-assumed"),
    ]
    d_d = [
        int_dist(near, "sr2b-together-count-given"),
        int_dist(total - 2 * near, "sr2b-together-subtracted-twice"),
        int_dist(total - factorial(n - 1), "sr2b-repeats-ignored"),
    ]
    if unit == "letter":
        stem = phrase(
            rng,
            (
                "The letters of the town name {w} are arranged in a row in every possible order.",
                "All the letters of {w}, the name of an Australian town, are rearranged in a row in every possible order.",
                "Consider every way of arranging the letters of the town name {w} in a row.",
            ),
            w=word,
        )
    else:
        stem = phrase(
            rng,
            (
                "A locker code is made by using all of the digits {w}, each as many times as it is listed, in a row.",
                "Every digit in the list {w} is used, each as often as it is listed, to make a {k}-digit code.",
                "A {k}-digit padlock code uses exactly the digits {w}, in any order.",
            ),
            w=", ".join(word),
            k=n,
        )
    f_c = rational_text(prob)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"How many different arrangements are there of all the {unit}s?",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=str(total),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"{n}! ÷ "
                    + ("(" + " × ".join(f"{c}!" for c in counts) + ")" if len(counts) > 1 else f"{counts[0]}!")
                    + f" = {total}."
                ],
                distractors=pick(str(total), [c for c in a_d if int(c.text) != total]),
            ),
            dict(
                prompt=f"How many of these arrangements have the {many} {letter}s next to each other?",
                answer_kind="number",
                answer=near,
                answer_text=str(near),
                answer_latex=str(near),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Treat the {many} {letter}s as one block, which makes {n - copies + 1} objects to arrange (with any other repeated letters still repeated): {near}."
                ],
                distractors=pick(str(near), [c for c in b_d if int(c.text) != near and int(c.text) > 0]),
            ),
            dict(
                prompt=f"One of the arrangements is chosen at random. Find the probability that the {many} {letter}s are next to each other.",
                answer_kind="fraction",
                answer=exact_value(prob),
                answer_text=f_c,
                answer_latex=_frac_latex(prob),
                marks=2,
                sensible_range=(0, 1),
                follow_through="part_b/part_a",
                working=[f"Probability = {near} ÷ {total} = {f_c}."],
                distractors=pick(f_c, [c for c in c_d if c.text != f_c]),
            ),
            dict(
                prompt=f"How many of the arrangements have the {many} {letter}s not next to each other?",
                answer_kind="number",
                answer=apart,
                answer_text=str(apart),
                answer_latex=str(apart),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_a-part_b",
                working=[f"{total} - {near} = {apart}."],
                distractors=pick(str(apart), [c for c in d_d if int(c.text) != apart and int(c.text) > 0]),
            ),
        ],
    )


def _frac_dist(value: Fraction, tag: str) -> Distractor:
    return Distractor(rational_text(value), tag, exact_value(value))


def _frac_latex(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else rf"\frac{{{value.numerator}}}{{{value.denominator}}}"
