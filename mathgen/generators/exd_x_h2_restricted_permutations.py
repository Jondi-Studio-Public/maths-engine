"""Exam-tier arrangements of the letters of a word that has repeated letters, under a positional rule.

The practice tiers arrange a handful of labelled objects in a row with one pair together or apart (and, at the
top tier, one repeated object). The exam tier moves to the letters of a real word where several letters repeat
AND a whole class of letters (the vowels) is restricted, so the learner must choose between the block method,
the gap method and the end-places method, and must divide out the repeats in each:

* ``apart``: no two vowels next to each other (arrange the consonants, then choose gaps for the vowels).
* ``together``: all the vowels in one unbroken block (treat the block as one object, then arrange inside it).
* ``ends``: the word begins and ends with a consonant (fill the ends first, then arrange the rest).

The answer is built from the letter counts by formula, then re-counted by listing every distinct arrangement of
the word before the question is returned.
"""

from __future__ import annotations

import random
from collections import Counter
from math import comb, factorial

from sympy.utilities.iterables import multiset_permutations

from ..core import check, generator, pick_distractors

_VOWELS = frozenset("AEIOU")

# Words of at most seven letters, each with at least one repeated letter and at least two vowels, so every
# rule below has a non-zero count and the listing check stays cheap.
_WORDS = (
    "BANANA",
    "TOMATO",
    "POTATO",
    "LETTER",
    "COFFEE",
    "BALLOON",
    "ANTENNA",
    "ALGEBRA",
    "ADDRESS",
    "SUCCESS",
    "CARROT",
    "MAMMAL",
    "REFEREE",
    "BUTTER",
    "PEPPER",
    "BUBBLE",
    "CANNON",
    "TUNNEL",
)


def _arrangements(counts: Counter) -> int:
    """Distinct orderings of a multiset: n! over the product of each repeat's factorial."""
    total = factorial(sum(counts.values()))
    for c in counts.values():
        total //= factorial(c)
    return total


def _ways(n: int) -> str:
    return "1 way" if n == 1 else f"{n} ways"


def _split(word: str) -> tuple[Counter, Counter]:
    vowels = Counter(ch for ch in word if ch in _VOWELS)
    consonants = Counter(ch for ch in word if ch not in _VOWELS)
    return vowels, consonants


def _apart(word: str) -> int:
    vowels, consonants = _split(word)
    v, c = sum(vowels.values()), sum(consonants.values())
    if v > c + 1:
        return 0
    return _arrangements(consonants) * comb(c + 1, v) * _arrangements(vowels)


def _together(word: str) -> int:
    vowels, consonants = _split(word)
    c = sum(consonants.values())
    with_block = Counter(consonants)
    with_block["#"] = 1  # the vowel block is one extra object
    check(sum(with_block.values()) == c + 1, "block count")
    return _arrangements(with_block) * _arrangements(vowels)


def _ends(word: str) -> int:
    """Both end letters consonants: sum over every ordered pair of end letters of the middle arrangements."""
    letters = Counter(word)
    total = 0
    for first in letters:
        if first in _VOWELS:
            continue
        for last in letters:
            if last in _VOWELS:
                continue
            rest = Counter(letters)
            rest[first] -= 1
            rest[last] -= 1
            if min(rest.values()) < 0:
                continue
            rest = +rest
            total += _arrangements(rest)
    return total


def _count_by_listing(word: str, rule: str) -> int:
    """Independent route: list every distinct arrangement of the word and test the rule directly."""
    count = 0
    for letters in multiset_permutations(list(word)):
        is_vowel = [ch in _VOWELS for ch in letters]
        if rule == "apart":
            ok = not any(a and b for a, b in zip(is_vowel, is_vowel[1:]))
        elif rule == "together":
            positions = [i for i, flag in enumerate(is_vowel) if flag]
            ok = positions[-1] - positions[0] == len(positions) - 1
        else:
            ok = not is_vowel[0] and not is_vowel[-1]
        count += ok
    return count


_RULE_TEXT = {
    "apart": "no two vowels are next to each other",
    "together": "all the vowels are together in one unbroken block",
    "ends": "the arrangement begins and ends with a consonant",
}

_PATTERNS = {
    "apart": (
        "How many distinct arrangements of the letters of the word {word} are there in which {rule}?",
        "The letters of {word} are rearranged. In how many distinct ways is it true that {rule}? (A, E, I, O and U "
        "are the vowels.)",
    ),
    "together": (
        "How many distinct arrangements of the letters of the word {word} are there in which {rule}?",
        "The letters of {word} are rearranged. In how many distinct ways is it true that {rule}? (A, E, I, O and U "
        "are the vowels.)",
    ),
    "ends": (
        "How many distinct arrangements of the letters of the word {word} are there in which {rule}?",
        "The letters of {word} are rearranged. In how many distinct ways is it true that {rule}? (A, E, I, O and U "
        "are the vowels.)",
    ),
}

_FORMULA = {"apart": _apart, "together": _together, "ends": _ends}


def _all_distinct(word: str, rule: str) -> int:
    """The same rule's count if every letter were treated as different (the repeats ignored)."""
    vowel_pool = iter("AEIOU")
    consonant_pool = iter("BCDFGHJKLMNPQRSTVWXYZ")
    rebuilt = "".join(next(vowel_pool) if ch in _VOWELS else next(consonant_pool) for ch in word)
    return _FORMULA[rule](rebuilt)


@generator(
    "exd_x_h2_restricted_permutations",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_restricted_permutations(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier restricted arrangements of a word with repeated letters."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        word = rng.choice(_WORDS)
        rule = rng.choice(("apart", "together", "ends"))
        answer = _FORMULA[rule](word)
        if answer > 0:
            break
    check(_count_by_listing(word, rule) == answer, "formula disagrees with listing every arrangement")

    vowels, consonants = _split(word)
    v, c = sum(vowels.values()), sum(consonants.values())
    total = _arrangements(Counter(word))
    text = rng.choice(_PATTERNS[rule]).format(word=word, rule=_RULE_TEXT[rule])

    cands = [total, _all_distinct(word, rule)]
    if rule == "apart":
        cands.append(_together(word))
        cands.append(_arrangements(consonants) * comb(c + 1, v))  # vowel order inside the gaps forgotten
        working = [
            f"Arrange the {c} consonants first: {_ways(_arrangements(consonants))}. Choose {v} of the {c + 1} gaps for "
            f"the vowels: C({c + 1},{v}) = {comb(c + 1, v)}. Arrange the vowels in those gaps: "
            f"{_ways(_arrangements(vowels))}. Total = {answer}."
        ]
    elif rule == "together":
        cands.append(_apart(word))
        cands.append(_arrangements(Counter({**consonants, "#": 1})))  # order inside the block forgotten
        working = [
            f"Treat the vowels as one block with the {c} consonants: {_arrangements(Counter({**consonants, '#': 1}))} "
            f"arrangements. The vowels can be ordered inside the block in {_ways(_arrangements(vowels))}. "
            f"Total = {answer}."
        ]
    else:
        cands.append(
            c * (c - 1) * _arrangements(Counter(word)) // (len(word) * (len(word) - 1))
        )  # ends treated as distinct
        cands.append(_arrangements(vowels) * _arrangements(consonants))  # classes arranged separately
        working = [
            f"Sum over each ordered pair of different end consonants the arrangements of the {len(word) - 2} "
            f"remaining letters (dividing by the repeats). Total = {answer}."
        ]
    distractors = pick_distractors(str(answer), [str(x) for x in cands if x > 0])
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
