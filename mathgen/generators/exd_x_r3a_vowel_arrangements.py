"""Exam source of `senior-solve-problems-involving-permutations-and-restrictions-with-or`: a word's letters, a vowel rule.

The practice tiers arrange a few labelled objects with one pair together or apart. An exam question takes a real word
in which letters repeat and restricts the whole class of vowels, then builds the count in stages and finishes with a
probability:

* ``apart`` (no two vowels next to each other): (a) arrangements of the consonants, (b) ways to place the vowels in
  the gaps, in order, (c) total arrangements, (d) probability that an arrangement chosen at random obeys the rule;
* ``together`` (all vowels in one unbroken block): (a) orders of the vowels inside the block, (b) arrangements of the
  consonants and the block, (c) total arrangements, (d) the probability.

Parts (c) and (d) follow through from the earlier parts.

## Construction

Counts come from the letter multiset by formula. An independent route counts the same arrangements by choosing the
POSITIONS the vowels occupy (non-adjacent for ``apart``, a run for ``together``) and arranging the letters in them.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb, factorial

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rational_text
from ._exd_r3_distinct import distinct_part_answers

_VOWELS = frozenset("AEIOU")
_WORDS = tuple(
    """BRISBANE MELBOURNE CANBERRA TOOWOOMBA MOORABBIN BUNDABERG GEELONG NEWCASTLE BANANA TOMATO POTATO LETTER COFFEE BALLOON ANTENNA ALGEBRA ADDRESS SUCCESS CARROT MAMMAL REFEREE BUTTER PEPPER BUBBLE CANNON TUNNEL COMMITTEE SENTENCE PARALLEL TOMORROW BASSOON COTTAGE LETTUCE MATTRESS SUMMER BOTTLE CABBAGE CAPTAIN DOLLAR HAMMER KANGAROO PLATYPUS QUOKKA POSSUM COCKATOO BROOME TAMWORTH BALLARAT DEVONPORT BALMAIN CRONULLA ENMORE NEWTOWN BANKSTOWN FRANKSTON DANDENONG WERRIBEE SUNSHINE CARNEGIE ELWOOD PRAHRAN ESSENDON FOOTSCRAY PAKENHAM BERRIES BANANAS PUMPKIN POTATOES TOMATOES CARROTS ZUCCHINI AVOCADO MANDARIN PINEAPPLE FOOTBALL CRICKET NETBALL TENNIS CRICKETER SWIMMING ATHLETICS BASEBALL TEACHER STUDENT LIBRARY SCIENCE GEOGRAPHY GEOMETRY""".split()
)


def _arrangements(counts: Counter) -> int:
    total = factorial(sum(counts.values()))
    for c in counts.values():
        total //= factorial(c)
    return total


def _positions_route(word: str, rule: str, members) -> int:
    """Independent count: choose the positions of the vowels, then arrange vowels and consonants in their places."""
    n = len(word)
    vowels = Counter(ch for ch in word if ch in members)
    consonants = Counter(ch for ch in word if ch not in members)
    v = sum(vowels.values())
    per_choice = _arrangements(vowels) * _arrangements(consonants)
    count = 0
    for chosen in combinations(range(n), v):
        if rule == "apart":
            ok = all(b - a > 1 for a, b in zip(chosen, chosen[1:]))
        else:
            ok = chosen[-1] - chosen[0] == v - 1
        count += per_choice if ok else 0
    return count


_TILE_CONSONANTS = "BCDGHKLMNPRSTW"


def _tile_set(rng) -> str:
    """A multiset of 7 to 9 letter tiles, with some repeated letters, as a sorted string."""
    v = rng.randint(2, 4)
    c = rng.randint(max(v, 4), 6)
    vowels = [rng.choice("AEIOU") for _ in range(v)]
    base = rng.sample(_TILE_CONSONANTS, c - 2)
    consonants = base + [rng.choice(base), rng.choice(base)]
    return "".join(sorted(vowels + consonants))


def _frac_part(prompt, value, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=rational_text(value),
        answer_latex=latex_answer(rational_text(value)),
        marks=marks,
        sensible_range=(0, 1),
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _count_part(prompt, value, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _tag_ints(cands, answer):
    out = [Distractor(str(v), mid, v) for v, mid in cands if v != answer and v > 0]
    return pick_distractors(str(answer), out, 3)


@generator("exd_x_r3a_vowel_arrangements", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_vowel_arrangements(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a word and rule with enough distractors"):
        if rng.random() < 0.3:
            word = rng.choice(_WORDS)
            tiles = False
        else:
            word = _tile_set(rng)
            tiles = True
        rule = rng.choice(["apart", "together"])
        repeated = sorted(ch for ch, k in Counter(word).items() if k >= 2)
        if rng.random() < 0.5:
            members = _VOWELS
            group, rule_apart, rule_together, rest = (
                "vowels",
                "no two vowels are next to each other",
                ("all the vowels are together in one block"),
                "consonants",
            )
        else:
            letter = rng.choice(repeated)
            members = frozenset(letter)
            group = f"copies of the letter {letter}"
            rule_apart = f"no two copies of the letter {letter} are next to each other"
            rule_together = f"all the copies of the letter {letter} are together in one block"
            rest = "other letters"
        vowels = Counter(ch for ch in word if ch in members)
        consonants = Counter(ch for ch in word if ch not in members)
        if not vowels or not consonants:
            continue
        n, v, c = len(word), sum(vowels.values()), sum(consonants.values())
        all_arr = _arrangements(Counter(word))
        arr_v, arr_c = _arrangements(vowels), _arrangements(consonants)
        if rule == "apart":
            a_val = arr_c
            b_val = comb(c + 1, v) * arr_v
            a_c = [
                (factorial(c), "repeated-letters-not-divided"),
                (all_arr, "restriction-ignored"),
                (arr_c * arr_v, "vowels-arranged-with-consonants"),
                (factorial(c - 1), "one-letter-fewer-arranged"),
            ]
            b_c = [
                (comb(c + 1, v), "order-of-vowels-omitted"),
                (comb(c, v) * arr_v, "gaps-counted-without-the-end-gap"),
                (factorial(v) * comb(c + 1, v), "repeated-letters-not-divided"),
                (arr_v, "gap-choice-omitted"),
            ]
        else:
            a_val = arr_v
            b_val = factorial(c + 1) // _prod_fact(consonants)
            a_c = [
                (factorial(v), "repeated-letters-not-divided"),
                (all_arr, "restriction-ignored"),
                (arr_c, "wrong-class-of-letters"),
                (factorial(v) * arr_c, "repeated-letters-not-divided"),
            ]
            b_c = [
                (factorial(c + 1), "repeated-letters-not-divided"),
                (arr_c, "block-not-counted-as-an-object"),
                (factorial(n) // _prod_fact(Counter(word)), "restriction-ignored"),
                (factorial(c), "block-not-counted-as-an-object"),
            ]
        total = a_val * b_val
        prob = Fraction(total, all_arr)
        check(total == _positions_route(word, rule, members), f"counting by positions disagrees for {word} ({rule})")
        if c + 1 < v or not 0 < prob < 1 or total == all_arr:
            continue
        keys = {a_val, b_val, total}
        if len(keys) < 3 or total > 200000:
            continue
        c_c = [
            (a_val + b_val, "counts-added-not-multiplied"),
            (all_arr, "restriction-ignored"),
            (all_arr - total, "complement-reported"),
            (a_val * (b_val // arr_v if rule == "apart" else 1), "a-count-omitted"),
        ]
        d_c = [
            Distractor(
                rational_text(Fraction(total, factorial(n))),
                "repeated-letters-ignored-in-sample-space",
                Fraction(total, factorial(n)),
            ),
            Distractor(rational_text(1 - prob), "complement-reported", 1 - prob),
            Distractor(
                rational_text(Fraction(all_arr, total)) if all_arr > total else rational_text(prob / 2),
                "probability-inverted",
                Fraction(all_arr, total),
            ),
            Distractor(rational_text(Fraction(a_val, all_arr)), "partial-count-over-total", Fraction(a_val, all_arr)),
        ]
        d_c = [d for d in d_c if 0 < d.value < 1]
        a_d, b_d, c_d = _tag_ints(a_c, a_val), _tag_ints(b_c, b_val), _tag_ints(c_c, total)
        d_d = pick_distractors(rational_text(prob), d_c, 3)
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break

    rule_text = rule_apart if rule == "apart" else rule_together
    # The vowel definition matters only when the restricted group is the vowels.
    vowel_note = " The letters A, E, I, O and U are vowels and the others are consonants." if group == "vowels" else ""
    if tiles:
        listing = ", ".join(word)
        stem = (
            f"{n} letter tiles show {listing}. The tiles are arranged in a row, and arrangements that look the same "
            "are counted once." + vowel_note
        )
        subject = "the tiles"
    else:
        stem = (
            f"The {n} letters of the word {word} are arranged in a row, and arrangements that look the same are "
            "counted once." + vowel_note
        )
        subject = word
    if rule == "apart":
        pa = f"In how many different ways can the {c} {rest} be arranged in a row on their own?"
        pb = (
            f"One arrangement of the {c} {rest} is fixed. The {v} {group} are now inserted into it so that "
            f"{rule_text}. In how many different ways can the {group} be placed, in order?"
        )
    else:
        pa = f"In how many different orders can the {v} {group} be arranged among themselves?"
        pb = (
            f"The {group} must stay together as one unbroken block. In how many different ways can the {c} {rest} "
            "and this one block be arranged in a row?"
        )
    pc = f"Using your answers to (a) and (b), find the number of different arrangements of {subject} in which {rule_text}."
    pd = (
        f"One arrangement of {subject} is chosen at random. Using your answer to (c), find the probability "
        f"that {rule_text}."
    )
    return dict(
        text=stem,
        parts=[
            _count_part(pa, a_val, 1, a_d, [f"Count the distinct orderings: {a_val}."]),
            _count_part(pb, b_val, 3, b_d, [f"Count the placements: {b_val}."]),
            _count_part(
                pc,
                total,
                2,
                c_d,
                [f"{a_val} × {b_val} = {total}."],
                follow_through="part_a*part_b",
            ),
            _frac_part(
                pd,
                prob,
                3,
                d_d,
                [f"All distinct arrangements: {all_arr}. Probability = {total}/{all_arr} = {rational_text(prob)}."],
                follow_through=f"part_c/{all_arr}",
            ),
        ],
    )


def _prod_fact(counts: Counter) -> int:
    out = 1
    for c in counts.values():
        out *= factorial(c)
    return out
