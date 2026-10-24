"""Everyday chance (AC9M3P01): list the possible outcomes of a simple event, and describe a chance from counts.

- ``eyz_list_outcomes`` (skill ``ey-list-possible-outcomes``, ``set``): the learner types every possible outcome
  of one chance event. Easy is a coin, a die, a spinner of named colours or a bag of counters (the colours, not
  the counters); medium lists the outcomes that fit a condition (even numbers on a die, more than a number, a
  multiple on numbered cards); hard picks one letter or digit from a word or number with repeats, or lists the
  outcomes that are not some stated ones. Every wrong list is the list one named slip reaches.
- ``eyz_chance_words`` (skill ``ey-describe-chance-from-counts``, ``mcq``): the learner chooses one of impossible,
  unlikely, even chance, likely or certain, from the counts in a bag, a spinner or a die, or from how often
  something happened in the last 10 times. A chance is only asked where it sits clearly in one word's range: at
  most 3 in 10 is unlikely, exactly 5 in 10 is even, at least 7 in 10 (and not all) is likely, so no draw is a
  matter of opinion. The five words are always offered.
- ``eyz_chance_words_exam`` (exam tier): a bag in three parts: how many counters in all, how many are not red
  (following from the total), and the word for the chance of picking one that is not red.

Answers are built backwards from a chosen word, so nothing is solved at generation time.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator
from ..slots import NAMES
from ._spread_kit import pick_distractors, whole_part, uses_rng


def _bag(items) -> str:
    """ "5 purple and 3 blue counters", or, when a count is 1, "5 purple counters and 1 blue counter"."""
    if all(c >= 2 for c, _ in items):
        parts = [f"{c} {col}" for c, col in items]
        return (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}") + " counters"
    parts = [f"{c} {col} counter{'' if c == 1 else 's'}" for c, col in items]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _a(word: str) -> str:
    """The word with its indefinite article: "an octagon", "a square"."""
    return ("an " if word[:1] in "aeiou" else "a ") + word


_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


# =============================================================================== list the possible outcomes

_COLOURS = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "white")


def _set_text(items) -> str:
    return "{" + ", ".join(str(i) for i in items) + "}"


def _ordered(items):
    return sorted(items, key=lambda v: (0, v) if isinstance(v, int) else (1, str(v)))


def _outcome_question(text: str, outcomes: list, wrong: list[tuple[list, str]], working: list[str]) -> dict:
    """A ``set`` question. ``wrong`` is ``[(list of outcomes, misconception id), ...]``, likeliest first."""
    answer = _ordered(outcomes)
    seen = {tuple(answer)}
    distractors = []
    for items, tag in wrong:
        key = tuple(_ordered(items)) if len(set(items)) == len(items) else tuple(items)
        if not items or key in seen:
            continue
        seen.add(key)
        distractors.append(Distractor(_set_text(key), tag, list(key)))
    picked = pick_distractors(
        _set_text(answer),
        distractors,
        n=4,
        unordered=True,
        lo=None,
        hi=None,
        unbounded="lists of outcomes, not numbers",
    )
    check(len(picked) >= 3, "an outcomes question needs three wrong lists")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=_set_text(answer),
        answer_latex=f"plain:{_set_text(answer)}",
        answer_kind="set",
        working=working,
        distractors=picked,
    )


def _drop_one(items: list, tag: str, rng) -> list[tuple[list, str]]:
    out = []
    for i in rng.sample(range(len(items)), min(2, len(items))):
        out.append((items[:i] + items[i + 1 :], tag))
    return out


def _easy_outcomes(rng) -> dict:
    kind = rng.choice(("coin", "die", "spinner", "spinner", "bag", "bag"))
    if kind == "coin":
        text = rng.choice(
            (
                "A coin is tossed once. List all the possible outcomes.",
                "Toss a coin. List every outcome that could happen.",
            )
        )
        outs = ["heads", "tails"]
        wrong = [
            (["heads"], "eyz-misses-an-outcome"),
            (["heads", "tails", "edge"], "eyz-adds-an-impossible-outcome"),
            (["heads", "heads", "tails"], "eyz-lists-repeats-as-outcomes"),
            (["tails"], "eyz-misses-an-outcome"),
            (["heads", "tails", "both"], "eyz-adds-an-impossible-outcome"),
        ]
        return _outcome_question(text, outs, wrong, ["A coin can land heads or tails."])
    if kind == "die":
        text = rng.choice(
            ("A die is rolled once. List all the possible outcomes.", "Roll a die. List every number it could show.")
        )
        outs = [1, 2, 3, 4, 5, 6]
        wrong = [
            ([1, 2, 3, 4, 5], "eyz-misses-an-outcome"),
            ([2, 3, 4, 5, 6], "eyz-misses-an-outcome"),
            ([0, 1, 2, 3, 4, 5, 6], "eyz-adds-an-impossible-outcome"),
            ([1, 2, 3, 4, 5, 6, 7], "eyz-adds-an-impossible-outcome"),
            ([1, 2, 3, 4, 5, 6, 6], "eyz-lists-repeats-as-outcomes"),
        ]
        return _outcome_question(text, outs, wrong, ["A die has the numbers 1 to 6, so any of them can come up."])
    if kind == "spinner":
        n = rng.randint(3, 5)
        cols = rng.sample(_COLOURS, n)
        words = ", ".join(cols[:-1]) + f" and {cols[-1]}"
        text = rng.choice(
            (
                f"A spinner has {n} equal sections: {words}. List all the possible outcomes.",
                f"A spinner is {words}, in equal sections. List every outcome.",
            )
        )
        spare = [c for c in _COLOURS if c not in cols]
        wrong = _drop_one(cols, "eyz-misses-an-outcome", rng)
        wrong.append((cols + [spare[0]], "eyz-adds-an-impossible-outcome"))
        wrong.append((cols + [cols[0]], "eyz-lists-repeats-as-outcomes"))
        wrong.append((cols[:2] if n > 3 else cols[:1], "eyz-misses-an-outcome"))
        return _outcome_question(text, cols, wrong, [f"The spinner can stop on any of its {n} colours."])
    n = rng.randint(2, 3)
    cols = rng.sample(_COLOURS, n)
    counts = rng.sample(range(1, 5), n)
    desc = _bag(list(zip(counts, cols)))
    text = f"A bag has {desc}. One is picked. List the colours it could be."
    every = [col for c, col in zip(counts, cols) for _ in range(c)]
    spare = [c for c in _COLOURS if c not in cols]
    wrong = [(every, "eyz-lists-every-counter")]
    wrong += _drop_one(cols, "eyz-misses-an-outcome", rng)
    wrong.append((cols + [spare[0]], "eyz-adds-an-impossible-outcome"))
    wrong.append(([col for c, col in zip(counts, cols) for _ in range(min(c, 2))], "eyz-lists-every-counter"))
    return _outcome_question(
        text,
        cols,
        wrong,
        [
            f"{_a(cols[0]).capitalize()} counter, {_a(cols[1])} counter"
            + (f" or {_a(cols[2])} counter" if n == 3 else "")
            + " could come out. List each colour once."
        ],
    )


def _medium_outcomes(rng) -> dict:
    kind = rng.choice(("die-even", "die-more", "die-less", "cards-multiple", "cards-more"))
    if kind == "die-even":
        odd = rng.random() < 0.5
        outs = [1, 3, 5] if odd else [2, 4, 6]
        word = "an odd" if odd else "an even"
        text = rng.choice(
            (
                f"A die is rolled. List the outcomes that are {word} number.",
                f"Roll a die. Which numbers are {word} number? List them.",
            )
        )
        other = [2, 4, 6] if odd else [1, 3, 5]
        wrong = [
            (other, "eyz-lists-the-other-group"),
            (outs[:-1], "eyz-misses-an-outcome"),
            (outs[1:], "eyz-misses-an-outcome"),
            ([1, 2, 3, 4, 5, 6], "eyz-lists-every-outcome"),
            (outs + [other[0]], "eyz-adds-a-number-that-does-not-fit"),
        ]
        return _outcome_question(
            text, outs, wrong, [f"The {'odd' if odd else 'even'} numbers on a die are {_set_text(outs)}."]
        )
    if kind in ("die-more", "die-less"):
        more = kind == "die-more"
        k = rng.randint(2, 4) if more else rng.randint(3, 5)
        outs = list(range(k + 1, 7)) if more else list(range(1, k))
        word = "more than" if more else "less than"
        text = rng.choice(
            (f"A die is rolled. List the outcomes that are {word} {k}.", f"Roll a die. List every number {word} {k}.")
        )
        boundary = sorted(outs + [k])
        other = list(range(1, k + 1)) if more else list(range(k + 1, 7))
        wrong = [
            (boundary, "eyz-includes-the-boundary-number"),
            (other, "eyz-lists-the-other-group"),
            (outs[:-1] or [k], "eyz-misses-an-outcome"),
            (outs[1:], "eyz-misses-an-outcome"),
            ([1, 2, 3, 4, 5, 6], "eyz-lists-every-outcome"),
        ]
        return _outcome_question(
            text, outs, wrong, [f"The numbers {word} {k} on a die are {_set_text(outs)}. {k} itself is not {word} {k}."]
        )
    top = rng.choice((8, 10, 12))
    if kind == "cards-multiple":
        m = rng.choice((2, 3, 4, 5))
        outs = [n for n in range(1, top + 1) if n % m == 0]
        if len(outs) < 2:
            m, outs = 2, [n for n in range(1, top + 1) if n % 2 == 0]
        text = f"Cards are numbered 1 to {top}. One is picked. List the outcomes that are multiples of {m}."
        wrong = [
            ([n for n in range(1, top + 1) if n % m != 0], "eyz-lists-the-other-group"),
            (outs[:-1], "eyz-misses-an-outcome"),
            ([m] + outs[:1] if m not in outs[:1] else outs[1:], "eyz-lists-the-factor-not-the-multiples"),
            (outs + [outs[-1] + 1 if outs[-1] + 1 <= top else 1], "eyz-adds-a-number-that-does-not-fit"),
            (list(range(1, top + 1)), "eyz-lists-every-outcome"),
        ]
        return _outcome_question(text, outs, wrong, [f"Count in {m}s up to {top}: {_set_text(outs)}."])
    k = rng.randint(top - 5, top - 2)
    outs = list(range(k + 1, top + 1))
    text = f"Cards are numbered 1 to {top}. One is picked. List the outcomes that are more than {k}."
    wrong = [
        (sorted(outs + [k]), "eyz-includes-the-boundary-number"),
        (list(range(1, k + 1)), "eyz-lists-the-other-group"),
        (outs[:-1] or [k], "eyz-misses-an-outcome"),
        (outs[1:], "eyz-misses-an-outcome"),
        (list(range(1, top + 1)), "eyz-lists-every-outcome"),
    ]
    return _outcome_question(text, outs, wrong, [f"The numbers more than {k} up to {top} are {_set_text(outs)}."])


_WORDS = (
    "BANANA",
    "MELBOURNE",
    "KOALA",
    "CANBERRA",
    "TOOWOOMBA",
    "PARRAMATTA",
    "ADELAIDE",
    "BRISBANE",
    "WOLLONGONG",
    "MISSISSIPPI",
    "TASMANIA",
    "LAUNCESTON",
)
_NUMBERS = (2023, 3114, 5225, 7707, 1991, 4004, 6336, 8118)


def _hard_outcomes(rng) -> dict:
    kind = rng.choice(("word", "word", "digits", "not"))
    if kind == "word":
        word = rng.choice(_WORDS)
        outs = sorted(set(word))
        text = f"A letter is picked from the word {word}. List the possible outcomes."
        letters = list(word)
        spare = next(c for c in "XZQJ" if c not in word)
        wrong = [
            (letters, "eyz-lists-repeats-as-outcomes"),
            (outs[:-1], "eyz-misses-an-outcome"),
            (outs + [spare], "eyz-adds-an-impossible-outcome"),
            (outs[1:], "eyz-misses-an-outcome"),
            (sorted(set(letters[: len(letters) // 2])), "eyz-reads-part-of-the-word"),
        ]
        return _outcome_question(
            text, outs, wrong, [f"{word} has the letters {_set_text(outs)}. A repeated letter is one outcome."]
        )
    if kind == "digits":
        n = rng.choice(_NUMBERS)
        digs = [int(d) for d in str(n)]
        outs = sorted(set(digs))
        text = f"A digit is picked from the number {n}. List the possible outcomes."
        wrong = [
            (digs, "eyz-lists-repeats-as-outcomes"),
            (outs[:-1], "eyz-misses-an-outcome"),
            (outs + [next(d for d in range(10) if d not in outs)], "eyz-adds-an-impossible-outcome"),
            (outs[1:], "eyz-misses-an-outcome"),
            (list(range(10)), "eyz-lists-every-outcome"),
        ]
        return _outcome_question(text, outs, wrong, [f"The digits of {n} are {_set_text(outs)}, each listed once."])
    out_a, out_b = sorted(rng.sample(range(1, 7), 2))
    outs = [v for v in range(1, 7) if v not in (out_a, out_b)]
    text = f"A die is rolled. List the outcomes that are not {out_a} or {out_b}."
    wrong = [
        ([out_a, out_b], "eyz-lists-the-other-group"),
        (outs[:-1], "eyz-misses-an-outcome"),
        (outs + [out_a], "eyz-adds-a-number-that-does-not-fit"),
        (outs[1:], "eyz-misses-an-outcome"),
        ([1, 2, 3, 4, 5, 6], "eyz-lists-every-outcome"),
    ]
    return _outcome_question(text, outs, wrong, [f"Take {out_a} and {out_b} away from 1 to 6: {_set_text(outs)}."])


@generator("eyz_list_outcomes", level=3, presets=_PRESETS)
@uses_rng
def eyz_list_outcomes(rng, tier="easy"):
    _tier(tier)
    return {"easy": _easy_outcomes, "medium": _medium_outcomes, "hard": _hard_outcomes}[tier](rng)


# =============================================================================== describe a chance in words

WORDS = ("impossible", "unlikely", "even chance", "likely", "certain")
_PHRASE = {
    "impossible": "impossible",
    "unlikely": "unlikely",
    "even chance": "an even chance",
    "likely": "likely",
    "certain": "certain",
}
_BINS = {
    "impossible": lambda p: p == 0,
    "unlikely": lambda p: 0 < p <= Fraction(3, 10),
    "even chance": lambda p: p == Fraction(1, 2),
    "likely": lambda p: Fraction(7, 10) <= p < 1,
    "certain": lambda p: p == 1,
}


def _word_of(p: Fraction) -> str | None:
    """The one word whose range holds ``p``, or None if ``p`` is in a gap (an opinion, so never asked)."""
    hits = [w for w, test in _BINS.items() if test(p)]
    return hits[0] if len(hits) == 1 else None


def sentence(event: str, word: str) -> str:
    """``"Picking red is unlikely."``: the option a learner reads, the event first."""
    return f"{event[0].upper()}{event[1:]} is {_PHRASE[word]}."


def word_in(text: str) -> str:
    """The chance word a sentence from :func:`sentence` states."""
    for word in WORDS:
        if text.endswith(f" is {_PHRASE[word]}."):
            return word
    raise ValueError(text)


def _word_options(event: str, answer: str) -> list[Distractor]:
    i = WORDS.index(answer)
    out = []
    if i + 1 < len(WORDS):
        out.append(Distractor(sentence(event, WORDS[i + 1]), "eyz-rates-the-chance-too-high"))
    if i - 1 >= 0:
        out.append(Distractor(sentence(event, WORDS[i - 1]), "eyz-rates-the-chance-too-low"))
    mirror = len(WORDS) - 1 - i
    if abs(mirror - i) > 1:
        out.append(Distractor(sentence(event, WORDS[mirror]), "eyz-opposite-chance"))
    taken = {d.text for d in out}
    for w in WORDS:
        if w != answer and sentence(event, w) not in taken:
            out.append(Distractor(sentence(event, w), "eyz-ignores-the-counts"))
    return out


def _word_question(text: str, event: str, word: str, working: list[str]) -> dict:
    answer = sentence(event, word)
    chosen = pick_distractors(
        answer, _word_options(event, word), n=4, lo=None, hi=None, unbounded="sentences, not numbers"
    )
    check(len(chosen) == 4, "four wrong sentences")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        answer_kind="mcq",
        working=working,
        distractors=chosen,
    )


def _share(word: str, rng, total: int | None = None) -> tuple[int, int]:
    """``(favourable, total)`` whose chance falls clearly in ``word``'s range, with a total from 6 to 10."""
    for _ in attempts(500, "a share in the word's range"):
        t = total or rng.choice((6, 8, 10))
        n = {"impossible": 0, "certain": t}.get(word, rng.randint(2, t - 2))
        if _word_of(Fraction(n, t)) == word:
            return n, t
    raise AssertionError("unreachable")


def _gloss(word: str) -> str:
    return {
        "impossible": "none of them",
        "unlikely": "a small share",
        "even chance": "exactly half",
        "likely": "most of them",
        "certain": "all of them",
    }[word]


def _bag_working(word: str, n: int, t: int, colour: str) -> list[str]:
    return [f"{n} of the {t} counters are {colour}.", f"{n} out of {t} is {_gloss(word)}, so it is {_PHRASE[word]}."]


def _easy_chance(rng) -> dict:
    word = rng.choice(WORDS)
    a, b = rng.sample(_COLOURS, 2)
    if word == "impossible":
        t = rng.choice((6, 8, 10))
        n = rng.randint(2, t - 2)
        other = rng.choice([c for c in _COLOURS if c not in (a, b)])
        text = f"A bag has {_bag([(n, a), (t - n, b)])}. Which sentence about picking {other} is true?"
        return _word_question(
            text, f"picking {other}", word, [f"There are no {other} counters, so picking one is impossible."]
        )
    if word == "certain":
        t = rng.choice((5, 6, 8, 10))
        text = f"A bag has {t} {a} counters. Which sentence about picking {a} is true?"
        return _word_question(text, f"picking {a}", word, [f"Every counter is {a}, so picking {a} is certain."])
    n, t = _share(word, rng)
    text = f"A bag has {_bag([(n, a), (t - n, b)])}. Which sentence about picking {a} is true?"
    return _word_question(text, f"picking {a}", word, _bag_working(word, n, t, a))


_EVIDENCE = (
    ("It rained on {n} of the last 10 days. Which sentence about rain tomorrow is true?", "rain tomorrow"),
    (
        "The bus was late on {n} of the last 10 days. Which sentence about tomorrow is true?",
        "the bus being late tomorrow",
    ),
    (
        "A spinner landed on red {n} times in 10 spins. Which sentence about the next spin is true?",
        "red on the next spin",
    ),
    (
        "Mia hit the target {n} times in 10 throws. Which sentence about the next throw is true?",
        "a hit on the next throw",
    ),
    (
        "A shop sold out of bread on {n} of the last 10 days. Which sentence about tomorrow is true?",
        "bread selling out tomorrow",
    ),
)


def _medium_chance(rng) -> dict:
    word = rng.choice(("unlikely", "even chance", "likely"))
    if rng.random() < 0.5:
        n = {"unlikely": rng.choice((2, 3)), "even chance": 5, "likely": rng.choice((7, 8, 9))}[word]
        template, event = rng.choice(_EVIDENCE)
        return _word_question(
            template.format(n=n),
            event,
            word,
            [f"{n} out of 10 is {_gloss(word)}, so it is {_PHRASE[word]}."],
        )
    # Three colours: pick one colour, or "not" a colour.
    for _ in attempts(500, "three colours with a clear chance"):
        counts = [rng.randint(2, 6) for _ in range(3)]
        t = sum(counts)
        if not 8 <= t <= 12:
            continue
        cols = rng.sample(_COLOURS, 3)
        k = rng.randrange(3)
        if rng.random() < 0.5:
            p, what = Fraction(counts[k], t), f"{_a(cols[k])} counter"
        else:
            p, what = Fraction(t - counts[k], t), f"a counter that is not {cols[k]}"
        if _word_of(p) == word:
            break
    desc = _bag(list(zip(counts, cols)))
    text = f"A bag has {desc}. Which sentence about picking {what} is true?"
    favourable = p.numerator * (t // p.denominator)
    fit = f"not {cols[k]}" if "not" in what else cols[k]
    return _word_question(
        text,
        f"picking {what}",
        word,
        [
            f"{favourable} of the {t} counters are {fit}.",
            f"{favourable} out of {t} is {_gloss(word)}, so it is {_PHRASE[word]}.",
        ],
    )


_DIE_EVENTS = (
    ("a number less than 7", "certain"),
    ("a number greater than 6", "impossible"),
    ("a 7", "impossible"),
    ("an even number", "even chance"),
    ("an odd number", "even chance"),
    ("a 6", "unlikely"),
    ("a 1", "unlikely"),
    ("a number greater than 1", "likely"),
    ("a number less than 6", "likely"),
    ("a number between 0 and 7", "certain"),
)
_DIE_WHY = {
    "certain": "Every number on a die fits, so it is certain.",
    "impossible": "No number on a die fits, so it is impossible.",
    "even chance": "3 of the 6 numbers fit, which is exactly half.",
    "unlikely": "Only 1 of the 6 numbers fits, so it is unlikely.",
    "likely": "5 of the 6 numbers fit, so it is likely.",
}


def _hard_chance(rng) -> dict:
    if rng.random() < 0.5:
        what, word = rng.choice(_DIE_EVENTS)
        text = f"A die is rolled. Which sentence about rolling {what} is true?"
        return _word_question(text, f"rolling {what}", word, [_DIE_WHY[word]])
    # A bag of three colours asked as "red or blue" (two colours together), so the learner adds before judging.
    for _ in attempts(1000, "two colours together with a clear chance"):
        word = rng.choice(("unlikely", "even chance", "likely"))
        counts = [rng.randint(2, 6) for _ in range(3)]
        t = sum(counts)
        if not 8 <= t <= 12:
            continue
        cols = rng.sample(_COLOURS, 3)
        p = Fraction(counts[0] + counts[1], t)
        if _word_of(p) == word:
            break
    desc = _bag(list(zip(counts, cols)))
    text = f"A bag has {desc}. Which sentence about picking {cols[0]} or {cols[1]} is true?"
    s = counts[0] + counts[1]
    return _word_question(
        text,
        f"picking {cols[0]} or {cols[1]}",
        word,
        [
            f"{counts[0]} + {counts[1]} = {s} of the {t} counters fit.",
            f"{s} out of {t} is {_gloss(word)}, so it is {_PHRASE[word]}.",
        ],
    )


@generator("eyz_chance_words", level=3, presets=_PRESETS)
@uses_rng
def eyz_chance_words(rng, tier="easy"):
    _tier(tier)
    return {"easy": _easy_chance, "medium": _medium_chance, "hard": _hard_chance}[tier](rng)


@generator("eyz_chance_words_exam", level=3, presets={"exam": dict(tier="exam")})
@uses_rng
def eyz_chance_words_exam(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(NAMES)
    colour = rng.choice(_COLOURS)
    other = rng.choice([c for c in _COLOURS if c != colour])
    for _ in attempts(1000, "a bag whose not-red chance is clear"):
        t = rng.randint(6, 12)
        red = rng.randint(1, t - 1)
        word = _word_of(Fraction(t - red, t))
        if word and word != "certain":
            break
    blue = t - red
    desc = _bag([(red, colour), (blue, other)])
    event = f"picking a counter that is not {colour}"
    part_c = dict(
        prompt=f"Which sentence about {event} is true?",
        answer_kind="mcq",
        answer=sentence(event, word),
        answer_text=sentence(event, word),
        answer_latex=f"plain:{sentence(event, word)}",
        marks=1,
        working=[
            f"{blue} of the {t} counters are not {colour}.",
            f"{blue} out of {t} is {_gloss(word)}, so it is {_PHRASE[word]}.",
        ],
        distractors=pick_distractors(
            sentence(event, word), _word_options(event, word), n=4, lo=None, hi=None, unbounded="sentences, not numbers"
        ),
    )
    return dict(
        text=f"{name} has a bag with {desc}. {name} picks one without looking.",
        parts=[
            whole_part(
                "How many counters are in the bag?",
                t,
                1,
                [
                    (red, "eyz-gives-one-colour-not-the-total"),
                    (blue, "eyz-gives-one-colour-not-the-total"),
                    (t + 1, None),
                    (t - 1, None),
                    (t + 2, None),
                ],
                lo=2,  # a bag with two colours holds at least two counters
                hi=t + 3,
                working=[f"{red} + {blue} = {t}."],
            ),
            whole_part(
                f"How many counters are not {colour}?",
                blue,
                1,
                [
                    (red, "eyz-counts-the-colour-not-the-rest"),
                    (t, "eyz-gives-one-colour-not-the-total"),
                    (blue + 1, None),
                    (blue - 1, None),
                    (blue + 2, None),
                ],
                lo=1,  # the counters that are not red: some, but not all of them
                hi=t - 1,
                follow_through=f"part_a - {red}",
                working=[f"{t} - {red} = {blue}."],
            ),
            part_c,
        ],
    )
