"""Building the sample space of a two-stage experiment: `r3j_sample_space` (easy, medium, hard) and the exam chain
`exd_x_r3j_sample_space_events`.

``build-sample-space`` (``statistics-probability``). Outcomes are ordered pairs (first stage, second stage), written
the way the question asks, for example (H, 3) or (2, 5).

* easy: list every outcome of a coin and a three-sector spinner (a set of ordered pairs).
* medium: two dice or spinners; list the ordered pairs that satisfy a stated property (a sum, a product, a
  difference, or the larger number).
* hard: list the outcomes of a compound event (an "or" of two conditions, a three-toss condition, or "exactly one"
  / "neither") as a set.
* exam (``exd_x_r3j_sample_space_events``): the number of outcomes, the outcomes in an event, the probability of the
  event and the probability that it does not happen, following through from the size of the sample space.

Each event is rebuilt a second way, by brute-force filtering of the product, against the property written as a
formula; counts also agree with the multiplication principle where it applies.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import PEOPLE, choice_question, fraction_part, number_part, set_part, single_set

_STAGE_NOTE = "Write each outcome as an ordered pair, for example (H, 1), and separate the pairs with commas."


def _space(first, second):
    return [(a, b) for a in first for b in second]


def _easy(rng):
    name = rng.choice(PEOPLE)
    first, second = rng.choice(
        [
            (["H", "T"], [1, 2, 3]),
            (["H", "T"], ["R", "G", "B"]),
            (["R", "G"], [1, 2, 3]),
            (["A", "B", "C"], [1, 2]),
        ]
    )
    sample = _space(first, second)
    if first == ["H", "T"] and isinstance(second[0], int):
        setup = f"{name} tosses a coin and spins a spinner with three equal sectors numbered 1, 2 and 3."
    elif first == ["H", "T"]:
        setup = f"{name} tosses a coin and spins a spinner with three equal sectors coloured red (R), green (G) and blue (B)."
    elif first == ["R", "G"]:
        setup = (
            f"{name} spins a spinner with a red (R) sector and a green (G) sector, then a spinner numbered 1, 2 and 3."
        )
    else:
        setup = f"{name} picks a card marked A, B or C from a bag, then tosses a coin that is marked 1 on one side and 2 on the other."
    text = f"{setup} List the sample space for the two stages. {_STAGE_NOTE}"
    n1, n2 = len(first), len(second)
    check(len(sample) == n1 * n2, "sample size disagrees with the multiplication principle")
    slips = [
        (sample[:-1], "r3j-sample-outcome-omitted"),
        (sample[: len(sample) // 2] + sample[len(sample) // 2 + 1 :], "r3j-sample-outcome-omitted"),
        (list(first) + list(second), "r3j-sample-stages-listed-separately"),
        (sample[:-1] + [(second[0], first[0])], "r3j-sample-pair-order-swapped"),
        (
            sample
            + [
                (first[0], max(x for x in second if isinstance(x, int)) + 1)
                if isinstance(second[0], int)
                else (first[0], "X")
            ],
            "r3j-sample-impossible-outcome-added",
        ),
    ]
    working = [
        f"Pair each of the {n1} first outcomes with each of the {n2} second outcomes: {n1} x {n2} = {n1 * n2} pairs."
    ]
    return single_set(text, sample, working, slips)


# property name -> (description text, predicate, wrong predicates with their tags)
def _property(kind, k):
    if kind == "sum":
        return (
            f"the two numbers add to {k}",
            lambda a, b: a + b == k,
            [
                (lambda a, b: a * b == k, "r3j-sample-product-for-sum"),
                (lambda a, b: a + b == k + 1, "r3j-sample-target-off-by-one"),
                (lambda a, b: abs(a - b) == k, "r3j-sample-difference-for-sum"),
            ],
        )
    if kind == "product":
        return (
            f"the two numbers multiply to {k}",
            lambda a, b: a * b == k,
            [
                (lambda a, b: a + b == k, "r3j-sample-sum-for-product"),
                (lambda a, b: a * b == k + 1, "r3j-sample-target-off-by-one"),
                (lambda a, b: a * b <= k, "r3j-sample-at-most-for-exactly"),
            ],
        )
    if kind == "difference":
        return (
            f"the two numbers differ by {k}",
            lambda a, b: abs(a - b) == k,
            [
                (lambda a, b: a - b == k, "r3j-sample-difference-one-way-only"),
                (lambda a, b: a + b == k, "r3j-sample-sum-for-difference"),
                (lambda a, b: abs(a - b) <= k, "r3j-sample-at-most-for-exactly"),
            ],
        )
    return (
        f"the larger of the two numbers is {k}",
        lambda a, b: max(a, b) == k,
        [
            (lambda a, b: min(a, b) == k, "r3j-sample-smaller-for-larger"),
            (lambda a, b: a == k, "r3j-sample-first-number-only"),
            (lambda a, b: max(a, b) <= k, "r3j-sample-at-most-for-exactly"),
        ],
    )


def _draw_event(rng, dice_a, dice_b, lo=2, hi=6):
    sample = _space(range(1, dice_a + 1), range(1, dice_b + 1))
    for _ in attempts(500, "an event with a few outcomes and clear wrong routes"):
        kind = rng.choice(["sum", "sum", "product", "difference", "larger"])
        k = {
            "sum": rng.randint(3, dice_a + dice_b - 1),
            "product": rng.choice([4, 6, 8, 9, 10, 12]),
            "difference": rng.randint(1, min(dice_a, dice_b) - 1),
            "larger": rng.randint(3, max(dice_a, dice_b)),
        }[kind]
        describe, pred, wrongs = _property(kind, k)
        event = [o for o in sample if pred(*o)]
        if not lo <= len(event) <= hi or len(event) == len(sample):
            continue
        wrong_sets = []
        for wp, tag in wrongs:
            w = [o for o in sample if wp(*o)]
            if w and w != event:
                wrong_sets.append((w, tag))
        mirror = sorted({tuple(sorted(o)) for o in event})
        if len(mirror) < len(event):
            wrong_sets.append((mirror, "r3j-sample-ordered-pairs-collapsed"))
        if len(wrong_sets) >= 3:
            return sample, event, describe, wrong_sets
    raise AssertionError  # pragma: no cover


def _medium(rng):
    dice = rng.choice([(4, 4), (6, 6), (4, 6), (3, 5)])
    sample, event, describe, wrong_sets = _draw_event(rng, *dice)
    name = rng.choice(PEOPLE)
    if dice[0] == dice[1] and dice[0] in (4, 6):
        setup = f"{name} rolls two fair {'six' if dice[0] == 6 else 'four'}-sided dice, one blue and one red."
    else:
        setup = f"{name} spins a spinner numbered 1 to {dice[0]} and then a spinner numbered 1 to {dice[1]}."
    text = (
        f"{setup} List all the outcomes in which {describe}. Write each outcome as an ordered pair "
        "(first number, second number), and separate the pairs with commas."
    )
    working = [
        f"There are {dice[0]} x {dice[1]} = {len(sample)} outcomes in the sample space.",
        f"The pairs where {describe} are: " + ", ".join(f"({a}, {b})" for a, b in event) + ".",
    ]
    return single_set(text, event, working, wrong_sets)


def _fmt(outcomes):
    return ", ".join(o if isinstance(o, str) else f"({o[0]}, {o[1]})" for o in outcomes)


def _hard_dice_list(rng):
    for _ in attempts(500, "a compound event with several clear wrong lists"):
        sides = rng.choice([4, 5, 5, 6])
        sample = _space(range(1, sides + 1), range(1, sides + 1))
        t = rng.randint(sides + 2, 2 * sides - 1)
        d = rng.randint(2, sides - 2)
        only_sum = [o for o in sample if o[0] + o[1] >= t]
        only_diff = [o for o in sample if abs(o[0] - o[1]) == d]
        both = [o for o in only_sum if o in only_diff]
        event = [o for o in sample if o in only_sum or o in only_diff]
        one_way = [o for o in event if o in only_sum or o[0] > o[1]]
        if both and len(event) <= 14 and len({tuple(x) for x in (event, only_sum, only_diff, both, one_way)}) == 5:
            break
    else:  # pragma: no cover
        raise AssertionError
    name = rng.choice(PEOPLE)
    who = (
        f"{name} rolls two fair {sides}-sided dice, numbered 1 to {sides}"
        if sides in (4, 6)
        else f"{name} spins two spinners, each numbered 1 to {sides}"
    )
    text = (
        f"{who}, and records the pair of numbers shown (first, second). "
        f"List the outcomes with a sum of at least {t} or a difference of exactly {d}. "
        "(The difference means the larger number minus the smaller.) "
        "Write each outcome as an ordered pair, and separate the pairs with commas."
    )
    check(len(event) == len(only_sum) + len(only_diff) - len(both), "inclusion-exclusion disagrees with filtering")
    slips = [
        (only_sum, "r3j-sample-one-condition-only"),
        (only_diff, "r3j-sample-one-condition-only"),
        (both, "r3j-sample-and-for-or"),
        (one_way, "r3j-sample-difference-one-way-only"),
    ]
    working = [
        f"Sum at least {t}: {_fmt(only_sum)}.",
        f"Difference exactly {d}: {_fmt(only_diff)}.",
        f"The union, listing each outcome once, is {_fmt(event)}: {len(event)} outcomes.",
    ]
    return single_set(text, event, working, slips)


def _hard_coin_list(rng):
    flips = 3
    low = 2
    sample = ["".join(o) for o in product("HT", repeat=flips)]
    kind = rng.choice(["at least", "no more than", "not"])
    if kind == "at least":
        event = [o for o in sample if o.count("T") >= low]
        desc = f"at least {low} tails"
        wrong = [
            ([o for o in sample if o.count("T") == low], "r3j-sample-exactly-for-at-least"),
            ([o for o in sample if o.count("T") <= low], "r3j-sample-at-most-for-at-least"),
            (event[:-1], "r3j-sample-outcome-omitted"),
            ([o for o in sample if o.count("T") >= low - 1], "r3j-sample-one-condition-only"),
        ]
    elif kind == "no more than":
        event = [o for o in sample if o.count("H") <= 1]
        desc = "no more than 1 head"
        wrong = [
            ([o for o in sample if o.count("H") == 1], "r3j-sample-exactly-for-at-least"),
            ([o for o in sample if o.count("H") >= 1], "r3j-sample-at-most-for-at-least"),
            (event[:-1], "r3j-sample-outcome-omitted"),
            ([o for o in sample if o.count("H") <= 2], "r3j-sample-one-condition-only"),
        ]
    else:
        event = [o for o in sample if o[0] != o[1]]
        desc = "the first two tosses different"
        wrong = [
            ([o for o in sample if o[0] == o[1]], "r3j-sample-complement-listed"),
            ([o for o in sample if o[0] != o[2]], "r3j-sample-wrong-positions-compared"),
            (event[:-1], "r3j-sample-outcome-omitted"),
            (sorted({o[:2] for o in event}), "r3j-sample-stages-listed-separately"),
        ]
    name = rng.choice(PEOPLE)
    text = (
        f"{name} tosses a coin {flips} times and records the results in order, for example HTH. "
        f"List all the outcomes with {desc}. Separate the outcomes with commas."
    )
    wrong = [(w, t) for w, t in wrong if w and sorted(w) != sorted(event)]
    working = [f"The {len(sample)} possible outcomes are {_fmt(sample)}.", f"Those with {desc} are {_fmt(event)}."]
    return single_set(text, event, working, wrong)


def _hard_none_listed(rng):
    for _ in attempts(500, "a neither/exactly-one event with clear wrong lists"):
        a_sides, b_sides = rng.choice([(4, 4), (5, 5), (4, 5), (6, 4)])
        k = rng.randint(2, min(a_sides, b_sides))
        sample = _space(range(1, a_sides + 1), range(1, b_sides + 1))
        mode = rng.choice(["exactly one", "neither"])
        shows = lambda o: (o[0] == k) + (o[1] == k)  # noqa: E731
        event = [o for o in sample if shows(o) == (1 if mode == "exactly one" else 0)]
        if mode == "exactly one":
            wrong = [
                ([o for o in sample if shows(o) >= 1], "r3j-sample-at-least-one-for-exactly-one"),
                ([o for o in sample if shows(o) == 2], "r3j-sample-and-for-or"),
                ([o for o in sample if o[0] == k], "r3j-sample-first-number-only"),
                ([o for o in sample if shows(o) == 0], "r3j-sample-complement-listed"),
            ]
            desc = f"exactly one of the spinners shows {k}"
        else:
            wrong = [
                ([o for o in sample if shows(o) >= 1], "r3j-sample-complement-listed"),
                ([o for o in sample if o[0] != k], "r3j-sample-first-number-only"),
                ([o for o in sample if shows(o) == 0 and o[0] != o[1]], "r3j-sample-outcome-omitted"),
                (event[:-1], "r3j-sample-outcome-omitted"),
            ]
            desc = f"neither spinner shows {k}"
        wrong = [(w, t) for w, t in wrong if w and sorted(w) != sorted(event)]
        if 3 <= len(event) <= 12 and len(wrong) >= 3:
            break
    else:  # pragma: no cover
        raise AssertionError
    name = rng.choice(PEOPLE)
    text = (
        f"{name} spins a spinner numbered 1 to {a_sides} and then a spinner numbered 1 to {b_sides}. "
        f"List all the outcomes in which {desc}. Write each outcome as an ordered pair "
        "(first number, second number), and separate the pairs with commas."
    )
    working = [
        f"There are {a_sides} x {b_sides} = {len(sample)} outcomes in the sample space.",
        f"The outcomes in which {desc} are {_fmt(event)}.",
    ]
    return single_set(text, event, working, wrong)


@generator(
    "r3j_sample_space",
    level=9,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_sample_space(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_dice_list, _hard_coin_list, _hard_none_listed])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3j_sample_space_events", level=13, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_sample_space_events(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "spinners with a clean event"):
        a, b = rng.choice([(4, 5), (3, 6), (4, 6), (5, 5), (3, 5), (4, 4)])
        sample, event, describe, wrong_sets = _draw_event(rng, a, b, lo=3, hi=6)
        n, k = len(sample), len(event)
        if Fraction(k, n) != Fraction(1, 2) and k < n:
            break
    name = rng.choice(PEOPLE)
    spinners = (
        f"two spinners, each numbered 1 to {a},"
        if a == b
        else f"a spinner numbered 1 to {a} and a spinner numbered 1 to {b},"
    )
    text = (
        f"{name} spins {spinners} once each, and records the pair of numbers as (first spinner, second spinner). "
        f"Every outcome is equally likely. Let E be the event that {describe}."
    )
    p = Fraction(k, n)
    parts = [
        number_part(
            "How many outcomes are in the sample space?",
            Fraction(n),
            0,
            1,
            [f"{a} x {b} = {n}."],
            [
                (a + b, "r3j-sample-added-not-multiplied"),
                (n - 1, "r3j-sample-count-one-short"),
                (2 * (a + b), "r3j-sample-added-not-multiplied"),
            ],
            sensible_range=(1, 100),
        ),
        set_part(
            "List the outcomes in E.",
            event,
            2,
            ["The pairs where " + describe + ": " + ", ".join(f"({x}, {y})" for x, y in event) + "."],
            wrong_sets,
        ),
        fraction_part(
            "Find P(E).",
            p,
            2,
            [f"P(E) = {k}/{n}."],
            [
                (Fraction(k, a + b), "r3j-sample-denominator-added"),
                (Fraction(1, k), "r3j-sample-one-over-count"),
                (Fraction(n - k, n), "r3j-sample-complement-given"),
                (Fraction(k + 1, n), "r3j-sample-count-one-over"),
            ],
            follow_through=f"{k}/part_a",
        ),
        fraction_part(
            "Find the probability that E does not happen.",
            1 - p,
            1,
            [f"P(not E) = 1 - {k}/{n} = {n - k}/{n}."],
            [
                (p, "r3j-sample-event-given-for-complement"),
                (Fraction(1, n - k), "r3j-sample-one-over-count"),
                (Fraction(n - k - 1, n), "r3j-sample-count-one-short"),
                (1 - Fraction(k + 1, n), "r3j-sample-count-one-over"),
            ],
            follow_through="1 - part_c",
        ),
    ]
    return dict(text=text, parts=parts)
