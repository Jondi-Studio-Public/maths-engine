"""Exam tier for four early-years number and pattern skills: short connected word problems.

The practice tiers of these skills ask for one fact or one step. A national-assessment problem at this level is
several steps long and the learner must decide what each step is, so each generator here is a chain of two to
four parts sharing one story, where later parts use earlier answers and carry `follow_through`:

- `exd_x_pr1_story_chain`: join and separate in three stages (a comparison, a journey with a limit, a canteen
  order) -- which quantity to combine next is the learner's decision.
- `exd_x_pr1_share_chain`: share a collection equally, then reason about several of the groups, then about what
  is left.
- `exd_x_pr1_mystery_chain`: undo one or two operations to find a starting number, then use it twice.
- `exd_x_pr1_pattern_chain`: find the length of a repeating unit, count the complete repeats in a long pattern
  and count one kind of item in all of it, including a partly finished last repeat.

Every question is built backwards from chosen whole numbers, so each key is exact, and every distractor is the
result of a named, plausible error. Each part's distractors come from that part's own errors only.
"""

from __future__ import annotations

import random
import re

from mathgen.answers import Distractor
from mathgen.core import VerificationError, attempts, check, generator, latex_answer, pick_distractors
from mathgen.slots import NAMES

_FILL = 3  # distractors offered per part


def _distractors(key: int, candidates: list[tuple[int, str | None]]) -> list[Distractor]:
    """Distinct positive whole-number wrong answers, in priority order, each tagged with its error."""
    cands = [Distractor(str(v), mid, v) for v, mid in candidates if isinstance(v, int) and v > 0 and v != key]
    return pick_distractors(str(key), cands, _FILL)


def _part(prompt, value, marks, working, candidates, follow_through=None):
    check(isinstance(value, int) and value > 0, "a part's key must be a positive whole number")
    ds = _distractors(value, candidates)
    check(len(ds) == _FILL, f"only {len(ds)} distractors for {value}: {candidates}")
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=ds,
        **({"follow_through": follow_through} if follow_through else {}),
    )


def _digits_clear(stem: str, parts: list[dict]) -> bool:
    """True when no part's key appears as a number in the stem or in that part's own prompt."""
    for p in parts:
        seen = set(re.findall(r"\d+", stem + " " + p["prompt"]))
        if p["answer_text"] in seen:
            return False
    return True


def _off_by_one(key: int):
    return [(key + 1, "story-off-by-one"), (key - 1, "story-off-by-one")]


# ---------------------------------------------------------------------------------------------------------------
# Joining and separating in three stages
# ---------------------------------------------------------------------------------------------------------------

_THINGS = ("stickers", "marbles", "footy cards", "shells", "gumnuts", "bottle caps")
_TRAINS = (
    ("Central", "Redfern", "Newtown"),
    ("Flinders Street", "Richmond", "Camberwell"),
    ("Roma Street", "Milton", "Toowong"),
    ("Perth", "Claisebrook", "Maylands"),
    ("Adelaide", "Thebarton", "Woodville"),
)
_FOODS = ("sausage rolls", "sandwiches", "muffins", "meat pies", "fruit cups")


def _story_compare(rng):
    n1, n2 = rng.sample(NAMES, 2)
    obj = rng.choice(_THINGS)
    A = rng.randint(8, 25)
    d = rng.randint(3, 12)
    a = A + d
    b = A + a
    g = rng.randint(3, 15)
    c = b - g
    stem = f"{n1} has {A} {obj}. {n2} has {d} more {obj} than {n1}."
    return stem, [
        _part(
            f"How many {obj} does {n2} have?",
            a,
            1,
            [f"{A} + {d} = {a}."],
            [(A - d, "story-operation-reversed"), (A, "story-one-quantity-ignored"), (d, "story-one-quantity-ignored")]
            + _off_by_one(a),
        ),
        _part(
            f"How many {obj} do {n1} and {n2} have altogether?",
            b,
            1,
            [f"{A} + {a} = {b}."],
            [(a, "story-one-quantity-ignored"), (2 * A, None), (A + d, "story-one-quantity-ignored")] + _off_by_one(b),
            f"{A}+part_a",
        ),
        _part(
            f"{n1} and {n2} give {g} of their {obj} to the school fete. How many {obj} do they have left?",
            c,
            2,
            [f"{b} - {g} = {c}."],
            [
                (b + g, "story-operation-reversed"),
                (b, "story-one-quantity-ignored"),
                (a - g, "story-one-quantity-ignored"),
            ]
            + _off_by_one(c),
            f"part_b-{g}",
        ),
    ]


def _story_train(rng):
    s1, s2, s3 = rng.choice(_TRAINS)
    A = rng.randint(15, 45)
    u = rng.randint(8, 30)
    a = A + u
    v = rng.randint(5, a - 8)
    b = a - v
    cap = a + rng.randint(5, 25)  # seats for everyone who was ever on board
    room = cap - b
    stem = (
        f"A train leaves {s1} with {A} passengers. At {s2}, {u} more people get on. "
        f"At {s3}, {v} people get off. The train has seats for {cap} people."
    )
    return stem, [
        _part(
            f"How many passengers are on the train after it leaves {s2}?",
            a,
            1,
            [f"{A} + {u} = {a}."],
            [(A - u, "story-operation-reversed"), (A, "story-one-quantity-ignored"), (u, "story-one-quantity-ignored")]
            + _off_by_one(a),
        ),
        _part(
            f"How many passengers are on the train after it leaves {s3}?",
            b,
            1,
            [f"{a} - {v} = {b}."],
            [
                (a + v, "story-operation-reversed"),
                (a, "story-one-quantity-ignored"),
                (A - v, "story-one-quantity-ignored"),
            ]
            + _off_by_one(b),
            f"part_a-{v}",
        ),
        _part(
            "How many more passengers could get on before every seat is taken?",
            room,
            2,
            [f"{cap} - {b} = {room}."],
            [
                (cap - a, "story-one-quantity-ignored"),
                (cap + b, "story-operation-reversed"),
                (b, "story-earlier-quantity-reported"),
            ]
            + _off_by_one(room),
            f"{cap}-part_b",
        ),
    ]


def _story_canteen(rng):
    food = rng.choice(_FOODS)
    A = rng.randint(30, 60)
    s = rng.randint(8, 25)
    a = A - s
    m = rng.randint(10, 30)
    b = a + m
    short = rng.randint(4, 25)
    T = b + short
    stem = (
        f"The school canteen makes {A} {food} for lunch orders. It sells {s} {food} before recess and then "
        f"makes {m} more. Lunch orders need {T} {food} in total."
    )
    return stem, [
        _part(
            f"How many {food} are left after the sales before recess?",
            a,
            1,
            [f"{A} - {s} = {a}."],
            [(A + s, "story-operation-reversed"), (A, "story-one-quantity-ignored"), (s, "story-one-quantity-ignored")]
            + _off_by_one(a),
        ),
        _part(
            f"How many {food} does the canteen have after it makes more?",
            b,
            1,
            [f"{a} + {m} = {b}."],
            [
                (a - m, "story-operation-reversed"),
                (a, "story-one-quantity-ignored"),
                (A + m, "story-one-quantity-ignored"),
            ]
            + _off_by_one(b),
            f"part_a+{m}",
        ),
        _part(
            f"How many more {food} are needed to fill all the lunch orders?",
            short,
            2,
            [f"{T} - {b} = {short}."],
            [
                (T - a, "story-one-quantity-ignored"),
                (T + b, "story-operation-reversed"),
                (b, "story-earlier-quantity-reported"),
            ]
            + _off_by_one(short),
            f"{T}-part_b",
        ),
    ]


@generator("exd_x_pr1_story_chain", level=3, presets={"exam": dict(tier="exam")})
def exd_x_pr1_story_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Three-stage join and separate problem. See module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a story whose keys differ from its stated numbers"):
        stem, parts = rng.choice((_story_compare, _story_train, _story_canteen))(rng)
        if _digits_clear(stem, parts):
            break
    return dict(text=stem, parts=parts)


# ---------------------------------------------------------------------------------------------------------------
# Sharing equally, then several groups, then what is left
# ---------------------------------------------------------------------------------------------------------------

_PACKS = (
    ("oranges", "bag", "bags"),
    ("mandarins", "bag", "bags"),
    ("cupcakes", "tray", "trays"),
    ("pencils", "box", "boxes"),
    ("stickers", "packet", "packets"),
    ("plums", "tray", "trays"),
)
_SELL = (
    "sold at the school market",
    "given to the netball team",
    "taken to the school fete",
    "sent to the local nursing home",
)


@generator("exd_x_pr1_share_chain", level=3, presets={"exam": dict(tier="exam")})
def exd_x_pr1_share_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Equal sharing, then several groups, then the rest. See module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a sharing problem whose keys differ from its stated numbers"):
        name = rng.choice(NAMES)
        n = rng.randint(4, 9)
        size = rng.randint(3, 15)
        total = n * size
        if total > 120:
            continue
        k = rng.randint(2, n - 2)
        part_b = k * size
        left = total - part_b
        if left == part_b:  # keep (b) and (c) from sharing a value
            continue
        if rng.random() < 0.5:
            items, one, many = rng.choice(_PACKS)
            fate = rng.choice(_SELL)
            stem = f"{name} packs {total} {items} equally into {n} {many}."
            q_a = f"How many {items} are in each {one}?"
            q_b = f"{k} of the {many} are {fate}. How many {items} is that?"
            q_c = f"How many {items} are still in the other {many}?"
        else:
            items = rng.choice(("pencils", "stickers", "marbles", "tickets", "bookmarks"))
            stem = f"{total} {items} are shared equally among {n} children."
            q_a = f"How many {items} does each child get?"
            q_b = f"{k} of the children put their {items} together. How many {items} do they have?"
            q_c = f"How many {items} do the other children have altogether?"
        parts = [
            _part(
                q_a,
                size,
                1,
                [f"{total} ÷ {n} = {size}."],
                [
                    (n, "groups-count-as-size"),
                    (total - n, "groups-total-minus-count"),
                    (size + 1, "story-off-by-one"),
                    (size - 1, "story-off-by-one"),
                ],
            ),
            _part(
                q_b,
                part_b,
                1,
                [f"{k} × {size} = {part_b}."],
                [
                    (size, "groups-one-group-for-several"),
                    (k + size, "story-operation-reversed"),
                    (size * (n - k), "story-earlier-quantity-reported"),
                    (part_b + size, "story-off-by-one"),
                    (part_b - size, "story-off-by-one"),
                ],
                f"{k}*part_a",
            ),
            _part(
                q_c,
                left,
                2,
                [f"{total} - {part_b} = {left}."],
                [
                    (part_b, "story-earlier-quantity-reported"),
                    (n - k, "groups-count-for-items"),
                    (total - size, "groups-one-group-for-several"),
                    (left + size, "story-off-by-one"),
                    (left - size, "story-off-by-one"),
                ],
                f"{total}-part_b",
            ),
        ]
        if _digits_clear(stem, parts):
            break
    return dict(text=stem, parts=parts)


# ---------------------------------------------------------------------------------------------------------------
# Undo an operation to find a starting number, then use it
# ---------------------------------------------------------------------------------------------------------------


@generator("exd_x_pr1_mystery_chain", level=3, presets={"exam": dict(tier="exam")})
def exd_x_pr1_mystery_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Find a starting number from what was done to it, then use it twice. See module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a mystery-number problem whose keys differ from its stated numbers"):
        n1, n2 = rng.sample(NAMES, 2)
        variant = rng.choice(("add", "take", "both"))
        x = rng.randint(6, 30)  # the starting number
        if variant == "add":
            b = rng.randint(3, 15)
            result = x + b
            told = f"adds {b} to it and gets {result}"
            cands = [
                (result + b, "inverse-operation-not-reversed"),
                (b, "names-known-addend"),
                (result, "states-the-result"),
            ]
            working = [f"{result} - {b} = {x}."]
        elif variant == "take":
            b = rng.randint(3, min(15, x - 1))
            result = x - b
            if result < 1:
                continue
            told = f"takes {b} away from it and gets {result}"
            cands = [
                (result - b, "inverse-operation-not-reversed"),
                (b, "names-known-addend"),
                (result, "states-the-result"),
            ]
            working = [f"{result} + {b} = {x}."]
        else:
            b = rng.randint(6, 18)
            s = rng.randint(2, b - 1)
            result = x + b - s
            told = f"adds {b} to it, then takes away {s}, and gets {result}"
            cands = [
                (result + b - s, "inverse-operation-not-reversed"),
                (result - b - s, "inverse-operation-not-reversed"),
                (result - b, "story-one-quantity-ignored"),
                (result, "states-the-result"),
            ]
            working = [f"{b} - {s} = {b - s}, so the number went up by {b - s}.", f"{result} - {b - s} = {x}."]
        e = rng.randint(2, 12)
        y = x + e
        total = x + y
        stem = f"{n1} thinks of a number. {n1} {told}."
        parts = [
            _part(f"What number did {n1} think of?", x, 2, working, cands + _off_by_one(x)),
            _part(
                f"{n2} thinks of a number that is {e} more than {n1}'s number. What is {n2}'s number?",
                y,
                1,
                [f"{x} + {e} = {y}."],
                [
                    (e, "story-one-quantity-ignored"),
                    (x - e, "story-operation-reversed"),
                    (x, "story-earlier-quantity-reported"),
                ]
                + _off_by_one(y),
                f"part_a+{e}",
            ),
            _part(
                f"What is the total of {n1}'s number and {n2}'s number?",
                total,
                1,
                [f"{x} + {y} = {total}."],
                [(y, "story-one-quantity-ignored"), (x, "story-one-quantity-ignored"), (2 * x, None)]
                + _off_by_one(total),
                "part_a+part_b",
            ),
        ]
        if _digits_clear(stem, parts):
            break
    return dict(text=stem, parts=parts)


# ---------------------------------------------------------------------------------------------------------------
# A long repeating pattern: unit length, complete repeats, one kind of item
# ---------------------------------------------------------------------------------------------------------------

_COLOURS = ("red", "blue", "green", "yellow", "white", "orange", "purple")
_OBJECTS = (
    ("necklace", "beads", "bead"),
    ("bracelet", "beads", "bead"),
    ("fence", "palings", "paling"),
    ("garland", "flags", "flag"),
)


def _smallest_period(seq: list[str]) -> int:
    return next(p for p in range(1, len(seq) + 1) if all(seq[i] == seq[i % p] for i in range(len(seq))))


@generator("exd_x_pr1_pattern_chain", level=3, presets={"exam": dict(tier="exam")})
def exd_x_pr1_pattern_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Unit length, complete repeats, then one colour counted through a partial repeat. See module docstring."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a pattern whose keys differ from its stated numbers"):
        u = rng.randint(3, 6)
        palette = rng.sample(_COLOURS, rng.randint(2, 3))
        unit = [rng.choice(palette) for _ in range(u)]
        if _smallest_period(unit * 2) != u or len(set(unit)) < 2:
            continue
        target = rng.choice(sorted(set(unit)))
        per_unit = unit.count(target)
        rem = rng.randint(1, u - 1)
        extra = unit[:rem].count(target)
        if extra == 0:
            continue
        reps = rng.randint(3, 9)
        count = reps * u + rem
        total_target = reps * per_unit + extra
        thing, plural, single = rng.choice(_OBJECTS)
        shown = ", ".join(unit * 2) + ", and so on"
        stem = (
            f"A {thing} is made from {plural} that follow a repeating pattern: {shown}. "
            f"The {thing} has {count} {plural} altogether."
        )
        try:
            parts = [
                _part(
                    f"How many {plural} are in one repeat of the pattern?",
                    u,
                    1,
                    [f"The pattern starts again after {', '.join(unit)}, which is {u} {plural}."],
                    [
                        (2, "pattern-unit-assumed-two"),
                        (2 * u, None),
                        (u - 1, "story-off-by-one"),
                        (u + 1, "story-off-by-one"),
                    ],
                ),
                _part(
                    f"How many complete repeats of the pattern are on the {thing}?",
                    reps,
                    2,
                    [f"{count} ÷ {u} = {reps} remainder {rem}, so there are {reps} complete repeats."],
                    [
                        (reps + 1, "pattern-partial-unit-counted-whole"),
                        (rem, "division-remainder-as-answer"),
                        (count // 2, "pattern-unit-assumed-two"),
                        (reps - 1, "story-off-by-one"),
                    ],
                    f"floor({count}/part_a)",
                ),
                _part(
                    f"How many {target} {plural} are on the {thing}?",
                    total_target,
                    2,
                    [
                        f"Each repeat has {per_unit} {target}, so {reps} repeats have {reps * per_unit}.",
                        f"The last {rem} {single if rem == 1 else plural} {'is' if rem == 1 else 'are'} "
                        f"{', '.join(unit[:rem])}, which has {extra} {target}.",
                        f"{reps * per_unit} + {extra} = {total_target}.",
                    ],
                    [
                        (reps * per_unit, "pattern-partial-unit-dropped"),
                        ((reps + 1) * per_unit, "pattern-partial-unit-counted-whole"),
                        (reps, "story-earlier-quantity-reported"),
                        (total_target + 1, "story-off-by-one"),
                        (total_target - 1, "story-off-by-one"),
                    ],
                    f"{per_unit}*part_b+{extra}",
                ),
            ]
        except VerificationError:
            continue  # too few distinct wrong answers for this draw
        if _digits_clear(stem, parts):
            break
    return dict(text=stem, parts=parts)
