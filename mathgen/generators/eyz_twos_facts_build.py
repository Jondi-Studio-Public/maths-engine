"""Twos facts, doubles and related facts (AC9M2A03), as typed whole-number answers.

- ``eyz_twos_facts`` (skill ``ey-twos-doubles-related-facts``, ``number``): easy is one step from a picture of two
  equal groups (double a number to 10, two bags of n, n + n); medium is the twos multiplication fact either way
  round, the missing factor, and the half or related division fact; hard builds a new fact from a known one (two
  steps further along the twos), doubles and halves beyond 10, and near doubles.
- ``eyz_twos_facts_exam`` (exam tier): a three-part story with two to a unit (pairs, wheels, seats). (a) double
  the units, (b) hence the total after more units arrive (follows through from (a)), (c) hence share the total
  equally between two (follows through from (b)).

Every answer is built from a chosen number, so nothing is solved at generation time. Every stem is at most 15
words so it can be read aloud, and every distractor is the value one named error reaches.
"""

from __future__ import annotations

from ..core import VerificationError, attempts, check, generator
from ..slots import NAMES
from ._ey2a_kit import tagged_numbers
from ._pr6_chain_kit import check_chain, whole_part

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

# What the picture of "2 groups" holds, as (plural, container) for a short read-aloud stem.
_THINGS = (
    ("apples", "bags"),
    ("shells", "buckets"),
    ("pencils", "cups"),
    ("stickers", "pages"),
    ("beads", "strings"),
    ("marbles", "jars"),
    ("grapes", "bowls"),
    ("biscuits", "plates"),
)


def _question(text: str, answer: int, cands: list[tuple[int, str | None]], working: list[str], *, high: int) -> dict:
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        answer_kind="number",
        working=working,
        distractors=tagged_numbers(answer, cands, low=0, high=high),
    )


def _double_wrongs(n: int) -> list[tuple[int, str | None]]:
    """What a learner reaches for 'double n' by a slip: add two, say n again, square it, or miss by one."""
    return [
        (n + 2, "eyz-adds-two-for-double"),
        (n, "eyz-gives-the-number-again"),
        (n * n, "eyz-squares-the-number"),
        (2 * n + 1, "eyz-one-too-many-in-double"),
        (2 * n - 1, "eyz-one-too-few-in-double"),
    ]


# ============================================================================================ easy


def _easy(rng) -> dict:
    n = rng.randint(3, 10)
    kind = rng.choice(("double", "double", "groups", "add", "doubles-of"))
    ans = 2 * n
    if kind == "double":
        text = rng.choice((f"What is double {n}?", f"Double {n}. What is the answer?", f"Find double {n}."))
        working = [f"Double {n} is {n} + {n} = {ans}."]
    elif kind == "groups":
        things, box = rng.choice(_THINGS)
        text = rng.choice(
            (
                f"2 {box} with {n} {things} in each. How many {things} in all?",
                f"There are 2 {box}. Each has {n} {things}. How many {things} altogether?",
            )
        )
        working = [f"2 groups of {n} is {n} + {n} = {ans}."]
    elif kind == "add":
        text = rng.choice((f"What is {n} + {n}?", f"{n} + {n} = ?"))
        working = [f"{n} + {n} is double {n}, which is {ans}."]
    else:
        text = rng.choice((f"What is 2 groups of {n}?", f"2 lots of {n}. How many?", f"What is 2 times {n}?"))
        working = [f"2 groups of {n} is double {n}, which is {ans}."]
    return _question(text, ans, _double_wrongs(n), working, high=30)


# ============================================================================================ medium


def _medium(rng) -> dict:
    kind = rng.choice(("fact", "fact", "missing", "half", "divide", "share"))
    n = rng.randint(2, 12)
    ans_product = 2 * n
    if kind == "fact":
        a, b = rng.choice(((2, n), (n, 2)))
        text = rng.choice((f"What is {a} × {b}?", f"{a} × {b} = ?"))
        return _question(
            text,
            ans_product,
            _double_wrongs(n),
            [f"{a} × {b} is 2 groups of {n}, which is {ans_product}."],
            high=40,
        )
    if kind == "missing":
        text = rng.choice((f"2 × ? = {ans_product}. What is the missing number?", f"? × 2 = {ans_product}. Find ?"))
        cands = [
            (ans_product, "eyz-gives-the-product"),
            (2, "eyz-names-the-given-factor"),
            (n + 1, "eyz-off-by-one-half"),
            (n - 1, "eyz-off-by-one-half"),
            (ans_product * 2, "doubled-instead-of-halved"),
        ]
        return _question(text, n, cands, [f"2 × {n} = {ans_product}, so the missing number is {n}."], high=60)
    cands = [
        (ans_product, "halving-skipped"),
        (ans_product * 2, "doubled-instead-of-halved"),
        (n + 1, "eyz-off-by-one-half"),
        (n - 1, "eyz-off-by-one-half"),
        (2, "eyz-names-the-given-factor"),
    ]
    if kind == "half":
        text = rng.choice((f"What is half of {ans_product}?", f"Half of {ans_product} is what number?"))
        working = [f"{n} + {n} = {ans_product}, so half of {ans_product} is {n}."]
    elif kind == "divide":
        text = rng.choice((f"What is {ans_product} ÷ 2?", f"{ans_product} ÷ 2 = ?"))
        working = [f"2 × {n} = {ans_product}, so {ans_product} ÷ 2 = {n}."]
    else:
        text = rng.choice(
            (
                f"Share {ans_product} stickers equally between 2 children. How many each?",
                f"{ans_product} grapes are shared equally by 2 people. How many each?",
                f"Pack {ans_product} pencils into 2 cups, the same in each. How many in a cup?",
            )
        )
        working = [f"{ans_product} ÷ 2 = {n}, because 2 × {n} = {ans_product}."]
    return _question(text, n, cands, working, high=60)


# ============================================================================================ hard


def _hard_step(rng) -> dict:
    """Build a new twos fact from a known one, a few steps along the count by twos."""
    a = rng.randint(3, 9)
    step = rng.choice((-3, -2, -1, 1, 2, 3))
    b = a + step
    if not 2 <= b <= 12:
        b = a + abs(step)
    k = b - a
    known, ans = 2 * a, 2 * b
    text = rng.choice(
        (
            f"2 × {a} = {known}. Use it to work out 2 × {b}.",
            f"You know 2 × {a} = {known}. What is 2 × {b}?",
            f"{a} × 2 = {known}. Use it to find {b} × 2.",
        )
    )
    word = "more" if k > 0 else "fewer"
    cands = [
        (known + k, "eyz-changes-by-the-step-once"),
        (known + (2 if k > 0 else -2), "eyz-changes-by-one-group-only"),
        (known, "eyz-keeps-the-known-fact"),
        (ans + 1, "eyz-one-too-many-in-double"),
        (ans - 1, "eyz-one-too-few-in-double"),
        (b, "eyz-gives-the-number-again"),
    ]
    working = [f"{b} is {abs(k)} {word} than {a}, so add {abs(2 * k)} {'on' if k > 0 else 'off'}: {known} gives {ans}."]
    return _question(text, ans, cands, working, high=40)


def _hard_big_double(rng) -> dict:
    n = rng.randint(11, 49)
    ans = 2 * n
    tens, ones = divmod(n, 10)
    text = rng.choice((f"What is double {n}?", f"Double {n}. What is the answer?", f"Find double {n}."))
    cands: list[tuple[int, str | None]] = [
        (2 * tens * 10 + ones, "eyz-doubles-the-tens-only"),
        (2 * ones + tens * 10, "eyz-doubles-the-ones-only"),
        (n + 2, "eyz-adds-two-for-double"),
        (ans + 10, "eyz-extra-ten-when-regrouping"),
        (ans - 10, "eyz-lost-ten-when-regrouping"),
    ]
    if 2 * ones >= 10:
        cands.insert(0, (int(f"{2 * tens}{2 * ones}"), "eyz-doubles-digits-side-by-side"))
    working = [
        f"Double {tens * 10} is {2 * tens * 10}. Double {ones} is {2 * ones}.",
        f"{2 * tens * 10} + {2 * ones} = {ans}.",
    ]
    return _question(text, ans, cands, working, high=1000)


def _hard_big_half(rng) -> dict:
    n = rng.randint(11, 30)
    whole = 2 * n
    text = rng.choice((f"What is half of {whole}?", f"{whole} ÷ 2 = ?", f"What is {whole} ÷ 2?"))
    cands: list[tuple[int, str | None]] = [
        (whole, "halving-skipped"),
        (whole * 2, "doubled-instead-of-halved"),
        (n + 1, "eyz-off-by-one-half"),
        (n - 1, "eyz-off-by-one-half"),
        (n + 10, "eyz-extra-ten-when-regrouping"),
        (n - 10, "eyz-lost-ten-when-regrouping"),
    ]
    tens, ones = divmod(whole, 10)
    working = [f"{n} + {n} = {whole}, so half of {whole} is {n}."]
    if tens % 2 == 0 and ones % 2 == 0:
        working = [
            f"Half of {tens * 10} is {tens * 5}. Half of {ones} is {ones // 2}.",
            f"{tens * 5} + {ones // 2} = {n}.",
        ]
    return _question(text, n, cands, working, high=200)


def _hard_near_double(rng) -> dict:
    n = rng.randint(3, 12)
    up = rng.random() < 0.5
    other = n + 1 if up else n - 1
    ans = n + other
    text = rng.choice((f"What is {n} + {other}?", f"{n} + {other} = ?", f"{other} + {n}. What is the total?"))
    base = 2 * min(n, other)
    cands = [
        (base, "eyz-forgets-the-extra-one"),
        (2 * max(n, other), "eyz-uses-the-larger-double"),
        (ans + 1, "eyz-one-too-many-in-double"),
        (ans - 1, "eyz-one-too-few-in-double"),
        (n + 2, "eyz-adds-two-for-double"),
    ]
    low, high = min(n, other), max(n, other)
    working = [f"{low} + {low} = {2 * low}, and {high} is 1 more than {low}, so {2 * low} + 1 = {ans}."]
    return _question(text, ans, cands, working, high=40)


def _hard(rng) -> dict:
    return rng.choice((_hard_step, _hard_step, _hard_big_double, _hard_big_half, _hard_near_double))(rng)


@generator("eyz_twos_facts", level=2, presets=_PRESETS)
def eyz_twos_facts(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# ============================================================================================ exam

_SCENES = (
    dict(
        stem="{name} has {p} pairs of {things}. There are 2 {things} in each pair.",
        a="How many {things} does {name} have?",
        b="{name} buys {q} more pairs. How many {things} now?",
        c="{name} shares them equally with a sister. How many each?",
        things=("socks", "gloves", "shoes", "earrings"),
    ),
    dict(
        stem="A shop has {p} {things}. Each one has 2 wheels.",
        a="How many wheels are on the {things}?",
        b="{q} more {things} arrive. How many wheels now?",
        c="The wheels are packed equally into 2 boxes. How many in each box?",
        things=("bikes", "scooters", "motorbikes"),
    ),
    dict(
        stem="A class has {p} desks. Each desk seats 2 children.",
        a="How many children can sit at the desks?",
        b="{q} more desks are added. How many children can sit now?",
        c="The children are split into 2 equal teams. How many in each team?",
        things=("desks",),
    ),
    dict(
        stem="{name} packs {p} lunch boxes. Each box gets 2 {things}.",
        a="How many {things} are packed?",
        b="{q} more boxes are packed. How many {things} in all?",
        c="The {things} are split equally between 2 trays. How many on each tray?",
        things=("apples", "crackers", "mandarins", "muesli bars"),
    ),
)


@generator("eyz_twos_facts_exam", level=2, presets={"exam": dict(tier="exam")})
def eyz_twos_facts_exam(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    scene = rng.choice(_SCENES)
    name = rng.choice(NAMES)
    things = rng.choice(scene["things"])
    for _ in attempts(1000, "a chain whose answers are not printed"):
        p, q = rng.randint(3, 24), rng.randint(2, 14)
        a, b, c = 2 * p, 2 * (p + q), p + q
        text = scene["stem"].format(name=name, p=p, things=things)
        fmt = dict(name=name, p=p, q=q, things=things)
        full = text + " " + scene["b"].format(**fmt)
        try:
            check_chain(full, [a, b, c])
        except VerificationError:  # a repeated or printed answer: draw again
            continue
        break
    check(b == a + 2 * q and c * 2 == b, "the chain is consistent")
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["a"].format(**fmt),
                a,
                1,
                [
                    (p + 2, "eyz-adds-two-for-double"),
                    (p, "eyz-gives-the-number-again"),
                    (p * p, "eyz-squares-the-number"),
                    (a + 1, None),
                ],
                working=[f"2 × {p} = {a}."],
            ),
            whole_part(
                scene["b"].format(**fmt),
                b,
                1,
                [
                    (a + q, "eyz-adds-the-units-not-the-twos"),
                    (2 * q, "story-one-quantity-ignored"),
                    (a + 2, "eyz-changes-by-one-group-only"),
                    (b + 2, None),
                    (b - 2, None),
                ],
                follow_through=f"part_a + 2*{q}",
                working=[f"{q} more groups of 2 is 2 × {q} = {2 * q}.", f"{a} + {2 * q} = {b}."],
            ),
            whole_part(
                scene["c"].format(**fmt),
                c,
                1,
                [
                    (b, "halving-skipped"),
                    (2 * b, "doubled-instead-of-halved"),
                    (c + 1, "eyz-off-by-one-half"),
                    (c - 1, "eyz-off-by-one-half"),
                ],
                follow_through="part_b/2",
                working=[f"{c} + {c} = {b}, so {b} ÷ 2 = {c}."],
            ),
        ],
    )
