"""Generates `fr2b_check_reasonable` questions for `f10-check-given-answer-reasonable` (AC9M3N05, AC9M4N07, AC9M5N08).

`answer: number`; every question is two parts: (a) the learner types an estimate (a `number` part), then (b) judges
the answer someone else has stated against it (an `mcq` part of four options).

Scope: using an estimate to judge a stated answer, not to compute one. The neighbouring skill
`f10-estimate-counts-and-check-reasonableness` rounds and computes; this one decides whether a stated answer
survives the check. Each stated answer is either exactly right or wrong by a slip an estimate can catch, and the
generator only keeps a draw where the estimate separates the two by more than three times the most the rounding can
move it, so the verdict never rests on a close call.

Ladder (reasoning escalation, not size):
  easy:   two three-digit numbers rounded to the nearest ten, added or subtracted; a wrong answer is out by 100.
  medium: two four-digit numbers rounded to the nearest hundred, added or subtracted; a wrong answer is out by 1 000.
  hard:   money: a number of items at a price with cents, change from a note, price rounded to the nearest dollar; a
          wrong answer forgets the quantity, gives the cost instead of the change, or slips a decimal place.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._fr2b_kit import build, int_dist, tagged

_TIERS = ("easy", "medium", "hard")
_NAMES = ("Mia", "Tom", "Aisha", "Liam", "Priya", "Jack", "Zoe", "Noah", "Isla", "Ravi")

OK = "It is reasonable: it is close to the estimate"
HIGH = "It is too high: it is much more than the estimate"
LOW = "It is too low: it is much less than the estimate"
CANT = "You cannot tell: an estimate cannot check an exact answer"

_VERDICT_TEXT = {"ok": OK, "high": HIGH, "low": LOW}


def _verdict(stated: Fraction, estimate: Fraction, tolerance: Fraction) -> str:
    if abs(stated - estimate) <= tolerance:
        return "ok"
    return "high" if stated > estimate else "low"


def _verdict_options(key: str) -> list[Distractor]:
    if key == "ok":
        return [
            Distractor(HIGH, "fr2b-rejects-a-reasonable-answer"),
            Distractor(LOW, "fr2b-rejects-a-reasonable-answer"),
            Distractor(CANT, "fr2b-thinks-an-estimate-cannot-check"),
        ]
    other = HIGH if key == "low" else LOW
    return [
        Distractor(OK, "fr2b-accepts-an-answer-without-comparing"),
        Distractor(other, "fr2b-compares-in-the-wrong-direction"),
        Distractor(CANT, "fr2b-thinks-an-estimate-cannot-check"),
    ]


def _round_to(value: int, unit: int) -> int:
    """Nearest multiple of `unit`, halves up."""
    return (value + unit // 2) // unit * unit


def _floor_to(value: int, unit: int) -> int:
    return value // unit * unit


def _parts(estimate_prompt, estimate, est_distractors, verdict_key, name, working):
    return [
        dict(
            prompt=estimate_prompt,
            answer_kind="number",
            answer=estimate,
            answer_text=str(estimate),
            marks=2,
            sensible_range=(0, None),
            working=working[:1],
            distractors=est_distractors,
        ),
        dict(
            prompt=f"Is {name}'s answer reasonable? Compare it with your estimate.",
            answer_kind="mcq",
            answer=_VERDICT_TEXT[verdict_key],
            answer_text=_VERDICT_TEXT[verdict_key],
            marks=1,
            working=working[1:],
            distractors=_verdict_options(verdict_key),
        ),
    ]


def _whole_number_estimates(a, b, unit, subtract, estimate):
    ra, rb = _round_to(a, unit), _round_to(b, unit)
    big_a, big_b = _round_to(a, unit * 10), _round_to(b, unit * 10)
    sign = -1 if subtract else 1
    options = [
        (_floor_to(a, unit) + sign * _floor_to(b, unit), "fr2b-estimate-rounds-down-instead-of-to-nearest"),
        (ra + sign * b, "fr2b-estimate-rounds-only-one-number"),
        (ra - sign * rb, "fr2b-estimate-uses-the-wrong-operation"),
        (big_a + sign * big_b, "fr2b-estimate-rounds-to-the-wrong-place"),
        (a + sign * b, "fr2b-estimate-does-not-round"),
    ]
    return [Distractor(str(v), mid, v) for v, mid in options if v > 0 and v != estimate]


def _whole_number_tier(rng, unit, low, high, slip, tolerance, contexts, ask):
    """Easy and medium: add or subtract two whole numbers rounded to `unit`."""
    for _ in attempts(2000, "a draw whose estimate separates the right answer from the wrong ones"):
        a, b = rng.randint(low, high), rng.randint(low, high)
        subtract = rng.random() < 0.5
        if subtract and a <= b + 2 * unit:
            continue
        exact = a - b if subtract else a + b
        if a % unit == 0 or b % unit == 0 or abs(a - b) < 2 * unit * 3:
            continue
        shape = rng.choice(("right", "right", "high", "low"))
        stated = exact if shape == "right" else exact + (slip if shape == "high" else -slip)
        ra, rb = _round_to(a, unit), _round_to(b, unit)
        estimate = ra - rb if subtract else ra + rb
        if estimate <= 0 or stated <= 0:
            continue
        key = _verdict(Fraction(stated), Fraction(estimate), Fraction(tolerance))
        reasonable = key == "ok"
        if reasonable != (shape == "right"):
            continue
        if not reasonable and abs(stated - estimate) <= 3 * tolerance:
            continue
        if abs(exact - estimate) > tolerance:
            continue
        if len({d.text for d in _whole_number_estimates(a, b, unit, subtract, estimate)}) < 3:
            continue
        break
    check(abs(exact - estimate) <= tolerance, "the estimate is not within the rounding bound of the exact answer")
    name = rng.choice(_NAMES)
    setting = rng.choice(contexts["sub" if subtract else "add"])
    stem = setting.format(a=_g(a), b=_g(b), name=name, s=_g(stated))
    wrong_estimates = _whole_number_estimates(a, b, unit, subtract, estimate)
    unit_name = {10: "ten", 100: "hundred"}[unit]
    working = [
        f"{_g(a)} rounds to {_g(ra)} and {_g(b)} rounds to {_g(rb)}, so the estimate is "
        f"{_g(ra)} {'-' if subtract else '+'} {_g(rb)} = {_g(estimate)}.",
        f"The estimate is {_g(estimate)} and {name} says {_g(stated)}, which is "
        + (
            "close, so it is reasonable."
            if reasonable
            else f"{'much more' if key == 'high' else 'much less'} than the estimate."
        ),
    ]
    prompt = ask["sub" if subtract else "add"].format(unit_name=unit_name)
    return dict(
        text=stem,
        parts=_parts(prompt, estimate, tagged(str(estimate), wrong_estimates), key, name, working),
        working=[],
    )


def _g(n: int) -> str:
    return f"{n:,}".replace(",", " ")


_EASY_CONTEXTS = {
    "add": [
        "{a} people visited a school fair on Saturday and {b} on Sunday. {name} says {s} people visited altogether.",
        "A bakery sold {a} pies in June and {b} pies in July. {name} says it sold {s} pies in those two months.",
    ],
    "sub": [
        "A library had {a} books on its shelves. {b} were borrowed. {name} says {s} books are left on the shelves.",
        "A farmer had {a} apples to sell and sold {b} of them. {name} says {s} apples are left.",
    ],
}
_EASY_ASK = {
    "add": "Round each number to the nearest {unit_name}, then add them to estimate the total.",
    "sub": "Round each number to the nearest {unit_name}, then subtract to estimate how many are left.",
}
_MEDIUM_CONTEXTS = {
    "add": [
        "A truck drove {a} km in March and {b} km in April. {name} says it drove {s} km altogether.",
        "A football club sold {a} tickets to its first game and {b} to its second. {name} says it sold {s} tickets in total.",
    ],
    "sub": [
        "A stadium holds {a} people. {b} seats are empty. {name} says {s} people are in the stadium.",
        "A warehouse had {a} parcels. {b} were sent out. {name} says {s} parcels are left in the warehouse.",
    ],
}
_MEDIUM_ASK = {
    "add": "Round each number to the nearest {unit_name}, then add them to estimate the total.",
    "sub": "Round each number to the nearest {unit_name}, then subtract to estimate the answer.",
}


def _easy(rng):
    return _whole_number_tier(rng, 10, 120, 890, 100, 10, _EASY_CONTEXTS, _EASY_ASK)


def _medium(rng):
    return _whole_number_tier(rng, 100, 1200, 8900, 1000, 100, _MEDIUM_CONTEXTS, _MEDIUM_ASK)


def _dollars(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def _hard(rng):
    for _ in attempts(3000, "a purchase whose estimate separates the right change from the wrong ones"):
        quantity = rng.randint(3, 9)
        price = rng.randint(3, 19) * 100 + rng.choice((95, 95, 99, 50, 25, 75))
        note = rng.choice((50, 100, 100))
        cost = quantity * price
        if cost >= note * 100 - 300:
            continue
        change = note * 100 - cost
        rounded_price = _round_to(price, 100)
        estimate = note - quantity * rounded_price // 100
        tolerance = Fraction(quantity, 2)
        if estimate <= 0 or abs(Fraction(change, 100) - estimate) > tolerance:
            continue
        shape = rng.choice(("right", "right", "no-quantity", "cost", "slip", "slip"))
        stated_cents = {
            "right": change,
            "no-quantity": note * 100 - price,
            "cost": cost,
            "slip": change + rng.choice((1000, -1000)),
        }[shape]
        if stated_cents <= 0:
            continue
        stated = Fraction(stated_cents, 100)
        key = _verdict(stated, Fraction(estimate), tolerance)
        if (key == "ok") != (shape == "right"):
            continue
        if key != "ok" and abs(stated - estimate) <= 3 * tolerance:
            continue
        break
    name = rng.choice(_NAMES)
    item = rng.choice(("exercise books", "pencil cases", "water bottles", "sports caps"))
    stem = phrase(
        rng,
        [
            "{name} buys {q} {item} at ${p} each and pays with a ${n} note. {name} says the change is ${s}.",
            "At the school shop, {name} buys {q} {item} for ${p} each and pays with a ${n} note. {name} works out the change as ${s}.",
        ],
        name=name,
        q=quantity,
        item=item,
        p=_dollars(price),
        n=note,
        s=_dollars(stated_cents),
    )
    wrong = [
        int_dist(note - quantity * (price // 100), "fr2b-estimate-rounds-down-instead-of-to-nearest"),
        int_dist(note - rounded_price // 100, "fr2b-estimate-forgets-the-quantity"),
        int_dist(quantity * rounded_price // 100, "fr2b-estimate-gives-the-cost-not-the-change"),
        int_dist(note + quantity * rounded_price // 100, "fr2b-estimate-uses-the-wrong-operation"),
    ]
    wrong = [d for d in wrong if int(d.text) > 0]
    working = [
        f"${_dollars(price)} rounds to ${rounded_price // 100}, so {quantity} items cost about ${quantity * rounded_price // 100} "
        f"and the change from ${note} is about ${estimate}.",
        f"The estimate is ${estimate} and {name} says ${_dollars(stated_cents)}, which is "
        + (
            "close, so it is reasonable."
            if key == "ok"
            else f"{'much more' if key == 'high' else 'much less'} than the estimate."
        ),
    ]
    prompt = f"Round the price to the nearest dollar, then estimate the change from ${note} in dollars."
    return dict(
        text=stem,
        parts=_parts(prompt, estimate, tagged(str(estimate), wrong), key, name, working),
        working=[],
    )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_check_reasonable", level=4, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_check_reasonable(rng, tier="easy"):
    """Estimate, then judge a stated answer against the estimate; tiers in the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
