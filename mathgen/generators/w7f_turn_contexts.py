"""``w7f_turn_contexts`` (AC9M3M05, ``f10-angle-turn-right-reference``): turns met in other settings, counted in right
angles, so the skill is no longer a single "a quarter turn is how many right angles" shape.

Ladder (reasoning escalation, not size):
  easy:   a clock hand or a compass pointer moves clockwise between two of its four main positions; count the
          right angles it turned through.
  medium: some whole turns and then a part turn; the learner must know a whole turn is four right angles.
  hard:   a net turn (clockwise then back anticlockwise) or the turn left to finish a whole turn.

The key is built in degrees (a right angle is 90 degrees) and checked in fractions of a whole turn.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator
from mathgen.generators._stat_display_common import graded_wrong, number_text

TOPIC = "w7f_turn_contexts"
TIERS = ("easy", "medium", "hard")

_CLOCK = (12, 3, 6, 9)
_COMPASS = ("north", "east", "south", "west")
_PART = {1: "a quarter turn", 2: "a half turn", 3: "a three-quarter turn"}
_DEVICES = ("wheel", "dial", "gate", "turntable", "propeller")


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _dev(rng):
    return _pick(rng, _DEVICES)


def _article(w):
    return "an" if w[0] in "aeiou" else "a"


def _easy(rng):
    k = rng.randint(1, 3)
    if rng.random() < 0.5:
        i = rng.randrange(4)
        a, b = _CLOCK[i], _CLOCK[(i + k) % 4]
        text = (
            f"The minute hand of a clock points at the {a}. It moves clockwise until it points at the {b}. "
            "How many right angles has it turned through?"
        )
        wrong_way = 4 - k
        reads_numbers = abs(b - a) if abs(b - a) != k else None
        cand = [(wrong_way, "turn-goes-the-wrong-way"), (k + 1, "turn-counts-the-start-position")]
        if reads_numbers:
            cand.append((reads_numbers, "turn-reads-the-clock-numbers"))
        cand.append((b if b != k else None, "turn-reads-the-clock-numbers"))
        working = [f"From the {a} to the {b} the hand passes {k} of the four quarter-turn marks.", f"That is {k}."]
    else:
        i = rng.randrange(4)
        a, b = _COMPASS[i], _COMPASS[(i + k) % 4]
        text = f"A robot faces {a}. It turns clockwise until it faces {b}. How many right angles has it turned through?"
        cand = [
            (4 - k, "turn-goes-the-wrong-way"),
            (k + 1, "turn-counts-the-start-position"),
            (k * 2, "turn-doubles-the-count"),
        ]
        working = [f"Clockwise from {a} to {b} is {k} quarter {'turn' if k == 1 else 'turns'}.", f"That is {k}."]
    deg = 90 * k
    return text, k, deg, cand, working


def _medium(rng):
    n = rng.randint(1, 6)
    e = rng.randint(1, 3)
    dev = _dev(rng)
    text = (
        f"{_article(dev).capitalize()} {dev} makes {n} full {'turn' if n == 1 else 'turns'} and then {_PART[e]}. "
        "How many right angles has it turned through altogether?"
    )
    key = 4 * n + e
    deg = 360 * n + 90 * e
    cand = [
        (n + e, "turn-adds-the-turn-counts"),
        (4 + e, "turn-counts-the-full-turns-as-one"),
        (4 * n, "turn-drops-the-part-turn"),
        (4 * (n + 1), "turn-rounds-up-to-a-full-turn"),
        (2 * n + e, "turn-counts-a-full-turn-as-two"),
    ]
    working = [
        f"{n} full {'turn' if n == 1 else 'turns'} is {n} x 4 = {4 * n} right angles.",
        f"{_PART[e]} adds {e}: {key}.",
    ]
    return text, key, deg, cand, working


def _hard(rng):
    dev = _dev(rng)
    if rng.random() < 0.7:
        n = rng.randint(1, 5)
        e = rng.randint(1, 3)
        total = 4 * n + e
        back = rng.randint(2, total - 1)
        text = (
            f"{_article(dev).capitalize()} {dev} turns clockwise through {n} full {'turn' if n == 1 else 'turns'} and "
            f"{_PART[e]}. It then turns back anticlockwise through {back} right angles. "
            "How many right angles clockwise is it from where it started?"
        )
        key = total - back
        deg = 360 * n + 90 * e - 90 * back
        cand = [
            (total + back, "turn-adds-the-way-back"),
            (total, "turn-ignores-the-way-back"),
            (back, "turn-gives-the-way-back"),
            (n + e - back if n + e > back else None, "turn-adds-the-turn-counts"),
            (abs(4 * n - back) if 4 * n != back else None, "turn-drops-the-part-turn"),
        ]
        working = [
            f"Clockwise: {n} x 4 + {e} = {total} right angles.",
            f"Back anticlockwise {back}: {total} - {back} = {key}.",
        ]
        return text, key, deg, cand, working
    r = rng.randint(5, 23)
    while r % 4 == 0:
        r += 1
    text = (
        f"{_article(dev).capitalize()} {dev} has turned through {r} right angles. How many more right angles "
        "will make it a whole number of full turns?"
    )
    key = 4 - r % 4
    deg = 90 * key
    check((r + key) % 4 == 0, "the turn does not finish a whole turn")
    cand = [
        (r % 4, "turn-gives-the-turn-already-made"),
        (r // 4, "turn-counts-the-full-turns"),
        (4 - r // 4 if r // 4 < 4 else None, "turn-counts-the-full-turns"),
        (key + 1, "turn-counts-the-start-position"),
        (4, "turn-gives-a-whole-turn"),
    ]
    working = [
        f"{r} = 4 x {r // 4} + {r % 4}, so {r % 4} right angles are past the last whole turn.",
        f"4 - {r % 4} = {key}.",
    ]
    return text, key, deg, cand, working


@generator(TOPIC, level=3, presets={t: {"tier": t} for t in TIERS})
def w7f_turn_contexts(rng, tier="easy"):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(100, "a turn question"):
        text, key, deg, cand, working = build(rng)
        # second route: the angle in degrees as a fraction of a whole turn, then in right angles
        check(Fraction(deg, 90) == key, f"{TOPIC}: degrees give {Fraction(deg, 90)} but the key is {key}")
        picked, seen = [], {key}
        for v, tag in [*cand, *[(key + d, "turn-counts-the-start-position") for d in (1, 2, 3)]]:
            if v is None or v <= 0 or v in seen or not graded_wrong(TOPIC, key, v):
                continue
            seen.add(v)
            picked.append(Distractor(number_text(v), tag, v))
            if len(picked) == 3:
                break
        if len(picked) < 3:
            continue
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=key,
            answer_text=str(key),
            answer_latex=str(key),
            working=working,
            distractors=picked,
        )
