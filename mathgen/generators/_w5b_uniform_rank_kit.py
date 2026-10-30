"""Wrong options for the Sylow and M/M/1 + EOQ builds, placed so the key has no fixed place among them.

Slips derived from the key (`key + 1`, `key / 2`, a doubled key) crowd round it, which makes the key the most central
option and gives a learner who never reads the question an edge (`scripts/key_strategy_edges.py`). `balanced` draws
how many wrong options lie below the key from 0 to `keep` uniformly, takes named slips first on each side, and makes up
a short side with neutral fillers of the key's own shape (whole numbers beside a whole key, fractions beside a
fraction), so the key's rank is uniform and the shape of the key is shared by its neighbours.

Not a generator. The random source is the one `uses_rng` installed for the running generator.
"""

from __future__ import annotations

import random
from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import VerificationError
from ..grading import GRADERS
from ._spread_kit import _RNG


def ftext(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def accepted(kind: str, key: Fraction, reply: Fraction) -> bool:
    probe = SimpleNamespace(
        answer=exact_value(key),
        answer_text=ftext(key),
        text="",
        answer_alt=None,
        answer_form=None,
        answer_kind=kind,
    )
    return bool(GRADERS[kind](probe, ftext(reply)).correct)


def _shape(v: Fraction) -> str:
    return "int" if v.denominator == 1 else "frac"


def _filler(rng: random.Random, key: Fraction, below: bool, lo, hi, taken) -> Fraction | None:
    """A neutral value on one side of the key, of the key's shape, within [lo, hi]."""
    shape = _shape(key)
    for _ in range(60):
        if below:
            target = key * Fraction(rng.randint(15, 92), 100)
        else:
            target = key * Fraction(rng.randint(108, 260), 100)
        if shape == "int":
            v = Fraction(round(target) + rng.choice((-1, 0, 0, 1)))
        else:
            d = key.denominator * rng.choice((1, 1, 2, 3))
            v = Fraction(round(target * d), d)
            if v.denominator == 1:
                continue
        if v == key or v in taken or (below and v >= key) or (not below and v <= key):
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            continue
        return v
    return None


def balanced(key, slips, keep, *, kind="fraction", lo=None, hi=None) -> list[Distractor]:
    """`keep` wrong options for `key`: (value, misconception id) slips first, then fillers, the key's rank uniform.

    A slip equal to the key, outside [lo, hi] or accepted by the `kind` grader is dropped. A slip of the other shape (a
    fraction beside a whole key) is used only when no slip of the key's shape is left, so the key is never the only
    whole number or the only fraction."""
    rng = _RNG.get() or random.Random(repr((str(key), [str(s[0]) for s in slips if s])))
    key = Fraction(key)
    seen = {key}
    same, other = [], []
    for item in slips:
        if item is None or item[0] is None:
            continue
        v, tag = Fraction(item[0]), item[1]
        if v in seen or (lo is not None and v < lo) or (hi is not None and v > hi) or accepted(kind, key, v):
            continue
        seen.add(v)
        (same if _shape(v) == _shape(key) else other).append((v, tag))
    rng.shuffle(same)
    rng.shuffle(other)
    below_slips = [s for s in same if s[0] < key]
    above_slips = [s for s in same if s[0] > key]
    want_below = rng.randint(0, keep)
    chosen: list[tuple[Fraction, str | None]] = []
    taken = set(seen)
    for side_below, want in ((True, want_below), (False, keep - want_below)):
        pool = below_slips if side_below else above_slips
        got = pool[:want]
        chosen += got
        for _ in range(want - len(got)):
            v = _filler(rng, key, side_below, lo, hi, taken)
            if v is None:
                break
            taken.add(v)
            chosen.append((v, None))
    # a side with no room (the key near a bound) hands its places to the other side, then to the other shape
    for side_below in (True, False):
        for _ in range(keep):
            if len(chosen) >= keep:
                break
            v = _filler(rng, key, side_below, lo, hi, taken)
            if v is None:
                break
            taken.add(v)
            chosen.append((v, None))
    for s in other + same:
        if len(chosen) >= keep:
            break
        if s not in chosen:
            chosen.append(s)
    if not any(tag for _v, tag in chosen):  # at least one named slip, so a wrong reply is diagnosed
        named = [s for s in same + other if s[1]]
        if named:
            chosen[-1] = named[0]
    chosen = [c for i, c in enumerate(chosen) if c[0] not in {x[0] for x in chosen[:i]}]
    if len(chosen) < 2:
        raise VerificationError("fewer than two wrong options")
    return [Distractor(ftext(v), tag, exact_value(v)) for v, tag in chosen[:keep]]
