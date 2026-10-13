"""Shared helpers for the second-year probability and statistics generators in the `r4i_*` modules.

Every generator here works from closed forms in exact fractions or plain floats (no SymPy in a sampling loop); the
slow independent re-derivations (quadrature, simulation of the defining sum, a SymPy solve) live in the tests.

* answers: `frac_fields`, `num_fields`, `int_fields` build the answer fields of a `fraction` answer, a decimal
  `number` stated to a fixed number of places, and a whole-number `number`;
* distractors: `frac_distractors`, `num_distractors`, `int_distractors` turn `(value, misconception)` pairs into
  tagged `Distractor`s that the grader rejects by construction, with no repeats and with an `avoid` list for
  values a learner who follows through from a wrong earlier part would reach;
* chains: `chain_avoid` works out those follow-through values for a whole exam chain, one wrong part at a time and
  also carried on through the later parts, so no later distractor can equal a correct follow-through answer;
* `redraw` re-runs a generator body when a draw leaves too few distractors.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, latex_answer, round_half_up, stated_precision_string

_SQRT2 = math.sqrt(2.0)


def phi(z: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def fr(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def ftext(value: Fraction) -> str:
    """A fraction written the way a learner types it: `3`, `3/4`, `-1/2`."""
    value = fr(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def dec(value, places: int) -> str:
    """`value` rounded half up (on its exact binary value) to `places` places, shown with exactly that many."""
    exact = Fraction(value) if not isinstance(value, float) else Fraction(repr(value))
    rounded = round_half_up(exact, places)
    return str(int(rounded)) if places == 0 else stated_precision_string(rounded, places)


def clean(value: float, places: int) -> float:
    """`value` as the float a learner would type at `places` places."""
    return float(dec(value, places))


def stable(value: float, places: int, margin: float = 0.08) -> bool:
    """True when `value` is not within `margin` of a rounding boundary at `places` places."""
    scaled = value * 10.0**places
    return abs((scaled - math.floor(scaled)) - 0.5) > margin


# -- answer fields ---------------------------------------------------------------------------------------------


def frac_fields(value) -> dict:
    value = fr(value)
    text = ftext(value)
    return dict(answer_kind="fraction", answer=exact_value(value), answer_text=text, answer_latex=latex_answer(text))


def num_fields(value: float, places: int) -> dict:
    text = dec(value, places)
    return dict(answer_kind="number", answer=float(text), answer_text=text, answer_latex=text)


def int_fields(value: int) -> dict:
    text = str(int(value))
    return dict(answer_kind="number", answer=int(value), answer_text=text, answer_latex=text)


# -- distractors -----------------------------------------------------------------------------------------------


def frac_distractors(true, candidates, *, limit=3, positive=False, at_most=None, at_least=None, avoid=()):
    """Tagged exact-rational distractors; none equals `true` or anything in `avoid`."""
    true = fr(true)
    seen = {true} | {fr(v) for v in avoid}
    valid = []
    for value, tag in candidates:
        if value is None:
            continue
        value = fr(value)
        if value in seen or (positive and value <= 0):
            continue
        if (at_most is not None and value > at_most) or (at_least is not None and value < at_least):
            continue
        seen.add(value)
        valid.append((value, tag))
    picked = valid[:limit]
    # the key must not be the only option of its shape (a whole number among fractions, or the reverse)
    whole = true.denominator == 1
    if picked and not any((v.denominator == 1) == whole for v, _ in picked):
        same = [item for item in valid[limit:] if (item[0].denominator == 1) == whole]
        if same:
            picked = picked[:-1] + [same[0]]
        else:
            return picked[:0]
    return [Distractor(ftext(v), tag, exact_value(v)) for v, tag in picked]


def num_distractors(true, candidates, places, *, limit=3, avoid=(), signed=False, at_most=None):
    """Tagged decimal distractors at the answer's own number of places, each at least 2.5 units from the key, from
    `avoid` and from each other (the typed grader accepts anything within half a unit of the key)."""
    unit = 10.0**-places
    kept = [clean(true, places)] + [clean(a, places) for a in avoid]
    out = []
    for value, tag in candidates:
        if value is None or not math.isfinite(value):
            continue
        shown = clean(value, places)
        if (not signed and shown <= 0) or (at_most is not None and shown > at_most):
            continue
        if any(abs(shown - k) < 2.5 * unit for k in kept):
            continue
        kept.append(shown)
        text = dec(value, places)
        out.append(Distractor(text, tag, float(text)))
        if len(out) == limit:
            break
    return out


def int_distractors(true, candidates, *, limit=3, avoid=(), lo=1):
    """Tagged whole-number distractors."""
    kept = {int(true)} | {int(a) for a in avoid}
    out = []
    for value, tag in candidates:
        if value is None:
            continue
        value = int(value)
        if value in kept or value < lo:
            continue
        kept.add(value)
        out.append(Distractor(str(value), tag, value))
        if len(out) == limit:
            break
    return out


def need(distractors, count=2):
    """Fail the draw (so it is redrawn) when fewer than `count` distractors survived."""
    check(len(distractors) >= count, f"only {len(distractors)} distractors")
    return distractors


def stored(distractor):
    """A distractor's stored value as a Fraction (exact kinds) or a number."""
    value = distractor.value
    if isinstance(value, dict):
        return Fraction(value["p"], value["q"])
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value)
    return value


def redraw(fn):
    """Run a generator body again (same random stream) when a draw leaves too few distractors or a rounding check
    fails; the draw is cheap, padding the list with weak distractors is not."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(80):
            try:
                return fn(rng, *args, **kwargs)
            except VerificationError as error:
                if not str(error).startswith(("only ", "redraw")):
                    raise
        raise VerificationError("no draw left enough distractors")

    return wrapper


def retry(condition, what="redraw"):
    """Reject the current draw (see `redraw`) unless `condition`."""
    if not condition:
        raise VerificationError(f"redraw: {what}")


# -- exam chains -----------------------------------------------------------------------------------------------


def chain_avoid(order, fns, canon, wrong, target, rounders=None):
    """Values a learner would reach for part `target` by following through from a wrong earlier part.

    `order` is the part labels in order; `fns[label]` is a function of the dict of values (label -> value) giving
    that part's value (None for a part that does not follow through); `canon` holds the canonical values (as stated,
    i.e. already rounded); `wrong[label]` lists the wrong values of the part's distractors; `rounders[label]` states a
    computed value the way the part states it. Each wrong value is followed through the target directly (the other
    parts correct) and carried through every part between (each consistent with the one before).
    """
    rounders = rounders or {}
    ti = order.index(target)

    def step(label, values):
        fn = fns.get(label)
        if fn is None:
            return None
        try:
            raw = fn(values)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return None if raw is None else rounders.get(label, lambda v: v)(raw)

    out = []
    for j in order[:ti]:
        for w in wrong.get(j, ()):
            values = dict(canon)
            values[j] = w
            direct = step(target, values)
            if direct is not None:
                out.append(direct)
            carried = dict(canon)
            carried[j] = w
            for m in order[order.index(j) + 1 : ti + 1]:
                if fns.get(m) is not None:
                    value = step(m, carried)
                    if value is None:
                        break
                    carried[m] = value
            else:
                out.append(carried[target])
    return out
