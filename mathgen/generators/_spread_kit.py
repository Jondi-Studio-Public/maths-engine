"""Option-order helpers for the `dg_*`, `eyz_*`, `gt_*` and `sc_*` builds: wrong options picked so that the key has no
fixed place among them.

The shared pickers take the first three usable candidates of a priority list, so a generator that lists its slips
"likeliest first" gives the key the same rank on every seed (always the smallest option, always second-largest...).
`mixed` reorders a candidate list so that its first three hold a random number of values below the key and the rest
above it. The draw comes from the generator's own `rng` (see `uses_rng`), so a question is still a pure function of its seed. Nothing here registers a topic.
"""

from __future__ import annotations

import contextvars
import functools
import random
import re
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import pick_distractors as _pick_distractors
from ._ey2a_kit import tagged_numbers as _tagged_numbers
from ._pr6_chain_kit import whole_part as _whole_part
from ._r3d_kit import number_fields as _number_fields
from ._r3d_kit import number_part as _number_part
from ._sr2b_kit import pick as _pick

NEAR = "near-miss-value"  # the catalogue id of a plain near miss used to fill a side of the options


class _Required:
    def __repr__(self):
        return "<required>"


REQUIRED = _Required()  # the default of lo= and hi=: a call that leaves a bound out is a TypeError, not an open range


def check_bounds(lo, hi, unbounded):
    """Every call must say how far its options may reach. `lo=` and `hi=` are the least and greatest value an option can
    take (None for one open end); both None needs `unbounded="why"`. A named slip outside the range is dropped, not offered."""
    if lo is REQUIRED or hi is REQUIRED:
        raise TypeError("pass lo= and hi= (the feasible range of the options), or unbounded='reason' with both None")
    if lo is None and hi is None and not unbounded:
        raise TypeError("lo and hi are both None: give the feasible range, or unbounded='reason'")


_RNG: contextvars.ContextVar = contextvars.ContextVar("spread_rng", default=None)


def uses_rng(fn):
    """Decorator for a registered generator: the `rng` it was given is the one `mixed` draws its option split from
    while the generator runs, so the split varies from seed to seed like everything else the generator draws."""

    @functools.wraps(fn)
    def run(rng, *args, **kwargs):
        token = _RNG.set(rng)
        try:
            return fn(rng, *args, **kwargs)
        finally:
            _RNG.reset(token)

    return run


_SURD = re.compile(r"[0-9sqrt()*/+\-. ]{1,40}")
_PI = re.compile(r"\s*(-?)\s*(\d*)\s*(?:pi|π)\s*(?:/\s*(\d+))?\s*")
_NUMBER = re.compile(r"\s*\$?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*")


def _value(item):
    """The number an item stands for, or None when it is not a plain number."""
    if isinstance(item, tuple):
        item = item[0]
    if item is None:
        return None
    if hasattr(item, "expr") or hasattr(item, "is_number"):  # a SymPy value, or a candidate wrapping one
        try:
            expr = sympy.sympify(getattr(item, "expr", item))
            return Fraction(float(expr)).limit_denominator(10**9) if expr.is_number else None
        except (TypeError, ValueError, sympy.SympifyError):
            return None
    for attr in ("value", "text"):
        if hasattr(item, attr):
            got = getattr(item, attr)
            if isinstance(got, str):
                item = got
            elif isinstance(got, (int, float, Fraction)) and not isinstance(got, bool):
                item = got
            else:
                item = getattr(item, "text", None)
            break
    if isinstance(item, bool):
        return None
    if isinstance(item, (int, float, Fraction)):
        return Fraction(item).limit_denominator(10**6)
    if isinstance(item, str):
        if m := _PI.fullmatch(item):  # a multiple of pi: ordered by its coefficient
            coeff = Fraction(int(m[2] or 1), int(m[3] or 1))
            return -coeff if m[1] else coeff
        if "sqrt" in item and _SURD.fullmatch(item):  # a surd such as 3*sqrt(2) or -sqrt(5)
            try:
                return Fraction(float(sympy.sympify(item))).limit_denominator(10**9)
            except (sympy.SympifyError, TypeError, ValueError):
                return None
        m = _NUMBER.fullmatch(item)
        if m:
            try:
                return Fraction(m[1])
            except (ValueError, ZeroDivisionError):
                return None
    return None


def round_step(key, *, always: bool = False) -> int:
    """The step near misses of `key` should take: 10 beside a multiple of 10, 5 beside a multiple of 5, else 1.

    A lone round number among the options is a tell, so about two times in five the fillers beside a round key are
    round too. `always` is for a graph read against round gridlines, where 151 beside 150 is no reading of it. A small
    key keeps step 1: stepping by 10 from 20 leaves no room below it."""
    v = _value(key)
    if v is None or v.denominator != 1 or v == 0:
        return 1
    step = 10 if v % 10 == 0 else 5 if v % 5 == 0 else 1
    if always:  # round gridlines: tens only from 40, fives from 15, so a small key keeps room below it
        step = 10 if step == 10 and abs(v) >= 40 else 5 if step in (5, 10) and abs(v) >= 15 else 1
        return step
    if abs(v) < 2 * step:
        return 1
    rng = _RNG.get()
    return step if rng is not None and rng.random() < 0.4 else 1


def _round10(item) -> bool:
    v = _value(item)
    return v is not None and v.denominator == 1 and v != 0 and v % 10 == 0


def round_extras(key, make) -> list:
    """`make(value)` for the multiples of 10 either side of a round key (10 apart), so a round key can keep company."""
    v = _value(key)
    if v is None or v.denominator != 1 or v <= 0 or v % 10 != 0:
        return []
    return [make(int(v) + d) for d in (-10, 10) if v + d > 0]


def mixed(key, items, count: int = 3, fillers=(), *, lo=REQUIRED, hi=REQUIRED, unbounded=None, slip_range=None) -> list:
    """`items` reordered so that its first `count` hold a random number below the key and the rest above it.

    Tagged items come first on each side; `fillers` (plain near misses, tried nearest first) make up a side that has too
    few. Any entry that is not a plain number leaves the list as it came. The unchosen items follow in their original
    order, so a pick that rejects some of the chosen still finds the rest."""

    check_bounds(lo, hi, unbounded)

    def feasible(item, lo=lo, hi=hi):
        v = _value(item)
        return v is None or ((lo is None or v >= lo) and (hi is None or v <= hi))

    # `slip_range` widens the bounds for NAMED slips only, when the slip is the very mistake of leaving the range
    # (an inverse-trig angle outside its principal range); fillers always keep to [lo, hi]
    s_lo, s_hi = slip_range if slip_range is not None else (lo, hi)

    # a value outside [lo, hi] cannot be reached by the question's own structure (a fourth turning point of a parabola's
    # square, a pyramid with two vertices): offering it gives the key away by elimination
    items = [i for i in items if feasible(i, s_lo, s_hi)]
    fillers = [f for f in fillers if feasible(f)]
    key = _value(key)
    values = [_value(i) for i in items]
    if key is None:
        return items
    first = {}
    for i, v in zip(items, values):
        if v is not None:
            first.setdefault(v, i)  # a value offered twice is one option
    below = [i for v, i in first.items() if v < key]
    above = [i for v, i in first.items() if v > key]
    rng = _RNG.get() or random.Random(repr((str(key), [str(v) for v in values])))
    rng.shuffle(below)
    rng.shuffle(above)
    taken = {v for v in values if v is not None}
    distinct = {}
    for f in fillers:  # a filler offered twice is one option
        if _value(f) is not None and _value(f) not in taken:
            distinct.setdefault(_value(f), f)
    fill = sorted(distinct.values(), key=lambda f: abs(_value(f) - key))
    fill_below = [f for f in fill if _value(f) < key]
    fill_above = [f for f in fill if _value(f) > key]
    pool_below, pool_above = below + fill_below, above + fill_above
    if len(pool_below) + len(pool_above) < count:
        return items
    kmin, kmax = max(0, count - len(pool_above)), min(count, len(pool_below))
    if (
        kmin == 0 and kmax < count
    ):  # little room below the key: lean on the options that fit, so it is not always lowest
        k = rng.choices(range(kmin, kmax + 1), weights=[1 + 0.4 * j for j in range(kmax - kmin + 1)])[0]
    elif kmin > 0 and kmax == count:  # little room above: lean on the options below
        k = rng.choices(range(kmin, kmax + 1), weights=[1 + 0.4 * (kmax - kmin - j) for j in range(kmax - kmin + 1)])[0]
    else:
        k = rng.randint(kmin, kmax)
    chosen = pool_below[:k] + pool_above[: count - k]
    if _round10(key) and not any(_round10(c) for c in chosen) and rng.random() < 0.4:
        # about two times in five a round key keeps company: a round option takes the place of a plain near miss
        spare = [c for c in pool_below + pool_above if _round10(c) and not any(c is x for x in chosen)]
        if spare:
            extra = rng.choice(spare)
            same = [i for i, c in enumerate(chosen) if (_value(c) < key) == (_value(extra) < key)] or list(
                range(len(chosen))
            )
            plain = [i for i in same if any(chosen[i] is f for f in fill)] or same
            chosen[rng.choice(plain)] = extra
    rng.shuffle(chosen)
    rest = [i for i in items if not any(i is c for c in chosen)]
    spare_fill = [f for f in fill if not any(f is c for c in chosen)]  # nearest first
    return chosen + rest + spare_fill


def tagged_numbers(answer, candidates, *, low=REQUIRED, high=REQUIRED, step: int = 1, ok=None):
    check_bounds(low, high, None)
    usable, seen = [], {answer}
    for value, tag in candidates:
        if value in seen or not low <= value <= high:
            continue
        seen.add(value)
        usable.append((value, tag))
    if step == 1:
        step = round_step(answer)
    near = [
        (answer + d * step, NEAR)
        for d in (-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8)
        if low <= answer + d * step <= high and (ok is None or ok(answer + d * step))
    ]
    near += [
        (v, NEAR) for v, _ in round_extras(answer, lambda v: (v, NEAR)) if low <= v <= high and (ok is None or ok(v))
    ]
    return _tagged_numbers(answer, mixed(answer, usable, fillers=near, lo=low, hi=high), low=low, high=high)[:3]


def whole_part(prompt, value, marks, candidates, *, step: int = 1, lo=REQUIRED, hi=REQUIRED, unbounded=None, **kw):
    check_bounds(lo, hi, unbounded)
    if step == 1:
        step = round_step(value)
    near = [(value + d * step, NEAR) for d in (-4, -3, -2, -1, 1, 2, 3, 4) if value + d * step > 0]
    near += round_extras(value, lambda v: (v, NEAR))
    return _whole_part(
        prompt, value, marks, mixed(value, candidates, fillers=near, lo=lo, hi=hi, unbounded=unbounded), **kw
    )


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def number_fields(
    value, dp, wrong, *, nonneg: bool = False, near_misses: bool = True, lo=REQUIRED, hi=REQUIRED, unbounded=None, **kw
):
    """`near_misses=False` for a question whose neighbours of the key can themselves be right (a size that is 'needed',
    'large enough', or whose stated rounding they share)."""
    check_bounds(lo, hi, unbounded)
    step = Fraction(round_step(value) if dp == 0 else 1, 10**dp)
    near = [
        (Fraction(value) + d * step, NEAR)
        for d in (-3, -2, -1, 1, 2, 3)
        if near_misses and (not nonneg or Fraction(value) + d * step > 0)
    ]
    if near_misses and dp == 0:
        near += round_extras(value, lambda v: (Fraction(v), NEAR))
    return _number_fields(
        value, dp, mixed(value, wrong, fillers=near, lo=lo, hi=hi, unbounded=unbounded), nonneg=nonneg, **kw
    )


def number_part(
    prompt,
    value,
    dp,
    marks,
    working,
    wrong,
    *,
    nonneg: bool = False,
    near_misses: bool = True,
    lo=REQUIRED,
    hi=REQUIRED,
    unbounded=None,
    **kw,
):
    check_bounds(lo, hi, unbounded)
    step = Fraction(round_step(value) if dp == 0 else 1, 10**dp)
    near = [
        (Fraction(value) + d * step, NEAR)
        for d in (-3, -2, -1, 1, 2, 3)
        if near_misses and (not nonneg or Fraction(value) + d * step > 0)
    ]
    if near_misses and dp == 0:
        near += round_extras(value, lambda v: (Fraction(v), NEAR))
    return _number_part(
        prompt,
        value,
        dp,
        marks,
        working,
        mixed(value, wrong, fillers=near, lo=lo, hi=hi, unbounded=unbounded),
        nonneg=nonneg,
        **kw,
    )


def _choose(answer_text, allow_zero, ordered, count, **kw):
    """`_sr2b_kit.pick`, which never offers a plain 0, unless the question is a count that can be 0 (`allow_zero`)."""
    if not allow_zero:
        return _pick(answer_text, ordered, count, **kw)
    from ._sr2b_kit import TooFewDistractors

    out = _pick_distractors(answer_text, ordered, count)
    if len(out) < kw.get("minimum", 2):
        raise TooFewDistractors(f"too few distractors for {answer_text}")
    return out


def pick(
    answer_text,
    candidates,
    count: int = 3,
    *,
    lo=REQUIRED,
    hi=REQUIRED,
    unbounded=None,
    fillers=(),
    allow_zero=False,
    slip_range=None,
    **kw,
):
    """`_sr2b_kit.pick` over a candidate list reordered to give the key a random rank (plain near misses fill a side)."""
    check_bounds(lo, hi, unbounded)
    key = _value(answer_text)
    near = list(fillers)
    if _is_pi(answer_text, candidates):
        # a multiple of pi: its neighbours are the angles a student could reach by a slip, a sixth, quarter or third away
        from ._sr2b_kit import pi_text

        dens = tuple(dict.fromkeys((6, 4, 3, *((key.denominator,) if key.denominator > 1 else ()))))
        for den in dens:
            for k in (-2, -1, 1, 2):
                v = key + Fraction(k, den)
                near.append(Distractor(pi_text(v), NEAR, pi_text(v)))
        return _choose(
            answer_text,
            allow_zero,
            mixed(
                answer_text, candidates, count, fillers=near, lo=lo, hi=hi, unbounded=unbounded, slip_range=slip_range
            ),
            count,
            **kw,
        )
    if key is not None and "/" not in str(answer_text) and _NUMBER.fullmatch(str(answer_text)):
        places = _decimals(str(answer_text))
        step = Fraction(round_step(key) if places == 0 else 1, 10**places)
        near += [
            Distractor(_shown(key + d * step, places), NEAR, _py(key + d * step))
            for d in (-3, -2, -1, 1, 2, 3)
            if key + d * step != 0 or (lo is not None and lo <= 0)  # 0 is a possible count when the caller allows it
        ]
    if key is not None and _NUMBER.fullmatch(str(answer_text)) and _decimals(str(answer_text)) == 0:
        near += round_extras(key, lambda v: Distractor(str(v), NEAR, v))
    return _choose(
        answer_text,
        allow_zero,
        mixed(answer_text, candidates, count, fillers=near, lo=lo, hi=hi, unbounded=unbounded, slip_range=slip_range),
        count,
        **kw,
    )


def _is_pi(answer_text, candidates) -> bool:
    texts = [str(answer_text), *(getattr(c, "text", str(c)) for c in candidates)]
    return any(_PI.fullmatch(t) and ("pi" in t or "π" in t) for t in texts) and _value(answer_text) is not None


def pick_distractors(
    answer_text, candidates, n: int = 3, *, lo=REQUIRED, hi=REQUIRED, unbounded=None, near_misses: bool = False, **kw
):
    """`core.pick_distractors` over a reordered candidate list; `near_misses` adds whole-number neighbours of a whole key."""
    check_bounds(lo, hi, unbounded)
    key = _value(answer_text)
    fillers = []
    if near_misses and key is not None and key.denominator == 1:
        step = round_step(key)
        fillers = [
            Distractor(str(int(key) + d * step), NEAR, int(key) + d * step)
            for d in (-3, -2, -1, 1, 2, 3)
            if int(key) + d * step >= (0 if lo is None else lo)
        ]
        fillers += round_extras(key, lambda v: Distractor(str(v), NEAR, v))
    return _pick_distractors(
        answer_text, mixed(answer_text, candidates, n, fillers=fillers, lo=lo, hi=hi, unbounded=unbounded), n, **kw
    )


def _py(v):
    return int(v) if v.denominator == 1 else float(v)


def _shown(v: Fraction, places: int) -> str:
    return str(int(v)) if places == 0 else f"{float(v):.{places}f}"
