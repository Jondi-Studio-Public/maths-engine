"""Generates `w6b_trig_equation_interval` (easy, medium, hard) and the exam chain `exd_x_w6b_trig_equation_interval`.

``trig-equations-solutions-in-radians`` (``trigonometry``, answer ``list``): every solution of ``f(a(x - b)) = c`` for
``f`` one of sin, cos or tan, over a stated interval, in radians. The reply is a list, graded as a set of values.

* easy: ``sin x = c``, ``cos x = c`` or ``tan x = c`` for an exact c that is not 0, 1 or -1, on 0 ≤ x ≤ 2π. The solutions
  are asked as multiples of π (write 7/10 for 7π/10), which keeps the reply exact and typed.
* medium: either a multiple angle, ``f(ax) = c`` with a = 2, 3, 4 or 1/2, or a shift, ``f(x - b) = c``, on 0 ≤ x ≤ 2π,
  with exact c (zero and ±1 included).
* hard: ``f(a(x - b)) = c`` with both a stretch and a shift on 0 ≤ x ≤ 2π, -π ≤ x ≤ π or an inexact c
  (a calculator value) solved to 2 decimal places in radians, with the interval stated.
* exam (``exd_x_w6b_trig_equation_interval``): ``f(a(x - b)) = c`` for sin or cos on 0 ≤ x ≤ 2π or -π ≤ x ≤ π: the number of complete
  periods in the interval, the number of solutions (following through from the periods), the smallest solution and all
  of them.

Solutions are generated from the two solution families in exact fractions of π and then re-derived by scanning the
function numerically for sign changes over the interval, which must find the same number of roots at the same places.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import check, generator, phrase
from ._w6b_kit import dist, need, retry, want

# principal angle t0 (in units of pi) for each exact c: sin c = c, cos c = c, tan c = c
_SIN = {
    "0": (Fraction(0), 0.0),
    "1/2": (Fraction(1, 6), 0.5),
    "√2/2": (Fraction(1, 4), math.sqrt(2) / 2),
    "√3/2": (Fraction(1, 3), math.sqrt(3) / 2),
    "1": (Fraction(1, 2), 1.0),
}
_COS = {
    "1": (Fraction(0), 1.0),
    "√3/2": (Fraction(1, 6), math.sqrt(3) / 2),
    "√2/2": (Fraction(1, 4), math.sqrt(2) / 2),
    "1/2": (Fraction(1, 3), 0.5),
    "0": (Fraction(1, 2), 0.0),
}
_TAN = {
    "0": (Fraction(0), 0.0),
    "√3/3": (Fraction(1, 6), math.sqrt(3) / 3),
    "1": (Fraction(1, 4), 1.0),
    "√3": (Fraction(1, 3), math.sqrt(3)),
}
_TABLE = {"sin": _SIN, "cos": _COS, "tan": _TAN}
_FUNC = {"sin": math.sin, "cos": math.cos, "tan": math.tan}


def _exact_c(fn, key, negative):
    """(display, value in pi units of the principal angle, float c) for sin/cos/tan = ±key."""
    t0, value = _TABLE[fn][key]
    if not negative:
        return key, t0, value
    if fn == "sin":
        return "-" + key, -t0, -value
    if fn == "cos":
        return "-" + key, 1 - t0, -value
    return "-" + key, -t0, -value


def pi_text(frac: Fraction) -> str:
    """A multiple of π as text: π/6, 5π/6, π, 2π, 0."""
    frac = Fraction(frac)
    if frac == 0:
        return "0"
    sign = "-" if frac < 0 else ""
    frac = abs(frac)
    head = "" if frac.numerator == 1 else str(frac.numerator)
    return f"{sign}{head}π" if frac.denominator == 1 else f"{sign}{head}π/{frac.denominator}"


def k_text(frac: Fraction) -> str:
    frac = Fraction(frac)
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def _solutions(fn, t0, a, b, lo, hi, *, families="all", periods="all"):
    """Solutions in units of pi of f(a(x - b)) = c with principal angle t0, inside [lo, hi] (all in pi units)."""
    if fn == "sin":
        bases = [(t0, 2), (1 - t0, 2)]
    elif fn == "cos":
        bases = [(t0, 2), (-t0, 2)]
    else:
        bases = [(t0, 1)]
    if families == "first":
        bases = bases[:1]
    found = set()
    span = range(-60, 61) if periods == "all" else range(0, 1)
    for base, step in bases:
        for n in span:
            u = base + step * n
            x = u / a + b
            if lo <= x <= hi:
                found.add(x)
    return sorted(found)


def _scan_roots(fn, c_value, a, b, lo, hi, points=6000):
    """Second route: the roots of f(a(x - b)) - c by sign changes over [lo*pi, hi*pi] (poles of tan skipped)."""
    f = _FUNC[fn]
    a_f, b_f, lo_f, hi_f = float(a), float(b) * math.pi, float(lo) * math.pi, float(hi) * math.pi
    xs = [lo_f + (hi_f - lo_f) * i / points for i in range(points + 1)]
    values = [f(a_f * (x - b_f)) - c_value for x in xs]
    roots = []
    for i in range(points + 1):
        if abs(values[i]) < 1e-9 and not (fn == "tan" and abs(values[i]) > 50):
            roots.append(xs[i] / math.pi)
    for i in range(points):
        v0, v1 = values[i], values[i + 1]
        if fn == "tan" and (abs(v0) > 50 or abs(v1) > 50) and v0 * v1 < 0:
            continue  # a pole, not a root
        if v0 * v1 < 0:
            roots.append((xs[i] + (xs[i + 1] - xs[i]) * v0 / (v0 - v1)) / math.pi)
    return roots


def _verify(fn, c_value, a, b, lo, hi, sols):
    roots = _scan_roots(fn, c_value, a, b, lo, hi)
    # a root touching zero without a sign change (sin x = 1) is found by value
    exact = [float(s) for s in sols]
    for s in sols:
        x = float(s) * math.pi
        check(abs(_FUNC[fn](float(a) * (x - float(b) * math.pi)) - c_value) < 1e-9, f"x = {s}π does not solve")
    if abs(c_value) < 1 - 1e-12 or fn == "tan":
        merged = []
        for r in sorted(roots):
            if not merged or abs(r - merged[-1]) > 1e-3:
                merged.append(r)
        check(len(merged) == len(exact), f"scan found {len(merged)} roots, the list has {len(exact)}")
        for r, e in zip(merged, exact):
            check(abs(r - e) < 1e-3, "scan and list disagree")


def _equation_text(fn, a, b):
    """f(x), f(ax), f(x - b), f(a(x - b)) as text."""
    a = Fraction(a)
    shift = "" if b == 0 else f" - {pi_text(b)}" if b > 0 else f" + {pi_text(-b)}"
    if a == 1:
        inner = "x" + shift
    elif b == 0:
        inner = f"{a.numerator}x" if a.denominator == 1 else "x/2"
    elif a.denominator == 1:
        inner = f"{a.numerator}(x{shift})"
    else:
        inner = f"(x{shift})/2"
    return f"{fn}({inner})"


def _interval_text(lo, hi):
    return f"{pi_text(lo) if lo != 0 else '0'} ≤ x ≤ {pi_text(hi)}"


_INSTRUCTION = " Give each solution as a multiple of π (for example, enter 7/10 for 7π/10), separated by commas."
_DECIMAL_INSTRUCTION = " Give each solution in radians, correct to 2 decimal places, separated by commas."

_STEMS = (
    "Solve {eq} = {c} for {interval}.{how}",
    "Find all values of x with {interval} that satisfy {eq} = {c}.{how}",
    "List every solution of the equation {eq} = {c} in the interval {interval}.{how}",
    "The equation {eq} = {c} is to be solved over {interval}. State all of its solutions.{how}",
)


def _list_answer(sols):
    texts = [k_text(s) for s in sols]
    return texts, ", ".join(texts)


def _dist_exact(sols, tag):
    texts, joined = _list_answer(sols)
    return dist(joined, tag, texts)


def _from_bases(bases, a, b, lo, hi):
    """Sorted x = (base + step n)/a + b inside [lo, hi], for (base, step) in units of pi."""
    return sorted({x for n in range(-30, 31) for base, step in bases if lo <= (x := (base + step * n) / a + b) <= hi})


def _looks(text):
    """What a learner can see of one option, as {feature: value}: the shape of its numbers, not their meaning."""
    items = [Fraction(t.strip()) if "/" in t or t.strip().lstrip("-").isdigit() else float(t) for t in text.split(",")]
    feats = {
        "n_items": len(items),
        "digits": min(sum(ch.isdigit() for ch in text), 8),
        "signs": tuple((v > 0) - (v < 0) for v in items),
        "whole": tuple(v == int(v) for v in items),
    }
    if all(v == int(v) for v in items):
        feats["parity"] = tuple(int(v) % 2 for v in items)
        feats["pow2"] = tuple(v > 0 and int(v) & (int(v) - 1) == 0 for v in items)
    return feats, items


def _key_stands_apart(key, wrong, anyone=False):
    """True when the key alone holds some visible feature value (length, digits, signs, wholeness, parity, nearness).

    The pools of wrong options come from the key by formulas, so the key is easily the only one of its sort; a redraw
    keeps it from being the odd one out, which a learner could pick without doing any maths. With ``anyone`` a wrong
    option that stands apart counts too, since a learner could cross it off.
    """
    texts = [key, *wrong]
    looks = [_looks(t) for t in texts]
    feats = [dict(f) for f, _ in looks]
    if all(len(items) == 1 for _, items in looks):
        values = [float(items[0]) for _, items in looks]
        spread = [round(sum(abs(v - w) for w in values), 9) for v in values]
        for i, f in enumerate(feats):
            f["near1"] = min(sum(abs(v - values[i]) <= 1 for j, v in enumerate(values) if j != i), 3)
            f["near2"] = min(sum(abs(v - values[i]) <= 2 for j, v in enumerate(values) if j != i), 3)
            low, high = min(spread), max(spread)
            f["centrality"] = (
                "central"
                if spread[i] == low and spread.count(low) == 1
                else "outlier"
                if spread[i] == high and spread.count(high) == 1
                else "mid"
            )
    common = set.intersection(*(set(f) for f in feats))
    who = range(len(feats)) if anyone else (0,)
    return any(sum(f[name] == feats[i][name] for f in feats) == 1 for name in common for i in who)


def _list_options(rng, joined, cand, n):
    """``n`` wrong lists in a random order, those as long as the answer first: the key is never the only list of its length."""
    size = joined.count(",")
    pool = list(cand)
    rng.shuffle(pool)
    pool.sort(key=lambda d: d.text.count(",") != size)
    out = need(joined, pool, n, unordered=True)
    want(sum(d.text.count(",") == size for d in out) >= min(2, n - 1), "too few wrong lists as long as the answer")
    want(not _key_stands_apart(joined, [d.text for d in out]), "the key is the odd one out")
    return out


def _digits(value):
    return sum(ch.isdigit() for ch in k_text(value))


def _alike(value, key):
    """Same number of digits and, for whole numbers, the same parity: nothing but the value tells the key apart."""
    value, key = Fraction(value), Fraction(key)
    if _digits(value) != _digits(key):
        return False
    return (value - key) % 2 == 0 if value.denominator == key.denominator == 1 else True


def _offset_options(rng, key, tag, n=3):
    """``n`` whole-number wrong options a few either side of the key, so no option is set apart by its size."""
    steps = [d for d in (-4, -3, -2, -1, 1, 2, 3, 4) if key + d >= 1]
    draws = [[str(key + d) for d in rng.sample(steps, n)] for _ in range(60)]
    # no option stands apart if possible, else at least not the key
    texts = next((t for t in draws if not _key_stands_apart(str(key), t, anyone=True)), None)
    texts = texts or next((t for t in draws if not _key_stands_apart(str(key), t)), draws[-1])
    return [dist(t, tag, int(t)) for t in texts]


def _ranked_options(rng, key, pool, n):
    """``n`` wrong options from ``pool`` with the key's rank among all the options drawn uniformly (0 is the smallest).

    ``pool`` holds `Distractor`s whose text is a number; each wrong value is used once, whichever tag reaches it first.
    """
    seen = {Fraction(key)}
    uniq = []
    for d in pool:
        v = Fraction(d.text)
        if v not in seen:
            seen.add(v)
            uniq.append((v, d))
    want(len(uniq) >= n, "too few distinct wrong values")
    rank = rng.randrange(n + 1)
    fits = [c for c in itertools.combinations(uniq, n) if sum(v < Fraction(key) for v, _ in c) == rank]
    want(bool(fits), "no choice of wrong values puts the key at this rank")

    # at least two wrong options are as long as the key (digits and parity too), so the key is not the one that stands apart
    def like(c):
        return sum(_alike(v, key) for v, _ in c)

    fits = [c for c in fits if like(c) >= min(2, n - 1)] or fits

    def apart(c, anyone):
        return _key_stands_apart(k_text(key), [d.text for _, d in c], anyone=anyone)

    # keep the key from being the odd one out; a wrong option that stands apart is crossed off by a learner, so avoid
    # those most of the time too (a pool too small to manage it gives way, in that order)
    fits = [c for c in fits if not apart(c, False)] or fits
    fits = [c for c in fits if not apart(c, True)] or fits
    chosen = rng.choice(fits)
    return [d for _, d in chosen]


def _candidates_exact(fn, t0, a, b, lo, hi, answer, positive_t0=None):
    """Tagged wrong solution lists; `positive_t0` is the angle when the sign of a negative c is ignored."""
    cand = []

    def add(sols, tag):
        if sols and sols != answer:
            cand.append(_dist_exact(sols, tag))

    add(_solutions(fn, t0, a, b, lo, hi, periods="first"), "trig-eq-solutions-stop-at-first-period")
    add(_solutions(fn, t0, a, b, lo, hi, families="first"), "trig-eq-solutions-miss-second-family")
    if fn == "sin":
        add(_from_bases([(t0, 2), (1 + t0, 2)], a, b, lo, hi), "trig-eq-sine-second-solution-uses-pi-plus")
        add(_from_bases([(t0, 2), (-t0, 2)], a, b, lo, hi), "trig-eq-sine-second-solution-uses-two-pi-minus")
    elif fn == "cos":
        add(_from_bases([(t0, 2), (1 - t0, 2)], a, b, lo, hi), "trig-eq-cosine-second-solution-uses-pi-minus")
        add(_from_bases([(t0, 2), (1 + t0, 2)], a, b, lo, hi), "trig-eq-cosine-second-solution-uses-pi-plus")
    else:
        add(_from_bases([(t0, 2)], a, b, lo, hi), "trig-eq-tangent-period-is-two-pi")
        add(_from_bases([(t0, 2), (1 - t0, 2)], a, b, lo, hi), "trig-eq-tangent-solved-like-sine")
        add(_from_bases([(-t0, 1)], a, b, lo, hi), "trig-eq-tangent-sign-of-angle-lost")
    swapped = Fraction(1, 2) - t0 if t0 != 0 else Fraction(1, 2)
    if swapped != t0:
        add(_solutions(fn, swapped, a, b, lo, hi), "trig-eq-special-angle-mixed-up")
    if positive_t0 is not None and positive_t0 != t0:
        add(_solutions(fn, positive_t0, a, b, lo, hi), "trig-eq-negative-value-treated-as-positive")
    if a != 1:
        add(_from_bases(_bases(fn, t0), 1, b, lo, hi), "trig-eq-multiple-angle-not-divided")
    if b != 0:
        add(_solutions(fn, t0, a, 0, lo, hi), "trig-eq-shift-ignored")
        add(_solutions(fn, t0, a, -b, lo, hi), "trig-eq-shift-wrong-way")
        if a != 1:
            add(_solutions(fn, t0, a, b / a, lo, hi), "trig-eq-shift-scaled-by-the-multiple")
    return cand


def _bases(fn, t0):
    if fn == "sin":
        return [(t0, 2), (1 - t0, 2)]
    if fn == "cos":
        return [(t0, 2), (-t0, 2)]
    return [(t0, 1)]


def _build_exact(rng, fn, a, b, lo, hi, *, allow_special, tier_name):
    special = ("0",) if fn == "tan" else ("0", "1")
    key = rng.choice([k for k in _TABLE[fn] if allow_special or k not in special])
    negative = key != "0" and rng.random() < 0.5
    disp, t0, c_value = _exact_c(fn, key, negative)
    sols = _solutions(fn, t0, a, b, lo, hi)
    want(1 <= len(sols) <= 8, "no solutions, or too many to type")
    _verify(fn, c_value, a, b, lo, hi, sols)
    # a clean display of c in the stem
    cand = _candidates_exact(fn, t0, a, b, lo, hi, sols, _TABLE[fn][key][0] if negative else None)
    texts, joined = _list_answer(sols)
    want(not any(len(t.split("/")[-1]) > 2 for t in texts if "/" in t), "an awkward denominator")
    working = _working_exact(fn, disp, t0, a, b, sols)
    return dict(
        text=phrase(
            rng,
            _STEMS,
            eq=_equation_text(fn, a, b),
            c=disp,
            interval=_interval_text(lo, hi),
            how=_INSTRUCTION,
        ),
        answer_kind="list",
        answer=[Fraction(t) for t in texts],
        answer_text=joined,
        working=working,
        distractors=_list_options(rng, joined, cand, 4),
    )


def _x_of_u(a):
    """x in terms of u when u = a(x - b): u/a, which is 2u for a = 1/2."""
    a = Fraction(a)
    if a == 1:
        return "u"
    return f"u/{a.numerator}" if a.denominator == 1 else f"{a.denominator}u"


def _working_exact(fn, disp, t0, a, b, sols):
    first = {
        "sin": "sin u = c gives u = α and u = π - α",
        "cos": "cos u = c gives u = α and u = -α",
        "tan": "tan u = c gives u = α",
    }[fn]
    line = f"Put u = {_unit_text(a, b)}. Then {first}, where α = {pi_text(t0)} is the angle in the first period, each plus whole periods."
    return [
        line,
        f"Solve x = {_x_of_u(a)}{' + ' + pi_text(b) if b else ''} and keep the values inside the interval.",
        "The solutions are x = " + ", ".join(pi_text(s) for s in sols) + ".",
    ]


def _unit_text(a, b):
    a = Fraction(a)
    if a == 1 and b == 0:
        return "x"
    if b == 0:
        return f"{a.numerator}x" if a.denominator == 1 else "x/2"
    if a == 1:
        return f"x - {pi_text(b)}"
    return f"{a.numerator}(x - {pi_text(b)})" if a.denominator == 1 else f"(x - {pi_text(b)})/2"


# -- easy -----------------------------------------------------------------------------------------------------


def _easy(rng):
    fn = rng.choice(["sin", "cos", "tan", "sin", "cos"])
    return _build_exact(
        rng, fn, Fraction(1), Fraction(0), Fraction(0), Fraction(2), allow_special=False, tier_name="easy"
    )


# -- medium ---------------------------------------------------------------------------------------------------


def _medium(rng):
    fn = rng.choice(["sin", "cos", "tan", "sin", "cos"])
    if rng.random() < 0.55:
        a, b = Fraction(rng.choice([2, 2, 3, 4])) if fn != "tan" else Fraction(rng.choice([2, 3])), Fraction(0)
        if rng.random() < 0.12 and fn != "tan":
            a = Fraction(1, 2)
    else:
        a, b = Fraction(1), Fraction(rng.choice([1, 1, 2, 3, 4, 6]), rng.choice([6, 4, 3, 2]))
        if b > 1:
            b = Fraction(1, rng.choice([2, 3, 4, 6]))
    return _build_exact(rng, fn, a, b, Fraction(0), Fraction(2), allow_special=True, tier_name="medium")


# -- hard -----------------------------------------------------------------------------------------------------


def _decimal_solutions(fn, c, a, b, lo, hi):
    """Solutions in radians (floats) of f(a(x - b)) = c on [lo pi, hi pi], c inexact; b in pi units."""
    t0 = {"sin": math.asin, "cos": math.acos, "tan": math.atan}[fn](c)
    if fn == "sin":
        bases = [(t0, 2 * math.pi), (math.pi - t0, 2 * math.pi)]
    elif fn == "cos":
        bases = [(t0, 2 * math.pi), (-t0, 2 * math.pi)]
    else:
        bases = [(t0, math.pi)]
    lo_f, hi_f = float(lo) * math.pi, float(hi) * math.pi
    found = []
    for base, step in bases:
        for n in range(-40, 41):
            x = (base + step * n) / float(a) + float(b) * math.pi
            if lo_f - 1e-12 <= x <= hi_f + 1e-12:
                found.append(x)
    return sorted(found)


def _decimal_candidates(fn, c, a, b, lo, hi, sols):
    out = []

    def add(values, tag):
        texts = [f"{v:.2f}" for v in values]
        if values and texts != [f"{v:.2f}" for v in sols] and len(set(texts)) == len(texts):
            out.append(dist(", ".join(texts), tag, [round(v + 1e-12, 2) for v in values]))

    t0 = {"sin": math.asin, "cos": math.acos, "tan": math.atan}[fn](c)
    add(_decimal_wrong_second(fn, t0, a, b, lo, hi), "trig-eq-second-solution-uses-the-wrong-quadrant")
    first_only = _decimal_first_family(fn, t0, a, b, lo, hi)
    add(first_only, "trig-eq-solutions-miss-second-family")
    if b != 0:
        add(_decimal_solutions(fn, c, a, 0, lo, hi), "trig-eq-shift-ignored")
    if a != 1:
        lo_f, hi_f = float(lo) * math.pi, float(hi) * math.pi
        undivided = sorted(
            {
                base + step * n + float(b) * math.pi
                for base, step in _float_bases(fn, t0)
                for n in range(-40, 41)
                if lo_f <= base + step * n + float(b) * math.pi <= hi_f
            }
        )
        add(undivided, "trig-eq-multiple-angle-not-divided")
    if c < 0:
        add(_decimal_solutions(fn, -c, a, b, lo, hi), "trig-eq-negative-value-treated-as-positive")
    if fn == "tan":
        add(_decimal_from(t0, math.pi - t0, 2 * math.pi, a, b, lo, hi), "trig-eq-tangent-solved-like-sine")
        add(_decimal_from(-t0, None, math.pi, a, b, lo, hi), "trig-eq-tangent-sign-of-angle-lost")
    if lo == 0 and a == 1 and b == 0:
        degrees = math.degrees(t0)
        pair = {"sin": [degrees, 180 - degrees], "cos": [degrees, 360 - degrees], "tan": [degrees, degrees + 180]}[fn]
        add(sorted(v % 360 for v in pair), "trig-eq-calculator-in-degree-mode")
    return out


def _decimal_from(first, second, step, a, b, lo, hi):
    """x = (base + step n)/a + b in [lo pi, hi pi] for the given bases (a second of None is left out)."""
    lo_f, hi_f = float(lo) * math.pi, float(hi) * math.pi
    bases = [first] if second is None else [first, second]
    return sorted(
        {
            x
            for base in bases
            for n in range(-40, 41)
            if lo_f - 1e-12 <= (x := (base + step * n) / float(a) + float(b) * math.pi) <= hi_f + 1e-12
        }
    )


def _float_bases(fn, t0):
    if fn == "sin":
        return [(t0, 2 * math.pi), (math.pi - t0, 2 * math.pi)]
    if fn == "cos":
        return [(t0, 2 * math.pi), (-t0, 2 * math.pi)]
    return [(t0, math.pi)]


def _decimal_wrong_second(fn, t0, a, b, lo, hi):
    lo_f, hi_f = float(lo) * math.pi, float(hi) * math.pi
    if fn == "sin":
        bases = [(t0, 2 * math.pi), (math.pi + t0, 2 * math.pi)]
    elif fn == "cos":
        bases = [(t0, 2 * math.pi), (math.pi - t0, 2 * math.pi)]
    else:
        bases = [(t0, 2 * math.pi)]
    return sorted(
        {
            x
            for base, step in bases
            for n in range(-40, 41)
            if lo_f <= (x := (base + step * n) / float(a) + float(b) * math.pi) <= hi_f
        }
    )


def _decimal_first_family(fn, t0, a, b, lo, hi):
    lo_f, hi_f = float(lo) * math.pi, float(hi) * math.pi
    base, step = _float_bases(fn, t0)[0]
    return sorted(
        {x for n in range(-40, 41) if lo_f <= (x := (base + step * n) / float(a) + float(b) * math.pi) <= hi_f}
    )


def _hard_decimal(rng):
    fn = rng.choice(["sin", "cos", "tan"])
    if fn == "tan":
        c = rng.choice([0.4, 0.6, 0.8, 1.5, 2, 2.5, 3, 4]) * rng.choice([-1, 1])
    else:
        c = rng.choice([0.15, 0.2, 0.3, 0.35, 0.4, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]) * rng.choice([-1, 1])
    a = Fraction(rng.choice([1, 1, 2, 3]))
    b = rng.choice([Fraction(0), Fraction(0), Fraction(1, 6), Fraction(1, 4), Fraction(1, 3)])
    lo, hi = rng.choice([(Fraction(0), Fraction(2)), (Fraction(-1), Fraction(1)), (Fraction(0), Fraction(2))])
    sols = _decimal_solutions(fn, c, a, b, lo, hi)
    want(1 <= len(sols) <= 8, "an awkward number of solutions")
    for s in sols:
        want(abs((s * 100) % 1 - 0.5) > 0.12, "a solution sits on a rounding boundary")
    for s in sols:
        check(abs(_FUNC[fn](float(a) * (s - float(b) * math.pi)) - c) < 1e-9, "a solution does not solve the equation")
    # second route: sign-change scan over the interval
    roots = _scan_roots(fn, c, a, b, lo, hi)
    merged = []
    for r in sorted(roots):
        if not merged or abs(r - merged[-1]) > 1e-3:
            merged.append(r)
    check(
        len(merged) == len(sols) and all(abs(m * math.pi - s) < 1e-3 for m, s in zip(merged, sols)),
        "the scan disagrees",
    )
    texts = [f"{s:.2f}" for s in sols]
    want(len(set(texts)) == len(texts), "two solutions round to the same text")
    joined = ", ".join(texts)
    cand = _decimal_candidates(fn, c, a, b, lo, hi, sols)
    c_text = f"{c:g}"
    return dict(
        text=phrase(
            rng,
            _STEMS,
            eq=_equation_text(fn, a, b),
            c=c_text,
            interval=_interval_text(lo, hi),
            how=_DECIMAL_INSTRUCTION,
        ),
        answer_kind="list",
        answer=[round(s + 1e-12, 2) for s in sols],
        answer_text=joined,
        working=[
            f"Let u = {_unit_text(a, b)}. A calculator in radian mode gives the first angle {math.asin(c) if fn == 'sin' else math.acos(c) if fn == 'cos' else math.atan(c):.4f}.",
            "Use the second solution family, add whole periods, and solve for x.",
            f"Keep the values in the interval: x = {joined}.",
        ],
        distractors=_list_options(rng, joined, cand, 4),
    )


def _hard_exact(rng):
    fn = rng.choice(["sin", "cos", "tan"])
    a = Fraction(rng.choice([2, 3, 4])) if fn != "tan" else Fraction(rng.choice([2, 3]))
    if rng.random() < 0.15 and fn != "tan":
        a = Fraction(1, 2)
    b = Fraction(rng.choice([1, 1, 2, 3, 5]), rng.choice([6, 4, 3, 2, 12]))
    b = b if b < 1 else Fraction(1, 3)
    lo, hi = rng.choice([(Fraction(-1), Fraction(1)), (Fraction(0), Fraction(2)), (Fraction(-1), Fraction(1))])
    if a >= 3 and (hi - lo) > 2:
        hi = lo + 2
    return _build_exact(rng, fn, a, b, lo, hi, allow_special=True, tier_name="hard")


def _hard(rng):
    return rng.choice([_hard_exact, _hard_decimal, _hard_exact, _hard_decimal])(rng)


@generator(
    "w6b_trig_equation_interval",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6b_trig_equation_interval(rng, tier="easy"):
    """Trigonometric equations solved over an interval, answered as a list; see the module docstring."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return retry(rng, _easy)
    if tier == "medium":
        return retry(rng, _medium)
    return retry(rng, _hard)


# -- exam -----------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "Consider the equation {eq} = {c} for {interval}.",
    "The function f is defined by f(x) = {eq} on {interval}, and f(x) = {c} is to be solved.",
    "A trigonometric equation, {eq} = {c}, is set on the interval {interval}.",
)


def _exam(rng):
    fn = rng.choice(["sin", "cos"])
    a = Fraction(rng.choice([2, 3, 4]))
    b = Fraction(rng.randrange(1, 12), 12) if rng.random() < 0.85 else Fraction(0)
    key = rng.choice([k for k in _TABLE[fn] if k not in ("0", "1")])
    negative = rng.random() < 0.5
    disp, t0, c_value = _exact_c(fn, key, negative)
    lo, hi = rng.choice([(Fraction(0), Fraction(2)), (Fraction(-1), Fraction(1))])
    sols = _solutions(fn, t0, a, b, lo, hi)
    want(len(sols) == 2 * int(a), "a solution sits on the end of the interval")
    _verify(fn, c_value, a, b, lo, hi, sols)
    cand = _candidates_exact(fn, t0, a, b, lo, hi, sols, _TABLE[fn][key][0] if negative else None)
    n_a = int(a)

    # the smallest solution that is not negative: a negative key or option would stand out on an interval below zero
    def first_nonneg(values):
        return next((v for v in values if v >= 0), None)

    non_neg = first_nonneg(sols)
    want(non_neg is not None and non_neg > 0, "no positive solution to ask for")
    first_of = {}
    for d in reversed(cand):
        v = first_nonneg([Fraction(x) for x in d.text.split(",")])
        if v is not None and d.misconception != "trig-eq-solutions-miss-second-family":
            first_of[d.misconception] = v
    smallest_cands = [_kfdist(v, tag) for tag, v in first_of.items()]
    smallest_cands.append(_kfdist(sols[sols.index(non_neg) + 1], "trig-eq-smallest-solution-is-the-second"))
    smallest_cands.append(_kfdist(t0, "trig-eq-smallest-solution-is-the-principal-angle"))
    smallest_cands.append(_kfdist(t0 * a, "trig-eq-multiple-angle-not-divided"))
    smallest_cands = [
        d for d in smallest_cands if 0 < Fraction(d.text) <= hi
    ]  # a value outside the interval is a giveaway
    count = len(sols)
    texts, joined = _list_answer(sols)
    smallest = non_neg
    eq = _equation_text(fn, a, b)
    return dict(
        text=phrase(rng, _EXAM_STEMS, eq=eq, c=disp, interval=_interval_text(lo, hi)),
        parts=[
            dict(
                prompt=(
                    f"How many complete periods of the graph of y = {_equation_text(fn, a, b)} fit in the interval "
                    f"{_interval_text(lo, hi)}?"
                ),
                answer_kind="number",
                answer=n_a,
                answer_text=str(n_a),
                marks=1,
                sensible_range=(0, None),
                working=[f"The period is 2π/{n_a}, and 2π ÷ (2π/{n_a}) = {n_a}, so {n_a} periods fit."],
                distractors=_offset_options(rng, n_a, "trig-eq-cycles-off-by-one"),
            ),
            dict(
                prompt="Hence state how many solutions the equation has in this interval.",
                answer_kind="number",
                answer=count,
                answer_text=str(count),
                marks=1,
                follow_through="2*part_a",
                sensible_range=(0, None),
                working=[f"Each of the {n_a} periods holds two solutions: {count}."],
                distractors=_offset_options(rng, count, "trig-eq-count-off-by-one"),
            ),
            dict(
                prompt=f"Find the smallest {'' if lo == 0 else 'non-negative '}solution, as a multiple of π.",
                answer_kind="fraction",
                answer=exact_value(smallest),
                answer_text=k_text(smallest),
                marks=2,
                sensible_range=(0, None) if smallest > 0 else None,
                working=["Solve for the angle inside the brackets, divide by the multiple and add the shift back."],
                distractors=_ranked_options(rng, smallest, smallest_cands, 3),
            ),
            dict(
                prompt="Find all the solutions. " + _INSTRUCTION.strip(),
                answer_kind="list",
                answer=[Fraction(t) for t in texts],
                answer_text=joined,
                marks=3,
                working=_working_exact(fn, disp, t0, a, b, sols),
                distractors=_list_options(rng, joined, cand, 3),
            ),
        ],
    )


def _kfdist(frac, tag):
    frac = Fraction(frac)
    return Distractor(k_text(frac), tag, exact_value(frac))


def _intdist(value, tag):
    return dist(str(value), tag, value)


@generator("exd_x_w6b_trig_equation_interval", level=16, presets={"exam": dict(tier="exam")})
def exd_x_w6b_trig_equation_interval(rng, tier="exam"):
    """Exam chain: number of periods, number of solutions (following through), smallest solution and every solution."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _exam, 300)
