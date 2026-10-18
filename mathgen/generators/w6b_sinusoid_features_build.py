"""Generates `w6b_sinusoid_features` (easy, medium, hard) and the exam chain `exd_x_w6b_sinusoid_features`.

``sinusoid-features-from-equation`` (``trigonometry``, answer ``expression``): reading the amplitude, period,
horizontal translation, vertical translation, extreme values and first maximum from the equation of a sine, cosine or
tangent function, with exact answers (multiples of π where needed).

* easy: one feature of ``y = a sin(b(x - c)) + d`` or the cosine form, with a > 0 and the bracket already factored: the
  amplitude, period, horizontal translation (signed: a bracket (x + c) is a shift to the left), vertical translation or the
  maximum or minimum value.
* medium: the bracket is expanded, ``y = a sin(bx + k) + d``, and a may be negative: the horizontal translation is -k/b,
  the amplitude is |a| and the extreme values depend on the sign of a.
* hard: the smallest positive x at which the maximum or minimum occurs, the y-intercept, or the smallest positive
  vertical asymptote of ``y = a tan(b(x - c)) + d``.
* exam (``exd_x_w6b_sinusoid_features``): the amplitude, period and horizontal translation of
  ``y = a sin(bx + k) + d`` and the maximum value (following through from the amplitude).

Every answer is checked a second way: the function is evaluated numerically, its extremes found by a fine scan, and its
period and shift confirmed by comparing values one period (or one shift) apart.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import check, generator, phrase
from ._w6b_kit import need, retry, want

# -- text helpers -----------------------------------------------------------------------------------------------


def pi_text(frac) -> str:
    frac = Fraction(frac)
    if frac == 0:
        return "0"
    sign = "-" if frac < 0 else ""
    frac = abs(frac)
    head = "" if frac.numerator == 1 else str(frac.numerator)
    return f"{sign}{head}π" if frac.denominator == 1 else f"{sign}{head}π/{frac.denominator}"


def pi_sympy(frac) -> str:
    frac = Fraction(frac)
    if frac == 0:
        return "0"
    return f"({frac.numerator})*pi/{frac.denominator}" if frac.denominator != 1 else f"({frac.numerator})*pi"


def num_text(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _coef(a: int) -> str:
    return "" if a == 1 else "-" if a == -1 else str(a)


def _b_text(b: Fraction) -> str:
    return str(b.numerator) if b.denominator == 1 else f"{b.numerator}/{b.denominator}"


def _signed_pi(c: Fraction) -> str:
    return f" - {pi_text(c)}" if c > 0 else f" + {pi_text(-c)}"


def equation_text(fn, a, b, c, d, *, expanded=False) -> str:
    """``y = a fn(b(x - c)) + d`` (or ``fn(bx + k)`` when expanded) as plain text."""
    b = Fraction(b)
    c = Fraction(c)
    if expanded:
        k = -b * c
        inner = f"{_b_text(b) if b != 1 else ''}x" + (_signed_pi(-k) if k != 0 else "")
        inner = inner.replace(" - -", " - ")
    elif c == 0:
        inner = f"{_b_text(b) if b != 1 else ''}x"
    elif b == 1:
        inner = f"x{_signed_pi(c)}"
    elif b.denominator == 1:
        inner = f"{b.numerator}(x{_signed_pi(c)})"
    else:
        inner = (
            f"{b.numerator}(x{_signed_pi(c)})/{b.denominator}"
            if b.numerator != 1
            else f"(x{_signed_pi(c)})/{b.denominator}"
        )
    tail = "" if d == 0 else f" + {d}" if d > 0 else f" - {-d}"
    return f"y = {_coef(a)}{fn}({inner}){tail}"


def _expr(frac, tag, *, pi=True):
    """A tagged option: text, and the SymPy string the grader compares against."""
    text = pi_text(frac) if pi else num_text(frac)
    return Distractor(text, tag, pi_sympy(frac) if pi else str(Fraction(frac)))


def _value_fields(frac, *, pi):
    frac = Fraction(frac)
    if pi:
        return dict(answer=pi_sympy(frac), answer_text=pi_text(frac))
    return dict(answer=str(frac), answer_text=num_text(frac))


# -- numeric second route ---------------------------------------------------------------------------------------


def _f(fn, a, b, c, d, x):
    inner = float(b) * (x - float(c) * math.pi)
    return float(a) * {"sin": math.sin, "cos": math.cos, "tan": math.tan}[fn](inner) + float(d)


def _scan_extremes(fn, a, b, c, d):
    period = 2 * math.pi / float(b)
    xs = [i * period / 4000 * 2 for i in range(4001)]
    ys = [_f(fn, a, b, c, d, x) for x in xs]
    return max(ys), min(ys)


def _verify_periodic(fn, a, b, c, d, period_frac, shift_frac):
    """Period: the function repeats after 2π/b; shift: the key point of y = a fn(bx) + d moves by c."""
    period = float(period_frac) * math.pi
    for x in (0.1, 0.73, 1.9, 2.8):
        check(abs(_f(fn, a, b, c, d, x) - _f(fn, a, b, c, d, x + period)) < 1e-9, "not periodic with that period")
        check(abs(_f(fn, a, b, c, d, x + float(shift_frac) * math.pi) - _f(fn, a, b, 0, d, x)) < 1e-9, "shift is wrong")


# -- features ---------------------------------------------------------------------------------------------------


def _first_extreme(fn, a, b, c, kind):
    """Smallest x > 0, in units of π, where y = a fn(b(x - c)) + d is maximal (kind='max') or minimal ('min')."""
    # inner angle t = b(x - c), in units of pi, where fn a*fn(t) is maximal
    want_high = (kind == "max") == (a > 0)
    if fn == "sin":
        t0 = Fraction(1, 2) if want_high else Fraction(3, 2)
    else:
        t0 = Fraction(0) if want_high else Fraction(1)
    period_t = Fraction(2)
    best = None
    for n in range(-30, 31):
        x = (t0 + period_t * n) / b + c
        if x > 0 and (best is None or x < best):
            best = x
    return best


def _verify_first_extreme(fn, a, b, c, d, x_frac, kind):
    x = float(x_frac) * math.pi
    target = max(_scan_extremes(fn, a, b, c, d)) if False else None
    hi, lo = _scan_extremes(fn, a, b, c, d)
    value = _f(fn, a, b, c, d, x)
    check(abs(value - (hi if kind == "max" else lo)) < 1e-9, "not an extreme at that x")
    period = 2 * math.pi / float(b)
    # nothing smaller and positive is also an extreme
    step = period / 400
    t = step
    while t < x - 1e-9:
        check(abs(_f(fn, a, b, c, d, t) - (hi if kind == "max" else lo)) > 1e-6, "a smaller positive extreme exists")
        t += step


_SHIFTS = [Fraction(n, m) for n, m in ((1, 6), (1, 4), (1, 3), (1, 2), (1, 12), (2, 3), (3, 4), (5, 6))]


def _draw(rng, *, kind, expanded=False, negative_ok=False):
    fn = rng.choice(["sin", "cos"]) if kind != "tan" else "tan"
    a = rng.choice([1, 2, 3, 4, 5])
    if negative_ok and rng.random() < 0.5:
        a = -a
    if expanded:
        b = Fraction(rng.choice([2, 3, 4]))
    else:
        b = (
            Fraction(rng.choice([2, 3, 4, 2, 3]))
            if rng.random() < 0.8
            else Fraction(rng.choice([1, 2]), rng.choice([2, 3]))
        )
        if b.denominator != 1 and b.numerator != 1:
            b = Fraction(1, 2)
    c = rng.choice(_SHIFTS) * rng.choice([-1, 1])
    d = rng.choice([-3, -2, -1, 0, 1, 2, 3, 4])
    return fn, a, b, c, d


# each feature: (answer fraction/number, uses_pi, prompt patterns, distractor factory)
_AMPLITUDE = (
    "State the amplitude of the function {eq}.",
    "What is the amplitude of {eq}?",
    "The graph of {eq} oscillates about its midline. Find its amplitude.",
)
_PERIOD = (
    "State the period of the function {eq}.",
    "Find the period of {eq}.",
    "How long is one full cycle of the graph of {eq}?",
)
_SHIFT = (
    "The graph of {eq} is a translation of a basic {fn} curve. State the horizontal translation, taking a shift to the "
    "right as positive and a shift to the left as negative.",
    "Compared with y = {a_fn}({bx}) the graph of {eq} has been moved horizontally. By how much? Give a negative value "
    "for a move to the left.",
    "Find the horizontal translation (right is positive) that takes the graph of y = {coef}{fn}({bx}) + {dtext} onto the graph "
    "of {eq}.",
)
_VERTICAL = (
    "State the vertical translation of the graph of {eq}, that is, the amount by which its midline is above the x-axis.",
    "The midline of the graph of {eq} is the line y = k. Find k.",
    "Find the value that is halfway between the maximum and the minimum of {eq}.",
)
_MAX = (
    "Find the maximum value of {eq}.",
    "What is the greatest value that y takes on the graph of {eq}?",
    "State the largest value of y for {eq}.",
)
_MIN = (
    "Find the minimum value of {eq}.",
    "What is the smallest value that y takes on the graph of {eq}?",
    "State the least value of y for {eq}.",
)


def _feature_pack(rng, fn, a, b, c, d, feature, *, expanded):
    eq = equation_text(fn, a, b, c, d, expanded=expanded)
    amp = abs(a)
    period = Fraction(2) / b
    k = -b * c
    d = Fraction(d)
    if feature == "amplitude":
        cand = [
            _expr(Fraction(-a), "sinusoid-amplitude-keeps-the-sign", pi=False),
            _expr(Fraction(2 * amp), "sinusoid-amplitude-is-the-full-height", pi=False),
            _expr(b, "sinusoid-amplitude-is-the-multiple-of-x", pi=False),
            _expr(Fraction(amp) + d, "sinusoid-amplitude-includes-the-vertical-shift", pi=False),
        ]
        out = dict(
            **_value_fields(amp, pi=False),
            distractors=need(num_text(amp), cand, 4),
            working=[f"The amplitude is |{a}| = {amp}."],
        )
        return phrase(rng, _AMPLITUDE, eq=eq), out
    if feature == "period":
        cand = [
            _expr(2 * b, "sinusoid-period-multiplies-by-the-multiple"),
            _expr(Fraction(1) / b, "sinusoid-period-uses-pi-not-two-pi"),
            _expr(b, "sinusoid-period-is-the-multiple-of-x"),
            _expr(Fraction(2), "sinusoid-period-ignores-the-multiple"),
        ]
        out = dict(
            **_value_fields(period, pi=True),
            distractors=need(pi_text(period), cand, 4),
            working=[f"The period is 2π ÷ {_b_text(b)} = {pi_text(period)}."],
        )
        return phrase(rng, _PERIOD, eq=eq), out
    if feature == "shift":
        shift = c
        wrong_sign = -shift
        cand = [
            _expr(wrong_sign, "sinusoid-shift-sign-reversed"),
            _expr(shift * b, "sinusoid-shift-multiplied-by-the-multiple"),
            _expr(shift / b, "sinusoid-shift-divided-by-the-multiple"),
        ]
        if expanded:
            cand = [
                _expr(k, "sinusoid-shift-is-the-constant-in-the-bracket"),
                _expr(-k, "sinusoid-shift-is-the-negated-constant-in-the-bracket"),
                _expr(k / b, "sinusoid-shift-sign-reversed"),
                _expr(k * b, "sinusoid-shift-multiplied-by-the-multiple"),
            ]
        text = (
            phrase(
                rng,
                _SHIFT,
                eq=eq,
                fn=fn,
                a_fn=fn,
                bx=("x" if b == 1 else f"{_b_text(b)}x"),
                coef=_coef(a),
                dtext=num_text(d),
            )
            if False
            else None
        )
        stems = (
            "The graph of {eq} is a translation of the graph of y = {coef}{fn}({bx}){dtail}. State the horizontal translation, "
            "taking a shift to the right as positive and a shift to the left as negative.",
            "Compared with y = {coef}{fn}({bx}){dtail}, the graph of {eq} has been moved horizontally. By how much? Give a "
            "negative value for a move to the left.",
            "Find the horizontal translation (right is positive) that takes the graph of y = {coef}{fn}({bx}){dtail} onto the graph "
            "of {eq}.",
        )
        dtail = "" if d == 0 else f" + {num_text(d)}" if d > 0 else f" - {num_text(-d)}"
        text = phrase(rng, stems, eq=eq, fn=fn, bx=("x" if b == 1 else f"{_b_text(b)}x"), coef=_coef(a), dtail=dtail)
        out = dict(
            **_value_fields(shift, pi=True),
            distractors=need(pi_text(shift), cand, 4),
            working=[
                f"Write the bracket as {_b_text(b)}(x - ({pi_text(shift)})): the graph moves {pi_text(shift)} horizontally."
            ],
        )
        return text, out
    if feature == "vertical":
        cand = [
            _expr(-d, "sinusoid-vertical-shift-sign-reversed", pi=False),
            _expr(d + amp, "sinusoid-vertical-shift-is-the-maximum", pi=False),
            _expr(Fraction(amp), "sinusoid-vertical-shift-is-the-amplitude", pi=False),
            _expr(d - amp, "sinusoid-vertical-shift-is-the-minimum", pi=False),
        ]
        out = dict(
            **_value_fields(d, pi=False),
            distractors=need(num_text(d), cand, 4),
            working=[f"The midline is y = {num_text(d)}."],
        )
        return phrase(rng, _VERTICAL, eq=eq), out
    if feature == "max":
        top = amp + d
        cand = [
            _expr(a + d, "sinusoid-extreme-uses-the-signed-amplitude", pi=False),
            _expr(Fraction(amp), "sinusoid-extreme-forgets-the-vertical-shift", pi=False),
            _expr(d - amp, "sinusoid-extreme-takes-the-other-end", pi=False),
            _expr(2 * amp + d, "sinusoid-extreme-uses-the-full-height", pi=False),
        ]
        out = dict(
            **_value_fields(top, pi=False),
            distractors=need(num_text(top), cand, 4),
            working=[f"Maximum = |a| + d = {amp} + ({num_text(d)}) = {num_text(top)}."],
        )
        return phrase(rng, _MAX, eq=eq), out
    check(feature == "min", feature)
    bottom = d - amp
    cand = [
        _expr(-a + d, "sinusoid-extreme-uses-the-signed-amplitude", pi=False),
        _expr(Fraction(-amp), "sinusoid-extreme-forgets-the-vertical-shift", pi=False),
        _expr(d + amp, "sinusoid-extreme-takes-the-other-end", pi=False),
        _expr(d - 2 * amp, "sinusoid-extreme-uses-the-full-height", pi=False),
    ]
    out = dict(
        **_value_fields(bottom, pi=False),
        distractors=need(num_text(bottom), cand, 4),
        working=[f"Minimum = d - |a| = {num_text(d)} - {amp} = {num_text(bottom)}."],
    )
    return phrase(rng, _MIN, eq=eq), out


def _build(rng, features, *, expanded, negative_ok):
    fn, a, b, c, d = _draw(rng, kind="wave", expanded=expanded, negative_ok=negative_ok)
    feature = rng.choice(features)
    want(not (feature == "shift" and c == 0), "no shift to ask about")
    # second route
    hi, lo = _scan_extremes(fn, a, b, c, d)
    check(abs(hi - (abs(a) + d)) < 1e-6 and abs(lo - (d - abs(a))) < 1e-6, "extremes differ from |a| + d and d - |a|")
    _verify_periodic(fn, a, b, c, d, Fraction(2) / b, c)
    text, body = _feature_pack(rng, fn, a, b, c, d, feature, expanded=expanded)
    return dict(text=text, answer_kind="expression", **body)


def _easy(rng):
    return _build(
        rng,
        ["amplitude", "period", "shift", "vertical", "max", "min", "shift", "period"],
        expanded=False,
        negative_ok=False,
    )


def _medium(rng):
    return _build(rng, ["shift", "shift", "amplitude", "max", "min", "period"], expanded=True, negative_ok=True)


_FIRST_STEMS = (
    "Find the smallest positive value of x at which {eq} reaches its {which} value.",
    "For {eq}, what is the least x > 0 at which y is at its {which}?",
    "The graph of {eq} first reaches its {which} value, for positive x, at x = ? Give the exact value.",
)
_INTERCEPT_STEMS = (
    "Find the y-intercept of the graph of {eq}.",
    "At what value of y does the graph of {eq} cross the y-axis?",
    "State the value of y when x = 0 for {eq}.",
)
_ASYMPTOTE_STEMS = (
    "Find the smallest positive value of x at which the graph of {eq} has a vertical asymptote.",
    "The graph of {eq} has vertical asymptotes. Find the first one to the right of the y-axis.",
    "For {eq}, state the smallest positive x for which the function is undefined.",
)


def _hard_extreme(rng):
    fn, a, b, c, d = _draw(rng, kind="wave", expanded=rng.random() < 0.5, negative_ok=True)
    expanded = rng.random() < 0.5
    kind = rng.choice(["max", "min"])
    x = _first_extreme(fn, a, b, c, kind)
    _verify_first_extreme(fn, a, b, c, d, x, kind)
    other = "min" if kind == "max" else "max"
    x_other = _first_extreme(fn, a, b, c, other)
    quarter = 1 / b / 2
    cand = [
        _expr(x_other, "sinusoid-first-extreme-is-the-opposite-one"),
        _expr(c, "sinusoid-first-extreme-is-the-shift"),
        _expr(
            x + Fraction(1) / b if x - Fraction(1) / b <= 0 else x - Fraction(1) / b,
            "sinusoid-first-extreme-off-by-a-half-period",
        ),
        _expr(x + 2 / b if x - 2 / b <= 0 else x - 2 / b, "sinusoid-first-extreme-is-not-the-first"),
        _expr((x - c) * b + c, "sinusoid-first-extreme-multiplies-instead-of-dividing"),
    ]
    which = "greatest" if kind == "max" else "least"
    eq = equation_text(fn, a, b, c, d, expanded=expanded)
    return dict(
        text=phrase(rng, _FIRST_STEMS, eq=eq, which=which),
        answer_kind="expression",
        **_value_fields(x, pi=True),
        distractors=need(pi_text(x), cand, 4),
        working=[
            f"{fn} reaches its {'largest' if (kind == 'max') == (a > 0) else 'smallest'} value when the bracket is {'π/2' if fn == 'sin' and (kind == 'max') == (a > 0) else '3π/2' if fn == 'sin' else '0' if (kind == 'max') == (a > 0) else 'π'} (plus whole turns).",
            f"Solving for x and taking the smallest positive value gives x = {pi_text(x)}.",
        ],
    )


def _hard_intercept(rng):
    fn, a, b, c, d = _draw(rng, kind="wave", expanded=False, negative_ok=True)
    angle = -b * c  # in units of pi
    f = (lambda t: math.sin(t * math.pi)) if fn == "sin" else (lambda t: math.cos(t * math.pi))
    base = f(float(angle))
    exact = _exact_trig(fn, angle)
    want(exact is not None, "an angle with no simple exact value")
    value = a * exact + d
    check(abs(float(value) - _f(fn, a, b, c, d, 0.0)) < 1e-9, "y-intercept differs from the numeric value")
    want(Fraction(value).denominator in (1, 2), "an awkward fraction")
    wrong_sign = _exact_trig(fn, -angle)
    cand = []
    if wrong_sign is not None:
        cand.append(_expr(a * wrong_sign + d, "sinusoid-intercept-sign-of-angle-reversed", pi=False))
    cand.append(_expr(Fraction(d) + a, "sinusoid-intercept-takes-the-maximum", pi=False))
    cand.append(_expr(Fraction(d), "sinusoid-intercept-is-the-vertical-shift", pi=False))
    cand.append(_expr(a * exact, "sinusoid-intercept-forgets-the-vertical-shift", pi=False))
    cand.append(
        _expr(
            Fraction(a) * _exact_trig(fn, angle / b) + d if _exact_trig(fn, angle / b) is not None else Fraction(a) + d,
            "sinusoid-intercept-divides-the-angle-by-the-multiple",
            pi=False,
        )
    )
    eq = equation_text(fn, a, b, c, d)
    return dict(
        text=phrase(rng, _INTERCEPT_STEMS, eq=eq),
        answer_kind="expression",
        **_value_fields(value, pi=False),
        distractors=need(num_text(value), cand, 4),
        working=[f"Put x = 0: y = {a}{fn}({pi_text(angle)}) + {num_text(d)} = {num_text(value)}."],
    )


def _exact_trig(fn, angle):
    """sin or cos of ``angle`` π (a Fraction) as a Fraction when it is 0, ±1/2 or ±1, else None."""
    angle = Fraction(angle) % 2
    table_sin = {
        Fraction(0): 0,
        Fraction(1, 6): Fraction(1, 2),
        Fraction(1, 2): 1,
        Fraction(5, 6): Fraction(1, 2),
        Fraction(1): 0,
        Fraction(7, 6): Fraction(-1, 2),
        Fraction(3, 2): -1,
        Fraction(11, 6): Fraction(-1, 2),
    }
    if fn == "sin":
        v = table_sin.get(angle)
    else:
        v = table_sin.get((angle + Fraction(1, 2)) % 2)
    return None if v is None else Fraction(v)


def _hard_asymptote(rng):
    a = rng.choice([1, 2, 3, -1, -2])
    b = Fraction(rng.choice([2, 3, 4]))
    c = rng.choice(_SHIFTS) * rng.choice([-1, 1])
    d = rng.choice([-2, -1, 0, 1, 3])
    best = None
    for n in range(-30, 31):
        x = (Fraction(1, 2) + n) / b + c
        if x > 0 and (best is None or x < best):
            best = x
    # second route: tan blows up at that x
    inner = float(b) * (float(best) * math.pi - float(c) * math.pi)
    check(abs(math.cos(inner)) < 1e-9, "not an asymptote")
    cand = [
        _expr(c, "sinusoid-asymptote-is-the-shift"),
        _expr(
            best + Fraction(1, 2) / b if best - Fraction(1, 2) / b <= 0 else best - Fraction(1, 2) / b,
            "sinusoid-asymptote-off-by-a-half-period",
        ),
        _expr(c + Fraction(1) / b, "sinusoid-asymptote-uses-the-period-not-half"),
        _expr((best - c) * b + c, "sinusoid-asymptote-multiplies-instead-of-dividing"),
    ]
    eq = equation_text("tan", a, b, c, d)
    return dict(
        text=phrase(rng, _ASYMPTOTE_STEMS, eq=eq),
        answer_kind="expression",
        **_value_fields(best, pi=True),
        distractors=need(pi_text(best), cand, 4),
        working=[
            f"tan is undefined when the bracket is π/2 + nπ, so {_b_text(b)}(x - ({pi_text(c)})) = π/2 + nπ.",
            f"The smallest positive x is {pi_text(best)}.",
        ],
    )


def _hard(rng):
    return rng.choice([_hard_extreme, _hard_extreme, _hard_intercept, _hard_asymptote])(rng)


@generator(
    "w6b_sinusoid_features",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6b_sinusoid_features(rng, tier="easy"):
    """Features of sinusoids read from their equations; see the module docstring."""
    if tier == "easy":
        return retry(rng, _easy)
    if tier == "medium":
        return retry(rng, _medium)
    check(tier == "hard", f"unknown tier {tier!r}")
    return retry(rng, _hard)


# -- exam -----------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "The graph of {eq} is drawn for all real x.",
    "A function is defined by {eq}.",
    "A sinusoidal model is given by {eq}, where x is in radians.",
)


def _exam(rng):
    fn = rng.choice(["sin", "cos"])
    a = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    b = Fraction(rng.choice([2, 3, 4]))
    c = rng.choice(_SHIFTS) * rng.choice([-1, 1])
    d = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    want(abs(a) != abs(d), "amplitude and shift coincide")
    hi, lo = _scan_extremes(fn, a, b, c, d)
    check(abs(hi - (abs(a) + d)) < 1e-6, "numeric maximum differs")
    _verify_periodic(fn, a, b, c, d, Fraction(2) / b, c)
    eq = equation_text(fn, a, b, c, d, expanded=True)
    amp, period, top = abs(a), Fraction(2) / b, abs(a) + d
    k = -b * c
    amp_cand = [
        _expr(Fraction(a), "sinusoid-amplitude-keeps-the-sign", pi=False),
        _expr(Fraction(2 * amp), "sinusoid-amplitude-is-the-full-height", pi=False),
        _expr(b, "sinusoid-amplitude-is-the-multiple-of-x", pi=False),
    ]
    period_cand = [
        _expr(2 * b, "sinusoid-period-multiplies-by-the-multiple"),
        _expr(Fraction(1) / b, "sinusoid-period-uses-pi-not-two-pi"),
        _expr(Fraction(2), "sinusoid-period-ignores-the-multiple"),
    ]
    shift_cand = [
        _expr(k, "sinusoid-shift-is-the-constant-in-the-bracket"),
        _expr(-k, "sinusoid-shift-is-the-negated-constant-in-the-bracket"),
        _expr(k / b, "sinusoid-shift-sign-reversed"),
        _expr(k * b, "sinusoid-shift-multiplied-by-the-multiple"),
    ]
    max_cand = [
        _expr(a + d, "sinusoid-extreme-uses-the-signed-amplitude", pi=False),
        _expr(Fraction(amp), "sinusoid-extreme-forgets-the-vertical-shift", pi=False),
        _expr(d - amp, "sinusoid-extreme-takes-the-other-end", pi=False),
        _expr(2 * amp + d, "sinusoid-extreme-uses-the-full-height", pi=False),
    ]
    d_text = f"({d})"
    return dict(
        text=phrase(rng, _EXAM_STEMS, eq=eq),
        parts=[
            dict(
                prompt="State the amplitude of the function.",
                answer_kind="expression",
                **_value_fields(amp, pi=False),
                marks=1,
                sensible_range=(0, None),
                working=[f"The amplitude is |{a}| = {amp}."],
                distractors=need(num_text(amp), amp_cand, 3),
            ),
            dict(
                prompt="State the period of the function.",
                answer_kind="expression",
                **_value_fields(period, pi=True),
                marks=1,
                sensible_range=(0, None),
                working=[f"The period is 2π ÷ {_b_text(b)} = {pi_text(period)}."],
                distractors=need(pi_text(period), period_cand, 3),
            ),
            dict(
                prompt=(
                    f"The graph is a horizontal translation of the graph of y = {_coef(a)}{fn}({_b_text(b)}x) + {d}"
                    if d > 0
                    else f"The graph is a horizontal translation of the graph of y = {_coef(a)}{fn}({_b_text(b)}x) - {-d}"
                )
                + ". State the translation, with a shift to the right as positive.",
                answer_kind="expression",
                **_value_fields(c, pi=True),
                marks=2,
                working=[f"Factor the bracket: {_b_text(b)}x + ({pi_text(k)}) = {_b_text(b)}(x - ({pi_text(c)}))."],
                distractors=need(pi_text(c), shift_cand, 3),
            ),
            dict(
                prompt="Hence find the maximum value of the function.",
                answer_kind="expression",
                **_value_fields(top, pi=False),
                marks=1,
                follow_through=f"part_a + {d_text}",
                working=[f"Maximum = amplitude + vertical shift = {amp} + ({d}) = {top}."],
                distractors=need(num_text(top), max_cand, 3),
            ),
        ],
    )


@generator("exd_x_w6b_sinusoid_features", level=16, presets={"exam": dict(tier="exam")})
def exd_x_w6b_sinusoid_features(rng, tier="exam"):
    """Exam chain: amplitude, period, horizontal translation and (following through) the maximum of a sinusoid."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return retry(rng, _exam)
