"""Generates `s2c3_fourier_series` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_fourier_series`.

``uc2-fourier-series-full-period`` (``fourier-analysis-and-pdes``, answer ``number``): Fourier coefficients and the
sum of the Fourier series of a piecewise function with period 2L, read from the integrals over [-L, L].

* easy: one coefficient (a0, an or bn, never a zero one) of a step function with two constant values, the step at
  x = 0 or at +-L/2.
* medium: one coefficient of a function with a linear piece, so the integral needs integration by parts.
* hard: either the value the series converges to at a stated x (a jump, the end of the period, or a point outside
  [-L, L) that has to be brought back by the period), or one coefficient of a function whose pieces break at +-L/2.
* exam (``exd_x_s2c3_fourier_series``): (a) a_n, (b) b_n, (c) the amplitude of the n-th harmonic from (a) and (b)
  (follows through), (d) the value of the series at a jump or at the end of the period.

Every coefficient is the closed form of the integral, checked against composite Simpson quadrature; every series
value is checked against a 160-term partial sum built from Simpson-integrated coefficients.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._s2c3_common import (
    fixed,
    key_ok,
    number_part,
    places_text,
    prompt_fields,
    stem_fields,
    wrong_numbers,
)
from ._s2c3_fourier import HALF_PERIODS, Fn, Piece, coefficient_name, multiple_of_L

DP = 3
_SERIES = (
    "f(x) = a₀/2 + Σ (aₙ cos(nπx/L) + bₙ sin(nπx/L))",
    r"f(x) \sim \frac{a_0}{2}+\sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{L}+b_n\sin\frac{n\pi x}{L}\right)",
)


def _draw_L(rng):
    return rng.choice(HALF_PERIODS)


def _definition(fn: Fn, rng) -> list:
    """The shared opening: the function, its period and the series convention."""
    plain, tex = fn.describe()
    period_plain, period_tex = multiple_of_L(Fraction(2), fn.L_text, fn.L_tex)
    return [
        "Let f be the function of period ",
        (period_plain, period_tex),
        " with ",
        (plain, tex),
        ". Its Fourier series on [−L, L] is ",
        _SERIES,
        ", where L = ",
        (fn.L_text, fn.L_tex),
        ".",
    ]


def _step_fn(rng) -> Fn:
    L, lt, ltex = _draw_L(rng)
    for _ in attempts(100, "a step function"):
        c1, c2 = rng.randint(-4, 5), rng.randint(-4, 5)
        if c1 == c2 or c1 == 0 and c2 == 0:
            continue
        p = rng.choice([Fraction(0), Fraction(0), Fraction(1, 2), Fraction(-1, 2)])
        return Fn(L, lt, ltex, [Piece(Fraction(-1), p, c1, 0), Piece(p, Fraction(1), c2, 0)])
    raise AssertionError  # pragma: no cover


def _linear_fn(rng, breakpoints=(Fraction(0),)) -> Fn:
    L, lt, ltex = _draw_L(rng)
    for _ in attempts(200, "a piecewise linear function"):
        p = rng.choice(breakpoints)
        s1 = rng.choice([-2, -1, 0, 1, 2])
        s2 = rng.choice([-2, -1, 0, 1, 2])
        if s1 == 0 and s2 == 0:
            continue
        # keep |f| modest whatever L is: slopes are scaled by 1/L rounded to a whole number of units
        c1, c2 = rng.randint(-3, 4), rng.randint(-3, 4)
        if (c1, s1) == (c2, s2):
            continue
        fn = Fn(L, lt, ltex, [Piece(Fraction(-1), p, c1, s1), Piece(p, Fraction(1), c2, s2)])
        if max(abs(fn.at_left_end(q)) for q in fn.pieces) > 14 or max(abs(fn.at_right_end(q)) for q in fn.pieces) > 14:
            continue
        return fn
    raise AssertionError  # pragma: no cover


def _coef_candidates(fn: Fn, kind: str, n: int):
    """(value, misconception id) for the errors a learner makes computing this coefficient."""
    true = fn.coefficient(kind, n)
    L = fn.L
    out = []
    if kind == "a" and n == 0:
        out.append((true / 2, "fourier-a0-half-confused"))
        out.append((true * 2, "fourier-normalisation-doubled"))
        out.append((fn.integral("a", 0.0, Fraction(0), Fraction(1)) / L, "fourier-one-piece-only"))
        out.append((fn.integral("a", 0.0) / (2 * L), "fourier-normalisation-halved"))
        return out
    out.append((true / 2, "fourier-normalisation-halved"))
    out.append((fn.integral(kind, n * math.pi / L, Fraction(0), Fraction(1)) / L, "fourier-one-piece-only"))
    if L != math.pi:
        out.append((fn.integral(kind, float(n)) / L, "fourier-period-scaling-dropped"))
    other = "b" if kind == "a" else "a"
    out.append((fn.coefficient(other, n), "fourier-sine-cosine-swap"))
    out.append((-true, "fourier-coefficient-sign-flip"))
    return out


def _coefficient_question(rng, fn: Fn, kind: str, n: int):
    """The single-answer dict for 'find this coefficient'."""
    name, name_tex = coefficient_name(kind, n)
    true = fn.coefficient(kind, n)
    lo, hi = Fraction(-1), Fraction(1)
    check(abs(fn.check_coefficient(kind, n, 1 / fn.L, lo, hi) - true) < 1e-7, "Simpson quadrature disagrees")
    cands = _coef_candidates(fn, kind, n)
    d = wrong_numbers(true, DP, cands, 4)
    check(len(d) >= 2, "too few distractors")
    return true, dict(
        **stem_fields(
            *_definition(fn, rng),
            " Find ",
            (name, name_tex),
            f", correct to {places_text(DP)}.",
        ),
        **number_part(true, DP),
        working=[
            f"{name} = (1/L) ∫ f(x) {'cos' if kind == 'a' else 'sin'}({'nπx/L' if n else '0'}) dx over [−L, L], "
            f"split at the break in f, = {fixed(true, DP)}."
        ],
        distractors=d,
    )


def _pick_coefficient(rng, fn: Fn, allow_a0=True):
    for _ in attempts(100, "a clearly non-zero coefficient"):
        kind = rng.choice(["a", "b", "b", "a"] + (["a0"] if allow_a0 else []))
        n = 0 if kind == "a0" else rng.randint(1, 3)
        kind = "a" if kind == "a0" else kind
        v = fn.coefficient(kind, n)
        if key_ok(v, DP, smallest=0.08) and abs(v) < 40:
            return kind, n
    raise AssertionError  # pragma: no cover


def _easy(rng):
    fn = _step_fn(rng)
    kind, n = _pick_coefficient(rng, fn)
    return _coefficient_question(rng, fn, kind, n)[1]


def _medium(rng):
    fn = _linear_fn(rng)
    kind, n = _pick_coefficient(rng, fn)
    return _coefficient_question(rng, fn, kind, n)[1]


# -- hard: the value of the series --------------------------------------------------------------------------------


def _value_candidates(fn: Fn, a: Fraction, value, left, right, how):
    r = fn.reduced(a)
    out = []
    if how in ("jump", "endpoint"):
        out.append((left, "fourier-jump-one-sided-value"))
        out.append((right, "fourier-jump-one-sided-value"))
        out.append((0.0, "fourier-jump-value-zero"))
    if not -1 <= a < 1:
        out.append((_unreduced(fn, a), "fourier-periodicity-ignored"))
    out.append((_mean_value(fn), "fourier-mean-value-only"))
    first, last = fn.pieces[0], fn.pieces[-1]
    out.append(((fn.at_left_end(first) + fn.at_right_end(last)) / 2, "fourier-jump-average-wrong-pair"))
    return out


def _unreduced(fn: Fn, a: Fraction) -> float:
    """What a learner gets by feeding x = a L into the formula of the nearest piece without using the period."""
    piece = fn.pieces[-1] if a >= 1 else fn.pieces[0]
    return fn.piece_value(piece, float(a) * fn.L)


def _mean_value(fn: Fn) -> float:
    return fn.integral("a", 0.0) / (2 * fn.L)


def _draw_value_point(rng, fn: Fn):
    """(a, how): an x = a L, with how in 'jump', 'endpoint', 'continuous', whose value genuinely needs the theorem."""
    breaks = fn.breakpoints()
    for _ in attempts(200, "a series-value point"):
        mode = rng.choice(["jump", "endpoint", "shifted-jump", "shifted-continuous", "shifted-endpoint"])
        shift = 2 * rng.choice([-2, -1, 1, 2]) if mode.startswith("shifted") else 0
        if mode.endswith("jump"):
            base = rng.choice(breaks)
        elif mode.endswith("endpoint"):
            base = Fraction(rng.choice([-1, 1]))
        else:
            base = rng.choice([Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 4), Fraction(1, 2)])
            if any(abs(base - b) < Fraction(1, 4) for b in breaks + [Fraction(-1), Fraction(1)]):
                continue
        a = base + shift
        value, left, right, how = fn.series_value(a)
        if mode.endswith("continuous") != (how == "continuous"):
            continue
        if how != "continuous" and abs(left - right) < 1:
            continue
        return a, value, left, right, how
    raise AssertionError  # pragma: no cover


def _check_series_value(fn: Fn, a: Fraction, value: float) -> None:
    tolerance = 0.12 * (1 + max(abs(fn.at_left_end(p)) + abs(fn.at_right_end(p)) for p in fn.pieces))
    check(abs(fn.partial_sum(a) - value) < tolerance, "the partial sum does not approach the claimed series value")


def _x_text(fn: Fn, a: Fraction):
    plain, tex = multiple_of_L(a, fn.L_text, fn.L_tex)
    return plain, tex


def _hard_value(rng):
    fn = _linear_fn(rng, breakpoints=(Fraction(0), Fraction(1, 2), Fraction(-1, 2)))
    for _ in attempts(100, "a series value that is a safe key"):
        a, value, left, right, how = _draw_value_point(rng, fn)
        if key_ok(value, DP, smallest=0.0):
            break
    _check_series_value(fn, a, value)
    cands = _value_candidates(fn, a, value, left, right, how)
    d = wrong_numbers(value, DP, cands, 4)
    check(len(d) >= 2, "too few distractors")
    x_plain, x_tex = _x_text(fn, a)
    return dict(
        **stem_fields(
            *_definition(fn, rng)[:4],
            ". To what value does the Fourier series of f converge at x = ",
            (x_plain, x_tex),
            f"? Give the value correct to {places_text(DP)}.",
        ),
        **number_part(value, DP),
        working=[
            f"Bring x = {x_plain} back into [−L, L) using the period, "
            + (
                f"where f jumps from {fixed(left, DP)} to {fixed(right, DP)}, so the series converges to the average {fixed(value, DP)}."
                if how != "continuous"
                else f"where f is continuous, so the series converges to f there, {fixed(value, DP)}."
            )
        ],
        distractors=d,
    )


def _hard_breaks(rng):
    fn = _linear_fn(rng, breakpoints=(Fraction(1, 2), Fraction(-1, 2)))
    kind, n = _pick_coefficient(rng, fn, allow_a0=False)
    return _coefficient_question(rng, fn, kind, n)[1]


@generator(
    "s2c3_fourier_series",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_fourier_series(rng, tier="easy"):
    """Full-period Fourier coefficients and series values; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_value, _hard_value, _hard_breaks])(rng)


# -- exam -----------------------------------------------------------------------------------------------------------


@generator("exd_x_s2c3_fourier_series", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_fourier_series(rng, tier="exam"):
    """Exam chain: a_n, b_n, the harmonic amplitude from both, and the series value at a jump."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam function"):
        fn = _linear_fn(rng, breakpoints=(Fraction(0), Fraction(1, 2), Fraction(-1, 2)))
        n = rng.randint(1, 2)
        a_val, b_val = fn.coefficient("a", n), fn.coefficient("b", n)
        amp = math.hypot(round(a_val, DP), round(b_val, DP))
        if not (key_ok(a_val, DP, 0.1) and key_ok(b_val, DP, 0.1) and key_ok(amp, DP, 0.1)):
            continue
        try:
            a, value, left, right, how = _draw_value_point(rng, fn)
        except AssertionError:
            continue
        if how == "continuous" or not key_ok(value, DP):
            continue
        break
    else:
        raise AssertionError  # pragma: no cover
    for kind, val in (("a", a_val), ("b", b_val)):
        check(
            abs(fn.check_coefficient(kind, n, 1 / fn.L, Fraction(-1), Fraction(1)) - val) < 1e-7, "quadrature differs"
        )
    _check_series_value(fn, a, value)
    an, an_tex = coefficient_name("a", n)
    bn, bn_tex = coefficient_name("b", n)
    ca = wrong_numbers(a_val, DP, _coef_candidates(fn, "a", n), 3)
    cb = wrong_numbers(b_val, DP, _coef_candidates(fn, "b", n), 3)
    ra, rb = round(a_val, DP), round(b_val, DP)
    c_cands = [
        (abs(ra) + abs(rb), "fourier-amplitude-added-linearly"),
        (ra * ra + rb * rb, "fourier-amplitude-no-root"),
        (max(abs(ra), abs(rb)), "fourier-amplitude-larger-only"),
        (math.hypot(ra, rb) / 2, "fourier-normalisation-halved"),
    ]
    cc = wrong_numbers(amp, DP, c_cands, 3)
    cd = wrong_numbers(value, DP, _value_candidates(fn, a, value, left, right, how), 3)
    check(min(len(ca), len(cb), len(cc), len(cd)) >= 2, "too few distractors")
    x_plain, x_tex = _x_text(fn, a)
    return dict(
        **stem_fields(*_definition(fn, rng)),
        parts=[
            dict(
                **prompt_fields("Find ", (an, an_tex), f", correct to {places_text(DP)}."),
                **number_part(a_val, DP),
                marks=3,
                working=[f"{an} = (1/L) ∫ f(x) cos(nπx/L) dx, integrating each piece by parts, = {fixed(a_val, DP)}."],
                distractors=ca,
            ),
            dict(
                **prompt_fields("Find ", (bn, bn_tex), f", correct to {places_text(DP)}."),
                **number_part(b_val, DP),
                marks=3,
                working=[f"{bn} = (1/L) ∫ f(x) sin(nπx/L) dx, integrating each piece by parts, = {fixed(b_val, DP)}."],
                distractors=cb,
            ),
            dict(
                **prompt_fields(
                    f"The n = {n} harmonic of the series is {an} cos(nπx/L) + {bn} sin(nπx/L). "
                    f"Find its amplitude, correct to {places_text(DP)}."
                ),
                **number_part(amp, DP),
                marks=1,
                sensible_range=(0, None),
                follow_through="sqrt(part_a**2 + part_b**2)",
                working=[f"Amplitude = √({an}² + {bn}²) = {fixed(amp, DP)}."],
                distractors=cc,
            ),
            dict(
                **prompt_fields(
                    "To what value does the Fourier series of f converge at x = ",
                    (x_plain, x_tex),
                    f"? Give the value correct to {places_text(DP)}.",
                ),
                **number_part(value, DP),
                marks=2,
                working=[
                    f"f jumps from {fixed(left, DP)} to {fixed(right, DP)} there, so the series gives the average, {fixed(value, DP)}."
                ],
                distractors=cd,
            ),
        ],
    )
