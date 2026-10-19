"""Generates `s2c3_fourier_half_range` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_fourier_half_range`.

``uc2-fourier-half-range`` (``fourier-analysis-and-pdes``, answer ``number``): the half-range Fourier sine and cosine
series of a function given on [0, L], as the odd and the even periodic extension.

* easy: one coefficient (b_n of the sine series, or a_0 or a_n of the cosine series) of a single linear piece.
* medium: one coefficient of a function with two linear pieces that break at L/2 or at a quarter point.
* hard: the value a half-range series converges to at a stated x: a jump, the end points 0 and L, a negative x or a
  point beyond L, so the odd or even extension and the period have to be used.
* exam (``exd_x_s2c3_fourier_half_range``): (a) the sine coefficient b_n, (b) the cosine coefficient a_n, (c) the value
  of the sine series and (d) the value of the cosine series at the same point outside (0, L).

Every coefficient is the closed-form integral checked against Simpson quadrature; every series value is checked
against a 160-term partial sum built from Simpson-integrated coefficients.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import VerificationError, attempts, check, generator
from ._s2c3_common import fixed, key_ok, number_part, places_text, prompt_fields, stem_fields, wrong_numbers
from ._s2c3_fourier import (
    HALF_PERIODS,
    Fn,
    Piece,
    coefficient_name,
    half_range_extension,
    half_range_partial_sum,
    half_range_value,
    multiple_of_L,
)

DP = 3
_NAME = {"b": "sine", "a": "cosine"}
_FORMULA = {
    "b": ("f(x) = Σ bₙ sin(nπx/L)", r"f(x) = \sum_{n=1}^{\infty} b_n\sin\frac{n\pi x}{L}"),
    "a": ("f(x) = a₀/2 + Σ aₙ cos(nπx/L)", r"f(x) = \frac{a_0}{2}+\sum_{n=1}^{\infty} a_n\cos\frac{n\pi x}{L}"),
}


def _definition(fn: Fn) -> list:
    plain, tex = fn.describe(close_last=True)
    return ["Let f be defined on [0, L], where L = ", (fn.L_text, fn.L_tex), ", by ", (plain, tex), ". "]


def _series_sentence(series: str) -> list:
    return [f"Expand f in a half-range Fourier {_NAME[series]} series, ", _FORMULA[series], ". "]


def _draw_fn(rng, pieces: int, breaks=(Fraction(1, 2),)) -> Fn:
    L, lt, ltex = rng.choice(HALF_PERIODS)
    for _ in attempts(200, "a function on [0, L]"):
        if pieces == 1:
            c0, c1 = rng.randint(-3, 5), rng.choice([-2, -1, 0, 1, 2])
            if c1 == 0 and c0 == 0 or (c0 == 0 and c1 == 0):
                continue
            ps = [Piece(Fraction(0), Fraction(1), c0, c1)]
        else:
            p = rng.choice(breaks)
            c0, c1, d0, d1 = (
                rng.randint(-3, 5),
                rng.choice([-2, -1, 0, 1, 2]),
                rng.randint(-3, 5),
                rng.choice([-2, -1, 0, 1, 2]),
            )
            if (c0, c1) == (d0, d1):
                continue
            ps = [Piece(Fraction(0), p, c0, c1), Piece(p, Fraction(1), d0, d1)]
        fn = Fn(L, lt, ltex, ps)
        if max(max(abs(fn.at_left_end(q)), abs(fn.at_right_end(q))) for q in ps) > 14:
            continue
        return fn
    raise AssertionError  # pragma: no cover


def _norm(fn: Fn) -> float:
    return 2 / fn.L


def _coefficient(fn: Fn, series: str, n: int) -> float:
    import math

    return fn.integral(series, n * math.pi / fn.L, Fraction(0), Fraction(1)) * _norm(fn)


def _coef_candidates(fn: Fn, series: str, n: int):
    import math

    true = _coefficient(fn, series, n)
    other = "a" if series == "b" else "b"
    out = [(true / 2, "fourier-half-range-factor-dropped")]
    if series == "a" and n == 0:
        out += [(true * 2, "fourier-normalisation-doubled"), (true / 4, "fourier-a0-half-confused")]
        return out
    out.append((_coefficient(fn, other, n), "fourier-sine-cosine-swap"))
    if fn.L != math.pi:
        out.append(
            (fn.integral(series, float(n), Fraction(0), Fraction(1)) * _norm(fn), "fourier-period-scaling-dropped")
        )
    out.append((-true, "fourier-coefficient-sign-flip"))
    out.append((true * 2, "fourier-normalisation-doubled"))
    return out


def _coefficient_question(rng, fn: Fn, series: str, n: int) -> dict:
    name, name_tex = coefficient_name("b" if series == "b" else "a", n)
    true = _coefficient(fn, series, n)
    check(abs(fn.check_coefficient(series, n, _norm(fn), Fraction(0), Fraction(1)) - true) < 1e-7, "quadrature differs")
    d = wrong_numbers(true, DP, _coef_candidates(fn, series, n), 4)
    check(len(d) >= 2, "too few distractors")
    return dict(
        **stem_fields(
            *_definition(fn), *_series_sentence(series), "Find ", (name, name_tex), f", correct to {places_text(DP)}."
        ),
        **number_part(true, DP),
        working=[f"{name} = (2/L) ∫₀ᴸ f(x) {'sin' if series == 'b' else 'cos'}(nπx/L) dx = {fixed(true, DP)}."],
        distractors=d,
    )


def _pick(rng, fn: Fn, allow_a0: bool):
    for _ in attempts(100, "a clearly non-zero coefficient"):
        series = rng.choice(["b", "b", "a"])
        n = 0 if (series == "a" and allow_a0 and rng.random() < 0.3) else rng.randint(1, 3)
        v = _coefficient(fn, series, n)
        if key_ok(v, DP, 0.08) and abs(v) < 40:
            return series, n
    raise AssertionError  # pragma: no cover


def _easy(rng):
    fn = _draw_fn(rng, 1)
    return _coefficient_question(rng, fn, *_pick(rng, fn, True))


def _medium(rng):
    fn = _draw_fn(rng, 2, (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)))
    return _coefficient_question(rng, fn, *_pick(rng, fn, True))


# -- series values --------------------------------------------------------------------------------------------------


def _draw_point(rng, fn: Fn, kinds=("negative", "end", "origin", "jump", "beyond", "periodic")):
    breaks = fn.breakpoints()
    for _ in attempts(300, "a series-value point"):
        kind = rng.choice(kinds)
        q = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1, 3), Fraction(2, 3)])
        if kind in ("negative", "beyond", "periodic") and q in breaks:
            continue
        a = {
            "negative": -q,
            "end": Fraction(1),
            "origin": Fraction(0),
            "jump": breaks[0] * rng.choice([1, -1, 1]) if breaks else Fraction(1, 2),
            "beyond": 1 + q,
            "periodic": 2 + q,
        }[kind]
        return kind, a
    raise AssertionError  # pragma: no cover


def _check_value(fn: Fn, series: str, a: Fraction, value: float) -> None:
    tolerance = 0.12 * (1 + max(abs(fn.at_left_end(p)) + abs(fn.at_right_end(p)) for p in fn.pieces))
    check(abs(half_range_partial_sum(fn, series, a) - value) < tolerance, "the partial sum does not approach the value")


def _value_candidates(fn: Fn, series: str, a: Fraction, value: float):
    other = "a" if series == "b" else "b"
    out = []
    left, right = half_range_value(fn, series, a)[1:]
    if abs(left - right) > 1e-6:
        out += [(left, "fourier-jump-one-sided-value"), (right, "fourier-jump-one-sided-value")]
    out.append((half_range_value(fn, other, a)[0], "fourier-wrong-extension"))
    # the formula of the nearest piece pushed outside [0, L], or f(L) read where the series is not f
    piece = fn.pieces[-1] if a > 0 else fn.pieces[0]
    out.append((fn.piece_value(piece, float(a) * fn.L), "fourier-periodicity-ignored"))
    if series == "b" and a in (0, 1):
        out.append((fn.value_at(a) if a == 0 else fn.at_right_end(fn.pieces[-1]), "fourier-sine-end-not-zero"))
    out.append((0.0, "fourier-jump-value-zero"))
    return out


def _x_text(fn: Fn, a: Fraction):
    return multiple_of_L(a, fn.L_text, fn.L_tex)


def _value_dict(rng, fn: Fn, series: str, kind: str, a: Fraction) -> dict:
    value = half_range_value(fn, series, a)[0]
    _check_value(fn, series, a, value)
    d = wrong_numbers(value, DP, _value_candidates(fn, series, a, value), 4)
    check(len(d) >= 2, "too few distractors")
    x_plain, x_tex = _x_text(fn, a)
    return dict(
        **stem_fields(
            *_definition(fn),
            *_series_sentence(series),
            "To what value does this series converge at x = ",
            (x_plain, x_tex),
            f"? Give the value correct to {places_text(DP)}.",
        ),
        **number_part(value, DP),
        working=[
            f"The {_NAME[series]} series is the {'odd' if series == 'b' else 'even'} extension of f with period 2L; "
            f"at x = {x_plain} it converges to {fixed(value, DP)}."
        ],
        distractors=d,
    )


def _hard(rng):
    for _ in attempts(200, "a hard half-range value"):
        fn = _draw_fn(rng, 2, (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)))
        series = rng.choice(["b", "a"])
        kind, a = _draw_point(rng, fn)
        value = half_range_value(fn, series, a)[0]
        zero_ok = kind in ("end", "origin") and series == "b"
        if not key_ok(value, DP, 0.0 if zero_ok else 0.08):
            continue
        if kind == "jump" and abs(half_range_value(fn, series, a)[1] - half_range_value(fn, series, a)[2]) < 1:
            continue
        try:
            return _value_dict(rng, fn, series, kind, a)
        except VerificationError:
            continue  # too few wrong options for this point: draw another
    raise AssertionError  # pragma: no cover


@generator(
    "s2c3_fourier_half_range",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_fourier_half_range(rng, tier="easy"):
    """Half-range sine and cosine series; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


@generator("exd_x_s2c3_fourier_half_range", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_fourier_half_range(rng, tier="exam"):
    """Exam chain: b_n, a_n, then the sine- and cosine-series values at one point outside (0, L)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam function"):
        fn = _draw_fn(rng, 2, (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)))
        n = rng.randint(1, 3)
        bn, an = _coefficient(fn, "b", n), _coefficient(fn, "a", n)
        kind, a = _draw_point(rng, fn, kinds=("negative", "beyond"))
        a2 = fn.breakpoints()[0]
        vs, (vc, vc_left, vc_right) = half_range_value(fn, "b", a)[0], half_range_value(fn, "a", a2)
        if (
            all(key_ok(v, DP, 0.08) for v in (bn, an, vs, vc))
            and abs(vc_left - vc_right) >= 1
            and len(wrong_numbers(vs, DP, _value_candidates(fn, "b", a, vs), 3)) >= 2
            and len(wrong_numbers(vc, DP, _value_candidates(fn, "a", a2, vc), 3)) >= 2
        ):
            break
    else:
        raise AssertionError  # pragma: no cover
    for series, v in (("b", bn), ("a", an)):
        check(
            abs(fn.check_coefficient(series, n, _norm(fn), Fraction(0), Fraction(1)) - v) < 1e-7, "quadrature differs"
        )
    _check_value(fn, "b", a, vs)
    _check_value(fn, "a", a2, vc)
    bname, btex = coefficient_name("b", n)
    aname, atex = coefficient_name("a", n)
    x_plain, x_tex = _x_text(fn, a)
    x2_plain, x2_tex = _x_text(fn, a2)
    parts = []
    specs = [
        (
            prompt_fields("In its half-range sine series, find ", (bname, btex), f", correct to {places_text(DP)}."),
            bn,
            _coef_candidates(fn, "b", n),
            3,
            f"{bname} = (2/L) ∫₀ᴸ f(x) sin(nπx/L) dx = {fixed(bn, DP)}.",
        ),
        (
            prompt_fields("In its half-range cosine series, find ", (aname, atex), f", correct to {places_text(DP)}."),
            an,
            _coef_candidates(fn, "a", n),
            3,
            f"{aname} = (2/L) ∫₀ᴸ f(x) cos(nπx/L) dx = {fixed(an, DP)}.",
        ),
        (
            prompt_fields(
                "To what value does the half-range sine series converge at x = ",
                (x_plain, x_tex),
                f"? Give the value correct to {places_text(DP)}.",
            ),
            vs,
            _value_candidates(fn, "b", a, vs),
            2,
            f"The sine series is the odd extension of f; it converges to {fixed(vs, DP)} at x = {x_plain}.",
        ),
        (
            prompt_fields(
                "To what value does the half-range cosine series converge at x = ",
                (x2_plain, x2_tex),
                f"? Give the value correct to {places_text(DP)}.",
            ),
            vc,
            _value_candidates(fn, "a", a2, vc),
            2,
            f"At x = {x2_plain} f jumps from {fixed(vc_left, DP)} to {fixed(vc_right, DP)}, so the cosine series converges to the average {fixed(vc, DP)}.",
        ),
    ]
    for fields, value, cands, marks, working in specs:
        d = wrong_numbers(value, DP, cands, 3)
        check(len(d) >= 2, "too few distractors")
        parts.append(dict(**fields, **number_part(value, DP), marks=marks, working=[working], distractors=d))
    return dict(**stem_fields(*_definition(fn)[:-1], "."), parts=parts)
