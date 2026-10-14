"""Richardson extrapolation: `r4h_richardson_extrapolation` (easy, medium, hard) and the exam chain
`exd_x_r4h_richardson_extrapolation`.

``uc2-richardson-extrapolation`` (``numerical-and-computational-mathematics``). A formula ``A(h)`` whose error is
``c₁h^p + c₂h^q + …`` is combined with the same formula at ``h/2``: ``(2^p A(h/2) − A(h)) / (2^p − 1)`` removes the
``h^p`` term.

* easy: forward differences ``(f(x+h) − f(x))/h`` for a cubic at ``h`` and ``h/2`` (error order 1), extrapolated once.
* medium: central differences for a quintic at ``h`` and ``h/2`` (error order 2), extrapolated once.
* hard: three trapezoid-rule values (as decimals) whose error has the powers ``h², h⁴, h⁶``; extrapolate twice and
  give the result to 5 decimal places.
* exam: a central difference for a quintic: ``D(h)``, ``D(h/2)``, the extrapolated value (follow-through from the two)
  and its error against the exact derivative (follow-through from the extrapolated value).

Everything is exact (`Fraction`). The test recomputes each answer a second way from the text.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    dec,
    dtxt,
    margin_ok,
    number_part,
    ft_symbol,
    finish_parts,
    poly_eval,
    pn,
    poly_prompt,
    rat,
    single_from,
    tx,
    with_redraw,
)

MISC = {
    "num2t-rich-finer-only": "Reporting the estimate at the smaller step size and not combining it with the larger-step estimate.",
    "num2t-rich-weights-swapped": "Swapping the roles of the two estimates, so the coarse estimate gets the larger weight.",
    "num2t-rich-average": "Averaging the two estimates instead of extrapolating, so the leading error is only halved.",
    "num2t-rich-wrong-order-weights": "Using the weights for the wrong error order, such as (4D(h/2) − D(h))/3 for a first-order formula.",
    "num2t-rich-first-order-weights": "Using the first-order weights 2D(h/2) − D(h) for a second-order formula.",
    "num2t-rich-coarse-only": "Reporting the estimate at the larger step size.",
    "num2t-rich-one-level-only": "Stopping after the first level of extrapolation when two error terms were to be removed.",
    "num2t-rich-second-level-weights": "Using the weights 4 and 3 again at the second level instead of 16 and 15.",
    "num2t-rich-levels-in-wrong-order": "Combining the first and last estimates directly instead of extrapolating the two improved estimates.",
    "num2t-rich-no-division-by-step": "Leaving out the division by the step size in a difference quotient.",
    "num2t-rich-no-division-by-double-step": "Dividing a central difference by h instead of 2h.",
    "num2t-rich-forward-for-central": "Using a forward difference where a central difference was specified.",
    "num2t-rich-error-sign": "Reporting the exact value minus the estimate instead of the estimate minus the exact value.",
    "num2t-rich-error-of-coarse": "Computing the error of the coarse central difference instead of the extrapolated value.",
    "num2t-rich-derivative-evaluated-elsewhere": "Evaluating the exact derivative at the wrong point when measuring the error.",
    "num2t-rich-fine-step-quartered": "Taking the finer step as h/4 instead of h/2.",
}
M_FINE = "num2t-rich-finer-only"
M_SWAP = "num2t-rich-weights-swapped"
M_AVG = "num2t-rich-average"
M_W2 = "num2t-rich-wrong-order-weights"
M_W1 = "num2t-rich-first-order-weights"
M_COARSE = "num2t-rich-coarse-only"
M_ONE = "num2t-rich-one-level-only"
M_W2L = "num2t-rich-second-level-weights"
M_ORDER = "num2t-rich-levels-in-wrong-order"
M_NODIV = "num2t-rich-no-division-by-step"
M_NO2H = "num2t-rich-no-division-by-double-step"
M_FWD = "num2t-rich-forward-for-central"
M_SIGN = "num2t-rich-error-sign"
M_ERR_COARSE = "num2t-rich-error-of-coarse"
M_ERR_PT = "num2t-rich-derivative-evaluated-elsewhere"
M_QUARTER = "num2t-rich-fine-step-quartered"

_H = {
    Fraction(2): "2",
    Fraction(1): "1",
    Fraction(1, 8): "0.125",
    Fraction(1, 10): "0.1",
    Fraction(1, 2): "0.5",
    Fraction(1, 4): "0.25",
    Fraction(1, 5): "0.2",
    Fraction(2, 5): "0.4",
}


def fwd(c, x, h):
    return (poly_eval(c, x + h) - poly_eval(c, x)) / h


def ctr(c, x, h):
    return (poly_eval(c, x + h) - poly_eval(c, x - h)) / (2 * h)


def deriv(c):
    return [k * c[k] for k in range(1, len(c))]


def extrap(coarse, fine, p):
    return (2**p * fine - coarse) / (2**p - 1)


def _poly(rng, degree):
    for _ in attempts(500, "a polynomial"):
        c = [rng.randint(-5, 5) for _ in range(degree + 1)]
        c[degree] = rng.choice([1, 2, 3, -1, -2])
        c[degree - 1] = rng.choice([-3, -2, -1, 1, 2, 3]) if degree > 3 else c[degree - 1]
        if sum(1 for v in c if v) >= 3 and c[1] != 0:
            return c
    raise AssertionError("unreachable")


def _draw(rng, degree):
    c = _poly(rng, degree)
    x0 = Fraction(rng.choice([1, 2, -1, 0, 1, 2]))
    h = rng.choice([Fraction(1), Fraction(1, 2), Fraction(2, 5), Fraction(1)])
    return c, x0, h


@generator(
    "r4h_richardson_extrapolation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_richardson_extrapolation(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "a Richardson question with enough distractors")


def _single(rng, tier):
    if tier == "hard":
        return _hard(rng)
    if tier == "easy":
        c, x0, h = _draw(rng, 3)
        coarse, fine = fwd(c, x0, h), fwd(c, x0, h / 2)
        value = extrap(coarse, fine, 1)
        if not margin_ok(value, 5):
            raise Redraw
        check(value == 2 * fine - coarse, "weights")
        cands = [
            (M_FINE, fine),
            (M_SWAP, 2 * coarse - fine),
            (M_AVG, (coarse + fine) / 2),
            (M_W2, (4 * fine - coarse) / 3),
            (M_COARSE, coarse),
        ]
        part = number_part("", value, 5, cands, 1, min_options=3)
        fine_h = _half(h)
        px = _x(x0)
        stem = rng.choice(
            [
                f"For «f(x) = {poly_prompt(c)}», the forward difference is «(f(x + h) − f(x))/h». Take «x = {px}», "
                f"evaluate the forward difference at «h = {_H[h]}» and «h = {fine_h}», and combine the two by "
                f"Richardson extrapolation to cancel the leading error term. Give the extrapolated estimate of «f′({px})» "
                "to 5 decimal places.",
                f"Estimate «f′({px})» for «f(x) = {poly_prompt(c)}» from the forward differences «(f(x + h) − f(x))/h» "
                f"with «h = {_H[h]}» and «h = {fine_h}», improved by one step of Richardson extrapolation. Give the "
                "improved estimate to 5 decimal places.",
                f"The function «f(x) = {poly_prompt(c)}» has forward difference quotients at «x = {px}» for step sizes "
                f"«{_H[h]}» and «{fine_h}». Combine them by Richardson extrapolation so that the leading error term "
                f"cancels, and give the resulting estimate of «f′({px})» to 5 decimal places.",
            ]
        )
        return single_from(
            part,
            stem,
            working=[
                f"D(h) = {tx(coarse)} and D(h/2) = {tx(fine)}.",
                f"The error is O(h), so the combination is 2D(h/2) − D(h) = {tx(value)}.",
            ],
        )
    c, x0, h = _draw(rng, 5)
    coarse, fine = ctr(c, x0, h), ctr(c, x0, h / 2)
    value = extrap(coarse, fine, 2)
    if not margin_ok(value, 5):
        raise Redraw
    cands = [
        (M_FINE, fine),
        (M_SWAP, (4 * coarse - fine) / 3),
        (M_AVG, (coarse + fine) / 2),
        (M_W1, 2 * fine - coarse),
        (M_COARSE, coarse),
    ]
    part = number_part("", value, 5, cands, 1, min_options=3)
    px = _x(x0)
    stem = rng.choice(
        [
            f"For «f(x) = {poly_prompt(c)}», the central difference is «(f(x + h) − f(x − h))/(2h)». Take «x = {px}», "
            f"evaluate it at «h = {_H[h]}» and at half that step, and combine the two by Richardson extrapolation to "
            f"cancel the leading error term. Give the extrapolated estimate of «f′({px})» to 5 decimal places.",
            f"Estimate «f′({px})» for «f(x) = {poly_prompt(c)}» using central differences «(f(x + h) − f(x − h))/(2h)» "
            f"with «h = {_H[h]}» and «h = {_half(h)}», improved by one step of Richardson extrapolation. Give the "
            "improved estimate to 5 decimal places.",
            f"Central difference quotients for «f(x) = {poly_prompt(c)}» at «x = {px}» are formed with step sizes "
            f"«{_H[h]}» and «{_half(h)}». Use Richardson extrapolation to cancel the leading error term and give the "
            f"resulting estimate of «f′({px})» to 5 decimal places.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"D(h) = {tx(coarse)} and D(h/2) = {tx(fine)}.",
            f"The error is O(h²), so the combination is (4D(h/2) − D(h))/3 = {tx(value)}.",
        ],
    )


def _x(v: Fraction) -> str:
    return (str(v.numerator) if v.denominator == 1 else str(float(v))).replace("-", "−")


def _half(h: Fraction) -> str:
    return _H.get(h / 2, str(float(h / 2)))


def _hard(rng):
    for _ in attempts(500, "a trapezoid table"):
        h = rng.choice([Fraction(1), Fraction(1, 2)])
        exact = Fraction(rng.randint(100, 900), 100)
        c2, c4, c6 = (Fraction(rng.randint(-20, 20), 10) for _ in range(3))
        if c2 == 0 or c4 == 0 or c6 == 0:
            continue
        t = [exact + c2 * (h / 2**k) ** 2 + c4 * (h / 2**k) ** 4 + c6 * (h / 2**k) ** 6 for k in range(3)]
        t = [Fraction(round(v * 10**4), 10**4) for v in t]
        r1a = extrap(t[0], t[1], 2)
        r1b = extrap(t[1], t[2], 2)
        value = (16 * r1b - r1a) / 15
        if margin_ok(value, 5, 0.15):
            break
    cands = [
        (M_ONE, r1b),
        (M_W2L, (4 * r1b - r1a) / 3),
        (M_ORDER, (16 * t[2] - t[0]) / 15),
        (M_AVG, (r1a + r1b) / 2),
        (M_COARSE, t[2]),
    ]
    part = number_part("", value, 5, cands, 1, min_options=3)
    hs = [_H.get(h / 2**k, str(float(h / 2**k))) for k in range(3)]
    t_text = [f"{float(v):.4f}" for v in t]
    stem = rng.choice(
        [
            f"The trapezoid rule applied to an integral gives «T(h) = {t_text[0]}» with step «h = {hs[0]}», "
            f"«T(h/2) = {t_text[1]}» and «T(h/4) = {t_text[2]}». The error of each is a series in even powers, "
            "«c₂h² + c₄h⁴ + c₆h⁶ + …». Use Richardson extrapolation twice, removing the «h²» term and then the «h⁴» "
            "term, to estimate the integral to 5 decimal places.",
            f"Three trapezoid estimates of one integral, with steps «{hs[0]}», «{hs[1]}» and «{hs[2]}», are "
            f"«{t_text[0]}», «{t_text[1]}» and «{t_text[2]}». The trapezoid error has only even powers of the step. "
            "Extrapolate twice to remove the «h²» and «h⁴» terms, and give the integral to 5 decimal places.",
            f"A trapezoid-rule table for an integral reads «{t_text[0]}», «{t_text[1]}», «{t_text[2]}» at steps "
            f"«{hs[0]}», «{hs[1]}», «{hs[2]}». Given that the error expands in powers «h², h⁴, h⁶», apply Richardson "
            "extrapolation at both levels and report the best estimate to 5 decimal places.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"First level: (4T(h/2) − T(h))/3 = {dec(r1a, 5)} and (4T(h/4) − T(h/2))/3 = {dec(r1b, 5)}.",
            f"Second level removes h⁴: (16·{dec(r1b, 5)} − {dec(r1a, 5)})/15 = {dec(value, 5)}.",
        ],
    )


@generator("exd_x_r4h_richardson_extrapolation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_richardson_extrapolation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Richardson question")


def _exam(rng):
    c, x0, h = _draw(rng, 5)
    exact = poly_eval(deriv(c), x0)
    d1, d2 = ctr(c, x0, h), ctr(c, x0, h / 2)
    value = extrap(d1, d2, 2)
    err = value - exact
    if err == 0 or not all(margin_ok(v, 5) for v in (d1, d2, value, err)):
        raise Redraw
    fx = lambda s: poly_eval(c, s)  # noqa: E731
    a_cands = [
        (M_NO2H, (fx(x0 + h) - fx(x0 - h)) / h),
        (M_FWD, (fx(x0 + h) - fx(x0)) / h),
        (M_NODIV, fx(x0 + h) - fx(x0 - h)),
    ]
    p_a = number_part(
        f"Evaluate the central difference «D(h)» at «h = {_H[h]}».",
        d1,
        5,
        a_cands,
        1,
        working=[f"D(h) = (f({_x(x0 + h)}) − f({_x(x0 - h)}))/{tx(2 * h)} = {tx(d1)}."],
        value_range=(-(10**5), 10**5),
    )
    hh = h / 2
    b_cands = [
        (M_NO2H, (fx(x0 + hh) - fx(x0 - hh)) / hh),
        (M_FWD, (fx(x0 + hh) - fx(x0)) / hh),
        (M_QUARTER, ctr(c, x0, h / 4)),
        (M_NODIV, fx(x0 + hh) - fx(x0 - hh)),
    ]
    p_b = number_part(
        f"Evaluate «D(h/2)», the same central difference at half the step.",
        d2,
        5,
        b_cands,
        1,
        working=[f"D(h/2) = {tx(d2)}."],
        value_range=(-(10**5), 10**5),
    )
    a, b = ft_symbol("part_a"), ft_symbol("part_b")
    p_c = number_part(
        "Combine your two estimates by Richardson extrapolation to cancel the «h²» term.",
        value,
        5,
        [(M_FINE, d2), (M_W1, 2 * d2 - d1), (M_SWAP, (4 * d1 - d2) / 3), (M_AVG, (d1 + d2) / 2)],
        2,
        ft=str((4 * b - a) / 3),
        working=[f"(4·{pn(d2)} − {pn(d1)})/3 = {tx(value)}."],
        value_range=(-(10**5), 10**5),
    )
    k = rat(exact)
    cc = ft_symbol("part_c")
    p_d = number_part(
        f"Find the error of your extrapolated value as an estimate of «f′({_x(x0)})», the estimate minus the exact "
        "value.",
        err,
        5,
        [
            (M_SIGN, -err),
            (M_ERR_COARSE, d1 - exact),
            (M_ERR_PT, value - poly_eval(deriv(c), x0 + h)),
        ],
        2,
        ft=str(cc - k),
        working=[f"f′({_x(x0)}) = {tx(exact)}, so the error is {tx(value)} − {pn(exact)} = {tx(err)}."],
        value_range=(-(10**5), 10**5),
    )
    px = _x(x0)
    stem = rng.choice(
        [
            f"For «f(x) = {poly_prompt(c)}», the central difference «D(h) = (f(x + h) − f(x − h))/(2h)» is used at "
            f"«x = {px}» with «h = {_H[h]}» to estimate «f′({px})».",
            f"The derivative of «f(x) = {poly_prompt(c)}» at «x = {px}» is estimated by the central difference "
            f"«D(h) = (f(x + h) − f(x − h))/(2h)» with «h = {_H[h]}», then improved by extrapolation.",
            f"Let «D(h) = (f(x + h) − f(x − h))/(2h)» for «f(x) = {poly_prompt(c)}». The estimate of «f′({px})» "
            f"starts from «h = {_H[h]}».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)
