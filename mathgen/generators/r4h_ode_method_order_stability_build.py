"""Order and stability of a one-step method for ``y' = −λy``: `r4h_ode_method_order_stability` (easy, medium, hard)
and the exam chain `exd_x_r4h_ode_method_order_stability`.

``uc2-ode-method-order-and-stability`` (``numerical-and-computational-mathematics``). A one-step method applied to
``y' = −λy`` gives ``yₙ₊₁ = R(hλ) yₙ``. The method has order p when ``R(z) − e^{−z} = O(z^(p+1))`` and is stable
for a step size when ``|R(hλ)| ≤ 1``.

* easy: explicit Euler on ``y' = Ay`` with a triangular 2×2 matrix; the largest step for which the iteration stays
  bounded, ``2/|λ|max`` (a fraction).
* medium: the θ-method ``yₙ₊₁ = yₙ + h[(1−θ)f(yₙ) + θ f(yₙ₊₁)]`` with θ below one half; the largest stable step,
  ``2/(λ(1−2θ))`` (a fraction).
* hard: the order of a method given only its amplification factor ``R(z)`` (a whole number).
* exam: a named method with a stated ``λ`` and ``h``: a classification (multiple choice, one of five order-and-stability
  statements), ``R(hλ)``, ``y₃`` (follow-through from ``R``) and the error constant of ``R``.

Everything is exact (`Fraction`). Orders come from the Taylor coefficients of ``R`` against those of ``e^{−z}``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    finish_parts,
    fr,
    fraction_part,
    integer_part,
    mat,
    poly_prompt,
    shown,
    single_from,
    with_redraw,
)

MISC = {
    "num2t-ode-stability-halved-bound": "Using the bound 1/λ for the explicit Euler step instead of 2/λ, requiring the factor to stay positive rather than within [−1, 1].",
    "num2t-ode-stability-smallest-eigenvalue": "Using the eigenvalue of smallest size instead of the largest when finding the step limit for a system.",
    "num2t-ode-stability-trace": "Using the trace of the matrix in place of its eigenvalue of largest size.",
    "num2t-ode-stability-product": "Using the product of the eigenvalues in place of the largest one.",
    "num2t-ode-stability-explicit-formula": "Using the explicit Euler limit 2/λ for a method that is partly implicit.",
    "num2t-ode-stability-one-over": "Dropping the 2 in the numerator, giving 1/(λ(1 − 2θ)).",
    "num2t-ode-stability-theta-not-doubled": "Using 1 − θ instead of 1 − 2θ in the stability limit.",
    "num2t-ode-stability-theta-sign": "Using 1 + 2θ instead of 1 − 2θ in the stability limit.",
    "num2t-ode-order-one-too-high": "Reporting the exponent of the first error term (order plus one) as the order.",
    "num2t-ode-order-one-too-low": "Reporting the number of the highest matching power of e^(−z) less one.",
    "num2t-ode-order-polynomial-degree": "Taking the degree of the amplification polynomial as the order.",
    "num2t-ode-order-always-one": "Calling every method that is not exact first order.",
    "num2t-ode-order-term-count": "Counting the matching terms of the series, constant term included, as the order.",
    "num2t-ode-order-misjudged": "Judging the order of a method from its number of stages or its look rather than from its amplification factor.",
    "num2t-ode-stability-misjudged": "Judging stability from the order of the method rather than from the size of its amplification factor.",
    "num2t-ode-r-explicit-formula": "Using the explicit amplification factor 1 − z for a method that is partly implicit.",
    "num2t-ode-r-backward-formula": "Using the backward Euler factor 1/(1 + z) for a method that is not fully implicit.",
    "num2t-ode-r-theta-swapped": "Swapping θ and 1 − θ in the amplification factor.",
    "num2t-ode-r-heun-no-half": "Dropping the factor 1/2 on the quadratic term of the Heun amplification factor.",
    "num2t-ode-r-heun-sign": "Taking the Heun amplification factor as 1 + z + z²/2.",
    "num2t-ode-steps-one-too-few": "Applying the amplification factor one time too few.",
    "num2t-ode-steps-one-too-many": "Applying the amplification factor one time too many.",
    "num2t-ode-steps-multiplied": "Multiplying the amplification factor by the number of steps instead of raising it to that power.",
    "num2t-ode-const-sign": "Giving the error constant with the wrong sign (R − e^(−z) instead of e^(−z) − R).",
    "num2t-ode-const-wrong-power": "Reading the mismatch in the coefficient of a later power of z as the error constant.",
    "num2t-ode-const-factorial-only": "Giving the exponential's coefficient 1/(p+1)! as the error constant, without subtracting R's own coefficient.",
}
M_HALF = "num2t-ode-stability-halved-bound"
M_MIN = "num2t-ode-stability-smallest-eigenvalue"
M_TRACE = "num2t-ode-stability-trace"
M_PROD = "num2t-ode-stability-product"
M_EXPL = "num2t-ode-stability-explicit-formula"
M_ONE = "num2t-ode-stability-one-over"
M_THETA = "num2t-ode-stability-theta-not-doubled"
M_PLUS = "num2t-ode-stability-theta-sign"
M_O_HI = "num2t-ode-order-one-too-high"
M_O_LO = "num2t-ode-order-one-too-low"
M_O_DEG = "num2t-ode-order-polynomial-degree"
M_O_ONE = "num2t-ode-order-always-one"
M_O_TERMS = "num2t-ode-order-term-count"
M_ORD = "num2t-ode-order-misjudged"
M_STAB = "num2t-ode-stability-misjudged"


# ---------------------------------------------------------------------------------------------------------------
# series


def series(num, den, n: int):
    """The first n + 1 Taylor coefficients of num(z)/den(z), each a list of coefficients (lowest power first),
    den[0] != 0."""
    out = []
    for k in range(n + 1):
        nk = fr(num[k]) if k < len(num) else Fraction(0)
        acc = nk - sum((fr(den[j]) * out[k - j] for j in range(1, min(k, len(den) - 1) + 1)), Fraction(0))
        out.append(acc / fr(den[0]))
    return out


def exp_coeff(k: int) -> Fraction:
    return Fraction((-1) ** k, math.factorial(k))


def order_of(num, den) -> int:
    """The order of the method with amplification factor num/den: the largest p with matching coefficients through
    z^p."""
    coeffs = series(num, den, 8)
    p = 0
    while p < 8 and coeffs[p + 1] == exp_coeff(p + 1):
        p += 1
    check(coeffs[0] == 1, "R(0) is not 1")
    return p


def error_constant(num, den):
    """(p, C) with e^{-z} - R(z) = C z^(p+1) + ..."""
    p = order_of(num, den)
    c = series(num, den, p + 1)[p + 1]
    return p, exp_coeff(p + 1) - c


def pade(m: int, k: int):
    """The (m, k) Pade approximant of e^{-z}: numerator and denominator coefficients, lowest power first."""
    nm = [
        Fraction(
            math.factorial(m + k - j) * math.factorial(m),
            math.factorial(m + k) * math.factorial(j) * math.factorial(m - j),
        )
        * (-1) ** j
        for j in range(m + 1)
    ]
    dn = [
        Fraction(
            math.factorial(m + k - j) * math.factorial(k),
            math.factorial(m + k) * math.factorial(j) * math.factorial(k - j),
        )
        for j in range(k + 1)
    ]
    return nm, dn


def _eval(coeffs, z) -> Fraction:
    out = Fraction(0)
    for c in reversed(coeffs):
        out = out * z + fr(c)
    return out


def _fn_text(coeffs) -> str:
    return poly_prompt(coeffs).replace("x", "z")


def r_text(num, den) -> str:
    if len(den) == 1 and den[0] == 1:
        return _fn_text(num)
    return f"({_fn_text(num)})/({_fn_text(den)})"


# ---------------------------------------------------------------------------------------------------------------
# methods


def theta_method(theta: Fraction):
    return [Fraction(1), -(1 - theta)], [Fraction(1), theta]


HEUN = ([Fraction(1), Fraction(-1), Fraction(1, 2)], [Fraction(1)])


def _frac_text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# ---------------------------------------------------------------------------------------------------------------
# practice


@generator(
    "r4h_ode_method_order_stability",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_ode_method_order_stability(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "an order-and-stability question")


def _single(rng, tier):
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


def _easy(rng):
    a, b = rng.sample([1, 2, 3, 4, 5, 6, 8, 10], 2)
    c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    top, low = max(a, b), min(a, b)
    value = Fraction(2, top)
    swap = rng.random() < 0.5
    rows = [[-a, c], [0, -b]] if not swap else [[-a, 0], [c, -b]]
    cands = [
        (M_MIN, Fraction(2, low)),
        (M_HALF, Fraction(1, top)),
        (M_TRACE, Fraction(2, a + b)),
        (M_PROD, Fraction(2, a * b)),
        (M_HALF, Fraction(1, low)),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    am = shown(mat(rows))
    stem = rng.choice(
        [
            f"The system «y′ = Ay» with «A = {am}» is solved by the explicit Euler method «yₙ₊₁ = yₙ + h·A·yₙ». Find "
            "the largest step size «h» for which the iterates stay bounded, whatever the starting vector. Give it as "
            "a fraction.",
            f"Explicit Euler, «yₙ₊₁ = yₙ + h·A·yₙ», is applied to «y′ = Ay» where «A = {am}». For what largest "
            "step size «h» do the iterates stay bounded for every starting vector? Answer as a fraction.",
            f"For the linear system with «A = {am}», explicit Euler steps «yₙ₊₁ = yₙ + h·A·yₙ» are taken. Determine the "
            "greatest «h» that keeps every solution of the iteration bounded, as a fraction.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"A is triangular, so its eigenvalues are its diagonal entries −{a} and −{b}.",
            "Euler multiplies each mode by 1 − hλ, which stays within [−1, 1] when h|λ| ≤ 2.",
            f"The binding eigenvalue has size {top}, so h ≤ 2/{top} = {_frac_text(value)}.",
        ],
    )


def _medium(rng):
    theta = rng.choice([Fraction(1, 4), Fraction(1, 3), Fraction(1, 10), Fraction(1, 5), Fraction(3, 8)])
    lam = rng.choice([2, 3, 4, 5, 6, 8, 10])
    value = Fraction(2) / (lam * (1 - 2 * theta))
    cands = [
        (M_EXPL, Fraction(2, lam)),
        (M_ONE, 1 / (lam * (1 - 2 * theta))),
        (M_THETA, Fraction(2) / (lam * (1 - theta))),
        (M_PLUS, Fraction(2) / (lam * (1 + 2 * theta))),
        (M_HALF, Fraction(1, lam)),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    th = _frac_text(theta)
    stem = rng.choice(
        [
            f"The equation «y′ = −{lam}y» is solved by the method «yₙ₊₁ = yₙ + h[(1 − θ)f(yₙ) + θ f(yₙ₊₁)]» with "
            f"«θ = {th}», where «f(y) = −{lam}y». Find the largest step size «h» for which «|yₙ₊₁| ≤ |yₙ|» for "
            "every starting value. Give it as a fraction.",
            f"The θ-method «yₙ₊₁ = yₙ + h[(1 − θ)f(yₙ) + θ f(yₙ₊₁)]» with «θ = {th}» is applied to «y′ = −{lam}y». "
            "What is the largest step size «h» for which the numerical solution never grows from one step to the "
            "next? Answer as a fraction.",
            f"With «θ = {th}», the method «yₙ₊₁ = yₙ + h[(1 − θ)f(yₙ) + θ f(yₙ₊₁)]» solves «y′ = −{lam}y». Find, as a "
            "fraction, the greatest «h» that guarantees «|yₙ₊₁| ≤ |yₙ|» whatever «y₀» is.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"yₙ₊₁ = R yₙ with R = (1 − (1 − θ)z)/(1 + θz) and z = {lam}h.",
            "|R| ≤ 1 for z > 0 reduces to (1 − 2θ)z ≤ 2.",
            f"So h ≤ 2/({lam}(1 − 2θ)) = {_frac_text(value)}.",
        ],
    )


def _hard_factor(rng):
    """(numerator, denominator) with a computed order between 1 and 4, and a spread of kinds."""
    kind = rng.choice(["poly", "poly", "pade", "pade", "theta"])
    if kind == "theta":
        theta = rng.choice(
            [Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(3, 4), Fraction(1, 4), Fraction(1, 2)]
        )
        return theta_method(theta)
    if kind == "poly":
        p = rng.choice([1, 2, 2, 3, 3, 4])
        num = [exp_coeff(k) for k in range(p + 1)]
        delta = rng.choice(
            [Fraction(1, 2), Fraction(1, 3), Fraction(-1, 2), Fraction(1, 4), Fraction(-1, 3), Fraction(1, 6)]
        )
        num.append(exp_coeff(p + 1) + delta / math.factorial(p + 1))
        if rng.random() < 0.4:
            num.append(Fraction(rng.choice([-1, 1]), rng.choice([12, 24, 30, 60])))
        return num, [Fraction(1)]
    m, k = rng.choice([(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 2)])
    num, den = pade(m, k)
    if rng.random() < 0.55:
        num = list(num)
        num[-1] = num[-1] * rng.choice([Fraction(3, 2), Fraction(1, 2), Fraction(-1), Fraction(2)])
        if num[-1] == 0:
            raise Redraw
    return num, den


def _hard(rng):
    num, den = _hard_factor(rng)
    p = order_of(num, den)
    if not 1 <= p <= 4:
        raise Redraw
    degree = len(num) - 1
    cands = [
        (M_O_HI, p + 1),
        (M_O_LO, p - 1),
        (M_O_DEG, degree if len(den) == 1 else degree + len(den) - 1),
        (M_O_TERMS, p + 2),
        (M_O_ONE, 1),
        (M_O_HI, p + 2 if p == 1 else 2 * p),
    ]
    part = integer_part("", p, cands, 1, n=4, min_options=3)
    rt = r_text(num, den)
    stem = rng.choice(
        [
            f"A one-step method applied to «y′ = −λy» gives «yₙ₊₁ = R(z)yₙ» with «z = hλ» and «R(z) = {rt}». Find the "
            "order of the method: the largest «p» for which «R(z) − e^(−z)» is «O(z^(p+1))».",
            f"The amplification factor of a one-step method for «y′ = −λy» is «R(z) = {rt}», where «z = hλ». What is "
            "the order of accuracy of the method?",
            f"Applying a method to the test equation «y′ = −λy» gives «yₙ₊₁ = R(hλ)yₙ» with «R(z) = {rt}». Compare "
            "«R(z)» with «e^(−z)» and state the order of the method.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            "Expand R(z) in powers of z and compare with e^(−z) = 1 − z + z²/2 − z³/6 + ….",
            f"The coefficients agree through z^{p} and first differ at z^{p + 1}, so the order is {p}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain

LABELS = [
    "First order, stable only for small enough steps",
    "First order, stable for every step size",
    "Second order, stable only for small enough steps",
    "Second order, stable for every step size",
    "First order, unstable for every step size",
]


def _method(rng):
    """(description for the stem, num, den, class index, per-method candidate list builder)."""
    which = rng.choice(["euler", "theta-low", "heun", "trap", "backward", "theta-high"])
    if which == "euler":
        return which, "yₙ₊₁ = yₙ + h f(yₙ)", theta_method(Fraction(0)), 0
    if which == "theta-low":
        th = rng.choice([Fraction(1, 4), Fraction(1, 3)])
        return which, f"yₙ₊₁ = yₙ + h[(1 − θ)f(yₙ) + θ f(yₙ₊₁)], θ = {_frac_text(th)}", theta_method(th), 0
    if which == "heun":
        return which, "yₙ₊₁ = yₙ + (h/2)[f(yₙ) + f(yₙ + h f(yₙ))]", HEUN, 2
    if which == "trap":
        return which, "yₙ₊₁ = yₙ + (h/2)[f(yₙ) + f(yₙ₊₁)]", theta_method(Fraction(1, 2)), 3
    if which == "backward":
        return which, "yₙ₊₁ = yₙ + h f(yₙ₊₁)", theta_method(Fraction(1)), 1
    th = rng.choice([Fraction(3, 4), Fraction(2, 3)])
    return which, f"yₙ₊₁ = yₙ + h[(1 − θ)f(yₙ) + θ f(yₙ₊₁)], θ = {_frac_text(th)}", theta_method(th), 1


@generator("exd_x_r4h_ode_method_order_stability", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_ode_method_order_stability(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam order-and-stability question")


def _exam(rng):
    which, scheme, (num, den), cls = _method(rng)
    lam = rng.choice([2, 3, 4, 5, 6, 8, 10])
    h = rng.choice([Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 2), Fraction(1, 8)])
    z = lam * h
    y0 = rng.choice([1, 2, 3, 4, 5, 8])
    r = _eval(num, z) / _eval(den, z)
    if r in (0, 1, -1) or abs(r) > 6:
        raise Redraw
    steps = 3
    y3 = y0 * r**steps
    p, const = error_constant(num, den)
    check(p == (1 if cls in (0, 1) else 2), "order and class disagree")
    if const == 0:
        raise Redraw
    key = LABELS[cls]
    wrong = []
    for i, label in enumerate(LABELS):
        if i == cls:
            continue
        right_order = (i in (0, 1, 4)) == (cls in (0, 1))
        wrong.append(Distractor(label, M_ORD if not right_order else M_STAB))
    p_a = dict(
        prompt="Which statement describes the method, for the model problem «y′ = −λy» with «λ > 0»?",
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        marks=2,
        working=[f"R(z) agrees with e^(−z) through z^{p}, so the order is {p}; the stability follows from |R(z)| ≤ 1."],
        distractors=wrong,
    )
    zt = _frac_text(z)
    p_b = fraction_part(
        f"Applied to «y′ = −{lam}y» with «h = {_frac_text(h)}», the method gives «yₙ₊₁ = R yₙ». Find the multiplier «R».",
        r,
        _r_cands(which, z, num, den),
        1,
        working=[f"z = hλ = {zt}, so R = {_frac_text(r)}."],
        value_range=(-6, 6),
    )
    p_c = fraction_part(
        f"With «y₀ = {y0}», use your «R» to find «y₃».",
        y3,
        [
            ("num2t-ode-steps-one-too-few", y0 * r**2),
            ("num2t-ode-steps-one-too-many", y0 * r**4),
            ("num2t-ode-steps-multiplied", y0 * 3 * r),
            ("num2t-ode-steps-one-too-few", y0 * r),
        ],
        2,
        ft=f"{y0}*part_b**3",
        working=[f"y₃ = {y0}·({_frac_text(r)})³ = {_frac_text(y3)}."],
        value_range=(-(10**4), 10**4),
    )
    p_d = fraction_part(
        "Write «e^(−z) − R(z) = C z^q + …» with «C ≠ 0» and «q» the power of the first term that does not cancel. Find «C».",
        const,
        [
            ("num2t-ode-const-sign", -const),
            ("num2t-ode-const-wrong-power", exp_coeff(p + 2) - series(num, den, p + 2)[p + 2]),
            ("num2t-ode-const-factorial-only", exp_coeff(p + 1)),
            ("num2t-ode-const-sign", const * (p + 1) if p >= 1 else None),
        ],
        2,
        working=[f"Compare the coefficients of z^{p + 1}: C = {_frac_text(const)}."],
        value_range=(-2, 2),
    )
    stem = rng.choice(
        [
            f"A one-step method is «{scheme}». It is applied to the test problem «y′ = −λy» with «λ > 0», where "
            "«f(y) = −λy».",
            f"Consider the scheme «{scheme}» for «y′ = f(y)». It is used on the test equation «y′ = −λy», with «λ > 0».",
            f"The method «{scheme}» is applied to «y′ = −λy» (so «f(y) = −λy»), where «λ > 0».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)


def _r_cands(which, z, num, den):
    cands = [("num2t-ode-r-explicit-formula", 1 - z), ("num2t-ode-r-backward-formula", 1 / (1 + z))]
    if which == "heun":
        cands = [
            ("num2t-ode-r-explicit-formula", 1 - z),
            ("num2t-ode-r-heun-no-half", 1 - z + z * z),
            ("num2t-ode-r-heun-sign", 1 + z + z * z / 2),
            ("num2t-ode-r-backward-formula", 1 / (1 + z)),
        ]
    elif which in ("theta-low", "theta-high", "trap"):
        theta = den[1]  # den = 1 + theta z
        swapped = (1 - theta * z) / (1 + (1 - theta) * z)
        cands.append(("num2t-ode-r-theta-swapped", swapped))
        cands.append(("num2t-ode-r-heun-sign", (1 + (1 - theta) * z) / (1 + theta * z)))
    return cands
