"""Generates `ucv1_trig_substitution` questions and the exam chain `exd_x_fy_trig_substitution`.

``ucv1-trig-substitution-integrals`` (``integration``, answer ``number``): definite integrals that a trigonometric
substitution (x = a sin θ, x = a tan θ or x = a sec θ) makes elementary. Every answer is a decimal correct to 3 places.

* easy: a standard form, ∫ dx/√(a² - x²) from 0 to a·s, or ∫ dx/(a² + x²) from 0 to a·t: the limits have to be
  converted to angles and the constant 1/a kept.
* medium: ∫ √(a² - x²) dx from 0 to a·s (needs sin²-cos² after the substitution, then a double angle), or
  ∫ dx/(a² + x²)^(3/2) (needs sec³ against sec² to cancel).
* hard: ∫ dx/(x√(x² - a²)) between two limits (the secant substitution), ∫ x²/√(a² - x²) dx from 0 to a·s, or a
  quadratic denominator x² - 2hx + h² + a² that needs completing the square before the substitution.
* exam (``exd_x_fy_trig_substitution``): ∫ x²/√(a² - x²) dx from 0 to a·s. (a) the upper limit in θ as a multiple of π,
  (b) the constant k in k∫ sin²θ dθ, (c) the value of ∫ sin²θ dθ over the new limits (from (a)), (d) the integral
  (from (b) and (c)).

Every value is re-derived by numerical quadrature (composite Simpson) of the integrand, which shares nothing with the closed form.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import exact_value
from ..core import attempts, check, generator, pick_distractors, rational_text
from ._fy1cd_kit import F, decimal_text, fraction_dist, int_dist, pick_decimal, say, surd

_S2, _S3 = math.sqrt(2), math.sqrt(3)

# s = sin of the upper angle: (value, coefficient of u in a, root, angle as a multiple of pi)
_SINES = [
    (0.5, (Fraction(1, 2), 1), Fraction(1, 6)),
    (_S2 / 2, (Fraction(1, 2), 2), Fraction(1, 4)),
    (_S3 / 2, (Fraction(1, 2), 3), Fraction(1, 3)),
]


class _Retry(Exception):
    """A draw whose values leave too few distinct wrong answers."""


def _need(ds) -> None:
    if len(ds) < 2:
        raise _Retry


def _retrying(build, rng):
    for _ in attempts(100, "a draw with enough distinct wrong answers"):
        try:
            return build(rng)
        except _Retry:
            continue


def _numeric(fn, lo, hi, panels: int = 800) -> float:
    """Composite Simpson's rule: a numerical value of the integral that shares nothing with the closed form."""
    lo, hi = float(lo), float(hi)
    h = (hi - lo) / panels
    total = fn(lo) + fn(hi)
    for i in range(1, panels):
        total += fn(lo + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def _verify(value, fn, lo, hi, what: str = "the integral") -> None:
    """The closed form against numerical quadrature of the integrand."""
    got = _numeric(fn, lo, hi)
    check(abs(got - float(value)) < 1e-7, f"{what}: quadrature {got}, closed form {float(value)}")


def _limit(a, coef, root) -> tuple[float, str, str]:
    """a times coef times the square root of `root` as a value, plain text and TeX."""
    c = Fraction(a) * coef
    plain, tex = surd(c, root) if root != 1 else surd(c, 1)
    return float(c) * math.sqrt(root), plain, tex


def _integral(integrand: F, lo: tuple[str, str], hi: tuple[str, str]) -> F:
    return F(
        f"∫ {integrand[0]} dx from x = {lo[0]} to x = {hi[0]}",
        rf"\int_{{{lo[1]}}}^{{{hi[1]}}}{integrand[1]}\,dx",
    )


def _stem(integral: F, places: int = 3):
    return say("Evaluate ", integral, f", correct to {places} decimal places.")


def _result(stem, value, ds, working, places: int = 3):
    number, text = decimal_text(value, places)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=number,
        answer_text=text,
        answer_latex=text,
        working=working,
        distractors=ds,
    )


def _k(a: int) -> str:
    """The coefficient a as a prefix: nothing for 1, so '1 sin θ' is written 'sin θ'."""
    return "" if a == 1 else f"{a} "


def _sq(a: int) -> str:
    return str(a * a)


# -- easy -------------------------------------------------------------------------------------------------------


def _easy_arcsin(rng):
    a = rng.choice([1, 2, 3, 4, 5, 6])
    s, (coef, root), p = rng.choice(_SINES)
    u, uplain, utex = _limit(a, coef, root)
    value = math.asin(s)
    check(abs(value - math.pi * float(p)) < 1e-12, "arcsin differs from the stated angle")
    _verify(value, lambda x: 1 / math.sqrt(a * a - x * x), 0, u)
    integrand = F(f"1/√({_sq(a)} - x²)", rf"\frac{{1}}{{\sqrt{{{_sq(a)}-x^{{2}}}}}}")
    stem = _stem(_integral(integrand, ("0", "0"), (uplain, utex)))
    cands = [
        (math.acos(s), "trigsub-inverse-swapped"),
        (u, "trigsub-limits-not-converted"),
        (math.asin(s) / a, "trigsub-spurious-factor"),
        (math.atan(s), "trigsub-wrong-inverse"),
        (math.asin(s) * a, "trigsub-spurious-factor"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {_k(a)}sin θ. Then dx = {_k(a)}cos θ dθ and √({_sq(a)} - x²) = {_k(a)}cos θ, so the integrand becomes 1 dθ.",
            f"The limits x = 0 and x = {uplain} become θ = 0 and θ = {rational_text(p)}π.",
            f"The value is {rational_text(p)}π = {decimal_text(value, 3)[1]}.",
        ],
    )


def _easy_arctan(rng):
    a = rng.choice([2, 3, 4, 5])
    t, tplain, ttex = rng.choice([(1, "1", "1"), (_S3, "√3", r"\sqrt{3}")])
    u = a * t
    uplain = str(a) if t == 1 else f"{a}√3"
    utex = str(a) if t == 1 else rf"{a}\sqrt{{3}}"
    value = math.atan(t) / a
    _verify(value, lambda x: 1 / (a * a + x * x), 0, u)
    integrand = F(f"1/({_sq(a)} + x²)", rf"\frac{{1}}{{{_sq(a)}+x^{{2}}}}")
    stem = _stem(_integral(integrand, ("0", "0"), (uplain, utex)))
    cands = [
        (math.atan(t), "trigsub-spurious-factor"),
        (math.atan(u) / a, "trigsub-limits-not-converted"),
        (math.atan(t) / (a * a), "trigsub-spurious-factor"),
        (math.atan(t) * a, "trigsub-spurious-factor"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {a} tan θ. Then dx = {a} sec² θ dθ and {_sq(a)} + x² = {_sq(a)} sec² θ, so the integrand becomes (1/{a}) dθ.",
            f"The limits become θ = 0 and θ = arctan({'1' if t == 1 else '√3'}) = {'π/4' if t == 1 else 'π/3'}.",
            f"The value is (1/{a})({'π/4' if t == 1 else 'π/3'}) = {decimal_text(value, 3)[1]}.",
        ],
    )


# -- medium -----------------------------------------------------------------------------------------------------


def _medium_sqrt(rng):
    a = rng.choice([1, 2, 3, 4, 5, 6])
    s, (coef, root), p = rng.choice(_SINES)
    u, uplain, utex = _limit(a, coef, root)
    theta = math.asin(s)
    c = math.sqrt(1 - s * s)
    value = Fraction(a * a, 2) * (theta + s * c)
    _verify(value, lambda x: math.sqrt(a * a - x * x), 0, u)
    integrand = F(f"√({_sq(a)} - x²)", rf"\sqrt{{{_sq(a)}-x^{{2}}}}")
    stem = _stem(_integral(integrand, ("0", "0"), (uplain, utex)))
    h = Fraction(a * a, 2)
    cands = [
        (h * theta, "trigsub-double-angle-term-dropped"),
        (h * (theta - s * c), "trigsub-double-angle-sign"),
        (2 * h * (theta + s * c), "trigsub-half-dropped"),
        (Fraction(a, 2) * (theta + s * c), "trigsub-constant-not-squared"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {_k(a)}sin θ. The integrand becomes {_sq(a)} cos² θ dθ = {Fraction(a * a, 2)}(1 + cos 2θ) dθ.",
            f"Integrating gives {Fraction(a * a, 2)}(θ + sin θ cos θ) from 0 to {rational_text(p)}π.",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


def _medium_power(rng):
    a = rng.choice([2, 3, 4, 5])
    t, tplain, ttex = rng.choice([(1, "1", "1"), (_S3, "√3", r"\sqrt{3}")])
    u = a * t
    uplain = str(a) if t == 1 else ("√3" if a == 1 else f"{a}√3")
    utex = str(a) if t == 1 else (r"\sqrt{3}" if a == 1 else rf"{a}\sqrt{{3}}")
    value = u / (a * a * math.sqrt(a * a + u * u))
    _verify(value, lambda x: 1 / (a * a + x * x) ** 1.5, 0, u)
    base = f"{_sq(a)} + x²"
    integrand = F(f"1/({base})^(3/2)", rf"\frac{{1}}{{\left({_sq(a)}+x^{{2}}\right)^{{3/2}}}}")
    stem = _stem(_integral(integrand, ("0", "0"), (uplain, utex)))
    theta = math.atan(t)
    cands = [
        (theta / a**3, "trigsub-power-ignored"),
        (u / (a * a * (a * a + u * u)), "trigsub-root-dropped"),
        (u / (a * math.sqrt(a * a + u * u)), "trigsub-wrong-power-of-a"),
        (math.cos(theta) / (a * a), "trigsub-sin-cos-swapped"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {a} tan θ. Then ({base})^(3/2) = {a}³ sec³ θ and dx = {a} sec² θ dθ, so the integrand becomes (cos θ/{a * a}) dθ.",
            f"Integrating gives (sin θ)/{a * a} from 0 to arctan({'1' if t == 1 else '√3'}).",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


# -- hard -------------------------------------------------------------------------------------------------------

# (lo/a, hi/a, arcsec lo, arcsec hi, needs a divisible by 3)
_SEC_PAIRS = [
    (("√2", _S2), ("2", 2), math.pi / 4, math.pi / 3, False),
    (("2/√3", 2 / _S3), ("√2", _S2), math.pi / 6, math.pi / 4, True),
    (("2/√3", 2 / _S3), ("2", 2), math.pi / 6, math.pi / 3, True),
]


def _limit_text(name: str, a: int) -> tuple[str, str]:
    if name == "2/√3":
        k = a // 3
        return surd(2 * k, 3)
    if name == "√2":
        return surd(a, 2)
    return str(2 * a), str(2 * a)


def _hard_secant(rng):
    lo, hi, alo, ahi, three = rng.choice(_SEC_PAIRS)
    a = rng.choice([3, 6] if three else [1, 2, 3, 4])
    lo_v, hi_v = a * lo[1], a * hi[1]
    value = (ahi - alo) / a
    _verify(value, lambda x: 1 / (x * math.sqrt(x * x - a * a)), lo_v, hi_v)
    integrand = F(f"1/(x√(x² - {_sq(a)}))", rf"\frac{{1}}{{x\sqrt{{x^{{2}}-{_sq(a)}}}}}")
    stem = _stem(_integral(integrand, _limit_text(lo[0], a), _limit_text(hi[0], a)))
    cands = [
        (ahi - alo, "trigsub-spurious-factor"),
        ((alo - ahi) / a, "trigsub-sign-reversed-limits"),
        ((math.atan(hi[1]) - math.atan(lo[1])) / a, "trigsub-sec-as-tan"),
        (a * (ahi - alo), "trigsub-spurious-factor"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {a} sec θ. Then dx = {a} sec θ tan θ dθ and √(x² - {_sq(a)}) = {a} tan θ, so the integrand becomes (1/{a}) dθ.",
            f"The limits become θ = arcsec({lo[0]}) and θ = arcsec({hi[0]}).",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


def _hard_x_squared(rng):
    a = rng.choice([2, 3, 4])
    s, (coef, root), p = rng.choice(_SINES)
    u, uplain, utex = _limit(a, coef, root)
    theta = math.asin(s)
    c = math.sqrt(1 - s * s)
    h = Fraction(a * a, 2)
    value = h * (theta - s * c)
    _verify(value, lambda x: x * x / math.sqrt(a * a - x * x), 0, u)
    integrand = F(f"x²/√({_sq(a)} - x²)", rf"\frac{{x^{{2}}}}{{\sqrt{{{_sq(a)}-x^{{2}}}}}}")
    stem = _stem(_integral(integrand, ("0", "0"), (uplain, utex)))
    cands = [
        (h * (theta + s * c), "trigsub-double-angle-sign"),
        (h * theta, "trigsub-double-angle-term-dropped"),
        (2 * h * (theta - s * c), "trigsub-half-dropped"),
        (Fraction(a, 2) * (theta - s * c), "trigsub-constant-not-squared"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Put x = {a} sin θ. The integrand becomes {_sq(a)} sin² θ dθ = {h}(1 - cos 2θ) dθ.",
            f"Integrating gives {h}(θ - sin θ cos θ) from 0 to {rational_text(p)}π.",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


def _hard_complete_square(rng):
    a = rng.choice([2, 3, 4, 5])
    h = rng.randint(1, 4)
    t, tplain, ttex = rng.choice([(1, "1", "1"), (_S3, "√3", r"\sqrt{3}")])
    quad_plain = f"x² - {2 * h}x + {h * h + a * a}"
    quad_tex = f"x^{{2}}-{2 * h}x+{h * h + a * a}"
    hi_plain = f"{h + a}" if t == 1 else f"{h} + {a}√3"
    hi_tex = f"{h + a}" if t == 1 else rf"{h}+{a}\sqrt{{3}}"
    hi_value = h + a * t
    value = math.atan(t) / a
    _verify(value, lambda x: 1 / (x * x - 2 * h * x + h * h + a * a), h, hi_value)
    integrand = F(f"1/({quad_plain})", rf"\frac{{1}}{{{quad_tex}}}")
    stem = _stem(_integral(integrand, (str(h), str(h)), (hi_plain, hi_tex)))
    cands = [
        (math.atan(t), "trigsub-spurious-factor"),
        ((math.atan(hi_value / a) - math.atan(Fraction(h, a))) / a, "trigsub-limits-not-shifted"),
        (math.atan(t) / (a * a), "trigsub-spurious-factor"),
        ((math.pi / 2 - math.atan(t)) / a, "trigsub-inverse-swapped"),
    ]
    ds = pick_decimal(value, cands, 3)
    _need(ds)
    return _result(
        stem,
        value,
        ds,
        [
            f"Complete the square: {quad_plain} = (x - {h})² + {a * a}.",
            f"Put x - {h} = {a} tan θ. The integrand becomes (1/{a}) dθ and the limits become θ = 0 and θ = arctan({'1' if t == 1 else '√3'}).",
            f"The value is {decimal_text(value, 3)[1]}.",
        ],
    )


@generator(
    "ucv1_trig_substitution",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_trig_substitution(rng, tier="easy"):
    """Definite integrals by trigonometric substitution; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retrying(rng.choice([_easy_arcsin, _easy_arctan]), rng)
    if tier == "medium":
        return _retrying(rng.choice([_medium_sqrt, _medium_power]), rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retrying(rng.choice([_hard_secant, _hard_x_squared, _hard_complete_square]), rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _exam(rng):
    a = rng.choice(list(range(2, 13)))
    s, (coef, root), p = rng.choice(_SINES)
    u, uplain, utex = _limit(a, coef, root)
    theta = math.pi * float(p)
    check(abs(math.asin(s) - theta) < 1e-12, "the upper angle differs")
    k = a * a
    inner = theta / 2 - math.sin(2 * theta) / 4
    value = k * inner
    _verify(value, lambda x: x * x / math.sqrt(a * a - x * x), 0, u)
    _verify(inner, lambda t: math.sin(t) ** 2, 0, theta)
    inner_n, inner_text = decimal_text(inner, 4)
    value_n, value_text = decimal_text(value, 3)
    pf = p
    p_frac = fraction_dist
    a_dist = pick_distractors(
        rational_text(p),
        [
            p_frac(Fraction(1, 2) - pf, "trigsub-inverse-swapped"),
            p_frac(pf / 2, "trigsub-limit-half-angle"),
            p_frac(pf * 2, "trigsub-limit-double-angle"),
            p_frac(pf + Fraction(1, 2), "trigsub-inverse-swapped"),
        ],
        4,
    )
    b_dist = pick_distractors(
        str(k),
        [
            int_dist(a, "trigsub-constant-not-squared"),
            int_dist(a**3, "trigsub-constant-cubed"),
            int_dist(1, "trigsub-constant-dropped"),
            int_dist(2 * a, "trigsub-constant-not-squared"),
        ],
        4,
    )
    c_cands = [
        (theta / 2, "trigsub-double-angle-term-dropped"),
        (theta / 2 + math.sin(2 * theta) / 4, "trigsub-double-angle-sign"),
        (math.sin(theta) ** 2, "trigsub-evaluates-instead-of-integrates"),
        (theta - math.sin(2 * theta) / 4, "trigsub-half-dropped"),
    ]
    c_dist = pick_decimal(inner, c_cands, 4, n=4, gap=0.0005)
    d_cands = [
        (inner, "trigsub-constant-dropped"),
        (2 * k * inner, "trigsub-half-dropped"),
        (k * inner / 2, "trigsub-half-dropped"),
        (k * (theta / 2 + math.sin(2 * theta) / 4), "trigsub-double-angle-sign"),
    ]
    d_dist = pick_decimal(value, d_cands, 3)

    # Offering what a learner's own earlier (wrong) answers lead to as a wrong option would deny the follow-through
    # marks, so those values are not offered.
    b_values = [k] + [int(d.text) for d in b_dist]
    c_values = [inner_n] + [float(d.text) for d in c_dist]
    own = {decimal_text(bv * cv, 3)[1] for bv in b_values for cv in c_values} - {value_text}
    d_dist = [d for d in d_dist if d.text not in own]
    own_c = {
        decimal_text(math.pi * float(Fraction(d.text)) / 2 - math.sin(2 * math.pi * float(Fraction(d.text))) / 4, 4)[1]
        for d in a_dist
    } - {inner_text}
    c_dist = [d for d in c_dist if d.text not in own_c]
    for ds in (a_dist, b_dist, c_dist, d_dist):
        _need(ds)

    integrand = F(f"x²/√({_sq(a)} - x²)", rf"\frac{{x^{{2}}}}{{\sqrt{{{_sq(a)}-x^{{2}}}}}}")
    stem = say("Let ", F("I = ", "I="), _integral(integrand, ("0", "0"), (uplain, utex)), ".")
    return dict(
        text=stem[0],
        latex=stem[1],
        parts=[
            dict(
                prompt=f"Use the substitution x = {a} sin θ. The upper limit x = {uplain} becomes θ = θ₁. Find θ₁/π.",
                prompt_latex=rf"\text{{Use the substitution }}x={a}\sin\theta\text{{. The upper limit }}x={utex}\text{{ becomes }}\theta=\theta_1\text{{. Find }}\theta_1/\pi\text{{.}}",
                answer_kind="fraction",
                answer=exact_value(pf),
                answer_text=rational_text(pf),
                answer_latex=rational_text(pf),
                marks=2,
                working=[f"sin θ₁ = {uplain}/{a}, so θ₁ = {rational_text(pf)}π."],
                distractors=a_dist,
            ),
            dict(
                prompt="The substitution turns I into k times the integral of sin² θ dθ from 0 to θ₁. Find k.",
                prompt_latex=r"\text{The substitution turns }I\text{ into }k\int_{0}^{\theta_1}\sin^{2}\theta\,d\theta\text{. Find }k\text{.}",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                answer_latex=str(k),
                marks=2,
                working=[f"x² = {k} sin² θ, and dx/√({k} - x²) = dθ, so k = {k}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the integral of sin² θ dθ from 0 to θ₁, correct to 4 decimal places.",
                prompt_latex=r"\text{Find }\int_{0}^{\theta_1}\sin^{2}\theta\,d\theta\text{, correct to 4 decimal places.}",
                answer_kind="number",
                answer=inner_n,
                answer_text=inner_text,
                answer_latex=inner_text,
                marks=2,
                follow_through="pi*part_a/2 - sin(2*pi*part_a)/4",
                working=["sin² θ = (1 - cos 2θ)/2, so the integral is θ/2 - (sin 2θ)/4 evaluated at θ₁."],
                distractors=c_dist,
            ),
            dict(
                prompt="Hence find I, correct to 3 decimal places.",
                answer_kind="number",
                answer=value_n,
                answer_text=value_text,
                answer_latex=value_text,
                marks=2,
                follow_through="part_b*part_c",
                working=[f"I = {k} × {inner_text} = {value_text}."],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_fy_trig_substitution", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_trig_substitution(rng, tier="exam"):
    """Exam chain: the limit in θ, the constant, ∫ sin² θ dθ and the integral of x²/√(a² - x²)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retrying(_exam, rng)
