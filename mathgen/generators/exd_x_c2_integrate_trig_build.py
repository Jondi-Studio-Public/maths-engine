"""Exam-tier source for ``integrate-trig``: integrands that cannot be integrated as they stand.

Two families, both of which need an identity before any antiderivative is available, so the
learner chooses the method:

- a squared sine or cosine, ``c sin^2(kx)`` or ``c cos^2(kx)``, using the half-angle identity;
- a product of two sines and/or cosines with different frequencies, ``c sin(ax)cos(bx)`` and the like,
  using the product-to-sum identities.

Built backwards from the antiderivative ``F``; the integrand is then differentiated back from ``F`` as an
independent check (``F' == integrand``), and the stated integrand is the one the learner sees.

Distractors:

- squared case: treating ``sin^2`` with the power rule (``sin^3 / 3k``), using the identity with the
  wrong sign, or dropping the inner ``1/2`` that comes from integrating ``cos(2kx)``;
- product case: multiplying the two separate antiderivatives, dropping the factor of one half from the
  identity, or flipping the sign of the second term.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, cos, diff, sin

from ..core import check, check_distinct, fmt_signed, generator, phrase, pick_distractors

_X = Symbol("x")

_PATTERNS = (
    "Find ∫ {integrand} dx.",
    "Evaluate the indefinite integral ∫ {integrand} dx.",
    "Determine ∫ {integrand} dx.",
)

_FN = {"sin": sin, "cos": cos}


# Trigonometric product identities defeat sympy's ``simplify``, so the derivative check is numeric: the two
# expressions must agree to 20 significant figures at several fixed rational points.
def _display(expr) -> str:
    """Learner-facing text: ``^`` for powers, never Python's ``**``."""
    return str(expr).replace("**", "^")


_PROBES = (Rational(3, 7), Rational(11, 5), Rational(-9, 4), Rational(13, 3))


def _agree(left, right) -> bool:
    return all(abs((left - right).subs(_X, t).evalf(40)) < 1e-20 for t in _PROBES)


def _arg_text(k: int) -> str:
    return "x" if k == 1 else f"{k}x"


def _squared(rng: random.Random) -> tuple[str, object, list]:
    name = rng.choice(["sin", "cos"])
    k = rng.randint(2, 5)
    c = rng.randint(2, 6)
    sign = -1 if name == "sin" else 1  # sin^2 = (1 - cos 2t)/2, cos^2 = (1 + cos 2t)/2
    fn = _FN[name]
    integrand = c * fn(k * _X) ** 2
    antiderivative = c * (_X / 2 + sign * sin(2 * k * _X) / (4 * k))
    wrong = [
        c * fn(k * _X) ** 3 / (3 * k),
        c * (_X / 2 - sign * sin(2 * k * _X) / (4 * k)),
        c * (_X / 2 + sign * sin(2 * k * _X) / 4),
    ]
    text = fmt_signed(c, f"{name}^2({_arg_text(k)})")
    return text, integrand, [antiderivative, wrong, integrand]


def _product(rng: random.Random) -> tuple[str, object, list]:
    first, second = rng.choice([("sin", "cos"), ("cos", "cos"), ("sin", "sin")])
    a, b = rng.sample(range(1, 7), 2)
    c = rng.choice([2, 4, 6])
    # sin a cos b = (sin(a+b) + sin(a-b))/2 ; cos a cos b = (cos(a-b) + cos(a+b))/2 ; sin a sin b = (cos(a-b) - cos(a+b))/2
    s, d = a + b, a - b
    if (first, second) == ("sin", "cos"):
        parts = (-cos(s * _X) / (2 * s), -cos(d * _X) / (2 * d))
        flipped = (-cos(s * _X) / (2 * s), cos(d * _X) / (2 * d))
    elif (first, second) == ("cos", "cos"):
        parts = (sin(s * _X) / (2 * s), sin(d * _X) / (2 * d))
        flipped = (sin(s * _X) / (2 * s), -sin(d * _X) / (2 * d))
    else:
        parts = (-sin(s * _X) / (2 * s), sin(d * _X) / (2 * d))
        flipped = (-sin(s * _X) / (2 * s), -sin(d * _X) / (2 * d))
    antiderivative = c * (parts[0] + parts[1])
    integrand = c * _FN[first](a * _X) * _FN[second](b * _X)
    antider = {"sin": lambda k: -cos(k * _X) / k, "cos": lambda k: sin(k * _X) / k}
    separate = c * antider[first](a) * antider[second](b)
    wrong = [separate, 2 * antiderivative, c * (flipped[0] + flipped[1])]
    text = fmt_signed(c, f"{first}({_arg_text(a)}){second}({_arg_text(b)})")
    return text, integrand, [antiderivative, wrong, integrand]


def _exam(rng: random.Random) -> dict:
    integrand_text, integrand, (antiderivative, wrong, _) = (_squared if rng.random() < 0.5 else _product)(rng)
    check(_agree(diff(antiderivative, _X), integrand), "derivative of the antiderivative is not the integrand")
    check_distinct(antiderivative, *wrong)
    text = phrase(rng, _PATTERNS, integrand=integrand_text)
    answer_text = f"{_display(antiderivative)} + C"
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=str(antiderivative),
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=pick_distractors(answer_text, [f"{_display(w)} + C" for w in wrong]),
    )


@generator("exd_x_c2_integrate_trig", level=15, presets={"exam": dict(tier="exam")})
def exd_x_c2_integrate_trig(rng: random.Random, tier: str = "exam") -> dict:
    """Integrate sin^2/cos^2 or a product of sines and cosines of different frequencies."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
