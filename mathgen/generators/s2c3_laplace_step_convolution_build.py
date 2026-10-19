"""Generates `s2c3_laplace_step` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_laplace_step`.

``uc2-laplace-step-and-convolution`` (``ordinary-differential-equations``, answer ``expression``): the second shift
theorem and the convolution theorem.

* easy: the transform of a signal that switches on at ``t = a`` with a body written in ``t`` (not in ``t - a``): a
  linear body, ``c t^2 + d`` or ``c e^(kt)``. The body has to be re-expressed in ``t - a`` before the shift theorem
  applies.
* medium: the inverse transform of ``e^(-as) N(s) / ((s - r1)(s - r2))`` or ``e^(-as)(Ps + Q)/(s^2 + b^2)``, as an
  expression in ``t`` valid for ``t > a``.
* hard: a convolution ``(g * h)(t)`` of two standard functions, or the inverse transform of the product of their
  transforms, found with the convolution theorem.
* exam (``exd_x_s2c3_laplace_step``): ``y'' + py' + qy = A u(t - T)`` with zero initial conditions: (a) ``Y(s)``,
  (b) the long-time value ``A/q`` (follows through from (a)), (c) ``y(t)`` for ``t >= T`` (follows through from
  (b)), (d) ``y`` at a stated time to three decimal places (follows through from (c)).

Every claimed transform is checked by direct numerical integration of ``f(t) e^(-s0 t)`` (Simpson), a different route
from the table the answer is read from, every convolution by integrating ``g(tau) h(t - tau)`` directly, and the exam
chain by substituting the solution into the differential equation (central differences) and the initial values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text, rounds_unambiguously
from ._s2c3_common import fixed, key_ok, number_part, simpson, wrong_numbers
from ._ucv1_texsay import say_marked

_t, _s = sympy.symbols("t s")


def _machine(expr) -> str:
    return str(expr).replace("**", "^")


def _rat(f: Fraction):
    return sympy.Rational(f.numerator, f.denominator)


def _coef(c: Fraction) -> tuple[bool, str]:
    """(negative, prefix) for the magnitude of a coefficient: '' for 1, '3', or '(2/5)'."""
    mag = abs(c)
    if mag == 1:
        return c < 0, ""
    if mag.denominator == 1:
        return c < 0, str(mag.numerator)
    return c < 0, f"({mag.numerator}/{mag.denominator})"


def _coef_over(c: int, den: str) -> tuple[bool, str]:
    return c < 0, f"{abs(c)}/{den}"


def _join(items: list[tuple[bool, str]]) -> str:
    out = ""
    for i, (neg, body) in enumerate(items):
        out += (("-" if neg else "") + body) if i == 0 else (" - " if neg else " + ") + body
    return out or "0"


# -- time-domain terms ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tm:
    """c * basis(arg) with arg = t - a (or t when a = 0). kind: one, t, t2, exp, sin, cos."""

    c: Fraction
    kind: str
    p: int = 0
    a: int = 0

    def _arg_sym(self):
        return _t - self.a if self.a else _t

    def sym(self):
        arg, c = self._arg_sym(), _rat(self.c)
        return {
            "one": c,
            "t": c * arg,
            "t2": c * arg**2,
            "exp": c * sympy.exp(self.p * arg),
            "sin": c * sympy.sin(self.p * arg),
            "cos": c * sympy.cos(self.p * arg),
        }[self.kind]

    def value(self, t: float) -> float:
        x = t - self.a
        c = float(self.c)
        return {
            "one": c,
            "t": c * x,
            "t2": c * x * x,
            "exp": c * math.exp(self.p * x),
            "sin": c * math.sin(self.p * x),
            "cos": c * math.cos(self.p * x),
        }[self.kind]

    def arg(self) -> str:
        if self.a < 0:
            return f"(t + {-self.a})"
        return f"(t - {self.a})" if self.a else "t"

    def scaled_arg(self) -> str:
        if self.p == 1:
            return self.arg()[1:-1] if self.a else "t"
        inner = self.arg()
        return f"{self.p}{inner}" if self.p > 0 else f"-{abs(self.p)}{inner}"

    def basis(self) -> str:
        if self.kind == "one":
            return ""
        if self.kind == "t":
            return self.arg()
        if self.kind == "t2":
            return f"{self.arg()}²"
        func = {"exp": "e^", "sin": "sin", "cos": "cos"}[self.kind]
        if self.kind == "exp":
            if self.p == 1 and not self.a:
                return "e^t"
            if self.p == -1:
                return f"e^(-{self.arg()})" if self.a else "e^(-t)"
            return f"e^({self.scaled_arg()})"
        return f"{func}({self.scaled_arg()})"

    def text(self) -> tuple[bool, str]:
        neg, pre = _coef(self.c)
        basis = self.basis()
        if self.kind == "one":
            return neg, pre or "1"
        return neg, f"{pre}{basis}"


def time_text(terms: list[Tm]) -> str:
    return _join([t.text() for t in terms if t.c != 0])


def time_sym(terms: list[Tm]):
    return sum((t.sym() for t in terms), sympy.Integer(0))


def time_value(terms: list[Tm], t: float) -> float:
    return sum(term.value(t) for term in terms)


def _differs_t(a: list[Tm], b: list[Tm], start: float = 0.0) -> bool:
    return any(abs(time_value(a, start + p) - time_value(b, start + p)) > 1e-6 for p in (0.37, 0.91, 1.43, 2.2))


def _laplace_check(terms: list[Tm], start: int, s0: float, expected: float) -> None:
    """Integrate f(t) e^(-s0 t) over [start, start + 30/s0] numerically and compare with the claimed transform value."""
    top = start + 40.0 / s0
    got = simpson(lambda x: sum(_vec(term, x) for term in terms) * np.exp(-s0 * x), start, top, 12000)
    check(abs(got - expected) <= 1e-6 * (1 + abs(expected)), "numerical transform differs from the claimed one")


def _vec(term: Tm, x):
    y = x - term.a
    c = float(term.c)
    return {
        "one": c * np.ones_like(x),
        "t": c * y,
        "t2": c * y * y,
        "exp": c * np.exp(term.p * y),
        "sin": c * np.sin(term.p * y),
        "cos": c * np.cos(term.p * y),
    }[term.kind]


def _s_value(expr, s0: float) -> float:
    return float(expr.subs(_s, sympy.Rational(s0).limit_denominator(1000)))


# -- easy: a switched-on body in t ---------------------------------------------------------------------------------


def _poly_text(items: list[tuple[int, str]]) -> str:
    return join_terms(items)


def _num_s(coeffs: dict[int, int]) -> str:
    """Numerator in s from {power: coefficient}, highest power first."""
    items = []
    for power in sorted(coeffs, reverse=True):
        label = "" if power == 0 else ("s" if power == 1 else f"s^{power}")
        items.append((coeffs[power], label))
    return join_terms(items)


def _shifted_text(a: int, num: str, den: str, single: bool) -> str:
    body = num if single else f"({num})"
    return f"e^(-{a}s){body}/{den}" if den else f"e^(-{a}s){body}"


def exp_s(a: int, sign: int = -1) -> str:
    """e^(-as) (sign -1) or e^(as) (sign +1), with e^(-s) for a = 1."""
    if sign == 0:
        return ""
    mag = "" if a == 1 else str(a)
    return f"e^({'-' if sign < 0 else ''}{mag}s)"


def _poly_s(sign: int, a: int, coeffs: dict[int, int], m: int) -> str:
    """exp factor (sign: -1, +1 or 0 for none), numerator {power: coefficient}, denominator s^m, written so that a
    learner (and the grader) reads the numerator as a whole: a lone constant goes in front of the exponential."""
    terms = {k: v for k, v in coeffs.items() if v}
    num = _num_s(coeffs)
    den = "s" if m == 1 else f"s^{m}"
    factor = exp_s(a, sign)
    if not factor:
        return f"{num if len(terms) == 1 else f'({num})'}/{den}"
    if len(terms) == 1 and 0 in terms:
        c = terms[0]
        return f"{'-' if c < 0 else ''}{'' if abs(c) == 1 else abs(c)}{factor}/{den}"
    return f"{factor}({num})/{den}"


def _poly_sym(sign: int, a: int, coeffs: dict[int, int], m: int):
    lead = sympy.exp(sign * a * _s) if sign else 1
    return lead * sum(v * _s**k for k, v in coeffs.items()) / _s**m


def _easy(rng):
    family = rng.choice(["linear", "linear", "quadratic", "exponential"])
    a = rng.choice([1, 2, 3, 4])
    wrong: list[tuple[str, object, str]] = []  # (display text, sympy, misconception)
    if family == "linear":
        c1 = rng.choice([1, 2, 3, 4, -1, -2, -3])
        c0 = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
        g_text = join_terms([(c1, "t"), (c0, "")])
        g_sym = c0 + c1 * _t
        k0 = c0 + c1 * a  # the constant once the body is written in (t - a)
        correct = {1: k0, 0: c1}
        answer_text = _poly_s(-1, a, correct, 2)
        answer = _poly_sym(-1, a, correct, 2)
        s0 = 3.0
        for sign, co, tag in [
            (-1, {1: c0, 0: c1}, "lt2-body-not-shifted"),
            (0, {1: c0, 0: c1}, "lt2-switch-factor-dropped"),
            (1, correct, "lt2-shift-factor-sign"),
            (-1, {1: c0 - c1 * a, 0: c1}, "lt2-body-shifted-wrong-way"),
        ]:
            wrong.append((_poly_s(sign, a, co, 2), _poly_sym(sign, a, co, 2), tag))
    elif family == "quadratic":
        c2 = rng.choice([1, 2, 3, 4])
        c0 = rng.choice([-2, -1, 0, 1, 2, 3])
        g_text = join_terms([(c2, "t²"), (c0, "")])
        g_sym = c2 * _t**2 + c0
        correct = {2: c2 * a * a + c0, 1: 2 * a * c2, 0: 2 * c2}
        answer_text = _poly_s(-1, a, correct, 3)
        answer = _poly_sym(-1, a, correct, 3)
        s0 = 3.0
        unshifted = {2: c0, 1: 0, 0: 2 * c2}
        reversed_way = {2: c2 * a * a + c0, 1: -2 * a * c2, 0: 2 * c2}
        for sign, co, tag in [
            (-1, unshifted, "lt2-body-not-shifted"),
            (0, unshifted, "lt2-switch-factor-dropped"),
            (1, correct, "lt2-shift-factor-sign"),
            (-1, reversed_way, "lt2-body-shifted-wrong-way"),
        ]:
            wrong.append((_poly_s(sign, a, co, 3), _poly_sym(sign, a, co, 3), tag))
    else:
        c = rng.choice([1, 2, 3, 4, 5, -1, -2, -3])
        k = rng.choice([-2, -1, 1, 2, 3])
        g_text = join_terms([(c, f"e^({k}t)" if k != 1 else "e^t")])
        g_sym = c * sympy.exp(k * _t)
        answer = c * sympy.exp(-a * (_s - k)) / (_s - k)
        pole = f"(s - {k})" if k > 0 else f"(s + {-k})"
        cf_neg, cf = _coef(Fraction(c))
        lead = ("-" if cf_neg else "") + cf

        def exp_text(tail: str) -> str:
            return f"{lead}e^({tail})/{pole}"

        shown = f"-{a}{pole}" if a != 1 else f"-{pole}"
        answer_text = exp_text(shown)
        s0 = float(max(k, 0) + 3)
        wrong = [
            (f"{lead}{exp_s(a)}/{pole}", c * sympy.exp(-a * _s) / (_s - k), "lt2-body-not-shifted"),
            (f"{lead}/{pole}", c / (_s - k), "lt2-switch-factor-dropped"),
            ("", None, ""),
            (
                f"{lead}e^(" + (f"{a}" if a != 1 else "") + f"{pole})/{pole}",
                c * sympy.exp(a * (_s - k)) / (_s - k),
                "lt2-shift-factor-sign",
            ),
        ]
        # body shifted the wrong way: e^(-as - ka), written out in full
        wrong[2] = (
            f"{lead}e^(-{a if a != 1 else ''}s {'-' if k > 0 else '+'} {abs(a * k)})/{pole}",
            c * sympy.exp(-a * _s - a * k) / (_s - k),
            "lt2-body-shifted-wrong-way",
        )
    # the independent check: integrate g(t) e^(-s0 t) from a upward and compare with F(s0)
    top = a + 40.0 / s0
    g_fn = sympy.lambdify(_t, g_sym, "numpy")
    got = simpson(lambda x: g_fn(x) * np.ones_like(x) * np.exp(-s0 * x), a, top, 12000)
    want = _s_value(answer, s0)
    check(abs(got - want) <= 1e-6 * (1 + abs(want)), "the switched-on transform does not match direct integration")
    scored = [
        Distractor(text, tag, text)
        for text, expr, tag in wrong
        if any(abs(_s_value(expr, p) - _s_value(answer, p)) > 1e-9 for p in (5, 7, 11))
    ]
    distractors = pick_distractors(answer_text, scored, 4)
    check(len(distractors) >= 3, "too few switched-on distractors")
    when = f"t ≥ {a}"
    stem, latex = say_marked(
        rng.choice(
            [
                "A signal is switched on at «t = {a}»: «f(t) = {g}» for «{when}», and «f(t) = 0» before then. Find «F(s) = L{{f(t)}}».",
                "Let «f(t) = 0» for «t < {a}» and «f(t) = {g}» for «{when}». Find the Laplace transform «F(s)».",
            ]
        ).format(a=a, g=g_text, when=when)
    )
    return dict(
        text=stem,
        latex=latex,
        answer=_machine(answer),
        answer_text=answer_text,
        answer_kind="expression",
        distractors=distractors,
        working=[
            f"Write the body in terms of t − {a} before applying the second shift theorem, then multiply by "
            f"{exp_s(a)}: F(s) = {answer_text}."
        ],
    )


# -- medium: inverse transforms of a switched-on transform ---------------------------------------------------------


def _medium(rng):
    a = rng.choice([1, 2, 3, 4])
    if rng.random() < 0.55:
        r1, r2 = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        r1, r2 = sorted((r1, r2))
        for _ in attempts(100, "a medium numerator"):
            C1, C2 = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            n1, n0 = C1 + C2, -(C1 * r2 + C2 * r1)
            if n1 and n0:
                break
        num = join_terms([(n1, "s"), (n0, "")])
        pole = lambda r: f"(s - {r})" if r > 0 else f"(s + {-r})"  # noqa: E731
        den = f"{pole(r1)}{pole(r2)}"
        given = f"{exp_s(a)}({num})/({den})"
        answer = [Tm(Fraction(C1), "exp", r1, a), Tm(Fraction(C2), "exp", r2, a)]
        s0 = 5.0
        F = lambda s: math.exp(-a * s) * (n1 * s + n0) / ((s - r1) * (s - r2))  # noqa: E731
        swapped = [Tm(Fraction(C2), "exp", r1, a), Tm(Fraction(C1), "exp", r2, a)]
        undivided = [
            Tm(Fraction(n1 * r1 + n0), "exp", r1, a),
            Tm(Fraction(n1 * r2 + n0), "exp", r2, a),
        ]
        cands = [
            ([Tm(Fraction(C1), "exp", r1, 0), Tm(Fraction(C2), "exp", r2, 0)], "lt2-no-shift-in-time"),
            ([Tm(Fraction(C1), "exp", r1, -a), Tm(Fraction(C2), "exp", r2, -a)], "lt2-shift-sign-in-time"),
            (swapped, "laplace-inverse-coefficient-swap"),
            (undivided, "laplace-inverse-coefficient-undivided"),
            ([Tm(Fraction(C1), "exp", -r1, a), Tm(Fraction(C2), "exp", -r2, a)], "laplace-inverse-pole-sign"),
        ]
        working = [
            f"Split N(s)/(({pole(r1)})({pole(r2)})) into partial fractions: "
            f"{_join([_coef_over(C1, pole(r1)), _coef_over(C2, pole(r2))])}; "
            f"the factor e^(-{a}s) delays the result by {a}, so f(t) = {time_text(answer)} for t > {a}."
        ]
    else:
        b = rng.choice([2, 3, 4, 5])
        P = rng.choice([-3, -2, -1, 0, 1, 2, 3, 4])
        Qb = rng.choice([-3, -2, -1, 1, 2, 3])
        Q = Qb * b
        num = join_terms([(P, "s"), (Q, "")])
        given = f"{exp_s(a)}({num})/(s² + {b * b})" if P != 0 else f"{Q}{exp_s(a)}/(s² + {b * b})"
        answer = [Tm(Fraction(P), "cos", b, a), Tm(Fraction(Qb), "sin", b, a)]
        answer = [t for t in answer if t.c != 0]
        s0 = 4.0
        F = lambda s: math.exp(-a * s) * (P * s + Q) / (s * s + b * b)  # noqa: E731
        cands = [
            ([Tm(Fraction(P), "cos", b, 0), Tm(Fraction(Qb), "sin", b, 0)], "lt2-no-shift-in-time"),
            ([Tm(Fraction(P), "cos", b, -a), Tm(Fraction(Qb), "sin", b, -a)], "lt2-shift-sign-in-time"),
            ([Tm(Fraction(P), "sin", b, a), Tm(Fraction(Qb), "cos", b, a)], "lt2-sin-cos-swapped-in-inverse"),
            ([Tm(Fraction(P), "cos", b, a), Tm(Fraction(Q), "sin", b, a)], "lt2-sine-not-divided-by-b"),
            ([Tm(Fraction(P), "cos", b * b, a), Tm(Fraction(Qb), "sin", b * b, a)], "lt2-frequency-is-b-squared"),
        ]
        working = [
            f"Write (Ps + Q)/(s² + {b * b}) as P cos({b}t) + (Q/{b}) sin({b}t) and delay by {a}: "
            f"f(t) = {time_text(answer)} for t > {a}."
        ]
    check(given is not None and len(answer) >= 1, "empty inverse problem")
    _laplace_check(answer, a, s0, F(s0))
    answer_text = time_text(answer)
    scored = []
    for terms, tag in cands:
        terms = [t for t in terms if t.c != 0]
        if terms and _differs_t(answer, terms, a):
            text = time_text(terms)
            scored.append(Distractor(text, tag, text))
    distractors = pick_distractors(answer_text, scored, 4)
    check(len(distractors) >= 3, "too few inverse distractors")
    stem, latex = say_marked(
        rng.choice(
            [
                "Find the function «f(t)» for «t > {a}» whose Laplace transform is «F(s) = {F}».",
                "A function «f(t)» has Laplace transform «F(s) = {F}». Find «f(t)» for «t > {a}».",
            ]
        ).format(a=a, F=given)
    )
    return dict(
        text=stem,
        latex=latex,
        answer=_machine(time_sym(answer)),
        answer_text=answer_text,
        answer_kind="expression",
        distractors=distractors,
        working=working,
    )


# -- hard: the convolution theorem ---------------------------------------------------------------------------------

_PAIRS = ["exp-sin", "exp-cos", "t-exp", "exp-exp", "t-sin"]


def _e(rate: int) -> str:
    return "e^t" if rate == 1 else ("e^(-t)" if rate == -1 else f"e^({rate}t)")


def _hard_terms(pair: str, p: int, q: int, b: int):
    """(g text, h text, g sym, h sym, F text, answer terms) of one convolution pair. p, q rates; b frequency."""
    P2 = p * p + b * b
    if pair == "exp-sin":
        return (
            _e(p),
            f"sin({b}t)",
            sympy.exp(p * _t),
            sympy.sin(b * _t),
            [
                Tm(Fraction(b, P2), "exp", p),
                Tm(Fraction(-b, P2), "cos", b),
                Tm(Fraction(-p, P2) * Fraction(1), "sin", b),
            ],
        )
    if pair == "exp-cos":
        return (
            _e(p),
            f"cos({b}t)",
            sympy.exp(p * _t),
            sympy.cos(b * _t),
            [Tm(Fraction(p, P2), "exp", p), Tm(Fraction(-p, P2), "cos", b), Tm(Fraction(b, P2), "sin", b)],
        )
    if pair == "t-exp":
        return (
            "t",
            _e(p),
            _t,
            sympy.exp(p * _t),
            [Tm(Fraction(1, p * p), "exp", p), Tm(Fraction(-1, p * p), "one"), Tm(Fraction(-1, p), "t")],
        )
    if pair == "exp-exp":
        return (
            _e(p),
            _e(q),
            sympy.exp(p * _t),
            sympy.exp(q * _t),
            [Tm(Fraction(1, p - q), "exp", p), Tm(Fraction(-1, p - q), "exp", q)],
        )
    return (
        "t",
        f"sin({b}t)",
        _t,
        sympy.sin(b * _t),
        [Tm(Fraction(1, b), "t"), Tm(Fraction(-1, b * b), "sin", b)],
    )


def _conv_numeric(g_sym, h_sym, t0: float) -> float:
    g = sympy.lambdify(_t, g_sym, "numpy")
    h = sympy.lambdify(_t, h_sym, "numpy")
    return simpson(lambda tau: g(tau) * np.ones_like(tau) * h(t0 - tau) * np.ones_like(tau), 0.0, t0, 4000)


def _product_text(pair: str, p: int, q: int, b: int) -> str:
    pole = lambda r: f"(s - {r})" if r > 0 else f"(s + {-r})"  # noqa: E731
    if pair == "exp-sin":
        return f"{b}/({pole(p)}(s² + {b * b}))"
    if pair == "exp-cos":
        return f"s/({pole(p)}(s² + {b * b}))"
    if pair == "t-exp":
        return f"1/(s²{pole(p)})"
    if pair == "exp-exp":
        return f"1/({pole(p)}{pole(q)})"
    return f"{b}/(s²(s² + {b * b}))"


def _hard(rng):
    for _ in attempts(200, "a convolution problem"):
        pair = rng.choice(_PAIRS)
        p = rng.choice([-3, -2, -1, 1, 2, 3])
        q = rng.choice([-3, -2, -1, 1, 2, 3])
        b = rng.choice([1, 2, 3])
        if pair == "exp-exp" and p == q:
            continue
        g_text, h_text, g_sym, h_sym, answer = _hard_terms(pair, p, q, b)
        answer = [t for t in answer if t.c != 0]
        for t0 in (0.9, 1.7):
            want = time_value(answer, t0)
            got = _conv_numeric(g_sym, h_sym, t0)
            check(abs(got - want) <= 1e-6 * (1 + abs(want)), "closed form differs from the convolution integral")
        in_s = rng.random() < 0.35
        P2 = p * p + b * b
        cands = _hard_candidates(pair, p, q, b, answer)
        answer_text = time_text(answer)
        scored = []
        for terms, tag in cands:
            terms = [t for t in terms if t.c != 0]
            if terms and _differs_t(answer, terms):
                text = time_text(terms)
                scored.append(Distractor(text, tag, text))
        distractors = pick_distractors(answer_text, scored, 4)
        if len(distractors) < 3:
            continue
        if in_s:
            given = _product_text(pair, p, q, b)
            stem, latex = say_marked(
                rng.choice(
                    [
                        "Find the inverse Laplace transform of «F(s) = {F}».",
                        "Find «f(t)» given that «L{{f(t)}} = {F}».",
                    ]
                ).format(F=given)
            )
            # check the s-domain claim numerically as well
            s0 = float(abs(p) + abs(q) + 3)
            _laplace_check(answer, 0, s0, _product_value(pair, p, q, b, s0))
            working = [
                "The transform is a product of two standard transforms, so f is the convolution of the two matching "
                f"functions ({g_text} and {h_text}): f(t) = {answer_text}."
            ]
        else:
            stem, latex = say_marked(
                rng.choice(
                    [
                        "Find the convolution of «g(t) = {g}» and «h(t) = {h}» as a function of «t».",
                        "Two functions are «g(t) = {g}» and «h(t) = {h}». Find their convolution as a function of «t».",
                    ]
                ).format(g=g_text, h=h_text)
            )
            working = [
                f"By the convolution theorem the transform of the convolution is G(s)H(s); inverting by partial fractions gives {answer_text}."
            ]
        return dict(
            text=stem,
            latex=latex,
            answer=_machine(time_sym(answer)),
            answer_text=answer_text,
            answer_kind="expression",
            distractors=distractors,
            working=working,
        )
    raise AssertionError  # pragma: no cover


def _product_value(pair: str, p: int, q: int, b: int, s0: float) -> float:
    if pair == "exp-sin":
        return b / ((s0 - p) * (s0 * s0 + b * b))
    if pair == "exp-cos":
        return s0 / ((s0 - p) * (s0 * s0 + b * b))
    if pair == "t-exp":
        return 1 / (s0 * s0 * (s0 - p))
    if pair == "exp-exp":
        return 1 / ((s0 - p) * (s0 - q))
    return b / (s0 * s0 * (s0 * s0 + b * b))


def _hard_candidates(pair: str, p: int, q: int, b: int, answer: list[Tm]):
    out = []
    # a transient term dropped
    out.append((answer[1:], "lt2-convolution-term-dropped"))
    out.append((answer[:-1], "lt2-convolution-term-dropped"))
    # one sign slipped
    for i in range(len(answer)):
        flipped = [Tm(-t.c if j == i else t.c, t.kind, t.p, t.a) for j, t in enumerate(answer)]
        out.append((flipped, "lt2-convolution-sign-slip"))
    # the common denominator left out
    if pair in ("exp-sin", "exp-cos"):
        P2 = p * p + b * b
        out.append(([Tm(t.c * P2, t.kind, t.p, t.a) for t in answer], "lt2-convolution-denominator-forgotten"))
    # sin and cos exchanged
    swapped = [Tm(t.c, {"sin": "cos", "cos": "sin"}.get(t.kind, t.kind), t.p, t.a) for t in answer]
    if swapped != answer:
        out.append((swapped, "lt2-sin-cos-swapped-in-inverse"))
    # the frequency taken as b squared / rate sign
    out.append(([Tm(t.c, t.kind, -t.p if t.kind == "exp" else t.p, t.a) for t in answer], "laplace-inverse-pole-sign"))
    return out


# -- generator -----------------------------------------------------------------------------------------------------


@generator(
    "s2c3_laplace_step",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_laplace_step(rng, tier="easy"):
    """Second shift theorem and convolution; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------

_T_FRACTIONS = [Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)]


def _step_response(r1: int, r2: int, A: int, T: int) -> list[Tm]:
    """y(t) = A u(t-T) / ... for t >= T, as terms in (t - T): (A/q)[1 + r2/(r1-r2) e^(r1 u) - r1/(r1-r2) e^(r2 u)]."""
    q = r1 * r2
    return [
        Tm(Fraction(A, q), "one"),
        Tm(Fraction(A * r2, q * (r1 - r2)), "exp", r1, T),
        Tm(Fraction(-A * r1, q * (r1 - r2)), "exp", r2, T),
    ]


def _check_ode(terms: list[Tm], p: int, q: int, A: int, T: int) -> None:
    """Substitute y(t) back: y'' + p y' + q y = A for t > T (central differences), and y(T) = y'(T) = 0."""
    y = lambda t: time_value(terms, t)  # noqa: E731
    h = 1e-4
    for t0 in (T + 0.4, T + 1.3, T + 2.9):
        d1 = (y(t0 + h) - y(t0 - h)) / (2 * h)
        d2 = (y(t0 + h) - 2 * y(t0) + y(t0 - h)) / (h * h)
        check(abs(d2 + p * d1 + q * y(t0) - A) < 1e-4 * (1 + abs(A)), "the solution does not satisfy the equation")
    check(abs(y(T)) < 1e-9, "y(T) is not 0")
    check(abs((y(T + 1e-6) - y(T)) / 1e-6) < 1e-3, "y'(T) is not 0")


@generator("exd_x_s2c3_laplace_step", level=21, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_laplace_step(rng, tier="exam"):
    """Exam chain: y'' + py' + qy = A u(t - T), y(0) = y'(0) = 0."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam step-response problem"):
        r1, r2 = sorted(rng.sample([-5, -4, -3, -2, -1], 2))
        p, q = -(r1 + r2), r1 * r2
        A = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
        T = rng.choice([1, 2, 3])
        t0 = Fraction(T) + rng.choice(_T_FRACTIONS)
        y_terms = _step_response(r1, r2, A, T)
        _check_ode(y_terms, p, q, A, T)
        value = time_value(y_terms, float(t0))
        if not key_ok(value, 3, 0.02):
            continue
        # independent check of Y(s): integrate y(t) e^(-s0 t) from T and compare with A e^(-T s0)/(s0 (s0^2 + p s0 + q))
        s0 = 3.0
        Y = lambda s: A * math.exp(-T * s) / (s * (s * s + p * s + q))  # noqa: E731
        _laplace_check(y_terms, T, s0, Y(s0))
        final = Fraction(A, q)
        a_text = f"{A}e^(-{T}s)/(s(s^2 + {p}s + {q}))"
        Ysym = A * sympy.exp(-T * _s) / (_s * (_s**2 + p * _s + q))

        def y_text(has_s=True, sign=-1, exp=True, mid=p):
            lead = f"{A}e^({'-' if sign < 0 else ''}{T}s)" if exp else f"{A}"
            quad = join_terms([(1, "s^2"), (mid, "s"), (q, "")])
            tail = f"s({quad})" if has_s else quad
            return f"{lead}/({tail})"

        a_cands = [
            (y_text(has_s=False), "lt2-step-treated-as-impulse"),
            (y_text(sign=1), "lt2-shift-factor-sign"),
            (y_text(exp=False), "lt2-switch-factor-dropped"),
            (y_text(mid=-p), "lt2-damping-term-sign"),
            (f"{A}e^(-{T}s)/(s(s^2 + {q}))", "lt2-damping-term-dropped"),
        ]
        a_cands = [(txt, tag) for txt, tag in a_cands if txt != a_text]
        a_dist = pick_distractors(a_text, [Distractor(txt, tag, txt) for txt, tag in a_cands], 4)
        # (b) the long-time value
        b_text = rational_text(final)
        b_cands = [
            (Fraction(A), "lt2-final-value-no-divide"),
            (Fraction(q, A), "lt2-final-value-inverted"),
            (Fraction(A, p) if p else None, "lt2-final-value-wrong-coefficient"),
            (Fraction(A, p + q), "lt2-final-value-wrong-coefficient"),
            (Fraction(A, q) * Fraction(-1), "lt2-final-value-sign"),
        ]
        b_dist = pick_distractors(
            b_text,
            [Distractor(rational_text(v), tag, exact_value(v)) for v, tag in b_cands if v is not None and v != final],
            4,
        )
        # (c) y(t) for t >= T
        c_text = time_text(y_terms)
        base = y_terms
        shiftless = [Tm(t.c, t.kind, t.p, 0) for t in base]
        wrong_way = [Tm(t.c, t.kind, t.p, -T if t.a else 0) for t in base]
        swapped = [
            base[0],
            Tm(base[1].c, "exp", r2, T),
            Tm(base[2].c, "exp", r1, T),
        ]
        no_one = base[1:]
        c_cands = [
            (shiftless, "lt2-no-shift-in-time"),
            (wrong_way, "lt2-shift-sign-in-time"),
            (swapped, "laplace-inverse-coefficient-swap"),
            (no_one, "lt2-steady-term-dropped"),
        ]
        c_dist = []
        d_cands = []
        for terms, tag in c_cands:
            if _differs_t(base, terms, T):
                text = time_text(terms)
                c_dist.append(Distractor(text, tag, text))
                d_cands.append((time_value(terms, float(t0)), tag))
        c_dist = pick_distractors(c_text, c_dist, 4)
        d_cands.append((float(final), "lt2-final-value-used-for-y"))
        d_dist = wrong_numbers(value, 3, d_cands, 4)
        if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) >= 2:
            break
    else:
        raise AssertionError  # pragma: no cover
    ode = f"y″ + {p}y′ + {q}y = {A}u(t - {T})" if p != 1 else f"y″ + y′ + {q}y = {A}u(t - {T})"
    stem, stem_latex = say_marked(
        rng.choice(
            (
                f"A function «y(t)» satisfies «{ode}», where «u» is the unit step function (0 before its argument "
                f"reaches 0, 1 after), with «y(0) = 0» and «y′(0) = 0».",
                f"Let «y(t)» solve «{ode}», with «y(0) = 0» and «y′(0) = 0», where «u» is the unit step function (0 "
                f"before its argument reaches 0, 1 after).",
                f"The output «y(t)» of a system is governed by «{ode}» and starts from rest: «y(0) = 0» and "
                f"«y′(0) = 0». Here «u» is the unit step function (0 before its argument reaches 0, 1 after).",
            )
        )
    )

    def prompt(template: str, **slots) -> dict:
        text, latex = say_marked(template.format(**slots))
        return dict(prompt=text, prompt_latex=latex)

    ft_c = f"part_b*(1 + ({r2})/({r1 - r2})*exp({r1}*(t-{T})) - ({r1})/({r1 - r2})*exp({r2}*(t-{T})))"
    t0_text = rational_text(t0)
    t0_ft = str(t0.numerator) if t0.denominator == 1 else f"{t0.numerator}/{t0.denominator}"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Take the Laplace transform of both sides and find «Y(s) = L{{y(t)}}»."),
                answer_kind="expression",
                answer=_machine(Ysym),
                answer_text=a_text,
                marks=3,
                working=[
                    f"Transforming with zero initial conditions gives (s² + {p}s + {q})Y = {A}e^(-{T}s)/s, so Y = {a_text}."
                ],
                distractors=a_dist,
            ),
            dict(
                **prompt("Find the value that «y(t)» approaches as «t» becomes very large."),
                answer_kind="fraction",
                answer=exact_value(final),
                answer_text=b_text,
                marks=2,
                follow_through="Subs(part_a*s, (s,), (0,))",
                working=[f"The final value theorem gives lim s·Y(s) = {A}/{q} = {b_text}."],
                distractors=b_dist,
            ),
            dict(
                **prompt("Hence find «y(t)» for «t ≥ {T}».", T=T),
                answer_kind="expression",
                answer=_machine(time_sym(y_terms)),
                answer_text=c_text,
                marks=3,
                follow_through=ft_c,
                working=[f"Split Y(s) into partial fractions and delay by {T}: y(t) = {c_text} for t ≥ {T}."],
                distractors=c_dist,
            ),
            dict(
                **prompt("Find «y({t0})», correct to 3 decimal places.", t0=t0_text),
                **number_part(value, 3),
                marks=1,
                sensible_range=(None, None),
                follow_through=f"Subs(part_c, (t, e), ({t0_ft}, E))",
                working=[f"y({t0_text}) = {fixed(value, 3)}."],
                distractors=d_dist,
            ),
        ],
    )
