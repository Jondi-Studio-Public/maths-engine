"""Generates `laplace_transform` questions (easy, medium, hard) and the exam chain `exd_x_uc_laplace`.

``ucv1-laplace-transform`` (``ordinary-differential-equations``, answer ``expression``): forward and inverse Laplace
transforms from a table, and the transform method for an initial value problem.

* easy: the transform of a three-term combination such as 3e^(2t) - 4t² + 5 (answer an expression in s).
* medium: the first shift theorem (e^(at) sin(bt), e^(at) cos(bt), t e^(at), t² e^(at), with a plain term beside it),
  or an inverse transform that needs one partial-fraction split.
* hard: an inverse transform that needs a repeated root, (ps + q)/(s - a)² or N(s)/(s(s - a)²), or completing the
  square, (ps + q)/(s² + 2ms + n) with n - m² a perfect square.
* exam (``exd_x_uc_laplace``): solve y' + ay = k e^(-bt), y' + ay = k, or y'' + by' + cy = 0 with initial values:
  (a) Y(s), (b) one partial-fraction coefficient (follows through from (a)), (c) y(t) (follows through from (b)),
  (d) y at a stated t to 3 decimal places (follows through from (c)).

Every time-domain function is a list of table terms, and every claimed transform is checked by direct numerical
integration of f(t)e^(-s0 t) over [0, 12] (Simpson), a different route from the table the answer is read from.
The exam chain is checked by substituting y(t) back into the differential equation exponential by exponential and
into the initial values, in exact fractions, and by integrating y(t)e^(-s0 t) against Y(s0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ._ucv1_texsay import say_marked
from ..core import (
    attempts,
    check,
    generator,
    join_terms,
    pick_distractors,
    rational_text,
    rounds_unambiguously,
)

_FACT = {0: 1, 1: 1, 2: 2, 3: 6}
_t, _s = sympy.symbols("t s")
_SUPER = {2: "²", 3: "³"}

# ---------------------------------------------------------------------------------------------------------------
# Time-domain terms


@dataclass(frozen=True)
class _T:
    """One table term c * basis(t). `kind` is const, pow, exp, texp, t2exp, sin, cos, expsin or expcos."""

    c: Fraction
    kind: str
    n: int = 0  # power of t, for `pow`
    a: int = 0  # rate in e^(at)
    b: int = 0  # frequency in sin(bt), cos(bt)

    def value(self, t: float) -> float:
        c = float(self.c)
        k = self.kind
        if k == "const":
            return c
        if k == "pow":
            return c * t**self.n
        if k == "exp":
            return c * math.exp(self.a * t)
        if k == "texp":
            return c * t * math.exp(self.a * t)
        if k == "t2exp":
            return c * t * t * math.exp(self.a * t)
        if k == "sin":
            return c * math.sin(self.b * t)
        if k == "cos":
            return c * math.cos(self.b * t)
        if k == "expsin":
            return c * math.exp(self.a * t) * math.sin(self.b * t)
        return c * math.exp(self.a * t) * math.cos(self.b * t)

    def laplace(self, s: Fraction) -> Fraction:
        """The table transform, exactly, at a rational s."""
        c, k, a, b = self.c, self.kind, self.a, self.b
        if k == "const":
            return c / s
        if k == "pow":
            return c * _FACT[self.n] / s ** (self.n + 1)
        if k == "exp":
            return c / (s - a)
        if k == "texp":
            return c / (s - a) ** 2
        if k == "t2exp":
            return 2 * c / (s - a) ** 3
        if k == "sin":
            return c * b / (s * s + b * b)
        if k == "cos":
            return c * s / (s * s + b * b)
        if k == "expsin":
            return c * b / ((s - a) ** 2 + b * b)
        return c * (s - a) / ((s - a) ** 2 + b * b)

    def sym(self):
        """The term as a SymPy expression in t (machine form of the answer)."""
        c, k, a, b = sympy.Rational(self.c.numerator, self.c.denominator), self.kind, self.a, self.b
        if k == "const":
            return c
        if k == "pow":
            return c * _t**self.n
        grow = sympy.exp(a * _t) if k in ("exp", "texp", "t2exp", "expsin", "expcos") else 1
        if k in ("exp", "texp", "t2exp"):
            return c * _t ** {"exp": 0, "texp": 1, "t2exp": 2}[k] * grow
        trig = sympy.sin(b * _t) if k.endswith("sin") else sympy.cos(b * _t)
        return c * grow * trig

    def basis(self, pretty: bool = False) -> str:
        k = self.kind
        if k == "const":
            return ""
        if k == "pow":
            return "t" if self.n == 1 else (f"t{_SUPER[self.n]}" if pretty else f"t^{self.n}")
        e = _exp_text(self.a)
        if k == "exp":
            return e
        if k == "texp":
            return f"t {e}"
        if k == "t2exp":
            return f"t{_SUPER[2]} {e}" if pretty else f"t^2 {e}"
        trig = f"{'sin' if k.endswith('sin') else 'cos'}({_freq_text(self.b)})"
        return trig if k in ("sin", "cos") else f"{e} {trig}"


def _exp_text(a: int) -> str:
    if a == 1:
        return "e^t"
    if a == -1:
        return "e^(-t)"
    return f"e^({a}t)"


def _freq_text(b: int) -> str:
    return "t" if b == 1 else f"{b}t"


def _coef_body(c: Fraction, basis: str) -> tuple[bool, str]:
    """(is_negative, text without sign) of c * basis, the way a student writes it."""
    mag = abs(c)
    if not basis:
        return c < 0, rational_text(mag)
    if mag == 1:
        return c < 0, basis
    if mag.denominator == 1:
        return c < 0, f"{mag.numerator}{basis}"
    return c < 0, f"({mag.numerator}/{mag.denominator}){basis}"


def _join(items: list[tuple[bool, str]]) -> str:
    """Signed pieces joined as a student writes a sum: ``a - b + c``."""
    out = []
    for i, (neg, body) in enumerate(items):
        if i == 0:
            out.append(f"-{body}" if neg else body)
        else:
            out.append(f"{'-' if neg else '+'} {body}")
    return " ".join(out)


def _merged(terms: list[_T]) -> list[_T]:
    """Like terms collected (same basis), dropping any that cancel, in first-appearance order."""
    order: dict[tuple, Fraction] = {}
    for t in terms:
        key = (t.kind, t.n, t.a, t.b)
        order[key] = order.get(key, Fraction(0)) + t.c
    return [_T(c, *key) for key, c in order.items() if c != 0]


def _time_text(terms: list[_T], pretty: bool = False) -> str:
    return _join([_coef_body(t.c, t.basis(pretty)) for t in _merged(terms)])


def _time_value(terms: list[_T], t: float) -> float:
    return sum(term.value(t) for term in terms)


# ---------------------------------------------------------------------------------------------------------------
# s-domain terms


@dataclass(frozen=True)
class _S:
    """K * numerator / denominator. numerator: "" (1), "s" or a shift ("sh", a) for (s - a).
    denominator: ("pow", r, k) for (s - r)^k, or ("quad", r, b2) for (s - r)^2 + b2."""

    K: Fraction
    num: object
    den: tuple

    def value(self, s: Fraction) -> Fraction:
        n = 1 if self.num == "" else (s if self.num == "s" else s - self.num[1])
        kind, r, x = self.den
        d = (s - r) ** x if kind == "pow" else (s - r) ** 2 + x
        return self.K * n / d

    def sym(self):
        K = sympy.Rational(self.K.numerator, self.K.denominator)
        n = 1 if self.num == "" else (_s if self.num == "s" else _s - self.num[1])
        kind, r, x = self.den
        d = (_s - r) ** x if kind == "pow" else (_s - r) ** 2 + x
        return K * n / d

    def text(self) -> tuple[bool, str]:
        mag = abs(self.K)
        assert mag.denominator == 1
        k = mag.numerator
        if self.num == "":
            top = str(k)
        else:
            core = "s" if self.num == "s" else _shifted(self.num[1])
            top = core if k == 1 else (f"{k}{core}")
        return self.K < 0, f"{top}/{_den_text(self.den)}"


def _shifted(a: int) -> str:
    if a == 0:
        return "s"
    return f"(s - {a})" if a > 0 else f"(s + {-a})"


def _den_text(den: tuple) -> str:
    kind, r, x = den
    base = _shifted(r)
    if kind == "pow":
        return base if x == 1 else (f"{base}^{x}" if r != 0 else f"s^{x}")
    sq = "s^2" if r == 0 else f"{base}^2"
    return f"({sq} + {x})"


def _s_terms(term: _T) -> _S:
    c, k, a, b = term.c, term.kind, term.a, term.b
    if k == "const":
        return _S(c, "", ("pow", 0, 1))
    if k == "pow":
        return _S(c * _FACT[term.n], "", ("pow", 0, term.n + 1))
    if k == "exp":
        return _S(c, "", ("pow", a, 1))
    if k == "texp":
        return _S(c, "", ("pow", a, 2))
    if k == "t2exp":
        return _S(2 * c, "", ("pow", a, 3))
    if k == "sin":
        return _S(c * b, "", ("quad", 0, b * b))
    if k == "cos":
        return _S(c, "s", ("quad", 0, b * b))
    if k == "expsin":
        return _S(c * b, "", ("quad", a, b * b))
    return _S(c, ("sh", a), ("quad", a, b * b))


def _s_text(terms: list[_S]) -> str:
    """The sum, with terms that share a numerator and denominator collected."""
    order: dict[tuple, Fraction] = {}
    for t in terms:
        key = (t.num, t.den)
        order[key] = order.get(key, Fraction(0)) + t.K
    return _join([_S(K, *key).text() for key, K in order.items() if K != 0])


def _s_value(terms: list[_S], s: Fraction) -> Fraction:
    return sum((t.value(s) for t in terms), Fraction(0))


def _machine(expr) -> str:
    """A SymPy expression as the parseable string kept in `answer` (the display form is `answer_text`)."""
    return str(expr).replace("**", "^")


def _differs_s(answer: list[_S], other: list[_S]) -> bool:
    return any(_s_value(answer, Fraction(p)) != _s_value(other, Fraction(p)) for p in (5, 7, 11))


def _differs_t(answer: list[_T], other: list[_T]) -> bool:
    return any(abs(_time_value(answer, p) - _time_value(other, p)) > 1e-6 for p in (0.37, 0.91, 1.43))


# ---------------------------------------------------------------------------------------------------------------
# Direct numerical integration, the independent route


def _quad(f, s0: float, upper: float = 12.0, steps: int = 1200) -> float:
    """Simpson's rule for the integral of f(t) e^(-s0 t) over [0, upper]."""
    h = upper / steps
    total = f(0.0) + f(upper) * math.exp(-s0 * upper)
    for i in range(1, steps):
        t = i * h
        total += (4 if i % 2 else 2) * f(t) * math.exp(-s0 * t)
    return total * h / 3


def _s0_for(terms: list[_T]) -> int:
    return max([0] + [t.a for t in terms]) + 3


def _check_transform(terms: list[_T], s_value) -> None:
    """`s_value(s0)` must equal the integral of f(t)e^(-s0 t); `terms` is the time-domain function."""
    s0 = _s0_for(terms)
    direct = _quad(lambda t: _time_value(terms, t), s0)
    claimed = float(s_value(Fraction(s0)))
    check(
        abs(direct - claimed) <= 1e-6 * max(1.0, abs(claimed)),
        f"transform disagrees with integration: {direct} {claimed}",
    )


def _nonzero(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# ---------------------------------------------------------------------------------------------------------------
# easy: transform of a three-term combination

_EASY_ORDER = ("exp", "pow", "sin", "cos", "const")


def _draw_basic(rng, kind: str, coef_hi: int = 6) -> _T:
    c = Fraction(_nonzero(rng, -coef_hi, coef_hi))
    if kind == "const":
        return _T(Fraction(rng.randint(1, 9)), "const")
    if kind == "pow":
        return _T(c, "pow", n=rng.randint(1, 3))
    if kind == "exp":
        return _T(c, "exp", a=_nonzero(rng, -4, 4))
    return _T(c, kind, b=rng.randint(2, 5))


def _basic_slips(term: _T) -> list[tuple[str, list[_S]]]:
    """Plausible wrong transforms of one table term, as (misconception id, s-domain terms)."""
    c, k, a, b = term.c, term.kind, term.a, term.b
    out: list[tuple[str, list[_S]]] = []
    if k == "const":
        out.append(("laplace-constant-as-number", [_S(c, "", ("pow", 0, 0))]))
    elif k == "pow":
        n = term.n
        if n > 1:
            out.append(("laplace-forgot-factorial", [_S(c, "", ("pow", 0, n + 1))]))
        out.append(("laplace-power-off-by-one", [_S(c * _FACT[n], "", ("pow", 0, n))]))
    elif k == "exp":
        out.append(("laplace-exp-sign", [_S(c, "", ("pow", -a, 1))]))
    elif k == "sin":
        out.append(("laplace-sin-cos-swap", [_S(c, "s", ("quad", 0, b * b))]))
        out.append(("laplace-sin-missing-b", [_S(c, "", ("quad", 0, b * b))]))
        out.append(("laplace-quad-b-not-squared", [_S(c * b, "", ("quad", 0, b))]))
    elif k == "cos":
        out.append(("laplace-sin-cos-swap", [_S(c * b, "", ("quad", 0, b * b))]))
        out.append(("laplace-quad-b-not-squared", [_S(c, "s", ("quad", 0, b))]))
    return out


def _easy(rng):
    for _ in attempts(100, "an easy Laplace combination"):
        kinds = sorted(rng.sample(_EASY_ORDER, 3), key=_EASY_ORDER.index)
        terms = [_draw_basic(rng, kind) for kind in kinds]
        if len({(t.kind, t.a if t.kind == "exp" else 0) for t in terms}) == 3:
            break
    answer = [_s_terms(t) for t in terms]
    _check_transform(terms, lambda s0: _s_value(answer, s0))
    candidates = []
    for i, term in enumerate(terms):
        for tag, wrong in _basic_slips(term):
            candidates.append((tag, answer[:i] + wrong + answer[i + 1 :]))
    return terms, answer, candidates


# ---------------------------------------------------------------------------------------------------------------
# medium: first shift theorem, or one partial-fraction split

_SHIFT_KINDS = ("expsin", "expcos", "texp", "t2exp")


def _shift_term(rng) -> _T:
    kind = rng.choice(_SHIFT_KINDS)
    c = Fraction(_nonzero(rng, -5, 5))
    a = _nonzero(rng, -4, 4)
    if kind in ("expsin", "expcos"):
        return _T(c, kind, a=a, b=rng.randint(2, 4))
    return _T(c, kind, a=a)


def _shift_slips(term: _T) -> list[tuple[str, list[_S]]]:
    c, k, a, b = term.c, term.kind, term.a, term.b
    out = []
    if k == "expsin":
        out.append(("laplace-no-shift", [_S(c * b, "", ("quad", 0, b * b))]))
        out.append(("laplace-exp-sign", [_S(c * b, "", ("quad", -a, b * b))]))
        out.append(("laplace-sin-missing-b", [_S(c, "", ("quad", a, b * b))]))
        out.append(("laplace-shift-not-squared", [_S(c * b, "", ("pow", a, 2))]))
    elif k == "expcos":
        out.append(("laplace-no-shift", [_S(c, "s", ("quad", 0, b * b))]))
        out.append(("laplace-exp-sign", [_S(c, ("sh", -a), ("quad", -a, b * b))]))
        out.append(("laplace-shift-numerator-only", [_S(c, "s", ("quad", a, b * b))]))
        out.append(("laplace-sin-cos-swap", [_S(c * b, "", ("quad", a, b * b))]))
    elif k == "texp":
        out.append(("laplace-shift-not-squared", [_S(c, "", ("pow", a, 1))]))
        out.append(("laplace-exp-sign", [_S(c, "", ("pow", -a, 2))]))
        out.append(("laplace-no-shift", [_S(c, "", ("pow", 0, 2))]))
    else:
        out.append(("laplace-forgot-factorial", [_S(c, "", ("pow", a, 3))]))
        out.append(("laplace-exp-sign", [_S(2 * c, "", ("pow", -a, 3))]))
        out.append(("laplace-shift-not-squared", [_S(2 * c, "", ("pow", a, 2))]))
    return out


def _medium_shift(rng):
    for _ in attempts(100, "a shift-theorem question"):
        shifted = _shift_term(rng)
        plain = rng.choice([_draw_basic(rng, "const"), _draw_basic(rng, "pow"), _draw_basic(rng, "sin")])
        if plain.kind == "pow":
            plain = _T(plain.c, "pow", n=rng.randint(1, 2))
        terms = [shifted, plain]
        if rng.random() < 0.5:
            terms.reverse()
        break
    answer = [_s_terms(t) for t in terms]
    _check_transform(terms, lambda s0: _s_value(answer, s0))
    candidates = []
    for i, term in enumerate(terms):
        slips = _shift_slips(term) if term.kind in _SHIFT_KINDS else _basic_slips(term)
        for tag, wrong in slips:
            candidates.append((tag, answer[:i] + wrong + answer[i + 1 :]))
    return terms, answer, candidates


# ---------------------------------------------------------------------------------------------------------------
# inverse transforms: the given F(s) is printed as numerator / denominator


def _numerator_text(n1: int, n0: int) -> str:
    return join_terms([(n1, "s"), (n0, "")])


def _quad_text(r: int, x: int) -> str:
    """s^2 + 2ms + n for (s - r)^2 + x, expanded."""
    return join_terms([(1, "s^2"), (-2 * r, "s"), (r * r + x, "")])


def _frac_text(num: str, den: str, single_term: bool) -> str:
    return f"({num})/({den})" if not single_term else f"{num}/({den})"


def _inverse_check(answer: list[_T], n1: int, n0: int, den_value) -> None:
    """The printed F(s) = (n1 s + n0)/den(s) must be the transform of the answer, by integration."""
    s0 = _s0_for(answer)
    direct = _quad(lambda t: _time_value(answer, t), s0)
    claimed = float((n1 * s0 + n0) / den_value(Fraction(s0)))
    check(
        abs(direct - claimed) <= 1e-6 * max(1.0, abs(claimed)),
        f"inverse disagrees with integration: {direct} {claimed}",
    )


def _medium_inverse(rng):
    for _ in attempts(200, "an inverse with two simple poles"):
        a, b = sorted(rng.sample(range(-4, 5), 2))
        A, B = _nonzero(rng, -6, 6), _nonzero(rng, -6, 6)
        if A + B != 0 and (A, B) != (1, 1):
            break
    n1, n0 = A + B, -(A * b + B * a)
    answer = [_T(Fraction(A), "exp", a=a), _T(Fraction(B), "exp", a=b)]
    _inverse_check(answer, n1, n0, lambda s: (s - a) * (s - b))
    expanded = rng.random() < 0.5
    den_text = join_terms([(1, "s^2"), (-(a + b), "s"), (a * b, "")]) if expanded else f"{_shifted(a)}{_shifted(b)}"
    stem = _frac_text(_numerator_text(n1, n0), den_text, n1 == 0 or n0 == 0)
    candidates = [
        ("laplace-inverse-coefficient-swap", [_T(Fraction(A), "exp", a=b), _T(Fraction(B), "exp", a=a)]),
        ("laplace-inverse-pole-sign", [_T(Fraction(A), "exp", a=-a), _T(Fraction(B), "exp", a=-b)]),
        (
            "laplace-inverse-coefficient-undivided",
            [_T(Fraction(n1 * a + n0), "exp", a=a), _T(Fraction(n1 * b + n0), "exp", a=b)],
        ),
        ("laplace-inverse-term-dropped", [_T(Fraction(A), "exp", a=a)]),
        ("laplace-inverse-numerator-split", [_T(Fraction(n1), "exp", a=a), _T(Fraction(n0), "exp", a=b)]),
    ]
    return f"F(s) = {stem}", answer, candidates


def _hard_repeated(rng):
    a = _nonzero(rng, -4, 4)
    expanded = rng.random() < 0.5
    if rng.random() < 0.4:  # (ps + q)/(s - a)^2
        for _ in attempts(100, "a repeated-root numerator"):
            p, C = _nonzero(rng, -5, 5), _nonzero(rng, -6, 6)
            q = C - p * a
            if q != 0:
                break
        answer = [_T(Fraction(p), "exp", a=a), _T(Fraction(C), "texp", a=a)]
        den_text = join_terms([(1, "s^2"), (-2 * a, "s"), (a * a, "")]) if expanded else f"{_shifted(a)}^2"
        _inverse_check(answer, p, q, lambda s: (s - a) ** 2)
        stem = _frac_text(_numerator_text(p, q), den_text, q == 0)
        candidates = [
            ("laplace-repeated-root-no-t", [_T(Fraction(p), "exp", a=a), _T(Fraction(C), "exp", a=a)]),
            ("laplace-repeated-root-numerator", [_T(Fraction(p), "exp", a=a), _T(Fraction(q), "texp", a=a)]),
            ("laplace-inverse-pole-sign", [_T(Fraction(p), "exp", a=-a), _T(Fraction(C), "texp", a=-a)]),
            ("laplace-inverse-term-dropped", [_T(Fraction(C), "texp", a=a)]),
        ]
        return f"F(s) = {stem}", answer, candidates
    for _ in attempts(200, "a three-term repeated-root split"):
        A, B, C = _nonzero(rng, -4, 4), _nonzero(rng, -4, 4), _nonzero(rng, -5, 5)
        if A + B != 0:
            break
    n2, n1, n0 = A + B, C - 2 * a * A - a * B, a * a * A
    answer = [_T(Fraction(A), "const"), _T(Fraction(B), "exp", a=a), _T(Fraction(C), "texp", a=a)]
    s0 = _s0_for(answer)
    direct = _quad(lambda t: _time_value(answer, t), s0)
    claimed = float((n2 * s0 * s0 + n1 * s0 + n0) / (s0 * (s0 - a) ** 2))
    check(abs(direct - claimed) <= 1e-6 * max(1.0, abs(claimed)), "three-term inverse disagrees with integration")
    num_text = join_terms([(n2, "s^2"), (n1, "s"), (n0, "")])
    den_text = f"s{_shifted(a)}^2" if not expanded else join_terms([(1, "s^3"), (-2 * a, "s^2"), (a * a, "s")])
    single = sum(1 for v in (n2, n1, n0) if v) == 1
    stem = _frac_text(num_text, den_text, single)
    candidates = [
        (
            "laplace-repeated-root-no-t",
            [_T(Fraction(A), "const"), _T(Fraction(B), "exp", a=a), _T(Fraction(C), "exp", a=a)],
        ),
        (
            "laplace-inverse-pole-sign",
            [_T(Fraction(A), "const"), _T(Fraction(B), "exp", a=-a), _T(Fraction(C), "texp", a=-a)],
        ),
        ("laplace-inverse-term-dropped", [_T(Fraction(A), "const"), _T(Fraction(C), "texp", a=a)]),
        (
            "laplace-inverse-coefficient-swap",
            [_T(Fraction(B), "const"), _T(Fraction(A), "exp", a=a), _T(Fraction(C), "texp", a=a)],
        ),
    ]
    return f"F(s) = {stem}", answer, candidates


def _hard_square(rng):
    for _ in attempts(200, "a complete-the-square inverse"):
        m = _nonzero(rng, -3, 3)
        w = rng.randint(1, 4)
        A, B = _nonzero(rng, -5, 5), _nonzero(rng, -5, 5)
        if A + B != 0:
            break
    # f = e^(-mt)(A cos wt + B sin wt); F = (A s + A m + B w)/(s^2 + 2ms + m^2 + w^2)
    a = -m
    n1, n0 = A, A * m + B * w
    answer = [_T(Fraction(A), "expcos", a=a, b=w), _T(Fraction(B), "expsin", a=a, b=w)]
    s0 = _s0_for(answer)
    direct = _quad(lambda t: _time_value(answer, t), s0)
    claimed = float((n1 * s0 + n0) / (s0 * s0 + 2 * m * s0 + m * m + w * w))
    check(
        abs(direct - claimed) <= 1e-6 * max(1.0, abs(claimed)), "complete-the-square inverse disagrees with integration"
    )
    den_text = join_terms([(1, "s^2"), (2 * m, "s"), (m * m + w * w, "")])
    stem = _frac_text(_numerator_text(n1, n0), den_text, n0 == 0)
    candidates = [
        (
            "laplace-square-sine-undivided",
            [_T(Fraction(A), "expcos", a=a, b=w), _T(Fraction(n0 - A * m), "expsin", a=a, b=w)],
        ),
        (
            "laplace-square-numerator-unshifted",
            [_T(Fraction(A), "expcos", a=a, b=w), _T(Fraction(n0, w), "expsin", a=a, b=w)],
        ),
        ("laplace-inverse-pole-sign", [_T(Fraction(A), "expcos", a=m, b=w), _T(Fraction(B), "expsin", a=m, b=w)]),
        ("laplace-sin-cos-swap", [_T(Fraction(B), "expcos", a=a, b=w), _T(Fraction(A), "expsin", a=a, b=w)]),
        ("laplace-square-no-exponential", [_T(Fraction(A), "cos", b=w), _T(Fraction(B), "sin", b=w)]),
    ]
    return f"F(s) = {stem}", answer, candidates


# ---------------------------------------------------------------------------------------------------------------
# registered single-answer topic

_FORWARD_STEMS = [
    "Find the Laplace transform of «f(t) = {f}».",
    "Use the table of Laplace transforms to find «L{{f(t)}}» for «f(t) = {f}».",
    "Find «F(s) = L{{f(t)}}», where «f(t) = {f}».",
]
_INVERSE_STEMS = [
    "Find the inverse Laplace transform of «{F}».",
    "Find «f(t) = L⁻¹{{F(s)}}», given «{F}».",
    "Given «{F}», find the function «f(t)» whose Laplace transform is «F(s)».",
]


@generator(
    "laplace_transform",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def laplace_transform(rng, tier="easy"):
    if tier == "easy":
        terms, answer, candidates = _easy(rng)
        forward = True
    elif tier == "medium":
        if rng.random() < 0.5:
            terms, answer, candidates = _medium_shift(rng)
            forward = True
        else:
            given, answer, candidates = _medium_inverse(rng)
            forward = False
    else:
        pick = rng.random()
        given, answer, candidates = (_hard_square if pick < 0.45 else _hard_repeated)(rng)
        forward = False

    if forward:
        answer_text = _s_text(answer)
        machine = _machine(sum((t.sym() for t in answer), sympy.Integer(0)))
        stem, stem_latex = say_marked(rng.choice(_FORWARD_STEMS).format(f=f"{_time_text(terms, pretty=True)}"))
        scored = [
            Distractor(_s_text(w), tag, _s_text(w)) for tag, w in candidates if _s_text(w) and _differs_s(answer, w)
        ]
    else:
        answer_text = _time_text(answer)
        machine = _machine(sum((t.sym() for t in answer), sympy.Integer(0)))
        stem, stem_latex = say_marked(rng.choice(_INVERSE_STEMS).format(F=given.replace("^2", "²").replace("^3", "³")))
        scored = [
            Distractor(_time_text(w), tag, _time_text(w))
            for tag, w in candidates
            if _time_text(w) and _differs_t(answer, w)
        ]
    distractors = pick_distractors(answer_text, scored, 4)
    check(len(distractors) >= 2, "too few Laplace distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine,
        answer_text=answer_text,
        answer_kind="expression",
        distractors=distractors,
        working=[
            f"Apply the standard transforms term by term (with the shift theorem or partial fractions where needed) "
            f"to reach {answer_text}."
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _lin(r: int) -> str:
    """The factor (s - r), written as a student writes it; a bare s when r = 0."""
    return _shifted(r)


def _y_text(n1: int, n0: int, r1: int, r2: int) -> str:
    num = _numerator_text(n1, n0)
    single = (n1 == 0) != (n0 == 0) or (n1 == 0 and n0 == 0)
    den = f"{_lin(r1)}{_lin(r2)}"
    return f"{num}/({den})" if single else f"({num})/({den})"


def _exp_t(r: int) -> str:
    return "" if r == 0 else _exp_text(r)


def _y_of_t(c1: Fraction, r1: int, c2: Fraction, r2: int) -> str:
    items = []
    for c, r in ((c1, r1), (c2, r2)):
        if c != 0:
            items.append(_coef_body(c, _exp_t(r)))
    return _join(items)


def _y_value(c1: Fraction, r1: int, c2: Fraction, r2: int, t: float) -> float:
    return float(c1) * math.exp(r1 * t) + float(c2) * math.exp(r2 * t)


def _ode_stem(kind: str, a: int, b: int, c: int, k: int, y0: int, v0: int) -> str:
    if kind == "second":
        left = join_terms([(1, "y''"), (b, "y'"), (c, "y")])
        return f"{left} = 0, y(0) = {y0}, y'(0) = {v0}"
    left = join_terms([(1, "y'"), (a, "y")])
    if kind == "exp":
        neg, body = _coef_body(Fraction(k), _exp_text(-b))
        right = f"-{body}" if neg else body
    else:
        right = str(k)
    return f"{left} = {right}, y(0) = {y0}"


_EXAM_STEMS = [
    "Use Laplace transforms to solve the initial value problem «{ode}».",
    "A function «y(t)» satisfies «{ode}». Solve for «y(t)» with Laplace transforms.",
    "Consider the initial value problem «{ode}», which is to be solved by Laplace transforms.",
]
_B_PROMPTS = [
    "Write «Y(s)» in partial fractions as «{A}/{f1} + {B}/{f2}». Find the value of «{which}».",
    "Express «Y(s)» in the form «{A}/{f1} + {B}/{f2}», and find «{which}».",
]
_C_PROMPTS = ["Hence find «y(t)».", "Hence find the solution «y(t)».", "Use your answer to find «y(t)»."]


def _prompt(template: str, **slots) -> dict:
    text, latex = say_marked(template.format(**slots))
    return dict(prompt=text, prompt_latex=latex)


def _draw_exam(rng):
    for _ in attempts(1000, "an exam Laplace initial value problem"):
        kind = rng.choice(["exp", "const", "second"])
        y0 = rng.randint(-4, 6)
        v0 = 0
        a = b = c = k = 0
        if kind == "exp":
            a, b, k = _nonzero(rng, -3, 4), rng.randint(1, 4), _nonzero(rng, -12, 12)
            if a == b:
                continue
            r1, r2 = -a, -b
            n1, n0 = y0, y0 * b + k
        elif kind == "const":
            a, k = _nonzero(rng, -4, 4), _nonzero(rng, -12, 12)
            r1, r2 = 0, -a
            n1, n0 = y0, k
        else:
            r1, r2 = sorted(rng.sample([v for v in range(-5, 4) if v != 0], 2))
            b, c = -(r1 + r2), r1 * r2
            v0 = rng.randint(-8, 8)
            n1, n0 = y0, v0 + b * y0
        C1 = Fraction(n1 * r1 + n0, r1 - r2)
        C2 = Fraction(n1 * r2 + n0, r2 - r1)
        if C1 == 0 or C2 == 0 or C1.denominator > 6 or C2.denominator > 6:
            continue
        if kind == "second" and y0 == 0 and v0 == 0:
            continue
        t0 = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)])
        value = _y_value(C1, r1, C2, r2, float(t0))
        if not (0.02 <= abs(value) <= 400 and rounds_unambiguously(value, 3, 0.1)):
            continue
        return kind, a, b, c, k, y0, v0, r1, r2, n1, n0, C1, C2, t0, value
    raise AssertionError("unreachable")


def _check_exam_ode(kind, a, b, c, k, y0, v0, r1, r2, C1, C2) -> None:
    """Substitute y = C1 e^(r1 t) + C2 e^(r2 t) into the equation, one exponential at a time, in exact fractions."""
    check(C1 + C2 == y0, "y(0) is wrong")
    if kind == "exp":
        check(C1 * (r1 + a) == 0 and C2 * (r2 + a) == k and r2 == -b, "y' + ay is not k e^(-bt)")
    elif kind == "const":
        check(C1 * a == k and C2 * (r2 + a) == 0, "y' + ay is not k")
    else:
        check(C1 * r1 + C2 * r2 == v0, "y'(0) is wrong")
        for C, r in ((C1, r1), (C2, r2)):
            check(C * (r * r + b * r + c) == 0, "y'' + by' + cy is not 0")


class _Redraw(Exception):
    """A draw whose parts cannot each offer two distinct wrong answers."""


@generator("exd_x_uc_laplace", level=20, presets={"exam": dict(tier="exam")})
def exd_x_uc_laplace(rng, tier="exam"):
    """Exam tier: an initial value problem solved by Laplace transforms, in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam Laplace question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    kind, a, b, c, k, y0, v0, r1, r2, n1, n0, C1, C2, t0, value = _draw_exam(rng)
    _check_exam_ode(kind, a, b, c, k, y0, v0, r1, r2, C1, C2)
    # Y(s) against the solution by integration, at one s0 past both poles.
    s0 = max(r1, r2, 0) + 3
    direct = _quad(lambda t: _y_value(C1, r1, C2, r2, t), s0)
    check(abs(direct - (n1 * s0 + n0) / ((s0 - r1) * (s0 - r2))) < 1e-6, "Y(s) disagrees with integration of y(t)")

    ode = _ode_stem(kind, a, b, c, k, y0, v0)
    y_text = _y_text(n1, n0, r1, r2)

    # (a) Y(s)
    if kind == "second":
        wrong_a = [
            ("laplace-sy0-term-dropped", _y_text(0, n0, r1, r2)),
            ("laplace-derivative-initial-sign", _y_text(-n1, -n0, r1, r2)),
            ("laplace-sy0-term-dropped", _y_text(n1, v0, r1, r2)),
            ("laplace-derivative-initial-sign", _y_text(n1, v0 - b * y0, r1, r2)),
        ]
    elif kind == "exp":
        wrong_a = [
            ("laplace-derivative-initial-sign", _y_text(-y0, -y0 * b + k, r1, r2)),
            ("laplace-initial-condition-dropped", _y_text(0, k, r1, r2)),
            ("laplace-forcing-ignored", f"{y0}/({_lin(r1)})"),
            ("laplace-exp-sign", _y_text(y0, -y0 * b + k, r1, b)),
        ]
    else:
        wrong_a = [
            ("laplace-derivative-initial-sign", _y_text(-y0, k, r1, r2)),
            ("laplace-initial-condition-dropped", _y_text(0, k, r1, r2)),
            ("laplace-forcing-ignored", f"{y0}/({_lin(r2)})"),
            ("laplace-constant-as-number", f"{y0 + k}/({_lin(r2)})"),
        ]
    a_dist = pick_distractors(y_text, [Distractor(t, tag, t) for tag, t in wrong_a], 4)

    # (b) one partial-fraction coefficient
    ask_first = rng.random() < 0.5
    pole = r1 if ask_first else r2
    other = r2 if ask_first else r1
    coef = C1 if ask_first else C2
    other_coef = C2 if ask_first else C1
    num_at = lambda x: n1 * x + n0  # noqa: E731
    b_wrong = [
        ("laplace-residue-undivided", Fraction(num_at(pole))),
        ("laplace-residue-sign", Fraction(num_at(-pole), pole - other)),
        ("laplace-residue-difference-order", Fraction(num_at(pole), other - pole)),
        ("laplace-residue-other-coefficient", other_coef),
    ]
    coef_text = rational_text(coef)
    b_dist = pick_distractors(
        coef_text, [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in b_wrong if v != coef], 4
    )

    # (c) y(t); follows through from (b)
    c_text = _y_of_t(C1, r1, C2, r2)
    c_wrong = [
        ("laplace-inverse-coefficient-swap", _y_of_t(C2, r1, C1, r2)),
        ("laplace-inverse-pole-sign", _y_of_t(C1, -r1, C2, -r2)),
        ("laplace-inverse-term-dropped", _y_of_t(coef, pole, Fraction(0), other)),
        (
            "laplace-inverse-coefficient-swap",
            _y_of_t(Fraction(y0) - C1, r1, C1, r2) if y0 != C2 else _y_of_t(C1, r1, C1, r2),
        ),
    ]
    c_dist = pick_distractors(c_text, [Distractor(t, tag, t) for tag, t in c_wrong], 4)

    # (d) y(t0) to 3 decimal places; follows through from (c)
    t0f = float(t0)
    d_text = f"{value:.3f}"
    alt_values = [
        ("laplace-inverse-coefficient-swap", _y_value(C2, r1, C1, r2, t0f)),
        ("laplace-inverse-pole-sign", _y_value(C1, -r1, C2, -r2, t0f)),
        ("laplace-inverse-term-dropped", float(coef) * math.exp(pole * t0f)),
        ("laplace-inverse-term-dropped", float(other_coef) * math.exp(other * t0f)),
    ]
    d_cands = [
        Distractor(f"{v:.3f}", tag, round(v, 3))
        for tag, v in alt_values
        if abs(v) < 1e6 and abs(round(v, 3) - round(value, 3)) > 0.0015
    ]
    d_dist = pick_distractors(d_text, d_cands, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    f1, f2 = _lin(r1), _lin(r2)
    t0_text = rational_text(t0) if t0.denominator == 1 else f"{float(t0):g}"
    t0_ft = str(t0.numerator) if t0.denominator == 1 else f"{t0.numerator}/{t0.denominator}"
    if kind == "second":
        lhs = join_terms([(1, "s²Y"), (-y0, "s"), (-v0, ""), (b, f"({join_terms([(1, 'sY'), (-y0, '')])})"), (c, "Y")])
        a_working = [
            f"Transforming the equation, using y(0) = {y0} and y'(0) = {v0}, gives {lhs} = 0, so Y = {y_text}."
        ]
    else:
        a_working = [f"Transforming the equation and using y(0) = {y0} gives Y = {y_text}."]
    which = "A" if ask_first else "B"
    stem, stem_latex = say_marked(rng.choice(_EXAM_STEMS).format(ode=ode))
    follow_c = f"part_b*exp({pole}*t) + ({y0} - part_b)*exp({other}*t)"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **_prompt("Take the Laplace transform of both sides and find «Y(s) = L{{y(t)}}»."),
                answer_kind="expression",
                answer=_machine(sympy.Rational(1) * (n1 * _s + n0) / ((_s - r1) * (_s - r2))),
                answer_text=y_text,
                marks=3,
                working=a_working,
                distractors=a_dist,
            ),
            dict(
                **_prompt(rng.choice(_B_PROMPTS), A="A", B="B", f1=f1, f2=f2, which=which),
                answer_kind="fraction",
                answer=exact_value(coef),
                answer_text=coef_text,
                marks=2,
                follow_through=f"Subs(part_a*({_lin(pole)}), (s,), ({pole},))",
                working=[
                    f"{which} = [({n1})s + ({n0})]/({pole} - ({other})) evaluated at s = {pole}, which is {coef_text}."
                ],
                distractors=b_dist,
            ),
            dict(
                **_prompt(rng.choice(_C_PROMPTS)),
                answer_kind="expression",
                answer=_machine(
                    sympy.Rational(C1.numerator, C1.denominator) * sympy.exp(r1 * _t)
                    + sympy.Rational(C2.numerator, C2.denominator) * sympy.exp(r2 * _t)
                ),
                answer_text=c_text,
                marks=2,
                follow_through=follow_c,
                working=[f"Inverting term by term, y(t) = {c_text}."],
                distractors=c_dist,
            ),
            dict(
                **_prompt("Find «y({t0})», correct to 3 decimal places.", t0=t0_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=d_text,
                marks=1,
                follow_through=f"Subs(part_c, (t, e), ({t0_ft}, E))",
                working=[f"y({t0_text}) = {d_text}."],
                distractors=d_dist,
            ),
        ],
    )
