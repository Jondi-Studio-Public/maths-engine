"""Function-sequence families for the real-analysis generators (`ucv1_riemann_interchange_build.py`,
`ucv1_uniform_convergence_build.py`).

A family is a sequence f_n on a closed interval [a, b] with a pointwise limit f. Each one carries what a marker
needs: the exact supremum error s_n = sup |f_n - f| as a closed form in n, the point where it is attained (or
approached), whether s_n tends to 0 (uniform convergence), and, where it has a closed form, the integral of f_n
over [a, b] and of f.

The closed forms are checked when a family is built: s_n against a grid of 801 points on [a, b] for n = 1 to 6
(the grid never exceeds s_n, and the claimed point reaches it), and the integral against Simpson's rule.

Underscore-prefixed, so the generator registry does not import it as a generator module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from fractions import Fraction

from ..core import check
from ._free_proof_kit import Fm, frac_plain, frac_tex


@dataclass
class Family:
    key: str
    fn: Fm  # f_n(x)
    lim: Fm  # f(x) (for a piecewise limit, the pointwise description)
    a: float
    b: float
    interval: str  # "0 ≤ x ≤ 1"
    interval_tex: str
    f_n: Callable[[int, float], float]
    f: Callable[[float], float]
    sup: Callable[[int], float]
    sup_fm: Fm  # s_n in closed form
    uniform: bool
    argmax: Callable[[int], float] | None  # where |f_n - f| reaches (or, if open, approaches) s_n
    attained: bool = True
    int_fn: Callable[[int], float] | None = None
    int_fn_fm: Fm | None = None
    int_lim: float | None = None
    int_lim_fm: Fm | None = None
    # an easy-to-state valid (not necessarily sharp) bound: |f_n - f| <= bound(n) for all x
    bound: Callable[[int], float] | None = None
    bound_fm: Fm | None = None
    notes: dict = field(default_factory=dict)


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _simpson(g: Callable[[float], float], a: float, b: float, m: int = 2000) -> float:
    h = (b - a) / m
    s = g(a) + g(b)
    for i in range(1, m):
        s += g(a + i * h) * (4 if i % 2 else 2)
    return s * h / 3


def _verify(fam: Family) -> Family:
    grid = [fam.a + (fam.b - fam.a) * i / 800 for i in range(801)]
    for n in range(1, 7):
        s = fam.sup(n)
        mx = max(abs(fam.f_n(n, x) - fam.f(x)) for x in grid)
        check(mx <= s + 1e-9, f"{fam.key}: grid value above the claimed supremum at n={n}")
        if fam.argmax is not None:
            xs = fam.argmax(n)
            check(fam.a - 1e-12 <= xs <= fam.b + 1e-12, f"{fam.key}: argmax outside the interval")
            if fam.attained:
                check(abs(abs(fam.f_n(n, xs) - fam.f(xs)) - s) < 1e-9, f"{fam.key}: argmax does not reach the supremum")
        check(mx >= s - max(0.05 * s, 0.02) or not fam.attained, f"{fam.key}: supremum too far above the grid maximum")
        if fam.bound is not None:
            check(fam.bound(n) + 1e-12 >= s, f"{fam.key}: stated bound below the supremum")
    if fam.int_fn is not None:
        for n in (1, 2, 5):
            check(
                abs(_simpson(lambda x, n=n: fam.f_n(n, x), fam.a, fam.b) - fam.int_fn(n)) < 1e-5,
                f"{fam.key}: integral of f_n",
            )
        check(abs(_simpson(fam.f, fam.a, fam.b) - fam.int_lim) < 1e-5, f"{fam.key}: integral of f")
    check(fam.uniform == (fam.sup(10**6) < 1e-2), f"{fam.key}: s_n and the uniform flag disagree")
    if fam.int_lim is not None:
        fam.int_lim_fm = _fraction_fm(fam.int_lim)
    return fam


def _fraction_fm(x: float) -> Fm:
    f = Fraction(x).limit_denominator(1000)
    check(abs(float(f) - x) < 1e-9, "the integral of the limit is a simple fraction")
    return Fm(frac_tex(f.numerator, f.denominator), frac_plain(f.numerator, f.denominator))


def frac_of(x: float) -> Fraction:
    return Fraction(x).limit_denominator(1000)


def _plus(base: Fm, term: Fm) -> Fm:
    """`base + term`, written as a subtraction when `term` starts with a minus sign."""
    if term.plain.startswith("-"):
        return Fm(base.tex + term.tex, base.plain + " - " + term.plain[1:])
    return Fm(base.tex + "+" + term.tex, base.plain + " + " + term.plain)


def with_offset(fam: Family, p: int, q: int) -> Family:
    """The family with p x^2 + q added to both f_n and f: the error f_n - f, so its supremum, is unchanged, and the
    limit function (and so the common value of the integrals) is no longer 0 or x^2."""
    from ._crit_kit import poly_fm

    check(fam.uniform and fam.int_fn is not None, "an offset needs a uniform family with integrals")
    g_fm = poly_fm([q, 0, p], "x")
    a, b = fam.a, fam.b

    def g(x: float) -> float:
        return p * x * x + q

    gi_val = p * (b**3 - a**3) / 3 + q * (b - a)
    gi = frac_of(gi_val)
    check(abs(float(gi) - gi_val) < 1e-9, "the integral of the offset is a simple fraction")
    gi_fm = Fm(frac_tex(gi.numerator, gi.denominator), frac_plain(gi.numerator, gi.denominator))
    new = Family(
        key=fam.key + f"_off{p}_{q}",
        fn=_plus(fam.fn, g_fm),
        lim=g_fm if fam.lim.plain == "0" else _plus(fam.lim, g_fm),
        a=fam.a,
        b=fam.b,
        interval=fam.interval,
        interval_tex=fam.interval_tex,
        f_n=lambda n, x: fam.f_n(n, x) + g(x),
        f=lambda x: fam.f(x) + g(x),
        sup=fam.sup,
        sup_fm=fam.sup_fm,
        uniform=True,
        argmax=fam.argmax,
        attained=fam.attained,
        int_fn=lambda n: fam.int_fn(n) + gi_val,
        int_fn_fm=fam.int_fn_fm if gi == 0 else _plus(fam.int_fn_fm, gi_fm),
        int_lim=fam.int_lim + gi_val,
        bound=fam.bound,
        bound_fm=fam.bound_fm,
        notes=dict(fam.notes),
    )
    new.int_lim_fm = _fraction_fm(new.int_lim)
    return _verify(new)


def scaled(fam: Family, c: int) -> Family:
    """The family multiplied by the whole number c >= 2 (f_n and f both): the supremum error is c times larger and
    a persistent gap, if there is one, is c times as big. Only for a limit of 0."""
    check(fam.lim.plain == "0" and c >= 2, "scaling needs a zero limit")
    sup_fm = (
        Fm(f"{c}\\cdot {fam.sup_fm.tex}", f"{c} × {fam.sup_fm.plain}") if fam.sup_fm.plain not in ("1",) else Fm(str(c))
    )
    notes = dict(fam.notes)
    if "gap" in notes:
        notes["gap"] = c * notes["gap"]
    new = Family(
        key=fam.key + f"_x{c}",
        fn=Fm(f"{c}\\cdot " + fam.fn.tex, f"{c} × " + fam.fn.plain),
        lim=fam.lim,
        a=fam.a,
        b=fam.b,
        interval=fam.interval,
        interval_tex=fam.interval_tex,
        f_n=lambda n, x: c * fam.f_n(n, x),
        f=lambda x: c * fam.f(x),
        sup=lambda n: c * fam.sup(n),
        sup_fm=sup_fm,
        uniform=fam.uniform,
        argmax=fam.argmax,
        attained=fam.attained,
        int_fn=(lambda n: c * fam.int_fn(n)) if fam.int_fn else None,
        int_fn_fm=Fm(f"{c}\\cdot " + fam.int_fn_fm.tex, f"{c} × " + fam.int_fn_fm.plain) if fam.int_fn_fm else None,
        int_lim=0.0 if fam.int_lim is not None else None,
        bound=(lambda n: c * fam.bound(n)) if fam.bound else None,
        bound_fm=Fm(f"{c}\\cdot " + fam.bound_fm.tex, f"{c} × " + fam.bound_fm.plain) if fam.bound_fm else None,
        notes=notes,
    )
    return _verify(new)


def _interval(a, b) -> tuple[str, str]:
    return f"{_num(a)} ≤ x ≤ {_num(b)}", f"{_num(a)}\\le x\\le {_num(b)}"


# ---------------------------------------------------------------------------
# Uniformly convergent families
# ---------------------------------------------------------------------------
def power_over_n(k: int, b: int) -> Family:
    """f_n(x) = x^k / n on [0, b]; f = 0; s_n = b^k / n."""
    iv, ivt = _interval(0, b)
    kk = "" if k == 1 else f"^{k}"
    return _verify(
        Family(
            key=f"power_over_n_{k}_{b}",
            fn=Fm(f"\\frac{{x{'' if k == 1 else '^{' + str(k) + '}'}}}{{n}}", f"x{kk}/n"),
            lim=Fm("0"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: x**k / n,
            f=lambda x: 0.0,
            sup=lambda n: b**k / n,
            sup_fm=Fm(f"\\frac{{{b**k}}}{{n}}", f"{b**k}/n"),
            uniform=True,
            argmax=lambda n: float(b),
            int_fn=lambda n: b ** (k + 1) / ((k + 1) * n),
            int_fn_fm=Fm(f"\\frac{{{b ** (k + 1)}}}{{{k + 1}n}}", f"{b ** (k + 1)}/({k + 1}n)"),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: b**k / n,
            bound_fm=Fm(f"\\frac{{{b**k}}}{{n}}", f"{b**k}/n"),
        )
    )


def square_plus_over_n(b: int) -> Family:
    """f_n(x) = x^2 + x/n on [0, b]; f = x^2; s_n = b/n."""
    iv, ivt = _interval(0, b)
    return _verify(
        Family(
            key=f"square_plus_over_n_{b}",
            fn=Fm("x^{2}+\\frac{x}{n}", "x^2 + x/n"),
            lim=Fm("x^{2}", "x^2"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: x * x + x / n,
            f=lambda x: x * x,
            sup=lambda n: b / n,
            sup_fm=Fm(f"\\frac{{{b}}}{{n}}", f"{b}/n"),
            uniform=True,
            argmax=lambda n: float(b),
            int_fn=lambda n: b**3 / 3 + b * b / (2 * n),
            int_fn_fm=Fm(f"\\frac{{{b**3}}}{{3}}+\\frac{{{b * b}}}{{2n}}", f"{b**3}/3 + {b * b}/(2n)"),
            int_lim=b**3 / 3,
            int_lim_fm=Fm(f"\\frac{{{b**3}}}{{3}}", f"{b**3}/3"),
            bound=lambda n: b / n,
            bound_fm=Fm(f"\\frac{{{b}}}{{n}}", f"{b}/n"),
        )
    )


def cos_over_n(c: int, b: int) -> Family:
    """f_n(x) = cos(c x) / n on [0, b]; f = 0; s_n = 1/n (attained at x = 0)."""
    iv, ivt = _interval(0, b)
    cx = "x" if c == 1 else f"{c}x"
    return _verify(
        Family(
            key=f"cos_over_n_{c}_{b}",
            fn=Fm(f"\\frac{{\\cos {cx}}}{{n}}", f"cos({cx})/n"),
            lim=Fm("0"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: math.cos(c * x) / n,
            f=lambda x: 0.0,
            sup=lambda n: 1 / n,
            sup_fm=Fm("\\frac{1}{n}", "1/n"),
            uniform=True,
            argmax=lambda n: 0.0,
            int_fn=lambda n: math.sin(c * b) / (c * n),
            int_fn_fm=Fm(
                f"\\frac{{\\sin {c * b}}}{{{c}n}}" if c != 1 else f"\\frac{{\\sin {b}}}{{n}}",
                f"sin({c * b})/({c}n)" if c != 1 else f"sin({b})/n",
            ),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: 1 / n,
            bound_fm=Fm("\\frac{1}{n}", "1/n"),
        )
    )


def shifted_square(b: int) -> Family:
    """f_n(x) = (x + 1/n)^2 on [0, b]; f = x^2; s_n = 2b/n + 1/n^2."""
    iv, ivt = _interval(0, b)
    return _verify(
        Family(
            key=f"shifted_square_{b}",
            fn=Fm("\\left(x+\\frac{1}{n}\\right)^{2}", "(x + 1/n)^2"),
            lim=Fm("x^{2}", "x^2"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: (x + 1 / n) ** 2,
            f=lambda x: x * x,
            sup=lambda n: 2 * b / n + 1 / n**2,
            sup_fm=Fm(f"\\frac{{{2 * b}}}{{n}}+\\frac{{1}}{{n^{{2}}}}", f"{2 * b}/n + 1/n^2"),
            uniform=True,
            argmax=lambda n: float(b),
            int_fn=lambda n: ((b + 1 / n) ** 3 - (1 / n) ** 3) / 3,
            int_fn_fm=Fm(
                f"\\frac{{1}}{{3}}\\left(\\left({b}+\\frac{{1}}{{n}}\\right)^{{3}}-\\frac{{1}}{{n^{{3}}}}\\right)",
                f"((({b} + 1/n)^3) - 1/n^3)/3",
            ),
            int_lim=b**3 / 3,
            int_lim_fm=Fm(f"\\frac{{{b**3}}}{{3}}", f"{b**3}/3"),
            bound=lambda n: 2 * b / n + 1 / n**2,
            bound_fm=Fm(f"\\frac{{{2 * b}}}{{n}}+\\frac{{1}}{{n^{{2}}}}", f"{2 * b}/n + 1/n^2"),
        )
    )


def root_abs(b: int) -> Family:
    """f_n(x) = sqrt(x^2 + 1/n^2) on [-b, b]; f = |x|; s_n = 1/n (attained at x = 0)."""
    iv, ivt = _interval(-b, b)
    return _verify(
        Family(
            key=f"root_abs_{b}",
            fn=Fm("\\sqrt{x^{2}+\\frac{1}{n^{2}}}", "sqrt(x^2 + 1/n^2)"),
            lim=Fm("|x|"),
            a=-b,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: math.sqrt(x * x + 1 / n**2),
            f=lambda x: abs(x),
            sup=lambda n: 1 / n,
            sup_fm=Fm("\\frac{1}{n}", "1/n"),
            uniform=True,
            argmax=lambda n: 0.0,
            bound=lambda n: 1 / n,
            bound_fm=Fm("\\frac{1}{n}", "1/n"),
        )
    )


def rational_decay(b: int) -> Family:
    """f_n(x) = x / (1 + n x^2) on [0, b]; f = 0; s_n = 1 / (2 sqrt(n)) at x = 1/sqrt(n) (b >= 1)."""
    iv, ivt = _interval(0, b)
    return _verify(
        Family(
            key=f"rational_decay_{b}",
            fn=Fm("\\frac{x}{1+nx^{2}}", "x/(1 + n x^2)"),
            lim=Fm("0"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: x / (1 + n * x * x),
            f=lambda x: 0.0,
            sup=lambda n: 1 / (2 * math.sqrt(n)),
            sup_fm=Fm("\\frac{1}{2\\sqrt{n}}", "1/(2 sqrt(n))"),
            uniform=True,
            argmax=lambda n: 1 / math.sqrt(n),
            int_fn=lambda n: math.log(1 + n * b * b) / (2 * n),
            int_fn_fm=Fm(
                f"\\frac{{\\ln(1+{b * b}n)}}{{2n}}" if b > 1 else "\\frac{\\ln(1+n)}{2n}",
                f"ln(1 + {b * b}n)/(2n)" if b > 1 else "ln(1 + n)/(2n)",
            ),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: 1 / (2 * math.sqrt(n)),
            bound_fm=Fm("\\frac{1}{2\\sqrt{n}}", "1/(2 sqrt(n))"),
        )
    )


def power_below_one(num: int, den: int) -> Family:
    """f_n(x) = x^n on [0, p/q] with p/q < 1; f = 0; s_n = (p/q)^n at the right endpoint."""
    b = num / den
    iv = f"0 ≤ x ≤ {num}/{den}"
    ivt = f"0\\le x\\le \\frac{{{num}}}{{{den}}}"
    return _verify(
        Family(
            key=f"power_below_one_{num}_{den}",
            fn=Fm("x^{n}", "x^n"),
            lim=Fm("0"),
            a=0,
            b=b,
            interval=iv,
            interval_tex=ivt,
            f_n=lambda n, x: x**n,
            f=lambda x: 0.0,
            sup=lambda n: b**n,
            sup_fm=Fm(f"\\left(\\frac{{{num}}}{{{den}}}\\right)^{{n}}", f"({num}/{den})^n"),
            uniform=True,
            argmax=lambda n: b,
            int_fn=lambda n: b ** (n + 1) / (n + 1),
            int_fn_fm=Fm(
                f"\\frac{{1}}{{n+1}}\\left(\\frac{{{num}}}{{{den}}}\\right)^{{n+1}}", f"({num}/{den})^(n+1)/(n+1)"
            ),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: b**n,
            bound_fm=Fm(f"\\left(\\frac{{{num}}}{{{den}}}\\right)^{{n}}", f"({num}/{den})^n"),
            notes={"ratio": (num, den)},
        )
    )


def power_times_complement() -> Family:
    """f_n(x) = x^n (1 - x) on [0, 1]; f = 0; s_n = n^n / (n+1)^(n+1) at x = n/(n+1)."""
    return _verify(
        Family(
            key="power_times_complement",
            fn=Fm("x^{n}(1-x)", "x^n (1 - x)"),
            lim=Fm("0"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: x**n * (1 - x),
            f=lambda x: 0.0,
            sup=lambda n: math.exp(n * math.log(n) - (n + 1) * math.log(n + 1)),
            sup_fm=Fm("\\frac{n^{n}}{(n+1)^{n+1}}", "n^n/(n+1)^(n+1)"),
            uniform=True,
            argmax=lambda n: n / (n + 1),
            int_fn=lambda n: 1 / ((n + 1) * (n + 2)),
            int_fn_fm=Fm("\\frac{1}{(n+1)(n+2)}", "1/((n+1)(n+2))"),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: 1 / (n + 1),
            bound_fm=Fm("\\frac{1}{n+1}", "1/(n+1)"),
        )
    )


def power_over_n_unit() -> Family:
    """f_n(x) = x^n / n on [0, 1]; f = 0; s_n = 1/n at the endpoint x = 1."""
    return _verify(
        Family(
            key="power_over_n_unit",
            fn=Fm("\\frac{x^{n}}{n}", "x^n/n"),
            lim=Fm("0"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: x**n / n,
            f=lambda x: 0.0,
            sup=lambda n: 1 / n,
            sup_fm=Fm("\\frac{1}{n}", "1/n"),
            uniform=True,
            argmax=lambda n: 1.0,
            int_fn=lambda n: 1 / (n * (n + 1)),
            int_fn_fm=Fm("\\frac{1}{n(n+1)}", "1/(n(n+1))"),
            int_lim=0.0,
            int_lim_fm=Fm("0"),
            bound=lambda n: 1 / n,
            bound_fm=Fm("\\frac{1}{n}", "1/n"),
        )
    )


# ---------------------------------------------------------------------------
# Pointwise but not uniformly convergent families
# ---------------------------------------------------------------------------
def power_unit() -> Family:
    """f_n(x) = x^n on [0, 1]; f = 0 for x < 1 and 1 at x = 1; the supremum of the error over x < 1 is 1."""
    return _verify(
        Family(
            key="power_unit",
            fn=Fm("x^{n}", "x^n"),
            lim=Fm("0\\text{ for }0\\le x<1,\\ 1\\text{ for }x=1", "0 for 0 ≤ x < 1 and 1 at x = 1"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: x**n,
            f=lambda x: 1.0 if x == 1 else 0.0,
            sup=lambda n: 1.0,
            sup_fm=Fm("1"),
            uniform=False,
            argmax=lambda n: 1 - 1e-7,
            attained=False,
            notes={"witness": "x_n = 2^(-1/n)", "gap": 0.5, "witness_fn": lambda n: 2 ** (-1 / n)},
        )
    )


def spike_rational(c: int = 1) -> Family:
    """f_n(x) = n x / (1 + n^2 x^2) on [0, 1]; f = 0; s_n = 1/2 at x = 1/n."""
    return _verify(
        Family(
            key="spike_rational",
            fn=Fm("\\frac{nx}{1+n^{2}x^{2}}", "n x/(1 + n^2 x^2)"),
            lim=Fm("0"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: n * x / (1 + n * n * x * x),
            f=lambda x: 0.0,
            sup=lambda n: 0.5,
            sup_fm=Fm("\\frac{1}{2}", "1/2"),
            uniform=False,
            argmax=lambda n: 1 / n,
            notes={"witness": "x_n = 1/n", "gap": 0.5, "witness_fn": lambda n: 1 / n},
        )
    )


def step_rational() -> Family:
    """f_n(x) = n x / (1 + n x) on [0, 1]; f = 0 at x = 0 and 1 for x > 0; the supremum of the error is 1."""
    return _verify(
        Family(
            key="step_rational",
            fn=Fm("\\frac{nx}{1+nx}", "n x/(1 + n x)"),
            lim=Fm("0\\text{ at }x=0,\\ 1\\text{ for }0<x\\le 1", "0 at x = 0 and 1 for 0 < x ≤ 1"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: n * x / (1 + n * x),
            f=lambda x: 0.0 if x == 0 else 1.0,
            sup=lambda n: 1.0,
            sup_fm=Fm("1"),
            uniform=False,
            argmax=lambda n: 1e-9 / n,
            attained=False,
            notes={"witness": "x_n = 1/n", "gap": 0.5, "witness_fn": lambda n: 1 / n},
        )
    )


def spike_exponential() -> Family:
    """f_n(x) = n x e^(-n x) on [0, 1]; f = 0; s_n = 1/e at x = 1/n."""
    return _verify(
        Family(
            key="spike_exponential",
            fn=Fm("nxe^{-nx}", "n x e^(-n x)"),
            lim=Fm("0"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: n * x * math.exp(-n * x),
            f=lambda x: 0.0,
            sup=lambda n: math.exp(-1),
            sup_fm=Fm("e^{-1}", "e^(-1)"),
            uniform=False,
            argmax=lambda n: 1 / n,
            notes={"witness": "x_n = 1/n", "gap": math.exp(-1), "witness_fn": lambda n: 1 / n},
        )
    )


def power_difference() -> Family:
    """f_n(x) = x^n (1 - x^n) on [0, 1]; f = 0; s_n = 1/4 at x = 2^(-1/n)."""
    return _verify(
        Family(
            key="power_difference",
            fn=Fm("x^{n}(1-x^{n})", "x^n (1 - x^n)"),
            lim=Fm("0"),
            a=0,
            b=1,
            interval="0 ≤ x ≤ 1",
            interval_tex="0\\le x\\le 1",
            f_n=lambda n, x: x**n * (1 - x**n),
            f=lambda x: 0.0,
            sup=lambda n: 0.25,
            sup_fm=Fm("\\frac{1}{4}", "1/4"),
            uniform=False,
            argmax=lambda n: 2 ** (-1 / n),
            notes={"witness": "x_n = 2^(-1/n)", "gap": 0.25, "witness_fn": lambda n: 2 ** (-1 / n)},
        )
    )


UNIFORM_SIMPLE = [
    lambda rng: power_over_n(rng.choice((1, 2, 3)), rng.choice((1, 2, 3))),
    lambda rng: square_plus_over_n(rng.choice((1, 2, 3))),
    lambda rng: cos_over_n(rng.choice((1, 2, 3)), rng.choice((1, 2, 3))),
]
UNIFORM_CALCULUS = [
    lambda rng: shifted_square(rng.choice((1, 2, 3))),
    lambda rng: root_abs(rng.choice((1, 2, 3))),
    lambda rng: rational_decay(rng.choice((1, 2, 3))),
]
UNIFORM_HARD = [
    lambda rng: power_below_one(*rng.choice(((1, 2), (2, 3), (3, 4), (4, 5), (3, 5)))),
    lambda rng: power_times_complement(),
    lambda rng: power_over_n_unit(),
]
NOT_UNIFORM = [
    lambda rng: power_unit(),
    lambda rng: spike_rational(),
    lambda rng: step_rational(),
    lambda rng: spike_exponential(),
    lambda rng: power_difference(),
]
