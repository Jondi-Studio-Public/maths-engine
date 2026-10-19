"""Generates `r3b_exact_definite_integral` questions (easy, medium, hard) and the exam chain
`r3b_exam_exact_definite_integral`.

Skill ``definite-integral-exp-log-exact`` (``integration``, answer ``expression``): exact values of definite integrals
of exponential and reciprocal functions, left in terms of ``e`` and ``ln``.

* easy: a single term over limits that make one end simple: ``a e^(kx)`` from 0 to n, or ``a/x`` from 1 to n.
* medium: limits with neither end at the simple value (``c e^(kx)`` from a to b, ``c/(px + q)`` over a pair of limits,
  ``a/x + b`` from 1 to n), so both ends of the antiderivative are evaluated.
* hard: limits written as logarithms so that the exponentials collapse to whole numbers and the answer is a
  fraction (``e^(k ln n) = n^k``), or the average value of an exponential or reciprocal function over an interval.
* exam (``r3b_exam_exact_definite_integral``): two regions under one curve, with limits 0, ln p and ln q (or 1, r^i
  and r^j for ``c/x``): (a) and (b) the two areas, (c) the ratio of the second to the first, (d) the average value
  over the whole interval.

Each exact value is compared with Simpson's rule on 2000 strips.
"""

from __future__ import annotations

from sympy import E, Integer, Rational, exp, log

from ..core import attempts, check, fmt_signed, generator, latex_from_inline
from ._r3b_kit import Cand as _Cand
from ._r3b_kit import (
    Itg,
    Redraw,
    check_definite,
    exact_fields,
    frac_fields,
    integral_math,
    integral_text,
    part,
    pick_expr,
    pick_fractions,
    retry,
    say,
    shown,
    tex,
    x,
)


def Cand(expr, tag, const=False):  # noqa: N802 - exact values never carry a constant of integration
    return _Cand(expr, tag, const)


_FIND = (
    "Find the exact value of {I}.",
    "Evaluate {I}, leaving your answer in exact form.",
    "Calculate the exact value of {I}.",
)


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _result(rng, f, lo, hi, value, cands, working):
    check_definite(value, f.value, lo, hi)
    text, latex = say(rng, _FIND, {"I": integral_text(f, lo, hi)}, {"I": integral_math(f, lo, hi)})
    return dict(
        text=text,
        latex=latex,
        working=working,
        distractors=pick_expr(value, False, cands, rng=rng),
        **exact_fields(value),
    )


def _sgn(n) -> str:
    return "+" if n > 0 else "-"


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy_exp(rng):
    k = rng.choice([1, 2, 3, 4, -1, -2])
    a = _nonzero(rng, -8, 8)
    n = rng.choice([1, 2, 3])
    f = Itg(a * exp(k * x))
    value = Rational(a, k) * (exp(k * n) - 1)
    cands = [
        Cand(Rational(a, k) * exp(k * n), "r3b-def-lower-limit-ignored", const=False),
        Cand(a * (exp(k * n) - 1), "r3b-exp-integral-not-divided", const=False),
        Cand(Rational(a, k) * (1 - exp(k * n)), "r3b-def-limits-reversed", const=False),
        Cand(Rational(a, k) * exp(k * n) - 1, "r3b-def-lower-term-not-scaled", const=False),
    ]
    working = [f"The antiderivative is {Rational(a, k)}e^({fmt_signed(k, 'x')}). Evaluate at {n} and at 0: e^0 = 1."]
    return _result(rng, f, 0, n, value, cands, working)


def _easy_log(rng):
    a = _nonzero(rng, -9, 9)
    n = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
    f = Itg(a / x, text=f"{a}/x", latex=rf"\frac{{{a}}}{{x}}")
    value = a * log(n)
    cands = [
        Cand(a * log(n) - a, "r3b-def-ln-one-as-one", const=False),
        Cand(a * log(n - 1) if n > 2 else None, "r3b-def-limit-reduced-by-one"),
        Cand(log(n), "r3b-recip-coefficient-dropped", const=False),
        Cand(-a * log(n), "r3b-def-limits-reversed", const=False),
    ]
    return _result(rng, f, 1, n, value, cands, [f"{a}ln|x| from 1 to {n} is {a}ln {n} - ({a})ln 1, and ln 1 = 0."])


def _easy(rng):
    return rng.choice([_easy_exp, _easy_log])(rng)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium_exp(rng):
    k = rng.choice([2, 3, 4, -1, -2, 5])
    c = _nonzero(rng, -8, 8)
    a, b = rng.choice([(1, 2), (1, 3), (2, 3), (-1, 1), (-2, 1), (0, 2)][:5])
    f = Itg(c * exp(k * x))
    value = Rational(c, k) * (exp(k * b) - exp(k * a))
    cands = [
        Cand(Rational(c, k) * exp(k * (b - a)), "r3b-def-limits-subtracted-inside"),
        Cand(Rational(c, k) * (exp(k * b) - exp(a)), "r3b-def-lower-limit-not-scaled"),
        Cand(c * (exp(k * b) - exp(k * a)), "r3b-exp-integral-not-divided"),
        Cand(Rational(c, k) * (exp(k * a) - exp(k * b)), "r3b-def-limits-reversed"),
    ]
    cands = [Cand(cd.expr, cd.tag, False) for cd in cands]
    return _result(
        rng, f, a, b, value, cands, [f"Antiderivative {Rational(c, k)}e^({k}x); evaluate at {b} and at {a}."]
    )


def _medium_log(rng):
    p = rng.choice([2, 3, 4, 5])
    q = _nonzero(rng, -3, 5)
    c = _nonzero(rng, -9, 9)
    a = rng.choice([0, 1, 2])
    b = a + rng.choice([1, 2, 3])
    lo, hi = p * a + q, p * b + q
    if lo <= 0 or hi <= 0:
        raise Redraw("the denominator must stay positive on the interval")
    den = p * x + q
    ltxt = f"{p}x {'+' if q > 0 else '-'} {abs(q)}"
    f = Itg(c / den, text=f"{c}/({ltxt})", latex=rf"\frac{{{c}}}{{{ltxt}}}")
    value = Rational(c, p) * log(Rational(hi, lo))
    cands = [
        Cand(c * log(Rational(hi, lo)), "r3b-recip-not-divided"),
        Cand(Rational(c, p) * log(Rational(b, a)) if a else None, "r3b-def-log-of-limits"),
        Cand(Rational(c, p) * log(hi), "r3b-def-lower-limit-ignored"),
        Cand(Rational(c, p) * (log(hi) + log(lo)), "r3b-def-logs-added"),
        Cand(Rational(c, p) * log(Rational(hi - lo, 1)) if hi - lo != 1 else None, "r3b-def-log-of-width"),
    ]
    cands = [Cand(cd.expr, cd.tag, False) for cd in cands]
    return _result(
        rng,
        f,
        a,
        b,
        value,
        cands,
        [f"Antiderivative ({Rational(c, p)})ln|{ltxt}|; the values of {ltxt} at the limits are {lo} and {hi}."],
    )


def _medium_mixed(rng):
    a = _nonzero(rng, -8, 8)
    b = _nonzero(rng, -6, 6)
    n = rng.choice([2, 3, 4, 5, 6])
    f = Itg(a / x + b, text=f"{a}/x {_sgn(b)} {abs(b)}", latex=rf"\frac{{{a}}}{{x}} {_sgn(b)} {abs(b)}")
    value = a * log(n) + b * (n - 1)
    cands = [
        Cand(a * log(n) + b, "r3b-def-constant-not-integrated"),
        Cand(a * log(n) + b * n, "r3b-def-lower-limit-ignored"),
        Cand(a * log(n) - b * (n - 1), "r3b-def-constant-sign"),
        Cand(b * (n - 1), "r3b-recip-term-dropped"),
    ]
    cands = [Cand(cd.expr, cd.tag, False) for cd in cands]
    return _result(
        rng, f, 1, n, value, cands, [f"Antiderivative {a}ln|x| {_sgn(b)} {abs(b)}x; evaluate between 1 and {n}."]
    )


def _medium(rng):
    return rng.choice([_medium_exp, _medium_log, _medium_mixed])(rng)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_collapse(rng):
    k = rng.choice([1, 2, 3])
    a = _nonzero(rng, -8, 8)
    p, q = rng.choice([(1, 2), (2, 3), (2, 4), (3, 5), (1, 3), (2, 5)])
    f = Itg(a * exp(k * x))
    lo, hi = log(p), log(q)
    value = Rational(a, k) * (q**k - p**k)
    cands = [
        Cand(Rational(a, k) * (q - p), "r3b-def-exponent-k-ignored"),
        Cand(a * (q**k - p**k), "r3b-exp-integral-not-divided"),
        Cand(Rational(a, k) * (q**k - p), "r3b-def-lower-limit-not-scaled"),
        Cand(Rational(a, k) * (q**k + p**k), "r3b-def-limits-added"),
        Cand(Rational(a, k) * (k * log(q) - k * log(p)), "r3b-def-logs-not-collapsed"),
    ]
    return _result(
        rng, f, lo, hi, value, cands, [f"Antiderivative {Rational(a, k)}e^({k}x). Use e^({k} ln q) = q^{k}."]
    )


def _hard_two_terms(rng):
    a = _nonzero(rng, -6, 6)
    b = _nonzero(rng, -6, 6)
    n = rng.choice([2, 3, 4])
    f = Itg(a * exp(2 * x) + b * exp(x))
    value = Rational(a, 2) * (n**2 - 1) + b * (n - 1)
    cands = [
        Cand(a * (n**2 - 1) + b * (n - 1), "r3b-exp-integral-not-divided"),
        Cand(Rational(a, 2) * (n - 1) + b * (n - 1), "r3b-def-exponent-k-ignored"),
        Cand(Rational(a, 2) * n**2 + b * n, "r3b-def-lower-limit-ignored"),
        Cand(Rational(a, 2) * (n**2 - 1) - b * (n - 1), "r3b-def-constant-sign"),
    ]
    return _result(
        rng,
        f,
        0,
        log(n),
        value,
        cands,
        [f"Antiderivative {Rational(a, 2)}e^(2x) {_sgn(b)} {abs(b)}e^x; e^(2 ln {n}) = {n**2}."],
    )


def _hard_average(rng):
    if rng.random() < 0.5:
        k = rng.choice([1, 2, 3])
        a = _nonzero(rng, -8, 8)
        n = rng.choice([2, 3, 4, 5])
        f = Itg(a * exp(k * x))
        total = Rational(a, k) * (n**k - 1)
        value = total / log(n)
        cands = [
            Cand(total, "r3b-avg-not-divided-by-width"),
            Cand(total / n, "r3b-avg-width-taken-as-upper-limit"),
            Cand(total * log(n), "r3b-avg-multiplied-by-width"),
            Cand(Rational(a, k) * (n**k - 1) / (n - 1), "r3b-avg-width-taken-as-exponential-width"),
        ]
        lo, hi = 0, log(n)
        fx = f"{a}e^({k}x)" if a != 1 else f"e^({k}x)"
    else:
        c = _nonzero(rng, -8, 8)
        a = rng.choice([1, 2])
        b = rng.choice([4, 5, 6, 8, 9])
        f = Itg(c / x, text=f"{c}/x", latex=rf"\frac{{{c}}}{{x}}")
        total = c * log(Rational(b, a))
        value = total / (b - a)
        cands = [
            Cand(total, "r3b-avg-not-divided-by-width"),
            Cand(total / b, "r3b-avg-width-taken-as-upper-limit"),
            Cand(total / (b + a), "r3b-avg-width-taken-as-sum"),
            Cand(c * log(b - a) / (b - a), "r3b-def-log-of-width"),
        ]
        lo, hi = a, b
        fx = f"{c}/x"
    check_definite(total, f.value, lo, hi)
    text = f"Find the exact average value of f(x) = {fx} over the interval from x = {_t(lo)} to x = {_t(hi)}."
    latex = latex_from_inline(
        rf"Find the exact average value of \(f(x) = {f.latex}\) over the interval from \(x = {_tex(lo)}\) to \(x = {_tex(hi)}\)."
    )
    return dict(
        text=text,
        latex=latex,
        working=["Average value = (1 / (b - a)) times the integral of f from a to b."],
        distractors=pick_expr(value, False, [Cand(cd.expr, cd.tag, False) for cd in cands], rng=rng),
        **exact_fields(value),
    )


def _t(v) -> str:
    return shown(Integer(v) if isinstance(v, int) else v)


def _tex(v) -> str:
    return tex(Integer(v) if isinstance(v, int) else v)


def _hard(rng):
    return rng.choice([_hard_collapse, _hard_two_terms, _hard_average])(rng)


@generator(
    "r3b_exact_definite_integral",
    level=15,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_exact_definite_integral(rng, tier="easy"):
    """Exact definite integrals of exponentials and reciprocals; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


def _exp_chain(rng):
    k = rng.choice([1, 2, 3])
    amp = _nonzero(rng, 2, 9)
    p, q = rng.choice([(2, 3), (2, 4), (2, 5), (3, 4), (3, 6), (2, 6), (3, 5)])
    f = amp * exp(k * x)
    r1 = Rational(amp, k) * (p**k - 1)
    r2 = Rational(amp, k) * (q**k - p**k)
    ratio = r2 / r1
    total = r1 + r2
    check_definite(r1, f, 0, log(p))
    check_definite(r2, f, log(p), log(q))
    avg = total / log(q)
    fx = f"{amp}e^({k}x)" if k != 1 else f"{amp}e^x"
    stem = (
        f"Let f(x) = {fx}. The region R1 is bounded by the graph of y = f(x), the x-axis and the lines x = 0 and x = ln({p}). "
        f"The region R2 is bounded by the graph of y = f(x), the x-axis and the lines x = ln({p}) and x = ln({q})."
    )
    stem_latex = rf"Let \(f(x) = {tex(f)}\). The region \(R_1\) is bounded by the graph of \(y = f(x)\), the \(x\)-axis and the lines \(x = 0\) and \(x = \ln {p}\). The region \(R_2\) is bounded by the graph of \(y = f(x)\), the \(x\)-axis and the lines \(x = \ln {p}\) and \(x = \ln {q}\)."
    return (
        stem,
        stem_latex,
        r1,
        r2,
        ratio,
        avg,
        total,
        q,
        (
            [
                Cand(amp * (p**k - 1), "r3b-exp-integral-not-divided"),
                Cand(Rational(amp, k) * p**k, "r3b-def-lower-limit-ignored"),
                Cand(Rational(amp, k) * (p - 1), "r3b-def-exponent-k-ignored"),
                Cand(Rational(amp, k) * (1 - p**k), "r3b-def-limits-reversed"),
            ],
            [
                Cand(Rational(amp, k) * (q**k - p), "r3b-def-lower-limit-not-scaled"),
                Cand(Rational(amp, k) * q**k, "r3b-def-lower-limit-ignored"),
                Cand(amp * (q**k - p**k), "r3b-exp-integral-not-divided"),
                Cand(Rational(amp, k) * (q**k - 1), "r3b-def-region-boundary-ignored"),
            ],
        ),
        (
            [
                (r1 / r2, "r3b-chain-ratio-inverted"),
                (r2 - r1, "r3b-chain-difference-for-ratio"),
                (r2 / total, "r3b-chain-share-for-ratio"),
                (r2 * r1, "r3b-chain-product-for-ratio"),
            ],
            [
                Cand(total, "r3b-avg-not-divided-by-width"),
                Cand(total / q, "r3b-avg-width-taken-as-upper-limit"),
                Cand(total / (log(q) - log(p)), "r3b-avg-width-taken-as-second-region"),
                Cand(r2 / log(q), "r3b-chain-region-left-out"),
            ],
        ),
    )


def _log_chain(rng):
    r = rng.choice([2, 3])
    i, j = rng.choice([(1, 3), (2, 3), (1, 4), (2, 5), (3, 4)])
    c = _nonzero(rng, 2, 9)
    p, q = r**i, r**j
    f = c / x
    r1 = c * i * log(r)
    r2 = c * (j - i) * log(r)
    ratio = Rational(j - i, i)
    total = c * j * log(r)
    check_definite(r1, f, 1, p)
    check_definite(r2, f, p, q)
    avg = total / (q - 1)
    stem = (
        f"Let f(x) = {c}/x. The region R1 is bounded by the graph of y = f(x), the x-axis and the lines x = 1 and x = {p}. "
        f"The region R2 is bounded by the graph of y = f(x), the x-axis and the lines x = {p} and x = {q}."
    )
    stem_latex = rf"Let \(f(x) = \frac{{{c}}}{{x}}\). The region \(R_1\) is bounded by the graph of \(y = f(x)\), the \(x\)-axis and the lines \(x = 1\) and \(x = {p}\). The region \(R_2\) is bounded by the graph of \(y = f(x)\), the \(x\)-axis and the lines \(x = {p}\) and \(x = {q}\)."
    return (
        stem,
        stem_latex,
        r1,
        r2,
        ratio,
        avg,
        total,
        q,
        (
            [
                Cand(c * log(r), "r3b-def-ln-limits-confused"),
                Cand(c * log(p) - c, "r3b-def-ln-one-as-one"),
                Cand(c * log(p - 1) if p > 2 else None, "r3b-def-limit-reduced-by-one"),
                Cand(log(p), "r3b-recip-coefficient-dropped"),
            ],
            [
                Cand(c * log(q), "r3b-def-region-boundary-ignored"),
                Cand(c * log(q - p), "r3b-def-log-of-width"),
                Cand(c * (j - i), "r3b-def-logs-not-collapsed"),
            ],
        ),
        (
            [
                (Rational(i, j - i), "r3b-chain-ratio-inverted"),
                (j - i - i, "r3b-chain-difference-for-ratio"),
                (Rational(j - i, j), "r3b-chain-share-for-ratio"),
                (Rational(q, p), "r3b-chain-limits-ratio-for-area-ratio"),
            ],
            [
                Cand(total, "r3b-avg-not-divided-by-width"),
                Cand(total / q, "r3b-avg-width-taken-as-upper-limit"),
                Cand(total / (q + 1), "r3b-avg-width-taken-as-sum"),
                Cand(r2 / (q - 1), "r3b-chain-region-left-out"),
            ],
        ),
    )


@generator("r3b_exam_exact_definite_integral", level=15, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_exact_definite_integral(rng, tier="exam"):
    """Exam chain: two areas under one curve, their ratio and the average value over both."""
    check(tier == "exam", f"unknown tier {tier!r}")
    exp_variant = rng.random() < 0.5
    stem, stem_latex, r1, r2, ratio, avg, total, q, (c1, c2), (ratio_c, avg_c) = (
        _exp_chain if exp_variant else _log_chain
    )(rng)
    check(ratio == r2 / r1 or exp_variant is False, "ratio mismatch")
    if exp_variant:
        fields1, fields2 = frac_fields(r1), frac_fields(r2)
        pa = part(
            "Find the exact area of R1.",
            "Find the exact area of \\(R_1\\).",
            2,
            ["Integrate f(x) from the first limit to the second."],
            fields1,
            pick_fractions(r1, [(cd.expr, cd.tag) for cd in c1], rng=rng),
        )
        pb = part(
            "Find the exact area of R2.",
            "Find the exact area of \\(R_2\\).",
            2,
            ["Integrate f(x) between the second and third limits."],
            fields2,
            pick_fractions(r2, [(cd.expr, cd.tag) for cd in c2], rng=rng),
        )
        follow_a = follow_b = None
    else:
        pa = part(
            "Find the exact area of R1.",
            "Find the exact area of \\(R_1\\).",
            2,
            ["Integrate f(x) from the first limit to the second."],
            exact_fields(r1),
            pick_expr(r1, False, [Cand(cd.expr, cd.tag, False) for cd in c1], rng=rng),
        )
        pb = part(
            "Find the exact area of R2.",
            "Find the exact area of \\(R_2\\).",
            2,
            ["Integrate f(x) between the second and third limits."],
            exact_fields(r2),
            pick_expr(r2, False, [Cand(cd.expr, cd.tag, False) for cd in c2], rng=rng),
        )
    pc = part(
        "Find the ratio of the area of R2 to the area of R1, as a fraction.",
        "Find the ratio of the area of \\(R_2\\) to the area of \\(R_1\\), as a fraction.",
        2,
        ["Divide the area of R2 by the area of R1; the common factor cancels."],
        frac_fields(ratio),
        pick_fractions(ratio, ratio_c, rng=rng),
        follow_through="part_b/part_a",
    )
    pd = part(
        "Find the exact average value of f over the interval covered by R1 and R2 together.",
        "Find the exact average value of \\(f\\) over the interval covered by \\(R_1\\) and \\(R_2\\) together.",
        2,
        ["Average value = (area of R1 + area of R2) divided by the width of the interval."],
        exact_fields(avg),
        pick_expr(avg, False, avg_c, rng=rng),
        follow_through=("(part_a+part_b)/log(%d)" % q) if exp_variant else ("(part_a+part_b)/(%d-1)" % q),
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
