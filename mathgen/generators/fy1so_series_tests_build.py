"""Generates `fy1so_series_tests` (easy, medium, hard) and the exam chain `fy1so_exd_series_tests`.

Skill ``ucv1-series-comparison-tests`` (domain ``functions-sequences``, answer ``mcq``): deciding whether a series of
positive or alternating terms converges, by the p-series test, comparison and limit comparison, the integral test, and the
absolute and conditional convergence of alternating series. The geometric, ratio-test, telescoping and radius-of-convergence
questions belong to ``ucv1-series-convergence``.

Every answer is a statement, a verdict together with the fact that justifies it, and each wrong option is a different
statement that is false for a named reason, so exactly one option is true.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: a p-series, possibly written with roots (1/(n√n), 3/n^(2/3)). The learner reads off the exponent p, then compares
  it with 1. Wrong options misread p (inverted, shifted by one, or one factor dropped).
- ``medium``: a series whose terms behave like c/n^p (a ratio of polynomials, or a root), settled by limit comparison, and
  series settled by direct comparison with a bounded numerator (sin²n or 2 + cos n). The learner finds the comparison series
  from the degrees; the wrong options misjudge the degree difference, reverse the verdict or reverse the inequality.
- ``hard``: classify Σ(-1)^n a_n as absolutely convergent, conditionally convergent or divergent. The terms may fail to
  tend to 0, or Σ|a_n| may be a p-series on either side of 1, or need the integral test (1/(n ln n), 1/(n (ln n)²)).
- ``exam`` (topic ``fy1so_exd_series_tests``): a four-part chain on a_n = c/(n + b)^q (q a whole number of at least 2).
  (a) the improper integral from 1 to infinity of c/(x + b)^q dx, (b) the integral-test upper bound on the sum, a_1 plus (a)
  (follows through from (a)), (c) the partial sum S_3, (d) the sharper upper bound S_3 plus the integral from 3 (follows
  through from (c)).

Each behaviour (the exponent p, the constant c) is checked against the numeric slope of log a_n between n = 10^5 and
10^6, and the exam bounds are checked against a long partial sum.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import Link, M, build_part, build_single, plain, say, tex_rational

TOPIC = "fy1so_series_tests"
EXAM_TOPIC = "fy1so_exd_series_tests"


def _exp_tex(p: Fraction) -> str:
    """The exponent of n as TeX braces content: 2, 3/2."""
    return str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}"


def _term_tex(c: Fraction, p: Fraction, var: str = "n") -> str:
    """c/n^p as TeX: 3/n², 1/n, 3/(2n²) for a fractional c."""
    power = var if p == 1 else f"{var}^{{{_exp_tex(p)}}}"
    if c.denominator == 1:
        return rf"\frac{{{c.numerator}}}{{{power}}}"
    return rf"\frac{{{c.numerator}}}{{{c.denominator}{power}}}"


def _behaves(c: Fraction, p: Fraction) -> str:
    """c/n^p as learner text: 3/n², 1/n, 2/n^(3/2)."""
    return plain(_term_tex(c, p))


def _series_of(p: Fraction) -> str:
    return "Σ " + plain(_term_tex(Fraction(1), p))


def _verdict(p: Fraction) -> str:
    return "Converges" if p > 1 else "Diverges"


def _slope(term, ns=(10**5, 10**6)) -> float:
    """-d log a_n / d log n between two large n: the exponent p a term behaves like."""
    a, b = (math.log(abs(term(n))) for n in ns)
    return -(b - a) / (math.log(ns[1]) - math.log(ns[0]))


# -- easy: p-series ----------------------------------------------------------------------------------------------------


def _rule(converges: bool) -> str:
    return "p-series converge when p > 1" if converges else "p-series diverge when p ≤ 1"


def _p_statement(p: Fraction, flip: bool = False) -> str:
    """A verdict on a p-series with the rule it rests on; `flip` states the rule the wrong way round."""
    converges = p > 1
    shown = (not converges) if flip else converges
    return f"{'Converges' if shown else 'Diverges'}: a p-series with p = {rational_text(p)}, and {_rule(shown)}"


def _easy(rng: random.Random):
    kind = rng.choice(["power", "power", "root", "frac"])
    c = rng.choice([1, 1, 2, 3, 5])
    if kind == "power":
        p = rng.choice(
            [
                Fraction(1, 2),
                Fraction(2, 3),
                Fraction(3, 4),
                Fraction(1),
                Fraction(5, 4),
                Fraction(4, 3),
                Fraction(3, 2),
            ]
            + [Fraction(2), Fraction(5, 2), Fraction(3)]
        )
        body = (
            rf"\frac{{{c}}}{{n^{{{_exp_tex(p)}}}}}" if p.denominator != 1 else rf"\frac{{{c}}}{{n^{{{p.numerator}}}}}"
        )
        if p == 1:
            body = rf"\frac{{{c}}}{{n}}"
        term = lambda n: c / n ** float(p)  # noqa: E731
        misreads = [
            (1 / p, "series-p-exponent-inverted"),
            (p - 1, "series-p-exponent-lowered"),
            (p + 1, "series-p-exponent-raised"),
        ]
    elif kind == "root":
        k = rng.choice([1, 2, 3])
        r = rng.choice([2, 3])
        p = Fraction(k) + Fraction(1, r)
        root = r"\sqrt{n}" if r == 2 else r"\sqrt[3]{n}"
        body = rf"\frac{{{c}}}{{n^{{{k}}}\,{root}}}" if k > 1 else rf"\frac{{{c}}}{{n\,{root}}}"
        term = lambda n: c / (n**k * n ** (1 / r))  # noqa: E731
        misreads = [
            (Fraction(k), "series-p-root-factor-dropped"),
            (Fraction(1, r), "series-p-power-factor-dropped"),
            (Fraction(k * r), "series-p-root-index-multiplied"),
        ]
    else:
        m = rng.choice([1, 2, 3, 4, 5, 6, 7])
        r = rng.choice([2, 3])
        p = Fraction(m, r)
        if p.denominator == 1:
            m, r = rng.choice([(1, 2), (3, 2), (5, 2), (1, 3), (2, 3), (4, 3), (5, 3)])
            p = Fraction(m, r)
        body = rf"\frac{{{c}}}{{n^{{{m}/{r}}}}}"
        term = lambda n: c / n ** float(p)  # noqa: E731
        misreads = [
            (Fraction(m), "series-p-root-index-ignored"),
            (Fraction(r, m), "series-p-exponent-inverted"),
            (Fraction(m + r, r), "series-p-exponent-raised"),
        ]
    check(abs(_slope(term) - float(p)) < 1e-3, f"p = {p} disagrees with the numeric slope")
    answer = _p_statement(p)
    wrongs = []
    for q, tag in misreads:
        if q > 0 and q != p:
            wrongs.append((_p_statement(Fraction(q)), tag))
    # the same exponent with the test the wrong way round
    wrongs.insert(1, (_p_statement(p, flip=True), "series-p-test-reversed"))
    if p <= 1 and c:
        wrongs.append(("Converges: the terms tend to 0", "series-terms-tend-to-zero-misused"))
    ask = rng.choice(
        [
            "Which statement about the series is correct?",
            "Decide whether the series converges, and choose the correct statement.",
            "Which of the following is true of the series?",
        ]
    )
    prompt = say("Consider ", M(rf"\sum_{{n=1}}^{{\infty}} {body}"), ". " + ask)
    working = [
        f"The terms are {c}/n^{rational_text(p)}, so the series is {c} times a p-series with p = {rational_text(p)}.",
        f"A p-series converges when p > 1 and diverges when p ≤ 1. Here p {'>' if p > 1 else '≤'} 1, so it {_verdict(p).lower()}.",
    ]
    return prompt, Link(prompt, "mcq", answer, working=working, wrongs=wrongs)


# -- medium: comparison and limit comparison ---------------------------------------------------------------------------


def _lim_statement(c: Fraction, p: Fraction) -> str:
    return (
        f"{_verdict(p)}: the terms behave like {_behaves(c, p)}, and {_series_of(p)} "
        f"{'converges' if p > 1 else 'diverges'}"
    )


def _medium_limit(rng: random.Random):
    family = rng.choice(["lin_cubic", "quad_quartic", "lin_quad", "root_top", "root_bottom", "shifted"])
    b = rng.choice([1, 2, 3, 5])
    d = rng.choice([1, 2, 3, 4, 7])
    if family == "lin_cubic":
        a = rng.choice([1, 2, 3, 5])
        tex = rf"\frac{{{a}n + {b}}}{{n^{{3}} + {d}n}}"
        c, p = Fraction(a), Fraction(2)
        term = lambda n: (a * n + b) / (n**3 + d * n)  # noqa: E731
    elif family == "quad_quartic":
        a = rng.choice([2, 3, 5])
        g = rng.choice([1, 2, 3])
        tex = (
            rf"\frac{{{a}n^{{2}} + {b}}}{{{g}n^{{4}} + {d}}}"
            if g > 1
            else rf"\frac{{{a}n^{{2}} + {b}}}{{n^{{4}} + {d}}}"
        )
        c, p = Fraction(a, g), Fraction(2)
        term = lambda n: (a * n**2 + b) / (g * n**4 + d)  # noqa: E731
    elif family == "lin_quad":
        a = rng.choice([1, 2, 3, 4])
        g = rng.choice([1, 2, 3])
        tex = rf"\frac{{{a}n + {b}}}{{{g}n^{{2}} + {d}}}" if g > 1 else rf"\frac{{{a}n + {b}}}{{n^{{2}} + {d}}}"
        c, p = Fraction(a, g), Fraction(1)
        term = lambda n: (a * n + b) / (g * n**2 + d)  # noqa: E731
    elif family == "root_top":
        tex = rf"\frac{{\sqrt{{n}}}}{{n^{{2}} + {d}}}"
        c, p = Fraction(1), Fraction(3, 2)
        term = lambda n: math.sqrt(n) / (n**2 + d)  # noqa: E731
    elif family == "root_bottom":
        a = rng.choice([1, 2, 3])
        tex = (
            rf"\frac{{{a}n + {b}}}{{\sqrt{{n^{{3}} + {d}}}}}"
            if a > 1
            else rf"\frac{{n + {b}}}{{\sqrt{{n^{{3}} + {d}}}}}"
        )
        c, p = Fraction(a), Fraction(1, 2)
        term = lambda n: (a * n + b) / math.sqrt(n**3 + d)  # noqa: E731
    else:
        a = rng.choice([1, 2, 3])
        tex = rf"\frac{{{a}}}{{n^{{2}} + {b}n + {d}}}"
        c, p = Fraction(a), Fraction(2)
        term = lambda n: a / (n**2 + b * n + d)  # noqa: E731
    check(abs(_slope(term) - float(p)) < 1e-3, f"p = {p} disagrees with the numeric slope")
    big = 10**6
    check(
        abs(term(big) * big ** float(p) - float(c)) < 1e-4 * max(1, float(c)),
        f"c = {c} disagrees with the numeric limit",
    )
    answer = _lim_statement(c, p)
    # the degree difference misjudged by one, on whichever side flips the verdict
    shifted = p - 1 if (p - 1 > 0 and (p - 1 > 1) != (p > 1)) else p + 1
    wrongs = [
        (_lim_statement(c, shifted), "series-comparison-degree-difference-misread"),
        (
            f"{'Diverges' if p > 1 else 'Converges'}: the terms behave like {_behaves(c, p)}, and {_series_of(p)} "
            f"{'converges' if p > 1 else 'diverges'}",
            "series-comparison-verdict-reversed",
        ),
    ]
    if p > 1:
        wrongs.append((_lim_statement(c, Fraction(1)), "series-comparison-harmonic-assumed"))
        wrongs.append((_lim_statement(c, Fraction(1, 2)), "series-comparison-degree-difference-misread"))
    else:
        wrongs.append(("Converges: the terms tend to 0", "series-terms-tend-to-zero-misused"))
        wrongs.append((_lim_statement(c, Fraction(2)), "series-comparison-degree-difference-misread"))
    ask = rng.choice(
        [
            "Use limit comparison to decide which statement is correct.",
            "Which statement, justified by limit comparison, is correct?",
            "Decide whether the series converges. Which justification is correct?",
        ]
    )
    prompt = say("Consider ", M(rf"\sum_{{n=1}}^{{\infty}} {tex}"), ". " + ask)
    working = [
        f"For large n the terms behave like {_behaves(c, p)}: the ratio of a_n to 1/n^{rational_text(p)} tends to {rational_text(c)}.",
        f"{_series_of(p)} {'converges' if p > 1 else 'diverges'} (p {'>' if p > 1 else '≤'} 1), so by limit comparison the series {'converges' if p > 1 else 'diverges'}.",
    ]
    return prompt, Link(prompt, "mcq", answer, working=working, wrongs=wrongs)


def _medium_direct(rng: random.Random):
    below = rng.random() < 0.5
    d = rng.choice([1, 2, 3, 4, 5])
    if below:
        a_tex = rf"\frac{{(\sin n)^{{2}}}}{{n^{{2}} + {d}}}"
        fact = "0 ≤ a_n ≤ 1/n²"
        answer = f"Converges: {fact}, and Σ 1/n² converges"
        wrongs = [
            (f"Diverges: {fact}, and Σ 1/n² converges", "series-comparison-verdict-reversed"),
            ("Converges: a_n ≥ 1/n², and Σ 1/n² converges", "series-comparison-direction-wrong"),
            ("Diverges: a_n ≥ 1/n², and Σ 1/n² diverges", "series-comparison-verdict-reversed"),
            ("Converges: the terms tend to 0", "series-terms-tend-to-zero-misused"),
        ]
        working = [
            "Since 0 ≤ sin²n ≤ 1 and n² + " + str(d) + " > n², we have 0 ≤ a_n ≤ 1/n².",
            "Σ 1/n² converges, so by direct comparison the series converges.",
        ]
    else:
        k = rng.choice([1, 2, 3])
        a_tex = rf"\frac{{{k + 1} + \cos n}}{{n}}"
        fact = f"a_n ≥ {k}/n"
        answer = f"Diverges: {fact}, and Σ 1/n diverges"
        wrongs = [
            (f"Converges: {fact}, and Σ 1/n diverges", "series-comparison-verdict-reversed"),
            ("Converges: a_n ≤ 1/n, and Σ 1/n diverges", "series-comparison-direction-wrong"),
            ("Converges: the terms tend to 0", "series-terms-tend-to-zero-misused"),
            ("Converges: a_n ≤ 3/n, and Σ 1/n converges", "series-comparison-harmonic-assumed"),
        ]
        working = [
            f"Since cos n ≥ -1, a_n ≥ {k}/n > 0.",
            f"Σ {k}/n diverges (a multiple of the harmonic series), so by direct comparison the series diverges.",
        ]
    prompt = say(
        "Consider ",
        M(rf"\sum_{{n=1}}^{{\infty}} a_{{n}}"),
        " where ",
        M(rf"a_{{n}} = {a_tex}"),
        ". Use direct comparison to decide which statement is correct.",
    )
    return prompt, Link(prompt, "mcq", answer, working=working, wrongs=wrongs)


def _medium(rng: random.Random):
    return _medium_limit(rng) if rng.random() < 0.75 else _medium_direct(rng)


# -- hard: absolute and conditional convergence ------------------------------------------------------------------------

_ABS = "Absolutely convergent"
_COND = "Conditionally convergent"
_DIV = "Divergent"


def _classes(truth: str):
    """The two wrong classifications for a series whose truth is `truth`, each with the misconception behind it."""
    if truth == _ABS:
        return [(_COND, "series-alternating-means-conditional"), (_DIV, "series-absolute-terms-misjudged")]
    if truth == _COND:
        return [
            (_ABS, "series-alternating-test-taken-as-absolute"),
            (_DIV, "series-divergent-because-absolute-diverges"),
        ]
    return [(_COND, "series-nth-term-test-missed"), (_ABS, "series-nth-term-test-missed")]


def _hard(rng: random.Random):
    kind = rng.choice(["power", "power", "ratio", "log", "root"])
    if kind == "power":
        p = rng.choice(
            [
                Fraction(1, 2),
                Fraction(1, 3),
                Fraction(2, 3),
                Fraction(3, 4),
                Fraction(1),
                Fraction(4, 3),
                Fraction(3, 2),
                Fraction(2),
                Fraction(5, 2),
                Fraction(3),
            ]
        )
        tex = rf"\frac{{(-1)^{{n}}}}{{n^{{{_exp_tex(p)}}}}}" if p != 1 else r"\frac{(-1)^{n}}{n}"
        truth = _ABS if p > 1 else _COND
        start = 1
        reason = f"Σ|a_n| is a p-series with p = {rational_text(p)}, which {'converges' if p > 1 else 'diverges'}" + (
            "." if p > 1 else ", but the terms 1/n^p decrease to 0, so the alternating series test gives convergence."
        )
        term = lambda n: 1 / n ** float(p)  # noqa: E731
        slope = float(p)
    elif kind == "ratio":
        a = rng.choice([1, 2, 3])
        g = rng.choice([1, 2, 3, 5])
        b = rng.choice([1, 2, 3])
        tex = (
            rf"(-1)^{{n}}\,\frac{{{'' if a == 1 else a}n}}{{{g}n + {b}}}"
            if g > 1
            else rf"(-1)^{{n}}\,\frac{{{'' if a == 1 else a}n}}{{n + {b}}}"
        )
        truth = _DIV
        start = 1
        limit = Fraction(a, g)
        reason = f"|a_n| tends to {rational_text(limit)}, not 0, so the terms do not tend to 0 and the series diverges."
        term = None
        slope = None
    elif kind == "log":
        s = rng.choice([1, 2])
        tex = rf"\frac{{(-1)^{{n}}}}{{n\,\ln n}}" if s == 1 else rf"\frac{{(-1)^{{n}}}}{{n\,(\ln n)^{{2}}}}"
        truth = _COND if s == 1 else _ABS
        start = 2
        reason = (
            "By the integral test Σ 1/(n ln n) diverges (the integral of 1/(x ln x) is ln(ln x)), but the terms decrease to 0."
            if s == 1
            else "By the integral test Σ 1/(n (ln n)²) converges (the integral of 1/(x (ln x)²) is -1/ln x)."
        )
        term = None
        slope = None
    else:
        shape = rng.choice(["quad_over_quad", "sqrt_den", "sqrt_num", "cubic"])
        d = rng.choice([1, 2, 3])
        if shape == "quad_over_quad":
            tex, p = rf"\frac{{(-1)^{{n}}\,n}}{{n^{{2}} + {d}}}", Fraction(1)
            term = lambda n: n / (n**2 + d)  # noqa: E731
        elif shape == "sqrt_den":
            tex, p = rf"\frac{{(-1)^{{n}}}}{{\sqrt{{n^{{2}} + {d}}}}}", Fraction(1)
            term = lambda n: 1 / math.sqrt(n**2 + d)  # noqa: E731
        elif shape == "sqrt_num":
            tex, p = rf"\frac{{(-1)^{{n}}\sqrt{{n}}}}{{n + {d}}}", Fraction(1, 2)
            term = lambda n: math.sqrt(n) / (n + d)  # noqa: E731
        else:
            tex, p = rf"\frac{{(-1)^{{n}}\,n}}{{n^{{3}} + {d}}}", Fraction(2)
            term = lambda n: n / (n**3 + d)  # noqa: E731
        truth = _ABS if p > 1 else _COND
        start = 1
        reason = (
            f"|a_n| behaves like 1/n^{rational_text(p)}, so by limit comparison Σ|a_n| {'converges' if p > 1 else 'diverges'}"
            + ("." if p > 1 else ", but the terms decrease to 0, so the alternating series test gives convergence.")
        )
        slope = float(p)
    if term is not None and slope is not None:
        check(abs(_slope(term) - slope) < 1e-3, f"the behaviour of {tex} disagrees with the numeric slope")
        # the terms decrease from the start (needed for the alternating series test)
        check(all(term(n + 1) <= term(n) + 1e-15 for n in range(2 + start, 40)), "the terms do not decrease")
    wrongs = [(label, tag) for label, tag in _classes(truth)]
    prompt = say(
        "Classify the series ",
        M(rf"\sum_{{n={start}}}^{{\infty}} {tex}"),
        " as absolutely convergent, conditionally convergent or divergent.",
    )
    return prompt, Link(prompt, "mcq", truth, working=[reason], wrongs=wrongs)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_series_tests(rng: random.Random, tier: str = "easy") -> dict:
    """A convergence verdict with its justification: a p-series, comparison, or absolute versus conditional convergence."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain on the integral test -----------------------------------------------------------------------


def _term(c: int, b: int, q: int, n: int) -> Fraction:
    return Fraction(c, (n + b) ** q)


def _tail_integral(c: int, b: int, q: int, start: int) -> Fraction:
    """The integral from `start` to infinity of c/(x + b)^q dx."""
    return Fraction(c, (q - 1) * (start + b) ** (q - 1))


def _exam(rng: random.Random):
    for _ in attempts(100, "an integral-test chain"):
        c = rng.choice([1, 2, 3, 4, 5, 6])
        b = rng.choice([0, 1, 2, 3])
        q = rng.choice([2, 3, 4])
        break
    x = sympy.Symbol("x")
    integral = _tail_integral(c, b, q, 1)
    # second route: the antiderivative differentiates back to the integrand and vanishes at infinity
    antiderivative = -sympy.Rational(c, q - 1) / (x + b) ** (q - 1)
    check(
        sympy.simplify(sympy.diff(antiderivative, x) - sympy.Rational(c) / (x + b) ** q) == 0, "antiderivative is wrong"
    )
    check(sympy.limit(antiderivative, x, sympy.oo) == 0, "the antiderivative does not vanish at infinity")
    check(-antiderivative.subs(x, 1) == sympy.Rational(integral.numerator, integral.denominator), "integral disagrees")
    a1 = _term(c, b, q, 1)
    upper = a1 + integral
    s3 = sum((_term(c, b, q, n) for n in range(1, 4)), Fraction(0))
    sharper = s3 + _tail_integral(c, b, q, 3)
    # second route: a long partial sum with the integral tail brackets the sum, and both bounds hold
    partial = math.fsum(c / (n + b) ** q for n in range(1, 2001))
    tail = float(_tail_integral(c, b, q, 2000))
    estimate = partial + tail
    check(estimate <= float(sharper) <= float(upper) and float(upper) - float(sharper) > 0, "the bounds do not nest")
    check(
        float(s3) < estimate - 0 and estimate - float(s3) <= float(_tail_integral(c, b, q, 3)) + 1e-9,
        "the sum is outside S_3 and the tail bound",
    )

    def a_tex(n: str) -> str:
        base = n if b == 0 else f"({n} + {b})"
        return rf"\frac{{{c}}}{{{base}^{{{q}}}}}"

    stem = say(
        "Let ",
        M(rf"a_{{n}} = {a_tex('n')}"),
        " for ",
        M(r"n \ge 1"),
        ", and let ",
        M(r"S = \sum_{n=1}^{\infty} a_{n}"),
        ". The function ",
        M(rf"f(x) = {a_tex('x')}"),
        " is positive and decreasing for ",
        M(r"x \ge 1"),
        ", with ",
        M("f(n) = a_{n}"),
        ".",
    )
    p_a = say("Find ", M(r"\int_{1}^{\infty} f(x)\,dx"), " exactly.")
    p_b = say(
        "Hence write down the upper bound ",
        M(r"a_{1} + \int_{1}^{\infty} f(x)\,dx"),
        " that the integral test gives for ",
        M("S"),
        ".",
    )
    p_c = say("Find the partial sum ", M(r"S_{3} = a_{1} + a_{2} + a_{3}"), " exactly.")
    p_d = say(
        "Hence find the sharper upper bound ",
        M(r"S_{3} + \int_{3}^{\infty} f(x)\,dx"),
        " for ",
        M("S"),
        ".",
    )
    k = q - 1
    a_wrongs = [
        (Fraction(c, (1 + b) ** k), "series-integral-power-rule-slip"),
        (Fraction(1, k * (1 + b) ** k), "series-integral-constant-dropped"),
        (Fraction(c, q * (1 + b) ** q), "series-integral-power-rule-slip"),
        (Fraction(c, k * b**k) if b else None, "series-integral-wrong-lower-limit"),
        (a1, "series-integral-replaced-by-first-term"),
        (-integral, "series-integral-sign-slip"),
    ]
    b_wrongs = [
        (a1, "series-bound-integral-omitted"),
        (integral, "series-bound-first-term-omitted"),
        (a1 + _tail_integral(c, b, q, 0) if b else None, "series-integral-wrong-lower-limit"),
        (integral + _term(c, b, q, 2), "series-bound-wrong-first-term"),
    ]
    s2 = sum((_term(c, b, q, n) for n in range(1, 3)), Fraction(0))
    s4 = s3 + _term(c, b, q, 4)
    c_wrongs = [
        (s2, "series-partial-sum-terms-miscounted"),
        (s4, "series-partial-sum-terms-miscounted"),
        (sum((_term(c, b, q, n) for n in range(2, 5)), Fraction(0)), "series-partial-sum-terms-miscounted"),
        (3 * a1, "series-partial-sum-terms-miscounted"),
    ]
    d_wrongs = [
        (s3 + _tail_integral(c, b, q, 4), "series-tail-integral-start-off-by-one"),
        (s3 + _tail_integral(c, b, q, 2), "series-tail-integral-start-off-by-one"),
        (s3, "series-tail-integral-omitted"),
        (_tail_integral(c, b, q, 3), "series-partial-sum-omitted"),
        (s3 + _term(c, b, q, 4), "series-tail-integral-omitted"),
    ]
    ft_b = f"part_a + {a1.numerator}/{a1.denominator}"
    ft_d = f"part_c + {c}/({k}*{3 + b}**{k})"
    links = [
        Link(
            p_a,
            "fraction",
            integral,
            2,
            [f"The antiderivative is -{c}/({k}(x + {b})^{k}), so the integral from 1 to infinity is {integral}."],
            a_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_b,
            "fraction",
            upper,
            1,
            [f"a_1 = {a1}, so the bound is {a1} + {integral} = {upper}."],
            b_wrongs,
            ft=ft_b,
            sensible=(0, None),
        ),
        Link(
            p_c,
            "fraction",
            s3,
            2,
            [f"S_3 = {a1} + {_term(c, b, q, 2)} + {_term(c, b, q, 3)} = {s3}."],
            c_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_d,
            "fraction",
            sharper,
            2,
            [
                f"The integral from 3 to infinity is {_tail_integral(c, b, q, 3)}, so the bound is {s3} + {_tail_integral(c, b, q, 3)} = {sharper}."
            ],
            d_wrongs,
            ft=ft_d,
            sensible=(0, None),
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_series_tests(rng: random.Random) -> dict:
    """The integral test on c/(n + b)^q: the improper integral, the bound it gives, S_3 and the sharper bound."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
