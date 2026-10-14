"""Generates `s2c2_frobenius_indicial` questions (easy, medium, hard) and the exam chain
`exd_x_s2c2_frobenius_indicial`.

``uc2-frobenius-indicial`` (``ordinary-differential-equations``, answer ``fraction``): the Frobenius method at the
regular singular point x = 0 of  A x^2 y'' + x (B0 + B1 x + B2 x^2) y' + (C0 + C1 x + C2 x^2) y = 0, with a series
y = x^r (a_0 + a_1 x + a_2 x^2 + ...).

* easy: the larger root of the indicial equation A r(r - 1) + B0 r + C0 = 0. The higher terms of the equation are
  present but do not matter, and the leading coefficient A is not 1.
* medium: a_1 / a_0 for the larger root, from the recurrence. Half the questions are written as
  a x y'' + (b + c x) y' + d y = 0, which has to be multiplied by x to reach the standard form.
* hard: a_2 / a_0 for the smaller root. The roots do not differ by an integer, so both Frobenius series exist, and
  the recurrence has two terms in a_(n-1) and a_(n-2).
* exam (``exd_x_s2c2_frobenius_indicial``): (a) the larger root r1, (b) the smaller root r2 (follows through from
  (a), because the roots add to 1 - B0 / A), (c) a_1 / a_0 for r1 (follows through from (a)), (d) a_1 / a_0 for r2
  (follows through from (b)).

The coefficients come from the recurrence  I(r + n) a_n = -sum over k of (B_k (r + n - k) + C_k) a_(n-k),  where
I(s) = A s(s - 1) + B0 s + C0. SymPy then substitutes x^r (a_0 + a_1 x + a_2 x^2) into the differential equation
and checks that the residual vanishes through the order the truncation determines.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text
from ._s2_options import keep_answer_shape, spot_check
from ._ucv1_texsay import say_marked

x = sympy.Symbol("x", positive=True)


def indicial(a, b0, c0, s):
    return a * s * (s - 1) + b0 * s + c0


def _rational_roots(a, b0, c0):
    """The real roots of A s(s-1) + B0 s + C0 = 0 as Fractions when both are rational, else None."""
    qa, qb, qc = Fraction(a), Fraction(b0 - a), Fraction(c0)
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return None
    from math import isqrt

    num, den = isqrt(disc.numerator), isqrt(disc.denominator)
    if num * num != disc.numerator or den * den != disc.denominator:
        return None
    root = Fraction(num, den)
    return sorted([(-qb - root) / (2 * qa), (-qb + root) / (2 * qa)])


def coefficients(a, bs, cs, r, upto, *, sign=1, shift_b=False, n_for_r=False, drop_last=False):
    """a_0 = 1, a_1, ..., a_upto of the Frobenius series for exponent r, with optional slips."""
    out = [Fraction(1)]
    for n in range(1, upto + 1):
        total = Fraction(0)
        for k in range(1, n + 1):
            if drop_last and k == n and n >= 2:
                continue
            bk = bs[k] if k < len(bs) else 0
            ck = cs[k] if k < len(cs) else 0
            point = r + n - k + (1 if shift_b else 0)
            total += (bk * point + ck) * out[n - k]
        arg = Fraction(n) if n_for_r else r + n
        denom = indicial(a, bs[0], cs[0], arg)
        out.append(sign * -total / denom)
    return out


def _verify(a, bs, cs, r, coeffs):
    """Substitute x^r (a_0 + a_1 x + ...) into the equation: the residual must vanish through x^(len - 1)."""
    for k in range(len(coeffs)):
        # The residual's coefficient of x^(r+k), in exact arithmetic, for every exponent the series reaches.
        total = a * (r + k) * (r + k - 1) * coeffs[k]
        for i in range(k + 1):
            bi = bs[i] if i < len(bs) else 0
            ci = cs[i] if i < len(cs) else 0
            total += (bi * (r + k - i) + ci) * coeffs[k - i]
        check(total == 0, f"the series does not satisfy the equation at x^{k}")
    if not spot_check(a, tuple(bs), tuple(cs), r):
        return
    rs = sympy.Rational(r.numerator, r.denominator)
    series = sum(sympy.Rational(c.numerator, c.denominator) * x**k for k, c in enumerate(coeffs))
    y = x**rs * series
    poly = lambda cc: sum(sympy.Rational(c) * x**k for k, c in enumerate(cc))  # noqa: E731
    residual = a * x**2 * sympy.diff(y, x, 2) + x * poly(bs) * sympy.diff(y, x) + poly(cs) * y
    reduced = sympy.expand(sympy.powsimp(sympy.expand(residual / x**rs)))
    p = sympy.Poly(reduced, x)
    for k in range(0, len(coeffs)):
        check(p.coeff_monomial(x**k) == 0, f"the series does not satisfy the equation at x^{k}")


_POWER = {0: "", 1: "x", 2: "x²", 3: "x³"}


def _poly_in_x(coeffs):
    return join_terms([(int(c), _POWER[k]) for k, c in enumerate(coeffs) if c])


def _signed_term(coef, mono, deriv, first):
    """One term ``coef mono deriv`` of the equation, with its sign, as a student writes it."""
    magnitude = abs(coef)
    head = "" if (magnitude == 1 and mono) else str(magnitude)
    body = f"{head}{mono}{' ' if deriv and mono else ''}{deriv}"
    if not mono and not deriv:
        body = str(magnitude)
    if first:
        return f"-{body}" if coef < 0 else body
    return f"{'-' if coef < 0 else '+'} {body}"


def _standard_text(a, bs, cs):
    """A x² y'' + x (B0 + B1 x + B2 x²) y' + (C0 + C1 x + C2 x²) y = 0, with single terms written plainly."""
    pieces = [f"{'' if a == 1 else a}x² y''"]
    b_terms = [(k, c) for k, c in enumerate(bs) if c]
    if len(b_terms) == 1:
        k, c = b_terms[0]
        pieces.append(_signed_term(int(c), _POWER[k + 1], "y'", False))
    else:
        pieces.append(f"+ x({_poly_in_x(bs)}) y'")
    c_terms = [(k, c) for k, c in enumerate(cs) if c]
    if len(c_terms) == 1:
        k, c = c_terms[0]
        pieces.append(_signed_term(int(c), _POWER[k], "y", False))
    else:
        pieces.append(f"+ ({_poly_in_x(cs)}) y")
    return " ".join(pieces) + " = 0"


def _x_form_text(a, b, c, d):
    """a x y'' + (b + c x) y' + d y = 0, the standard form multiplied through by 1/x."""
    return f"{'' if a == 1 else a}x y'' + ({join_terms([(b, ''), (c, 'x')])}) y' {_signed_term(d, '', 'y', False)} = 0"


def _draw_roots(rng, *, integer_difference_ok=False, allow_unit_a=False):
    """(A, B0, C0, r_small, r_large) with rational roots and integer coefficients."""
    for _ in attempts(4000, "an indicial equation with rational roots"):
        a = rng.choice([1, 2, 2, 3, 4] if allow_unit_a else [2, 2, 3, 4])
        m1, m2 = rng.randint(-6, 6), rng.randint(-6, 6)
        if m1 == m2 or (m1 * m2) % a:
            continue
        if not integer_difference_ok and (m1 - m2) % a == 0:
            continue
        b0, c0 = a - m1 - m2, (m1 * m2) // a
        if abs(b0) > 8 or abs(c0) > 12:
            continue
        r1, r2 = sorted([Fraction(m1, a), Fraction(m2, a)])
        roots = _rational_roots(a, b0, c0)
        check(roots == [r1, r2], "the indicial roots are not the ones built")
        return a, b0, c0, r1, r2
    raise AssertionError("unreachable")


def _nonzero(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _root_candidates(a, b0, c0, answer, which):
    """Wrong values for a root of A s(s-1) + B0 s + C0 = 0 (`which` is "larger" or "smaller")."""
    r_small, r_large = _rational_roots(a, b0, c0)
    other = r_small if which == "larger" else r_large
    cands = [
        ("ode2-indicial-wrong-root-chosen", other),
        ("ode2-indicial-sign-error", -answer),
    ]
    unscaled = _rational_roots(1, b0, c0)  # leaving the leading coefficient out
    if unscaled:
        cands.append(("ode2-indicial-leading-coefficient-ignored", unscaled[1] if which == "larger" else unscaled[0]))
    dropped = _rational_roots_dropped(a, b0, c0)  # r^2 instead of r(r - 1)
    if dropped:
        cands.append(("ode2-indicial-r-minus-one-dropped", dropped[1] if which == "larger" else dropped[0]))
    cands.append(("ode2-indicial-root-sum-given", 1 - Fraction(b0, a)))
    return cands


def _rational_roots_dropped(a, b0, c0):
    from math import isqrt

    qa, qb, qc = Fraction(a), Fraction(b0), Fraction(c0)
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return None
    num, den = isqrt(disc.numerator), isqrt(disc.denominator)
    if num * num != disc.numerator or den * den != disc.denominator:
        return None
    root = Fraction(num, den)
    return sorted([(-qb - root) / (2 * qa), (-qb + root) / (2 * qa)])


def _slipped(a, bs, cs, r, upto, **slip):
    """a_upto under a named slip, or None when the slip makes a denominator zero."""
    try:
        return coefficients(a, bs, cs, r, upto, **slip)[upto]
    except ZeroDivisionError:
        return None


def _ratio_candidates(a, bs, cs, r, wrong_root, upto):
    """Wrong values for a_upto / a_0 at exponent r (those a slip cannot compute are left out)."""
    out = [
        ("ode2-frobenius-sign-slip", _slipped(a, bs, cs, r, upto, sign=-1)),
        ("ode2-frobenius-coefficient-index-shifted", _slipped(a, bs, cs, r, upto, shift_b=True)),
        ("ode2-frobenius-n-for-r-plus-n", _slipped(a, bs, cs, r, upto, n_for_r=True)),
        ("ode2-indicial-wrong-root-chosen", _slipped(a, bs, cs, wrong_root, upto)),
    ]
    if upto >= 2:
        out.append(("ode2-frobenius-term-left-out", _slipped(a, bs, cs, r, upto, drop_last=True)))
    return [(tag, v) for tag, v in out if v is not None]


def _draw(rng, tier):
    for _ in attempts(4000, "a Frobenius problem with clean coefficients"):
        if tier == "easy":
            a, b0, c0, r_small, r_large = _draw_roots(rng, integer_difference_ok=True, allow_unit_a=True)
            bs = [b0, _nonzero(rng, -3, 3), rng.choice([0, 0, 1])]
            cs = [c0, _nonzero(rng, -3, 3), rng.choice([0, 0, -1])]
            return dict(a=a, bs=bs, cs=cs, r_small=r_small, r_large=r_large, ask="root", form="standard")
        if tier == "medium":
            if rng.random() < 0.5:
                # a x y'' + (b + c x) y' + d y = 0: roots 0 and 1 - b/a
                a = rng.choice([2, 3, 4])
                b = _nonzero(rng, -7, 7)
                if b % a == 0:
                    continue
                c, d = _nonzero(rng, -3, 3), _nonzero(rng, -3, 3)
                bs, cs = [b, c], [0, d]
                roots = sorted([Fraction(0), 1 - Fraction(b, a)])
                return dict(
                    a=a, bs=bs, cs=cs, r_small=roots[0], r_large=roots[1], ask="a1", form="x", xform=(a, b, c, d)
                )
            a, b0, c0, r_small, r_large = _draw_roots(rng, integer_difference_ok=True)
            bs = [b0, _nonzero(rng, -3, 3)]
            cs = [c0, _nonzero(rng, -3, 3)]
            return dict(a=a, bs=bs, cs=cs, r_small=r_small, r_large=r_large, ask="a1", form="standard")
        a, b0, c0, r_small, r_large = _draw_roots(rng)
        bs = [b0, _nonzero(rng, -3, 3), _nonzero(rng, -2, 2)]
        cs = [c0, _nonzero(rng, -3, 3), _nonzero(rng, -2, 2)]
        return dict(a=a, bs=bs, cs=cs, r_small=r_small, r_large=r_large, ask="a2", form="standard")
    raise AssertionError("unreachable")


@generator(
    "s2c2_frobenius_indicial",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2c2_frobenius_indicial(rng, tier="easy"):
    for _ in attempts(200, "a Frobenius question with enough distractors"):
        spec = _draw(rng, tier)
        a, bs, cs = spec["a"], spec["bs"], spec["cs"]
        r_small, r_large = spec["r_small"], spec["r_large"]
        if spec["ask"] == "root":
            answer, cands = r_large, _root_candidates(a, bs[0], cs[0], r_large, "larger")
            question = "Find the larger root of the indicial equation."
            working = [
                f"The indicial equation is {a}r(r - 1) + ({bs[0]})r + ({cs[0]}) = 0, with roots {rational_text(r_small)} and {rational_text(r_large)}."
            ]
        else:
            upto = 1 if spec["ask"] == "a1" else 2
            r = r_large if upto == 1 else r_small
            coeffs = coefficients(a, bs, cs, r, upto)
            answer = coeffs[upto]
            _verify(a, bs, cs, r, coeffs)
            cands = _ratio_candidates(a, bs, cs, r, r_small if upto == 1 else r_large, upto)
            which = "larger" if upto == 1 else "smaller"
            question = f"Take the series with the {which} exponent r and a_0 = 1. Find a_{upto}."
            working = [
                f"The exponents are {rational_text(r_small)} and {rational_text(r_large)}; "
                f"for r = {rational_text(r)} the recurrence gives a_{upto} = {rational_text(answer)}."
            ]
        distractors = pick_distractors(
            rational_text(answer),
            [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in cands if v != answer],
            4,
        )
        if len(distractors) < 3:
            continue
        break
    equation = _x_form_text(*spec["xform"]) if spec["form"] == "x" else _standard_text(a, bs, cs)
    text, latex = say_marked(
        f"The differential equation «{equation}» has a regular singular point at «x = 0», "
        f"and a Frobenius series solution «y = x^r(a_0 + a_1 x + a_2 x² + …)». {question}"
    )
    return dict(
        text=text,
        latex=latex,
        answer_kind="fraction",
        answer=exact_value(answer),
        answer_text=rational_text(answer),
        distractors=distractors,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


@generator("exd_x_s2c2_frobenius_indicial", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c2_frobenius_indicial(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(60, "an exam question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    a, b0, c0, r_small, r_large = _draw_roots(rng)
    b1, c1 = _nonzero(rng, -3, 3), _nonzero(rng, -3, 3)
    bs, cs = [b0, b1], [c0, c1]
    if b1 * r_large + c1 == 0 or b1 * r_small + c1 == 0:
        raise _Redraw
    a1_large = coefficients(a, bs, cs, r_large, 1)[1]
    a1_small = coefficients(a, bs, cs, r_small, 1)[1]
    _verify(a, bs, cs, r_large, coefficients(a, bs, cs, r_large, 2))
    _verify(a, bs, cs, r_small, coefficients(a, bs, cs, r_small, 2))
    check(r_small + r_large == 1 - Fraction(b0, a), "the roots do not add to 1 - B0/A")

    def fraction_dist(correct, wrong):
        return pick_distractors(
            rational_text(correct),
            [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in wrong if v != correct],
            4,
        )

    a_dist = fraction_dist(r_large, _root_candidates(a, b0, c0, r_large, "larger"))
    b_dist = fraction_dist(r_small, _root_candidates(a, b0, c0, r_small, "smaller"))
    c_dist = fraction_dist(a1_large, _ratio_candidates(a, bs, cs, r_large, r_small, 1))
    d_dist = fraction_dist(a1_small, _ratio_candidates(a, bs, cs, r_small, r_large, 1))
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    equation = _standard_text(a, bs, cs)
    stem, stem_latex = say_marked(
        f"The differential equation «{equation}» has a regular singular point at «x = 0». "
        "It has two Frobenius series solutions «y = x^r(a_0 + a_1 x + a_2 x² + …)» with «a_0 = 1», "
        "one for each root «r» of the indicial equation."
    )

    def prompt(template):
        text, tex = say_marked(template)
        return dict(prompt=text, prompt_latex=tex)

    def ratio_ft(symbol):
        denominator = f"({a}*({symbol} + 1)*{symbol} + ({b0})*({symbol} + 1) + ({c0}))"
        return f"-(({b1})*{symbol} + ({c1}))/{denominator}"

    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt("Find the larger root «r_1» of the indicial equation."),
                answer_kind="fraction",
                answer=exact_value(r_large),
                answer_text=rational_text(r_large),
                marks=2,
                working=[
                    f"{a}r(r - 1) + ({b0})r + ({c0}) = 0 has roots {rational_text(r_small)} and {rational_text(r_large)}."
                ],
                distractors=a_dist,
            ),
            dict(
                **prompt("Find the smaller root «r_2»."),
                answer_kind="fraction",
                answer=exact_value(r_small),
                answer_text=rational_text(r_small),
                marks=1,
                follow_through=f"1 - ({b0})/({a}) - part_a",
                working=[f"The roots add to 1 - ({b0})/{a}, so r_2 = {rational_text(r_small)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt("For the series with exponent «r_1», find «a_1»."),
                answer_kind="fraction",
                answer=exact_value(a1_large),
                answer_text=rational_text(a1_large),
                marks=2,
                follow_through=ratio_ft("part_a"),
                working=[f"The recurrence at n = 1 gives a_1 = {rational_text(a1_large)}."],
                distractors=c_dist,
            ),
            dict(
                **prompt("For the series with exponent «r_2», find «a_1»."),
                answer_kind="fraction",
                answer=exact_value(a1_small),
                answer_text=rational_text(a1_small),
                marks=2,
                follow_through=ratio_ft("part_b"),
                working=[f"The recurrence at n = 1 gives a_1 = {rational_text(a1_small)}."],
                distractors=d_dist,
            ),
        ],
    )
