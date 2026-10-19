"""Generates `u2_complex_equations` questions and the exam chain `exd_x_u2_complex_equations`.

``ucv1-complex-equations-division`` (``complex-analysis``, answer ``expression``): dividing complex numbers in Cartesian
form and solving equations in a complex unknown. Loci are not asked.

* easy: (a + bi)/(c + di) in the form x + yi.
* medium: an equation that mixes z and its conjugate, A z + B z̄ = C, solved for z; or a linear equation with complex
  coefficients, (a + bi) z + c = d, which needs a division as its last step.
* hard: the square root with positive real part of a + bi (found by equating real and imaginary parts), or the root of
  a quadratic with complex coefficients that has the larger imaginary part (the discriminant is itself a complex square).
* exam: z = (p + qi)^2 is the solution of A z + B z̄ = C. (a) z, (b) 1/z in Cartesian form (follows through from (a)),
  (c) the square root of z with positive real part.

Every answer is re-derived with SymPy: the quotient is multiplied back, the equation is substituted into, and the
square root is squared.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, rational_text
from ._u2_kit import keep, lin, redraws

TOPIC = "u2_complex_equations"
EXAM_TOPIC = "exd_x_u2_complex_equations"

CONJ = "̄"  # combining overline, written after the z


def zbar():
    return "z" + CONJ


def _q(v):
    v = Fraction(v)
    return sympy.Rational(v.numerator, v.denominator)


def cx(re, im):
    """The SymPy value re + im*I."""
    return _q(re) + _q(im) * sympy.I


def _coef(v: Fraction, unit: str) -> str:
    """`v` times `unit` as a learner types it: i, -i, 3i, (3/2)i."""
    v = Fraction(v)
    if v == 1:
        return unit
    if v == -1:
        return "-" + unit
    if v.denominator == 1:
        return f"{v.numerator}{unit}"
    sign = "-" if v < 0 else ""
    return f"{sign}({abs(v.numerator)}/{v.denominator}){unit}"


def cx_text(re, im) -> str:
    """a + bi with the sign of b between the terms; a pure real or pure imaginary number has one term."""
    re, im = Fraction(re), Fraction(im)
    if im == 0:
        return rational_text(re)
    if re == 0:
        return _coef(im, "i")
    sign = "+" if im > 0 else "-"
    return f"{rational_text(re)} {sign} {_coef(abs(im), 'i')}"


def cx_dist(re, im, tag):
    """A distractor for the value re + im i."""
    return Distractor(cx_text(re, im), tag, sympy.sstr(cx(re, im)))


def _conj_lhs(a, b):
    """'3z + 2z̄', 'z - z̄', '-z + 2z̄'."""
    first = {1: "", -1: "-"}.get(a, str(a)) + "z"
    second = ("" if abs(b) == 1 else str(abs(b))) + zbar()
    return f"{first} {'+' if b > 0 else '-'} {second}"


def _answer(re, im):
    value = cx(re, im)
    return dict(
        answer_kind="expression",
        answer=sympy.sstr(value),
        answer_text=cx_text(re, im),
        answer_latex="plain:" + cx_text(re, im),
    )


def _signed_term(re, im, var: str = "") -> str:
    """`` + 2z`` / `` - iz`` / `` + (3 - i)``: a complex coefficient added to an expression, as written on paper."""
    re, im = Fraction(re), Fraction(im)
    if im == 0:
        sign, body = ("-" if re < 0 else "+"), _coef(abs(re), "") if var else rational_text(abs(re))
    elif re == 0:
        sign, body = ("-" if im < 0 else "+"), _coef(abs(im), "i")
    else:
        # A bracketed complex coefficient: subtract (3 + i)z rather than add (-3 - i)z.
        if re < 0 and im < 0:
            return f" - ({cx_text(-re, -im)}){var}"
        return f" + ({cx_text(re, im)}){var}"
    return f" {sign} {body}{var}"


def _bi(b) -> str:
    """The imaginary part's size followed by i, leaving out a coefficient of 1: ``i``, ``3i``."""
    return "i" if b == 1 else f"{b}i"


def _lit(re, im):
    """A complex number written in a stem, in brackets: (3 + 4i)."""
    return f"({cx_text(re, im)})"


def divide(a, b, c, d):
    """(a + bi)/(c + di) as (re, im) Fractions."""
    den = c * c + d * d
    return Fraction(a * c + b * d, den), Fraction(b * c - a * d, den)


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a division with a non-trivial denominator"):
        a, b = rng.randint(-6, 7), rng.randint(-6, 7)
        c, d = rng.randint(-4, 4), rng.randint(-4, 4)
        if 0 in (a, b, c, d) or abs(c) == abs(d) and rng.random() < 0.6:
            continue
        den = c * c + d * d
        if den in (2, 8):  # keep the denominators distinctive
            continue
        re, im = divide(a, b, c, d)
        if im == 0 or re == 0:
            continue
        break
    check(sympy.expand(cx(re, im) * cx(c, d)) == cx(a, b), "quotient times divisor differs from the dividend")
    check(sympy.expand(sympy.radsimp(cx(a, b) / cx(c, d)) - cx(re, im)) == 0, "SymPy's quotient differs")
    cands = [
        cx_dist(*(Fraction(a * c - b * d, den), Fraction(b * c + a * d, den)), "complex-div-conjugate-not-used"),
        cx_dist(*(Fraction(a * c - b * d, den), Fraction(b * c - a * d, den)), "complex-i-squared-sign"),
        cx_dist(Fraction(a, c), Fraction(b, d), "complex-div-by-parts"),
        cx_dist(
            *(Fraction(a * c + b * d, abs(c * c - d * d) or 1), Fraction(b * c - a * d, abs(c * c - d * d) or 1)),
            "complex-div-denominator-difference",
        ),
        cx_dist(re, -im, "complex-div-conjugate-result"),
    ]
    text = f"Write {_lit(a, b)} / {_lit(c, d)} in the form x + yi, where x and y are real."
    text = text.replace(" / ", " ÷ ")
    working = [
        f"Multiply top and bottom by the conjugate {_lit(c, -d)}: the denominator becomes {abs(c)}² + {abs(d)}² = {den}.",
        f"The numerator {_lit(a, b)}{_lit(c, -d)} = {cx_text(a * c + b * d, b * c - a * d)}, so the quotient is {cx_text(re, im)}.",
    ]
    return dict(text=text, working=working, distractors=keep(cx_text(re, im), cands, 4), **_answer(re, im))


# -- medium --------------------------------------------------------------------------------------------------------


def _medium_conjugate(rng):
    for _ in attempts(500, "a conjugate equation with a clean solution"):
        a, b = rng.choice([1, 2, 3, 4, -1, -2]), rng.choice([1, 2, 3, -1, -2, -3])
        x, y = rng.randint(-5, 5), rng.randint(-5, 5)
        if 0 in (x, y) or a + b == 0 or a - b == 0 or a == b:
            continue
        break
    c_re, c_im = (a + b) * x, (a - b) * y
    # Independent route: substitute the answer back, with SymPy's own conjugate.
    zv = cx(x, y)
    check(
        sympy.expand(a * zv + b * sympy.conjugate(zv) - cx(c_re, c_im)) == 0,
        "substitution does not satisfy the equation",
    )
    left = _conj_lhs(a, b)
    cands = [
        cx_dist(Fraction(c_re, a + b), -Fraction(c_im, a - b), "complex-eq-conjugate-answer"),
        cx_dist(Fraction(c_re, a + b), Fraction(c_im, a + b), "complex-eq-treat-conjugate-as-z"),
        cx_dist(Fraction(c_re, a - b), Fraction(c_im, a - b), "complex-eq-sign-conjugate-part"),
        cx_dist(Fraction(c_im, a - b), Fraction(c_re, a + b), "complex-eq-parts-swapped"),
    ]
    text = f"Find the complex number z such that {left} = {cx_text(c_re, c_im)}."
    working = [
        f"Write z = x + yi, so {zbar()} = x - yi. The left side is {lin(a + b, 0, 'x')} {'+' if a - b > 0 else '-'} {lin(abs(a - b), 0, 'y')}i.",
        f"Equating real and imaginary parts: {lin(a + b, 0, 'x')} = {c_re} and {lin(a - b, 0, 'y')} = {c_im}, so z = {cx_text(x, y)}.",
    ]
    return dict(text=text, working=working, distractors=keep(cx_text(x, y), cands, 4), **_answer(x, y))


def _medium_linear(rng):
    for _ in attempts(500, "a linear equation with a non-integer solution"):
        a, b = rng.randint(1, 4), rng.randint(-4, 4)
        c, d = rng.randint(1, 5), rng.randint(-5, 5)
        if b == 0 or d == 0:
            continue
        # (a + bi) z + c = d + 0i forms are too bare: put a complex right-hand side e + fi.
        e, f = rng.randint(-6, 6), rng.randint(-6, 6)
        if f == 0:
            continue
        re, im = divide(e - c, f - d, a, b)
        break
    value = cx(re, im)
    check(sympy.expand(cx(a, b) * value + cx(c, d) - cx(e, f)) == 0, "substitution does not satisfy the equation")
    den = a * a + b * b
    cands = [
        cx_dist(*divide(e + c, f + d, a, b), "complex-eq-sign-constant"),
        cx_dist(*divide(e - c, f - d, a, -b), "complex-div-conjugate-not-used"),
        cx_dist(Fraction(e - c, a), Fraction(f - d, b), "complex-div-by-parts"),
        cx_dist(
            Fraction((e - c) * a - (f - d) * b, den), Fraction((f - d) * a + (e - c) * b, den), "complex-i-squared-sign"
        ),
    ]
    text = f"Solve {_lit(a, b)}z + {_lit(c, d)} = {cx_text(e, f)} for z, giving z in the form x + yi."
    working = [
        f"Subtract {_lit(c, d)}: {_lit(a, b)}z = {cx_text(e - c, f - d)}.",
        f"Divide by {_lit(a, b)} using its conjugate: z = {cx_text(re, im)}.",
    ]
    return dict(text=text, working=working, distractors=keep(cx_text(re, im), cands, 4), **_answer(re, im))


def _medium(rng):
    return rng.choice([_medium_conjugate, _medium_linear, _medium_conjugate])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------


def _hard_sqrt(rng):
    for _ in attempts(500, "a perfect complex square"):
        p, q = rng.randint(1, 5), rng.choice([v for v in range(-5, 6) if v])
        if abs(p) == abs(q):
            continue
        break
    a, b = p * p - q * q, 2 * p * q
    check(sympy.expand(cx(p, q) ** 2) == cx(a, b), "square differs")
    check(sympy.expand(cx(-p, -q) ** 2) == cx(a, b), "the other root differs")
    cands = [
        cx_dist(-p, -q, "complex-sqrt-wrong-branch"),
        cx_dist(p, -q, "complex-sqrt-sign-imaginary"),
        cx_dist(q, p, "complex-sqrt-parts-swapped"),
    ]
    text = f"Find the square root of {cx_text(a, b)} that has a positive real part, giving it in the form x + yi."
    working = [
        f"Let (x + yi)² = {cx_text(a, b)}. Then x² - y² = {a} and 2xy = {b}.",
        f"The solution with x > 0 is x = {p}, y = {q}: z = {cx_text(p, q)}.",
    ]
    return dict(text=text, working=working, distractors=keep(cx_text(p, q), cands, 4), **_answer(p, q))


def _hard_quadratic(rng):
    for _ in attempts(500, "a quadratic with two roots of different imaginary parts"):
        r1 = (rng.randint(-4, 4), rng.randint(-4, 4))
        r2 = (rng.randint(-4, 4), rng.randint(-4, 4))
        if r1[1] == r2[1] or r1 == r2 or 0 in (r1[1], r2[1]):
            continue
        break
    hi, lo = (r1, r2) if r1[1] > r2[1] else (r2, r1)
    s = (hi[0] + lo[0], hi[1] + lo[1])
    prod = cx(*hi) * cx(*lo)
    for root in (hi, lo):
        check(sympy.expand(cx(*root) ** 2 - cx(*s) * cx(*root) + prod) == 0, "a root does not satisfy the quadratic")
    cands = [
        cx_dist(*lo, "complex-quadratic-other-root"),
        cx_dist(hi[0], -hi[1], "complex-quadratic-conjugate-root"),
        cx_dist(-hi[0], -hi[1], "complex-quadratic-sign-root"),
        cx_dist(Fraction(s[0], 2), Fraction(s[1], 2), "complex-quadratic-midpoint"),
        cx_dist(lo[0], -lo[1], "complex-quadratic-conjugate-root"),
    ]
    mid = sympy.nsimplify(prod)
    mid_re, mid_im = int(sympy.re(mid)), int(sympy.im(mid))
    text = (
        f"The equation z²{_signed_term(-s[0], -s[1], 'z')}{_signed_term(mid_re, mid_im)} = 0 has two distinct roots. "
        "Find the root with the larger imaginary part."
    )
    disc = cx(*s) ** 2 - 4 * prod
    working = [
        f"The discriminant is {_lit(*s)}² - 4{_lit(mid_re, mid_im)} = {cx_text(sympy.re(sympy.expand(disc)), sympy.im(sympy.expand(disc)))}, "
        f"which equals ({cx_text(hi[0] - lo[0], hi[1] - lo[1])})².",
        f"So z = ({_lit(*s)} ± ({cx_text(hi[0] - lo[0], hi[1] - lo[1])}))/2, and the root with the larger imaginary part is {cx_text(*hi)}.",
    ]
    return dict(text=text, working=working, distractors=keep(cx_text(*hi), cands, 4), **_answer(*hi))


def _hard(rng):
    return rng.choice([_hard_sqrt, _hard_quadratic])(rng)


@generator(
    TOPIC,
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@redraws
def u2_complex_equations(rng, tier="easy"):
    """Division and equations in a complex unknown; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_u2_complex_equations(rng, tier="exam"):
    """w = (k + bi)/(c + di) with k real: the k that makes Re w = Im w, w itself, then A z + B conj(z) = (uk, v + k)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a division chain with integer k"):
        b, c, d = rng.randint(-5, 6), rng.randint(-4, 4), rng.randint(-4, 4)
        if 0 in (b, c, d) or c + d == 0 or (b * (c - d)) % (c + d) != 0 or c == d:
            continue
        k = b * (c - d) // (c + d)
        if k in (0, b):
            continue
        if (
            sum(
                1
                for num, quot in ((b * (c + d), c - d), (-b * (c + d), c - d), (b * c, d), (-b * d, c))
                if quot and num % quot == 0 and num // quot not in (k, -k, b)
            )
            < 1
        ):
            continue
        a_, b_ = rng.choice([2, 3, 4]), rng.choice([-3, -2, -1, 1, 2])
        if a_ + b_ == 0 or a_ == b_:
            continue
        break
    den = c * c + d * d
    w_re, w_im = Fraction(k * c + b * d, den), Fraction(b * c - k * d, den)
    check(w_re == w_im, "Re w differs from Im w")
    check(sympy.expand(cx(w_re, w_im) * cx(c, d) - (k + b * sympy.I)) == 0, "SymPy's quotient differs")
    for _ in attempts(100, "a right-hand side with a non-real solution"):
        u, v = rng.choice([1, 2, 3, -1, -2]), rng.randint(-4, 4)
        if v + k != 0:
            break
    z_re, z_im = Fraction(u * k, a_ + b_), Fraction(v + k, a_ - b_)
    zv = cx(z_re, z_im)
    check(
        sympy.expand(a_ * zv + b_ * sympy.conjugate(zv) - (u * k + (v + k) * sympy.I)) == 0,
        "substitution does not satisfy the equation",
    )

    ka_cands = [Distractor(str(-k), "complex-eq-sign-constant", -k), Distractor(str(b), "complex-div-by-parts", b)]
    for num, quot, tag in (
        (b * (c + d), c - d, "complex-eq-parts-swapped"),  # the conjugate's signs used for the imaginary part
        (-b * (c + d), c - d, "complex-eq-sign-conjugate-part"),  # real part equal to minus the imaginary part
        (b * c, d, "complex-eq-treat-conjugate-as-z"),  # imaginary part set to zero
        (-b * d, c, "complex-eq-treat-conjugate-as-z"),  # real part set to zero
    ):
        if quot and num % quot == 0:
            ka_cands.append(Distractor(str(num // quot), tag, num // quot))
    wr, wi = Fraction(k * c - b * d, den), Fraction(b * c + k * d, den)
    kb_cands = [
        cx_dist(wr, wi, "complex-div-conjugate-not-used"),
        cx_dist(w_re, -w_im, "complex-div-conjugate-result"),
        cx_dist(
            Fraction(k * c + b * d, abs(c * c - d * d) or 1),
            Fraction(b * c - k * d, abs(c * c - d * d) or 1),
            "complex-div-denominator-difference",
        ),
    ]
    kc_cands = [
        cx_dist(z_re, -z_im, "complex-eq-conjugate-answer"),
        cx_dist(Fraction(u * k, a_ - b_), Fraction(v + k, a_ + b_), "complex-eq-sign-conjugate-part"),
        cx_dist(Fraction(v + k, a_ - b_), z_re, "complex-eq-parts-swapped"),
    ]
    u_text = {1: "", -1: "-"}.get(u, str(u))
    rhs_text = f"{u_text}k + ({v} + k)i" if v >= 0 else f"{u_text}k + (k - {-v})i"
    left = _conj_lhs(a_, b_)
    ft_b = f"(part_a*{c} + {b * d})/{den} + I*({b * c} - part_a*{d})/{den}".replace("+ -", "- ")
    ft_c = f"{u}*part_a/{a_ + b_} + I*({v} + part_a)/{a_ - b_}"
    return dict(
        text=f"Let k be a real number and let w = (k {'+' if b > 0 else '-'} {_bi(abs(b))}) ÷ {_lit(c, d)}.",
        parts=[
            dict(
                prompt="Find the value of k for which the real part of w equals its imaginary part.",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=3,
                working=[
                    f"Multiplying by the conjugate: w = ((k {'+' if b > 0 else '-'} {_bi(abs(b))}){_lit(c, -d)})/{den}.",
                    f"Re w = ({lin(c, b * d)})/{den} and Im w = ({lin(-d, b * c)})/{den}; equating them gives k = {k}.",
                ],
                distractors=keep(str(k), ka_cands),
            ),
            dict(
                prompt="Write w in the form x + yi for this value of k.",
                marks=3,
                working=[
                    f"With k = {k}, w = ({k} {'+' if b > 0 else '-'} {_bi(abs(b))})({cx_text(c, -d)})/{den} = {cx_text(w_re, w_im)}."
                ],
                distractors=keep(cx_text(w_re, w_im), kb_cands),
                follow_through=ft_b,
                **_answer(w_re, w_im),
            ),
            dict(
                prompt=f"With this value of k, solve {left} = {rhs_text} for z, giving z in the form x + yi.",
                marks=3,
                working=[
                    f"With k = {k} the right side is {cx_text(u * k, v + k)}; z = x + yi gives {lin(a_ + b_, 0, 'x')} = {u * k} and {lin(a_ - b_, 0, 'y')} = {v + k}.",
                ],
                distractors=keep(cx_text(z_re, z_im), kc_cands),
                follow_through=ft_c,
                **_answer(z_re, z_im),
            ),
        ],
    )
