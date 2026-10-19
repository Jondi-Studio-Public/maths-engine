"""Generates `r3g_multiple_angle` questions (easy, medium, hard) and the exam chain `r3g_exam_multiple_angle`.

Skill ``de-moivre-multiple-angle`` (``trigonometry``, answer ``expression``): De Moivre's theorem used to expand
``(cos θ + i sin θ)^n`` and read off a multiple angle in terms of powers of sine and cosine, and the reverse, a power in
terms of multiple angles from ``(z + 1/z)^n``. Integral powers of one complex number are a different skill.

* easy: ``cos 3θ`` or ``sin 3θ`` as a fraction, given ``cos θ`` for an acute θ (the other ratio comes from a Pythagorean
  triple; the real or imaginary part of ``(cos θ + i sin θ)^3`` is the value).
* medium: ``cos 4θ`` or ``sin 4θ`` from the fourth power, given ``sin θ`` for an acute θ.
* hard: a coefficient in either direction: the coefficient of ``cos^k θ`` in ``cos nθ`` (n = 4 to 6), or the coefficient
  of one multiple angle in ``cos^n θ`` or ``sin^n θ`` (n = 3 to 6), including the constant term.
* exam (``r3g_exam_multiple_angle``): ``cos θ`` given for an acute θ; (a) sin θ, (b) cos 3θ, (c) sin 3θ (each
  following through from (a)) and (d) tan 3θ (following through from (b) and (c)).

Every value is confirmed a second way: the multiple angle is evaluated with ``math`` at the angle the ratio defines, and
the coefficients are checked against SymPy's Chebyshev polynomials or by evaluating both sides at several angles.
"""

from __future__ import annotations

import math

import sympy as sp

from ..core import attempts, check, generator, ordinal
from ._r3g_kit import ex, latex, pick, retry_short, typed, vary_symbol

R = sp.Rational
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(n: int) -> str:
    return str(n).translate(_SUP)


def _acute_pair(rng):
    p, q, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        p, q = q, p
    return R(p, h), R(q, h)  # (sin, cos) of an acute angle


def _expand_power(c, s, n: int):
    """(real, imaginary) parts of (c + i s)^n by the binomial theorem."""
    z = sp.expand((c + sp.I * s) ** n)
    return sp.re(z), sp.im(z)


def _binomial_parts(c, s, n: int):
    real = sum(sp.binomial(n, k) * (-1) ** (k // 2) * c ** (n - k) * s**k for k in range(0, n + 1, 2))
    imag = sum(sp.binomial(n, k) * (-1) ** ((k - 1) // 2) * c ** (n - k) * s**k for k in range(1, n + 1, 2))
    return real, imag


def _multiple(c, s, n: int, fn: str):
    real, imag = _binomial_parts(c, s, n)
    theta = math.atan2(float(s), float(c))
    value = real if fn == "cos" else imag
    check(
        abs(float(value) - (math.cos(n * theta) if fn == "cos" else math.sin(n * theta))) < 1e-9,
        "multiple angle differs from the float",
    )
    return value


def _frame(rng, given: str, ask: str, n: int) -> str:
    """The stem in one of three framings: the plain statement, an expansion, or the power of a complex number."""
    variant = rng.randrange(3)
    if variant == 0:
        return f"Angle θ is acute and {given}. Use De Moivre's theorem to find {ask}."
    if variant == 1:
        return f"Given that θ is acute and {given}, expand (cos θ + i sin θ){sup(n)} to find {ask}."
    return f"The complex number z = cos θ + i sin θ has an acute argument θ with {given}. By working out z{sup(n)}, find {ask}."


def _easy(rng):
    s, c = _acute_pair(rng)
    fn = rng.choice(("cos", "sin"))
    value = _multiple(c, s, 3, fn)
    if fn == "cos":
        stem = _frame(rng, f"cos θ = {typed(c)}", "cos 3θ", 3)
        cands = [
            ex(c**3 + 3 * c * s**2, "r3g-multiple-angle-i-squared-sign"),
            ex(c**3 - 2 * c * s**2, "r3g-multiple-angle-binomial-coefficient"),
            ex(3 * c, "r3g-multiple-angle-multiplied-ratio"),
            ex(3 * c**2 * s - s**3, "r3g-multiple-angle-parts-swapped"),
        ]
        working = [
            f"sin θ = {typed(s)} because θ is acute.",
            "(cos θ + i sin θ)³ has real part cos³θ - 3 cos θ sin²θ, and by De Moivre's theorem this is cos 3θ.",
            f"cos 3θ = {typed(value)}.",
        ]
    else:
        stem = _frame(rng, f"cos θ = {typed(c)}", "sin 3θ", 3)
        cands = [
            ex(3 * c**2 * s + s**3, "r3g-multiple-angle-i-squared-sign"),
            ex(3 * c * s**2 - c**3, "r3g-multiple-angle-parts-swapped"),
            ex(3 * s, "r3g-multiple-angle-multiplied-ratio"),
            ex(c**3 - 3 * c * s**2, "r3g-multiple-angle-parts-swapped"),
        ]
        working = [
            f"sin θ = {typed(s)} because θ is acute.",
            "(cos θ + i sin θ)³ has imaginary part 3 cos²θ sin θ - sin³θ, and this is sin 3θ.",
            f"sin 3θ = {typed(value)}.",
        ]
    text = typed(value)
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=working,
        distractors=pick(text, cands, 3),
    )


def _medium(rng):
    s, c = _acute_pair(rng)
    fn = rng.choice(("cos", "sin"))
    value = _multiple(c, s, 4, fn)
    if fn == "cos":
        cands = [
            ex(c**4 + 6 * c**2 * s**2 + s**4, "r3g-multiple-angle-i-squared-sign"),
            ex(c**4 - 4 * c**2 * s**2 + s**4, "r3g-multiple-angle-binomial-coefficient"),
            ex(4 * c**3 * s - 4 * c * s**3, "r3g-multiple-angle-parts-swapped"),
            ex(4 * c, "r3g-multiple-angle-multiplied-ratio"),
        ]
        working = [
            f"sin θ = {typed(s)} because θ is acute.",
            "The real part of (cos θ + i sin θ)⁴ is cos⁴θ - 6 cos²θ sin²θ + sin⁴θ, which is cos 4θ.",
        ]
    else:
        cands = [
            ex(4 * c**3 * s + 4 * c * s**3, "r3g-multiple-angle-i-squared-sign"),
            ex(c**4 - 6 * c**2 * s**2 + s**4, "r3g-multiple-angle-parts-swapped"),
            ex(4 * c**3 * s - 2 * c * s**3, "r3g-multiple-angle-binomial-coefficient"),
            ex(4 * s, "r3g-multiple-angle-multiplied-ratio"),
        ]
        working = [
            f"cos θ = {typed(c)} because θ is acute.",
            "The imaginary part of (cos θ + i sin θ)⁴ is 4 cos³θ sin θ - 4 cos θ sin³θ, which is sin 4θ.",
        ]
    given = f"sin θ = {typed(s)}"
    text = typed(value)
    working.append(f"{fn} 4θ = {text}.")
    return dict(
        text=_frame(rng, given, f"{fn} 4θ", 4),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=working,
        distractors=pick(text, cands, 3),
    )


# -- hard -------------------------------------------------------------------------------------------------------


def laurent_coefficients(fn: str, n: int) -> dict[int, sp.Expr]:
    """{j: coefficient of z**j} for (z + 1/z)**n / 2**n (cos) or ((z - 1/z) / (2 i))**n (sin), z = e^(i theta)."""
    z = sp.Symbol("z")
    base = (z + 1 / z) / 2 if fn == "cos" else (z - 1 / z) / (2 * sp.I)
    poly = sp.Poly(sp.expand(base**n * z**n), z)
    out = {}
    for (power,), coeff in poly.terms():
        out[power - n] = sp.nsimplify(sp.simplify(coeff))
    return out


def power_to_multiples(fn: str, n: int) -> dict[tuple[str, int], sp.Rational]:
    """{(basis, j): coefficient} with fn^n = sum coeff * basis(j theta); basis is cos or sin, j = 0 is the constant."""
    lc = laurent_coefficients(fn, n)
    out = {}
    for j in range(0, n + 1):
        plus, minus = lc.get(j, 0), lc.get(-j, 0)
        if j == 0:
            if plus != 0:
                out[("cos", 0)] = sp.nsimplify(plus)
            continue
        cos_coeff = sp.nsimplify(sp.simplify(plus + minus))
        sin_coeff = sp.nsimplify(sp.simplify(sp.I * (plus - minus)))
        if cos_coeff != 0:
            check(sin_coeff == 0, "a power of one function should use only one basis")
            out[("cos", j)] = cos_coeff
        if sin_coeff != 0:
            check(cos_coeff == 0, "a power of one function should use only one basis")
            out[("sin", j)] = sin_coeff
    return out


def _check_power_expansion(fn: str, n: int, terms) -> None:
    for theta in (0.3, 0.9, 1.7, 2.6):
        lhs = getattr(math, fn)(theta) ** n
        rhs = sum(
            float(c) * (math.cos(j * theta) if basis == "cos" else math.sin(j * theta))
            for (basis, j), c in terms.items()
        )
        check(abs(lhs - rhs) < 1e-9, f"{fn}^{n} expansion differs at {theta}")


def _hard_power(rng):
    fn = rng.choice(("cos", "sin"))
    n = rng.randint(3, 6)
    terms = power_to_multiples(fn, n)
    _check_power_expansion(fn, n, terms)
    keys = sorted(terms, key=lambda k: -k[1])
    pick_key = rng.choice(keys)
    basis, j = pick_key
    value = terms[pick_key]
    shown_basis = "constant term" if j == 0 else f"{basis} {j}θ" if j > 1 else f"{basis} θ"
    others = [terms[k] for k in keys if k != pick_key]
    listing = " + ".join(
        (f"A{idx} " + (f"{b} {jj}θ" if jj > 1 else f"{b} θ")) if jj else f"A{idx}"
        for idx, (b, jj) in enumerate(keys, 1)
    )
    del listing
    cands = [
        ex(value / 2, "r3g-multiple-angle-paired-terms-halved"),
        ex(value * 2, "r3g-multiple-angle-paired-terms-doubled"),
        ex(
            sp.binomial(n, (n - j) // 2) if (n - j) % 2 == 0 else sp.binomial(n, (n - j + 1) // 2),
            "r3g-multiple-angle-binomial-without-power-of-two",
        ),
        ex(-value, "r3g-multiple-angle-i-squared-sign"),
    ]
    cands += [ex(o, "r3g-multiple-angle-wrong-term") for o in others[:2]]
    if j == 0:
        stem = (
            f"Use De Moivre's theorem to write {fn}{sup(n)}θ as a sum of multiples of cosine. Find the constant term."
        )
    else:
        stem = (
            f"Use De Moivre's theorem to write {fn}{sup(n)}θ as a sum of "
            f"{'sine' if basis == 'sin' else 'cosine'} terms of multiple angles. "
            f"Find the coefficient of {basis} {j}θ."
            if j > 1
            else f"Use De Moivre's theorem to write {fn}{sup(n)}θ as a sum of {'sine' if basis == 'sin' else 'cosine'} terms of "
            f"multiple angles. Find the coefficient of {basis} θ."
        )
    del shown_basis
    text = typed(value)
    return dict(
        text=stem,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=[
            f"With z = cos θ + i sin θ, {fn} θ = {'(z + 1/z)/2' if fn == 'cos' else '(z - 1/z)/(2i)'}.",
            f"Expand the {ordinal(n)} power by the binomial theorem and pair z^k with z^-k using 2 cos kθ = z^k + z^-k and 2i sin kθ = z^k - z^-k.",
            f"The required coefficient is {text}.",
        ],
        distractors=pick(text, cands, 3),
    )


def _cos_poly_text(poly) -> str:
    """The polynomial in c = cos θ as the learner writes it: 32cos⁶θ - 48cos⁴θ + 18cos²θ - 1."""
    out = []
    for (power,), coef in sorted(poly.terms(), reverse=True):
        body = "" if power == 0 else "cos θ" if power == 1 else f"cos{sup(power)}θ"
        mag = abs(coef)
        shown = f"{mag}{body}" if (mag != 1 or not body) else body
        out.append(f"-{shown}" if not out and coef < 0 else shown if not out else f"{'-' if coef < 0 else '+'} {shown}")
    return " ".join(out)


def _hard_multiple(rng):
    n = rng.choice((4, 5, 6))
    c = sp.Symbol("c")
    s2 = 1 - c**2
    cos_n = sp.expand(
        sum(sp.binomial(n, k) * (-1) ** (k // 2) * c ** (n - k) * s2 ** (k // 2) for k in range(0, n + 1, 2))
    )
    check(sp.expand(cos_n - sp.chebyshevt(n, c)) == 0, "cos n theta differs from the Chebyshev polynomial")
    poly = sp.Poly(cos_n, c)
    powers = sorted(k for (k,) in [m for m in poly.monoms()])
    k = rng.choice([p for p in powers if p not in (n,)] or powers)
    value = poly.coeff_monomial(c**k)
    cands = [
        ex(value * -1, "r3g-multiple-angle-i-squared-sign"),
        ex(sp.binomial(n, n - k), "r3g-multiple-angle-binomial-coefficient"),
        ex(poly.coeff_monomial(c**n), "r3g-multiple-angle-wrong-term"),
        ex(value * 2, "r3g-multiple-angle-paired-terms-doubled"),
        ex(value // 2 if value % 2 == 0 else value + 1, "r3g-multiple-angle-wrong-term"),
    ]
    power_text = f"cos{sup(k)}θ" if k > 1 else "cos θ" if k == 1 else None
    text = typed(value)
    return dict(
        text=(
            f"Use De Moivre's theorem and sin²θ = 1 - cos²θ to write cos {n}θ as a polynomial in cos θ. "
            + (f"Find the coefficient of {power_text}." if power_text else "Find the constant term.")
        ),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex(value),
        working=[
            f"cos {n}θ is the real part of (cos θ + i sin θ){sup(n)}: the terms with an even power of sin θ.",
            "Replace sin²θ by 1 - cos²θ in each and collect powers of cos θ.",
            f"cos {n}θ = {_cos_poly_text(poly)}; the {'coefficient of ' + power_text if power_text else 'constant term'} is {text}.",
        ],
        distractors=pick(text, cands, 3),
    )


@generator(
    "r3g_multiple_angle",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
@vary_symbol("θ", ("θ", "φ", "α"))
def r3g_multiple_angle(rng, tier="easy"):
    """Multiple angles and powers of sine and cosine from De Moivre's theorem; tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_power, _hard_power, _hard_multiple])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_multiple_angle", level=12, presets={"exam": dict(tier="exam")})
@retry_short
@vary_symbol("θ", ("θ", "φ", "α"))
def r3g_exam_multiple_angle(rng, tier="exam"):
    """Exam chain: sin θ, cos 3θ, sin 3θ (from sin θ) and tan 3θ (from the other two)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(100, "a nonzero cos 3θ"):
        s, c = _acute_pair(rng)
        cos3 = _multiple(c, s, 3, "cos")
        sin3 = _multiple(c, s, 3, "sin")
        if cos3 != 0 and sin3 != 0:
            break
    tan3 = sin3 / cos3
    ft_cos = f"({typed(c)})^3 - 3*({typed(c)})*part_a^2"
    ft_sin = f"3*({typed(c)})^2*part_a - part_a^3"

    def part(prompt, value, marks, working, cands, follow=None):
        text = typed(value)
        d = dict(
            prompt=prompt,
            answer_kind="expression",
            answer=text,
            answer_text=text,
            answer_latex=latex(value),
            marks=marks,
            working=working,
            distractors=pick(text, cands, 3),
        )
        if follow:
            d["follow_through"] = follow
        return d

    return dict(
        text=[
            f"Angle θ is acute and cos θ = {typed(c)}.",
            f"Let θ be an acute angle with cos θ = {typed(c)}.",
            f"The complex number z = cos θ + i sin θ has an acute argument θ and cos θ = {typed(c)}.",
        ][rng.randrange(3)],
        parts=[
            part(
                "Find sin θ.",
                s,
                1,
                [f"sin²θ = 1 - {typed(c**2)} = {typed(s**2)}, and sin θ > 0 because θ is acute."],
                [
                    ex(-s, "r3g-multiple-angle-sign-of-root"),
                    ex(1 - c, "r3g-multiple-angle-pythagorean-slip"),
                    ex(s**2, "r3g-multiple-angle-root-not-taken"),
                    ex(c, "r3g-multiple-angle-parts-swapped"),
                ],
            ),
            part(
                "Use De Moivre's theorem to find cos 3θ.",
                cos3,
                2,
                [f"cos 3θ = cos³θ - 3 cos θ sin²θ = {typed(cos3)}."],
                [
                    ex(c**3 + 3 * c * s**2, "r3g-multiple-angle-i-squared-sign"),
                    ex(c**3 - 2 * c * s**2, "r3g-multiple-angle-binomial-coefficient"),
                    ex(3 * c, "r3g-multiple-angle-multiplied-ratio"),
                    ex(3 * c**2 * s - s**3, "r3g-multiple-angle-parts-swapped"),
                ],
                ft_cos,
            ),
            part(
                "Use De Moivre's theorem to find sin 3θ.",
                sin3,
                2,
                [f"sin 3θ = 3 cos²θ sin θ - sin³θ = {typed(sin3)}."],
                [
                    ex(3 * c**2 * s + s**3, "r3g-multiple-angle-i-squared-sign"),
                    ex(3 * c * s**2 - c**3, "r3g-multiple-angle-parts-swapped"),
                    ex(3 * s, "r3g-multiple-angle-multiplied-ratio"),
                    ex(3 * c**2 * s - 2 * s**3, "r3g-multiple-angle-binomial-coefficient"),
                ],
                ft_sin,
            ),
            part(
                "Hence find tan 3θ.",
                tan3,
                1,
                [f"tan 3θ = sin 3θ ÷ cos 3θ = {typed(tan3)}."],
                [
                    ex(1 / tan3, "r3g-multiple-angle-ratio-inverted"),
                    ex(-tan3, "r3g-multiple-angle-i-squared-sign"),
                    ex(3 * s / c, "r3g-multiple-angle-multiplied-ratio"),
                    ex(sin3 * cos3, "r3g-multiple-angle-ratio-inverted"),
                ],
                "part_c/part_b",
            ),
        ],
    )
