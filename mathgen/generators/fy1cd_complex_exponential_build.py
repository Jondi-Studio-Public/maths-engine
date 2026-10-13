"""Generates `ucv1_complex_exponential` questions and the exam chain `exd_x_fy_complex_exponential`.

``ucv1-complex-exponential-form`` (``complex-analysis``, answer ``fraction``): complex numbers in exponential (Euler)
form r e^(iθ). Every answer is the argument θ written as a multiple of π (θ/π, a fraction), reduced to -π < θ ≤ π.

* easy: write a Cartesian number such as -3 + 3√3 i or 4i in exponential form: its argument. The quadrant has to be
  read from the signs.
* medium: z₁ and z₂ are given in exponential form; find the argument of a combination such as z₁²z₂ or z₁/z₂². The
  arguments add, subtract and multiply, and the result usually falls outside (-π, π] and has to be brought back.
* hard: solve zⁿ = w, with w in exponential or Cartesian form, and give the argument of the root with the least positive
  argument: the n roots are spread 2π/n apart and the principal root is not always the one asked for.
* exam (``exd_x_fy_complex_exponential``): z = a + b√3 i. (a) |z|, (b) its argument, (c) the argument of zⁿ (following
  through from (b)), (d) zⁿ itself, a real number (following through from (a) and (c)).

Every key is re-derived with SymPy: ``arg`` of the exact Cartesian number, and the n-th roots by solving zⁿ = w.
"""

from __future__ import annotations

from fractions import Fraction

import cmath
import math

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, pick_distractors, rational_text
from ._fy1cd_kit import F, fraction_dist, int_dist, pi_text, reduce_arg, say, surd

_S3 = sympy.sqrt(3)

# (real coef, real has sqrt3, imaginary coef, imaginary has sqrt3, argument/pi, modulus factor, modulus root)
_DIRECTIONS = [
    (1, 0, 0, 0, Fraction(0), 1),
    (0, 0, 1, 0, Fraction(1, 2), 1),
    (-1, 0, 0, 0, Fraction(1), 1),
    (0, 0, -1, 0, Fraction(-1, 2), 1),
    (1, 0, 1, 0, Fraction(1, 4), 2),
    (-1, 0, 1, 0, Fraction(3, 4), 2),
    (1, 0, -1, 0, Fraction(-1, 4), 2),
    (-1, 0, -1, 0, Fraction(-3, 4), 2),
    (1, 0, 1, 1, Fraction(1, 3), 1),
    (-1, 0, 1, 1, Fraction(2, 3), 1),
    (1, 0, -1, 1, Fraction(-1, 3), 1),
    (-1, 0, -1, 1, Fraction(-2, 3), 1),
    (1, 1, 1, 0, Fraction(1, 6), 1),
    (-1, 1, 1, 0, Fraction(5, 6), 1),
    (1, 1, -1, 0, Fraction(-1, 6), 1),
    (-1, 1, -1, 0, Fraction(-5, 6), 1),
]
_SQRT3_FAMILY = [d for d in _DIRECTIONS if d[1] or d[3]]
_ANGLES = [Fraction(s * a, b) for a, b in ((1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6)) for s in (1, -1)]


def _term(c: int, root: bool, imag: bool) -> tuple[str, str, str]:
    """One term of a Cartesian number: its sign, its plain text and its TeX (without the sign)."""
    mag = abs(c)
    head = "" if (mag == 1 and (root or imag)) else str(mag)
    if root:
        plain, tex = head + "√3" + (" i" if imag else ""), head + r"\sqrt{3}" + (r"\,i" if imag else "")
    else:
        plain, tex = head + ("i" if imag else ""), head + ("i" if imag else "")
    return ("-" if c < 0 else "+"), plain, tex


def cartesian(k: int, d) -> F:
    """k times (x + y i) for a direction row, as a formula."""
    terms = []
    if d[0]:
        terms.append(_term(d[0] * k, bool(d[1]), False))
    if d[2]:
        terms.append(_term(d[2] * k, bool(d[3]), True))
    plain = tex = ""
    for n, (sign, p, t) in enumerate(terms):
        if n == 0:
            lead = "-" if sign == "-" else ""
            plain, tex = lead + p, lead + t
        else:
            plain += f" {sign} {p}"
            tex += f" {sign} {t}"
    return F(plain, tex)


def _value(k: int, d) -> sympy.Expr:
    re = d[0] * k * (_S3 if d[1] else 1)
    im = d[2] * k * (_S3 if d[3] else 1)
    return re + im * sympy.I


def expo(r: int, p: Fraction) -> F:
    """r e^(iθ) with θ = pπ, as a formula (p is never 0)."""
    plain_angle, tex_angle = pi_text(abs(p))
    sign = "-" if p < 0 else ""
    head = "" if r == 1 else str(r)
    return F(f"{head}e^({sign}i{plain_angle})", f"{head}e^{{{sign}i\\,{tex_angle}}}")


def _angle_of(expr) -> Fraction:
    value = sympy.simplify(sympy.arg(expr) / sympy.pi)
    check(value.is_Rational, f"argument of {expr} is not a rational multiple of pi")
    return Fraction(int(value.p), int(value.q))


def _ref(p: Fraction) -> Fraction:
    """The angle between the point and the real axis, in units of pi (the 'first quadrant' angle)."""
    return min(abs(p), 1 - abs(p))


def _arg_candidates(p: Fraction):
    """Wrong arguments for the argument of a Cartesian number, each tagged with the error."""
    ref = _ref(p)
    return [
        (ref, "cx-arg-ignores-quadrant"),
        (-p, "cx-arg-conjugate-sign"),
        (Fraction(1, 2) - ref, "cx-arg-swaps-real-imag"),
        (reduce_arg(p - Fraction(1, 2)), "cx-arg-quarter-turn-off"),
        (reduce_arg(p + Fraction(1, 2)), "cx-arg-quarter-turn-off"),
    ] + ([(p + 2, "cx-arg-not-reduced")] if p < 0 else [])


def _fractions(answer: Fraction, candidates, n: int = 4) -> list[Distractor]:
    out = [fraction_dist(v, tag) for v, tag in candidates]
    return pick_distractors(rational_text(answer), out, n)


# -- easy -------------------------------------------------------------------------------------------------------


def _easy(rng):
    d = rng.choice([row for row in _DIRECTIONS if row[4] != 0])  # a positive real has no same-shape wrong argument
    k = rng.randint(1, 5)
    z = cartesian(k, d)
    p = d[4]
    check(_angle_of(_value(k, d)) == p, "SymPy's argument differs")
    stem = say(
        rng.choice(["Write ", "Express "]),
        F(f"z = {z[0]}", f"z={z[1]}"),
        " in exponential form ",
        F("re^(iθ)", r"re^{i\theta}"),
        " with ",
        F("-π < θ ≤ π", r"-\pi<\theta\le\pi"),
        ". Find ",
        F("θ/π", r"\theta/\pi"),
        ".",
    )
    cands = _arg_candidates(p)
    ds = _fractions(p, cands)
    check(len(ds) >= 2, "too few distractors")
    plain_angle = pi_text(p)[0]
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="fraction",
        answer=exact_value(p),
        answer_text=rational_text(p),
        answer_latex=rational_text(p),
        working=[
            f"Here z = {z[0]}, which lies {'in the ' + _quadrant(d) if _quadrant(d) else 'on an axis'}, so θ = {plain_angle}.",
            f"θ/π = {rational_text(p)}.",
        ],
        distractors=ds,
    )


def _quadrant(d) -> str:
    x, y = d[0], d[2]
    if x == 0 or y == 0:
        return ""
    return {
        (1, 1): "first quadrant",
        (-1, 1): "second quadrant",
        (-1, -1): "third quadrant",
        (1, -1): "fourth quadrant",
    }[(1 if x > 0 else -1, 1 if y > 0 else -1)]


# -- medium -----------------------------------------------------------------------------------------------------

# (plain, TeX, coefficient on arg z1, on arg z2, how the wrong versions go)
_COMBOS = [
    ("z₁z₂", "z_1z_2", 1, 1),
    ("z₁²z₂", "z_1^2z_2", 2, 1),
    ("z₁z₂²", "z_1z_2^2", 1, 2),
    ("z₁/z₂", r"\frac{z_1}{z_2}", 1, -1),
    ("z₁²/z₂", r"\frac{z_1^2}{z_2}", 2, -1),
    ("z₁/z₂²", r"\frac{z_1}{z_2^2}", 1, -2),
    ("z₁ times the conjugate of z₂", r"z_1\overline{z_2}", 1, -1),
]


def _medium(rng):
    for _ in attempts(500, "arguments whose combination is not trivial"):
        a, b = rng.choice(_ANGLES), rng.choice(_ANGLES)
        plain, tex, ca, cb = rng.choice(_COMBOS)
        raw = ca * a + cb * b
        if raw != 0 and abs(raw) != 1:
            break
    answer = reduce_arg(raw)
    r1, r2 = rng.randint(1, 4), rng.randint(1, 4)
    z1, z2 = (
        sympy.exp(sympy.I * sympy.pi * sympy.Rational(a.numerator, a.denominator)),
        sympy.exp(sympy.I * sympy.pi * sympy.Rational(b.numerator, b.denominator)),
    )
    combo = {
        "z₁z₂": z1 * z2,
        "z₁²z₂": z1**2 * z2,
        "z₁z₂²": z1 * z2**2,
        "z₁/z₂": z1 / z2,
        "z₁²/z₂": z1**2 / z2,
        "z₁/z₂²": z1 / z2**2,
        "z₁ times the conjugate of z₂": z1 * sympy.conjugate(z2),
    }[plain]
    check(_angle_of(combo) == answer, "SymPy's argument differs")
    stem = say(
        "Let ",
        F(f"z₁ = {expo(r1, a)[0]}", f"z_1={expo(r1, a)[1]}"),
        " and ",
        F(f"z₂ = {expo(r2, b)[0]}", f"z_2={expo(r2, b)[1]}"),
        ". Write ",
        F(plain, tex),
        " in the form ",
        F("re^(iθ)", r"re^{i\theta}"),
        " with ",
        F("-π < θ ≤ π", r"-\pi<\theta\le\pi"),
        " and find ",
        F("θ/π", r"\theta/\pi"),
        ".",
    )
    cands = []
    if raw != answer:
        cands.append((raw, "cx-arg-not-reduced"))
    if ca != 1:
        cands.append((reduce_arg(a + cb * b), "cx-arg-power-ignored"))
    if cb not in (1, -1):
        cands.append((reduce_arg(ca * a + (1 if cb > 0 else -1) * b), "cx-arg-power-ignored"))
    if cb < 0:
        cands.append((reduce_arg(ca * a - cb * b), "cx-quotient-adds-args"))
    else:
        cands.append((reduce_arg(ca * a - cb * b), "cx-product-subtracts-args"))
    cands.append((reduce_arg(-raw), "cx-arg-conjugate-sign"))
    cands.append((reduce_arg(a), "cx-arg-ignores-second-factor"))
    cands.append((reduce_arg(b), "cx-arg-ignores-second-factor"))
    ds = _fractions(answer, cands)
    check(len(ds) >= 2, "too few distractors")
    sa, sb = rational_text(a), rational_text(b)
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="fraction",
        answer=exact_value(answer),
        answer_text=rational_text(answer),
        answer_latex=rational_text(answer),
        working=[
            f"The arguments of z₁ and z₂ are {sa}π and {sb}π, so the argument of {plain} is ({ca})({sa})π + ({cb})({sb})π = {rational_text(raw)}π.",
            f"Adding or subtracting 2π to bring it into (-π, π] gives θ/π = {rational_text(answer)}.",
        ],
        distractors=ds,
    )


# -- hard -------------------------------------------------------------------------------------------------------


def _root_args(phi: Fraction, n: int) -> list[Fraction]:
    return sorted(reduce_arg((phi + 2 * k) / n) for k in range(n))


def _w_formula(s: int, phi: Fraction, cartesian_ok: bool) -> F:
    if cartesian_ok and phi in (Fraction(1), Fraction(1, 2), Fraction(-1, 2)):
        plain = {Fraction(1): f"-{s}", Fraction(1, 2): f"{s}i", Fraction(-1, 2): f"-{s}i"}[phi]
        return F(plain, plain.replace("i", r"\,i"))
    return expo(s, phi)


def _hard(rng):
    n = rng.choice([3, 4, 5, 6])
    phi = rng.choice([Fraction(1), Fraction(1, 2), Fraction(-1, 2)] + _ANGLES)
    base = rng.randint(1, 3)
    s = base**n
    roots = _root_args(phi, n)
    positive = [r for r in roots if r > 0]
    answer = min(positive)
    # Independent route: each claimed root, raised to the n-th power numerically, must give w, and there are n of them.
    wc = complex(s * cmath.cos(float(phi) * cmath.pi), s * cmath.sin(float(phi) * cmath.pi))
    check(len(set(roots)) == n, "the roots are not distinct")
    check(
        all(abs((base * cmath.exp(1j * cmath.pi * float(r))) ** n - wc) < 1e-9 * s for r in roots),
        "a numerical root does not satisfy z^n = w",
    )
    wf = _w_formula(s, phi, True)
    stem = say(
        "Of the solutions of ",
        F(f"z{SUPN[n]} = {wf[0]}", f"z^{{{n}}}={wf[1]}"),
        ", the one with the smallest positive argument is written ",
        F("re^(iθ)", r"re^{i\theta}"),
        " with ",
        F("-π < θ ≤ π", r"-\pi<\theta\le\pi"),
        ". Find ",
        F("θ/π", r"\theta/\pi"),
        ".",
    )
    principal = reduce_arg(phi / n)
    cands = [
        (principal, "cx-root-skips-2pi-k"),
        (reduce_arg((phi + 2) / n), "cx-root-wrong-k"),
        (reduce_arg(phi / n + 2), "cx-root-divides-only-2k"),
        (max(roots), "cx-root-wrong-k"),
        (phi / n + Fraction(2, n) if phi / n + Fraction(2, n) <= 1 else phi / n, "cx-root-divides-only-2k"),
        (reduce_arg(phi * n), "cx-root-power-instead-of-root"),
        (reduce_arg(answer + Fraction(2, n)), "cx-root-wrong-k"),
        (reduce_arg(answer - Fraction(2, n)), "cx-root-wrong-k"),
    ]
    ds = _fractions(answer, cands)
    check(len(ds) >= 2, "too few distractors")
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="fraction",
        answer=exact_value(answer),
        answer_text=rational_text(answer),
        answer_latex=rational_text(answer),
        working=[
            f"The solutions have arguments ({rational_text(phi)}π + 2kπ)/{n} for k = 0, 1, ..., {n - 1}.",
            "Reduced into (-π, π] they are " + ", ".join(f"{rational_text(r)}π" for r in roots) + ".",
            f"The smallest positive one is {rational_text(answer)}π.",
        ],
        distractors=ds,
    )


SUPN = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶"}


@generator(
    "ucv1_complex_exponential",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_complex_exponential(rng, tier="easy"):
    """The argument of a complex number, a combination, or a root, in exponential form; see the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

# (direction row, powers n for which n p is a whole number)
_AXIS_I = [d for d in _DIRECTIONS if d[0] == 0 and d[2] != 0]
_K_RANGES = {2: range(3, 10), 3: range(2, 4), 4: range(2, 7), 6: range(2, 3)}
_AXIS_K = {2: range(3, 10), 4: range(2, 7), 6: range(2, 5)}
_EXAM_CHOICES = [(d, n) for d in _SQRT3_FAMILY for n in (3, 6) if (d[4] * n).denominator == 1] + [
    (d, n) for d in _AXIS_I for n in (2, 4, 6)
]
_FT_REDUCE = "{x} - 2*ceiling(({x} - 1)/2)"


@generator("exd_x_fy_complex_exponential", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_complex_exponential(rng, tier="exam"):
    """Exam chain: the modulus, the argument, the argument of zⁿ and the value of zⁿ for z = a + b√3 i."""
    check(tier == "exam", f"unknown tier {tier!r}")
    d, n = rng.choice(_EXAM_CHOICES)
    axis = d in _AXIS_I
    k = rng.choice(list(_AXIS_K[n] if axis else _K_RANGES[n]))
    value = _value(k, d)
    z = cartesian(k, d)
    p = d[4]
    modulus = k if axis else 2 * k
    check(sympy.simplify(sympy.Abs(value) - modulus) == 0, "SymPy's modulus differs")
    check(_angle_of(value) == p, "SymPy's argument differs")
    raw = n * p
    arg_n = reduce_arg(raw)
    check(arg_n in (0, 1), "z^n is not real")
    check(_angle_of(sympy.expand(value**n)) == arg_n, "SymPy's argument of z^n differs")
    power = sympy.expand(value**n)
    check(power.is_real, "z^n is not real")
    power_value = int(power)
    check(power_value == modulus**n * (1 if arg_n == 0 else -1), "z^n differs from r^n cos(n theta)")

    if axis:
        a_cands = [
            (modulus * modulus, "cx-modulus-forgot-root"),
            (0, "cx-modulus-ignores-imaginary-part"),
            (-modulus, "cx-modulus-negative"),
            (1, "cx-modulus-drops-scale"),
        ]
    else:
        a_cands = [
            (4 * k * k, "cx-modulus-forgot-root"),
            (k, "cx-modulus-real-part-only"),
            (2, "cx-modulus-drops-scale"),
            (4 * k, "cx-modulus-forgot-root"),
        ]
    a_dist = pick_distractors(str(modulus), [int_dist(v, t) for v, t in a_cands], 4)
    b_dist = _fractions(p, _arg_candidates(p))
    c_dist = _fractions(
        arg_n,
        [
            (raw, "cx-arg-not-reduced"),
            (p, "cx-arg-power-ignored"),
            (Fraction(1) - arg_n, "cx-arg-sign-after-power"),
            (raw + 2 if raw < 0 else raw, "cx-arg-not-reduced"),
            (raw - 2 if raw > 0 else raw, "cx-arg-not-reduced"),
            (p / n, "cx-power-divides-arg"),
        ],
    )
    sign = 1 if arg_n == 0 else -1
    d_dist = pick_distractors(
        str(power_value),
        [
            int_dist(sign * n * modulus, "cx-power-as-multiplication"),
            int_dist(sign * modulus ** (n - 1), "cx-power-exponent-off-by-one"),
            int_dist(sign * modulus ** (n + 1), "cx-power-exponent-off-by-one"),
            int_dist(-power_value, "cx-power-sign"),
        ],
        4,
    )
    # A later distractor that a learner's own earlier (wrong) answers would lead to by following through is a right
    # answer for them, so it is not offered as a wrong one.
    own_c = {reduce_arg(n * Fraction(b.text)) for b in b_dist} | {arg_n}
    c_dist = [c for c in c_dist if Fraction(c.text) not in own_c - {arg_n}]
    a_values = [modulus] + [int(a.text) for a in a_dist]
    c_values = [arg_n] + [Fraction(c.text) for c in c_dist]
    own_d = {
        int(round(av**n * math.cos(math.pi * float(cv)))) for av in a_values for cv in c_values if cv.denominator == 1
    }
    d_dist = [d for d in d_dist if int(d.text) not in own_d - {power_value}]
    for label, ds in (("a", a_dist), ("b", b_dist), ("c", c_dist), ("d", d_dist)):
        check(len(ds) >= 2, f"too few ({label}) distractors: {ds}")

    stem = say("Let ", F(f"z = {z[0]}", f"z={z[1]}"), ".")
    ang = pi_text(p)[0]
    return dict(
        text=stem[0],
        latex=stem[1],
        parts=[
            dict(
                prompt="Find |z|.",
                prompt_latex=r"\text{Find }|z|\text{.}",
                answer_kind="number",
                answer=modulus,
                answer_text=str(modulus),
                answer_latex=str(modulus),
                marks=2,
                sensible_range=(0, None),
                working=[f"|z| is the square root of the sum of the squares of the parts of z, which is {modulus}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Write z in the form re^(iθ) with -π < θ ≤ π. Find θ/π.",
                prompt_latex=r"\text{Write }z\text{ in the form }re^{i\theta}\text{ with }-\pi<\theta\le\pi\text{. Find }\theta/\pi\text{.}",
                answer_kind="fraction",
                answer=exact_value(p),
                answer_text=rational_text(p),
                answer_latex=rational_text(p),
                marks=2,
                working=[f"Reading the quadrant from the signs, θ = {ang}, so θ/π = {rational_text(p)}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Hence write zⁿ in the form re^(iφ) with -π < φ ≤ π. Find φ/π.".replace("zⁿ", f"z{SUPN[n]}"),
                prompt_latex=rf"\text{{Hence write }}z^{{{n}}}\text{{ in the form }}re^{{i\varphi}}\text{{ with }}-\pi<\varphi\le\pi\text{{. Find }}\varphi/\pi\text{{.}}",
                answer_kind="fraction",
                answer=exact_value(arg_n),
                answer_text=rational_text(arg_n),
                answer_latex=rational_text(arg_n),
                marks=2,
                follow_through=_FT_REDUCE.format(x=f"{n}*part_b"),
                working=[
                    f"The argument of z{SUPN[n]} is {n}θ = {rational_text(raw)}π, which is {rational_text(arg_n)}π after adding a multiple of 2π."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find z{SUPN[n]}, which is a real number.",
                prompt_latex=rf"\text{{Hence find }}z^{{{n}}}\text{{, which is a real number.}}",
                answer_kind="number",
                answer=power_value,
                answer_text=str(power_value),
                answer_latex=str(power_value),
                marks=2,
                follow_through=f"part_a**{n}*cos(pi*part_c)",
                working=[
                    f"z{SUPN[n]} = |z|{SUPN[n]} e^(iφ) = {modulus}{SUPN[n]} × {'1' if arg_n == 0 else '(-1)'} = {power_value}."
                ],
                distractors=d_dist,
            ),
        ],
    )
