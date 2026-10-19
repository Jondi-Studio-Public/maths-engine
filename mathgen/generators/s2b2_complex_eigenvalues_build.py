"""Generates `s2b2_complex_eigenvalues_2x2` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_complex_eigenvalues_2x2`.

``uc2-complex-eigenvalues-2x2`` (``linear-algebra``): a real 2×2 matrix with no real eigenvalue has a conjugate pair
a ± bi. Over the reals it is A = P C P⁻¹ with C = [[a, -b], [b, a]] a rotation-scaling matrix: it scales by the
modulus r = √(a² + b²) and rotates by the argument θ, where a + bi = r(cos θ + i sin θ).

* easy (``expression``): the imaginary part b of the eigenvalue a + bi with b > 0.
* medium (``expression``): either the modulus of the eigenvalues, √(a² + b²) = √det A, or the modulus of the
  eigenvalues of A², A³ or A⁻¹, which are the powers of the eigenvalues of A.
* hard (``expression``): either the argument of a + bi in (-π, π], exactly (the quadrant matters when a < 0), or an
  entry of a power Aⁿ with a = ±b or a = 0, where the angle is a multiple of π/4 and Aⁿ = P Cⁿ P⁻¹ is an integer
  matrix.
* exam: (a) a, (b) b, (c) the modulus, (d) the argument, with (c) and (d) following through from (a) and (b).

A is P C P⁻¹ for a dense unimodular P. Every key is checked with SymPy's ``eigenvals`` and ``arg`` and with ``cmath``,
and Aⁿ is checked against exact repeated multiplication.
"""

from __future__ import annotations

import cmath
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, phrase, pick_distractors
from ._s2b2_kit import (
    tidied,
    fm,
    inverse,
    ints_distractor,
    ints_text,
    mat_str,
    matrix_distractors,
    mmul,
    mpow,
    number_distractor,
    rows_value,
    sym,
    unimodular,
)

I = sympy.I


def _draw(rng, a_values=None, b_values=None, cap=16, min_r2=0):
    """(a, b, A, P): A = P C P⁻¹ with C = [[a, -b], [b, a]], b > 0, A a dense integer matrix."""
    a_values = list(a_values if a_values is not None else range(-4, 5))
    b_values = list(b_values if b_values is not None else range(1, 6))
    for _ in attempts(2000, "a dense integer matrix with complex eigenvalues"):
        a, b = rng.choice(a_values), rng.choice(b_values)
        c = fm([[a, -b], [b, a]])
        p = unimodular(rng, 2, 2)
        m = mmul(mmul(p, c), inverse(p))
        flat = [x for row in m for x in row]
        if any(x.denominator != 1 for x in flat) or max(abs(x) for x in flat) > cap or flat.count(0) > 1:
            continue
        if m == c or a * a + b * b < min_r2:
            continue
        return a, b, m, p
    raise AssertionError  # pragma: no cover


def _verify_eigenvalues(m, a, b):
    eig = sym(m).eigenvals()
    check(set(eig) == {a + b * I, a - b * I}, "SymPy's eigenvalues differ")
    values = [complex(a, b), complex(a, -b)]
    tr, dt = float(m[0][0] + m[1][1]), float(m[0][0] * m[1][1] - m[0][1] * m[1][0])
    roots = [(tr + s * cmath.sqrt(tr * tr - 4 * dt)) / 2 for s in (1, -1)]
    check(all(min(abs(r - v) for v in values) < 1e-9 for r in roots), "the quadratic formula differs")


def _argument_expr(a, b):
    """The exact argument of a + bi, in (-π, π], as a SymPy expression (b > 0 here)."""
    if a == 0:
        return sympy.pi / 2
    if a > 0:
        return sympy.atan(sympy.Rational(b, a))
    return sympy.pi - sympy.atan(sympy.Rational(b, -a))


def _expr_text(expr):
    return str(expr).replace("**", "^")


def _decimal(x, places=3):
    return round(float(x), places)


def _argument_candidates(a, b, theta):
    """Tagged wrong arguments for a + bi (b > 0)."""
    out = []
    if a != 0:
        out.append((sympy.atan(sympy.Rational(b, a)), "complex-argument-quadrant-ignored"))
    out.append((sympy.atan(sympy.Rational(a, b)), "complex-argument-ratio-inverted"))
    out.append((-theta, "complex-conjugate-chosen"))
    out.append((sympy.pi / 2 - theta, "complex-argument-complement"))
    if a > 0:
        out.append((sympy.pi - theta, "complex-argument-quadrant-ignored"))
    if a < 0:
        out.append((sympy.atan(sympy.Rational(b, -a)), "complex-argument-quadrant-ignored"))
    return out


def _expression_options(candidates, key_value):
    """Tagged expression options `(expr, tag)`, dropping any within 0.01 of the key's value or repeated."""
    out, seen = [], set()
    for expr, tag in candidates:
        value = float(expr)
        text = _expr_text(expr)
        if abs(value - key_value) < 0.01 or text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
    return out


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The real matrix A = {m} has no real eigenvalues. Its eigenvalue with positive imaginary part is a + bi. Find b.",
    "Find the imaginary part b of the eigenvalue a + bi of A = {m} that has b > 0.",
    "A = {m} has a complex conjugate pair of eigenvalues a ± bi with b > 0. Find b.",
)


def _easy(rng):
    a, b, m, _p = _draw(rng, b_values=range(1, 8), cap=20)
    _verify_eigenvalues(m, a, b)
    tr = int(m[0][0] + m[1][1])
    dt = int(m[0][0] * m[1][1] - m[0][1] * m[1][0])
    text = str(b)
    values = [
        (a, "complex-real-imag-swapped"),
        (2 * b, "complex-quadratic-formula-halving"),
        (dt, "complex-imaginary-part-is-determinant"),
        (abs(tr), "complex-real-part-is-trace"),
        (b * b, "complex-imaginary-part-squared"),
        (-b, "complex-conjugate-chosen"),
    ]
    options, seen = [], {b}
    for v, tag in values:
        if v in seen or v == 0:
            continue
        seen.add(v)
        options.append(Distractor(str(v), tag, str(v)))
    distractors = pick_distractors(text, options, 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(m)),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=text,
        working=[
            f"The trace is {tr} and the determinant is {dt}, so λ² - {tr}λ + {dt} = 0, with discriminant {tr * tr - 4 * dt} < 0.",
            f"λ = ({tr} ± √({tr * tr - 4 * dt}))/2 = {a} ± {b}i, so b = {b}.",
        ],
        distractors=distractors,
    )


# -- medium -------------------------------------------------------------------------------------------------------

_MODULUS_STEMS = (
    "The real matrix A = {m} has complex eigenvalues. Find their modulus. Give an exact value (write sqrt for a square "
    "root).",
    "Find the modulus |λ| of the complex eigenvalues of A = {m}, exactly (write sqrt for a square root).",
)
_POWER_MODULUS_STEMS = (
    "The real matrix A = {m} has complex eigenvalues. Find the modulus of the eigenvalues of {what}. Give an exact "
    "value (write sqrt for a square root).",
    "Find the modulus of the complex eigenvalues of {what}, where A = {m}, exactly (write sqrt for a square root).",
)


def _medium_modulus(rng):
    a, b, m, _p = _draw(rng, min_r2=5)
    _verify_eigenvalues(m, a, b)
    r2 = a * a + b * b
    dt = int(m[0][0] * m[1][1] - m[0][1] * m[1][0])
    check(dt == r2, "det A differs from a² + b²")
    modulus = sympy.sqrt(r2)
    key = float(modulus)
    check(abs(abs(complex(a, b)) - key) < 1e-12, "cmath's modulus differs")
    tr = int(m[0][0] + m[1][1])
    candidates = [
        (sympy.Integer(r2), "complex-modulus-squared-forgot-root"),
        (sympy.Integer(abs(a) + b), "complex-modulus-sum-not-root"),
        (
            sympy.sqrt(abs(a * a - b * b)) if a * a != b * b else sympy.Integer(abs(a)),
            "complex-modulus-difference-of-squares",
        ),
        (sympy.Integer(abs(tr)) if tr else sympy.Integer(b), "complex-modulus-is-trace"),
        (sympy.sqrt(abs(tr * tr - 4 * dt)), "complex-modulus-from-discriminant"),
        (sympy.Integer(b), "complex-real-imag-swapped"),
    ]
    text = _expr_text(modulus)
    distractors = pick_distractors(text, _expression_options(candidates, key), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _MODULUS_STEMS, m=mat_str(m)),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(modulus),
        answer_alt=_decimal(key),
        working=[
            f"The eigenvalues are {a} ± {b}i.",
            f"|λ| = √({a}² + {b}²) = √{r2}; equivalently |λ|² = det A = {dt}.",
        ],
        distractors=distractors,
    )


def _medium_power_modulus(rng):
    a, b, m, _p = _draw(rng, min_r2=5)
    _verify_eigenvalues(m, a, b)
    r2 = a * a + b * b
    k = rng.choice([2, 3, -1])
    z = sympy.Integer(a) + sympy.Integer(b) * I
    key_expr = sympy.simplify(sympy.sqrt(r2) ** k)
    key = float(key_expr)
    check(abs(abs(complex(a, b) ** k) - key) < 1e-9, "cmath's modulus of the power differs")
    check(abs(float(sympy.Abs(z**k)) - key) < 1e-9, "SymPy's modulus of the power differs")
    # SymPy: the eigenvalues of A**k are the k-th powers of those of A.
    powered = sym(m) ** k
    check(set(powered.eigenvals()) == {sympy.expand(z**k), sympy.expand(sympy.conjugate(z) ** k)}, "eigenvalues of Aᵏ")
    r = sympy.sqrt(r2)
    if k > 0:
        candidates = [
            (r, "complex-modulus-power-ignored"),
            (k * r, "complex-modulus-power-as-multiple"),
            (sympy.Integer(r2) ** k, "complex-modulus-squared-power"),
            (sympy.Integer(r2), "complex-modulus-squared-forgot-root"),
            (r ** (k + 1), "complex-modulus-power-ignored"),
        ]
        what = "A" + {2: "²", 3: "³"}[k]
    else:
        candidates = [
            (r, "complex-modulus-inverse-not-reciprocated"),
            (-1 / r, "complex-modulus-inverse-not-reciprocated"),
            (sympy.Integer(1) / r2, "complex-modulus-squared-forgot-root"),
            (sympy.Integer(r2), "complex-modulus-squared-forgot-root"),
        ]
        what = "A⁻¹"
    text = _expr_text(key_expr)
    distractors = pick_distractors(text, _expression_options(candidates, key), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _POWER_MODULUS_STEMS, m=mat_str(m), what=what),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(key_expr),
        answer_alt=_decimal(key),
        working=[
            f"The eigenvalues of A are {a} ± {b}i, with modulus √{r2}.",
            f"The eigenvalues of {what} are the {'reciprocals' if k < 0 else 'powers'} (λ{'⁻¹' if k < 0 else {2: '²', 3: '³'}[k]}) of those, so their modulus is {text}.",
        ],
        distractors=distractors,
    )


def _medium(rng):
    return rng.choice([_medium_modulus, _medium_power_modulus])(rng)


# -- hard ---------------------------------------------------------------------------------------------------------

_ARG_STEMS = (
    "The real matrix A = {m} has complex eigenvalues. Find the argument θ of its eigenvalue with positive imaginary "
    "part, with -π < θ ≤ π. Give an exact value (write atan for arctan and pi for π).",
    "Write the eigenvalue of A = {m} with positive imaginary part in polar form r(cos θ + i sin θ), with -π < θ ≤ π. Find "
    "θ exactly (write atan for arctan and pi for π).",
)
_POWER_STEMS = (
    "The real matrix A = {m} has complex eigenvalues. Find the ({i}, {j}) entry of A{n}.",
    "Find the ({i}, {j}) entry of the matrix A{n}, where A = {m}.",
)
_SUPER = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸"}


def _hard_argument(rng):
    a, b, m, _p = _draw(rng, a_values=[x for x in range(-4, 5)], b_values=range(1, 6))
    _verify_eigenvalues(m, a, b)
    theta = _argument_expr(a, b)
    key = float(theta)
    check(abs(cmath.phase(complex(a, b)) - key) < 1e-12, "cmath's argument differs")
    check(abs(float(sympy.arg(a + b * I)) - key) < 1e-12, "SymPy's argument differs")
    candidates = _argument_candidates(a, b, theta)
    text = _expr_text(theta)
    distractors = pick_distractors(text, _expression_options(candidates, key), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _ARG_STEMS, m=mat_str(m)),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(theta),
        answer_alt=_decimal(key),
        working=[
            f"The eigenvalue with positive imaginary part is {a} + {b}i.",
            f"It lies in the {'first' if a > 0 else 'second'} quadrant (real part {'positive' if a > 0 else 'negative'}), so θ = {text}.",
        ],
        distractors=distractors,
    )


def _power_matrix(a, b, n):
    z = complex(a, b) ** n
    c = [[round(z.real), -round(z.imag)], [round(z.imag), round(z.real)]]
    return c


def _hard_power(rng):
    k = rng.choice([1, 1, 2])
    sign = rng.choice([-1, 1])
    a, b = sign * k, k
    n = rng.choice([4, 5, 6, 7, 8] if k == 1 else [4, 5, 6])
    for _ in attempts(500, "a dense matrix with eigenvalues ±k + ki"):
        _a, _b, m, p = _draw(rng, a_values=[a], b_values=[b], cap=12)
        break
    _verify_eigenvalues(m, a, b)
    exact = mpow(m, n)
    # Second route: A^n = P C^n P^-1 with C^n read from (a + bi)^n.
    cn = fm(_power_matrix(a, b, n))
    via_c = mmul(mmul(p, cn), inverse(p))
    check(via_c == exact, "P C^n P⁻¹ differs from A^n")
    check(sym(m) ** n == sym(exact), "SymPy's A^n differs")
    i, j = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    key = int(exact[i][j])
    text = str(key)
    candidates = [
        ([[Fraction(int(m[r][c]) ** n) for c in range(2)] for r in range(2)], "matrix-power-entrywise"),
        ([[Fraction(n) * m[r][c] for c in range(2)] for r in range(2)], "matrix-power-as-multiple"),
        (mpow(m, n - 1), "matrix-power-off-by-one"),
        (mpow(m, n + 1), "matrix-power-off-by-one"),
        (mpow(m, n // 2 * 2), "complex-power-angle-rounded"),
        (mscale_exact(exact, -1), "sign-error-moving-term"),
    ]
    options, seen = [], {key}
    for cand, tag in candidates:
        v = int(cand[i][j])
        if v not in seen:
            seen.add(v)
            options.append(Distractor(str(v), tag, str(v)))
    distractors = pick_distractors(text, options, 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _POWER_STEMS, m=mat_str(m), n=_SUPER[n], i=i + 1, j=j + 1),
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=text,
        working=[
            f"The eigenvalues are {a} ± {b}i, so A = PCP⁻¹ with C = [[{a}, {-b}], [{b}, {a}]].",
            f"({a} + {b}i){n} = {complex(a, b) ** n:g}, so C{n} = {mat_str(cn)} and A{n} = P C{n} P⁻¹ = {mat_str(exact)}; "
            f"its ({i + 1}, {j + 1}) entry is {text}.",
        ],
        distractors=distractors,
    )


def mscale_exact(m, s):
    return [[Fraction(s) * v for v in row] for row in m]


def _hard(rng):
    return rng.choice([_hard_argument, _hard_power])(rng)


@generator(
    "s2b2_complex_eigenvalues_2x2",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_complex_eigenvalues_2x2(rng, tier="easy"):
    """Complex eigenvalues of a real 2×2 matrix and the rotation-scaling form; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_complex_eigenvalues_2x2", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_complex_eigenvalues_2x2(rng, tier="exam"):
    """Exam chain: the real and imaginary parts, then the modulus and argument following through from them."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, b, m, _p = _draw(rng, a_values=[-4, -3, -2, -1, 1, 2, 3, 4], min_r2=5)
    _verify_eigenvalues(m, a, b)
    tr = int(m[0][0] + m[1][1])
    dt = int(m[0][0] * m[1][1] - m[0][1] * m[1][0])
    modulus = sympy.sqrt(a * a + b * b)
    theta = _argument_expr(a, b)
    mod_text, theta_text = _expr_text(modulus), _expr_text(theta)
    mod_key, theta_key = float(modulus), float(theta)
    a_dist = pick_distractors(
        str(a),
        [
            number_distractor(tr, "complex-real-part-is-trace"),
            number_distractor(-a, "sign-flip-both-roots"),
            number_distractor(b, "complex-real-imag-swapped"),
            number_distractor(dt, "eigval-determinant-as-eigenvalue"),
            number_distractor(2 * a + 1, "eigval-trace-difference-sign"),
        ],
        3,
    )
    b_dist = pick_distractors(
        str(b),
        [
            number_distractor(2 * b, "complex-quadratic-formula-halving"),
            number_distractor(-b, "complex-conjugate-chosen"),
            number_distractor(a, "complex-real-imag-swapped"),
            number_distractor(b * b, "complex-imaginary-part-squared"),
            number_distractor(b + 1, "complex-quadratic-formula-halving"),
        ],
        3,
    )
    r2 = a * a + b * b
    c_options = _expression_options(
        [
            (sympy.Integer(r2), "complex-modulus-squared-forgot-root"),
            (sympy.Integer(abs(a) + b), "complex-modulus-sum-not-root"),
            (
                sympy.sqrt(abs(a * a - b * b)) if a * a != b * b else sympy.Integer(abs(a) + 1),
                "complex-modulus-difference-of-squares",
            ),
            (sympy.sqrt(abs(tr * tr - 4 * dt)), "complex-modulus-from-discriminant"),
        ],
        mod_key,
    )
    c_dist = pick_distractors(mod_text, c_options, 3)
    d_options = _expression_options(_argument_candidates(a, b, theta), theta_key)
    d_dist = pick_distractors(theta_text, d_options, 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=phrase(
            rng,
            (
                "The real matrix A = {m} has complex eigenvalues. Let a + bi be the eigenvalue with positive imaginary part.",
                "Let A = {m}, a real matrix with no real eigenvalue. Write a + bi, with b > 0, for one of its eigenvalues.",
                "A = {m} has a pair of complex conjugate eigenvalues; let a + bi be the one with positive imaginary part.",
            ),
            m=mat_str(m),
        ),
        parts=[
            dict(
                prompt="Find a.",
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                marks=1,
                working=[f"a is half the trace: {tr}/2 = {a}."],
                distractors=a_dist,
                sensible_range=(-20, 20),
            ),
            dict(
                prompt="Find b.",
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                marks=2,
                working=[f"a² + b² = det A = {dt}, so b² = {b * b} and b = {b} (taking b > 0)."],
                distractors=b_dist,
                sensible_range=(0, 20),
            ),
            dict(
                prompt="Find the modulus |a + bi|, exactly (write sqrt for a square root).",
                answer_kind="expression",
                answer=mod_text,
                answer_text=mod_text,
                answer_latex=latex_answer(modulus),
                answer_alt=_decimal(mod_key),
                marks=1,
                follow_through="sqrt(part_a**2 + part_b**2)",
                working=[f"|a + bi| = √({a}² + {b}²) = {mod_text}."],
                distractors=c_dist,
            ),
            dict(
                prompt="Find the argument θ of a + bi, with -π < θ ≤ π, exactly (write atan for arctan and pi for π).",
                answer_kind="expression",
                answer=theta_text,
                answer_text=theta_text,
                answer_latex=latex_answer(theta),
                answer_alt=_decimal(theta_key),
                marks=2,
                follow_through="atan(part_b/part_a) + pi*(1 - sign(part_a))/2",
                working=[
                    f"a + bi = {a} + {b}i is in the {'first' if a > 0 else 'second'} quadrant, so θ = {theta_text}."
                ],
                distractors=d_dist,
            ),
        ],
    )
