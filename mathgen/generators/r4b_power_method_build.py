"""The power method: `r4b_power_method` (easy, medium, hard) and the exam chain `exd_x_r4b_power_method`.

``uc2-power-method-step`` (``numerical-and-computational-mathematics``). Every question fixes the scaling in its
wording so the answer is unique: after each multiplication by A the vector is divided by its entry of largest
magnitude (keeping that entry's sign), so the entry of largest magnitude becomes 1, and that entry is the eigenvalue
estimate. Every draw has a unique entry of largest magnitude at every step.

* easy: a 2x2 matrix, two steps, the second eigenvalue estimate to 4 decimal places.
* medium: a 3x3 matrix, two steps, the second eigenvalue estimate to 4 decimal places.
* hard: the same, but the answer is the Rayleigh quotient of the second scaled vector, to 4 decimal places.
* exam: a 3x3 matrix with a stated dominant eigenvalue, in four parts: the first estimate (whole number), the second
  entry of the first scaled vector, the second estimate from both, and its distance from the stated eigenvalue.

A is built as P·D·P⁻¹ with D diagonal, so its dominant eigenvalue is known; each draw is checked by applying A to the
first column of P, by repeating the steps in floating point and by running the iteration until it settles on that
eigenvalue.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator
from ._r4b_kit import Redraw, fraction_part, fr, margin_ok, number_distractors, number_part, with_redraw
from ._ucv1_texsay import say_marked
from .r4b_lu_factorisation_build import answer_dict, mmul, shown, tagged

M_UNSCALED = "num2-power-method-vector-not-scaled"
M_MIN_ENTRY = "num2-power-method-scaled-by-smallest-entry"
M_SUM = "num2-power-method-scaled-by-sum-of-entries"
M_TRANSPOSE = "num2-power-method-transpose-used"
M_EXTRA = "num2-power-method-extra-step-taken"
M_ABS = "num2-power-method-sign-of-scale-lost"
M_EARLY = "num2-power-method-estimate-from-earlier-step"
M_UNSCALED_EST = "num2-power-method-estimate-without-scaling"
M_OTHER_COMPONENT = "num2-power-method-ratio-of-another-component"
M_RAYLEIGH_SCALE = "num2-power-method-scale-factor-for-rayleigh-quotient"
M_RAYLEIGH_NO_DENOM = "num2-power-method-rayleigh-denominator-missed"
M_RAYLEIGH_EARLY = "num2-power-method-rayleigh-from-earlier-vector"
M_RAYLEIGH_NORM = "num2-power-method-rayleigh-norm-of-product"
M_ERR_SIGN = "num2-power-method-error-sign"
M_ERR_EIG = "num2-power-method-eigenvalue-for-error"


def matvec(a, v):
    return [sum(Fraction(a[i][j]) * v[j] for j in range(len(v))) for i in range(len(a))]


def top(v):
    """Index of the entry of largest magnitude, or None when two entries tie for it."""
    mags = [abs(x) for x in v]
    best = max(mags)
    idx = [i for i, m in enumerate(mags) if m == best]
    return idx[0] if len(idx) == 1 else None


def scaled_step(a, v):
    """(y, m, x): y = A v, m its entry of largest magnitude, x = y / m."""
    y = matvec(a, v)
    k = top(y)
    check(k is not None and y[k] != 0, "no unique largest entry")
    m = y[k]
    return y, m, [t / m for t in y]


def rayleigh(a, v):
    av = matvec(a, v)
    return sum(v[i] * av[i] for i in range(len(v))) / sum(t * t for t in v)


def _unimodular(rng, n):
    for _ in attempts(500, "a unimodular matrix"):
        lo = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        up = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i):
                lo[i][j] = Fraction(rng.randint(-2, 2) if n == 2 else rng.randint(-1, 2))
            for j in range(i + 1, n):
                up[i][j] = Fraction(rng.randint(-2, 2) if n == 2 else rng.randint(-1, 2))
        p = mmul(lo, up)
        if sum(1 for row in p for v in row if v == 0) <= n:
            return p, lo, up
    raise AssertionError("unreachable")


def _inverse_unimodular(lo, up):
    """P⁻¹ = U⁻¹ L⁻¹ for the unit triangular factors, by back substitution."""
    n = len(lo)

    def inv_unit(t, lower):
        out = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        order = range(n) if lower else range(n - 1, -1, -1)
        for c in range(n):
            for i in order:
                if i == c:
                    continue
                if lower and i < c or not lower and i > c:
                    continue
                span = range(c, i) if lower else range(i + 1, c + 1)
                out[i][c] = -sum(t[i][k] * out[k][c] for k in span)
        return out

    return mmul(inv_unit(up, False), inv_unit(lo, True))


def _matrix(rng, n, entry_cap):
    for _ in attempts(5000, "a matrix with a dominant eigenvalue"):
        lam1 = rng.randint(3, 8)
        pool = [v for v in range(-lam1 + 1, lam1) if v != 0]
        others = rng.sample(pool, n - 1)
        p, lo, up = _unimodular(rng, n)
        pinv = _inverse_unimodular(lo, up)
        d = [[Fraction(([lam1] + others)[i]) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
        a = mmul(mmul(p, d), pinv)
        flat = [x for row in a for x in row]
        if any(x.denominator != 1 for x in flat) or max(abs(x) for x in flat) > entry_cap:
            continue
        if sum(1 for x in flat if x == 0) > (n if n == 3 else 0):
            continue
        return [[int(x) for x in row] for row in a], lam1, [[int(x) for x in row] for row in p]
    raise AssertionError("unreachable")


def _starts(rng, n):
    if n == 2:
        return rng.choice([[1, 1], [1, 0], [2, 1], [1, 2], [1, -1], [3, 1], [1, 3], [2, -1], [0, 1]])
    return rng.choice([[1, 1, 1], [1, 0, 1], [2, 1, 1], [1, 1, 0], [1, -1, 1], [0, 1, 1], [1, 2, 1], [2, 1, 0]])


def _verify(a, lam1, p, v0, trail):
    """A p₁ = λ₁ p₁ for the first column of P, the steps again in floating point, and the iteration settling on λ₁."""
    n = len(a)
    col = [Fraction(p[i][0]) for i in range(n)]
    check(matvec(a, col) == [lam1 * t for t in col], "the first column of P is not a dominant eigenvector")
    fv = [float(t) for t in v0]
    for y, m, x in trail:
        fy = [sum(a[i][j] * fv[j] for j in range(n)) for i in range(n)]
        fm = max(fy, key=abs)
        check(
            all(abs(fy[i] - float(y[i])) < 1e-6 for i in range(n)) and abs(fm - float(m)) < 1e-6,
            "float route disagrees",
        )
        fv = [t / fm for t in fy]
    for _ in range(80):
        fy = [sum(a[i][j] * fv[j] for j in range(n)) for i in range(n)]
        fm = max(fy, key=abs)
        fv = [t / fm for t in fy]
    check(abs(fm - lam1) < 1e-6, "the iteration does not settle on lam1 from this start")


def _draw(rng, n, steps, cap=12):
    for _ in attempts(3000, "a power-method draw"):
        a, lam1, p = _matrix(rng, n, cap)
        v0 = [Fraction(t) for t in _starts(rng, n)]
        v, ok = v0, True
        trail = []
        for _s in range(steps):
            y = matvec(a, v)
            k = top(y)
            if k is None or y[k] == 0:
                ok = False
                break
            y, m, v = scaled_step(a, v)
            trail.append((y, m, v))
            if any(t == 0 for t in v) or any(abs(t) > 10**4 for t in y):
                ok = False
                break
        if not ok:
            continue
        if trail[0][1].denominator != 1 or (n == 3 and all(t.denominator == 1 for t in trail[0][2])):
            continue
        if steps >= 2 and not (abs(lam1 - trail[1][1]) < abs(lam1 - trail[0][1]) and trail[1][1] > 0):
            continue  # the second estimate should be a visible improvement on the first
        try:
            _verify(a, lam1, p, v0, trail)
        except Exception:
            continue
        return a, lam1, v0, trail
    raise AssertionError("unreachable")


@generator(
    "r4b_power_method",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_power_method(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a power-method question with enough distractors")


_SCALE = "Scale each result by dividing by its entry of largest magnitude, so that entry becomes 1."


def _next_scale(a, v):
    """The scale factor of one more step, or None when that step has no unique largest entry."""
    try:
        return scaled_step(a, v)[1]
    except VerificationError:
        return None


def _col(v):
    return [[t] for t in v]


def _single(rng, tier):
    n = 2 if tier == "easy" else 3
    a, lam1, v0, trail = _draw(rng, n, 2, cap=9 if n == 2 else 12)
    (y1, m1, x1), (y2, m2, x2) = trail
    m3 = _next_scale(a, x2)
    ask_start = rng.choice(
        [
            f"Let «A = {shown(a)}» and «x₀ = {shown(_col(v0))}». The power method multiplies by «A» and scales the "
            f"result at each step. {_SCALE} The eigenvalue estimate after a step is the entry used to scale. ",
            f"The power method is applied to «A = {shown(a)}» with starting vector «x₀ = {shown(_col(v0))}»: at each "
            f"step, multiply by «A» and scale the result. {_SCALE} The eigenvalue estimate after a step is the entry "
            "used to scale. ",
            f"Starting from «x₀ = {shown(_col(v0))}», run the power method for «A = {shown(a)}», multiplying by «A» "
            f"and then scaling at every step. {_SCALE} The entry used to scale is the eigenvalue estimate for that step. ",
        ]
    )
    if tier in ("easy", "medium"):
        at = [[a[j][i] for j in range(n)] for i in range(n)]
        vt = [Fraction(t) for t in v0]
        for _s in range(2):
            try:
                _y, mt, vt = scaled_step(at, vt)
            except Exception:
                mt = None
                break
        k2 = top(y2)
        cands = [
            (M_EARLY, m1),
            (M_EXTRA, m3),
            (M_UNSCALED_EST, m1 * m2),
            (M_TRANSPOSE, mt),
        ] + [(M_OTHER_COMPONENT, y2[j] / x1[j]) for j in range(n) if j != k2]
        if m2 < 0:
            cands.append((M_ABS, abs(m2)))
        stem = say_marked(ask_start + "Find the eigenvalue estimate after the second step, to 4 decimal places.")
        if not margin_ok(m2, 4, 0.12) or m2.denominator == 1:
            raise Redraw
        part = number_part(
            "",
            m2,
            4,
            cands,
            1,
            working=[
                f"Step 1: A x₀ = {shown(_col(y1))}, scaled by {m1}, gives x₁ = {shown(_col(x1))}.",
                f"Step 2: A x₁ = {shown(_col(y2))}; its entry of largest magnitude, {m2}, is the estimate.",
            ],
        )
        dist = part["distractors"]
        if len(dist) < 3:
            raise Redraw
        return dict(
            text=stem[0],
            latex=stem[1],
            answer=part["answer"],
            answer_text=part["answer_text"],
            answer_kind="number",
            distractors=dist,
            working=part["working"],
        )
    value = rayleigh(a, x2)
    check(value == sympy_rayleigh(a, x2), "SymPy's Rayleigh quotient disagrees")
    if not margin_ok(value, 4, 0.12):
        raise Redraw
    av = matvec(a, x2)
    cands = [
        (M_RAYLEIGH_SCALE, m3),
        (M_RAYLEIGH_NO_DENOM, sum(x2[i] * av[i] for i in range(3))),
        (M_RAYLEIGH_EARLY, rayleigh(a, x1)),
        (M_RAYLEIGH_NORM, sum(t * t for t in av) / sum(t * t for t in x2)),
        (M_EARLY, m2),
    ]
    stem = say_marked(
        rng.choice(
            [
                f"Let «A = {shown(a)}» and «x₀ = {shown(_col(v0))}». The power method multiplies by «A» and scales the "
                f"result at each step. {_SCALE} After two steps, find the Rayleigh quotient «x₂ᵀAx₂ / x₂ᵀx₂» of the "
                "scaled vector «x₂», to 4 decimal places.",
                f"The power method is applied to «A = {shown(a)}» with starting vector «x₀ = {shown(_col(v0))}»: at "
                f"each step, multiply by «A» and scale the result. {_SCALE} Find the Rayleigh quotient "
                "«x₂ᵀAx₂ / x₂ᵀx₂» of the scaled vector «x₂» after two steps, to 4 decimal places.",
                f"Starting from «x₀ = {shown(_col(v0))}», run two steps of the power method for «A = {shown(a)}», "
                f"multiplying by «A» and then scaling. {_SCALE} What is the Rayleigh quotient «x₂ᵀAx₂ / x₂ᵀx₂» of "
                "the scaled vector «x₂», to 4 decimal places?",
            ]
        )
    )
    part = number_part(
        "",
        value,
        4,
        cands,
        1,
        working=[
            f"x₁ = {shown(_col(x1))} and x₂ = {shown(_col(x2))}.",
            f"x₂ᵀAx₂ = {sum(x2[i] * av[i] for i in range(3))} and x₂ᵀx₂ = {sum(t * t for t in x2)}.",
        ],
    )
    dist = part["distractors"]
    if len(dist) < 3:
        raise Redraw
    return dict(
        text=stem[0],
        latex=stem[1],
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="number",
        distractors=dist,
        working=part["working"],
    )


def sympy_rayleigh(a, v):
    sa = sympy.Matrix(a)
    sv = sympy.Matrix([sympy.Rational(t.numerator, t.denominator) for t in v])
    r = (sv.T * sa * sv)[0] / (sv.T * sv)[0]
    return Fraction(int(r.p), int(r.q))


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_power_method", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4b_power_method(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam power-method question")


_ORD = ["first", "second", "third"]


def _rt(v) -> str:
    v = fr(v)
    return f"Rational({v.numerator}, {v.denominator})"


def _exam(rng):
    for _ in attempts(500, "an exam power-method draw"):
        a, lam1, v0, trail = _draw(rng, 3, 2)
        (y1, m1, x1), (y2, m2, x2) = trail
        k1 = top(y1)
        k2 = top(y2)
        err = abs(lam1 - m2)
        if err != 0 and m1 != m2 and err < abs(lam1 - m1):
            break
    q = (k1 + 1) % 3
    r = (k1 + 2) % 3
    stem, latex = say_marked(
        f"Let «A = {shown(a)}» and «x₀ = {shown(_col(v0))}». The dominant eigenvalue of «A» is «λ = {lam1}». "
        f"The power method multiplies by «A» and scales the result at each step. {_SCALE} The eigenvalue estimate "
        "after a step is the entry used to scale."
    )
    p_a = fraction_part(
        "Find the eigenvalue estimate «m₁» after the first step.",
        m1,
        [(M_UNSCALED_EST, Fraction(sum(y1))), (M_ABS, abs(m1)), (M_OTHER_COMPONENT, y1[(k1 + 1) % 3]), (M_EXTRA, m2)],
        1,
        working=[f"A x₀ = {shown(_col(y1))}; its entry of largest magnitude is {m1}."],
    )
    p_b = fraction_part(
        f"Using your «m₁», find the {_ORD[q]} entry of the scaled vector «x₁».",
        x1[q],
        [
            (M_UNSCALED, y1[q]),
            (M_MIN_ENTRY, y1[q] / min((t for t in y1 if t != 0), key=abs)),
            (M_OTHER_COMPONENT, x1[r]),
            (M_SUM, y1[q] / sum(y1) if sum(y1) else None),
        ],
        1,
        ft=f"{_rt(y1[q])}/part_a",
        working=[f"x₁ = {shown(_col(x1))}, so the {_ORD[q]} entry is {x1[q]}."],
    )
    row = a[k2]
    lead_term = Fraction(row[k1]) if True else 0
    # (A x1)_k2 = row[k1]*1 + row[q]*x1[q] + row[r]*x1[r], x1[r] = y1[r]/m1; x1[k1] = 1.
    ft_c = f"{row[k1]} + {row[q]}*part_b + {row[r]}*{_rt(y1[r])}/part_a"
    cands_c = [
        (M_EARLY, m1),
        (M_EXTRA, _next_scale(a, x2)),
        (M_UNSCALED_EST, m1 * m2),
        (M_OTHER_COMPONENT, y2[(k2 + 1) % 3] / x1[(k2 + 1) % 3]),
        (M_OTHER_COMPONENT, y2[(k2 + 2) % 3] / x1[(k2 + 2) % 3]),
    ]
    p_c = fraction_part(
        "Using your «m₁» and your entry of «x₁», find the eigenvalue estimate «m₂» after the second step.",
        m2,
        cands_c,
        2,
        ft=ft_c,
        working=[f"A x₁ = {shown(_col(y2))}; its entry of largest magnitude is {m2}."],
    )
    p_d = fraction_part(
        "Using your «m₂», find the distance «|λ − m₂|» of the estimate from the dominant eigenvalue.",
        err,
        [
            (M_ERR_SIGN, m2 - lam1 if m2 > lam1 else lam1 - m2),
            (M_ERR_EIG, Fraction(lam1)),
            (M_EARLY, abs(lam1 - m1)),
            ("num2-power-method-error-as-ratio", m2 / lam1),
        ],
        1,
        ft=f"Abs({lam1} - part_c)",
        working=[f"|{lam1} − {m2}| = {err}."],
        value_range=(0, None),
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
