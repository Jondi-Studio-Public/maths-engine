"""Generates `s2c3_sturm_liouville` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_sturm_liouville`.

``uc2-sturm-liouville-eigenvalues`` (``fourier-analysis-and-pdes``, answer ``number``): the eigenvalue problem
``y'' + lambda y = 0`` on an interval [a, b] with mixed, Neumann or periodic end conditions.

* easy: Dirichlet at one end and Neumann at the other, on [0, L] or a shifted interval [a, b]; find the n-th eigenvalue.
* medium: either Neumann at both ends or periodic ends, where lambda = 0 is an eigenvalue (and, for periodic ends, every
  other one is double), so the index of the eigenvalue asked for has to be counted carefully; or the value of a
  normalised eigenfunction at a point.
* hard: either a Robin end, ``y(0) = 0`` and ``y'(L) + h y(L) = 0``, whose n-th eigenvalue is the square of the n-th
  root of ``mu cos(mu L) + h sin(mu L) = 0``; or the coefficient of a linear function in the eigenfunction expansion
  (orthogonality, with the norm L/2).
* exam (``exd_x_s2c3_sturm_liouville``): the problem ``y(0) = 0, y'(pi) = 0``: (a) lambda_1, (b) lambda_n (follows through
  from (a)), (c) the normalised n-th eigenfunction at a point, (d) the coefficient of f(x) = 1 or f(x) = x in the
  eigenfunction expansion (follows through from (b)).

Every eigenvalue is checked by substituting the eigenfunction back into the differential equation and both end
conditions numerically, and the index is checked by counting sign changes for the Robin problem and by listing the
eigenvalues in order for the others.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..core import VerificationError, attempts, check, generator
from ._s2c3_common import fixed, key_ok, number_part, places_text, prompt_fields, simpson, stem_fields, wrong_numbers
from ._s2c3_fourier import Fn, Piece, multiple_of_L

DP = 3
_ODE = ("y″ + λy = 0", r"y''+\lambda y=0")
_ENDS = {"D": ("y(%s) = 0", "y(%s)=0"), "N": ("y′(%s) = 0", "y'(%s)=0")}


# -- the spectrum --------------------------------------------------------------------------------------------------------


def wavenumbers(kind: str, ell: float, count: int) -> list[float]:
    """The first `count` wavenumbers k (lambda = k^2), smallest first, repeated eigenvalues listed once per eigenfunction."""
    if kind == "DD":
        return [n * math.pi / ell for n in range(1, count + 1)]
    if kind in ("DN", "ND"):
        return [(2 * n - 1) * math.pi / (2 * ell) for n in range(1, count + 1)]
    if kind == "NN":
        return [n * math.pi / ell for n in range(0, count)]
    out = [0.0]  # periodic: 0, then each 2 m pi / l twice
    m = 1
    while len(out) < count:
        out += [2 * m * math.pi / ell] * 2
        m += 1
    return out[:count]


def eigenfunction(kind: str, a: float, ell: float, k: float):
    """The un-normalised eigenfunction for wavenumber k."""
    if kind in ("DD", "DN"):
        return lambda x: math.sin(k * (x - a))
    if kind in ("ND", "NN"):
        return lambda x: math.cos(k * (x - a))
    return lambda x: math.cos(k * (x - a))  # periodic: the cosine member of the pair


def _check_eigenpair(kind: str, a: float, b: float, k: float) -> None:
    """Substitute back: y'' + k^2 y = 0 and both end conditions, by central differences."""
    y = eigenfunction(kind, a, b - a, k)
    h = 1e-4 * (b - a)
    mid = a + 0.37 * (b - a)
    check(abs((y(mid + h) - 2 * y(mid) + y(mid - h)) / (h * h) + k * k * y(mid)) < 1e-3 * (1 + k * k), "not a solution")
    deriv = lambda x: (y(x + h) - y(x - h)) / (2 * h)  # noqa: E731
    ends = {"DD": ("D", "D"), "DN": ("D", "N"), "ND": ("N", "D"), "NN": ("N", "N"), "PER": ("P", "P")}[kind]
    for x, e in zip((a, b), ends):
        if e == "D":
            check(abs(y(x)) < 1e-9, "Dirichlet condition fails")
        elif e == "N":
            check(abs(deriv(x)) < 1e-6 * (1 + k), "Neumann condition fails")
    if kind == "PER":
        check(abs(y(a) - y(b)) < 1e-9 and abs(deriv(a) - deriv(b)) < 1e-6 * (1 + k), "periodic conditions fail")


def _interval(rng):
    """(a, b, text, tex, is_pi): the interval of the problem."""
    r = rng.random()
    if r < 0.2:
        return 0.0, math.pi, "[0, π]", r"[0,\pi]", True
    if r < 0.6:
        L = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
        return 0.0, float(L), f"[0, {L}]", f"[0,{L}]", False
    a = rng.choice([-2, -1, 1, 2, 3])
    L = rng.choice([2, 3, 4, 5, 6])
    return float(a), float(a + L), f"[{a}, {a + L}]", f"[{a},{a + L}]", False


def _end_text(letter: str, point: str):
    plain, tex = _ENDS[letter]
    return plain % point, tex % point


def _problem_items(kind: str, a_text: str, b_text: str, interval) -> list:
    ends = {"DD": ("D", "D"), "DN": ("D", "N"), "ND": ("N", "D"), "NN": ("N", "N")}
    if kind == "PER":
        cond = [("y(%s) = y(%s) and y′(%s) = y′(%s)" % (a_text, b_text, a_text, b_text), "")]
        return [
            "Consider the eigenvalue problem ",
            _ODE,
            f" on {interval[2]} with ",
            (
                f"y({a_text}) = y({b_text}) and y′({a_text}) = y′({b_text})",
                rf"y({a_text})=y({b_text})\text{{ and }}y'({a_text})=y'({b_text})",
            ),
            ". ",
        ]
    left, right = ends[kind]
    lp, lt = _end_text(left, a_text)
    rp, rt = _end_text(right, b_text)
    return [
        "Consider the eigenvalue problem ",
        _ODE,
        f" on {interval[2]} with ",
        (f"{lp} and {rp}", rf"{lt}\text{{ and }}{rt}"),
        ". ",
    ]


def _pt(v: float) -> str:
    return "π" if abs(v - math.pi) < 1e-12 else f"{v:g}"


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(200, "an easy eigenvalue problem"):
        a, b, text, tex, is_pi = _interval(rng)
        kind = rng.choice(["DN", "ND"])
        n = rng.choice([2, 3, 4, 5])
        ell = b - a
        ks = wavenumbers(kind, ell, n)
        _check_eigenpair(kind, a, b, ks[-1])
        v = ks[-1] ** 2
        if not key_ok(v, DP, 0.05):
            continue
        cands = [
            ((n * math.pi / ell) ** 2, "sl-dirichlet-formula-used"),
            (((2 * n + 1) * math.pi / (2 * ell)) ** 2, "sl-index-off-by-one"),
            (((2 * n - 1) * math.pi / ell) ** 2, "sl-half-factor-dropped"),
            (((2 * n - 1) * math.pi / (2 * ell)), "sl-eigenvalue-not-squared"),
            (((2 * n - 1) * math.pi / (2 * b)) ** 2 if a != 0 and b != 0 else None, "sl-interval-endpoint-for-length"),
            (((2 * n - 3) * math.pi / (2 * ell)) ** 2, "sl-index-off-by-one"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        return dict(
            **stem_fields(
                *_problem_items(kind, _pt(a), _pt(b), (None, None, text)),
                f"The eigenvalues are λ₁ < λ₂ < λ₃ < …. Find λ{'₀₁₂₃₄₅₆₇₈₉'[n]}, correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"The mixed ends give λₙ = ((2n − 1)π/(2ℓ))² with ℓ = b − a = {_pt(ell)}, so λ{n} = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- medium ----------------------------------------------------------------------------------------------------------------


def _medium_index(rng):
    for _ in attempts(200, "an indexed eigenvalue"):
        a, b, text, tex, is_pi = _interval(rng)
        kind = rng.choice(["NN", "PER"])
        ell = b - a
        m = rng.choice([3, 4, 5]) if kind == "NN" else rng.choice([4, 5, 6, 7])
        ks = wavenumbers(kind, ell, m + 2)
        _check_eigenpair(kind, a, b, ks[m - 1])
        # the list is strictly increasing for NN, and non-decreasing with exact repeats for PER
        check(all(x <= y + 1e-12 for x, y in zip(ks, ks[1:])), "eigenvalues are not in order")
        v = ks[m - 1] ** 2
        if not key_ok(v, DP, 0.05):
            continue
        if kind == "NN":
            cands = [
                ((m * math.pi / ell) ** 2, "sl-zero-eigenvalue-forgotten"),
                (((m - 2) * math.pi / ell) ** 2 if m > 2 else None, "sl-index-off-by-one"),
                ((m * math.pi / (2 * ell)) ** 2, "sl-half-factor-wrongly-used"),
                (((m - 1) * math.pi / ell), "sl-eigenvalue-not-squared"),
            ]
            kind_text = "both ends insulated"
        else:
            j = (m) // 2  # the m-th listed value is 2 j pi / l
            cands = [
                ((m * math.pi / ell) ** 2, "sl-multiplicity-ignored"),
                ((2 * (m - 1) * math.pi / ell) ** 2, "sl-multiplicity-ignored"),
                ((2 * ((m + 1) // 2) * math.pi / ell) ** 2, "sl-zero-eigenvalue-forgotten"),
                ((2 * (m // 2 + 1) * math.pi / ell) ** 2, "sl-index-off-by-one"),
                (((m - 1) * math.pi / ell) ** 2, "sl-multiplicity-ignored"),
            ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        note = (
            "List the eigenvalues in increasing order, with each eigenvalue repeated once for every independent "
            "eigenfunction it has: λ₁ ≤ λ₂ ≤ λ₃ ≤ …. "
        )
        return dict(
            **stem_fields(
                *_problem_items(kind, _pt(a), _pt(b), (None, None, text)),
                note,
                f"Find λ{'₀₁₂₃₄₅₆₇₈₉'[m]}, correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                "λ = 0 is an eigenvalue"
                + (
                    ", and every other eigenvalue (2jπ/ℓ)² has two eigenfunctions, a cosine and a sine"
                    if kind == "PER"
                    else ", with the others (jπ/ℓ)²"
                )
                + f"; counting in order, λ{m} = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _medium_eigenfunction(rng):
    for _ in attempts(300, "a normalised eigenfunction value"):
        a, b, text, tex, is_pi = _interval(rng)
        kind = rng.choice(["DD", "DN", "ND"])
        n = rng.choice([1, 2, 3])
        ell = b - a
        k = wavenumbers(kind, ell, n)[-1]
        _check_eigenpair(kind, a, b, k)
        q = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
        x0 = a + float(q) * ell
        amp = math.sqrt(2 / ell)
        y = eigenfunction(kind, a, ell, k)
        # normalisation: the integral of (amp y)^2 over the interval is 1
        wave = np.sin if kind in ("DD", "DN") else np.cos
        norm = simpson(lambda x: (amp * wave(k * (x - a))) ** 2, a, b)
        check(abs(norm - 1.0) < 1e-6, "eigenfunction is not normalised")
        v = amp * y(x0)
        if not key_ok(v, DP, 0.05):
            continue
        trig = (lambda t: math.sin(t)) if kind in ("DD", "DN") else (lambda t: math.cos(t))
        cands = [
            (y(x0), "sl-normalisation-forgotten"),
            (math.sqrt(1 / ell) * y(x0), "sl-normalisation-wrong-constant"),
            ((2 / ell) * y(x0), "sl-normalisation-root-dropped"),
            (amp * trig(k * x0), "sl-interval-shift-ignored") if a != 0 else (None, ""),
            (amp * trig(n * math.pi * (x0 - a) / ell), "sl-dirichlet-formula-used") if kind != "DD" else (None, ""),
            (
                amp * trig((2 * n - 1) * math.pi * (x0 - a) / (2 * ell)) if kind == "DD" else None,
                "sl-mixed-formula-used",
            ),
        ]
        d = wrong_numbers(v, DP, [(val, tag) for val, tag in cands if val is not None], 4)
        if len(d) < 3:
            continue
        ends = {"DD": ("y(a) = 0", "y(b) = 0"), "DN": ("", ""), "ND": ("", "")}
        func = "sin" if kind in ("DD", "DN") else "cos"
        x_text = _pt(x0) if not is_pi else multiple_of_L(q, "π", r"\pi")[0]
        return dict(
            **stem_fields(
                *_problem_items(kind, _pt(a), _pt(b), (None, None, text)),
                f"Let φ{'₀₁₂₃₄₅₆₇₈₉'[n]} be the eigenfunction for the {['', 'smallest', 'second smallest', 'third smallest'][n]} eigenvalue, "
                f"scaled so that its integral of φ² over {text} is 1 and its leading coefficient is positive "
                f"(a positive multiple of {func}). Find φ{'₀₁₂₃₄₅₆₇₈₉'[n]}({x_text}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"φ = √(2/ℓ) {func}(k(x − a)) with k = {fixed(k, 4)} and ℓ = {_pt(ell)} (since the integral of {func}² is ℓ/2), "
                f"so φ({x_text}) = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- hard -----------------------------------------------------------------------------------------------------------------------


def robin_roots(L: float, h: float, count: int) -> list[float]:
    """The first `count` positive roots mu of mu cos(mu L) + h sin(mu L) = 0, by bisection in
    ((2n - 1) pi / (2L), n pi / L)."""
    roots = []
    for n in range(1, count + 1):
        lo, hi = (2 * n - 1) * math.pi / (2 * L) + 1e-12, n * math.pi / L - 1e-12
        g = lambda m: m * math.cos(m * L) + h * math.sin(m * L)  # noqa: E731
        check(g(lo) * g(hi) < 0, "no sign change in the bracket")
        for _ in range(200):
            mid = (lo + hi) / 2
            if g(lo) * g(mid) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append((lo + hi) / 2)
    return roots


def _count_roots_below(L: float, h: float, mu: float) -> int:
    """Independent count by scanning for sign changes of g on (0, mu)."""
    g = lambda m: m * math.cos(m * L) + h * math.sin(m * L)  # noqa: E731
    steps = 20000
    prev, count = g(1e-9), 0
    for i in range(1, steps):
        m = mu * i / steps * 0.999999
        cur = g(m)
        if prev * cur < 0:
            count += 1
        prev = cur
    return count


def _hard_robin(rng):
    for _ in attempts(200, "a Robin problem"):
        L = rng.choice([1, 2, 3, 4])
        h = rng.choice([0.5, 1, 2, 3])
        n = rng.choice([1, 2, 3])
        mu = robin_roots(L, h, n)[-1]
        check(abs(mu * math.cos(mu * L) + h * math.sin(mu * L)) < 1e-9, "not a root")
        check(_count_roots_below(L, h, mu) == n - 1, "wrong root index")
        v = mu * mu
        if not key_ok(v, DP, 0.05):
            continue
        # tan(mu L) = +mu/h: the sign slip, found as the n-th root of mu cos(mu L) - h sin(mu L)
        slip = _nth_root(lambda m: m * math.cos(m * L) - h * math.sin(m * L), L, n)
        cands = [
            (mu, "sl-eigenvalue-not-squared"),
            (((2 * n - 1) * math.pi / (2 * L)) ** 2, "sl-robin-condition-ignored"),
            ((n * math.pi / L) ** 2, "sl-dirichlet-limit-used"),
            (slip * slip if slip else None, "sl-robin-sign-error"),
            (robin_roots(L, h, n + 1)[-1] ** 2, "sl-index-off-by-one"),
            (robin_roots(L, h, n - 1)[-1] ** 2 if n > 1 else None, "sl-index-off-by-one"),
        ]
        d = wrong_numbers(v, DP, [(val, tag) for val, tag in cands if val is not None], 4)
        if len(d) < 3:
            continue
        return dict(
            **stem_fields(
                "Consider the eigenvalue problem ",
                _ODE,
                f" on [0, {L}] with ",
                (
                    f"y(0) = 0 and y′({L}) + {_num(h)}y({L}) = 0" if h != 1 else f"y(0) = 0 and y′({L}) + y({L}) = 0",
                    rf"y(0)=0\text{{ and }}y'({L})+{'' if h == 1 else _num(h)}y({L})=0",
                ),
                f". The eigenvalues are λ₁ < λ₂ < λ₃ < …, all positive. Find λ{'₀₁₂₃₄₅₆₇₈₉'[n]}, correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"With λ = μ² the solution y = sin(μx) must satisfy μ cos({L}μ) + {_num(h)} sin({L}μ) = 0, i.e. tan({L}μ) = −μ/{_num(h)}. "
                f"The {['', 'first', 'second', 'third'][n]} positive root is μ = {fixed(mu, 4)}, so λ{n} = μ² = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _num(h: float) -> str:
    return f"{h:g}"


def _nth_root(g, L: float, n: int):
    """The n-th positive root of g found by scanning for sign changes (None if not found in range)."""
    steps, top = 40000, (n + 2) * math.pi / L
    prev, found = g(1e-9), 0
    for i in range(1, steps + 1):
        m = top * i / steps
        cur = g(m)
        if prev * cur < 0:
            found += 1
            if found == n:
                lo, hi = top * (i - 1) / steps, m
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if g(lo) * g(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                return (lo + hi) / 2
        prev = cur
    return None


def _expansion_coefficient(L: float, kind: str, n: int, c0: int, c1: int) -> float:
    """c_n = integral of f phi_n / (L/2) for phi_n = sin(k x) (DN) or cos(k x) (ND), f = c0 + c1 x on [0, L]."""
    fn = Fn(L, str(L), str(L), [Piece(Fraction(0), Fraction(1), c0, c1)])
    k = (2 * n - 1) * math.pi / (2 * L)
    return fn.integral("b" if kind == "DN" else "a", k) * 2 / L


def _hard_expansion(rng):
    for _ in attempts(300, "an expansion coefficient"):
        L = rng.choice([1, 2, 3, 4, 5])
        kind = rng.choice(["DN", "ND"])
        n = rng.choice([1, 2, 3])
        c0, c1 = rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 1, 2, -1, -2])
        if c0 == 0 and c1 == 0:
            continue
        v = _expansion_coefficient(L, kind, n, c0, c1)
        k = (2 * n - 1) * math.pi / (2 * L)
        phi = (lambda x: math.sin(k * x)) if kind == "DN" else (lambda x: math.cos(k * x))
        numeric = simpson(lambda x: (c0 + c1 * x) * (np.sin(k * x) if kind == "DN" else np.cos(k * x)), 0, L) / simpson(
            lambda x: (np.sin(k * x) if kind == "DN" else np.cos(k * x)) ** 2, 0, L
        )
        check(abs(numeric - v) < 1e-7, "quadrature differs")
        if not key_ok(v, DP, 0.03):
            continue
        raw = v * L / 2
        other = "ND" if kind == "DN" else "DN"
        cands = [
            (raw, "sl-norm-not-divided"),
            (v / 2, "sl-norm-wrong"),
            (_expansion_coefficient(L, other, n, c0, c1), "sl-sine-cosine-swap"),
            (_dirichlet_coefficient(L, kind, n, c0, c1), "sl-dirichlet-formula-used"),
            (-v, "sl-coefficient-sign-flip"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        from ._s2c3_common import linear_text

        f_text = linear_text(c0, c1)
        left, right = ("D", "N") if kind == "DN" else ("N", "D")
        lp, lt = _end_text(left, "0")
        rp, rt = _end_text(right, str(L))
        func = "sin" if kind == "DN" else "cos"
        sub = "₀₁₂₃₄₅₆₇₈₉"
        return dict(
            **stem_fields(
                "Let φₙ be the eigenfunction for the n-th smallest eigenvalue of ",
                _ODE,
                f" on [0, {L}] with ",
                (f"{lp} and {rp}", rf"{lt}\text{{ and }}{rt}"),
                f", with leading coefficient 1 (so φₙ = {func}(kₙx) for a suitable kₙ). Write ",
                (f"f(x) = {f_text}", f"f(x)={f_text}"),
                f" as f(x) = Σ cₙφₙ(x) on [0, {L}]. Find c{sub[n]}, correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"kₙ = (2n − 1)π/(2·{L}), so k{n} = {fixed(k, 4)}. The φₙ are orthogonal with ∫φₙ² = {L}/2, "
                f"so c{n} = (2/{L}) ∫₀^{L} f(x) {func}(k{n}x) dx = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _dirichlet_coefficient(L: float, kind: str, n: int, c0: int, c1: int) -> float:
    """The coefficient a learner gets by using k = n pi / L instead of (2n - 1) pi / (2L) (still dividing by L/2)."""
    fn = Fn(L, str(L), str(L), [Piece(Fraction(0), Fraction(1), c0, c1)])
    return fn.integral("b" if kind == "DN" else "a", n * math.pi / L) * 2 / L


@generator(
    "s2c3_sturm_liouville",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_sturm_liouville(rng, tier="easy"):
    """Eigenvalues and eigenfunctions of y'' + lambda y = 0; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_index, _medium_eigenfunction])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_robin, _hard_expansion])(rng)


# -- exam --------------------------------------------------------------------------------------------------------------------


@generator("exd_x_s2c3_sturm_liouville", level=21, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_sturm_liouville(rng, tier="exam"):
    """Exam chain on y(0) = 0, y'(pi) = 0: lambda_1, lambda_n, a normalised eigenfunction value, an expansion coefficient."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "an exam problem"):
        n = rng.choice([2, 3, 4])
        q = rng.choice([Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4)])
        use_x = rng.random() < 0.5
        lam1 = 0.25
        lam_n = (n - 0.5) ** 2
        k = n - 0.5
        x0 = float(q) * math.pi
        amp = math.sqrt(2 / math.pi)
        phi = amp * math.sin(k * x0)
        c_n = (
            _expansion_coefficient(math.pi, "DN", n, 0, 1) if use_x else _expansion_coefficient(math.pi, "DN", n, 1, 0)
        )
        if not (key_ok(phi, DP, 0.05) and key_ok(c_n, DP, 0.03)):
            continue
        a_d = wrong_numbers(
            lam1,
            4,
            [
                (1.0, "sl-dirichlet-formula-used"),
                (0.5, "sl-eigenvalue-not-squared"),
                (2.25, "sl-index-off-by-one"),
                (1.0, "sl-half-factor-dropped"),
            ],
            3,
        )
        b_d = wrong_numbers(
            lam_n,
            4,
            [
                (float(n * n), "sl-dirichlet-formula-used"),
                ((n + 0.5) ** 2, "sl-index-off-by-one"),
                (float((2 * n - 1) ** 2), "sl-half-factor-dropped"),
                (n - 0.5, "sl-eigenvalue-not-squared"),
                (lam1 * n * n, "sl-index-squared-from-first"),
            ],
            3,
        )
        c_d = wrong_numbers(
            phi,
            DP,
            [
                (math.sin(k * x0), "sl-normalisation-forgotten"),
                (math.sqrt(1 / math.pi) * math.sin(k * x0), "sl-normalisation-wrong-constant"),
                ((2 / math.pi) * math.sin(k * x0), "sl-normalisation-root-dropped"),
                (amp * math.sin(n * x0), "sl-dirichlet-formula-used"),
            ],
            3,
        )
        raw = c_n * math.pi / 2
        d_d = wrong_numbers(
            c_n,
            DP,
            [
                (raw, "sl-norm-not-divided"),
                (c_n / 2, "sl-norm-wrong"),
                (-c_n, "sl-coefficient-sign-flip"),
                (
                    _dirichlet_coefficient(math.pi, "DN", n, 0 if use_x else 1, 1 if use_x else 0),
                    "sl-dirichlet-formula-used",
                ),
            ],
            3,
        )
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break
    else:
        raise AssertionError  # pragma: no cover
    check(
        abs(_expansion_coefficient(math.pi, "DN", n, 0 if use_x else 1, 1 if use_x else 0) - c_n) < 1e-12, "coefficient"
    )
    _check_eigenpair("DN", 0.0, math.pi, k)
    x_plain, x_tex = multiple_of_L(q, "π", r"\pi")
    sub = "₀₁₂₃₄₅₆₇₈₉"
    f_text = "x" if use_x else "1"
    ft_d = f"2*(-1)**({n}+1)/(part_b*pi)" if use_x else f"2/(sqrt(part_b)*pi)"
    return dict(
        **stem_fields(
            "Consider the eigenvalue problem ",
            _ODE,
            " on [0, π] with ",
            ("y(0) = 0 and y′(π) = 0", r"y(0)=0\text{ and }y'(\pi)=0"),
            f". Its eigenvalues are λ₁ < λ₂ < λ₃ < …, with eigenfunctions φₙ (leading coefficient 1).",
        ),
        parts=[
            dict(
                **prompt_fields("Find the smallest eigenvalue λ₁, correct to 4 decimal places."),
                **number_part(lam1, 4),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "The end conditions give φₙ = sin(kₙx) with cos(kₙπ) = 0, so kₙ = n − ½ and λ₁ = (½)² = 0.2500."
                ],
                distractors=a_d,
            ),
            dict(
                **prompt_fields(f"Hence find λ{sub[n]}, correct to 4 decimal places."),
                **number_part(lam_n, 4),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{(2 * n - 1) ** 2}*part_a",
                working=[f"λₙ = (n − ½)² = {(2 * n - 1)}²λ₁ = {fixed(lam_n, 4)}."],
                distractors=b_d,
            ),
            dict(
                **prompt_fields(
                    f"Let φ{sub[n]} be scaled so that the integral of φ{sub[n]}² over [0, π] is 1 and its leading coefficient is positive. "
                    f"Find φ{sub[n]}(",
                    (x_plain, x_tex),
                    f"), correct to {places_text(DP)}.",
                ),
                **number_part(phi, DP),
                marks=3,
                working=[f"φ = √(2/π) sin(({n} − ½)x), so φ{sub[n]}({x_plain}) = {fixed(phi, DP)}."],
                distractors=c_d,
            ),
            dict(
                **prompt_fields(
                    f"Write f(x) = {f_text} as Σ cₙφₙ(x) on [0, π], with φₙ = sin(kₙx) as above. Find c{sub[n]}, correct to {places_text(DP)}."
                ),
                **number_part(c_n, DP),
                marks=3,
                follow_through=ft_d,
                working=[
                    f"The φₙ are orthogonal with ∫φₙ² = π/2, so c{n} = (2/π)∫₀^π {f_text} sin(({n} − ½)x) dx = {fixed(c_n, DP)}."
                ],
                distractors=d_d,
            ),
        ],
    )
