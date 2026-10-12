"""Generates `line_integral` questions (easy, medium, hard) and the exam chain `exd_x_uc_line_integral`.

``ucv1-line-integral`` (``multivariable-and-vector-calculus``, answer ``expression``): line integrals in the plane.

* easy: a scalar line integral, ∫_C f ds, of a linear polynomial over a straight segment between lattice points.
  The answer is exact and usually has a surd (the length of the segment).
* medium: the work ∫_C F·dr of a polynomial field along a parametrised curve r(t) (a parabola, a cubic or a line).
* hard: a conservative field. The curve is awkward or not given, and the potential makes the work short: the work is
  φ(B) - φ(A). Half the hard questions hide a parameter k that must first be found by making the field conservative.
* exam (``exd_x_uc_line_integral``): either (a) the potential φ, (b) the work from A to B, (c) the work along a
  second curve between other points; or (a) the value of k that makes the field conservative, (b) the potential,
  (c) the work from A to B, (d) the work along a second curve. Every later part follows through from the potential.

Every answer is worked out twice: exactly, with fractions, and numerically by six-point Gauss-Legendre on the
parametrised integral. For the hard and exam questions the exact route is the potential difference and the numerical
route integrates F·r' along the curve, so the two share nothing.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, pick_distractors, rational_text, value_safe_str
from ._ucv1_multiint import Poly, gauss_1d
from ._ucv1_texsay import say_marked

X, Y = Poly.var(0), Poly.var(1)
T = Poly.var(0)  # the parameter t uses index 0 of its own polynomials


def _nonzero(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# ---------------------------------------------------------------------------------------------------------------
# surd values: c * sqrt(m)


def _reduce(c: Fraction, n: int) -> tuple[Fraction, int]:
    """c * sqrt(n) with the squares taken out of n."""
    k = 1
    for f in range(int(n**0.5) + 1, 1, -1):
        if n % (f * f) == 0:
            k = f
            break
    return c * k, n // (k * k)


def _surd_text(c: Fraction, m: int) -> str:
    if c == 0:
        return "0"
    if m == 1:
        return rational_text(c)
    mag = abs(c)
    sign = "-" if c < 0 else ""
    if mag.denominator == 1:
        lead = "" if mag == 1 else str(mag.numerator)
        return f"{sign}{lead}√{m}"
    return f"{sign}{mag.numerator}√{m}/{mag.denominator}"


def _surd_machine(c: Fraction, m: int) -> str:
    return value_safe_str(sympy.Rational(c.numerator, c.denominator) * sympy.sqrt(m))


def _same_surd(a: tuple[Fraction, int], b: tuple[Fraction, int]) -> bool:
    return a[1] == b[1] and a[0] == b[0] or (a[0] == 0 and b[0] == 0)


def _surd_float(c: Fraction, m: int) -> float:
    return float(c) * m**0.5


# ---------------------------------------------------------------------------------------------------------------
# easy: scalar line integral over a segment


_EASY_STEMS = [
    "Evaluate «∫C f ds», where «f(x, y) = {f}» and C is the straight line segment from «{A}» to «{B}». Give the exact value.",
    "Find the exact value of the line integral «∫C ({f}) ds» along the straight line segment C from «{A}» to «{B}».",
    "Let C be the line segment from «{A}» to «{B}». Evaluate the scalar line integral of «f(x, y) = {f}» over C, exactly.",
]


def _prompt(template: str, **slots) -> dict:
    text, latex = say_marked(template.format(**slots))
    return dict(prompt=text, prompt_latex=latex)


def _stem(template: str, **slots) -> dict:
    text, latex = say_marked(template.format(**slots))
    return dict(text=text, latex=latex)


def _pt(p) -> str:
    return f"({p[0]}, {p[1]})"


def _easy(rng):
    for _ in attempts(500, "an easy line integral"):
        A = (rng.randint(-3, 4), rng.randint(-3, 4))
        B = (rng.randint(-3, 4), rng.randint(-3, 4))
        dx, dy = B[0] - A[0], B[1] - A[1]
        n = dx * dx + dy * dy
        if n == 0 or n > 50:
            continue
        p, q, r = _nonzero(rng, -4, 5), _nonzero(rng, -4, 5), rng.choice([0, 0, 1, 2, 3, -2])
        f = p * X + q * Y + r
        break
    xt, yt = A[0] + dx * T, A[1] + dy * T
    along = f.compose({0: xt, 1: yt})  # f(r(t)), a polynomial in t
    inner = along.antiderivative(0)
    I = inner.value((1, 0, 0)) - inner.value((0, 0, 0))
    c, m = _reduce(I, n)  # the integral is I * sqrt(n)
    # second route: Gauss-Legendre on the unit interval, with the speed taken from the coordinates.
    speed = (float(dx) ** 2 + float(dy) ** 2) ** 0.5
    numeric = gauss_1d(lambda t: float(f.value((A[0] + dx * t, A[1] + dy * t, 0))) * speed, 0.0, 1.0)
    check(abs(numeric - _surd_float(c, m)) < 1e-9 * max(1.0, abs(numeric)), "line integral disagrees with quadrature")

    B_end = f.value((B[0], B[1], 0))
    ds_as_dx = I * abs(dx if dx else dy)
    candidates = [
        ("line-integral-forgot-speed", _reduce(I, 1)),
        ("line-integral-speed-squared", _reduce(I * n, 1)),
        ("line-integral-endpoint-only", _reduce(B_end, n)),
        ("line-integral-ds-as-dx", _reduce(ds_as_dx, 1)),
    ]
    answer = (c, m)
    scored = [(tag, v) for tag, v in candidates if not _same_surd(v, answer)]
    return dict(f=f, A=A, B=B, answer=answer, scored=scored)


# ---------------------------------------------------------------------------------------------------------------
# medium: work along a parametrised curve

_CURVES = [
    # (x(t), y(t), t range)
    (lambda: T, lambda: T**2, (0, 1)),
    (lambda: T, lambda: T**2, (0, 2)),
    (lambda: T, lambda: T**2, (-1, 1)),
    (lambda: T**2, lambda: T, (0, 1)),
    (lambda: T**2, lambda: T, (0, 2)),
    (lambda: T, lambda: T**3, (0, 1)),
    (lambda: T, lambda: T**3, (-1, 1)),
    (lambda: T + 1, lambda: 2 * T, (0, 2)),
    (lambda: 2 * T, lambda: 1 - T, (0, 3)),
    (lambda: T + 1, lambda: T**2 - T, (0, 2)),
    (lambda: 2 * T - 1, lambda: T**2, (0, 2)),
]


def _field(rng) -> tuple[Poly, Poly]:
    """A polynomial field that is NOT conservative (so the path matters), degree at most 2."""
    for _ in attempts(200, "a non-conservative polynomial field"):
        monos = [X * Y, X, Y, X**2, Y**2, Poly.const(1)]
        P = sum((rng.choice([-3, -2, -1, 1, 2, 3]) * rng.choice(monos) for _ in range(2)), Poly())
        Q = sum((rng.choice([-3, -2, -1, 1, 2, 3]) * rng.choice(monos) for _ in range(2)), Poly())
        if P.t and Q.t and (P.diff(1) - Q.diff(0)).t:
            return P, Q
    raise AssertionError("unreachable")


def _clean(p: Poly, names: str) -> str:
    text = p.text(names)
    return f"({text})" if len(p.t) > 1 or text.startswith("-") else text


def _clean_vec(P: Poly, Q: Poly, names: str = "xyz") -> str:
    """P i + Q j, with brackets only around a sum or a negative term."""
    return f"{_clean(P, names)} i + {_clean(Q, names)} j"


def _vec_t(xt: Poly, yt: Poly) -> str:
    return _clean_vec(xt, yt, "t")


def _work(P: Poly, Q: Poly, xt: Poly, yt: Poly, a: int, b: int) -> Fraction:
    integrand = P.compose({0: xt, 1: yt}) * xt.diff(0) + Q.compose({0: xt, 1: yt}) * yt.diff(0)
    anti = integrand.antiderivative(0)
    return anti.value((b, 0, 0)) - anti.value((a, 0, 0))


def _work_numeric(P, Q, xt, yt, a, b) -> float:
    def f(t):
        pt = (float(xt.value((t, 0, 0))), float(yt.value((t, 0, 0))), 0.0)
        return float(P.value(pt)) * float(xt.diff(0).value((t, 0, 0))) + float(Q.value(pt)) * float(
            yt.diff(0).value((t, 0, 0))
        )

    return gauss_1d(f, float(a), float(b))


_MEDIUM_STEMS = [
    "Find the work done by the force field «F(x, y) = {F}» in moving a particle along the curve C given by «r(t) = {r}, {lo} ≤ t ≤ {hi}». Give the exact value.",
    "Evaluate «∫C F · dr», where «F(x, y) = {F}» and C is the curve «r(t) = {r}» for «{lo} ≤ t ≤ {hi}».",
    "A particle moves along C: «r(t) = {r}, {lo} ≤ t ≤ {hi}», under the force «F(x, y) = {F}». Find the work done by F, exactly.",
]


def _medium(rng):
    for _ in attempts(300, "a medium line integral"):
        P, Q = _field(rng)
        xf, yf, (lo, hi) = rng.choice(_CURVES)
        xt, yt = xf(), yf()
        W = _work(P, Q, xt, yt, lo, hi)
        if W == 0 or W.denominator > 30:
            continue
        break
    numeric = _work_numeric(P, Q, xt, yt, lo, hi)
    check(abs(numeric - float(W)) < 1e-8 * max(1.0, abs(numeric)), "work disagrees with quadrature")

    # the plausible wrong routes
    comp = {0: xt, 1: yt}
    wrong_r = (P.compose(comp) * xt + Q.compose(comp) * yt).antiderivative(0)
    r_for_rprime = wrong_r.value((hi, 0, 0)) - wrong_r.value((lo, 0, 0))
    wrong_dr = (P.compose(comp) + Q.compose(comp)).antiderivative(0)
    forgot_dr = wrong_dr.value((hi, 0, 0)) - wrong_dr.value((lo, 0, 0))
    swapped = (Q.compose(comp) * xt.diff(0) + P.compose(comp) * yt.diff(0)).antiderivative(0)
    comps_swapped = swapped.value((hi, 0, 0)) - swapped.value((lo, 0, 0))
    A = (xt.value((lo, 0, 0)), yt.value((lo, 0, 0)))
    B = (xt.value((hi, 0, 0)), yt.value((hi, 0, 0)))
    endpoint_field = P.value((B[0], B[1], 0)) * (B[0] - A[0]) + Q.value((B[0], B[1], 0)) * (B[1] - A[1])
    candidates = [
        ("line-integral-r-for-r-prime", r_for_rprime),
        ("line-integral-forgot-dr", forgot_dr),
        ("line-integral-direction-sign", -W),
        ("line-integral-endpoint-field", endpoint_field),
        ("line-integral-components-swapped", comps_swapped),
    ]
    return dict(P=P, Q=Q, xt=xt, yt=yt, lo=lo, hi=hi, W=W, candidates=candidates)


# ---------------------------------------------------------------------------------------------------------------
# hard and exam: conservative fields, built from a potential

_PHI_MONOS = [
    ("x2y", Poly.mono(1, 2, 1)),
    ("xy2", Poly.mono(1, 1, 2)),
    ("x3", Poly.mono(1, 3)),
    ("y3", Poly.mono(1, 0, 3)),
    ("x2", Poly.mono(1, 2)),
    ("y2", Poly.mono(1, 0, 2)),
    ("xy", Poly.mono(1, 1, 1)),
]


def _draw_phi(rng, need_xy2: bool = False):
    """A potential of degree 3 with at least one mixed cubic term; returns (phi, coefficients by name)."""
    for _ in attempts(300, "a potential function"):
        names = rng.sample([n for n, _ in _PHI_MONOS], rng.randint(3, 4))
        if need_xy2 and "xy2" not in names:
            names[0] = "xy2"
        if not ({"x2y", "xy2"} & set(names)) or len(set(names)) < 3:
            continue
        coeffs = {n: _nonzero(rng, -3, 4) for n in set(names)}
        phi = sum((dict(_PHI_MONOS)[n] * c for n, c in coeffs.items()), Poly())
        P, Q = phi.diff(0), phi.diff(1)
        if P.t and Q.t:
            return phi, coeffs
    raise AssertionError("unreachable")


def _curve(rng, avoid: set):
    """r(t) = (x(t), y(t)) on [0, 1], awkward (not a straight line), ending at lattice points not in `avoid`."""
    for _ in attempts(300, "an awkward curve"):
        x0, y0 = rng.randint(-2, 3), rng.randint(-2, 3)
        ax = [rng.randint(-2, 3), rng.randint(-2, 2), rng.choice([0, 0, 1, -1])]
        ay = [rng.randint(-2, 3), rng.randint(-2, 2), rng.choice([0, 0, 1, -1])]
        if ax[1] == 0 and ay[1] == 0 and ax[2] == 0 and ay[2] == 0:
            continue
        xt = x0 + ax[0] * T + ax[1] * T**2 + ax[2] * T**3
        yt = y0 + ay[0] * T + ay[1] * T**2 + ay[2] * T**3
        # not a straight line: some curvature
        if not (xt.degree(0) >= 2 or yt.degree(0) >= 2):
            continue
        a_pt = (xt.value((0, 0, 0)), yt.value((0, 0, 0)))
        b_pt = (xt.value((1, 0, 0)), yt.value((1, 0, 0)))
        if a_pt == b_pt or a_pt in avoid or b_pt in avoid:
            continue
        return xt, yt, a_pt, b_pt
    raise AssertionError("unreachable")


def _potential_wrongs(phi: Poly, P: Poly, Q: Poly) -> list[tuple[str, Poly]]:
    """Wrong potentials a learner might write, as (misconception, polynomial)."""
    only_p = P.antiderivative(0)  # integrates P dx and stops: the function of y alone is missing
    only_q = Q.antiderivative(1)  # integrates Q dy and stops
    both = only_p + only_q  # adds the two integrals, counting each mixed term twice
    return [
        ("line-integral-potential-missing-y-part", only_p),
        ("line-integral-potential-missing-x-part", only_q),
        ("line-integral-potential-double-counted", both),
        ("line-integral-potential-sign", -phi),
    ]


def _poly_typed(p: Poly) -> str:
    return p.text("xy", style="typed")


_HARD_STEMS = [
    "Find the work done by the force field «F(x, y) = {F}» in moving a particle along the curve C given by «r(t) = {r}, 0 ≤ t ≤ 1». Give the exact value.",
    "Evaluate «∫C F · dr», where «F(x, y) = {F}» and C is the curve «r(t) = {r}, 0 ≤ t ≤ 1».",
    "A particle moves along C: «r(t) = {r}, 0 ≤ t ≤ 1», under the force «F(x, y) = {F}». Find the work done by F, exactly.",
]
_HARD_K_STEMS = [
    "The force field «F(x, y) = {F}» is conservative. Find the work done by F in moving a particle from «{A}» to «{B}».",
    "The field «F(x, y) = {F}» is conservative, where «k» is a constant. Find «∫C F · dr» for any path C from «{A}» to «{B}».",
]


def _field_with_k(phi: Poly, coeffs: dict):
    """F = grad phi with the coefficient of y^2 in P (from the x y^2 term) replaced by the symbol k."""
    P, Q = phi.diff(0), phi.diff(1)
    c2 = coeffs["xy2"]
    p_text = (P - Poly.mono(c2, 0, 2)).text()
    k_term = "ky²"
    # the P component, with the k y² term written last
    p_full = f"{p_text} + {k_term}" if p_text != "0" else k_term
    return p_full, Q


def _hard(rng):
    if rng.random() < 0.5:
        phi, coeffs = _draw_phi(rng)
        P, Q = phi.diff(0), phi.diff(1)
        xt, yt, A, B = _curve(rng, set())
        W = phi.value((B[0], B[1], 0)) - phi.value((A[0], A[1], 0))
        if W == 0:
            return None
        # second route: integrate F . r' along the curve numerically
        numeric = _work_numeric(P, Q, xt, yt, 0, 1)
        check(abs(numeric - float(W)) < 1e-7 * max(1.0, abs(numeric)), "potential difference disagrees with quadrature")
        pa, pb = phi.value((A[0], A[1], 0)), phi.value((B[0], B[1], 0))
        wrong_phis = _potential_wrongs(phi, P, Q)
        cands = [("line-integral-direction-sign", pa - pb), ("line-integral-endpoints-added", pa + pb)]
        for tag, wp in wrong_phis[:2]:
            cands.append((tag, wp.value((B[0], B[1], 0)) - wp.value((A[0], A[1], 0))))
        text = rng.choice(_HARD_STEMS).format(F=_clean_vec(P, Q), r=_vec_t(xt, yt))
        return dict(text=text, W=W, candidates=cands)
    # a parameter k that must first make the field conservative
    phi, coeffs = _draw_phi(rng, need_xy2=True)
    P, Q = phi.diff(0), phi.diff(1)
    for _ in attempts(100, "two lattice points"):
        A = (rng.randint(-2, 3), rng.randint(-2, 3))
        B = (rng.randint(-2, 3), rng.randint(-2, 3))
        if A != B:
            break
    W = phi.value((B[0], B[1], 0)) - phi.value((A[0], A[1], 0))
    if W == 0:
        return None
    c2 = coeffs["xy2"]
    # second route: the work along the straight segment A -> B, integrated numerically
    xt, yt = A[0] + (B[0] - A[0]) * T, A[1] + (B[1] - A[1]) * T
    numeric = _work_numeric(P, Q, xt, yt, 0, 1)
    check(abs(numeric - float(W)) < 1e-7 * max(1.0, abs(numeric)), "k-field work disagrees with quadrature")
    cands = []
    for tag, k_wrong in (("line-integral-parameter-sign", -c2), ("line-integral-parameter-derivative-slip", 2 * c2)):
        if k_wrong == c2:
            continue
        wp = phi + Poly.mono(k_wrong - c2, 1, 2)
        cands.append((tag, wp.value((B[0], B[1], 0)) - wp.value((A[0], A[1], 0))))
    pa, pb = phi.value((A[0], A[1], 0)), phi.value((B[0], B[1], 0))
    cands += [("line-integral-direction-sign", pa - pb), ("line-integral-endpoints-added", pa + pb)]
    p_full, _ = _field_with_k(phi, coeffs)
    field = f"({p_full}) i + {_clean(Q, 'xyz')} j"
    text = rng.choice(_HARD_K_STEMS).format(F=field, A=_pt(A), B=_pt(B))
    return dict(text=text, W=W, candidates=cands)


def _value_distractors(answer: Fraction, candidates, max_denominator: int = 24) -> list[Distractor]:
    out = []
    for tag, v in candidates:
        v = Fraction(v)
        if v != answer and v.denominator <= max_denominator:
            out.append(Distractor(rational_text(v), tag, exact_value(v)))
    return out


@generator(
    "line_integral",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def line_integral(rng, tier="easy"):
    for _ in attempts(30, "a line integral with enough distractors"):
        out = _single(rng, tier)
        if out is not None:
            return out
    raise AssertionError("unreachable")


def _single(rng, tier):
    if tier == "easy":
        d = _easy(rng)
        c, m = d["answer"]
        answer_text = _surd_text(c, m)
        machine = _surd_machine(c, m)
        text, latex = say_marked(rng.choice(_EASY_STEMS).format(f=d["f"].text(), A=_pt(d["A"]), B=_pt(d["B"])))
        dist = [Distractor(_surd_text(*v), tag, _surd_text(*v)) for tag, v in d["scored"]]
        working = [f"With r(t) = A + t(B - A), ds = |r'(t)| dt, and the integral is {answer_text}."]
    elif tier == "medium":
        d = _medium(rng)
        W = d["W"]
        answer_text = rational_text(W)
        machine = answer_text
        text, latex = say_marked(
            rng.choice(_MEDIUM_STEMS).format(
                F=_clean_vec(d["P"], d["Q"]), r=_vec_t(d["xt"], d["yt"]), lo=d["lo"], hi=d["hi"]
            )
        )
        dist = _value_distractors(W, d["candidates"])
        working = [
            f"Work = the integral of F(r(t)) · r'(t) dt from t = {d['lo']} to t = {d['hi']}, which is {answer_text}."
        ]
    else:
        d = _hard(rng)
        if d is None:
            return None
        W = d["W"]
        answer_text = rational_text(W)
        machine = answer_text
        text, latex = say_marked(d["text"])
        dist = _value_distractors(W, d["candidates"])
        working = [f"The field is conservative, with a potential φ, so the work is φ(B) - φ(A) = {answer_text}."]
    distractors = pick_distractors(answer_text, dist, 4)
    if len(distractors) < 3:
        return None
    return dict(
        text=text,
        latex=latex,
        answer=machine,
        answer_text=answer_text,
        answer_kind="expression",
        distractors=distractors,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = [
    "A particle moves in the force field «F(x, y) = {F}», with force in newtons and distance in metres.",
    "The force field «F(x, y) = {F}» acts on a particle. Force is measured in newtons and distance in metres.",
    "A robot arm moves a tool through the force field «F(x, y) = {F}» (newtons, metres).",
]
_A_PROMPTS = [
    "Show that F is conservative by finding its potential function «φ(x, y)», the function with «F = ∇φ» and «φ(0, 0) = 0».",
    "Find the potential function «φ(x, y)» with «F = ∇φ» and «φ(0, 0) = 0». (The existence of «φ» shows that F is conservative.)",
]
_B_PROMPTS = [
    "Hence find the work done by F, in joules, in moving the particle from «{A}» to «{B}» along any path.",
    "Use «φ» to find the work done by F, in joules, as the particle moves from «{A}» to «{B}».",
]
_C_PROMPTS = [
    "Find the work done by F, in joules, along the curve C: «r(t) = {r}, 0 ≤ t ≤ 1».",
    "The particle is moved instead along C: «r(t) = {r}, 0 ≤ t ≤ 1». Find the work done by F, in joules.",
]


class _Redraw(Exception):
    """A draw that cannot give each part two distinct wrong answers."""


@generator("exd_x_uc_line_integral", level=20, presets={"exam": dict(tier="exam")})
def exd_x_uc_line_integral(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam line integral with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    with_k = rng.random() < 0.5
    phi, coeffs = _draw_phi(rng, need_xy2=with_k)
    P, Q = phi.diff(0), phi.diff(1)
    A = (rng.randint(-1, 2), rng.randint(-1, 2))
    for _ in attempts(100, "endpoints for part (b)"):
        B = (rng.randint(-1, 3), rng.randint(-1, 3))
        if B != A:
            break
    xt, yt, C0, C1 = _curve(rng, {A, B})
    at = lambda p, pt: p.value((pt[0], pt[1], 0))  # noqa: E731
    W_ab = at(phi, B) - at(phi, A)
    W_c = at(phi, C1) - at(phi, C0)
    if W_ab == 0 or W_c == 0 or abs(W_ab) > 400 or abs(W_c) > 400:
        raise _Redraw
    # second route to the work: F . r' along each path, numerically
    seg_x, seg_y = A[0] + (B[0] - A[0]) * T, A[1] + (B[1] - A[1]) * T
    for W, (x_, y_) in ((W_ab, (seg_x, seg_y)), (W_c, (xt, yt))):
        numeric = _work_numeric(P, Q, x_, y_, 0, 1)
        check(abs(numeric - float(W)) < 1e-7 * max(1.0, abs(numeric)), "exam work disagrees with quadrature")
    check(P.diff(1) == Q.diff(0), "the field built from phi is not conservative")

    phi_text = _poly_typed(phi)
    phi_machine = value_safe_str(phi.sympy("xy"))
    wrongs = _potential_wrongs(phi, P, Q)
    phi_dist = [Distractor(_poly_typed(w), tag, _poly_typed(w)) for tag, w in wrongs if w != phi]

    def work_cands(a_pt, b_pt, path=None):
        pa, pb = at(phi, a_pt), at(phi, b_pt)
        out = [("line-integral-direction-sign", pa - pb), ("line-integral-endpoints-added", pa + pb)]
        if path is None:
            out.append(
                ("line-integral-endpoint-field", at(P, b_pt) * (b_pt[0] - a_pt[0]) + at(Q, b_pt) * (b_pt[1] - a_pt[1]))
            )
            out.append(("line-integral-forgot-initial-potential", pb))
        else:
            comp = {0: path[0], 1: path[1]}
            wr = (P.compose(comp) * path[0] + Q.compose(comp) * path[1]).antiderivative(0)
            out.append(("line-integral-r-for-r-prime", wr.value((1, 0, 0)) - wr.value((0, 0, 0))))
            wd = (P.compose(comp) + Q.compose(comp)).antiderivative(0)
            out.append(("line-integral-forgot-dr", wd.value((1, 0, 0)) - wd.value((0, 0, 0))))
        return out

    b_dist = pick_distractors(rational_text(W_ab), _value_distractors(W_ab, work_cands(A, B)), 4)
    c_dist = pick_distractors(rational_text(W_c), _value_distractors(W_c, work_cands(C0, C1, (xt, yt))), 4)
    a_dist = pick_distractors(phi_text, phi_dist, 4)
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        raise _Redraw

    field = _clean_vec(P, Q)
    work_b = f"φ({B[0]}, {B[1]}) - φ({A[0]}, {A[1]}) = {rational_text(W_ab)}"
    ft_b = f"Subs(part_{{p}}, (x, y), ({B[0]}, {B[1]})) - Subs(part_{{p}}, (x, y), ({A[0]}, {A[1]}))"
    ft_c = f"Subs(part_{{p}}, (x, y), ({C1[0]}, {C1[1]})) - Subs(part_{{p}}, (x, y), ({C0[0]}, {C0[1]}))"
    prompts_b = _prompt(rng.choice(_B_PROMPTS), A=_pt(A), B=_pt(B))
    prompt_c = _prompt(rng.choice(_C_PROMPTS), r=_vec_t(xt, yt))

    def part_b(label_potential: str, marks_b=2):
        return dict(
            **prompts_b,
            answer_kind="fraction",
            answer=exact_value(W_ab),
            answer_text=rational_text(W_ab),
            marks=marks_b,
            follow_through=ft_b.format(p=label_potential),
            working=[work_b + "."],
            distractors=b_dist,
        )

    def part_c(label_potential: str):
        return dict(
            **prompt_c,
            answer_kind="fraction",
            answer=exact_value(W_c),
            answer_text=rational_text(W_c),
            marks=2,
            follow_through=ft_c.format(p=label_potential),
            working=[
                f"F is conservative, so the work is φ({C1[0]}, {C1[1]}) - φ({C0[0]}, {C0[1]}) = {rational_text(W_c)}."
            ],
            distractors=c_dist,
        )

    if not with_k:
        return dict(
            **_stem(rng.choice(_EXAM_STEMS), F=field),
            parts=[
                dict(
                    **_prompt(rng.choice(_A_PROMPTS)),
                    answer_kind="expression",
                    answer=phi_machine,
                    answer_text=phi_text,
                    marks=3,
                    working=["Integrate P with respect to x, add a function of y, and match the y-derivative to Q."],
                    distractors=a_dist,
                ),
                part_b("a"),
                part_c("a"),
            ],
        )

    # the parameter variant
    c2 = coeffs["xy2"]
    p_full, _ = _field_with_k(phi, coeffs)
    field_k = f"({p_full}) i + {_clean(Q, 'xyz')} j"
    k_wrongs = [("line-integral-parameter-sign", -c2), ("line-integral-parameter-derivative-slip", 2 * c2)]
    k_wrongs += [("line-integral-parameter-other-coefficient", Fraction(coeffs[n])) for n in coeffs if n != "xy2"][:1]
    k_dist = pick_distractors(
        rational_text(c2), [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in k_wrongs if v != c2], 3
    )
    if len(k_dist) < 2:
        raise _Redraw
    k_expr = phi.sympy("xy")
    k_symbol = sympy.Symbol("k")
    phi_k = (phi - Poly.mono(c2, 1, 2)).sympy("xy") + k_symbol * sympy.Symbol("x") * sympy.Symbol("y") ** 2
    ft_phi = str(phi_k).replace("k", "part_a")
    check(sympy.expand(phi_k.subs(k_symbol, c2) - k_expr) == 0, "the potential with k is wrong")
    return dict(
        **_stem(rng.choice(_EXAM_STEMS) + " Here «k» is a constant.", F=field_k),
        parts=[
            dict(
                **_prompt("Find the value of «k» for which F is conservative."),
                answer_kind="fraction",
                answer=exact_value(Fraction(c2)),
                answer_text=rational_text(Fraction(c2)),
                marks=2,
                working=["F is conservative when ∂P/∂y = ∂Q/∂x, which gives k."],
                distractors=k_dist,
            ),
            dict(
                **_prompt(
                    "For this value of «k», find the potential function «φ(x, y)» with «F = ∇φ» and «φ(0, 0) = 0»."
                ),
                answer_kind="expression",
                answer=phi_machine,
                answer_text=phi_text,
                marks=3,
                follow_through=ft_phi,
                working=["Integrate P with respect to x, add a function of y, and match the y-derivative to Q."],
                distractors=a_dist,
            ),
            part_b("b", 2),
            part_c("b"),
        ],
    )
