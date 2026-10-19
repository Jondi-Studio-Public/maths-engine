"""Generates `s2c3_laplace_rectangle` questions (easy, medium, hard) and the exam chain `exd_x_s2c3_laplace_rectangle`.

``uc2-laplace-equation-rectangle`` (``fourier-analysis-and-pdes``, answer ``number``): the steady temperature
``u(x, y)`` of a rectangular plate, ``u_xx + u_yy = 0`` on ``0 < x < a, 0 < y < b``, with three edges at 0 and one edge
held at a profile ``f``.

If the hot edge has length l (along it) and the plate is d deep across it, the solution is
``u = sum B_n sin(n pi s / l) sinh(n pi r / l) / sinh(n pi d / l)``, where s runs along the hot edge, r is the distance
from the opposite (zero) edge, and ``B_n = (2/l) integral of f sin(n pi s / l)``. Which side is hot varies, so the
learner has to find l and d from the picture, not assume the top edge.

* easy: the hot edge carries a sum of two sine modes; find u at a point inside the plate.
* medium: the hot edge carries a constant, a ramp or a tent profile; find one coefficient B_n.
* hard: either two opposite edges each carrying one sine mode (add two solutions), or the temperature at the centre of
  the plate for a constant, ramp or tent edge, to 2 decimal places, summing as many terms as matter.
* exam (``exd_x_s2c3_laplace_rectangle``): the top edge held at a constant c: (a) B_1, (b) B_3 (follows through from
  (a): B_3 = B_1 / 3), (c) the first mode's contribution at the centre (follows through from (a)), (d) u at the centre
  to 2 decimal places.

Keys are checked by central differences (u_xx + u_yy = 0 at the point, zero on the three cold edges, and the series
matching the edge profile at interior points of the hot edge).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._s2c3_common import (
    fixed,
    key_ok,
    linear_text,
    number_part,
    places_text,
    prompt_fields,
    stem_fields,
    wrong_numbers,
)
from ._s2c3_fourier import Fn, Piece

DP = 3
_TERMS = 60
_SIZES = ((3, 4), (4, 3), (4, 5), (5, 4), (3, 3), (4, 4), (6, 4), (4, 6), (5, 5), (3, 5), (5, 3))
_LAPLACE = ("∂²u/∂x² + ∂²u/∂y² = 0", r"\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}=0")


def _ratio(n: float, r: float, ell: float, d: float, kind: str = "sinh") -> float:
    """sinh(n pi r / l) / sinh(n pi d / l), computed without overflow (or cosh/cosh)."""
    a, b = n * math.pi * r / ell, n * math.pi * d / ell
    if kind == "cosh":
        return math.exp(a - b) * (1 + math.exp(-2 * a)) / (1 + math.exp(-2 * b))
    return math.exp(a - b) * (-math.expm1(-2 * a)) / (-math.expm1(-2 * b))


class Plate:
    """A rectangle a by b with edge `edge` in {top, right, bottom, left} carrying the profile with coefficients `coef`."""

    def __init__(self, a, b, edge, coef):
        self.a, self.b, self.edge, self.coef = a, b, edge, coef
        self.ell, self.d = (a, b) if edge in ("top", "bottom") else (b, a)

    def along_and_distance(self, x, y):
        """(s, r): s along the hot edge, r from the opposite edge."""
        return {
            "top": (x, y),
            "bottom": (x, self.b - y),
            "right": (y, x),
            "left": (y, self.a - x),
        }[self.edge]

    def value(
        self,
        x,
        y,
        *,
        ratio="sinh",
        ell=None,
        d=None,
        terms=_TERMS,
        first_only=False,
        wrong_r=False,
        skip_ratio=False,
        unnormalised=False,
    ):
        ell = ell or self.ell
        d = d or self.d
        s, r = self.along_and_distance(x, y)
        if wrong_r:
            r = self.d - r
        total = 0.0
        for n in range(1, 2 if first_only else terms + 1):
            b = self.coef(n)
            if b == 0:
                continue
            if skip_ratio:
                factor = 1.0
            elif unnormalised:
                factor = math.sinh(min(n * math.pi * r / ell, 700))
            else:
                factor = _ratio(n, r, ell, d, ratio)
            total += b * math.sin(n * math.pi * s / ell) * factor
        return total


def _edge_phrase(a, b, edge):
    hot = {"top": f"y = {b}", "bottom": "y = 0", "right": f"x = {a}", "left": "x = 0"}[edge]
    cold = {
        "top": f"x = 0, x = {a} and y = 0",
        "bottom": f"x = 0, x = {a} and y = {b}",
        "right": f"x = 0, y = 0 and y = {b}",
        "left": f"x = {a}, y = 0 and y = {b}",
    }[edge]
    var = "x" if edge in ("top", "bottom") else "y"
    return hot, cold, var


def _intro(a, b, edge, profile_items: list, variant: int = 0) -> list:
    hot, cold, var = _edge_phrase(a, b, edge)
    x_or_y = {"top": f"u(x, {b})", "bottom": "u(x, 0)", "right": f"u({a}, y)", "left": "u(0, y)"}[edge]
    openings = (
        f"A thin rectangular plate occupies 0 ≤ x ≤ {a}, 0 ≤ y ≤ {b} (lengths in metres). Its steady temperature "
        f"u(x, y), in °C, satisfies ",
        f"A rectangular sheet of metal lies in the region 0 ≤ x ≤ {a}, 0 ≤ y ≤ {b} (lengths in metres). Its steady "
        f"temperature u(x, y), in °C, satisfies ",
        f"Consider a flat rectangular plate with 0 ≤ x ≤ {a}, 0 ≤ y ≤ {b} (lengths in metres). Its steady "
        f"temperature u(x, y), in °C, satisfies ",
    )
    return [
        openings[variant],
        _LAPLACE,
        f". The edges {cold} are held at 0 °C, and the edge {hot} is held at {x_or_y} = ",
        *profile_items,
        ". ",
    ]


def _sine_text(coefs: dict, ell: int, var: str) -> tuple[str, str]:
    plain, tex = "", ""
    for i, (n, c) in enumerate(sorted(coefs.items())):
        mag = abs(c)
        sign = ("-" if c < 0 else "") if i == 0 else (" - " if c < 0 else " + ")
        lead = "" if mag == 1 else f"{mag} "
        arg = f"{'' if n == 1 else n}π{var}/{ell}"
        plain += f"{sign}{lead}sin({arg})"
        tex += f"{sign}{'' if mag == 1 else mag}\\sin\\frac{{{'' if n == 1 else n}\\pi {var}}}{{{ell}}}"
    return plain, tex


def _profile_fn(rng, ell: int, kind: str) -> Fn:
    if kind == "const":
        return Fn(
            float(ell), str(ell), str(ell), [Piece(Fraction(0), Fraction(1), rng.choice([20, 40, 50, 60, 80, 100]), 0)]
        )
    if kind == "ramp":
        slope = rng.choice([5, 10, 15, 20])
        while slope * ell > 100:
            slope //= 2
        slope = max(slope, 1)
        return Fn(float(ell), str(ell), str(ell), [Piece(Fraction(0), Fraction(1), 0, slope)])
    s = rng.choice([5, 10, 15, 20])
    while s * ell > 160:
        s //= 2
    s = max(s, 1)
    return Fn(
        float(ell),
        str(ell),
        str(ell),
        [Piece(Fraction(0), Fraction(1, 2), 0, s), Piece(Fraction(1, 2), Fraction(1), s * ell, -s)],
    )


_DEG = r"\,{}^{\circ}\mathrm{C}"


def _profile_items(fn: Fn, kind: str, var: str) -> list:
    """The profile as sentence items: a plain constant, or maths for a ramp or a tent."""
    if kind == "const":
        return [f"{fn.pieces[0].c0} °C"]
    texts = [linear_text(p.c0, p.c1, var) for p in fn.pieces]
    if kind == "ramp":
        return [(f"{texts[0]} °C", f"{texts[0]}{_DEG}")]
    half, ell = f"{fn.L / 2:g}", f"{int(fn.L)}"
    return [
        (f"{texts[0]} °C", f"{texts[0]}{_DEG}"),
        f" for 0 ≤ {var} ≤ {half} and ",
        (f"{texts[1]} °C", f"{texts[1]}{_DEG}"),
        f" for {half} ≤ {var} ≤ {ell}",
    ]


def _fn_coef(fn: Fn, n: int) -> float:
    return fn.integral("b", n * math.pi / fn.L) * 2 / fn.L


def _check_plate(p: Plate, x, y, fn: Fn | None) -> None:
    """Laplace's equation at the point, zero on the cold edges, and the edge profile at interior points of the hot edge."""
    h = 1e-3 * min(p.a, p.b)
    u = p.value
    lap = (u(x + h, y) + u(x - h, y) + u(x, y + h) + u(x, y - h) - 4 * u(x, y)) / (h * h)
    check(abs(lap) < 1e-3 * (1 + abs(u(x, y)) / (h * h) ** 0.0), "u does not satisfy Laplace's equation")
    for xx, yy in ((0.0, 0.4 * p.b), (p.a, 0.4 * p.b), (0.4 * p.a, 0.0), (0.4 * p.a, p.b)):
        s, r = p.along_and_distance(xx, yy)
        if r < 1e-9 or abs(r - p.d) > 1e-9:  # a cold edge, not the hot one
            check(abs(u(xx, yy)) < 1e-9, "a cold edge is not at zero")
    if fn is not None:
        for q in (0.3, 0.5, 0.7):
            s = q * p.ell
            xs, ys = {"top": (s, p.b), "bottom": (s, 0.0), "right": (p.a, s), "left": (0.0, s)}[p.edge]
            target = fn.value_at(Fraction(q).limit_denominator(100))
            check(abs(u(xs, ys, terms=2000) - target) < 0.02 * (1 + abs(target)), "the edge profile is not matched")


def _point(rng, p: Plate):
    xs = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
    ys = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
    return float(xs * p.a), float(ys * p.b), xs, ys


def _frac_text(v: Fraction, whole: int) -> str:
    val = v * whole
    return str(val.numerator) if val.denominator == 1 else f"{float(val):g}"


def _value_candidates(p: Plate, x, y):
    return [
        (p.value(x, y, skip_ratio=True), "laplace-hyperbolic-factor-dropped"),
        (p.value(x, y, ell=p.d, d=p.ell), "laplace-lengths-swapped"),
        (p.value(x, y, unnormalised=True), "laplace-hyperbolic-not-normalised"),
        (p.value(x, y, wrong_r=True), "laplace-distance-from-hot-edge"),
        (p.value(x, y, ratio="cosh"), "laplace-cosh-for-sinh"),
        (p.value(x, y, first_only=True), "laplace-first-mode-only"),
    ]


def _significant(p: Plate, x, y, threshold: float) -> int:
    s, r = p.along_and_distance(x, y)
    return sum(
        1
        for n in range(1, _TERMS + 1)
        if abs(p.coef(n) * math.sin(n * math.pi * s / p.ell) * _ratio(n, r, p.ell, p.d)) > threshold
    )


# -- easy -----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(300, "an easy plate"):
        a, b = rng.choice(_SIZES)
        edge = rng.choice(["top", "right", "bottom", "left"])
        p0 = Plate(a, b, edge, lambda n: 0.0)
        m, n = sorted(rng.sample([1, 2, 3], 2))
        coefs = {m: rng.choice([-1, 1]) * rng.randint(2, 20) * 5, n: rng.choice([-1, 1]) * rng.randint(2, 20) * 5}
        p = Plate(a, b, edge, lambda j, coefs=coefs: float(coefs.get(j, 0)))
        x, y, xs, ys = _point(rng, p)
        v = p.value(x, y)
        s, r = p.along_and_distance(x, y)
        if not key_ok(v, DP, 0.1) or abs(v) > 200:
            continue
        # both modes must matter at the point
        if any(abs(c * math.sin(j * math.pi * s / p.ell) * _ratio(j, r, p.ell, p.d)) < 0.3 for j, c in coefs.items()):
            continue
        _check_plate(p, x, y, None)
        d = wrong_numbers(v, DP, _value_candidates(p, x, y)[:5], 4)
        if len(d) < 3:
            continue
        var = "x" if edge in ("top", "bottom") else "y"
        plain, tex = _sine_text(coefs, p.ell, var)
        return dict(
            **stem_fields(
                *_intro(a, b, edge, [(plain + " °C", tex + r"\,{}^{\circ}\mathrm{C}")]),
                f"Find u({_frac_text(xs, a)}, {_frac_text(ys, b)}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"Only the hot edge is non-zero, so u = Σ Bₙ sin(nπs/{p.ell}) sinh(nπr/{p.ell})/sinh(nπ·{p.d}/{p.ell}) "
                f"with s along the hot edge and r measured from the opposite edge; the sine modes of the edge give "
                f"B{m} and B{n} directly, so u = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- medium: one coefficient -----------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(300, "a medium plate"):
        a, b = rng.choice(_SIZES)
        edge = rng.choice(["top", "right", "bottom", "left"])
        p = Plate(a, b, edge, lambda n: 0.0)
        kind = rng.choice(["const", "ramp", "tent", "const"])
        fn = _profile_fn(rng, p.ell, kind)
        n = rng.choice([1, 2, 3, 4])
        v = _fn_coef(fn, n)
        if not key_ok(v, DP, 0.08) or abs(v) > 300:
            continue
        check(abs(fn.check_coefficient("b", n, 2 / fn.L, Fraction(0), Fraction(1)) - v) < 1e-7, "quadrature differs")
        other_d = float(p.d)
        cands = [
            (v / 2, "fourier-half-range-factor-dropped"),
            (-v, "fourier-coefficient-sign-flip"),
            (fn.integral("a", n * math.pi / fn.L) * 2 / fn.L, "fourier-sine-cosine-swap"),
            (
                fn.integral("b", n * math.pi / other_d, Fraction(0), Fraction(1)) * 2 / other_d,
                "laplace-lengths-swapped",
            ),
            (v * _ratio(n, p.d / 2, p.ell, p.d), "laplace-coefficient-includes-hyperbolic-factor"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        var = "x" if edge in ("top", "bottom") else "y"
        return dict(
            **stem_fields(
                *_intro(a, b, edge, _profile_items(fn, kind, var)),
                "Write the solution as ",
                (
                    "u = Σ Bₙ sin(nπs/ℓ) sinh(nπr/ℓ)/sinh(nπd/ℓ)",
                    r"u=\sum_{n=1}^{\infty}B_n\sin\frac{n\pi s}{\ell}\,\frac{\sinh(n\pi r/\ell)}{\sinh(n\pi d/\ell)}",
                ),
                f", where s runs along the hot edge (length ℓ = {p.ell}), r is the distance from the opposite edge and d = {p.d}. "
                f"Find B{'₀₁₂₃₄₅₆₇₈₉'[n]}, correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[f"Bₙ = (2/ℓ) ∫₀^ℓ f(s) sin(nπs/ℓ) ds with ℓ = {p.ell}, so B{n} = {fixed(v, DP)}."],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------------------


class TwoEdges:
    """u = top solution + bottom solution on a plate with opposite edges y = b and y = 0 carrying one sine mode each."""

    def __init__(self, a, b, mt, ct, mb, cb):
        self.a, self.b, self.mt, self.ct, self.mb, self.cb = a, b, mt, ct, mb, cb

    def parts(self):
        top = Plate(self.a, self.b, "top", lambda n: float(self.ct) if n == self.mt else 0.0)
        bottom = Plate(self.a, self.b, "bottom", lambda n: float(self.cb) if n == self.mb else 0.0)
        return top, bottom

    def value(self, x, y, **kw):
        top, bottom = self.parts()
        return top.value(x, y, **kw) + bottom.value(x, y, **kw)


def _hard_superposition(rng):
    for _ in attempts(300, "a two-edge plate"):
        a, b = rng.choice(_SIZES)
        mt, mb = rng.sample([1, 2, 3], 2)
        ct, cb = rng.choice([-1, 1]) * rng.randint(2, 18) * 5, rng.choice([-1, 1]) * rng.randint(2, 18) * 5
        two = TwoEdges(a, b, mt, ct, mb, cb)
        top, bottom = two.parts()
        x, y, xs, ys = _point(rng, top)
        v = two.value(x, y)
        t_val, b_val = top.value(x, y), bottom.value(x, y)
        if not key_ok(v, DP, 0.1) or abs(t_val) < 0.5 or abs(b_val) < 0.5 or abs(v) > 200:
            continue
        check(abs(two.value(x, y) - t_val - b_val) < 1e-12, "superposition differs")
        for pl in (top, bottom):
            _check_plate(pl, x, y, None)
        # laplace holds for the sum, and the cold sides are zero
        h = 1e-3
        lap = sum(two.value(x + dx, y + dy) for dx, dy in ((h, 0), (-h, 0), (0, h), (0, -h))) - 4 * v
        check(abs(lap / (h * h)) < 1e-2 * (1 + abs(v) / (h * h) * 1e-3), "sum is not harmonic")
        cands = [
            (t_val, "laplace-one-edge-only"),
            (b_val, "laplace-one-edge-only"),
            (
                top.value(x, y, skip_ratio=True) + bottom.value(x, y, skip_ratio=True),
                "laplace-hyperbolic-factor-dropped",
            ),
            (top.value(x, y) + _bottom_from_y(bottom, x, y), "laplace-distance-from-hot-edge"),
            (t_val - b_val, "laplace-sign-between-edges"),
        ]
        d = wrong_numbers(v, DP, cands, 4)
        if len(d) < 3:
            continue
        pt, tt = _sine_text({mt: ct}, a, "x")
        pb, tb = _sine_text({mb: cb}, a, "x")
        return dict(
            **stem_fields(
                f"A thin rectangular plate occupies 0 ≤ x ≤ {a}, 0 ≤ y ≤ {b} (lengths in metres). Its steady temperature "
                f"u(x, y), in °C, satisfies ",
                _LAPLACE,
                f". The edges x = 0 and x = {a} are held at 0 °C, the edge y = {b} at u(x, {b}) = ",
                (pt + " °C", tt + r"\,{}^{\circ}\mathrm{C}"),
                " and the edge y = 0 at u(x, 0) = ",
                (pb + " °C", tb + r"\,{}^{\circ}\mathrm{C}"),
                f". Find u({_frac_text(xs, a)}, {_frac_text(ys, b)}), correct to {places_text(DP)}.",
            ),
            **number_part(v, DP),
            working=[
                f"Add the solution for the top edge ({fixed(t_val, DP)}) and for the bottom edge, which uses the distance "
                f"b − y from the top ({fixed(b_val, DP)}): u = {fixed(v, DP)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


def _bottom_from_y(bottom: Plate, x, y):
    return bottom.value(x, bottom.b - y)


def _centre_problem(rng, kind_choices=("const", "ramp", "tent"), edges=("top", "right", "bottom", "left"), dp=2):
    for _ in attempts(500, "a centre-of-plate problem"):
        a, b = rng.choice([(3, 4), (4, 5), (3, 5), (5, 6), (4, 6), (6, 5), (5, 4), (4, 3), (5, 3), (6, 4)])
        edge = rng.choice(edges)
        kind = rng.choice(kind_choices)
        ell, d = (a, b) if edge in ("top", "bottom") else (b, a)
        if d < 1.1 * ell:
            continue  # keep the plate deep enough across the hot edge that a few terms are enough
        fn = _profile_fn(rng, ell, kind)
        p = Plate(a, b, edge, lambda n, fn=fn: _fn_coef(fn, n))
        x, y = a / 2, b / 2
        v = p.value(x, y)
        n_terms = _significant(p, x, y, 0.004)
        if not key_ok(v, dp, 0.2) or not 2 <= n_terms <= 4 or abs(v) > 120:
            continue
        return p, fn, kind, x, y, v
    raise AssertionError  # pragma: no cover


def _hard_centre(rng):
    for _ in attempts(100, "a hard centre problem"):
        p, fn, kind, x, y, v = _centre_problem(rng)
        _check_plate(p, x, y, fn)
        cands = [(val, tag) for val, tag in _value_candidates(p, x, y)]
        d = wrong_numbers(v, 2, cands, 4)
        if len(d) < 3:
            continue
        var = "x" if p.edge in ("top", "bottom") else "y"
        return dict(
            **stem_fields(
                *_intro(p.a, p.b, p.edge, _profile_items(fn, kind, var)),
                f"Find the temperature at the centre of the plate, u({p.a / 2:g}, {p.b / 2:g}), correct to 2 decimal places.",
            ),
            **number_part(v, 2),
            working=[
                f"Expand the edge profile in sine modes of length {p.ell}, give mode n the factor sinh(nπr/{p.ell})/sinh(nπ·{p.d}/{p.ell}) "
                f"with r = {p.along_and_distance(x, y)[1]:g}, and sum the terms that matter: u = {fixed(v, 2)}."
            ],
            distractors=d,
        )
    raise AssertionError  # pragma: no cover


@generator(
    "s2c3_laplace_rectangle",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c3_laplace_rectangle(rng, tier="easy"):
    """Laplace's equation on a rectangle with one hot edge; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_superposition, _hard_centre])(rng)


@generator("exd_x_s2c3_laplace_rectangle", level=21, presets={"exam": dict(tier="exam")})
def exd_x_s2c3_laplace_rectangle(rng, tier="exam"):
    """Exam chain: B_1, B_3 (from B_1), the first mode's contribution at the centre, and u at the centre."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(400, "an exam plate"):
        a, b = rng.choice(
            [(3, 4), (4, 5), (4, 6), (5, 6), (3, 5), (5, 7), (2, 3), (3, 6), (4, 7), (2, 5), (6, 7), (5, 8)]
        )
        c = rng.choice([10, 20, 30, 40, 50, 60, 75, 80, 90, 100])
        fn = Fn(float(a), str(a), str(a), [Piece(Fraction(0), Fraction(1), c, 0)])
        p = Plate(a, b, "top", lambda n, fn=fn: _fn_coef(fn, n))
        B1, B3 = p.coef(1), p.coef(3)
        x, y = a / 2, b / 2
        first = B1 * math.sin(math.pi / 2) * _ratio(1, b / 2, a, b)
        v = p.value(x, y)
        if not (key_ok(B1, DP, 5) and key_ok(B3, DP, 1) and key_ok(first, DP, 1) and key_ok(v, 2, 1)):
            continue
        if not 2 <= _significant(p, x, y, 0.004) <= 4:
            continue
        a_d = wrong_numbers(
            B1,
            DP,
            [
                (B1 / 2, "fourier-half-range-factor-dropped"),
                (2 * c / math.pi, "laplace-constant-edge-coefficient-halved"),
                (4 * c / math.pi * math.pi, "laplace-pi-dropped"),
                (-B1, "fourier-coefficient-sign-flip"),
                (4 * c / (math.pi * b / a), "laplace-lengths-swapped"),
            ],
            3,
        )
        b_d = wrong_numbers(
            B3,
            DP,
            [
                (B1 / 9, "laplace-odd-mode-squared"),
                (B1 / 2, "laplace-even-mode-reasoning"),
                (B1 * 3, "laplace-odd-mode-inverted"),
                (B1, "laplace-all-modes-equal"),
                (-B3, "fourier-coefficient-sign-flip"),
            ],
            3,
        )
        c_d = wrong_numbers(
            first,
            DP,
            [
                (B1, "laplace-hyperbolic-factor-dropped"),
                (B1 * math.sinh(math.pi * b / (2 * a)), "laplace-hyperbolic-not-normalised"),
                (B1 * _ratio(1, b / 2, a, b, "cosh"), "laplace-cosh-for-sinh"),
                (B1 * _ratio(1, b / 2, b, a), "laplace-lengths-swapped"),
            ],
            3,
        )
        d_d = wrong_numbers(v, 2, _value_candidates(p, x, y), 3)
        if min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2:
            break
    else:
        raise AssertionError  # pragma: no cover
    _check_plate(p, x, y, fn)
    check(abs(B1 / 3 - B3) < 1e-9, "B3 is not B1/3 for a constant edge")
    check(abs(fn.check_coefficient("b", 1, 2 / fn.L, Fraction(0), Fraction(1)) - B1) < 1e-7, "quadrature differs")
    k = Fraction(b, 2 * a)
    return dict(
        **stem_fields(
            *_intro(a, b, "top", [f"{c} °C"], rng.randrange(3)),
            "Write the solution as ",
            (
                "u = Σ Bₙ sin(nπx/a) sinh(nπy/a)/sinh(nπb/a)",
                r"u=\sum_{n=1}^{\infty}B_n\sin\frac{n\pi x}{a}\,\frac{\sinh(n\pi y/a)}{\sinh(n\pi b/a)}",
            ),
            f", where a = {a} and b = {b}.",
        ),
        parts=[
            dict(
                **prompt_fields(f"Find B₁, correct to {places_text(DP)}."),
                **number_part(B1, DP),
                marks=3,
                sensible_range=(0, None),
                working=[f"B₁ = (2/a) ∫₀ᵃ {c} sin(πx/a) dx = 4·{c}/π = {fixed(B1, DP)}."],
                distractors=a_d,
            ),
            dict(
                **prompt_fields(f"Hence find B₃, correct to {places_text(DP)}."),
                **number_part(B3, DP),
                marks=2,
                follow_through="part_a/3",
                working=[f"For a constant edge Bₙ = 4·{c}/(nπ) for odd n, so B₃ = B₁/3 = {fixed(B3, DP)}."],
                distractors=b_d,
            ),
            dict(
                **prompt_fields(
                    f"The first mode contributes B₁ sin(πx/a) sinh(πy/a)/sinh(πb/a) to u. Find this contribution at the centre "
                    f"of the plate, ({a / 2:g}, {b / 2:g}), correct to {places_text(DP)}."
                ),
                **number_part(first, DP),
                marks=2,
                follow_through=f"part_a*sinh(pi*({k}))/sinh(pi*({Fraction(b, a)}))",
                working=[
                    f"At the centre sin(π/2) = 1 and sinh(π·{b}/(2·{a}))/sinh(π·{b}/{a}) gives {fixed(first, DP)}."
                ],
                distractors=c_d,
            ),
            dict(
                **prompt_fields("Find the temperature u at the centre of the plate, correct to 2 decimal places."),
                **number_part(v, 2),
                marks=3,
                working=[f"Summing the odd modes at the centre (the later terms are tiny) gives u = {fixed(v, 2)}."],
                distractors=d_d,
            ),
        ],
    )
