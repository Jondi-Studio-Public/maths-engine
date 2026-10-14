"""Newton's method for optimisation: `r4h_newton_optimisation` (easy, medium, hard) and the exam chain
`exd_x_r4h_newton_optimisation`.

``uc2-newton-method-optimisation-step`` (``optimisation``). One Newton step towards a minimiser of ``f`` moves from
``x`` to ``x − H(x)⁻¹ ∇f(x)`` (in one variable, ``x − f′(x)/f″(x)``).

* easy: one variable, a quartic, one step (a fraction).
* medium: two variables, a polynomial with an ``x⁴`` term, one step (a column vector).
* hard: two variables: the value of ``f`` at the point reached by one step (a fraction).
* exam: two variables in four parts: ``det H`` at the start, the new ``x`` and ``y`` (follow-through from ``det H``) and
  the change in ``f`` (follow-through from the two coordinates).

Exact arithmetic with `Fraction`. The Hessian at the start is positive definite in every two-variable question, so
the step is a descent step.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    finish_parts,
    fraction_part,
    ft_symbol,
    number_part,
    poly_eval,
    poly_prompt,
    rat,
    single_from,
    tx,
    with_redraw,
)

MISC = {
    "num2t-newton-opt-root-finding-step": "Using Newton's root-finding step x − f(x)/f′(x) instead of x − f′(x)/f″(x).",
    "num2t-newton-opt-step-sign": "Adding the Newton correction instead of subtracting it.",
    "num2t-newton-opt-no-second-derivative": "Taking a gradient step x − f′(x), leaving out the division by the second derivative.",
    "num2t-newton-opt-ratio-inverted": "Dividing the second derivative by the first instead of the first by the second.",
    "num2t-newton-opt-correction-only": "Reporting the correction f′(x)/f″(x) rather than the new point.",
    "num2t-newton-opt-derivative-at-wrong-point": "Evaluating the derivatives at the origin instead of at the starting point.",
    "num2t-newton-opt-gradient-step": "Using the gradient step x − ∇f(x) with no Hessian.",
    "num2t-newton-opt-sign-reversed": "Adding H⁻¹∇f to the starting point instead of subtracting it.",
    "num2t-newton-opt-hessian-not-inverted": "Multiplying the gradient by the Hessian instead of its inverse.",
    "num2t-newton-opt-diagonal-hessian": "Ignoring the mixed second derivative, so the step treats the variables separately.",
    "num2t-newton-opt-inverse-unscaled": "Using the adjugate of the Hessian without dividing by its determinant.",
    "num2t-newton-opt-inverse-no-swap": "Inverting a 2×2 Hessian without swapping its diagonal entries.",
    "num2t-newton-opt-hessian-rows-swapped": "Applying the inverse Hessian with the components of the gradient in the wrong order.",
    "num2t-newton-opt-value-at-start": "Giving f at the starting point instead of at the new point.",
    "num2t-newton-opt-value-at-gradient-step": "Evaluating f at the gradient-step point.",
    "num2t-newton-opt-value-at-sign-reversed": "Evaluating f at the point reached by adding the Newton correction.",
    "num2t-newton-opt-value-at-diagonal": "Evaluating f at the point from a diagonal-Hessian step.",
    "num2t-newton-opt-det-main-diagonal-only": "Giving the product of the diagonal entries as the Hessian determinant.",
    "num2t-newton-opt-det-sum-of-products": "Adding the product of the off-diagonal entries instead of subtracting it.",
    "num2t-newton-opt-det-trace": "Giving the trace of the Hessian in place of its determinant.",
    "num2t-newton-opt-det-first-derivatives": "Building the determinant from the first partial derivatives.",
    "num2t-newton-opt-coordinate-unchanged": "Giving the starting coordinate: the step is not applied.",
    "num2t-newton-opt-coordinate-other-sign": "Adding the correction to the coordinate.",
    "num2t-newton-opt-coordinate-no-determinant": "Leaving out the division by the determinant in the new coordinate.",
    "num2t-newton-opt-coordinate-swapped-cross": "Using the mixed second derivative with the wrong gradient component.",
    "num2t-newton-opt-change-new-value": "Giving f at the new point instead of the change in f.",
    "num2t-newton-opt-change-sign": "Giving the change in f as the old value minus the new value.",
    "num2t-newton-opt-change-start-value": "Giving f at the start as the change.",
}
M_ROOT = "num2t-newton-opt-root-finding-step"
M_SIGN = "num2t-newton-opt-step-sign"
M_NO2 = "num2t-newton-opt-no-second-derivative"
M_INV = "num2t-newton-opt-ratio-inverted"
M_CORR = "num2t-newton-opt-correction-only"
M_ORIGIN = "num2t-newton-opt-derivative-at-wrong-point"
M_GRAD = "num2t-newton-opt-gradient-step"
M_SIGN2 = "num2t-newton-opt-sign-reversed"
M_HMUL = "num2t-newton-opt-hessian-not-inverted"
M_DIAG = "num2t-newton-opt-diagonal-hessian"
M_ADJ = "num2t-newton-opt-inverse-unscaled"
M_NOSWAP = "num2t-newton-opt-inverse-no-swap"
M_ORDER = "num2t-newton-opt-hessian-rows-swapped"


# ---------------------------------------------------------------------------------------------------------------
# one variable


def d1(c):
    return [k * c[k] for k in range(1, len(c))]


def _easy(rng):
    for _ in attempts(2000, "a quartic"):
        c = [rng.randint(-4, 4) for _ in range(5)]
        c[4] = rng.choice([1, 1, 2])
        c[3] = rng.choice([0, 0, -2, -1, 1, 2])
        c[2] = rng.choice([-3, -2, -1, 1, 2, 3, 4])
        c[1] = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        x0 = Fraction(rng.choice([-2, -1, 1, 2, 3]))
        g, h = poly_eval(d1(c), x0), poly_eval(d1(d1(c)), x0)
        f0 = poly_eval(c, x0)
        if h <= 0 or g == 0 or f0 == 0 or poly_eval(d1(c), Fraction(0)) == g:
            continue
        x1 = x0 - g / h
        if abs(x1) > 30 or x1 == x0:
            continue
        cands = [
            (M_ROOT, x0 - f0 / g),
            (M_SIGN, x0 + g / h),
            (M_NO2, x0 - g),
            (M_INV, x0 - h / g),
            (M_CORR, g / h),
            (M_ORIGIN, -poly_eval(d1(c), 0) / poly_eval(d1(d1(c)), 0) if poly_eval(d1(d1(c)), 0) != 0 else None),
        ]
        try:
            part = fraction_part("", x1, cands, 1, min_options=3)
        except Redraw:
            continue
        break
    ft = poly_prompt(c)
    x0t = tx(x0)
    stem = rng.choice(
        [
            f"Newton's method is used to minimise «f(x) = {ft}». Starting from «x₀ = {x0t}», find the next iterate «x₁» "
            "as a fraction or whole number.",
            f"Take one step of Newton's method for optimisation from «x₀ = {x0t}» on «f(x) = {ft}». What is «x₁»?",
            f"To find a local minimum of «f(x) = {ft}», one iteration of the Newton optimisation method is made from "
            f"the guess «{x0t}». Give the new guess as a fraction or whole number.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"f′(x₀) = {tx(g)} and f″(x₀) = {tx(h)}.",
            f"x₁ = x₀ − f′(x₀)/f″(x₀) = {x0t} − {tx(g / h)} = {tx(x1)}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# two variables: polynomials as {(i, j): coefficient}


def ev(f, x, y):
    return sum(Fraction(c) * Fraction(x) ** i * Fraction(y) ** j for (i, j), c in f.items())


def dx(f):
    return {(i - 1, j): c * i for (i, j), c in f.items() if i}


def dy(f):
    return {(i, j - 1): c * j for (i, j), c in f.items() if j}


def _var(i, j) -> str:
    sup = {2: "²", 3: "³", 4: "⁴"}
    return ("x" + sup.get(i, "") if i else "") + ("y" + sup.get(j, "") if j else "")


def f_text(f) -> str:
    order = sorted(f, key=lambda k: (-(k[0] + k[1]), -k[0]))
    out = []
    for n, key in enumerate(order):
        c = f[key]
        mag = abs(c)
        body = _var(*key)
        txt = str(mag) if not body else (body if mag == 1 else f"{mag}{body}")
        sign = "−" if c < 0 else "+"
        out.append(f"−{txt}" if (n == 0 and c < 0) else txt if n == 0 else f"{sign} {txt}")
    return " ".join(out)


class Setup:
    def __init__(self, f, x0, y0):
        self.f, self.x0, self.y0 = f, Fraction(x0), Fraction(y0)
        self.fx, self.fy = dx(f), dy(f)
        self.fxx, self.fxy, self.fyy = dx(self.fx), dy(self.fx), dy(self.fy)
        self.gx, self.gy = ev(self.fx, x0, y0), ev(self.fy, x0, y0)
        self.h11, self.h12, self.h22 = ev(self.fxx, x0, y0), ev(self.fxy, x0, y0), ev(self.fyy, x0, y0)
        self.det = self.h11 * self.h22 - self.h12**2

    def step(self):
        return (
            (self.h22 * self.gx - self.h12 * self.gy) / self.det,
            (self.h11 * self.gy - self.h12 * self.gx) / self.det,
        )

    def new(self):
        sx, sy = self.step()
        return self.x0 - sx, self.y0 - sy

    def at(self, x, y):
        return ev(self.f, x, y)


def draw_setup(rng):
    for _ in attempts(5000, "a two-variable function"):
        f = {
            (4, 0): rng.choice([1, 1, 2]),
            (2, 0): rng.choice([-2, -1, 1, 2, 3]),
            (0, 2): rng.choice([1, 2, 3, 4]),
            (1, 1): rng.choice([-2, -1, 1, 2, 3]),
            (1, 0): rng.choice([-3, -2, -1, 1, 2, 3]),
            (0, 1): rng.choice([-3, -2, -1, 1, 2, 3]),
        }
        if rng.random() < 0.5:
            f[(0, 4)] = 1
        if rng.random() < 0.4:
            f[(2, 1)] = rng.choice([-1, 1])
        s = Setup(f, rng.choice([-1, 0, 1, 2]), rng.choice([-1, 0, 1, 2]))
        if s.h11 <= 0 or s.det <= 0 or s.det > 150 or (s.gx == 0 and s.gy == 0):
            continue
        nx, ny = s.new()
        if (nx, ny) == (s.x0, s.y0) or max(abs(nx), abs(ny)) > 12:
            continue
        if nx.denominator > 160 or ny.denominator > 160:
            continue
        return s
    raise AssertionError("unreachable")


def _alt_points(s):
    """Wrong-method points for the medium and hard tiers, as (tag, (x, y))."""
    sx, sy = s.step()
    out = [
        (M_SIGN2, (s.x0 + sx, s.y0 + sy)),
        (M_GRAD, (s.x0 - s.gx, s.y0 - s.gy)),
        (M_HMUL, (s.x0 - (s.h11 * s.gx + s.h12 * s.gy), s.y0 - (s.h12 * s.gx + s.h22 * s.gy))),
    ]
    if s.h11 and s.h22:
        out.append((M_DIAG, (s.x0 - s.gx / s.h11, s.y0 - s.gy / s.h22)))
    out.append((M_ADJ, (s.x0 - (s.h22 * s.gx - s.h12 * s.gy), s.y0 - (s.h11 * s.gy - s.h12 * s.gx))))
    out.append((M_NOSWAP, (s.x0 - (s.h11 * s.gx - s.h12 * s.gy) / s.det, s.y0 - (s.h22 * s.gy - s.h12 * s.gx) / s.det)))
    out.append((M_ORDER, (s.x0 - (s.h22 * s.gy - s.h12 * s.gx) / s.det, s.y0 - (s.h11 * s.gx - s.h12 * s.gy) / s.det)))
    return out


def _plausible(v):
    return abs(v) <= 400 and v.denominator <= 500


@generator(
    "r4h_newton_optimisation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_newton_optimisation(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "a Newton optimisation question")


def _single(rng, tier):
    if tier == "easy":
        return _easy(rng)
    s = draw_setup(rng)
    nx, ny = s.new()
    ft, x0t, y0t = f_text(s.f), tx(s.x0), tx(s.y0)
    if tier == "medium":
        which = rng.choice(["x", "y"])
        value = nx if which == "x" else ny
        pick = 0 if which == "x" else 1
        cands = [(tag, pt[pick]) for tag, pt in _alt_points(s) if _plausible(pt[pick])]
        part = fraction_part("", value, cands, 1, min_options=3)
        coord = f"«{which}₁»"
        stem = rng.choice(
            [
                f"Newton's method is used to minimise «f(x, y) = {ft}». From «(x₀, y₀) = ({x0t}, {y0t})», find the "
                f"{which}-coordinate {coord} of the next iterate, as a fraction or whole number.",
                f"Take one Newton step for optimisation on «f(x, y) = {ft}», starting at «({x0t}, {y0t})». What is "
                f"{coord}, the new {which}-coordinate?",
                f"One iteration of the Newton optimisation method is applied to «{ft}» from the point «({x0t}, {y0t})». "
                f"Give the new {which}-coordinate {coord} as a fraction or whole number.",
            ]
        )
        return single_from(
            part,
            stem,
            working=[
                f"∇f = ({tx(s.gx)}, {tx(s.gy)}) and H = [[{tx(s.h11)}, {tx(s.h12)}], [{tx(s.h12)}, {tx(s.h22)}]], det H = {tx(s.det)}.",
                f"x₁ = x₀ − H⁻¹∇f = ({tx(nx)}, {tx(ny)}).",
            ],
        )
    value = s.at(nx, ny)
    cands = [
        ("num2t-newton-opt-value-at-start", s.at(s.x0, s.y0)),
        ("num2t-newton-opt-value-at-gradient-step", s.at(s.x0 - s.gx, s.y0 - s.gy)),
        ("num2t-newton-opt-value-at-sign-reversed", s.at(s.x0 + (s.x0 - nx), s.y0 + (s.y0 - ny))),
    ]
    if s.h11 and s.h22:
        cands.append(("num2t-newton-opt-value-at-diagonal", s.at(s.x0 - s.gx / s.h11, s.y0 - s.gy / s.h22)))
    cands = [(t, v) for t, v in cands if v is not None and _plausible(v)]
    part = fraction_part("", value, cands, 1, min_options=3)
    stem = rng.choice(
        [
            f"One Newton optimisation step is taken on «f(x, y) = {ft}» from «({x0t}, {y0t})». Find the value of «f» at "
            "the new point, as a fraction or whole number.",
            f"Starting at «({x0t}, {y0t})», Newton's method for minimising «{ft}» moves to a new point. What is «f» there?",
            f"For «f(x, y) = {ft}», apply Newton's method for optimisation once, starting from «({x0t}, {y0t})», and "
            "evaluate «f» at the point you reach. Give it as a fraction or whole number.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"The step takes ({x0t}, {y0t}) to ({tx(nx)}, {tx(ny)}).",
            f"f({tx(nx)}, {tx(ny)}) = {tx(value)}.",
        ],
    )


@generator("exd_x_r4h_newton_optimisation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_newton_optimisation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Newton optimisation question")


def _exam(rng):
    s = draw_setup(rng)
    nx, ny = s.new()
    ft, x0t, y0t = f_text(s.f), tx(s.x0), tx(s.y0)
    f0 = s.at(s.x0, s.y0)
    f1 = s.at(nx, ny)
    change = f1 - f0
    h11, h12, h22, gx, gy = s.h11, s.h12, s.h22, s.gx, s.gy
    p_a = fraction_part(
        "Find the determinant of the Hessian matrix of «f» at the starting point.",
        s.det,
        [
            ("num2t-newton-opt-det-main-diagonal-only", h11 * h22),
            ("num2t-newton-opt-det-sum-of-products", h11 * h22 + h12 * h12),
            ("num2t-newton-opt-det-trace", h11 + h22),
            ("num2t-newton-opt-det-first-derivatives", gx * gy - h12 * h12),
        ],
        1,
        working=[
            f"H = {f'[[{tx(h11)}, {tx(h12)}], [{tx(h12)}, {tx(h22)}]]'}, so det H = {tx(h11)}·{tx(h22)} − {tx(h12)}² = {tx(s.det)}."
        ],
        value_range=(0, None),
    )
    da = ft_symbol("part_a")
    p_b = fraction_part(
        "Use your determinant to find the new «x»-coordinate «x₁» after one Newton step.",
        nx,
        [
            ("num2t-newton-opt-coordinate-unchanged", s.x0),
            ("num2t-newton-opt-coordinate-other-sign", s.x0 + (s.x0 - nx)),
            ("num2t-newton-opt-coordinate-no-determinant", s.x0 - (h22 * gx - h12 * gy)),
            ("num2t-newton-opt-coordinate-swapped-cross", s.x0 - (h22 * gx - h12 * gx) / s.det),
            (M_DIAG, s.x0 - gx / h11),
        ],
        1,
        ft=str(rat(s.x0) - (rat(h22 * gx - h12 * gy)) / da),
        working=[f"x₁ = {x0t} − ({tx(h22)}·{tx(gx)} − {tx(h12)}·{tx(gy)})/{tx(s.det)} = {tx(nx)}."],
    )
    p_c = fraction_part(
        "Use your determinant to find the new «y»-coordinate «y₁».",
        ny,
        [
            ("num2t-newton-opt-coordinate-unchanged", s.y0),
            ("num2t-newton-opt-coordinate-other-sign", s.y0 + (s.y0 - ny)),
            ("num2t-newton-opt-coordinate-no-determinant", s.y0 - (h11 * gy - h12 * gx)),
            ("num2t-newton-opt-coordinate-swapped-cross", s.y0 - (h11 * gy - h12 * gy) / s.det),
            (M_DIAG, s.y0 - gy / h22),
        ],
        1,
        ft=str(rat(s.y0) - (rat(h11 * gy - h12 * gx)) / da),
        working=[f"y₁ = {y0t} − ({tx(h11)}·{tx(gy)} − {tx(h12)}·{tx(gx)})/{tx(s.det)} = {tx(ny)}."],
    )
    xb, yc = ft_symbol("part_b"), ft_symbol("part_c")
    expr = sum((rat(c) * xb**i * yc**j for (i, j), c in s.f.items()), rat(0)) - rat(f0)
    p_d = number_part(
        "Find the change in «f» from the start to the new point, «f(x₁, y₁) − f(x₀, y₀)», using your coordinates. "
        "Give the answer correct to 4 decimal places.",
        change,
        4,
        [
            ("num2t-newton-opt-change-new-value", f1),
            ("num2t-newton-opt-change-sign", -change),
            ("num2t-newton-opt-change-start-value", f0),
        ],
        2,
        ft=str(expr),
        working=[f"f(x₀, y₀) = {tx(f0)} and f(x₁, y₁) = {tx(f1)}, a change of {tx(change)}."],
    )
    stem = rng.choice(
        [
            f"Newton's method for optimisation is applied to «f(x, y) = {ft}», starting at «(x₀, y₀) = ({x0t}, {y0t})».",
            f"A minimum of «f(x, y) = {ft}» is sought by Newton's method from the point «({x0t}, {y0t})».",
            f"Consider «f(x, y) = {ft}» and one Newton optimisation step from «({x0t}, {y0t})».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)
