"""Generates `s2a1_reverse_order` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_reverse_order`.

``uc2-reverse-order-of-integration`` (``multivariable-and-vector-calculus``, answer ``expression``): an iterated
integral whose inner integral has no elementary antiderivative in the order given, so it can only be done after the
order of integration is reversed. The inner integrand depends on the inner variable alone (e^(y²), e^(y³), e^(y⁴),
√(1+y³), 1/√(1+y³), sin(y)/y). The outer variable runs from 0 to V = φ(U) and the inner from φ⁻¹(outer) to U, so
the region is {0 ≤ v ≤ φ(u), 0 ≤ u ≤ U} and the reversed integral is ∫₀^U φ(u)·h(u) du, which a substitution does.

* easy: e^(y²) over the triangle under a line through the origin, written as an iterated integral.
* medium: the same idea with a parabola bound (e^(y³), √(1+y³), 1/√(1+y³)) and with sin(y)/y between 0 and π/2 or π.
* hard: a cube-root bound (e^(y⁴)), sin(y)/y over a parabolic region, and questions that give the REGION (a triangle
  or a curve and two lines) instead of the iterated integral, so the learner must draw it and choose the order.
* exam (``exd_x_s2a1_reverse_order``): (a) the area of the region of integration, (b) the reversed inner integral at
  one point, (c) the value of the double integral, (d) the mean value of the integrand over the region
  (follows through from (a) and (c)).

Each key is worked out twice. The exact route is the closed-form primitive G of φ·h (checked to differentiate back to
φ·h). The second route is in tests/test_s2a1_reverse.py: it integrates in the ORIGINAL order, which never uses the
reversal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy

from ..core import VerificationError, attempts, check, generator
from ..core import rational_text
from ._s2a1_common import to_fraction, exact_answer, exact_part, exact_wrong, gauss_nested, stem, tidy
from ._ucv1_calc_markup import tex_value, value_string, value_text

u = sympy.Symbol("u", positive=True)
S = sympy.Integer
PI = sympy.pi


# ---------------------------------------------------------------------------------------------------------------
# families


@dataclass(frozen=True)
class Family:
    key: str
    h: object  # sympy expression in u (the integrand, without the coefficient)
    h_num: object  # numpy version
    phi_power: int  # phi(u) = m u^phi_power; m is only free when phi_power == 1
    primitive: object  # function (m) -> G(u) with G' = phi h
    plain: str  # integrand text with {c} and {u}
    tex: str
    integer_us: tuple
    pi_us: tuple
    factor_slip: int  # the constant that substitution forgets: the value comes out this many times too large


FAMILIES = {
    "e2": Family(
        "e2",
        sympy.exp(u**2),
        lambda a: np.exp(a**2),
        1,
        lambda m: m * sympy.exp(u**2) / 2,
        "{c}e^({u}²)",
        r"{c}e^{{{u}^{{2}}}}",
        (1, 2, 3),
        (),
        2,
    ),
    "e3": Family(
        "e3",
        sympy.exp(u**3),
        lambda a: np.exp(a**3),
        2,
        lambda m: sympy.exp(u**3) / 3,
        "{c}e^({u}³)",
        r"{c}e^{{{u}^{{3}}}}",
        (1, 2, 3),
        (),
        3,
    ),
    "e4": Family(
        "e4",
        sympy.exp(u**4),
        lambda a: np.exp(a**4),
        3,
        lambda m: sympy.exp(u**4) / 4,
        "{c}e^({u}⁴)",
        r"{c}e^{{{u}^{{4}}}}",
        (1,),
        (),
        4,
    ),
    "s1": Family(
        "s1",
        sympy.sqrt(1 + u**3),
        lambda a: np.sqrt(1 + a**3),
        2,
        lambda m: 2 * (1 + u**3) ** sympy.Rational(3, 2) / 9,
        "{c}√(1+{u}³)",
        r"{c}\sqrt{{1+{u}^{{3}}}}",
        (1, 2, 3),
        (),
        3,
    ),
    "s2": Family(
        "s2",
        1 / sympy.sqrt(1 + u**3),
        lambda a: 1 / np.sqrt(1 + a**3),
        2,
        lambda m: 2 * sympy.sqrt(1 + u**3) / 3,
        "{c1}/√(1+{u}³)",
        r"\frac{{{c1}}}{{\sqrt{{1+{u}^{{3}}}}}}",
        (1, 2, 3),
        (),
        3,
    ),
    "si1": Family(
        "si1",
        sympy.sin(u) / u,
        lambda a: np.sin(a) / a,
        1,
        lambda m: -m * sympy.cos(u),
        "{c}sin({u})/{u}",
        r"{c}\frac{{\sin {u}}}{{{u}}}",
        (),
        (PI / 2, PI),
        1,
    ),
    "si2": Family(
        "si2",
        sympy.sin(u) / u,
        lambda a: np.sin(a) / a,
        2,
        lambda m: sympy.sin(u) - u * sympy.cos(u),
        "{c}sin({u})/{u}",
        r"{c}\frac{{\sin {u}}}{{{u}}}",
        (),
        (PI / 2, PI),
        1,
    ),
}


_OPENINGS = ("Evaluate the iterated integral",)


class Problem:
    """One iterated integral: coefficient c, family, slope m (phi = m u^k), the inner bound U and the letters."""

    def __init__(self, fam: Family, c: int, m: int, U, outer: str, inner: str):
        self.fam, self.c, self.m, self.U, self.outer, self.inner = fam, c, m, sympy.sympify(U), outer, inner
        self.opening = _OPENINGS[0]
        self.k = fam.phi_power
        self.V = sympy.simplify(m * self.U**self.k)
        self.G = fam.primitive(m)
        self.value = sympy.simplify(c * (self.G.subs(u, self.U) - self.G.subs(u, 0)))
        self.area = sympy.simplify(m * self.U ** (self.k + 1) / (self.k + 1))

    # -- exact checks ------------------------------------------------------------------------------------------

    def verify(self) -> None:
        phi_h = self.m * u**self.k * self.fam.h
        for point in (0.4, 0.9, 1.3):
            slope = float(sympy.diff(self.G, u).subs(u, point))
            check(abs(slope - float(phi_h.subs(u, point))) < 1e-9, "the primitive does not differentiate back")

    # -- text --------------------------------------------------------------------------------------------------

    def lower_plain(self) -> str:
        v = self.outer
        return {1: (v if self.m == 1 else f"{v}/{self.m}"), 2: f"√{v}", 3: f"∛{v}"}[self.k]

    def lower_tex(self) -> str:
        v = self.outer
        if self.k == 1:
            return v if self.m == 1 else rf"\frac{{{v}}}{{{self.m}}}"
        return rf"\sqrt{{{v}}}" if self.k == 2 else rf"\sqrt[3]{{{v}}}"

    def h_plain(self, coefficient=True) -> str:
        c = "" if self.c == 1 else str(self.c)
        text = self.fam.plain.format(c=c if coefficient else "", c1=c if coefficient and c else 1, u=self.inner)
        if self.fam.key.startswith("si") and c:
            text = text.replace(c, c + " ", 1)
        return text

    def h_tex(self) -> str:
        c = "" if self.c == 1 else str(self.c)
        return self.fam.tex.format(c=c, c1=self.c, u=self.inner)

    def iterated_stem(self) -> dict:
        v, w = self.outer, self.inner
        V, U = _lim(self.V), _lim(self.U)
        V_tex, U_tex = tex_value(self.V), tex_value(self.U)
        text = (
            f"{self.opening} ∫ from {v} = 0 to {V} of ∫ from {w} = {self.lower_plain()} to {U} of "
            f"{self.h_plain()} d{w} d{v}."
        )
        latex = (
            rf"\text{{{self.opening} }}"
            rf"\int_{{0}}^{{{V_tex}}}\int_{{{self.lower_tex()}}}^{{{U_tex}}} {self.h_tex()}\,\mathrm{{d}}{w}\,\mathrm{{d}}{v}"
            r"\text{.}"
        )
        return dict(text=text, latex=latex)

    def region_words(self) -> str:
        """The region {0 ≤ v ≤ m u^k, 0 ≤ u ≤ U} in words, with vertices or curves in (x, y) order."""
        v, w, U, V = self.outer, self.inner, self.U, self.V
        if self.k == 1:
            pts = [(0, 0), (0, U), (V, U)] if v == "x" else [(0, 0), (U, 0), (U, V)]
            a, b, c = (f"«({_lim(p)}, {_lim(q)})»" for p, q in pts)
            return f"the triangle with vertices {a}, {b} and {c}"
        power = {2: "²", 3: "³"}[self.k]
        return f"the region in the first quadrant bounded by the curve «{v} = {w}{power}», the line «{w} = {_lim(U)}» and the «{w}»-axis"

    def region_stem(self) -> dict:
        return stem(f"Evaluate «∬ {self.h_plain()} dA», where R is {self.region_words()}.")

    # -- wrong answers -----------------------------------------------------------------------------------------

    def candidates(self):
        fam, c, m, U, V = self.fam, self.c, self.m, self.U, self.V
        G0, GU = self.G.subs(u, 0), self.G.subs(u, U)
        value = self.value
        out = [
            ("reverse-order-limits-swapped", -value),
            ("reverse-order-substitution-factor", value * fam.factor_slip),
            ("reverse-order-substitution-factor", value / fam.factor_slip) if fam.factor_slip != 1 else None,
            ("reverse-order-lower-limit-forgotten", c * GU),
            ("reverse-order-lower-limit-added", c * (GU + G0)),
            ("reverse-order-coefficient-dropped", value / c) if c != 1 else None,
            ("reverse-order-coefficient-squared", value * c) if c != 1 else None,
            ("reverse-order-swapped-extent", c * (self.G.subs(u, V) - G0))
            if V != U and fam.key in ("e2", "si1")
            else None,
        ]
        if fam.key == "e2" and m != 1:
            out.append(("reverse-order-slope-misread", c * (sympy.exp(U**2) - 1) / (2 * m)))
            out.append(("reverse-order-slope-dropped", c * (sympy.exp(U**2) - 1) / 2))
        if fam.key == "si1" and m != 1:
            out.append(("reverse-order-slope-dropped", value / m))
        if fam.key == "si2":
            out.append(("reverse-order-product-rule-slip", c * (sympy.sin(U) + U * sympy.cos(U))))
        return [item for item in out if item is not None]


# ---------------------------------------------------------------------------------------------------------------
# answers and distractors


def _lim(value) -> str:
    return value_text(value).replace("π^2", "π²")


# ---------------------------------------------------------------------------------------------------------------
# tiers


def _letters(rng):
    return ("x", "y") if rng.random() < 0.65 else ("y", "x")


def _pick_u(rng, fam: Family, integers_only=False):
    pool = list(fam.integer_us) if (integers_only or not fam.pi_us) else list(fam.integer_us) + list(fam.pi_us)
    return rng.choice(pool or list(fam.pi_us))


def _easy(rng):
    outer, inner = _letters(rng)
    problem = Problem(
        FAMILIES["e2"], rng.choice(range(1, 10)), rng.choice(range(1, 7)), rng.choice([1, 2, 3]), outer, inner
    )
    problem.opening = rng.choice(_OPENINGS)
    return problem, problem.iterated_stem()


def _medium(rng):
    outer, inner = _letters(rng)
    key = rng.choice(["e3", "s1", "s2", "si1", "si1"])
    fam = FAMILIES[key]
    m = rng.choice([1, 2, 3, 4, 5]) if key == "si1" else 1
    problem = Problem(fam, rng.choice(range(1, 10)), m, _pick_u(rng, fam), outer, inner)
    problem.opening = rng.choice(_OPENINGS)
    return problem, problem.iterated_stem()


def _hard(rng):
    outer, inner = _letters(rng)
    pick = rng.random()
    if pick < 0.3:
        fam, m = FAMILIES["e4"], 1
        problem = Problem(fam, rng.choice([1, 2, 3]), m, _pick_u(rng, fam), outer, inner)
        return problem, problem.iterated_stem()
    if pick < 0.5:
        fam = FAMILIES["si2"]
        problem = Problem(fam, rng.choice([1, 2, 3]), 1, _pick_u(rng, fam), outer, inner)
        return problem, problem.iterated_stem()
    key = rng.choice(["e2", "e2", "e3", "e3", "s1", "s2"])
    fam = FAMILIES[key]
    m = rng.choice([2, 3, 4]) if key == "e2" else 1
    problem = Problem(fam, rng.choice([1, 2, 3, 5]), m, _pick_u(rng, fam, integers_only=True), outer, inner)
    return problem, problem.region_stem()


@generator(
    "s2a1_reverse_order",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a1_reverse_order(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(60, "a reverse-order integral with enough distractors"):
        problem, out = build(rng)
        try:
            problem.verify()
        except VerificationError:
            continue
        value = problem.value
        wrong = exact_wrong(value, problem.candidates(), 4, allow_negative=True)
        if len(wrong) >= 3 and float(value) > 0:
            break
    kind, machine, answer_text = exact_answer(value)
    v, w, U = problem.outer, problem.inner, value_text(problem.U)
    bound = (str(problem.m) if problem.m != 1 else "") + w + {1: "", 2: "²", 3: "³"}[problem.k]
    working = [
        f"The inner integral of {problem.h_plain(False)} with respect to {w} has no elementary antiderivative, so reverse the order.",
        f"The region is 0 ≤ {w} ≤ {U} and 0 ≤ {v} ≤ {bound}.",
        f"Integrate {v} first: its length {bound} becomes a factor of the integrand.",
        f"Then substitute and evaluate between 0 and {U} to get {answer_text}.",
    ]
    return dict(
        **out,
        answer=machine,
        answer_text=answer_text,
        answer_latex=tex_value(tidy(value)),
        answer_kind=kind,
        distractors=wrong,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    outer, inner = _letters(rng)
    key = rng.choice(["e2", "e2", "e3", "s1", "s2"])
    fam = FAMILIES[key]
    m = rng.choice([1, 2, 3, 4]) if key == "e2" else 1
    problem = Problem(fam, rng.choice([1, 1, 2, 3]), m, rng.choice([1, 2]) if key != "e3" else 2, outer, inner)
    problem.verify()
    v, w = outer, inner
    c, U, V, k = problem.c, problem.U, problem.V, problem.k
    check(V != U, "the region is a square")
    area = problem.area
    # (b) the reversed inner integral at w = 1: phi(1) * c * h(1)
    g1 = sympy.simplify(c * problem.m * fam.h.subs(u, 1))
    value = problem.value
    # candidate wrong areas
    a_c = [
        ("reverse-order-bounding-rectangle", U * V),
        ("reverse-order-curve-area-as-triangle", U * V / 2)
        if k > 1
        else ("reverse-order-complement-area", U * V - area),
        ("reverse-order-complement-area", U * V - area),
        ("reverse-order-slope-dropped", U**2 / 2),
    ]
    b_c = [
        ("reverse-order-constant-inner-limit", c * V * fam.h.subs(u, 1)),
        ("reverse-order-derivative-of-limit", c * k * problem.m * fam.h.subs(u, 1)),
        ("reverse-order-coefficient-dropped", fam.h.subs(u, 1)),
        ("reverse-order-slope-misread", c * fam.h.subs(u, 1) / problem.m),
        ("reverse-order-constant-inner-limit", c * U * fam.h.subs(u, 1)),
    ]
    c_c = problem.candidates()
    d_c = [
        ("region-average-not-divided", value),
        ("reverse-order-mean-inverted", area / value),
        ("reverse-order-mean-wrong-area", value / (U * V)),
    ] + [(t, cv / area) for t, cv in problem.candidates()]
    stem_dict = problem.iterated_stem()
    stem_dict = dict(
        text=stem_dict["text"]
        .replace("Evaluate the iterated integral", "Let I be the iterated integral", 1)
        .rstrip(".")
        + ", and let R be its region of integration.",
        latex=stem_dict["latex"]
        .replace(r"\text{Evaluate the iterated integral }", r"\text{Let }I=", 1)
        .replace(r"\text{.}", r"\text{, and let }R\text{ be its region of integration.}"),
    )
    parts = [
        exact_part(
            "Find the area of the region of integration R, in square units.",
            area,
            1,
            ["Describe R by 0 ≤ " + v + " ≤ the bound, 0 ≤ " + w + " ≤ " + _lim(U) + " and integrate 1 over it."],
            a_c,
            sensible_range=(0, None),
        ),
        exact_part(
            f"Reverse the order of integration, so that «{v}» is integrated first. Find the value of that inner integral when «{w} = 1».",
            g1,
            2,
            [f"Integrating the integrand over «{v}» from 0 to the bound at «{w} = 1» multiplies it by the bound."],
            b_c,
        ),
        exact_part(
            "Find the exact value of I.",
            value,
            3,
            [f"I = ∫ from {w} = 0 to {_lim(U)} of (the inner integral) d{w}, done by a substitution."],
            c_c,
        ),
        exact_part(
            "Find the mean value of the integrand over R.",
            value / area,
            1,
            ["Mean value = I / area of R."],
            d_c,
            follow_through=f"part_c/({rational_text(to_fraction(area))})",
            sensible_range=(0, None),
        ),
    ]
    check(all(parts), "a part has too few distractors")
    return dict(**stem_dict, parts=parts)


@generator("exd_x_s2a1_reverse_order", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_reverse_order(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam reverse-order chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
