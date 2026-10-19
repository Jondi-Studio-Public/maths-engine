"""Shared machinery for the second-year vector-calculus builds (`s2a3_*`): flux, Green, divergence, Stokes.

Everything here works on a *parametrised domain* `Dom`: a rectangle in some parameters, where a parameter that is an
angle is carried as the pair (cos, sin) of exact symbols and ranges over a whole number of quarter-turns. That is
enough for every shape the builds use: a rectangle, a triangle (a rectangle squashed by one parameter), a disc, a
sphere, a cylinder, a cone, a paraboloid bowl. On such a domain

* `integrate(expr, dom)` is EXACT: it expands the integrand into monomials and integrates each one with the Wallis
  formula for the trigonometric ones, so the answers come out as ``a + b*pi`` without a symbolic integrator;
* `numeric(expr, dom)` is a second, independent route: Gauss-Legendre quadrature of the same integrand with the
  angles turned back into ``cos`` and ``sin``.

A `Surf` is a parametrised surface (flux), a `Curve` a parametrised curve (line integral) and a `Solid` a
parametrised solid (volume integral). Each can be integrated exactly and numerically, and the generators check the
two agree before they believe either.

The module also holds the writing helpers shared by the builds: polynomials as plain text and LaTeX, a field as
``P i + Q j + R k``, a stem whose maths fragments are marked « » and set as maths, and the distractor filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import types
from functools import lru_cache

from sympy.polys.rings import PolyElement
from types import SimpleNamespace

import numpy as np
import sympy as sp

from ..answers import Distractor
from ..core import _escape_prose, attempts, check, paper_notation, value_safe_str
from ..grading import grade

x, y, z = sp.symbols("x y z")
u, v, w, t = sp.symbols("u v w t")
# cos and sin of an angle parameter, as exact symbols: (C, S) for the first angle, (CP, SP) for a second one
C, S, CP, SP = sp.symbols("C S CP SP")


# ---------------------------------------------------------------------------------------------------------------
# domains and exact / numerical integration


@dataclass(frozen=True)
class Dom:
    """A rectangle in the parameters `vars`. `ranges[i]` is (lo, hi); for an angle (a key of `angles`) it is a pair of
    whole quarter-turns, so (0, 4) is a full turn. `angles[var] = (c, s)` names the symbols standing for cos and sin."""

    vars: tuple
    ranges: tuple
    angles: dict

    def gens(self):
        out = []
        for var in self.vars:
            out.extend(self.angles[var] if var in self.angles else [var])
        return out

    def mid(self):
        point = {}
        for var, (lo, hi) in zip(self.vars, self.ranges):
            if var in self.angles:
                c, s = self.angles[var]
                a = float(sp.pi) * (lo + hi) / 4
                point[c], point[s] = np.cos(a), np.sin(a)
            else:
                point[var] = (lo + hi) / 2
        return point


def D(expr, var, dom: Dom):
    """d(expr)/d(var), where an angle parameter's cos and sin symbols differentiate into one another."""
    out = sp.diff(expr, var)
    if var in dom.angles:
        c, s = dom.angles[var]
        out += sp.diff(expr, c) * (-s) + sp.diff(expr, s) * c
    return out


def _dfact(n: int) -> int:
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


@lru_cache(maxsize=None)
def _wallis(a: int, b: int):
    """Integral of cos^a sin^b over the first quarter-turn, exactly."""
    value = sp.Rational(_dfact(a - 1) * _dfact(b - 1), _dfact(a + b))
    return value * sp.pi / 2 if a % 2 == 0 and b % 2 == 0 else value


def _quarters(a: int, b: int, q0: int, q1: int):
    total = sp.Integer(0)
    for k in range(q0, q1):
        sign = (-1 if k % 4 in (1, 2) else 1) ** a * (-1 if k % 4 in (2, 3) else 1) ** b
        total += sign * _wallis(a, b)
    return total


#: The cross-checks: the quadrature of every exact integral and the second exact route of each theorem. Off while
#: questions are generated, so that `make()` stays fast; the tests switch it on (`tests/test_s2a3_cross_checks.py`) and
#: require every route to agree.
CHECKS = False

_GL = {n: np.polynomial.legendre.leggauss(n) for n in (16, 40)}


def _numeric_poly(terms, dom: Dom) -> float:
    """Gauss-Legendre quadrature of the polynomial in the domain's generators, with each angle's pair of generators
    evaluated as its cosine and sine (more nodes for an angle, whose integrand is a trigonometric polynomial)."""
    axes, wts, values, axis_of = [], [], [], []
    for i, (var, (lo, hi)) in enumerate(zip(dom.vars, dom.ranges)):
        nodes, weights = _GL[40 if var in dom.angles else 16]
        if var in dom.angles:
            lo, hi = float(sp.pi) * lo / 2, float(sp.pi) * hi / 2
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        node = mid + half * nodes
        axes.append(node)
        wts.append(half * weights)
        pair = (np.cos(node), np.sin(node)) if var in dom.angles else (node,)
        values.extend(pair)
        axis_of.extend([i] * len(pair))
    shape = [len(a) for a in axes]
    grids = []
    for vals, ax in zip(values, axis_of):
        sh = [1] * len(shape)
        sh[ax] = shape[ax]
        grids.append(vals.reshape(sh))
    total = np.zeros(shape)
    for monom, coeff in terms:
        term = float(coeff) * np.ones(shape)
        for g, e in zip(grids, monom):
            if e:
                term = term * g**e
        total += term
    for i, wt in enumerate(wts):
        sh = [1] * len(shape)
        sh[i] = len(wt)
        total = total * wt.reshape(sh)
    return float(total.sum())


@lru_cache(maxsize=None)
def _ring(gens: tuple):
    return sp.ring(gens, sp.QQ)[0]


def compose(parts, dom: Dom):
    """The polynomial sum of g(sub) * factor over `parts`, each (g, sub, factor): g a polynomial in the keys of the
    substitution `sub`, each replaced by a polynomial in the domain's generators, and `factor` another. Worked out in a
    polynomial ring, which is far faster than expanding the sympy expressions."""
    R = _ring(tuple(dom.gens()))
    total = R.zero
    for g, sub, factor in parts:
        keys = list(sub)
        images = [R.from_expr(sp.sympify(sub[k])) for k in keys]
        gp = sp.Poly(sp.sympify(g), *keys)
        check(gp.domain.is_QQ or gp.domain.is_ZZ, f"integrand has stray symbols {gp.free_symbols}")
        acc = R.zero
        for monom, coeff in gp.terms():
            term = R.from_expr(coeff)
            for image, e in zip(images, monom):
                if e:
                    term = term * image**e
            acc += term
        total += acc * R.from_expr(sp.sympify(factor))
    return total


def integrate(expr, dom: Dom, verify: bool = True):
    """The exact integral of the polynomial `expr` in the domain's generators over the domain. With the cross-checks on
    (`CHECKS`) it is also computed by quadrature and the two must agree: the exact route (Wallis formulas on each
    monomial) and the numerical route share nothing but the polynomial."""
    gens = dom.gens()
    if isinstance(expr, PolyElement):
        terms = [(m, expr.ring.domain.to_sympy(c)) for m, c in expr.terms()]
    else:
        poly = sp.Poly(expr, *gens)  # expands in a polynomial ring, far faster than `sp.expand`
        check(poly.domain.is_QQ or poly.domain.is_ZZ, f"integrand has stray symbols {poly.free_symbols - set(gens)}")
        terms = poly.terms()
    if not terms or terms == [((0,) * len(gens), 0)]:
        return sp.Integer(0)
    by_power: dict[int, sp.Rational] = {}
    for monom, coeff in terms:
        term, k, power = sp.Rational(coeff), 0, 0
        for var, (lo, hi) in zip(dom.vars, dom.ranges):
            if var in dom.angles:
                piece = _quarters(monom[k], monom[k + 1], lo, hi)
                k += 2
                ratio = piece / sp.pi
                if piece == 0:
                    term = sp.Integer(0)
                elif ratio.is_Rational:
                    term, power = term * ratio, power + 1
                else:
                    term *= piece
            else:
                n = monom[k]
                term *= sp.Rational(hi ** (n + 1) - lo ** (n + 1), n + 1)
                k += 1
        by_power[power] = by_power.get(power, sp.Integer(0)) + term
    total = sum((c * sp.pi**k for k, c in by_power.items()), sp.Integer(0))
    if verify and CHECKS:
        approx = _numeric_poly(terms, dom)
        value = float(total)
        check(abs(approx - value) <= 1e-9 * max(1.0, abs(value)), f"exact {value} but quadrature {approx}")
    return total


# ---------------------------------------------------------------------------------------------------------------
# surfaces, curves, solids


def _sub_mid(expr, dom):
    mid = {k: sp.Float(v) for k, v in dom.mid().items()}
    return float(sp.sympify(expr).xreplace(mid))


@dataclass(frozen=True)
class Surf:
    r: sp.Matrix  # a 3-vector of expressions in the parameters
    dom: Dom

    def normal(self):
        if "_normal" not in self.__dict__:
            a, b = self.dom.vars
            self.__dict__["_normal"] = D(self.r, a, self.dom).cross(D(self.r, b, self.dom))
        return self.__dict__["_normal"]

    def at_mid(self):
        return [_sub_mid(c, self.dom) for c in self.r]

    def normal_at_mid(self):
        if "_normal_mid" not in self.__dict__:
            self.__dict__["_normal_mid"] = [_sub_mid(c, self.dom) for c in self.normal()]
        return self.__dict__["_normal_mid"]


def orient(surf: Surf, reference) -> int:
    """+1 or -1: the sign that makes the parametrisation's normal point along `reference`, a vector or a function of
    the point on the surface (at its middle) giving a vector."""
    n = surf.normal_at_mid()
    ref = reference(surf.at_mid()) if callable(reference) else reference
    dot = sum(a * b for a, b in zip(n, ref))
    check(abs(dot) > 1e-9, "the normal is perpendicular to the reference direction at the middle of the piece")
    return 1 if dot > 0 else -1


def outward_from(inside):
    """A reference for `orient`: away from a point inside a convex solid."""
    return lambda p: [a - b for a, b in zip(p, inside)]


def flux_integrand(F, surf: Surf, sign: int):
    sub = dict(zip((x, y, z), surf.r))
    field = sp.Matrix([sp.sympify(c).xreplace(sub) for c in F])
    return sign * field.dot(surf.normal())


def flux(F, surf: Surf, reference, what: str = "flux", verify: bool = True):
    """The exact flux of F through `surf` in the direction `reference`, confirmed by quadrature."""
    sign = orient(surf, reference)
    sub = dict(zip((x, y, z), surf.r))
    normal = surf.normal()
    parts = [(c, sub, sign * n) for c, n in zip(F, normal)]
    return integrate(compose(parts, surf.dom), surf.dom, verify)


@dataclass(frozen=True)
class Curve:
    r: sp.Matrix
    dom: Dom  # one parameter


def line_integrand(F, curve: Curve):
    sub = dict(zip((x, y, z), curve.r))
    var = curve.dom.vars[0]
    field = [sp.sympify(c).xreplace(sub) for c in F]
    return sum(f * D(c, var, curve.dom) for f, c in zip(field, curve.r))


def line(F, curve: Curve, what: str = "line integral", verify: bool = True):
    """The exact ∫ F · dr along `curve` in the direction of its parameter, confirmed by quadrature."""
    sub = dict(zip((x, y, z), curve.r))
    var = curve.dom.vars[0]
    parts = [(f, sub, D(c, var, curve.dom)) for f, c in zip(F, curve.r)]
    return integrate(compose(parts, curve.dom), curve.dom, verify)


def segment(a, b):
    """The straight segment from point a to point b (2 or 3 coordinates), t from 0 to 1."""
    dom = Dom((t,), ((0, 1),), {})
    return Curve(sp.Matrix([sp.Integer(p) + (sp.Integer(q) - sp.Integer(p)) * t for p, q in zip(a, b)]), dom)


def circle_curve(centre, radius, plane_z=None, quarters=(0, 4), height=None):
    """A circle (or arc) traced anticlockwise as seen from above: (cx + R C, cy + R S[, height])."""
    dom = Dom((t,), (tuple(quarters),), {t: (C, S)})
    comps = [centre[0] + radius * C, centre[1] + radius * S]
    if height is not None:
        comps.append(height)
    return Curve(sp.Matrix(comps), dom)


@dataclass(frozen=True)
class Solid:
    r: sp.Matrix
    dom: Dom  # three parameters

    def jacobian(self):
        if "_jacobian" not in self.__dict__:
            a, b, c = self.dom.vars
            self.__dict__["_jacobian"] = sp.Matrix.hstack(
                D(self.r, a, self.dom), D(self.r, b, self.dom), D(self.r, c, self.dom)
            ).det()
        return self.__dict__["_jacobian"]


def solid_integral(g, solid: Solid, what: str = "volume integral", verify: bool = True):
    """Exact ∭ g dV over a parametrised solid, confirmed by quadrature."""
    jac = solid.jacobian()
    sign = 1 if _sub_mid(jac, solid.dom) > 0 else -1
    sub = dict(zip((x, y, z), solid.r))
    return integrate(compose([(sign * sp.sympify(g), sub, jac)], solid.dom), solid.dom, verify)


def region_integral(g, region: Surf, what: str = "area integral", verify: bool = True):
    """Exact ∬ g dA over a flat region parametrised as r = (x(u, v), y(u, v), 0), confirmed by quadrature."""
    jac = region.normal()[2]
    sign = 1 if _sub_mid(jac, region.dom) > 0 else -1
    sub = dict(zip((x, y), region.r[:2]))
    return integrate(compose([(sign * sp.sympify(g), sub, jac)], region.dom), region.dom, verify)


# ---------------------------------------------------------------------------------------------------------------
# the shapes the builds use


@lru_cache(maxsize=None)
def rect_region(x0, x1, y0, y1) -> Surf:
    return Surf(sp.Matrix([u, v, 0]), Dom((u, v), ((x0, x1), (y0, y1)), {}))


@lru_cache(maxsize=None)
def tri_region(a, b, shape: str) -> Surf:
    """A right triangle with legs a (along x) and b (along y).

    ``corner``: x, y >= 0 under the hypotenuse. ``under``: 0 <= y <= (b/a) x, x <= a."""
    if shape == "corner":
        return Surf(sp.Matrix([a * u * (1 - v), b * u * v, 0]), Dom((u, v), ((0, 1), (0, 1)), {}))
    return Surf(sp.Matrix([a * u, b * u * v, 0]), Dom((u, v), ((0, 1), (0, 1)), {}))


@lru_cache(maxsize=None)
def polar_region(radius, q0=0, q1=4, centre=(0, 0)) -> Surf:
    """The sector of a disc: radius in [0, R], angle from q0 to q1 quarter-turns."""
    return Surf(
        sp.Matrix([centre[0] + u * C, centre[1] + u * S, 0]),
        Dom((u, v), ((0, radius), (q0, q1)), {v: (C, S)}),
    )


def graph_over(region: Surf, f) -> Surf:
    """The graph z = f(x, y) over a planar region."""
    sub = {x: region.r[0], y: region.r[1]}
    return Surf(sp.Matrix([region.r[0], region.r[1], sp.sympify(f).xreplace(sub)]), region.dom)


# ---------------------------------------------------------------------------------------------------------------
# writing: polynomials and fields in plain Unicode and LaTeX

_SUP = {2: "²", 3: "³"}


def _num(c: sp.Rational) -> str:
    return str(c.p) if c.q == 1 else f"{c.p}/{c.q}"


def poly_text(expr, gens=(x, y, z)) -> str:
    """A polynomial the way a learner reads it: ``x²y + 3x - 2``."""
    poly = sp.Poly(sp.expand(expr), *gens)
    if poly.is_zero:
        return "0"
    out = []
    for monom, coeff in poly.terms():
        body = "".join(f"{g}{_SUP.get(e, '^' + str(e)) if e > 1 else ''}" for g, e in zip(gens, monom) if e)
        mag = abs(coeff)
        lead = "" if (mag == 1 and body) else _num(mag)
        if body and mag.q != 1:
            lead = f"({lead})"
        piece = lead + body
        out.append(("-" if coeff < 0 else "+", piece))
    first_sign, first = out[0]
    text = ("-" if first_sign == "-" else "") + first
    for sign, piece in out[1:]:
        text += f" {sign} {piece}"
    return text


def _wrap_needed(expr) -> bool:
    expr = sp.expand(expr)
    return len(sp.Add.make_args(expr)) > 1 or expr.could_extract_minus_sign()


class M(str):
    """A maths fragment: the plain text, with its LaTeX beside it."""

    latex: str

    def __new__(cls, plain: str, latex: str):
        obj = super().__new__(cls, plain)
        obj.latex = latex
        return obj


def mpoly(expr, gens=(x, y, z)) -> M:
    return M(poly_text(expr, gens), sp.latex(sp.expand(expr), order="lex"))


def mvec(comps, units="ijk") -> M:
    """``P i + Q j + R k``: a component that is a sum goes in brackets, a single negative term becomes a subtraction,
    and zero components are left out."""
    plain, tex = "", ""
    for c, unit in zip(comps, units):
        c = sp.expand(c)
        if c == 0:
            continue
        multi = len(sp.Add.make_args(c)) > 1
        neg = (not multi) and c.could_extract_minus_sign()
        body = -c if neg else c
        p, l = poly_text(body), sp.latex(body, order="lex")
        if multi:
            p, l = f"({p})", f"\\left({l}\\right)"
        elif body == 1:
            p, l = "", ""
        item_p, item_l = f"{p} {unit}".strip(), f"{l}\\mathbf{{{unit}}}"
        if not plain:
            plain, tex = ("-" if neg else "") + item_p, ("-" if neg else "") + item_l
        else:
            plain += (" - " if neg else " + ") + item_p
            tex += (" - " if neg else " + ") + item_l
    return M(plain, tex)


def tex_math(s: str) -> str:
    """LaTeX for a plain Unicode maths fragment written by these builds."""
    s = re.sub(r"(∬|∭|∮|∫)([A-Z])(?![A-Za-z])", r"\1_{\2}", s)
    s = re.sub(
        r"(?<![A-Za-z\\])d([SAVxyztr])(?![A-Za-z])",
        lambda m: r"\,\mathrm{d}" + (r"\mathbf{r}" if m.group(1) == "r" else m.group(1)),
        s,
    )
    s = re.sub(r"(?<![A-Za-z\\{])([ijk])(?![A-Za-z(])", r"\\mathbf{\1}", s)
    s = re.sub(r"(?<![A-Za-z\\])(curl|div)(?![A-Za-z])", r"\\operatorname{\1}", s)
    s = re.sub(
        r"(?<![A-Za-z\\])(sin|cos)([²³])?",
        lambda m: "\\" + m.group(1) + ("^{" + {"²": "2", "³": "3"}[m.group(2)] + "}" if m.group(2) else ""),
        s,
    )
    s = re.sub(r"\^\(([^()]*)\)", lambda m: "^{" + m.group(1) + "}", s)
    s = re.sub(r"\^(-?\d+)", r"^{\1}", s)
    s = s.replace("²", "^{2}").replace("³", "^{3}")
    s = re.sub(r"√(\d+)", r"\\sqrt{\1}", s)
    for symbol, command in (
        ("≤", r"\le "),
        ("≥", r"\ge "),
        ("∬", r"\iint "),
        ("∭", r"\iiint "),
        ("∮", r"\oint "),
        ("∫", r"\int "),
        ("∇", r"\nabla "),
        ("·", r"\cdot "),
        ("×", r"\times "),
        ("π", r"\pi "),
        ("θ", r"\theta "),
        ("φ", r"\varphi "),
        ("ρ", r"\rho "),
        ("∂", r"\partial "),
    ):
        s = s.replace(symbol, command)
    return s


_MARK = re.compile(r"«(.*?)»", re.S)
_SLOT = re.compile(r"\{(\w+)\}")


def _fragment(template: str, slots: dict) -> tuple[str, str]:
    plain, tex, pos = [], [], 0
    for m in _SLOT.finditer(template):
        literal = template[pos : m.start()]
        plain.append(literal)
        tex.append(tex_math(literal))
        slot = slots[m.group(1)]
        plain.append(str(slot))
        tex.append(slot.latex if isinstance(slot, M) else tex_math(str(slot)))
        pos = m.end()
    literal = template[pos:]
    plain.append(literal)
    tex.append(tex_math(literal))
    return "".join(plain), "".join(tex)


def say(template: str, **slots) -> tuple[str, str]:
    """(text, latex) for a sentence in which each maths fragment sits between « and ». ``{name}`` inside a fragment is
    filled from `slots` (an `M` keeps its own LaTeX)."""
    text, latex = [], []
    parts = _MARK.split(template)
    for i, part in enumerate(parts):
        if i % 2:
            p, l = _fragment(part, slots)
            text.append(p)
            latex.append(l + "{}")
        elif part:
            part = part.format(**{k: str(v) for k, v in slots.items()}) if _SLOT.search(part) else part
            text.append(part)
            latex.append(r"\text{" + _escape_prose(part) + "}")
    return "".join(text), "".join(latex)


def prompt(template: str, **slots) -> dict:
    text, latex = say(template, **slots)
    return dict(prompt=text, prompt_latex=latex)


def stem(template: str, **slots) -> dict:
    text, latex = say(template, **slots)
    return dict(text=text, latex=latex)


# ---------------------------------------------------------------------------------------------------------------
# answers and distractors


def _gens_of(expr):
    return tuple(g for g in (x, y, z) if g in expr.free_symbols)


def val_text(expr) -> str:
    expr = sp.sympify(expr)
    if expr.free_symbols:
        return poly_text(expr, _gens_of(expr))
    return paper_notation(value_safe_str(sp.simplify(expr) if expr.has(sp.sqrt) else expr))


def machine(expr) -> str:
    expr = sp.sympify(expr)
    if expr.free_symbols:
        return value_safe_str(sp.expand(expr))
    return value_safe_str(sp.simplify(expr) if expr.has(sp.sqrt) else expr)


def same(a, b) -> bool:
    a, b = sp.sympify(a), sp.sympify(b)
    if not (a.free_symbols or b.free_symbols):
        return abs(float(sp.N(a - b, 30))) < 1e-9
    return sp.expand(a - b) == 0


def accepted_by_grader(answer, text: str, reply: str) -> bool:
    target = SimpleNamespace(answer=machine(answer), answer_text=text, answer_alt=None, answer_form=None)
    return grade("expression", target, reply).correct


def distractors_for(answer, candidates, n: int = 4) -> list[Distractor]:
    """Wrong values as `Distractor`s: candidates are (misconception id, value) in priority order. A value equal to the
    answer, a repeat, an infinity, and anything the real grader would accept are dropped, each checked."""
    text = val_text(answer)
    out, seen = [], [answer]
    for tag, value in candidates:
        try:
            if isinstance(value, types.FunctionType):
                value = value()  # a candidate worked out only if it is reached
            value = sp.sympify(value)
            if value.has(sp.zoo, sp.nan, sp.oo) or value.has(sp.I) or any(same(value, s) for s in seen):
                continue
        except (TypeError, ValueError):
            continue
        shown = val_text(value)
        if len(shown) > (90 if value.free_symbols else 28):
            continue
        seen.append(value)
        out.append(Distractor(shown, tag, machine(value)))
        if len(out) == n:
            break
    return out


def answer_fields(answer) -> dict:
    answer = sp.sympify(answer)
    return dict(
        answer=machine(answer),
        answer_text=val_text(answer),
        answer_latex=sp.latex(sp.expand(answer), order="lex"),
        answer_kind="expression",
    )


def single_question(rng, draw, tier: str, *, tries: int = 40):
    """Build a one-answer question: `draw(rng, tier)` returns a dict with text, latex, answer (a SymPy value),
    candidates ((misconception id, value) pairs) and working, or None to redraw. Needs three distractors."""
    for _ in attempts(tries, "a question with three usable distractors"):
        d = draw(rng, tier)
        if d is None:
            continue
        dist = distractors_for(d["answer"], d["candidates"], 4)
        if len(dist) >= 3:
            return dict(
                text=d["text"], latex=d["latex"], working=d["working"], distractors=dist, **answer_fields(d["answer"])
            )
    raise AssertionError("unreachable")


def part_dict(prompt_fields: dict, answer, candidates, marks: int, working, follow_through=None, **extra):
    """One part of an exam chain, with distractors from that part's own errors."""
    dist = distractors_for(answer, candidates, 4)
    if len(dist) < 2:
        raise Redraw
    part = dict(**prompt_fields, **answer_fields(answer), marks=marks, working=working, distractors=dist, **extra)
    if follow_through:
        part["follow_through"] = follow_through
    return part


class Redraw(Exception):
    """A draw that cannot give a part two distinct wrong answers."""


def exam_question(rng, build, *, tries: int = 40):
    for _ in attempts(tries, "an exam chain with enough distractors"):
        try:
            return build(rng)
        except Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# random polynomials


def monomials(gens, max_degree: int, min_degree: int = 1):
    return list(_monomials(tuple(gens), max_degree, min_degree))


@lru_cache(maxsize=None)
def _monomials(gens, max_degree: int, min_degree: int):
    out = []
    for exps in np.ndindex(*(max_degree + 1,) * len(gens)):
        if min_degree <= sum(exps) <= max_degree:
            term = sp.Integer(1)
            for g, e in zip(gens, exps):
                term *= g**e
            out.append(term)
    return tuple(out)


def rand_poly(rng, gens, max_degree: int, terms: int, coeffs=(-3, 3), min_degree: int = 1, need=()):
    """A polynomial with `terms` distinct monomials of the allowed degrees, plus any in `need`."""
    pool = monomials(gens, max_degree, min_degree)
    picked = list(need) + [m for m in rng.sample(pool, min(terms, len(pool))) if m not in need]
    lo, hi = coeffs
    return sum(
        (rng.choice([c for c in range(lo, hi + 1) if c]) * m for m in picked[: max(terms, len(need))]), sp.Integer(0)
    )


# ---------------------------------------------------------------------------------------------------------------
# surfaces and solids of the standard shapes (angles: v is theta, u or w the other parameter)


def box_face(axis: int, level, extents) -> Surf:
    """The face of the box [0, a] x [0, b] x [0, c] on the plane coordinate `axis` = `level`."""
    return _box_face(axis, level, tuple(extents))


@lru_cache(maxsize=None)
def _box_face(axis, level, extents) -> Surf:
    others = [i for i in range(3) if i != axis]
    comps = [None, None, None]
    comps[axis] = sp.Integer(level)
    comps[others[0]], comps[others[1]] = u, v
    return Surf(sp.Matrix(comps), Dom((u, v), ((0, extents[others[0]]), (0, extents[others[1]])), {}))


@lru_cache(maxsize=None)
def cylinder_side(a, h) -> Surf:
    return Surf(sp.Matrix([a * C, a * S, w]), Dom((v, w), ((0, 4), (0, h)), {v: (C, S)}))


@lru_cache(maxsize=None)
def disc_cap(a, height) -> Surf:
    return Surf(sp.Matrix([u * C, u * S, height]), Dom((u, v), ((0, a), (0, 4)), {v: (C, S)}))


@lru_cache(maxsize=None)
def sphere_piece(a, theta_q=(0, 4), phi_q=(0, 2)) -> Surf:
    """A piece of the sphere of radius a: theta (about the z-axis) and phi (from the z-axis) in quarter-turns."""
    return Surf(
        sp.Matrix([a * SP * C, a * SP * S, a * CP]),
        Dom((u, v), (tuple(phi_q), tuple(theta_q)), {u: (CP, SP), v: (C, S)}),
    )


@lru_cache(maxsize=None)
def cone_side(h) -> Surf:
    """The cone z = sqrt(x^2 + y^2) up to the plane z = h."""
    return Surf(sp.Matrix([u * C, u * S, u]), Dom((u, v), ((0, h), (0, 4)), {v: (C, S)}))


@lru_cache(maxsize=None)
def bowl_side(h) -> Surf:
    """The paraboloid z = x^2 + y^2 up to the plane z = h (radius sqrt(h) is not a whole number, so the parameter is
    the height of the circle: r = s, z = s^2 with s up to the radius)."""
    return Surf(sp.Matrix([u * C, u * S, u**2]), Dom((u, v), ((0, h), (0, 4)), {v: (C, S)}))


@lru_cache(maxsize=None)
def box_solid(a, b, c) -> Solid:
    return Solid(sp.Matrix([u, v, w]), Dom((u, v, w), ((0, a), (0, b), (0, c)), {}))


@lru_cache(maxsize=None)
def cylinder_solid(a, h) -> Solid:
    return Solid(sp.Matrix([u * C, u * S, w]), Dom((u, v, w), ((0, a), (0, 4), (0, h)), {v: (C, S)}))


@lru_cache(maxsize=None)
def ball_solid(a, theta_q=(0, 4), phi_q=(0, 2)) -> Solid:
    return Solid(
        sp.Matrix([u * SP * C, u * SP * S, u * CP]),
        Dom((u, w, v), ((0, a), tuple(phi_q), tuple(theta_q)), {w: (CP, SP), v: (C, S)}),
    )


@lru_cache(maxsize=None)
def cone_solid(h) -> Solid:
    """z from the cone z = sqrt(x^2 + y^2) up to z = h, with s in [0, 1] running from the cone to the lid."""
    return Solid(sp.Matrix([u * C, u * S, u + (h - u) * w]), Dom((u, v, w), ((0, h), (0, 4), (0, 1)), {v: (C, S)}))


@lru_cache(maxsize=None)
def bowl_solid(radius, h) -> Solid:
    """The solid under the plane z = h and above the paraboloid z = x^2 + y^2, over the disc of the given radius
    (so h = radius^2)."""
    return Solid(
        sp.Matrix([u * C, u * S, u**2 + (h - u**2) * w]), Dom((u, v, w), ((0, radius), (0, 4), (0, 1)), {v: (C, S)})
    )
