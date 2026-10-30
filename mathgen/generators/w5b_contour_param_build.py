"""Generates `w5b_contour_param` and `exd_x_w5b_contour_param`: contour integrals by parametrisation.

One second-year skill, ``uc2-contour-integrals-parametrisation`` (domain ``complex-analysis``). It is the companion of
``uc2-contour-integrals-residue-theorem`` (``w5a_contour``), which never parametrises: here the path is given, the
integrand is often NOT analytic (``conj(z)``, ``Re(z)``, ``Im(z)``, ``|z|``), and the answer is found by substituting
``z = z(t)`` and integrating, so the antiderivative shortcut and the closed-loop-is-zero shortcut are the traps.

Every answer is an exact value ``A + B*pi`` with ``A`` and ``B`` Gaussian rationals, kept as a pair of `C` while it is
being worked out, and only the finished value is handed to SymPy for printing.

Exact route: the integrand is a polynomial in ``z`` and ``conj(z)`` (or ``|z|`` where that is a constant or linear
along the path). On a segment ``z = a + (b - a) t`` it is a polynomial in ``t``. On a circle ``z = c + r e^(it)`` it is
a Laurent polynomial in ``u = e^(it)``, and every angle is a multiple of ``pi/2`` so ``u`` takes values in
``{1, i, -1, -i}`` and no trigonometry is needed.

Second route (checked in `make()`): composite five-point Gauss-Legendre quadrature of ``f(z(t)) z'(t)`` on the
parametrisation itself, and for the closed-loop questions Green's theorem (``oint conj(z) dz = 2i * area``,
``oint Re(z) dz = i * area``, ``oint Im(z) dz = -area``, ``oint z^n dz = 0``).

Tiers:

- ``easy``: ``z^n``, ``conj(z)``, ``Re(z)``, ``Im(z)`` or ``|z|`` along one straight segment, or ``z^n`` / ``conj(z)`` /
  ``|z|`` along a quarter or half circle. The parametrisation is stated.
- ``medium``: ``conj(z)``, ``conj(z)^2``, ``Re(z)``, ``Im(z)`` or ``|z|`` along an arc of a circle or a two-segment
  path, where the antiderivative shortcut, the straight chord and the other arc all give a different answer.
- ``hard``: ``lambda (z - c)^m conj(z - c)^k`` once round a circle (``2*pi*i*lambda*r^(m+k+1)`` when ``m - k = -1``,
  otherwise 0), and the difference of the integrals of ``a z^n + b conj(z)`` (or ``Re``, ``Im``) along two different
  routes between the same two points.
- ``exam`` (topic ``exd_x_w5b_contour_param``): a four-part chain on ``f(z) = lambda z^m conj(z)^k``: the chord, the arc
  of ``|z| = r`` between the same two points, the ML bound for the arc, and how far the arc integral falls short of it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction as Fr

import sympy

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from ._w5a_support import (
    _SUP,
    C,
    I_,
    X,
    astr,
    atext,
    cx_paren,
    cx_x,
    dist_list,
    msay,
    pmul,
    ppow,
    qsym,
    single,
    zminus_x,
)

from ._spread_kit import _RNG, uses_rng
from ._w5b_pde_cp_balance_kit import fair

TOPIC = "w5b_contour_param"
EXAM_TOPIC = "exd_x_w5b_contour_param"

# -- values: A + B*pi ---------------------------------------------------------------------------------------------------

V = tuple  # (A, B), each a C


def v_add(x: V, y: V) -> V:
    return (x[0] + y[0], x[1] + y[1])


def v_neg(x: V) -> V:
    return (-x[0], -x[1])


def v_sub(x: V, y: V) -> V:
    return v_add(x, v_neg(y))


def v_scale(x: V, s) -> V:
    return (x[0] * s, x[1] * s)


def vsym(x: V):
    return sympy.expand(x[0].sym() + x[1].sym() * sympy.pi)


def vcomplex(x: V) -> complex:
    return complex(x[0]) + complex(x[1]) * math.pi


_ZERO: V = (C(0), C(0))


def i_pow(n: int) -> C:
    return (C(1), C(0, 1), C(-1), C(0, -1))[n % 4]


# -- integrands --------------------------------------------------------------------------------------------------------


@dataclass
class F:
    """An integrand: a polynomial in z and conj(z) (`terms`), or |z| (`abs_`), or lam (z-c)^m conj(z-c)^k (`wk`)."""

    text: X
    terms: dict = field(default_factory=dict)  # (p, q) -> C : coef z^p conj(z)^q
    abs_: bool = False
    wk: tuple | None = None  # (lam, m, k, c)
    kind: str = "poly"  # "pow", "conj", "re", "im", "abs", "mix", "wk"
    analytic: bool = False

    def __call__(self, z: complex) -> complex:
        if self.abs_:
            return abs(z) + 0j
        if self.wk is not None:
            lam, m, k, c = self.wk
            w = z - complex(c)
            return complex(lam) * w**m * w.conjugate() ** k
        total = 0j
        for (p, q), coef in self.terms.items():
            total += complex(coef) * z**p * z.conjugate() ** q
        return total


def _sup(n: int) -> str:
    return _SUP.get(str(n), f"^{n}") if n > 1 else ""


def _coef_text(c: int, lead: bool = True) -> str:
    if c == 1:
        return "" if lead else "+ "
    if c == -1:
        return "-" if lead else "- "
    return f"{c}" if lead else (f"+ {c}" if c > 0 else f"- {-c}")


def f_pow(n: int) -> F:
    if n >= 1:
        t = X("z" + _sup(n), f"z^{{{n}}}" if n > 1 else "z")
    else:
        t = X("1/z" + _sup(-n), rf"\frac{{1}}{{z^{{{-n}}}}}" if n < -1 else r"\frac{1}{z}")
    return F(t, {(n, 0): C(1)}, kind="pow", analytic=True)


def f_conj(k: int) -> F:
    return F(X("conj(z)" + _sup(k), r"\overline{z}" + (f"^{{{k}}}" if k > 1 else "")), {(0, k): C(1)}, kind="conj")


def f_re() -> F:
    return F(X("Re(z)", r"\operatorname{Re}(z)"), {(1, 0): C(Fr(1, 2)), (0, 1): C(Fr(1, 2))}, kind="re")


def f_im() -> F:
    return F(X("Im(z)", r"\operatorname{Im}(z)"), {(1, 0): C(0, Fr(-1, 2)), (0, 1): C(0, Fr(1, 2))}, kind="im")


def f_abs() -> F:
    return F(X("|z|", r"|z|"), abs_=True, kind="abs")


def f_atom(kind: str) -> F:
    return {"conj": lambda: f_conj(1), "re": f_re, "im": f_im}[kind]()


def f_mix(alpha: int, n: int, beta: int, atom: str) -> F:
    """alpha z^n + beta atom(z)."""
    a = f_atom(atom)
    pz = f_pow(n)
    t1 = X(_coef_text(alpha) + pz.text.p, _coef_text(alpha) + pz.text.t)
    sep = "+" if beta > 0 else "-"
    mag = abs(beta)
    mt = "" if mag == 1 else f"{mag} "
    mtt = "" if mag == 1 else f"{mag}"
    t2p = f"{mt}{a.text.p}"
    t2t = f"{mtt}{a.text.t}"
    terms = {(n, 0): C(alpha)}
    for key, coef in a.terms.items():
        terms[key] = terms.get(key, C(0)) + coef * beta
    return F(X(f"{t1.p} {sep} {t2p}", f"{t1.t} {sep} {t2t}"), terms, kind="mix")


def _join_factors(pieces: list) -> str:
    """Factors written side by side, with a space before conj( so a coefficient never runs into it."""
    out = ""
    for piece in pieces:
        out += (" " if out and piece.startswith("conj(") else "") + piece
    return out


def f_wk(lam: int, m: int, k: int, c: C) -> F:
    """lam (z-c)^m conj(z-c)^k, written out with the bracket (z - c)."""
    w = zminus_x(c)
    wb = w.p if c.is_zero() else f"({w.p})"
    wt = w.t if c.is_zero() else f"({w.t})"
    num_p, num_t = [], []
    if abs(lam) != 1:
        num_p.append(str(abs(lam)))
        num_t.append(str(abs(lam)))
    if m > 0:
        num_p.append(wb + _sup(m))
        num_t.append(wt + (f"^{{{m}}}" if m > 1 else ""))
    if k > 0:
        num_p.append(f"conj({w.p})" + _sup(k))
        num_t.append(rf"\overline{{{w.t}}}" + (f"^{{{k}}}" if k > 1 else ""))
    np_ = _join_factors(num_p) or "1"
    nt_ = "".join(num_t) or "1"
    if m < 0:
        dp = wb + _sup(-m)
        dt = wt + (f"^{{{-m}}}" if -m > 1 else "")
        body = X(f"{np_}/{dp}", rf"\frac{{{nt_}}}{{{dt}}}")
    else:
        body = X(np_, nt_)
    if lam < 0:
        body = X("-" + body.p, "-" + body.t)
    return F(body, wk=(lam, m, k, c), kind="wk")


# -- paths ------------------------------------------------------------------------------------------------------------


@dataclass
class Seg:
    a: C
    b: C


@dataclass
class Arc:
    c: C
    r: Fr
    n0: int  # start angle n0 * pi/2
    n1: int  # end angle n1 * pi/2; n1 > n0 is anticlockwise

    def start(self) -> C:
        return self.c + i_pow(self.n0) * self.r

    def end(self) -> C:
        return self.c + i_pow(self.n1) * self.r


@dataclass
class Poly:
    pts: list


def _pt(c: C) -> X:
    return cx_paren(c) if (c.re != 0 and c.im != 0) else cx_x(c)


def _circle(c: C, r: Fr) -> X:
    inner = zminus_x(c)
    return X(f"|{inner.p}| = {r.numerator}", f"|{inner.t}| = {r.numerator}")


def path_x(path) -> X:
    """The path as a noun phrase."""
    if isinstance(path, Seg):
        a, b = _pt(path.a), _pt(path.b)
        return X(
            f"the straight line segment from {a.p} to {b.p}",
            rf"\text{{the straight line segment from }}{a.t}\text{{ to }}{b.t}",
        )
    if isinstance(path, Poly):
        p = [_pt(q) for q in path.pts]
        txt = f"the path made of the straight segment from {p[0].p} to {p[1].p} followed by the straight segment from {p[1].p} to {p[2].p}"
        return X(txt, rf"\text{{{txt}}}")
    circ = _circle(path.c, path.r)
    way = "anticlockwise" if path.n1 > path.n0 else "clockwise"
    if abs(path.n1 - path.n0) == 4:
        txt = f"the circle {circ.p}, traversed once {way}"
        return X(txt, rf"\text{{the circle }}{circ.t}\text{{, traversed once {way}}}")
    s, e = _pt(path.start()), _pt(path.end())
    return X(
        f"the arc of the circle {circ.p} from {s.p} to {e.p}, traversed {way}",
        rf"\text{{the arc of the circle }}{circ.t}\text{{ from }}{s.t}\text{{ to }}{e.t}\text{{, traversed {way}}}",
    )


def _pi_mult(n: int) -> str:
    """n * pi/2 as text: 0, π/2, π, 3π/2, -π/2 ..."""
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n % 2 == 0:
        return sign + ("π" if n == 2 else f"{n // 2}π")
    return sign + ("π/2" if n == 1 else f"{n}π/2")


def _lin(a: C, d: C) -> str:
    """a + d t as text: 1 + (2 - i)t, (3 + 4i)t, 2t."""
    dd = cx_x(d).p
    dt = dd if (d.im == 0 and d.re > 0 and d.re.denominator == 1) else f"({dd})"
    if a.is_zero():
        return f"{dt}t"
    return f"{cx_x(a).p} + {dt}t"


def arc_param(path: "Arc") -> str:
    """z(t) = c + r e^(it) for an anticlockwise arc, c + r e^(-it) for a clockwise one, with t increasing."""
    anti = path.n1 > path.n0
    base = (
        ("e^(it)" if anti else "e^(-it)") if path.r == 1 else f"{path.r.numerator}" + ("e^(it)" if anti else "e^(-it)")
    )
    pre = "" if path.c.is_zero() else f"{cx_x(path.c).p} + "
    lo, hi = (path.n0, path.n1) if anti else (-path.n0, -path.n1)
    return f"z(t) = {pre}{base} for {_pi_mult(lo)} ≤ t ≤ {_pi_mult(hi)}"


def hint_x(path) -> str:
    if isinstance(path, Seg):
        return f" Use z(t) = {_lin(path.a, path.b - path.a)} for 0 ≤ t ≤ 1."
    if isinstance(path, Arc):
        return f" Use {arc_param(path)}."
    return ""


# -- the exact integrator ---------------------------------------------------------------------------------------------


def _norm_exact(z: C) -> int:
    n2 = z.abs2()
    check(n2.denominator == 1, "norm not an integer square")
    root = math.isqrt(n2.numerator)
    check(root * root == n2.numerator, "|z| is not exact")
    return root


def _seg(f: F, a: C, b: C, with_dz: bool = True) -> V:
    d = b - a
    if f.abs_:
        # |z| is linear along a ray from the origin: (|a| + |b|)/2 on average
        check(
            a.is_zero() or b.is_zero() or ((a.conj() * b).im == 0 and (a.conj() * b).re > 0),
            "segment is not on a ray from 0",
        )
        mean = Fr(_norm_exact(a) + _norm_exact(b), 2)
        return ((d * mean if with_dz else C(mean)), C(0))
    tot = C(0)
    ab, db = a.conj(), d.conj()
    for (p, q), coef in f.terms.items():
        check(p >= 0 and q >= 0, "negative power on a segment")
        poly = pmul(ppow([a, d], p), ppow([ab, db], q))
        part = C(0)
        for k, ck in enumerate(poly):
            part = part + ck / (k + 1)
        tot = tot + coef * part
    return ((tot * d if with_dz else tot), C(0))


def _zlau(c: C, r: Fr, p: int) -> dict:
    if p >= 0:
        return {k: C(math.comb(p, k)) * (c ** (p - k)) * (C(r) ** k) for k in range(p + 1)}
    check(c.is_zero(), "negative power about a centre")
    return {p: C(r) ** p}


def _clau(c: C, r: Fr, q: int) -> dict:
    cb = c.conj()
    return {-k: C(math.comb(q, k)) * (cb ** (q - k)) * (C(r) ** k) for k in range(q + 1)}


def _lmul(x: dict, y: dict) -> dict:
    out: dict = {}
    for i, a in x.items():
        for j, b in y.items():
            out[i + j] = out.get(i + j, C(0)) + a * b
    return {k: v for k, v in out.items() if not v.is_zero()}


def laurent(f: F, c: C, r: Fr) -> dict:
    """f(c + r u) as {j: coefficient of u^j}."""
    if f.abs_:
        check(c.is_zero(), "|z| is constant only on circles about 0")
        return {0: C(r)}
    if f.wk is not None:
        lam, m, k, c0 = f.wk
        check(c0 == c, "the integrand is not centred on this circle")
        return {m - k: C(lam) * (C(r) ** (m + k))}
    out: dict = {}
    for (p, q), coef in f.terms.items():
        for j, v in _lmul(_zlau(c, r, p), _clau(c, r, q)).items():
            out[j] = out.get(j, C(0)) + coef * v
    return out


def _arc_lau(lau: dict, r: Fr, n0: int, n1: int, with_dz: bool = True) -> V:
    a, b = C(0), C(0)
    for j, kappa in lau.items():
        if with_dz:
            m = j + 1
            if m == 0:
                b = b + kappa * I_ * r * Fr(n1 - n0, 2)
            else:
                a = a + kappa * r * (i_pow(n1 * m) - i_pow(n0 * m)) / m
        elif j == 0:
            b = b + kappa * Fr(n1 - n0, 2)
        else:
            a = a + kappa * (i_pow(n1 * j) - i_pow(n0 * j)) / (I_ * j)
    return (a, b)


def path_int(f: F, path, with_dz: bool = True) -> V:
    if isinstance(path, Seg):
        return _seg(f, path.a, path.b, with_dz)
    if isinstance(path, Poly):
        tot = _ZERO
        for a, b in zip(path.pts, path.pts[1:]):
            tot = v_add(tot, _seg(f, a, b, with_dz))
        return tot
    return _arc_lau(laurent(f, path.c, path.r), path.r, path.n0, path.n1, with_dz)


# -- the numeric second route ----------------------------------------------------------------------------------------

_GL_X = (0.0, 0.5384693101056831, -0.5384693101056831, 0.9061798459386640, -0.9061798459386640)
_GL_W = (0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891)


def _gauss(fn, t0: float, t1: float, panels: int) -> complex:
    total = 0j
    h = (t1 - t0) / panels
    for k in range(panels):
        mid = t0 + (k + 0.5) * h
        for x, w in zip(_GL_X, _GL_W):
            total += w * fn(mid + 0.5 * h * x)
    return total * 0.5 * h


def num_int(f: F, path) -> complex:
    if isinstance(path, Poly):
        return sum((num_int(f, Seg(a, b)) for a, b in zip(path.pts, path.pts[1:])), 0j)
    if isinstance(path, Seg):
        a, b = complex(path.a), complex(path.b)
        return _gauss(lambda t: f(a + (b - a) * t) * (b - a), 0.0, 1.0, 6)
    c, r = complex(path.c), float(path.r)
    import cmath

    return _gauss(
        lambda t: f(c + r * cmath.exp(1j * t)) * 1j * r * cmath.exp(1j * t),
        path.n0 * math.pi / 2,
        path.n1 * math.pi / 2,
        24,
    )


def verified(f: F, path) -> V:
    """The exact integral, checked against quadrature on the parametrisation."""
    v = path_int(f, path)
    got, want = num_int(f, path), vcomplex(v)
    check(abs(got - want) < 1e-7 * max(1.0, abs(want)), f"quadrature {got} disagrees with exact {want}")
    return v


def endpoints(path) -> tuple:
    if isinstance(path, Seg):
        return path.a, path.b
    if isinstance(path, Poly):
        return path.pts[0], path.pts[-1]
    return path.start(), path.end()


# -- distractors -----------------------------------------------------------------------------------------------------


def _shortcut(f: F, path, held: bool) -> V | None:
    """The antiderivative shortcut. held=False: every non-analytic atom replaced by z (z^(p+q+1)/(p+q+1) between the
    ends). held=True: conj(z)^q treated as a constant (conj(z)^q z^(p+1)/(p+1))."""
    a, b = endpoints(path)
    if f.wk is not None:
        return None
    tot = C(0)
    if f.abs_:
        if held:
            return (C(_norm_exact(b)) * (b - a), C(0)) if isinstance(path, Seg) else None
        return ((b**2 - a**2) / 2, C(0))
    for (p, q), coef in f.terms.items():
        if held:
            if q == 0:
                continue
            tot = tot + coef * (b.conj() ** q * b ** (p + 1) - a.conj() ** q * a ** (p + 1)) / (p + 1)
        else:
            e = p + q + 1
            tot = tot + coef * (b**e - a**e) / e
    if held and not any(q > 0 for (_p, q) in f.terms):
        return None
    return (tot, C(0))


def wrongs_for(f: F, path, true: V) -> list:
    """(value, misconception) candidates for a single path integral."""
    out: list = []
    val = vsym(true)

    def add(v, tag):
        if v is not None:
            out.append((vsym(v) if isinstance(v, tuple) else v, tag))

    if not f.analytic:
        add(_shortcut(f, path, False), "cx2-param-nonanalytic-as-z")
        add(
            _shortcut(f, path, True),
            "cx2-param-conjugate-held-constant" if not f.abs_ else "cx2-param-modulus-held-constant",
        )
    add(path_int(f, path, with_dz=False), "cx2-param-dz-dropped")
    if f.kind == "pow" and f.terms and next(iter(f.terms))[0] >= 1:
        n = next(iter(f.terms))[0]
        a, b = endpoints(path)
        add(((b ** (n + 1) - a ** (n + 1)) / n, C(0)), "cx2-param-power-rule-slip")
        add((b**n - a**n, C(0)), "cx2-param-power-rule-slip")
        add((b ** (n + 1) - a ** (n + 1), C(0)), "cx2-param-power-rule-slip")
    if isinstance(path, Arc):
        add(v_scale(true, Fr(1) / path.r), "cx2-param-radius-dropped-from-dz")
        other = Arc(path.c, path.r, path.n0, path.n1 - 4 if path.n1 > path.n0 else path.n1 + 4)
        if abs(path.n1 - path.n0) != 4:
            add(path_int(f, other), "cx2-param-wrong-arc")
            if not f.abs_ and f.wk is None:
                add(_seg(f, path.start(), path.end()), "cx2-param-path-independence-assumed")
    if isinstance(path, Poly) and not f.abs_:
        add(_seg(f, path.pts[0], path.pts[-1]), "cx2-param-path-independence-assumed")
    if f.kind in ("re", "im"):
        add(path_int(f_im() if f.kind == "re" else f_re(), path), "cx2-param-re-im-swapped")
    out.append((-val, "cx2-orientation-ignored"))
    return out


# -- phrasing ---------------------------------------------------------------------------------------------------------


def _ask(rng, f: F, path, hint: str = "", way: str = "Find the exact value") -> tuple:
    px = path_x(path)
    integ = X(f"∫_C {f.text.p} dz", rf"\int_C {f.text.t}\,\mathrm{{d}}z")
    shapes = [
        lambda: msay(
            "Let ",
            X(f"f(z) = {f.text.p}", f"f(z) = {f.text.t}"),
            ". Evaluate ",
            X("∫_C f(z) dz", r"\int_C f(z)\,\mathrm{d}z"),
            " exactly, where C is ",
            px,
            "." + hint,
        ),
        lambda: msay(way + " of ", integ, ", where C is ", px, "." + hint),
        lambda: msay("The contour C is ", px, ". Calculate the exact value of ", integ, "." + hint),
    ]
    return shapes[rng.randrange(3)]()


# -- easy -----------------------------------------------------------------------------------------------------------

_GI = [C(a, b) for a in range(-3, 4) for b in range(-3, 4)]
_RAYS = [C(3, 4), C(4, 3), C(5), C(0, 5), C(-3, 4), C(4, -3), C(-5), C(0, -5), C(6, 8), C(12, 5), C(5, 12), C(-4, -3)]


def _abs_segment(rng) -> Seg:
    w = rng.choice(_RAYS)
    s0, s1 = rng.sample([0, 1, 2, 3], 2)
    return Seg(w * min(s0, s1), w * max(s0, s1)) if rng.random() < 0.5 else Seg(w * max(s0, s1), w * min(s0, s1))


def _finish(rng, f: F, path, hint: str = "", way: str = "Find the exact value"):
    """The finished question, or None when the integral is zero (a guessable key): the caller draws again."""
    true = verified(f, path)
    value = vsym(true)
    if value == 0:
        return None
    working = [
        _working_param(f, path),
        f"Substituting and integrating gives {atext(value)}.",
    ]
    return single(
        _ask(rng, f, path, hint, way),
        value,
        working,
        wrongs_for(f, path, true),
        dist=_spread(value, wrongs_for(f, path, true), rng),
    )


def _retry(rng, draw):
    for _ in attempts(200, "a non-zero integral"):
        out = draw(rng)
        if out is not None:
            return out


def _working_param(f: F, path) -> str:
    if isinstance(path, Seg):
        return f"Parametrise z(t) = {cx_x(path.a).p} + ({cx_x(path.b - path.a).p})t, 0 ≤ t ≤ 1, so dz = ({cx_x(path.b - path.a).p}) dt."
    if isinstance(path, Poly):
        return "Parametrise each straight leg as z(t) = start + (end - start)t, 0 ≤ t ≤ 1, integrate each and add."
    anti = path.n1 > path.n0
    rr = "" if path.r == 1 else str(path.r.numerator)
    return (
        f"Parametrise {arc_param(path)}, so dz = {'' if anti else '-'}{rr}i e^({'' if anti else '-'}it) dt "
        f"and conj(z) = {cx_x(path.c.conj()).p + ' + ' if not path.c.is_zero() else ''}{rr}e^({'-' if anti else ''}it)."
    )


def _easy_segment(rng):
    kind = rng.choice(["pow", "pow", "conj", "re", "im", "abs"])
    if kind == "abs":
        path, f = _abs_segment(rng), f_abs()
    else:
        for _ in attempts(200, "a segment"):
            a, b = rng.sample(_GI, 2)
            if a != b:
                break
        path = Seg(a, b)
        f = {"pow": lambda: f_pow(rng.choice([1, 2, 2, 3])), "conj": lambda: f_conj(1), "re": f_re, "im": f_im}[kind]()
    return _finish(rng, f, path, hint_x(path))


def _easy_arc(rng):
    kind = rng.choice(["pow", "conj", "abs"])
    c = (
        C(0)
        if kind == "abs" or rng.random() < 0.4
        else rng.choice([C(1), C(-1), C(0, 1), C(1, 1), C(2), C(0, -1), C(-1, 1)])
    )
    r = Fr(rng.choice([1, 2, 3]))
    span = rng.choice([1, 2, 2, 3])
    n0 = rng.randrange(4)
    sign = 1 if rng.random() < 0.7 else -1
    path = Arc(c, r, n0, n0 + sign * span)
    f = {"pow": lambda: f_pow(rng.choice([1, 2, 3])), "conj": lambda: f_conj(1), "abs": f_abs}[kind]()
    return _finish(rng, f, path, hint_x(path))


def _easy(rng):
    return _retry(rng, _easy_segment if rng.random() < 0.6 else _easy_arc)


# -- medium ---------------------------------------------------------------------------------------------------------


def _medium_arc(rng):
    kind = rng.choice(["conj", "conj", "re", "im", "abs", "conj2"])
    c = (
        C(0)
        if kind == "abs" or rng.random() < 0.35
        else rng.choice([C(1), C(-1), C(0, 1), C(1, 1), C(2), C(0, -1), C(-1, 1), C(1, -1)])
    )
    r = Fr(rng.choice([1, 2, 3]))
    span = rng.choice([1, 2, 2, 3])
    n0 = rng.randrange(4)
    sign = 1 if rng.random() < 0.65 else -1
    path = Arc(c, r, n0, n0 + sign * span)
    f = {"conj": lambda: f_conj(1), "conj2": lambda: f_conj(2), "re": f_re, "im": f_im, "abs": f_abs}[kind]()
    return _finish(rng, f, path)


def _medium_poly(rng):
    kind = rng.choice(["conj", "conj", "re", "im", "abs"])
    if kind == "abs":
        # a V through the origin: in along one ray to 0, out along another
        for _ in attempts(100, "two different rays"):
            w1, w2 = rng.sample(_RAYS, 2)
            w1, w2 = w1 * rng.choice([1, 1, 2]), w2 * rng.choice([1, 1, 2])
            if (w1.conj() * w2).im != 0 or (w1.conj() * w2).re < 0:
                break
        path, f = Poly([w1, C(0), w2]), f_abs()
    else:
        for _ in attempts(300, "a two-leg path"):
            a, d, b = rng.sample(_GI, 3)
            if (d - a).re * (b - d).im - (d - a).im * (b - d).re != 0:
                break
        path = Poly([a, d, b])
        f = {"conj": lambda: f_conj(1), "re": f_re, "im": f_im}[kind]()
    return _finish(rng, f, path)


def _medium(rng):
    return _retry(rng, _medium_arc if rng.random() < 0.6 else _medium_poly)


# -- hard ------------------------------------------------------------------------------------------------------------

_RESONANT = [(-1, 0), (0, 1), (1, 2), (2, 3), (0, 1), (-1, 0)]
_PLAIN = [(1, 0), (2, 0), (-2, 0), (0, 2), (1, 1), (0, 3), (1, 3), (-1, 1), (0, 0)]


def _hard_closed(rng):
    for _ in attempts(400, "a closed-circle integrand"):
        resonant = rng.random() < 0.75
        m, k = rng.choice(_RESONANT if resonant else _PLAIN)
        if (m, k) == (0, 0):
            continue
        c = rng.choice([C(0), C(1), C(-1), C(0, 1), C(1, 1), C(2), C(-1, -1), C(0, -2)])
        r = Fr(rng.choice([1, 2, 3]))
        lam = rng.choice([1, 1, 2, 3, -1, -2])
        if resonant and abs(lam) == 1 and rng.random() < 0.5:
            lam = rng.choice([2, 3, -2])
        n0 = rng.randrange(4)
        anticlockwise = rng.random() < 0.6
        path = Arc(c, r, n0, n0 + (4 if anticlockwise else -4))
        f = f_wk(lam, m, k, c)
        break
    true = verified(f, path)
    s = 1 if anticlockwise else -1
    # the closed form, a third route
    closed = vsym((C(0), C(0)))
    if m - k == -1:
        closed = sympy.expand(s * 2 * sympy.pi * sympy.I * lam * qsym(r) ** (m + k + 1))
    check(sympy.expand(vsym(true) - closed) == 0, "closed form disagrees with the parametrised integral")
    value = closed
    two_pi_i = 2 * sympy.pi * sympy.I
    base = qsym(r) ** (m + k + 1)
    wr = [
        (sympy.expand(s * two_pi_i * lam * base), "cx2-param-every-power-gives-2-pi-i"),
        (sympy.expand(s * two_pi_i * lam * qsym(r) ** (m + k)), "cx2-param-radius-power-slip"),
        (sympy.expand(s * two_pi_i * lam * qsym(r) ** (m + k + 2)), "cx2-param-radius-power-slip"),
        (sympy.expand(-s * two_pi_i * lam * base), "cx2-orientation-ignored"),
        (sympy.expand(s * 2 * sympy.pi * lam * base), "forgets-i-in-residue-theorem"),
        (sympy.expand(s * two_pi_i * base), "cx2-param-every-power-gives-2-pi-i"),
        (sympy.Integer(0), "cx2-param-closed-loop-assumed-zero"),
    ]
    if m - k != -1:
        wr.insert(1, (sympy.expand(s * two_pi_i * lam), "cx2-param-every-power-gives-2-pi-i"))
    way = "anticlockwise" if anticlockwise else "clockwise"
    if m - k == -1:
        working = [
            f"On the circle z - c = {r.numerator}e^(it), so (z - c)^m conj(z - c)^k dz has the factor e^(i(m - k + 1)t) = 1 here.",
            f"The integrand becomes {lam} r^{m + k} e^(-it) times {r.numerator}i e^(it) dt, so the integral is {atext(value)}.",
        ]
    else:
        working = [
            f"On the circle z - c = {r.numerator}e^(it), so the integrand times dz is a constant times e^(i(m - k + 1)t) with m - k + 1 = {m - k + 1}, not 0.",
            "The integral of e^(int) over a full turn is 0 for n not 0, so the integral is 0 whichever way the circle is traversed.",
        ]
    return single(_ask(rng, f, path, "", "Evaluate the exact value"), value, working, wr, dist=_spread(value, wr, rng))


def _signed_area_loop(path2, a: C, b: C) -> tuple:
    """Signed area (A, Bpi as Fractions) of the loop: chord a->b, then the route 2 back from b to a."""
    if isinstance(path2, Poly):
        d = path2.pts[1]
        # polygon a, b, d
        s = (a.re * b.im - b.re * a.im) + (b.re * d.im - d.re * b.im) + (d.re * a.im - a.re * d.im)
        return (s / 2, Fr(0))
    span = abs(path2.n1 - path2.n0)
    sin_t = {1: 1, 2: 0, 3: -1}[span]
    seg_const = -path2.r * path2.r * sin_t / 2
    seg_pi = path2.r * path2.r * span / 4
    sign = -1 if path2.n1 > path2.n0 else 1  # chord forward + anticlockwise arc back is clockwise
    return (sign * seg_const, sign * seg_pi)


def _hard_routes(rng):
    atom = rng.choice(["conj", "conj", "re", "im"])
    alpha = rng.choice([1, 2, 3, -1])
    n = rng.choice([1, 2, 3])
    beta = rng.choice([1, 2, 3, -1, -2])
    f = f_mix(alpha, n, beta, atom)
    use_arc = rng.random() < 0.45
    if use_arc:
        c = rng.choice([C(0), C(1), C(0, 1), C(-1, 1), C(1, -1), C(2)])
        r = Fr(rng.choice([1, 2]))
        span = rng.choice([1, 2, 3])
        n0 = rng.randrange(4)
        sign = 1 if rng.random() < 0.6 else -1
        arc = Arc(c, r, n0, n0 + sign * span)
        a, b = arc.start(), arc.end()
        r1, r2 = Seg(a, b), arc
    else:
        for _ in attempts(300, "a triangle"):
            a, d, b = rng.sample(_GI, 3)
            cross = (d - a).re * (b - a).im - (d - a).im * (b - a).re
            if cross != 0:
                break
        r1, r2 = Seg(a, b), Poly([a, d, b])
    i1, i2 = verified(f, r1), verified(f, r2)
    diff = v_sub(i1, i2)
    # Green's theorem: the difference is the integral round the loop (route 1 out, route 2 back)
    area = _signed_area_loop(r2, a, b)
    kappa = {"conj": C(0, 2), "re": C(0, 1), "im": C(-1)}[atom]
    green = (kappa * beta * area[0], kappa * beta * area[1])
    check(sympy.expand(vsym(diff) - vsym(green)) == 0, "Green's theorem disagrees with the two routes")
    value = vsym(diff)
    check(value != 0, "a zero route difference is guessable")
    area_sym = sympy.Rational(area[0].numerator, area[0].denominator) + qsym(area[1]) * sympy.pi
    wr = [
        (sympy.Integer(0), "cx2-param-path-independence-assumed"),
        (-value, "cx2-orientation-ignored"),
        (sympy.expand(value / 2), "cx2-param-area-factor-slip"),
        (sympy.expand(value * 2), "cx2-param-area-factor-slip"),
        (sympy.expand(beta * area_sym), "cx2-param-area-factor-slip"),
        (vsym(i1), "cx2-param-path-independence-assumed"),
        (vsym(i2), "cx2-param-path-independence-assumed"),
    ]
    if atom != "conj":
        wr.append((sympy.expand(beta * area_sym * 2 * sympy.I), "cx2-param-area-factor-slip"))
    px1, px2 = path_x(r1), path_x(r2)
    stem_shapes = [
        lambda: msay(
            X(f"Let f(z) = {f.text.p}.", f"f(z) = {f.text.t}"),
            " Let I1 be the integral of f(z) along ",
            px1,
            ", and I2 the integral of f(z) along ",
            px2,
            ". Find the exact value of I1 - I2.",
        ),
        lambda: msay(
            "Two routes join ",
            _pt(a),
            " to ",
            _pt(b),
            ": C1 is ",
            px1,
            " and C2 is ",
            px2,
            ". With ",
            X(f"f(z) = {f.text.p}", f"f(z) = {f.text.t}"),
            ", find the exact value of the integral of f along C1 minus the integral of f along C2.",
        ),
        lambda: msay(
            "Calculate the exact difference between the integral of ",
            X(f.text.p, f.text.t),
            " along ",
            px1,
            " and its integral along ",
            px2,
            ".",
        ),
    ]
    stem = stem_shapes[rng.randrange(3)]()
    working = [
        f"I1 = {atext(vsym(i1))} and I2 = {atext(vsym(i2))} by parametrising each route.",
        f"So I1 - I2 = {atext(value)}. Equivalently it is the integral round the closed loop (C1 out, C2 back): "
        "only the non-analytic part contributes, and it equals a constant times the signed area enclosed.",
    ]
    return single(stem, value, working, wr, dist=_spread(value, wr, rng))


def _hard(rng):
    return _hard_closed(rng) if rng.random() < 0.5 else _hard_routes(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def w5b_contour_param(rng: random.Random, tier: str = "easy") -> dict:
    """A contour integral by parametrisation z = z(t): an exact expression in i and pi."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# -- exam ------------------------------------------------------------------------------------------------------------


def _spread(answer, cands: list, rng) -> list:
    """Four wrong expressions chosen from a dozen so that no learner trick finds the key: the slips first, then the key
    reflected or rotated (a sign or a quarter turn wrong) as fillers that are written like the key."""
    a = sympy.sympify(answer)
    re_a, im_a = sympy.expand(sympy.re(a)), sympy.expand(sympy.im(a))
    fillers = [
        (-re_a + sympy.I * im_a, "cx2-arithmetic-slip"),
        (re_a - sympy.I * im_a, "cx2-arithmetic-slip"),
        (sympy.expand(sympy.I * a), "cx2-arithmetic-slip"),
        (sympy.expand(-sympy.I * a), "cx2-arithmetic-slip"),
        (2 * a, "cx2-arithmetic-slip"),
        (a / 2, "cx2-arithmetic-slip"),
    ]
    pool = dist_list(a, [*cands, *fillers], 10)
    return fair(atext(a), pool, 4, rng)


def _dec_dist(value: Fr, cands: list, dp: int = 2) -> list:
    """Decimal distractors (stated precision) that the typed number grader cannot confuse with `value`."""
    tol = stated_precision_tolerance(dp)
    out = []
    for v, tag in cands:
        v = Fr(v).limit_denominator(10**9)
        out.append(Distractor(stated_precision_string(round_half_up(v, dp), dp), tag, float(round_half_up(v, dp))))
    key = stated_precision_string(round_half_up(value, dp), dp)
    for off in (1, -1, 2, -2, 3, -3, 5, -5):
        v = round_half_up(value + off, dp)
        if v > 0:
            out.append(Distractor(stated_precision_string(v, dp), "cx2-arithmetic-slip", float(v)))
    chosen = pick_distractors(key, out, 10, tolerance=tol + Fr(1, 1000))
    rng = _RNG.get()
    return chosen[:4] if rng is None else fair(key, chosen, 4, rng)


def _exam_f(lam: int, m: int, k: int) -> F:
    pieces = [("-" if lam < 0 else "") + ("" if abs(lam) == 1 else str(abs(lam)))]
    tex = list(pieces)
    if m:
        pieces.append("z" + _sup(m))
        tex.append("z" + (f"^{{{m}}}" if m > 1 else ""))
    text = ""
    for piece in pieces:
        text += (" " if text and piece.startswith("conj(") else "") + piece
    ctext = text
    if k:
        ctext = (text + " " if text.strip("-") else text) + "conj(z)" + _sup(k)
        tex.append(r"\overline{z}" + (f"^{{{k}}}" if k > 1 else ""))
    return F(X(ctext, "".join(tex)), {(m, k): C(lam)}, kind="mix")


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_w5b_contour_param(rng: random.Random, tier: str = "exam") -> dict:
    """The chord, the arc of |z| = r, the ML bound for the arc, and how far the arc integral is under the bound."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(800, "an exam question"):
        r = Fr(rng.choice([1, 2, 3, 4, 5, 6]))
        m, k = rng.choice([(m, k) for k in range(1, 6) for m in range(0, 6) if m + k <= 6])
        lam = rng.choice([v for v in range(-11, 12) if v])
        span = rng.choice([1, 2, 3])
        n0 = rng.randrange(4)
        sign = 1 if rng.random() < 0.65 else -1
        arc = Arc(C(0), r, n0, n0 + sign * span)
        a, b = arc.start(), arc.end()
        f = _exam_f(lam, m, k)
        chord = Seg(a, b)
        i_chord, i_arc = verified(f, chord), verified(f, arc)
        mod_f = abs(lam) * r ** (m + k)
        bound = float(mod_f * r * Fr(span, 2)) * math.pi
        ival = abs(vcomplex(i_arc))
        slack = bound - ival
        ties = [abs((v * 100) % 1 - 0.5) > 2e-3 for v in (bound, slack)]
        if all(ties) and slack > 0.05 and vsym(i_chord) != 0 and vsym(i_arc) != 0 and vsym(i_chord) != vsym(i_arc):
            break
    length = r * Fr(span, 2)  # times pi
    pi_fr = Fr(314159265358979, 10**14)
    bound_fr = Fr(bound).limit_denominator(10**9)
    slack_fr = Fr(slack).limit_denominator(10**9)
    ival_fr = Fr(ival).limit_denominator(10**9)
    bound_t = stated_precision_string(round_half_up(bound_fr, 2), 2)
    slack_t = stated_precision_string(round_half_up(slack_fr, 2), 2)
    px_arc, px_chord = path_x(arc), path_x(chord)
    ftx = X(f.text.p, f.text.t)
    stem_variants = [
        lambda: msay(
            "Let ", X(f"f(z) = {ftx.p}", f"f(z) = {ftx.t}"), ". Let C1 be ", px_chord, " and let C2 be ", px_arc, "."
        ),
        lambda: msay(
            "Two paths join ",
            _pt(a),
            " to ",
            _pt(b),
            ": C1 is ",
            px_chord,
            " and C2 is ",
            px_arc,
            ". Take ",
            X(f"f(z) = {ftx.p}", f"f(z) = {ftx.t}"),
            ".",
        ),
        lambda: msay(
            "Take ", X(f"f(z) = {ftx.p}", f"f(z) = {ftx.t}"), ", with C1 = ", px_chord, " and C2 = ", px_arc, "."
        ),
    ]
    stem = stem_variants[rng.randrange(3)]()
    d = b - a
    chord_val, arc_val = vsym(i_chord), vsym(i_arc)
    da = _spread(chord_val, wrongs_for(f, chord, i_chord), rng)
    db = _spread(arc_val, wrongs_for(f, arc, i_arc), rng)
    check(len(da) >= 2 and len(db) >= 2, "too few expression distractors")
    chord_len = Fr(math.sqrt(float(d.abs2()))).limit_denominator(10**9)
    cands_c = [
        (mod_f * chord_len, "cx2-param-ml-length-chord"),
        (mod_f * r * 2 * pi_fr, "cx2-param-ml-length-circle"),
        (length * pi_fr, "cx2-param-ml-max-omitted"),
        (mod_f * length, "cx2-param-ml-max-omitted"),
        (mod_f * length * 2 * pi_fr, "cx2-param-ml-length-circle"),
        (mod_f * (r + 1) * length * pi_fr, None),
    ]
    dc = _dec_dist(bound_fr, cands_c)
    cands_d = [
        (-slack_fr, "cx2-param-ml-slack-sign"),
        (ival_fr, "cx2-param-ml-slack-sign"),
        (bound_fr, "cx2-param-ml-slack-sign"),
        (bound_fr + ival_fr, "cx2-param-ml-slack-sign"),
        (slack_fr + 1, None),
        (slack_fr * 2, None),
    ]
    dd = _dec_dist(slack_fr, cands_d)
    check(len(dc) >= 2 and len(dd) >= 2, "too few decimal distractors")
    arcname = "anticlockwise" if arc.n1 > arc.n0 else "clockwise"
    rt = _rtxt(r)
    mod_sym = sympy.Rational(mod_f.numerator, mod_f.denominator)
    parts = [
        dict(
            prompt=f"Parametrise C1 by z(t) = {_lin(a, d)} for 0 ≤ t ≤ 1 and find the exact value of the integral of f along C1.",
            answer_kind="expression",
            answer=astr(chord_val),
            answer_text=atext(chord_val),
            answer_latex=_alatex(chord_val),
            marks=3,
            working=[f"dz = ({cx_x(d).p}) dt; integrating the polynomial in t gives {atext(chord_val)}."],
            distractors=da,
        ),
        dict(
            prompt=f"Parametrise C2 by {arc_param(arc)} and find the exact value of the integral of f along C2.",
            answer_kind="expression",
            answer=astr(arc_val),
            answer_text=atext(arc_val),
            answer_latex=_alatex(arc_val),
            marks=3,
            working=[
                f"On C2 ({arcname}), z = {rt}e^(i\u03b8) with \u03b8 = {'t' if arc.n1 > arc.n0 else '-t'}, so conj(z) = {rt}e^(-i\u03b8) and dz = {rt}i e^(i\u03b8) d\u03b8.",
                f"The integral is {atext(arc_val)}.",
            ],
            distractors=db,
        ),
        dict(
            prompt="Use the ML inequality to find the bound M L for the integral along C2, where M is the maximum of |f| on C2 and L is the length of C2. Give M L correct to 2 decimal places.",
            answer_kind="number",
            answer=float(round_half_up(bound_fr, 2)),
            answer_text=bound_t,
            answer_latex=bound_t,
            marks=2,
            sensible_range=(0, None),
            working=[
                f"|f(z)| = {abs(lam)}|z|^{m + k} = {atext(mod_sym)} on |z| = {r.numerator}, so M = {atext(mod_sym)}.",
                f"L = {r.numerator} × {_pi_mult(abs(arc.n1 - arc.n0))} and M L = {bound_t}.",
            ],
            distractors=dc,
        ),
        dict(
            prompt="By how much does the bound M L from the previous part exceed the modulus of your answer to the integral along C2? Give your answer correct to 2 decimal places.",
            answer_kind="number",
            answer=float(round_half_up(slack_fr, 2)),
            answer_text=slack_t,
            answer_latex=slack_t,
            marks=2,
            sensible_range=(0, None),
            working=[
                f"|integral| = {ival:.4f}, so M L - |integral| = {bound_t} - {ival:.4f}, which is {slack_t} to 2 decimal places."
            ],
            distractors=dd,
        ),
    ]
    return dict(text=stem[0], latex=stem[1], parts=parts)


def _rtxt(r: Fr) -> str:
    return "" if r == 1 else str(r.numerator)


def _alatex(v) -> str:
    from ..core import latex_answer

    return latex_answer(sympy.sympify(v))
