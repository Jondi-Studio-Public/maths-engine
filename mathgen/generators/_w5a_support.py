"""Shared pieces for the second-year complex-analysis builds (`w5a_*_build.py`).

Everything a question needs is built backwards from a chosen answer, so the arithmetic happens on exact Gaussian
rationals (`C`, a pair of `Fraction`s) and only the finished answer is handed to SymPy for printing and grading.
That keeps `make()` fast: no `simplify`, no `integrate`, no `residue`.

* `C` and the `p*` / `sdiv` helpers: exact complex arithmetic, polynomials and power-series division.
* `X`, `msay` and the `*_x` renderers: every piece of mathematics is written once as a `(plain, TeX)` pair so the
  learner-facing text and the `latex` field never drift apart.
* `atext` / `astr` / `alatex`: an exact answer as the text a learner types, as a SymPy string, and as TeX.
* `dist_list`, `Link`, `link_part`, `single`: tagged distractors and the shapes `make()` returns.
* `circle_mean`: the numeric second route. `(1/2*pi*i)` times a contour integral over a circle is the mean of
  `f(z) (z - c)` over equally spaced points, which converges geometrically for a function analytic near the circle.
"""

from __future__ import annotations

import cmath
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Callable, NamedTuple

import sympy

from ..answers import Distractor, exact_value
from ..core import _escape_prose, check, latex_answer, rational_text


def pn(n) -> str:
    """A number for substituting into a formula: negative values are bracketed, (-2)."""
    t = str(n)
    return f"({t})" if t.startswith("-") else t


def join_signed(items) -> str:
    """Terms joined with their signs absorbed: join_signed(["3", "-4", "2i"]) is "3 - 4 + 2i"."""
    out = ""
    for k, item in enumerate(items):
        item = str(item)
        if k == 0:
            out = item
        elif item.startswith("-"):
            out += " - " + item[1:]
        else:
            out += " + " + item
    return out


def shift(a, var: str = "z") -> str:
    """The text of var - a for an integer a: z - 2, z + 1, or z for 0."""
    a = int(a)
    return var if a == 0 else f"{var} - {a}" if a > 0 else f"{var} + {-a}"


# -- exact complex arithmetic ------------------------------------------------------------------------------------------


class C:
    """A Gaussian rational: re + im * i, both `Fraction`s."""

    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = Fr(re)
        self.im = Fr(im)

    @staticmethod
    def of(v) -> "C":
        return v if isinstance(v, C) else C(v)

    def __add__(self, o):
        o = C.of(o)
        return C(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __sub__(self, o):
        o = C.of(o)
        return C(self.re - o.re, self.im - o.im)

    def __rsub__(self, o):
        return C.of(o) - self

    def __neg__(self):
        return C(-self.re, -self.im)

    def __mul__(self, o):
        o = C.of(o)
        return C(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = C.of(o)
        d = o.re * o.re + o.im * o.im
        check(d != 0, "division by zero")
        n = self * o.conj()
        return C(n.re / d, n.im / d)

    def __rtruediv__(self, o):
        return C.of(o) / self

    def __pow__(self, n: int):
        if n < 0:
            return C(1) / (self ** (-n))
        out = C(1)
        for _ in range(n):
            out = out * self
        return out

    def __eq__(self, o):
        o = C.of(o) if isinstance(o, (int, Fr, C)) else None
        return o is not None and self.re == o.re and self.im == o.im

    def __hash__(self):
        return hash((self.re, self.im))

    def __repr__(self):
        return f"C({self.re}, {self.im})"

    def conj(self) -> "C":
        return C(self.re, -self.im)

    def abs2(self) -> Fr:
        return self.re * self.re + self.im * self.im

    def is_real(self) -> bool:
        return self.im == 0

    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def __complex__(self):
        return complex(float(self.re), float(self.im))

    def sym(self):
        return sympy.Rational(self.re.numerator, self.re.denominator) + sympy.I * sympy.Rational(
            self.im.numerator, self.im.denominator
        )


I_ = C(0, 1)


def qsym(f) -> sympy.Rational:
    f = Fr(f)
    return sympy.Rational(f.numerator, f.denominator)


# polynomials are lists of C, lowest power first


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else C(0)) + (b[k] if k < len(b) else C(0)) for k in range(n)]


def pmul(a, b):
    out = [C(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = out[i + j] + x * y
    return out


def ppow(a, n: int):
    out = [C(1)]
    for _ in range(n):
        out = pmul(out, a)
    return out


def peval(p, z):
    out = C(0)
    for c in reversed(p):
        out = out * z + c
    return out


def pshift(p, a):
    """Coefficients of p(a + t) in t."""
    out = [C(0)]
    for c in reversed(p):
        out = padd(pmul(out, [C.of(a), C(1)]), [c])
    return out


def pders(p):
    return [p[k] * k for k in range(1, len(p))] or [C(0)]


def sdiv(num, den, n: int):
    """The first n power-series coefficients of num/den (den[0] != 0)."""
    out: list[C] = []
    for k in range(n):
        s = num[k] if k < len(num) else C(0)
        for j in range(1, min(k, len(den) - 1) + 1):
            s = s - den[j] * out[k - j]
        out.append(s / den[0])
    return out


@dataclass
class Rat:
    """f(z) = num(z) / prod (z - a)^m over `poles = [(a, m), ...]`, with num a polynomial (lowest power first)."""

    num: list
    poles: list = field(default_factory=list)

    def at(self, z: complex) -> complex:
        cache = self.__dict__.get("_cache")
        if cache is None:
            cache = self.__dict__["_cache"] = (
                [complex(c) for c in self.num],
                [(complex(a), m) for a, m in self.poles],
            )
        coeffs, poles = cache
        top = 0j
        for c in reversed(coeffs):
            top = top * z + c
        bottom = 1 + 0j
        for a, m in poles:
            bottom *= (z - a) ** m
        return top / bottom

    def laurent(self, j: int, extra: int = 3):
        """Laurent coefficients about pole j: a dict {power: C} for powers -m .. extra - 1."""
        a, m = self.poles[j]
        gnum = pshift(self.num, a)
        gden = [C(1)]
        for i, (b, n) in enumerate(self.poles):
            if i != j:
                gden = pmul(gden, ppow([a - b, C(1)], n))
        ser = sdiv(gnum, gden, m + extra)
        return {k - m: ser[k] for k in range(m + extra)}

    def residue(self, j: int) -> C:
        return self.laurent(j, 1)[-1]


def circle_mean(fn: Callable[[complex], complex], centre: complex, radius: float, n: int = 96) -> complex:
    """(1 / 2*pi*i) times the counter-clockwise integral of fn over the circle: the mean of fn(z)(z - c)."""
    total = 0j
    for k in range(n):
        w = radius * cmath.exp(2j * math.pi * k / n)
        total += fn(centre + w) * w
    return total / n


# -- paired plain / TeX text -------------------------------------------------------------------------------------------


class X(NamedTuple):
    """A piece of mathematics: the learner-facing text and its TeX."""

    p: str
    t: str


def msay(*parts) -> tuple[str, str]:
    """(text, latex) for a sentence made of prose strings and `X` formulas."""
    text: list[str] = []
    tex: list[str] = []
    for part in parts:
        if isinstance(part, X):
            text.append(part.p)
            tex.append(part.t)
        elif part:
            text.append(part)
            tex.append(r"\text{" + _escape_prose(part) + "}")
    return re.sub(r" {2,}", " ", "".join(text)).strip(), "".join(tex)


_SUP = {"2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def _frac_x(f: Fr, sign: bool = True) -> X:
    f = Fr(f)
    mag = abs(f)
    neg = "-" if (f < 0 and sign) else ""
    if mag.denominator == 1:
        return X(f"{neg}{mag.numerator}", f"{neg}{mag.numerator}")
    return X(f"{neg}{mag.numerator}/{mag.denominator}", rf"{neg}\frac{{{mag.numerator}}}{{{mag.denominator}}}")


def cx_x(c) -> X:
    """A Gaussian rational as text: 3, -2i, 1 + 2i, 1/2 - i, 3/2 + 5i/2."""
    c = C.of(c)
    a, b = c.re, c.im
    if b == 0:
        return _frac_x(a)
    mag = abs(b)
    if mag == 1:
        ip, it = "i", "i"
    elif mag.denominator == 1:
        ip, it = f"{mag.numerator}i", f"{mag.numerator}i"
    else:
        n_ = "" if mag.numerator == 1 else str(mag.numerator)
        ip, it = f"{n_}i/{mag.denominator}", rf"\frac{{{n_ or 1}}}{{{mag.denominator}}}i"
    if a == 0:
        return X(("-" if b < 0 else "") + ip, ("-" if b < 0 else "") + it)
    re_part = _frac_x(a)
    op = "-" if b < 0 else "+"
    return X(f"{re_part.p} {op} {ip}", f"{re_part.t} {op} {it}")


def cx_paren(c) -> X:
    """The same, in brackets when it is not a single signed term."""
    c = C.of(c)
    x = cx_x(c)
    if c.re != 0 and c.im != 0:
        return X(f"({x.p})", f"({x.t})")
    return x


def zminus_x(a, var: str = "z") -> X:
    """var - a written out: z, z - 2, z + 3i, z - 1 + i."""
    a = C.of(a)
    p, t = var, var
    for part, unit in ((-a.re, ""), (-a.im, "i")):
        if part == 0:
            continue
        mag = abs(part)
        sign = "+" if part > 0 else "-"
        if unit and mag == 1:
            body = X("i", "i")
        elif unit:
            body = cx_x(C(0, mag))
        else:
            body = _frac_x(mag)
        p += f" {sign} {body.p}"
        t += f" {sign} {body.t}"
    return X(p, t)


def fac_x(a, k: int = 1, var: str = "z") -> X:
    """(var - a)^k, with brackets unless it is a bare variable to the first power."""
    a = C.of(a)
    base = zminus_x(a, var)
    if a.is_zero():
        pp, tt = base.p, base.t
    else:
        pp, tt = f"({base.p})", f"({base.t})"
    if k == 1:
        return X(pp, tt)
    sup = _SUP.get(str(k))
    return X(pp + (sup if sup else f"^{k}"), tt + f"^{{{k}}}")


def prod_x(factors: list[X]) -> X:
    return X("".join(f.p for f in factors), "".join(f.t for f in factors))


def _signed_tail(c: C) -> tuple[str, str]:
    """A constant with two parts written as signed terms: + 2 - i, - 1/2 + 3i/2."""
    re_x, im_x = _frac_x(abs(c.re)), cx_x(C(0, abs(c.im)))
    sr, si = ("-" if c.re < 0 else "+"), ("-" if c.im < 0 else "+")
    return f"{sr} {re_x.p} {si} {im_x.p}", f"{sr} {re_x.t} {si} {im_x.t}"


def poly_x(coeffs, var: str = "z") -> X:
    """A polynomial, highest power first, from coefficients lowest power first."""
    coeffs = [C.of(c) for c in coeffs]
    pieces_p: list[str] = []
    pieces_t: list[str] = []
    for power in range(len(coeffs) - 1, -1, -1):
        c = coeffs[power]
        if c.is_zero():
            continue
        if c.is_real():
            neg, mag = c.re < 0, abs(c.re)
            coef = X("", "") if (mag == 1 and power > 0) else _frac_x(mag)
            if mag.denominator != 1 and power > 0:
                coef = X(f"({coef.p})", f"({coef.t})")
        elif c.re == 0:
            neg, mag = c.im < 0, abs(c.im)
            coef = X("i", "i") if mag == 1 else cx_x(C(0, mag))
            if mag.denominator != 1 and power > 0:
                coef = X(f"({coef.p})", f"({coef.t})")
        else:
            neg, coef = False, cx_paren(c)
        if power == 0 and pieces_p and not c.is_real() and c.re != 0:
            tail_p, tail_t = _signed_tail(c)
            pieces_p.append(tail_p)
            pieces_t.append(tail_t)
            continue
        if power == 0:
            seg_p, seg_t = coef.p, coef.t
        elif power == 1:
            seg_p, seg_t = coef.p + var, coef.t + var
        else:
            sup = _SUP.get(str(power))
            seg_p = coef.p + var + (sup if sup else f"^{power}")
            seg_t = coef.t + var + f"^{{{power}}}"
        if not pieces_p:
            pieces_p.append(("-" if neg else "") + seg_p)
            pieces_t.append(("-" if neg else "") + seg_t)
        else:
            pieces_p.append(f"{'-' if neg else '+'} {seg_p}")
            pieces_t.append(f"{'-' if neg else '+'} {seg_t}")
    if not pieces_p:
        return X("0", "0")
    return X(" ".join(pieces_p), " ".join(pieces_t))


def _atomic(s: str) -> bool:
    """A single number or symbol (with an optional power), or a bracketed group: safe to divide by without brackets."""
    return bool(re.fullmatch(r"(\d+(\.\d+)?|[A-Za-zπ])[²³⁴⁵⁶⁷⁸⁹]?", s)) or _enclosed(s)


def _enclosed(s: str) -> bool:
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for k, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and k < len(s) - 1:
            return False
    return True


def frac_x(num: X, den: X) -> X:
    n = num.p if _atomic(num.p) or " " not in num.p else f"({num.p})"
    d = den.p if _atomic(den.p) else f"({den.p})"
    return X(f"{n}/{d}", rf"\frac{{{num.t}}}{{{den.t}}}")


def rat_x(rat: Rat, var: str = "z") -> X:
    """The quotient num / product of powers, as one fraction."""
    den = prod_x([fac_x(a, m, var) for a, m in rat.poles])
    return frac_x(poly_x(rat.num, var), den)


def tex_of(s: str) -> str:
    return s


# -- answers: text, SymPy string, TeX ----------------------------------------------------------------------------------


def astr(v) -> str:
    """The SymPy string stored as an expression answer."""
    return sympy.sstr(sympy.sympify(v))


def _exp_arg(arg: str) -> str:
    if re.fullmatch(r"\d+", arg):
        return "e" if arg == "1" else f"e^{arg}"
    return f"e^({arg})"


def atext(v) -> str:
    """An exact value the way a learner types it: 2πi, ln(2)/2 + 3πi/4, e^(-π/2), (1 + i)/2, π(√2)/2."""
    s = sympy.sstr(sympy.sympify(v)).replace("**", "^")
    s = re.sub(r"exp\(([^()]*)\)", lambda m: _exp_arg(m.group(1)), s)
    s = re.sub(r"\bE\b", "e", s)
    s = s.replace("log(", "ln(").replace("pi", "π")
    s = re.sub(r"\bI\b", "i", s)
    s = re.sub(r"sqrt\((\d+)\)", r"√\1", s)
    s = re.sub(r"\^2(?![\d(])", "²", s)
    s = re.sub(r"\^3(?![\d(])", "³", s)
    s = re.sub(r"(?<=[\w)²³π])\*(?=[\w(√π])", "", s)
    s = s.replace("*", "")
    s = s.replace("iπ", "πi")
    s = re.sub(r"√(\d+)π", r"π(√\1)", s)
    return s


def alatex(v) -> str:
    return latex_answer(sympy.sympify(v))


def val_x(v) -> X:
    """An exact value as an `X`, for use inside prompts."""
    return X(atext(v), sympy.latex(sympy.sympify(v), ln_notation=True))


# -- distractors -------------------------------------------------------------------------------------------------------

_PROBES = (
    {
        "x": sympy.Rational(37, 100),
        "y": sympy.Rational(113, 100),
        "z": sympy.Rational(7, 10) + sympy.I * sympy.Rational(3, 10),
    },
    {
        "x": sympy.Rational(-21, 10),
        "y": sympy.Rational(17, 10),
        "z": sympy.Rational(-6, 5) + sympy.I * sympy.Rational(9, 10),
    },
)


def same_val(a, b) -> bool:
    a, b = sympy.sympify(a), sympy.sympify(b)
    free = a.free_symbols | b.free_symbols
    if not free:
        try:
            return abs(complex(a) - complex(b)) < 1e-9 * max(1.0, abs(complex(b)))
        except (TypeError, ValueError):
            return a == b
    for probe in _PROBES:
        sub = {s: probe.get(s.name, sympy.Rational(13, 10)) for s in free}
        try:
            if abs(complex(sympy.N((a - b).subs(sub), 15))) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _vkey(v):
    """The numeric fingerprint of a value (its complex value at each probe), computed once per value."""
    free = v.free_symbols
    probes = (None,) if not free else _PROBES
    out = []
    for probe in probes:
        try:
            if probe is None:
                out.append(complex(v))
            else:
                sub = {s: probe.get(s.name, sympy.Rational(13, 10)) for s in free}
                out.append(complex(sympy.N(v.subs(sub), 15)))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def _keys_match(k1, k2) -> bool:
    if k1 is None or k2 is None or len(k1) != len(k2):
        return False
    return all(abs(a - b) < 1e-9 * max(1.0, abs(b)) for a, b in zip(k1, k2))


def _shape(v) -> str | None:
    v = sympy.sympify(v)
    if v.is_Integer:
        return "int"
    if v.is_Rational:
        return "frac"
    return None


def dist_list(answer, cands, n: int = 4, finite_only: bool = True) -> list[Distractor]:
    """Up to n distinct wrong expression options from `(value, misconception)` candidates."""
    answer = sympy.sympify(answer)
    want = _shape(answer)
    ordered = sorted(
        enumerate(cands), key=lambda ic: (0 if want is not None and _shape(ic[1][0]) == want else 1, ic[0])
    )
    seen = [(answer, _vkey(answer))]
    out: list[Distractor] = []
    for _i, (value, tag) in ordered:
        value = sympy.sympify(value)
        if finite_only and (value.has(sympy.zoo, sympy.nan, sympy.oo)):
            continue
        key = _vkey(value)
        if key is None:
            if any(value == s_ for s_, _k in seen):
                continue
        elif any(_keys_match(key, k_) for _s, k_ in seen):
            continue
        seen.append((value, key))
        out.append(Distractor(atext(value), tag, astr(value)))
        if len(out) == n:
            break
    return out


def frac_dist(answer, cands, n: int = 4) -> list[Distractor]:
    """Wrong options for a part graded as a fraction (an exact rational)."""
    answer = sympy.sympify(answer)
    seen = [answer]
    want = _shape(answer)
    ordered = sorted(enumerate(cands), key=lambda ic: (0 if _shape(ic[1][0]) == want else 1, ic[0]))
    out: list[Distractor] = []
    for _i, (value, tag) in ordered:
        value = sympy.sympify(value)
        if not value.is_Rational or any(same_val(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(rational_text(value), tag, exact_value(value)))
        if len(out) == n:
            break
    return out


def int_dist(answer: int, cands, n: int = 4) -> list[Distractor]:
    seen = {int(answer)}
    out: list[Distractor] = []
    for value, tag in cands:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(Distractor(str(value), tag, value))
        if len(out) == n:
            break
    return out


def single(
    prompt,
    value,
    working,
    wrongs,
    kind: str = "expression",
    n: int = 4,
    fallback_tag: str | None = "cx2-arithmetic-slip",
    answer_text: str | None = None,
    answer_latex: str | None = None,
    dist: list | None = None,
) -> dict:
    """The dict a single-answer generator returns. `fallback_tag` names the slip behind simple scalings of the answer,
    used only when the listed wrong answers leave fewer than three distinct options."""
    if dist is None:
        dist = dist_list(value, wrongs, n)
    if len(dist) < 3 and fallback_tag is not None:
        v = sympy.sympify(value)
        extra = [
            (-v, fallback_tag),
            (v * 2, fallback_tag),
            (v / 2, fallback_tag),
            (v * 3, fallback_tag),
            (v + 1, fallback_tag),
            (v - 1, fallback_tag),
            (v + sympy.I, fallback_tag),
            (v + 2, fallback_tag),
        ]
        dist = dist_list(value, list(wrongs) + extra, n)
    check(len(dist) >= 3, "fewer than three distinct distractors")
    return dict(
        text=prompt[0],
        latex=prompt[1],
        answer=astr(value),
        answer_text=answer_text if answer_text is not None else atext(value),
        answer_latex=answer_latex if answer_latex is not None else alatex(value),
        answer_kind=kind,
        working=working,
        distractors=dist,
    )


# -- exam chains -------------------------------------------------------------------------------------------------------


@dataclass
class Link:
    prompt: tuple  # (text, latex)
    value: object  # int for "number", SymPy value otherwise
    kind: str  # "number", "fraction" or "expression"
    marks: int
    working: list
    wrongs: list
    ft: str | None = None
    sensible: tuple | None = None


def link_part(link: Link) -> dict:
    v = link.value
    if link.kind == "number":
        distractors = int_dist(v, link.wrongs)
        answer, text, latex = int(v), str(int(v)), str(int(v))
    elif link.kind == "fraction":
        v = sympy.sympify(v)
        distractors = frac_dist(v, link.wrongs)
        if len(distractors) < 3:
            extra = [
                (-v, "calc-sign-slip"),
                (v * 2, "cx2-arithmetic-slip"),
                (v / 2, "cx2-arithmetic-slip"),
                (v + 1, "cx2-arithmetic-slip"),
                (v - 1, "cx2-arithmetic-slip"),
            ]
            distractors = frac_dist(v, list(link.wrongs) + extra)
        answer, text, latex = exact_value(v), rational_text(v), alatex(v)
    else:
        distractors = dist_list(v, link.wrongs)
        if len(distractors) < 3:
            sv = sympy.sympify(v)
            extra = [
                (-sv, "calc-sign-slip"),
                (sv * 2, "cx2-arithmetic-slip"),
                (sv / 2, "cx2-arithmetic-slip"),
                (sv + 1, "cx2-arithmetic-slip"),
                (sv - 1, "cx2-arithmetic-slip"),
            ]
            distractors = dist_list(v, list(link.wrongs) + extra)
        answer, text, latex = astr(v), atext(v), alatex(v)
    check(len(distractors) >= 2, f"fewer than two distractors for {link.prompt[0]!r}")
    part = dict(
        prompt=link.prompt[0],
        prompt_latex=link.prompt[1],
        answer_kind=link.kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex,
        marks=link.marks,
        working=link.working,
        distractors=distractors,
    )
    if link.ft:
        part["follow_through"] = link.ft
    if link.sensible and link.kind in ("number", "fraction"):
        part["sensible_range"] = link.sensible
    return part


def ineq_part(prompt, text: str, marks: int, working, cands) -> dict:
    """An inequality part. Candidates are spelled like the key, so a different string is a different set."""
    options, seen = [], {text}
    for wrong, tag in cands:
        if wrong in seen:
            continue
        seen.add(wrong)
        options.append(Distractor(wrong, tag, wrong))
    picked = options[:3]
    check(len(picked) >= 2, "fewer than two usable inequality distractors")
    return dict(
        prompt=prompt[0],
        prompt_latex=prompt[1],
        answer_kind="inequality",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(text, plain=True),
        marks=marks,
        working=working,
        distractors=picked,
    )


def exam_dict(stem, links) -> dict:
    return dict(
        text=stem[0], latex=stem[1], parts=[link_part(link) if isinstance(link, Link) else link for link in links]
    )


def numeric_close(got: complex, want, what: str, tol: float = 1e-7) -> None:
    want = (
        complex(sympy.N(sympy.sympify(want), 15)) if not isinstance(want, (int, float, complex, Fr)) else complex(want)
    )
    check(abs(got - want) < tol * max(1.0, abs(want)), f"{what}: numeric {got}, exact {want}")
