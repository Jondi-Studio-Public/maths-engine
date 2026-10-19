"""Shared helpers for the second-year vector-calculus builds (`s2a2_*`): fields, surfaces and their question text.

* A formula is built as a SymPy expression and printed twice from the same tree: `plain()` for the learner-facing text
  (``x²y + e^(yz)``, Unicode, no Python notation) and `tex()` for the question's LaTeX field.
* `say()` joins prose and `M(...)` formulas into the pair ``(text, latex)`` a generator returns.
* Answers are stored the way the expression grader reads them: `machine()` is SymPy's own spelling (the stored
  `answer`), `shown()` the same with ``^`` for ``**`` (the stored `answer_text`). A vector answer is a bracketed
  tuple, ``(2*x*y, z^2, -x)``, which the expression grader compares component by component, in order.
* The numerical helpers (`fd_partial`, `gl2`, ...) are the independent second route every generator uses to confirm
  an exact SymPy result.
"""

from __future__ import annotations

import functools
import random
import re
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import sympy

from ..answers import Distractor, exact_value
from ..core import _escape_prose, attempts, rational_text
from ..grading import grade
from ..grading.safe_parse import sympify_safely

x, y, z = sympy.symbols("x y z")
u, v = sympy.symbols("u v")
XYZ = (x, y, z)
UV = (u, v)

_SUP = {2: "²", 3: "³"}
_FUNC_PLAIN = {sympy.sin: "sin", sympy.cos: "cos", sympy.tan: "tan", sympy.log: "ln"}
_FUNC_TEX = {sympy.sin: r"\sin", sympy.cos: r"\cos", sympy.tan: r"\tan", sympy.log: r"\ln"}


# ---------------------------------------------------------------------------------------------------------------
# printing


def _is_var_like(e) -> bool:
    return e.is_Symbol or e is sympy.pi or (e.is_Pow and e.base.is_Symbol and e.exp.is_Integer and e.exp > 0)


def _wrap(s: str, atomic: bool) -> str:
    return s if atomic else f"({s})"


def _render(e, tex: bool) -> tuple[str, bool]:
    """``(text, atomic)``: the text of `e` and whether it can stand as a factor or a base with no brackets."""
    if e.is_Integer:
        return str(int(e)), e >= 0
    if e.is_Rational:
        p, q = int(e.p), int(e.q)
        if tex:
            return (("-" if p < 0 else "") + rf"\frac{{{abs(p)}}}{{{q}}}"), False
        return f"{p}/{q}", False
    if e.is_Symbol:
        return str(e), True
    if e is sympy.pi:
        return (r"\pi" if tex else "π"), True
    if e is sympy.E:
        return "e", True
    if e.is_Add:
        parts = []
        for term in e.as_ordered_terms():
            s, _ = _render(term, tex)
            parts.append(s)
        out = parts[0]
        for s in parts[1:]:
            out += f" - {s[1:]}" if s.startswith("-") else f" + {s}"
        return out, False
    if e.is_Mul:
        coeff, rest = e.as_coeff_Mul()
        factors = list(sympy.Mul.make_args(rest))
        factors.sort(
            key=lambda f: (
                (-1, "", "")
                if f is sympy.pi
                else (0, str(f.free_symbols), str(f))
                if _is_var_like(f)
                else (1, "", str(f))
            )
        )
        pieces = []
        for f in factors:
            s, atomic = _render(f, tex)
            pieces.append((_wrap(s, atomic or f.is_Pow), f))
        text = ""
        for i, (s, f) in enumerate(pieces):
            if i:
                prev = pieces[i - 1][1]
                text += "" if (not tex and _is_var_like(prev) and _is_var_like(f)) else " "
            text += s
        if coeff == 1:
            return text, len(pieces) == 1 and pieces[0][1].is_Symbol
        if coeff == -1:
            return "-" + text, False
        cs, _ = _render(coeff, tex)
        if coeff.is_Rational and not coeff.is_Integer:
            neg = coeff < 0
            cs = ("-" if neg else "") + (_render(abs(coeff), tex)[0] if tex else f"({abs(coeff.p)}/{coeff.q})")
        joiner = "" if (not tex and _is_var_like(factors[0])) else " "
        return cs + joiner + text, False
    if e.is_Pow:
        base, exp = e.base, e.exp
        if exp == sympy.Rational(1, 2):
            s, atomic = _render(base, tex)
            if tex:
                return rf"\sqrt{{{s}}}", True
            return "√" + (s if atomic and not s.startswith("-") else f"({s})"), True
        bs, atomic = _render(base, tex)
        bs = _wrap(bs, atomic and not bs.startswith("-"))
        if exp.is_Integer and exp > 0:
            if tex:
                return f"{bs}^{{{int(exp)}}}", True
            return bs + (_SUP.get(int(exp)) or f"^{int(exp)}"), True
        es, _ = _render(exp, tex)
        return (f"{bs}^{{{es}}}" if tex else f"{bs}^({es})"), True
    if isinstance(e, sympy.exp):
        s, atomic = _render(e.args[0], tex)
        if tex:
            return f"e^{{{s}}}", True
        return ("e^" + s if e.args[0].is_Symbol else f"e^({s})"), True
    if e.func in _FUNC_PLAIN:
        s, _ = _render(e.args[0], tex)
        return (f"{_FUNC_TEX[e.func]}({s})" if tex else f"{_FUNC_PLAIN[e.func]}({s})"), True
    raise ValueError(f"cannot print {e!r}")


def plain(e) -> str:
    """The learner-facing text of an expression."""
    return _render(sympy.sympify(e), False)[0]


def tex(e) -> str:
    """The LaTeX of an expression."""
    return _render(sympy.sympify(e), True)[0]


def vec_plain(comps) -> str:
    return "⟨" + ", ".join(plain(c) for c in comps) + "⟩"


def vec_tex(comps) -> str:
    return r"\left\langle " + r",\, ".join(tex(c) for c in comps) + r" \right\rangle"


class M(str):
    """A formula inside a `say()` sentence: the plain text, with the LaTeX alongside."""

    tex: str

    def __new__(cls, plain_text: str, tex_text: str):
        obj = super().__new__(cls, plain_text)
        obj.tex = tex_text
        return obj


def m(e) -> M:
    """A scalar formula."""
    return M(plain(e), tex(e))


def mv(comps) -> M:
    """A vector written ⟨a, b, c⟩."""
    return M(vec_plain(comps), vec_tex(comps))


def raw(plain_text: str, tex_text: str | None = None) -> M:
    """A fixed piece of notation; the LaTeX defaults to the text."""
    return M(plain_text, plain_text if tex_text is None else tex_text)


def say(*parts) -> tuple[str, str]:
    """``(text, latex)`` for a sentence of prose strings and `M` formulas."""
    text, latex = [], []
    for p in parts:
        if isinstance(p, M):
            text.append(str(p))
            latex.append(p.tex)
        elif p:
            text.append(p)
            latex.append(r"\text{" + _escape_prose(p) + "}")
    return "".join(text), "".join(latex)


def cat(*parts) -> M:
    """Several `M` pieces (or plain strings, taken as text) written one after another."""
    ms = [p if isinstance(p, M) else M(p, p) for p in parts]
    return M("".join(str(p) for p in ms), "".join(p.tex for p in ms))


_SLOT = re.compile(r"\{(\w+)\}")


def fill(template: str, **slots) -> tuple[str, str]:
    """``say()`` for a template whose ``{name}`` slots are `M` formulas (other text is prose)."""
    pieces = []
    pos = 0
    for match in _SLOT.finditer(template):
        pieces.append(template[pos : match.start()])
        value = slots[match.group(1)]
        pieces.append(value if isinstance(value, M) else str(value))
        pos = match.end()
    pieces.append(template[pos:])
    return say(*pieces)


def fdef(name: str = "F") -> M:
    """``F(x, y, z) = `` with the name in bold in the LaTeX."""
    return M(f"{name}(x, y, z) = ", rf"\mathbf{{{name}}}(x, y, z) = ")


def sdef(name: str, expr) -> M:
    """``f(x, y, z) = <expression>`` for a scalar function."""
    return M(f"{name}(x, y, z) = {plain(expr)}", f"{name}(x, y, z) = {tex(expr)}")


def bold(name: str) -> M:
    return M(name, rf"\mathbf{{{name}}}")


def divop(name: str = "F") -> M:
    return M(f"∇ · {name}", rf"\nabla \cdot \mathbf{{{name}}}")


def curlop(name: str = "F") -> M:
    return M(f"∇ × {name}", rf"\nabla \times \mathbf{{{name}}}")


R_OF_UV = M("r(u, v) = ", r"\mathbf{r}(u, v) = ")


def pt_plain(p) -> str:
    return "(" + ", ".join(plain(c) for c in p) + ")"


def pt_tex(p) -> str:
    return "(" + ", ".join(tex(c) for c in p) + ")"


def mpt(p) -> M:
    return M(pt_plain(p), pt_tex(p))


# ---------------------------------------------------------------------------------------------------------------
# answers


def machine(e) -> str:
    """SymPy's spelling, the stored `answer` of an expression (Euler's number written ``exp(1)``)."""
    return re.sub(r"\bE\b", "exp(1)", sympy.sstr(sympy.sympify(e)))


def shown(e) -> str:
    """The stored `answer_text` of an expression: SymPy's spelling with ``^`` for ``**``."""
    return machine(e).replace("**", "^")


def nice(answer_text: str) -> str:
    """A stored answer text read back and printed the way the question text is (``4x`` for ``4*x``), for worked steps."""
    value = sympify_safely(answer_text.replace("^", "**"))
    if isinstance(value, tuple):
        return pt_plain(value)
    return plain(value)


def tup_machine(comps) -> str:
    return "(" + ", ".join(machine(c) for c in comps) + ")"


def tup_shown(comps) -> str:
    return tup_machine(comps).replace("**", "^")


def expr_answer(e) -> dict:
    return dict(answer=machine(e), answer_text=shown(e), answer_kind="expression")


def vec_answer(comps) -> dict:
    return dict(answer=tup_machine(comps), answer_text=tup_shown(comps), answer_kind="expression")


def dexpr(e, tag: str) -> Distractor:
    return Distractor(shown(e), tag, shown(e))


def dvec(comps, tag: str) -> Distractor:
    return Distractor(tup_shown(comps), tag, tup_shown(comps))


def frac_answer(value) -> dict:
    """An exact rational answer (kind ``fraction``)."""
    f = Fraction(int(sympy.Rational(value).p), int(sympy.Rational(value).q))
    return dict(answer=exact_value(f), answer_text=rational_text(f), answer_kind="fraction")


def dfrac(value, tag: str) -> Distractor:
    r = sympy.Rational(value)
    f = Fraction(int(r.p), int(r.q))
    return Distractor(rational_text(f), tag, exact_value(f))


def _target(answer, answer_text: str):
    return SimpleNamespace(answer=answer, answer_text=answer_text, answer_alt=None, answer_form=None)


_PROBE_VALUES = (
    sympy.Rational(3, 2),
    sympy.Rational(-5, 7),
    sympy.Rational(2, 3),
    sympy.Rational(-7, 4),
    sympy.Rational(4, 5),
    sympy.Rational(9, 7),
    sympy.Rational(1, 3),
    sympy.Rational(11, 5),
    sympy.Rational(-3, 2),
    sympy.Rational(5, 6),
)


def _read(text: str):
    return sympify_safely(text.replace("^", "**"))


def differs(a_text: str, b_text: str) -> bool:
    """Whether two answer texts (expressions, or bracketed tuples of them) are different values.

    An independent numerical probe at several rational points, used by the generators to keep a distractor from being
    a second spelling of the answer; the tests grade every distractor with the real grader as well."""
    a, b = _read(a_text), _read(b_text)
    if isinstance(a, tuple) != isinstance(b, tuple):
        return True
    if isinstance(a, tuple):
        return len(a) != len(b) or any(differs_expr(p, q) for p, q in zip(a, b))
    return differs_expr(a, b)


def differs_expr(a, b) -> bool:
    a, b = sympy.sympify(a), sympy.sympify(b)
    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    for shift in range(3):
        point = {s: _PROBE_VALUES[(i + 3 * shift) % len(_PROBE_VALUES)] for i, s in enumerate(symbols)}
        va, vb = complex(sympy.N(a.xreplace(point), 20)), complex(sympy.N(b.xreplace(point), 20))
        if abs(va - vb) > 1e-9 * max(1.0, abs(va), abs(vb)):
            return True
    return False


def usable(answer, answer_text: str, candidates, kind: str = "expression", n: int = 4) -> list[Distractor]:
    """The first `n` candidates that are wrong, one per distinct text, none equal to the answer's text.

    A ``fraction`` candidate is judged by the real grader; an ``expression`` candidate by an independent numerical
    probe (`differs`), which is much cheaper than the grader's own symbolic comparison. Either way a distractor that
    is secretly a second spelling of the answer is dropped."""
    target = _target(answer, answer_text)
    out, seen = [], {answer_text}
    for c in candidates:
        if c.text in seen:
            continue
        if kind == "expression":
            if not differs(answer_text, c.text):
                continue
        elif grade(kind, target, c.text).correct:
            continue
        seen.add(c.text)
        out.append(c)
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------------------------------------------------------
# random polynomials and fields


def _monomials(variables, max_degree: int, min_degree: int = 1):
    out = []
    n = len(variables)

    def build(i, remaining, acc):
        if i == n:
            deg = sum(acc)
            if min_degree <= deg <= max_degree:
                term = sympy.Integer(1)
                for var, p in zip(variables, acc):
                    term *= var**p
                out.append(term)
            return
        for p in range(remaining + 1):
            build(i + 1, remaining - p, acc + [p])

    build(0, max_degree, [])
    return out


def rand_poly(
    rng: random.Random,
    variables=XYZ,
    terms: int = 3,
    max_degree: int = 3,
    lo: int = -4,
    hi: int = 5,
    min_degree: int = 1,
    need=None,
):
    """A polynomial of `terms` distinct monomials with non-zero integer coefficients.

    `need`, if given, is a monomial that must be among them."""
    pool = _monomials(variables, max_degree, min_degree)
    chosen = rng.sample(pool, min(terms, len(pool)))
    if need is not None and need not in chosen:
        chosen[0] = need
    total = sympy.Integer(0)
    for mono in chosen:
        total += rng.choice([c for c in range(lo, hi + 1) if c != 0]) * mono
    return total


def nz(rng: random.Random, lo: int = -4, hi: int = 5) -> int:
    return rng.choice([c for c in range(lo, hi + 1) if c != 0])


def lattice_point(rng: random.Random, lo: int = -2, hi: int = 3, nonzero: bool = False):
    for _ in attempts(100, "a lattice point"):
        p = tuple(sympy.Integer(rng.randint(lo, hi)) for _ in range(3))
        if not nonzero or all(c != 0 for c in p):
            return p
    raise AssertionError("unreachable")


def at(e, p, variables=XYZ):
    """The value of `e` at the point `p`."""
    return sympy.sympify(e).xreplace({var: sympy.sympify(c) for var, c in zip(variables, p)})


def div(F, variables=XYZ):
    return sum(sympy.diff(c, var) for c, var in zip(F, variables))


def curl(F):
    P, Q, R = F
    return (
        sympy.diff(R, y) - sympy.diff(Q, z),
        sympy.diff(P, z) - sympy.diff(R, x),
        sympy.diff(Q, x) - sympy.diff(P, y),
    )


def grad(f, variables=XYZ):
    return tuple(sympy.diff(f, var) for var in variables)


def dot(a, b):
    return sum(p * q for p, q in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def pint(e, var):
    """The antiderivative of a polynomial in `var` (other symbols are coefficients), with no constant."""
    e = sympy.expand(e)
    if e == 0:
        return sympy.Integer(0)
    return sympy.Poly(e, var).integrate().as_expr()


def simp(e):
    """A tidy form: expanded and cancelled."""
    return sympy.simplify(sympy.expand(e))


def tidy(F):
    return tuple(simp(c) for c in F)


def same(a, b) -> bool:
    return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0


# ---------------------------------------------------------------------------------------------------------------
# numerical second routes


@functools.lru_cache(maxsize=512)
def _fn(e, variables):
    f = sympy.lambdify(variables, e, "math")
    return lambda *a: float(f(*a))


def fd_partial(e, variables, point, axis: int, h: float = 1e-3) -> float:
    """Fourth-order central difference of `e` with respect to variables[axis] at `point`, as a float."""
    f = _fn(e, variables)
    p = [float(c) for c in point]

    def at_shift(s):
        q = list(p)
        q[axis] += s
        return f(*q)

    return (-at_shift(2 * h) + 8 * at_shift(h) - 8 * at_shift(-h) + at_shift(-2 * h)) / (12 * h)


def fd_div(F, point, variables=XYZ) -> float:
    return sum(fd_partial(c, variables, point, i) for i, c in enumerate(F))


def fd_curl(F, point, variables=XYZ):
    d = [[fd_partial(c, variables, point, j) for j in range(3)] for c in F]  # d[i][j] = dF_i / dx_j
    return (d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1])


def fd_grad(e, point, variables=XYZ):
    return tuple(fd_partial(e, variables, point, i) for i in range(3))


def close(a, b, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def close_vec(a, b, tol: float = 1e-6) -> bool:
    return all(close(p, q, tol) for p, q in zip(a, b))


def gl1(f, lo: float, hi: float, n: int = 24) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return float(half * sum(w * f(mid + half * t) for t, w in zip(nodes, weights)))


def gl2(f, u_range, v_range, n: int = 24) -> float:
    """Gauss-Legendre over a region. `v_range` may be a pair of numbers or a pair of functions of u."""
    ulo, uhi = u_range

    def inner(uv):
        vlo, vhi = (v_range[0](uv), v_range[1](uv)) if callable(v_range[0]) else v_range
        return gl1(lambda t: f(uv, t), vlo, vhi, n)

    return gl1(inner, ulo, uhi, n)


def line_integral_numeric(F, r, t, a, b, n: int = 40) -> float:
    """∫ F(r(t)) · r'(t) dt from a to b by Gauss-Legendre, with every function lambdified (no symbolic substitution)."""
    fn = sympy.lambdify(XYZ, list(F), "math")
    rn = sympy.lambdify(t, list(r), "math")
    dn = sympy.lambdify(t, [sympy.diff(c, t) for c in r], "math")

    def integrand(s):
        pos = rn(s)
        vel = dn(s)
        return sum(f * v for f, v in zip(fn(*pos), vel))

    return gl1(lambda s: float(integrand(s)), float(a), float(b), n)


def float_of(e) -> float:
    return float(sympy.N(e, 30))
