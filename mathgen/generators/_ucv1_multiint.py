"""Exact polynomial arithmetic for the multivariable integral builds (`ucv1_line_integral_build`,
`ucv1_triple_integral_build`).

A `Poly` is a polynomial in up to three variables (x, y, z are indices 0, 1, 2; a parameter t is index 0 of a
separate `Poly`) with `Fraction` coefficients, stored as ``{(i, j, k): coefficient}``. It does exactly what those two
generators need and nothing more: sum, product, power, composition, an antiderivative, an exact definite integral
between polynomial limits, evaluation, and the learner-facing text of a polynomial.

`gauss_nested` is a second, numerical route to a (multiple) integral over limits given as polynomials in the outer
variables: six-point Gauss-Legendre in each variable, exact for any polynomial of degree 11 or less in each. The
generators use it to check what the exact route produced.
"""

from __future__ import annotations

from fractions import Fraction

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# Six-point Gauss-Legendre on [-1, 1].
_GL_NODES = (
    -0.9324695142031521,
    -0.6612093864662645,
    -0.2386191860831969,
    0.2386191860831969,
    0.6612093864662645,
    0.9324695142031521,
)
_GL_WEIGHTS = (
    0.1713244923791704,
    0.3607615730481386,
    0.4679139345726910,
    0.4679139345726910,
    0.3607615730481386,
    0.1713244923791704,
)


class Poly:
    __slots__ = ("t",)

    def __init__(self, terms=None):
        self.t = {k: Fraction(v) for k, v in (terms or {}).items() if v != 0}

    # -- construction ------------------------------------------------------------------------------------------
    @staticmethod
    def const(c) -> "Poly":
        return Poly({(0, 0, 0): Fraction(c)})

    @staticmethod
    def var(i: int) -> "Poly":
        e = [0, 0, 0]
        e[i] = 1
        return Poly({tuple(e): 1})

    @staticmethod
    def mono(c, i=0, j=0, k=0) -> "Poly":
        return Poly({(i, j, k): Fraction(c)})

    # -- arithmetic --------------------------------------------------------------------------------------------
    def __add__(self, other):
        other = _poly(other)
        out = dict(self.t)
        for k, v in other.t.items():
            out[k] = out.get(k, 0) + v
        return Poly(out)

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.t.items()})

    def __sub__(self, other):
        return self + (-_poly(other))

    def __rsub__(self, other):
        return _poly(other) + (-self)

    def __mul__(self, other):
        other = _poly(other)
        out: dict = {}
        for k1, v1 in self.t.items():
            for k2, v2 in other.t.items():
                k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
                out[k] = out.get(k, 0) + v1 * v2
        return Poly(out)

    __rmul__ = __mul__

    def __pow__(self, n: int):
        result = Poly.const(1)
        for _ in range(n):
            result = result * self
        return result

    def __eq__(self, other):
        return isinstance(other, Poly) and self.t == other.t

    def __hash__(self):
        return hash(frozenset(self.t.items()))

    # -- calculus ----------------------------------------------------------------------------------------------
    def diff(self, i: int) -> "Poly":
        out: dict = {}
        for k, v in self.t.items():
            if k[i]:
                e = list(k)
                e[i] -= 1
                out[tuple(e)] = out.get(tuple(e), 0) + v * k[i]
        return Poly(out)

    def antiderivative(self, i: int) -> "Poly":
        out: dict = {}
        for k, v in self.t.items():
            e = list(k)
            e[i] += 1
            out[tuple(e)] = v / e[i]
        return Poly(out)

    def compose(self, images: dict) -> "Poly":
        """Replace variable i by the polynomial images[i] (all at once). Variables not named stay."""
        total = Poly()
        cache: dict = {}
        for k, v in self.t.items():
            term = Poly.const(v)
            for i in range(3):
                if k[i]:
                    if i in images:
                        if (i, k[i]) not in cache:
                            cache[(i, k[i])] = images[i] ** k[i]
                        term = term * cache[(i, k[i])]
                    else:
                        e = [0, 0, 0]
                        e[i] = k[i]
                        term = term * Poly({tuple(e): 1})
            total = total + term
        return total

    def integrate(self, i: int, lo: "Poly", hi: "Poly") -> "Poly":
        """The exact integral over variable i from lo to hi (limits free of variable i)."""
        anti = self.antiderivative(i)
        return anti.compose({i: _poly(hi)}) - anti.compose({i: _poly(lo)})

    def value(self, point) -> Fraction:
        """Evaluate at a point (x, y, z); a missing coordinate counts as 0."""
        total = 0
        for k, v in self.t.items():
            term = v
            for i in range(3):
                if k[i]:
                    term = term * point[i] ** k[i]
            total += term
        return total

    def uses(self, i: int) -> bool:
        return any(k[i] for k in self.t)

    def degree(self, i: int) -> int:
        return max((k[i] for k in self.t), default=0)

    # -- text --------------------------------------------------------------------------------------------------
    def text(self, names: str = "xyz", style: str = "pretty") -> str:
        """The polynomial as written on a page. style 'pretty' uses superscripts (x²y); 'typed' uses ^ (x^2y)."""
        if not self.t:
            return "0"
        pieces = []
        for k in sorted(self.t, key=lambda e: (-sum(e), tuple(-x for x in e))):
            c = self.t[k]
            body = ""
            for i, power in enumerate(k):
                if power:
                    body += names[i]
                    if power > 1:
                        body += str(power).translate(_SUP) if style == "pretty" else f"^{power}"
            mag = abs(c)
            if not body:
                text = _frac_text(mag)
            elif mag == 1:
                text = body
            elif mag.denominator == 1:
                text = f"{mag.numerator}{body}"
            else:
                text = f"({mag.numerator}/{mag.denominator}){body}"
            pieces.append((c < 0, text))
        out = []
        for n, (neg, text) in enumerate(pieces):
            if n == 0:
                out.append(f"-{text}" if neg else text)
            else:
                out.append(f"{'-' if neg else '+'} {text}")
        return " ".join(out)

    def sympy(self, names: str = "xyz"):
        import sympy

        symbols = sympy.symbols(" ".join(names))
        if not isinstance(symbols, tuple):
            symbols = (symbols,)
        total = sympy.Integer(0)
        for k, v in self.t.items():
            term = sympy.Rational(v.numerator, v.denominator)
            for i, power in enumerate(k):
                if power:
                    term = term * symbols[i] ** power
            total += term
        return total


def _frac_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _poly(x) -> Poly:
    return x if isinstance(x, Poly) else Poly.const(x)


def gauss_nested(f, bounds, point=()) -> float:
    """Float integral of f(*coordinates) over the nested region bounds[k] = (lo, hi), each a `Poly` in the earlier
    coordinates (x, y, z in order). Six-point Gauss-Legendre in each variable."""
    k = len(point)
    if k == len(bounds):
        return f(*point)
    padded = point + (0.0,) * (3 - len(point))
    lo, hi = float(bounds[k][0].value(padded)), float(bounds[k][1].value(padded))
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    total = 0.0
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        total += weight * gauss_nested(f, bounds, point + (mid + half * node,))
    return total * half


def gauss_1d(f, lo: float, hi: float) -> float:
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return half * sum(w * f(mid + half * x) for x, w in zip(_GL_NODES, _GL_WEIGHTS))
