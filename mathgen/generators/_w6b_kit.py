"""Small helpers shared by the ``w6b_*`` build modules: exact complex and surd text, tagged distractors, tier plumbing.

Nothing here draws random numbers, so a helper can never change which question a seed gives.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, pick_distractors

I = sympy.I


def dist(text, tag, value=None):
    """A tagged wrong option. ``value`` is what a reply with this error would be graded as (default: its own text)."""
    return Distractor(str(text), tag, str(text) if value is None else value)


def fdist(frac, tag):
    """A tagged wrong fraction option."""
    frac = Fraction(frac)
    text = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    return Distractor(text, tag, exact_value(frac))


class ThinOptions(VerificationError):
    """Too few distinct wrong options survived for this draw; `retry` redraws the question."""


def need(answer_text, candidates, n=3, minimum=2, **kw):
    """``pick_distractors`` that insists on ``minimum`` survivors, else raises `ThinOptions`."""
    out = pick_distractors(str(answer_text), list(candidates), n, **kw)
    if len(out) < minimum:
        raise ThinOptions(f"fewer than {minimum} distractors for {answer_text!r}")
    return out


def want(condition, message="redraw"):
    """Ask for a redraw (`retry`) unless ``condition`` holds: for properties of the random draw, not of the maths."""
    if not condition:
        raise ThinOptions(message)


def retry(rng, build, tries=40):
    """``build(rng)``, redrawn (same rng stream) while it reports `ThinOptions`; at most ``tries`` draws."""
    for _ in range(tries):
        try:
            return build(rng)
        except ThinOptions:
            continue
    raise VerificationError(f"no draw with enough distractors in {tries} tries")


def tier_of(tier, table):
    if tier not in table:
        raise ValueError(f"unknown tier {tier!r}")
    return table[tier]


def rat_text(r) -> str:
    """A rational as a student writes it: ``3``, ``-1/2``."""
    r = sympy.Rational(r)
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def _surd_term(coeff: sympy.Rational, radicand: int) -> str:
    sign = "-" if coeff < 0 else ""
    mag = abs(coeff)
    if mag.q == 1:
        head = "" if mag.p == 1 else str(mag.p)
        return f"{sign}{head}√{radicand}"
    head = "" if mag.p == 1 else str(mag.p)
    return f"{sign}{head}√{radicand}/{mag.q}"


def num_text(x) -> str:
    """An exact real number of the form ``q + r1√n1 + r2√n2`` as plain text: ``1 - 2√3``, ``√2/2``, ``-3/2``."""
    x = sympy.expand(sympy.nsimplify(x) if not isinstance(x, sympy.Basic) else x)
    rational = sympy.Integer(0)
    surds: list[tuple[int, sympy.Rational]] = []
    for term in sympy.Add.make_args(x):
        coeff, rest = term.as_coeff_Mul()
        if rest == 1:
            rational += coeff
            continue
        check(rest.is_Pow and rest.exp == sympy.Rational(1, 2) and rest.base.is_Integer, f"not a surd term: {term}")
        surds.append((int(rest.base), sympy.Rational(coeff)))
    surds.sort()
    pieces: list[str] = []
    if rational != 0 or not surds:
        pieces.append(rat_text(rational))
    for radicand, coeff in surds:
        text = _surd_term(coeff, radicand)
        if not pieces:
            pieces.append(text)
        elif text.startswith("-"):
            pieces.append(f"- {text[1:]}")
        else:
            pieces.append(f"+ {text}")
    return " ".join(pieces)


def cx_text(z) -> str:
    """A complex number ``a + bi`` as plain text: ``-2 + 3i``, ``1 - 2i``, ``-i``, ``2 + (√3/2)i``.

    A coefficient of ``i`` that is not a whole number is bracketed, so a reply reads one way only.
    """
    z = sympy.expand(z)
    re_, im_ = sympy.expand(sympy.re(z)), sympy.expand(sympy.im(z))
    if im_ == 0:
        return num_text(re_)
    if isinstance(im_, sympy.Add):
        # A sum of a whole number and a surd: keep its own signs inside the brackets, and add it.
        imag = f"({num_text(im_)})i"
        return imag if re_ == 0 else f"{num_text(re_)} + {imag}"
    negative = bool(im_.is_negative)
    mag = sympy.expand(-im_ if negative else im_)
    if mag == 1:
        imag = "i"
    elif mag.is_Integer:
        imag = f"{mag}i"
    else:
        imag = f"({num_text(mag)})i"
    if re_ == 0:
        return f"-{imag}" if negative else imag
    return f"{num_text(re_)} {'-' if negative else '+'} {imag}"


def plus(text: str) -> str:
    """``+ text``, bracketed when ``text`` starts with a minus sign, so working never reads ``+ -3``."""
    return f"+ ({text})" if text.startswith("-") else f"+ {text}"


def cis(deg: int):
    """Exact ``cos + i sin`` of a whole number of degrees."""
    rad = sympy.Rational(deg, 180) * sympy.pi
    return sympy.expand(sympy.cos(rad) + I * sympy.sin(rad))


def same_point(z, w) -> bool:
    return sympy.simplify(sympy.expand(z - w)) == 0


def cx_value(z) -> str:
    """The exact complex number as a SymPy string (``sqrt(3) + I``), the form the expression grader compares against."""
    return str(sympy.expand(z))


def cdist(z, tag):
    """A tagged complex-number option: shown as ``cx_text(z)``, graded as ``cx_value(z)``."""
    return dist(cx_text(z), tag, cx_value(z))


def gaussian(a: int, b: int):
    return sympy.Integer(a) + sympy.Integer(b) * I
