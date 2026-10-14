"""The polynomial interpolation error bound: `r4h_interpolation_error_bound` (easy, medium, hard) and the exam chain
`exd_x_r4h_interpolation_error_bound`.

``uc2-interpolation-error-bound`` (``numerical-and-computational-mathematics``). For an interpolating polynomial of
degree n through n + 1 nodes, ``|f(x) − p(x)| ≤ M/(n+1)! · |(x−x₀)⋯(x−xₙ)|`` where ``M`` bounds ``|f^(n+1)|`` on the
interval spanned by the nodes. Every function is one whose derivatives are known in closed form, and whose bound ``M``
is an exact rational number: ``sin kx`` and ``cos kx`` (``M = k^(n+1)``), ``ln x`` and ``1/x`` on an interval to the
right of zero (largest at the left end) and a power ``x^p`` (largest at the right end).

* easy: ``sin kx`` or ``cos kx``, three nodes, the bound on the error at a stated point (a fraction).
* medium: ``ln x``, ``1/x`` or ``x^p``, three or four nodes, the bound at a stated point, where the student has to
  find where the derivative is largest.
* hard: linear interpolation in a table: the fewest equal subintervals of the stated interval that guarantee the stated
  tolerance (a whole number).
* exam: three parts: the bound ``M``, the node product at the stated point, and the error bound built from them.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    fr,
    fraction_part,
    integer_part,
    finish_parts,
    ft_symbol,
    rat,
    single_from,
    with_redraw,
)

MISC = {
    "num2t-ieb-derivative-order-off-by-one": "Bounding the n-th derivative instead of the (n+1)-th when n + 1 nodes are used.",
    "num2t-ieb-factorial-omitted": "Leaving out the factor 1/(n+1)! in the interpolation error bound.",
    "num2t-ieb-wrong-factorial": "Dividing by n! instead of (n+1)! in the interpolation error bound.",
    "num2t-ieb-derivative-max-wrong-end": "Taking the largest value of the derivative at the wrong end of the interval, where it is smallest.",
    "num2t-ieb-derivative-factorial-dropped": "Dropping the factorial that comes from repeated differentiation, writing (-1)^n / x^(n+1) for the derivative of ln x.",
    "num2t-ieb-node-factor-missed": "Leaving one node out of the product (x − x₀)(x − x₁)⋯(x − xₙ).",
    "num2t-ieb-node-spacing-cubed": "Replacing the node product by the cube of the node spacing.",
    "num2t-ieb-max-product-wrong": "Taking the largest node product on a spacing h as h² instead of h²/4 for linear interpolation.",
    "num2t-ieb-half-factor-missed": "Bounding the linear interpolation error by M h²/4, forgetting the 1/8 in M h²/8.",
    "num2t-ieb-nodes-for-subintervals": "Giving the number of nodes (one more than the number of subintervals).",
    "num2t-ieb-tolerance-not-squared": "Solving for h instead of h², so the number of subintervals comes from the square root of the right quantity.",
    "num2t-ieb-derivative-order-too-high": "Bounding a derivative one order higher than the (n+1)-th.",
    "num2t-ieb-node-product-with-evaluation-point": "Multiplying the node product by the evaluation point as an extra factor.",
    "num2t-ieb-factorial-too-high": "Dividing by (n+2)! instead of (n+1)! in the interpolation error bound.",
    "num2t-ieb-derivative-at-wrong-order": "Using the first derivative of the function in place of the second when bounding linear interpolation.",
}
M_ORDER = "num2t-ieb-derivative-order-off-by-one"
M_NOFACT = "num2t-ieb-factorial-omitted"
M_WRONGFACT = "num2t-ieb-wrong-factorial"
M_WRONGEND = "num2t-ieb-derivative-max-wrong-end"
M_DFACT = "num2t-ieb-derivative-factorial-dropped"
M_NODE = "num2t-ieb-node-factor-missed"
M_SPACING = "num2t-ieb-node-spacing-cubed"
M_MAXPROD = "num2t-ieb-max-product-wrong"
M_HALF = "num2t-ieb-half-factor-missed"
M_NODES = "num2t-ieb-nodes-for-subintervals"
M_SQRT = "num2t-ieb-tolerance-not-squared"
M_D1 = "num2t-ieb-derivative-at-wrong-order"

SINE_NOTE = " (take any sine or cosine factor to be at most 1)"
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _t(v) -> str:
    """A node or point as text: whole numbers and halves, quarters as decimals."""
    v = fr(v)
    if v.denominator == 1:
        return str(v.numerator)
    return str(float(v)).rstrip("0")


def _fx(fam) -> str:
    kind, p = fam
    if kind == "sin":
        return "sin " + ("" if p == 1 else str(p)) + "x"
    if kind == "cos":
        return "cos " + ("" if p == 1 else str(p)) + "x"
    if kind == "ln":
        return "ln x"
    if kind == "inv":
        return "1/x"
    return f"x{str(p).translate(_SUP)}"


def deriv_bound(fam, m: int, a: Fraction, b: Fraction) -> Fraction:
    """A bound for |f^(m)| on [a, b] that is attained or is the standard one (|sin|, |cos| ≤ 1)."""
    kind, p = fam
    if kind in ("sin", "cos"):
        return Fraction(p) ** m
    if kind == "ln":
        return Fraction(math.factorial(m - 1)) / a**m
    if kind == "inv":
        return Fraction(math.factorial(m)) / a ** (m + 1)
    return Fraction(math.factorial(p), math.factorial(p - m)) * b ** (p - m)


def _wrong_end(fam, m, a, b):
    kind, p = fam
    if kind == "ln":
        return Fraction(math.factorial(m - 1)) / b**m
    if kind == "inv":
        return Fraction(math.factorial(m)) / b ** (m + 1)
    if kind == "pow":
        return Fraction(math.factorial(p), math.factorial(p - m)) * a ** (p - m)
    return None


def omega(nodes, x) -> Fraction:
    out = Fraction(1)
    for t in nodes:
        out *= Fraction(x) - t
    return out


def _nodes(rng, count, *, positive=True):
    start = Fraction(rng.choice([1, 1, 2, 3])) if positive else Fraction(rng.choice([0, 1, 2]))
    gap = rng.choice([Fraction(1, 2), Fraction(1), Fraction(1)])
    return [start + gap * k for k in range(count)], gap


def _point(rng, nodes, gap):
    """A point strictly inside the node interval and not a node."""
    for _ in attempts(200, "an interpolation point"):
        x = nodes[0] + gap * Fraction(rng.randint(1, 4 * (len(nodes) - 1) - 1), 4)
        if x not in nodes:
            return x
    raise AssertionError("unreachable")


def _fam_easy(rng):
    return (rng.choice(["sin", "cos"]), rng.choice([2, 3, 4]))


def _fam_medium(rng, n1):
    kind = rng.choice(["ln", "inv", "pow", "pow"])
    if kind == "pow":
        return ("pow", n1 + rng.choice([2, 3]))
    return (kind, 0)


def bound_at(fam, nodes, x):
    n1 = len(nodes)
    m = deriv_bound(fam, n1, nodes[0], nodes[-1])
    return m, abs(omega(nodes, x)), m * abs(omega(nodes, x)) / math.factorial(n1)


def _bound_candidates(fam, nodes, x, m, w, value):
    n1 = len(nodes)
    cands = [
        (M_NOFACT, m * w),
        (M_WRONGFACT, m * w / math.factorial(n1 - 1)),
        (M_ORDER, deriv_bound(fam, n1 - 1, nodes[0], nodes[-1]) * w / math.factorial(n1)),
    ]
    we = _wrong_end(fam, n1, nodes[0], nodes[-1])
    if we is not None:
        cands.append((M_WRONGEND, we * w / math.factorial(n1)))
    if fam[0] == "ln":
        cands.append((M_DFACT, w / nodes[0] ** n1 / math.factorial(n1)))
    short = Fraction(1)
    for t in nodes[:-1]:
        short *= abs(Fraction(x) - t)
    cands.append((M_NODE, m * short / math.factorial(n1)))
    gap = nodes[1] - nodes[0]
    cands.append((M_SPACING, m * gap**n1 / math.factorial(n1)))
    return cands


@generator(
    "r4h_interpolation_error_bound",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_interpolation_error_bound(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "an error-bound question with enough distractors")


def _single(rng, tier):
    if tier == "hard":
        return _hard(rng)
    count = 3 if tier == "easy" else rng.choice([3, 4])
    fam = _fam_easy(rng) if tier == "easy" else _fam_medium(rng, count)
    nodes, gap = _nodes(rng, count, positive=tier != "easy" or True)
    x = _point(rng, nodes, gap)
    m, w, value = bound_at(fam, nodes, x)
    cands = _bound_candidates(fam, nodes, x, m, w, value)
    part = fraction_part("", value, cands, 1, value_range=(0, None), min_options=3)
    n = count - 1
    names = ", ".join(_t(t) for t in nodes)
    note = SINE_NOTE if fam[0] in ("sin", "cos") else ""
    pt = _t(x)
    stem = rng.choice(
        [
            f"The function «f(x) = {_fx(fam)}» is approximated by the polynomial «p» of degree at most {n} that passes "
            f"through «f» at the nodes «x = {names}». Use the standard interpolation error formula, with the derivative "
            f"bounded by its largest size on the node interval{note}, to bound «|f({pt}) − p({pt})|». Give the bound "
            "as a fraction in lowest terms.",
            f"Interpolate «f(x) = {_fx(fam)}» at the nodes «x = {names}» with a polynomial «p» of degree at most {n}. "
            f"Bounding the derivative by its largest size on the node interval{note}, what bound does the interpolation "
            f"error formula give for «|f({pt}) − p({pt})|»? Answer as a fraction in lowest terms.",
            f"A polynomial «p» of degree at most {n} agrees with «f(x) = {_fx(fam)}» at «x = {names}». Take the "
            f"derivative bound «M» to be the largest size of the derivative on the node interval{note}. Find the "
            f"error bound for «|f({pt}) − p({pt})|» as a fraction in lowest terms.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"|f(x) − p(x)| ≤ M/{count}! · |(x − x₀)(x − x₁)⋯(x − x{n})| with M a bound for |f^({count})| on [{_t(nodes[0])}, {_t(nodes[-1])}].",
            f"M = {m} and the node product at x = {_t(x)} has size {w}, so the bound is {value}.",
        ],
    )


def _hard(rng):
    fam = rng.choice([("ln", 0), ("inv", 0), ("pow", 4), ("pow", 5), ("sin", 2), ("sin", 3), ("cos", 2)])
    a = Fraction(rng.choice([1, 2, 3]))
    width = Fraction(rng.choice([1, 2, 3, 4]))
    b = a + width
    m2 = deriv_bound(fam, 2, a, b)
    tol = Fraction(1, rng.choice([100, 200, 500, 1000, 2000, 10000]))
    need = m2 * width**2 / (8 * tol)
    n_sub = math.isqrt(need.numerator // need.denominator)
    while Fraction(n_sub) ** 2 < need:
        n_sub += 1
    check(Fraction(n_sub) ** 2 >= need and Fraction(n_sub - 1) ** 2 < need, "N is not the least")
    if Fraction(n_sub) ** 2 == need or n_sub < 4 or n_sub > 400:
        raise Redraw
    cands = [
        (M_MAXPROD, _least(m2 * width**2 / (2 * tol))),
        (M_HALF, _least(m2 * width**2 / (4 * tol))),
        (M_NODES, n_sub + 1),
        (M_SQRT, _least_lin(m2 * width**2 / (8 * tol))),
        (M_D1, _least(deriv_bound(fam, 1, a, b) * width**2 / (8 * tol))),
    ]
    if fam[0] in ("ln", "inv", "pow"):
        wrong = _wrong_end(fam, 2, a, b)
        cands.append((M_WRONGEND, _least(wrong * width**2 / (8 * tol))))
    part = integer_part("", n_sub, cands, 1)
    note = SINE_NOTE if fam[0] in ("sin", "cos") else ""
    tol_t = _t_dec(tol)
    stem = rng.choice(
        [
            f"A table of «f(x) = {_fx(fam)}» is to be built on the interval «[{_t(a)}, {_t(b)}]» with equally spaced "
            "entries, and values between entries are found by linear interpolation. "
            f"Using the standard error bound{note}, find the fewest equal subintervals that guarantee an error of at "
            f"most «{tol_t}» everywhere on the interval.",
            f"Values of «f(x) = {_fx(fam)}» for «x» in «[{_t(a)}, {_t(b)}]» are to be read from a table with equal "
            f"steps and linear interpolation between entries. By the standard error bound{note}, how many equal "
            f"subintervals are enough to keep the error within «{tol_t}», and what is the fewest?",
            f"Linear interpolation between equally spaced table entries is used for «f(x) = {_fx(fam)}» on "
            f"«[{_t(a)}, {_t(b)}]». Find the smallest number of equal subintervals for which the error bound{note} "
            f"is at most «{tol_t}».",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            "Linear interpolation on a subinterval of width h has error at most M h²/8, with M a bound for |f″|.",
            f"M = {m2}, so the width must satisfy h² ≤ {8 * tol / m2}; with h = {width}/N this needs N ≥ {_sqrt_text(need)}, so N = {n_sub}.",
        ],
    )


def _least(need: Fraction) -> int:
    n = max(1, math.isqrt(need.numerator // need.denominator))
    while Fraction(n) ** 2 < need:
        n += 1
    return n


def _least_lin(need: Fraction) -> int:
    n = max(1, need.numerator // need.denominator)
    while Fraction(n) < need:
        n += 1
    return n


def _sqrt_text(need: Fraction) -> str:
    return f"{math.sqrt(need):.2f}"


def _t_dec(tol: Fraction) -> str:
    return f"{float(tol):.10f}".rstrip("0")


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4h_interpolation_error_bound", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_interpolation_error_bound(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam error-bound question")


def _exam(rng):
    count = 3
    fam = _fam_medium(rng, count)
    nodes, gap = _nodes(rng, count)
    x = _point(rng, nodes, gap)
    m, w, value = bound_at(fam, nodes, x)
    n1 = count
    a_cands = [
        (M_WRONGEND, _wrong_end(fam, n1, nodes[0], nodes[-1])),
        (M_ORDER, deriv_bound(fam, n1 - 1, nodes[0], nodes[-1])),
        (M_DFACT, 1 / nodes[0] ** n1 if fam[0] == "ln" else None),
        ("num2t-ieb-derivative-order-too-high", deriv_bound(fam, n1 + 1, nodes[0], nodes[-1])),
    ]
    p_a = fraction_part(
        f"Find the least value of «M» that bounds «|f‴(t)|» for every «t» in «[{_t(nodes[0])}, {_t(nodes[-1])}]».",
        m,
        a_cands,
        1,
        working=[f"f‴ is largest in size at the {'right' if fam[0] == 'pow' else 'left'} end, giving M = {m}."],
        value_range=(0, None),
    )
    b_cands = [
        (M_NODE, _short(nodes, x)),
        (M_SPACING, (nodes[1] - nodes[0]) ** 3),
        ("num2t-ieb-node-product-with-evaluation-point", abs(omega(nodes, x) * x)),
    ]
    p_b = fraction_part(
        f"Find «|(x − {_t(nodes[0])})(x − {_t(nodes[1])})(x − {_t(nodes[2])})|» at «x = {_t(x)}».",
        w,
        b_cands,
        1,
        working=[f"|({_t(x)} − {_t(nodes[0])})({_t(x)} − {_t(nodes[1])})({_t(x)} − {_t(nodes[2])})| = {w}."],
        value_range=(0, None),
    )
    c_cands = [
        (M_NOFACT, m * w),
        (M_WRONGFACT, m * w / 2),
        ("num2t-ieb-factorial-too-high", m * w / 24),
    ]
    ft = str(rat(1) / 6 * ft_symbol("part_a") * ft_symbol("part_b"))
    p_c = fraction_part(
        f"Using your «M» and your node product, find the bound on «|f({_t(x)}) − p({_t(x)})|» from the interpolation error formula.",
        value,
        c_cands,
        2,
        ft=ft,
        working=[f"The bound is M/3! · {w} = {value}."],
        value_range=(0, None),
    )
    names = ", ".join(_t(t) for t in nodes)
    stem = rng.choice(
        [
            f"The function «f(x) = {_fx(fam)}» is approximated by the quadratic «p» that passes through «f» at the "
            f"nodes «x = {names}».",
            f"A quadratic «p» interpolates «f(x) = {_fx(fam)}» at the nodes «x = {names}».",
            f"Let «p» be the polynomial of degree at most 2 with «p(x) = f(x)» at «x = {names}», where «f(x) = {_fx(fam)}».",
        ]
    )
    return finish_parts([p_a, p_b, p_c], stem)


def _short(nodes, x):
    out = Fraction(1)
    for t in nodes[:-1]:
        out *= abs(Fraction(x) - t)
    return out
