"""Generates `s2c1_variation_of_parameters` questions (easy, medium, hard) and the exam chain
`exd_x_s2c1_variation_of_parameters`.

``uc2-variation-of-parameters`` (``ordinary-differential-equations``, answer ``expression``): a particular solution of
``y'' + b y' + c y = g(x)`` where the forcing term is one undetermined coefficients cannot match (sec, csc, tan,
e^(ax)/x, ln x, 1/(1 + e^x), ...), found by variation of parameters with every constant of integration taken as 0.

* easy: ``y'' + k²y = A sec(kx)`` or ``A csc(kx)`` (the learner must find cos(kx), sin(kx) and W = k first).
* medium: a repeated root, ``y'' - 2ay' + a²y = A e^(ax)/x``, ``A e^(ax)/x²`` or ``A e^(ax) ln(x)`` (y1 = e^(ax),
  y2 = x e^(ax), W = e^(2ax)).
* hard: ``A tan(kx)``, ``A sec²(kx)``, ``A e^(ax)/(1 + x²)`` and two distinct real roots with a ``1/(1 + e^x)`` forcing.
* exam (``exd_x_s2c1_variation_of_parameters``): the two solutions of the homogeneous equation are given; (a) the
  Wronskian, (b) u1' or u2' (follows through from (a)), (c) the particular solution, (d) its value at a stated x
  (follows through from (c)).

Every template carries its Wronskian, u1', u2' and the integrated particular solution in closed form, and each is
re-checked numerically: W against y1 y2' - y1' y2, u1' and u2' against the variation-of-parameters formulas, and the
particular solution by substituting it back into the differential equation at three points inside its domain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, rounds_unambiguously
from ._s2c1_kit import (
    X,
    expr_options,
    readable,
    machine,
    number_options,
    plain,
    prompt_part,
    agree,
    same_function,
    say_marked,
    solves_ode,
)

x = X
_R = sympy.Rational


@dataclass(frozen=True)
class _Case:
    b: int  # y'' + b y' + c y = g
    c: int
    g: sympy.Expr
    g_text: str
    y1: sympy.Expr
    y2: sympy.Expr
    W: sympy.Expr
    u1: sympy.Expr  # integral of u1' with the constant taken as 0
    u2: sympy.Expr
    closed: sympy.Expr  # y_p in its simplest closed form
    domain: str  # "for 0 < x < π/4"
    points: tuple  # probe points inside the domain
    x0s: tuple  # evaluation points for the exam, inside the domain
    d1: sympy.Expr = None  # u1' in closed form
    d2: sympy.Expr = None  # u2' in closed form

    @property
    def u1p(self):
        return self.d1

    @property
    def u2p(self):
        return self.d2

    @property
    def yp(self):
        return self.closed


def _ode_lhs(b: int, c: int) -> str:
    return join_terms([(1, "y''"), (b, "y'"), (c, "y")])


def _coef(a: int, body: str) -> str:
    return body if a == 1 else f"{a}{body}"


def _exp_text(a: int) -> str:
    return "e^x" if a == 1 else ("e^(-x)" if a == -1 else f"e^({a}x)")


def _over_one_plus_exp(A: int, p: int) -> str:
    """A e^(px)/(1 + e^x) in the form a learner reads."""
    top = "" if p == 0 else _exp_text(p)
    if not top:
        return f"{A}/(1 + e^x)" if A != 1 else "1/(1 + e^x)"
    return f"{_coef(A, top)}/(1 + e^x)"


def _kx(k: int) -> str:
    return "x" if k == 1 else f"{k}x"


def _inside(k: float, upper: float):
    """Three rational probe points strictly inside (0, upper / k)."""
    return tuple(_R(int(round(f * 100)), 100 * k) for f in (0.2 * upper, 0.5 * upper, 0.8 * upper))


# ---------------------------------------------------------------------------------------------------------------
# The templates. Each returns a `_Case`; `A` multiplies the forcing term.


def _sec(k: int, A: int) -> _Case:
    kx = k * x
    return _Case(
        0, k * k, A * sympy.sec(kx), _coef(A, f"sec({_kx(k)})"), sympy.cos(kx), sympy.sin(kx), _R(k),
        A * sympy.log(sympy.cos(kx)) / k**2, A * x / k,
        sympy.expand(A * (sympy.cos(kx) * sympy.log(sympy.cos(kx)) / k**2 + x * sympy.sin(kx) / k)),
        f"for 0 < x < π/{2 * k}" if k > 1 else "for 0 < x < π/2",
        _inside(k, 1.5), (_R(2, 10), _R(3, 10), _R(4, 10)),
        d1=-A * sympy.tan(kx) / k, d2=_R(A, k),
    )  # fmt: skip


def _csc(k: int, A: int) -> _Case:
    kx = k * x
    return _Case(
        0, k * k, A * sympy.csc(kx), _coef(A, f"csc({_kx(k)})"), sympy.cos(kx), sympy.sin(kx), _R(k),
        -A * x / k, A * sympy.log(sympy.sin(kx)) / k**2,
        sympy.expand(A * (-x * sympy.cos(kx) / k + sympy.sin(kx) * sympy.log(sympy.sin(kx)) / k**2)),
        f"for 0 < x < π/{k}" if k > 1 else "for 0 < x < π",
        _inside(k, 3.0), (_R(2, 10), _R(3, 10), _R(4, 10)),
        d1=_R(-A, k), d2=A * sympy.cot(kx) / k,
    )  # fmt: skip


def _tan(k: int, A: int) -> _Case:
    kx = k * x
    # u1 = -(ln(sec + tan) - sin)/k^2 and u2 = -cos/k^2, so that y_p = -cos(kx) ln(sec(kx) + tan(kx))/k^2
    return _Case(
        0, k * k, A * sympy.tan(kx), _coef(A, f"tan({_kx(k)})"), sympy.cos(kx), sympy.sin(kx), _R(k),
        -A * (sympy.log(sympy.sec(kx) + sympy.tan(kx)) - sympy.sin(kx)) / k**2, -A * sympy.cos(kx) / k**2,
        -A * sympy.cos(kx) * sympy.log(sympy.sec(kx) + sympy.tan(kx)) / k**2,
        f"for 0 < x < π/{2 * k}" if k > 1 else "for 0 < x < π/2",
        _inside(k, 1.5), (_R(2, 10), _R(3, 10), _R(4, 10)),
        d1=-A * sympy.sin(kx) * sympy.tan(kx) / k, d2=A * sympy.sin(kx) / k,
    )  # fmt: skip


def _sec2(k: int, A: int) -> _Case:
    kx = k * x
    return _Case(
        0, k * k, A * sympy.sec(kx) ** 2, _coef(A, f"sec^2({_kx(k)})"), sympy.cos(kx), sympy.sin(kx), _R(k),
        -A * sympy.sec(kx) / k**2, A * sympy.log(sympy.sec(kx) + sympy.tan(kx)) / k**2,
        sympy.expand(A * (-1 + sympy.sin(kx) * sympy.log(sympy.sec(kx) + sympy.tan(kx))) / k**2),
        f"for 0 < x < π/{2 * k}" if k > 1 else "for 0 < x < π/2",
        _inside(k, 1.5), (_R(2, 10), _R(3, 10), _R(4, 10)),
        d1=-A * sympy.tan(kx) * sympy.sec(kx) / k, d2=A * sympy.sec(kx) / k,
    )  # fmt: skip


def _repeated(a: int, kind: str, A: int) -> _Case:
    e = sympy.exp(a * x)
    common = dict(b=-2 * a, c=a * a, y1=e, y2=x * e, W=sympy.exp(2 * a * x))
    pts = (_R(7, 10), _R(13, 10), _R(19, 10))
    x0s = (_R(3, 2), _R(2), _R(5, 2))
    if kind == "inv":
        return _Case(g=A * e / x, g_text=_coef(A, f"{_exp_text(a)}/x"), u1=-A * x, u2=A * sympy.log(x),
                     closed=sympy.expand(A * x * e * (sympy.log(x) - 1)), domain="for x > 0", points=pts, x0s=x0s, d1=_R(-A), d2=A / x, **common)  # fmt: skip
    if kind == "inv2":
        return _Case(g=A * e / x**2, g_text=_coef(A, f"{_exp_text(a)}/x^2"), u1=-A * sympy.log(x), u2=-A / x,
                     closed=sympy.expand(-A * e * (sympy.log(x) + 1)), domain="for x > 0", points=pts, x0s=x0s, d1=-A / x, d2=A / x**2, **common)  # fmt: skip
    if kind == "log":
        return _Case(g=A * e * sympy.log(x), g_text=_coef(A, f"{_exp_text(a)} ln(x)"),
                     u1=A * (-(x**2) * sympy.log(x) / 2 + x**2 / 4), u2=A * (x * sympy.log(x) - x),
                     closed=sympy.expand(A * x**2 * e * (2 * sympy.log(x) - 3) / 4), domain="for x > 0", points=pts, x0s=x0s, d1=-A * x * sympy.log(x), d2=A * sympy.log(x), **common)  # fmt: skip
    check(kind == "atan", f"unknown repeated-root template {kind!r}")
    return _Case(g=A * e / (1 + x**2), g_text=_coef(A, f"{_exp_text(a)}/(1 + x^2)"),
                 u1=-A * sympy.log(1 + x**2) / 2, u2=A * sympy.atan(x),
                 closed=sympy.expand(A * e * (x * sympy.atan(x) - sympy.log(1 + x**2) / 2)), domain="", points=pts, x0s=x0s, d1=-A * x / (1 + x**2), d2=A / (1 + x**2), **common)  # fmt: skip


def _distinct(m: int, A: int) -> _Case:
    """y'' - (2m + 1)y' + m(m + 1)y = A e^((m+1)x)/(1 + e^x); the roots are m and m + 1."""
    y1, y2 = sympy.exp(m * x), sympy.exp((m + 1) * x)
    L = sympy.log(1 + sympy.exp(x))
    return _Case(
        -(2 * m + 1), m * (m + 1), A * y2 / (1 + sympy.exp(x)), _over_one_plus_exp(A, m + 1),
        y1, y2, sympy.exp((2 * m + 1) * x), -A * L, A * (x - L),
        sympy.expand(A * y1 * (sympy.exp(x) * x - (1 + sympy.exp(x)) * L)),
        "", (_R(-1, 2), _R(1, 2), _R(3, 2)), (_R(1, 2), _R(1), _R(3, 2)),
        d1=-A * sympy.exp(x) / (1 + sympy.exp(x)), d2=A / (1 + sympy.exp(x)),
    )  # fmt: skip


def _distinct_neg(A: int) -> _Case:
    """y'' + 3y' + 2y = A/(1 + e^x), the classic: y1 = e^(-x), y2 = e^(-2x)."""
    y1, y2 = sympy.exp(-x), sympy.exp(-2 * x)
    L = sympy.log(1 + sympy.exp(x))
    return _Case(
        3, 2, A / (1 + sympy.exp(x)), _over_one_plus_exp(A, 0),
        y1, y2, -sympy.exp(-3 * x), A * L, A * (L - sympy.exp(x)),
        sympy.expand(A * y2 * ((1 + sympy.exp(x)) * L - sympy.exp(x))),
        "", (_R(-1, 2), _R(1, 2), _R(3, 2)), (_R(1, 2), _R(1), _R(3, 2)),
        d1=A * sympy.exp(x) / (1 + sympy.exp(x)), d2=-A * sympy.exp(2 * x) / (1 + sympy.exp(x)),
    )  # fmt: skip


def _build_case(case: _Case) -> _Case:
    """Check a template against the defining formulas, each by a route that does not use the template's own claim."""
    pts = case.points
    wronskian = case.y1 * sympy.diff(case.y2, x) - sympy.diff(case.y1, x) * case.y2
    check(agree(wronskian, case.W, x, pts), "W is not y1 y2' - y1' y2")
    check(agree(case.d1, -case.y2 * case.g / case.W, x, pts), "u1' is not -y2 g/W")
    check(agree(case.d2, case.y1 * case.g / case.W, x, pts), "u2' is not y1 g/W")
    check(agree(sympy.diff(case.u1, x), -case.y2 * case.g / case.W, x, pts), "u1' is not the derivative of u1")
    check(agree(sympy.diff(case.u2, x), case.y1 * case.g / case.W, x, pts), "u2' is not the derivative of u2")
    for y in (case.y1, case.y2):
        check(solves_ode(y, (1, case.b, case.c), 0, x, pts), "a homogeneous solution fails the equation")
    check(agree(case.u1 * case.y1 + case.u2 * case.y2, case.yp, x, pts), "y_p is not u1 y1 + u2 y2")
    check(solves_ode(case.yp, (1, case.b, case.c), case.g, x, pts), "y_p does not satisfy the equation")
    return case


_EASY = ("sec", "csc")
_MEDIUM = ("inv", "inv2", "log")


def _reciprocal(case: _Case, name: str, k: int, A: int) -> _Case:
    """The same forcing term written as ``A/cos(kx)`` or ``A/sin(kx)`` instead of with sec or csc."""
    return replace(case, g_text=f"{A}/{name}({_kx(k)})")


def _draw(rng, tier: str) -> _Case:
    A = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    if tier == "easy":
        k = rng.choice([1, 2, 3, 4, 5])
        sec = rng.random() < 0.5
        case = _build_case((_sec if sec else _csc)(k, A))
        return _reciprocal(case, "cos" if sec else "sin", k, A) if rng.random() < 0.5 else case
    if tier == "medium":
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        return _build_case(_repeated(a, rng.choice(_MEDIUM), A))
    pick = rng.random()
    if pick < 0.25:
        return _build_case(_tan(rng.choice([1, 2, 3, 4]), A))
    if pick < 0.45:
        return _build_case(_sec2(rng.choice([1, 2, 3, 4]), A))
    if pick < 0.6:
        return _build_case(_repeated(rng.choice([-3, -2, -1, 1, 2, 3]), "atan", A))
    if pick < 0.85:
        return _build_case(_distinct(rng.choice([-3, -2, -1, 0, 1, 2, 3]), A))
    return _build_case(_distinct_neg(A))


def _exam_draw(rng) -> _Case:
    return _draw(rng, rng.choice(["easy", "medium", "hard"]))


# ---------------------------------------------------------------------------------------------------------------
# Wrong answers


def _wrong_particular(case: _Case):
    u1, u2, y1, y2 = case.u1, case.u2, case.y1, case.y2
    wrongs = [
        ("vop-sign-on-first-term", -u1 * y1 + u2 * y2),
        ("vop-swapped-solutions", u1 * y2 + u2 * y1),
        ("vop-wronskian-reversed", -case.yp),
        ("vop-sign-on-second-term", u1 * y1 - u2 * y2),
        ("vop-one-term-only", u2 * y2),
    ]
    if case.W.is_number and case.W != 1:
        wrongs.append(("vop-wronskian-forgotten", case.W * case.yp))
    return [(tag, sympy.expand(w)) for tag, w in wrongs]


@generator(
    "s2c1_variation_of_parameters",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2c1_variation_of_parameters(rng, tier="easy"):
    for _ in attempts(30, "a variation-of-parameters question whose answer the grader can read"):
        case = _draw(rng, tier)
        yp = case.yp
        yp_text = readable(yp)
        if yp_text is not None:
            break
    domain = f" {case.domain}" if case.domain else ""
    template = rng.choice(
        [
            "Find a particular solution of «{ode} = {g}»{dom} by variation of parameters, taking every constant of "
            "integration to be 0.",
            "Use variation of parameters to find the particular solution «y_p(x)» of «{ode} = {g}»{dom}. Take every "
            "constant of integration to be 0.",
            "The differential equation «{ode} = {g}»{dom} has a particular solution that variation of parameters "
            "gives when every constant of integration is taken to be 0. Find it.",
        ]
    )
    stem, stem_latex = say_marked(template.format(ode=_ode_lhs(case.b, case.c), g=case.g_text, dom=domain))
    distractors = expr_options(yp, _wrong_particular(case), 4)
    check(len(distractors) >= 2, "too few variation-of-parameters distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        answer=machine(yp),
        answer_text=yp_text,
        answer_kind="expression",
        distractors=distractors,
        working=[
            f"The homogeneous solutions are y1 = {plain(case.y1)} and y2 = {plain(case.y2)}, with Wronskian "
            f"W = {plain(case.W)}.",
            f"u1' = -y2 g/W and u2' = y1 g/W integrate to u1 = {plain(case.u1)} and u2 = {plain(case.u2)}, "
            f"so y_p = u1 y1 + u2 y2 = {yp_text}.",
        ],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


class _Redraw(Exception):
    pass


def _value_at(expr, point: Fraction) -> float:
    try:
        return float(sympy.sympify(expr).subs(x, sympy.Rational(point.numerator, point.denominator)).evalf(30))
    except TypeError:  # outside the real domain of the particular solution
        raise _Redraw from None


@generator("exd_x_s2c1_variation_of_parameters", level=22, presets={"exam": dict(tier="exam")})
def exd_x_s2c1_variation_of_parameters(rng, tier="exam"):
    """Exam tier: variation of parameters in four connected parts."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(40, "an exam variation-of-parameters question with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _exam_once(rng):
    case = _exam_draw(rng)
    yp, y1, y2, W, g = case.yp, case.y1, case.y2, case.W, case.g
    x0 = rng.choice(case.x0s)
    value = _value_at(yp, x0)
    if not (0.01 <= abs(value) <= 500 and rounds_unambiguously(value, 3, 0.1)):
        raise _Redraw

    # (a) the Wronskian
    w_wrong = [
        ("vop-wronskian-reversed", -W),
        ("vop-wronskian-sum", y1 * sympy.diff(y2, x) + sympy.diff(y1, x) * y2),
        ("vop-wronskian-wrong-arrangement", y1 * y2 - sympy.diff(y1, x) * sympy.diff(y2, x)),
        ("vop-wronskian-no-derivative", y1 * sympy.diff(y2, x) - y1 * y2),
    ]
    a_dist = expr_options(W, w_wrong, 4, simplify=True)

    # (b) u1' (or u2')
    first = rng.random() < 0.5
    if first:
        target, label, formula = case.u1p, "u1'", -y2 * g
        b_wrong = [
            ("vop-sign-on-first-term", -target),
            ("vop-swapped-solutions", case.u2p),
            ("vop-wronskian-forgotten", formula),
            ("vop-wronskian-multiplied", formula * W),
        ]
    else:
        target, label, formula = case.u2p, "u2'", y1 * g
        b_wrong = [
            ("vop-sign-on-first-term", -target),
            ("vop-swapped-solutions", case.u1p),
            ("vop-wronskian-forgotten", formula),
            ("vop-wronskian-multiplied", formula * W),
        ]
    b_dist = expr_options(target, b_wrong, 4, simplify=True)

    # (c) the particular solution
    c_dist = expr_options(yp, _wrong_particular(case), 4)

    # (d) its value at x0
    d_text = f"{value:.3f}"
    alt = []
    for d in c_dist:
        alt.append((d.misconception, _value_at(sympy.sympify(d.value), x0)))
    d_dist = number_options(value, 3, alt, 4)
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw
    from ..grading.parts import reply_value

    # Part (d) follows through from (c): its reply must be readable as a value, which the parser
    # declines for a very long expression.
    yp_text = readable(yp)
    if yp_text is None or reply_value("expression", plain(W)) is None:
        raise _Redraw

    x0_text = f"{float(x0):g}"
    x0_ft = str(x0.numerator) if x0.denominator == 1 else f"{x0.numerator}/{x0.denominator}"
    domain = f" {case.domain}" if case.domain else ""
    stem, stem_latex = say_marked(
        f"The equation «{_ode_lhs(case.b, case.c)} = {case.g_text}»{domain} has the homogeneous solutions "
        f"«y1 = {plain(y1)}» and «y2 = {plain(y2)}»."
    )
    sign = "-" if first else ""
    ft_b = f"{sign}({machine(y2 if first else y1)})*({machine(g)})/Subs(part_a, (e,), (E,))"
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            dict(
                **prompt_part("Find the Wronskian «W = y1 y2' - y1' y2» of y1 and y2."),
                answer_kind="expression",
                answer=machine(W),
                answer_text=plain(W),
                marks=2,
                working=[f"W = {plain(W)}."],
                distractors=a_dist,
            ),
            dict(
                **prompt_part(
                    "For variation of parameters write the particular solution as «y_p = u1 y1 + u2 y2». "
                    "Find «{label}».",
                    label=label,
                ),
                answer_kind="expression",
                answer=machine(target),
                answer_text=plain(target),
                marks=2,
                follow_through=ft_b,
                working=[f"{label} = {'-y2 g' if first else 'y1 g'}/W = {plain(target)}."],
                distractors=b_dist,
            ),
            dict(
                **prompt_part("Integrate, taking every constant of integration to be 0, to find «y_p(x)»."),
                answer_kind="expression",
                answer=machine(yp),
                answer_text=yp_text,
                marks=3,
                working=[f"u1 = {plain(case.u1)}, u2 = {plain(case.u2)}, so y_p = {yp_text}."],
                distractors=c_dist,
            ),
            dict(
                **prompt_part("Find «y_p({x0})», correct to 3 decimal places.", x0=x0_text),
                answer_kind="number",
                answer=round(value, 3),
                answer_text=d_text,
                marks=1,
                follow_through=f"Subs(part_c, (x, e), ({x0_ft}, E))",
                working=[f"y_p({x0_text}) = {d_text}."],
                distractors=d_dist,
            ),
        ],
    )
