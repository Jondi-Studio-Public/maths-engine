"""A one-pole contour integral, and the Cauchy-Riemann check.

- ``p15e_contour_integral_residue`` (``ucv1-contour-integral-residue``,
  ``answer: expression``) -- evaluate a positively oriented contour
  integral that encloses exactly one simple pole of
  f(z) = c / (z^2 + Bz + C), via the residue theorem
  (integral = 2*pi*i*Res(f, pole)). Every design keeps B = -2p, C = p^2+k^2
  so the two poles are p +/- ik: a CONJUGATE pair, so the enclosed
  residue is always PURELY IMAGINARY and 2*pi*i*Res is always REAL --
  grader defect: the graded key never needs a literal `I`.
  Construction computes the residue via the CLOSED-FORM quotient-rule
  formula Res = c / g'(z0) (g(z) = z^2+Bz+C) -- no symbolic verification
  runs inside `make()`, which is the hot path
  (per-topic test sweeps call it on the order of a thousand times per generator,
  so a slow `make()` taxes every test run; see `scripts/check_generator_speed.py`). The independent
  route -- re-deriving the residue via the direct limit definition
  ``lim_{z->z0} (z-z0) f(z)`` with `sympy.limit`, a genuinely different and
  much slower algorithm -- lives ENTIRELY in
  ``tests/test_p15e_complex_oracle.py``, run once per suite (60 seeds/tier)
  rather than on every `make()` call; see that file's own docstring.
- ``p15e_cr_equations_check`` (``ucv1-cr-equations-check``, ``answer:
  mcq``) -- at a stated point, decide whether a complex function
  u(x,y) + i v(x,y) satisfies the Cauchy-Riemann equations (u_x = v_y and
  u_y = -v_x). All four classes drawn; the independent
  check in the oracle test recomputes every partial derivative with SymPy
  from the prompt text alone, never touching this module.

Brand new module; no existing generator is edited.
"""

from __future__ import annotations

import sympy
from sympy import I, diff, im, re, sin, symbols

from ..core import (
    check,
    generator,
    join_terms,
    phrase,
    pick_distractors,
)

_MAX_ATTEMPTS = 500
_Z = symbols("z")
_X, _Y = symbols("x y", real=True)


def _fmt(expr) -> str:
    """Display only -- every caller already passes a fully `sympy.expand`-ed
    value, so this only needs to stringify it, not re-simplify it.
    `sympy.simplify` was measured costing most of a CR exam draw's ~220 ms
    here (`trigsimp`, on the tier's own sin/cosh/sinh values) for zero
    display benefit: `expand` is idempotent on an already-expanded
    expression, so the string is unchanged for every non-trig value this
    module ever formats, and for the trig ones it is dramatically cheaper."""
    return str(sympy.expand(expr)).replace("**", "^")


def _text_fmt(expr) -> str:
    return str(sympy.expand(expr)).replace("**", "^").replace("*", "·")


def _eq(a, b) -> bool:
    """Exact equality for the plain rational/pi/I combinations this module's
    values always are -- `sympy.expand` alone (no `cancel`/`gcd_terms`)
    settles it, since none of these values is an unevaluated fraction with a
    symbolic denominator. Replaces `symbolically_equal`'s general-purpose
    (and, for a bare constant with no free symbols, unconditionally
    `simplify`-based -- see its docstring) route in this module's hot loop:
    profiling a batch of `make()` calls showed `symbolically_equal` costing
    ~15 ms of a ~840 ms exam-tier draw, essentially all inside
    `sympy.cancel`, which a plain `expand` here never touches."""
    return sympy.expand(a - b) == 0


# ---------------------------------------------------------------------------
# Skill 1: ucv1-contour-integral-residue
# ---------------------------------------------------------------------------

_C_POOL = tuple(v for v in range(-9, 10) if v != 0)
_K_POOL = tuple(range(1, 7))
_P_POOL = tuple(range(-6, 7))
_P_NONZERO_POOL = tuple(v for v in range(-6, 7) if v != 0)


def _quotient_residue(c, b_coef, z0):
    """Closed-form residue at a simple pole of c / (z^2 + Bz + C): the
    quotient rule Res = c / g'(z0), g(z) = z^2 + Bz + C. No symbolic
    verification here -- see the module docstring; the oracle test
    re-derives this same number via the direct limit definition instead."""
    g_prime = 2 * z0 + b_coef
    return sympy.expand(c / g_prime)


def _pole_text(k) -> str:
    """ "i" for k=1, "5i" for k=5 -- never "1i"."""
    return "i" if k == 1 else f"{k}i"


def _full_pole_text(p, k) -> str:
    """ "i" (p=0,k=1), "3i" (p=0,k=3), "5 + 2i" (p=5,k=2) -- never "0 + 1i"."""
    return _pole_text(k) if p == 0 else f"{p} + {_pole_text(k)}"


def _shift_text(p) -> str:
    """ "z" (p=0), "(z - 5)" (p=5), "(z + 3)" (p=-3) -- never "(z - (5))"."""
    if p == 0:
        return "z"
    return f"(z - {p})" if p > 0 else f"(z + {-p})"


def _quad_text(b_coef, c_coef) -> str:
    """ "z^2 + bz + c" via `join_terms`, so a zero `b_coef` drops its term
    ("z^2 + 1", never "z^2 + (0)z + (1)") and a negative coefficient reads
    as "- 4z" rather than "+ (-4)z"."""
    return join_terms([(1, "z^2"), (b_coef, "z"), (c_coef, "")])


def _offer(answer, value, offered, formatted, label):
    if _eq(value, answer) or any(_eq(value, v) for v in offered):
        return
    offered.append(value)
    formatted.append(_fmt(value))


_EASY_PATTERNS = [
    "Let f(z) = {c} / (z^2 + {kk}). Let C be a positively oriented simple "
    "closed contour that encloses the pole z = {k} but not z = {kc}. "
    "Evaluate the contour integral of f(z) around C using the residue "
    "theorem, integral = 2·pi·i·Res(f, {k}).",
    "The function f(z) = {c} / (z^2 + {kk}) has simple poles at z = "
    "+/-{kbare}. C is a positively oriented simple closed contour enclosing "
    "z = {k} only. Find the contour integral of f(z) around C via "
    "integral = 2·pi·i·Res(f, {k}).",
    "Given f(z) = {c} / (z^2 + {kk}), and a positively oriented simple "
    "closed contour C enclosing only the pole z = {k}, compute the "
    "contour integral of f(z) around C, using integral = 2·pi·i·Res(f, {k}).",
]

_MEDIUM_PATTERNS = [
    "Let f(z) = {c} / ({shift}^2 + {kk}). Let C be a positively "
    "oriented simple closed contour that encloses the pole z = {pole} "
    "but not z = {conj}. Evaluate the contour integral of f(z) around "
    "C using the residue theorem.",
    "The function f(z) = {c} / ({shift}^2 + {kk}) has simple poles at "
    "z = {p} +/- {kbare}. C is a positively oriented simple closed contour "
    "enclosing z = {pole} only. Find the contour integral of f(z) "
    "around C.",
    "Given f(z) = {c} / ({shift}^2 + {kk}), and a positively oriented "
    "simple closed contour C enclosing only the pole z = {pole}, "
    "compute the contour integral of f(z) around C.",
]

_HARD_PATTERNS = [
    "Let f(z) = {c} / ({quad}). C is a positively "
    "oriented simple closed contour enclosing only the pole z = {pole} "
    "(not its conjugate). First verify by substitution that this z "
    "satisfies {quad} = 0, then evaluate the "
    "contour integral of f(z) around C using the residue theorem.",
    "The function f(z) = {c} / ({quad}) has a "
    "conjugate pair of simple poles. C is a positively oriented simple "
    "closed contour enclosing only z = {pole}. Check that this z is a "
    "root of {quad} = 0, then find the contour "
    "integral of f(z) around C.",
    "Given f(z) = {c} / ({quad}), and C a "
    "positively oriented simple closed contour enclosing only z = "
    "{pole}, verify this z solves {quad} = 0, then "
    "compute the contour integral of f(z) around C via the residue "
    "theorem.",
]

_EXAM_PATTERNS = [
    "Let f(z) = {c} / ({quad}). C is a positively oriented simple closed "
    "contour lying entirely in the upper half-plane (Im(z) > 0) and "
    "enclosing the one pole of f there. Evaluate the contour integral of "
    "f(z) around C.",
    "The function f(z) = {c} / ({quad}) has two simple poles; C is a "
    "positively oriented simple closed contour lying entirely in the upper "
    "half-plane and enclosing the one pole there. Compute the contour "
    "integral of f(z) around C.",
    "Given f(z) = {c} / ({quad}), and C a positively oriented simple "
    "closed contour lying entirely in the upper half-plane enclosing the "
    "one pole of f there, evaluate the contour integral of f(z) around C.",
]


def _p15e_contour_easy(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        k = rng.choice(_K_POOL)
        z0 = I * k
        residue = _quotient_residue(c, 0, z0)
        answer = sympy.expand(2 * sympy.pi * I * residue)
        check(im(answer) == 0, "easy: contour integral is not real")

        offered, formatted = [], []
        _offer(answer, residue, offered, formatted, "forgets-2-pi-i")
        _offer(answer, -answer, offered, formatted, "sign-flip")
        _offer(answer, sympy.expand(2 * sympy.pi * I * (c / z0)), offered, formatted, "drops-the-2-in-derivative")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        pole = _pole_text(k)
        text = phrase(rng, _EASY_PATTERNS, c=c, kk=k * k, k=pole, kc=f"-{pole}", kbare=pole)
        working = [
            f"Res(f, {pole}) = {c} / (2 × {pole}) = {_text_fmt(residue)}.",
            f"integral = 2πi × Res(f, {pole}) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p15e_contour_medium(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        k = rng.choice(_K_POOL)
        p = rng.choice(_P_NONZERO_POOL)
        b_coef = -2 * p
        c_coef = p * p + k * k
        z0 = p + I * k
        check(sympy.expand(z0**2 + b_coef * z0 + c_coef) == 0, "medium: pole fails its own characteristic equation")
        residue = _quotient_residue(c, b_coef, z0)
        answer = sympy.expand(2 * sympy.pi * I * residue)
        check(im(answer) == 0, "medium: contour integral is not real")

        offered, formatted = [], []
        _offer(answer, residue, offered, formatted, "forgets-2-pi-i")
        _offer(answer, -answer, offered, formatted, "sign-flip")
        other_pole = p - I * k
        _offer(
            answer,
            sympy.expand(2 * sympy.pi * I * _quotient_residue(c, b_coef, other_pole)),
            offered,
            formatted,
            "wrong-pole",
        )
        _offer(answer, sympy.expand(2 * sympy.pi * I * (c / z0)), offered, formatted, "drops-the-2-in-derivative")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        pole = _full_pole_text(p, k)
        conj = f"{p} - {_pole_text(k)}"
        text = phrase(
            rng, _MEDIUM_PATTERNS, c=c, shift=_shift_text(p), kk=k * k, p=p, pole=pole, conj=conj, kbare=_pole_text(k)
        )
        working = [
            f"Res(f, {pole}) = {c} / (2({pole}) - 2({p})) = {_text_fmt(residue)}.",
            f"integral = 2πi × Res(f, {pole}) = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p15e_contour_hard(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        k = rng.choice(_K_POOL)
        p = rng.choice(_P_POOL)
        b_coef = -2 * p
        c_coef = p * p + k * k
        z0 = p + I * k
        check(sympy.expand(z0**2 + b_coef * z0 + c_coef) == 0, "hard: pole fails its own characteristic equation")
        residue = _quotient_residue(c, b_coef, z0)
        answer = sympy.expand(2 * sympy.pi * I * residue)
        check(im(answer) == 0, "hard: contour integral is not real")

        offered, formatted = [], []
        _offer(answer, residue, offered, formatted, "forgets-2-pi-i")
        _offer(answer, -answer, offered, formatted, "sign-flip")
        other_pole = p - I * k
        _offer(
            answer,
            sympy.expand(2 * sympy.pi * I * _quotient_residue(c, b_coef, other_pole)),
            offered,
            formatted,
            "wrong-pole",
        )
        _offer(answer, sympy.expand(2 * sympy.pi * I * (c / z0)), offered, formatted, "drops-the-2-in-derivative")

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        quad = _quad_text(b_coef, c_coef)
        pole = _full_pole_text(p, k)
        text = phrase(rng, _HARD_PATTERNS, c=c, quad=quad, pole=pole)
        working = [
            f"({pole})^2 + ({b_coef})({pole}) + ({c_coef}) = 0, confirmed by substitution.",
            f"Res(f, {pole}) = {_text_fmt(residue)}; integral = 2πi × Res = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("hard: could not draw a usable question")


def _p15e_contour_exam(rng):
    for _ in range(_MAX_ATTEMPTS):
        c = rng.choice(_C_POOL)
        k = rng.choice(_K_POOL)
        p = rng.choice(_P_POOL)
        b_coef = -2 * p
        c_coef = p * p + k * k
        z0 = p + I * k  # the root with Im(z) > 0 -- k is always drawn > 0
        check(sympy.expand(z0**2 + b_coef * z0 + c_coef) == 0, "exam: pole fails its own characteristic equation")
        residue = _quotient_residue(c, b_coef, z0)
        answer = sympy.expand(2 * sympy.pi * I * residue)
        check(im(answer) == 0, "exam: contour integral is not real")

        offered, formatted = [], []
        _offer(answer, residue, offered, formatted, "forgets-2-pi-i")
        _offer(answer, -answer, offered, formatted, "sign-flip")
        other_pole = p - I * k
        _offer(
            answer,
            sympy.expand(2 * sympy.pi * I * _quotient_residue(c, b_coef, other_pole)),
            offered,
            formatted,
            "wrong-pole",
        )
        forgets_halving = 2 * p + 2 * I * k  # the quadratic formula's numerator, not divided by 2
        _offer(
            answer,
            sympy.expand(2 * sympy.pi * I * _quotient_residue(c, b_coef, forgets_halving)),
            offered,
            formatted,
            "forgets-to-halve-the-quadratic-formula",
        )

        if not formatted:
            continue
        answer_text = _fmt(answer)
        distractors = pick_distractors(answer_text, formatted)
        if not distractors:
            continue

        quad = _quad_text(b_coef, c_coef)
        pole = _full_pole_text(p, k)
        text = phrase(rng, _EXAM_PATTERNS, c=c, quad=quad)
        working = [
            f"z = [-({b_coef}) ± sqrt(({b_coef})^2 - 4({c_coef}))] / 2 = {p} ± {_pole_text(k)}; "
            f"only z = {pole} has Im(z) > 0.",
            f"Res(f, {pole}) = {_text_fmt(residue)}; integral = 2πi × Res = {_text_fmt(answer)}.",
        ]
        return dict(
            text=text, latex=text, answer=answer_text, answer_text=answer_text, working=working, distractors=distractors
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p15e_contour_integral_residue",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15e_contour_integral_residue(rng, tier="easy"):
    if tier == "easy":
        return _p15e_contour_easy(rng)
    if tier == "medium":
        return _p15e_contour_medium(rng)
    if tier == "hard":
        return _p15e_contour_hard(rng)
    if tier == "exam":
        return _p15e_contour_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Skill 2: ucv1-cr-equations-check
# ---------------------------------------------------------------------------

_CR_OPTIONS = (
    "Both Cauchy-Riemann equations hold",
    "Only u_x = v_y fails",
    "Only u_y = -v_x fails",
    "Both Cauchy-Riemann equations fail",
)

# Each kind below is built as an (a-part) + (b-part) sum, and picks its class
# from a fact that holds for ANY choice of the two parts' own analytic
# character, not merely the specific formulas used here:
#   - a holomorphic part (a function of z ALONE) contributes exactly 0 to
#     BOTH u_x-v_y and u_y+v_x, whatever that holomorphic function is (the
#     Cauchy-Riemann identity holds for every holomorphic function).
#   - an antiholomorphic part (a function of zc = x-iy ALONE) generally
#     contributes something position-dependent; a LINEAR antiholomorphic
#     part b*(zc-r) contributes a POSITION-INDEPENDENT 2b to u_x-v_y and 0
#     identically to u_y+v_x (checked once by hand below); a nonlinear one
#     (used for "anti_holomorphic" itself) does not have a closed form
#     simple enough to hand-prove, so its class is instead verified FRESH
#     every draw via `_cr_class` and resampled on any mismatch, exactly like
#     every other "unlucky coincidence" guard in this module (never a hard
#     `check` that could crash on a rare miss).
# holomorphic:      f(z) = a*sin(z-r) + b*(z-r)   -- entire in z alone (sin
#                    and the identity are both entire) -> CR holds
#                    identically -> class 0, ALWAYS (no resample needed).
# anti_holomorphic:  f(z) = a*sin(zc-r) + b*(zc-r) -- entire in zc alone
#                    -> antiholomorphic -> verified fresh, class 3 on
#                    almost every draw (rare coincidences resampled away).
# swap_first_fails:  f(z) = a*sin(z-r) + b*(zc-r)  -- u_x-v_y = 2b (b drawn
#                    nonzero, so never zero); u_y+v_x = 0 identically
#                    (proved once, below) -> class 1, ALWAYS.
# swap_second_fails: f(z) = a*sin(z-r) + i*b*(zc-r) -- u_x-v_y = 0
#                    identically; u_y+v_x = 2b (nonzero) -> class 2, ALWAYS.
#
# Proof of the linear b-part's contribution (b real, r real, z=x+iy):
#   b*(zc-r) = b*(x-r) - i*b*y  =>  u_b = b*(x-r), v_b = -b*y.
#   u_b,x = b, v_b,y = -b  =>  u_b,x - v_b,y = 2b.
#   u_b,y = 0, v_b,x = 0   =>  u_b,y + v_b,x = 0.
#   i*b*(zc-r) = i*b*(x-r) + b*y  =>  u_b = b*y, v_b = b*(x-r).
#   u_b,x = 0, v_b,y = 0   =>  u_b,x - v_b,y = 0.
#   u_b,y = b, v_b,x = b   =>  u_b,y + v_b,x = 2b.
# A holomorphic a-part's own contribution is 0 to both by the CR identity,
# so summing a holomorphic a-part with either linear b-part above gives
# EXACTLY the "b-part alone" numbers, whatever entire function sin(...) is
# swapped in for -- this is why the "a" side may safely be transcendental
# (a genuine chain-rule differentiation step) without upsetting the class guarantee.
_KIND_EXPECTED_CLASS = {
    "holomorphic": 0,
    "anti_holomorphic": 3,
    "swap_first_fails": 1,
    "swap_second_fails": 2,
}
_CR_KINDS = ("holomorphic", "anti_holomorphic", "swap_first_fails", "swap_second_fails")


def _is_zero(expr) -> bool:
    """Whether a fully-substituted (no free symbols) expression is exactly
    zero, checked NUMERICALLY (`evalf` at 30 digits) rather than via
    `sympy.simplify`/`trigsimp`: every value `_cr_class` calls this on is an
    exact combination of rationals and trig/hyperbolic functions of integer
    arguments, so a genuine identity (e.g. a holomorphic part's CR
    cancellation) evaluates to 0 to 30 digits and a genuine nonzero value
    (e.g. the linear b-part's constant "2b") is never within light-years of
    0 by coincidence. `trigsimp`'s symbolic search was measured costing
    ~9.5 s of a 20-draw exam-tier profile; this route is exact for this
    module's inputs and orders of magnitude cheaper."""
    value = complex(expr.evalf(30))
    return abs(value) < 1e-15


def _cr_class(u, v, x0, y0):
    """The four-way class of the Cauchy-Riemann equations at (x0, y0),
    computed by direct SymPy differentiation -- the only route this
    generator has to a class at all; every kind's own algebraic guarantee
    above is checked against this same call, never assumed."""
    ux = diff(u, _X).subs({_X: x0, _Y: y0})
    uy = diff(u, _Y).subs({_X: x0, _Y: y0})
    vx = diff(v, _X).subs({_X: x0, _Y: y0})
    vy = diff(v, _Y).subs({_X: x0, _Y: y0})
    first_holds = _is_zero(ux - vy)  # u_x = v_y
    second_holds = _is_zero(uy + vx)  # u_y = -v_x
    if first_holds and second_holds:
        return 0  # both hold
    if not first_holds and second_holds:
        return 1  # only u_x = v_y fails
    if first_holds and not second_holds:
        return 2  # only u_y = -v_x fails
    return 3  # both fail


def _cr_from_poly(a, b, r, kind):
    """Build u(x,y) + i v(x,y) for f(z) built from `kind` -- see the class
    table above. `r` is a real translation (0 at the easy tier, nonzero
    from medium on): translating a function by a real constant does not
    change any of its partial derivatives, so it never changes the CR
    class, only the DISPLAYED u, v (a genuinely harder read, not a
    different mathematical fact)."""
    z = _X + I * _Y
    zc = _X - I * _Y
    zt, zct = z - r, zc - r
    if kind == "holomorphic":
        f = a * zt**2 + b * zt
    elif kind == "anti_holomorphic":
        f = a * zct**2 + b * zct
    elif kind == "swap_first_fails":
        f = a * zt**2 + b * zct
    elif kind == "swap_second_fails":
        f = a * zt**2 + I * b * zct
    else:
        raise ValueError(kind)
    f = sympy.expand(f)
    u = sympy.expand(re(f))
    v = sympy.expand(im(f))
    return u, v


def _cr_from_trig(a, b, r, kind):
    """EXAM tier only: same four kinds, but the "a"-part is `a*sin(z-r)`
    (or its conjugate), an entire function that is NOT a polynomial --
    expanding sin((x-r)+iy) into u,v needs the chain rule and the complex
    trig identity sin(p+iq) = sin(p)cosh(q) + i cos(p)sinh(q), a genuine
    escalation beyond every other tier's pure polynomial differentiation
    ("a term whose partial derivative needs the product or
    chain rule"). The class guarantee is unaffected -- see the docstring
    above the kind table: a holomorphic a-part contributes 0 to both CR
    differences whatever entire function it is."""
    z = _X + I * _Y
    zc = _X - I * _Y
    zt, zct = z - r, zc - r
    if kind == "holomorphic":
        f = a * sin(zt) + b * zt
    elif kind == "anti_holomorphic":
        f = a * sin(zct) + b * zct
    elif kind == "swap_first_fails":
        f = a * sin(zt) + b * zct
    elif kind == "swap_second_fails":
        f = a * sin(zt) + I * b * zct
    else:
        raise ValueError(kind)
    f = sympy.expand(f, complex=True)
    u = sympy.expand(re(f))
    v = sympy.expand(im(f))
    return u, v


_CR_PATTERNS = [
    "Let f(z) = u(x,y) + i v(x,y) where u(x,y) = {u} and v(x,y) = {v}. At "
    "the point (x, y) = ({x0}, {y0}), decide whether f satisfies the "
    "Cauchy-Riemann equations: (a) Both hold. (b) Only u_x = v_y fails. "
    "(c) Only u_y = -v_x fails. (d) Both fail.",
    "For u(x,y) = {u} and v(x,y) = {v}, at the point (x, y) = ({x0}, {y0}), "
    "which describes the Cauchy-Riemann equations u_x = v_y and "
    "u_y = -v_x: (a) Both hold. (b) Only u_x = v_y fails. (c) Only "
    "u_y = -v_x fails. (d) Both fail?",
    "Given u(x,y) = {u} and v(x,y) = {v}, check the Cauchy-Riemann "
    "equations at (x, y) = ({x0}, {y0}): does (a) both hold, (b) only "
    "u_x = v_y fail, (c) only u_y = -v_x fail, or (d) both fail?",
]


def _shift_arg(r, base: str) -> str:
    """ "z", "z - 6", "z + 6" -- `base` is "z" or "conj(z)"."""
    if r == 0:
        return base
    return f"{base} - {r}" if r > 0 else f"{base} + {-r}"


def _f_text(a, b, r, kind) -> str:
    """f(z), shown to the learner directly (never pre-expanded into u, v):
    via `join_terms`, so the join between the a-part and b-part always has
    a correct sign ("... - 2i(...)", never "... + -2i(...)")."""
    a_is_conj = kind == "anti_holomorphic"
    b_is_conj = kind != "holomorphic"
    a_body = f"sin({_shift_arg(r, 'conj(z)' if a_is_conj else 'z')})"
    b_shift = _shift_arg(r, "conj(z)" if b_is_conj else "z")
    b_body = f"i({b_shift})" if kind == "swap_second_fails" else f"({b_shift})"
    return join_terms([(a, a_body), (b, b_body)])


# Exam tier ONLY: f(z) is given directly, in z and conj(z) -- never the pre-expanded u(x,y), v(x,y)
# every other tier hands over -- so finding u, v is itself a genuine reasoning step this tier adds
# (each exam tier needs a step the hard tier never asks for). The point is a bare
# coordinate pair exactly as at every other tier: no unrelated arithmetic padding.
_EXAM_CR_PATTERNS = [
    "Let f(z) = {f_text}. At the point (x, y) = ({x0}, {y0}), first find "
    "u(x,y) and v(x,y) (write z = x+iy, conj(z) = x-iy), then decide "
    "whether f satisfies the Cauchy-Riemann equations there: (a) Both "
    "hold. (b) Only u_x = v_y fails. (c) Only u_y = -v_x fails. (d) Both "
    "fail.",
    "For f(z) = {f_text}, with z = x+iy, find u(x,y) and v(x,y), then at "
    "the point (x, y) = ({x0}, {y0}) decide which describes the "
    "Cauchy-Riemann equations u_x = v_y and u_y = -v_x: (a) Both hold. "
    "(b) Only u_x = v_y fails. (c) Only u_y = -v_x fails. (d) Both fail?",
    "Given f(z) = {f_text} (z = x+iy), write f in terms of u(x,y) and "
    "v(x,y), then check the Cauchy-Riemann equations at (x, y) = "
    "({x0}, {y0}): does (a) both hold, (b) only u_x = v_y fail, (c) only "
    "u_y = -v_x fail, or (d) both fail?",
]


def _p15e_cr_common(rng, a_hi, pt_hi, r_hi, *, exam_point=False):
    for _ in range(_MAX_ATTEMPTS):
        a = rng.choice([v for v in range(-a_hi, a_hi + 1) if v != 0])
        b = rng.choice([v for v in range(-a_hi, a_hi + 1) if v != 0])
        x0 = rng.randint(-pt_hi, pt_hi)
        y0 = rng.randint(-pt_hi, pt_hi)
        r = rng.randint(-r_hi, r_hi) if r_hi else 0
        kind = rng.choice(_CR_KINDS)
        builder = _cr_from_trig if exam_point else _cr_from_poly
        u, v = builder(a, b, r, kind)
        cls = _cr_class(u, v, x0, y0)
        if cls != _KIND_EXPECTED_CLASS[kind]:
            continue  # a rare position/parameter coincidence -- resample

        answer_text = _CR_OPTIONS[cls]
        others = [opt for opt in _CR_OPTIONS if opt != answer_text]

        u_text = _text_fmt(u)
        v_text = _text_fmt(v)
        if exam_point:
            f_text = _f_text(a, b, r, kind)
            text = phrase(rng, _EXAM_CR_PATTERNS, f_text=f_text, x0=x0, y0=y0)
        else:
            text = phrase(rng, _CR_PATTERNS, u=u_text, v=v_text, x0=x0, y0=y0)
        working = [
            f"u(x,y) = {u_text}, v(x,y) = {v_text}."
            if exam_point
            else f"u_x = {_text_fmt(diff(u, _X))}, u_y = {_text_fmt(diff(u, _Y))}, "
            f"v_x = {_text_fmt(diff(v, _X))}, v_y = {_text_fmt(diff(v, _Y))} "
            f"(evaluated at x={x0}, y={y0}).",
            f"Classification: {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=f"plain:{answer_text}",
            working=working,
            distractors=pick_distractors(answer_text, others),
        )
    raise RuntimeError("cr: could not draw a usable question")


@generator(
    "p15e_cr_equations_check",
    level=18,
    presets={
        "easy": dict(a_hi=4, pt_hi=2, r_hi=0, exam_point=False),
        "medium": dict(a_hi=7, pt_hi=4, r_hi=0, exam_point=False),
        "hard": dict(a_hi=7, pt_hi=4, r_hi=4, exam_point=False),
        "exam": dict(a_hi=9, pt_hi=5, r_hi=6, exam_point=True),
    },
)
def p15e_cr_equations_check(rng, a_hi=4, pt_hi=2, r_hi=0, exam_point=False):
    return _p15e_cr_common(rng, a_hi, pt_hi, r_hi, exam_point=exam_point)
