"""Exam-tier source for `senior-integrate-using-the-trigonometric-identities-mathrm-s-mathrm`: rewrite with an
identity, integrate, then evaluate a definite integral.

f(x) is p sin^2(kx), p cos^2(kx) or p sin(kx) cos(kx). With a = p/2 the identities give

    p sin^2(kx)         = a - a cos(2kx)       integral  a x - a sin(2kx)/(2k)  + C
    p cos^2(kx)         = a + a cos(2kx)       integral  a x + a sin(2kx)/(2k)  + C
    p sin(kx) cos(kx)   = a sin(2kx)           integral  -a cos(2kx)/(2k)       + C

and the chain is

(a) the constant a                                   fraction,   1 mark
(b) the indefinite integral of f                     expression, 2 marks   built from part_a
(c) the exact value of the integral from 0 to L      expression, 2 marks   F(L) - F(0) from part_b, L = pi/(4k) or pi/(2k)

The upper limit makes 2kL equal to pi/2 or pi, so every trigonometric value needed is exactly 0, 1 or -1 and the
arithmetic is done with exact fractions (an antiderivative is a triple: coefficient of x, coefficient of the
trigonometric term, and which function it is). Every wrong option in (b) and (c) is a slip a learner makes carrying
the earlier error forward (a missing 1/2k, a swapped sign, the half left out, a sine evaluated as a cosine at the
limit, the constant forgotten). The key is checked by differentiating (b) and by Simpson's rule on (c), both with
plain floating point, and (b) is also checked symbolically in the tests.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import bracket_compound, check, generator, latex_answer, pick_distractors, rational_text
from ._exd_distinct_parts import distinct_part_answers

# (value of the function at 0, at pi/2, at pi)
_AT = {"sin": (0, 1, 0), "cos": (1, 0, -1)}



def _form_with(form: str, a, a_text: str) -> str:
    """The form "a - a cos(4x)" or "a sin(4x)" with the constant filled in; a coefficient of 1 is left off."""
    coef = "" if a == 1 else (bracket_compound(a_text) + " ")
    first, _, tail = form.partition(" ")  # "a", "- a cos(4x)" or "sin(4x)"
    if tail.startswith(("sin", "cos")):
        return f"{coef}{tail}"
    op, _, trig = tail.partition(" a ")
    return f"{a_text} {op} {coef}{trig}"

    return form

def _coef_text(coef: Fraction, atom: str) -> str:
    """ "3*atom/2", "-atom", "atom/4" ... for a non-zero coefficient."""
    head = "-" if coef < 0 else ""
    num, den = abs(coef.numerator), coef.denominator
    body = atom if num == 1 else f"{num}*{atom}"
    return f"{head}{body}" + (f"/{den}" if den != 1 else "")


def _join(terms: list[str]) -> str:
    out = terms[0]
    for t in terms[1:]:
        out += f" - {t[1:]}" if t.startswith("-") else f" + {t}"
    return out


def _integral_text(a_x: Fraction, a_trig: Fraction, trig: str, k: int, *, constant: bool = True) -> str:
    arg = f"{2 * k}*x"
    terms = []
    if a_x != 0:
        terms.append(_coef_text(a_x, "x"))
    terms.append(_coef_text(a_trig, f"{trig}({arg})"))
    return _join(terms) + (" + C" if constant else "")


def _antiderivative(family: str, a: Fraction, k: int, *, flip: bool = False, scale: int = 1):
    """(coefficient of x, coefficient of the trigonometric term, which function), with optional slips: `flip`
    reverses the sign of the trigonometric term, `scale` multiplies it (2k when the 1/(2k) is forgotten)."""
    sign = -1 if flip else 1
    if family == "sin2":
        return a, -sign * scale * a / (2 * k), "sin"
    if family == "cos2":
        return a, sign * scale * a / (2 * k), "sin"
    return Fraction(0), -sign * scale * a / (2 * k), "cos"


def _definite(antiderivative, k: int, upper_over: int, *, swap: bool = False):
    """(coefficient of pi, rational part) of F(L) - F(0) for L = pi/(upper_over k); `swap` evaluates the other of
    sine and cosine at the limits, the slip of reading one function's value for the other's."""
    a_x, a_trig, trig = antiderivative
    values = _AT["cos" if swap and trig == "sin" else "sin" if swap else trig]
    top = values[1] if upper_over == 4 else values[2]  # 2kL = pi/2 or pi
    return a_x / (upper_over * k), a_trig * (top - values[0])


def _value_text(pi_coef: Fraction, rest: Fraction) -> str:
    terms = []
    if pi_coef != 0:
        terms.append(_coef_text(pi_coef, "pi"))
    if rest != 0:
        terms.append(rational_text(rest))
    return _join(terms) if terms else "0"


def _simpson(f, lo: float, hi: float, n: int = 200) -> float:
    h = (hi - lo) / n
    return (f(lo) + f(hi) + sum((4 if i % 2 else 2) * f(lo + i * h) for i in range(1, n))) * h / 3


def _options(key, candidates, n: int):
    out, seen = [], [key]
    for tag, signature, text in candidates:
        if signature in seen:
            continue
        seen.append(signature)
        out.append(Distractor(text, tag, value=text))
    return out[:n]


@generator("exd_x_r3e_trig_identity_integral", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_trig_identity_integral(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    family = rng.choice(["sin2", "cos2", "sincos"])
    p, k = rng.randint(2, 12), rng.randint(1, 5)
    upper_over = rng.choice([2, 4])  # L = pi / (upper_over * k)
    a = Fraction(p, 2)
    L = math.pi / (upper_over * k)

    def f(v: float) -> float:
        return {
            "sin2": p * math.sin(k * v) ** 2,
            "cos2": p * math.cos(k * v) ** 2,
            "sincos": p * math.sin(k * v) * math.cos(k * v),
        }[family]

    F = _antiderivative(family, a, k)
    pi_coef, rest = _definite(F, k, upper_over)

    # Independent checks, in floating point: F' is f, and Simpson's rule reproduces the definite integral.
    a_x, a_trig, trig = F
    for probe in (0.31, 0.77, 1.4):
        deriv = float(a_x) + float(a_trig) * 2 * k * (
            math.cos(2 * k * probe) if trig == "sin" else -math.sin(2 * k * probe)
        )
        check(abs(deriv - f(probe)) < 1e-9, "F' is not f")
    check(abs(_simpson(f, 0.0, L) - (float(pi_coef) * math.pi + float(rest))) < 1e-7, "definite integral disagrees")

    # -- (a) ----------------------------------------------------------------------------------------------------
    a_text = rational_text(a)
    a_dist = pick_distractors(
        a_text,
        [
            Distractor(rational_text(Fraction(p)), "forgot-half-in-power-reduction", value=Fraction(p)),
            Distractor(rational_text(Fraction(p, 4)), "double-angle-identity-misapplied", value=Fraction(p, 4)),
            Distractor(rational_text(Fraction(2 * p)), "double-angle-identity-misapplied", value=Fraction(2 * p)),
            Distractor(rational_text(-a), "power-reduction-sign-swapped", value=-a),
        ],
        3,
    )

    # -- (b) ----------------------------------------------------------------------------------------------------
    key_b = _integral_text(*F[:2], F[2], k)
    wrong_forms = [
        ("trig-antiderivative-sign-slip", _antiderivative(family, a, k, flip=True)),
        ("integral-inner-coefficient-not-divided", _antiderivative(family, a, k, scale=2 * k)),
        ("forgot-half-in-power-reduction", _antiderivative(family, Fraction(p), k)),
    ]
    b_cands = [(tag, G, _integral_text(*G[:2], G[2], k)) for tag, G in wrong_forms]
    rng.shuffle(b_cands)
    # The constant left out is a slip of its own; the grader refuses a reply without + C.
    no_constant = _integral_text(*F[:2], F[2], k, constant=False)
    b_dist = _options(F, b_cands, 2) + [Distractor(no_constant, "forgot-constant-of-integration", value=no_constant)]

    # -- (c) ----------------------------------------------------------------------------------------------------
    key_c = _value_text(pi_coef, rest)
    c_slips = [(tag, _definite(G, k, upper_over)) for tag, G in wrong_forms]
    c_slips.append(("dropped-trig-term-at-limit", _definite(F, k, upper_over, swap=True)))
    rng.shuffle(c_slips)
    c_dist = _options((pi_coef, rest), [(t, v, _value_text(*v)) for t, v in c_slips], 3)
    check(len(a_dist) >= 3 and len(b_dist) == 3 and len(c_dist) >= 2, "too few distractors")

    kx = "" if k == 1 else str(k)
    shown = {"sin2": f"sin^2({kx}x)", "cos2": f"cos^2({kx}x)", "sincos": f"sin({kx}x)cos({kx}x)"}[family]
    func_latex = {
        "sin2": rf"{p}\sin^{{2}}({kx}x)",
        "cos2": rf"{p}\cos^{{2}}({kx}x)",
        "sincos": rf"{p}\sin({kx}x)\cos({kx}x)",
    }[family]
    form = {
        "sin2": f"a - a cos({2 * k}x)",
        "cos2": f"a + a cos({2 * k}x)",
        "sincos": f"a sin({2 * k}x)",
    }[family]
    upper = f"π/{upper_over * k}"
    ft_b = {
        "sin2": f"part_a*x - part_a*sin({2 * k}*x)/{2 * k} + C",
        "cos2": f"part_a*x + part_a*sin({2 * k}*x)/{2 * k} + C",
        "sincos": f"-part_a*cos({2 * k}*x)/{2 * k} + C",
    }[family]
    return dict(
        text=f"The function f is defined by f(x) = {p} {shown}.",
        latex=rf"\text{{The function }}f\text{{ is defined by }}f(x) = {func_latex}\text{{.}}",
        parts=[
            dict(
                prompt=f"Write f(x) in the form {form}, where a is a constant. Find a.",
                answer_kind="fraction",
                answer={"p": a.numerator, "q": a.denominator},
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                working=[f"Using the identity, f(x) = {_form_with(form, a, a_text)}, so a = {a_text}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the indefinite integral of f(x).",
                answer_kind="expression",
                answer=key_b,
                answer_text=key_b,
                answer_latex=latex_answer(key_b),
                marks=2,
                working=[f"The integral of f(x) is {key_b}."],
                distractors=b_dist,
                follow_through=ft_b,
            ),
            dict(
                prompt=f"Hence find the exact value of the integral of f(x) from x = 0 to x = {upper}.",
                answer_kind="expression",
                answer=key_c,
                answer_text=key_c,
                answer_latex=latex_answer(key_c),
                marks=2,
                working=[f"Evaluate the integral from 0 to {upper}: the value is {key_c}."],
                distractors=c_dist,
                follow_through=f"Subs(part_b, (x,), (pi/{upper_over * k},)) - Subs(part_b, (x,), (0,))",
            ),
        ],
    )
