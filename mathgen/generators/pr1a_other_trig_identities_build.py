"""Rubric-assessed proof skill: the learner proves a triple- or quadruple-angle /
power-reduction identity beyond the angle-sum, double-angle, Pythagorean and product families (each derived by
CITING those families as given, never re-proving them), then applies one to evaluate a stated trigonometric
expression exactly, or to a short prove-then-use composite.

This module keeps only a cheap, generator-local numeric sanity probe: every
reference line is evaluated with the `math` module at three fixed-but-irregular points (or, for an "apply"
item with no free symbol, at the item's own fixed angle) and checked to agree with the target.

CAS-confirmability, measured directly on this base:
- The triple-angle chains (cos 3x, sin 3x) and their power-reduction
  rearrangements (cos^3 x, sin^3 x) confirm at every step for a PLAIN base
  symbol, both as a full derivation and as a bare start->target jump.
- The quadruple-angle chains (cos 4x, sin 4x, cos^4 x) confirm at every step
  for a plain OR scaled base symbol (multiplier up to 4 measured), both
  full and bare.
- A SCALED triple-angle argument (e.g. cos 6t = 4cos(2t)^3 - 3cos(2t), the
  medium tier's illustrative example) does NOT confirm, at any step,
  including the very first (`cos(6*t)` vs `cos(3*(2*t))`, which sympy's Mul
  flattens to the same literal `cos(6*t)` -- the failure is inside
  `symbolically_equal`'s `simplify` call recognising the scaled triple-angle
  polynomial, not a wording issue). Measured directly; dropped from the
  pool. `tan(3*x)` in terms of `tan(2*x)`/`tan(x)` also does not confirm
  on this base, so no tan family is built at all.
- A numeric apply value only confirms for CERTAIN denominators: `pi/9`,
  `4*pi/9`, `5*pi/9`, `8*pi/9`, `pi/12`, `5*pi/12`, `7*pi/12`, `11*pi/12`,
  `pi/18`, `7*pi/18`, `11*pi/18`, `17*pi/18` all confirm (both cos- and
  sin-triple); `2*pi/9`, `7*pi/9`, `5*pi/18`, `13*pi/18` do NOT (`simplify`
  returns an un-reduced cubic in the base cosine/sine, `.is_zero` is `None`)
  even though the identity is exactly as true. The skill's own example,
  `4cos^3(pi/9) - 3cos(pi/9) = 1/2`, is in the confirming set; `2*pi/9`
  (an earlier note had it passing) does NOT confirm on THIS base.
"""

from __future__ import annotations

import math
import random
import re

from ..core import check, generator, phrase

# ---------------------------------------------------------------------------
# Safe symbol names (both sympy.sympify and mathgen.grading._parse_expression),
# Latin only -- no Greek, spelled out or otherwise.
VARS: tuple[tuple[str, str, str], ...] = (
    ("x", "x", "x"),
    ("t", "t", "t"),
    ("u", "u", "u"),
    ("v", "v", "v"),
)

# Fixed-but-irregular probe points (radians) for the cheap in-generator
# numeric check.
PROBE_POINTS: tuple[float, ...] = (0.41, 1.07, -0.53)

# Multiplier pool for the hard tier's quadruple-angle families only (measured
# to confirm through the CAS at m=1..4; the triple-angle families never take
# a multiplier -- see the module docstring).
QUAD_MULTIPLIERS: tuple[int, ...] = (1, 1, 2, 3, 4)


def _disp(var_display: str, m: int) -> str:
    return var_display if m == 1 else f"{m}{var_display}"


def _sym(var_name: str, m: int) -> str:
    return var_name if m == 1 else f"{m}*{var_name}"


def _readable_expr_text(sympy_text: str) -> str:
    """Typeable ASCII rendering of one of this module's own sympy-text
    expressions: `**` -> `^`, `*` dropped entirely."""
    return sympy_text.replace("**", "^").replace("*", "")


_TRIG_LATEX_RE = re.compile(r"\b(sin|cos|tan)\(")


def _to_latex_expr(sympy_text: str) -> str:
    """This module's own sympy-text to proper LaTeX, for an APPLY item's
    rendered expression only (latex keeps real math notation,
    unlike the typeable-ASCII `text` field)."""
    out = sympy_text.replace("**", "^")
    out = _TRIG_LATEX_RE.sub(lambda m: f"\\{m.group(1)}(", out)
    out = out.replace("pi", r"\pi")
    return out.replace("*", "")


def _pi_display(frac: str) -> str:
    """`"pi/12"` -> `"π/12"`, `"5*pi/12"` -> `"5π/12"` -- this module's
    own fixed-form radian fractions only."""
    return frac.replace("*pi", "π").replace("pi", "π")


# ---------------------------------------------------------------------------
# Givens the learner may cite (never re-derive). Every function argument
# bracketed, powers with `^`, no unicode superscript.
# ---------------------------------------------------------------------------
SIN_SUM_GIVEN = "sin(A + B) = sin(A)cos(B) + cos(A)sin(B)"
COS_SUM_GIVEN = "cos(A + B) = cos(A)cos(B) - sin(A)sin(B)"
PYTH_GIVEN = "sin(A)^2 + cos(A)^2 = 1"
SIN2_GIVEN = "sin(2A) = 2sin(A)cos(A) (the previously established double-angle result)"
COS2_2COS2M1_GIVEN = "cos(2A) = 2cos(A)^2 - 1 (the previously established double-angle result)"
COS2_1M2SIN2_GIVEN = "cos(2A) = 1 - 2sin(A)^2 (the previously established double-angle result)"
TRIPLE_COS_GIVEN = "cos(3A) = 4cos(A)^3 - 3cos(A) (the previously established triple-angle result)"
TRIPLE_SIN_GIVEN = "sin(3A) = 3sin(A) - 4sin(A)^3 (the previously established triple-angle result)"

SIN_SUM_GIVEN_LATEX = r"\sin(A + B) = \sin A\cos B + \cos A\sin B"
COS_SUM_GIVEN_LATEX = r"\cos(A + B) = \cos A\cos B - \sin A\sin B"
PYTH_GIVEN_LATEX = r"\sin^2 A + \cos^2 A = 1"
SIN2_GIVEN_LATEX = r"\sin 2A = 2\sin A\cos A \text{ (previously established)}"
COS2_2COS2M1_GIVEN_LATEX = r"\cos 2A = 2\cos^2 A - 1 \text{ (previously established)}"
COS2_1M2SIN2_GIVEN_LATEX = r"\cos 2A = 1 - 2\sin^2 A \text{ (previously established)}"
TRIPLE_COS_GIVEN_LATEX = r"\cos 3A = 4\cos^3 A - 3\cos A \text{ (previously established)}"
TRIPLE_SIN_GIVEN_LATEX = r"\sin 3A = 3\sin A - 4\sin^3 A \text{ (previously established)}"


# ---------------------------------------------------------------------------
# PROVE identity families. Each entry is a dict of pure-string builders (the
# sympy-text `start`/`target`/`reference_lines`/traps) plus a `domain_all_real`
# flag and a pair of `math`-only callables used for the cheap per-item
# numeric probe. Triple-angle families take no multiplier (measured not to
# confirm scaled); quadruple-angle families may.
# ---------------------------------------------------------------------------


def _fam_cos3(s: str) -> dict:
    base = s
    triple = f"3*{s}"
    start = f"cos({triple})"
    target = f"4*cos({base})**3 - 3*cos({base})"
    ref = [
        start,
        f"cos(2*{base} + {base})",
        f"cos(2*{base})*cos({base}) - sin(2*{base})*sin({base})",
        f"(2*cos({base})**2 - 1)*cos({base}) - 2*sin({base})*cos({base})*sin({base})",
        f"2*cos({base})**3 - cos({base}) - 2*(1 - cos({base})**2)*cos({base})",
        target,
    ]
    return dict(
        key="cos3",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS_SUM_GIVEN, COS2_2COS2M1_GIVEN, PYTH_GIVEN],
        givens_latex=[COS_SUM_GIVEN_LATEX, COS2_2COS2M1_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_traps=[
            ("triples_the_function", f"3*cos({base})"),
            ("sign_slip_in_triple_angle", f"4*cos({base})**3 + 3*cos({base})"),
        ],
        wrong_cos2x_index=3,
        wrong_cos2x_line=f"(1 - 2*cos({base})**2)*cos({base}) - 2*sin({base})*cos({base})*sin({base})",
        math_left=lambda v: math.cos(3 * v),
        math_right=lambda v: 4 * math.cos(v) ** 3 - 3 * math.cos(v),
        identity_latex=lambda bd, td: rf"\cos {td} = 4\cos^3 {bd} - 3\cos {bd}",
        domain_all_real=True,
    )


def _fam_sin3(s: str) -> dict:
    base = s
    triple = f"3*{s}"
    start = f"sin({triple})"
    target = f"3*sin({base}) - 4*sin({base})**3"
    ref = [
        start,
        f"sin(2*{base} + {base})",
        f"sin(2*{base})*cos({base}) + cos(2*{base})*sin({base})",
        f"(2*sin({base})*cos({base}))*cos({base}) + (1 - 2*sin({base})**2)*sin({base})",
        f"2*sin({base})*(1 - sin({base})**2) + sin({base}) - 2*sin({base})**3",
        target,
    ]
    return dict(
        key="sin3",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[SIN_SUM_GIVEN, COS2_1M2SIN2_GIVEN, PYTH_GIVEN],
        givens_latex=[SIN_SUM_GIVEN_LATEX, COS2_1M2SIN2_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_traps=[
            ("triples_the_function", f"3*sin({base})"),
            ("sign_slip_in_triple_angle", f"3*sin({base}) + 4*sin({base})**3"),
        ],
        wrong_cos2x_index=3,
        wrong_cos2x_line=f"(2*sin({base})*cos({base}))*cos({base}) + (2*sin({base})**2 - 1)*sin({base})",
        math_left=lambda v: math.sin(3 * v),
        math_right=lambda v: 3 * math.sin(v) - 4 * math.sin(v) ** 3,
        identity_latex=lambda bd, td: rf"\sin {td} = 3\sin {bd} - 4\sin^3 {bd}",
        domain_all_real=True,
    )


def _fam_cos3_power(s: str) -> dict:
    base = s
    triple = f"3*{s}"
    start = f"cos({base})**3"
    target = f"(3*cos({base}) + cos({triple}))/4"
    ref = [
        start,
        f"(4*cos({base})**3)/4",
        f"(4*cos({base})**3 - 3*cos({base}) + 3*cos({base}))/4",
        target,
    ]
    return dict(
        key="cos3_power",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[TRIPLE_COS_GIVEN],
        givens_latex=[TRIPLE_COS_GIVEN_LATEX],
        auth_traps=[
            ("wrong_cos2x_form_midway", f"(3*cos({base}) - cos({triple}))/4"),
        ],
        wrong_cos2x_index=2,
        wrong_cos2x_line=f"(3*cos({base}) - cos({triple}))/4",
        math_left=lambda v: math.cos(v) ** 3,
        math_right=lambda v: (3 * math.cos(v) + math.cos(3 * v)) / 4,
        identity_latex=lambda bd, td: rf"\cos^3 {bd} = \dfrac{{3\cos {bd} + \cos {td}}}{{4}}",
        domain_all_real=True,
    )


def _fam_sin3_power(s: str) -> dict:
    base = s
    triple = f"3*{s}"
    start = f"sin({base})**3"
    target = f"(3*sin({base}) - sin({triple}))/4"
    ref = [
        start,
        f"(4*sin({base})**3)/4",
        f"(-(3*sin({base}) - 4*sin({base})**3) + 3*sin({base}))/4",
        target,
    ]
    return dict(
        key="sin3_power",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[TRIPLE_SIN_GIVEN],
        givens_latex=[TRIPLE_SIN_GIVEN_LATEX],
        auth_traps=[
            ("wrong_cos2x_form_midway", f"(3*sin({base}) + sin({triple}))/4"),
        ],
        wrong_cos2x_index=2,
        wrong_cos2x_line=f"(3*sin({base}) + sin({triple}))/4",
        math_left=lambda v: math.sin(v) ** 3,
        math_right=lambda v: (3 * math.sin(v) - math.sin(3 * v)) / 4,
        identity_latex=lambda bd, td: rf"\sin^3 {bd} = \dfrac{{3\sin {bd} - \sin {td}}}{{4}}",
        domain_all_real=True,
    )


def _fam_cos4(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    quad = _sym(s, 4 * m)
    start = f"cos({quad})"
    target = f"8*cos({base})**4 - 8*cos({base})**2 + 1"
    ref = [
        start,
        f"cos(2*({double}))",
        f"2*cos({double})**2 - 1",
        f"2*(2*cos({base})**2 - 1)**2 - 1",
        target,
    ]
    return dict(
        key="cos4",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_2COS2M1_GIVEN],
        givens_latex=[COS2_2COS2M1_GIVEN_LATEX],
        auth_traps=[
            ("wrong_cos2x_form_midway", f"1 - 2*cos({double})**2"),
            ("quadruples_the_function", f"4*cos({base})"),
        ],
        wrong_cos2x_index=2,
        wrong_cos2x_line=f"1 - 2*cos({double})**2",
        math_left=lambda v: math.cos(4 * m * v),
        math_right=lambda v: 8 * math.cos(m * v) ** 4 - 8 * math.cos(m * v) ** 2 + 1,
        identity_latex=lambda bd, qd: rf"\cos {qd} = 8\cos^4 {bd} - 8\cos^2 {bd} + 1",
        domain_all_real=True,
    )


def _fam_sin4(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    quad = _sym(s, 4 * m)
    start = f"sin({quad})"
    target = f"4*sin({base})*cos({base})*(1 - 2*sin({base})**2)"
    ref = [
        start,
        f"sin(2*({double}))",
        f"2*sin({double})*cos({double})",
        f"2*(2*sin({base})*cos({base}))*(1 - 2*sin({base})**2)",
        target,
    ]
    return dict(
        key="sin4",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[SIN2_GIVEN, COS2_1M2SIN2_GIVEN],
        givens_latex=[SIN2_GIVEN_LATEX, COS2_1M2SIN2_GIVEN_LATEX],
        auth_traps=[
            ("wrong_cos2x_form_midway", f"2*(2*sin({base})*cos({base}))*(1 + 2*sin({base})**2)"),
            ("quadruples_the_function", f"4*sin({base})"),
        ],
        wrong_cos2x_index=3,
        wrong_cos2x_line=f"2*(2*sin({base})*cos({base}))*(1 + 2*sin({base})**2)",
        math_left=lambda v: math.sin(4 * m * v),
        math_right=lambda v: 4 * math.sin(m * v) * math.cos(m * v) * (1 - 2 * math.sin(m * v) ** 2),
        identity_latex=lambda bd, qd: rf"\sin {qd} = 4\sin {bd}\cos {bd}(1 - 2\sin^2 {bd})",
        domain_all_real=True,
    )


def _fam_cos4_power(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    quad = _sym(s, 4 * m)
    start = f"cos({base})**4"
    target = f"(3 + 4*cos({double}) + cos({quad}))/8"
    ref = [
        start,
        f"(cos({base})**2)**2",
        f"((1 + cos({double}))/2)**2",
        f"(1 + 2*cos({double}) + cos({double})**2)/4",
        f"(1 + 2*cos({double}) + (1 + cos({quad}))/2)/4",
        target,
    ]
    return dict(
        key="cos4_power",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_2COS2M1_GIVEN],
        givens_latex=[COS2_2COS2M1_GIVEN_LATEX],
        auth_traps=[
            ("wrong_cos2x_form_midway", f"(3 - 4*cos({double}) + cos({quad}))/8"),
        ],
        wrong_cos2x_index=3,
        wrong_cos2x_line=f"(1 - 2*cos({double}) + cos({double})**2)/4",
        math_left=lambda v: math.cos(m * v) ** 4,
        math_right=lambda v: (3 + 4 * math.cos(2 * m * v) + math.cos(4 * m * v)) / 8,
        identity_latex=lambda bd, dd, qd: rf"\cos^4 {bd} = \dfrac{{3 + 4\cos {dd} + \cos {qd}}}{{8}}",
        domain_all_real=True,
    )


EASY_PROVE_BUILDERS = (_fam_cos3, _fam_sin3)
MEDIUM_PROVE_BUILDERS = (_fam_cos3_power, _fam_sin3_power)
HARD_QUAD_BUILDERS = (_fam_cos4, _fam_sin4, _fam_cos4_power)


# ---------------------------------------------------------------------------
# APPLY items: evaluate a stated expression by recognising it as a triple
# angle. Angle pool restricted to the denominators/numerators measured to
# confirm through the CAS (module docstring) -- `2*pi/9`, `7*pi/9`,
# `5*pi/18`, `13*pi/18` are measured NOT to confirm and are excluded.
# (kind, base fraction, degree, exact-value sympy text, exact-value readable)
# ---------------------------------------------------------------------------
COS_APPLY_POOL: tuple[tuple[str, int, str, str], ...] = (
    ("pi/9", 20, "1/2", "1/2"),
    ("4*pi/9", 80, "-1/2", "-1/2"),
    ("5*pi/9", 100, "1/2", "1/2"),
    ("8*pi/9", 160, "-1/2", "-1/2"),
    ("pi/12", 15, "sqrt(2)/2", "√2/2"),
    ("5*pi/12", 75, "-sqrt(2)/2", "-√2/2"),
    ("7*pi/12", 105, "sqrt(2)/2", "√2/2"),
    ("11*pi/12", 165, "-sqrt(2)/2", "-√2/2"),
    ("pi/18", 10, "sqrt(3)/2", "√3/2"),
    ("7*pi/18", 70, "-sqrt(3)/2", "-√3/2"),
    ("11*pi/18", 110, "sqrt(3)/2", "√3/2"),
    ("17*pi/18", 170, "-sqrt(3)/2", "-√3/2"),
)
SIN_APPLY_POOL: tuple[tuple[str, int, str, str], ...] = (
    ("pi/9", 20, "sqrt(3)/2", "√3/2"),
    ("4*pi/9", 80, "-sqrt(3)/2", "-√3/2"),
    ("5*pi/9", 100, "-sqrt(3)/2", "-√3/2"),
    ("8*pi/9", 160, "sqrt(3)/2", "√3/2"),
    ("pi/12", 15, "sqrt(2)/2", "√2/2"),
    ("5*pi/12", 75, "-sqrt(2)/2", "-√2/2"),
    ("7*pi/12", 105, "-sqrt(2)/2", "-√2/2"),
    ("11*pi/12", 165, "sqrt(2)/2", "√2/2"),
    ("pi/18", 10, "1/2", "1/2"),
    ("7*pi/18", 70, "-1/2", "-1/2"),
    ("11*pi/18", 110, "-1/2", "-1/2"),
    ("17*pi/18", 170, "1/2", "1/2"),
)


def eval_base(base_frac: str) -> float:
    """A tiny, fixed-form radian evaluator for the pool's own `"k*pi/n"`
    strings -- never a general expression parser."""
    text = base_frac.replace(" ", "")
    if text.startswith("pi/"):
        return math.pi / int(text[3:])
    if text.startswith("-pi/"):
        return -math.pi / int(text[4:])
    k, rest = text.split("*pi/")
    return int(k) * math.pi / int(rest)


def _apply_family(kind: str, angle: tuple[str, int, str, str]) -> dict:
    base_frac, deg, target_val, target_readable = angle
    base_rad = _pi_display(base_frac)
    if kind == "cos":
        start = f"4*cos({base_frac})**3 - 3*cos({base_frac})"
        ref = [start, f"cos(3*{base_frac})", target_val]
        text_expr = f"4cos({base_rad})^3 - 3cos({base_rad})"
    else:
        start = f"3*sin({base_frac}) - 4*sin({base_frac})**3"
        ref = [start, f"sin(3*{base_frac})", target_val]
        text_expr = f"3sin({base_rad}) - 4sin({base_rad})^3"
    return dict(
        key=f"apply_{kind}",
        start=start,
        target=target_val,
        reference_lines=ref,
        text_expr=text_expr,
        target_readable=target_readable,
        deg=deg,
    )


APPLY_KINDS = ("cos", "sin")


# ---------------------------------------------------------------------------
# EXAM composite items: prove a short corollary by citing the triple-angle
# result, then implicitly "use" it ("prove-then-use
# composite"). Both measured to confirm through the CAS (module docstring).
# Domain-restricted (a division by cos/sin term) wording.
# ---------------------------------------------------------------------------


def _fam_composite_cos_ratio(s: str) -> dict:
    base = s
    start = f"cos(3*{base})/cos({base})"
    target = f"2*cos(2*{base}) - 1"
    ref = [
        start,
        f"(4*cos({base})**3 - 3*cos({base}))/cos({base})",
        f"4*cos({base})**2 - 3",
        f"2*(2*cos({base})**2 - 1) - 1",
        target,
    ]
    return dict(
        key="composite_cos_ratio",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[TRIPLE_COS_GIVEN, COS2_2COS2M1_GIVEN],
        givens_latex=[TRIPLE_COS_GIVEN_LATEX, COS2_2COS2M1_GIVEN_LATEX],
        auth_traps=[("sign_slip_in_triple_angle", f"(4*cos({base})**3 + 3*cos({base}))/cos({base})")],
        wrong_cos2x_index=3,
        wrong_cos2x_line=f"2*(1 - 2*cos({base})**2) - 1",
        divide_condition=f"cos({base}) is not 0",
        divide_condition_latex=rf"\cos {base} \neq 0",
        math_left=lambda v: math.cos(3 * v) / math.cos(v),
        math_right=lambda v: 2 * math.cos(2 * v) - 1,
        identity_latex=lambda bd, dd: rf"\dfrac{{\cos 3{bd}}}{{\cos {bd}}} = 2\cos {dd} - 1",
        domain_all_real=False,
    )


def _fam_composite_ratio_diff(s: str) -> dict:
    base = s
    start = f"sin(3*{base})/sin({base}) - cos(3*{base})/cos({base})"
    target = "2"
    ref = [
        start,
        f"(3*sin({base}) - 4*sin({base})**3)/sin({base}) - (4*cos({base})**3 - 3*cos({base}))/cos({base})",
        f"(3 - 4*sin({base})**2) - (4*cos({base})**2 - 3)",
        f"6 - 4*sin({base})**2 - 4*cos({base})**2",
        f"6 - 4*(sin({base})**2 + cos({base})**2)",
        "6 - 4",
        target,
    ]
    return dict(
        key="composite_ratio_diff",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[TRIPLE_SIN_GIVEN, TRIPLE_COS_GIVEN, PYTH_GIVEN],
        givens_latex=[TRIPLE_SIN_GIVEN_LATEX, TRIPLE_COS_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_traps=[
            (
                "sign_slip_in_triple_angle",
                f"(3*sin({base}) + 4*sin({base})**3)/sin({base}) - (4*cos({base})**3 - 3*cos({base}))/cos({base})",
            )
        ],
        wrong_cos2x_index=None,
        wrong_cos2x_line=None,
        divide_condition=f"sin({base}) is not 0 and cos({base}) is not 0",
        divide_condition_latex=rf"\sin {base} \neq 0 \text{{ and }} \cos {base} \neq 0",
        math_left=lambda v: math.sin(3 * v) / math.sin(v) - math.cos(3 * v) / math.cos(v),
        math_right=lambda v: 2.0,
        identity_latex=lambda bd: rf"\dfrac{{\sin 3{bd}}}{{\sin {bd}}} - \dfrac{{\cos 3{bd}}}{{\cos {bd}}} = 2",
        domain_all_real=False,
    )


EXAM_COMPOSITE_BUILDERS = (_fam_composite_cos_ratio, _fam_composite_ratio_diff)


# ---------------------------------------------------------------------------
# Marking-dict assembly, shared by every prove item. Two template pools:
# `_ALLREAL` for families defined for every real value of the variable,
# `_RESTRICTED` for a family with a division by a trig term, whose prompt
# must not claim "for all real x". Every template ends in
# the shared notation sentence.
# ---------------------------------------------------------------------------

NOTATION_SENTENCE = (
    "Write one expression per line; put brackets round every angle, as in sin(x), and use ^ for powers, as in cos(x)^2."
)
NOTATION_SENTENCE_LATEX = (
    r"\text{Write one expression per line; put brackets round every angle, "
    r"as in sin(x), and use \textasciicircum{} for powers, as in "
    r"cos(x)\textasciicircum{}2.}"
)


def _text_verbatim(s: str) -> str:
    r"""Wrap a typeable-notation string in `\text{...}`, escaping `^` as
    `\textasciicircum{}` (the KaTeX-verified pattern from the sibling
    modules)."""
    return r"\text{" + s.replace("^", r"\textasciicircum{}") + "}"


_PROVE_TEXT_TEMPLATES_ALLREAL = (
    "Prove that {identity}, for all real {var_text}. You may cite, as given: {givens}. "
    "Start from the left-hand side, {start_text}, and finish at {target_text}. {notation}",
    "Show that {identity} for every real value of {var_text}. Given results you may use: {givens}. "
    "Begin your proof with {start_text} and end at {target_text}. {notation}",
    "Using {givens} as given results, prove the identity {identity}, true for all real {var_text}. "
    "Work from {start_text} through to {target_text}. {notation}",
)
_PROVE_TEXT_TEMPLATES_RESTRICTED = (
    "Prove that {identity}, given that {condition}. You may cite, as given: {givens}. "
    "Start from the left-hand side, {start_text}, and finish at {target_text}. {notation}",
    "Show that {identity}, given that {condition}. Given results you may use: {givens}. "
    "Begin your proof with {start_text} and end at {target_text}. {notation}",
    "Using {givens} as given results, prove the identity {identity}, given that {condition}. "
    "Work from {start_text} through to {target_text}. {notation}",
)
_PROVE_LATEX_TEMPLATES_ALLREAL = (
    r"\text{{Prove that }} {identity}, \text{{ for all real }} {var_text}. "
    r"\text{{You may cite, as given: }} {givens}. "
    r"\text{{Start from the left-hand side, }} {start_text}, \text{{ and finish at }} {target_text}. {notation}",
    r"\text{{Show that }} {identity} \text{{ for every real value of }} {var_text}. "
    r"\text{{Given results you may use: }} {givens}. "
    r"\text{{Begin your proof with }} {start_text} \text{{ and end at }} {target_text}. {notation}",
    r"\text{{Using }} {givens} \text{{ as given results, prove the identity }} {identity}, "
    r"\text{{ true for all real }} {var_text}. "
    r"\text{{Work from }} {start_text} \text{{ through to }} {target_text}. {notation}",
)
_PROVE_LATEX_TEMPLATES_RESTRICTED = (
    r"\text{{Prove that }} {identity}, \text{{ given that }} {condition}. "
    r"\text{{You may cite, as given: }} {givens}. "
    r"\text{{Start from the left-hand side, }} {start_text}, \text{{ and finish at }} {target_text}. {notation}",
    r"\text{{Show that }} {identity}, \text{{ given that }} {condition}. "
    r"\text{{Given results you may use: }} {givens}. "
    r"\text{{Begin your proof with }} {start_text} \text{{ and end at }} {target_text}. {notation}",
    r"\text{{Using }} {givens} \text{{ as given results, prove the identity }} {identity}, "
    r"\text{{ given that }} {condition}. "
    r"\text{{Work from }} {start_text} \text{{ through to }} {target_text}. {notation}",
)


def _key_steps(givens: list[str]) -> list[dict]:
    return [
        {
            "criterion_id": "correctness",
            "description": "every line is an exact multiple-angle rearrangement of the previous one",
        },
        {
            "criterion_id": "justified_steps",
            "description": f"cites the given result(s) actually used ({'; '.join(givens)}) rather than re-deriving them",
        },
        {
            "criterion_id": "communication",
            "description": "each line is a clearly legible expression, one per line, ending at the stated target",
        },
    ]


def _prove_traps(fam: dict) -> list[dict]:
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]
    traps = []
    for auth_name, auth_wrong in fam["auth_traps"]:
        traps.append(
            {
                "name": auth_name,
                "question": "does every line follow from a genuine multiple-angle rearrangement of the one before it?",
                "example_lines": [start, auth_wrong, target],
                "expected_check": "authoritative_failure",
            }
        )
    if fam.get("wrong_cos2x_line") is not None:
        idx = fam["wrong_cos2x_index"]
        corrupted = list(ref)
        corrupted[idx] = fam["wrong_cos2x_line"]
        traps.append(
            {
                "name": "wrong_cos2x_form_midway",
                "question": "does the response substitute the correct double- or triple-angle form at each step?",
                "example_lines": corrupted,
                "expected_check": "authoritative_failure",
            }
        )
    traps.append(
        {
            "name": "starts_from_rhs",
            "question": "does the response start from the stated left-hand side?",
            "example_lines": list(reversed(ref)),
            "expected_check": "non_authoritative_failure",
        }
    )
    traps.append(
        {
            "name": "cites_what_it_proves",
            "question": "does the response show the multiple-angle substitution step, or only assert the identity is true?",
            "example_lines": [start, target],
            "expected_check": "passes_chain",
        }
    )
    if fam.get("divide_condition"):
        traps.append(
            {
                "name": "divides_by_zero_unstated",
                "question": f"is {fam['divide_condition']} (needed for the division) stated anywhere in the response?",
                "example_lines": list(ref),
                "expected_check": "passes_chain",
            }
        )
    return traps


def _make_prove_question(fam: dict, var_text: str, var_latex: str, disp_args: tuple, rng: random.Random) -> dict:
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]

    # Cheap in-generator sanity probe: the reference chain's
    # two endpoints agree numerically at 3 points.
    for v in PROBE_POINTS:
        left, right = fam["math_left"](v), fam["math_right"](v)
        check(abs(left - right) < 1e-6, f"{fam['key']}: reference chain does not hold at probe point {v}")

    start_text = _readable_expr_text(start)
    target_text = _readable_expr_text(target)
    identity_text = f"{start_text} = {target_text}"
    identity_latex = fam["identity_latex"](*disp_args)
    givens = fam["givens"]
    givens_latex = fam["givens_latex"]

    if fam["domain_all_real"]:
        text_templates, latex_templates = _PROVE_TEXT_TEMPLATES_ALLREAL, _PROVE_LATEX_TEMPLATES_ALLREAL
        extra = {}
        extra_latex = {}
    else:
        text_templates, latex_templates = _PROVE_TEXT_TEMPLATES_RESTRICTED, _PROVE_LATEX_TEMPLATES_RESTRICTED
        extra = {"condition": fam["divide_condition"]}
        extra_latex = {"condition": fam["divide_condition_latex"]}

    # `text` and `latex` share ONE template choice.
    idx = rng.randrange(len(text_templates))
    text = text_templates[idx].format(
        identity=identity_text,
        var_text=var_text,
        givens="; ".join(givens),
        start_text=start_text,
        target_text=target_text,
        notation=NOTATION_SENTENCE,
        **extra,
    )
    latex = latex_templates[idx].format(
        identity=identity_latex,
        var_text=var_latex,
        givens="; ".join(givens_latex),
        start_text=_text_verbatim(start_text),
        target_text=_text_verbatim(target_text),
        notation=NOTATION_SENTENCE_LATEX,
        **extra_latex,
    )
    marking = {
        "start": start,
        "target": target,
        "symbols": {fam["_var_name"]: "real"},
        "reference_lines": ref,
        "key_steps": _key_steps(givens),
        "traps": _prove_traps(fam),
        "checks": [
            {
                "type": "equivalence_chain",
                "start": start,
                "target": target,
                "symbols": {fam["_var_name"]: "real"},
                "authoritative": True,
            }
        ],
    }
    return dict(text=text, latex=latex, answer=marking, answer_text=identity_text)


_APPLY_TEXT_TEMPLATES = (
    "Evaluate {expr} (that is, {deg}°) exactly, by recognising it as a triple angle. "
    "Give your answer as an exact value. {notation}",
    "Show that {expr} (that is, {deg}°) equals {target_readable} by recognising the triple-angle form, "
    "then evaluating exactly. {notation}",
    "{expr} (that is, {deg}°) is a triple-angle expression. Evaluate it exactly, "
    "giving a surd or fraction where needed. {notation}",
)
_APPLY_LATEX_TEMPLATES = (
    r"\text{{Evaluate }} {expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) exactly, "
    r"by recognising it as a triple angle. Give your answer as an exact value.}} {notation}",
    r"\text{{Show that }} {expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) equals }} {target_latex} "
    r"\text{{ by recognising the triple-angle form, then evaluating exactly.}} {notation}",
    r"{expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) is a triple-angle expression. Evaluate it exactly, "
    r"giving a surd or fraction where needed.}} {notation}",
)


def _apply_math_value(kind: str, v: float) -> float:
    if kind == "cos":
        return 4 * math.cos(v) ** 3 - 3 * math.cos(v)
    return 3 * math.sin(v) - 4 * math.sin(v) ** 3


def _latex_target(readable: str) -> str:
    """`"1/2"` / `"√3/2"` / `"-√2/2"` -> proper LaTeX, for this pool's own
    handful of exact values."""
    neg = readable.startswith("-")
    body = readable[1:] if neg else readable
    if body.startswith("√"):
        num, _, den = body[1:].partition("/")
        out = rf"\dfrac{{\sqrt{{{num}}}}}{{{den}}}"
    else:
        num, _, den = body.partition("/")
        out = rf"\dfrac{{{num}}}{{{den}}}"
    return f"-{out}" if neg else out


def _make_apply_question(kind: str, angle: tuple, rng: random.Random) -> dict:
    fam = _apply_family(kind, angle)
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]
    base_frac = angle[0]

    v = eval_base(base_frac)
    # Cheap in-generator sanity probe: evaluate the stated expression with
    # `math` and compare to the pool's own stored exact value.
    computed = _apply_math_value(kind, v)
    # Robust check: the stored exact value via a tiny table lookup (this
    # pool's own handful of values), never eval'ing sympy text.
    check(
        abs(computed - _stored_float(fam["target"])) < 1e-9,
        f"{fam['key']}: apply target does not match the stored value",
    )

    expr_latex = _to_latex_expr(start)
    target_latex = _latex_target(fam["target_readable"])
    idx = rng.randrange(len(_APPLY_TEXT_TEMPLATES))
    text = _APPLY_TEXT_TEMPLATES[idx].format(
        expr=fam["text_expr"], deg=fam["deg"], target_readable=fam["target_readable"], notation=NOTATION_SENTENCE
    )
    latex = _APPLY_LATEX_TEMPLATES[idx].format(
        expr_latex=expr_latex, deg=fam["deg"], target_latex=target_latex, notation=NOTATION_SENTENCE_LATEX
    )

    wrong_value = f"-({target})"
    traps = [
        {
            "name": "wrong_special_value",
            "question": "is the final numeric value exactly correct, including its sign?",
            "example_lines": ref[:-1] + [wrong_value],
            "expected_check": "non_authoritative_failure",
        },
        {
            "name": "cites_what_it_proves",
            "question": "does the response show the triple-angle recognition, or only assert the value?",
            "example_lines": [start, target],
            "expected_check": "passes_chain",
        },
    ]
    marking = {
        "start": start,
        "target": target,
        "symbols": {},
        "reference_lines": ref,
        "key_steps": [
            {
                "criterion_id": "correctness",
                "description": "recognises the stated expression as a triple-angle form and evaluates it exactly",
            },
            {
                "criterion_id": "justified_steps",
                "description": "names the triple-angle identity used before substituting the special-angle value",
            },
            {
                "criterion_id": "communication",
                "description": "the final line is the single exact value, in simplest surd or fraction form",
            },
        ],
        "traps": traps,
        "checks": [
            {"type": "equivalence_chain", "start": start, "target": target, "symbols": {}, "authoritative": True}
        ],
    }
    answer_text = f"{fam['text_expr']} = {fam['target_readable']}"
    return dict(text=text, latex=latex, answer=marking, answer_text=answer_text)


def _stored_float(sympy_text: str) -> float:
    """A tiny, fixed-form evaluator for this pool's own handful of exact
    value strings (`1/2`, `-1/2`, `sqrt(2)/2`, `-sqrt(2)/2`, `sqrt(3)/2`,
    `-sqrt(3)/2`, `2`) -- never a general expression parser."""
    table = {
        "1/2": 0.5,
        "-1/2": -0.5,
        "sqrt(2)/2": math.sqrt(2) / 2,
        "-sqrt(2)/2": -math.sqrt(2) / 2,
        "sqrt(3)/2": math.sqrt(3) / 2,
        "-sqrt(3)/2": -math.sqrt(3) / 2,
        "2": 2.0,
    }
    return table[sympy_text]


def _make_composite_question(fam: dict, var_text: str, var_latex: str, rng: random.Random) -> dict:
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]
    for v in PROBE_POINTS:
        left, right = fam["math_left"](v), fam["math_right"](v)
        check(abs(left - right) < 1e-6, f"{fam['key']}: reference chain does not hold at probe point {v}")

    start_text = _readable_expr_text(start)
    target_text = _readable_expr_text(target)
    identity_text = f"{start_text} = {target_text}"
    identity_latex = (
        fam["identity_latex"](var_latex, var_latex)
        if fam["key"] == "composite_cos_ratio"
        else fam["identity_latex"](var_latex)
    )
    givens = fam["givens"]
    givens_latex = fam["givens_latex"]

    text_templates, latex_templates = _PROVE_TEXT_TEMPLATES_RESTRICTED, _PROVE_LATEX_TEMPLATES_RESTRICTED
    idx = rng.randrange(len(text_templates))
    text = text_templates[idx].format(
        identity=identity_text,
        var_text=var_text,
        givens="; ".join(givens),
        start_text=start_text,
        target_text=target_text,
        notation=NOTATION_SENTENCE,
        condition=fam["divide_condition"],
    )
    latex = latex_templates[idx].format(
        identity=identity_latex,
        var_text=var_latex,
        givens="; ".join(givens_latex),
        start_text=_text_verbatim(start_text),
        target_text=_text_verbatim(target_text),
        notation=NOTATION_SENTENCE_LATEX,
        condition=fam["divide_condition_latex"],
    )
    marking = {
        "start": start,
        "target": target,
        "symbols": {fam["_var_name"]: "real"},
        "reference_lines": ref,
        "key_steps": _key_steps(givens),
        "traps": _prove_traps(fam),
        "checks": [
            {
                "type": "equivalence_chain",
                "start": start,
                "target": target,
                "symbols": {fam["_var_name"]: "real"},
                "authoritative": True,
            }
        ],
    }
    return dict(text=text, latex=latex, answer=marking, answer_text=identity_text)


def _prove_item(builders: tuple, var_pool: tuple, rng: random.Random) -> dict:
    builder = rng.choice(builders)
    var_name, var_text, var_latex = rng.choice(var_pool)
    fam = builder(var_name)
    fam["_var_name"] = var_name
    return _make_prove_question(fam, var_text, var_latex, (var_latex, var_latex), rng)


def _quad_item(var_pool: tuple, rng: random.Random) -> dict:
    builder = rng.choice(HARD_QUAD_BUILDERS)
    var_name, var_text, var_latex = rng.choice(var_pool)
    m = rng.choice(QUAD_MULTIPLIERS)
    fam = builder(var_name, m)
    fam["_var_name"] = var_name
    base_d, double_d, quad_d = _disp(var_latex, m), _disp(var_latex, 2 * m), _disp(var_latex, 4 * m)
    disp_args = (base_d, quad_d) if builder is not _fam_cos4_power else (base_d, double_d, quad_d)
    return _make_prove_question(fam, var_text, var_latex, disp_args, rng)


def _composite_item(var_pool: tuple, rng: random.Random) -> dict:
    builder = rng.choice(EXAM_COMPOSITE_BUILDERS)
    var_name, var_text, var_latex = rng.choice(var_pool)
    fam = builder(var_name)
    fam["_var_name"] = var_name
    return _make_composite_question(fam, var_text, var_latex, rng)


@generator(
    "pr1a_other_trig_identity_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pr1a_other_trig_identity_proof(rng, variant="easy"):
    """Prove a triple- or quadruple-angle / power-reduction identity (citing
    the angle-sum, double-angle and Pythagorean formulas as given, never
    re-proving them), then apply one to evaluate a stated trigonometric
    expression exactly, or use it in a short composite -- ACMSM049's kept
    contract.
    """
    if variant == "easy":
        if rng.random() < 0.5:
            return _prove_item(EASY_PROVE_BUILDERS, VARS, rng)
        kind = rng.choice(APPLY_KINDS)
        pool = COS_APPLY_POOL if kind == "cos" else SIN_APPLY_POOL
        angle = rng.choice(pool)
        return _make_apply_question(kind, angle, rng)
    if variant == "medium":
        return _prove_item(MEDIUM_PROVE_BUILDERS, VARS, rng)
    if variant == "hard":
        return _quad_item(VARS, rng)
    if variant == "exam":
        if rng.random() < 0.55:
            kind = rng.choice(APPLY_KINDS)
            pool = COS_APPLY_POOL if kind == "cos" else SIN_APPLY_POOL
            angle = rng.choice(pool)
            return _make_apply_question(kind, angle, rng)
        return _composite_item(VARS, rng)
    raise ValueError(f"unknown variant: {variant}")
