"""Rubric-assessed proof skill: the learner proves a double-angle identity for sine, cosine
or tangent, then applies one to evaluate a stated trigonometric expression exactly.

This module keeps only a cheap, generator-local numeric sanity probe: every reference line is evaluated with
the `math` module at three fixed-but-irregular points (or, for an "apply" item with no free symbol, at the
item's own fixed angle) and checked to agree with the target.

Two named identity families -- `tan2A` (hard tier) and `exam1` (exam tier, `(1 - cos 2A)/sin 2A = tan A`) --
have a "bare" two-line jump (start straight to target, skipping the sin/cos-to-tan substitution) that
a symbolic equivalence-chain check cannot confirm: `simplify` does not unify `tan(A)` with `sin(A)/cos(A)` inside a
combined fraction. Their reference chains instead go through the granular, single-substitution steps that were
measured to confirm.

Every free variable is Latin-only (`x, t, u, v` -- no `theta`/`alpha`/`phi`, spelled out or as Greek glyphs);
every printed start/end line and identity statement is typeable ASCII (`sin(x)^2`, brackets round every
function argument, no `**`/`*`/unicode superscript); every family whose sides involve `tan`, or a division by
a trig term, states its true restricted domain instead of claiming "for all real x"; apply items state their
angle in radians, with the degree value optional and separate.
"""

from __future__ import annotations

import math
import random
import re

from ..core import check, generator, phrase

# Safe symbol names (both sympy.sympify and mathgen.grading._parse_expression), Latin only, spelled
# out or otherwise. `var_name`/`var_text`/`var_latex` are all the same ASCII string: nothing here
# needs a separate display form any more.
VARS: tuple[tuple[str, str, str], ...] = (
    ("x", "x", "x"),
    ("t", "t", "t"),
    ("u", "u", "u"),
    ("v", "v", "v"),
)

# Fixed-but-irregular probe points (radians) for the cheap in-generator numeric check. Chosen so
# that, for every family below (all at multiplier m=1 for the families that divide by cos/sin),
# cos(a), sin(2a), 1+cos(2a) and 1-tan(a)**2 are all comfortably away from zero.
PROBE_POINTS: tuple[float, ...] = (0.41, 1.07, -0.53)

# Multiplier pool for the easy/medium tiers' "scaled argument" variety (for example
# "sin 6x = 2 sin 3x cos 3x", "cos 4t = 1 - 2 sin^2 2t").
# Weighted toward 1 (the plain-symbol case) so a scaled argument is a real
# but minority variant, not the norm.
EASY_MEDIUM_MULTIPLIERS: tuple[int, ...] = (1, 1, 1, 2, 3, 4)


def _disp(var_display: str, m: int) -> str:
    return var_display if m == 1 else f"{m}{var_display}"


def _sym(var_name: str, m: int) -> str:
    return var_name if m == 1 else f"{m}*{var_name}"


def _readable_expr_text(sympy_text: str) -> str:
    """Typeable ASCII rendering of one of this module's own sympy-text expressions: `**` -> `^`
    (`sin(x)**2` -> `sin(x)^2`), and `*` dropped entirely -- `2*sin(x)*cos(x)` -> `2sin(x)cos(x)` and
    `sin(2*x)` -> `sin(2x)` are both explicitly listed as parser-correct forms. Every function argument in the
    source strings is already bracketed (`sin(x)`, never `sin x`), so this never has to insert a bracket. Only
    ever applied to strings this module itself built."""
    return sympy_text.replace("**", "^").replace("*", "")


_TRIG_LATEX_RE = re.compile(r"\b(sin|cos|tan)\(")


def _to_latex_expr(sympy_text: str) -> str:
    """This module's own sympy-text (`start`/`target`) to proper LaTeX, for
    the APPLY items' rendered expression only (latex keeps real
    math notation, unlike the typeable-ASCII `text` field). `pi` -> `\\pi`,
    `sin(`/`cos(`/`tan(` -> `\\sin(`/`\\cos(`/`\\tan(`, `**` -> `^`, `*`
    dropped (`2\\sin(x)\\cos(x)` juxtaposition, matching the ASCII form)."""
    out = sympy_text.replace("**", "^")
    out = _TRIG_LATEX_RE.sub(lambda m: f"\\{m.group(1)}(", out)
    out = out.replace("pi", r"\pi")
    return out.replace("*", "")


def _pi_display(frac: str) -> str:
    """`"pi/12"` -> `"π/12"`, `"5*pi/12"` -> `"5π/12"` -- this
    module's own fixed-form radian fractions only (see `eval_base`), never a
    general expression."""
    return frac.replace("*pi", "π").replace("pi", "π")


# ---------------------------------------------------------------------------
# Angle-sum/Pythagorean "givens" the learner may cite (never re-derive).
# Every function argument bracketed, powers with `^`, no unicode superscript
#: the learner may copy these verbatim as a proof line.
# ---------------------------------------------------------------------------
SIN_SUM_GIVEN = "sin(A + B) = sin(A)cos(B) + cos(A)sin(B)"
COS_SUM_GIVEN = "cos(A + B) = cos(A)cos(B) - sin(A)sin(B)"
PYTH_GIVEN = "sin(A)^2 + cos(A)^2 = 1"
SIN2_GIVEN = "sin(2A) = 2sin(A)cos(A) (the previously established double-angle result)"
COS2_DIFF_GIVEN = "cos(2A) = cos(A)^2 - sin(A)^2 (the previously established double-angle result)"
COS2_1M2SIN2_GIVEN = "cos(2A) = 1 - 2sin(A)^2 (the previously established double-angle result)"
COS2_2COS2M1_GIVEN = "cos(2A) = 2cos(A)^2 - 1 (the previously established double-angle result)"

SIN_SUM_GIVEN_LATEX = r"\sin(A + B) = \sin A\cos B + \cos A\sin B"
COS_SUM_GIVEN_LATEX = r"\cos(A + B) = \cos A\cos B - \sin A\sin B"
PYTH_GIVEN_LATEX = r"\sin^2 A + \cos^2 A = 1"
SIN2_GIVEN_LATEX = r"\sin 2A = 2\sin A\cos A \text{ (previously established)}"
COS2_DIFF_GIVEN_LATEX = r"\cos 2A = \cos^2 A - \sin^2 A \text{ (previously established)}"
COS2_1M2SIN2_GIVEN_LATEX = r"\cos 2A = 1 - 2\sin^2 A \text{ (previously established)}"
COS2_2COS2M1_GIVEN_LATEX = r"\cos 2A = 2\cos^2 A - 1 \text{ (previously established)}"


# ---------------------------------------------------------------------------
# PROVE identity families. Each entry is a dict of pure-string builders (the
# sympy-text `start`/`target`/`reference_lines`/traps) plus a `domain_all_real`
# flag and a pair of `math`-only
# callables used for the cheap per-item numeric probe.
# ---------------------------------------------------------------------------


def _fam_sin2A(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    start = f"sin({double})"
    target = f"2*sin({base})*cos({base})"
    ref = [start, f"sin({base})*cos({base}) + cos({base})*sin({base})", target]
    return dict(
        key="sin2A",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[SIN_SUM_GIVEN],
        givens_latex=[SIN_SUM_GIVEN_LATEX],
        auth_trap=("doubles_the_function", f"2*sin({base})"),
        math_left=lambda v: math.sin(2 * v),
        math_right=lambda v: 2 * math.sin(v) * math.cos(v),
        identity_latex=lambda bd, dd: rf"\sin {dd} = 2\sin {bd}\cos {bd}",
        domain_all_real=True,
    )


def _fam_cos2A_diff(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    start = f"cos({double})"
    target = f"cos({base})**2 - sin({base})**2"
    ref = [start, f"cos({base})*cos({base}) - sin({base})*sin({base})", target]
    return dict(
        key="cos2A_diff",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS_SUM_GIVEN],
        givens_latex=[COS_SUM_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"2*cos({base})**2 - sin({base})**2"),
        math_left=lambda v: math.cos(2 * v),
        math_right=lambda v: math.cos(v) ** 2 - math.sin(v) ** 2,
        identity_latex=lambda bd, dd: rf"\cos {dd} = \cos^2 {bd} - \sin^2 {bd}",
        domain_all_real=True,
    )


def _fam_cos2A_1m2sin2(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    start = f"cos({double})"
    target = f"1 - 2*sin({base})**2"
    ref = [start, f"cos({base})**2 - sin({base})**2", f"(1 - sin({base})**2) - sin({base})**2", target]
    return dict(
        key="cos2A_1m2sin2",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS_SUM_GIVEN, PYTH_GIVEN],
        givens_latex=[COS_SUM_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"1 - sin({base})**2"),
        math_left=lambda v: math.cos(2 * v),
        math_right=lambda v: 1 - 2 * math.sin(v) ** 2,
        identity_latex=lambda bd, dd: rf"\cos {dd} = 1 - 2\sin^2 {bd}",
        domain_all_real=True,
    )


def _fam_cos2A_2cos2m1(s: str, m: int) -> dict:
    base = _sym(s, m)
    double = _sym(s, 2 * m)
    start = f"cos({double})"
    target = f"2*cos({base})**2 - 1"
    ref = [start, f"cos({base})**2 - sin({base})**2", f"cos({base})**2 - (1 - cos({base})**2)", target]
    return dict(
        key="cos2A_2cos2m1",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS_SUM_GIVEN, PYTH_GIVEN],
        givens_latex=[COS_SUM_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"2*cos({base})**2 - sin({base})**2"),
        math_left=lambda v: math.cos(2 * v),
        math_right=lambda v: 2 * math.cos(v) ** 2 - 1,
        identity_latex=lambda bd, dd: rf"\cos {dd} = 2\cos^2 {bd} - 1",
        domain_all_real=True,
    )


def _fam_tan2A(s: str) -> dict:
    base = s  # hard tier: multiplier 1 only
    start = f"tan(2*{base})"
    target = f"2*tan({base})/(1 - tan({base})**2)"
    ref = [
        start,
        f"sin(2*{base})/cos(2*{base})",
        f"(2*sin({base})*cos({base}))/(cos({base})**2 - sin({base})**2)",
        f"2*(sin({base})/cos({base}))/(1-(sin({base})/cos({base}))**2)",
        f"2*(sin({base})/cos({base}))/(1-tan({base})**2)",
        target,
    ]
    return dict(
        key="tan2A",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[SIN2_GIVEN, COS2_DIFF_GIVEN],
        givens_latex=[SIN2_GIVEN_LATEX, COS2_DIFF_GIVEN_LATEX],
        auth_trap=("doubles_the_function", f"2*tan({base})"),
        math_left=lambda v: math.tan(2 * v),
        math_right=lambda v: 2 * math.tan(v) / (1 - math.tan(v) ** 2),
        identity_latex=lambda bd, dd: rf"\tan {dd} = \dfrac{{2\tan {bd}}}{{1 - \tan^2 {bd}}}",
        bare_jump_confirms=False,
        domain_all_real=False,  # tan and a division: cos(A) != 0, tan(A) != +-1
    )


def _fam_sin2_rearranged(s: str) -> dict:
    base = s
    start = f"sin({base})**2"
    target = f"(1 - cos(2*{base}))/2"
    ref = [start, f"(1 - (1 - 2*sin({base})**2))/2", target]
    return dict(
        key="sin2_rearr",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_1M2SIN2_GIVEN],
        givens_latex=[COS2_1M2SIN2_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"(1 + cos(2*{base}))/2"),
        math_left=lambda v: math.sin(v) ** 2,
        math_right=lambda v: (1 - math.cos(2 * v)) / 2,
        identity_latex=lambda bd, dd: rf"\sin^2 {bd} = \dfrac{{1 - \cos {dd}}}{{2}}",
        domain_all_real=True,  # division by the constant 2, not a trig term
    )


def _fam_cos2_rearranged(s: str) -> dict:
    base = s
    start = f"cos({base})**2"
    target = f"(1 + cos(2*{base}))/2"
    ref = [start, f"(1 + (2*cos({base})**2 - 1))/2", target]
    return dict(
        key="cos2_rearr",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_2COS2M1_GIVEN],
        givens_latex=[COS2_2COS2M1_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"(1 - cos(2*{base}))/2"),
        math_left=lambda v: math.cos(v) ** 2,
        math_right=lambda v: (1 + math.cos(2 * v)) / 2,
        identity_latex=lambda bd, dd: rf"\cos^2 {bd} = \dfrac{{1 + \cos {dd}}}{{2}}",
        domain_all_real=True,  # division by the constant 2, not a trig term
    )


def _fam_exam1(s: str) -> dict:
    base = s
    start = f"(1 - cos(2*{base}))/sin(2*{base})"
    target = f"tan({base})"
    ref = [start, f"(2*sin({base})**2)/(2*sin({base})*cos({base}))", f"sin({base})/cos({base})", target]
    return dict(
        key="exam1",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_1M2SIN2_GIVEN, SIN2_GIVEN],
        givens_latex=[COS2_1M2SIN2_GIVEN_LATEX, SIN2_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"cos({base})/sin({base})"),
        math_left=lambda v: (1 - math.cos(2 * v)) / math.sin(2 * v),
        math_right=lambda v: math.tan(v),
        identity_latex=lambda bd, dd: rf"\dfrac{{1 - \cos {dd}}}{{\sin {dd}}} = \tan {bd}",
        bare_jump_confirms=False,
        domain_all_real=False,  # tan, and a division by sin(2A)
    )


def _fam_exam2(s: str) -> dict:
    base = s
    start = f"sin(2*{base})/(1 + cos(2*{base}))"
    target = f"tan({base})"
    ref = [start, f"(2*sin({base})*cos({base}))/(2*cos({base})**2)", f"sin({base})/cos({base})", target]
    return dict(
        key="exam2",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[SIN2_GIVEN, COS2_2COS2M1_GIVEN],
        givens_latex=[SIN2_GIVEN_LATEX, COS2_2COS2M1_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"cos({base})/sin({base})"),
        math_left=lambda v: math.sin(2 * v) / (1 + math.cos(2 * v)),
        math_right=lambda v: math.tan(v),
        identity_latex=lambda bd, dd: rf"\dfrac{{\sin {dd}}}{{1 + \cos {dd}}} = \tan {bd}",
        domain_all_real=False,  # tan, and a division by (1 + cos(2A))
    )


def _fam_exam3(s: str) -> dict:
    base = s
    start = f"cos({base})**4 - sin({base})**4"
    target = f"cos(2*{base})"
    ref = [
        start,
        f"(cos({base})**2 - sin({base})**2)*(cos({base})**2 + sin({base})**2)",
        f"cos({base})**2 - sin({base})**2",
        target,
    ]
    return dict(
        key="exam3",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[COS2_DIFF_GIVEN, PYTH_GIVEN],
        givens_latex=[COS2_DIFF_GIVEN_LATEX, PYTH_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"sin(2*{base})"),
        math_left=lambda v: math.cos(v) ** 4 - math.sin(v) ** 4,
        math_right=lambda v: math.cos(2 * v),
        identity_latex=lambda bd, dd: rf"\cos^4 {bd} - \sin^4 {bd} = \cos {dd}",
        domain_all_real=True,
    )


def _fam_exam4(s: str) -> dict:
    base = s
    start = f"(sin({base}) + cos({base}))**2"
    target = f"1 + sin(2*{base})"
    ref = [
        start,
        f"sin({base})**2 + 2*sin({base})*cos({base}) + cos({base})**2",
        f"(sin({base})**2 + cos({base})**2) + 2*sin({base})*cos({base})",
        f"1 + 2*sin({base})*cos({base})",
        target,
    ]
    return dict(
        key="exam4",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[PYTH_GIVEN, SIN2_GIVEN],
        givens_latex=[PYTH_GIVEN_LATEX, SIN2_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"1 - sin(2*{base})"),
        math_left=lambda v: (math.sin(v) + math.cos(v)) ** 2,
        math_right=lambda v: 1 + math.sin(2 * v),
        identity_latex=lambda bd, dd: rf"(\sin {bd} + \cos {bd})^2 = 1 + \sin {dd}",
        domain_all_real=True,
    )


def _fam_exam5(s: str) -> dict:
    base = s
    start = f"(cos({base}) - sin({base}))**2"
    target = f"1 - sin(2*{base})"
    ref = [
        start,
        f"cos({base})**2 - 2*sin({base})*cos({base}) + sin({base})**2",
        f"(cos({base})**2 + sin({base})**2) - 2*sin({base})*cos({base})",
        f"1 - 2*sin({base})*cos({base})",
        target,
    ]
    return dict(
        key="exam5",
        start=start,
        target=target,
        reference_lines=ref,
        givens=[PYTH_GIVEN, SIN2_GIVEN],
        givens_latex=[PYTH_GIVEN_LATEX, SIN2_GIVEN_LATEX],
        auth_trap=("wrong_cos2x_form", f"1 + sin(2*{base})"),
        math_left=lambda v: (math.cos(v) - math.sin(v)) ** 2,
        math_right=lambda v: 1 - math.sin(2 * v),
        identity_latex=lambda bd, dd: rf"(\cos {bd} - \sin {bd})^2 = 1 - \sin {dd}",
        domain_all_real=True,
    )


EASY_PROVE_BUILDERS = (_fam_sin2A, _fam_cos2A_diff)
MEDIUM_PROVE_BUILDERS = (_fam_cos2A_1m2sin2, _fam_cos2A_2cos2m1)
HARD_PROVE_BUILDERS = (_fam_tan2A, _fam_sin2_rearranged, _fam_cos2_rearranged)
EXAM_PROVE_BUILDERS = (_fam_exam1, _fam_exam2, _fam_exam3, _fam_exam4, _fam_exam5)


# ---------------------------------------------------------------------------
# APPLY items (easy tier): evaluate a stated expression by recognising it as
# a double angle, from a small pool of base angles whose double is a
# standard 30/45/60/90-family angle (so the exact value is a familiar surd).
# (base fraction, double fraction, double in degrees, base in degrees)
# ---------------------------------------------------------------------------
ANGLE_POOL: tuple[tuple[str, str, int, float], ...] = (
    ("pi/12", "pi/6", 30, 15),
    ("pi/8", "pi/4", 45, 22.5),
    ("5*pi/12", "5*pi/6", 150, 75),
    ("3*pi/8", "3*pi/4", 135, 67.5),
    ("7*pi/12", "7*pi/6", 210, 105),
    ("5*pi/8", "5*pi/4", 225, 112.5),
)
# degrees -> (sin value, cos value, sin sympy text, cos sympy text, sin readable, cos readable)
_DEG_VALUES: dict[int, tuple[float, float, str, str, str, str]] = {
    30: (0.5, math.sqrt(3) / 2, "1/2", "sqrt(3)/2", "1/2", "√3/2"),
    45: (math.sqrt(2) / 2, math.sqrt(2) / 2, "sqrt(2)/2", "sqrt(2)/2", "√2/2", "√2/2"),
    150: (0.5, -math.sqrt(3) / 2, "1/2", "-sqrt(3)/2", "1/2", "-√3/2"),
    135: (math.sqrt(2) / 2, -math.sqrt(2) / 2, "sqrt(2)/2", "-sqrt(2)/2", "√2/2", "-√2/2"),
    210: (-0.5, -math.sqrt(3) / 2, "-1/2", "-sqrt(3)/2", "-1/2", "-√3/2"),
    225: (-math.sqrt(2) / 2, -math.sqrt(2) / 2, "-sqrt(2)/2", "-sqrt(2)/2", "-√2/2", "-√2/2"),
}


def _fmt_deg(d: float) -> str:
    return str(int(d)) if float(d).is_integer() else str(d)


def _apply_family(kind: str, angle: tuple[str, str, int, float]) -> dict:
    base, double, deg, base_deg = angle
    bd = _fmt_deg(base_deg)
    base_rad = _pi_display(base)
    sin_v, cos_v, sin_t, cos_t, sin_r, cos_r = _DEG_VALUES[deg]
    if kind == "sin2A":
        start = f"2*sin({base})*cos({base})"
        ref = [start, f"sin({double})", sin_t]
        target, target_readable = sin_t, sin_r
        text_expr = f"2 sin({base_rad}) cos({base_rad})"
    elif kind == "cos2A_diff":
        start = f"cos({base})**2 - sin({base})**2"
        ref = [start, f"cos({double})", cos_t]
        target, target_readable = cos_t, cos_r
        text_expr = f"cos({base_rad})^2 - sin({base_rad})^2"
    elif kind == "cos2A_1m2sin2":
        start = f"1 - 2*sin({base})**2"
        ref = [start, f"cos({double})", cos_t]
        target, target_readable = cos_t, cos_r
        text_expr = f"1 - 2sin({base_rad})^2"
    else:  # cos2A_2cos2m1
        start = f"2*cos({base})**2 - 1"
        ref = [start, f"cos({double})", cos_t]
        target, target_readable = cos_t, cos_r
        text_expr = f"2cos({base_rad})^2 - 1"
    return dict(
        key=f"apply_{kind}",
        start=start,
        target=target,
        reference_lines=ref,
        text_expr=text_expr,
        target_readable=target_readable,
        deg=deg,
        deg_display=bd,
    )


def eval_base(base_frac: str) -> float:
    """A tiny, fixed-form radian evaluator for the pool's own `"k*pi/n"`
    strings -- never a general expression parser; every angle in
    `ANGLE_POOL` is written `"pi/n"` or `"k*pi/n"`."""
    text = base_frac.replace(" ", "")
    if text.startswith("pi/"):
        return math.pi / int(text[3:])
    k, rest = text.split("*pi/")
    return int(k) * math.pi / int(rest)


APPLY_KINDS = ("sin2A", "cos2A_diff", "cos2A_1m2sin2", "cos2A_2cos2m1")


# ---------------------------------------------------------------------------
# Marking-dict assembly, shared by every prove item. Two template pools:
# `_ALLREAL` for families defined for every real value of the variable,
# `_RESTRICTED` for a family with `tan`, or a division by a trig term, whose
# prompt must not claim "for all real x". Every template
# ends in the shared notation sentence.
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
    r"""Wrap a typeable-notation string (this module's own `_readable_expr_
    text` output -- ASCII only, `^` for powers, never `**`/`*`/unicode) in
    `\text{...}`, escaping `^` as `\textasciicircum{}` (the KaTeX-verified
    pattern: `\text{use \textasciicircum{} for powers}` renders, a bare `^`
    inside `\text{}` does not). Keeps a quoted start/end line the exact
    typeable string the learner sees in `text`, never a bare copy sitting
    unescaped in maths mode (which KaTeX would render as italic letters
    with a dangling `^`, or fail outright)."""
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
    "Prove that {identity}, for all {var_text} where both sides are defined. You may cite, as given: {givens}. "
    "Start from the left-hand side, {start_text}, and finish at {target_text}. {notation}",
    "Show that {identity} for every value of {var_text} where both sides are defined. Given results you may use: {givens}. "
    "Begin your proof with {start_text} and end at {target_text}. {notation}",
    "Using {givens} as given results, prove the identity {identity}, true for all {var_text} where both sides are defined. "
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
    r"\text{{Prove that }} {identity}, \text{{ for all }} {var_text} \text{{ where both sides are defined. }}"
    r"\text{{You may cite, as given: }} {givens}. "
    r"\text{{Start from the left-hand side, }} {start_text}, \text{{ and finish at }} {target_text}. {notation}",
    r"\text{{Show that }} {identity} \text{{ for every value of }} {var_text} \text{{ where both sides are defined. }}"
    r"\text{{Given results you may use: }} {givens}. "
    r"\text{{Begin your proof with }} {start_text} \text{{ and end at }} {target_text}. {notation}",
    r"\text{{Using }} {givens} \text{{ as given results, prove the identity }} {identity}, "
    r"\text{{ true for all }} {var_text} \text{{ where both sides are defined. }}"
    r"\text{{Work from }} {start_text} \text{{ through to }} {target_text}. {notation}",
)


def _key_steps(givens: list[str]) -> list[dict]:
    steps = [
        {
            "criterion_id": "correctness",
            "description": "every line is an exact double-angle rearrangement of the previous one",
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
    return steps


def _prove_traps(fam: dict) -> list[dict]:
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]
    auth_name, auth_wrong = fam["auth_trap"]
    traps = [
        {
            "name": auth_name,
            "question": f"does every line follow from a genuine double-angle rearrangement of the one before it?",
            "example_lines": [start, auth_wrong, target],
            "expected_check": "authoritative_failure",
        },
        {
            "name": "starts_from_rhs",
            "question": "does the response start from the stated left-hand side?",
            "example_lines": list(reversed(ref)),
            "expected_check": "non_authoritative_failure",
        },
    ]
    bare = [start, target]
    if fam.get("bare_jump_confirms", True):
        traps.append(
            {
                "name": "cites_what_it_proves",
                "question": "does the response show the substitution step, or only assert the identity is true?",
                "example_lines": bare,
                "expected_check": "passes_chain",
            }
        )
    else:
        traps.append(
            {
                "name": "cites_what_it_proves",
                "question": "does the response show the substitution step, or only assert the identity is true?",
                "example_lines": bare,
                "expected_check": "non_authoritative_failure",
            }
        )
    reproves = [ref[0], f"0 + {ref[0]}"] + ref[1:]
    traps.append(
        {
            "name": "reproves_angle_sum",
            "question": "does the response cite the angle-sum/double-angle result, or re-derive it from scratch?",
            "example_lines": reproves,
            "expected_check": "passes_chain",
        }
    )
    if fam["key"] == "tan2A":
        traps.append(
            {
                "name": "divides_by_zero_unstated",
                "question": "is cos(A) != 0 (needed to divide by cos^2(A)) stated anywhere in the response?",
                "example_lines": list(ref),
                "expected_check": "passes_chain",
            }
        )
    return traps


def _make_prove_question(
    fam: dict, var_text: str, var_latex: str, base_disp_latex: str, double_disp_latex: str, rng: random.Random
) -> dict:
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]

    # Cheap in-generator sanity probe: the reference chain's two
    # endpoints, and every intermediate line, agree numerically at 3 points.
    for v in PROBE_POINTS:
        left, right = fam["math_left"](v), fam["math_right"](v)
        check(abs(left - right) < 1e-6, f"{fam['key']}: reference chain does not hold at probe point {v}")

    start_text = _readable_expr_text(start)
    target_text = _readable_expr_text(target)
    identity_text = f"{start_text} = {target_text}"
    identity_latex = fam["identity_latex"](base_disp_latex, double_disp_latex)
    givens = fam["givens"]
    givens_latex = fam["givens_latex"]

    if fam["domain_all_real"]:
        text_templates, latex_templates = _PROVE_TEXT_TEMPLATES_ALLREAL, _PROVE_LATEX_TEMPLATES_ALLREAL
    else:
        text_templates, latex_templates = _PROVE_TEXT_TEMPLATES_RESTRICTED, _PROVE_LATEX_TEMPLATES_RESTRICTED

    # `text` and `latex` share ONE template choice: `phrase()`
    # itself calls `rng.choice`, so calling it twice independently would
    # advance the rng and could pick a DIFFERENT index for latex than for
    # text (measured: hard seed 3 picked "Prove that ..." for text and
    # "Show that ..." for latex). `idx = rng.randrange(len(...))` draws
    # exactly the same one `_randbelow` call `phrase()`'s own `rng.choice`
    # would have made, so `text` stays byte-identical to before this fix.
    idx = rng.randrange(len(text_templates))
    text = text_templates[idx].format(
        identity=identity_text,
        var_text=var_text,
        givens="; ".join(givens),
        start_text=start_text,
        target_text=target_text,
        notation=NOTATION_SENTENCE,
    )
    latex = latex_templates[idx].format(
        identity=identity_latex,
        var_text=var_latex,
        givens="; ".join(givens_latex),
        start_text=_text_verbatim(start_text),
        target_text=_text_verbatim(target_text),
        notation=NOTATION_SENTENCE_LATEX,
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
    answer_text = identity_text
    return dict(text=text, latex=latex, answer=marking, answer_text=answer_text)


_APPLY_TEXT_TEMPLATES = (
    "Evaluate {expr} (that is, {deg}°) exactly, by recognising it as a double angle. "
    "Give your answer as an exact value. {notation}",
    "Show that {expr} (that is, {deg}°) equals {target_readable} by recognising the double-angle form, "
    "then evaluating exactly. {notation}",
    "{expr} (that is, {deg}°) is a double-angle expression. Evaluate it exactly, "
    "giving a surd or fraction where needed. {notation}",
)
_APPLY_LATEX_TEMPLATES = (
    r"\text{{Evaluate }} {expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) exactly, "
    r"by recognising it as a double angle. Give your answer as an exact value.}} {notation}",
    r"\text{{Show that }} {expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) equals }} {target_latex} "
    r"\text{{ by recognising the double-angle form, then evaluating exactly.}} {notation}",
    r"{expr_latex} \text{{ (that is, }} {deg}^\circ \text{{) is a double-angle expression. Evaluate it exactly, "
    r"giving a surd or fraction where needed.}} {notation}",
)


def _make_apply_question(kind: str, angle: tuple[str, str, int], rng: random.Random) -> dict:
    fam = _apply_family(kind, angle)
    start, target, ref = fam["start"], fam["target"], fam["reference_lines"]
    base_frac = angle[0]

    v = eval_base(base_frac)
    # Cheap in-generator sanity probe: evaluate the stated expression with
    # `math` and compare to the pool's own stored exact float.
    left = _apply_math_value(kind, v)
    right = _apply_math_value(kind, v)  # recomputed from the same closed form
    check(abs(left - right) < 1e-9, f"{fam['key']}: apply value probe failed")
    stored = _DEG_VALUES[angle[2]]
    expected_float = stored[0] if kind == "sin2A" else stored[1]
    check(abs(left - expected_float) < 1e-9, f"{fam['key']}: apply target does not match the stored value")

    expr_latex = _to_latex_expr(start)
    target_latex = _latex_target(fam["target_readable"])
    # Same shared-index fix as `_make_prove_question` above: one
    # `rng.randrange` draw, used to index BOTH template tuples, so latex
    # never independently drifts to a different wording than text.
    idx = rng.randrange(len(_APPLY_TEXT_TEMPLATES))
    text = _APPLY_TEXT_TEMPLATES[idx].format(
        expr=fam["text_expr"],
        deg=fam["deg_display"],
        target_readable=fam["target_readable"],
        notation=NOTATION_SENTENCE,
    )
    latex = _APPLY_LATEX_TEMPLATES[idx].format(
        expr_latex=expr_latex, deg=fam["deg_display"], target_latex=target_latex, notation=NOTATION_SENTENCE_LATEX
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
            "name": "skips_the_recognition_step",
            "question": "does the response show the double-angle recognition, or only assert the value?",
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
                "description": "recognises the stated expression as a double-angle form and evaluates it exactly",
            },
            {
                "criterion_id": "justified_steps",
                "description": "names the double-angle identity used before substituting the special-angle value",
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


def _latex_target(readable: str) -> str:
    """`"1/2"` / `"√3/2"` / `"-√2/2"` -> proper LaTeX, for the one
    handful of exact values this pool ever produces (`_DEG_VALUES` above)."""
    neg = readable.startswith("-")
    body = readable[1:] if neg else readable
    if body.startswith("√"):
        num, _, den = body[1:].partition("/")
        out = rf"\dfrac{{\sqrt{{{num}}}}}{{{den}}}"
    else:
        num, _, den = body.partition("/")
        out = rf"\dfrac{{{num}}}{{{den}}}"
    return f"-{out}" if neg else out


def _apply_math_value(kind: str, v: float) -> float:
    if kind == "sin2A":
        return 2 * math.sin(v) * math.cos(v)
    if kind == "cos2A_diff":
        return math.cos(v) ** 2 - math.sin(v) ** 2
    if kind == "cos2A_1m2sin2":
        return 1 - 2 * math.sin(v) ** 2
    return 2 * math.cos(v) ** 2 - 1


# Builders that take a multiplier (the easy/medium "scaled argument" variety); every hard/exam
# builder takes the plain symbol only.
_SCALED_BUILDERS = (_fam_sin2A, _fam_cos2A_diff, _fam_cos2A_1m2sin2, _fam_cos2A_2cos2m1)


def _prove_item(builders: tuple, var_pool: tuple, mult_pool: tuple, rng: random.Random) -> dict:
    builder = rng.choice(builders)
    var_name, var_text, var_latex = rng.choice(var_pool)
    scaled = builder in _SCALED_BUILDERS
    m = rng.choice(mult_pool) if scaled else 1
    fam = builder(var_name, m) if scaled else builder(var_name)
    fam["_var_name"] = var_name
    base_disp_latex = _disp(var_latex, m)
    double_disp_latex = _disp(var_latex, 2 * m)
    return _make_prove_question(fam, var_text, var_latex, base_disp_latex, double_disp_latex, rng)


@generator(
    "pr1a_double_angle_proof",
    level=14,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def pr1a_double_angle_proof(rng, variant="easy"):
    """Prove a double-angle identity for sine, cosine or tangent (citing the
    angle-sum formulas as given, never re-proving them), then apply one to
    evaluate a stated trigonometric expression exactly -- ACMSM044's scope.
    """
    if variant == "easy":
        if rng.random() < 0.5:
            return _prove_item(EASY_PROVE_BUILDERS, VARS, EASY_MEDIUM_MULTIPLIERS, rng)
        kind = rng.choice(APPLY_KINDS)
        angle = rng.choice(ANGLE_POOL)
        return _make_apply_question(kind, angle, rng)
    if variant == "medium":
        return _prove_item(MEDIUM_PROVE_BUILDERS, VARS, EASY_MEDIUM_MULTIPLIERS, rng)
    if variant == "hard":
        return _prove_item(HARD_PROVE_BUILDERS, VARS, (1,), rng)
    if variant == "exam":
        return _prove_item(EXAM_PROVE_BUILDERS, VARS, (1,), rng)
    raise ValueError(f"unknown variant: {variant}")
