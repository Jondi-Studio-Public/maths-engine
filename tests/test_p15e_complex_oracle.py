"""Independent oracle for the p15e complex-analysis questions, such as the contour integral.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p15e_complex_build`) and re-derives each answer from
scratch:

- the contour integral by parsing the denominator f(z) = c / (...) directly
  out of the prompt (whatever its exact algebraic shape -- "z^2 + 16",
  "(z - 1)^2 + 16", "z^2 - 4z + 20" are all valid sympy expressions once
  "^" -> "**" and implicit "4z" -> "4*z"), solving it for its roots with
  `sympy.solve` (never the generator's own quotient-rule formula), computing
  the residue at the enclosed root via the direct limit definition
  `lim_{z->z0} (z-z0) f(z)` with `sympy.limit` (a genuinely different and
  much slower algorithm than the generator's closed form), and identifying
  the enclosed root either from the pole the prompt states explicitly
  (easy/medium/hard) or, at the exam tier -- which states NO pole, only
  that the contour lies in the upper half-plane -- from Im(z) > 0 alone,
  exactly as a learner would have to.
- the Cauchy-Riemann class via `sympy.diff` on u, v either parsed straight
  out of the prompt (easy/medium/hard, which state u(x,y)/v(x,y) directly)
  or, at the exam tier -- which states only f(z) in z and conj(z) -- by
  parsing that formula and expanding it with z = x+iy, conj(z) = x-iy
  substituted in, via `sympy.expand(..., complex=True)` (the same
  substitution a learner would do by hand, done here independently of the
  generator's own kind-based algebra).
"""

import re
from dataclasses import replace

import pytest
import sympy as sp
from sympy import I

import mathgen
from mathgen.skills import load_skills

CONTOUR_ID = "ucv1-contour-integral-residue"
CR_ID = "ucv1-cr-equations-check"
SEEDS = range(60)

_Z = sp.symbols("z")
_X, _Y = sp.symbols("x y", real=True)


def _desanitize(s: str) -> str:
    return s.replace("^", "**").replace("·", "*")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


CONTOUR_SOURCES = _skill_sources(CONTOUR_ID)
CR_SOURCES = _skill_sources(CR_ID)
CR_SOURCES["exam"] = replace(CR_SOURCES["exam"], topic="p15e_cr_equations_check", params={})


# ==========================================================================
# ucv1-contour-integral-residue
# ==========================================================================

_C_RE = re.compile(r"f\(z\) = (-?\d+) /")


def _extract_denominator(text: str) -> str:
    """The content of the single balanced `(...)` following "f(z) = C / ",
    whatever its internal shape -- handles the medium tier's own nested
    parens ("(z - 1)^2 + 16") without assuming a fixed number of them."""
    marker = "/ ("
    idx = text.index(marker) + len(marker) - 1  # index of the opening '('
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[idx + 1 : i]
    raise AssertionError(f"unbalanced parens in: {text}")


def _to_sympy_poly(token: str):
    token = token.replace("^", "**")
    token = re.sub(r"(\d)(z)", r"\1*\2", token)
    return sp.sympify(token, locals={"z": _Z})


_POLE_RE = re.compile(r"z = (-?\d+ \+ \d*i|\d*i)\b")


def _to_complex_pole(token: str):
    token = token.strip()
    m = re.match(r"^(-?\d+)\s*\+\s*(\d*)i$", token)
    if m:
        p = int(m.group(1))
        k = int(m.group(2)) if m.group(2) else 1
        return p + k * I
    m = re.match(r"^(\d*)i$", token)
    if m:
        k = int(m.group(1)) if m.group(1) else 1
        return k * I
    raise AssertionError(f"could not parse pole token: {token!r}")


def _enclosed_root(text: str, roots):
    if "upper half-plane" in text:
        # Exam tier: no pole is stated at all -- the learner (and this
        # oracle) must derive both roots and keep the one with Im(z) > 0.
        chosen = [r for r in roots if sp.im(r) > 0]
        assert len(chosen) == 1, f"expected exactly one root with Im(z) > 0 among {roots}"
        return chosen[0]
    idx = text.index("enclos")
    m = _POLE_RE.search(text, idx)
    assert m, f"could not find an explicit enclosed pole after 'enclos' in: {text}"
    stated = _to_complex_pole(m.group(1))
    chosen = [r for r in roots if sp.simplify(r - stated) == 0]
    assert len(chosen) == 1, f"stated pole {stated} does not match exactly one root of {roots}"
    return chosen[0]


def _solve_contour(text: str):
    c_m = _C_RE.search(text)
    assert c_m, f"could not parse the numerator c from: {text}"
    c = int(c_m.group(1))
    denom = _to_sympy_poly(_extract_denominator(text))
    roots = sp.solve(denom, _Z)
    assert len(roots) == 2, f"expected a quadratic with two roots, got {roots} from: {text}"
    z0 = _enclosed_root(text, roots)
    f = c / denom
    residue = sp.simplify(sp.limit((_Z - z0) * f, _Z, z0))
    return sp.simplify(2 * sp.pi * I * residue)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_contour_answer(difficulty, seed):
    src = CONTOUR_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_contour(q.text)
    stated = sp.sympify(_desanitize(q.answer_text))
    assert sp.simplify(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_contour_exam_states_no_pole_and_no_leaked_capital_i():
    """The exam tier must not print a solved pole (the learner derives it from the quadratic) and must never leak
    a sympy-style capital `I` into displayed text."""
    for seed in SEEDS:
        q = mathgen.make("p15e_contour_integral_residue", seed=seed, difficulty="exam")
        assert "upper half-plane" in q.text
        assert not _POLE_RE.search(q.text), f"exam tier leaked an explicit pole: {q.text}"
        assert re.search(r"(?<![a-zA-Z])I(?![a-zA-Z])", q.text) is None, f"exam tier leaked a bare capital I: {q.text}"
        assert "1i" not in q.text


# ==========================================================================
# ucv1-cr-equations-check
# ==========================================================================

_CR_OPTIONS = (
    "Both Cauchy-Riemann equations hold",
    "Only u_x = v_y fails",
    "Only u_y = -v_x fails",
    "Both Cauchy-Riemann equations fail",
)

_POINT_RE = re.compile(r"\(x, y\) = \((-?\d+), (-?\d+)\)")
_CR_UV_RE = re.compile(r"u\(x,y\) = (.+?) and v\(x,y\) = (.+?)[,.]")
_F_TEXT_RE = re.compile(r"f\(z\) = (.+?) \(z = x\+iy\)|f\(z\) = (.+?), with z = x\+iy|f\(z\) = (.+?)\. At the point")


def _sanitize_uv(token: str) -> str:
    return token.strip().rstrip(".").replace("^", "**").replace("·", "*")


def _cr_class_from_uv(u_expr, v_expr, x0, y0):
    ux = sp.diff(u_expr, _X).subs({_X: x0, _Y: y0})
    uy = sp.diff(u_expr, _Y).subs({_X: x0, _Y: y0})
    vx = sp.diff(v_expr, _X).subs({_X: x0, _Y: y0})
    vy = sp.diff(v_expr, _Y).subs({_X: x0, _Y: y0})
    first_holds = sp.simplify(ux - vy) == 0
    second_holds = sp.simplify(uy + vx) == 0
    if first_holds and second_holds:
        return _CR_OPTIONS[0]
    if not first_holds and second_holds:
        return _CR_OPTIONS[1]
    if first_holds and not second_holds:
        return _CR_OPTIONS[2]
    return _CR_OPTIONS[3]


def _extract_f_text(text: str) -> str:
    m = re.search(r"f\(z\) = (.+?)(?: \(z = x\+iy\)|, with z = x\+iy|\. At the point| \(z = x\+iy\))", text)
    assert m, f"could not parse f(z) from: {text}"
    return m.group(1)


_ZSYM, _ZCSYM = sp.symbols("z zc")


def _solve_cr_exam(text: str, x0: int, y0: int):
    """Parses f(z), given in z and conj(z) exactly as the exam prompt
    states it, and independently expands it with z = x+iy, conj(z) = x-iy
    substituted in -- the same hand derivation a learner does -- rather
    than reusing any of the generator's own kind-based algebra."""
    f_text = _extract_f_text(text)
    token = f_text.replace("conj(z)", "zc")

    def _i_repl(m):
        coef = m.group(1)
        if coef in ("", None):
            return "I*("
        if coef == "-":
            return "-I*("
        return f"{coef}*I*("

    token = re.sub(r"(-?\d*)i\(", _i_repl, token)
    token = re.sub(r"(\d)(?=[a-zA-Z(])", r"\1*", token)
    parsed = sp.sympify(token, locals={"z": _ZSYM, "zc": _ZCSYM, "I": sp.I, "sin": sp.sin})
    f_expr = parsed.subs({_ZSYM: _X + I * _Y, _ZCSYM: _X - I * _Y})
    f_expr = sp.expand(f_expr, complex=True)
    u = sp.expand(sp.re(f_expr))
    v = sp.expand(sp.im(f_expr))
    return _cr_class_from_uv(u, v, x0, y0)


def _solve_cr(text: str):
    point_m = _POINT_RE.search(text)
    assert point_m, f"could not parse the evaluation point from: {text}"
    x0, y0 = int(point_m.group(1)), int(point_m.group(2))

    uv = _CR_UV_RE.search(text)
    if uv:
        # `locals={"x": _X, "y": _Y}` is essential: plain `sp.sympify` would
        # mint its OWN assumption-free `x`/`y` symbols, distinct from this
        # module's `real=True` `_X`/`_Y` despite the identical name, so
        # `diff(..., _X)` would silently treat the parsed expression's `x`
        # as an unrelated constant and return 0 for every derivative.
        u_expr = sp.sympify(_sanitize_uv(uv.group(1)), locals={"x": _X, "y": _Y})
        v_expr = sp.sympify(_sanitize_uv(uv.group(2)), locals={"x": _X, "y": _Y})
        return _cr_class_from_uv(u_expr, v_expr, x0, y0)

    # Exam tier: no pre-expanded u, v in the prompt -- only f(z).
    return _solve_cr_exam(text, x0, y0)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_cr_classification(difficulty, seed):
    src = CR_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_cr(q.text)
    assert derived == q.answer_text, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {q.answer_text}"
    )


def test_oracle_cr_exam_has_no_unrelated_linear_system():
    """The exam's evaluation point must be a bare (x, y) pair, never an unrelated "solve x+y=s, x-y=d first"
    padding step."""
    for seed in SEEDS:
        q = mathgen.make("p15e_cr_equations_check", seed=seed, difficulty="exam")
        assert "x - y" not in q.text and "x + y" not in q.text
        assert _POINT_RE.search(q.text)


def test_oracle_cr_exam_requires_a_chain_rule_step():
    """The exam's f(z) must involve a genuine product/ chain-rule step (here, the complex trig identity for
    sin(x+iy)), never a pure polynomial identical in structure to the easier tiers."""
    for seed in SEEDS:
        q = mathgen.make("p15e_cr_equations_check", seed=seed, difficulty="exam")
        assert "sin(" in q.text


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard", "exam"]
    total = len(tiers) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds x 2 skills).")
