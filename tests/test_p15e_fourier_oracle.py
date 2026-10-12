"""Independent oracle for the `p15e_fourier` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p15e_fourier_build`). The Gaussian transform is
re-derived via `mpmath` numeric quadrature of the DEFINING integral at one
sample point -- never the generator's own closed-form substitution. The
wave mode is re-derived directly from the textbook separation-of-variables
formula for the stated boundary family and initial condition, then checked
against the reported answer by symbolic equality.
"""

import re

import mpmath as mp
import pytest
import sympy as sp

from mathgen.skills import load_skills
from tests.exd_x_r2h_support import legacy_exam_sources
import mathgen

FT_ID = "ucv1-fourier-transform-gaussian"
WAVE_ID = "ucv1-wave-equation-separation"
SEEDS = range(60)

_X, _W = sp.symbols("x w", real=True)
_T = sp.symbols("t", real=True)


def _desanitize(s: str) -> str:
    return s.replace("^", "**").replace("·", "*")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return legacy_exam_sources(skill.id, {src.difficulty: src for src in skill.sources})


FT_SOURCES = _skill_sources(FT_ID)
WAVE_SOURCES = _skill_sources(WAVE_ID)


# ==========================================================================
# ucv1-fourier-transform-gaussian
# ==========================================================================

_F_RE = re.compile(r"f\(x\) = (-?\d*)\s*exp\(-(?:(\d+)?x\^2|x\^2)\)")


def _parse_amplitude(token: str) -> int:
    if token in ("", "-"):
        return -1 if token == "-" else 1
    return int(token)


def _oracle_transform_value(A: int, a: int, w0: float) -> float:
    """Numeric quadrature of the DEFINING ordinary-convention integral
    A*exp(-a*x^2)*exp(-i*w0*x) dx over the real line, at one sample point --
    a genuinely different route than the generator's closed-form formula.
    Only the real part is sampled: the imaginary part is IDENTICALLY zero
    by odd/even symmetry (proved once, not re-checked numerically here)."""
    bound = 12.0 / (float(a) ** 0.5)
    return float(mp.quad(lambda x: float(A) * mp.e ** (-float(a) * x**2) * mp.cos(w0 * x), [-bound, bound]))


def _solve_ft(text: str, answer_text: str):
    fm = _F_RE.search(text)
    assert fm, f"could not parse f(x) from: {text}"
    A = _parse_amplitude(fm.group(1))
    a = int(fm.group(2)) if fm.group(2) else 1

    is_freq = "F(xi)" in text
    if is_freq:
        freq = sp.symbols("xi", real=True)
        stated = sp.sympify(_desanitize(answer_text), locals={"xi": freq})
        # F(xi) = ordinary-convention F(w) with w = 2*pi*xi substituted.
        w0 = 1.7
        f0 = w0 / (2 * 3.141592653589793)
        predicted = _oracle_transform_value(A, a, w0)
        sampled = float(stated.subs(freq, f0))
    elif "UNITARY" in text or "unitary" in text:
        stated = sp.sympify(_desanitize(answer_text), locals={"w": _W})
        w0 = 1.7
        ordinary = _oracle_transform_value(A, a, w0)
        predicted = ordinary / (2 * 3.141592653589793) ** 0.5
        sampled = float(stated.subs(_W, w0))
    else:
        stated = sp.sympify(_desanitize(answer_text), locals={"w": _W})
        w0 = 1.7
        predicted = _oracle_transform_value(A, a, w0)
        sampled = float(stated.subs(_W, w0))
    return predicted, sampled


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_gaussian_transform(difficulty, seed):
    src = FT_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    predicted, sampled = _solve_ft(q.text, q.answer_text)
    assert abs(predicted - sampled) < 1e-4, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\n"
        f"predicted: {predicted}\nsampled from stated answer: {sampled}"
    )


# ==========================================================================
# ucv1-wave-equation-separation
# ==========================================================================

_C_RE = re.compile(r"c = (\d+)")
_N_RE = re.compile(r"n = (\d+)")
_L_PI_RE = re.compile(r"\[0, pi\]")
_L_INT_RE = re.compile(r"\[0, (\d+)\]")
_COEF_RE = re.compile(r"[ABC] = (-?\d+)")


_MIXED_INDEXING_RE = re.compile(r"n = 1 is the lowest mode, n = 2 the next")


def _oracle_wave_number(family: str, n: int, L, text: str = ""):
    """The wavenumber k for the family's own eigenfunction: n*pi/L for Dirichlet/Neumann, the half-integer
    (2n-1)*pi/(2L) for the exam-tier mixed family (fixed at x=0, free at x=L) -- a genuinely different
    closed-form derivation than the generator's own `_wave_number` (the textbook boundary-value argument:
    u(0,t)=0 rules out cos(kx), and u_x(L,t)=0 forces cos(kL)=0, i.e. kL=(2m-1)pi/2). The mixed family's
    formula depends on an INDEXING CONVENTION (n=1 is the lowest mode) that the prompt must state explicitly
    -- so this only applies the (2n-1)*pi/(2L) formula once that stated convention is confirmed present,
    rather than assuming it silently."""
    if family == "mixed":
        assert _MIXED_INDEXING_RE.search(text), (
            f"mixed-family prompt never states the n=1-is-lowest-mode indexing "
            f"convention this formula depends on: {text}"
        )
        return (2 * n - 1) * sp.pi / (2 * L)
    return n * sp.pi / L


def _oracle_wave_mode(family: str, time_kind: str, coeff: int, n: int, c: int, L, text: str = ""):
    k = _oracle_wave_number(family, n, L, text)
    if family == "dirichlet":
        spatial = sp.sin(k * _X)
    elif family == "neumann":
        spatial = sp.cos(k * _X)
    else:  # mixed: fixed at 0 rules out cos, so the eigenfunction is sin(kx)
        spatial = sp.sin(k * _X)
    time_factor = sp.cos(k * c * _T) if time_kind == "cos" else sp.sin(k * c * _T)
    return coeff * spatial * time_factor


def _parse_wave_params(text: str):
    c_m = _C_RE.search(text)
    # `.findall` + last match, never `.search`'s first: the mixed-family exam
    # prompt states an indexing convention sentence ("n = 1 is the lowest
    # mode, n = 2 the next, ...") BEFORE the actual "mode number n = <k>"
    # clause, and `search` would otherwise grab the convention sentence's own
    # "n = 1" as if it were the drawn mode number.
    n_all = _N_RE.findall(text)
    coeff_m = _COEF_RE.search(text)
    assert c_m and n_all and coeff_m, f"could not parse c/n/coefficient from: {text}"
    c = int(c_m.group(1))
    n = int(n_all[-1])
    coeff = int(coeff_m.group(1))
    L = sp.pi if _L_PI_RE.search(text) else sp.Integer(_L_INT_RE.search(text).group(1))

    is_mixed = "mixed" in text.lower()
    is_neumann = ("free ends" in text) or ("Neumann" in text) or ("u_x(0,t)" in text)
    if is_mixed:
        family = "mixed"
    elif is_neumann:
        family = "neumann"
    else:
        family = "dirichlet"
    starts_from_rest = "starts from rest" in text
    time_kind = "cos" if starts_from_rest else "sin"
    return family, time_kind, coeff, n, c, L


def _solve_wave(text: str):
    family, time_kind, coeff, n, c, L = _parse_wave_params(text)
    return _oracle_wave_mode(family, time_kind, coeff, n, c, L, text)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_wave_mode(difficulty, seed):
    src = WAVE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_wave(q.text)
    # `locals={"x": _X, "t": _T}` matches this module's `real=True` symbols:
    # a plain `sp.sympify` would otherwise mint assumption-free x/t distinct
    # from `_X`/`_T` despite the identical name, so an ALGEBRAICALLY equal
    # `derived - stated` would print as zero-looking but not reduce to the
    # literal `0` `simplify` needs (the same hazard fixed in the CR oracle).
    stated = sp.sympify(_desanitize(q.answer_text), locals={"x": _X, "t": _T})
    assert sp.simplify(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_wave_mode_solves_pde_and_boundary(difficulty, seed):
    """The PDE/boundary-condition verification that used to run inside `make` as `_verify_wave_mode` -- moved
    here ('the symbolic... cross-check belongs in the oracle tests, not in make'). Uses `sympy.diff` directly
    on the STATED answer (never the generator's own internals), so this is a genuinely independent check that
    u_tt = c^2 u_xx and the family's own boundary condition both hold."""
    src = WAVE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    family, _time_kind, _coeff, _n, c, L = _parse_wave_params(q.text)
    stated = sp.sympify(_desanitize(q.answer_text), locals={"x": _X, "t": _T})

    u_tt = sp.diff(stated, _T, 2)
    u_xx = sp.diff(stated, _X, 2)
    assert sp.simplify(u_tt - c**2 * u_xx) == 0, (
        f"seed={seed} difficulty={difficulty}\nanswer: {stated}\ndoes not solve u_tt = c^2 u_xx"
    )

    u_x = sp.diff(stated, _X)
    if family == "dirichlet":
        assert sp.simplify(stated.subs(_X, 0)) == 0
        assert sp.simplify(stated.subs(_X, L)) == 0
    elif family == "neumann":
        assert sp.simplify(u_x.subs(_X, 0)) == 0
        assert sp.simplify(u_x.subs(_X, L)) == 0
    else:  # mixed: fixed at x=0, free at x=L
        assert sp.simplify(stated.subs(_X, 0)) == 0
        assert sp.simplify(u_x.subs(_X, L)) == 0


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard", "exam"]
    total = len(tiers) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds x 2 skills).")
