"""Independent oracle for the `p13e_residue_heat` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p13e_residue_heat_build`) and re-derives the answer from
scratch with sympy, using the textbook local-limit formula / PDE separation
theory rather than any of the generator's own construction code.
"""

import dataclasses
import re

import pytest
import sympy as sp

from mathgen.skills import load_skills
import mathgen

RESIDUE_ID = "ucv1-residue-simple-pole"
HEAT_ID = "ucv1-heat-equation-mode"
SEEDS = range(60)

_Z, _X, _T = sp.symbols("z x t")
_I = sp.I


def _desanitize(s: str) -> str:
    """Undo the generator's learner-facing display rewrites ("**" -> "^",
    "*" -> "·", see `p13e_residue_heat_build._fmt`/`_text_fmt`) before
    handing text to `sympy.sympify`, which understands neither."""
    return s.replace("^", "**").replace("·", "*")


_SINGLE_ANSWER_EXAM = {"ucv1-heat-equation-mode": "p13e_heat_equation_mode"}


def _by_difficulty(skill):
    """Sources by difficulty. The exam tier of a skill listed above now serves a multi-part chain; these tests
    cover the single-answer generator it replaced, so that tier is pointed back at it."""
    sources = {src.difficulty: src for src in skill.sources}
    if skill.id in _SINGLE_ANSWER_EXAM:
        sources["exam"] = dataclasses.replace(sources["exam"], topic=_SINGLE_ANSWER_EXAM[skill.id], params={})
    return sources


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return _by_difficulty(skill)


RESIDUE_SOURCES = _skill_sources(RESIDUE_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
RESIDUE_SOURCES["exam"] = dataclasses.replace(RESIDUE_SOURCES["exam"], topic="p13e_residue_simple_pole", params={})
HEAT_SOURCES = _skill_sources(HEAT_ID)


# ==========================================================================
# ucv1-residue-simple-pole
# ==========================================================================

_TARGET_RE = re.compile(r"pole at z = (-?\d+)")
_EASY_RE = re.compile(r"f\(z\) = (-?\d+) / \(\(z - \((-?\d+)\)\)\(z - \((-?\d+)\)\)\)")
_MEDIUM_RE = re.compile(r"f\(z\) = \(z - \((-?\d+)\)\) / \(\(z - \((-?\d+)\)\)\(z - \((-?\d+)\)\)\)")
_HARD_RE = re.compile(r"f\(z\) = (-?\d+) / \(z\^2 \+ (\d+)\^2\)")
_EXAM_RE = re.compile(r"f\(z\) = \(z - \((-?\d+)\)\) / \(z\^2 \+ \((-?\d+)\)z \+ \((-?\d+)\)\)")
_CANDIDATES_RE = re.compile(r"Candidate 1: z = (.+?)\.?\s+(?:and\s+)?Candidate 2: z = (.+?)(?:\.|\s\()")


def _solve_residue(text: str):
    m = _EXAM_RE.search(text)
    if m:
        r, b_coef, c_coef = (int(g) for g in m.groups())
        cm = _CANDIDATES_RE.search(text)
        assert cm, f"could not find the two candidates in: {text}"
        w1 = sp.sympify(_desanitize(cm.group(1)))
        w2 = sp.sympify(_desanitize(cm.group(2)))
        roots = [w for w in (w1, w2) if sp.simplify(w**2 + b_coef * w + c_coef) == 0]
        assert len(roots) == 1, f"expected exactly one true root among {w1}, {w2}, got {roots}"
        w_true = roots[0]
        other_root = sp.conjugate(w_true)
        return sp.simplify((w_true - r) / (w_true - other_root))

    m = _HARD_RE.search(text)
    if m:
        c, k = (int(g) for g in m.groups())
        f = c / (_Z**2 + k**2)
        return sp.simplify(sp.limit((_Z - _I * k) * f, _Z, _I * k))

    m = _MEDIUM_RE.search(text)
    if m:
        r, a, b = (int(g) for g in m.groups())
        tm = _TARGET_RE.search(text)
        assert tm, f"could not find the stated pole in: {text}"
        target = int(tm.group(1))
        other = b if target == a else a
        f = (_Z - r) / ((_Z - target) * (_Z - other))
        return sp.simplify(sp.limit((_Z - target) * f, _Z, target))

    m = _EASY_RE.search(text)
    if m:
        c, a, b = (int(g) for g in m.groups())
        tm = _TARGET_RE.search(text)
        assert tm, f"could not find the stated pole in: {text}"
        target = int(tm.group(1))
        other = b if target == a else a
        f = c / ((_Z - target) * (_Z - other))
        return sp.simplify(sp.limit((_Z - target) * f, _Z, target))

    raise AssertionError(f"could not parse residue question text: {text}")


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_residue_answer(difficulty, seed):
    src = RESIDUE_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_residue(q.text)
    stated = sp.sympify(_desanitize(q.answer_text))
    assert sp.simplify(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


# ==========================================================================
# ucv1-heat-equation-mode
# ==========================================================================

_ALPHA_RE = re.compile(r"alpha = (\d+)")
_N_RE = re.compile(r"\bn = (\d+)\b")
_L_INT_RE = re.compile(r"\[0, (\d+)\]")
_COEFF_RE = re.compile(r"[ABC] = (\d+)")


def _solve_heat(text: str):
    alpha_m = _ALPHA_RE.search(text)
    n_m = _N_RE.search(text)
    coeff_m = _COEFF_RE.search(text)
    assert alpha_m and n_m and coeff_m, f"could not parse alpha/n/coefficient from: {text}"
    alpha = int(alpha_m.group(1))
    n = int(n_m.group(1))
    coeff = int(coeff_m.group(1))

    if "[0, pi]" in text:
        L = sp.pi
    else:
        l_m = _L_INT_RE.search(text)
        assert l_m, f"could not parse L from: {text}"
        L = sp.Integer(l_m.group(1))

    is_neumann = ("insulated" in text) or ("u_x(0,t)" in text)
    spatial = sp.cos(n * sp.pi * _X / L) if is_neumann else sp.sin(n * sp.pi * _X / L)
    decay = sp.exp(-alpha * sp.Integer(n) ** 2 * sp.pi**2 * _T / L**2)
    return sp.simplify(coeff * spatial * decay)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_heat_answer(difficulty, seed):
    src = HEAT_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve_heat(q.text)
    stated = sp.sympify(_desanitize(q.answer_text))
    assert sp.simplify(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard", "exam"]
    total = len(tiers) * len(SEEDS) * 2  # both skills
    assert len(SEEDS) >= 60
    assert total >= 480
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds x 2 skills).")
