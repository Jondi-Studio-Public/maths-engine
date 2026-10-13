"""Independent oracle for the `ucv1-energy-estimate` questions: it rebuilds u(x,t) from the rendered text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p18a_energy_estimate_build`). Rebuilds u(x,t) from the
stated A/k/n(or n+1/2)/interval/family, and re-derives E(t) by calling
`sympy.integrate` directly on u(x,t)^2 over the stated interval -- the
general dispatcher, not the generator's own double-angle-antiderivative
shortcut -- a genuinely different computational route from both pieces of
the generator's own construction/check.

Seed count is the module constant `N_SEEDS`, overridable by
`P18A_ORACLE_SEEDS`, default 250 per tier
(>= 1000 questions over the four tiers).
"""

import os
import re

import pytest
import sympy as sp

from mathgen.skills import load_skills
from types import SimpleNamespace
import mathgen

SKILL_ID = "ucv1-energy-estimate"
N_SEEDS = int(os.environ.get("P18A_ORACLE_SEEDS", "250"))
SEEDS = range(N_SEEDS)
TIERS = ["easy", "medium", "hard", "exam"]

_X, _T = sp.symbols("x t")


def _desanitize(s: str) -> str:
    """Undo the generator's learner-facing display rewrites ("**" -> "^",
    "*" -> "·"), the same convention `test_p13e_residue_heat_oracle.py` uses,
    before handing text to `sympy.sympify`."""
    return s.replace("^", "**").replace("·", "*")


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources()
# The skill's exam tier is now a multi-part chain (tests/test_exd_x_r2g_heat_energy_decay.py). These tests keep covering the
# generator's own earlier exam preset, which is unchanged.
SOURCES["exam"] = SimpleNamespace(topic="p18a_energy_estimate", difficulty="exam", params={}, stages=["exam"])

_K_RE = re.compile(r"k = (-?\d+)")
_INTERVAL_RE = re.compile(r"\[(-?\d+),\s*(-?\d+)\]")
_AMPLITUDE_RE = re.compile(r"u\(x,(?:t|0)\) = (-?\d+)")

# The generator prints the REDUCED wavenumber/decay-rate fraction
# directly (e.g. "sin(7*pi*x/12)", "exp(-45*pi^2*t/4)", "sin(pi*x)"), never
# the raw n/L or n^2/L^2 pair, so this oracle parses the reduced coefficients
# straight off the text and integrates the literal printed function -- an
# even more direct "does the stated answer match what is actually printed"
# check than reconstructing n and L separately would give. `(?:\(x[^)]*\)|x)`
# swallows an offset x-term ("(x - 2)"/"(x + 3)"/"x") without needing its
# value: the true offset `a` is already recovered from `_INTERVAL_RE`.
_MODE_RE = re.compile(
    r"(?P<trig>sin|cos)\("
    r"(?:(?P<num>\d+)·)?pi·(?:\(x[^)]*\)|x)(?:/(?P<den>\d+))?\)"
    r"·exp\(-(?:(?P<enum>\d+)·)?pi\^2·t(?:/(?P<eden>\d+))?\)"
)
# The IC-given exam variant prints only the spatial factor (no exp yet, by
# design -- the learner must derive the decay rate from k and this
# wavenumber before integrating).
_IC_SIN_RE = re.compile(r"u\(x,0\) = -?\d+·sin\((?:(?P<num>\d+)·)?pi·x(?:/(?P<den>\d+))?\)")


def _frac(num, den):
    return sp.Rational(int(num) if num else 1, int(den) if den else 1)


def _rederive_energy(text: str):
    a, b = (int(g) for g in _INTERVAL_RE.search(text).groups())
    A = int(_AMPLITUDE_RE.search(text).group(1))

    is_neumann_zero = "mode number n = 0" in text or "n = 0 mode" in text
    if is_neumann_zero:
        u = A
        return sp.simplify(sp.integrate(u**2, (_X, a, b)))

    m = _MODE_RE.search(text)
    if m is not None:
        # Main branches: the printed sin/cos coefficient IS m/L already,
        # reduced -- use it directly for the spatial factor. But the printed
        # EXP coefficient is cross-checked against k and that same sin/cos
        # coefficient (derived_rate = k*(m/L)^2) rather than trusted outright:
        # trusting the printed exponent alone would accept an internally
        # inconsistent question (a stated k that disagrees with the mode's
        # own decay), which is exactly the "match a corrupted key" failure
        # this oracle exists to catch.
        k = int(_K_RE.search(text).group(1))
        trig_func = sp.sin if m.group("trig") == "sin" else sp.cos
        sin_coef = _frac(m.group("num"), m.group("den"))
        exp_coef = _frac(m.group("enum"), m.group("eden"))
        derived_rate = k * sin_coef**2
        assert sp.simplify(exp_coef - derived_rate) == 0, (
            f"printed exp coefficient {exp_coef} disagrees with k*(m/L)^2 = "
            f"{derived_rate} (k={k}, sin/cos coefficient={sin_coef}): {text}"
        )
        spatial = trig_func(sin_coef * sp.pi * (_X - a))
        u = A * spatial * sp.exp(-derived_rate * sp.pi**2 * _T)
        return sp.simplify(sp.integrate(u**2, (_X, a, b)))

    # IC-given exam variant: only the spatial factor is printed; derive the
    # decay rate independently from k and the PDE (u_t = k u_xx forces
    # exp(-k*(sin_coef*pi)^2*t) for a mode sin(sin_coef*pi*(x-a))).
    ic = _IC_SIN_RE.search(text)
    k = int(_K_RE.search(text).group(1))
    sin_coef = _frac(ic.group("num"), ic.group("den"))
    spatial = sp.sin(sin_coef * sp.pi * (_X - a))
    u = A * spatial * sp.exp(-k * sin_coef**2 * sp.pi**2 * _T)
    return sp.simplify(sp.integrate(u**2, (_X, a, b)))


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_energy_answer(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _rederive_energy(q.text)
    stated = sp.sympify(_desanitize(q.answer_text))
    assert sp.simplify(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert N_SEEDS >= 250
    assert total >= 1000
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {N_SEEDS} seeds).")
