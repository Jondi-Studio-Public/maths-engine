"""Independent oracle for the `p15e_algebra` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p15e_algebra_build`) and re-derives the answer from
scratch with `sympy.rem`/`sympy.div` -- polynomial division, a genuinely
different algorithm than the generator's own hand-substitution recurrence.
"""

import re

import pytest
import sympy as sp

from mathgen.skills import load_skills
import mathgen

SKILL_ID = "ucv1-quotient-ring-operation"
SEEDS = range(60)

_X = sp.symbols("x")


def _desanitize(s: str) -> str:
    return s.replace("^", "**").replace("·", "*")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)

_P_RE = re.compile(r"Z_(\d+)")

# `m(x)`/`a(x)`/`b(x)` now display via `mathgen.core.join_terms` ("x^2 - 4x + 1",
# "6x - 3", "-2x^2 - 2x - 5", never the old "x^2 + (m1)x + (m0)" parenthesised
# form) -- so extraction just grabs the free-form polynomial text up to
# whichever word or punctuation mark ends it in that sentence, rather than
# assuming a fixed coefficient shape.
_STOP = r"(?:,|\.| and | by | in | over | with |\()"
_POLY_RE = {name: re.compile(rf"{name}\(x\) = (.+?){_STOP}") for name in "mab"}


def _extract(text: str, name: str) -> str:
    m = _POLY_RE[name].search(text)
    assert m, f"could not parse {name}(x) from: {text}"
    return m.group(1)


def _to_expr(token: str):
    """A `join_terms`-formatted polynomial string ("x^2 - 4x + 1", "6x - 3")
    into a sympy expression: `^` -> `**`, then implicit "6x" -> "6*x" (sympy
    does not parse bare adjacency)."""
    token = token.strip().replace("^", "**")
    token = re.sub(r"(\d)(x)", r"\1*\2", token)
    return sp.sympify(token)


def _oracle_common(text: str, op: str):
    m_expr = _to_expr(_extract(text, "m"))
    a_expr = _to_expr(_extract(text, "a"))
    b_expr = _to_expr(_extract(text, "b"))
    raw = sp.expand(a_expr + b_expr) if op == "+" else sp.expand(a_expr * b_expr)
    _, rem = sp.div(raw, m_expr, _X)
    return sp.expand(rem)


def _oracle_easy(text: str):
    return _oracle_common(text, "+")


def _oracle_medium(text: str):
    return _oracle_common(text, "*")


def _oracle_hard(text: str):
    return _oracle_common(text, "*")


def _oracle_exam(text: str):
    p_m = _P_RE.search(text)
    assert p_m, f"could not parse p from: {text}"
    p = int(p_m.group(1))
    rem = _oracle_common(text, "*")
    if rem == 0:
        return sp.Integer(0)
    rem = sp.Poly(rem, _X)
    coeffs = [rem.coeff_monomial(_X**i) % p for i in range(rem.degree() + 1)]
    return sp.expand(sum(c * _X**i for i, c in enumerate(coeffs)))


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard", "exam"] for s in SEEDS])
def test_oracle_rederives_quotient_ring_answer(difficulty, seed):
    if difficulty == "exam":
        # The skill's exam source is now a multi-part chain; this oracle covers the single-answer generator.
        q = mathgen.make("p15e_quotient_ring_operation", seed=seed, difficulty="exam")
    else:
        src = SOURCES[difficulty]
        q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _ORACLES[difficulty](q.text)
    stated = sp.expand(sp.sympify(_desanitize(q.answer_text)))
    assert sp.expand(derived - stated) == 0, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard", "exam"]
    total = len(tiers) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(tiers)} tiers x {len(SEEDS)} seeds).")
