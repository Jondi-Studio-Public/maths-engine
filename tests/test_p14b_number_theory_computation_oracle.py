"""Independent oracle for the `p14b_number_theory_computation` questions: it re-derives each answer from the
rendered question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p14b_number_theory_computation_build`) for n, m and e
(evaluating e's expression form when needed) and recomputes `pow(m, e, n)`
directly -- Python's own three-argument modular exponentiation, standard
library, the cleanest independent check available for this skill.
"""

from dataclasses import replace
import re

import pytest

import mathgen
from mathgen.skills import load_skills

NTC_ID = "ucv1-number-theory-computation"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_M_RE = re.compile(r"m = (\d+)")
# Never followed by "^": "e = 2^10 + 26" must fall through to the expression
# branch below, not stop early at "e = 2".
_E_RE = re.compile(r"e = (\d+)(?!\^)")
_EXPR_RE = re.compile(r"2\^(\d+) \+ (\d+)")
# "n = " is ambiguous in some templates ("n = {p} × {q} = {n}" states
# "n = " TWICE, and the first occurrence is p, not n) -- always derive n from
# the stated factorisation "{p}(^2)? × {q}" instead.
_FACTORS_RE = re.compile(r"(\d+)(\^2)?\s*×\s*(\d+)")


def _extract_n(text: str) -> int:
    fm = _FACTORS_RE.search(text)
    assert fm, text
    p, squared, q = int(fm.group(1)), fm.group(2), int(fm.group(3))
    return (p * p * q) if squared else (p * q)


def _solve(text: str) -> int:
    n = _extract_n(text)
    m = int(_M_RE.search(text).group(1))
    e_match = _E_RE.search(text)
    if e_match is not None:
        e = int(e_match.group(1))
    else:
        expr_match = _EXPR_RE.search(text)
        assert expr_match, text
        e = 2 ** int(expr_match.group(1)) + int(expr_match.group(2))
    return pow(m, e, n)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


NTC_SOURCES = _skill_sources(NTC_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
NTC_SOURCES["exam"] = replace(NTC_SOURCES["exam"], topic="p14b_number_theory_computation", params={})


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_number_theory_answer(difficulty, seed):
    src = NTC_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = int(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_would_catch_a_corrupted_answer():
    src = NTC_SOURCES["easy"]
    q = mathgen.make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    corrupted = derived + 1
    assert derived != corrupted


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
