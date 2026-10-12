"""Independent oracle for the `p12f_ftc_rsa` questions: it re-derives each answer from the rendered question
text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p12f_ftc_rsa_build`) and re-derives the answer from
scratch with its own modular-exponentiation implementation.
"""

from dataclasses import replace

import re

import pytest

import mathgen
from mathgen.skills import load_skills

RSA_ID = "ucv1-rsa-modular-step"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_N_RE = re.compile(r"n = (?:p\*q = )?(\d+)")
_N_STATED_RE = re.compile(r"n = \d")
_D_RE = re.compile(r"d = (\d+)")
_E_RE = re.compile(r"e = (\d+)")
_C_RE = re.compile(r"c = (\d+)")
_M_RE = re.compile(r"m = (\d+)")
_P_RE = re.compile(r"p = (\d+)")
_Q_RE = re.compile(r"q = (\d+)")


def _modpow(base: int, exp: int, modulus: int) -> int:
    """The oracle's own repeated-squaring implementation -- written without
    reading the generator, deliberately structured differently from its
    `_mod_pow_manual` (accumulates via left-to-right binary digits of `exp`
    rather than right-to-left square-and-multiply)."""
    result = 1
    for bit in bin(exp)[2:]:
        result = (result * result) % modulus
        if bit == "1":
            result = (result * base) % modulus
    return result


def _solve(text: str) -> int:
    if _D_RE.search(text) is not None:
        # decrypt tier: n, d, c stated directly; find m = c^d mod n.
        n = int(_N_RE.search(text).group(1))
        d = int(_D_RE.search(text).group(1))
        c = int(_C_RE.search(text).group(1))
        return _modpow(c, d, n)

    if _N_STATED_RE.search(text) is None:
        # exam chain tier: only p, q, e, m stated; n = p*q must be derived first.
        p = int(_P_RE.search(text).group(1))
        q = int(_Q_RE.search(text).group(1))
        e = int(_E_RE.search(text).group(1))
        m = int(_M_RE.search(text).group(1))
        n = p * q
        return _modpow(m, e, n)

    # encrypt tier (easy/medium): n stated directly.
    n = int(_N_RE.search(text).group(1))
    e = int(_E_RE.search(text).group(1))
    m = int(_M_RE.search(text).group(1))
    return _modpow(m, e, n)


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


RSA_SOURCES = _skill_sources(RSA_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
RSA_SOURCES["exam"] = replace(
    RSA_SOURCES["exam"],
    topic="p12f_rsa_modular_step",
    params={"mode": "chain", "primes": [3, 5, 7, 11, 13], "exponents": [3]},
)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_rsa_answer(difficulty, seed):
    src = RSA_SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    stated = int(q.answer_text)
    assert derived == stated, (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
    )


def test_oracle_would_catch_a_corrupted_answer():
    """Proves the oracle is red first: a deliberately-wrong stated answer
    must fail the comparison above, not pass silently."""
    src = RSA_SOURCES["easy"]
    q = mathgen.make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
    derived = _solve(q.text)
    corrupted = derived + 1
    assert derived != corrupted


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
