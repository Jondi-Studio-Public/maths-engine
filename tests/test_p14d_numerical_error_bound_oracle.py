"""Independent oracle for the `p14d_numerical_error_bound` questions: it re-derives each answer from the rendered
question text.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p14d_numerical_error_bound_build`) and re-derives the
absolute and relative error bound from scratch with the plain
max(ahat - lo, hi - ahat) formula -- a genuinely different route from the
generator's own internal check, which proves the SAME direct formula against
the half-width-plus-offset identity. The oracle here re-derives everything,
independently, straight from the rendered text.
"""

import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.skills import load_skills

SKILL_ID = "ucv1-numerical-error-bound"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_FRAC = r"-?\d+(?:/\d+)?"
_AHAT_RE = re.compile(rf"ahat = ({_FRAC})")
_BRACKET_RE = re.compile(rf"\[({_FRAC}), ({_FRAC})\]")
_CENTRE_RADIUS_RE = re.compile(rf"certified to lie within ({_FRAC}) of ({_FRAC})")
_LABELLED_PAIR_RE = re.compile(rf"(\w+) = ({_FRAC}), (\w+) = ({_FRAC})")


def _parse_frac(text: str) -> Fraction:
    if "/" in text:
        p, q = text.split("/")
        return Fraction(int(p), int(q))
    return Fraction(int(text))


def _solve(text: str):
    ahat_m = _AHAT_RE.search(text)
    assert ahat_m, f"could not find ahat in: {text}"
    ahat = _parse_frac(ahat_m.group(1))

    bracket_m = _BRACKET_RE.search(text)
    if bracket_m:
        lo, hi = _parse_frac(bracket_m.group(1)), _parse_frac(bracket_m.group(2))
    else:
        cr_m = _CENTRE_RADIUS_RE.search(text)
        assert cr_m, f"could not find a bracket or centre/radius phrase in: {text}"
        radius, centre = _parse_frac(cr_m.group(1)), _parse_frac(cr_m.group(2))
        lo, hi = centre - radius, centre + radius

    assert lo < ahat < hi, f"ahat not strictly inside [lo, hi]: {lo} < {ahat} < {hi}"
    absolute_bound = max(ahat - lo, hi - ahat)
    relative_bound = absolute_bound / abs(ahat)
    return absolute_bound, relative_bound


def _parse_answer_pair(answer_text: str):
    m = _LABELLED_PAIR_RE.fullmatch(answer_text)
    assert m, f"could not parse answer_text: {answer_text}"
    label1, v1, label2, v2 = m.groups()
    return {label1: _parse_frac(v1), label2: _parse_frac(v2)}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_error_bound(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived_abs, derived_rel = _solve(q.text)
    stated = _parse_answer_pair(q.answer_text)
    assert derived_abs == stated["absolute_bound"], (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived absolute_bound: {derived_abs}\nstated: {stated}"
    )
    assert derived_rel == stated["relative_bound"], (
        f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived relative_bound: {derived_rel}\nstated: {stated}"
    )


def test_oracle_coverage():
    total = len(TIERS) * len(SEEDS)
    assert len(SEEDS) >= 60
    assert total >= 240
    print(f"\nOracle re-derived {total} questions ({len(TIERS)} tiers x {len(SEEDS)} seeds).")
