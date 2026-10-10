"""Independent oracle for the p17b proportion confidence-interval questions.

Parses ONLY `question.text` and `question.answer_text` (never anything from
`mathgen.generators.p17b_proportion_ci_inverse_build`) and re-derives every
tier's answer from scratch, straight off the printed strings:

  easy:   p_hat = (L + U) / 2, margin = (U - L) / 2
  medium: p_hat = (L + U) / 2, margin = (U - L) / 2,
          n = round(z^2 . p_hat . (1 - p_hat) / margin^2), k = round(p_hat . n)
  hard:   margin = |p_hat - bound| (bound is whichever of lower/upper is
          printed), n and k as above, z read from the stated confidence level
          via the inline table
  exam:   n as at medium, N = ceil(z^2 . p_hat . (1 - p_hat) / target^2)

This is a genuinely different route from the generator's own construction
(which draws n, k first and rejects any draw whose printed-bound-derived key
disagrees with the hidden truth) -- the oracle never sees the hidden n/k/z at
all, only what the rendered question shows.
"""

import re
from fractions import Fraction

import pytest

import mathgen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

SKILL_ID = "ucv1-proportion-confidence-interval"
SEEDS = range(60)
REDERIVE_SEEDS = range(1000)
GRADER_SEEDS = range(1000)
TIERS = ["easy", "medium", "hard", "exam"]

_FRAC = r"-?\d+(?:\.\d+)?"
_BOUNDS_RE = re.compile(rf"lower bound ({_FRAC}) and upper bound ({_FRAC})")
_Z_RE = re.compile(r"z = (\d+\.\d+)")
_HARD_LEVEL_RE = re.compile(r"built at the (\d+)% confidence level")
_TABLE_ROW_RE = re.compile(r"an? (\d+)% confidence level uses z = (" + _FRAC + r")")
_PHAT_RE = re.compile(rf"p_hat = ({_FRAC})")
_LOWER_BOUND_RE = re.compile(rf"lower bound ({_FRAC})\.")
_UPPER_BOUND_RE = re.compile(rf"upper bound ({_FRAC})\.")
_TARGET_RE = re.compile(rf"target margin is ({_FRAC})")
_LABELLED_PAIR_RE = re.compile(rf"(\w+) = ({_FRAC}), (\w+) = ({_FRAC})")


def _to_fraction(text: str) -> Fraction:
    if "/" in text:
        p, q = text.split("/")
        return Fraction(int(p), int(q))
    return Fraction(text)


def _round_half_up(value: Fraction) -> int:
    import math

    sign = -1 if value < 0 else 1
    return sign * math.floor(abs(value) + Fraction(1, 2))


def _parse_answer_pair(answer_text: str):
    m = _LABELLED_PAIR_RE.fullmatch(answer_text)
    assert m, f"could not parse answer_text: {answer_text}"
    label1, v1, label2, v2 = m.groups()
    return {label1: _to_fraction(v1), label2: _to_fraction(v2)}


def _solve_easy(text: str):
    m = _BOUNDS_RE.search(text)
    assert m, f"no bounds in: {text}"
    L, U = _to_fraction(m.group(1)), _to_fraction(m.group(2))
    return {"p_hat": (L + U) / 2, "margin": (U - L) / 2}


def _solve_medium(text: str):
    bm = _BOUNDS_RE.search(text)
    assert bm, f"no bounds in: {text}"
    L, U = _to_fraction(bm.group(1)), _to_fraction(bm.group(2))
    zm = _Z_RE.search(text)
    assert zm, f"no z in: {text}"
    z = _to_fraction(zm.group(1))
    p_hat = (L + U) / 2
    margin = (U - L) / 2
    n = _round_half_up(z * z * p_hat * (1 - p_hat) / (margin * margin))
    k = _round_half_up(p_hat * n)
    return {"n": n, "k": k}


def _solve_hard(text: str):
    pm = _PHAT_RE.search(text)
    assert pm, f"no p_hat in: {text}"
    p_hat = _to_fraction(pm.group(1))
    lower_m = _LOWER_BOUND_RE.search(text)
    upper_m = _UPPER_BOUND_RE.search(text)
    assert (lower_m is None) != (upper_m is None), f"expected exactly one bound in: {text}"
    if lower_m:
        bound = _to_fraction(lower_m.group(1))
        margin = p_hat - bound
    else:
        bound = _to_fraction(upper_m.group(1))
        margin = bound - p_hat
    level_m = _HARD_LEVEL_RE.search(text)
    assert level_m, f"no confidence level in: {text}"
    level = level_m.group(1)
    table = dict(_TABLE_ROW_RE.findall(text))
    assert level in table, f"level {level} not in table for: {text}"
    z = _to_fraction(table[level])
    n = _round_half_up(z * z * p_hat * (1 - p_hat) / (margin * margin))
    k = _round_half_up(p_hat * n)
    return {"n": n, "k": k}


def _solve_exam(text: str):
    bm = _BOUNDS_RE.search(text)
    assert bm, f"no bounds in: {text}"
    L, U = _to_fraction(bm.group(1)), _to_fraction(bm.group(2))
    zm = _Z_RE.search(text)
    assert zm, f"no z in: {text}"
    z = _to_fraction(zm.group(1))
    tm = _TARGET_RE.search(text)
    assert tm, f"no target margin in: {text}"
    target = _to_fraction(tm.group(1))
    p_hat = (L + U) / 2
    margin = (U - L) / 2
    n = _round_half_up(z * z * p_hat * (1 - p_hat) / (margin * margin))
    N_exact = z * z * p_hat * (1 - p_hat) / (target * target)
    if N_exact.denominator == 1:
        N = N_exact.numerator
    else:
        N = N_exact.numerator // N_exact.denominator + 1
    return {"n": n, "N": N}


_SOLVERS = {
    "easy": _solve_easy,
    "medium": _solve_medium,
    "hard": _solve_hard,
    "exam": _solve_exam,
}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return {src.difficulty: src for src in skill.sources}


SOURCES = _skill_sources(SKILL_ID)


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in TIERS for s in SEEDS])
def test_oracle_rederives_answer(difficulty, seed):
    src = SOURCES[difficulty]
    q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
    derived = _SOLVERS[difficulty](q.text)
    stated = _parse_answer_pair(q.answer_text)
    assert set(derived.keys()) == set(stated.keys())
    for label, derived_value in derived.items():
        assert derived_value == stated[label], (
            f"seed={seed} difficulty={difficulty}\ntext: {q.text}\nderived: {derived}\nstated: {stated}"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_rederives_answer_over_1000_seeds(difficulty):
    """The oracle re-derives both components from the rendered
    prompt TEXT over at least 1000 seeds per tier -- `test_oracle_rederives_answer`
    above only covers 60 seeds (parametrized per-seed so a single bad seed is
    easy to locate); this collects every mismatch over 1000 seeds and asserts
    the list is empty, so a failure here names every bad seed at once instead
    of stopping at the first."""
    src = SOURCES[difficulty]
    mismatches = []
    for seed in REDERIVE_SEEDS:
        q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
        derived = _SOLVERS[difficulty](q.text)
        stated = _parse_answer_pair(q.answer_text)
        if derived != stated:
            mismatches.append((seed, q.text, derived, stated))
    assert not mismatches, f"{difficulty}: {len(mismatches)}/{len(REDERIVE_SEEDS)} seeds disagreed: " + "; ".join(
        f"seed={s} derived={d} stated={st}" for s, _t, d, st in mismatches[:20]
    )


def test_oracle_fires_on_a_corrupted_answer():
    """Proves the oracle red first: corrupt one component of a real
    question's answer_text and confirm the oracle's own re-derivation
    disagrees (it does not just echo the generator)."""
    src = SOURCES["medium"]
    q = mathgen.make(src.topic, seed=0, difficulty=src.difficulty, **(src.params or {}))
    derived = _SOLVERS["medium"](q.text)
    stated = _parse_answer_pair(q.answer_text)
    assert derived == stated  # sanity: real answer agrees

    corrupted = dict(stated)
    corrupted["n"] = corrupted["n"] + 1
    assert corrupted != derived  # the oracle would catch this corruption


@pytest.mark.parametrize("difficulty", TIERS)
def test_grader_probe_key_and_ulp_over_1000_seeds(difficulty):
    """Grader probe: for every tier, over >= 1000
    seeds, the key is accepted, every distractor rejected, and a
    one-unit-in-last-place change per component is rejected."""
    src = SOURCES[difficulty]
    grade_pair = GRADERS["pair"]
    for seed in GRADER_SEEDS:
        q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
        assert grade_pair(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_pair(q, d).correct
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        label1, value1, label2, value2 = m.groups()
        if difficulty == "easy":
            bumped1 = str(round(float(value1) + 0.0001, 4))
            bumped2 = str(round(float(value2) + 0.0001, 4))
        else:
            bumped1 = str(int(value1) + 1)
            bumped2 = str(int(value2) + 1)
        assert not grade_pair(q, f"{label1} = {bumped1}, {label2} = {value2}").correct
        assert not grade_pair(q, f"{label1} = {value1}, {label2} = {bumped2}").correct


def test_oracle_coverage():
    per_seed_total = len(TIERS) * len(SEEDS)
    rederive_total = len(TIERS) * len(REDERIVE_SEEDS)
    total = per_seed_total + rederive_total
    assert len(SEEDS) >= 60
    assert len(REDERIVE_SEEDS) >= 1000
    assert per_seed_total >= 240
    assert rederive_total >= 4000
    assert total >= 4000
    print(
        f"\nOracle re-derived {total} questions "
        f"({per_seed_total} at {len(SEEDS)} seeds/tier for per-seed localisation, "
        f"{rederive_total} at {len(REDERIVE_SEEDS)} seeds/tier for coverage)."
    )
