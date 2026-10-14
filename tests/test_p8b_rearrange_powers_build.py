"""Widened, narrowed again: tests for the `rearrange_powers_roots` topic, which extends the LIVE
`rearrange-simple-formulas` skill (`skills/equations/rearrange-simple-formulas.yaml`) with formulas whose
target sits inside a power or a square root.

See `mathgen/generators/p8b_rearrange_powers_build.py`'s module docstring for the full grading-gap writeup and
`_CASES` for the current bank. This file's pool-size test now asserts 7, not 10 or 3, over the same 60 seeds;
there is still no per-seed numeric draw, only `rng.choice` among the fixed cases, so a "distinct question"
count over 60 seeds is a pool-coverage check, not a variety-of-numbers check.

That was REVERTED: it broke the real grading pipeline. `answer_text` is exactly what a learner who matches the
canonical answer verbatim submits, and `mathgen/grading/typed.py:_parse_expression` cannot round-trip unicode
`√` (mis-parses silently) or unicode `π` (parses as an unrelated free symbol, not `sympy.pi`). Direct
measurement: 134/160 real-grader accepts failed over this topic's own seed sweep with the unicode substitution
in place, and `scripts/regrade_sweep.py` independently caught the same defect (`false_rejects=34` for
`rearrange-simple-formulas`). So `answer_text` stays ASCII (`pi`, `sqrt(...)`) same as `Question.answer`, and
the test below that types `answer_text` into the real grader does so verbatim, with no reversal helper needed.
"""

import functools

import pytest
from sympy import Eq, Integer, Mul, Pow, pi, solve, sqrt, symbols, sympify, Rational

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent, symbolically_equal

SEEDS = range(60)


def _combine_like_radicals(expr):
    """Mechanically build the "natural alternative form" a learner would type for a canonical answer that is a
    Mul of same-index radical factors -- e.g. `sqrt(P)/sqrt(R)` -> `sqrt(P/R)` -- WITHOUT the case's own
    positivity assumptions (fresh, assumption-free sympy objects, exactly what
    `mathgen/grading/typed.py:_parse_expression` produces from a learner's typed text). Returns `None` when
    there is nothing to combine (fewer than two same-index radical factors), so the caller can skip that case
    rather than assert a no-op. This is the exact transform whose absence-of-a-test let
    `physics_c`/`physics_I`/`vertex_x` ship with a false-reject defect (works generically off whatever
    `_CASES` currently contains), not the three known-bad formulas by name, so a future case with this shape
    fails this test automatically instead of shipping unnoticed."""
    if not expr.is_Mul:
        return None
    groups = {}
    other = []
    for factor in expr.args:
        if isinstance(factor, Pow) and factor.exp.is_Rational and factor.exp.q != 1:
            key = abs(factor.exp)
            num, den = groups.setdefault(key, ([], []))
            (num if factor.exp > 0 else den).append(factor.base)
        else:
            other.append(factor)
    if not groups:
        return None
    factors = list(other)
    changed = False
    for key, (num, den) in groups.items():
        if len(num) + len(den) <= 1:
            factors.append(Pow(num[0], key) if num else Pow(den[0], -key))
            continue
        changed = True
        combined_base = Mul(*num, evaluate=True) if num else Integer(1)
        if den:
            combined_base = combined_base / Mul(*den, evaluate=True)
        factors.append(Pow(combined_base, key, evaluate=False))
    return Mul(*factors, evaluate=False) if changed else None


@functools.lru_cache(maxsize=1)
def _skills():
    # load_skills() rescans every YAML file (~563ms); cache it at module
    # scope so no per-seed loop pays that cost.
    return load_skills()


def _skill(skill_id):
    _domains, skills = _skills()
    for s in skills:
        if s.id == skill_id:
            return s
    raise AssertionError(f"skill {skill_id!r} not found")


# --- determinism / pool size ------------------------------------------------


def test_pool_size_is_at_least_seven():
    """7 (formula, target) cases exist by design. 60 seeds is enough to see all 7 with
    `rng.choice`."""
    seen = set()
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        seen.add((q.text, q.answer_text))
    assert len(seen) >= 7, seen


# --- answer_form must never be set -------------


def test_answer_form_is_never_set():
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        assert q.answer_form is None
    # also confirm it is omitted from the serialised dict entirely, exactly
    # like every other topic that never sets it.
    q = make("rearrange_powers_roots", seed=0, difficulty="hard")
    assert "answer_form" not in q.to_dict()


# --- distractor hygiene ------------------------------------------------------


def test_distractors_distinct_nonempty_and_never_equal_answer():
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        assert len(q.distractors) >= 2
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))


def test_answer_and_distractors_formatted_alike():
    """All options are plain sympy-derived algebraic expressions -- none is
    identifiable by shape (no option is, say, a decimal among fractions)."""
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        for opt in [q.answer_text, *q.distractors]:
            assert all(ch not in opt for ch in ("=", ":"))


# --- real grader: accept canonical, reject distractors and a wrong answer --


def test_real_grader_accepts_canonical_and_rejects_distractors():
    """Types `answer_text` (and every distractor) into the real grader
    VERBATIM -- exactly what a learner who matches the canonical text sees
    and submits. `answer_text` is ASCII (`pi`/`sqrt(...)`), same as
    `Question.answer`, so this round-trips (see module docstring for why the
    unicode-display attempt was reverted). Swept over all 60 seeds, which per
    `test_pool_size_is_at_least_ten` covers every one of the 10 cases, not a
    sample -- the display defect this test guards was per-case, not
    per-seed."""
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        assert grade_expression(q, q.answer_text).correct
        for distractor in q.distractors:
            assert not grade_expression(q, distractor).correct


def test_alternative_radical_form_is_accepted_for_every_case():
    """Pins the grading-gap PROPERTY, not the three
    formulas that were dropped for violating it: for every case in the
    current bank whose canonical answer contains two or more same-index
    radical factors combined by multiplication/division (a split root, e.g.
    `sqrt(A)/sqrt(pi)`), the single-radical form a learner would naturally
    write instead (`sqrt(A/pi)`) must ALSO be graded correct. Built
    mechanically via `_combine_like_radicals` (assumption-free, mirroring
    what a learner's typed text actually parses to), not a hand-picked list
    of strings -- so a future case added to `_CASES` with this shape fails
    this test automatically, the way `physics_c`/`physics_I`/`vertex_x`
    should have but didn't have a test to catch.

    Green again with `physics_I` removed."""
    checked_any_radical_case = False
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        canonical = sympify(q.answer)
        alt_expr = _combine_like_radicals(canonical)
        if alt_expr is None:
            continue  # nothing to combine for this case -- not what this test checks
        checked_any_radical_case = True
        alt = str(alt_expr)
        assert grade_expression(q, alt).correct, (q.text, q.answer, alt)
    assert checked_any_radical_case, "no seed drew a case with a combinable split radical"


def test_real_grader_rejects_a_deliberately_wrong_answer():
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        assert not grade_expression(q, "999999").correct


def test_equal_but_differently_arranged_answer_is_accepted():
    """Regression guard required: a rearranged formula has no single required shape. Find an `l`-target question
    (canonical `T^2g/(4pi^2)`) and confirm an algebraically-equal but differently-arranged string also grades
    correct."""
    checked = False
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        if q.text.startswith("Make l the subject"):
            assert q.answer_text == "T^2g/(4pi^2)"
            alt = "g*T**2/4/pi**2"
            assert grade_expression(q, alt).correct
            checked = True
            break
    assert checked, "no seed in range drew the l-target case"


def test_equal_but_differently_arranged_answer_is_accepted_for_a_new_case():
    """The SUVAT case has no such hazard: `a` appears only linearly, so any equivalent rearrangement grades
    correct."""
    checked = False
    for seed in SEEDS:
        q = make("rearrange_powers_roots", seed=seed, difficulty="hard")
        if q.text.startswith("Make a the subject of the formula s = ut"):
            # `solve()`'s Add-canonicalisation for this particular equation
            # is not guaranteed structurally stable across processes (see
            # the note above `test_solve_output_for_sphere_r`), so
            # this checks the VALUE of `answer_text`, not its exact string,
            # unlike the pre-existing `l`-case test above (whose
            # `T^2g/(4pi^2)` form has no internal `Add` to reorder).
            import sympy
            from mathgen.grading.typed import _parse_expression

            s, u, t = sympy.symbols("s u t")
            canonical_value = _parse_expression(q.answer_text)
            assert symbolically_equal(canonical_value, 2 * (s - t * u) / t**2)
            alt = "2*s/t**2 - 2*u/t"
            assert alt != q.answer_text  # genuinely a different arrangement
            assert grade_expression(q, alt).correct
            checked = True
            break
    assert checked, "no seed in range drew the suvat-a case"


# --- independent check fires on a corrupted answer --------------------------


def test_check_equivalent_raises_on_a_corrupted_answer():
    A, r = symbols("A r", positive=True)
    solutions = solve(Eq(A, pi * r**2), r)
    assert len(solutions) == 1
    correct = solutions[0]
    corrupted = correct + 1  # deliberately wrong
    with pytest.raises(VerificationError):
        check_equivalent(A.subs(r, corrupted), (pi * r**2).subs(r, corrupted))


def test_check_distinct_raises_when_candidate_equals_answer():
    A, r = symbols("A r", positive=True)
    solutions = solve(Eq(A, pi * r**2), r)
    correct = solutions[0]
    with pytest.raises(VerificationError):
        check_distinct(correct, correct)  # a candidate equal to the answer


# --- solve() output shape, for every case -----------------------------------


def test_solve_reduces_r_to_a_single_positive_root():
    A, r = symbols("A r", positive=True)
    solutions = solve(Eq(A, pi * r**2), r)
    assert solutions == [sqrt(A) / sqrt(pi)]


def test_solve_output_for_l_and_g():
    T, l, g = symbols("T l g", positive=True)
    l_solutions = solve(Eq(T, 2 * pi * sqrt(l / g)), l)
    g_solutions = solve(Eq(T, 2 * pi * sqrt(l / g)), g)
    assert l_solutions == [T**2 * g / (4 * pi**2)]
    assert g_solutions == [4 * pi**2 * l / T**2]


# NOTE: these new-case checks compare `solve()`'s output via
# `symbolically_equal` (value equality), not a strict `==` list comparison.
# `solve()`'s automatic Add/Mul canonicalisation for these formulas is not
# always structurally stable across processes with different hash seeds
# (confirmed empirically: `2*(s - t*u)/t**2` vs `(2*s - 2*t*u)/t**2` for the
# SUVAT-a case, same value, under pytest-xdist's separate worker processes)
# -- the two pre-existing tests above keep `==` because their outputs
# (`sqrt(A)/sqrt(pi)`, `T**2*g/(4*pi**2)`) have no internal `Add` for the
# canonicaliser to reorder and are stable in practice.


def test_solve_output_for_sphere_r():
    V, r = symbols("V r", positive=True)
    solutions = solve(Eq(V, Rational(4, 3) * pi * r**3), r)
    assert len(solutions) == 1
    assert symbolically_equal(solutions[0], 6 ** Rational(1, 3) * V ** Rational(1, 3) / (2 * pi ** Rational(1, 3)))


def test_solve_output_for_pythagoras_a():
    a, b, c = symbols("a b c", positive=True)
    solutions = solve(Eq(c, sqrt(a**2 + b**2)), a)
    assert len(solutions) == 1
    assert symbolically_equal(solutions[0], sqrt(-(b**2) + c**2))


def test_solve_output_for_suvat_a_and_u():
    s, u, a = symbols("s u a", real=True)
    t = symbols("t", positive=True)
    a_solutions = solve(Eq(s, u * t + Rational(1, 2) * a * t**2), a)
    u_solutions = solve(Eq(s, u * t + Rational(1, 2) * a * t**2), u)
    assert len(a_solutions) == 1
    assert len(u_solutions) == 1
    assert symbolically_equal(a_solutions[0], 2 * (s - t * u) / t**2)
    assert symbolically_equal(u_solutions[0], -a * t / 2 + s / t)


# --- skill loads correctly ---------------------------------------------------


def test_skill_loads_with_new_source_and_unchanged_declared_fields():
    skill = _skill("rearrange-simple-formulas")
    assert skill.answer == "expression"
    # placement_eligible/prerequisites/etc. are declared fields this test
    # must not touch -- assert they are unchanged from
    # what the YAML already stated, not a new-skill convention.
    assert skill.prerequisites == ["variables-on-both-sides", "use-formulas"]
    topics = {src.topic for src in skill.sources}
    assert "rearrange_powers_roots" in topics
    assert "rearrange" in topics  # existing source untouched
    new_sources = [src for src in skill.sources if src.topic == "rearrange_powers_roots"]
    assert len(new_sources) == 1
    assert new_sources[0].difficulty == "hard"
    assert set(new_sources[0].stages) == {5, 6, "maintenance"}
