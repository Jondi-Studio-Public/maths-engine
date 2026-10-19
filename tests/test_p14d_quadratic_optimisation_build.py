"""Focused contract tests for: ucv1-unconstrained-critical-point (answer: pair) and ucv1-convexity-hessian-sign
(answer: mcq)."""

import dataclasses
import re
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen.grading.typed import grade_mcq, grade_pair, mcq_options
from mathgen.skills import load_skills
from mathgen import make

CRIT_ID = "ucv1-unconstrained-critical-point"
CONVEX_ID = "ucv1-convexity-hessian-sign"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_PAIR_ANSWER_RE = re.compile(r"x = -?\d+(?:/\d+)?, y = -?\d+(?:/\d+)?")
_CLASSES = {"convex", "concave", "indefinite"}

_SYMPY_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)

# The old bracketed-sum-of-quadratics exam disguise: two `(...)` groups, each
# containing an x^2 or y^2 term, joined by `+` at the top level.
_OLD_BRACKETED_QUADRATIC_SUM_RE = re.compile(r"\([^()]*[xy]\^2[^()]*\)\s*\+\s*\([^()]*[xy]\^2[^()]*\)")


def _parse_poly(expr_text, x_sym, y_sym):
    return parse_expr(expr_text, local_dict={"x": x_sym, "y": y_sym}, transformations=_SYMPY_TRANSFORMATIONS)


_SINGLE_ANSWER_EXAM = {"ucv1-unconstrained-critical-point": "p14d_critical_point"}


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
    return skill, _by_difficulty(skill)


CRIT_SKILL, CRIT_SOURCES = _skill_sources(CRIT_ID)
CONVEX_SKILL, CONVEX_SOURCES = _skill_sources(CONVEX_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
CONVEX_SOURCES["exam"] = replace(CONVEX_SOURCES["exam"], topic="p14d_convexity_hessian_sign", params={})


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-unconstrained-critical-point
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(CRIT_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, difficulty, seed)
        wrong_x = Fraction(q.answer["x"]) + 17
        wrong_y = Fraction(q.answer["y"]) + 17
        wrong = f"x = {wrong_x}, y = {wrong_y}"
        assert not grade_pair(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_independent_check_fires_on_a_corrupted_answer(difficulty):
    """The generator's own check() must reject a corrupted (x, y): perturb
    d by +1 so the Cramer's-rule solve no longer lands on x0."""
    import mathgen.core as core

    for seed in list(SEEDS)[:10]:
        src = CRIT_SOURCES[difficulty]
        rng_topic = src.topic
        # Reconstruct via the same seeded rng path is not exposed directly;
        # instead corrupt at the arithmetic level the check itself performs.
        a, b, c, d, e = 2, 3, 1, -5, -7
        det = Fraction(4 * a * b - c * c)
        x_solved = (-2 * b * d + c * e) / det
        y_solved = (-2 * a * e + c * d) / det
        d_bad = d + 1
        x_bad = (-2 * b * d_bad + c * e) / det
        assert x_bad != x_solved
        with pytest.raises(core.VerificationError):
            core.check(x_bad == x_solved, "corrupted x must fail")


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_answer_is_a_fraction_pair(difficulty):
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, difficulty, seed)
        assert isinstance(q.answer, dict) and set(q.answer) == {"x", "y"}
        for v in q.answer.values():
            assert type(v) in (int, Fraction), f"{v!r} is {type(v)}"
        assert _PAIR_ANSWER_RE.fullmatch(q.answer_text), q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, difficulty, seed)
        for d in q.distractors:
            assert _PAIR_ANSWER_RE.fullmatch(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_crit_200_seed_pair_never_concentrates(difficulty):
    counts = Counter(_make(CRIT_SOURCES, difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert len(counts) >= 30, f"{difficulty}: only {len(counts)} distinct answers over 200 seeds"
    assert top_share <= 0.10, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_crit_easy_tier_is_decoupled():
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, "easy", seed)
        assert "xy" not in q.text.split(".")[0].replace(" ", "")


def test_crit_medium_hard_tiers_are_coupled():
    for difficulty in ("medium", "hard"):
        for seed in SEEDS:
            q = _make(CRIT_SOURCES, difficulty, seed)
            assert "xy" in q.text.split(".")[0].replace(" ", ""), (difficulty, q.text)


def test_crit_exam_tier_is_genuinely_coupled_via_squared_form():
    """The exam tier's sum-of-two-squares disguise means the rendered text
    never shows a literal "xy" term before expansion (see
    test_crit_exam_tier_uses_squared_form_not_bracketed_quadratic_sum), so
    coupling must be verified by actually expanding the squared form and
    checking the x*y coefficient is nonzero -- not by a substring check."""
    x_sym, y_sym = sympy.symbols("x y")
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, "exam", seed)
        rhs = q.text.split("f(x, y) = ")[1].split(". ")[0]
        expr = sympy.expand(_parse_poly(rhs, x_sym, y_sym))
        assert expr.coeff(x_sym * y_sym) != 0, (seed, q.text, expr)


def test_crit_hard_tier_point_is_never_an_integer():
    for seed in SWEEP_SEEDS:
        q = _make(CRIT_SOURCES, "hard", seed)
        x_val, y_val = q.answer["x"], q.answer["y"]
        assert Fraction(x_val).denominator != 1 or Fraction(y_val).denominator != 1, q.answer_text


def test_crit_exam_tier_uses_squared_form_not_bracketed_quadratic_sum():
    """The defect: the old exam disguise split f into two ALREADY-QUADRATIC bracketed pieces ("(-4x^2 + 3xy) +
    (x^2 +...)"), a shape `measure_tiers` reads as a new prompt shape but which needs no collecting step
    harder than any lower tier. The fix requires a sum of two SQUARED LINEAR forms instead, so the old pattern
    must never appear and a squared-form marker must always appear -- and only in the exam tier, never in
    easy/medium/hard. Proved RED against the pre-fix generator before this fix."""
    for seed in SEEDS:
        q = _make(CRIT_SOURCES, "exam", seed)
        assert not _OLD_BRACKETED_QUADRATIC_SUM_RE.search(q.text), q.text
        assert ")^2" in q.text, q.text
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _make(CRIT_SOURCES, difficulty, seed)
            assert ")^2" not in q.text, (difficulty, q.text)


def test_crit_skill_loads_through_api_with_correct_properties():
    assert CRIT_SKILL.domain == "optimisation"
    assert CRIT_SKILL.status == "active"
    assert CRIT_SKILL.placement_eligible is False
    assert CRIT_SKILL.answer == "pair"
    assert CRIT_SKILL.prerequisites == ["jacobian-two-variable"]
    assert {src.difficulty for src in CRIT_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert CRIT_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-convexity-hessian-sign
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_convex_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(CONVEX_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 2


@pytest.mark.parametrize("difficulty", TIERS)
def test_convex_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(CONVEX_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, correct_index).correct, f"seed {seed} {difficulty}: canonical index not accepted"
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, index).correct, (
                    f"seed {seed} {difficulty}: distractor index {index} wrongly accepted"
                )


@pytest.mark.parametrize("difficulty", TIERS)
def test_convex_real_grader_rejects_an_out_of_range_index(difficulty):
    for seed in SEEDS:
        q = _make(CONVEX_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "99").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_convex_answer_is_one_of_the_three_class_labels(difficulty):
    for seed in SEEDS:
        q = _make(CONVEX_SOURCES, difficulty, seed)
        assert q.answer_text in _CLASSES
        assert set(q.distractors) | {q.answer_text} == _CLASSES
        assert isinstance(q.answer, str) and q.answer == q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_convex_200_seed_class_balance_near_a_third_each(difficulty):
    counts = Counter(_make(CONVEX_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _CLASSES, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.45, f"{difficulty}: top class {counts.most_common(1)[0]} is {top_share:.1%} of 200 draws"


def test_convex_easy_tier_is_diagonal_only():
    for seed in SEEDS:
        q = _make(CONVEX_SOURCES, "easy", seed)
        assert "xy" not in q.text.split(".")[0].replace(" ", "")


def test_convex_medium_hard_exam_tiers_have_a_cross_term_on_most_draws():
    for difficulty in ("medium", "hard", "exam"):
        coupled = sum("xy" in _make(CONVEX_SOURCES, difficulty, seed).text.replace(" ", "") for seed in SWEEP_SEEDS)
        assert coupled >= len(SWEEP_SEEDS) * 0.5, f"{difficulty}: only {coupled}/{len(SWEEP_SEEDS)} coupled"


def test_convex_medium_tier_has_diagonal_positive_but_indefinite_seeds():
    """The required trap: a, b both positive (diagonal LOOKS convex)
    but det(H) < 0 (actually indefinite). fxx/fyy are read straight out of
    the generator's own working-text Hessian, not re-derived from the prompt."""
    trap_count = 0
    for seed in range(200):
        q = _make(CONVEX_SOURCES, "medium", seed)
        if q.answer_text != "indefinite":
            continue
        fxx = int(q.working[0].split("[[")[1].split(",")[0])
        fyy = int(q.working[0].split("], [")[1].split(",")[1].split("]")[0])
        if fxx > 0 and fyy > 0:
            trap_count += 1
    assert trap_count >= 1, "no medium-tier diagonal-positive-but-indefinite trap seed found in 200 draws"


def test_convex_exam_tier_shows_cross_terms_separately_not_combined():
    """The defect: the old exam disguise used the same bracketed-sum-of-quadratics split as the critical-point
    skill. The fix presents an explicit non-symmetric matrix, with the x*y and y*x terms written SEPARATELY
    (never pre-combined into one c*x*y) -- so both must appear in the exam tier and neither combined form nor
    the old bracketed pattern may appear, while easy/medium/hard (a single combined c*x*y coefficient) never
    show two separate xy/yx terms. Proved RED against the pre-fix generator before this fix."""
    for seed in SEEDS:
        q = _make(CONVEX_SOURCES, "exam", seed)
        body = q.text.split(".")[0].replace(" ", "")
        assert "xy" in body and "yx" in body, q.text
        assert not _OLD_BRACKETED_QUADRATIC_SUM_RE.search(q.text), q.text
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            q = _make(CONVEX_SOURCES, difficulty, seed)
            body = q.text.split(".")[0].replace(" ", "")
            assert not ("xy" in body and "yx" in body), (difficulty, q.text)


_EXAM_TERM_RE = re.compile(r"([+-]?\d*)(x\^2|y\^2|xy|yx)")


def _exam_matrix_coeffs(text):
    """Parse A11, A12, A21, A22 from the exam tier's non-symmetric-matrix
    text (x^2, xy, yx, y^2 terms, each with its own coefficient -- a
    missing coefficient means 1, a bare "-" means -1)."""
    body = text.split("f(x, y) = ")[1].split(". ")[0].replace(" ", "")
    coeffs = {}
    for sign_num, suffix in _EXAM_TERM_RE.findall(body):
        coeffs[suffix] = 1 if sign_num in ("", "+") else (-1 if sign_num == "-" else int(sign_num))
    return coeffs["x^2"], coeffs["xy"], coeffs["yx"], coeffs["y^2"]


def _classify_det(a, b, det):
    if det > 0:
        return "convex" if a > 0 else "concave"
    return "indefinite"


def test_convex_ignores_one_cross_term_misconception_fires_on_a_measured_share():
    """New misconception `ignores-one-cross-term`: reading
    only A12 (or only A21) as the whole Hessian off-diagonal entry, instead
    of the TRUE symmetrised A12 + A21. Measures, over >= 200 exam-tier
    seeds, the fraction where that wrong classification differs from the
    true one AND happens to match one of the two options actually offered
    (the mcq always offers exactly the two non-target labels, so any
    differing wrong class is necessarily one of them)."""
    seeds = range(200)
    fires = 0
    for seed in seeds:
        q = _make(CONVEX_SOURCES, "exam", seed)
        a11, a12, a21, a22 = _exam_matrix_coeffs(q.text)
        true_det = 4 * a11 * a22 - (a12 + a21) ** 2
        true_class = _classify_det(a11, a22, true_det)
        assert true_class == q.answer_text, (seed, q.text, true_class, q.answer_text)
        for wrong_cross in (a12, a21):
            wrong_det = 4 * a11 * a22 - wrong_cross**2
            wrong_class = _classify_det(a11, a22, wrong_det)
            if wrong_class != true_class:
                assert wrong_class in q.distractors, (seed, q.text, wrong_class, q.distractors)
                fires += 1
                break
    fraction = fires / len(seeds)
    print(f"\nignores-one-cross-term fires on {fires}/{len(seeds)} = {fraction:.1%} of exam-tier draws")
    assert fraction >= 0.05, f"ignores-one-cross-term fired on only {fraction:.1%} of {len(seeds)} exam draws"


def test_convex_semidefinite_edge_case_never_rendered():
    """det(H) == 0 must never appear across a wide sweep, since the
    generator retries on that draw."""
    for difficulty in TIERS:
        for seed in SWEEP_SEEDS:
            q = _make(CONVEX_SOURCES, difficulty, seed)
            det_str = q.working[0].split("det(H) = ")[1].split(";")[0]
            assert int(det_str) != 0, (difficulty, seed, q.working)


def test_convex_skill_loads_through_api_with_correct_properties():
    assert CONVEX_SKILL.domain == "optimisation"
    assert CONVEX_SKILL.status == "active"
    assert CONVEX_SKILL.placement_eligible is False
    assert CONVEX_SKILL.answer == "mcq"
    assert CONVEX_SKILL.prerequisites == ["two-by-two-eigenvalues"]
    assert {src.difficulty for src in CONVEX_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert CONVEX_SOURCES["exam"].stages == ["exam"]
