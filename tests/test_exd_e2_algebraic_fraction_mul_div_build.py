"""Tests for algebraic-fraction-multiply-divide."""

import re

import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import symbolically_equal

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.
_x, _y = Symbol("x"), Symbol("y")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_EASY_RE = re.compile(r"^\((\d+)x/(\d+)\) × \((\d+)/\((\d+)x\)\)$")
_MEDIUM_RE = re.compile(r"^\((\d+)x/(\d+)\) / \((\d+)x/(\d+)\)$")
_HARD_RE = re.compile(r"^\((\d+)xy/(\d+)\) / \((\d+)x/\((\d+)y\)\)$")
_EXAM_CHAIN3_RE = re.compile(r"^\((\d+)xy/(\d+)\) / \((\d+)x/\((\d+)y\)\) (×|/) \((\d+)/(\d+)\)$")
_EXAM_POWER_RE = re.compile(r"^\((\d+)y/(\d+)\)\^2 / \((\d+)y/(\d+)\)$")

_SENTENCE_SUFFIXES = (
    ", giving your answer in simplest form.",
    " to a single term.",
    ".",
)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": _x, "y": _y})


def _all_questions(tier, seeds=SEEDS):
    return [make("exd_e2_algebraic_fraction_mul_div", seed=s, difficulty=tier) for s in seeds]


def _strip_sentence_frame(text: str) -> str:
    """The bare expression a question's text wraps -- strips the "Simplify "
    prefix and whichever of the generator's three sentence frames was
    chosen for this seed, so the per-tier regexes below only ever see the
    mathematical expression itself, never the wording variety around it."""
    prefix = "Simplify "
    assert text.startswith(prefix), f"unrecognised sentence prefix: {text!r}"
    body = text[len(prefix) :]
    for suffix in _SENTENCE_SUFFIXES:
        if body.endswith(suffix):
            return body[: -len(suffix)]
    raise AssertionError(f"unrecognised sentence frame: {text!r}")


def _own_unreduced_text(q) -> str:
    """The question's own literal uncancelled product/quotient text, parsed
    from `q.text` with a regex (never the generator's internal variables),
    rebuilt with explicit `*`/`**` so sympy parses it exactly as the
    question's own drawn numbers describe it -- never simplified, never
    cancelled."""
    expr = _strip_sentence_frame(q.text)
    m = _EASY_RE.match(expr)
    if m:
        a, b, c, d = m.groups()
        return f"({a}*x/{b})*({c}/({d}*x))"
    m = _MEDIUM_RE.match(expr)
    if m:
        a, b, c, d = m.groups()
        return f"({a}*x/{b})/({c}*x/{d})"
    m = _HARD_RE.match(expr)
    if m:
        m1, n1, m2, n2 = m.groups()
        return f"({m1}*x*y/{n1})/({m2}*x/({n2}*y))"
    m = _EXAM_CHAIN3_RE.match(expr)
    if m:
        m1, n1, m2, n2, op2, m3, n3 = m.groups()
        op = "*" if op2 == "×" else "/"
        return f"({m1}*x*y/{n1})/({m2}*x/({n2}*y)){op}({m3}/{n3})"
    m = _EXAM_POWER_RE.match(expr)
    if m:
        m1, n1, m2, n2 = m.groups()
        return f"({m1}*y/{n1})**2/({m2}*y/{n2})"
    raise AssertionError(f"unrecognised prompt shape {expr!r} (from {q.text!r})")


class TestAlgebraicFractionMulDiv:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("exd_e2_algebraic_fraction_mul_div", seed=seed, difficulty=tier)
            q2 = make("exd_e2_algebraic_fraction_mul_div", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_expanded(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "expanded", f"{tier}: {q.text!r} should be answer_form=expanded"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_never_contains_a_bracket(self, tier):
        # The design constraint: the fully-cancelled answer is always a bare
        # monomial numerator over a bare positive integer denominator.
        for q in _all_questions(tier):
            assert "(" not in q.answer_text, f"{tier}: answer contains a bracket: {q.answer_text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike_no_bracket_no_double_star(self, tier):
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            for o in options:
                assert "(" not in o, f"{tier}: bracketed option {o!r} in {q.text!r}"
                assert "**" not in o, f"{tier}: Python exponent notation in option {o!r} in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for q in _all_questions(tier):
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"{tier}: grader rejected canonical {q.answer_text!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for q in _all_questions(tier):
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"{tier}: grader accepted distractor {d!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, tier):
        for q in _all_questions(tier, seeds=range(10)):
            wrong = "x + 999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
    def test_forgot_to_invert_exercisable_over_200_seeds(self, tier):
        # Measured, not assumed, on every OTHER tier: every one of them is a divide (exam's power
        # variant is forced to a division specifically so this holds there too, see the generator
        # module's own docstring). forgot-to-invert is the only misconception that leaves x^2
        # (medium, hard, exam-chain3) or y^3 (exam-power) in the option text -- every other
        # misconception's variable part matches the answer's own shape.
        pool = set()
        for q in _all_questions(tier, seeds=range(200)):
            pool.update(d for d in q.distractors if "x^2" in d or "y^3" in d)
        assert pool, f"{tier}: forgot-to-invert never observed over 200 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_misconceptions_measured_exercisable_over_200_seeds(self, tier):
        distractor_pool = set()
        for q in _all_questions(tier, seeds=range(200)):
            distractor_pool.update(q.distractors)
        assert len(distractor_pool) > 1, f"{tier}: distractor pool too small over 200 seeds: {distractor_pool}"

    def test_exam_clears_three_distinct_shapes(self):
        from mathgen.text_quality import prompt_shape

        shapes = {prompt_shape(q) for q in _all_questions("exam")}
        assert len(shapes) >= 3, f"exam: only {len(shapes)} distinct prompt shapes over 60 seeds (need >= 3): {shapes}"


class TestOwnFormRejectedProof:
    """'s mandatory proof, run on every generated question (not a sample): the question's own literal uncancelled
    product/quotient text (a) is PROVED value-equal to the canonical answer by ordinary sympy equivalence (the
    hazard's precondition -- if this fails, the "rejected" assertion below would prove nothing about the form
    check specifically), and (b) is REJECTED by the real grader against that question's own answer_text."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unreduced_form_is_value_equal_but_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            unreduced_text = _own_unreduced_text(q)
            unreduced_expr = _parse(unreduced_text)
            checked += 1
            assert symbolically_equal(unreduced_expr, q.answer), (
                f"{tier}: {unreduced_text!r} is not value-equal to {q.answer!r} for {q.text!r} -- "
                "construction bug, the rejection proof below would be meaningless"
            )
            result = GRADERS["expression"](q, unreduced_text)
            assert not result.correct, (
                f"{tier}: grader accepted the question's own unreduced form "
                f"{unreduced_text!r} for {q.text!r} (answer {q.answer_text!r})"
            )
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(f"{tier}: own-unreduced-form value-equal-but-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "algebraic-fraction-multiply-divide")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"simplify-algebraic-fractions"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "exd_e2_algebraic_fraction_mul_div"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
