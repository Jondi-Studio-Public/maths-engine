"""Tests for simplify-radicals-of-monomials."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import symbolically_equal

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.

_RADICAND_RE = re.compile(r"∛\(([^)]+)\) ÷ √\(([^)]+)\)")
_COEF_RE = re.compile(r"^-?\d+")
_VAR_TOKEN_RE = re.compile(r"([a-c])(?:\^(\d+))?")


def _all_questions(tier, seeds=SEEDS):
    return [make("exd_e2_simplify_radicals_of_monomials", seed=s, difficulty=tier) for s in seeds]


def _extract_radicands(text: str) -> tuple[str, str]:
    """The two radicand substrings (cube, square) from a question's own
    prompt text, via regex -- never the generator's internal draw
    variables: the text is built from the question's own
    drawn numbers."""
    m = _RADICAND_RE.search(text)
    assert m, f"unrecognised prompt shape (no radicand pair found): {text!r}"
    return m.group(1), m.group(2)


def _parse_radicand(text: str) -> tuple[int, dict[str, int]]:
    """One radicand string ('8000a^6b^3' implicit-multiplication style) ->
    (coefficient, {var: exp}). Generic over variable COUNT (0-3), unlike a
    single fixed-shape regex -- hard's own num_vars varies question to
    question."""
    m = _COEF_RE.match(text)
    assert m, f"unrecognised radicand coefficient in {text!r}"
    coef = int(m.group(0))
    exps: dict[str, int] = {}
    for var, exp in _VAR_TOKEN_RE.findall(text):
        exps[var] = int(exp) if exp else 1
    return coef, exps


def _integer_cbrt_signed(n: int) -> int:
    """The exact (possibly negative) integer cube root of a perfect cube --
    asserts n really is one rather than silently rounding, so a construction
    bug fails loudly here rather than producing a meaningless proof."""
    sign = -1 if n < 0 else 1
    magnitude = abs(n)
    r = round(magnitude ** (1 / 3))
    for cand in (r - 1, r, r + 1):
        if cand**3 == magnitude:
            return sign * cand
    raise AssertionError(f"not a perfect cube: {n}")


def _integer_sqrt(n: int) -> int:
    r = round(n**0.5)
    for cand in (r - 1, r, r + 1):
        if cand * cand == n:
            return cand
    raise AssertionError(f"not a perfect square: {n}")


def _fmt_pow_py(var: str, exp: int) -> str:
    return var if exp == 1 else f"{var}**{exp}"


def _extracted_uncancelled_text(cube_text: str, square_text: str) -> str:
    """The PRIMARY, required rejection-target form (the hazard):
    each root's own coefficient/exponents extracted separately, never
    combined/cancelled against the other root -- always a compound (hence
    bracketed) denominator by this module's own construction (every
    variable's d_v >= 1 unconditionally, see the generator module
    docstring), so this form is provably value-equal to the canonical by
    ordinary polynomial arithmetic and rejected only by the form check."""
    cube_coef, cube_exps = _parse_radicand(cube_text)
    square_coef, square_exps = _parse_radicand(square_text)
    cbrt_coef = _integer_cbrt_signed(cube_coef)
    sqrt_coef = _integer_sqrt(square_coef)
    numer = [str(cbrt_coef)] + [_fmt_pow_py(v, cube_exps[v] // 3) for v in sorted(cube_exps)]
    denom = [str(sqrt_coef)] + [_fmt_pow_py(v, square_exps[v] // 2) for v in sorted(square_exps)]
    return f"{'*'.join(numer)}/({'*'.join(denom)})"


def _monomial_text_py(coef: int, exps: dict[str, int]) -> str:
    """(coef, {var: exp}) -> explicit-'*' Python text ('8000*a**6*b**3'),
    parseable by plain `sympy.sympify` (no implicit-multiplication
    transformation) -- unlike the generator's own `.text`, which
    deliberately omits '*' (see that module's `_monomial_text` docstring)."""
    parts = [str(coef)] + [_fmt_pow_py(v, exps[v]) for v in sorted(exps)]
    return "*".join(parts)


def _fully_unextracted_text(cube_text: str, square_text: str) -> str:
    """The BONUS, honestly-caveated rejection-target form: the literal
    cbrt(...)/sqrt(...) text, never parsed/extracted at all -- rebuilt with
    explicit '*' from the PARSED (coefficient, {var: exp}) representation
    (never a naive string replace on the generator's own '*'-free text)."""
    cube_coef, cube_exps = _parse_radicand(cube_text)
    square_coef, square_exps = _parse_radicand(square_text)
    return f"cbrt({_monomial_text_py(cube_coef, cube_exps)})/sqrt({_monomial_text_py(square_coef, square_exps)})"


class TestSimplifyRadicalsOfMonomials:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("exd_e2_simplify_radicals_of_monomials", seed=seed, difficulty=tier)
            q2 = make("exd_e2_simplify_radicals_of_monomials", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
            for dist in q.distractors:
                assert dist != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_expanded(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "expanded", f"{tier}: {q.text!r} should be answer_form=expanded"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_never_contains_a_bracket(self, tier):
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
    def test_options_have_at_most_one_denominator_pow(self, tier):
        for q in _all_questions(tier):
            for o in [q.answer_text] + list(q.distractors):
                assert o.count("/") <= 1, f"{tier}: option {o!r} has more than one '/' in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for q in _all_questions(tier):
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"{tier}: grader rejected canonical {q.answer_text!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for q in _all_questions(tier):
            for dist in q.distractors:
                result = GRADERS["expression"](q, dist)
                assert not result.correct, f"{tier}: grader accepted distractor {dist!r} for {q.text!r}"

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

    @pytest.mark.parametrize("tier", TIERS)
    def test_misconceptions_measured_exercisable_over_200_seeds(self, tier):
        distractor_pool = set()
        for q in _all_questions(tier, seeds=range(200)):
            distractor_pool.update(q.distractors)
        assert len(distractor_pool) > 1, f"{tier}: distractor pool too small over 200 seeds: {distractor_pool}"

    @pytest.mark.parametrize("tier", ["easy", "medium"])
    def test_no_denominator_variable_at_easy_or_medium(self, tier):
        for q in _all_questions(tier):
            assert "/" not in q.answer_text, f"{tier}: unexpected denominator in {q.answer_text!r}"

    @pytest.mark.parametrize("tier", ["hard", "exam"])
    def test_denominator_variable_present_every_question(self, tier):
        for q in _all_questions(tier):
            assert "/" in q.answer_text, f"{tier}: expected a denominator variable in {q.answer_text!r}"

    def test_hard_uses_two_or_three_variables(self):
        var_counts = set()
        for q in _all_questions("hard"):
            cube_text, square_text = _extract_radicands(q.text)
            _, square_exps = _parse_radicand(square_text)
            var_counts.add(len(square_exps))
        assert var_counts == {2, 3}, f"hard: expected both 2- and 3-variable questions over 60 seeds, got {var_counts}"

    def test_exam_always_three_variables(self):
        for q in _all_questions("exam"):
            cube_text, square_text = _extract_radicands(q.text)
            _, square_exps = _parse_radicand(square_text)
            assert len(square_exps) == 3, f"exam: expected 3 variables, got {square_exps} in {q.text!r}"

    def test_exam_negative_coefficient_exercised_over_60_seeds(self):
        negatives = [q for q in _all_questions("exam") if q.answer_text.startswith("-")]
        assert negatives, "exam: negative-coefficient answer never observed over 60 seeds"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_no_negative_coefficient_below_exam(self, tier):
        for q in _all_questions(tier):
            assert not q.answer_text.startswith("-"), f"{tier}: unexpected negative coefficient in {q.answer_text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_exam_clears_three_distinct_shapes(self, tier):
        from mathgen.text_quality import prompt_shape

        shapes = {prompt_shape(q) for q in _all_questions(tier)}
        assert len(shapes) >= 3, (
            f"{tier}: only {len(shapes)} distinct prompt shapes over 60 seeds (need >= 3): {shapes}"
        )


class TestOwnFormRejectedProof:
    """The PRIMARY, required check: the roots-extracted-but-uncancelled form is (a) proved value-equal to the
    canonical by ordinary sympy equivalence (the hazard's precondition), and (b) proved rejected by the real
    grader. The BONUS, honestly-caveated check: the fully unextracted cbrt(...)/sqrt(...) form is also
    rejected, but its rejection is NOT claimed to prove the form check specifically -- see the module
    docstring's own measured note that it is not even value-equal for generic symbols."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_extracted_uncancelled_form_is_value_equal_but_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            cube_text, square_text = _extract_radicands(q.text)
            extracted = _extracted_uncancelled_text(cube_text, square_text)
            checked += 1
            assert symbolically_equal(extracted, q.answer), (
                f"{tier}: {extracted!r} is not value-equal to {q.answer!r} for {q.text!r} -- "
                "construction bug, the rejection proof below would be meaningless"
            )
            result = GRADERS["expression"](q, extracted)
            assert not result.correct, (
                f"{tier}: grader accepted the question's own extracted-uncancelled form "
                f"{extracted!r} for {q.text!r} (answer {q.answer_text!r})"
            )
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(
            f"{tier}: extracted-uncancelled value-equal-but-rejected proof (PRIMARY): "
            f"{checked}/{checked} passed, 0 failed"
        )

    @pytest.mark.parametrize("tier", TIERS)
    def test_fully_unextracted_form_rejected_bonus(self, tier):
        checked = 0
        value_equal_count = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            cube_text, square_text = _extract_radicands(q.text)
            full = _fully_unextracted_text(cube_text, square_text)
            checked += 1
            if symbolically_equal(full, q.answer):
                value_equal_count += 1
            result = GRADERS["expression"](q, full)
            assert not result.correct, (
                f"{tier}: grader accepted the question's own fully-unextracted radical form "
                f"{full!r} for {q.text!r} (answer {q.answer_text!r})"
            )
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(
            f"{tier}: fully-unextracted-radical rejected proof (BONUS, not form-check-specific): "
            f"{checked}/{checked} rejected, {value_equal_count}/{checked} were ALSO value-equal "
            "(generally 0 -- generic symbols cannot prove the root identity)"
        )


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "simplify-radicals-of-monomials")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"simplify-surds"}
        stages_covered = set()
        for source in skill.sources:
            # The exam tier is now a multi-part chain (tests/test_exd_x_r2j_radical_monomial_chain.py).
            # ...and the single-part exam question is back as a second exam source, key exam-single (bf2fca7).
            chain = source.difficulty == "exam" and source.key != "exam-single"
            assert source.topic == (
                "exd_x_r2j_radical_monomial_chain" if chain else "exd_e2_simplify_radicals_of_monomials"
            )
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 2  # the multi-part chain, then the exam-single source
        assert exam_sources[-1].key == "exam-single"
        assert all(s.stages == ["exam"] for s in exam_sources)
