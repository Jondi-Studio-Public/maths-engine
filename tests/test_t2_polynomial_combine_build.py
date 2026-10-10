"""Tests for add/subtract-polynomials and polynomial-linear-combination. See the skill YAMLs' descriptions."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the sweep width.

_ADD_SUB_RE = re.compile(r"^A = (.+?)\. B = (.+?)\. C = (.+?)\. Find A \+ B - C\.$")
_LINCOMB_RE = re.compile(r"^P = (.+?)\. Q = (.+?)\. R = (.+?)\. Find (.+?)\.$")
_TERM_RE = re.compile(r"\s*([+-])\s*(\d+)?(x(?:\^(\d+))?)?")


def _parse_poly(display: str) -> dict:
    """{degree: coefficient}, reimplemented locally against this module's own
    display convention (own test file -- unlike the independent oracle, it is
    fine for this to know the shape join_terms/_poly_display produce)."""
    padded = display if display[0] in "+-" else "+" + display
    coeffs: dict[int, int] = {}
    for m in _TERM_RE.finditer(padded):
        sign_s, mag_s, xpart, deg_s = m.groups()
        sign = 1 if sign_s == "+" else -1
        if xpart is None:
            degree, mag = 0, int(mag_s) if mag_s is not None else 0
        elif deg_s is not None:
            degree, mag = int(deg_s), int(mag_s) if mag_s is not None else 1
        else:
            degree, mag = 1, int(mag_s) if mag_s is not None else 1
        coeffs[degree] = coeffs.get(degree, 0) + sign * mag
    return coeffs


def _all_questions(topic, tier, seeds=SEEDS):
    return [make(topic, seed=s, difficulty=tier) for s in seeds]


TOPICS = ["polynomial_add_subtract", "polynomial_linear_combination"]


class TestBothTopicsShared:
    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, topic, tier):
        for seed in range(20):
            q1 = make(topic, seed=seed, difficulty=tier)
            q2 = make(topic, seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, topic, tier):
        for q in _all_questions(topic, tier):
            assert len(q.distractors) > 0, f"{topic} {tier}: no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{topic} {tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{topic} {tier}: duplicate distractors"

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_expanded_on_every_question(self, topic, tier):
        for q in _all_questions(topic, tier):
            assert q.answer_form == "expanded", f"{topic} {tier}: {q.text!r} missing answer_form=expanded"

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, topic, tier):
        for q in _all_questions(topic, tier):
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"{topic} {tier}: grader rejected canonical {q.answer_text!r} for {q.text!r}"

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, topic, tier):
        for q in _all_questions(topic, tier):
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"{topic} {tier}: grader accepted distractor {d!r} for {q.text!r}"

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, topic, tier):
        for q in _all_questions(topic, tier, seeds=range(10)):
            wrong = "999x^2 + 999x - 999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{topic} {tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("topic", TOPICS)
    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, topic, tier):
        texts = {q.text for q in _all_questions(topic, tier)}
        assert len(texts) >= 20, f"{topic} {tier}: only {len(texts)} distinct questions (need >= 20)"


class TestOwnFormRejectedProof:
    """Instead: the question's own bracketed construction copied straight back -- algebraically equal, but
    carrying a bracket, which answer_form="expanded" must reject."""

    def test_add_subtract_rejects_its_own_bracketed_construction(self):
        checked = 0
        for q in _all_questions("polynomial_add_subtract", "hard", seeds=PROOF_SEEDS):
            m = _ADD_SUB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            a, b, c = m.groups()
            checked += 1
            own_form = f"({a}) + ({b}) - ({c})"
            result = GRADERS["expression"](q, own_form)
            assert not result.correct, f"accepted own bracketed form {own_form!r} for {q.text!r}"
        assert checked >= 20, f"only {checked} questions sampled (need >= 20)"
        print(f"add-subtract-polynomials: own-form-rejected proof: {checked}/{checked} passed, 0 failed")

    def test_linear_combination_rejects_its_own_bracketed_construction(self):
        checked = 0
        for q in _all_questions("polynomial_linear_combination", "hard", seeds=PROOF_SEEDS):
            m = _LINCOMB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            p, qq, r, combo = m.groups()
            m1 = re.match(r"(\d+)?P", combo).group(1) or "1"
            m2 = re.search(r"([+-])\s*(\d+)?Q", combo)
            m3 = re.search(r"([+-])\s*(\d+)?R", combo)
            checked += 1
            own_form = f"{m1}({p}) + {(m2.group(2) or '1')}({qq}) - {(m3.group(2) or '1')}({r})"
            result = GRADERS["expression"](q, own_form)
            assert not result.correct, f"accepted own bracketed form {own_form!r} for {q.text!r}"
        assert checked >= 20, f"only {checked} questions sampled (need >= 20)"
        print(f"polynomial-linear-combination: own-form-rejected proof: {checked}/{checked} passed, 0 failed")


class TestHardTierAlignmentTrap:
    """The missing-power trap must fire on EVERY hard seed, asserted
    directly, not sampled: one of the three polynomials has one fewer
    displayed term than the other two, which share the same (maximal) term
    count."""

    def test_add_subtract_missing_power_every_hard_seed(self):
        for q in _all_questions("polynomial_add_subtract", "hard"):
            m = _ADD_SUB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            counts = sorted(len(_parse_poly(g)) for g in m.groups())
            assert counts[0] == counts[1] - 1 and counts[1] == counts[2], (
                f"missing-power trap did not fire (term counts {counts}) for {q.text!r}"
            )

    def test_linear_combination_missing_power_every_hard_seed(self):
        for q in _all_questions("polynomial_linear_combination", "hard"):
            m = _LINCOMB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            p, qq, r, _combo = m.groups()
            counts = sorted(len(_parse_poly(g)) for g in (p, qq, r))
            assert counts[0] == counts[1] - 1 and counts[1] == counts[2], (
                f"missing-power trap did not fire (term counts {counts}) for {q.text!r}"
            )

    def test_linear_combination_mixed_sign_r_every_hard_seed(self):
        for q in _all_questions("polynomial_linear_combination", "hard"):
            m = _LINCOMB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            _p, _qq, r, _combo = m.groups()
            coeffs = _parse_poly(r).values()
            assert any(c > 0 for c in coeffs) and any(c < 0 for c in coeffs), (
                f"R is not mixed-sign for hard-tier {q.text!r}"
            )


class TestMediumTierNegativeLeading:
    def test_add_subtract_negative_leading_every_medium_seed(self):
        for q in _all_questions("polynomial_add_subtract", "medium"):
            m = _ADD_SUB_RE.match(q.text)
            assert m, f"unrecognised question shape: {q.text!r}"
            leadings = []
            for g in m.groups():
                d = _parse_poly(g)
                leadings.append(d[max(d)])
            assert any(v < 0 for v in leadings), f"no negative leading coefficient among {leadings} in {q.text!r}"


class TestSkillsLoad:
    def test_add_subtract_polynomials_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "add-subtract-polynomials")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"combine-like-terms", "expand-single-brackets"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "polynomial_add_subtract"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}

    def test_polynomial_linear_combination_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "polynomial-linear-combination")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"combine-like-terms", "expand-single-brackets"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "polynomial_linear_combination"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
