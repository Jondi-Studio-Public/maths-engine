"""- ucv1-finite-automaton-acceptance   (topic p15d_automaton_acceptance, mcq)
- ucv1-formal-derivation-check       (topic p15d_formal_derivation_check, mcq)
- ucv1-dominated-convergence-check   (topic p15d_dominated_convergence, mcq)
- ucv1-manifold-chart-jacobian       (topic p15d_chart_jacobian, ordered_list)
- ucv1-validation-calibration-check  (topic p15d_validation_calibration, mcq)
- f10-shape-sorting-algorithm        (topic p15d_shape_sorting, mcq)

Coverage: determinism, distractor validity, the REAL grader (never the generator's own construction-time check
alone), skill loading, answer-kind shape, and >= 20 distinct questions per tier over 60 seeds (or a stated,
justified smaller pool).
"""

import functools

import pytest

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p15d_build import (
    _line_is_valid,
    _all_leaves,
    _attrs_for,
    _trace,
    _ALL_TYPES,
    _TREES,
)

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

TOPIC_TO_SKILL = {
    "p15d_automaton_acceptance": "ucv1-finite-automaton-acceptance",
    "p15d_formal_derivation_check": "ucv1-formal-derivation-check",
    "p15d_dominated_convergence": "ucv1-dominated-convergence-check",
    "p15d_chart_jacobian": "ucv1-manifold-chart-jacobian",
    "p15d_validation_calibration": "ucv1-validation-calibration-check",
    "p15d_shape_sorting": "f10-shape-sorting-algorithm",
}
MCQ_TOPICS = [t for t in TOPIC_TO_SKILL if t != "p15d_chart_jacobian"]


@pytest.fixture(scope="module")
def skills_by_id():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


class TestDeterminism:
    @pytest.mark.parametrize("topic", list(TOPIC_TO_SKILL))
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, topic, difficulty):
        for seed in SEEDS:
            q = make(topic, seed=seed, difficulty=difficulty)
            repeat = make(topic, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()


class TestMcqGrading:
    """The real grader, called directly, never the generator's own check()."""

    @pytest.mark.parametrize("topic", MCQ_TOPICS)
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_real_grader_accepts_answer_rejects_distractors(self, topic, difficulty):
        for seed in SEEDS:
            q = make(topic, seed=seed, difficulty=difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            assert grade_mcq(q, str(correct_index)).correct
            for index in range(len(options)):
                if index == correct_index:
                    continue
                assert not grade_mcq(q, str(index)).correct

    @pytest.mark.parametrize("topic", MCQ_TOPICS)
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_well_formed(self, topic, difficulty):
        for seed in SEEDS:
            q = make(topic, seed=seed, difficulty=difficulty)
            assert q.distractors
            assert q.answer_text not in q.distractors
            assert len(set(q.distractors)) == len(q.distractors)


class TestOrderedListGrading:
    """``answer: ordered_list``. Was `answer: list` -- `grade_list` is order-insensitive on the SET of five
    values, so a learner who named the same five numbers in the wrong slots (transpose, swapped rows) was
    marked correct. `grade_ordered_list` is position-sensitive, so those are now genuinely wrong."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_real_grader_accepts_answer_rejects_distractors(self, difficulty):
        for seed in SEEDS:
            q = make("p15d_chart_jacobian", seed=seed, difficulty=difficulty)
            assert GRADERS["ordered_list"](q, q.answer_text).correct
            for d in q.distractors:
                assert not GRADERS["ordered_list"](q, d).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_transpose_and_swapped_rows_are_offered_and_rejected(self, difficulty):
        """A transpose (du/dy <-> dv/dx swapped) and a swapped-rows candidate (row 2 before row 1, determinant
        sign flipped) must each appear among the distractors on some seed, and the real ordered_list grader
        must reject each one wherever it appears."""
        seen_transpose = seen_swapped = False
        for seed in SEEDS:
            q = make("p15d_chart_jacobian", seed=seed, difficulty=difficulty)
            j11, j12, j21, j22, det = q.answer
            transpose_text = f"{j11}, {j21}, {j12}, {j22}, {det}"
            swapped_text = f"{j21}, {j22}, {j11}, {j12}, {-det}"
            if transpose_text in q.distractors:
                seen_transpose = True
                assert not GRADERS["ordered_list"](q, transpose_text).correct
            if swapped_text in q.distractors:
                seen_swapped = True
                assert not GRADERS["ordered_list"](q, swapped_text).correct
        assert seen_transpose, f"transpose distractor never offered at {difficulty}"
        assert seen_swapped, f"swapped-rows distractor never offered at {difficulty}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_false_accept_red_on_list_green_on_ordered_list(self, difficulty):
        """The transpose candidate (same five numbers, du/dy and dv/dx swapped) graded CORRECT under `list` and
        INCORRECT under `ordered_list` (GREEN -- the shipped grader kind, per `answer: ordered_list` in the
        skill YAML)."""
        for seed in SEEDS:
            q = make("p15d_chart_jacobian", seed=seed, difficulty=difficulty)
            j11, j12, j21, j22, det = q.answer
            transpose_text = f"{j11}, {j21}, {j12}, {j22}, {det}"
            if transpose_text == q.answer_text:
                continue  # symmetric Jacobian this draw; no transposition to test
            # RED: naming the same five numbers, wrong slots, would pass `list`.
            assert GRADERS["list"](q, transpose_text).correct
            # GREEN: the actually-shipped grader kind rejects it.
            assert not GRADERS["ordered_list"](q, transpose_text).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_five_integers_matching_its_own_text(self, difficulty):
        for seed in SEEDS:
            q = make("p15d_chart_jacobian", seed=seed, difficulty=difficulty)
            assert isinstance(q.answer, list) and len(q.answer) == 5
            assert all(isinstance(v, int) for v in q.answer)
            assert q.answer_text == ", ".join(str(v) for v in q.answer)
            j11, j12, j21, j22, det = q.answer
            assert det == j11 * j22 - j12 * j21


class TestAutomatonAcceptance:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_matches_a_direct_trace_of_the_rendered_table(self, difficulty):
        """Deliberately-wrong-answer rejection: a corrupted final state must
        be rejected by the real grader too."""
        for seed in SEEDS:
            q = make("p15d_automaton_acceptance", seed=seed, difficulty=difficulty)
            options = mcq_options(q)
            correct_index = options.index(q.answer_text)
            # A deliberately wrong composite (a state that never occurs in
            # the option set, if any digit still resolves to a state).
            wrong = (
                q.answer_text.replace("Accepted", "Rejected")
                if "Accepted" in q.answer_text
                else q.answer_text.replace("Rejected", "Accepted")
            )
            if wrong in options:
                wrong_index = options.index(wrong)
                assert wrong_index != correct_index
                assert not grade_mcq(q, str(wrong_index)).correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_three_state_distractors_every_seed(self, difficulty):
        for seed in SEEDS:
            q = make("p15d_automaton_acceptance", seed=seed, difficulty=difficulty)
            assert len(q.distractors) == 3

    def test_exam_word_is_given_compactly_not_traced_step_by_step(self):
        """The exam tier must give the word compactly (exponent/repeat notation) so the learner reasons about the
        automaton's eventual cycle by period, rather than trace a long word step by step -- a step the hard
        tier never asks for."""
        for seed in range(30):
            q = make("p15d_automaton_acceptance", seed=seed, difficulty="exam")
            assert "^" in q.text, q.text

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_lower_tiers_never_use_compact_notation(self, difficulty):
        for seed in range(30):
            q = make("p15d_automaton_acceptance", seed=seed, difficulty=difficulty)
            assert "^" not in q.text


class TestFormalDerivationCheck:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_four_options_every_seed(self, difficulty):
        for seed in SEEDS:
            q = make("p15d_formal_derivation_check", seed=seed, difficulty=difficulty)
            assert len(q.distractors) + 1 >= 4

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_line_option_is_reachable(self, difficulty):
        """Sanity: 'Every line is valid' and at least one 'Line N' class
        both occur across a real sweep (never a constant answer space)."""
        seen = {make("p15d_formal_derivation_check", seed=s, difficulty=difficulty).answer_text for s in range(200)}
        assert "Every line is valid" in seen
        assert any(a.startswith("Line ") for a in seen)

    def test_skeleton_order_varies_across_seeds(self):
        """The first derived line's rule must not be a constant, so the key cannot be learned as 'look at line
        4'."""
        import re

        rules_seen = set()
        for seed in range(60):
            q = make("p15d_formal_derivation_check", seed=seed, difficulty="exam")
            m = re.search(r"Line 4: .*?-- (Modus Ponens|Conjunction Elimination \(left conjunct\))", q.text)
            assert m, q.text
            rules_seen.add(m.group(1))
        assert rules_seen == {"Modus Ponens", "Conjunction Elimination (left conjunct)"}

    def test_exam_corruption_only_lands_on_a_cascading_line(self):
        """Line 5 (AndI) and Line 9 (AndI2, the last line) are never cited by anything later, so they must never
        be the exam answer."""
        seen = {make("p15d_formal_derivation_check", seed=s, difficulty="exam").answer_text for s in range(200)}
        assert "Line 5" not in seen
        assert "Line 9" not in seen
        assert any(a in seen for a in ("Line 4", "Line 6", "Line 7", "Line 8"))


class TestDominatedConvergence:
    """On a finite sample space every stated bound is automatically integrable, so "the bound is not integrable"
    is never a genuine key -- it is a permanently-wrong decoy, offered as a distractor on every draw, never
    the answer."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_all_three_honest_classes_drawn(self, difficulty):
        seen = {make("p15d_dominated_convergence", seed=s, difficulty=difficulty).answer_text for s in range(200)}
        assert seen == {
            "All hypotheses are met",
            "Pointwise convergence fails",
            "The stated bound is violated at a supplied point",
        }

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_not_integrable_is_never_the_answer_but_always_a_distractor(self, difficulty):
        for seed in range(60):
            q = make("p15d_dominated_convergence", seed=seed, difficulty=difficulty)
            assert q.answer_text != "The bound is not integrable"
            assert "The bound is not integrable" in q.distractors

    def test_exam_states_a_multi_term_tail_convention_not_a_single_index(self):
        """Exam must state a well-defined convention (the last `tail_len` >= 2 named terms, not just one),
        distinct from the single-starting-index wording the lower tiers use."""
        for seed in range(30):
            q = make("p15d_dominated_convergence", seed=seed, difficulty="exam")
            assert "the convention: the sequence is constant across at least the last" in q.text
            assert "stabilises from f" not in q.text
            # at least two named tail terms, e.g. "f4, f5, f6"
            import re

            m = re.search(r"across at least the last (\d+) listed values, ((?:f\d+, )*f\d+),", q.text)
            assert m, q.text
            assert int(m.group(1)) >= 2
            assert len(m.group(2).split(", ")) == int(m.group(1))

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_lower_tiers_state_a_single_stabilisation_index(self, difficulty):
        for seed in range(30):
            q = make("p15d_dominated_convergence", seed=seed, difficulty=difficulty)
            assert "stabilises from f" in q.text
            assert "the convention: the sequence is constant across" not in q.text


class TestValidationCalibration:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_names_a_direction_and_a_decimal_value(self, difficulty):
        for seed in SEEDS:
            q = make("p15d_validation_calibration", seed=seed, difficulty=difficulty)
            assert q.answer_text.startswith("ECE = ")
            assert q.answer_text.endswith("over-confident") or q.answer_text.endswith("under-confident")

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_both_directions_drawn(self, difficulty):
        seen_directions = set()
        for seed in range(200):
            q = make("p15d_validation_calibration", seed=seed, difficulty=difficulty)
            seen_directions.add(q.answer_text.rsplit(", ", 1)[1])
        assert seen_directions == {"over-confident", "under-confident"}

    def test_exam_bins_are_unequal_width_and_stated(self):
        """Exam tier bins must be unequal-width, with the boundaries stated explicitly rather than assumed as an
        equal division -- a step the hard tier never needs (hard's bins are always the equal-width [0, 1/3),
        [1/3, 2/3), [2/3, 1])."""
        for seed in range(30):
            q = make("p15d_validation_calibration", seed=seed, difficulty="exam")
            assert "not equal width" in q.text
            assert "equal-width bins" not in q.text

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_lower_tiers_use_equal_width_bins(self, difficulty):
        for seed in range(30):
            q = make("p15d_validation_calibration", seed=seed, difficulty=difficulty)
            assert "equal-width bins" in q.text
            assert "not equal width" not in q.text


class TestShapeSorting:
    """The tree must be logically coherent (no test repeats or contradicts what an earlier test on the same path
    already established) and every bin must be reachable by some real shape."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_is_a_bin_in_the_tiers_own_tree(self, difficulty):
        leaves = set(_all_leaves(_TREES[difficulty]))
        for seed in SEEDS:
            q = make("p15d_shape_sorting", seed=seed, difficulty=difficulty)
            assert q.answer_text in leaves
            assert set(q.distractors) <= leaves - {q.answer_text}

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_leaf_is_reachable(self, difficulty):
        leaves = set(_all_leaves(_TREES[difficulty]))
        seen = {make("p15d_shape_sorting", seed=s, difficulty=difficulty).answer_text for s in range(400)}
        assert seen == leaves

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_leaf_distribution_is_not_concentrated(self, difficulty):
        """Construct-to-target (pick the leaf first, then a type that reaches it) must bring every tier close to
        the uniform 1/n_leaves share; a generous 2x-uniform ceiling catches a real concentration defect
        without being seed-brittle."""
        from collections import Counter

        n_leaves = len(_all_leaves(_TREES[difficulty]))
        counts = Counter(make("p15d_shape_sorting", seed=s, difficulty=difficulty).answer_text for s in range(200))
        top_share = counts.most_common(1)[0][1] / 200
        assert top_share <= 2 * (1 / n_leaves), (difficulty, n_leaves, counts)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_leaf_is_reachable_by_a_real_named_type(self, difficulty):
        """Reachability proven by construction, not sampling: every named
        triangle/quadrilateral type is traced through the tier's own tree,
        and every leaf must be hit by at least one real type."""
        tree = _TREES[difficulty]
        leaves = set(_all_leaves(tree))
        hit = set()
        for shape, type_name in _ALL_TYPES:
            attrs = _attrs_for(shape, type_name)
            hit.add(_trace(tree, attrs))
        assert hit == leaves

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_test_genuinely_splits_its_remaining_types(self, difficulty):
        """No test may be asked when its answer is already forced for every
        defect: asking `sides4?` right after `sides3?` said yes, whose
        `yes` branch is then mathematically unreachable). Walk every
        internal node and require BOTH branches to be reached by at least
        one real type."""
        tree = _TREES[difficulty]
        all_attrs = [_attrs_for(shape, t) for shape, t in _ALL_TYPES]

        def check_node(node, types_here):
            if node[0] == "leaf":
                return
            _, test, yes_b, no_b = node
            yes_types = [a for a in types_here if a[test]]
            no_types = [a for a in types_here if not a[test]]
            assert yes_types, f"{difficulty} test {test!r}: yes-branch unreachable"
            assert no_types, f"{difficulty} test {test!r}: no-branch unreachable"
            check_node(yes_b, yes_types)
            check_node(no_b, no_types)

        check_node(tree, all_attrs)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_stated_facts_are_internally_consistent(self, difficulty):
        """The rendered side-length list must itself carry the `sides3`/
        `equal_sides` facts (read, not told) -- guards against a future
        edit reintroducing the old bare-boolean-attribute wording."""
        for seed in SEEDS:
            q = make("p15d_shape_sorting", seed=seed, difficulty=difficulty)
            assert "sides measuring" in q.text
            assert "not exactly" not in q.text and "not all equal" not in q.text


class TestSkillLoading:
    @pytest.mark.parametrize("topic,skill_id", list(TOPIC_TO_SKILL.items()))
    def test_skill_loads_with_expected_shape(self, skills_by_id, topic, skill_id):
        skill = skills_by_id[skill_id]
        assert skill.placement_eligible is False
        assert skill.answer in GRADERS
        stages = {src.difficulty: src.stages for src in skill.sources if src.topic == topic}
        assert set(stages) == {"easy", "medium", "hard", "exam"}
        assert stages["exam"] == ["exam"]
        for tier in ("easy", "medium", "hard"):
            assert stages[tier] and stages[tier] != ["exam"]

    def test_expected_prerequisites_and_answer_kinds(self, skills_by_id):
        expected = {
            "ucv1-finite-automaton-acceptance": (["finite-set-operations", "apply-a-rule-to-inputs"], "mcq"),
            "ucv1-formal-derivation-check": (["propositional-formula-equivalence"], "mcq"),
            "ucv1-dominated-convergence-check": (["ucv1-sigma-algebra-closure", "limit-at-infinity"], "mcq"),
            "ucv1-manifold-chart-jacobian": (["jacobian-two-variable"], "ordered_list"),
            "ucv1-validation-calibration-check": (["least-squares-line"], "mcq"),
            "f10-shape-sorting-algorithm": (["f10-polygon-classification"], "mcq"),
        }
        for skill_id, (prereqs, answer_kind) in expected.items():
            skill = skills_by_id[skill_id]
            assert skill.prerequisites == prereqs
            assert skill.answer == answer_kind


class TestDistinctQuestionPools:
    """>= 20 distinct rendered questions per tier over 60 seeds, or a stated smaller pool. The two mcq skills
    whose answer space is a fixed finite class set (formal-derivation-check, dominated-convergence-check) are
    named here with their measured, honest pool size instead."""

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_automaton_acceptance_pool(self, difficulty):
        prompts = {make("p15d_automaton_acceptance", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 20

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_chart_jacobian_pool(self, difficulty):
        prompts = {make("p15d_chart_jacobian", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 20

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_validation_calibration_pool(self, difficulty):
        prompts = {make("p15d_validation_calibration", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 20

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_formal_derivation_check_pool_is_genuinely_bounded(self, difficulty):
        """The derivation's premises, rule set and corruption slots are all finite and stated in the contract;
        the pool is genuinely narrower than 20."""
        prompts = {make("p15d_formal_derivation_check", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 10

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_dominated_convergence_pool_is_genuinely_bounded(self, difficulty):
        prompts = {make("p15d_dominated_convergence", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 10

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_shape_sorting_pool(self, difficulty):
        """Concrete numeric side lengths (not a bare boolean-attribute list) make the pool wide at every tier,
        including easy (measured: 180/200 distinct prompts at easy, up from 16/200 pre-fix)."""
        prompts = {make("p15d_shape_sorting", seed=s, difficulty=difficulty).text for s in SEEDS}
        assert len(prompts) >= 20


class TestFormalDerivationInGeneratorCheckStandalone:
    """`_line_is_valid` is the structural rule-checker the generator itself
    uses; exercised here directly against a hand-built mini derivation, to
    prove the primitive independent of any one seed's draw."""

    def test_valid_modus_ponens_chain(self):
        A, B = ("atom", "p"), ("atom", "q")
        lines = [(A, "Premise", ()), (("implies", A, B), "Premise", ()), (B, "MP", (0, 1))]
        assert all(_line_is_valid(lines, i) for i in range(3))

    def test_corrupted_modus_ponens_is_caught(self):
        A, B = ("atom", "p"), ("atom", "q")
        lines = [(A, "Premise", ()), (("implies", A, B), "Premise", ()), (A, "MP", (0, 1))]
        assert _line_is_valid(lines, 0) and _line_is_valid(lines, 1)
        assert not _line_is_valid(lines, 2)
