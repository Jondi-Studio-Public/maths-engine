"""Covers:
- probability_complements_compounds (easy/medium/hard)
- random_equal_likelihood_outcomes (easy/medium/hard)
"""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.generators.p6d_probability_build import (
    probability_complements_compounds,
    random_equal_likelihood_outcomes,
    _format_fraction,
    _build_spinner,
    _is_balanced,
)
from mathgen.grading.typed import grade_fraction, grade_mcq, mcq_options


class TestProbabilityComplementsCompounds:
    """Tests for f10-probability-complements-compounds."""

    def test_determinism_easy(self):
        """Easy tier produces deterministic output."""
        q1 = make("probability_complements_compounds", seed=42, difficulty="easy")
        q2 = make("probability_complements_compounds", seed=42, difficulty="easy")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_determinism_medium(self):
        """Medium tier produces deterministic output."""
        q1 = make("probability_complements_compounds", seed=42, difficulty="medium")
        q2 = make("probability_complements_compounds", seed=42, difficulty="medium")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_determinism_hard(self):
        """Hard tier produces deterministic output."""
        q1 = make("probability_complements_compounds", seed=42, difficulty="hard")
        q2 = make("probability_complements_compounds", seed=42, difficulty="hard")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_easy_no_half_probability(self):
        """Easy tier never generates p=1/2 (required for forgot-to-complement distinctness)."""
        for seed in range(500):
            q = make("probability_complements_compounds", seed=seed, difficulty="easy")
            # Parse fraction from question text
            match = re.search(r"probability that .* is (\d+)/(\d+)", q.text)
            assert match, f"could not parse fraction in: {q.text}"
            n, d = int(match.group(1)), int(match.group(2))
            assert 2 * n != d, f"seed {seed}: found p=1/2 in easy tier: {n}/{d}"

    def test_medium_no_half_probability(self):
        """Medium tier never generates p=1/2 after reduction."""
        for seed in range(500):
            q = make("probability_complements_compounds", seed=seed, difficulty="medium")
            match = re.search(r"probability that .* is (\d+)/(\d+)", q.text)
            assert match, f"could not parse fraction in: {q.text}"
            n, d = int(match.group(1)), int(match.group(2))
            p_reduced = Fraction(n, d)
            assert p_reduced != Fraction(1, 2), f"seed {seed}: found p=1/2 in medium tier after reduction"

    def test_hard_no_half_probability(self):
        """Hard tier never generates p=1/2."""
        for seed in range(500):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")
            match = re.search(r"probability of .* is (\d+)/(\d+)", q.text)
            assert match, f"could not parse fraction in: {q.text}"
            n, d = int(match.group(1)), int(match.group(2))
            assert 2 * n != d, f"seed {seed}: found p=1/2 in hard tier: {n}/{d}"

    def test_hard_no_word_not(self):
        """Hard tier never contains the word 'not'."""
        for seed in range(100):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")
            assert re.search(r"\bnot\b", q.text) is None, f"seed {seed}: hard tier contains 'not': {q.text}"

    def test_distractors_distinct(self):
        """Every distractor is distinct from the answer."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("probability_complements_compounds", seed=seed, difficulty=preset)
                answer_frac = Fraction(q.answer["p"], q.answer["q"])
                for distractor in q.distractors:
                    d_frac = Fraction(distractor)
                    assert d_frac != answer_frac, (
                        f"seed {seed} {preset}: distractor {distractor} equals answer {q.answer_text}"
                    )

    def test_formatting_parity(self):
        """Answer and all distractors are formatted alike (n/d or n)."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("probability_complements_compounds", seed=seed, difficulty=preset)
                # Check answer format
                is_int = q.answer_text.isdigit() or (q.answer_text[0] == "-" and q.answer_text[1:].isdigit())
                if is_int:
                    pattern = r"^-?\d+$"
                else:
                    pattern = r"^\d+/\d+$"

                assert re.match(pattern, q.answer_text), (
                    f"seed {seed} {preset}: answer {q.answer_text} does not match expected format"
                )

                for distractor in q.distractors:
                    assert re.match(pattern, distractor), (
                        f"seed {seed} {preset}: distractor {distractor} does not match answer format"
                    )

    def test_misreads_numerator_differs_by_one_d(self):
        """Misreads-numerator distractor differs from complement by exactly 1/d."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("probability_complements_compounds", seed=seed, difficulty=preset)
                answer_frac = Fraction(q.answer["p"], q.answer["q"])

                # Parse p from question
                if preset == "hard":
                    pattern = r"probability of .* is (\d+)/(\d+)"
                else:
                    pattern = r"probability that .* is (\d+)/(\d+)"
                match = re.search(pattern, q.text)
                assert match, f"could not parse p in {preset}: {q.text}"
                n, d = int(match.group(1)), int(match.group(2))
                p = Fraction(n, d)
                d_reduced = Fraction(n, d).limit_denominator().denominator

                # The complement is 1 - p
                expected_c = 1 - p
                expected_c = Fraction(expected_c).limit_denominator()

                # Misreads value should be 1 - (n±1)/d, differing from c by 1/d_reduced
                # We can't easily extract which distractor is which, so we verify
                # that at least one distractor differs from c by exactly 1/d_reduced
                differences = set()
                for distractor in q.distractors:
                    d_frac = Fraction(distractor)
                    diff = abs(expected_c - d_frac)
                    if diff != 0:
                        differences.add(diff)

                # At least one should be 1/d_reduced
                expected_diff = Fraction(1, d_reduced)
                assert expected_diff in differences, (
                    f"seed {seed} {preset}: no distractor differs by 1/{d_reduced} from complement"
                )

    def test_real_grader_accepts_answer_easy(self):
        """Real grader accepts the canonical answer at easy tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="easy")
            result = grade_fraction(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text} for question: {q.text}"

    def test_real_grader_accepts_answer_medium(self):
        """Real grader accepts the canonical answer at medium tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="medium")
            result = grade_fraction(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text} for question: {q.text}"

    def test_real_grader_accepts_answer_hard(self):
        """Real grader accepts the canonical answer at hard tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")
            result = grade_fraction(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text} for question: {q.text}"

    def test_real_grader_rejects_all_distractors_easy(self):
        """Real grader rejects every distractor at easy tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="easy")
            for distractor in q.distractors:
                result = grade_fraction(q, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor {distractor}"

    def test_real_grader_rejects_all_distractors_medium(self):
        """Real grader rejects every distractor at medium tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="medium")
            for distractor in q.distractors:
                result = grade_fraction(q, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor {distractor}"

    def test_real_grader_rejects_all_distractors_hard(self):
        """Real grader rejects every distractor at hard tier."""
        for seed in range(40):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")
            for distractor in q.distractors:
                result = grade_fraction(q, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor {distractor}"

    def test_answer_is_fraction(self):
        """Answer is a valid Fraction."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("probability_complements_compounds", seed=seed, difficulty=preset)
                frac = Fraction(q.answer["p"], q.answer["q"])
                # Verify it's between 0 and 1
                assert 0 < frac < 1, f"seed {seed} {preset}: answer {frac} not in (0,1)"
                # Verify it's in lowest terms
                assert frac == frac.limit_denominator(), f"seed {seed} {preset}: answer {frac} not in lowest terms"

    def test_independent_check_corruption_caught(self):
        """Independent check catches corrupted answer."""
        # We'll verify this by checking the check_equivalent call works
        # by manually corrupting and checking
        from fractions import Fraction
        from mathgen.core import check_equivalent, VerificationError

        # Simulate a corrupted computation
        p = Fraction(1, 3)
        c_correct = 1 - p
        c_wrong = p  # corrupted answer

        # Correct answer should pass
        try:
            check_equivalent(c_correct + p, Fraction(1), "sum to one check")
        except VerificationError:
            pytest.fail("correct answer failed check_equivalent")

        # Corrupted answer should fail
        with pytest.raises(VerificationError):
            check_equivalent(c_wrong + p, Fraction(1), "sum to one check")

    def test_at_least_20_distinct_easy(self):
        """Easy tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("probability_complements_compounds", seed=seed, difficulty="easy")
            questions.add((q.text, q.answer_text, tuple(q.distractors)))
        assert len(questions) >= 20, f"only {len(questions)} distinct easy questions over 60 seeds"

    def test_at_least_20_distinct_medium(self):
        """Medium tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("probability_complements_compounds", seed=seed, difficulty="medium")
            questions.add((q.text, q.answer_text, tuple(q.distractors)))
        assert len(questions) >= 20, f"only {len(questions)} distinct medium questions over 60 seeds"

    def test_at_least_20_distinct_hard(self):
        """Hard tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")
            questions.add((q.text, q.answer_text, tuple(q.distractors)))
        assert len(questions) >= 20, f"only {len(questions)} distinct hard questions over 60 seeds"

    def test_easy_medium_negation_is_grammatical(self):
        """Easy and medium negations are grammatical, not mechanical 'does not happen'.

        Regression test for ungrammatical negation defect:
        - Question must never contain the literal substring "does not happen"
        - Question must always contain "does not " or "is not " immediately following
          one of the event names' subject nouns
        """
        for tier in ["easy", "medium"]:
            for seed in range(60):
                q = make("probability_complements_compounds", seed=seed, difficulty=tier)

                # Check "does not happen" never appears
                assert "does not happen" not in q.text, (
                    f"seed {seed} {tier}: question contains ungrammatical 'does not happen': {q.text}"
                )

                # Check that "does not " or "is not " appears in the question text
                has_grammatical_negation = "does not " in q.text or "is not " in q.text
                assert has_grammatical_negation, (
                    f"seed {seed} {tier}: question lacks grammatical negation ('does not ' or 'is not '): {q.text}"
                )

    def test_easy_medium_events_have_no_fixed_probability_object_and_use_explicit_negation(self):
        r"""Easy and medium tier events have no fixed-probability objects and use explicit negation.

        Regression test for contrived-probability and difficulty-progression defects:
        - No question should mention coin/card/die (fair coin = always 1/2, standard deck = always 1/2, standard die = always 1/2)
        - Every question must use explicit negation (e.g. "does not happen", "does not \w+")
        """
        for tier in ["easy", "medium"]:
            for seed in range(60):
                q = make("probability_complements_compounds", seed=seed, difficulty=tier)

                # Check no fixed-probability objects
                assert "coin" not in q.text, f"seed {seed} {tier}: question mentions coin (fixed P=1/2): {q.text}"
                assert "card" not in q.text, f"seed {seed} {tier}: question mentions card (fixed P=1/2): {q.text}"
                assert "die" not in q.text, f"seed {seed} {tier}: question mentions die (fixed P=1/2): {q.text}"

                # Check explicit negation: "does not <verb>" or "is not <adjective>"
                has_explicit_negation = (
                    re.search(r"does not \w+", q.text) is not None or re.search(r"is not \w+", q.text) is not None
                )
                assert has_explicit_negation, f"seed {seed} {tier}: question lacks explicit negation: {q.text}"

    def test_hard_events_have_no_fixed_probability_object(self):
        """Hard tier outcome pairs have no fixed-probability objects (coin/card/die).

        Regression test for contrived-probability defect:
        - No question should mention coin/card/die
        """
        for seed in range(60):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")

            assert "coin" not in q.text, f"seed {seed} hard: question mentions coin (fixed P=1/2): {q.text}"
            assert "card" not in q.text, f"seed {seed} hard: question mentions card (fixed P=1/2): {q.text}"
            assert "die" not in q.text, f"seed {seed} hard: question mentions die (fixed P=1/2): {q.text}"

    def test_hard_tier_outcome_words_are_grammatical_nouns_or_gerunds(self):
        """Hard tier outcome words must be grammatical nouns or gerunds in both template slots.

        Regression test for verb/noun mismatch defect:
        - Extract the word(s) immediately following "The probability of "
        - Each extracted outcome must be a member of the whitelist of grammatical words
        - This ensures future additions to outcome_pairs are grammatically sound
        """
        # Whitelist of outcome words that are grammatical in both:
        # (1) "exactly one of two ways: {outcome} or {outcome}" listing
        # (2) "The probability of {outcome} is ..." slot
        whitelist = {
            "winning",
            "losing",
            "passing",
            "failing",
            "rain",
            "no rain",
            "success",
            "failure",
            "approval",
            "rejection",
        }

        for seed in range(60):
            q = make("probability_complements_compounds", seed=seed, difficulty="hard")

            # Extract both outcomes using regex with non-greedy matching:
            # First: "The probability of X is" (outcome_1)
            # Second: "probability of Y?" (outcome_2)
            pattern1 = r"The probability of (.+?) is"
            match1 = re.search(pattern1, q.text)
            assert match1, f"seed {seed}: could not parse first outcome in: {q.text}"
            outcome_1_extracted = match1.group(1).strip()

            # Find second "probability of" by searching after the first match
            search_start = match1.end()
            second_part = q.text[search_start:]
            pattern2 = r"probability of (.+?)\?"
            match2 = re.search(pattern2, second_part)
            assert match2, f"seed {seed}: could not parse second outcome in: {q.text}"
            outcome_2_extracted = match2.group(1).strip()

            # Both extracted outcomes must be in the whitelist
            assert outcome_1_extracted in whitelist, (
                f"seed {seed}: outcome '{outcome_1_extracted}' not in whitelist: {q.text}"
            )
            assert outcome_2_extracted in whitelist, (
                f"seed {seed}: outcome '{outcome_2_extracted}' not in whitelist: {q.text}"
            )


class TestRandomEqualLikelihoodOutcomes:
    """Tests for f10-random-equal-likelihood-outcomes."""

    def test_determinism_easy(self):
        """Easy tier produces deterministic output."""
        q1 = make("random_equal_likelihood_outcomes", seed=42, difficulty="easy")
        q2 = make("random_equal_likelihood_outcomes", seed=42, difficulty="easy")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_determinism_medium(self):
        """Medium tier produces deterministic output."""
        q1 = make("random_equal_likelihood_outcomes", seed=42, difficulty="medium")
        q2 = make("random_equal_likelihood_outcomes", seed=42, difficulty="medium")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_determinism_hard(self):
        """Hard tier produces deterministic output."""
        q1 = make("random_equal_likelihood_outcomes", seed=42, difficulty="hard")
        q2 = make("random_equal_likelihood_outcomes", seed=42, difficulty="hard")
        assert q1.text == q2.text
        assert q1.answer_text == q2.answer_text
        assert q1.distractors == q2.distractors

    def test_has_four_options(self):
        """Question always has four spinner options."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)
                option_count = q.text.count("Spinner")
                assert option_count == 4, f"seed {seed} {preset}: expected 4 spinners, got {option_count} in: {q.text}"

    def test_exactly_one_balanced_option(self):
        """Exactly one spinner has equal-sized sectors."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)

                # Parse each option (format: "Spinner A: ... | Spinner B: ... | ...")
                options = re.findall(r"Spinner (\w): ([^|]+)", q.text)
                assert len(options) == 4, f"seed {seed} {preset}: could not parse 4 options in: {q.text}"

                balanced_count = 0
                for letter, labels_text in options:
                    labels = [label.strip() for label in labels_text.split(",")]
                    counter = Counter(labels)
                    is_balanced = len(set(counter.values())) == 1
                    if is_balanced:
                        balanced_count += 1

                assert balanced_count == 1, f"seed {seed} {preset}: expected 1 balanced option, got {balanced_count}"

    def test_answer_is_balanced_option(self):
        """The answer text is the balanced option."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)

                # Parse answer from answer_text
                match = re.search(r"Spinner (\w): (.+)$", q.answer_text, re.MULTILINE)
                assert match, f"seed {seed} {preset}: could not parse answer"
                labels_text = match.group(2)
                labels = [label.strip() for label in labels_text.split(",")]
                counter = Counter(labels)
                is_balanced = len(set(counter.values())) == 1
                assert is_balanced, f"seed {seed} {preset}: answer option is not balanced: {q.answer_text}"

    def test_distractors_are_unbalanced(self):
        """All distractors are unbalanced options."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)

                for distractor in q.distractors:
                    match = re.search(r"Spinner (\w): (.+)$", distractor, re.MULTILINE)
                    assert match, f"seed {seed} {preset}: could not parse distractor"
                    labels_text = match.group(2)
                    labels = [label.strip() for label in labels_text.split(",")]
                    counter = Counter(labels)
                    is_balanced = len(set(counter.values())) == 1
                    assert not is_balanced, f"seed {seed} {preset}: distractor is balanced: {distractor}"

    def test_real_grader_accepts_answer(self):
        """Real grader accepts the canonical answer."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)
                options = mcq_options(q)
                answer_index = options.index(q.answer_text)
                result = grade_mcq(q, str(answer_index))
                assert result.correct, f"seed {seed} {preset}: grader rejected answer index {answer_index}"

    def test_real_grader_rejects_all_distractors(self):
        """Real grader rejects all distractors."""
        for seed in range(40):
            for preset in ["easy", "medium", "hard"]:
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=preset)
                options = mcq_options(q)
                for i, option in enumerate(options):
                    if option != q.answer_text:
                        result = grade_mcq(q, str(i))
                        assert not result.correct, f"seed {seed} {preset}: grader accepted distractor at index {i}"

    def test_medium_has_close_but_unequal_option(self):
        """Medium tier has one distractor with close-but-unequal counts (max - min == 1)."""
        found_close_unequal = False
        for seed in range(100):
            q = make("random_equal_likelihood_outcomes", seed=seed, difficulty="medium")

            options = re.findall(r"Spinner (\w): ([^|]+)", q.text)
            for letter, labels_text in options:
                labels = [label.strip() for label in labels_text.split(",")]
                counter = Counter(labels)
                if len(set(counter.values())) > 1:  # unbalanced
                    counts = list(counter.values())
                    if max(counts) - min(counts) == 1:
                        found_close_unequal = True
                        break
            if found_close_unequal:
                break

        assert found_close_unequal, "medium tier should produce at least one option with max-min==1 over 100 seeds"

    def test_at_least_20_distinct_easy(self):
        """Easy tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("random_equal_likelihood_outcomes", seed=seed, difficulty="easy")
            questions.add((q.text, q.answer_text))
        assert len(questions) >= 20, f"only {len(questions)} distinct easy questions"

    def test_at_least_20_distinct_medium(self):
        """Medium tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("random_equal_likelihood_outcomes", seed=seed, difficulty="medium")
            questions.add((q.text, q.answer_text))
        assert len(questions) >= 20, f"only {len(questions)} distinct medium questions"

    def test_at_least_20_distinct_hard(self):
        """Hard tier produces at least 20 distinct questions over 60 seeds."""
        questions = set()
        for seed in range(60):
            q = make("random_equal_likelihood_outcomes", seed=seed, difficulty="hard")
            questions.add((q.text, q.answer_text))
        assert len(questions) >= 20, f"only {len(questions)} distinct hard questions"
