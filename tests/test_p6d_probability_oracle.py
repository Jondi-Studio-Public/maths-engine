"""Independent oracle for `probability_complements_compounds` and `random_equal_likelihood_outcomes`.

Probability oracle."""

import re
from collections import Counter
from fractions import Fraction

from mathgen import make


class TestProbabilityComplementsCompounds:
    """Computational oracle for probability_complements_compounds."""

    def test_complements_compounds_oracle(self):
        """Verify complement computation from question text alone."""
        challenged = []
        examined = 0

        for tier in ["easy", "medium", "hard"]:
            for seed in range(40):
                q = make("probability_complements_compounds", seed=seed, difficulty=tier)
                examined += 1

                # Extract every fraction n/d from the text
                fractions = re.findall(r"(\d+)/(\d+)", q.text)

                if len(fractions) != 1:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Expected exactly 1 fraction, found {len(fractions)}",
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                        }
                    )
                    continue

                # Independently compute the complement
                n, d = map(int, fractions[0])
                computed_complement = 1 - Fraction(n, d)

                # Parse answer_text as a fraction
                try:
                    answer_fraction = Fraction(q.answer_text)
                except (ValueError, ZeroDivisionError) as e:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Cannot parse answer_text as fraction: {e}",
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                            "computed": str(computed_complement),
                        }
                    )
                    continue

                # Verify answer matches computed complement
                if answer_fraction != computed_complement:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Answer {answer_fraction} != computed complement {computed_complement}",
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                            "computed": str(computed_complement),
                        }
                    )
                    continue

                # Verify all distractors differ from the correct value
                for i, distractor in enumerate(q.distractors):
                    try:
                        distractor_fraction = Fraction(distractor)
                    except (ValueError, ZeroDivisionError) as e:
                        challenged.append(
                            {
                                "seed": seed,
                                "tier": tier,
                                "issue": f"Cannot parse distractor {i} as fraction: {e}",
                                "text": q.text,
                                "answer_text": q.answer_text,
                                "distractors": q.distractors,
                                "distractor_index": i,
                            }
                        )
                        continue

                    if distractor_fraction == computed_complement:
                        challenged.append(
                            {
                                "seed": seed,
                                "tier": tier,
                                "issue": f"Distractor {i} equals correct answer: {distractor}",
                                "text": q.text,
                                "answer_text": q.answer_text,
                                "distractors": q.distractors,
                                "distractor_index": i,
                            }
                        )

        # Report findings
        assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
        print(f"probability_complements_compounds: examined {examined}, challenged 0")


class TestRandomEqualLikelihoodOutcomes:
    """Enumeration oracle for random_equal_likelihood_outcomes."""

    @staticmethod
    def is_balanced(labels_str):
        """Check if a spinner option's labels are balanced.

        Given comma-separated labels (without "Spinner X: " prefix),
        return True iff each distinct label appears the same number of times.
        """
        labels = [label.strip() for label in labels_str.split(", ")]
        if not labels:
            return False

        counter = Counter(labels)
        # Balanced iff all counts are equal
        return len(set(counter.values())) == 1

    def test_random_equal_likelihood_outcomes_oracle(self):
        """Verify exactly one spinner is balanced, derived from text alone."""
        challenged = []
        inconsistent = []
        examined = 0

        for tier in ["easy", "medium", "hard"]:
            for seed in range(150):
                q = make("random_equal_likelihood_outcomes", seed=seed, difficulty=tier)
                examined += 1

                # Extract all 4 spinner options from q.text
                # Format: "Spinner A: label1, label2, ... | Spinner B: ... | ..."
                # Split on " | " to get individual spinner strings
                spinner_pattern = r"Spinner\s+[A-D]:\s+[^|]+"
                matches = re.findall(spinner_pattern, q.text)

                if len(matches) != 4:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Expected exactly 4 spinners, found {len(matches)}",
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                        }
                    )
                    continue

                # Map full option text (with prefix) for each spinner
                options_by_letter = {}
                for option_str in matches:
                    option_str = option_str.strip()
                    # Extract letter and labels from "Spinner X: label1, label2, ..."
                    m = re.match(r"Spinner\s+([A-D]):\s+(.+)", option_str)
                    if not m:
                        challenged.append(
                            {
                                "seed": seed,
                                "tier": tier,
                                "issue": f"Cannot parse spinner option: {option_str}",
                                "text": q.text,
                                "answer_text": q.answer_text,
                                "distractors": q.distractors,
                            }
                        )
                        continue
                    letter, labels_str = m.groups()
                    options_by_letter[letter] = {
                        "full_text": option_str,
                        "labels_str": labels_str.strip(),
                    }

                # Check balance for each option
                balanced_letters = []
                for letter, option_data in options_by_letter.items():
                    if self.is_balanced(option_data["labels_str"]):
                        balanced_letters.append(letter)

                    # Check for internal inconsistency
                    labels = [label.strip() for label in option_data["labels_str"].split(", ")]
                    if not labels or len(labels) == 0:
                        inconsistent.append(
                            {
                                "seed": seed,
                                "tier": tier,
                                "letter": letter,
                                "issue": "empty label list",
                            }
                        )

                    counter = Counter(labels)
                    if any(count == 0 for count in counter.values()):
                        inconsistent.append(
                            {
                                "seed": seed,
                                "tier": tier,
                                "letter": letter,
                                "issue": "zero count in labels",
                            }
                        )

                # Assert exactly one is balanced
                if len(balanced_letters) != 1:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Expected exactly 1 balanced option, found {len(balanced_letters)}: {balanced_letters}",
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                        }
                    )
                    continue

                balanced_letter = balanced_letters[0]
                balanced_option_full_text = options_by_letter[balanced_letter]["full_text"]

                # Assert balanced option's full text equals answer_text
                if balanced_option_full_text != q.answer_text:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": f"Balanced option text doesn't match answer_text",
                            "balanced_option": balanced_option_full_text,
                            "answer_text": q.answer_text,
                            "text": q.text,
                            "distractors": q.distractors,
                        }
                    )
                    continue

                # Assert other three options are exactly the distractors
                distractor_texts = {
                    options_by_letter[letter]["full_text"] for letter in options_by_letter if letter != balanced_letter
                }
                expected_distractors = set(q.distractors)

                if distractor_texts != expected_distractors:
                    challenged.append(
                        {
                            "seed": seed,
                            "tier": tier,
                            "issue": "Distractor texts don't match q.distractors",
                            "distractor_texts": sorted(distractor_texts),
                            "expected_distractors": sorted(expected_distractors),
                            "text": q.text,
                            "answer_text": q.answer_text,
                            "distractors": q.distractors,
                        }
                    )

        # Report findings
        assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
        print(f"random_equal_likelihood_outcomes: examined {examined}, challenged 0")
        if inconsistent:
            print(f"  WARNING: found {len(inconsistent)} internally inconsistent options")
