"""Independent oracle for the informal-spread and everyday-set-description questions of p6d.

Spread sets oracle."""

import re
from mathgen.core import make


def parse_spread_threshold_and_rule(text: str):
    """
    Parse the rule and threshold from spread question text.
    Returns (rule_description, threshold_value).

    Three possible rules:
    1. "within X of every other value" -> max-min <= X
    2. "more than X away from every other value" -> has outlier check
    3. "difference of more than X between consecutive values" -> gap between clusters
    """
    # First check: "within X of every other value" pattern
    match = re.search(r"within\s+(\d+)\s+of\s+every\s+other", text)
    if match:
        threshold = int(match.group(1))
        return ("tightly_packed", threshold)

    # Second check: outlier - "more than X away from every other value"
    match = re.search(r"more than\s+(\d+)\s+away from every other value", text)
    if match:
        threshold = int(match.group(1))
        return ("has_outlier", threshold)

    # Third check: gap between clusters - "difference of more than X between"
    match = re.search(r"difference of more than\s+(\d+)\s+between", text)
    if match:
        threshold = int(match.group(1))
        return ("gap_between_clusters", threshold)

    return None


def check_tightly_packed(values, threshold):
    """Check if max(values) - min(values) <= threshold"""
    if not values:
        return False
    return max(values) - min(values) <= threshold


def check_has_outlier(values, threshold):
    """Check if any value's distance to EVERY other value exceeds threshold"""
    if len(values) < 2:
        return False

    for i, candidate in enumerate(values):
        is_outlier = True
        for j, other in enumerate(values):
            if i == j:  # Skip comparing with itself
                continue
            if abs(candidate - other) <= threshold:
                is_outlier = False
                break
        if is_outlier:
            return True
    return False


def check_gap_between_clusters(values, threshold):
    """Check if there's a consecutive gap > threshold with 2+ values on each side"""
    if len(values) < 4:
        return False

    sorted_vals = sorted(values)  # Sort but don't deduplicate
    if len(sorted_vals) < 4:
        return False

    for i in range(len(sorted_vals) - 1):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        if gap > threshold:
            # Count values strictly before and after gap
            before_count = i + 1
            after_count = len(sorted_vals) - (i + 1)
            if before_count >= 2 and after_count >= 2:
                return True
    return False


def test_spread_enumeration():
    """Test spread skill over 150+ seeds per tier"""
    challenged = []
    total = 0

    tiers = ["easy", "medium", "hard"]
    for tier in tiers:
        for seed in range(160):
            try:
                q = make("use_informal_ways_of_describing_spread_such_as", seed=seed, difficulty=tier)
            except Exception as e:
                challenged.append((tier, seed, f"Error generating: {str(e)[:50]}"))
                continue

            total += 1

            # Parse the rule from question text
            rule_info = parse_spread_threshold_and_rule(q.text)
            if not rule_info:
                challenged.append((tier, seed, f"Could not parse rule from: {q.text[:80]}"))
                continue

            rule_type, threshold = rule_info

            # Extract all 4 data sets from question text
            options = {}
            for letter in ["A", "B", "C", "D"]:
                pattern = f"Data set {letter}:\\s*([\\d,\\s]+?)(?:;|$)"
                match = re.search(pattern, q.text)
                if match:
                    values_str = match.group(1).strip()
                    values = [int(v.strip()) for v in values_str.split(",")]
                    options[letter] = values

            if len(options) != 4:
                challenged.append((tier, seed, f"Found only {len(options)} options"))
                continue

            # Apply rule to each option
            correct_letters = []
            for letter, values in options.items():
                if rule_type == "tightly_packed":
                    if check_tightly_packed(values, threshold):
                        correct_letters.append(letter)
                elif rule_type == "has_outlier":
                    if check_has_outlier(values, threshold):
                        correct_letters.append(letter)
                elif rule_type == "gap_between_clusters":
                    if check_gap_between_clusters(values, threshold):
                        correct_letters.append(letter)

            # Verify exactly one correct
            if len(correct_letters) != 1:
                answer_text = q.answer_text
                match = re.search(r"Data set ([A-D])", answer_text)
                correct_letter = match.group(1) if match else "?"
                challenged.append(
                    (
                        tier,
                        seed,
                        f"Expected 1 correct, found {len(correct_letters)}: {correct_letters}",
                        f"Answer says {correct_letter}",
                    )
                )
                continue

            oracle_correct = correct_letters[0]

            # Verify answer_text matches
            answer_match = re.search(r"Data set ([A-D]):\s*(.+?)$", q.answer_text)
            if not answer_match:
                challenged.append((tier, seed, f"No letter in answer: {q.answer_text}"))
                continue

            answer_letter = answer_match.group(1)
            answer_data = answer_match.group(2)

            if oracle_correct != answer_letter:
                challenged.append(
                    (
                        tier,
                        seed,
                        f"Oracle says {oracle_correct}, answer says {answer_letter}",
                        f"Rule: {rule_type}, threshold: {threshold}",
                    )
                )
                continue

            # Verify answer data matches the correct option
            oracle_option_text = f"Data set {oracle_correct}: {', '.join(map(str, options[oracle_correct]))}"
            if answer_data.strip() != ", ".join(map(str, options[oracle_correct])):
                challenged.append(
                    (
                        tier,
                        seed,
                        f"Answer text data mismatch",
                        f"Oracle: {options[oracle_correct]}, Answer: {answer_data}",
                    )
                )
                continue

            # Verify distractors are the other three options
            distractor_set = set(q.distractors)
            expected_distractors = set(
                f"Data set {l}: {', '.join(map(str, options[l]))}" for l in ["A", "B", "C", "D"] if l != oracle_correct
            )
            if distractor_set != expected_distractors:
                challenged.append(
                    (tier, seed, "Distractor mismatch", f"Expected: {expected_distractors}, Got: {distractor_set}")
                )

    print(f"Spread: {total} questions examined, {len(challenged)} challenged")
    if challenged:
        for item in challenged[:3]:
            print(f"  Challenge: {item}")

    assert len(challenged) == 0, f"Found {len(challenged)} challenged questions in spread skill"


def categorize_set_description(desc_text):
    """
    Given the description text from the question, determine its semantic category.
    Check specific phrases BEFORE generic fallbacks.
    Returns one of: intersection, union, diff_first_minus_second, complement,
                    symmetric_difference, all_three, diff_all_but_one, etc.
    """
    desc = desc_text.lower()

    # Check for "none of" first (complement with 3+ groups)
    if "none of" in desc or "not in any" in desc:
        return "complement"

    # Check specific 3-group patterns
    if "all three of" in desc:
        return "all_three"
    if " and " in desc and "but not" in desc and desc.count(" and ") >= 2:
        # "A and B but not C" (has 2+ "and"s in 3-group format)
        return "diff_all_but_one"

    # Check specific 2-group patterns (BEFORE 3-group patterns that include "but not")
    # "A and B but not C" should be caught by the 3-group check above
    if "but not" in desc:
        # Check if it's "A and B but not C" (3-group) or just "A but not B" (2-group)
        before_but = desc.split("but not")[0]
        if " and " in before_but:
            # "A and B but not C" -> has "and" before "but not" -> 3-group difference
            return "diff_all_but_one"
        # Just "A but not B" (2-group)
        return "diff_first_minus_second"

    if "exactly one" in desc:
        # "exactly one of A or B" or hard: "exactly one of A, B or C"
        return "symmetric_difference"

    if "neither" in desc and "nor" in desc:
        # "neither A nor B"
        return "complement"

    if "both" in desc and "and" in desc:
        # "both A and B"
        return "intersection"

    # Generic "A or B" / "at least one" (must be after specific checks)
    if "at least one" in desc or (" or " in desc and "exactly one" not in desc and "none of" not in desc):
        # "at least one of A or B" or "A or B"
        return "union"

    return "unknown"


def categorize_set_notation(notation_text):
    """
    Given a set notation phrase, determine its semantic category.
    Check specific markers, not generic words that appear in multiple types.
    For difference, distinguish which group is first (removed) vs second (doing removing).
    """
    text = notation_text.lower()

    # Exact markers take priority
    if "removed" in text or ("with" in text and "removed" in text):
        # "A with B removed" means A minus B (first group minus second)
        if " and " in text and "with" in text:
            # "A and B, with C removed" -> A ∩ B - C format
            return "diff_all_but_one"
        # "A with B removed" -> A - B (first minus second)
        return "diff_first_minus_second"

    if "outside" in text:
        # "everyone outside both" or "everyone outside all three"
        # Both are forms of complement
        return "complement"

    # Check if this is a 3-group intersection by counting group names or commas
    # "the intersection of X, Y and Z" -> all_three
    # "the intersection of X and Y" -> intersection (2-group)
    if "intersection" in text:
        # Count commas and "and" to determine if 3+ groups
        comma_count = text.count(",")
        if comma_count >= 1 or text.count(" and ") >= 2:
            # Likely 3+ groups: "X, Y and Z" has 1 comma
            return "all_three"
        return "intersection"

    if "union" in text:
        return "union"

    if "exactly one" in text:
        return "symmetric_difference"

    return "unknown"


def test_set_enumeration():
    """Test set skill over 150+ seeds per tier"""
    challenged = []
    boundary_failed = False
    boundary_questions_checked = 0
    total = 0

    tiers = ["easy", "medium", "hard"]
    for tier in tiers:
        for seed in range(160):
            try:
                q = make("use_everyday_occurrences_to_illustrate_set_descriptions_and", seed=seed, difficulty=tier)
            except Exception as e:
                challenged.append((tier, seed, f"Error generating: {str(e)[:50]}"))
                continue

            total += 1
            boundary_questions_checked += 1

            # BOUNDARY CHECK: no explicit set elements (no braces)
            all_text = q.text + " " + q.answer_text + " " + " ".join(q.distractors)
            if "{" in all_text or "}" in all_text:
                boundary_failed = True
                challenged.append((tier, seed, "BOUNDARY FAIL: found braces in question/answers"))
                continue

            # Extract the description from question text
            desc_match = re.search(r"following group\?\s*(.+?)\s*(?:\.?\s*Options:|$)", q.text)
            if not desc_match:
                challenged.append((tier, seed, "Could not extract description"))
                continue

            description = desc_match.group(1).strip()
            desc_category = categorize_set_description(description)

            if desc_category == "unknown":
                challenged.append((tier, seed, f"Unknown description: {description[:50]}"))
                continue

            # Extract the 4 option texts from question
            options_match = re.search(r"Options:\s*(.+?)$", q.text, re.DOTALL)
            if not options_match:
                challenged.append((tier, seed, "Could not find Options section"))
                continue

            options_text = options_match.group(1)

            # Parse each option
            options = {}
            for letter in ["A", "B", "C", "D"]:
                pattern = f"{letter}:\\s*([^;]*?)(?:;|$)"
                match = re.search(pattern, options_text)
                if match:
                    option_desc = match.group(1).strip()
                    options[letter] = option_desc

            if len(options) != 4:
                challenged.append((tier, seed, f"Found only {len(options)} options"))
                continue

            # Categorize each option
            option_categories = {}
            for letter, option_text in options.items():
                option_category = categorize_set_notation(option_text)
                option_categories[letter] = (option_category, option_text)

            # Find options in the same semantic category as the description
            semantic_matches = [letter for letter, (cat, text) in option_categories.items() if cat == desc_category]

            # Among semantic matches, find which one(s) exactly equal the answer text
            exact_matches = [letter for letter in semantic_matches if options[letter] == q.answer_text]

            # Verify exactly one exact match
            if len(exact_matches) != 1:
                challenged.append(
                    (
                        tier,
                        seed,
                        f"Found {len(exact_matches)} exact matches: {exact_matches} (semantic: {semantic_matches})",
                        f"Desc: '{description}' ({desc_category}), Answer: '{q.answer_text}'",
                    )
                )
                continue

            oracle_correct = exact_matches[0]

            # Verify distractors are the other three options
            distractors_from_options = {options[l] for l in ["A", "B", "C", "D"] if l != oracle_correct}
            distractors_from_question = set(q.distractors)
            if distractors_from_options != distractors_from_question:
                challenged.append(
                    (
                        tier,
                        seed,
                        "Distractor mismatch",
                        f"Expected: {distractors_from_options}, Got: {distractors_from_question}",
                    )
                )

    print(f"Set: {total} questions examined, {len(challenged)} challenged")
    print(
        f"Boundary check: {'PASS' if not boundary_failed else 'FAIL'} ({boundary_questions_checked} questions checked)"
    )
    if challenged:
        for item in challenged[:3]:
            print(f"  Challenge: {item}")

    assert not boundary_failed, f"Boundary check failed"
    assert len(challenged) == 0, f"Found {len(challenged)} challenged questions in set skill"


if __name__ == "__main__":
    test_spread_enumeration()
    test_set_enumeration()
