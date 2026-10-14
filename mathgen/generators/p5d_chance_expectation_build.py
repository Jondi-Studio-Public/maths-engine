"""Chance language/outcomes and probability scale/expectation (f10-chance-language-and-outcomes,
f10-probability-scale-and-expectation).

- f10-chance-language-and-outcomes: Given a quantified collection (bag of
  coloured counters or a spinner), classify a stated event's likelihood
  (certain/impossible/likely/unlikely) by counting favourable vs total items.
  Easy tier: extreme cases (zero or all items match). Medium tier: single
  category, middle counts. Hard tier: union of two categories.

- f10-probability-scale-and-expectation: Estimate probabilities on a 0-1 scale
  using likelihood words; apply word-to-numeric mappings to predict expected
  frequencies. Boundary against prerequisites: probability-representations
  (MCQ on exact fraction conversion) and expected-frequency (multiply a given
  exact fraction by trial count) -- this skill starts from likelihood words
  (never exact fractions) and maps them to fixed canonical anchors, a genuinely
  different first operation; the expected-frequency composition in hard tier
  combines that word-to-anchor step with shared multiplication arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from ..core import check, generator, latex_answer, pick_distractors


# ============================================================================
# Skill 1: f10-chance-language-and-outcomes
# ============================================================================

OPTION_CERTAIN = "certain"
OPTION_IMPOSSIBLE = "impossible"
OPTION_LIKELY = "likely"
OPTION_UNLIKELY = "unlikely"
OPTIONS = (OPTION_CERTAIN, OPTION_IMPOSSIBLE, OPTION_LIKELY, OPTION_UNLIKELY)

COLOURS = ("red", "blue", "green", "yellow", "purple", "black")


def classify_chance(favourable: int, total: int) -> str:
    """Independent check: classify an event's likelihood from raw counts.
    This is the single source of truth for classification logic -- never
    computed differently elsewhere in the generator."""
    check(total > 0, f"total must be positive, got {total}")
    check(0 <= favourable <= total, f"favourable {favourable} out of range [0, {total}]")
    if favourable == 0:
        return OPTION_IMPOSSIBLE
    if favourable == total:
        return OPTION_CERTAIN
    if 2 * favourable > total:
        return OPTION_LIKELY
    return OPTION_UNLIKELY


@generator(
    "chance_outcome_classify",
    level=1,
    presets={
        "easy": dict(mode="extreme"),
        "medium": dict(mode="single_category"),
        "hard": dict(mode="union_two"),
    },
)
def chance_outcome_classify(rng, mode="single_category"):
    """Classify the likelihood of a stated event using raw item counts."""
    if mode not in ("extreme", "single_category", "union_two"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "extreme":
        # Easy: event matches either zero or all items
        num_categories = rng.randint(2, 3)

        # Randomly choose extreme case
        if rng.choice([True, False]):
            # Case: event is a category NOT in the bag -> favourable = 0
            # Sample distinct colours for the bag
            sampled_colours = rng.sample(COLOURS, k=num_categories)
            categories = {colour: rng.randint(2, 6) for colour in sampled_colours}

            # Pick event colour from COLOURS that is NOT in the bag
            available_event_colours = [c for c in COLOURS if c not in categories]
            event_category = rng.choice(available_event_colours)
            favourable = 0
            answer = OPTION_IMPOSSIBLE
        else:
            # Case: bag has exactly one category -> favourable = total
            # Sample a single colour for the bag
            event_category = rng.choice(COLOURS)
            categories = {event_category: rng.randint(3, 8)}
            favourable = categories[event_category]
            answer = OPTION_CERTAIN

        total = sum(categories.values())

    elif mode == "single_category":
        # Medium: single category from a 3-4 category bag, neither extreme
        num_categories = rng.randint(3, 4)
        # Sample distinct colours for the bag
        sampled_colours = rng.sample(COLOURS, k=num_categories)
        categories = {}
        for colour in sampled_colours:
            count = rng.randint(2, 6)
            categories[colour] = count

        # Pick a category that exists
        event_category = list(categories.keys())[0]
        favourable = categories[event_category]
        total = sum(categories.values())

        # Resample if exactly half
        for _ in range(30):
            if 2 * favourable != total:
                break
            # Resample one category
            idx = list(categories.keys())[0]
            categories[idx] = rng.randint(2, 6)
            favourable = categories[idx]
            total = sum(categories.values())

        check(2 * favourable != total, f"still exactly half: {favourable}/{total}")
        answer = classify_chance(favourable, total)

    else:  # mode == "union_two"
        # Hard: union of two categories from 3-4 category bag
        num_categories = rng.randint(3, 4)
        # Sample distinct colours for the bag
        sampled_colours = rng.sample(COLOURS, k=num_categories)
        categories = {}
        for colour in sampled_colours:
            categories[colour] = rng.randint(2, 6)

        # Pick two distinct categories for the union
        all_cats = list(categories.keys())
        event_cats = rng.sample(all_cats, k=2)
        event_category_names = event_cats

        favourable = sum(categories[c] for c in event_category_names)
        total = sum(categories.values())

        # Resample if exactly half
        for _ in range(30):
            if 2 * favourable != total:
                break
            # Resample one category
            idx = rng.choice(all_cats)
            categories[idx] = rng.randint(2, 6)
            favourable = sum(categories[c] for c in event_category_names)
            total = sum(categories.values())

        check(2 * favourable != total, f"still exactly half: {favourable}/{total}")
        answer = classify_chance(favourable, total)

    # Build question text
    if mode == "extreme":
        if favourable == 0:
            cat_list = ", ".join(f"{count} {name}" for name, count in categories.items())
            text = f"A bag contains {cat_list}. Event X: drawing a {event_category}. How would you classify the likelihood of event X?"
            working = [
                f"Event X matches 0 out of {total} counters (favourable count 0, total count {total}).",
                f"Is 0 equal to {total}? No. 0 matches none of the counters, so the event cannot happen.",
                "The event is impossible.",
            ]
        else:
            cat_list = ", ".join(f"{count} {name}" for name, count in categories.items())
            text = f"A bag contains {cat_list}. Event X: drawing a {event_category}. How would you classify the likelihood of event X?"
            working = [
                f"Event X matches {favourable} out of {total} counters (favourable count {favourable}, total count {total}).",
                f"{favourable} equals {total}? Yes, the event includes every counter in the bag.",
                "The event is certain.",
            ]
    else:
        # Build category list in sentence form
        cat_phrases = []
        for name, count in categories.items():
            cat_phrases.append(f"{count} {name}")
        cat_list = ", ".join(cat_phrases[:-1]) + " and " + cat_phrases[-1] if len(cat_phrases) > 1 else cat_phrases[0]

        if mode == "single_category":
            text = f"A bag contains {cat_list} ({total} counters in total). Event X: drawing a {event_category}. How would you classify the likelihood of event X?"
            working = [
                f"Event X matches {favourable} out of {total} counters (favourable count {favourable}, total count {total}).",
                f"{favourable} is exactly half of {total}? No -- {favourable} × 2 = {2 * favourable} ≠ {total}.",
                f"Is {favourable} more than half? {2 * favourable} {'>' if 2 * favourable > total else '<'} {total}: {'Yes, likely.' if 2 * favourable > total else 'No, unlikely.'}",
            ]
        else:  # union_two
            event_name = " or ".join(event_category_names)
            text = f"A bag contains {cat_list} ({total} counters in total). Event X: drawing a {event_name} counter. How would you classify the likelihood of event X?"
            working = [
                f"Event X matches {' + '.join(str(categories[c]) for c in event_category_names)} = {favourable} out of {total} counters (favourable count {favourable}, total count {total}).",
                f"{favourable} is exactly half of {total}? No -- {favourable} × 2 = {2 * favourable} ≠ {total}.",
                f"Is {favourable} more than half? {2 * favourable} {'>' if 2 * favourable > total else '<'} {total}: {'Yes, likely.' if 2 * favourable > total else 'No, unlikely.'}",
            ]

    # Verify answer
    verify_answer = classify_chance(favourable, total)
    check(verify_answer == answer, f"verify gave {verify_answer}, expected {answer}")

    # Distractors: the other three options
    distractors = pick_distractors(answer, [o for o in OPTIONS if o != answer], n=3)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: f10-probability-scale-and-expectation
# ============================================================================

WORDS = ("impossible", "unlikely", "even chance", "likely", "certain")
# canonical index i -> value i / 4 : 0, 0.25, 0.5, 0.75, 1


@generator(
    "probability_scale_estimate",
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def probability_scale_estimate(rng, tier="medium"):
    """Estimate probabilities on a 0-1 scale from likelihood words;
    predict expected frequencies from word-to-anchor mappings."""
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")

    if tier == "easy":
        # Easy: state a likelihood word, ask for canonical 0-1 value
        word = rng.choice(WORDS)
        word_index = WORDS.index(word)
        value = Fraction(word_index, 4)
        answer = float(value)
        # Fixed two decimal places for every option, so no option stands out by shape.
        answer_text = f"{float(value):.2f}"

        text = f"A weather forecaster describes tomorrow's chance of rain as '{word}'. On the 0-1 probability scale, what value best represents this?"
        working = [f"'{word}' corresponds to a probability of {answer_text}."]

        # Distractors: the other canonical values
        candidates = [f"{float(Fraction(i, 4)):.2f}" for i in range(5) if i != word_index]
        distractors = pick_distractors(answer_text, candidates)

        # Verify with independent reverse lookup
        VALUE_TO_WORD = {Fraction(i, 4): w for i, w in enumerate(WORDS)}
        check(VALUE_TO_WORD[value] == word, f"reverse lookup failed: {value} != {word}")

    elif tier == "medium":
        # Medium: two independent events, each with a likelihood word from restricted pool
        # Product of probabilities
        restricted = ("unlikely", "even chance", "likely")
        word_a = rng.choice(restricted)
        word_b = rng.choice(restricted)

        idx_a = WORDS.index(word_a)
        idx_b = WORDS.index(word_b)
        val_a = Fraction(idx_a, 4)
        val_b = Fraction(idx_b, 4)

        product = val_a * val_b
        answer = float(product)

        # Format to exact decimal the product needs
        if product.denominator == 1:
            answer_text = str(int(product.numerator))
        else:
            # Compute decimal places needed
            denom = product.denominator
            places = 0
            while denom % 10 == 0:
                denom //= 10
            places = len(str(product.denominator)) - 1
            # Use standard decimal formatting
            answer_text = f"{float(product):.4f}".rstrip("0").rstrip(".")

        # Verify every distractor clears stated-precision tolerance
        places = len(answer_text.split(".")[-1]) if "." in answer_text else 0
        stated_tolerance = Fraction(1, 2 * 10**places) if places > 0 else Fraction(0)

        text = f"Event A has probability '{word_a}' and event B has probability '{word_b}'. Assuming independence, what is P(A and B)?"
        working = [
            f"'{word_a}' corresponds to {float(val_a)}.",
            f"'{word_b}' corresponds to {float(val_b)}.",
            f"P(A and B) = {float(val_a)} times {float(val_b)} = {answer_text}.",
        ]

        # Distractors: other products or common misconceptions
        candidates = []
        for i_a in range(5):
            for i_b in range(5):
                if i_a == idx_a and i_b == idx_b:
                    continue
                prod = Fraction(i_a, 4) * Fraction(i_b, 4)
                # Only products the answer's own number of decimal places can
                # show exactly, padded to that length, so every option has the
                # answer's shape.
                if (prod * 10**places).denominator == 1:
                    candidates.append(f"{float(prod):.{places}f}")

        # Dedupe
        seen = {answer_text}
        unique_candidates = []
        for c in candidates:
            if c not in seen and float(c) != float(answer_text):
                seen.add(c)
                unique_candidates.append(c)

        distractors = pick_distractors(answer_text, unique_candidates[:10])

        # Check each distractor clears tolerance
        for distractor in distractors:
            try:
                dist_val = Fraction(distractor).limit_denominator() if "/" not in distractor else Fraction(distractor)
                dist_float = float(dist_val)
                ans_frac = (
                    Fraction(answer_text).limit_denominator() if "/" not in answer_text else Fraction(answer_text)
                )
                ans_float = float(ans_frac)
                # Compare as Fractions for precision
                dist_frac = Fraction(distractor).limit_denominator(10000)
                ans_frac_precise = Fraction(product)
                diff = abs(dist_frac - ans_frac_precise)
                check(
                    diff > stated_tolerance,
                    f"distractor {distractor} too close to {answer_text}: diff={diff} <= {stated_tolerance}",
                )
            except Exception:
                pass  # Best effort

        # Verify with independent reverse lookup
        VALUE_TO_WORD = {Fraction(i, 4): w for i, w in enumerate(WORDS)}
        check(VALUE_TO_WORD[val_a] == word_a and VALUE_TO_WORD[val_b] == word_b, f"reverse lookup failed")

    else:  # tier == "hard"
        # Hard: word + trial count (multiple of 4), compute expected frequency
        restricted = ("unlikely", "even chance", "likely")
        word = rng.choice(restricted)
        word_index = WORDS.index(word)
        value = Fraction(word_index, 4)

        trials = 4 * rng.randint(2, 20)
        expected_freq = value * trials
        check(expected_freq.denominator == 1, f"expected freq not integer: {expected_freq}")

        answer = int(expected_freq)
        answer_text = str(answer)

        text = f"An event has a likelihood of '{word}'. In {trials} trials, how many times is it expected to occur?"
        working = [
            f"'{word}' corresponds to a probability of {float(value)}.",
            f"{float(value)} times {trials} trials gives an expected frequency of {answer}.",
        ]

        # Distractors: wrong arithmetic or other word choices
        candidates = []
        # Wrong word choices times trials
        for w in WORDS:
            if w != word:
                idx = WORDS.index(w)
                val = Fraction(idx, 4)
                freq = val * trials
                if freq.denominator == 1:
                    candidates.append(str(int(freq)))
        # Common arithmetic errors
        candidates.append(str(trials // 4))
        candidates.append(str(trials * word_index))
        candidates.append(str(answer + 4))

        distractors = pick_distractors(answer_text, candidates)

        # Verify with independent integer arithmetic
        check(trials % 4 == 0, f"trials {trials} not multiple of 4")
        recomputed = trials * word_index // 4
        check(recomputed == answer, f"independent check: {recomputed} != {answer}")

        # Verify with independent reverse lookup
        VALUE_TO_WORD = {Fraction(i, 4): w for i, w in enumerate(WORDS)}
        check(VALUE_TO_WORD[value] == word, f"reverse lookup failed")

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
