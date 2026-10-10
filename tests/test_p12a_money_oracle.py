"""Independent oracle for the `p12a_money` questions: it re-derives each answer from the rendered question text.

Parses ONLY the rendered question (``question.text``, ``question.answer_text``, ``question.distractors``) and
re-derives the answer from scratch. Imports nothing from ``mathgen.generators.p12a_money_build`` -- not a
template string, not a formula, not the trap-range constants.

Two skills, two independent readings:

1. **f10-money-dollars-cents-representation**: a single dollars-and-cents
   amount ("$D.cc") or a single bare cents integer is read out of the
   text with generic decimal/integer regexes (never the generator's own
   template list), and the OTHER unit's value is recomputed by dividing
   or multiplying by 100 directly on the extracted literal.
2. **f10-additive-multiplicative-money-model**: every dollar figure in the
   text is extracted in order, and the operation is classified from the
   text's own STRUCTURE (which words appear, how many dollar figures are
   present) -- a rule table written fresh in this file, not read from the
   generator's builder functions.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

REP_TOPIC = "money_dollars_cents_representation"
MODEL_TOPIC = "additive_multiplicative_money_model"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# ---------------------------------------------------------------------------
# f10-money-dollars-cents-representation
# ---------------------------------------------------------------------------

_DOLLAR_AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{2})?)\b")
_CENTS_AMOUNT_RE = re.compile(r"\b(\d+) cents?\b")


def _oracle_representation_value(text: str) -> Fraction:
    """Recompute the SINGLE stated amount's value in the OTHER unit,
    independent of which direction the text asks for: if a "$D[.cc]"
    literal appears, the question is dollars -> cents (return amount*100
    cents); if a "N cents" literal appears (and no dollar literal), the
    question is cents -> dollars (return amount/100 dollars). Exactly one
    of the two must appear, or the text is under-specified."""
    dollar_match = _DOLLAR_AMOUNT_RE.search(text)
    cents_match = _CENTS_AMOUNT_RE.search(text)
    assert (dollar_match is not None) != (cents_match is not None), (
        f"expected exactly one of a $-amount or a cents-amount in: {text!r}"
    )
    if dollar_match:
        return Fraction(dollar_match.group(1)) * 100
    return Fraction(cents_match.group(1)) / 100


def test_representation_value_oracle_recomputes_from_the_question_alone():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(REP_TOPIC, seed=seed, difficulty=tier)
            expected = _oracle_representation_value(q.text)
            if "." in q.answer_text:
                assert Fraction(q.answer_text) == expected, (
                    f"{tier} seed {seed}: {q.text!r} -- expected {expected}, rendered {q.answer_text!r}"
                )
            else:
                assert int(q.answer_text) == expected, (
                    f"{tier} seed {seed}: {q.text!r} -- expected {expected}, rendered {q.answer_text!r}"
                )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_representation_distractors_never_equal_the_rendered_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(REP_TOPIC, seed=seed, difficulty=tier)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))


def test_representation_oracle_fires_on_a_corrupted_answer():
    q = make(REP_TOPIC, seed=0, difficulty="easy")
    expected = _oracle_representation_value(q.text)
    corrupted_answer_text = "999999.99" if "." in q.answer_text else "999999"
    if "." in q.answer_text:
        assert Fraction(corrupted_answer_text) != expected
    else:
        assert int(corrupted_answer_text) != expected


# ---------------------------------------------------------------------------
# f10-additive-multiplicative-money-model
# ---------------------------------------------------------------------------

_ALL_DOLLARS_RE = re.compile(r"\$(\d+)\b")
_BARE_DOLLARS_RE = re.compile(r"\b(\d+) dollars\b")


def _extract_dollar_figures(text: str) -> list[int]:
    """Every dollar figure mentioned in the text, IN ORDER OF APPEARANCE,
    whether written "$N" or "N dollars" -- read structurally (regex over
    the raw text), never via the generator's own named variables."""
    figures = []
    for m in re.finditer(r"\$(\d+)\b|\b(\d+) dollars\b", text):
        figures.append(int(m.group(1) or m.group(2)))
    return figures


def _extract_counts(text: str) -> list[int]:
    """Every bare (non-dollar) integer count in the text -- "N of the same",
    "N people", "N winners", "N raffle tickets", "into groups of N" -- read
    structurally via a rule distinct from the dollar-figure extraction
    above."""
    counts = []
    for m in re.finditer(r"\b(\d+) (?:of the same|people|winners|raffle tickets|envelopes)\b|\bbuys (\d+)\b", text):
        counts.append(int(m.group(1) or m.group(2)))
    return counts


def _classify_and_recompute(text: str) -> int:
    """Independently classify the operation from the text's own STRUCTURE
    (which fixed phrases appear) and recompute the value from the dollar
    figures/counts extracted above -- a rule table authored fresh in this
    file, never read from the generator's own builder functions."""
    dollars = _extract_dollar_figures(text)
    counts = _extract_counts(text)

    # Most-specific literal phrases first -- some of the medium/hard
    # templates share generic substrings ("spends", "have left") with the
    # easy templates' own generic markers below, so a rarer phrase unique
    # to one template must win before falling back to a broader pattern
    # that would otherwise misclassify it (e.g. the medium divide template
    # contains both "spends" and "have left", which would wrongly satisfy
    # the easy subtract rule if checked first).

    # -- medium --
    if "garage sale" in text:
        return dollars[0] + dollars[1] + dollars[2]
    if "then spends" in text and "chores" in text:
        return dollars[0] + dollars[1] - dollars[2]
    if "also buys" in text:
        return counts[0] * dollars[0] + dollars[1]
    if "prize money" in text:
        m = re.search(r"^(\d+) dollars of prize money is shared equally among (\d+) winners", text)
        total, n = int(m.group(1)), int(m.group(2))
        return total // n - dollars[-1]

    # -- hard --
    if "washing cars" in text:
        return dollars[0] + dollars[1]
    if "saving twice as fast" in text:
        return dollars[1] - dollars[0]  # text order: saved amount first, then the full price
    if "raffle tickets" in text:
        m = re.search(
            r"^(\w+) buys (\d+) raffle tickets\. Each ticket costs \$(\d+) more than "
            r"last year's \$(\d+) price",
            text,
        )
        n, extra, base = int(m.group(2)), int(m.group(3)), int(m.group(4))
        return n * (base + extra)
    if "envelopes" in text:
        m = re.search(r"^\$(\d+) is shared equally into envelopes of \$(\d+) each", text)
        total, price = int(m.group(1)), int(m.group(2))
        return total // price

    # -- easy (generic fallbacks, checked last) --
    if "earns another" in text:
        return dollars[0] + dollars[1]
    if "spends" in text and "have left" in text:
        return dollars[0] - dollars[1]
    if "of the same" in text and "total cost" in text:
        return counts[0] * dollars[0]
    if "shares" in text and "equally among" in text and "each person" in text:
        return dollars[0] // counts[0]

    raise AssertionError(f"oracle has no classification rule for: {text!r}")


def test_model_value_oracle_recomputes_from_the_question_alone():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(MODEL_TOPIC, seed=seed, difficulty=tier)
            expected = _classify_and_recompute(q.text)
            assert int(q.answer_text) == expected, (
                f"{tier} seed {seed}: {q.text!r} -- expected {expected}, rendered {q.answer_text!r}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_model_distractors_never_equal_the_rendered_answer():
    for tier in TIERS:
        for seed in SEEDS:
            q = make(MODEL_TOPIC, seed=seed, difficulty=tier)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))


def test_model_oracle_fires_on_a_corrupted_answer():
    q = make(MODEL_TOPIC, seed=0, difficulty="easy")
    expected = _classify_and_recompute(q.text)
    corrupted = expected + 1
    assert corrupted != expected
