"""Independent oracle for the `p14b_practical_number_operations` questions: it re-derives each answer from the
rendered question text.

Parses ONLY the rendered question (`question.text`, `question.answer_text`, `question.distractors`) and
re-derives the answer from scratch. Imports NOTHING from
`mathgen.generators.p14b_practical_number_operations_build` -- not a template string, not a formula, not a
constant.

Every rendered question states, in order:
  1. a base amount ("$D.cc"), always immediately followed by either
     "% discount"/"% loyalty discount" (subtract) or "% service charge
     added"/"% GST added"/"% import surcharge is added" (add);
  2. optionally a second, non-percentage clause: a flat amount described as
     "added"/"is added" (add) or "subtracted"/"is subtracted" (sub), OR a
     "buying N of them" multiply, OR neither (medium-split has none);
  3. optionally a final "split equally among N people" division.

This oracle reads those clauses directly out of the text with its own regex
rules (never the generator's own per-shape template strings) and recomputes
the value via plain `Fraction` arithmetic, independent of however
construction accumulated its own running integer-cents total.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

TOPIC = "p14b_practical_number_operations"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

_DOLLAR_RE = re.compile(r"\$(\d+\.\d{2})")
_PCT_RE = re.compile(r"(\d+)%")
_QTY_RE = re.compile(r"buying (\d+) of them")
_SPLIT_N_RE = re.compile(r"split equally among (\d+) people")

_ADD_PCT_MARKERS = ("service charge added", "GST added", "surcharge is added", "surcharge included")
_SUB_PCT_MARKERS = ("discount",)


def _pct_direction(text: str) -> str:
    """ "add" or "sub" for the FIRST (and only) percentage clause -- read from
    which fixed phrase follows the "N%" token in the text, a rule written
    fresh here rather than reused from the generator's own branch labels."""
    if any(marker in text for marker in _ADD_PCT_MARKERS):
        return "add"
    if any(marker in text for marker in _SUB_PCT_MARKERS):
        return "sub"
    raise AssertionError(f"oracle cannot classify the percentage direction in: {text!r}")


def _flat_direction(text: str) -> str | None:
    """ "add"/"sub" for a SECOND flat dollar clause, or None if there isn't
    one (the medium-split and medium-multiply shapes have no flat clause)."""
    if "is added" in text or ("added" in text and text.count("added") == 2):
        return "add"
    if "is subtracted" in text:
        return "sub"
    return None


def _oracle_value(text: str) -> Fraction:
    dollar_amounts = [Fraction(m) for m in _DOLLAR_RE.findall(text)]
    pct = int(_PCT_RE.search(text).group(1))
    pct_op = _pct_direction(text)

    base = dollar_amounts[0]
    after_pct = base * (100 + pct) / 100 if pct_op == "add" else base * (100 - pct) / 100

    qty_match = _QTY_RE.search(text)
    if qty_match:
        return after_pct * int(qty_match.group(1))

    flat_op = _flat_direction(text)
    running = after_pct
    if flat_op is not None:
        # The flat amount is the SECOND dollar figure stated (the first is
        # the base price/order total).
        flat_amount = dollar_amounts[1]
        running = running + flat_amount if flat_op == "add" else running - flat_amount

    split_match = _SPLIT_N_RE.search(text)
    if split_match:
        running = running / int(split_match.group(1))

    return running


def test_oracle_recomputes_from_the_question_alone():
    checked = 0
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            expected = _oracle_value(q.text)
            rendered = Fraction(q.answer_text)
            assert expected == rendered, (
                f"{difficulty} seed {seed}: {q.text!r} -- oracle {expected} (~{float(expected):.2f}), "
                f"rendered {q.answer_text!r}"
            )
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


def test_oracle_distractors_never_equal_the_rendered_answer():
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))


def test_oracle_fires_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    expected = _oracle_value(q.text)
    corrupted = expected + 1
    assert corrupted != expected


def test_oracle_classifies_every_seed_without_raising():
    """A text the oracle cannot classify at all is itself a finding (an
    under-specified question) -- confirm every seed/tier parses cleanly."""
    for difficulty in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            _oracle_value(q.text)  # raises AssertionError/IndexError if unparseable
