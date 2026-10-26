"""Tests the exam chain for causation versus correlation: r squared, the unexplained variation, the design verdict
and the justified conclusion."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_gm064_causation_chain"
SKILL_ID = "senior-distinguish-between-causality-and-correlation-through-examples"


def oracle(q):
    """(a) to (d) re-derived from the rendered text: r, then the design from its wording."""
    text = q.text
    r = Fraction(re.search(r"r = (-?\d\.\d+)", text).group(1))
    explained = r * r * 100
    randomised = "randomly assigned" in text or "chosen at random" in text
    observational = "third variable" in text or "associated with both" in text
    assert randomised != observational, text
    n = int(re.search(r"\d+", text).group())
    # a randomised experiment supports a firm causal claim only when 30 or more took part
    return explained, 100 - explained, randomised and n >= 30


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_wired():
    check_wired(SKILL_ID, TOPIC)


def test_unknown_tier_is_refused():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="hard")


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "mcq", "mcq"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False]
        assert all(len(p.distractors) == 3 for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        explained, unexplained, randomised = oracle(q)
        a, b, c, d = canonical(q).values()
        c_yes = c.startswith("Yes")
        d_causal = d.startswith("Changes in") and " may cause" not in d
        pct = int(re.search(r"(\d+)% of the variation", d).group(1))
        if (Fraction(a), Fraction(b), c_yes, d_causal, pct) != (
            explained,
            unexplained,
            randomised,
            randomised,
            explained,
        ):
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_a_small_experiment_cannot_support_a_firm_causal_verdict(questions):
    """Among the experiments, the verdict follows n: firm for 30 or more, withheld for fewer, and the stem says why."""
    seen = set()
    for q in questions[:300]:
        if "randomly assigned" not in q.text and "chosen at random" not in q.text:
            continue
        n = int(re.search(r"\d+", q.text).group())
        assert "fewer than 30 as too small" in q.text
        c, d = canonical(q)["c"], canonical(q)["d"]
        assert c.startswith("Yes") == (n >= 30), (q.text, c)
        assert ("too few" in c) == (n < 30) and ("firm causal conclusion is not justified" in d) == (n < 30)
        seen.add(n < 30)
    assert seen == {True, False}


def test_both_verdicts_and_signs_occur(questions):
    assert {oracle(q)[2] for q in questions[:300]} == {True, False}
    signs = {q.text.split("r = ")[1].startswith("-") for q in questions[:300]}
    assert signs == {True, False}


def test_a_survey_names_a_third_variable_and_an_experiment_does_not(questions):
    for q in questions[:120]:
        randomised = "randomly assigned" in q.text or "chosen at random" in q.text
        c = canonical(q)["c"]
        assert (" may explain the association" in c) != randomised


def test_answer_space_and_shapes():
    check_answer_space(TOPIC)
    check_shapes(TOPIC, "causality_correlation_classify", {"question_type": "hard"})


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_explained_percentage():
    """A learner's own (a) carries into (b): 100 minus whatever they gave."""

    def carried(q):
        explained, _, _ = oracle(q)
        wrong = explained + 7
        return "a", str(int(wrong)), {"b": str(int(100 - wrong))}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def _key_length_rank(part) -> int:
    """How many of the part's options are strictly shorter than its key."""
    return sum(len(str(getattr(d, "text", d))) < len(part.answer_text) for d in part.distractors)


@pytest.mark.parametrize("randomised", [True, False], ids=["experiment", "survey"])
@pytest.mark.parametrize("label", ["c", "d"])
def test_key_length_gives_nothing_away(questions, label, randomised):
    """Within each design, over 300 seeds, the key lands at every length rank and no rank takes more than 40%."""
    mine = [q for q in questions[:300] if ("randomly assigned" in q.text or "chosen at random" in q.text) == randomised]
    ranks = Counter(_key_length_rank(next(p for p in q.parts if p.label == label)) for q in mine)
    assert set(ranks) == {0, 1, 2, 3}, ranks
    assert max(ranks.values()) <= 0.4 * len(mine), ranks
