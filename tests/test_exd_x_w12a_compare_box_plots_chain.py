"""Tests the exam chain for comparing two box plots: two IQRs, their difference, a justified conclusion."""

from __future__ import annotations

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

TOPIC = "exd_x_w12a_compare_box_plots_chain"
SKILL_ID = "f10-compare-box-plots"


def oracle(q):
    """Both IQRs, their gap, and who has the higher median and the larger IQR, from the plotted summaries."""
    first, second = q.figure["plots"]
    ia, ib = first["q3"] - first["q1"], second["q3"] - second["q1"]
    higher = first if first["median"] > second["median"] else second
    wider = first if ia > ib else second
    return ia, ib, abs(ia - ib), higher["label"], wider["label"]


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, "stat5_box_plot_compare", {"tier": "hard"})


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert q.figure["kind"] == "box_plot" and len(q.figure["plots"]) == 2
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        assert [p.marks for p in q.parts] == [1, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]
        assert all(len(p.distractors) == 3 for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        ia, ib, gap, higher, wider = oracle(q)
        a, b, c, d = canonical(q).values()
        names = [p["label"] for p in q.figure["plots"]]
        statement_ok = d.startswith(higher) and (f"{wider} has the larger IQR" in d or f"{wider} has both" in d)
        if (int(a), int(b), int(c)) != (ia, ib, gap) or not statement_ok or higher not in names:
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_the_shortcuts_point_the_other_way(questions):
    """The higher median has the lower whisker end, and the larger IQR the shorter range: the named errors differ."""
    for q in questions[:300]:
        first, second = q.figure["plots"]
        assert (first["median"] > second["median"]) != (first["max"] > second["max"])
        range_a, range_b = first["max"] - first["min"], second["max"] - second["min"]
        iqr_a, iqr_b = first["q3"] - first["q1"], second["q3"] - second["q1"]
        assert (iqr_a > iqr_b) != (range_a > range_b)


def test_all_four_conclusions_are_offered_and_all_orderings_occur(questions):
    seen = set()
    for q in questions[:300]:
        _, _, _, higher, wider = oracle(q)
        names = [p["label"] for p in q.figure["plots"]]
        seen.add((names.index(higher), names.index(wider)))
        assert len({q.parts[3].answer_text, *q.parts[3].distractors}) == 4
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_first_iqr():
    """A first IQR that is too big changes the difference to the learner's own gap."""

    def carried(q):
        ia, ib, _, _, _ = oracle(q)
        # A reply that merely repeats an earlier one earns no follow-through, so keep the gap clear of both.
        wrong = next(w for w in range(ia + 3, ia + 30) if abs(w - ib) not in (ib, w) and abs(w - ib) != ia)
        return "a", str(wrong), {"c": str(abs(wrong - ib))}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
