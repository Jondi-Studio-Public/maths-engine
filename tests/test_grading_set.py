"""Tests grading of the `set` answer kind: canonical form, labels in any order, missing or wrong entries, tuples
and displayed precision.

The `set` answer kind."""

from __future__ import annotations

import pytest

from mathgen.core import Question

from mathgen.grading import typed as grading


def _q(items):
    return Question(
        topic="test", level=0, seed=1, text="q", latex="", answer=items, answer_text=grading.set_answer_text(items)
    )


def _ok(items, given) -> bool:
    return grading.grade("set", _q(items), given).correct


def test_canonical():
    assert _q(["length", "angle"]).answer_text == "{length, angle}"
    assert _q([(3, 4), (2, 14)]).answer_text == "{(3, 4), (2, 14)}"


@pytest.mark.parametrize(
    "given", ["{length, angle}", "angle, length", "Angle and LENGTH", "length; angle", "angle\nlength"]
)
def test_labels_in_any_order_and_form(given):
    assert _ok(["length", "angle"], given)


@pytest.mark.parametrize("given", ["length", "length, angle, area", "length, length", "length, area", ""])
def test_labels_missing_extra_or_wrong(given):
    assert not _ok(["length", "angle"], given)


def test_tuples_keep_their_inner_commas():
    items = [(3, 4), (2, 14), (1, 24)]
    assert _ok(items, "(1, 24), (3, 4), (2, 14)")
    assert _ok(items, "{(2,14); (3,4); (1,24)}")
    assert not _ok(items, "(4, 3), (2, 14), (1, 24)")
    assert not _ok(items, "(3, 4), (2, 14)")


def test_numbers_to_displayed_precision():
    items = ["0.40", 3, "1/2"]
    assert _ok(items, "3, 0.4, 0.5")
    assert _ok(items, "{1/2, 3, 0.404}")
    assert not _ok(items, "3, 0.41, 1/2")
    assert not _ok(items, "3.04, 0.40, 1/2")


def test_mixed_labels_and_numbers():
    assert _ok(["heads", 2], "2 or Heads")
    assert not _ok(["heads", 2], "tails, 2")
