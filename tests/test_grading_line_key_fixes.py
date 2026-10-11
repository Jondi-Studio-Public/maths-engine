"""Labelled one-part line keys ("y = 3", "x = -2") on the figure skills.

Real grade path (`mathgen.grading.grade("pair", ...)`) on questions made by the
pv1d/pv1e generators whose key is a single labelled equation.
"""

import pytest
from mathgen import make
from mathgen.grading import grade

CASES = [
    ("pv1d_parabola_axis_y", "y"),
    ("pv1d_parabola_axis_x", "x"),
    ("pv1e_exp_asymptote", "y"),
    ("pv1e_log_asymptote", "x"),
]


def _questions():
    for topic, label in CASES:
        for seed in range(6):
            q = make(topic, seed=seed, difficulty="easy")
            yield pytest.param(q, label, id=f"{topic}-{seed}")


def _value(q, label):
    return q.answer[label]


@pytest.mark.parametrize("q,label", list(_questions()))
def test_label_stripped_by_design_bare_value_accepted(q, label):
    v = _value(q, label)
    other = "x" if label == "y" else "y"
    assert grade("pair", q, f"{label} = {v}").correct
    # The label is stripped by design (documented in the two pv1d tests): same as bare.
    assert grade("pair", q, f"{other} = {v}").correct
    # A bare value is accepted by design (documented in test_pv1d_parabola_build.py and
    # test_pv1d_sideways_circle_build.py).
    assert grade("pair", q, f"{v}").correct


@pytest.mark.parametrize("q,label", list(_questions()))
def test_unicode_minus_equivalent(q, label):
    v = _value(q, label)
    given = f"{label} = {v}".replace("-", "−")
    assert grade("pair", q, given).correct
    assert grade("pair", q, f"{label}={v}".replace("-", "\u2212")).correct


@pytest.mark.parametrize("q,label", list(_questions()))
def test_equivalent_line_forms_accepted(q, label):
    v = _value(q, label)
    other = "x" if label == "y" else "y"
    sign = "-" if v >= 0 else "+"
    assert grade("pair", q, f"{label} {sign} {abs(v)} = 0").correct
    assert grade("pair", q, f"{v} = {label}").correct
    assert grade("pair", q, f"{label} = +{v}" if v >= 0 else f"{label} = {v}").correct
    # still wrong
    assert not grade("pair", q, f"{other} {sign} {abs(v)} = 0").correct
    assert not grade("pair", q, f"{label} = {v + 1}").correct
    assert not grade("pair", q, f"{label} {sign} {abs(v) + 1} = 0").correct


def test_unicode_minus_on_two_part_key():
    q = make("pv1e_hyperbola_asymptotes", seed=0, difficulty="easy")
    assert "-" in q.answer_text
    assert grade("pair", q, q.answer_text.replace("-", "−")).correct
    assert not grade("pair", q, q.answer_text.replace("-", "−").replace("y = ", "y = 9")).correct


class _Key:
    """The two fields the pair grader reads from a one-equation key."""

    def __init__(self, label, value):
        self.answer = {label: value}
        self.answer_text = f"{label} = {value}"


@pytest.mark.parametrize("given", ["lambda = 17", "17 = lambda", "+lambda = 17"])
def test_a_keyword_label_keeps_the_original_verdict(given):
    # `lambda` is a Python keyword, so the expression parser cannot read it. The
    # rearrangement branch must stay out of the way instead of rejecting.
    assert grade("pair", _Key("lambda", 17), given).correct


@pytest.mark.parametrize("given", ["lambda = 16", "16 = lambda", "+lambda = 18"])
def test_a_keyword_label_still_rejects_wrong_values(given):
    assert not grade("pair", _Key("lambda", 17), given).correct
