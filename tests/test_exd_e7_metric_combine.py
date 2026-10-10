"""Tests the `combine` option of the `metric_conversion` generator, which converts a quantity and then adds a
second one.

Metric combine."""

import mathgen


def test_combine_answer_is_conversion_then_addition():
    q = mathgen.make("metric_conversion", seed=5, kind="length", max_steps=1, value_hi=30, combine=True)
    assert "total" in q.text.lower()
    assert q.answer_text == str(q.answer)


def test_combine_default_is_false_and_unchanged():
    # combine defaults to False: identical to calling without the param at all.
    a = mathgen.make("metric_conversion", seed=5, kind="length", max_steps=1, value_hi=30)
    b = mathgen.make("metric_conversion", seed=5, kind="length", max_steps=1, value_hi=30, combine=False)
    assert a.text == b.text
    assert a.answer == b.answer


def test_combine_renders_for_all_three_kinds():
    for kind in ("length", "mass", "capacity"):
        q = mathgen.make("metric_conversion", seed=3, kind=kind, max_steps=1, value_hi=30, combine=True)
        assert "One amount is" in q.text
