"""Exam tier of `significant-figures`: the area of a rectangle from two measurements, reported to the data's precision."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_n4_sig_fig_product"
SKILL = next(s for s in load_skills()[1] if s.id == "significant-figures")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _sig_figs(text: str) -> int:
    return len(text.replace(".", "").lstrip("0"))


def oracle(q):
    """Re-derive all three answers from the two measurements in the prompt, using decimal arithmetic."""
    length, width = re.search(r"length is ([\d.]+) \w+ and its width is ([\d.]+) ", q.stem).groups()
    product = Decimal(length) * Decimal(width)
    k = min(_sig_figs(length), _sig_figs(width))
    unit = Decimal(1).scaleb(product.adjusted() - k + 1)
    rounded = product.quantize(unit, rounding=ROUND_HALF_UP)
    return {"a": format(product, "f"), "b": str(k), "c": format(rounded, "f")}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_prompt_shapes_are_new():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("significant_figures", seed=s, variant="large_trailing_zero")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_the_two_measurements_differ_in_precision():
    for seed in range(100):
        length, width = re.search(r"length is ([\d.]+) \w+ and its width is ([\d.]+) ", _exam(seed).stem).groups()
        assert _sig_figs(length) != _sig_figs(width)


def test_shape():
    q = _exam(0)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [bool(p.follow_through) for p in q.parts] == [False, False, True]
    assert all(p.answer_kind == "number" and len(p.distractors) >= 2 for p in q.parts)


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_key_accepted_and_distractors_rejected():
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        canon = _canonical(q)
        full = grade(SKILL, q, canon)
        assert full.correct and full.marks == full.max_marks == 4, (seed, q.text)
        for part in q.parts:
            for d in part.distractors:
                shown += 1
                result = grade(SKILL, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                assert part.option_notes.get(d)
    assert shown >= 600


def _places(text: str) -> int:
    return len(text.partition(".")[2])


def test_follow_through_from_wrong_significant_figure_count():
    """A learner who picks the wrong count and then rounds their own product to it earns part (c).

    Follow-through restates its target to the canonical answer's decimal places, so it is checked here where the
    wrong count leaves the answer on the same decimal places as the key."""
    carried = 0
    for seed in range(200):
        q = _exam(seed)
        right = oracle(q)
        product = Decimal(right["a"])
        for wrong_k in (int(d) for d in q.parts[1].distractors):
            if wrong_k < 1:
                continue
            unit = Decimal(1).scaleb(product.adjusted() - wrong_k + 1)
            wrong_c = format(product.quantize(unit, rounding=ROUND_HALF_UP), "f")
            result = grade(SKILL, q, {"a": right["a"], "b": str(wrong_k), "c": wrong_c})
            assert result.parts["b"].marks == 0
            if wrong_c != right["c"] and _places(wrong_c) == _places(right["c"]):
                carried += 1
                assert result.parts["c"].marks == q.parts[2].marks, (seed, q.text, wrong_k, wrong_c)
                assert result.parts["c"].follow_through
    assert carried >= 5


def test_follow_through_from_a_decimal_point_slip():
    carried = 0
    for seed in range(60):
        q = _exam(seed)
        right = oracle(q)
        slipped = Decimal(right["a"]) * 10
        k = int(right["b"])
        unit = Decimal(1).scaleb(slipped.adjusted() - k + 1)
        wrong_c = format(slipped.quantize(unit, rounding=ROUND_HALF_UP), "f")
        result = grade(SKILL, q, {"a": format(slipped, "f"), "b": right["b"], "c": wrong_c})
        assert result.parts["a"].marks == 0
        carried += result.parts["c"].marks == q.parts[2].marks
    assert carried >= 50


def test_all_wrong_earns_nothing():
    result = grade(SKILL, _exam(3), {"a": "9999", "b": "9", "c": "8888"})
    assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
