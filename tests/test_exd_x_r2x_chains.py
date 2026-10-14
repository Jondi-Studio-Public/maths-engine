"""Tests the six round-two exam chains added to existing skills (`exd_x_r2x_*`).

Each chain has an independent oracle that re-derives every part from the rendered stem and prompts, follow-through
cases that start from a wrong earlier answer, and the shared grading, text and answer-space checks.
"""

from __future__ import annotations

import itertools
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest
import sympy

from mathgen.engine import grade
from mathgen.text_quality import find_text_defects
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_follow_through,
    check_grading,
    exam,
    skill,
)

N = 300


def _fmt(value, places: int) -> str:
    value = Fraction(value)
    q = Decimal(value.numerator) / Decimal(value.denominator)
    return str(q.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)) if places else str(int(round(value)))


def _places(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _frac(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _dec(text: str) -> Fraction:
    return Fraction(Decimal(text.replace(",", "")))


def _wired(skill_id: str, topic: str) -> None:
    sources = [s for s in skill(skill_id).sources if s.topic == topic]
    assert len(sources) == 1
    assert sources[0].difficulty == "exam" and list(sources[0].stages) == ["exam"]
    # The skill keeps its other sources, including the earlier tiers.
    assert {"easy", "medium", "hard"} <= {s.difficulty for s in skill(skill_id).sources} or skill_id.startswith(
        "senior-review"
    )


def _generic(skill_id: str, topic: str, parts: int) -> None:
    for seed in range(60):
        q = exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam(topic, seed).to_json()
        assert len(q.parts) == parts
        assert q.parts[0].follow_through is None
        assert sum(1 for p in q.parts if p.follow_through) >= 2
    check_grading(skill_id, topic)


# --------------------------------------------------------------------------------------------------------------------
# f10-conditional-probability-language

COND_TOPIC, COND_ID = "exd_x_r2x_conditional_probability_language", "f10-conditional-probability-language"


def _cond_oracle(q):
    from mathgen.generators.exd_x_r2x_conditional_probability_language import _SCENARIOS

    body = q.stem.split("Of them, ")[1].split(". (a)")[0] if ". (a)" in q.stem else q.stem.split("Of them, ")[1]
    cells = [int(x) for x in re.findall(r"(?:^|; |; and )(\d+) ", body)]
    assert len(cells) == 4, body
    prompts = {p.label: p.prompt for p in q.parts}
    sc = next(s for s in _SCENARIOS if s["noun"] in prompts["a"])
    total = int(re.search(r"of the (\d+)", prompts["a"]).group(1))
    assert total == sum(cells)
    cond_phrase = re.search(rf"{total} {sc['noun']} (.+?)\?", prompts["a"]).group(1)
    row = [r[0] for r in sc["rows"]].index(cond_phrase)
    ask = prompts["c"].split("Give the answer")[0]
    # "does not use the sauna" contains "use the sauna" only as a plural form, so test the singular phrases longest first.
    col = next(j for j, c in sorted(enumerate(sc["cols"]), key=lambda jc: -len(jc[1][1])) if c[1] + "?" in ask)
    n = [[cells[0], cells[1]], [cells[2], cells[3]]]
    group = sum(n[row])
    cell = n[row][col]
    target_total = n[0][col] + n[1][col]
    cond = Fraction(cell, group)
    plain = Fraction(target_total, total)
    return {
        "a": str(group),
        "b": _frac(cond),
        "c": _frac(plain),
        "d": _fmt(100 * (cond - plain), 1),
    }


def test_conditional_wired():
    _wired(COND_ID, COND_TOPIC)


def test_conditional_rederived():
    for seed in range(N):
        q = exam(COND_TOPIC, seed)
        assert _cond_oracle(q) == canonical(q), (seed, q.text)


def test_conditional_generic():
    _generic(COND_ID, COND_TOPIC, 4)


def test_conditional_answer_space():
    check_answer_space(COND_TOPIC, distinct=100, share=0.05)


def test_conditional_independent_and_dependent_both_occur():
    changes = [canonical(exam(COND_TOPIC, s))["d"] for s in range(N)]
    assert 0.15 < changes.count("0.0") / N < 0.45
    assert any(c.startswith("-") for c in changes) and any(c not in ("0.0",) and not c.startswith("-") for c in changes)


def test_conditional_follow_through():
    def carried(q):
        wrong = q.parts[0].distractors[0]
        cell = int(q.parts[1].follow_through.split("/")[0])
        return "a", wrong, {"b": _frac(Fraction(cell, int(wrong)))}

    check_follow_through(COND_ID, COND_TOPIC, carried)


# --------------------------------------------------------------------------------------------------------------------
# senior-review-metric-units-of-length-their-abbreviations-conversions

FENCE_TOPIC = "exd_x_r2x_metric_length_fencing"
FENCE_ID = "senior-review-metric-units-of-length-their-abbreviations-conversions"
_TO_M = {"km": Fraction(1000), "m": Fraction(1), "cm": Fraction(1, 100)}


def _fence_inputs(q):
    lengths = [(_dec(v), u) for v, u in re.findall(r"(\d+(?:\.\d+)?) (km|cm|m)\b", q.stem)]
    (v1, u1), (v2, u2), (vg, ug) = lengths[:3]
    rate = int(re.search(r"\$(\d+) per metre", q.stem).group(1))
    note = int(re.findall(r"\$([\d,]+)", q.stem)[-1].replace(",", ""))
    converted = next(v * _TO_M[u] for v, u in ((v1, u1), (v2, u2)) if u != "m")
    other = next(v for v, u in ((v1, u1), (v2, u2)) if u == "m")
    return converted, other, vg * _TO_M[ug], rate, note


def _fence_oracle(q):
    side, other, gate, rate, note = _fence_inputs(q)
    perimeter = 2 * (side + other)
    cost = rate * (perimeter - gate)
    places = _places(q.parts[0].answer_text)
    return {
        "a": _fmt(side, places),
        "b": _fmt(perimeter, places),
        "c": _fmt(cost, 2),
        "d": _fmt(note - cost, 2),
    }


def test_fence_wired():
    _wired(FENCE_ID, FENCE_TOPIC)


def test_fence_rederived():
    for seed in range(N):
        q = exam(FENCE_TOPIC, seed)
        assert _fence_oracle(q) == canonical(q), (seed, q.text)


def test_fence_generic():
    _generic(FENCE_ID, FENCE_TOPIC, 4)


def test_fence_answer_space_and_both_unit_pairs():
    check_answer_space(FENCE_TOPIC, distinct=100, share=0.05)
    stems = [exam(FENCE_TOPIC, s).stem for s in range(60)]
    assert any(" km " in s for s in stems) and any(" cm " in s for s in stems)


def test_fence_follow_through():
    def carried(q):
        wrong = q.parts[0].distractors[0]
        side, other, gate, rate, note = _fence_inputs(q)
        a = _dec(wrong)
        b = 2 * (a + other)
        c = rate * (b - gate)
        later = {"b": _fmt(b, _places(q.parts[1].answer_text))}
        if b > 0 and c > 0:
            later["c"] = _fmt(c, 2)
            if note - c > 0:
                later["d"] = _fmt(note - c, 2)
        return "a", wrong, later

    check_follow_through(FENCE_ID, FENCE_TOPIC, carried)


# --------------------------------------------------------------------------------------------------------------------
# solve-quadratics-factorable

QUAD_TOPIC, QUAD_ID = "exd_x_r2x_quadratic_nonmonic_roots", "solve-quadratics-factorable"
x = sympy.Symbol("x")


def _quad_poly(q):
    text = re.search(r"y = (.+?), where", q.stem).group(1).replace("^", "**")
    text = re.sub(r"(\d)x", r"\1*x", text)
    text = re.sub(r"(?<![\d*])-x", "-1*x", text)
    return sympy.sympify(text, locals={"x": x})


def _quad_oracle(q):
    poly = _quad_poly(q)
    roots = sorted(sympy.solve(poly, x))
    span = roots[1] - roots[0]
    xv = (roots[0] + roots[1]) / 2
    height = poly.subs(x, xv)
    return {
        "a": _fmt(Fraction(str(roots[0])), 2),
        "b": _fmt(Fraction(str(roots[1])), 2),
        "c": _fmt(Fraction(str(span)), 2),
        "d": _fmt(Fraction(str(height)), 2),
    }


def test_quad_wired():
    _wired(QUAD_ID, QUAD_TOPIC)


def test_quad_rederived():
    for seed in range(N):
        q = exam(QUAD_TOPIC, seed)
        assert _quad_oracle(q) == canonical(q), (seed, q.text)


def test_quad_generic():
    _generic(QUAD_ID, QUAD_TOPIC, 4)


def test_quad_answer_space_and_leading_coefficient_not_monic():
    check_answer_space(QUAD_TOPIC, distinct=100, share=0.05)
    for seed in range(60):
        lead = _quad_poly(exam(QUAD_TOPIC, seed)).coeff(x, 2)
        assert lead <= -2


def test_quad_follow_through():
    def carried(q):
        hi = _dec(q.parts[1].answer_text)
        lead = -_quad_poly(q).coeff(x, 2)
        for wrong in q.parts[0].distractors:
            span = hi - _dec(wrong)
            height = lead * span * span / 4
            # A span of zero or less, or a height that just repeats the span, carries no follow-through.
            if span > 0 and height != span:
                return "a", wrong, {"c": _fmt(span, 2), "d": _fmt(height, 2)}
        raise AssertionError(q.parts[0].distractors)

    check_follow_through(QUAD_ID, QUAD_TOPIC, carried)


# --------------------------------------------------------------------------------------------------------------------
# f10-direct-proportion-scale-model

PROP_TOPIC, PROP_ID = "exd_x_r2x_direct_proportion_table", "f10-direct-proportion-scale-model"


def _prop_inputs(q):
    nums = [_dec(v) for v in re.findall(r"(\d+(?:\.\d+)?)", q.stem)]
    pairs = list(zip(nums[0:6:2], nums[1:6:2]))
    xn = _dec(re.search(r"(\d+(?:\.\d+)?)", q.parts[1].prompt).group(1))
    delta = _dec(re.search(r"(\d+(?:\.\d+)?)", q.parts[2].prompt).group(1))
    return pairs, xn, delta


def _prop_oracle(q):
    pairs, xn, delta = _prop_inputs(q)
    ratios = {y / xv for xv, y in pairs}
    assert len(ratios) == 1, pairs
    k = ratios.pop()
    return {"a": _fmt(k, 2), "b": _fmt(k * xn, 2), "c": _fmt((k * xn + delta) / k, 0)}


def test_prop_wired():
    _wired(PROP_ID, PROP_TOPIC)


def test_prop_rederived():
    for seed in range(N):
        q = exam(PROP_TOPIC, seed)
        assert _prop_oracle(q) == canonical(q), (seed, q.text)


def test_prop_generic():
    _generic(PROP_ID, PROP_TOPIC, 3)


def test_prop_answer_space():
    check_answer_space(PROP_TOPIC, distinct=100, share=0.05)


def test_prop_follow_through():
    def carried(q):
        wrong = q.parts[0].distractors[0]
        _, xn, delta = _prop_inputs(q)
        k = _dec(wrong)
        b = k * xn
        later = {"b": _fmt(b, 2)}
        x_total = (b + delta) / k
        if x_total.denominator == 1 and b > 0:
            later["c"] = str(int(x_total))
        return "a", wrong, later

    check_follow_through(PROP_ID, PROP_TOPIC, carried)


# --------------------------------------------------------------------------------------------------------------------
# f10-compound-outcomes-replacement

COMP_TOPIC, COMP_ID = "exd_x_r2x_compound_outcomes_replacement", "f10-compound-outcomes-replacement"


def _comp_counts(q):
    a, _, b, _ = re.search(r"(\d+) (\w+) (?:counters|jelly beans|marbles) and (\d+) (\w+)", q.stem).groups()
    return int(a), int(b)


def _comp_oracle(q):
    a, b = _comp_counts(q)
    n = a + b
    balls = ["A"] * a + ["B"] * b
    draws = list(itertools.product(balls, repeat=2))
    ordered = list(itertools.permutations(range(n), 2))

    def prob(pool, event):
        return Fraction(sum(1 for d in pool if event(d)), len(pool))

    ab = prob(draws, lambda d: d == ("A", "B"))
    one = prob(draws, lambda d: d.count("A") == 1)
    any_a = prob(draws, lambda d: "A" in d)
    any_wo = prob(ordered, lambda d: "A" in (balls[d[0]], balls[d[1]]))
    return {"a": _frac(ab), "b": _frac(one), "c": _frac(any_a), "d": _frac(any_wo - any_a)}


def test_comp_wired():
    _wired(COMP_ID, COMP_TOPIC)


def test_comp_rederived():
    for seed in range(N):
        q = exam(COMP_TOPIC, seed)
        assert _comp_oracle(q) == canonical(q), (seed, q.text)


def test_comp_generic():
    _generic(COMP_ID, COMP_TOPIC, 4)


def test_comp_answer_space():
    check_answer_space(COMP_TOPIC, distinct=100, share=0.05)


def test_comp_follow_through():
    def carried(q):
        wrong = q.parts[0].distractors[0]
        a, b = _comp_counts(q)
        n = a + b
        b_val = 2 * Fraction(wrong)
        later = {"b": _frac(b_val)}
        if b_val < 1:
            c_val = b_val + Fraction(a * a, n * n)
            later["c"] = _frac(c_val)
        return "a", wrong, later

    check_follow_through(COMP_ID, COMP_TOPIC, carried)


# --------------------------------------------------------------------------------------------------------------------
# f10-boxplot-distribution

BOX_TOPIC, BOX_ID = "exd_x_r2x_boxplot_two_groups", "f10-boxplot-distribution"


def _box_groups(q):
    a = [int(v) for v in re.search(r"Group A: ([\d, ]+)\.", q.stem).group(1).split(", ")]
    b = [int(v) for v in re.search(r"Group B: ([\d, ]+)\.", q.stem).group(1).split(", ")]
    return a, b


def _halves(values):
    v = sorted(values)
    h = len(v) // 2
    return v[:h], v[h + len(v) % 2 :]


def _med(v):
    v = sorted(v)
    m = len(v) // 2
    return Fraction(v[m]) if len(v) % 2 else Fraction(v[m - 1] + v[m], 2)


def _box_oracle(q):
    a, b = _box_groups(q)
    lo, up = _halves(a)
    q1, q3 = _med(lo), _med(up)
    fence = q3 + Fraction(3, 2) * (q3 - q1)
    outliers = [v for v in a if v > fence]
    assert len(outliers) == 1
    rest = [v for v in a if v != outliers[0]]
    lo_b, up_b = _halves(b)
    iqr_b = _med(up_b) - _med(lo_b)
    return {
        "a": _fmt(q1, 1),
        "b": _fmt(q3, 1),
        "c": _fmt(fence, 1),
        "d": _fmt(Fraction(sum(rest), len(rest)), 1),
        "e": _fmt(abs(iqr_b - (q3 - q1)), 1),
    }


def test_box_wired():
    _wired(BOX_ID, BOX_TOPIC)


def test_box_rederived():
    for seed in range(N):
        q = exam(BOX_TOPIC, seed)
        assert _box_oracle(q) == canonical(q), (seed, q.text)


def test_box_generic():
    _generic(BOX_ID, BOX_TOPIC, 5)


def test_box_answer_space():
    check_answer_space(BOX_TOPIC, distinct=100, share=0.05)


def test_box_follow_through():
    def carried(q):
        a, b = _box_groups(q)
        q3 = _dec(q.parts[1].answer_text)
        lo_b, up_b = _halves(b)
        iqr_b = _med(up_b) - _med(lo_b)
        for wrong in q.parts[0].distractors:
            iqr_wrong = q3 - _dec(wrong)
            if iqr_wrong > 0:  # a zero or negative spread carries no follow-through
                return (
                    "a",
                    wrong,
                    {
                        "c": _fmt(q3 + Fraction(3, 2) * iqr_wrong, 1),
                        "e": _fmt(abs(iqr_b - iqr_wrong), 1),
                    },
                )
        raise AssertionError(q.parts[0].distractors)

    check_follow_through(BOX_ID, BOX_TOPIC, carried)


@pytest.mark.parametrize(
    "skill_id, topic",
    [
        (COND_ID, COND_TOPIC),
        (FENCE_ID, FENCE_TOPIC),
        (QUAD_ID, QUAD_TOPIC),
        (PROP_ID, PROP_TOPIC),
        (COMP_ID, COMP_TOPIC),
        (BOX_ID, BOX_TOPIC),
    ],
)
def test_existing_sources_kept_and_composed_answer_graded(skill_id, topic):
    sk = skill(skill_id)
    assert any(s.topic != topic for s in sk.sources if s.difficulty == "hard") or skill_id == FENCE_ID
    q = exam(topic, 3)
    result = grade(sk, q, q.answer_text)
    assert result.correct and result.marks == result.max_marks
