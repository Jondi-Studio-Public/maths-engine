"""Exam tier of `senior-review-the-conditions-for-similarity-of-two-dimensional`: use a condition, then the scale factor."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_sc6_support import (
    assert_answer_space,
    assert_follow_through,
    assert_grading,
    assert_new_shapes,
    assert_no_text_defects,
    assert_source_wired,
    canonical,
    exam,
    frac,
    skill_of,
)

SKILL = skill_of("senior-review-the-conditions-for-similarity-of-two-dimensional")
TOPIC = "exd_x_sc6_similarity"


def _dec(value: Fraction) -> str:
    from decimal import Decimal

    text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _lengths(stem: str) -> dict[str, Fraction]:
    return {m.group(1): Fraction(m.group(2)) for m in re.finditer(r"\b([A-Z]{2}) = (\d+(?:\.\d+)?) cm", stem)}


def shape_of(stem: str) -> str:
    return "sas" if "SAS" in stem else "sss" if "SSS" in stem else "aa"


def oracle(q) -> dict[str, str]:
    stem = q.stem
    sides = _lengths(stem)
    kind = shape_of(stem)
    if kind == "sas":
        n1, n2 = re.match(r"Triangles ([A-Z]{3}) and ([A-Z]{3}) ", stem).groups()
        small1, small2 = sides[n1[0] + n1[1]], sides[n1[0] + n1[2]]
        big1 = sides[n2[0] + n2[1]]
        area = Fraction(re.search(r"area of (\d+) cm²", stem).group(1))
        x = big1 * small2 / small1
        k = big1 / small1
        return {"a": _dec(x), "b": _dec(k), "c": _dec(area * k * k)}
    if kind == "sss":
        n1, n2 = re.search(r"Triangle ([A-Z]{3}) has sides .*? Triangle ([A-Z]{3}) has", stem).groups()
        first = [sides[n1[i] + n1[(i + 1) % 3]] for i in range(3)]
        second = {n2[i] + n2[(i + 1) % 3]: i for i in range(3)}
        known = {second[name]: v for name, v in sides.items() if name in second}
        j = next(i for i in range(3) if i not in known)
        i0 = next(iter(known))
        k = known[i0] / first[i0]
        y = k * first[j]
        return {"a": _dec(y), "b": _dec(k), "c": _dec(k * sum(first))}
    # aa
    a_ang, b_ang = (int(v) for v in re.findall(r"= (\d+)°", stem)[:2])
    n1, n2 = re.search(r"In triangle ([A-Z]{3}), .*? In triangle ([A-Z]{3}),", stem).groups()
    ab, bc, de = sides[n1[:2]], sides[n1[1:]], sides[n2[:2]]
    k = de / ab
    return {"a": str(180 - a_ang - b_ang), "b": _dec(k), "c": _dec(k * bc)}


def test_source_wired():
    assert_source_wired(SKILL, TOPIC)


def test_shapes_and_text():
    assert_new_shapes(SKILL)
    assert_no_text_defects(SKILL)


def test_answer_space():
    assert_answer_space(SKILL, minimum=100)


def test_all_three_shapes_occur():
    assert {shape_of(exam(SKILL, s).stem) for s in range(60)} == {"sas", "sss", "aa"}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(SKILL, seed)
        assert {k: frac(v) for k, v in oracle(q).items()} == {k: frac(v) for k, v in canonical(q).items()}, (
            seed,
            q.stem,
        )


def test_grader_accepts_key_rejects_distractors():
    assert_grading(SKILL)


def _r2(value: Fraction) -> str:
    """`value` rounded half up to 2 decimal places, as a learner would state it."""
    scaled = int(value * 100 + Fraction(1, 2))
    return f"{scaled // 100}.{scaled % 100:02d}"


def test_follow_through_into_scale_factor_and_last_part():
    checked = 0
    for seed in range(150):
        q = exam(SKILL, seed)
        kind = shape_of(q.stem)
        key = canonical(q)
        sides = _lengths(q.stem)
        if kind == "aa":
            # (c) follows (b): a wrong scale factor carries into the side length.
            n1 = re.search(r"In triangle ([A-Z]{3}),", q.stem).group(1)
            bc = sides[n1[1:]]
            wrong_b = q.parts[1].distractors[0]
            replies = {**key, "b": wrong_b, "c": _r2(frac(wrong_b) * bc)}
            assert_follow_through(SKILL, q, replies, "b")
        else:
            wrong_a = q.parts[0].distractors[0]
            if kind == "sas":
                n1 = re.match(r"Triangles ([A-Z]{3}) ", q.stem).group(1)
                k = frac(wrong_a) / sides[n1[0] + n1[2]]
                area = Fraction(re.search(r"area of (\d+) cm²", q.stem).group(1))
                c = area * _r2_value(k) ** 2
            else:
                n1 = re.search(r"Triangle ([A-Z]{3}) has sides", q.stem).group(1)
                n2 = re.search(r"Triangle ([A-Z]{3}) has [A-Z]", q.stem).group(1)
                first = [sides[n1[i] + n1[(i + 1) % 3]] for i in range(3)]
                target = re.search(r"length of ([A-Z]{2})", q.parts[0].prompt).group(1)
                j = [n2[i] + n2[(i + 1) % 3] for i in range(3)].index(target)
                k = frac(wrong_a) / first[j]
                c = _r2_value(k) * sum(first)
            replies = {"a": wrong_a, "b": _r2(k), "c": _r2(c)}
            assert_follow_through(SKILL, q, replies, "a")
        checked += 1
    assert checked == 150


def _r2_value(value: Fraction) -> Fraction:
    return Fraction(_r2(value))


def test_all_wrong_earns_nothing():
    for seed in range(30):
        result = grade(SKILL, exam(SKILL, seed), {"a": "999", "b": "998", "c": "997"})
        assert result.marks == 0


def test_deterministic():
    assert exam(SKILL, 7).to_dict() == exam(SKILL, 7).to_dict()
