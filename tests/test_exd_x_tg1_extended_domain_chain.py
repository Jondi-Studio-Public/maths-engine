"""Tests for the multi-part exam source `exd_x_tg1_extended_domain_chain` (senior-recognise-the-graphs-of-y-sin-x-y).

The oracle reads the function and the domain from the rendered stem, then counts features of the real curve by scanning
it on a fine grid (sign changes, slope changes, jumps); it never imports the generator module or uses residue classes.
"""

from __future__ import annotations

import collections
import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_tg1_extended_domain_chain"
SKILL_ID = "senior-recognise-the-graphs-of-y-sin-x-y"
SEEDS = range(300)
STEPS_PER_PI = 1200

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _pi(text):
    if text == "0":
        return Fraction(0)
    m = re.fullmatch(r"(-?)(\d*)π(?:/(\d+))?", text)
    sign = -1 if m.group(1) else 1
    return sign * Fraction(int(m.group(2) or 1), int(m.group(3) or 1))


def read(q):
    stem = q.stem
    if "y = sin x" in stem or "greatest value of 1" in stem:
        name = "sin"
    elif "y = cos x" in stem or "crosses the y-axis" in stem:
        name = "cos"
    else:
        assert "y = tan x" in stem or "asymptote at x" in stem
        name = "tan"
    lo, hi = (_pi(t) for t in re.search(r"(0|-?\d*π(?:/\d+)?) ≤ x ≤ (0|-?\d*π(?:/\d+)?)", stem).groups())
    return name, lo, hi


def scan(name, lo, hi):
    """Counts from a fine grid. Grid index i is x = i pi / STEPS_PER_PI; every feature lies on it."""
    i0, i1 = int(lo * STEPS_PER_PI), int(hi * STEPS_PER_PI)
    xs = [i * math.pi / STEPS_PER_PI for i in range(i0, i1 + 1)]
    f = getattr(math, name)
    ys = [f(x) for x in xs]
    near = lambda v, t: abs(v - t) < 1e-7  # noqa: E731
    if name == "tan":
        # asymptotes: where cosine is zero; tan blows up there so the grid straddles the jump
        asym = sum(1 for x in xs if abs(math.cos(x)) < 1e-9)
        zeros = sum(1 for y, x in zip(ys, xs) if abs(math.cos(x)) > 1e-3 and near(y, 0))
        hits = [i0 + k for k, (y, x) in enumerate(zip(ys, xs)) if abs(math.cos(x)) > 1e-3 and near(y, 1)]
        one_sum = sum(Fraction(4 * i, STEPS_PER_PI) for i in hits)  # in units of pi/4
        return dict(zeros=zeros, asym=asym, one=len(hits), pieces=asym + 1, sum=one_sum)
    zeros = sum(1 for y in ys if near(y, 0))
    turning = [k for k in range(1, len(ys) - 1) if (ys[k] - ys[k - 1]) * (ys[k + 1] - ys[k]) < 0]
    maxima = sum(1 for k in turning if ys[k] > 0)
    max_sum = sum(Fraction(2 * (i0 + k), STEPS_PER_PI) for k in turning if ys[k] > 0)  # in units of pi/2
    half = sum(1 for y in ys if near(y, 0.5))
    return dict(zeros=zeros, turning=len(turning), maxima=maxima, half=half, sum=max_sum)


def oracle(q):
    name, lo, hi = read(q)
    c = scan(name, lo, hi)
    if name == "tan":
        return {
            "a": str(c["zeros"]),
            "b": str(c["asym"]),
            "c": str(c["pieces"]),
            "d": str(c["one"]),
            "e": str(c["sum"]),
        }
    return {
        "a": str(c["zeros"]),
        "b": str(c["turning"]),
        "c": str(c["maxima"]),
        "d": str(c["half"]),
        "e": str(c["sum"]),
    }


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_is_wired_as_the_second_exam_source():
    sources = _SKILL.sources
    assert [s.difficulty for s in sources] == ["easy", "medium", "hard", "exam", "exam"]
    last = sources[-1]
    assert last.topic == TOPIC and last.key == "exam-single" and list(last.stages) == ["exam"]
    assert last.params == {"tier": "exam"}
    assert sources[-2].topic == "pv1f_trig_graph_id"


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="easy")


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == list("abcde")
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False, False]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_grid_scan_oracle_agrees_300_seeds(questions):
    bad = [(s, q.text) for s, q in zip(SEEDS, questions) if oracle(q) != _canonical(q)]
    assert bad == []


def test_all_three_functions_and_stem_styles_occur(questions):
    assert {read(q)[0] for q in questions} == {"sin", "cos", "tan"}
    styles = {
        "named" if "The graph of" in q.stem else "interval" if "Consider" in q.stem else "described" for q in questions
    }
    assert styles == {"named", "interval", "described"}


def test_domains_are_extended_and_include_half_multiples(questions):
    spans = [read(q)[2] - read(q)[1] for q in questions]
    assert min(spans) >= 2 and max(spans) >= 5
    ends = {read(q)[1].denominator for q in questions}
    assert ends >= {1, 2, 4}


def test_answer_space_and_prompt_shapes(questions):
    answers = collections.Counter(q.answer_text for q in questions)
    assert len(answers) >= 100 and max(answers.values()) <= 0.1 * len(questions)
    assert len({prompt_shape(q) for q in questions[:60]}) >= 3


def test_key_accepted_and_distractors_rejected(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
                assert p.option_notes.get(d) is None or p.option_notes[d] in CATALOGUE


def test_distractors_are_tagged_where_an_error_exists(questions):
    tagged = [d for q in questions[:100] for p in q.parts for d in p.distractors if d in p.option_notes]
    assert len(tagged) >= 300


def test_follow_through_from_a_wrong_part_b(questions):
    checked = 0
    for q in questions:
        canon = _canonical(q)
        wrong_b = int(canon["b"]) + 2  # same parity: the count of maxima stays a whole-number function of b
        name = read(q)[0]
        if name == "tan":
            carried = wrong_b + 1
        else:
            b_true, c_true = int(canon["b"]), int(canon["c"])
            first_is_max = b_true % 2 == 1 and c_true * 2 > b_true
            carried = (wrong_b + 1) // 2 if first_is_max else wrong_b // 2
        part_c = q.parts[2]
        if (
            str(carried) in part_c.distractors
            or str(carried) == canon["c"]
            or str(carried) in (canon["a"], str(wrong_b))
        ):
            continue
        result = grade(_SKILL, q, {**canon, "b": str(wrong_b), "c": str(carried)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == part_c.marks, (q.text, wrong_b, carried)
        assert result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 40  # many carried values coincide with a listed wrong option or an earlier reply and are skipped


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: "999" for p in q.parts})
        assert result.marks == 0 and not result.correct


def test_text_is_clean_and_deterministic(questions):
    for q in questions[:100]:
        assert find_text_defects(q) == [], q.text
    assert _exam(7).to_json() == _exam(7).to_json()


def test_no_wrong_option_is_impossible_for_what_the_part_asks(questions):
    for q in questions:
        a, b, c, d, e = q.parts
        name = read(q)[0]
        # maxima among alternating turning points: never all of them, and never more than the turning points
        if name != "tan":
            assert all(int(x) < int(b.answer_text) for x in c.distractors), (q.text, c.distractors)
        else:
            # pieces = asymptotes + 1, so the asymptote count itself is never a piece count worth offering
            assert b.answer_text not in c.distractors, (q.text, c.distractors)
        for p in q.parts[:4]:
            assert all(int(x) > 0 for x in p.distractors)
