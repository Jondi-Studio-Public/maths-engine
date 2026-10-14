"""`f10-choose-unit-and-convert`: topic `fr2c_choose_unit_convert` (easy, medium, hard) and the exam chain.

The medium and hard oracles convert every printed measure to the smallest unit with exact fractions.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from tests.fr2c_checks import check_follow_through, check_parts, check_single, skill

SKILL = "f10-choose-unit-and-convert"
TOPIC = "fr2c_choose_unit_convert"
EXAM = "fr2c_choose_unit_convert_exam"
TIERS = ("easy", "medium", "hard")
SMALLEST = {
    "mm": 1,
    "cm": 10,
    "m": 1000,
    "km": 1_000_000,
    "mg": 1,
    "g": 1000,
    "kg": 10**6,
    "t": 10**9,
    "mL": 1,
    "L": 1000,
}
MEASURE = re.compile(r"(\d+(?:\.\d+)?) (mm|cm|m|km|mg|g|kg|t|mL|L)\b")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _base(text):
    n, u = text.split()
    return Fraction(n) * SMALLEST[u]


def test_skill_wiring():
    sk = skill(SKILL)
    assert sk.answer == "mcq" and sk.status == "active"
    assert [(s.topic, s.difficulty) for s in sk.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert sk.sources[-1].stages == ["exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    check_single(SKILL, TOPIC, tier, min_options=3)


def test_easy_names_the_unit_and_offers_four_different_units():
    for seed in range(80):
        q = _q("easy", seed)
        assert len({q.answer_text, *q.distractors}) == 4
        assert q.answer_text.endswith(
            (
                "metres",
                "grams",
                "litres",
                "tonnes",
                "kilograms",
                "kilometres",
                "centimetres",
                "millimetres",
                "millilitres",
                "kilolitres",
                "milligrams",
            )
        )


def test_medium_key_is_the_extreme_measure_and_not_the_extreme_number():
    for seed in range(120):
        q = _q("medium", seed)
        measures = [m.group(0) for m in MEASURE.finditer(q.text.split(".")[0] + ".")] if False else None
        opts = [q.answer_text, *q.distractors]
        assert len(opts) == 4
        bases = {o: _base(o) for o in opts}
        word = re.search(r"(longest|shortest|heaviest|lightest|largest|smallest)", q.text).group(1)
        high = word in ("longest", "heaviest", "largest")
        best = max(bases, key=bases.get) if high else min(bases, key=bases.get)
        assert best == q.answer_text, (seed, q.text)
        numbers = {o: Fraction(o.split()[0]) for o in opts}
        extreme_number = max(numbers, key=numbers.get) if high else min(numbers, key=numbers.get)
        assert extreme_number != q.answer_text, (seed, q.text)
        assert all(o in q.text for o in opts)


def test_hard_total_or_remaining_in_the_smaller_unit():
    kinds = set()
    for seed in range(160):
        q = _q("hard", seed)
        text = q.text
        found = [(Fraction(n), u) for n, u in MEASURE.findall(text)]
        small_unit = q.answer_text.split()[1]
        scale = {"kg": 1000, "m": 100 if small_unit == "cm" else 1000, "L": 1000, "km": 1000, "g": 1000}
        big_unit = next(u for _n, u in found if u != small_unit)
        f = scale[big_unit]

        def small(n, u):
            return n * (f if u == big_unit else 1)

        values = [small(n, u) for n, u in found]
        if "is left" in text or "How much" in text:
            kinds.add("left")
            assert Fraction(q.answer_text.split()[0]) == values[0] - values[1] - values[2], text
        else:
            kinds.add("total")
            assert Fraction(q.answer_text.split()[0]) == sum(values), text
    assert kinds == {"left", "total"}


def test_exam_chain_shape_and_marks():
    check_parts(
        SKILL, EXAM, marks=[1, 2, 2, 2], kinds=["mcq", "number", "number", "number"], follow=[False, False, True, True]
    )


def test_exam_answers_recomputed_from_the_stem():
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        found = [(Fraction(n), u) for n, u in MEASURE.findall(q.text.split("(a)")[0])]
        small = next(u for _n, u in found if u in ("mL", "g", "cm"))
        big = next(u for _n, u in found if u not in ("mL", "g", "cm"))
        f = {"L": 1000, "kg": 1000, "m": 100}[big]
        total = sum(n * (f if u == big else 1) for n, u in found)
        parts = {p.label: p for p in q.parts}
        assert Fraction(parts["b"].answer_text) == total, q.text
        portion = int(re.search(r"uses (\d+) " + small, q.text).group(1))
        assert Fraction(parts["c"].answer_text) == total / portion
        used = int(re.search(r"makes (\d+) of", q.text).group(1))
        assert Fraction(parts["d"].answer_text) == total - used * portion


def test_exam_follow_through_from_a_wrong_total():
    def replace(q, parts):
        wrong = parts["b"].distractors[0]
        portion = int(re.search(r"uses (\d+) ", q.text).group(1))
        value = Fraction(wrong) / portion
        return (wrong, str(int(value + Fraction(1, 2)))) if value >= 1 else (None, None)

    check_follow_through(SKILL, EXAM, "b", replace, "c", seeds=range(60))
