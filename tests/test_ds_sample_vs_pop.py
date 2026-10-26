"""Tests `f10-random-versus-convenience-sample` (AC9M8ST02).

Every key is re-derived from the rendered stem (the table and the two sample lines are parsed back out of the text and
the asked quantity is recomputed), every distractor must grade wrong through `mathgen.engine.grade`, and every exam
chain is marked part by part with follow-through.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.ds_sample_vs_pop_build as gen
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SKILL = "f10-random-versus-convenience-sample"
SKILLS = {s.id: s for s in load_skills()[1]}
SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

# the verb phrase of each categorical target, back to its category
WHO = {who: cat for ctx in gen._PROP_CTX for cat, (who, _, _) in ctx["targets"].items()}


@functools.lru_cache(maxsize=None)
def single(tier: str, seed: int):
    src = next(s for s in SKILLS[SKILL].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=tier, **src.params)


@functools.lru_cache(maxsize=None)
def exam(seed: int):
    src = next(s for s in SKILLS[SKILL].sources if s.difficulty == "exam")
    return make(src.topic, seed=seed, difficulty="exam", **src.params)


# -- parsing the stem back -------------------------------------------------------------------------------------------


class Parsed:
    """The population table and the two samples, read from the text alone."""

    def __init__(self, text: str):
        self.text = text
        table = re.search(r"\(number of students\): (.*?)\. (?:Random sample)", text).group(1)
        self.numeric = re.match(r"\d", table) is not None
        if self.numeric:
            self.table = {int(v): int(c) for v, c in re.findall(r"(\d+) \((\d+)\)", table)}
        else:
            self.table = {v: int(c) for v, c in re.findall(r"(\w+) \((\d+)\)", table)}
        self.N = sum(self.table.values())
        rnd = re.search(r"Random sample \((\d+) students.*?\): (.*?)\. Convenience", text)
        cnv = re.search(r"Convenience sample \(.*?\): (.*?)\. (?!.*Convenience)", text)
        self.n = int(rnd.group(1))
        self.rnd_text, self.cnv_text = rnd.group(2), cnv.group(1)
        if self.numeric:
            self.rnd = [int(x) for x in self.rnd_text.split(", ")]
            self.cnv = [int(x) for x in self.cnv_text.split(", ")]
        else:
            self.rnd = {k: int(v) for k, v in (p.split(" ") for p in self.rnd_text.split(", "))}
            self.cnv = {k: int(v) for k, v in (p.split(" ") for p in self.cnv_text.split(", "))}
            self.target = self.target_of(text)

    @staticmethod
    def target_of(text):
        return WHO[
            next(
                w
                for w in sorted(WHO, key=len, reverse=True)
                if w in text.split("Convenience sample")[-1].split(". ", 1)[1]
            )
        ]

    # population value
    @property
    def pop(self) -> Fraction:
        if self.numeric:
            return Fraction(sum(v * c for v, c in self.table.items()), self.N)
        return Fraction(100 * self.table[self.target], self.N)

    def est(self, which: str) -> Fraction:
        s = self.rnd if which == "random" else self.cnv
        if self.numeric:
            return Fraction(sum(s), len(s))
        return Fraction(100 * s[self.target], sum(s.values()))

    def gap(self, which):
        return abs(self.est(which) - self.pop)


def expected(q) -> Fraction:
    """The answer to the asked quantity, from the stem alone."""
    p = Parsed(q.text)
    t = q.text
    # the sentences after the convenience sample line are the question
    ask = t.split("Convenience sample")[-1].split(". ", 1)[1]
    if ask.startswith("What is the mean of the whole school") or ask.startswith("What percentage of all the students"):
        return p.pop
    m = re.match(
        r"What is the mean of the (random|convenience) sample|What percentage of the (random|convenience)", ask
    )
    if m:
        return p.est(m.group(1) or m.group(2))
    m = re.match(
        r"How far is the mean of the (random|convenience)|By how many percentage points does the (random|convenience)",
        ask,
    )
    if m:
        return p.gap(m.group(1) or m.group(2))
    if "is nearer" in ask:
        return abs(p.gap("convenience") - p.gap("random"))
    if "put together" in ask:
        return abs((p.est("random") + p.est("convenience")) / 2 - p.pop)
    if "multiplies the convenience sample" in ask:
        return p.gap("convenience") * p.N
    if "applies the convenience sample" in ask:
        return p.gap("convenience") * p.N / 100
    raise AssertionError(f"unrecognised question: {ask!r}")


def as_fraction(text: str) -> Fraction:
    return Fraction(text)


# -- wiring ----------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    skill = SKILLS[SKILL]
    assert skill.status == "active" and skill.answer == "number"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]
    for prerequisite in skill.prerequisites:
        assert prerequisite in SKILLS
    listed = {m.id for m in skill.misconceptions}
    assert listed >= {k for k in CATALOGUE if k.startswith("ds-svp-")}


@pytest.mark.parametrize("topic,tier", [("ds_sample_vs_pop_gap", "extreme"), ("ds_sample_vs_pop_exam", "easy")])
def test_unknown_tier_is_a_value_error(topic, tier):
    fn = gen.ds_sample_vs_pop_gap if topic.endswith("gap") else gen.ds_sample_vs_pop_exam
    import random

    with pytest.raises(ValueError):
        fn(random.Random(0), tier=tier)


# -- single questions ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_clean_deterministic_and_keyed_from_the_stem(tier):
    for seed in SEEDS:
        q = single(tier, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == single(tier, seed).to_json()
        want = expected(q)
        got = Fraction(q.answer_text)
        assert got == want, (tier, seed, q.text, q.answer_text, want)


@pytest.mark.parametrize("tier", TIERS)
def test_key_graded_correct_and_every_distractor_wrong_and_tagged(tier):
    skill = SKILLS[SKILL]
    for seed in SEEDS:
        q = single(tier, seed)
        assert grade(skill, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade(skill, q, d).correct, (tier, seed, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (seed, d)


def test_samples_are_possible_draws_from_the_population():
    """No value (or category) is used more often in a sample than the population table holds it."""
    for tier in TIERS:
        for seed in SEEDS:
            p = Parsed(single(tier, seed).text)
            for sample in (p.rnd, p.cnv):
                used = Counter(sample) if p.numeric else sample
                assert all(k <= p.table[key] for key, k in used.items()), (tier, seed)
                assert sum(used.values()) == p.n


def test_the_convenience_sample_is_biased_and_the_random_one_is_not_identical():
    for tier in TIERS:
        for seed in SEEDS:
            p = Parsed(single(tier, seed).text)
            assert p.gap("convenience") > 0 and p.gap("random") > 0
            assert p.gap("convenience") != p.gap("random")


@pytest.mark.parametrize("tier", TIERS)
def test_three_prompt_shapes_and_a_wide_answer_space(tier):
    shapes = {prompt_shape(single(tier, s)) for s in range(80)}
    assert len(shapes) >= 3, shapes
    counts = Counter(single(tier, s).answer_text for s in range(200))
    assert len(counts) >= 12
    assert counts.most_common(1)[0][1] / 200 <= 0.2


def test_tiers_escalate_in_reasoning_not_size():
    """Easy asks one quantity; medium a gap; hard two samples at once (nearer-by, pooled, whole-school error)."""

    def asked(tier):
        out = set()
        for seed in range(80):
            t = single(tier, seed).text
            q = t.split("Convenience sample")[-1].split(". ", 1)[1]
            out.add(
                "hard"
                if ("nearer" in q or "put together" in q or "multiplies" in q or "applies" in q)
                else "gap"
                if ("far" in q or "percentage points" in q)
                else "value"
            )
        return out

    assert asked("easy") == {"value"}
    assert asked("medium") == {"gap"}
    assert asked("hard") == {"hard"}


# -- exam chain ------------------------------------------------------------------------------------------------------


def test_exam_chain_shape_and_marks():
    for seed in range(30):
        q = exam(seed)
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam(seed).to_json()
        assert all(p.sensible_range is not None for p in q.parts)


def test_exam_keys_are_rederived_from_the_stem():
    for seed in range(30):
        q = exam(seed)
        p = Parsed(q.text)
        a, b, c, d = (Fraction(x.answer_text) for x in q.parts)
        assert a == p.pop
        assert b == p.gap("random") and c == p.gap("convenience")
        if p.numeric:
            assert d == c * p.N
        else:
            assert d == c * p.N / 100


def test_exam_marking_with_follow_through():
    skill = SKILLS[SKILL]
    for seed in range(12):
        q = exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(skill, q, canon)
        assert full.correct and full.marks == full.max_marks
        for part in q.parts:
            assert len(part.distractors) >= 2
            for d in part.distractors:
                assert part.option_notes.get(d) in CATALOGUE
                result = grade(skill, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
        junk = grade(skill, q, {p.label: f"{7919 * (i + 1)}.5" for i, p in enumerate(q.parts)})
        assert junk.marks == 0


def test_exam_follow_through_formulas_give_the_keys():
    for seed in range(30):
        q = exam(seed)
        values = {}
        for part in q.parts:
            value = sympy.Rational(Fraction(part.answer_text).numerator, Fraction(part.answer_text).denominator)
            if part.follow_through:
                carried = sympy.sympify(part.follow_through, locals={"Abs": sympy.Abs, **values})
                assert sympy.simplify(carried - value) == 0, (seed, part.label, part.follow_through)
            values[f"part_{part.label}"] = value


def test_a_wrong_part_a_is_followed_through():
    skill = SKILLS[SKILL]
    for seed in range(30):
        q = exam(seed)
        mean = Parsed(q.text).numeric
        wrong_a = Fraction(q.parts[0].answer_text) + 1
        est_r = Parsed(q.text).est("random")
        est_c = Parsed(q.text).est("convenience")
        b = abs(est_r - wrong_a)
        c = abs(est_c - wrong_a)
        d = abs(c - b) if mean else c * Parsed(q.text).N / 100
        reply = {"a": str(wrong_a), "b": f"{float(b):.6f}", "c": f"{float(c):.6f}", "d": f"{float(d):.6f}"}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (seed, reply)


def test_exam_answers_are_varied():
    answers = {tuple(p.answer_text for p in exam(s).parts) for s in range(300)}
    assert len(answers) >= 100
    shapes = {prompt_shape(exam(s)) for s in range(60)}
    assert len(shapes) >= 3 or len({tuple(p.prompt[:20] for p in exam(s).parts) for s in range(60)}) >= 2


def test_exam_part_d_is_varied():
    distinct = {exam(s).parts[3].answer_text for s in range(300)}
    assert len(distinct) >= 120, len(distinct)
