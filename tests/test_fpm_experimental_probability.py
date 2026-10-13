"""Tests for the four Years 5-8 experimental-probability skills built in `fpm_experimental_probability_build`.

Each key is re-derived from the rendered question text (the oracles below parse the stem, never the generator's
own variables), every distractor must grade wrong through `mathgen.engine.grade`, and every exam chain is marked part
by part with follow-through.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fpm_experimental_probability_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

SKILLS = {s.id: s for s in load_skills()[1]}
SEEDS = range(150)
TIERS = ("easy", "medium", "hard")

REL = "f10-relative-frequency-estimate"
EXP = "f10-expected-versus-observed"
SAMPLE = "f10-sample-size-and-reliability"
COMPOUND = "f10-compound-event-frequencies"


def frac_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def single(skill_id: str, tier: str, seed: int):
    src = next(s for s in SKILLS[skill_id].sources if s.difficulty == tier)
    return make(src.topic, seed=seed, difficulty=tier, **src.params)


def exam(skill_id: str, seed: int):
    src = next(s for s in SKILLS[skill_id].sources if s.difficulty == "exam")
    return make(src.topic, seed=seed, difficulty="exam", **src.params)


# -- wiring -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,kind",
    [(REL, "fraction"), (EXP, "number"), (SAMPLE, "number"), (COMPOUND, "fraction")],
)
def test_skill_wiring(skill_id, kind):
    skill = SKILLS[skill_id]
    assert skill.status == "active" and skill.answer == kind
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [s.stages for s in skill.sources][-1] == ["exam"]
    for prerequisite in skill.prerequisites:
        assert prerequisite in SKILLS


# -- shared checks over every tier ---------------------------------------------------------------------------------


@pytest.mark.parametrize("skill_id", [REL, EXP, SAMPLE, COMPOUND])
@pytest.mark.parametrize("tier", TIERS)
def test_canonical_correct_and_distractors_wrong(skill_id, tier):
    skill = SKILLS[skill_id]
    for seed in SEEDS:
        q = single(skill_id, tier, seed)
        assert grade(skill, q, q.answer_text).correct, (q.text, q.answer_text)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3, (q.text, q.distractors)
        for d in q.distractors:
            assert not grade(skill, q, d).correct, (q.text, q.answer_text, d)
            assert q.option_notes.get(d) in CATALOGUE, (q.text, d, q.option_notes)
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))


@pytest.mark.parametrize("skill_id", [REL, EXP, SAMPLE, COMPOUND])
def test_tiers_differ_in_shape(skill_id):
    shapes = {tier: {prompt_shape(single(skill_id, tier, s)) for s in range(60)} for tier in TIERS}
    assert all(len(v) >= 2 for v in shapes.values()), {k: len(v) for k, v in shapes.items()}
    assert shapes["easy"] != shapes["hard"] and shapes["medium"] != shapes["hard"]


@pytest.mark.parametrize("skill_id", [REL, EXP, SAMPLE, COMPOUND])
@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(skill_id, tier):
    assert single(skill_id, tier, 7).to_dict() == single(skill_id, tier, 7).to_dict()


# -- 1. relative frequency -----------------------------------------------------------------------------------------


def _categories_in(event_text: str, cats):
    found = []
    for cat in cats:
        if re.search(rf"(?<![\w]){re.escape(cat)}(?![\w])", event_text):
            found.append(cat)
    return found


def rel_oracle(text: str) -> Fraction:
    event = text.split("probability that ")[1].split(". Give")[0].split("? Answer")[0]
    negated = bool(re.search(r"\bnot\b", event))
    if "The relative frequencies of the results were: " in text:
        body = text.split("The relative frequencies of the results were: ")[1].split(". The relative")[0]
        shares = {}
        for item in body.split(", "):
            name, pct = item.rsplit(" ", 1)
            shares[name] = int(pct.rstrip("%"))
        missing = re.search(r"relative frequency for (.+?) was not recorded", text).group(1)
        shares[missing] = 100 - sum(shares.values())
        assert shares[missing] > 0
        counts, total = shares, 100
    else:
        body = text.split("The results were: ")[1].split(". ")[0]
        counts = {}
        for item in body.split(", "):
            name, n = item.rsplit(" ", 1)
            counts[name] = int(n)
        total = sum(counts.values())
    chosen = _categories_in(event, list(counts))
    assert chosen, (text, event)
    inside = sum(counts[c] for c in chosen)
    return Fraction(total - inside if negated else inside, total)


@pytest.mark.parametrize("tier", TIERS)
def test_relative_frequency_oracle(tier):
    seen = Counter()
    for seed in range(300):
        q = single(REL, tier, seed)
        assert frac_text(rel_oracle(q.text)) == q.answer_text, (q.text, q.answer_text)
        seen[q.answer_text] += 1
    assert len(seen) >= 25


def test_relative_frequency_medium_covers_both_event_kinds():
    texts = [single(REL, "medium", s).text for s in range(120)]
    assert any(" or " in t.split("probability that ")[1] for t in texts)
    assert any(re.search(r"\bnot\b", t.split("probability that ")[1]) for t in texts)


def test_relative_frequency_exam_is_the_shared_chain():
    src = next(s for s in SKILLS[REL].sources if s.difficulty == "exam")
    assert (src.topic, src.params, src.stages) == ("exd_x_r2l_relfreq_chain", {"tier": "exam"}, ["exam"])
    q = exam(REL, 3)
    assert grade(SKILLS[REL], q, q.answer_text).correct


# -- 2. expected versus observed -----------------------------------------------------------------------------------

_DIE = {
    "a number greater than 4": {5, 6},
    "an even number": {2, 4, 6},
    "a 6": {6},
    "a number less than 3": {1, 2},
    "a multiple of 3": {3, 6},
    "a prime number": {2, 3, 5},
}
_UNION = {
    "an even number or a number greater than 4": {2, 4, 5, 6},
    "a multiple of 3 or an even number": {2, 3, 4, 6},
    "a prime number or an even number": {2, 3, 4, 5, 6},
    "an odd number or a number less than 3": {1, 2, 3, 5},
    "a number greater than 3 or a multiple of 3": {3, 4, 5, 6},
}


def _setting_probability(text: str) -> tuple[Fraction, int]:
    """(theoretical probability, number of trials) read from a stem."""
    if "die is rolled" in text:
        n = int(re.search(r"rolled (\d+) times", text).group(1))
        for phrase, faces in sorted({**_DIE, **_UNION}.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(phrase)} (?:came up|to come up)", text):
                return Fraction(len(faces), 6), n
        raise AssertionError(text)
    if "equal sections" in text:
        m, k, n = (
            int(x)
            for x in re.search(r"has (\d+) equal sections, (\d+) of them \w+\. It is spun (\d+) times", text).groups()
        )
        return Fraction(k, m), n
    if "centre angle" in text:
        theta = int(re.search(r"centre angle of (\d+)°", text).group(1))
        return Fraction(theta, 360), int(re.search(r"spun (\d+) times", text).group(1))
    n = int(re.search(r"(\d+) times in all", text).group(1))
    r, b, g = (int(x) for x in re.search(r"holds (\d+) red, (\d+) blue and (\d+) green", text).groups())
    removed = re.search(r"(\d+) red marbles? (?:is|are) removed", text)
    colour = re.search(r"a (\w+) marble (?:to be drawn|was drawn)", text).group(1)
    counts = {"red": r, "blue": b, "green": g}
    total = r + b + g
    if removed:
        x = int(removed.group(1))
        counts["red"] -= x
        total -= x
    return Fraction(counts[colour], total), n


def _observed(text: str) -> int:
    return int(re.search(r"In the experiment, .*? (\d+) times", text).group(1))


@pytest.mark.parametrize("tier", TIERS)
def test_expected_oracle(tier):
    for seed in range(250):
        q = single(EXP, tier, seed)
        p, n = _setting_probability(q.text)
        expected = p * n
        assert expected.denominator == 1
        key = int(expected) if tier == "easy" else abs(_observed(q.text) - int(expected))
        assert int(q.answer_text) == key, (q.text, q.answer_text, key)


def test_expected_hard_uses_a_second_step():
    kinds = Counter()
    for seed in range(150):
        text = single(EXP, "hard", seed).text
        kinds["angle" if "centre angle" in text else "removed" if "removed" in text else "or"] += 1
    assert set(kinds) == {"angle", "removed", "or"} and min(kinds.values()) >= 20, kinds


def test_expected_exam_chain():
    skill = SKILLS[EXP]
    for seed in range(120):
        q = exam(EXP, seed)
        p, n = _setting_probability(q.stem)
        expected, observed = int(p * n), _observed(q.stem)
        target = int(re.search(r"expect .*? (\d+) times", q.parts[3].prompt).group(1))
        want = {
            "a": frac_text(p),
            "b": str(expected),
            "c": str(abs(observed - expected)),
            "d": str(int(target / p)),
        }
        assert {x.label: x.answer_text for x in q.parts} == want, q.text
        assert [x.marks for x in q.parts] == [2, 1, 1, 2]
        assert [x.follow_through is not None for x in q.parts] == [False, True, True, True]
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_expected_exam_distractors_and_follow_through():
    skill = SKILLS[EXP]
    for seed in range(80):
        q = exam(EXP, seed)
        canon = {x.label: x.answer_text for x in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        p = Fraction(canon["a"]) if "/" in canon["a"] else Fraction(int(canon["a"]))
        _, n = _setting_probability(q.stem)
        wrong_a = Fraction(1, 2) if p != Fraction(1, 2) else Fraction(1, 3)
        wrong_b = wrong_a * n
        if wrong_b.denominator != 1:
            continue
        reply = {"a": frac_text(wrong_a), "b": str(int(wrong_b)), "c": str(abs(_observed(q.stem) - int(wrong_b)))}
        reply["d"] = str(int(Fraction(re.search(r"expect .*? (\d+) times", q.parts[3].prompt).group(1)) / wrong_a))
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert all(result.parts[x].marks == result.parts[x].max_marks for x in "bcd"), (q.text, reply, result)


# -- 3. sample size and reliability -------------------------------------------------------------------------------


def _reports(text: str):
    found = re.findall(r"(\d+) (?:times|free throws|kicks at goal)[^.]*\. ([^.]*)\.", text)
    return [(int(n), int(re.search(r"\d+", res).group(0))) for n, res in found]


def _pct_text(num: int, den: int) -> str:
    value = Fraction(100 * num, den)
    assert (value * 10).denominator == 1
    return f"{float(value):.1f}"


@pytest.mark.parametrize("tier", TIERS)
def test_sample_size_oracle(tier):
    kinds = Counter()
    for seed in range(250):
        q = single(SAMPLE, tier, seed)
        reports = _reports(q.text)
        if tier == "easy":
            assert len(reports) == 2
            want = _pct_text(sum(s for _, s in reports), sum(n for n, _ in reports))
        elif tier == "medium":
            assert len(reports) == 3
            n, s = max(reports)
            assert sorted(r[0] for r in reports)[-1] >= 4 * sorted(r[0] for r in reports)[0]
            want = _pct_text(s, n)
        elif "closer" in q.text:
            kinds["closer"] += 1
            theory = Fraction(re.search(r"or (\d+\.\d)%", q.text).group(1))
            small, pooled = (
                Fraction(100 * reports[0][1], reports[0][0]),
                Fraction(100 * sum(s for _, s in reports), sum(n for n, _ in reports)),
            )
            want = f"{float(abs(small - theory) - abs(pooled - theory)):.1f}"
            assert abs(small - theory) > abs(pooled - theory)
        else:
            kinds["pool"] += 1
            assert len(reports) == 3
            want = _pct_text(sum(s for _, s in reports), sum(n for n, _ in reports))
        assert q.answer_text == want, (q.text, q.answer_text, want)
    if tier == "hard":
        assert min(kinds.values()) >= 40, kinds


def test_sample_size_medium_largest_experiment_is_not_the_extreme_estimate():
    positions = Counter()
    for seed in range(200):
        q = single(SAMPLE, "medium", seed)
        reports = _reports(q.text)
        pcts = [Fraction(s, n) for n, s in reports]
        big = max(range(3), key=lambda i: reports[i][0])
        positions[big] += 1
        # the most reliable result is not always the highest or the lowest estimate
        assert len(set(pcts)) == 3
    assert min(positions.values()) >= 30, positions


def test_sample_size_larger_experiments_sit_closer_to_the_probability():
    """The seeded draws are real binomial draws, so the average gap shrinks as the experiment grows."""
    gaps_small, gaps_big = [], []
    for seed in range(300):
        q = single(SAMPLE, "hard", seed)
        if "closer" not in q.text:
            continue
        theory = Fraction(re.search(r"or (\d+\.\d)%", q.text).group(1))
        reports = _reports(q.text)
        gaps_small.append(abs(Fraction(100 * reports[0][1], reports[0][0]) - theory))
        gaps_big.append(abs(Fraction(100 * sum(s for _, s in reports), sum(n for n, _ in reports)) - theory))
    assert len(gaps_small) >= 60
    assert sum(gaps_big) < sum(gaps_small)


def test_sample_size_exam_chain():
    skill = SKILLS[SAMPLE]
    for seed in range(120):
        q = exam(SAMPLE, seed)
        reports = _reports(q.stem)
        total = sum(n for n, _ in reports)
        successes = sum(s for _, s in reports)
        theory = Fraction(re.search(r"or (\d+\.\d)%", q.stem).group(1))
        pooled = Fraction(100 * successes, total)
        future = int(re.search(r"in (\d+) further trials", q.parts[3].prompt).group(1))
        want = {
            "a": str(successes),
            "b": f"{float(pooled):.1f}",
            "c": f"{float(abs(pooled - theory)):.1f}",
            "d": str(int(pooled * future / 100)),
        }
        assert {x.label: x.answer_text for x in q.parts} == want, q.text
        assert [x.marks for x in q.parts] == [1, 2, 1, 2]
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == 6


def test_sample_size_exam_distractors_and_follow_through():
    skill = SKILLS[SAMPLE]
    checked = 0
    for seed in range(100):
        q = exam(SAMPLE, seed)
        canon = {x.label: x.answer_text for x in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        reports = _reports(q.stem)
        total = sum(n for n, _ in reports)
        theory = Fraction(re.search(r"or (\d+\.\d)%", q.stem).group(1))
        wrong_a = sum(s for _, s in reports) + 1
        wrong_b = Fraction(100 * wrong_a, total)
        future = int(re.search(r"in (\d+) further trials", q.parts[3].prompt).group(1))
        reply = {
            "a": str(wrong_a),
            "b": f"{float(wrong_b):.1f}",
            "c": f"{float(abs(wrong_b - theory)):.1f}",
            "d": str(int(wrong_b * future / 100)),
        }
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert all(result.parts[x].marks == result.parts[x].max_marks for x in "bcd"), (q.text, reply, result)
        checked += 1
    assert checked == 100


# -- 4. compound events --------------------------------------------------------------------------------------------


def _cells(text: str):
    body = re.search(r"Table \([^)]*\): (.*?)\. ", text).group(1)
    rows = [r.split(" — ") for r in body.split("; ")]
    return [int(rows[0][1]), int(rows[0][2]), int(rows[1][1]), int(rows[1][2])]


def compound_oracle(text: str) -> Fraction:
    if "Table (" in text:
        both, a_only, b_only, neither = _cells(text)
        total = both + a_only + b_only + neither
        if " and " in text.split("at random, ")[1].split(". Give")[0] and "(or both)" not in text:
            return Fraction(both, total)
        return Fraction(total - neither, total)
    total = int(re.search(r"\d+", text).group(0))
    n_a, n_b, both = (
        int(x)
        for x in re.search(
            r"A happened (\d+) times, B happened (\d+) times, and both A and B happened (\d+) times", text
        ).groups()
    )
    if "neither A nor B" in text:
        return Fraction(total - (n_a + n_b - both), total)
    if "exactly one" in text:
        return Fraction(n_a + n_b - 2 * both, total)
    assert "A happens but B does not" in text
    return Fraction(n_a - both, total)


@pytest.mark.parametrize("tier", TIERS)
def test_compound_oracle(tier):
    asks = Counter()
    for seed in range(300):
        q = single(COMPOUND, tier, seed)
        assert frac_text(compound_oracle(q.text)) == q.answer_text, (q.text, q.answer_text)
        asks[
            (
                "neither"
                if "neither" in q.text
                else "exactly"
                if "exactly one" in q.text
                else "but"
                if "but B" in q.text
                else tier
            )
        ] += 1
    if tier == "hard":
        assert {"neither", "exactly", "but"} <= set(asks) and min(asks.values()) >= 40, asks


def test_compound_exam_chain():
    skill = SKILLS[COMPOUND]
    for seed in range(120):
        q = exam(COMPOUND, seed)
        table = re.search(r"Table \([^)]*\): (.*?)\. \(", q.text + " ").group(1) if False else None
        body = re.search(r"Table \([^)]*\): (.*?)\. \(a\)", q.text).group(1)
        rows = [r.split(" — ") for r in body.split("; ")]
        a_only, b_only, neither = int(rows[0][2]), int(rows[1][1]), int(rows[1][2])
        total = int(re.search(r"\d+", q.text).group(0))
        both = total - a_only - b_only - neither
        union = total - neither
        future = int(re.search(r"next (\d+) ", q.parts[3].prompt).group(1))
        projected = Fraction(future * union, total)
        assert abs(projected % 1 - Fraction(1, 2)) > Fraction(1, 20)
        want = {
            "a": str(both),
            "b": frac_text(Fraction(both, total)),
            "c": frac_text(Fraction(union, total)),
            "d": str(int(projected + Fraction(1, 2))),
        }
        assert {x.label: x.answer_text for x in q.parts} == want, q.text
        assert [x.marks for x in q.parts] == [1, 1, 2, 2]
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == 6


def test_compound_exam_distractors_and_follow_through():
    skill = SKILLS[COMPOUND]
    for seed in range(100):
        q = exam(COMPOUND, seed)
        canon = {x.label: x.answer_text for x in q.parts}
        for part in q.parts:
            assert len(part.distractors) == 3
            for d in part.distractors:
                assert part.option_notes[d] in CATALOGUE
                assert grade(skill, q, {**canon, part.label: d}).parts[part.label].marks == 0, (q.text, part.label, d)
        body = re.search(r"Table \([^)]*\): (.*?)\. \(a\)", q.text).group(1)
        rows = [r.split(" — ") for r in body.split("; ")]
        a_only, b_only, neither = int(rows[0][2]), int(rows[1][1]), int(rows[1][2])
        total = int(re.search(r"\d+", q.text).group(0))
        wrong_a = total - a_only - b_only - neither + 1
        union = Fraction(a_only + b_only + wrong_a, total)
        if not 0 < union < 1:
            continue
        future = int(re.search(r"next (\d+) ", q.parts[3].prompt).group(1))
        projected = future * union
        if abs(projected % 1 - Fraction(1, 2)) < Fraction(1, 100):
            continue
        reply = {
            "a": str(wrong_a),
            "b": frac_text(Fraction(wrong_a, total)),
            "c": frac_text(union),
            "d": str(int(projected + Fraction(1, 2))),
        }
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert all(result.parts[x].marks == result.parts[x].max_marks for x in "bcd"), (q.text, reply, result)


@pytest.mark.parametrize(
    "skill_id,exam_topic",
    [
        (EXP, "fpm_expected_observed_exam"),
        (SAMPLE, "fpm_sample_size_exam"),
        (COMPOUND, "fpm_compound_frequencies_exam"),
    ],
)
def test_exam_wrong_everything_earns_nothing_and_text_is_clean(skill_id, exam_topic):
    skill = SKILLS[skill_id]
    for seed in range(40):
        q = exam(skill_id, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert grade(skill, q, {"a": "999", "b": "9999", "c": "99999", "d": "999999"}).marks == 0
        assert find_text_defects(q) == [], (q.text, find_text_defects(q))
    assert len({prompt_shape(exam(skill_id, s)) for s in range(40)}) >= 2
