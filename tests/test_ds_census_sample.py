"""Tests `f10-census-sample-experiment-practicalities`: the key is re-derived from the printed study, not from the generator."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

import pytest

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from tests import r3b_support as sup

SKILL = "f10-census-sample-experiment-practicalities"
TOPIC = "ds_census_sample_statements"
SIZE = {"easy": 6, "medium": 7, "hard": 8}
CUT = re.compile(r" \(([A-H])\) ")
METHOD = re.compile(r"^(A census|A random sample|A well-designed experiment|An observational study) (.*)\.$")
DESTRUCTIVE = ("until it goes flat", "to taste the soup", "until it burns out")
SEEDS = range(240)


def parse(text):
    """(the scene and instruction, {letter: statement}) from a question's text."""
    head, rest = text.split(" (A) ", 1)
    parts = CUT.split(" (A) " + rest)
    return head, dict(zip(parts[1::2], (p.strip() for p in parts[2::2])))


def truth(scene, st):
    """Whether one printed statement is true, worked out from the scene alone."""
    m = METHOD.match(st)
    if m:
        method, claim = m.group(1), m.group(2)
        if claim.startswith("can show that"):
            return method == "A well-designed experiment"
        if claim.startswith("cannot show that"):
            return method != "A well-designed experiment"
        if claim.startswith("changes something"):
            return method == "A well-designed experiment"
        if claim.startswith("changes nothing"):
            return method != "A well-designed experiment"
        assert method in ("A census", "A random sample"), st
        if claim.startswith("collects data from every"):
            return method == "A census"
        if claim.startswith("leaves out some"):
            return method == "A random sample"
        if claim.startswith("gives the exact value"):
            return method == "A census"
        if claim.startswith("can give a value that differs"):
            return method == "A random sample"
        raise AssertionError(st)
    destroys = any(w in st for w in DESTRUCTIVE)
    volunteers = "volunteers" in scene
    if " uses up the " in st:
        return destroys
    if " as it was." in st:
        return not destroys
    if "picked at random from" in st:
        return not volunteers
    if "chose themselves by volunteering" in st:
        return volunteers
    if st.startswith("These results may be biased"):
        return volunteers
    if st.startswith("These results cannot be biased"):
        return not volunteers
    # figures
    m = re.match(r"Checking all (\d+) .* would need (\d+) times as many \w+ as checking the (\d+) in the sample\.", st)
    if m:
        N, V, n = map(int, m.groups())
        return V * n == N
    m = re.match(r"Checking all (\d+) .* would (?:take|cost) \$?(\d+)(?: \w+)? in all\.", st)
    if m:
        N, V = map(int, m.groups())
        each = re.search(r"Each [\w -]+ (?:takes|costs) \$?(\d+)", scene)
        return N * int(each.group(1)) == V
    m = re.match(r"Each of the two groups has (\d+) ", st)
    if m:
        n = int(re.search(r"(?:asks for|picks|marks out) (\d+) ", scene).group(1))
        return 2 * int(m.group(1)) == n
    m = re.match(r"The experiment collects (\d+) .* in all\.", st)
    if m:
        n = int(re.search(r"(?:asks for|picks|marks out) (\d+) ", scene).group(1))
        r = int(re.search(r"(\d+) (?:times|quizzes|separate seasons)", scene).group(1))
        return n * r == int(m.group(1))
    m = re.match(r"The observer watched for (\d+) hours in all\.", st)
    if m:
        h, d = re.search(r"for (\d+) hours each \w+ for (\d+) ", scene).groups()
        return int(h) * int(d) == int(m.group(1))
    m = re.match(r"The \w+ counted over the (\d+) days add up to (\d+)\.", st)
    if m:
        counts = [int(x) for x in re.search(r"The counts were ([\d, ]+)\.", scene).group(1).split(", ")]
        return len(counts) == int(m.group(1)) and sum(counts) == int(m.group(2))
    raise AssertionError(f"unread statement: {st!r}")


def test_wired():
    data = sup.skill_yaml(SKILL)
    assert data["answer"] == "set" and data["status"] == "active"
    assert [s["difficulty"] for s in data["sources"]] == ["easy", "medium", "hard"]
    assert all(s["topic"] == TOPIC for s in data["sources"])
    assert "AC9M8ST01" in data["tags"] and "acara-f10-v9" in data["tags"]
    assert data["year_level"] == ["8"]
    assert sup.skill_object(SKILL).status == "active"


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_is_rederived_from_the_printed_study(tier):
    ks = Counter()
    for seed in SEEDS:
        q = sup.tier_question(TOPIC, tier, seed)
        scene, statements = parse(q.text)
        assert len(statements) == SIZE[tier], (seed, q.text)
        assert list(statements) == list("ABCDEFGH"[: SIZE[tier]])
        assert len(set(statements.values())) == SIZE[tier], "a statement is repeated"
        key = sorted(letter for letter, st in statements.items() if truth(scene, st))
        assert key == sorted(q.answer), (seed, q.text, q.answer)
        assert 2 <= len(key) <= SIZE[tier] - 2
        ks[len(key)] += 1
    assert len(ks) >= SIZE[tier] - 3, ks


@pytest.mark.parametrize("tier", sup.TIERS)
def test_graded_key_in_any_order_and_every_wrong_set_rejected(tier):
    sk = sup.skill_object(SKILL)
    for seed in range(60):
        q = sup.tier_question(TOPIC, tier, seed)
        letters = list(q.answer)
        assert grade(sk, q, q.answer_text).correct
        assert grade(sk, q, ", ".join(reversed(letters))).correct
        assert grade(sk, q, ", ".join(letters).lower()).correct
        assert len(q.distractors) >= 3 and len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text
            assert not grade(sk, q, d).correct, (seed, d)
            assert q.option_notes.get(d) in CATALOGUE, (seed, d)
        # also wrong: none, all, one missing, one extra
        everything = list("ABCDEFGH"[: SIZE[tier]])
        for wrong in ("", ", ".join(everything), ", ".join(letters[1:]), ", ".join(letters[:-1]), "Z"):
            assert not grade(sk, q, wrong).correct, (seed, wrong)
        extra = [c for c in everything if c not in letters][0]
        assert not grade(sk, q, ", ".join(letters + [extra])).correct


@pytest.mark.parametrize("tier", sup.TIERS)
def test_no_tell_in_length_wording_or_place(tier):
    true_len, false_len = [], []
    form = defaultdict(lambda: [0, 0])
    place = Counter()
    longest = Counter()
    for seed in range(300):
        q = sup.tier_question(TOPIC, tier, seed)
        scene, statements = parse(q.text)
        flags = {letter: truth(scene, st) for letter, st in statements.items()}
        for letter, st in statements.items():
            (true_len if flags[letter] else false_len).append(len(st))
            for w in (
                "cannot show",
                "can show",
                "changes nothing",
                "changes something",
                "leaves out",
                "every ",
                "exact",
                "differs",
            ):
                if w in st:
                    form[w][flags[letter]] += 1
            place[letter] += flags[letter]
        top = max(statements, key=lambda k: len(statements[k]))
        longest[flags[top]] += 1
    assert abs(statistics.mean(true_len) - statistics.mean(false_len)) < 0.05 * statistics.mean(true_len)
    for w, (f, t) in form.items():
        if f + t >= 60:
            assert 0.3 <= t / (f + t) <= 0.7, (w, f, t)
    share = sum(place.values())
    for letter, c in place.items():
        assert abs(c / share - 1 / SIZE[tier]) < 0.06, place
    assert 0.4 <= longest[True] / 300 <= 0.62, longest


def test_tiers_escalate():
    """Easy has properties only; medium adds a fact of the study and a figure; hard has two or more figures."""
    for seed in range(120):
        for tier in sup.TIERS:
            q = sup.tier_question(TOPIC, tier, seed)
            _, statements = parse(q.text)
            figures = sum(1 for s in statements.values() if re.search(r"\d", s))
            facts = sum(1 for s in statements.values() if not METHOD.match(s) and not re.search(r"\d", s))
            if tier == "easy":
                assert figures == 0 and facts == 0, q.text
            elif tier == "medium":
                assert figures >= 1 and figures + facts >= 2, q.text
            else:
                assert figures >= 2, q.text


@pytest.mark.parametrize("tier", sup.TIERS)
def test_prompt_variety_and_situations(tier):
    intros, scenes, kinds = set(), set(), Counter()
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        scene, _ = parse(q.text)
        intros.add(scene.rsplit(". ", 1)[-1])
        scenes.add(re.sub(r"\d+", "#", scene)[:40])
        for key, words in (
            ("census", "checks every"),
            ("sample", "checks only"),
            ("experiment", "put at random"),
            ("observation", "counts"),
        ):
            if words in scene:
                kinds[key] += 1
    assert len(intros) >= 3, intros
    assert len(scenes) >= 6, scenes
    assert set(kinds) >= {"census", "sample", "experiment", "observation"}, kinds
    assert len({sup.prompt_shape(sup.tier_question(TOPIC, tier, s)) for s in range(60)}) >= 3


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_and_deterministic(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier, seeds=range(60))


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="extreme")


def test_misconceptions_listed_and_catalogued():
    from mathgen.generators.ds_census_sample_build import MISCONCEPTIONS

    listed = {m["id"] for m in sup.skill_yaml(SKILL)["misconceptions"]}
    assert {m[0] for m in MISCONCEPTIONS} == listed
    assert listed <= set(CATALOGUE)
    used = set()
    for tier in sup.TIERS:
        for seed in range(150):
            used |= set(sup.tier_question(TOPIC, tier, seed).option_notes.values())
    assert used <= listed
    assert len(used) >= 14, used


def test_answer_space():
    for tier in sup.TIERS:
        sup.assert_answer_space(TOPIC, tier, seeds=range(200), distinct=min(12, 2 ** (SIZE[tier] - 2)), top=0.25)
