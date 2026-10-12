"""Early-years figure kinds (money, hundred chart, fraction model, counters, base ten) and the skills that use them.

Every skill is generated here through its own sources, at every tier and 60 seeds: the figure is valid,
does not show the key, the registered reader re-derives the key from the figure, and the real graders
accept the key and reject every distractor. Also the converted skills (counting, subitising, arrays).
"""

from __future__ import annotations

import functools
import random

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import REGISTRY
from mathgen.figure_kinds import early
from mathgen.grading.typed import grade, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts import skill_contract
from scripts.measure_tiers import classify_skill

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

NEW = {
    "f10-money-count-collection": ("money_count_collection", "number", "early-foundations"),
    "f10-money-compare-collections": ("money_compare_collections", "mcq", "early-foundations"),
    "f10-money-change-from-note": ("money_change_from_note", "mcq", "early-foundations"),
    "f10-hundred-chart-patterns": ("hundred_chart_read", "number", "early-foundations"),
    "f10-fraction-shaded-name": ("fraction_shaded_name", "fraction", "primary-number"),
    "f10-fraction-compare-models": ("fraction_compare_models", "mcq", "primary-number"),
    "f10-base-ten-blocks-number": ("base_ten_read", "number", "early-foundations"),
}
CONVERTED = {
    "count-to-5": ("count_objects", "mcq"),
    "count-to-10": ("count_objects", "mcq"),
    "f10-subitise-name-to-5": ("subitise_to_5", "number"),
    "f10-multiply-divide-one-digit-arrays": ("multiply_divide_one_digit_arrays", "number"),
}
ALL = {**{k: (v[0], v[1]) for k, v in NEW.items()}, **CONVERTED}


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(skill_id, difficulty, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == difficulty)
    return make(spec.topic, seed=seed, difficulty=difficulty, **(spec.params or {}))


def _tiers(skill_id):
    return [s.difficulty for s in _skills()[skill_id].sources]


def _is_right(skill_id, q, text):
    kind = _skills()[skill_id].answer
    if kind == "mcq":
        return grade("mcq", q, str(mcq_options(q).index(text))).correct
    return grade(kind, q, text).correct


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", sorted(ALL))
def test_skill_is_active_with_the_figure_gate(skill_id):
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.mental is None
    assert skill.answer == ALL[skill_id][1]
    assert {s.topic for s in skill.sources} == {ALL[skill_id][0]}


@pytest.mark.parametrize("skill_id", sorted(NEW))
def test_new_skill_shape(skill_id):
    skill = _skills()[skill_id]
    assert skill.domain == NEW[skill_id][2] and skill.placement_eligible is False
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert len(skill.misconceptions) >= 3


@pytest.mark.parametrize("skill_id", sorted(ALL))
def test_every_tier_draws_a_valid_figure_that_does_not_show_the_key(skill_id):
    for tier in _tiers(skill_id):
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text])


@pytest.mark.parametrize("skill_id", sorted(ALL))
def test_registered_reader_re_derives_the_key(skill_id):
    topic = ALL[skill_id][0]
    reader = figures.get_reader(topic)
    assert reader is not None
    for tier in _tiers(skill_id):
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("skill_id", sorted(ALL))
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id):
    for tier in _tiers(skill_id):
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert _is_right(skill_id, q, q.answer_text), (tier, seed)
            assert q.distractors and q.answer_text not in q.distractors
            for d in q.distractors:
                assert not _is_right(skill_id, q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id", sorted(ALL))
def test_deterministic(skill_id):
    for tier in _tiers(skill_id):
        for seed in range(20):
            spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
            again = make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))
            assert _q(skill_id, tier, seed).to_json() == again.to_json()


@pytest.mark.parametrize("skill_id", sorted(ALL))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


@pytest.mark.parametrize("skill_id", sorted(NEW))
def test_every_tier_adds_a_prompt_shape_and_has_variety(skill_id):
    verdict, adds = classify_skill(_skills()[skill_id], 60)
    assert verdict == "escalates", adds
    for tier in TIERS:
        qs = [_q(skill_id, tier, s) for s in SEEDS]
        assert len({q.text for q in qs}) >= 20
        assert len({prompt_shape(q) for q in qs}) >= 3


@pytest.mark.parametrize("topic", sorted({v[0] for v in ALL.values()}))
def test_unknown_tier_or_mode_raises(topic):
    if topic in ("count_objects",):
        return  # no tier argument
    with pytest.raises((ValueError, TypeError)):
        make(topic, seed=1, tier="bogus", mode="bogus", variant="bogus")


# ------------------------------------------------------------------ mcq key balance


@pytest.mark.parametrize(
    "skill_id", ["f10-money-compare-collections", "f10-fraction-compare-models", "f10-money-change-from-note"]
)
def test_mcq_key_position_is_balanced(skill_id):
    for tier in TIERS:
        counts: dict[str, int] = {}
        for seed in range(300):
            q = make(_skills()[skill_id].sources[TIERS.index(tier)].topic, seed=seed, difficulty=tier)
            counts[q.answer_text] = counts.get(q.answer_text, 0) + 1
        top = max(counts.values()) / 300
        assert top < 0.6, (tier, top)


# ------------------------------------------------------------------ tier shape


def test_money_count_tiers():
    for seed in SEEDS:
        easy = _q("f10-money-count-collection", "easy", seed)
        items = [i["denom"] for i in easy.figure["groups"][0]["items"]]
        assert len(set(items)) == 1 and 2 <= len(items) <= 5
        med = _q("f10-money-count-collection", "medium", seed)
        assert len({i["denom"] for i in med.figure["groups"][0]["items"]}) >= 1 and "$" not in med.text
        hard = _q("f10-money-count-collection", "hard", seed)
        assert "$" in hard.text and "more cents" in hard.text
        total = early.money_cents(hard.figure)
        assert 0 < total < int(hard.answer_text) + total  # the key is what is MISSING, not the total


def test_money_count_medium_often_turns_a_dollar_coin_into_cents():
    hits = sum(
        any(
            i["denom"] in ("$1", "$2")
            for i in _q("f10-money-count-collection", "medium", s).figure["groups"][0]["items"]
        )
        for s in SEEDS
    )
    assert 15 <= hits <= 45


def test_money_compare_tiers_and_traps():
    for seed in SEEDS:
        easy = _q("f10-money-compare-collections", "easy", seed).figure
        assert len(easy["groups"]) == 2
        assert all(i["type"] == "coin" and i["cents"] < 100 for g in easy["groups"] for i in g["items"])
        med = _q("f10-money-compare-collections", "medium", seed).figure
        assert any(i["type"] == "note" for g in med["groups"] for i in g["items"])
        hard = _q("f10-money-compare-collections", "hard", seed)
        groups = hard.figure["groups"]
        assert len(groups) == 3
        sizes = [len(g["items"]) for g in groups]
        top = [max(i["cents"] for i in g["items"]) for g in groups]
        key = "ABC".index(hard.answer_text[-1])
        assert sizes.index(max(sizes)) != key and top.index(max(top)) != key


def test_money_compare_easy_often_has_more_coins_worth_less():
    traps = 0
    for seed in range(200):
        fig = make("money_compare_collections", seed=seed, difficulty="easy").figure
        a, b = fig["groups"]
        ta, tb = early.money_cents(fig, 0), early.money_cents(fig, 1)
        if ta != tb and (ta > tb) != (len(a["items"]) > len(b["items"])):
            traps += 1
    assert traps >= 80


def test_money_change_tiers():
    for seed in SEEDS:
        easy = _q("f10-money-change-from-note", "easy", seed)
        assert len(easy.figure["groups"][0]["items"]) == 1 and ".50" not in easy.text
        med = _q("f10-money-change-from-note", "medium", seed)
        assert ".50" in med.text and len(med.figure["groups"][0]["items"]) == 1
        hard = _q("f10-money-change-from-note", "hard", seed)
        assert len(hard.figure["groups"][0]["items"]) >= 2


def test_hundred_chart_tiers():
    for seed in SEEDS:
        easy = _q("f10-hundred-chart-patterns", "easy", seed).figure
        assert len(easy["hidden"]) == 1 and easy["shaded"] == easy["hidden"] and not easy["marked"]
        med = _q("f10-hundred-chart-patterns", "medium", seed).figure
        assert len(med["marked"]) == 1 and not med["shaded"] and len(med["hidden"]) >= 5
        hard = _q("f10-hundred-chart-patterns", "hard", seed).figure
        assert len(hard["shaded"]) >= 4 and len(hard["hidden"]) == 1 and hard["hidden"][0] in hard["shaded"]
    starts = {_q("f10-hundred-chart-patterns", "hard", s).figure["start"] for s in SEEDS}
    assert starts == {0, 1}


def test_hundred_chart_hard_pattern_is_not_the_next_number():
    for seed in SEEDS:
        q = _q("f10-hundred-chart-patterns", "hard", seed)
        visible = sorted(set(q.figure["shaded"]) - set(q.figure["hidden"]))
        step = min(b - a for a, b in zip(visible, visible[1:]))
        assert step >= 2


def test_fraction_name_tiers():
    for seed in SEEDS:
        easy = _q("f10-fraction-shaded-name", "easy", seed).figure["models"][0]
        assert easy["parts"] <= 4 and easy["shaded"] == list(range(len(easy["shaded"])))
        med = _q("f10-fraction-shaded-name", "medium", seed).figure["models"][0]
        assert 5 <= med["parts"] <= 8
        hard = _q("f10-fraction-shaded-name", "hard", seed)
        assert (
            hard.figure["models"][0]["parts"] >= 8
            and "not" in hard.text
            or "unshaded" in hard.text
            or "no shading" in hard.text
        )
        m = hard.figure["models"][0]
        assert hard.answer_text == f"{m['parts'] - len(m['shaded'])}/{m['parts']}"


def test_fraction_compare_tiers_and_traps():
    from fractions import Fraction

    for seed in SEEDS:
        easy = _q("f10-fraction-compare-models", "easy", seed).figure["models"]
        assert easy[0]["parts"] == easy[1]["parts"]
        med = _q("f10-fraction-compare-models", "medium", seed).figure["models"]
        assert len(med[0]["shaded"]) == len(med[1]["shaded"]) == 1 and med[0]["parts"] != med[1]["parts"]
        q = _q("f10-fraction-compare-models", "hard", seed)
        a, b = q.figure["models"]
        assert a["parts"] != b["parts"]
        va, vb = Fraction(len(a["shaded"]), a["parts"]), Fraction(len(b["shaded"]), b["parts"])
        if va != vb:  # more shaded parts must not pick out the larger fraction
            assert len(a["shaded"]) == len(b["shaded"]) or ((len(a["shaded"]) > len(b["shaded"])) != (va > vb))
    keys = {_q("f10-fraction-compare-models", "hard", s).answer_text for s in SEEDS}
    assert keys == {"Model A", "Model B", "Same amount"}


def test_base_ten_tiers():
    for seed in SEEDS:
        easy = _q("f10-base-ten-blocks-number", "easy", seed).figure
        assert easy["hundreds"] == 0 and easy["tens"] >= 1
        med = _q("f10-base-ten-blocks-number", "medium", seed).figure
        assert med["hundreds"] >= 1 and med["tens"] <= 9 and med["ones"] <= 9
        hard = _q("f10-base-ten-blocks-number", "hard", seed).figure
        assert hard["ones"] >= 10 or hard["tens"] >= 10
    zeros = sum(
        1 for s in SEEDS if 0 in (_q("f10-base-ten-blocks-number", "medium", s).figure[k] for k in ("tens", "ones"))
    )
    assert zeros >= 15


def test_converted_tiers():
    for seed in SEEDS:
        assert _q("count-to-5", "easy", seed).figure["layout"] == "scatter"
        assert 1 <= len(_q("count-to-5", "easy", seed).figure["dots"]) <= 5
        assert 4 <= len(_q("count-to-10", "easy", seed).figure["dots"]) <= 10
        assert _q("f10-subitise-name-to-5", "easy", seed).figure["layout"] == "dice"
        assert _q("f10-subitise-name-to-5", "medium", seed).figure["layout"] == "scatter"


def test_text_never_states_the_counted_number_for_counting():
    for seed in SEEDS:
        for sid in ("count-to-5", "count-to-10", "f10-subitise-name-to-5"):
            q = _q(sid, "easy", seed)
            assert not any(ch.isdigit() for ch in q.text)


# ------------------------------------------------------------------ the validators


def _bad(spec):
    return figures.validate(spec)


def test_money_validator():
    ok = early.money(collections=[["5c", "$2"]], alt="coins")
    assert figures.validate(ok) == []
    assert _bad({**ok, "groups": []})
    spec = {
        "kind": "money",
        "alt": "x",
        "groups": [{"label": "", "items": [{"denom": "7c", "type": "coin", "cents": 7}]}],
    }
    assert any("unknown denomination" in p for p in _bad(spec))
    spec["groups"][0]["items"] = [{"denom": "$5", "type": "coin", "cents": 500}]
    assert any("do not match" in p for p in _bad(spec))
    spec["groups"][0]["items"] = [{"denom": "5c", "type": "coin", "cents": 5}] * 13
    assert any("1 to 12" in p for p in _bad(spec))
    two = {
        "kind": "money",
        "alt": "x",
        "groups": [{"label": "", "items": [{"denom": "5c", "type": "coin", "cents": 5}]}] * 2,
    }
    assert any("label" in p for p in _bad(two))


def test_money_texts_include_every_coin_so_a_total_equal_to_one_coin_is_caught():
    fig = early.money(collections=[["20c"]], alt="a coin")
    with pytest.raises(AssertionError):
        figures.assert_no_leak(fig, ["20"])
    figures.assert_no_leak(early.money(collections=[["10c", "10c"]], alt="coins"), ["20"])


def test_hundred_chart_validator():
    ok = early.hundred_chart(alt="a chart", hidden=[5, 40], shaded=[5], marked=[40])
    assert figures.validate(ok) == []
    assert any("outside" in p for p in _bad({**ok, "hidden": [101]}))
    assert any("outside" in p for p in _bad({**ok, "start": 0, "hidden": [100]}))
    assert any("start" in p for p in _bad({**ok, "start": 2}))
    assert any("entirely hidden" in p for p in _bad({**ok, "hidden": list(range(1, 11))}))
    assert any(
        "more than 30" in p for p in _bad({**ok, "hidden": [r * 10 + c + 1 for r in range(5) for c in range(0, 8)]})
    )
    assert any("repeats" in p for p in _bad({**ok, "shaded": [3, 3]}))
    shown = figures.labels(ok)
    assert "5" not in shown and "40" not in shown and "6" in shown


def test_fraction_model_validator():
    shape = early.fraction_shape("bar", 4, [0])
    assert figures.validate(early.fraction_model(models=[shape], alt="a bar")) == []
    for bad, needle in [
        (early.fraction_shape("circle", 7, [0]), "circle parts"),
        (early.fraction_shape("rectangle", 6, [0], rows=2, cols=2), "rows * cols"),
        (early.fraction_shape("bar", 4, [0, 1, 2, 3]), "at least one unshaded"),
        (early.fraction_shape("bar", 4, []), "at least one part"),
        (early.fraction_shape("bar", 4, [4]), "from 0 to parts-1"),
        (early.fraction_shape("bar", 1, [0]), "from 2 to 12"),
        (early.fraction_shape("triangle", 4, [0]), "shape must be"),
    ]:
        problems = _bad({"kind": "fraction_model", "alt": "x", "models": [bad]})
        assert any(needle in p for p in problems), (needle, problems)
    a, b = early.fraction_shape("bar", 4, [0], label="A"), early.fraction_shape("circle", 4, [0], label="B")
    assert any("same shape" in p for p in _bad({"kind": "fraction_model", "alt": "x", "models": [a, b]}))
    c = early.fraction_shape("bar", 4, [0], label="C")
    assert any("labelled A and B" in p for p in _bad({"kind": "fraction_model", "alt": "x", "models": [a, c]}))


def test_counters_validator():
    ok = early.counters(layout="ten_frame", alt="a ten frame", frames=[[1, 1, 2, 0, 0, 0, 0, 0, 0, 0]])
    assert figures.validate(ok) == []
    assert early.counters_total(ok) == 3
    assert figures.validate(early.counters(layout="array", alt="a", rows=3, cols=4)) == []
    assert early.counters_total(early.counters(layout="array", alt="a", rows=3, cols=4)) == 12
    assert any("1 to 10" in p for p in _bad({**early.counters(layout="array", alt="a", rows=3, cols=4), "rows": 11}))
    assert any(
        "ten frames" in p
        for p in _bad({"kind": "counters", "alt": "a", "layout": "ten_frame", "shape": "circle", "frames": []})
    )
    assert any("10 boxes" in p for p in _bad({**ok, "frames": [[1, 1]]}))
    assert any("layout" in p for p in _bad({**ok, "layout": "spiral"}))
    assert any("shape" in p for p in _bad({**ok, "shape": "heart"}))
    groups = early.counters(layout="groups", alt="g", groups=[3, 3, 0])
    assert early.counters_total(groups) == 6
    assert any("rings" in p for p in _bad({**groups, "groups": [10]}))
    dice = early.counters(layout="dice", alt="d", faces=[{"n": 6, "colour": 1}, {"n": 2, "colour": 2}])
    assert early.counters_total(dice) == 8
    assert any("faces" in p for p in _bad({**dice, "faces": [{"n": 7, "colour": 1}]}))
    spec = {
        "kind": "counters",
        "alt": "s",
        "layout": "scatter",
        "shape": "star",
        "dots": [{"at": [0.5, 0.5], "colour": 1}, {"at": [0.55, 0.5], "colour": 1}],
    }
    assert any("too close" in p for p in _bad(spec))
    spec["dots"][1]["at"] = [1.5, 0.5]
    assert any("unit square" in p for p in _bad(spec))


def test_scatter_dots_are_valid_in_every_arrangement():
    rng = random.Random(3)
    for arrangement, ns in (("tidy", range(1, 11)), ("free", range(1, 7)), ("clusters", range(2, 9))):
        for n in ns:
            for _ in range(30):
                dots = early.scatter_dots(rng, n, arrangement=arrangement)
                fig = early.counters(layout="scatter", alt="s", dots=dots)
                assert early.counters_total(fig) == n
    with pytest.raises(ValueError):
        early.scatter_dots(rng, 3, arrangement="bogus")


def test_base_ten_validator():
    ok = early.base_ten(hundreds=1, tens=14, ones=0, alt="blocks")
    assert figures.validate(ok) == [] and early.base_ten_value(ok) == 240
    assert any("at least one" in p for p in _bad({**ok, "hundreds": 0, "tens": 0}))
    assert any("tens" in p for p in _bad({**ok, "tens": 21}))
    assert any("hundreds" in p for p in _bad({**ok, "hundreds": 10}))


def test_every_new_kind_is_registered_and_draws_its_text_through_labels():
    for kind in ("money", "hundred_chart", "fraction_model", "counters", "base_ten"):
        assert kind in figures.kinds()
    assert "A" in figures.labels(
        early.fraction_model(
            models=[early.fraction_shape("bar", 4, [0], label="A"), early.fraction_shape("bar", 4, [1], label="B")],
            alt="two bars",
        )
    )


def test_registry_has_every_topic():
    for _sid, (topic, *_rest) in ALL.items():
        assert topic in REGISTRY
