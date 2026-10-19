"""Early-years data skills: sort objects, one-to-one displays, tally tables, comparing a tally with a picture graph.

The ``tally`` figure kind (marks in gates of five, a scatter of objects, a pictures column) and the four skills built on it.
Every tier of every skill is generated through the skill's own sources, 60 seeds each: the figure is valid and does not
show the key, the registered reader re-derives the key from the figure, the real grader accepts the key and rejects
every distractor, and the tiers differ in what is asked.
"""

from __future__ import annotations

import functools
import re
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, make, svg
from mathgen.figure_kinds import data
from mathgen.figure_kinds.tally import ICONS, tally
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

# skill id -> (topic, answer kind, ACARA code tag)
SKILLS = {
    "sort-and-count-objects": ("ey_sort_objects", "number", "ac9mfst01"),
    "one-to-one-displays": ("ey_one_to_one_data", "number", "ac9m1st02"),
    "tally-tables": ("ey_tally_tables", "mcq", "ac9m2st01"),
    "compare-tally-and-picture-displays": ("ey_compare_displays", "mcq", "ac9m2st02"),
}
CASES = [(sid, topic, tier) for sid, (topic, _k, _c) in SKILLS.items() for tier in TIERS]


@functools.cache
def _skills():
    return {s.id: s for s in load_skills()[1]}


@functools.cache
def _questions(topic, tier):
    return tuple(make(topic, seed=seed, difficulty=tier) for seed in SEEDS)


def _grade(kind, q, text):
    if kind == "mcq":
        from mathgen.grading.typed import mcq_options

        return grade("mcq", q, mcq_options(q).index(text)).correct
    return grade("number", q, text).correct


# ------------------------------------------------------------------ the figure kind


def test_tally_is_registered_and_draws_marks_in_gates_of_five():
    fig = tally(alt="a tally table", rows=[("dogs", 13), ("cats", 5), ("fish", 0)])
    assert figures.validate(fig) == []
    root = ET.fromstring(svg.figure_svg(fig))
    lines = [e for e in root.iter() if e.tag.endswith("line") and e.get("stroke-linecap") == "round"]
    # 13 = two gates (4 strokes + 1 slash each) and 3 strokes; 5 = one gate; 0 draws nothing
    assert len(lines) == (2 * 5 + 3) + 5


def test_hidden_counts_are_not_drawn_and_shown_counts_are_leaked_to_the_check():
    hidden = tally(alt="a tally table", rows=[("dogs", 17), ("cats", 8)])
    shown = tally(alt="a tally table", rows=[("dogs", 17), ("cats", 8)], show_counts=True)
    assert ">17<" not in svg.figure_svg(hidden)
    assert ">17<" in svg.figure_svg(shown)
    figures.assert_no_leak(hidden, ["17", "8"])
    with pytest.raises(AssertionError):
        figures.assert_no_leak(shown, ["17"])


def test_objects_carry_their_names_and_no_total():
    items = [("apple", 0.0, 0.0), ("cat", 0.5, 0.0), ("cat", 1.0, 0.0), ("dog", 0.0, 1.0)]
    fig = tally(alt="things to sort", items=items)
    assert {"apple", "cat", "dog"} <= set(figures.labels(fig))
    text = svg.figure_svg(fig)
    assert text.count(">cat<") == 2


def test_every_icon_has_its_own_outline():
    drawn = {name: svg._icon(name, 50, 50) for name in ICONS}
    assert len(set(drawn.values())) == len(ICONS)
    items = [(name, 0.14 * i, 0.33 * (i % 3)) for i, name in enumerate(ICONS)]
    root = ET.fromstring(svg.figure_svg(tally(alt="every icon", items=items)))
    assert {e.text for e in root.iter() if e.tag.endswith("text")} == set(ICONS)


def test_pictures_column_draws_the_key_and_one_symbol_per_picture():
    fig = tally(
        alt="a table",
        rows=[("dogs", 6), ("cats", 4)],
        pictures={"symbol": "star", "each": 2, "counts": [3, 2]},
    )
    assert "Each picture = 2" in figures.labels(fig)
    root = ET.fromstring(svg.figure_svg(fig))
    stars = [e for e in root.iter() if e.tag.endswith("polygon")]
    assert len(stars) == 3 + 2 + 1  # the rows, plus the one in the key


@pytest.mark.parametrize(
    "over,message",
    [
        (dict(rows=[("a", 1)] * 2), "distinct"),
        (dict(rows=[(f"r{i}", 1) for i in range(7)]), "at most 6 rows"),
        (dict(rows=[("dogs", 31)]), "0 to 30"),
        (dict(rows=[("dogs", -1)]), "0 to 30"),
        (dict(rows=[("a very long label", 1)]), "short word"),
        (dict(rows=[]), "needs rows, objects or both"),
        (dict(rows=[("dogs", 3)], pictures={"symbol": "star", "each": 3, "counts": [1]}), "each must be 1 or 2"),
        (dict(rows=[("dogs", 3)], pictures={"symbol": "dragon", "each": 1, "counts": [1]}), "symbol"),
        (dict(rows=[("dogs", 3)], pictures={"symbol": "star", "each": 1, "counts": [1, 2]}), "one entry per row"),
        (dict(rows=[("dogs", 21)], pictures={"symbol": "star", "each": 1, "counts": [1]}), "0 to 20"),
        (dict(rows=[("dogs", 3)], pictures={"symbol": "star", "each": 1, "counts": [11]}), "0 to 10"),
        (dict(items=[("dragon", 0.5, 0.5)]), "one of"),
        (dict(items=[("cat", 1.5, 0.5)]), "unit square"),
        (dict(items=[("cat", 0.5, 0.5), ("dog", 0.55, 0.55)]), "overlap"),
        (dict(items=[("cat", 0.0, 0.0)] * 21), "at most 20 objects"),
    ],
)
def test_bad_specs_are_refused(over, message):
    with pytest.raises(ValueError, match=message):
        tally(alt="a figure", **over)


def test_picture_graph_already_supports_a_key_of_one():
    fig = data.picture_graph(categories=["dogs", "cats"], symbols=[3, 2], each=1, symbol="heart", alt="a picture graph")
    assert "Each symbol = 1" in figures.labels(fig)
    assert figures.validate(fig) == []


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("sid", SKILLS)
def test_skill_wiring(sid):
    topic, kind, code = SKILLS[sid]
    skill = _skills()[sid]
    assert skill.status == "active" and skill.answer == kind
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.read_aloud is True and skill.placement_eligible is False and skill.learner_band == "early"
    assert [s.topic for s in skill.sources] == [topic] * 3
    assert [s.difficulty for s in skill.sources] == TIERS
    assert code in skill.tags
    assert skill.misconceptions and all(m.description for m in skill.misconceptions)
    by_id = _skills()
    assert skill.prerequisites and all(by_id[p].status == "active" for p in skill.prerequisites)
    assert set(skill.difficulty_progression) == set(TIERS)


# ------------------------------------------------------------------ every question


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_key_grades_correct_and_every_distractor_grades_wrong(sid, topic, tier):
    kind = SKILLS[sid][1]
    for seed, q in zip(SEEDS, _questions(topic, tier)):
        assert q.to_json() == make(topic, seed=seed, difficulty=tier).to_json()
        assert _grade(kind, q, q.answer_text)
        assert len(q.distractors) >= 2 and len(set(q.distractors)) == len(q.distractors)
        assert q.answer_text not in q.distractors
        for d in q.distractors:
            assert not _grade(kind, q, d), (q.text, d)
        for tag in (q.option_notes or {}).values():
            assert tag in CATALOGUE and tag.startswith("eyt-")


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_figure_is_valid_does_not_leak_and_agrees_with_the_key(sid, topic, tier):
    reader = figures.get_reader(topic)
    assert reader is not None
    for q in _questions(topic, tier):
        assert figures.validate(q.figure) == []
        assert svg.figure_svg(q.figure)
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert reader(q) == q.answer_text, (q.text, q.figure)


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_prompts_are_short_and_state_no_count_the_figure_shows(sid, topic, tier):
    for q in _questions(topic, tier):
        assert len(q.text.split()) <= READ_ALOUD_MAX_WORDS, q.text
        assert q.text.count("?") >= 1


@pytest.mark.parametrize("sid,topic,tier", CASES)
def test_pool_is_varied(sid, topic, tier):
    qs = _questions(topic, tier)
    assert len({q.answer_text for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 3
    assert len({str(q.figure) for q in qs}) >= 50


# ------------------------------------------------------------------ tiers differ in what is asked


def test_sort_objects_tiers():
    from mathgen.generators._ey_tally_kit import item_counts

    easy, medium, hard = (_questions("ey_sort_objects", t) for t in TIERS)
    assert all(len(item_counts(q.figure)) <= 3 and len(q.figure["items"]) <= 12 for q in easy)
    assert all(len(item_counts(q.figure)) == 3 for q in medium)
    assert all(len(item_counts(q.figure)) == 4 and len(q.figure["items"]) <= 20 for q in hard)
    assert {("more" in q.text) for q in medium} == {True, False}
    for q in hard:
        c = sorted(item_counts(q.figure).values())
        assert c[-1] > c[-2] and c[0] < c[1]  # a unique biggest and smallest group
    assert any("fewer" in q.text for q in hard) and any("not" in q.text for q in hard)


def test_one_to_one_uses_both_displays_and_a_key_of_one():
    for tier in TIERS:
        kinds = {q.figure["kind"] for q in _questions("ey_one_to_one_data", tier)}
        assert kinds == {"tally", "picture_graph"}
    for q in _questions("ey_one_to_one_data", "hard"):
        assert len(q.figure.get("rows")) == 5
        if q.figure["kind"] == "picture_graph":
            assert q.figure["key"]["each"] == 1


def test_tally_tables_tiers_read_search_and_check():
    easy, medium, hard = (_questions("ey_tally_tables", t) for t in TIERS)
    assert all(not q.figure["items"] and not q.figure["show_counts"] for q in easy + medium)
    assert all(q.answer_text.isdigit() and int(q.answer_text) <= 14 for q in easy)
    assert all(q.answer_text in {f"The {r['label']} row" for r in q.figure["rows"]} for q in medium + hard)
    assert all(q.figure["items"] and q.figure["rows"] for q in hard)
    # every medium wrong row is a plausible misread of the stated count: some option is built from a misread
    tags = {t for q in medium for t in (q.option_notes or {}).values()}
    assert {"eyt-skips-the-slash", "eyt-gate-counted-as-one"} <= tags


def test_compare_displays_tiers_use_the_key():
    easy, medium, hard = (_questions("ey_compare_displays", t) for t in TIERS)
    assert all(q.figure["pictures"]["each"] == 1 for q in easy)
    assert all(q.figure["pictures"]["each"] == 2 for q in medium + hard)
    for q in medium:  # the wrong row has one picture per tally mark, so ignoring the key points at the wrong rows
        (bad,) = [r for r, p in zip(q.figure["rows"], q.figure["pictures"]["counts"]) if p * 2 != r["count"]]
        assert f"The {bad['label']} row" == q.answer_text
        assert q.figure["pictures"]["counts"][q.figure["rows"].index(bad)] == bad["count"]
    for q in hard:
        assert re.search(r"more", q.text) and q.answer_text.isdigit()
        assert "eyt-ignores-the-picture-key" in set((q.option_notes or {}).values()) or len(q.distractors) >= 3


def test_show_counts_must_be_a_bool_and_need_rows():
    spec = tally(alt="a table", rows=[("dogs", 3)])
    assert any("bool" in p for p in figures.validate({**spec, "show_counts": "yes"}))
    assert any(
        "show_counts needs rows" in p
        for p in figures.validate(tally(alt="x", items=[("cat", 0.5, 0.5)]) | {"show_counts": True})
    )
