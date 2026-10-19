"""Pattern strips (``pattern_strip``) and the skills that use them (``continue-patterns`` and
``f10-pattern-strip-repeating-unit``).

Every tier of every topic is generated 60 times: the figure is valid, does not show the key, the registered reader
re-derives the key from the figure, the real grader accepts the key and rejects every distractor, and every prompt is
short enough to read aloud. The kind's validator refuses bad specs, and the renderer draws every token and keeps
two styles apart by more than colour.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET

import pytest

from mathgen import figures, make
from mathgen.figure_kinds import pattern as pat
from mathgen.grading.typed import grade, mcq_options
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from mathgen.svg import figure_svg
from scripts import skill_contract

TIERS = ("easy", "medium", "hard")
SEEDS = range(60)
TOPICS = ("ey_pattern_continue", "ey_pattern_match", "ey_pattern_unit")
SKILLS = {
    "continue-patterns": ("ac9mfa01", ["ey_pattern_continue", "ey_pattern_continue", "ey_pattern_match"]),
    "f10-pattern-strip-repeating-unit": (
        "ac9m1a02",
        ["ey_pattern_unit", "ey_pattern_unit", "ey_pattern_match", "ey_pattern_unit"],
    ),
}
SVG = "{http://www.w3.org/2000/svg}"


@functools.lru_cache(maxsize=1)
def _skills():
    return {s.id: s for s in load_skills()[1]}


@functools.lru_cache(maxsize=None)
def _q(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


def _is_right(skill_id, q, text):
    return grade(_skills()[skill_id].answer, q, mcq_options(q).index(text)).correct


# ------------------------------------------------------------------ the topics


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_every_tier_draws_a_valid_figure_that_does_not_show_the_key(topic, tier):
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert q.figure is not None and q.figure["kind"] == "pattern_strip"
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert len(q.distractors) >= 2 and q.answer_text not in q.distractors


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_the_reader_re_derives_the_key_from_the_figure(topic, tier):
    reader = figures.get_reader(topic)
    assert reader is not None
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert reader(q) == q.answer_text, (seed, q.text)


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_is_short_enough_to_read_aloud(topic, tier):
    for seed in SEEDS:
        q = _q(topic, tier, seed)
        assert len(q.text.split()) <= READ_ALOUD_MAX_WORDS, q.text


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_unknown_tier_refused(topic, tier):
    for seed in range(10):
        assert _q(topic, tier, seed).to_json() == make(topic, seed=seed, difficulty=tier).to_json()
    with pytest.raises(ValueError):
        make(topic, seed=1, tier="bogus")


def test_continue_options_are_described_in_words_a_child_hears():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("ey_pattern_continue", tier, seed)
            for option in (q.answer_text, *q.distractors):
                assert option.split()[0] in ("a", "an", "the"), option
                assert not any(ch.isdigit() for ch in option.replace("number", "")) or "number" in option


def test_continue_tiers_escalate():
    easy = [pat.structure(pat.row_keys(_q("ey_pattern_continue", "easy", s).figure)) for s in SEEDS]
    assert set(easy) == {"AB"}
    medium = {pat.structure(pat.row_keys(_q("ey_pattern_continue", "medium", s).figure)) for s in SEEDS}
    assert medium == {"AAB", "ABB", "ABC"}
    gaps = [
        any(c["type"] == "blank" for c in _q("ey_pattern_continue", "medium", s).figure["rows"][0]["cells"])
        for s in SEEDS
    ]
    assert any(gaps) and not all(gaps)  # a gap in the middle on some seeds, a question mark at the end on others
    hard = [_q("ey_pattern_continue", "hard", s).figure for s in SEEDS]
    for fig in hard:
        shapes = {k.split(":")[1] for k in pat.row_keys(fig) if k}
        fills = {k.split(":")[2] for k in pat.row_keys(fig) if k}
        assert len(shapes) >= 2 and len(fills) >= 2  # two things change together


def test_continue_hard_offers_the_right_shape_in_the_wrong_fill():
    hits = 0
    for seed in SEEDS:
        q = _q("ey_pattern_continue", "hard", seed)
        shape = q.answer_text.split()[-1]
        hits += any(d.split()[-1] == shape for d in q.distractors)
    assert hits >= 40


def test_match_decoys_differ_in_structure_and_hard_has_a_row_of_the_models_own_pictures():
    for tier in TIERS:
        for seed in SEEDS:
            q = _q("ey_pattern_match", tier, seed)
            structs = [pat.structure(pat.row_keys(q.figure, i)) for i in range(3)]
            assert len(set(structs)) == 3, (tier, seed, structs)
            lengths = [len(r["cells"]) for r in q.figure["rows"]]
            key = "ABC".index(q.answer_text[-1])
            assert lengths.count(lengths[key]) > 1 or min(lengths) < lengths[key] < max(lengths)
    for seed in SEEDS:
        q = _q("ey_pattern_match", "hard", seed)
        model = {w for w in q.text.rstrip("?").replace(" like ", " as ").rsplit(" as ", 1)[1].split(", ")}
        wrong = [r for r in "ABC" if f"Row {r}" in q.distractors]
        drawn = [{c["type"] for c in q.figure["rows"]["ABC".index(r)]["cells"]} for r in wrong]
        assert {"action"} in drawn, (
            seed,
            q.text,
            model,
        )  # the look-alike row is made of the model's own kind of picture


def test_unit_tiers_ask_three_different_questions():
    easy = _q("ey_pattern_unit", "easy", 1)
    assert easy.answer_text.isdigit() and "repeat" in easy.text
    medium = _q("ey_pattern_unit", "medium", 1)
    assert medium.figure["numbered"] is True and "number" in medium.text
    hard = _q("ey_pattern_unit", "hard", 1)
    assert hard.answer_text.startswith("Number ")


def test_unit_easy_distractors_include_the_whole_strip_count():
    hits = 0
    for seed in SEEDS:
        q = _q("ey_pattern_unit", "easy", seed)
        total = str(len(q.figure["rows"][0]["cells"]))
        hits += total in q.distractors
    assert hits >= 50


def test_unit_medium_asks_past_the_end_of_the_strip():
    for seed in SEEDS:
        q = _q("ey_pattern_unit", "medium", seed)
        k = [int(w) for w in q.text.replace("?", "").split() if w.isdigit()][0]
        assert k >= len(q.figure["rows"][0]["cells"]) + 2


def test_unit_hard_mistake_is_unique_and_after_the_first_repeat():
    for seed in SEEDS:
        q = _q("ey_pattern_unit", "hard", seed)
        place = int(q.answer_text.split()[-1])
        assert pat.read_mistake(q.figure) == place
        assert place > 2


# ------------------------------------------------------------------ the grader


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id):
    for src in _skills()[skill_id].sources:
        for seed in SEEDS:
            q = _q(src.topic, src.difficulty, seed)
            assert _is_right(skill_id, q, q.answer_text), (src.topic, src.difficulty, seed)
            for d in q.distractors:
                assert not _is_right(skill_id, q, d), (src.topic, src.difficulty, seed, d, q.answer_text)


# ------------------------------------------------------------------ the figure kind


def _strip(cells, **kw):
    return {"kind": "pattern_strip", "alt": "a pattern", "numbered": False, "rows": [{"cells": cells, **kw}]}


def test_the_validator_accepts_every_token_and_refuses_bad_specs():
    ok = _strip(
        [
            pat.shape_cell("heart", "striped"),
            pat.text_cell("12"),
            pat.text_cell("Q"),
            pat.action_cell("jump"),
            pat.stack_cell(10),
            pat.blank_cell(),
            pat.mark_cell(),
        ],
        bracket=[0, 2],
    )
    assert figures.validate(ok) == []
    bad = [
        _strip([pat.shape_cell("hexagon", "solid")]),
        _strip([pat.shape_cell("circle", "dotted")]),
        _strip([pat.text_cell("1234")]),
        _strip([{"type": "text", "text": "ab"}]),
        _strip([pat.action_cell("sing")]),
        _strip([pat.stack_cell(11)]),
        _strip([pat.stack_cell(0)]),
        _strip([{"type": "ghost"}]),
        _strip([]),
        _strip([pat.text_cell("1")] * 17),
        _strip([pat.mark_cell(), pat.blank_cell(), pat.mark_cell(), pat.text_cell("1")]),
        _strip([pat.mark_cell()]),
        _strip([pat.text_cell("1"), pat.text_cell("2")], bracket=[1, 2]),
        _strip([pat.text_cell("1"), pat.text_cell("2")], bracket=[1, 0]),
        _strip([pat.text_cell("1")], label=""),
        {**_strip([pat.text_cell("1")]), "rows": []},
        {**_strip([pat.text_cell("1")]), "rows": [{"cells": [pat.text_cell("1")]}] * 4},
        {**_strip([pat.text_cell("1")]), "numbered": "yes"},
        {**_strip([pat.text_cell("1")]), "alt": ""},
    ]
    for spec in bad:
        assert figures.validate(spec), spec


def test_several_rows_need_distinct_labels():
    row = {"cells": [pat.text_cell("1"), pat.text_cell("2")]}
    spec = {"kind": "pattern_strip", "alt": "rows", "numbered": False}
    assert figures.validate({**spec, "rows": [{**row}, {**row}]})
    assert figures.validate({**spec, "rows": [{**row, "label": "A"}, {**row, "label": "A"}]})
    assert figures.validate({**spec, "rows": [{**row, "label": "A"}, {**row, "label": "B"}]}) == []


def test_the_texts_feed_the_leak_check():
    fig = pat.pattern_strip(
        alt="a pattern",
        numbered=True,
        rows=[{"label": "A", "cells": [pat.text_cell("7"), pat.action_cell("clap"), pat.mark_cell()]}],
    )
    shown = figures.labels(fig)
    assert {"A", "7", "clap", "?", "1", "2", "3"} <= set(shown)
    with pytest.raises(AssertionError):
        figures.assert_no_leak(fig, ["clap"])


def test_reading_functions():
    cells = [pat.text_cell(x) for x in "xyyxyyxy"]
    fig = pat.pattern_strip(alt="a", rows=[{"cells": cells + [pat.mark_cell()]}])
    assert pat.read_gap(fig) == "text:y"
    assert pat.read_nth(fig, 12) == "text:y" and pat.read_nth(fig, 10) == "text:x"
    assert pat.structure(pat.row_keys(fig)[:-1]) == "ABB"
    assert pat.describe_key("shape:circle:open") == "an open circle"
    assert pat.describe_key("shape:star:solid") == "a solid star"
    assert pat.describe_key("text:5") == "the number 5" and pat.describe_key("text:M") == "the letter M"
    assert pat.describe_key("action:clap") == "a clap"
    with pytest.raises(ValueError):
        pat.describe_cell(pat.blank_cell())


def test_a_gap_nothing_shows_and_an_ambiguous_mistake_are_refused():
    two_gaps = pat.pattern_strip(
        alt="a", rows=[{"cells": [pat.text_cell("1"), pat.text_cell("2"), pat.blank_cell(), pat.mark_cell()]}]
    )
    with pytest.raises(ValueError):
        pat.read_gap(two_gaps)
    two = pat.pattern_strip(alt="a", rows=[{"cells": [pat.text_cell(x) for x in "xyxyxy"]}])
    with pytest.raises(ValueError):
        pat.read_mistake(two)  # no mistake at all
    clean = pat.pattern_strip(alt="a", rows=[{"cells": [pat.text_cell(x) for x in "xyxyyyxy"]}])
    assert pat.read_mistake(clean) == 5


# ------------------------------------------------------------------ the renderer


def _draw(cells, **kw):
    return ET.fromstring(figure_svg(pat.pattern_strip(alt="a pattern", rows=[{"cells": cells, **kw}])))


def test_every_token_is_drawn_and_unknowns_are_shown_as_unknowns():
    root = _draw([pat.shape_cell(s, "solid") for s in pat.SHAPES] + [pat.blank_cell(), pat.mark_cell()])
    assert len(list(root.iter(SVG + "polygon"))) >= 3 and list(root.iter(SVG + "path"))
    assert [t.text for t in root.iter(SVG + "text")] == ["?"]
    assert any(r.get("stroke-dasharray") for r in root.iter(SVG + "rect"))  # the blank is a dashed box
    words = _draw([pat.action_cell(a) for a in pat.ACTIONS])
    assert [t.text for t in words.iter(SVG + "text")] == list(pat.ACTIONS)  # an icon is always labelled
    text = _draw([pat.text_cell("12"), pat.text_cell("Q")])
    assert [t.text for t in text.iter(SVG + "text")] == ["12", "Q"]
    stack = _draw([pat.stack_cell(10), pat.stack_cell(1)])
    assert len(list(stack.iter(SVG + "circle"))) == 11


def test_fills_differ_by_more_than_colour():
    def paint(fill):
        root = _draw([pat.shape_cell("circle", fill)])
        circle = next(c for c in root.iter(SVG + "circle"))
        return circle.get("fill"), root

    solid, _ = paint("solid")
    open_, _ = paint("open")
    striped, root = paint("striped")
    assert solid == "currentColor" and open_ == "#ffffff"
    assert striped.startswith("url(#") and list(root.iter(SVG + "pattern"))
    assert len({solid, open_, striped}) == 3


def test_a_bracket_and_numbers_are_drawn_when_asked():
    plain = figure_svg(pat.pattern_strip(alt="a", rows=[{"cells": [pat.text_cell("1")] * 4}]))
    marked = figure_svg(pat.pattern_strip(alt="a", rows=[{"cells": [pat.text_cell("1")] * 4, "bracket": [0, 1]}]))
    numbered = ET.fromstring(
        figure_svg(pat.pattern_strip(alt="a", numbered=True, rows=[{"cells": [pat.text_cell("9")] * 3}]))
    )
    assert marked.count("<polyline") == plain.count("<polyline") + 1
    assert [t.text for t in numbered.iter(SVG + "text")] == ["9", "9", "9", "1", "2", "3"]


def test_sixteen_cells_in_three_rows_fit_inside_the_canvas():
    row = lambda label: {"label": label, "cells": [pat.action_cell("clap")] * 16}  # noqa: E731
    root = ET.fromstring(figure_svg(pat.pattern_strip(alt="a", rows=[row("A"), row("B"), row("C")])))
    box = [float(v) for v in root.get("viewBox").split()]
    for rect in root.iter(SVG + "rect"):
        assert float(rect.get("x")) >= 0 and float(rect.get("x")) + float(rect.get("width")) <= box[2] + 0.01
        assert float(rect.get("y")) + float(rect.get("height")) <= box[3] + 0.01


# ------------------------------------------------------------------ the skills


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_wiring(skill_id):
    code, topics = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False and skill.learner_band == "early" and skill.read_aloud is True
    assert code in skill.tags and skill.answer == "mcq"
    assert [s.topic for s in skill.sources] == topics
    assert [s.difficulty for s in skill.sources] == sorted((s.difficulty for s in skill.sources), key=TIERS.index)
    assert len(skill.misconceptions) >= 3 and all(m.description for m in skill.misconceptions)
    assert set(skill.difficulty_progression) == set(TIERS)
    by_id = _skills()
    assert all(by_id[p].status == "active" for p in skill.prerequisites)
    assert skill.year_level == (("F",) if code == "ac9mfa01" else ("1",))


def test_the_old_continue_patterns_id_and_its_dependants_survive():
    skills = _skills()
    assert skills["f10-pattern-copy-repeat-unit"].prerequisites == ["continue-patterns"]
    assert "continue-patterns" in skills["f10-pattern-strip-repeating-unit"].prerequisites


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)
