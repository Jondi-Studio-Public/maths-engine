"""Early-years "place the object" (``f10-place-objects-from-position-words``, topic ``ey_place_objects``, ``grid_cells``).

Each tier is generated through the skill's own sources at 60 seeds: the figure is valid, the registered reader re-derives
the key from the figure and prompt alone, an independent oracle (below) does the same from the figure's raw pictures, the
real grader accepts the key and rejects every distractor, and every prompt fits the read-aloud cap.
"""

from __future__ import annotations

import functools
import re

import pytest

from mathgen import figures, make, svg
from mathgen.generators import ey_place_objects_build as build
from mathgen.grading.typed import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import READ_ALOUD_MAX_WORDS, load_skills
from scripts import skill_contract

SKILL = "f10-place-objects-from-position-words"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
WORDS = {"house", "school", "church", "hospital", "tree", "lighthouse", "tent", "aeroplane"}
SYMBOL_WORD = {
    "house": "house",
    "school": "school",
    "church": "church",
    "hospital": "hospital",
    "tree": "tree",
    "lighthouse": "lighthouse",
    "campground": "tent",
    "airport": "aeroplane",
}


@functools.lru_cache(maxsize=1)
def _skill():
    return next(s for s in load_skills()[1] if s.id == SKILL)


@functools.lru_cache(maxsize=None)
def _q(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _cells(text):
    return {tuple(map(int, m)) for m in re.findall(r"\((\d+), (\d+)\)", text)}


def _oracle(q):
    """The squares to shade, worked out from the raw figure and the prompt with plain string handling."""
    pics = {(int(s["at"][0]), int(s["at"][1])): SYMBOL_WORD[s["symbol"]] for s in q.figure["symbols"]}
    spot = {w: c for c, w in pics.items()}
    step = {"above": (0, 1), "below": (0, -1), "left of": (-1, 0), "right of": (1, 0)}
    new: dict[str, tuple[int, int]] = {}
    cur = None  # the new picture the sentence is about
    alt = "|".join(sorted(WORDS))
    for clause in re.split(r"[.,:?]|\bthen\b", q.text.lower().replace("shade both", "")):
        words = re.findall(rf"\b(?:{alt})\b", clause)
        fresh = [w for w in words if w not in spot]
        if fresh and "just" not in clause.split(fresh[0])[0][-12:]:
            cur = fresh[0]
        if "between" in clause:
            a, b = spot[words[-2]], spot[words[-1]]
            new[cur] = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        for rel, vec in step.items():
            for m in re.finditer(rf"just {rel} the (\w+)", clause):
                base = new[m.group(1)] if m.group(1) in new else spot[m.group(1)]
                new[cur] = (base[0] + vec[0], base[1] + vec[1])
    return set(new.values())


def test_skill_wiring():
    skill = _skill()
    assert skill.status == "active" and skill.domain == "primary-geometry-time" and skill.answer == "grid_cells"
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False and skill.read_aloud is True and skill.learner_band == "early"
    assert [(s.topic, s.difficulty) for s in skill.sources] == [("ey_place_objects", t) for t in TIERS]
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"]]
    assert {"ac9mfsp02", "acara-f10-v9"} <= set(skill.tags)
    steps = [skill.difficulty_progression[t].steps for t in TIERS]
    assert steps == sorted(steps) and len(set(steps)) == 3


def test_every_new_misconception_is_listed_and_in_the_catalogue():
    listed = {m.id for m in _skill().misconceptions}
    assert {i for i, _ in build.MISCONCEPTIONS} <= listed <= set(CATALOGUE) | listed
    assert {i for i, _ in build.MISCONCEPTIONS} <= set(CATALOGUE)


@pytest.mark.parametrize("tier", TIERS)
def test_every_tagged_distractor_is_a_listed_misconception(tier):
    listed = {m.id for m in _skill().misconceptions}
    tags = set()
    for seed in SEEDS:
        q = _q(tier, seed)
        assert set(q.option_notes) <= set(q.distractors) and q.option_notes, (tier, seed)
        tags |= set(q.option_notes.values())
    assert tags <= listed and tags <= set(CATALOGUE)


@pytest.mark.parametrize("tier", TIERS)
def test_reader_and_oracle_agree_with_the_key_and_the_figure_is_valid(tier):
    reader = figures.get_reader("ey_place_objects")
    assert reader is not None
    for seed in SEEDS:
        q = _q(tier, seed)
        assert figures.validate(q.figure) == [], (tier, seed)
        assert reader(q) == q.answer_text, (tier, seed, q.text)
        assert _oracle(q) == _cells(q.answer_text), (tier, seed, q.text, q.answer_text)
        svg.figure_svg(q.figure)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade("grid_cells", q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) >= 3 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not grade("grid_cells", q, d).correct, (tier, seed, d, q.answer_text)
        # a shading that is only part of the key, or the key plus a square, must also fail
        cells = sorted(_cells(q.answer_text))
        if len(cells) == 2:
            assert not grade("grid_cells", q, f"({cells[0][0]}, {cells[0][1]})").correct
        extra = next(c for c in ((x, y) for x in range(5) for y in range(5)) if c not in cells)
        assert not grade("grid_cells", q, q.answer_text + f" ({extra[0]}, {extra[1]})").correct


@pytest.mark.parametrize("tier", TIERS)
def test_key_squares_are_empty_and_the_new_pictures_are_not_drawn(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        drawn = {(int(s["at"][0]), int(s["at"][1])) for s in q.figure["symbols"]}
        assert not (drawn & _cells(q.answer_text)), (tier, seed)
        named = re.findall(r"\b(" + "|".join(WORDS) + r")\b", q.text)
        shown = [SYMBOL_WORD[s["symbol"]] for s in q.figure["symbols"]]
        assert len(set(shown)) == len(shown)
        new = [w for w in named if w not in shown]
        assert len(set(new)) == (2 if tier == "hard" else 1), (tier, seed, q.text)
        assert q.figure["canvas"] == "grid"


@pytest.mark.parametrize("tier", TIERS)
def test_every_prompt_is_short_and_varied(tier):
    texts = {_q(tier, seed).text for seed in SEEDS}
    assert max(len(t.split()) for t in texts) <= READ_ALOUD_MAX_WORDS
    shapes = {re.sub(r"\b(" + "|".join(WORDS) + r"|above|below|left of|right of)\b", "_", t) for t in texts}
    assert len(shapes) >= 3, shapes
    assert len({repr(_q(tier, seed).figure) for seed in SEEDS}) >= 30
    assert len({_q(tier, seed).answer_text for seed in SEEDS}) >= 6


def test_tiers_escalate():
    for seed in SEEDS:
        easy, medium, hard = (_q(t, seed) for t in TIERS)
        eg, mg, hg = (q.figure["grid"] for q in (easy, medium, hard))
        assert sorted((eg["width"], eg["height"])) == [1, 5]
        assert eg["width"] * eg["height"] < mg["width"] * mg["height"] <= hg["width"] * hg["height"]
        assert len(_cells(easy.answer_text)) == len(_cells(medium.answer_text)) == 1
        assert len(_cells(hard.answer_text)) == 2


def test_hard_second_square_is_beside_the_first_not_a_drawn_picture():
    for seed in SEEDS:
        a, b = sorted(_cells(_q("hard", seed).answer_text))
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(family):
    skill_contract.FAMILIES[family](SKILL)


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make("ey_place_objects", seed=1, tier="bogus")
