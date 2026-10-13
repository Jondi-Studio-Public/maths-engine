"""Build test for the amplitude skill (ACMMM037, y = a sin x)."""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import pytest

import mathgen
from mathgen import figures
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1c_amplitude_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-amplitude-changes-and-the-graphs-of-y"
TOPIC = "pv1c_amplitude"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _grade(q, given):
    return GRADERS["fraction"](q, given).correct


def _value(q):
    return Fraction(q.answer_text)


def test_skill_shape():
    s = _skill()
    assert s.status == "active" and s.deferred_reason is None and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert s.placement_eligible is False
    assert s.answer == "fraction"
    assert s.mental is None
    assert len(s.sources) == 4
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {m.id for m in s.misconceptions} == {"sign-flipped", "peak-to-trough", "off-by-one-up", "off-by-one-down"}


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_amplitude


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic(tier):
    for seed in SEEDS:
        assert _make(tier, seed).to_json() == _make(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_figure_valid_and_no_leak(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert q.figure is not None
        assert figures.validate(q.figure) == []
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert q.figure["alt"] == "the graph of a sine curve on a coordinate grid"
        assert q.figure["points"] == [] and q.figure["lines"] == []
        assert 2 <= sum(len(p) for c in q.figure["curves"] for p in c["pieces"]) <= 200
        assert "amplitude" not in q.text.lower() and "amplitude" not in q.latex.lower()
        assert "**" not in q.text and "*" not in q.text


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_key_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert _grade(q, q.answer_text)
        assert _grade(q, f"{int(_value(q))}.0")
        for d in q.distractors:
            assert not _grade(q, d), (tier, seed, d)
        assert not _grade(q, str(float(_value(q)) + 1))
        assert not _grade(q, str(float(_value(q)) + 100))


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(re.fullmatch(r"-?\d+", v) for v in vals)
        counts.append(len(q.distractors))
    assert sum(c >= 2 for c in counts) >= 0.85 * len(counts)


@pytest.mark.parametrize("tier", TIERS)
def test_reader_agrees_with_key(tier):
    reader = figures.get_reader(TOPIC)
    for seed in range(1000):
        q = _make(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    qs = [_make(tier, seed) for seed in range(60)]
    assert len({q.text for q in qs}) >= 10
    assert len({(q.text, q.answer_text, repr(q.figure["curves"])) for q in qs}) >= 20
    assert len({prompt_shape(q) for q in qs}) >= 3


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration(tier):
    counts = {}
    for seed in range(200):
        a = _make(tier, seed).answer_text
        counts[a] = counts.get(a, 0) + 1
    assert max(counts.values()) <= 0.25 * 200


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._sample_fn

    def wrong_curve(a):  # the drawn curve no longer has the key's a
        return real(a + 1 if a != -1 else a + 2)

    monkeypatch.setattr(gen, "_sample_fn", wrong_curve)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the property checked here
def _lobe_direction(q):
    """+1 if the curve rises first from the left edge of the window, else -1."""
    fig = q.figure
    x_lo = fig["x_range"][0]
    return 1 if next(y for x, y in fig["curves"][0]["pieces"][0] if x == x_lo + 0.5) > 0 else -1


def _cue_pairs(tier, n=300):
    return {(_lobe_direction(_make(tier, s)), _make(tier, s).answer_text.startswith("-")) for s in range(n)}


@pytest.mark.parametrize("tier", ["hard", "exam"])
def test_first_lobe_direction_does_not_decide_the_sign(tier):
    """Odd and even window starts both occur, so a rising first lobe occurs with a < 0 and a
    falling one with a > 0: the learner must use the x labels."""
    assert len(_cue_pairs(tier)) == 4


@pytest.mark.parametrize("tier,pool", [("hard", "_S_HARD"), ("exam", "_S_EXAM")])
def test_property_is_red_against_the_unconstrained_draw(tier, pool, monkeypatch):
    """With only even window starts the lobe direction fixes the sign: the constraint is real."""
    monkeypatch.setattr(gen, pool, tuple(s for s in getattr(gen, pool) if s % 2 == 0))
    assert len(_cue_pairs(tier)) == 2


def test_signs_and_origin_by_tier():
    assert all(not _make("easy", s).answer_text.startswith("-") for s in range(200))
    for tier in ("medium", "hard", "exam"):
        neg = sum(_make(tier, s).answer_text.startswith("-") for s in range(200))
        assert 40 <= neg <= 160, (tier, neg)
    for tier in ("hard", "exam"):
        for s in range(100):
            lo, hi = _make(tier, s).figure["x_range"]
            assert lo > 0 or hi < 0  # x = 0 is not in the window
            assert hi - lo == (3 if tier == "hard" else 1)


def test_prompts_state_the_scale_and_name_the_family():
    for tier in TIERS:
        for seed in range(40):
            q = _make(tier, seed)
            assert "y = a sin x" in q.text and "π" in q.text and "1 unit" in q.text
            assert r"{y = a \sin x}" in q.latex and r"{\pi}" in q.latex
