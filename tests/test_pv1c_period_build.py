"""Build test for the period-of-y=sin(bx) skill (ACMMM038)."""

from __future__ import annotations

import functools
import re

import pytest

import mathgen
from mathgen.grading.typed import grade
from mathgen.skills import load_skills
from mathgen import figures
from mathgen.core import VerificationError
from mathgen.generators import pv1c_period_build as gen
from mathgen.text_quality import prompt_shape

SKILL_ID = "senior-examine-period-changes-and-the-graphs-of-y"
TOPIC = "pv1c_period"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _make(tier, seed):
    src = next(s for s in _skill().sources if s.difficulty == tier)
    return mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _ok(q, given):
    return grade("expression", q, given).correct


def _star(display):
    return re.sub(r"(\d)pi", r"\1*pi", display)


def test_skill_shape():
    s = _skill()
    assert s.status == "deferred" and s.needs_figure
    assert s.capability_gates == ["figure_stimulus"]
    assert s.placement_eligible is False
    assert s.answer == "expression"
    assert [x.difficulty for x in s.sources] == TIERS
    assert [x.stages for x in s.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert len(s.sources) == 4 and s.mental is None
    assert {m.id for m in s.misconceptions} == {"half-period", "double-period", "quarter-period", "frequency-b"}


def test_reader_registered():
    assert figures.get_reader(TOPIC) is gen.read_period


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
        figures.assert_no_leak(q.figure, [q.answer_text, q.answer, *q.distractors])
        assert q.figure["alt"] == "the graph of a sine curve on a coordinate grid"
        assert q.figure["x_tick_format"] == "pi" and q.figure["equal_scale"] is False
        assert q.figure["points"] == [] and q.figure["lines"] == [] and q.figure["segments"] == []
        assert len(q.figure["curves"]) == 1 and "label" not in q.figure["curves"][0]
        assert 2 <= sum(len(p) for c in q.figure["curves"] for p in c["pieces"]) <= 200


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_spellings_rejects_distractors_and_wrong_value(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert _ok(q, q.answer_text) and _ok(q, q.answer)
        assert _ok(q, q.answer_text.replace("pi", "π"))
        assert _ok(q, " " + q.answer_text.replace("pi", " pi") + " ")
        for d in q.distractors:
            assert not _ok(q, d), (tier, seed, d)
        assert not _ok(q, "5"), (tier, seed)  # a bare number, no pi
        assert not _ok(q, "0.6667")
        assert not _ok(q, _star(q.answer_text) + "+1")
        assert not _ok(q, "7*pi/5")


def test_grader_spellings_of_two_pi_over_three():
    q = None
    for seed in range(200):
        cand = _make("easy", seed)
        if cand.answer_text == "2pi/3":
            q = cand
            break
    assert q is not None
    for ok in ("2pi/3", "2π/3", "(2/3)pi", "2*pi/3"):
        assert _ok(q, ok), ok
    for bad in ("2/3", "4pi/3", "pi/3"):
        assert not _ok(q, bad), bad


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_and_enough(tier):
    counts = []
    for seed in range(200):
        q = _make(tier, seed)
        vals = [q.answer_text, *q.distractors]
        assert len(set(vals)) == len(vals)
        assert all(re.fullmatch(r"-?(\d+)?pi(/\d+)?|-?\d+(/\d+)?", v) for v in vals), vals
        assert any("pi" in d for d in q.distractors)  # one with the key's shape
        assert "*" not in q.text and "*" not in q.answer_text
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
    assert max(counts.values()) <= 0.25 * 200, counts


@pytest.mark.parametrize("tier", TIERS)
def test_in_generator_check_fires_on_a_corrupted_answer(tier, monkeypatch):
    real = gen._sine_curve

    def wrong_frequency(spec):  # the drawn curve no longer has the keyed period
        spec2 = gen.Spec(spec.variant, spec.k * 2, spec.sign, spec.lo, spec.hi, spec.x_grid, spec.step, spec.b)
        return real(spec2)

    monkeypatch.setattr(gen, "_sine_curve", wrong_frequency)
    with pytest.raises(VerificationError):
        for seed in range(10):
            _make(tier, seed)


# ------------------------------------------------ the properties checked here
def _falls_first(q):
    pts = sorted(tuple(p) for c in q.figure["curves"] for piece in c["pieces"] for p in piece)
    return pts[1][1] < 0


def _neg_share(tier, n=200):
    return sum(_falls_first(_make(tier, s)) for s in range(n)) / n


@pytest.mark.parametrize("tier", ["hard", "exam"])
def test_curves_fall_first_on_about_half_the_draws(tier):
    assert 0.35 <= _neg_share(tier) <= 0.65


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_easy_and_medium_always_rise_first(tier):
    assert _neg_share(tier, 100) == 0


def test_property_is_red_against_the_unconstrained_draw(monkeypatch):
    """With the falls-first share lifted to zero the property fails: the share is the constraint."""
    monkeypatch.setattr(gen, "_NEG_SHARE", 0.0)
    assert not (0.35 <= _neg_share("hard") <= 0.65)
    assert not (0.35 <= _neg_share("exam") <= 0.65)


def test_example_five_pi_over_six_is_never_a_key():
    for tier in TIERS:
        for seed in range(200):
            assert _make(tier, seed).answer_text != "5pi/6"


def test_frequency_distractor_never_equals_the_key_display():
    for tier in TIERS:
        for seed in range(200):
            q = _make(tier, seed)
            assert all(d != q.answer_text for d in q.distractors)


def test_medium_prompt_never_mentions_peaks():
    for seed in range(200):
        t = _make("medium", seed).text.lower()
        assert not any(w in t for w in ("peak", "highest", "maximum", "trough"))


def test_medium_prompt_states_which_zeros_define_a_cycle():
    for seed in range(200):
        q = _make("medium", seed)
        for t in (q.text, q.latex):
            low = " ".join(t.lower().split())
            assert "full cycle" in low and "upward" in low and "next" in low, t
            assert "half" not in low


def test_exam_window_is_zero_to_four_pi_and_shows_at_most_one_cycle():
    for seed in range(200):
        q = _make("exam", seed)
        assert q.figure["x_range"] == [0, 4]
        assert q.answer_text in {"4pi", "6pi", "8pi", "10pi", "12pi", "14pi"}


def test_hard_window_is_zero_to_two_pi():
    for seed in range(200):
        assert _make("hard", seed).figure["x_range"] == [0, 2]


def test_prompts_have_no_python_notation_and_latex_twin_has_pi_in_maths():
    for tier in TIERS:
        for seed in range(40):
            q = _make(tier, seed)
            assert "**" not in q.text and "*" not in q.text
            assert "pi" not in re.sub(r"\\\(.*?\\\)", "", q.latex)
