"""Hard-tier fix + exam source for `pythagorean-hypotenuse` (`kinds=[side_hyp]`) and `pythagorean-leg`
(`kinds=[side_leg]`), both on the pinned `trigonometry` topic (`mathgen/generators/geometry.py`).

The independent oracle below (`_oracle_hard`/`_oracle_exam`) parses `question.text` alone and re-derives the
third side from scratch with plain `math`/`Fraction` -- it imports nothing from `mathgen.generators.geometry`,
so a sign/formatting slip in the generator's own `check` calls cannot also be baked into the check that is
supposed to catch it. `test_oracle_catches_a_corrupted_answer` proves the oracle is not vacuously true by
feeding it a deliberately wrong `answer_text` and confirming it disagrees.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance
from mathgen.text_quality import prompt_shape

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "trigonometry"
SEEDS = range(40)
SHAPE_SEEDS = range(60)


# --- independent oracle: hard tier (exact_triple=False, decimal answer) ----

_HARD_HYP_RE = re.compile(
    r"A right-angled triangle has legs of length (\d+) and (\d+)\. "
    r"Find the length of the hypotenuse, to 1 decimal place\."
)
_HARD_LEG_RE = re.compile(
    r"A right-angled triangle has hypotenuse (\d+) and one leg of length (\d+)\. "
    r"Find the other leg, to 1 decimal place\."
)


def _oracle_hard(kind: str, text: str) -> str:
    """Re-derive the rounded third side from the rendered prompt alone,
    using plain `math.sqrt` -- a route that never touches
    `mathgen.generators.geometry`, `sympy`, or the generator's own rounding
    guard."""
    if kind == "side_hyp":
        match = _HARD_HYP_RE.search(text)
        assert match, f"could not parse hard-tier side_hyp prompt: {text!r}"
        a, b = (int(v) for v in match.groups())
        third = math.sqrt(a * a + b * b)
    else:
        match = _HARD_LEG_RE.search(text)
        assert match, f"could not parse hard-tier side_leg prompt: {text!r}"
        c, a = (int(v) for v in match.groups())
        third = math.sqrt(c * c - a * a)
    return f"{round(third, 1):.1f}"


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_oracle_agrees(kind):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        expected = _oracle_hard(kind, q.text)
        assert expected == q.answer_text, (
            f"kind={kind} seed={seed}: oracle recomputed {expected} from the prompt, "
            f"generator answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- independent oracle: exam tier (word problem, exact integer answer) ----

_HYP_EXAM_RES = [
    re.compile(
        r"anchored to the ground (\d+) m from the base of the mast and attached "
        r"to the mast at a point (\d+) m above the ground"
    ),
    re.compile(
        r"zip-line tower is (\d+) m tall\. A support cable runs from the top of "
        r"the tower to an anchor point on the ground (\d+) m from the base"
    ),
    re.compile(
        r"drone is flying (\d+) m directly above a point on the ground that is "
        r"(\d+) m from its pilot"
    ),
]
_LEG_EXAM_RES = [
    re.compile(
        r"kite string, (\d+) m long, is anchored to the ground\. The kite is "
        r"flying directly above a point on the ground that is (\d+) m from the anchor"
    ),
    re.compile(
        r"zip-line cable, (\d+) m long, runs from a platform down to the ground\. "
        r"The point where the cable meets the ground is (\d+) m from the base"
    ),
    re.compile(r"cable is (\d+) m long and the anchor is (\d+) m from the base of the tower"),
]


def _oracle_exam(kind: str, text: str) -> str:
    """Re-derive the exact third side from the word-problem prompt alone.
    For side_hyp the two matched numbers are the two legs (find the
    hypotenuse); for side_leg the first matched number is the hypotenuse
    and the second is the given leg (find the other leg) -- matching the
    order each scenario states them in the prompt, independent of which
    generator variable name produced them."""
    if kind == "side_hyp":
        for pattern in _HYP_EXAM_RES:
            match = pattern.search(text)
            if match:
                leg1, leg2 = (int(v) for v in match.groups())
                third_sq = leg1 * leg1 + leg2 * leg2
                break
        else:
            raise AssertionError(f"could not parse exam-tier side_hyp prompt: {text!r}")
    else:
        for pattern in _LEG_EXAM_RES:
            match = pattern.search(text)
            if match:
                hyp, leg = (int(v) for v in match.groups())
                third_sq = hyp * hyp - leg * leg
                break
        else:
            raise AssertionError(f"could not parse exam-tier side_leg prompt: {text!r}")
    root = math.isqrt(third_sq)
    assert root * root == third_sq, f"exam tier third side is not exact: {third_sq} from {text!r}"
    return str(root)


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_oracle_agrees(kind):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        expected = _oracle_exam(kind, q.text)
        assert expected == q.answer_text, (
            f"kind={kind} seed={seed}: oracle recomputed {expected} from the prompt, "
            f"generator answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == len(SEEDS)


class _FakeQuestion:
    """Just enough of `Question` for `_oracle_hard`/`_oracle_exam` to run
    against a hand-built (possibly corrupted) prompt/answer pair."""

    def __init__(self, text, answer_text):
        self.text = text
        self.answer_text = answer_text


def test_oracle_catches_a_corrupted_answer():
    """Proves the oracle is not vacuously true (the `inverse_one_to_one_
    function` failure mode): feed it a real question's prompt paired with a
    deliberately wrong `answer_text` and confirm it disagrees, for both the
    hard-tier and exam-tier routes and both kinds."""
    q = make(TOPIC, seed=0, kinds=("side_hyp",), exact_triple=False)
    corrupted = f"{round(float(q.answer_text) + 5.0, 1):.1f}"
    assert _oracle_hard("side_hyp", q.text) != corrupted

    q = make(TOPIC, seed=0, kinds=("side_leg",), exact_triple=False)
    corrupted = f"{round(float(q.answer_text) + 5.0, 1):.1f}"
    assert _oracle_hard("side_leg", q.text) != corrupted

    q = make(TOPIC, seed=0, difficulty="exam", kinds=("side_hyp",))
    corrupted = str(int(q.answer_text) + 3)
    assert _oracle_exam("side_hyp", q.text) != corrupted

    q = make(TOPIC, seed=0, difficulty="exam", kinds=("side_leg",))
    corrupted = str(int(q.answer_text) + 3)
    assert _oracle_exam("side_leg", q.text) != corrupted


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_deterministic_across_seeds(kind):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        q2 = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_deterministic_across_seeds(kind):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        q2 = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        assert q1.to_json() == q2.to_json()


# --- distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_distractors_distinct_and_clear_stated_precision(kind):
    tol = stated_precision_tolerance(1)
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1
        answer_value = float(q.answer_text)
        for d in q.distractors:
            assert abs(float(d) - answer_value) > tol, (
                f"kind={kind} seed={seed}: distractor {d!r} is within the stated "
                f"1 d.p. tolerance of the answer {q.answer_text!r}"
            )


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_distractors_distinct_and_never_equal_answer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


# --- formatting consistency (no option is the only decimal/integer on offer)


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_every_option_is_a_1dp_decimal(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        for text in [q.answer_text, *q.distractors]:
            assert re.fullmatch(r"\d+\.\d", text), f"kind={kind} seed={seed}: option {text!r} is not X.Y"
        # Only the true answer is guarded against a trailing ".0" (the
        # generator's own resample loop); a distractor may legitimately
        # land on one (e.g. an un-rooted radicand happens to end in .0).
        assert not q.answer_text.endswith(".0"), (
            f"kind={kind} seed={seed}: answer {q.answer_text!r} displays as a whole "
            f"number, hiding the decimal reasoning step this tier exists to add"
        )


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_every_option_is_a_plain_integer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        for text in [q.answer_text, *q.distractors]:
            assert text.isdigit(), f"kind={kind} seed={seed}: option {text!r} is not a plain integer"


# --- answer-kind checks -------------------------------------------------


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_answer_is_a_float_rounded_to_1dp_and_off_the_boundary(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        assert isinstance(q.answer, float)
        assert q.answer == round(q.answer, 1)
        scaled = abs(q.answer) * 10
        frac = scaled - math.floor(scaled)
        assert abs(frac - 0.5) > 0.05, f"kind={kind} seed={seed}: answer sits on the rounding boundary"


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_answer_is_a_positive_integer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        assert isinstance(q.answer, int)
        assert q.answer > 0
        assert q.answer_text == str(q.answer)


# --- variety: at least 20 distinct rendered questions per (kind, tier) ------


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_at_least_20_distinct_questions_over_60_seeds(kind):
    texts = {make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False).text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"kind={kind}: only {len(texts)} distinct hard-tier questions over 60 seeds"


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_at_least_20_distinct_questions_over_60_seeds(kind):
    texts = {make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,)).text for seed in SHAPE_SEEDS}
    assert len(texts) >= 20, f"kind={kind}: only {len(texts)} distinct exam-tier questions over 60 seeds"


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_clears_three_distinct_shapes(kind):
    shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))) for seed in SHAPE_SEEDS}
    assert len(shapes) >= 3, f"kind={kind}: only {len(shapes)} distinct prompt shapes over 60 seeds"


# --- real grader accept/reject --------------------------------------------


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_real_grader_accepts_canonical_and_rejects_distractors(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_hard_tier_real_grader_rejects_a_deliberately_wrong_answer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, kinds=(kind,), exact_triple=False)
        wrong = round(q.answer + 5.0, 1)
        assert not grade_number(q, f"{wrong:.1f}").correct


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_real_grader_accepts_canonical_and_rejects_distractors(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct


@pytest.mark.parametrize("kind", ["side_hyp", "side_leg"])
def test_exam_tier_real_grader_rejects_a_deliberately_wrong_answer(kind):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", kinds=(kind,))
        wrong = q.answer + 7
        assert not grade_number(q, str(wrong)).correct


# --- default-params behaviour is untouched -----------


def test_default_params_side_hyp_side_leg_are_still_exact_triples():
    """`exact_triple` defaults True and `word_problem` defaults False, so a
    bare call (no skill's own hard/exam params) must still draw from
    TRIPLES and answer with a plain integer -- the pinned-hash contract
    this change must not disturb."""
    for seed in range(10):
        q = make(TOPIC, seed=seed, kinds=("side_hyp",))
        assert isinstance(q.answer, int)
        assert "to 1 decimal place" not in q.text
        q = make(TOPIC, seed=seed, kinds=("side_leg",))
        assert isinstance(q.answer, int)
        assert "to 1 decimal place" not in q.text


# --- skill contract -----------------------------------------------------


@pytest.mark.parametrize(
    "skill_id,kind",
    [("pythagorean-hypotenuse", "side_hyp"), ("pythagorean-leg", "side_leg")],
)
def test_skill_contract_has_four_sources_including_exam(skill_id, kind):
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id[skill_id]

    assert skill.answer == "number"
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    for source in skill.sources:
        assert source.params.get("kinds") == [kind]

    hard_source = skill.sources[2]
    assert hard_source.params.get("exact_triple") is False
    assert hard_source.stages == [5, 6, "maintenance"]

    exam_source = skill.sources[3]
    assert exam_source.stages == ["exam"]
    assert "exact_triple" not in exam_source.params
