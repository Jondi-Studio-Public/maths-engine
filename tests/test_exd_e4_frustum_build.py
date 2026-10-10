"""Constructibility values (l1, w1, l2, w2, h[, H_total, H_small]) are pulled
back out of the RENDERED question text/working via regex, the same
convention `test_t5_annulus_frustum_build.py`'s own parsers use -- never
read off the generator's internal locals -- so these tests exercise what a
learner is actually shown. `mathgen.core` helpers (`round_half_up`,
`stated_precision_string`) are shared, public formatting utilities, not
generator internals, and are used here the same way
`test_t5_annulus_frustum_build.py` imports `round_half_up` from
`mathgen.core` directly.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import (
    VerificationError,
    check_distinct,
    check_equivalent,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

TOPIC = "rectangular_frustum_volume"
TIERS = ["easy", "medium", "hard", "exam"]
EXACT_TIERS = ["easy", "medium", "hard"]
SEEDS_40 = range(40)
SEEDS_60 = range(60)


# ---------------------------------------------------------------------------
# Text parsers -- (l1, w1, l2, w2, h) per tier, derived from the rendered
# text alone. Every tier's four phrasings are matched by their own regex;
# the first that matches wins.
# ---------------------------------------------------------------------------

_DIRECT_PATTERNS = [
    (
        r"base measuring (\d+) cm by (\d+) cm and a rectangular top measuring "
        r"(\d+) cm by (\d+) cm, with a height of (\d+) cm",
        ("l1", "w1", "l2", "w2", "h"),
    ),
    (
        r"base (\d+) cm by (\d+) cm, narrowing to a top (\d+) cm by (\d+) cm "
        r"over a height of (\d+) cm",
        ("l1", "w1", "l2", "w2", "h"),
    ),
    (
        r"base of a rectangular frustum measures (\d+) cm by (\d+) cm and its "
        r"top measures (\d+) cm by (\d+) cm; the height between the two faces "
        r"is (\d+) cm",
        ("l1", "w1", "l2", "w2", "h"),
    ),
    (
        r"is (\d+) cm tall, with a base of (\d+) cm by (\d+) cm and a top of "
        r"(\d+) cm by (\d+) cm",
        ("h", "l1", "w1", "l2", "w2"),
    ),
]

_SCALE_FACTOR_PATTERNS = [
    (
        r"base (\d+) cm by (\d+) cm and a height of (\d+) cm\. Each side of the "
        r"top rectangle is (\d+)/(\d+) the length",
        ("l1", "w1", "h", "p", "q"),
    ),
    (
        r"is (\d+) cm tall, with a base of (\d+) cm by (\d+) cm\. The top "
        r"rectangle is scaled down from the base by a factor of (\d+)/(\d+)",
        ("h", "l1", "w1", "p", "q"),
    ),
    (
        r"narrows from a base of (\d+) cm by (\d+) cm to a top whose sides are "
        r"each (\d+) out of every (\d+) of the base's corresponding side, over "
        r"a height of (\d+) cm",
        ("l1", "w1", "p", "q", "h"),
    ),
    (
        r"base of a rectangular frustum measures (\d+) cm by (\d+) cm; the top "
        r"rectangle is similar to the base with a linear scale factor of "
        r"(\d+)/(\d+), and the height between the two faces is (\d+) cm",
        ("l1", "w1", "p", "q", "h"),
    ),
]

_APEX_HEIGHT_PATTERNS = [
    (
        r"base (\d+) cm by (\d+) cm and a top (\d+) cm by (\d+) cm\. If the "
        r"four sloped edges were extended to meet at the pyramid's apex, the "
        r"apex would be (\d+) cm above the base",
        ("l1", "w1", "l2", "w2", "H_total"),
    ),
    (
        r"base measuring (\d+) cm by (\d+) cm and a top measuring (\d+) cm by "
        r"(\d+) cm\. The full pyramid this frustum was cut from has a total "
        r"height \(base to apex\) of (\d+) cm",
        ("l1", "w1", "l2", "w2", "H_total"),
    ),
    (
        r"base of a rectangular frustum is (\d+) cm by (\d+) cm and its top is "
        r"(\d+) cm by (\d+) cm; extending its sloped sides to the original apex "
        r"gives a total height of (\d+) cm",
        ("l1", "w1", "l2", "w2", "H_total"),
    ),
    (
        r"base measures (\d+) cm by (\d+) cm and its top measures (\d+) cm by "
        r"(\d+) cm\. Extended to a full pyramid, the apex sits (\d+) cm above "
        r"the base plane",
        ("l1", "w1", "l2", "w2", "H_total"),
    ),
]

_APEX_HEIGHT_DECIMAL_PATTERNS = [
    (
        r"base (\d+) cm by (\d+) cm\. Each side of the top rectangle is "
        r"(\d+)/(\d+) the length of the corresponding side of the base\. "
        r"Extending the sloped edges to the pyramid's apex gives a total height "
        r"of (\d+) cm",
        ("l1", "w1", "p", "q", "H_total"),
    ),
    (
        r"cut from a base measuring (\d+) cm by (\d+) cm; the top rectangle is "
        r"scaled down by a factor of (\d+)/(\d+) in both directions, and the "
        r"full pyramid's apex is (\d+) cm above the base",
        ("l1", "w1", "p", "q", "H_total"),
    ),
    (
        r"has a base of (\d+) cm by (\d+) cm, with the top's sides each (\d+) "
        r"out of every (\d+) of the base's corresponding side\. The apex of "
        r"the original pyramid sits (\d+) cm above the base",
        ("l1", "w1", "p", "q", "H_total"),
    ),
    (
        r"base of a rectangular frustum measures (\d+) cm by (\d+) cm and its "
        r"top is similar to the base with a linear scale factor of (\d+)/(\d+)"
        r"\. Extended to the full pyramid, the apex is (\d+) cm above the base "
        r"plane",
        ("l1", "w1", "p", "q", "H_total"),
    ),
]


def _match_any(text: str, patterns) -> dict:
    for pattern, names in patterns:
        m = re.search(pattern, text)
        if m:
            return dict(zip(names, (int(g) for g in m.groups())))
    raise AssertionError(f"could not parse dimensions from: {text!r}")


def _dims_for_tier(tier: str, text: str) -> tuple[int, int, int, int, int]:
    """(l1, w1, l2, w2, h) for any tier, derived entirely from the rendered
    text -- deriving l2/w2/h where the tier does not state them directly,
    via the same similar-triangles relation the question's own working uses
    (never the generator's internal locals)."""
    if tier == "easy":
        d = _match_any(text, _DIRECT_PATTERNS)
        return d["l1"], d["w1"], d["l2"], d["w2"], d["h"]
    if tier == "medium":
        d = _match_any(text, _SCALE_FACTOR_PATTERNS)
        l1, w1, p, q, h = d["l1"], d["w1"], d["p"], d["q"], d["h"]
        assert (l1 * p) % q == 0 and (w1 * p) % q == 0, "top dims not exact"
        l2, w2 = l1 * p // q, w1 * p // q
        return l1, w1, l2, w2, h
    if tier == "hard":
        d = _match_any(text, _APEX_HEIGHT_PATTERNS)
        l1, w1, l2, w2, H_total = d["l1"], d["w1"], d["l2"], d["w2"], d["H_total"]
        H_small_num = H_total * l2
        assert H_small_num % l1 == 0, "H_small not exact"
        H_small = H_small_num // l1
        h = H_total - H_small
        return l1, w1, l2, w2, h
    if tier == "exam":
        d = _match_any(text, _APEX_HEIGHT_DECIMAL_PATTERNS)
        l1, w1, p, q, H_total = d["l1"], d["w1"], d["p"], d["q"], d["H_total"]
        assert (l1 * p) % q == 0 and (w1 * p) % q == 0, "top dims not exact"
        l2, w2 = l1 * p // q, w1 * p // q
        H_small_num = H_total * l2
        assert H_small_num % l1 == 0, "H_small not exact"
        H_small = H_small_num // l1
        h = H_total - H_small
        return l1, w1, l2, w2, h
    raise ValueError(f"unknown tier {tier!r}")


def _h_total_h_small(l1: int, l2: int, h: int) -> tuple[Fraction, Fraction]:
    """H_total/H_small re-derived from (l1, l2, h) alone via similar
    triangles: H_total = h*l1/(l1-l2), H_small = H_total - h. Works
    uniformly for every tier once (l1, l2, h) are known."""
    H_total = Fraction(h * l1, l1 - l2)
    H_small = H_total - h
    return H_total, H_small


def _fmt_for_tier(value: Fraction, tier: str) -> str:
    if tier == "exam":
        return stated_precision_string(round_half_up(value, 2), 2)
    assert value.denominator == 1, f"expected exact integer, got {value}"
    return str(int(value))


# ---------------------------------------------------------------------------
# The swept constructibility test -- a named hazard. Every seed,
# every tier, not a sample.
# ---------------------------------------------------------------------------


def test_top_strictly_inside_base_every_seed_every_tier():
    for tier in TIERS:
        for seed in SEEDS_60:
            q = make(TOPIC, seed=seed, difficulty=tier)
            l1, w1, l2, w2, h = _dims_for_tier(tier, q.text)
            assert l2 < l1, f"{tier} seed={seed}: l2={l2} not strictly less than l1={l1} ({q.text!r})"
            assert w2 < w1, f"{tier} seed={seed}: w2={w2} not strictly less than w1={w1} ({q.text!r})"
            assert l2 * w1 == l1 * w2, (
                f"{tier} seed={seed}: scale ratio not uniform (l2*w1={l2 * w1} != l1*w2={l1 * w2}) ({q.text!r})"
            )
            assert h > 0, f"{tier} seed={seed}: h={h} not strictly positive ({q.text!r})"


# ---------------------------------------------------------------------------
# Determinism, distractors, real grader, answer precision, distinct
# questions, uniform formatting -- required for every build topic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_determinism_across_seeds(tier):
    for seed in SEEDS_40:
        q1 = make(TOPIC, seed=seed, difficulty=tier)
        q2 = make(TOPIC, seed=seed, difficulty=tier)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_within_tolerance(tier):
    tolerance = stated_precision_tolerance(2) if tier == "exam" else Fraction(0)
    for seed in SEEDS_40:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.distractors, f"{tier} seed={seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        answer_val = Fraction(q.answer_text)
        for d in q.distractors:
            d_val = Fraction(d)
            assert abs(d_val - answer_val) > tolerance, (
                f"{tier} seed={seed}: distractor {d} within tolerance of answer {q.answer_text}"
            )


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_answer_rejects_distractors_and_wrong(tier):
    grader = GRADERS["number"]
    for seed in SEEDS_40:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grader(q, q.answer_text)
        assert result.correct, f"{tier} seed={seed}: grader rejected answer {q.answer_text}"
        for d in q.distractors:
            r = grader(q, d)
            assert not r.correct, f"{tier} seed={seed}: grader accepted distractor {d}"
        if tier == "exam":
            wrong = f"{float(q.answer) + 1.0:.2f}"
        else:
            wrong = str(int(q.answer) + 1)
        rw = grader(q, wrong)
        assert not rw.correct, f"{tier} seed={seed}: grader accepted wrong answer {wrong}"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_identically(tier):
    exact_re = re.compile(r"^\d+$")
    exam_re = re.compile(r"^-?\d+\.\d\d$")
    pattern = exam_re if tier == "exam" else exact_re
    for seed in SEEDS_40:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert pattern.match(q.answer_text), f"{tier} seed={seed}: {q.answer_text!r} wrong format"
        for d in q.distractors:
            assert pattern.match(d), f"{tier} seed={seed}: distractor {d!r} wrong format"


@pytest.mark.parametrize("tier", TIERS)
def test_answer_satisfies_declared_kind(tier):
    """Exact tiers: the answer is a plain integer (int, no decimal point).
    Exam tier: the answer carries exactly 2 stated decimal places and is not
    sitting on a rounding boundary."""
    for seed in SEEDS_40:
        q = make(TOPIC, seed=seed, difficulty=tier)
        if tier == "exam":
            assert isinstance(q.answer, float)
            assert re.match(r"^-?\d+\.\d\d$", q.answer_text)
        else:
            assert isinstance(q.answer, int)
            assert "." not in q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(tier):
    texts = {make(TOPIC, seed=seed, difficulty=tier).text for seed in SEEDS_60}
    assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions over 60 seeds"


@pytest.mark.parametrize("tier", TIERS)
def test_at_least_4_distinct_prompt_shapes_over_60_seeds(tier):
    """The four hand-written templates per tier should show up as >= 4 distinct rendered shapes once digits are
    masked out."""
    shapes = {re.sub(r"\d+", "#", make(TOPIC, seed=seed, difficulty=tier).text) for seed in SEEDS_60}
    assert len(shapes) >= 4, f"{tier}: only {len(shapes)} digit-masked shapes over 60 seeds: {shapes}"


# ---------------------------------------------------------------------------
# The headline requirement, made testable: every seed, every tier, the
# rendered `working` carries H_total, H_small, both pyramid-volume
# intermediates and a subtraction step producing the final volume.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_working_shows_the_removed_apex_derivation_every_seed(tier):
    for seed in SEEDS_60:
        q = make(TOPIC, seed=seed, difficulty=tier)
        l1, w1, l2, w2, h = _dims_for_tier(tier, q.text)
        A1, A2 = l1 * w1, l2 * w2
        H_total, H_small = _h_total_h_small(l1, l2, h)
        assert H_total.denominator == 1, f"{tier} seed={seed}: H_total not exact ({H_total})"
        assert H_small.denominator == 1, f"{tier} seed={seed}: H_small not exact ({H_small})"

        vol_big = Fraction(A1 * int(H_total), 3)
        vol_small = Fraction(A2 * int(H_small), 3)
        V = vol_big - vol_small

        H_total_text = str(int(H_total))
        H_small_text = str(int(H_small))
        vol_big_text = _fmt_for_tier(vol_big, tier)
        vol_small_text = _fmt_for_tier(vol_small, tier)
        answer_text = _fmt_for_tier(V, tier)

        g = math.gcd(l2, l1)
        p_reduced, q_reduced = l2 // g, l1 // g

        joined = "\n".join(q.working)
        assert q.working, f"{tier} seed={seed}: no working"
        assert answer_text == q.answer_text, (
            f"{tier} seed={seed}: re-derived answer {answer_text} != rendered {q.answer_text}"
        )
        # Requirement (1): the base/top areas are shown, not just used silently.
        assert f"{l1} x {w1} = {A1} cm^2" in joined, (
            f"{tier} seed={seed}: base area {l1} x {w1} = {A1} not shown in working: {q.working!r}"
        )
        assert f"{l2} x {w2} = {A2} cm^2" in joined, (
            f"{tier} seed={seed}: top area {l2} x {w2} = {A2} not shown in working: {q.working!r}"
        )
        # The similarity ratio itself is visible (not just the resulting
        # heights) -- the reduced p/q form, since that is what the
        # generator's own similar-triangles sentence states.
        assert f"H_small/H_total = {p_reduced}/{q_reduced}" in joined, (
            f"{tier} seed={seed}: similarity ratio {p_reduced}/{q_reduced} not shown in working: {q.working!r}"
        )
        # Requirement (2): H_total and H_small, with actual numbers substituted.
        assert f"= {H_total_text} cm," in joined, (
            f"{tier} seed={seed}: H_total={H_total_text} not shown in working: {q.working!r}"
        )
        assert f"= {H_small_text} cm." in joined, (
            f"{tier} seed={seed}: H_small={H_small_text} not shown in working: {q.working!r}"
        )
        assert vol_big_text in joined, (
            f"{tier} seed={seed}: big-pyramid volume {vol_big_text} not shown in working: {q.working!r}"
        )
        assert vol_small_text in joined, (
            f"{tier} seed={seed}: removed-apex volume {vol_small_text} not shown in working: {q.working!r}"
        )
        subtraction = f"{vol_big_text} - {vol_small_text} = {answer_text}"
        assert subtraction in joined, (
            f"{tier} seed={seed}: subtraction step {subtraction!r} not found in working: {q.working!r}"
        )


# ---------------------------------------------------------------------------
# Independent-check firing on a corrupted value, exercised directly.
# ---------------------------------------------------------------------------


def test_check_equivalent_fires_on_corrupted_volume():
    A1, A2, G, h = 400, 100, 200, 12
    V = Fraction(h, 3) * (A1 + A2 + G)
    corrupted = V + Fraction(1, 1000)
    with pytest.raises(VerificationError):
        check_equivalent(V, corrupted, "expected mismatch")


def test_check_distinct_fires_when_candidate_equals_answer():
    V = Fraction(2800)
    with pytest.raises(VerificationError):
        check_distinct(V, V, msg="candidate equals the answer")


# ---------------------------------------------------------------------------
# Skill loading.
# ---------------------------------------------------------------------------


class TestSkillsLoad:
    def test_rectangular_frustum_volume_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "rectangular-frustum-volume"), None)
        assert skill is not None
        assert skill.domain == "measurement"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-use-formulas-to-find-the-volume-of-pyramids"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 4

        stages = [tuple(src.stages) for src in skill.sources]
        assert (0, 1, 2) in stages
        assert (3, 4) in stages
        assert (5, 6, "maintenance") in stages
        assert ("exam",) in stages

        difficulties = {src.difficulty for src in skill.sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}

        misconception_ids = {m.id for m in skill.misconceptions}
        assert misconception_ids == {
            "omits-apex-removal",
            "treats-as-prism",
            "uses-mean-area-slab",
        }
