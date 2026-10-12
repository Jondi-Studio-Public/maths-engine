"""The exam source added to `cartesian_polar_convert` (topic feeding
senior-convert-between-cartesian-and-polar-form).

The exam source is reached with ``difficulty="exam"`` (preset key "exam",
``variant="exam"``) and dispatches randomly to one of two shapes:

- ``_polar_exam_surd``: Cartesian -> polar where r is a genuine surd (the
  point is deliberately NOT drawn from a Pythagorean triple), r to 2 d.p.
  and theta to 1 d.p., quadrant-aware. Answer keys ``{"r", "theta"}``.
- ``_polar_exam_reverse_obtuse``: polar -> Cartesian with an obtuse theta
  (90 < theta < 180), x and y to 1 d.p. Answer keys ``{"x", "y"}``.

Every check here reads only what `mathgen.make` hands back (`.text`,
`.answer`, `.answer_text`, `.distractors`, `.working`) plus independently
recomputed values via `cmath`/`math` -- it does not import anything from
``mathgen.generators.p5b_complex_plane_build``.

**The tolerance gap, stated explicitly (do not gloss over this):** the
skill is graded by `mathgen.grading.grade_pair`, whose ONLY tolerance is
`_PAIR_TOLERANCE = Fraction(51, 1000) = 0.051` (confirmed by reading
`grade_pair`/`_labelled_pair_verdict` in `mathgen/grading/typed.py`). The prompt says
"r to 2 d.p." (matching how an exam would phrase it) and "theta to 1
d.p.", but the grader does NOT enforce those stated precisions (which would
need 0.005 and 0.05 respectively) -- it only enforces the coarser 0.051 on
every component. Every distractor test below checks the actual enforced
figure (0.051, as an exact `Fraction`, never a float subtraction), not the
stated display precision.
"""

from __future__ import annotations

import cmath
import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_pair
from mathgen.skills import load_skills
from mathgen import make
from mathgen.text_quality import prompt_shape

SEEDS = range(60)

#: The grader's actual enforced tolerance (mathgen/grading/typed.py::_PAIR_TOLERANCE),
#: as an exact Fraction -- never a float threshold.
GRADER_TOLERANCE = Fraction(51, 1000)

_SURD_XY_RE = re.compile(r"\((-?\d+), (-?\d+)\)")
_REVERSE_RTHETA_RE = re.compile(r"r = (-?\d+), θ = (-?\d+)°")
_SURD_WORKING_THETA_RE = re.compile(r"≈\s*(-?[\d.]+)°\s*≈")


def _make_exam(seed: int):
    return make("cartesian_polar_convert", seed=seed, difficulty="exam")


def _classify(q) -> str:
    keys = set(q.answer.keys())
    if keys == {"r", "theta"}:
        return "surd"
    if keys == {"x", "y"}:
        return "reverse_obtuse"
    raise AssertionError(f"unexpected answer keys: {keys!r} for text={q.text!r}")


def _parse_labelled_pair(text: str) -> dict[str, str]:
    """'r = 12.35, theta = 233.1' -> {'r': '12.35', 'theta': '233.1'}, string
    values kept as-is so callers can build exact Fractions from them."""
    out: dict[str, str] = {}
    for segment in text.split(","):
        label, _, value = segment.strip().partition(" = ")
        assert value, f"segment {segment!r} in {text!r} did not parse as 'label = value'"
        out[label.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------
# Determinism, shape, and both sub-shapes reachable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_determinism(seed):
    q1 = _make_exam(seed)
    q2 = _make_exam(seed)
    assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("seed", SEEDS)
def test_answer_is_pair_dict(seed):
    q = _make_exam(seed)
    assert isinstance(q.answer, dict)
    assert len(q.answer) == 2
    assert all(isinstance(v, (int, float)) for v in q.answer.values())


def test_both_exam_shapes_reachable():
    """Over 200 seeds, both the surd-r and the obtuse-reverse shape appear."""
    counts = {"surd": 0, "reverse_obtuse": 0}
    for seed in range(200):
        counts[_classify(_make_exam(seed))] += 1
    assert counts["surd"] > 0, counts
    assert counts["reverse_obtuse"] > 0, counts


def test_exam_source_clears_shape_bar():
    """Two wording banks (2 patterns each) across the two sub-shapes comfortably clears this once digits are
    masked out."""
    shapes = {prompt_shape(_make_exam(seed)) for seed in SEEDS}
    assert len(shapes) >= 3, f"only {len(shapes)} distinct prompt shapes: {sorted(shapes)}"


def test_at_least_20_distinct_questions_over_60_seeds():
    texts = {_make_exam(seed).text for seed in SEEDS}
    assert len(texts) >= 20, f"only {len(texts)} distinct texts over 60 seeds"


def test_exam_is_genuine_escalation_over_hard():
    """The exam source's shapes are absent from the hard (reverse) tier's
    shapes -- a genuine escalation in kind, not merely bigger numbers."""
    exam_shapes = {prompt_shape(_make_exam(seed)) for seed in SEEDS}
    hard_shapes = {prompt_shape(make("cartesian_polar_convert", seed=seed, difficulty="hard")) for seed in SEEDS}
    assert exam_shapes - hard_shapes, "exam tier adds no shape over hard"


# --------------------------------------------------------------------------
# Four quadrants, explicitly -- not a random sample (surd shape)
# --------------------------------------------------------------------------


def test_four_quadrants_theta_cross_checked_against_cmath():
    """At least one seed per quadrant for the surd shape, cross-checked
    against `cmath.polar` -- the same independent-check pattern
    `_polar_axis`/`_polar_triple` already use
    (`check(abs(theta_check - theta) < eps)`), applied here to the RENDERED
    artefact. The tolerance is 1e-3, not the internal-float 1e-6 those
    functions use, because the only unrounded theta this test can recover is
    the 4 d.p. printed in the working line -- this is an intentional,
    disclosed adaptation of the existing pattern to an external, read-only
    check, not a loosening of what the generator itself verifies internally
    (which uses 1e-6/1e-9 against the true float, see
    `_polar_exam_surd`'s own `check()` calls).
    """
    found: dict[tuple[int, int], tuple[int, object, int, int]] = {}
    for seed in range(800):
        q = _make_exam(seed)
        if _classify(q) != "surd":
            continue
        m = _SURD_XY_RE.search(q.text)
        assert m, q.text
        x, y = int(m.group(1)), int(m.group(2))
        quadrant = (1 if x > 0 else -1, 1 if y > 0 else -1)
        if quadrant not in found:
            found[quadrant] = (seed, q, x, y)
        if len(found) == 4:
            break

    assert len(found) == 4, f"only found quadrants {sorted(found)} over 800 seeds"

    for quadrant, (seed, q, x, y) in found.items():
        # Quadrant-range sanity on the REPORTED (rounded) theta.
        theta = q.answer["theta"]
        if quadrant == (1, 1):
            assert 0 < theta < 90, (seed, x, y, theta)
        elif quadrant == (-1, 1):
            assert 90 < theta < 180, (seed, x, y, theta)
        elif quadrant == (-1, -1):
            assert 180 < theta < 270, (seed, x, y, theta)
        else:
            assert 270 < theta < 360, (seed, x, y, theta)

        # Independent cross-check via cmath.polar against the unrounded
        # theta printed in the working line.
        working_line = next(w for w in q.working if "atan2" in w)
        wm = _SURD_WORKING_THETA_RE.search(working_line)
        assert wm, working_line
        theta_exact_rendered = float(wm.group(1))

        r_check, theta_rad_check = cmath.polar(complex(x, y))
        theta_check = math.degrees(theta_rad_check)
        if theta_check < 0:
            theta_check += 360
        assert abs(theta_check - theta_exact_rendered) < 1e-3, (
            f"quadrant {quadrant} seed {seed}: cmath.polar gives {theta_check}, "
            f"rendered working shows {theta_exact_rendered}"
        )

        # The bug class to avoid: a naive atan2(abs(y),
        # abs(x)) (first-quadrant-only) angle must NOT be what was reported,
        # for every non-Q1 quadrant.
        if quadrant != (1, 1):
            naive = math.degrees(math.atan2(abs(y), abs(x)))
            assert abs(theta - naive) > 1.0, (
                f"quadrant {quadrant} seed {seed}: reported theta {theta} suspiciously "
                f"close to the naive first-quadrant angle {naive} (wrong-quadrant bug?)"
            )


def test_surd_r_is_never_from_a_perfect_square():
    """r is a genuine surd: x^2 + y^2 recovered from the rendered text is
    never a perfect square (which is exactly the condition that would make r
    rational -- a Pythagorean triple or its scaled multiple)."""
    checked = 0
    for seed in range(200):
        q = _make_exam(seed)
        if _classify(q) != "surd":
            continue
        m = _SURD_XY_RE.search(q.text)
        assert m, q.text
        x, y = int(m.group(1)), int(m.group(2))
        s = x * x + y * y
        assert math.isqrt(s) ** 2 != s, f"seed {seed}: ({x},{y}) sums to a perfect square {s}"
        checked += 1
    assert checked >= 20, f"only {checked} surd-shaped seeds found in 200"


def test_surd_r_and_theta_stated_precision():
    """r is always stated to exactly 2 d.p. and theta to exactly 1 d.p. in
    answer_text, for every surd-shaped seed."""
    checked = 0
    for seed in range(60):
        q = _make_exam(seed)
        if _classify(q) != "surd":
            continue
        parsed = _parse_labelled_pair(q.answer_text)
        r_str, theta_str = parsed["r"], parsed["theta"]
        assert "." in r_str and len(r_str.split(".")[1]) == 2, r_str
        assert "." in theta_str and len(theta_str.split(".")[1]) == 1, theta_str
        checked += 1
    assert checked >= 10, f"only {checked} surd-shaped seeds in 60"


# --------------------------------------------------------------------------
# Obtuse-theta reverse shape
# --------------------------------------------------------------------------


def test_reverse_obtuse_theta_range_and_quadrant():
    checked = 0
    for seed in range(200):
        q = _make_exam(seed)
        if _classify(q) != "reverse_obtuse":
            continue
        m = _REVERSE_RTHETA_RE.search(q.text)
        assert m, q.text
        r, theta = int(m.group(1)), int(m.group(2))
        assert 90 < theta < 180, f"seed {seed}: theta={theta} is not obtuse"
        x, y = q.answer["x"], q.answer["y"]
        assert x < 0, f"seed {seed}: x={x} should be negative for an obtuse theta"
        assert y > 0, f"seed {seed}: y={y} should be positive for an obtuse theta"

        # Independent cross-check via cmath.rect.
        z = cmath.rect(r, math.radians(theta))
        assert abs(round(z.real, 1) - x) < 1e-9, (seed, r, theta, x, z.real)
        assert abs(round(z.imag, 1) - y) < 1e-9, (seed, r, theta, y, z.imag)
        checked += 1
    assert checked >= 20, f"only {checked} reverse_obtuse-shaped seeds in 200"


def test_reverse_obtuse_stated_precision():
    checked = 0
    for seed in range(60):
        q = _make_exam(seed)
        if _classify(q) != "reverse_obtuse":
            continue
        parsed = _parse_labelled_pair(q.answer_text)
        for v in parsed.values():
            assert "." in v and len(v.split(".")[1]) == 1, v
        checked += 1
    assert checked >= 5, f"only {checked} reverse_obtuse-shaped seeds in 60"


# --------------------------------------------------------------------------
# Distractors: distinct, non-empty, formatted alike, rejected by the real
# grader -- margin checked as an exact Fraction against the grader's ACTUAL
# tolerance (0.051), never a float subtraction and never the finer stated
# display precision the grader does not enforce.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_distractors_nonempty_and_exceed_grader_tolerance(seed):
    q = _make_exam(seed)
    assert q.distractors, f"seed {seed}: no distractors at all"
    canonical = _parse_labelled_pair(q.answer_text)
    for d in q.distractors:
        given = _parse_labelled_pair(d)
        assert set(given) == set(canonical), f"seed {seed}: label mismatch {d} vs {q.answer_text}"
        diffs = [abs(Fraction(given[k]) - Fraction(canonical[k])) for k in canonical]
        assert max(diffs) > GRADER_TOLERANCE, (
            f"seed {seed}: distractor {d} is within the grader's actual 0.051 tolerance "
            f"of {q.answer_text} (max component diff {max(diffs)})"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_distractors_formatted_like_the_answer(seed):
    """Same labels, same order, same decimal-place count as the canonical --
    no option identifiable by its shape alone."""
    q = _make_exam(seed)
    canonical_labels = [seg.strip().split(" = ")[0] for seg in q.answer_text.split(",")]
    canonical_decimals = [
        len(seg.strip().split(" = ")[1].split(".")[1]) if "." in seg else 0 for seg in q.answer_text.split(",")
    ]
    for d in q.distractors:
        d_labels = [seg.strip().split(" = ")[0] for seg in d.split(",")]
        d_decimals = [len(seg.strip().split(" = ")[1].split(".")[1]) if "." in seg else 0 for seg in d.split(",")]
        assert d_labels == canonical_labels, f"seed {seed}: {d!r} vs {q.answer_text!r}"
        assert d_decimals == canonical_decimals, f"seed {seed}: {d!r} vs {q.answer_text!r}"


@pytest.mark.parametrize("seed", SEEDS)
def test_real_grader_accepts_canonical(seed):
    q = _make_exam(seed)
    result = grade_pair(q, q.answer_text)
    assert result.correct, f"seed {seed}: grader rejected canonical {q.answer_text!r}"


@pytest.mark.parametrize("seed", SEEDS)
def test_real_grader_rejects_distractors(seed):
    q = _make_exam(seed)
    for d in q.distractors:
        result = grade_pair(q, d)
        assert not result.correct, f"seed {seed}: grader accepted distractor {d!r}"


@pytest.mark.parametrize("seed", SEEDS)
def test_real_grader_rejects_deliberately_wrong_answer(seed):
    """A hand-corrupted answer (first component pushed 1 whole unit away,
    exact Fraction arithmetic) is rejected by the real grader, every seed."""
    q = _make_exam(seed)
    canonical = _parse_labelled_pair(q.answer_text)
    keys = list(canonical.keys())
    corrupted = dict(canonical)
    bumped = Fraction(canonical[keys[0]]) + 1
    corrupted[keys[0]] = str(bumped)
    corrupted_text = ", ".join(f"{k} = {v}" for k, v in corrupted.items())
    result = grade_pair(q, corrupted_text)
    assert not result.correct, f"seed {seed}: grader wrongly accepted corrupted {corrupted_text!r}"


# --------------------------------------------------------------------------
# Independent oracle: recompute the answer from the rendered question text
# alone, via cmath -- never importing mathgen.generators.p5b_complex_plane_build.
# --------------------------------------------------------------------------


def _oracle_polar_from_cartesian(x: int, y: int) -> tuple[float, float]:
    r, theta_rad = cmath.polar(complex(x, y))
    theta = math.degrees(theta_rad)
    if theta < 0:
        theta += 360
    return round(r, 2), round(theta, 1)


def _oracle_cartesian_from_polar(r: int, theta_deg: int) -> tuple[float, float]:
    z = cmath.rect(r, math.radians(theta_deg))
    return round(z.real, 1), round(z.imag, 1)


def test_oracle_matches_rendered_answer():
    checked = {"surd": 0, "reverse_obtuse": 0}
    for seed in SEEDS:
        q = _make_exam(seed)
        shape = _classify(q)
        if shape == "surd":
            m = _SURD_XY_RE.search(q.text)
            assert m, q.text
            x, y = int(m.group(1)), int(m.group(2))
            r_oracle, theta_oracle = _oracle_polar_from_cartesian(x, y)
            assert abs(q.answer["r"] - r_oracle) < 1e-9, (seed, q.answer, r_oracle)
            assert abs(q.answer["theta"] - theta_oracle) < 1e-9, (seed, q.answer, theta_oracle)
        else:
            m = _REVERSE_RTHETA_RE.search(q.text)
            assert m, q.text
            r, theta = int(m.group(1)), int(m.group(2))
            x_oracle, y_oracle = _oracle_cartesian_from_polar(r, theta)
            assert abs(q.answer["x"] - x_oracle) < 1e-9, (seed, q.answer, x_oracle)
            assert abs(q.answer["y"] - y_oracle) < 1e-9, (seed, q.answer, y_oracle)
        checked[shape] += 1
    assert checked["surd"] >= 10, checked
    assert checked["reverse_obtuse"] >= 5, checked


def test_oracle_fires_on_corrupted_answer():
    """The independent oracle, exercised directly against a deliberately
    corrupted rendered answer, disagrees -- proving it is actually checking
    something rather than trivially agreeing."""
    surd_seed = next(s for s in range(60) if _classify(_make_exam(s)) == "surd")
    q = _make_exam(surd_seed)
    m = _SURD_XY_RE.search(q.text)
    x, y = int(m.group(1)), int(m.group(2))
    r_oracle, theta_oracle = _oracle_polar_from_cartesian(x, y)
    corrupted_r = q.answer["r"] + 5.0
    assert abs(corrupted_r - r_oracle) > GRADER_TOLERANCE, "oracle failed to detect corrupted r"

    reverse_seed = next(s for s in range(60) if _classify(_make_exam(s)) == "reverse_obtuse")
    q2 = _make_exam(reverse_seed)
    m2 = _REVERSE_RTHETA_RE.search(q2.text)
    r, theta = int(m2.group(1)), int(m2.group(2))
    x_oracle, y_oracle = _oracle_cartesian_from_polar(r, theta)
    corrupted_x = q2.answer["x"] + 5.0
    assert abs(corrupted_x - x_oracle) > GRADER_TOLERANCE, "oracle failed to detect corrupted x"


# --------------------------------------------------------------------------
# Skill loading
# --------------------------------------------------------------------------


def test_skill_loads_with_exam_source():
    _domains, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-convert-between-cartesian-and-polar-form"), None)
    assert skill is not None
    assert skill.answer == "pair"
    assert skill.prerequisites == ["senior-consider-complex-numbers-as-points-in-a-plane", "complex-modulus"]
    assert skill.placement_eligible is False

    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert len(exam_sources) == 1, f"expected exactly one exam source, got {exam_sources}"
    exam_source = exam_sources[0]
    assert exam_source.topic == "cartesian_polar_convert"
    assert exam_source.stages == ["exam"], exam_source.stages
    assert dict(exam_source.params) == {"variant": "exam"}

    # Every declared source is still present -- this change must not disturb
    # easy/medium/hard.
    difficulties = sorted(s.difficulty for s in skill.sources)
    assert difficulties == ["easy", "exam", "hard", "medium"], difficulties
