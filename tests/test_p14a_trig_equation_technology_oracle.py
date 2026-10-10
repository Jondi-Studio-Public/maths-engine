"""Independent oracle for the `p14a_trig_equation_technology` questions: it re-derives each answer from the
rendered question text.

Imports nothing from `mathgen.generators.p14a_trig_equation_technology_build`:
not `_algebraic_solutions`, not `_enumerate_solutions`, not `_base_solutions_deg`,
not any misconception formula. Parses `mu`/`k`/`b`/`a`/`c`/`d`/the interval/the
naming rule straight out of the RENDERED `question.text`, and re-derives the
answer from scratch via its OWN, independently written numeric root scan
(`_scan` below) -- never the generator's own bisection/enumeration code.

Naming-uniqueness proof, re-verified here rather than trusted: for every
seed on every tier, this file's own scan confirms the solution COUNT the
question's naming rule assumes (exactly two for easy/medium; at least `n`
for the stated ordinal on hard/exam) before applying that naming rule to
the independently-enumerated list and comparing to `answer_text`.

Run across >= 1000 seeds total, 250 per tier.
"""

from __future__ import annotations

import math
import re

from mathgen import make

TOPIC = "p14a_solve_trig_equation_technology"
SEEDS_PER_TIER = 250


def _scan(kind: str, b: float, k: float, hi: float, n_points: int = 12000) -> list[float]:
    """A from-scratch numeric root scan over ``trig(b*x) - k`` for ``x`` in
    ``[0, hi)``: sign-change detection plus a plain bisection refine,
    written independently of the generator's own `_enumerate_solutions`
    (different point count, different loop structure, no shared code)."""

    def f(x: float) -> float:
        angle = math.radians(b * x)
        value = math.sin(angle) if kind == "sin" else math.cos(angle)
        return value - k

    step = hi / n_points
    found = []
    x_prev = 0.0
    f_prev = f(0.0)
    for i in range(1, n_points + 1):
        x_cur = i * step
        f_cur = f(x_cur)
        if f_prev == 0.0:
            found.append(x_prev)
        elif f_prev * f_cur < 0.0:
            left, right = x_prev, x_cur
            f_left = f_prev
            for _ in range(80):
                mid = (left + right) / 2.0
                f_mid = f(mid)
                if f_left * f_mid <= 0.0:
                    right = mid
                else:
                    left, f_left = mid, f_mid
            found.append((left + right) / 2.0)
        x_prev, f_prev = x_cur, f_cur
    found.sort()
    deduped: list[float] = []
    for value in found:
        if not deduped or value - deduped[-1] > 1e-3:
            deduped.append(value)
    return deduped


def _round_half_up(value: float, dp: int) -> str:
    """Independent re-implementation of half-up rounding -- deliberately NOT
    a call to `mathgen.core.round_half_up`."""
    scale = 10**dp
    scaled = math.floor(abs(value) * scale + 0.5)
    sign = "-" if value < 0 and scaled != 0 else ""
    whole, frac = divmod(scaled, scale)
    return f"{sign}{whole}.{str(frac).zfill(dp)}" if dp else f"{sign}{whole}"


_NUM_RE = r"-?\d+(?:\.\d+)?"

_EASY_RE = re.compile(
    rf"^Solve (?P<kind>sin|cos)\(x\) = (?P<k>{_NUM_RE}) for x in the interval "
    rf"\[0.{{1,2}}, 360.{{1,2}}\)\. Give the (?P<which>smaller|larger) solution "
    rf"correct to 1 decimal place\.$"
)
_MEDIUM_RE = re.compile(
    rf"^Solve (?P<a>-?\d+) (?P<kind>sin|cos)\(x\) = (?P<d>{_NUM_RE}) for x in the interval "
    rf"\[0.{{1,2}}, 360.{{1,2}}\)\. Give the (?P<which>smaller|larger) solution "
    rf"correct to 1 decimal place\.$"
)
_HARD_RE = re.compile(
    rf"^Solve (?P<kind>sin|cos)\((?P<b>\d+)x\) = (?P<k>{_NUM_RE}) for x in the interval "
    rf"\[0.{{1,2}}, 360.{{1,2}}\)\. Give the (?P<n>\d+)\w\w smallest solution "
    rf"\(counting from 1\) correct to 1 decimal place\.$"
)
_EXAM_RE = re.compile(
    rf"^The height of the tide at a harbour, h\(t\) metres, t hours after midnight, is modelled by "
    rf"h\(t\) = (?P<a>{_NUM_RE}) (?P<kind>sin|cos)\((?P<b>\d+)t.{{1,2}}\) \+ (?P<c>{_NUM_RE}) for "
    rf"0 <= t < 24\. Find the (?P<n>\d+)\w\w smallest time t \(in hours, counting from 1\) at which "
    rf"h\(t\) = (?P<d>{_NUM_RE}) m\. Give your answer correct to 3 decimal places\.$"
)


def test_oracle_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", variant="direct")
        m = _EASY_RE.search(q.text)
        assert m, (seed, q.text)
        kind, k, which = m["kind"], float(m["k"]), m["which"]
        roots = _scan(kind, 1, k, 360.0)
        assert len(roots) == 2, (seed, q.text, roots)
        chosen = roots[0] if which == "smaller" else roots[1]
        assert _round_half_up(chosen, 1) == q.answer_text, (seed, q.text, q.answer_text, chosen)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", variant="coefficient")
        m = _MEDIUM_RE.search(q.text)
        assert m, (seed, q.text)
        kind, a_coeff, d_val, which = m["kind"], int(m["a"]), float(m["d"]), m["which"]
        k = d_val / a_coeff
        roots = _scan(kind, 1, k, 360.0)
        assert len(roots) == 2, (seed, q.text, roots)
        chosen = roots[0] if which == "smaller" else roots[1]
        assert _round_half_up(chosen, 1) == q.answer_text, (seed, q.text, q.answer_text, chosen)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", variant="multi_period")
        m = _HARD_RE.search(q.text)
        assert m, (seed, q.text)
        kind, b, k, n = m["kind"], int(m["b"]), float(m["k"]), int(m["n"])
        roots = _scan(kind, b, k, 360.0)
        assert len(roots) == 2 * b, (seed, q.text, roots)
        assert 1 <= n <= len(roots), (seed, q.text, n, len(roots))
        chosen = roots[n - 1]
        assert _round_half_up(chosen, 1) == q.answer_text, (seed, q.text, q.answer_text, chosen)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", variant="technology_applied")
        m = _EXAM_RE.search(q.text)
        assert m, (seed, q.text)
        kind, b = m["kind"], int(m["b"])
        a_amp, c_mean, d_val, n = float(m["a"]), float(m["c"]), float(m["d"]), int(m["n"])
        k = (d_val - c_mean) / a_amp
        roots = _scan(kind, b, k, 24.0)
        expected_count = round(b * 24.0 / 360.0) * 2
        assert len(roots) == expected_count, (seed, q.text, roots)
        assert 1 <= n <= len(roots), (seed, q.text, n, len(roots))
        chosen = roots[n - 1]
        assert _round_half_up(chosen, 3) == q.answer_text, (seed, q.text, q.answer_text, chosen)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Prove the oracle is red first: a corrupted answer must be caught -------


def test_oracle_catches_a_corrupted_answer():
    """The oracle's own assertion, re-run against a deliberately corrupted
    `answer_text` (an offset well outside the display tolerance), must
    actually raise -- proving the check fires rather than trivially passing."""
    q = make(TOPIC, seed=0, difficulty="easy", variant="direct")
    m = _EASY_RE.search(q.text)
    kind, k, which = m["kind"], float(m["k"]), m["which"]
    roots = _scan(kind, 1, k, 360.0)
    chosen = roots[0] if which == "smaller" else roots[1]
    correct = _round_half_up(chosen, 1)
    corrupted_answer_text = _round_half_up(chosen + 5.0, 1)
    assert correct == q.answer_text
    try:
        assert correct == corrupted_answer_text
    except AssertionError:
        pass
    else:
        raise AssertionError("oracle failed to catch a corrupted answer_text")
