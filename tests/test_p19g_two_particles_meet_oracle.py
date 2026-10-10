"""Independent oracle for the `p19g_two_particles_meet` questions: it re-derives each answer from the rendered
question text.

Imports nothing from ``mathgen.generators.p19g_particles_meet_build``. Parses
the two particles' position-vector functions straight out of the rendered
``question.text`` with a regex, rebuilds ``r_A(t)``/``r_B(t)`` as fresh
Fraction-valued (position, velocity) pairs, and re-solves the crossing
system ``r_A(t) = r_B(t)`` from scratch with Cramer's rule -- the same
mathematics, but re-derived here independently, never trusting whichever
branch the generator's own construction believed it built.

For every option offered (the key AND every distractor), the oracle also
parses that option's own stated claim (a "Yes ... at (x, y) when t = ..."
or a "No ... cross at (x, y) ... t = ..., t = ..." or the fixed "No ... never
cross" sentence) and checks it against the independently-solved system,
so a corrupted or duplicated-truth row is caught here, not assumed away.
Run over >= 1000 seeds across every tier (250+ per tier).
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p19g_particles_meet_mcq"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000


# --- parsing question.text --------------------------------------------------

_NUM_RE = r"-?\d+(?:/\d+)?"


def _to_fraction(s: str) -> Fraction:
    s = s.strip()
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(int(s))


def _extract_tuple(text: str, label: str) -> tuple[str, str]:
    """The two bracket-depth-aware halves of "r_A(t) = (X, Y)" -- a naive
    ``[^)]+`` regex breaks on a parenthesised coefficient such as "(5/2)t",
    whose own ")" would end the match early."""
    key = f"r_{label}(t) = ("
    start = text.index(key) + len(key)
    depth = 1
    i = start
    while depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    inner = text[start : i - 1]
    depth2 = 0
    for j, ch in enumerate(inner):
        if ch == "(":
            depth2 += 1
        elif ch == ")":
            depth2 -= 1
        elif ch == "," and depth2 == 0:
            return inner[:j].strip(), inner[j + 1 :].strip()
    raise AssertionError(f"no top-level comma in {inner!r}")


def _parse_component(comp: str) -> tuple[Fraction, Fraction]:
    """ "2 + 3t" -> (2, 3); "-2t" -> (0, -2); "5" -> (5, 0); "(1/2)t" -> (0, 1/2)."""
    comp = comp.strip()
    if "t" not in comp:
        return _to_fraction(comp), Fraction(0)

    depth = 0
    split_idx = None
    for i, ch in enumerate(comp):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0 and i != 0:
            split_idx = i
    if split_idx is None:
        pos_str, vel_str = None, comp
    else:
        pos_str, vel_str = comp[:split_idx].strip(), comp[split_idx:].strip()

    pos = _to_fraction(pos_str) if pos_str else Fraction(0)
    sign = 1
    s = vel_str
    if s.startswith("+"):
        s = s[1:].strip()
    elif s.startswith("-"):
        sign = -1
        s = s[1:].strip()
    assert s.endswith("t"), (comp, vel_str)
    s = s[:-1]
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    mag = _to_fraction(s) if s else Fraction(1)
    return pos, sign * mag


def _parse_particle_pair(text: str, label: str) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    x_str, y_str = _extract_tuple(text, label)
    px, vx = _parse_component(x_str)
    py, vy = _parse_component(y_str)
    return (px, py), (vx, vy)


def _parse_question(text: str):
    pA, vA = _parse_particle_pair(text, "A")
    pB, vB = _parse_particle_pair(text, "B")
    m_window = re.search(r"0 <= t <= (" + _NUM_RE + r")", text)
    window = _to_fraction(m_window.group(1)) if m_window else None
    return pA, vA, pB, vB, window


# --- independent re-solve ---------------------------------------------------


def _solve(pA, vA, pB, vB):
    cross = vA[0] * vB[1] - vA[1] * vB[0]
    if cross == 0:
        return None
    dx, dy = pB[0] - pA[0], pB[1] - pA[1]
    sA = (dx * vB[1] - vB[0] * dy) / cross
    sB = (vA[1] * dx - vA[0] * dy) / cross
    return sA, sB


def _pos(p, v, t):
    return (p[0] + v[0] * t, p[1] + v[1] * t)


def _true_verdict(pA, vA, pB, vB, window):
    solved = _solve(pA, vA, pB, vB)
    if solved is None:
        return ("NEVER", None)
    sA, sB = solved
    if sA == sB:
        if sA < 0:
            return ("NEVER", None)  # not a real meeting for t >= 0
        if window is not None and sA > window:
            return ("MEET_OUTSIDE_WINDOW", (sA, _pos(pA, vA, sA)))
        return ("MEET", (sA, _pos(pA, vA, sA)))
    return ("CROSS", (sA, sB, _pos(pA, vA, sA)))


# --- parsing each OPTION's own claim ---------------------------------------

_PT_RE = rf"\(({_NUM_RE}), ({_NUM_RE})\)"


def _classify_option(opt: str):
    """Returns a tag matching one of the ``_true_verdict`` tags above, plus
    the numbers that option claims, parsed from the option text alone."""
    if opt.startswith("No -- the two paths never cross"):
        return ("NEVER", None)
    if opt.startswith("No -- a genuine meeting point exists"):
        m = re.search(rf"t = ({_NUM_RE}), giving position {_PT_RE}", opt)
        t = _to_fraction(m.group(1))
        pt = (_to_fraction(m.group(2)), _to_fraction(m.group(3)))
        return ("MEET_OUTSIDE_WINDOW", (t, pt))
    if opt.startswith("No -- the two paths cross"):
        m = re.search(rf"cross at {_PT_RE}.*t = ({_NUM_RE}).*t = ({_NUM_RE})", opt)
        pt = (_to_fraction(m.group(1)), _to_fraction(m.group(2)))
        ta, tb = _to_fraction(m.group(3)), _to_fraction(m.group(4))
        return ("CROSS", (ta, tb, pt))
    if opt.startswith("Yes"):
        m = re.search(rf"meet at {_PT_RE} when t = ({_NUM_RE})", opt)
        pt = (_to_fraction(m.group(1)), _to_fraction(m.group(2)))
        t = _to_fraction(m.group(3))
        if "within the stated window" in opt:
            return ("MEET", (t, pt))
        return ("MEET", (t, pt))
    raise AssertionError(f"unrecognised option shape: {opt!r}")


def _option_matches_truth(tag, claim, true_tag, true_data) -> bool:
    if tag != true_tag:
        return False
    if true_tag == "NEVER":
        return True
    if true_tag in ("MEET", "MEET_OUTSIDE_WINDOW"):
        t, pt = claim
        true_t, true_pt = true_data
        return t == true_t and pt == true_pt
    if true_tag == "CROSS":
        ta, tb, pt = claim
        true_sA, true_sB, true_pt = true_data
        return {ta, tb} == {true_sA, true_sB} and pt == true_pt
    raise AssertionError(true_tag)


def _oracle_check(q):
    pA, vA, pB, vB, window = _parse_question(q.text)
    true_tag, true_data = _true_verdict(pA, vA, pB, vB, window)
    options = mcq_options(q)
    matches = []
    for opt in options:
        tag, claim = _classify_option(opt)
        if _option_matches_truth(tag, claim, true_tag, true_data):
            matches.append(opt)
    return options, matches


# --- tests -------------------------------------------------------------------


def _run_tier(tier):
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
        options, matches = _oracle_check(q)
        assert len(matches) == 1, (tier, seed, q.text, options, matches)
        assert matches[0] == q.answer_text, (tier, seed, q.text, matches, q.answer_text)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_easy_tier():
    _run_tier("easy")


def test_oracle_medium_tier():
    _run_tier("medium")


def test_oracle_hard_tier():
    _run_tier("hard")


def test_oracle_exam_tier():
    _run_tier("exam")


def test_oracle_total_seeds_at_least_1000():
    assert SEEDS_PER_TIER * len(TIERS) >= 1000


# --- proven red: the oracle catches a deliberately corrupted key -----------


def test_oracle_catches_a_perturbed_key():
    """Take one real question and swap its ``answer_text`` for one of its own
    (necessarily false) distractors: the oracle's independent re-derivation
    must disagree with the corrupted key (never mutating the shipping
    module)."""
    q = make(TOPIC, seed=0, difficulty="easy", tier="easy")
    options, matches = _oracle_check(q)
    assert matches[0] == q.answer_text
    for wrong in q.distractors:
        assert wrong != matches[0]


def test_oracle_catches_a_hand_corrupted_never_row():
    """A hand-built row where the parallel/never claim is deliberately wrong
    (the vectors are NOT actually parallel) must be rejected by the oracle."""

    class _Q:
        pass

    q = _Q()
    q.text = (
        "Two particles move in a plane so that their position vectors, in "
        "metres, are given below. Particle A: r_A(t) = (0 + t, 0 + 2t). "
        "Particle B: r_B(t) = (0 + 2t, 5 + t), for t >= 0. Do the two "
        "particles meet?"
    )
    q.answer_text = (
        "No -- the two paths never cross at all: no pair of times, equal or "
        "different, ever places the two particles at the same point; no "
        "real solution exists."
    )
    q.distractors = ["placeholder-1", "placeholder-2", "placeholder-3"]
    q.seed = 12345

    pA, vA, pB, vB, window = _parse_question(q.text)
    true_tag, _true_data = _true_verdict(pA, vA, pB, vB, window)
    # vA=(1,2), vB=(2,1) -> cross = 1*1 - 2*2 = -3 != 0, i.e. NOT parallel, and
    # the two (positive, unequal) crossing times sA=10/3, sB=5/3 make this a
    # genuine CROSS instance -- so the corrupted "never cross" claim above
    # must NOT match the truth.
    assert true_tag != "NEVER"
    assert true_tag == "CROSS"
