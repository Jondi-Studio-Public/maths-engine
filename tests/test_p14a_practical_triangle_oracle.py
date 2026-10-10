"""Independent oracle for `p14a_practical_triangle`, re-deriving the answer from the rendered question.

Parses only ``question.text`` / ``question.answer_text`` /
``question.distractors`` and re-derives the answer from scratch, via the
route stated in the generator's docstring. Does NOT import
``mathgen.generators.p14a_practical_triangle_build`` -- if this file ever
needs a name from that module, the check is not independent.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make

TOPIC = "p14a_practical_triangle"

# Each fact is looked up by its own anchored key-phrase regex, independent
# of which of the 3 sentence orderings the generator drew for that seed
# (case-insensitive on the leading word only, since a sentence-initial
# "Angle QPT" and a mid-sentence "angle QPT" are the same fact).
_D_RE = re.compile(r"PQ = (\d+) m")
_E_RE = re.compile(r"QT = (\d+) m")
_PHI_RE = re.compile(r"[Aa]ngle QPT = (\d+) degrees")
_PSI_RE = re.compile(r"[Aa]ngle PQT = (\d+) degrees")
_THETA_P_RE = re.compile(r"elevation from P to the top of the tower is (\d+) degrees")
_THETA_Q_RE = re.compile(r"elevation from Q to the top of the tower is (\d+) degrees")
_BQ_RE = re.compile(r"bearing of Q from P is (\d+) degrees")
_BT_RE = re.compile(r"bearing of T from P is (\d+) degrees")


def _num(regex: re.Pattern, text: str) -> int:
    m = regex.search(text)
    assert m, f"could not find {regex.pattern!r} in {text!r}"
    return int(m.group(1))


def _interior_from_bearings(bearing_q: int, bearing_t: int) -> int:
    """Independent re-derivation of the exam tier's interior angle, via unit
    compass vectors and the dot product/acos -- NEVER the modular
    difference-then-supplement route the generator itself uses. Rounds to
    the nearest whole degree since the construction's own phi is always an
    integer degree."""
    vq = (math.sin(math.radians(bearing_q)), math.cos(math.radians(bearing_q)))
    vt = (math.sin(math.radians(bearing_t)), math.cos(math.radians(bearing_t)))
    dot = max(-1.0, min(1.0, vq[0] * vt[0] + vq[1] * vt[1]))
    return round(math.degrees(math.acos(dot)))


def _expected_easy(text: str) -> float:
    d, phi, psi, theta = (_num(r, text) for r in (_D_RE, _PHI_RE, _PSI_RE, _THETA_P_RE))
    angle_t = 180 - phi - psi
    pt = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
    return pt * math.tan(math.radians(theta))


def _expected_medium(text: str) -> float:
    d, phi, psi, theta = (_num(r, text) for r in (_D_RE, _PHI_RE, _PSI_RE, _THETA_Q_RE))
    angle_t = 180 - phi - psi
    qt = d * math.sin(math.radians(phi)) / math.sin(math.radians(angle_t))
    return qt * math.tan(math.radians(theta))


def _expected_hard(text: str) -> float:
    d, e, psi, theta = (_num(r, text) for r in (_D_RE, _E_RE, _PSI_RE, _THETA_P_RE))
    pt = math.sqrt(d * d + e * e - 2 * d * e * math.cos(math.radians(psi)))
    return pt * math.tan(math.radians(theta))


def _expected_exam(text: str) -> float:
    d, psi, bq, bt, theta = (_num(r, text) for r in (_D_RE, _PSI_RE, _BQ_RE, _BT_RE, _THETA_P_RE))
    phi = _interior_from_bearings(bq, bt)
    angle_t = 180 - phi - psi
    pt = d * math.sin(math.radians(psi)) / math.sin(math.radians(angle_t))
    return pt * math.tan(math.radians(theta))


_EXPECTED = {
    "easy": _expected_easy,
    "medium": _expected_medium,
    "hard": _expected_hard,
    "exam": _expected_exam,
}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
@pytest.mark.parametrize("seed", range(60))
def test_practical_triangle_oracle(seed: int, difficulty: str):
    q = make(TOPIC, seed=seed, difficulty=difficulty)
    expected = _EXPECTED[difficulty](q.text)
    given = float(q.answer_text)
    assert abs(expected - given) < 0.05, (
        f"seed {seed}/{difficulty}: expected {expected:.1f}, answer_text says {q.answer_text}. Question: {q.text}"
    )
    for d in q.distractors:
        assert abs(float(d) - expected) > 0.051, (
            f"seed {seed}/{difficulty}: distractor {d} within tolerance of expected {expected:.1f}. Question: {q.text}"
        )
