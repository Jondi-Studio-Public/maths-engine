"""Re-derives each rendered question's graded pair from `question.text` alone via the Cramer determinant
formulas, never importing anything from `mathgen.generators.p14c_linear_system_determinant_build`.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make

SEEDS = range(80)
TOPIC = "p14c_linear_system_determinant"

_MAT_RE = re.compile(r"A = \[(-?\d+) (-?\d+); (-?\d+) (-?\d+)\]")
_B_RE = re.compile(r"b = \((-?\d+), (-?\d+)\)")
_XY_RE = re.compile(r"x = (-?\d+), y = (-?\d+)")
_D_PATTERNS = (
    re.compile(r"det\(A\) = D = (-?\d+)"),
    re.compile(r"D = det\(A\) = (-?\d+)"),
    re.compile(r"det\(A\) = (-?\d+)"),
)


def _system_from_text(text: str):
    a11, a12, a21, a22 = (int(v) for v in _MAT_RE.search(text).groups())
    b1, b2 = (int(v) for v in _B_RE.search(text).groups())
    return a11, a12, a21, a22, b1, b2


def _oracle_given(text: str) -> dict:
    a11, a12, a21, a22, b1, b2 = _system_from_text(text)
    D = a11 * a22 - a12 * a21
    Dx = b1 * a22 - a12 * b2
    Dy = a11 * b2 - b1 * a21
    return {"D": D, "D_x": Dx, "D_y": Dy}


def _oracle_exam(text: str) -> dict:
    x, y = (int(v) for v in _XY_RE.search(text).groups())
    D = None
    for pattern in _D_PATTERNS:
        m = pattern.search(text)
        if m:
            D = int(m.group(1))
            break
    assert D is not None, f"could not find D in {text!r}"
    return {"D_x": x * D, "D_y": y * D}


@pytest.mark.parametrize(
    "tier,label1,label2",
    [
        ("easy", "D_x", "D"),
        ("medium", "D_y", "D"),
        ("hard", "D_x", "D_y"),
    ],
)
def test_oracle_recomputes(tier, label1, label2):
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_values = _oracle_given(q.text)
        assert oracle_values[label1] == q.answer[label1], (
            f"{tier} seed={seed}: oracle {label1}={oracle_values[label1]} != "
            f"canonical {q.answer[label1]} (text={q.text!r})"
        )
        assert oracle_values[label2] == q.answer[label2], (
            f"{tier} seed={seed}: oracle {label2}={oracle_values[label2]} != "
            f"canonical {q.answer[label2]} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_recomputes_exam():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        oracle_values = _oracle_exam(q.text)
        assert oracle_values["D_x"] == q.answer["D_x"], (
            f"exam seed={seed}: oracle D_x={oracle_values['D_x']} != canonical {q.answer['D_x']} (text={q.text!r})"
        )
        assert oracle_values["D_y"] == q.answer["D_y"], (
            f"exam seed={seed}: oracle D_y={oracle_values['D_y']} != canonical {q.answer['D_y']} (text={q.text!r})"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    oracle_values = _oracle_given(q.text)
    corrupted = oracle_values["D_x"] + 1
    assert oracle_values["D_x"] != corrupted


def test_exam_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="exam")
    oracle_values = _oracle_exam(q.text)
    corrupted = oracle_values["D_x"] + 1
    assert oracle_values["D_x"] != corrupted
