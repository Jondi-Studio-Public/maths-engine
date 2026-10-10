"""Independent oracle for `pv0c_tangent_secant`.

Imports NOTHING from `mathgen.generators.pv0c_tangent_secant_build`. Parses
`question.text` with a regex and recomputes the requested length from
scratch using only AM x BM = TM^2 (and, for the centre variant,
TM^2 + r^2 = MO^2), then compares against `question.answer_text`.

Proven red: temporarily editing the generator's `answer_text` to `str(answer + 1)` (kept as a note here, not
committed) makes every one of these assertions fail; reverting restores green.
"""

from __future__ import annotations

import math
import re

from mathgen import make

TOPIC = "pv0c_tangent_secant"
SEEDS_PER_TIER = 300


def _num(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    assert m, f"pattern {pattern!r} not found in {text!r}"
    return int(m.group(1))


def _oracle_easy(text: str) -> int:
    am = _num(text, r"AM (?:is|=) (\d+) cm")
    bm = _num(text, r"BM (?:is|=) (\d+) cm")
    root = math.isqrt(am * bm)
    assert root * root == am * bm
    return root  # TM


def _oracle_medium(text: str) -> int:
    tm = _num(text, r"TM (?:is|=) (\d+) cm")
    asks_for_am = text.rstrip(".").endswith("AM, in cm")
    asks_for_bm = text.rstrip(".").endswith("BM, in cm")
    assert asks_for_am != asks_for_bm, text
    if asks_for_am:
        bm = _num(text, r"BM (?:is|=) (\d+) cm")
        assert (tm * tm) % bm == 0
        return (tm * tm) // bm  # AM
    am = _num(text, r"AM (?:is|=) (\d+) cm")
    assert (tm * tm) % am == 0
    return (tm * tm) // am  # BM


def _oracle_hard(text: str) -> int:
    bm = _num(text, r"BM (?:is|=) (\d+) cm")
    asks_for_tm = text.rstrip(".").endswith("TM, in cm")
    if asks_for_tm:
        ab = _num(text, r"AB (?:is|=) (\d+) cm")
        am = ab + bm
        root = math.isqrt(am * bm)
        assert root * root == am * bm
        return root  # TM
    tm = _num(text, r"TM (?:is|=) (\d+) cm")
    assert (tm * tm) % bm == 0
    am = (tm * tm) // bm
    return am - bm  # AB


def _oracle_exam(text: str) -> int:
    if "centre O" in text:
        r = _num(text, r"radius (\d+) cm")
        tm = _num(text, r"TM is (\d+) cm")
        mo_sq = tm * tm + r * r
        root = math.isqrt(mo_sq)
        assert root * root == mo_sq
        return root  # MO
    tm = _num(text, r"TM is (\d+) cm")
    ab = _num(text, r"chord AB is (\d+) cm")
    # x*(x + ab) = tm^2 (find BM) or x*(x - ab) = tm^2 (find AM), positive root
    disc_bm = ab * ab + 4 * tm * tm
    root_bm = math.isqrt(disc_bm)
    if "length of BM" in text:
        assert root_bm * root_bm == disc_bm
        assert (root_bm - ab) % 2 == 0
        return (root_bm - ab) // 2  # BM
    disc_am = ab * ab + 4 * tm * tm
    root_am = math.isqrt(disc_am)
    assert root_am * root_am == disc_am
    assert (root_am + ab) % 2 == 0
    return (root_am + ab) // 2  # AM


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


def test_oracle_rederives_answer_every_tier():
    total = 0
    for tier, oracle in _ORACLES.items():
        for seed in range(SEEDS_PER_TIER):
            q = make(TOPIC, seed=seed, difficulty=tier)
            derived = oracle(q.text)
            assert derived == int(q.answer_text), (
                f"tier={tier} seed={seed}: oracle derived {derived}, question said {q.answer_text}\n{q.text}"
            )
            total += 1
    assert total == SEEDS_PER_TIER * 4


def test_oracle_rejects_a_perturbed_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    derived = _oracle_easy(q.text)
    assert derived == int(q.answer_text)
    perturbed = str(int(q.answer_text) + 1)
    assert derived != int(perturbed)
