"""Independent oracle for senior-use-recursion-to-generate-an-arithmetic-sequence's from_two_terms hard tier.
Re-derives from rendered question text alone -- does not import mathgen.generators.p3c_recursion_build."""

from __future__ import annotations

import re

from mathgen import make

_PATTERN = re.compile(
    r"t(\d+) = (-?\d+) and t(\d+) = (-?\d+)\. Find the common difference "
    r"and the first term, then write the next (\d+) terms after t1"
)


def test_oracle_recomputes_the_terms_from_the_question_text():
    checked = 0
    for seed in range(300):
        q = make("arithmetic_recursion_generate", seed=seed, variant="from_two_terms")
        m = _PATTERN.search(q.text)
        assert m, f"unexpected question text shape: {q.text!r}"
        p, t_p, qpos, t_q, k = (int(v) for v in m.groups())
        gap = qpos - p
        assert (t_q - t_p) % gap == 0, f"seed={seed}: t_q - t_p not divisible by gap"
        d = (t_q - t_p) // gap
        t1 = t_p - (p - 1) * d
        expected_terms = [t1 + m_ * d for m_ in range(1, k + 1)]
        expected_text = ", ".join(str(t) for t in expected_terms)
        assert expected_text == q.answer_text, f"seed={seed}: {q.text!r} -> {q.answer_text!r} expected {expected_text}"
        checked += 1
    assert checked == 300
