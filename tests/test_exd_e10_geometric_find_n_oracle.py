"""Independent oracle for senior-use-the-formula-t-n-r-n-1's find_n hard tier. Re-derives from rendered question
text alone -- does not import mathgen.generators.p3c_logarithms_b_build."""

from __future__ import annotations

import re

from mathgen import make

_PATTERN = re.compile(
    r"The number (\d+) is a term of the geometric sequence with t1 = (\d+) and "
    r"common ratio r = (\d+)"
)


def test_oracle_recomputes_n_from_the_question_text():
    checked = 0
    for seed in range(300):
        q = make("geometric_nth_term_formula", seed=seed, variant="find_n")
        m = _PATTERN.search(q.text)
        assert m, f"unexpected question text shape: {q.text!r}"
        t_n, t1, r = (int(v) for v in m.groups())
        value = t1
        found_n = None
        for n in range(1, 20):
            if value == t_n:
                found_n = n
                break
            value *= r
        assert found_n is not None, f"seed={seed}: no n found for t1={t1} r={r} t_n={t_n}"
        assert str(found_n) == q.answer_text, f"seed={seed}: {q.text!r} -> {q.answer_text!r} expected {found_n}"
        checked += 1
    assert checked == 300
