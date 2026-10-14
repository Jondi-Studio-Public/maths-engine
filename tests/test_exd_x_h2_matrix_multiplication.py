"""Exam tier of the matrix-multiplication skill: AB - BA, a matrix polynomial, and solving AB = C for unknowns.

The oracle parses the matrices out of the rendered prompt with `ast.literal_eval` and recomputes with plain
nested loops (and a search over the integers for the unknowns). It shares no code with the generator.
"""

from __future__ import annotations

import ast
import re

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_matrix_multiplication"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_easy_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    easy = {prompt_shape(make("matrix_product_mcq", seed=s, difficulty="easy")) for s in range(60)}
    assert len(shapes - easy) >= 3, (shapes, easy)


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def _matrices(text: str) -> list:
    return [ast.literal_eval(m) for m in re.findall(r"\[\[[^A-Za-z]*?\]\]", text)]


def _oracle(text: str) -> str:
    if "[[x," in text:
        a_text, b_text, c_text = re.findall(r"\[\[.*?\]\]", text)
        c = ast.literal_eval(c_text)
        hits = []
        for x in range(-12, 13):
            for y in range(-12, 13):
                a = ast.literal_eval(a_text.replace("x", str(x)))
                b = ast.literal_eval(b_text.replace("y", str(y)))
                if _mul(a, b) == c:
                    hits.append((x, y))
        assert len(hits) == 1, (text, hits)
        return f"x = {hits[0][0]}, y = {hits[0][1]}"
    if "AB - BA" in text:
        a, b = _matrices(text)
        ab, ba = _mul(a, b), _mul(b, a)
        return str([[ab[i][j] - ba[i][j] for j in range(2)] for i in range(2)])
    (a,) = _matrices(text)
    k, m = (int(v) for v in re.search(r"A\^2 - (\d+)A \+ (\d+)I", text).groups())
    a2 = _mul(a, a)
    return str([[a2[i][j] - k * a[i][j] + (m if i == j else 0) for j in range(2)] for i in range(2)])


def test_key_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if expected != q.answer_text:
            mismatches.append((seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches[:3]


def test_grader_accepts_key_option_and_rejects_every_distractor_option():
    grade = GRADERS["mcq"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        options = mcq_options(q)
        assert len(options) >= 3 and len(set(options)) == len(options), seed
        for index, option in enumerate(options):
            if grade(q, index).correct != (option == q.answer_text):
                bad.append((seed, index, option, q.answer_text))
    assert bad == [], bad[:3]


def test_deterministic_and_all_three_kinds_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    kinds = set()
    for seed in range(60):
        t = _exam(seed).text
        kinds.add("unknowns" if "[[x," in t else ("commutator" if "AB - BA" in t else "polynomial"))
    assert kinds == {"unknowns", "commutator", "polynomial"}
