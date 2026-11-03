"""The exponential_g recipe: conformance, and each part's key and distractors against the real graders."""

import math
import re

import pytest

from mathgen import core, recipes
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE

RECIPE = "methods_exponential_g_1"
TOPIC = "recipe_" + RECIPE


def _g(a, b, c, x):
    return a ** (2 * x) - b * a**x + c


def _stem(q):
    m = re.search(r"g\(x\) = (\d+)\^\(2x\) - (\d+)\*\d+\^x \+ (\d+)", q.text)
    return tuple(int(v) for v in m.groups())


def _ln(a, text):
    return f"ln({text.split('(')[1][:-1]})/ln({a})"


def test_conformance():
    assert recipes.conformance(RECIPE, seeds=300) == []


@pytest.mark.parametrize("seed", range(60))
def test_keys_are_right_and_graded_right(seed):
    q = core.make(TOPIC, seed=seed)
    a, b, c = _stem(q)
    pa, pb, pc, pd = q.parts
    k = int(re.search(rf"log_{a}\((\d+)\)", pa.prompt).group(1))
    assert pa.answer == round(_g(a, b, c, math.log(k, a)))  # independent numeric evaluation
    assert abs(_g(a, b, c, pb.answer)) < 1e-9
    other = int(re.search(r"log_\d+\((\d+)\)", pc.answer_text).group(1))
    assert abs(_g(a, b, c, math.log(other, a))) < 1e-6
    d0 = int(re.search(r"g\(x\) = (-?\d+)", pd.prompt).group(1))
    big = int(re.search(r"log_\d+\((\d+)\)", pd.answer_text).group(1))
    assert _g(a, b, c, math.log(big, a)) == pytest.approx(d0, abs=1e-6)
    # the larger solution: the other solution of g(x) = d0 is smaller
    small = b - big
    assert _g(a, b, c, math.log(small, a)) == pytest.approx(d0, abs=1e-6) and small < big
    reply = {"a": str(pa.answer), "b": str(pb.answer), "c": _ln(a, pc.answer_text), "d": _ln(a, pd.answer_text)}
    assert grade("number", q, reply).correct
    for part, label in zip(q.parts, "abcd"):
        notes = part.option_notes or {}
        for text in part.distractors:
            wrong = dict(reply, **{label: text})
            assert not grade("number", q, wrong).parts[label].correct, (seed, label, text)
            assert text in notes and notes[text] in CATALOGUE, (seed, label, text)


@pytest.mark.parametrize("seed", range(120))
def test_options_do_not_leak_across_parts(seed):
    q = core.make(TOPIC, seed=seed)
    shown = [{p.answer_text, *p.distractors} for p in q.parts]
    for i, later in enumerate(q.parts):
        for j in range(i):
            # an earlier part's options never show a later part's key
            assert later.answer_text not in shown[j], (seed, "part", j, "shows part", i, "key")
            # and no logarithm option is repeated from an earlier part
            repeated = {t for t in later.distractors if "log" in t} & shown[j]
            assert not repeated, (seed, i, j, repeated)
