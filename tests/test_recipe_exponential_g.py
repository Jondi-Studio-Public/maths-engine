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
    assert pb.answer + math.log(other, a) == pytest.approx(math.log(int(pd.answer_text.split("(")[1][:-1]), a))
    reply = {
        "a": str(pa.answer),
        "b": str(pb.answer),
        "c": f"ln({other})/ln({a})",
        "d": f"ln({pd.answer_text.split('(')[1][:-1]})/ln({a})",
    }
    assert grade("number", q, reply).correct
    for part, label in zip(q.parts, "abcd"):
        for text in part.distractors:
            wrong = dict(reply, **{label: text})
            assert not grade("number", q, wrong).parts[label].correct, (seed, label, text)
            assert (part.option_notes or {}).get(text, "squared-as-doubled") in CATALOGUE
