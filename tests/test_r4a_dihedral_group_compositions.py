"""Tests `uc2-dihedral-group-compositions`: every element is read back out of the question TEXT and the product is
recomputed as a composition of permutations of the n-gon's vertices (r: x -> x + 1, s: x -> -x), then put into the form
r^i s^j; the typed ordered-list grader on every key and distractor; and the four-part exam chain."""

from __future__ import annotations

import re

import mathgen.generators.r4a_dihedral_group_compositions_build  # noqa: F401  (registers the topics)
from tests import r4a_support as sup

TOPIC = "r4a_dihedral_group_compositions"
SKILL_ID = "uc2-dihedral-group-compositions"


def _n(text):
    return int(re.search(r"dihedral group D_(\d+)", text).group(1))


def _element(text, n):
    """The vertex permutation (tuple of images of 0..n-1) of "r^3 s", "r", "s", "r^2" or "e"."""
    text = text.strip()
    out = tuple(range(n))
    if text == "e":
        return out
    for token in reversed(text.split()):  # the rightmost factor acts first
        if token == "s":
            step = tuple((-x) % n for x in range(n))
            out = tuple(step[v] for v in out)
        else:
            k = int(token[2:]) if "^" in token else 1
            out = tuple((v + k) % n for v in out)
    return out


def _compose(f, g):
    return tuple(f[g[x]] for x in range(len(g)))


def _inverse(f):
    out = [0] * len(f)
    for x, v in enumerate(f):
        out[v] = x
    return tuple(out)


def _normal_form(f):
    n = len(f)
    i = f[0] % n  # r^i s^j sends 0 to i
    reflection = f[1] != (i + 1) % n
    return [i, 1 if reflection else 0]


def _order(f):
    n, g, k = len(f), f, 1
    while g != tuple(range(n)):
        g = _compose(f, g)
        k += 1
    return k


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "ordered_list")


def test_easy_key_is_the_product_of_vertex_permutations():
    for seed in range(120):
        question = sup.q(TOPIC, "easy", seed)
        n = _n(question.text)
        x, y = re.search(r"\(([^()]*)\)\(([^()]*)\)", question.text.split("s r = r^(-1) s.")[1]).groups()
        product = _compose(_element(x, n), _element(y, n))
        assert question.answer == _normal_form(product), (seed, question.text)
        assert f"0 ≤ i < {n}" in question.text


def test_medium_key_is_the_word_reduced():
    for seed in range(120):
        question = sup.q(TOPIC, "medium", seed)
        n = _n(question.text)
        word = re.search(r"(?:word|product) ((?:(?:r\^\d+|s) ?)+)", question.text.split("s r = r^(-1) s.")[1]).group(1)
        assert len(word.split()) >= 5
        assert question.answer == _normal_form(_element(word, n)), (seed, question.text)


def test_hard_key_is_a_conjugate_or_a_commutator():
    seen = set()
    for seed in range(160):
        question = sup.q(TOPIC, "hard", seed)
        n = _n(question.text)
        body = question.text.split("s r = r^(-1) s.")[1]
        if "commutator" in body or "x y x^(-1) y^(-1)" in body:
            x, y = re.search(r"x = (.*?) and y = (.*?)\. ", body).groups()
            fx, fy = _element(x, n), _element(y, n)
            result = _compose(_compose(fx, fy), _compose(_inverse(fx), _inverse(fy)))
            seen.add("commutator")
        else:
            m = re.search(r"g = (.*?) and x = (.*?)\. ", body) or re.search(r"x = (.*?) by g = (.*?), that", body)
            g, x = (m.group(1), m.group(2)) if "where" in body else (m.group(2), m.group(1))
            fg, fx = _element(g, n), _element(x, n)
            result = _compose(_compose(fg, fx), _inverse(fg))
            seen.add("conjugate")
        assert question.answer == _normal_form(result), (seed, question.text)
    assert seen == {"commutator", "conjugate"}


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "ordered_list")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 15, "medium": 12, "hard": 12})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(
        SKILL_ID, TOPIC, "abcd", (2, 2, 2, 3), ("ordered_list", "number", "number", "number"), (0, 0, 1, 1)
    )


def test_exam_keys_by_vertex_permutations():
    for seed in range(120):
        question = sup.exam(TOPIC, seed)
        n = _n(question.stem)
        x, y = re.search(r"Let x = (.*?) and y = (.*?), and let", question.stem).groups()
        fx, fy = _element(x, n), _element(y, n)
        xy = _compose(fx, fy)
        a, b, c, d = question.parts
        assert a.answer == _normal_form(xy)
        assert int(b.answer) == _order(xy)
        assert int(c.answer) == 2 * n // _order(xy)
        commutator = _compose(xy, _compose(_inverse(fx), _inverse(fy)))
        assert int(d.answer) == _order(commutator)


def test_exam_follow_through_from_a_wrong_order():
    from mathgen.engine import grade

    s = sup.skill(SKILL_ID)
    checked = 0
    for seed in range(120):
        question = sup.exam(TOPIC, seed)
        n = _n(question.stem)
        b = int(question.parts[1].answer)
        wrong = b + 2 if b % 2 == 0 else b + 1
        sent = {p.label: p.answer_text for p in question.parts} | {"b": str(wrong)}
        if (2 * n) % wrong:
            continue
        sent["c"] = str(2 * n // wrong)
        sent["d"] = str(wrong // 2)  # the wrong order is even, so the commutator order is half of it
        if sent["c"] in (str(wrong), question.parts[2].answer_text) or sent["d"] == question.parts[3].answer_text:
            continue
        result = grade(s, question, sent)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, sent)
        assert result.parts["d"].marks == 3 and result.parts["d"].follow_through is True, (seed, sent)
        checked += 1
    assert checked >= 5, checked
