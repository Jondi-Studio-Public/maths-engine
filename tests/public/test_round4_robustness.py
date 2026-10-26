"""Round 4 robustness fixes: repeated labels, container input to the safe reader, seeds and ids, figure ranges."""

import time

import pytest
import sympy
from fastapi.testclient import TestClient

from mathgen import core, engine, service, svg
from mathgen.grading.safe_parse import sympify_safely
from mathgen.figure_kinds.data import bar_chart
from mathgen.server import Config, create_app

SKILLS = {s.id: s for s in engine.load_skills()[1]}
PAYLOAD = "__import__('os').getpid()"


@pytest.mark.parametrize("read", [sympify_safely, core._sympify])
def test_dicts_and_sets_are_read_by_the_safe_parser(read):
    for value in ({"a": PAYLOAD}, {PAYLOAD: 1}, {PAYLOAD}, frozenset({PAYLOAD})):
        with pytest.raises(sympy.SympifyError):
            read(value)
    assert read({"a": "1+1"}) == {sympy.Symbol("a"): 2}
    assert read({"1", "2"}) == {sympy.Integer(1), sympy.Integer(2)}


def test_a_long_run_of_repeated_labels_grades_quickly():
    skill = SKILLS["uc2-cosets-and-index"]
    question, _ = engine.make_with_id(skill, 1, source=0)
    start = time.monotonic()
    for count in (20_000, 40_000):
        assert not engine.grade(skill, question, "x=" * count).correct
    assert time.monotonic() - start < 3.0


def test_grade_with_deadline_is_public():
    assert "grade_with_deadline" in engine.__all__
    skill = SKILLS["simplify-surds"]
    question, _ = engine.make_with_id(skill, 7)
    assert engine.grade_with_deadline(skill, question, question.answer_text).correct


def test_make_with_id_refuses_seeds_it_cannot_name_and_names_a_random_one():
    skill = SKILLS["simplify-surds"]
    for bad in (True, 1.5, "7", b"7", float("nan")):
        with pytest.raises(TypeError):
            engine.make_with_id(skill, bad)
    question, qid = engine.make_with_id(skill, None)
    assert f"/{question.seed}/" in qid
    assert engine.regenerate(qid, [skill]).text == question.text


@pytest.mark.parametrize("seed", ["1_0", " 1", "١", "+1", "1.0", ""])
def test_parse_question_id_wants_plain_digits(seed):
    with pytest.raises(ValueError):
        engine.parse_question_id(f"skill/key/{seed}/abcdef")
    assert engine.parse_question_id("skill/key/-12/abcdef")[2] == -12


def test_year_university_matches_university_skills_in_any_case():
    lib = service.Library.load()
    hits = service.find_skills(lib, year="university")
    assert hits and len(hits) == len(service.find_skills(lib, year="University"))
    assert len(service.find_skills(lib, year="10a")) == len(service.find_skills(lib, year="10A"))
    assert len(service.find_skills(lib, year="f")) > 0


def test_a_boolean_reply_is_not_an_option_index():
    key = "k" * 24
    with TestClient(create_app(Config(keys=[(key, "learner")]))) as client:
        for payload in ({"id": "a/b/1/c", "reply": False}, {"id": "a/b/1/c", "reply": {"a": True}}):
            r = client.post("/v1/grade", json=payload, headers={"Authorization": f"Bearer {key}"})
            assert r.status_code == 422


@pytest.mark.parametrize("error", [OverflowError("too big"), ZeroDivisionError("zero")])
def test_a_renderer_overflow_is_the_documented_value_error(monkeypatch, error):
    spec = bar_chart(categories=["a"], series=[{"name": "", "values": [2]}], value_max=10, value_step=2, alt="x")

    def boom(spec, ids):
        raise error

    monkeypatch.setitem(svg._RENDERERS, "bar_chart", boom)
    with pytest.raises(ValueError, match="out of range"):
        svg.figure_svg(spec)
