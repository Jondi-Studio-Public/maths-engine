"""`mathgen.service`: the JSON-ready layer the server and the exports build on."""

import json

import pytest

from mathgen import service


@pytest.fixture(scope="module")
def lib():
    return service.Library.load()


def test_everything_returned_is_plain_json(lib):
    for skill in service.find_skills(lib)[::40]:
        for view in service.make_questions(lib, skill.id, seed=3, answers=True):
            assert json.loads(json.dumps(view)) == view


def test_derived_seeds_are_the_same_on_every_machine():
    # Pinned: a paper's questions come from these seeds, so changing the derivation changes every paper.
    assert service.derive_seed(42, "A", 0, 0, 0, 0) == service.derive_seed("42", "A", "0", "0", "0", "0")
    assert service.derive_seed(42, "A", 0, 0, 0, 0) == 2110393933


def test_curriculum_codes_and_years_come_from_tags(lib):
    tagged = [s for s in lib.skills if service.curriculum_codes(s)]
    assert len(tagged) > 100
    for skill in tagged:
        for code in service.curriculum_codes(skill):
            assert code == code.upper()


def test_filters_combine(lib):
    year8 = service.find_skills(lib, year="8")
    mcq8 = service.find_skills(lib, year="8", answer_kind="mcq")
    assert set(s.id for s in mcq8) <= set(s.id for s in year8)
    assert all(s.answer == "mcq" for s in mcq8)
    assert all("surd" in (s.id + s.name + s.description).casefold() for s in service.find_skills(lib, text="SURD"))


def test_make_questions_avoids_repeats_while_fresh_ones_come(lib):
    views = service.make_questions(lib, "simplify-surds", seed=1, count=10)
    assert len(views) == 10
    assert len({v["text"] for v in views}) == 10


def test_limits_are_enforced(lib):
    with pytest.raises(service.BadRequest):
        service.make_questions(lib, "simplify-surds", count=service.MAX_COUNT + 1)
    big = {"versions": 10, "sections": [{"items": [{"skill": "simplify-surds", "count": 50}]}]}
    with pytest.raises(service.BadRequest):
        service.build_paper(lib, big)
    with pytest.raises(service.NotFound):
        service.make_questions(lib, "simplify-surds", source="no-such-source")


def test_a_paper_without_answers_holds_none(lib):
    spec = {"seed": 9, "sections": [{"items": [{"skill": "simplify-surds", "count": 2}]}]}
    paper = service.build_paper(lib, spec, answers=False)
    assert not any("answer" in q for q in paper["versions"][0]["sections"][0]["questions"])
    keyed = service.build_paper(lib, spec, answers=True)
    assert [q["id"] for q in keyed["versions"][0]["sections"][0]["questions"]] == [
        q["id"] for q in paper["versions"][0]["sections"][0]["questions"]
    ]


def test_senior_skills_carry_the_acara_attribution(lib):
    senior = next(s for s in service.find_skills(lib) if s.id.startswith("senior-"))
    assert service.skill_summary(senior)["attribution"] == service.ACARA_ATTRIBUTION
    paper = service.build_paper(lib, {"seed": 1, "sections": [{"items": [{"skill": senior.id}]}]})
    assert service.ACARA_ATTRIBUTION in paper["attribution"]


def test_rubric_skills_are_not_graded_by_the_engine(lib):
    rubric = [s for s in lib.skills if s.answer == "rubric" and s.status == "active" and s.sources]
    if not rubric:
        pytest.skip("no active rubric skill in the library")
    (view,) = service.make_questions(lib, rubric[0].id, seed=1)
    assert view["gradeable"] is False
    with pytest.raises(service.BadRequest):
        service.grade_reply(lib, view["id"], "a proof")
