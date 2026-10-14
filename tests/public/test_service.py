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


def test_an_id_secret_hides_the_generator_seed(lib):
    secret = "x" * 16
    keyed = service.Library.load(id_secret=secret)
    other = service.Library.load(id_secret="y" * 16)
    (a,) = service.make_questions(keyed, "simplify-surds", seed=11, answers=True)
    assert service.make_questions(keyed, "simplify-surds", seed=11, answers=True) == [a]
    assert a["id"].split("/")[2] == "11"
    assert service.grade_reply(keyed, a["id"], a["answer"]["text"])["correct"] is True
    (plain,) = service.make_questions(lib, "simplify-surds", seed=11, answers=True)
    assert plain["id"] != a["id"]
    with pytest.raises(service.QuestionChanged):
        service.grade_reply(other, a["id"], a["answer"]["text"])  # another secret: not this server's id
    with pytest.raises(service.QuestionChanged):
        service.grade_reply(lib, a["id"], a["answer"]["text"])  # no secret: the plain seed 11 is another question
    with pytest.raises(ValueError):
        service.Library.load(id_secret="short")


def test_an_oversized_source_index_is_a_clean_not_found(lib):
    with pytest.raises(service.NotFound, match="no source number"):
        service.make_questions(lib, "simplify-surds", source=10**30)


def test_senior_skills_without_the_senior_prefix_are_attributed_too(lib):
    # `combinations` is tagged `acara-v8-4` and lives in senior-mathematics, but its id has no `senior-` prefix.
    combinations = lib.skill("combinations")
    assert "acara-senior-v8-4" in service.frameworks(combinations)
    assert service.skill_summary(combinations)["attribution"] == service.ACARA_ATTRIBUTION
    paper = service.build_paper(lib, {"seed": 1, "sections": [{"items": [{"skill": "combinations"}]}]})
    assert service.ACARA_ATTRIBUTION in paper["attribution"]
    assert "attribution" not in service.skill_summary(lib.skill("simplify-surds"))


def test_a_figure_the_renderer_cannot_draw_still_gives_a_question(monkeypatch, lib):
    # A figure kind newer than the SVG renderer must not stop the question being served.
    import mathgen.svg

    def cannot(spec, **_):
        raise ValueError("cannot render figure kind")

    monkeypatch.setattr(mathgen.svg, "figure_svg", cannot)
    (view,) = service.make_questions(lib, "f10-measure-angle-protractor", seed=1)
    assert view["figure"]["svg"] is None and view["figure"]["alt"] and "spec" not in view["figure"]


def test_a_multiple_choice_reply_is_read_as_option_text_first(lib):
    (view,) = service.make_questions(lib, "add-within-10", seed=0, answers=True)
    options, right = view["options"], view["answer"]["option"]
    assert options[right] == "8" and right != 8  # the case: the right option's text is not its index
    wrong = next(o for o in options if o != "8" and not (o.isdigit() and int(o) < len(options)))

    def mark(reply):
        return service.grade_reply(lib, view["id"], reply)["correct"]

    assert mark("8") and mark(" 8 ")  # the option 8, not the ninth option
    assert mark(right)  # an integer is always an index
    assert mark(chr(ord("A") + right))  # a letter, as the grader reads it
    assert mark(str(right))  # text that names no option falls back to the grader's index reading
    assert not mark(wrong)
