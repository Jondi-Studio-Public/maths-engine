"""`mathgen.mcp_server`: the MCP tools, driven through the MCP SDK's in-process client, in both roles."""

import asyncio

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from mathgen import mcp_server  # noqa: E402
from mathgen.server.pool import WorkerPool  # noqa: E402

PAPER = {"title": "Surds quiz", "seed": 1, "sections": [{"items": [{"skill": "simplify-surds", "count": 3}]}]}
#: Skills that exercise every shape a learner view can take: a figure, multiple choice, a table, several parts.
SHAPES = [
    ("count-to-5", None),
    ("integer-polynomial-quotient-remainder", None),
    ("f10-collect-sort-compare-categorical-data", None),
    ("permutation-composition", "exam"),
    ("simplify-surds", None),
]


@pytest.fixture(scope="module")
def pool():
    pool = WorkerPool(1)
    yield pool
    pool.close()


def call(server, tool, args=None):
    """`(structured result, is_error, text)` of one tool call."""

    async def go():
        async with Client(server) as client:
            result = await client.call_tool(tool, args or {})
            text = result.content[0].text if result.content else ""
            return result.structured_content, result.is_error, text

    return asyncio.run(go())


def tools(server):
    async def go():
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return asyncio.run(go())


@pytest.fixture(scope="module")
def learner(pool, tmp_path_factory):
    return mcp_server.build_server("learner", pool=pool, out_dir=tmp_path_factory.mktemp("learner"))


@pytest.fixture(scope="module")
def author(pool, tmp_path_factory):
    return mcp_server.build_server("author", pool=pool, out_dir=tmp_path_factory.mktemp("author"))


def _keys(data):
    """Every dict key anywhere in `data`."""
    if isinstance(data, dict):
        for k, v in data.items():
            yield k
            yield from _keys(v)
    elif isinstance(data, list):
        for v in data:
            yield from _keys(v)


def test_learner_tools_take_no_answer_flags(learner):
    found = tools(learner)
    assert set(found) == {
        "engine_info",
        "find_skills",
        "get_skill",
        "make_questions",
        "get_question",
        "grade_answer",
        "build_paper",
    }
    for tool in found.values():
        props = tool.input_schema.get("properties", {})
        assert "answers" not in props and "reveal" not in props, tool.name
    assert found["build_paper"].input_schema["properties"]["format"]["enum"] == ["json", "html"]


def test_author_tools_offer_answers(author):
    found = tools(author)
    assert "answers" in found["make_questions"].input_schema["properties"]
    assert "reveal" in found["grade_answer"].input_schema["properties"]
    assert set(found["build_paper"].input_schema["properties"]["format"]["enum"]) == set(mcp_server.AUTHOR_FORMATS)


def test_learner_cannot_smuggle_an_answer_flag(learner):
    data, is_error, text = call(learner, "make_questions", {"skill_id": "simplify-surds", "answers": True})
    # Either refused as an unknown argument or ignored; never an answer.
    assert is_error or "answer" not in set(_keys(data))


@pytest.mark.parametrize("skill_id, source", SHAPES)
def test_learner_views_never_hold_answers(learner, author, skill_id, source):
    args = {"skill_id": skill_id, "count": 2, "seed": 11, "include_figures": True}
    if source:
        args["source"] = source
    data, is_error, text = call(learner, "make_questions", args)
    assert not is_error, text
    keys = set(_keys(data))
    assert not keys & {"answer", "spec", "canonical", "working"}, keys
    full, _, _ = call(author, "make_questions", {**args, "answers": True})
    # Same questions in both roles: the learner view is the author view less its answers.
    assert [q["id"] for q in data["questions"]] == [q["id"] for q in full["questions"]]
    assert all("answer" in q or "parts" in q for q in full["questions"])

    qid = data["questions"][0]["id"]
    again, _, _ = call(learner, "get_question", {"question_id": qid})
    assert again["id"] == qid and "answer" not in set(_keys(again))


def test_learner_grading_never_reveals(learner, author):
    data, _, _ = call(author, "make_questions", {"skill_id": "simplify-surds", "seed": 5, "answers": True})
    q = data["questions"][0]
    right, wrong = q["answer"]["text"], "1"
    for reply in (right, wrong):
        result, is_error, text = call(learner, "grade_answer", {"question_id": q["id"], "reply": reply})
        assert not is_error, text
        assert "canonical" not in set(_keys(result))
    assert call(learner, "grade_answer", {"question_id": q["id"], "reply": right})[0]["correct"] is True
    assert call(learner, "grade_answer", {"question_id": q["id"], "reply": wrong})[0]["correct"] is False
    shown, _, _ = call(author, "grade_answer", {"question_id": q["id"], "reply": wrong, "reveal": True})
    assert shown["canonical"]


def test_multi_part_replies_are_marked_by_part(learner, author):
    data, _, _ = call(
        author,
        "make_questions",
        {"skill_id": "permutation-composition", "source": "exam", "seed": 3, "answers": True},
    )
    q = data["questions"][0]
    reply = {p["label"]: p["answer"]["text"] for p in q["parts"]}
    result, is_error, text = call(learner, "grade_answer", {"question_id": q["id"], "reply": reply})
    assert not is_error, text
    assert result["correct"] is True and result["marks"] == result["max_marks"]


def test_figures_are_alt_text_unless_asked(learner):
    plain, _, _ = call(learner, "make_questions", {"skill_id": "count-to-5", "seed": 2})
    drawn, _, _ = call(learner, "make_questions", {"skill_id": "count-to-5", "seed": 2, "include_figures": True})
    assert plain["questions"][0]["figure"].keys() == {"alt"}
    assert drawn["questions"][0]["figure"]["svg"].lstrip().startswith("<svg")


def test_find_and_describe_skills(learner):
    data, is_error, _ = call(learner, "find_skills", {"text": "surd", "limit": 3})
    assert not is_error and data["total"] >= len(data["skills"]) and len(data["skills"]) <= 3
    every, _, _ = call(learner, "find_skills", {"text": "surd", "limit": 100})
    assert every["total"] == data["total"] and "simplify-surds" in [s["id"] for s in every["skills"]]
    year8, _, _ = call(learner, "find_skills", {"curriculum_code": "AC9M8", "limit": 100})
    assert year8["skills"] and all(any(c.startswith("AC9M8") for c in s["curriculum_codes"]) for s in year8["skills"])
    skill, _, _ = call(learner, "get_skill", {"skill_id": "simplify-surds"})
    assert skill["id"] == "simplify-surds" and "prerequisites" in skill
    info, _, _ = call(learner, "engine_info")
    assert info["role"] == "learner" and info["paper_formats"] == ["json", "html"]


def test_errors_come_back_as_tool_errors(learner):
    for tool, args in [
        ("make_questions", {"skill_id": "no-such-skill"}),
        ("get_skill", {"skill_id": "no-such-skill"}),
        ("get_question", {"question_id": "nonsense"}),
        ("grade_answer", {"question_id": "simplify-surds/easy/1/000000000000", "reply": "1"}),
        ("make_questions", {"skill_id": "simplify-surds", "count": mcp_server.MAX_COUNT + 1}),
    ]:
        _, is_error, text = call(learner, tool, args)
        assert is_error, (tool, text)


def test_learner_papers_have_no_answers(learner):
    data, is_error, text = call(learner, "build_paper", {"spec": PAPER})
    assert not is_error, text
    assert not set(_keys(data["paper"])) & {"answer", "spec", "canonical"}
    _, is_error, _ = call(learner, "build_paper", {"spec": PAPER, "format": "html-key"})
    assert is_error


def test_author_files_land_in_the_output_folder_under_server_names(author, tmp_path):
    server = mcp_server.build_server("author", pool=author._maths_engine.pool, out_dir=tmp_path)
    spec = {**PAPER, "title": "../../etc/passwd", "versions": 2}
    paths = []
    for fmt, version in [("html", None), ("html-key", None), ("moodle", "B"), ("qti", "A")]:
        args = {"spec": spec, "format": fmt}
        if version:
            args["version"] = version
        data, is_error, text = call(server, "build_paper", args)
        assert not is_error, text
        paths.append(data["path"])
        assert data["bytes"] > 0
    assert len(set(paths)) == 4
    for path in paths:
        written = tmp_path / path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        assert written.is_file() and str(written.resolve()) == path
        assert written.name.startswith("etc-passwd")
    assert (tmp_path / paths[3].rsplit("/", 1)[-1]).read_bytes()[:2] == b"PK"


def test_roles_are_checked():
    with pytest.raises(ValueError):
        mcp_server.build_server("admin")


def test_command_line_parses_the_plugin_arguments(monkeypatch):
    seen = {}

    class Fake:
        _maths_engine = type("E", (), {"close": lambda self: None})()

        def run(self, transport):
            seen["transport"] = transport

    def fake_build(role, **kwargs):
        seen.update(role=role, **kwargs)
        return Fake()

    monkeypatch.setattr(mcp_server, "build_server", fake_build)
    assert mcp_server.main(["--role", "author", "--out-dir", "/tmp/x"]) == 0
    assert seen["role"] == "author" and seen["out_dir"] == "/tmp/x" and seen["transport"] == "stdio"
