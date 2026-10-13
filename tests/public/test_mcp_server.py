"""`mathgen.mcp_server`: the MCP tools, driven through the MCP SDK's in-process client, in both roles."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from mathgen import mcp_server  # noqa: E402
from mathgen.server import pool as pool_module  # noqa: E402
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
    # The SDK ignores an argument a tool does not declare, so the flag is dropped, never honoured.
    data, is_error, text = call(learner, "make_questions", {"skill_id": "simplify-surds", "answers": True})
    assert not is_error, text
    assert "answer" not in set(_keys(data))
    result, is_error, text = call(learner, "grade_answer", {"question_id": data["questions"][0]["id"], "reply": "1"})
    assert not is_error, text
    reveal, _, _ = call(
        learner, "grade_answer", {"question_id": data["questions"][0]["id"], "reply": "1", "reveal": True}
    )
    assert "canonical" not in set(_keys(reveal))


def _answer_texts(view):
    """The answer texts in an author view worth searching for: long enough not to turn up by chance."""
    blocks = [view.get("answer")] + [p.get("answer") for p in view.get("parts") or []]
    return {b["text"] for b in blocks if b and len(b.get("text") or "") >= 4}


@pytest.mark.parametrize("skill_id, source", [s for s in SHAPES if s[0] != "integer-polynomial-quotient-remainder"])
def test_no_answer_text_reaches_the_learner(learner, author, skill_id, source):
    """Value-level, not key-level: the author's answer text appears nowhere in what the learner role returns for
    the same questions (multiple choice is left out: its options hold the answer by design)."""
    args = {"skill_id": skill_id, "count": 3, "seed": 21, **({"source": source} if source else {})}
    full, _, _ = call(author, "make_questions", {**args, "answers": True})
    shown, _, _ = call(learner, "make_questions", args)
    seen = [json.dumps(shown, ensure_ascii=False)]
    for view in shown["questions"]:
        seen.append(json.dumps(call(learner, "get_question", {"question_id": view["id"]})[0], ensure_ascii=False))
        reply = {p["label"]: "1" for p in view["parts"]} if view.get("parts") else "1"
        seen.append(json.dumps(call(learner, "grade_answer", {"question_id": view["id"], "reply": reply})[0]))
    for view in full["questions"]:
        for text in _answer_texts(view):
            assert not any(text in blob for blob in seen), (view["id"], text)


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


def test_multi_part_grades_hide_every_parts_answer(learner, author):
    data, _, _ = call(
        author, "make_questions", {"skill_id": "permutation-composition", "source": "exam", "seed": 3, "answers": True}
    )
    q = data["questions"][0]
    right = {p["label"]: p["answer"]["text"] for p in q["parts"]}
    for reply in (right, {label: "1" for label in right}):
        result, is_error, text = call(learner, "grade_answer", {"question_id": q["id"], "reply": reply})
        assert not is_error, text
        assert set(result["parts"]) == set(right)
        assert "canonical" not in set(_keys(result))
    shown, _, _ = call(author, "grade_answer", {"question_id": q["id"], "reply": right, "reveal": True})
    assert all(part["canonical"] for part in shown["parts"].values())


def test_an_object_reply_to_a_one_answer_question_is_refused(learner):
    data, _, _ = call(learner, "make_questions", {"skill_id": "simplify-surds", "seed": 5})
    _, is_error, text = call(learner, "grade_answer", {"question_id": data["questions"][0]["id"], "reply": {"a": "1"}})
    assert is_error and "one answer" in text


def test_figures_are_alt_text_unless_asked(learner):
    plain, _, _ = call(learner, "make_questions", {"skill_id": "count-to-5", "seed": 2})
    drawn, _, _ = call(learner, "make_questions", {"skill_id": "count-to-5", "seed": 2, "include_figures": True})
    assert plain["questions"][0]["figure"].keys() == {"alt"}
    assert drawn["questions"][0]["figure"]["svg"].lstrip().startswith("<svg")


# One skill per statistical figure kind: each is a read-the-picture question, so a learner gets nothing to read
# unless the picture itself comes back.
CHARTS = {
    "bar_chart": "f3-read-a-column-graph",
    "box_plot": "f10-compare-box-plots",
    "dot_plot": "f6-read-a-dot-plot",
    "histogram": "f7-read-a-histogram",
    "picture_graph": "f2-read-a-picture-graph",
    "pie_chart": "f6-read-a-pie-chart",
}


@pytest.mark.parametrize("kind, skill_id", sorted(CHARTS.items()))
def test_every_chart_kind_reaches_the_learner_as_a_picture(learner, author, kind, skill_id):
    args = {"skill_id": skill_id, "seed": 3, "include_figures": True}
    data, is_error, text = call(learner, "make_questions", args)
    assert not is_error, text
    figure = data["questions"][0]["figure"]
    assert figure.keys() == {"alt", "svg"} and figure["svg"].lstrip().startswith("<svg")
    keyed, _, _ = call(author, "make_questions", {**args, "answers": True})
    assert keyed["questions"][0]["figure"]["spec"]["kind"] == kind


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


def test_json_papers_are_capped_and_drop_drawings(learner):
    big = {"sections": [{"items": [{"skill": "simplify-surds", "count": mcp_server.MAX_COUNT}]}], "versions": 2}
    _, is_error, text = call(learner, "build_paper", {"spec": big})
    assert is_error and "html" in text
    figures = {"seed": 4, "sections": [{"items": [{"skill": "count-to-5", "count": 2}]}]}
    plain, _, _ = call(learner, "build_paper", {"spec": figures})
    drawn, _, _ = call(learner, "build_paper", {"spec": figures, "include_figures": True})
    first = plain["paper"]["versions"][0]["sections"][0]["questions"][0]["figure"]
    assert first.keys() == {"alt"}
    assert drawn["paper"]["versions"][0]["sections"][0]["questions"][0]["figure"]["svg"]


def test_a_paper_spec_is_checked_before_the_engine_runs(learner):
    for spec in ({"sections": []}, {"sections": [{"items": [{"skill": "simplify-surds", "colour": "red"}]}]}):
        _, is_error, _ = call(learner, "build_paper", {"spec": spec})
        assert is_error, spec


def test_an_unwritable_output_folder_is_a_tool_error(pool, tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    server = mcp_server.build_server("learner", pool=pool, out_dir=blocker / "exports")
    _, is_error, text = call(server, "build_paper", {"spec": PAPER, "format": "html"})
    assert is_error and "could not write" in text


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
        written = tmp_path / Path(path).name
        assert written.is_file() and str(written.resolve()) == path
        assert written.name.startswith("etc-passwd")
    assert (tmp_path / Path(paths[3]).name).read_bytes()[:2] == b"PK"


class FakePool:
    """A pool whose every call raises `error`."""

    def __init__(self, error):
        self.error = error

    def run(self, name, timeout, **kwargs):
        raise self.error

    def close(self):
        pass


@pytest.mark.parametrize(
    "error, learner_says, author_says",
    [
        (pool_module.Timeout("x"), "took longer than", "took longer than"),
        (pool_module.Unavailable("x"), "busy", "busy"),
        (pool_module.Changed("old/1", "new/1"), "no longer generated", "no longer generated"),
        (pool_module.WorkerError("not the part's answer '42'"), mcp_server.LEARNER_FAILURE, "'42'"),
    ],
)
def test_engine_failures_become_tool_errors(error, learner_says, author_says):
    for role, says in (("learner", learner_says), ("author", author_says)):
        server = mcp_server.build_server(role, pool=FakePool(error))
        _, is_error, text = call(server, "make_questions", {"skill_id": "simplify-surds"})
        assert is_error and says in text, (role, text)
        if role == "learner":
            assert "42" not in text


def test_the_server_speaks_mcp_over_stdio(tmp_path):
    """The real transport: a subprocess on stdin and stdout, workers and all. A worker writing to stdout would
    break this, which is why `pool._worker_main` sends its stdout to stderr."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mathgen.mcp_server", "--role", "learner", "--out-dir", str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )

    async def go():
        async with Client(params, read_timeout_seconds=120) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            made = await client.call_tool("make_questions", {"skill_id": "simplify-surds", "seed": 9})
            qid = made.structured_content["questions"][0]["id"]
            graded = await client.call_tool("grade_answer", {"question_id": qid, "reply": "1"})
            return names, graded.structured_content

    names, graded = asyncio.run(go())
    assert "grade_answer" in names and graded["correct"] is False and "canonical" not in graded


def test_roles_are_checked():
    with pytest.raises(ValueError):
        mcp_server.build_server("admin")


def test_command_line_parses_the_plugin_arguments(monkeypatch):
    seen = {}

    class Fake:
        _maths_engine = type("E", (), {"close": lambda self: None, "warm_up": lambda self: None})()

        def run(self, transport):
            seen["transport"] = transport

    def fake_build(role, **kwargs):
        seen.update(role=role, **kwargs)
        return Fake()

    monkeypatch.setattr(mcp_server, "build_server", fake_build)
    monkeypatch.setattr(mcp_server.signal, "signal", lambda *a: None)
    assert mcp_server.main(["--role", "author", "--out-dir", "/tmp/x", "--workers", "2"]) == 0
    assert seen["role"] == "author" and seen["out_dir"] == "/tmp/x" and seen["transport"] == "stdio"
    assert seen["workers"] == 2


@pytest.mark.parametrize("flags", [["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "-3"], ["--workers", "0"]])
def test_command_line_refuses_bad_limits(flags):
    with pytest.raises(SystemExit):
        mcp_server.main(flags)
