"""The HTTP API (`mathgen.server`): keys and roles, no answers without an author key, grading by id, deadlines."""

import pytest
from fastapi.testclient import TestClient

from mathgen import service
from mathgen.server import Config, create_app
from mathgen.server.config import parse_keys
from mathgen.server.pool import WorkerPool

AUTHOR = "a" * 24
LEARNER = "l" * 24
A = {"Authorization": f"Bearer {AUTHOR}"}
L = {"X-API-Key": LEARNER}
# A reply SymPy never finishes simplifying (mathgen/grading/safe_parse.py); the deadline must stop it.
HANGS = "sqrt(acos(x)^2)"


@pytest.fixture(scope="module")
def lib():
    return service.Library.load()


@pytest.fixture(scope="module")
def pool():
    pool = WorkerPool(1)  # one is enough: the timeout test also proves a killed worker is replaced
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def client(pool):
    config = Config(keys=[(AUTHOR, "author"), (LEARNER, "learner")], timeout=3, max_body_bytes=20_000)
    with TestClient(create_app(config, pool=pool)) as c:
        yield c


def first(lib, predicate):
    return next(s for s in lib.skills if s.status == "active" and s.sources and predicate(s))


def ask(client, skill, seed=1, headers=L, **extra):
    r = client.post("/v1/questions", json={"skill": skill, "seed": seed, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["questions"]


def test_health_needs_no_key_and_names_the_library(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok" and body["skills"] > 700 and body["library"]


def test_every_other_route_needs_a_key(client):
    assert client.get("/v1/skills").status_code == 401
    assert client.get("/v1/skills", headers={"Authorization": "Bearer wrong-key-wrong-key"}).status_code == 401
    assert client.post("/v1/questions", json={"skill": "simplify-surds"}).status_code == 401


def test_skills_filter_by_acara_code_prefix(client):
    body = client.get("/v1/skills", params={"code": "AC9M8"}, headers=L).json()
    assert body["count"] == len(body["skills"]) > 0
    assert all(any(c.startswith("AC9M8") for c in s["curriculum_codes"]) for s in body["skills"])
    assert all("8" in s["years"] for s in body["skills"])


def test_unknown_skill_is_404(client):
    assert client.get("/v1/skills/no-such-skill", headers=L).status_code == 404
    assert client.post("/v1/questions", json={"skill": "no-such-skill"}, headers=L).status_code == 404


def test_the_same_seed_gives_the_same_questions(client):
    assert ask(client, "simplify-surds", seed=7, count=3) == ask(client, "simplify-surds", seed=7, count=3)
    assert ask(client, "simplify-surds", seed=7) != ask(client, "simplify-surds", seed=8)


def test_a_learner_key_cannot_ask_for_answers(client):
    assert client.post("/v1/questions", json={"skill": "simplify-surds", "answers": True}, headers=L).status_code == 403
    (q,) = ask(client, "simplify-surds")
    assert client.get("/v1/question", params={"id": q["id"], "answers": True}, headers=L).status_code == 403
    assert client.post("/v1/grade", json={"id": q["id"], "reply": "1", "reveal": True}, headers=L).status_code == 403
    paper = {"sections": [{"items": [{"skill": "simplify-surds"}]}]}
    for fmt in ("html-key", "moodle", "qti"):
        assert client.post("/v1/papers", json={**paper, "format": fmt}, headers=L).status_code == 403


def test_a_learner_view_carries_no_answer(client, lib):
    for skill in (first(lib, lambda s: s.answer == "number"), first(lib, lambda s: s.answer == "mcq")):
        (shown,) = ask(client, skill.id)
        (keyed,) = ask(client, skill.id, headers=A, answers=True)
        assert "answer" not in shown and "answer" in keyed
        assert {k: v for k, v in keyed.items() if k != "answer"} == shown


def test_grading_rebuilds_the_question_from_its_id(client):
    (q,) = ask(client, "simplify-surds", seed=7, headers=A, answers=True)
    right = client.post("/v1/grade", json={"id": q["id"], "reply": q["answer"]["text"]}, headers=L).json()
    wrong = client.post("/v1/grade", json={"id": q["id"], "reply": "2√3"}, headers=L).json()
    assert right["correct"] is True and wrong["correct"] is False
    assert "canonical" not in wrong
    revealed = client.post("/v1/grade", json={"id": q["id"], "reply": "2√3", "reveal": True}, headers=A).json()
    assert revealed["canonical"]


def test_multiple_choice_is_graded_by_option_index(client, lib):
    skill = first(lib, lambda s: s.answer == "mcq")
    (q,) = ask(client, skill.id, headers=A, answers=True)
    right = q["answer"]["option"]
    assert q["options"][right] == q["answer"]["text"]
    results = client.post(
        "/v1/grade/batch",
        json={"items": [{"id": q["id"], "reply": i} for i in range(len(q["options"]))]},
        headers=L,
    ).json()["results"]
    assert [r["correct"] for r in results] == [i == right for i in range(len(q["options"]))]


def test_a_multi_part_question_is_graded_part_by_part(client, lib):
    for skill in lib.skills:
        if skill.status != "active" or skill.answer == "rubric":
            continue
        exam = [s for s in skill.sources if s.difficulty == "exam"]
        if not exam:
            continue
        (q,) = ask(client, skill.id, headers=A, answers=True, source=exam[0].effective_key)
        if "parts" in q and all(p["answer_kind"] == "number" for p in q["parts"]):
            break
    else:
        pytest.skip("no all-number multi-part question in the library")
    reply = {p["label"]: p["answer"]["text"] for p in q["parts"]}
    body = client.post("/v1/grade", json={"id": q["id"], "reply": reply}, headers=L).json()
    assert body["correct"] is True and body["marks"] == body["max_marks"] == q["marks"]


def test_an_id_whose_content_changed_is_refused_not_marked(client):
    (q,) = ask(client, "simplify-surds")
    stale = q["id"].rsplit("/", 1)[0] + "/000000000000"
    r = client.post("/v1/grade", json={"id": stale, "reply": "1"}, headers=L)
    assert r.status_code == 409 and r.json()["detail"]["new_id"] == q["id"]
    assert client.post("/v1/grade", json={"id": "not-an-id", "reply": "1"}, headers=L).status_code == 422


def test_a_reply_that_never_finishes_times_out_and_the_server_carries_on(client, lib):
    skill = first(lib, lambda s: s.answer == "expression")
    (q,) = ask(client, skill.id)
    assert client.post("/v1/grade", json={"id": q["id"], "reply": HANGS}, headers=L).status_code == 504
    assert client.post("/v1/grade", json={"id": q["id"], "reply": "x"}, headers=L).status_code == 200


def test_oversized_bodies_and_replies_are_refused(client):
    (q,) = ask(client, "simplify-surds")
    assert client.post("/v1/grade", json={"id": q["id"], "reply": "1" * 2001}, headers=L).status_code == 413
    assert client.post("/v1/grade", json={"id": q["id"], "reply": "1" * 30_000}, headers=L).status_code == 413


def test_papers_are_reproducible_and_versions_differ(client):
    spec = {"seed": 5, "versions": 2, "sections": [{"items": [{"skill": "simplify-surds", "count": 3}]}]}
    one = client.post("/v1/papers", json=spec, headers=L).json()
    assert one == client.post("/v1/papers", json=spec, headers=L).json()
    a, b = one["versions"]
    ids = lambda v: [q["id"] for s in v["sections"] for q in s["questions"]]  # noqa: E731
    assert len(ids(a)) == 3 and ids(a) != ids(b)
    assert all("answer" not in q for v in one["versions"] for s in v["sections"] for q in s["questions"])


def test_keys_are_role_prefixed_and_long():
    assert parse_keys(f"author:{AUTHOR}, learner:{LEARNER}") == [(AUTHOR, "author"), (LEARNER, "learner")]
    for bad in ("short", "admin:" + "x" * 20, "author:tooshort"):
        with pytest.raises(ValueError):
            parse_keys(bad)


def test_a_learner_gets_a_figure_as_a_picture_not_its_data(client):
    # The spec of a read-the-picture question holds the answer (here the angle); the SVG draws only the rays.
    (q,) = ask(client, "f10-measure-angle-protractor")
    assert set(q["figure"]) == {"alt", "svg"} and q["figure"]["svg"].startswith("<svg")
    (keyed,) = ask(client, "f10-measure-angle-protractor", headers=A, answers=True)
    assert keyed["figure"]["spec"]["kind"] == "protractor"


def test_a_chunked_body_is_counted_as_it_arrives(client):
    def chunks():
        yield b'{"id": "x", "reply": "'
        for _ in range(30):
            yield b"1" * 1000
        yield b'"}'

    r = client.post("/v1/grade", content=chunks(), headers={**L, "content-type": "application/json"})
    assert r.status_code == 413


def test_a_worker_that_died_while_idle_is_replaced_without_failing_a_call(client, pool):
    (q,) = ask(client, "simplify-surds")
    for worker in list(pool._all):
        worker.process.kill()
        worker.process.join(5)
    assert client.post("/v1/grade", json={"id": q["id"], "reply": "1"}, headers=L).status_code == 200
    assert client.get("/v1/health").json()["workers"] == pool.size
