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


# --- figure route, headers, auth defaults, docs -------------------------------------------------------------


def test_figure_svg_serves_the_picture_and_nothing_else(client):
    (q,) = ask(client, "f10-measure-angle-protractor")
    r = client.get("/v1/figure.svg", params={"id": q["id"]}, headers=L)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert r.text.startswith("<svg")
    assert r.text == q["figure"]["svg"]
    assert "spec" not in q["figure"]
    (plain,) = ask(client, "simplify-surds")
    assert client.get("/v1/figure.svg", params={"id": plain["id"]}, headers=L).status_code == 404


def test_every_response_carries_the_security_headers(client):
    (q,) = ask(client, "f10-measure-angle-protractor")
    responses = [
        client.get("/v1/health"),
        client.get("/v1/skills"),  # a 401
        client.post("/v1/questions", json={"skill": "simplify-surds", "answers": True}, headers=A),
        client.get("/v1/figure.svg", params={"id": q["id"]}, headers=L),
        client.post(
            "/v1/papers",
            json={"sections": [{"items": [{"skill": "simplify-surds"}]}], "format": "html"},
            headers=L,
        ),
    ]
    for r in responses:
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["cache-control"] == "no-store"
    csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"
    assert responses[3].headers["content-security-policy"] == csp
    assert responses[4].headers["content-security-policy"] == csp
    assert "content-security-policy" not in responses[0].headers


def test_docs_are_off_unless_asked_for(client, pool):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    with TestClient(create_app(Config(keys=[(AUTHOR, "author")], docs=True), pool=pool)) as c:
        assert c.get("/openapi.json").status_code == 200


def test_no_keys_means_no_start_unless_open_is_chosen(pool):
    with pytest.raises(ValueError, match="MATHS_API_OPEN"):
        create_app(Config(), pool=pool)
    assert Config().open_role == "learner"
    with TestClient(create_app(Config(open=True), pool=pool)) as c:
        (q,) = c.post("/v1/questions", json={"skill": "simplify-surds"}).json()["questions"]
        assert "answer" not in q
        assert c.post("/v1/questions", json={"skill": "simplify-surds", "answers": True}).status_code == 403


def test_the_environment_sets_open_and_refuses_empty_key_lists(monkeypatch):
    for name in ("MATHS_API_KEYS", "MATHS_API_OPEN", "MATHS_API_OPEN_ROLE"):
        monkeypatch.delenv(name, raising=False)
    config = Config.from_env()
    assert config.open is False and config.open_role == "learner"
    with pytest.raises(ValueError, match="MATHS_API_OPEN"):
        config.check_auth()
    monkeypatch.setenv("MATHS_API_OPEN", "1")
    monkeypatch.setenv("MATHS_API_OPEN_ROLE", "author")
    config = Config.from_env()
    config.check_auth()
    assert config.open and config.open_role == "author"
    for bad in (",", " , ,", "   x"):
        with pytest.raises(ValueError):
            parse_keys(bad)
    assert parse_keys("   ") == []


def test_config_rejects_values_that_would_never_finish_or_never_start():
    for bad in (
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": -1},
        {"timeout": 0},
        {"paper_timeout": float("inf")},
        {"queue_timeout": -3},
        {"batch_timeout": float("nan")},
        {"workers": 0},
        {"workers": -2},
        {"open_role": "root"},
        {"id_secret": "short"},
        {"worker_memory_mb": -1},
    ):
        with pytest.raises(ValueError):
            Config(**bad)
    assert Config(timeout=2).batch_timeout == 20 and Config(workers=3).max_in_flight == 6


def test_requests_are_bounded_in_size_and_shape(client):
    q = ask(client, "simplify-surds")[0]
    assert client.post("/v1/questions", json={"skill": "x" * 201}, headers=L).status_code == 422
    assert (
        client.post("/v1/questions", json={"skill": "simplify-surds", "source": "x" * 101}, headers=L).status_code
        == 422
    )
    assert (
        client.post("/v1/questions", json={"skill": "simplify-surds", "source": 10**30}, headers=L).status_code == 422
    )
    assert client.post("/v1/questions", json={"skill": "simplify-surds", "source": 999}, headers=L).status_code == 404
    reply = {f"p{i}": "1" for i in range(21)}
    assert client.post("/v1/grade", json={"id": q["id"], "reply": reply}, headers=L).status_code == 422
    # Keys count towards the length as well as values.
    long_keys = {"k" * 1500: "1", "j" * 1500: "1"}
    assert client.post("/v1/grade", json={"id": q["id"], "reply": long_keys}, headers=L).status_code == 413


# --- load shedding, with a pool that behaves ---------------------------------------------------------------


class FakePool:
    """Stands in for the worker pool: each call sleeps `delay`, then returns or raises `outcome`."""

    size = 1
    live = 1
    version = {}

    def __init__(self, delay=0.0, outcome=None):
        self.delay, self.outcome, self.calls = delay, outcome, 0

    def run(self, name, timeout, *, wait=30.0, **kwargs):
        import time

        self.calls += 1
        time.sleep(self.delay)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return {"id": kwargs.get("qid"), "correct": True}


def fake_client(fake, **config):
    return TestClient(create_app(Config(keys=[(LEARNER, "learner")], **config), pool=fake))


ID = "simplify-surds/easy/1/abcdefabcdef"


def test_a_full_server_answers_503_with_retry_after():
    import threading

    fake = FakePool(delay=1.0)
    with fake_client(fake, max_in_flight=1) as c:
        slow = threading.Thread(
            target=lambda: c.post("/v1/grade", json={"id": ID, "reply": "1"}, headers=L), daemon=True
        )
        slow.start()
        import time

        time.sleep(0.4)
        r = c.post("/v1/grade", json={"id": ID, "reply": "1"}, headers=L)
        slow.join(5)
        assert r.status_code == 503 and r.headers["retry-after"]
        assert c.post("/v1/grade", json={"id": ID, "reply": "1"}, headers=L).status_code == 200  # a place is free again
        assert c.get("/v1/health").status_code == 200  # health never queues


def test_a_batch_has_one_total_deadline():
    fake = FakePool(delay=0.3)
    items = [{"id": ID, "reply": str(i)} for i in range(10)]
    with fake_client(fake, batch_timeout=1.0) as c:
        results = c.post("/v1/grade/batch", json={"items": items}, headers=L).json()["results"]
    graded = [r for r in results if "error" not in r]
    late = [r for r in results if r.get("error") == "timeout"]
    assert 1 <= len(graded) <= 4 and len(graded) + len(late) == 10
    assert fake.calls == len(graded)  # the late items were never run


def test_repeated_timeouts_on_one_skill_start_a_cool_down():
    from mathgen.server.pool import Timeout

    fake = FakePool(outcome=Timeout("grade took longer than 1 s"))
    other = "simplify-surds/hard/1/abcdefabcdef"
    with fake_client(fake, cooldown=30, cooldown_strikes=2) as c:
        post = lambda qid: c.post("/v1/grade", json={"id": qid, "reply": "1"}, headers=L)  # noqa: E731
        assert post(ID).status_code == 504
        assert post(ID).status_code == 504
        calls = fake.calls
        r = post(ID)
        assert r.status_code == 503 and int(r.headers["retry-after"]) > 0
        assert fake.calls == calls  # failed fast: no worker was used
        assert post(other).status_code == 504  # another source is unaffected
        batch = c.post("/v1/grade/batch", json={"items": [{"id": ID, "reply": "1"}]}, headers=L).json()
        assert batch["results"][0]["error"] == "unavailable" and fake.calls == calls + 1


# --- the id secret, through the API ------------------------------------------------------------------------


def test_with_an_id_secret_grading_works_but_the_public_engine_cannot_rebuild_the_question():
    from mathgen import engine

    secret = "s" * 24
    plain = service.Library.load()
    pool = WorkerPool(1, id_secret=secret)
    try:
        config = Config(keys=[(AUTHOR, "author"), (LEARNER, "learner")], id_secret=secret, timeout=3)
        with TestClient(create_app(config, pool=pool)) as c:
            (q,) = ask(c, "simplify-surds", seed=7, headers=A, answers=True)
            assert ask(c, "simplify-surds", seed=7, headers=A, answers=True) == [q]  # same seed, same question
            right = c.post("/v1/grade", json={"id": q["id"], "reply": q["answer"]["text"]}, headers=L).json()
            assert right["correct"] is True
            assert c.get("/v1/question", params={"id": q["id"]}, headers=L).json()["id"] == q["id"]
            # The seed field is a public nonce: the keyless engine rebuilds some other question from it, so
            # the hash does not match and it will not hand over this question's answer.
            assert q["id"].split("/")[2] == "7"
            with pytest.raises(engine.QuestionChanged) as caught:
                engine.regenerate(q["id"], plain.by_id)
            assert caught.value.question.text != q["text"] or caught.value.new_id != q["id"]
            # An id without the secret's stamp is refused, not marked.
            unkeyed = engine.make_with_id(plain.by_id["simplify-surds"], 7)[1]
            assert unkeyed != q["id"]
            assert c.post("/v1/grade", json={"id": unkeyed, "reply": "1"}, headers=L).status_code == 409
    finally:
        pool.close()


def test_papers_with_an_id_secret_use_the_derived_seed_as_the_nonce():
    secret = "p" * 20
    keyed = service.Library.load(id_secret=secret)
    plain = service.Library.load()
    spec = {"seed": 3, "sections": [{"items": [{"skill": "simplify-surds", "count": 2}]}]}
    a = service.build_paper(keyed, spec)
    assert a == service.build_paper(keyed, spec)
    b = service.build_paper(plain, spec)
    ids = lambda p: [q["id"] for q in p["versions"][0]["sections"][0]["questions"]]  # noqa: E731
    assert [i.split("/")[2] for i in ids(a)] == [i.split("/")[2] for i in ids(b)]  # same public nonces
    assert ids(a) != ids(b)  # but different questions behind them


# --- worker memory and default size ------------------------------------------------------------------------


def test_a_worker_memory_cap_turns_a_runaway_allocation_into_memory_error():
    import subprocess
    import sys

    pytest.importorskip("resource")  # POSIX only
    code = (
        "from mathgen.server.pool import _limit_memory\n"
        "_limit_memory(300)\n"
        "try:\n    bytearray(800 * 2**20)\nexcept MemoryError:\n    print('capped')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL
    )
    assert out.stdout.strip() == "capped", out.stderr


def test_workers_with_a_memory_cap_still_load_and_grade():
    pool = WorkerPool(1, memory_mb=Config().worker_memory_mb)
    try:
        with TestClient(create_app(Config(keys=[(LEARNER, "learner")]), pool=pool)) as c:
            (q,) = ask(c, "simplify-surds")
            assert c.post("/v1/grade", json={"id": q["id"], "reply": "1"}, headers=L).status_code == 200
    finally:
        pool.close()


def test_default_workers_follow_the_cpus_this_process_may_use(monkeypatch):
    import os

    from mathgen.server import pool as pool_module

    monkeypatch.setattr(pool_module, "_cgroup_cpus", lambda: None)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    assert pool_module.default_workers() == 3
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64, raising=False)
    assert pool_module.default_workers() == 8
    monkeypatch.setattr(pool_module, "_cgroup_cpus", lambda: 2)  # a container quota below the visible CPUs
    assert pool_module.default_workers() == 2
    monkeypatch.setattr(pool_module, "_cgroup_cpus", lambda: None)
    monkeypatch.setattr(os, "process_cpu_count", lambda: None, raising=False)
    assert 1 <= pool_module.default_workers() <= 8


def test_a_multi_part_grade_holds_no_part_answer_without_reveal(client):
    # A multi-part result keeps each part's own result, `canonical` included, keyed by label.
    (q,) = ask(client, "permutation-composition", headers=A, answers=True, source="exam")
    assert "parts" in q
    right = {p["label"]: p["answer"]["text"] for p in q["parts"]}
    wrong = {p["label"]: "0" for p in q["parts"]}

    def has_canonical(value):
        if isinstance(value, dict):
            return "canonical" in value or any(has_canonical(v) for v in value.values())
        return isinstance(value, list) and any(has_canonical(v) for v in value)

    for reply in (right, wrong):
        body = client.post("/v1/grade", json={"id": q["id"], "reply": reply}, headers=L).json()
        assert body["parts"] and not has_canonical(body)
        assert all(isinstance(p["marks"], int | str) for p in body["parts"].values())
    revealed = client.post("/v1/grade", json={"id": q["id"], "reply": wrong, "reveal": True}, headers=A).json()
    assert all(p["canonical"] for p in revealed["parts"].values())


def test_an_object_reply_to_a_one_answer_question_is_a_bad_request(client):
    (q,) = ask(client, "simplify-surds")
    assert client.post("/v1/grade", json={"id": q["id"], "reply": {"a": "1"}}, headers=L).status_code == 422
