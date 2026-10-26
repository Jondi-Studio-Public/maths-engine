"""Round 4 review fixes: a linear 'over' rewrite, case-blind bar labels and a bounded slow-reply batch."""

import time
from fractions import Fraction

from fastapi.testclient import TestClient

from mathgen.core import Question
from mathgen.grading import grade
from mathgen.grading import interactive as ia
from mathgen.grading import stem_checks
from mathgen.server import Config, create_app
from mathgen.server.pool import Timeout

AUTHOR = "a" * 24


def test_over_rewrite_is_linear_and_unchanged_on_normal_input():
    class Q:
        answer_text = "e^(2π)"
        text = ""

    start = time.monotonic()
    out = stem_checks.rewrite_reply("expression", Q(), "x" * 30000)
    assert out == "x" * 30000
    assert time.monotonic() - start < 2.0
    assert stem_checks.rewrite_reply("expression", Q(), "pi over 2 and 3 over 4") == "pi/2 and 3/4"


def _bars(reply):
    answer = ia.bars_answer(["Hockey", "Netball"], [Fraction(5, 2), 4], top=10, snap=Fraction(1, 2))
    q = Question(
        topic="t", level=0, seed=1, text="", latex="", answer=answer, answer_text="x", answer_latex="", working=[],
        difficulty="easy", distractors=[],
    )  # fmt: skip
    return grade("bars", q, reply).correct


def test_bar_category_names_ignore_case_and_outer_space():
    assert _bars("Hockey = 2.5, Netball = 4")
    assert _bars("hockey = 2.5, NETBALL = 4")
    assert _bars("  hockey =2.5 ;  netball: 4 ")
    assert not _bars("hockey = 4, netball = 2.5")
    assert not _bars("hockey = 2.5, soccer = 4")


class _SlowPool:
    size = 2
    live = 2
    calls = 0

    def run(self, name, timeout, **kwargs):
        type(self).calls += 1
        raise Timeout("too slow")

    def close(self):
        pass


def test_a_batch_of_slow_replies_stops_after_a_few_timeouts():
    _SlowPool.calls = 0
    items = [{"id": f"skill{i}/src/1/h", "reply": "x"} for i in range(20)]
    with TestClient(create_app(Config(keys=[(AUTHOR, "author")]), pool=_SlowPool())) as client:
        r = client.post("/v1/grade/batch", json={"items": items}, headers={"Authorization": f"Bearer {AUTHOR}"})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 20 and all(x["error"] == "timeout" for x in results)
    assert _SlowPool.calls == 3  # the default budget is three timeouts' worth of worker time
    assert "too many slow" in results[-1]["detail"]


def _batch(client, n, ids=8):
    items = [{"id": f"skill{i % ids}/src/1/h", "reply": "x"} for i in range(n)]
    return client.post("/v1/grade/batch", json={"items": items}, headers={"Authorization": f"Bearer {AUTHOR}"})


def test_the_batch_budget_is_worker_seconds_not_a_count():
    _SlowPool.calls = 0
    config = Config(keys=[(AUTHOR, "author")], timeout=2.0, batch_budget=4.0)
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        results = _batch(client, 10).json()["results"]
    assert _SlowPool.calls == 2 and len(results) == 10


def test_one_client_cycling_ids_is_cooled_down_across_batches():
    _SlowPool.calls = 0
    config = Config(keys=[(AUTHOR, "author")], cooldown_strikes=3)  # 6 timeouts across any ids start the client's
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        assert _batch(client, 20).status_code == 200  # 3 timeouts
        assert _batch(client, 20).status_code == 200  # 6 timeouts
        calls = _SlowPool.calls
        r = _batch(client, 20)
        r2 = client.post(
            "/v1/grade", json={"id": "other/src/1/h", "reply": "x"}, headers={"Authorization": f"Bearer {AUTHOR}"}
        )
    assert r.status_code == 503 and int(r.headers["retry-after"]) > 0
    assert r2.status_code == 503 and _SlowPool.calls == calls  # no worker was used


def test_batches_leave_half_the_workers_for_single_grades():
    import threading

    gate = threading.Event()
    entered = threading.Event()

    class _Blocking(_SlowPool):
        size = 2

        def run(self, name, timeout, **kwargs):
            entered.set()
            gate.wait(5)
            return {"correct": True}

    with TestClient(create_app(Config(keys=[(AUTHOR, "author")], workers=2), pool=_Blocking())) as client:
        out = {}
        t = threading.Thread(target=lambda: out.update(first=_batch(client, 1)))
        t.start()
        entered.wait(5)
        second = _batch(client, 1)  # a second batch while the first holds the only batch place
        gate.set()
        t.join()
    assert second.status_code == 503 and out["first"].status_code == 200
