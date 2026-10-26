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
    config = Config(keys=[(AUTHOR, "author")], client_strikes=6)  # 6 timeouts across any ids start the client's
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


def test_client_strikes_zero_turns_the_client_cool_down_off():
    _SlowPool.calls = 0
    config = Config(keys=[(AUTHOR, "author")], client_strikes=0)
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        for _ in range(4):
            assert _batch(client, 20).status_code == 200


def test_the_client_cool_down_is_off_by_default():
    assert Config().client_strikes == 0


def test_open_mode_names_clients_by_the_peer_address_and_ignores_the_forwarded_header():
    _SlowPool.calls = 0
    config = Config(open=True, cooldown_strikes=1, client_strikes=2)
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        spoof = {"X-Forwarded-For": "10.0.0.50"}
        items = [{"id": f"s{i}/b/1/h", "reply": "x"} for i in range(3)]
        client.post("/v1/grade/batch", json={"items": items}, headers=spoof)
        # a different claimed address does not make a different client: the peer is the same
        r = client.post("/v1/grade", json={"id": "z/y/1/h", "reply": "x"}, headers={"X-Forwarded-For": "10.0.0.51"})
    assert r.status_code == 503


def test_two_keys_are_two_clients():
    other = "b" * 24
    _SlowPool.calls = 0
    config = Config(keys=[(AUTHOR, "author"), (other, "learner")], client_strikes=2)
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        _batch(client, 3)
        r = client.post("/v1/grade", json={"id": "z/y/1/h", "reply": "x"}, headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 504  # not 503: the second key has its own count


def test_replies_just_under_half_the_deadline_still_spend_the_budget():
    class _Brisk(_SlowPool):
        def run(self, name, timeout, **kwargs):
            type(self).calls += 1
            time.sleep(0.45)
            return {"correct": True}

    _Brisk.calls = 0
    config = Config(keys=[(AUTHOR, "author")], timeout=1.0, batch_budget=3.0)
    with TestClient(create_app(config, pool=_Brisk())) as client:
        results = _batch(client, 20).json()["results"]
    assert _Brisk.calls <= 6 and len(results) == 20


def test_a_reply_that_nearly_times_out_spends_budget():
    class _Slow(_SlowPool):
        def run(self, name, timeout, **kwargs):
            type(self).calls += 1
            time.sleep(timeout * 0.8)
            return {"correct": True}

    _Slow.calls = 0
    config = Config(keys=[(AUTHOR, "author")], timeout=0.5, batch_budget=1.0)
    with TestClient(create_app(config, pool=_Slow())) as client:
        results = _batch(client, 6).json()["results"]
    assert _Slow.calls == 2 and sum("error" in r for r in results) == 4


def test_a_budget_smaller_than_a_deadline_is_not_spent_in_slivers():
    _SlowPool.calls = 0
    config = Config(keys=[(AUTHOR, "author")], timeout=5.0, batch_budget=12.0)
    with TestClient(create_app(config, pool=_SlowPool())) as client:
        _batch(client, 6)
    assert _SlowPool.calls == 2  # 10 s spent, 2 s left is under a deadline, so the rest are not run


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
        single = client.post(
            "/v1/grade", json={"id": "a/b/1/h", "reply": "x"}, headers={"Authorization": f"Bearer {AUTHOR}"}
        )
        gate.set()
        t.join()
    assert second.status_code == 503 and out["first"].status_code == 200
    assert single.status_code == 200  # a single grade is not held up by the batch


def test_a_reply_refused_after_running_still_spends_the_budget():
    from mathgen.server.pool import Changed

    class _Changed(_SlowPool):
        def run(self, name, timeout, **kwargs):
            type(self).calls += 1
            time.sleep(0.4)
            raise Changed("changed", "a/b/1/new")

    _Changed.calls = 0
    config = Config(keys=[(AUTHOR, "author")], timeout=1.0, batch_budget=2.0)
    with TestClient(create_app(config, pool=_Changed())) as client:
        results = _batch(client, 20).json()["results"]
    assert _Changed.calls <= 5 and len(results) == 20
