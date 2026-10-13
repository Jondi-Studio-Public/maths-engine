"""scripts/check_generator_speed.py times the combinations skills really use, and reports every failure."""

import time

import pytest

from mathgen import engine
from scripts import check_generator_speed as speed


def test_it_times_the_bare_default_of_every_topic_and_each_distinct_source_call():
    combos = speed.combinations()
    assert len(combos) == len(set(combos))
    bare = {topic for topic, difficulty, params in combos if difficulty is None and not params}
    assert {topic for _level, topic in speed.topics()} <= bare
    used = {
        (src.topic, src.difficulty, tuple(sorted(engine._tuplify(src.params).items())))
        for skill in engine.load_skills()[1]
        if skill.status == "active"
        for src in skill.sources
    }
    assert used <= set(combos)
    assert any(difficulty for _t, difficulty, _p in combos) and any(params for _t, _d, params in combos)


def test_describe_names_the_difficulty_and_params():
    assert speed.describe(("t", None, ())) == "t"
    assert speed.describe(("t", "easy", (("kinds", ("a", "b")),))) == "t [easy, kinds=('a', 'b')]"


def test_an_exception_is_a_problem_not_a_missing_sample(monkeypatch):
    def boom(topic, seed=None, difficulty=None, **params):
        raise ValueError(f"cannot make {topic} {params}")

    monkeypatch.setattr(speed, "make", boom)
    result = speed.time_combo(("t", "hard", (("n", 3),)), seeds=3)
    assert result["ms"] is None and len(result["problems"]) == 4  # the warm-up and 3 seeds
    assert "ValueError: cannot make t {'n': 3}" in result["problems"][1]


def test_a_call_that_never_returns_is_a_timeout_and_skips_the_rest(monkeypatch):
    def hang(topic, seed=None, difficulty=None, **params):
        while True:
            try:
                time.sleep(0.05)
            except Exception:  # noqa: BLE001 - the alarm must get through a generator's own except
                pass

    monkeypatch.setattr(speed, "make", hang)
    result = speed.time_combo(("t", None, ()), seeds=8, timeout_s=0.2)
    assert result["problems"] == ["warm-up: no answer after 0.2 s"]


def test_main_fails_naming_the_topic_and_params_and_passes_with_errors_empty(monkeypatch, capsys):
    monkeypatch.setattr(speed, "combinations", lambda: [("fast", None, ()), ("broken", "easy", (("k", 1),))])

    def fake_make(topic, seed=None, difficulty=None, **params):
        if topic == "broken":
            raise RuntimeError("nope")

    monkeypatch.setattr(speed, "make", fake_make)
    monkeypatch.setattr("sys.argv", ["check_generator_speed.py"])
    assert speed.main() == 1
    out = capsys.readouterr().out
    assert "errors: 1" in out and "broken [easy, k=1]" in out and "RuntimeError: nope" in out

    monkeypatch.setattr(speed, "combinations", lambda: [("fast", None, ())])
    assert speed.main() == 0
    assert "errors: []" in capsys.readouterr().out


def test_a_median_over_the_ceiling_still_fails(monkeypatch, capsys):
    monkeypatch.setattr(speed, "combinations", lambda: [("slow", None, ())])
    monkeypatch.setattr(speed, "make", lambda *a, **k: time.sleep(0.03))
    monkeypatch.setattr("sys.argv", ["check_generator_speed.py", "--seeds", "2", "--ceiling-ms", "10"])
    assert speed.main() == 1
    assert "OVER CEILING" in capsys.readouterr().out


@pytest.mark.parametrize("workers", ["0", "x"])
def test_workers_must_be_a_positive_integer(monkeypatch, workers):
    monkeypatch.setattr("sys.argv", ["check_generator_speed.py", "--workers", workers])
    with pytest.raises(SystemExit) as stopped:
        speed.main()
    assert stopped.value.code == 2


def test_one_untimed_warm_up_call_comes_before_the_timed_calls_and_is_not_in_the_median(monkeypatch):
    seen = []
    clock = iter([0.0, 0.0, 0.002, 0.0, 0.004])
    monkeypatch.setattr(speed.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(speed, "make", lambda topic, seed=None, difficulty=None, **p: seen.append(seed))
    result = speed.time_combo(("t", None, ()), seeds=2)
    assert seen == [2, 0, 1]  # the warm-up seed lies outside the timed ones
    assert result["ms"] == pytest.approx(3.0)  # median of 2 ms and 4 ms; the warm-up is not in it


def test_a_warm_up_failure_is_reported_as_one(monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("cold")

    monkeypatch.setattr(speed, "make", boom)
    assert speed.time_combo(("t", None, ()), seeds=1)["problems"][0].startswith("warm-up: ValueError: cold")
