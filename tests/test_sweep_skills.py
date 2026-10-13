"""scripts/sweep_skills.py: a failure of any kind is collected, never fatal to the sweep, and fails the run."""

import time
from types import SimpleNamespace

import pytest

from mathgen import engine
from scripts import sweep_skills as sweep

SKILL_ID = "simplify-surds"


@pytest.fixture(scope="module")
def skill():
    return next(s for s in engine.load_skills()[1] if s.id == SKILL_ID)


def test_a_clean_question_has_no_problems(skill):
    question = engine.make_for_skill(skill, 0, source=skill.sources[0])
    assert sweep.check_question(skill, question) == []


def test_a_distractor_that_grades_correct_is_reported(skill, monkeypatch):
    question = engine.make_for_skill(skill, 0, source=skill.sources[0])
    accept_all = SimpleNamespace(correct=True)
    monkeypatch.setattr(sweep.engine, "grade", lambda *_args: accept_all)
    problems = sweep.check_question(skill, question)
    assert len(problems) == len(question.distractors) >= 1
    assert all(p.startswith("distractor") for p in problems)


def test_an_answer_that_does_not_grade_correct_is_reported(skill, monkeypatch):
    question = engine.make_for_skill(skill, 0, source=skill.sources[0])
    monkeypatch.setattr(sweep.engine, "grade", lambda *_args: SimpleNamespace(correct=False))
    assert [p.split(" ")[0] for p in sweep.check_question(skill, question)] == ["answer_text"]


def test_text_defects_are_reported_only_when_asked_for(skill, monkeypatch):
    question = engine.make_for_skill(skill, 0, source=skill.sources[0])
    monkeypatch.setattr(sweep, "find_text_defects", lambda _q: ["star_operator in text: '7*x'"])
    assert sweep.check_question(skill, question) == []
    assert sweep.check_question(skill, question, text_quality=True) == ["text: star_operator in text: '7*x'"]


def test_an_exception_does_not_stop_the_remaining_seeds(skill, monkeypatch):
    def broken(_skill, seed, source=None):
        raise ValueError(f"boom {seed}")

    monkeypatch.setattr(sweep.engine, "make_for_skill", broken)
    checked, failures = sweep.sweep_source(skill, 0, seeds=3, timeout_s=5)
    assert checked == 3
    assert [(f.skill, f.seed) for f in failures] == [(SKILL_ID, 0), (SKILL_ID, 1), (SKILL_ID, 2)]
    assert "ValueError: boom 1" in failures[1].reason


def test_a_slow_question_times_out_and_the_next_seed_still_runs(skill, monkeypatch):
    real = sweep.engine.make_for_skill

    def slow_first(sk, seed, source=None):
        if seed == 0:
            while True:  # an `except Exception` in a generator must not be able to swallow the alarm
                try:
                    time.sleep(0.05)
                except Exception:  # noqa: BLE001
                    pass
        return real(sk, seed, source=source)

    monkeypatch.setattr(sweep.engine, "make_for_skill", slow_first)
    checked, failures = sweep.sweep_source(skill, 0, seeds=2, timeout_s=0.2)
    assert checked == 2
    assert [(f.seed, f.reason.split(":")[0]) for f in failures] == [(0, "timeout")]


def test_shards_partition_the_skills_without_overlap():
    ids = [s.id for s in engine.load_skills()[1]][:300]
    owner = [{i for i in ids if sweep.in_shard(i, n, 4)} for n in range(4)]
    assert set().union(*owner) == set(ids) and sum(len(o) for o in owner) == len(ids)


def test_the_run_lists_failures_and_exits_non_zero_and_writes_the_step_summary(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(sweep.engine, "grade", lambda *_args: SimpleNamespace(correct=False))
    assert sweep.main(["--skills", SKILL_ID, "--seeds", "1", "--workers", "1"]) == 1
    out = capsys.readouterr().out
    assert "FAILURES:" in out and SKILL_ID in out and "answer_text" in out
    text = summary.read_text(encoding="utf-8")
    assert "| skill | source | seed | reason |" in text and SKILL_ID in text


def test_a_clean_run_exits_zero_and_an_unknown_skill_is_a_usage_error(capsys):
    assert sweep.main(["--skills", SKILL_ID, "--seeds", "2", "--workers", "1"]) == 0
    assert "failures: []" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stopped:
        sweep.main(["--skills", "no-such-skill"])
    assert stopped.value.code == 2
