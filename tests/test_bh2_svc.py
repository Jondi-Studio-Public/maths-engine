"""Regression tests for the service, scheduling and student-text fixes of the 2026-10-19 bug hunt (lane svc)."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from mathgen import engine, service
from mathgen.core import mcq_options, paper_notation
from mathgen.scheduling import calibration, fsrs, review
from mathgen.server.config import Config

LIB = service.Library.load()


def _skill(sid):
    return LIB.skill(sid)


def _questions(sid, sources, seeds):
    skill = _skill(sid)
    for source in sources:
        for seed in seeds:
            yield engine.make_for_skill(skill, seed=seed, source=source)


def _texts(q):
    out = [q.text, q.answer_text, *q.distractors, *(q.working or [])]
    for part in q.parts or []:
        out += [part.text, part.answer_text, *part.working, *part.distractors]
    return [str(t) for t in out]


# -- D9: a note must not hand a learner the key ---------------------------------------------------------------------


@pytest.mark.parametrize("qid", ["ey-quarter-half-turns/hard/41", "uc2-quadric-surfaces-level-sets/easy/21"])
def test_learner_note_never_names_the_key(qid):
    skill_id, source, seed = qid.split("/")
    skill = _skill(skill_id)
    question = engine.make_for_skill(skill, seed=int(seed), source=source)
    full_id = engine.question_id(skill, source, int(seed), question)
    for option in range(len(mcq_options(question))):
        learner = service.grade_reply(LIB, full_id, option)
        author = service.grade_reply(LIB, full_id, option, reveal=True)
        if learner["note"]:
            assert author["canonical"].casefold() not in learner["note"].casefold()
        assert "canonical" not in learner


def test_note_without_the_key_is_kept():
    result = service.result_view(_fake_result("Turns the wrong way.", "a half turn"), reveal=False)
    assert result["note"] == "Turns the wrong way."
    revealed = service.result_view(_fake_result("Turns only a half turn.", "a half turn"), reveal=False)
    assert revealed["note"] is None


def _fake_result(note, canonical):
    from mathgen.grading.typed import GradeResult

    return GradeResult(correct=False, canonical=canonical, note=note, note_kind="misconception")


# -- D10-D12: messages ----------------------------------------------------------------------------------------------


def test_cooldown_message_has_no_none():
    pytest.importorskip("fastapi")
    from mathgen.server import app as app_module

    source = open(app_module.__file__, encoding="utf-8").read()
    assert 'key=f"{body.skill}/{body.source}"' not in source


def test_mcp_bad_skills_path_is_refused_at_start():
    pytest.importorskip("mcp")
    result = subprocess.run(
        [sys.executable, "-m", "mathgen.mcp_server", "--skills", "/nonexistent/skills"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode == 2
    assert "/nonexistent/skills" in result.stderr


@pytest.mark.parametrize("flag", ["--workers", "--timeout"])
def test_mcp_flag_errors_name_the_flag_not_a_function(flag):
    result = subprocess.run(
        [sys.executable, "-m", "mathgen.mcp_server", flag, "x"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode == 2
    assert f"argument {flag}" in result.stderr
    assert "_workers" not in result.stderr and "_positive_seconds" not in result.stderr


def test_server_env_errors_name_the_variable(monkeypatch):
    monkeypatch.setenv("MATHS_API_WORKERS", "abc")
    with pytest.raises(ValueError, match="MATHS_API_WORKERS"):
        Config.from_env()
    monkeypatch.delenv("MATHS_API_WORKERS")
    monkeypatch.setenv("MATHS_API_TIMEOUT", "soon")
    with pytest.raises(ValueError, match="MATHS_API_TIMEOUT"):
        Config.from_env()


# -- D14: edge cases ------------------------------------------------------------------------------------------------


def test_build_paper_versions_message_and_source_index():
    base = {"sections": [{"items": [{"skill": "simplify-surds", "count": 1}]}]}
    with pytest.raises(service.BadRequest, match="versions must be"):
        service.build_paper(LIB, {**base, "versions": 11})
    for bad in (-1, True):
        spec = {"sections": [{"items": [{"skill": "simplify-surds", "source": bad, "count": 1}]}]}
        with pytest.raises(service.BadRequest):
            service.build_paper(LIB, spec)


def test_unknown_part_key_is_refused():
    skill = _skill("misleading-graph-figure")
    question = engine.make_for_skill(skill, seed=1, source="easy")
    qid = engine.question_id(skill, "easy", 1, question)
    with pytest.raises(service.BadRequest, match="no such part"):
        service.grade_reply(LIB, qid, {"a": "1", "zz": 1})


NOW = datetime(2026, 1, 10)


def test_review_score_edge_cases():
    with pytest.raises(ValueError, match="outcome"):
        review.score("bogus", None, None, NOW)
    with pytest.raises(ValueError, match="maximum_interval_days"):
        review.score("advance", None, None, NOW, maximum_interval_days=0)
    memory, _ = fsrs.step(None, fsrs.GOOD, 0)
    skewed = review.score("advance", memory, NOW + timedelta(days=3), NOW)
    assert skewed["elapsed_days"] == 0.0


def test_fsrs_rejects_a_bool_rating():
    with pytest.raises(ValueError):
        fsrs.step(None, True, 0)


def test_calibration_edge_cases():
    rows = [{"retrievability": 0.8, "rating": 3}]
    with pytest.raises(ValueError, match="bins"):
        calibration.fsrs_calibration(rows, bins=0)
    with pytest.raises(ValueError, match="retrievability"):
        calibration.fsrs_calibration([{"retrievability": 1.5, "rating": 3}])


@pytest.mark.parametrize(
    "text, expected",
    [("1/2*x", "1/2 × x"), ("x/y*z", "x/y × z"), ("a/(b+1)*c", "a/(b+1) × c"), ("2*x", "2x"), ("x*y/2", "xy/2")],
)
def test_paper_notation_keeps_a_product_after_a_division_visible(text, expected):
    assert paper_notation(text) == expected
    assert paper_notation(paper_notation(text)) == expected


def test_moodle_category_has_no_slash_from_the_title():
    from mathgen.export import paper_moodle_xml

    paper = service.build_paper(
        LIB, {"title": "Year 7/8 Algebra", "seed": 1, "sections": [{"items": [{"skill": "simplify-surds"}]}]}
    )
    xml = paper_moodle_xml(paper)
    category = re.search(r"<category><text>(.*?)</text>", xml).group(1)
    assert "Year 7/8" not in category and category.startswith("$course$/top/Year 7-8 Algebra")


# -- D4/D5/D8: student text -----------------------------------------------------------------------------------------

BAD = re.compile(r"\b1x|\(1\)x|\+ \(0\)|- -|\+-|atan2|\d{10,}|`|\d\.\d{9,}")


@pytest.mark.parametrize(
    "sid",
    [
        "uc2-secant-iteration",
        "lagrange-constrained-point",
        "senior-review-cartesian-form",
        "senior-simulate-repeated-random-sampling-from-a-variety-of-acmsm139",
        "senior-use-technology-to-investigate-the-effect-of-the",
        "senior-convert-between-cartesian-and-polar-form",
        "senior-understand-motion-of-a-body-under-concurrent-forces",
        "senior-apply-approximation-strategies-for-calculations",
        "senior-calculate-the-area-between-curves-in-simple-cases",
        "uc2-group-table-check",
    ],
)
def test_working_and_text_are_plain(sid):
    skill = _skill(sid)
    sources = [s.effective_key for s in skill.sources]
    for q in _questions(sid, sources, range(1, 41)):
        for text in _texts(q):
            assert not BAD.search(text), (sid, text)


def test_area_between_curves_working_is_plain_and_right():
    skill = _skill("senior-calculate-the-area-between-curves-in-simple-cases")
    q = engine.make_for_skill(skill, seed=1239521655, source="easy")
    assert "An antiderivative of x is x²/2" in " ".join(q.working)
    assert q.working[-1].strip().endswith("= 10.5") and q.answer_text == "10.5"


def test_misleading_graph_honest_axis_note_does_not_claim_a_flaw():
    skill = _skill("misleading-graph-figure")
    for seed in range(1, 40):
        q = engine.make_for_skill(skill, seed=seed, source="easy")
        for part in q.parts:
            if part.answer_text.startswith("Nothing is misleading"):
                for d in part.distractors:
                    assert getattr(d, "tag", None) != "ir2a-blames-the-horizontal-axis"
