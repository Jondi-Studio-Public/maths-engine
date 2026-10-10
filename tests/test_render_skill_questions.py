"""Tests for scripts/render_skill_questions.py and mathgen/text_quality.py."""

from pathlib import Path

import pytest

from mathgen.skills import load_skills
from mathgen.core import Question
from mathgen.text_quality import find_text_defects
from scripts.render_skill_questions import main, render_skill, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def _question(text: str, answer_text: str = "1", distractors: list[str] | None = None) -> Question:
    return Question(
        topic="t",
        level=1,
        seed=0,
        text=text,
        latex=text,
        answer=1,
        answer_text=answer_text,
        distractors=distractors or [],
    )


@pytest.mark.parametrize(
    "text",
    [
        "Simplify: 5**3 × 5**3",
        "Estimate 192 * 544 by rounding each number.",
        "Work out 2*x + 3.",
        "10 equal parts make a whole. 8 parts are used, then 1 more parts are used.",
        "Sam has 3 apples. He gives away 1 apples.",
        "Write 1 hundredths as a decimal.",
    ],
)
def test_defective_prompts_are_flagged(text):
    assert find_text_defects(_question(text))


@pytest.mark.parametrize(
    "text",
    [
        "Simplify: 5^3 × 5^3",
        "Estimate 192 × 544 by rounding each number.",
        "10 equal parts make a whole. 8 parts are used, then 1 more part is used.",
        "Sam has 3 apples. He gives away 1 apple.",
        "What is 1 plus 2?",
        "1 is a factor of every number.",
        "The bag costs 1 times as much as the box.",
        "Round 11 apples to the nearest 10.",
        "The share is 0.1 units.",
        "Player 1 gets 3 points.",
        "Year 1 students line up in 4 rows.",
        "Table 1 lists the values.",
        "Team 1 scores 4 goals and Team 2 scores 6 goals.",
    ],
)
def test_clean_prompts_are_not_flagged(text):
    assert find_text_defects(_question(text)) == []


def test_sentence_initial_verb_is_not_treated_as_a_label():
    assert find_text_defects(_question("Write 1 hundredths as a decimal."))


def test_double_star_is_checked_in_answer_and_distractors_but_star_operator_only_in_prompt():
    q = _question("Simplify the power.", answer_text="4**8", distractors=["4**16", "2*sqrt(2)"])
    defects = find_text_defects(q)
    assert any(d.startswith("double_star in answer_text") for d in defects)
    assert any(d.startswith("double_star in distractor[0]") for d in defects)
    assert not any("distractor[1]" in d for d in defects)


def _load_skill(skill_id: str):
    _, skills = load_skills(ROOT / "skills")
    return {s.id: s for s in skills}[skill_id]


def test_render_covers_every_source_and_tier_at_the_requested_seed_count():
    skill = _load_skill("f10-division-remainder-context")
    rendered = render_skill(skill, seeds=3)
    assert [src.difficulty for src in rendered.sources] == ["easy", "medium", "hard"]
    assert all(len(src.rows) == 3 for src in rendered.sources)
    assert rendered.exceptions == 0
    assert rendered.questions == 9


def test_markdown_has_one_section_per_skill_and_one_table_per_tier():
    skills = [_load_skill(i) for i in ("f10-division-remainder-context", "f10-multidigit-multiplication")]
    rendered = [render_skill(s, seeds=2) for s in skills]
    md = to_markdown(rendered, seeds=2)
    for s in skills:
        assert f"## {s.id}" in md
    assert md.count("| seed | question | answer | distractors | flags |") == sum(len(r.sources) for r in rendered)
    assert "2 seeds per tier" in md


def test_cli_writes_the_file_and_exit_code_reflects_flags(tmp_path, capsys):
    out = tmp_path / "render.md"
    code = main(["--ids", "f10-division-remainder-context", "--seeds", "2", "--out", str(out)])
    assert out.exists()
    assert "## f10-division-remainder-context" in out.read_text(encoding="utf-8")
    assert code == 0, capsys.readouterr().err


def test_cli_rejects_unknown_ids(capsys):
    assert main(["--ids", "no-such-skill"]) == 2
    assert "unknown skill ids" in capsys.readouterr().err
