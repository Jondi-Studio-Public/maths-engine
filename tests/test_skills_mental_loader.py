"""Tests the skill loader's validation of mental-mode blocks: level structure, timings and generation.

The loader (mathgen/skills.py) checks a skill's own YAML `mental:` block for structure (1-5 levels,
positive target_s/limit_s, limit_s >= target_s) and for generation (each level's topic/params/difficulty
must produce a question for seeds 1..5)."""

from __future__ import annotations

from mathgen import skills as skills_mod


def _write_skill(tmp_path, mental_yaml: str):
    domain_dir = tmp_path / "arithmetic"
    domain_dir.mkdir()
    (domain_dir / "add-two.yaml").write_text(
        "id: add-two\n"
        "name: Add two\n"
        "domain: arithmetic\n"
        "status: active\n"
        "prerequisites: []\n"
        "answer: number\n"
        "sources:\n"
        "  - topic: addition\n"
        "    params: {}\n"
        "    difficulty: easy\n" + mental_yaml
    )
    return tmp_path


def test_valid_mental_block_generates_and_passes(tmp_path):
    root = _write_skill(
        tmp_path,
        """mental:
  levels:
    - {topic: addition, params: {lo: 1, hi: 20}, target_s: 3, limit_s: 8}
  entry_stage: 1
""",
    )
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert report.ok, report.errors
    skill = next(s for s in skill_list if s.id == "add-two")
    assert skill.mental.entry_stage == 1
    assert len(skill.mental.levels) == 1


def test_mental_block_with_too_many_levels_is_rejected(tmp_path):
    levels = "\n".join(
        f"    - {{topic: addition, params: {{lo: 1, hi: 20}}, target_s: {i}, limit_s: {i + 5}}}"
        for i in range(1, 7)  # 6 levels, max is 5
    )
    root = _write_skill(tmp_path, f"mental:\n  levels:\n{levels}\n")
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert not report.ok
    assert any("mental.levels must have 1 to 5" in e for e in report.errors)


def test_mental_block_missing_target_or_limit_is_rejected(tmp_path):
    root = _write_skill(
        tmp_path,
        """mental:
  levels:
    - {topic: addition, params: {lo: 1, hi: 20}, limit_s: 8}
""",
    )
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert not report.ok
    assert any("target_s must be a positive number" in e for e in report.errors)


def test_mental_block_limit_less_than_target_is_rejected(tmp_path):
    root = _write_skill(
        tmp_path,
        """mental:
  levels:
    - {topic: addition, params: {lo: 1, hi: 20}, target_s: 10, limit_s: 5}
""",
    )
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert not report.ok
    assert any("limit_s must be >= target_s" in e for e in report.errors)


def test_mental_block_that_cannot_generate_is_rejected_even_though_sources_are_fine(tmp_path):
    """A bad mental block on an otherwise valid skill is a validation
    error -- the skill's own
    `sources` (used for focus mode) are perfectly fine here."""
    root = _write_skill(
        tmp_path,
        """mental:
  levels:
    - {topic: addition, params: {lo: not-a-number}, target_s: 3, limit_s: 8}
""",
    )
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert not report.ok
    assert any("mental level 0" in e and "failed at seed" in e for e in report.errors)
    # the skill's normal source is untouched and would validate fine alone
    assert not any("source topic" in e for e in report.errors)


def test_skill_with_no_mental_block_is_unaffected(tmp_path):
    root = _write_skill(tmp_path, "")
    domains, skill_list = skills_mod.load_skills(root)
    report = skills_mod.validate(domains, skill_list)
    assert report.ok, report.errors
    skill = next(s for s in skill_list if s.id == "add-two")
    assert skill.mental is None
