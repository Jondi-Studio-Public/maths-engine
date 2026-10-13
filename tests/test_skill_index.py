"""The skill -> topic -> generator-module index: it is current, deterministic and points at real code."""

from pathlib import Path

import pytest
import yaml

from scripts import build_skill_index as index

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_index_is_current():
    reason = index.stale_reason()
    assert reason is None, (
        "The skill index is stale. A skill, a source topic or the module a generator lives in changed.\n"
        f"Regenerate it and commit the result:\n    {index.REGENERATE}\n\n{reason}"
    )


def test_rendering_twice_gives_identical_text():
    assert index.render() == index.render()


def test_every_skill_has_one_entry_sorted_by_id_and_every_path_is_relative_and_exists():
    data = yaml.safe_load(index.index_path().read_text(encoding="utf-8"))
    skills = index._skills()
    assert list(data) == sorted(s.id for s in skills)
    for skill in skills:
        entry = data[skill.id]
        assert set(entry) == {"domain", "status", "sources"}
        assert [src["topic"] for src in entry["sources"]] == list(dict.fromkeys(s.topic for s in skill.sources))
        for src in entry["sources"]:
            if src["module"] is not None:
                assert not Path(src["module"]).is_absolute() and "\\" not in src["module"]
                assert (ROOT / src["module"]).is_file(), src


def test_the_committed_file_carries_no_line_numbers():
    """A line number changes on every generator edit, which would make the file stale on almost every PR."""
    for line in index.render().splitlines():
        assert ".py:" not in line


def test_locate_gives_the_def_line_of_the_generator_even_when_the_decorator_spans_lines():
    checked = 0
    for _level, topic in sorted((lvl, t) for t, (lvl, _fn) in index.REGISTRY.items())[:200]:
        where = index.locate(topic)
        source = (ROOT / where.path).read_text(encoding="utf-8").splitlines()
        assert source[where.line - 1].lstrip().startswith(("def ", "async def ")), (topic, where)
        checked += 1
    assert checked


def test_where_finds_a_skill_a_topic_and_nothing():
    skill = index._skills()[0]
    found = index.where(skill.id)
    assert found and found[0].startswith(f"skill {skill.id}")
    assert any(".py:" in line for line in found[1:])
    topic = skill.sources[0].topic
    found = index.where(topic)
    assert found and found[0].startswith(f"topic {topic}: ") and skill.id in found[1]
    assert index.where("no-such-skill-or-topic") is None


def test_a_missing_or_edited_index_is_reported_with_the_command_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "index_path", lambda: tmp_path / "skills" / "gone.yaml")
    monkeypatch.setattr(index, "REPO_ROOT", tmp_path)
    assert index.REGENERATE in index.stale_reason("anything")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "gone.yaml").write_text("old\n", encoding="utf-8")
    reason = index.stale_reason("new\n")
    assert index.REGENERATE in reason and "-old" in reason and "+new" in reason


def test_check_and_where_exit_codes(capsys):
    assert index.main(["--check"]) == 0
    assert index.main(["--where", "no-such-skill-or-topic"]) == 1
    assert "no skill or topic named" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [[], ["--write", "--check"]])
def test_exactly_one_mode_is_required(flags):
    with pytest.raises(SystemExit) as stopped:
        index.main(flags)
    assert stopped.value.code == 2
