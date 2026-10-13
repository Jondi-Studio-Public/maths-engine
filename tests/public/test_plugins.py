"""The Agent Plugins under `plugins/` (https://agent-plugins.org/specification, v1.0.0) and the Claude Code
marketplace that lists them. The rules checked here are the spec's, written out, since a test may not fetch its
JSON Schemas."""

import json
import re
import tomllib
from pathlib import Path

import pytest

from mathgen import mcp_server

ROOT = Path(__file__).resolve().parents[2]
PLUGINS = ROOT / "plugins"
ROLE = {"maths-engine-tutor": "learner", "maths-engine-teacher": "author"}
PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
MANIFEST_FIELDS = {"$schema", "name", "version", "description", "author", "homepage", "repository", "license"}
MANIFEST_FIELDS |= {"keywords", "extensions"}
NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,62}[a-z0-9])?$")
SKILL_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def engine_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def test_the_plugins_are_the_two_roles():
    assert sorted(p.name for p in PLUGINS.iterdir() if p.is_dir()) == sorted(ROLE)


@pytest.mark.parametrize("name", sorted(ROLE))
def test_manifest_follows_the_spec(name):
    manifest = load(PLUGINS / name / "plugin.json")
    assert set(manifest) <= MANIFEST_FIELDS  # the schema is closed
    assert manifest["$schema"] == PLUGIN_SCHEMA
    assert manifest["name"] == name and NAME.match(name) and "--" not in name and ".." not in name
    assert manifest["version"] == engine_version()
    assert set(manifest["author"]) <= {"name", "email", "url"}
    assert all(isinstance(k, str) for k in manifest["keywords"])


@pytest.mark.parametrize("name", sorted(ROLE))
def test_mcp_config_follows_the_spec(name):
    config = load(PLUGINS / name / "mcp.json")
    assert set(config) == {"$schema", "mcpServers"}
    assert config["$schema"] == MCP_SCHEMA  # the same spec version as plugin.json
    (server,) = config["mcpServers"].values()
    assert set(server) <= {"type", "command", "args", "env", "cwd"} and server["type"] == "stdio"
    # One executable token, no placeholder; a bare name, since nothing is bundled.
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", server["command"])
    assert "env" not in server  # nothing secret, nothing reserved
    for arg in server["args"]:
        assert "${" not in arg or re.fullmatch(r"\$\{PLUGIN_DATA\}/[a-z/-]+", arg), arg


@pytest.mark.parametrize("name", sorted(ROLE))
def test_the_server_runs_in_the_plugins_role(name):
    args = load(PLUGINS / name / "mcp.json")["mcpServers"]["maths-engine"]["args"]
    start = args.index("mathgen-mcp")
    assert args[: start - 1] == ["--from"]
    assert args[start - 1].startswith("maths-engine[mcp] @ git+https://github.com/Jondi-Studio-Public/maths-engine@")
    flags = args[start + 1 :]
    assert flags[flags.index("--role") + 1] == ROLE[name]
    assert flags[flags.index("--out-dir") + 1] == "${PLUGIN_DATA}/exports"
    # Every flag the plugin passes is one the command accepts.
    seen = {}

    class Fake:
        _maths_engine = type("E", (), {"close": lambda self: None})()

        def run(self, transport):
            pass

    real = mcp_server.build_server
    try:
        mcp_server.build_server = lambda role, **kw: seen.update(role=role, **kw) or Fake()
        assert mcp_server.main(flags) == 0
    finally:
        mcp_server.build_server = real
    assert seen["role"] == ROLE[name]


def test_claude_code_launches_the_same_servers():
    """Claude Code reads its MCP config from the marketplace entry (it does not read mcp.json); the servers must be
    the portable mcp.json's, with Claude Code's name for the data folder."""
    entries = {p["name"]: p for p in load(ROOT / ".claude-plugin" / "marketplace.json")["plugins"]}
    for name in ROLE:
        portable = load(PLUGINS / name / "mcp.json")["mcpServers"]
        claude = entries[name]["mcpServers"]
        assert claude.keys() == portable.keys()
        for key, server in portable.items():
            assert claude[key]["command"] == server["command"]
            assert [a.replace("${CLAUDE_PLUGIN_DATA}", "${PLUGIN_DATA}") for a in claude[key]["args"]] == server["args"]


def test_claude_code_marketplace_lists_both_plugins():
    market = load(ROOT / ".claude-plugin" / "marketplace.json")
    entries = {p["name"]: p for p in market["plugins"]}
    assert entries.keys() == ROLE.keys()
    for name, entry in entries.items():
        assert entry["source"] == f"./plugins/{name}"
        manifest = load(PLUGINS / name / "plugin.json")
        assert entry["version"] == manifest["version"] and entry["description"] == manifest["description"]


def _frontmatter(text: str) -> dict:
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md starts with YAML frontmatter"
    import yaml

    return yaml.safe_load(match.group(1))


@pytest.mark.parametrize("name", sorted(ROLE))
def test_skills_follow_the_agent_skills_spec(name):
    skills = sorted((PLUGINS / name / "skills").iterdir())
    assert skills
    for folder in skills:
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
        meta = _frontmatter(text)
        assert meta["name"] == folder.name and SKILL_NAME.match(folder.name) and "--" not in folder.name
        assert 1 <= len(meta["description"]) <= 1024
        for link in re.findall(r"\]\((?!https?://)([^)#]+)\)", text):
            target = (folder / link).resolve()
            assert target.is_file() and folder.resolve() in target.parents, link


def test_shared_skill_is_the_same_in_both_plugins():
    tutor, teacher = (PLUGINS / n / "skills" / "maths-skill-finder" / "SKILL.md" for n in sorted(ROLE))
    assert tutor.read_bytes() == teacher.read_bytes()


def test_tutor_skills_never_name_answer_flags():
    """The tutor's server has no `answers` or `reveal` parameter; its skills must not tell an agent to use one."""
    for path in (PLUGINS / "maths-engine-tutor" / "skills").rglob("*"):
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"answers\s*=\s*true|reveal\s*=\s*true|html-key|moodle|qti", text, re.IGNORECASE), path


def test_skills_name_only_tools_the_server_has():
    tools = {"engine_info", "find_skills", "get_skill", "make_questions", "get_question", "grade_answer"}
    tools.add("build_paper")
    # Markdown under plugins/ counts as code in ci.yml's docs-only filter, so changing a skill runs this test.
    for path in [p for p in PLUGINS.rglob("*") if p.suffix == ".md"]:
        for called in re.findall(r"`([a-z_]+)\(", path.read_text(encoding="utf-8")):
            assert called in tools, (path, called)


def test_no_file_escapes_its_plugin():
    for path in PLUGINS.rglob("*"):
        assert not path.is_symlink(), path
