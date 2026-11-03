"""scripts/tag_technology.py adds `technology:` to skill YAML as plain text and leaves the rest alone."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from tag_technology import insert_tags  # noqa: E402

BLOCK = "id: a\nstatus: active\nsources:\n# a comment\n- topic: t1\n  params: {}\n  difficulty: easy\n- topic: t2\n  difficulty: hard\nmisconceptions: []\n"
FLOW = "id: a\nstatus: active\nsources:\n- {topic: t1, difficulty: easy}\n- {topic: t2, difficulty: hard}\ntags: []\n"
INLINE = "id: a\nstatus: active\nsources: [{topic: t1}, {topic: t2}]\ntags: []\n"


def _techs(text: str) -> list:
    return [s.get("technology") for s in yaml.safe_load(text)["sources"]]


def test_block_flow_and_inline_sources_get_their_own_tag():
    for text in (BLOCK, FLOW, INLINE):
        out = insert_tags(text, None, {0: "none", 1: "required"})
        assert _techs(out) == ["none", "required"]
        assert yaml.safe_load(out).get("technology") is None


def test_only_named_sources_are_tagged_and_comments_survive():
    out = insert_tags(BLOCK, None, {1: "required"})
    assert _techs(out) == [None, "required"]
    assert "# a comment" in out


def test_skill_tag_goes_after_status():
    out = insert_tags(BLOCK, "required", {})
    assert yaml.safe_load(out)["technology"] == "required"
    assert _techs(out) == [None, None]


def test_tagging_twice_changes_nothing():
    once = insert_tags(BLOCK, "required", {0: "none", 1: "required"})
    # A tagged skill and tagged sources: the second pass has nothing left to tag and adds no lines.
    assert insert_tags(once, "required", {}) == once
