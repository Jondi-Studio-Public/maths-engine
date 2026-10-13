"""The generator sweep parts cover every topic exactly once: see `tests/mathgen_sweep_cases.py`."""

from pathlib import Path

from tests.mathgen_sweep_cases import (
    BUILD_PARTS,
    BUILD_TOPICS,
    DEFAULT_PARTS,
    TOPICS,
    build_part_topics,
    default_part_topics,
)


def _check(kind, parts, part_topics, every):
    topics = [t for part in range(parts) for t in part_topics(part)]
    assert sorted(topics) == sorted(every)
    assert len(topics) == len(set(topics))
    for part in range(parts):
        assert Path(__file__).with_name(f"test_mathgen_sweep_{kind}_part_{part}.py").is_file()
    assert not Path(__file__).with_name(f"test_mathgen_sweep_{kind}_part_{parts}.py").exists()


def test_the_build_parts_cover_every_build_topic_once():
    _check("build", BUILD_PARTS, build_part_topics, BUILD_TOPICS)


def test_the_default_parts_cover_every_topic_once():
    _check("default", DEFAULT_PARTS, default_part_topics, TOPICS)
