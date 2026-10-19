"""scripts/ci_durations.py: the per-file seconds each CI shard logs for refreshing the shard weights."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ci_durations.py"
spec = importlib.util.spec_from_file_location("ci_durations", SCRIPT)
ci_durations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_durations)

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.test_a" name="test_one" time="1.5"/>
<testcase classname="tests.test_a" name="test_two[x]" time="0.5"/>
<testcase classname="tests.test_split" name="test_slow[1]" time="10.0"/>
<testcase classname="tests.test_split" name="test_slow[2]" time="5.0"/>
<testcase classname="tests.test_split" name="test_fast" time="1.0"/>
<testcase classname="tests.test_split.TestGroup" name="test_method" time="2.0"/>
<testcase classname="tests.sub.test_b" name="test_b" time="3.0"/>
<testcase classname="tests.test_missing" name="test_gone" time="9.0"/>
</testsuite></testsuites>
"""


@pytest.fixture
def tree(tmp_path):
    for rel in ("tests/test_a.py", "tests/test_split.py", "tests/sub/test_b.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#")
    report = tmp_path / "junit.xml"
    report.write_text(REPORT)
    return tmp_path, report


def test_seconds_are_summed_per_file_and_per_function_of_a_split_file(tree):
    root, report = tree
    totals = ci_durations.durations(report, root, {"tests/test_split.py::test_slow": 1.0})
    assert totals == {
        "tests/test_a.py": 2.0,
        "tests/test_split.py::test_slow": 15.0,
        "tests/test_split.py::test_fast": 1.0,
        "tests/test_split.py::TestGroup::test_method": 2.0,
        "tests/sub/test_b.py": 3.0,
    }, "a test whose file is not under the root is left out, not misfiled"


def test_main_prints_largest_first_with_a_total(tree, capsys):
    root, report = tree
    # Run from the real repo root, so point the report at files that exist there.
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_ci_durations" name="test_x" time="4.0"/>'
        '<testcase classname="tests.test_ci_shard" name="test_y" time="6.0"/></testsuite></testsuites>'
    )
    assert ci_durations.main(["ci_durations.py", str(report)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("ci-seconds: 6.0 tests/test_ci_shard.py")
    assert lines[1].startswith("ci-seconds: 4.0 tests/test_ci_durations.py")
    assert lines[-1] == "ci-total: 10.0"


@pytest.mark.parametrize("content", [None, "", "<not xml"])
def test_a_missing_or_broken_report_never_fails_the_shard(tmp_path, capsys, content):
    report = tmp_path / "junit.xml"
    if content is not None:
        report.write_text(content)
    assert ci_durations.main(["ci_durations.py", str(report)]) == 0
    assert capsys.readouterr().out.startswith("ci-seconds: no durations")


def test_wrong_usage_is_an_error(capsys):
    assert ci_durations.main(["ci_durations.py"]) == 2
