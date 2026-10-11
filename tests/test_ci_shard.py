"""The CI shards together cover every test file exactly once, and the sharder fails closed."""

import fnmatch
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ci_shard.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow_shard_count() -> int:
    """The shard count, read from the CI matrix, which is the one place it is defined."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    shards = workflow["jobs"]["test"]["strategy"]["matrix"]["shard"]
    assert shards == list(range(len(shards))), "the matrix must list shards 0..N-1 exactly once each"
    return len(shards)


SHARDS = _workflow_shard_count()

_spec = importlib.util.spec_from_file_location("ci_shard", SCRIPT)
ci_shard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_shard)


def test_real_tree_union_is_exact():
    expected = sorted(p.relative_to(ROOT).as_posix() for p in ci_shard.test_files(ROOT))
    shards = ci_shard.partition(ROOT, SHARDS)
    flat = [f for shard in shards for f in shard]
    assert len(flat) == len(set(flat)), "a file landed in two shards"
    assert sorted(flat) == expected
    assert all(shards), "an empty shard wastes a CI job"


def test_cli_matches_partition_and_is_deterministic():
    shards = ci_shard.partition(ROOT, SHARDS)
    for i in (0, SHARDS - 1):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), str(i), str(SHARDS)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert out == shards[i]


def test_temp_tree_balance(tmp_path):
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    sizes = {"a": 100, "b": 90, "c": 60, "d": 50, "e": 40, "f": 10}
    for name, size in sizes.items():
        (tmp_path / "tests" / f"test_{name}.py").write_text("#" * size)
    (tmp_path / "tests" / "sub" / "test_g.py").write_text("#" * 5)
    (tmp_path / "tests" / "conftest.py").write_text("#" * 500)
    shards = ci_shard.partition(tmp_path, 3)
    flat = sorted(f for s in shards for f in s)
    assert flat == sorted([f"tests/test_{n}.py" for n in sizes] + ["tests/sub/test_g.py"])
    loads = [sum((tmp_path / f).stat().st_size for f in s) for s in shards]
    assert max(loads) - min(loads) <= 40


def test_bad_arguments_rejected():
    r = subprocess.run([sys.executable, str(SCRIPT), "16", "16"], capture_output=True, text=True)
    assert r.returncode == 2


# ---------------------------------------------------------------------------
# Independent checks. The tests above compare ci_shard with its own test_files(),
# so they cannot notice a file that test_files() never finds. These do not call it.
# ---------------------------------------------------------------------------

# pytest's default python_files patterns, and the directories it does not recurse into by default.
_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
_PYTEST_SKIPS = (".*", "build", "dist", "node_modules", "venv", "*.egg", "*.egg-info", "__pycache__", "site-packages")


def _walk_test_files(base: Path, skip_top: tuple[str, ...] = ()) -> set[str]:
    """Test-pattern files under ``base`` found with os.walk, as posix paths relative to ``base``."""
    found = set()
    for dirpath, dirnames, filenames in os.walk(base):
        here = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if not any(fnmatch.fnmatch(d, pat) for pat in _PYTEST_SKIPS) and not (here == base and d in skip_top)
        ]
        for name in filenames:
            if any(fnmatch.fnmatch(name, pat) for pat in _TEST_FILE_PATTERNS):
                found.add((here / name).relative_to(base).as_posix())
    return found


def test_real_tree_union_matches_independent_discovery():
    expected = _walk_test_files(ROOT / "tests")
    expected = {f"tests/{name}" for name in expected}
    assert len(expected) > 100, "the independent walk found almost nothing; this check is broken"
    flat = [f for shard in ci_shard.partition(ROOT, SHARDS) for f in shard]
    assert len(flat) == len(set(flat)), "a file landed in two shards"
    assert set(flat) == expected, (
        f"missing from every shard: {sorted(expected - set(flat))[:5]}; "
        f"in a shard but not on disk: {sorted(set(flat) - expected)[:5]}"
    )


def _repository_files() -> set[str] | None:
    """Tracked and not-ignored files as posix paths, or None when this is not a git checkout."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {name for name in done.stdout.decode("utf-8", "replace").split("\0") if name}


def test_no_test_file_lives_outside_tests_dir():
    """CI runs the shards' explicit file lists, so a test file anywhere else would never run."""
    files = _repository_files()
    if files is None:  # a source archive: fall back to walking the tree, skipping virtualenvs and the like
        stray = sorted(_walk_test_files(ROOT, skip_top=("tests",)))
    else:
        stray = sorted(
            name
            for name in files
            if not name.startswith("tests/")
            and any(fnmatch.fnmatch(Path(name).name, pat) for pat in _TEST_FILE_PATTERNS)
        )
    assert not stray, f"test files outside tests/ are not in any CI shard: {stray}"


def test_discovery_agrees_with_pytest_on_a_synthetic_tree(tmp_path):
    """Run real ``pytest --collect-only`` over a small tree and compare the files it finds."""
    names = [
        "tests/test_a.py",
        "tests/b_test.py",
        "tests/sub/test_c.py",
        "tests/sub/deep/d_test.py",
        "tests/sub/deep/test_e.py",
        "tests/conftest.py",
        "tests/helper.py",
        "tests/testing_x.py",
        "tests/Test_upper.py",
        "tests/data_test.txt",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_ok():\n    pass\n" if name.endswith(".py") else "x\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = {line.split("::")[0] for line in result.stdout.splitlines() if "::" in line}
    ours = {p.relative_to(tmp_path).as_posix() for p in ci_shard.test_files(tmp_path)}
    assert collected, "pytest collected nothing from the synthetic tree"
    assert ours == collected


def test_test_files_are_absolute_and_partition_paths_are_relative():
    files = ci_shard.test_files(ROOT)
    assert files and all(p.is_absolute() for p in files)
    names = [name for shard in ci_shard.partition(ROOT, SHARDS) for name in shard]
    assert names and not any(Path(n).is_absolute() or "\\" in n for n in names)


def test_directory_named_like_a_test_file_is_not_a_test_file(tmp_path):
    (tmp_path / "tests" / "test_dir.py").mkdir(parents=True)
    (tmp_path / "tests" / "test_dir.py" / "test_inner.py").write_text("#")
    (tmp_path / "tests" / "test_real.py").write_text("#")
    assert [p.relative_to(tmp_path).as_posix() for p in ci_shard.test_files(tmp_path)] == [
        "tests/test_dir.py/test_inner.py",
        "tests/test_real.py",
    ]


# ---------------------------------------------------------------------------
# Fail closed: CI runs the output as the list of tests to run, so a bad request must
# produce a non-zero exit, nothing on stdout and one clear message, never a traceback.
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _assert_refused(result: subprocess.CompletedProcess, status: int) -> None:
    assert result.returncode == status, (result.returncode, result.stdout, result.stderr)
    assert result.stdout == "", "a refused request must print no file list"
    assert "ci_shard.py: error:" in result.stderr or "Usage:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "shard,count",
    [
        ("x", "16"),
        ("0", "x"),
        ("", "16"),
        ("0", ""),
        ("1.5", "16"),
        ("-1", "16"),
        ("0", "-4"),
        (" 3", "16"),
        ("3 ", "16"),
        ("+3", "16"),
        ("1_0", "16"),
        ("0x3", "16"),
        pytest.param("٣", "16", id="arabic-indic-digit-three"),  # int() accepts it
        pytest.param("3\n", "16", id="trailing-newline"),
        pytest.param("9" * 5000, "16", id="5000-digits"),
    ],
)
def test_non_integer_arguments_are_refused(shard, count):
    _assert_refused(_run_cli(shard, count), 2)


@pytest.mark.parametrize("shard,count", [("16", "16"), ("17", "16"), ("0", "0"), ("5", "3")])
def test_index_out_of_range_is_refused(shard, count):
    _assert_refused(_run_cli(shard, count), 2)


@pytest.mark.parametrize("args", [(), ("1",), ("1", "2", "3")])
def test_wrong_number_of_arguments_is_refused(args):
    _assert_refused(_run_cli(*args), 2)


def test_more_shards_than_test_files_is_refused():
    # One more shard than the tree has test files, counted rather than hard-coded: the
    # tree has outgrown any fixed guess (1000 stopped being "more" at 1000 test files).
    count = len(ci_shard.test_files(ROOT)) + 1
    _assert_refused(_run_cli(str(count - 1), str(count)), 1)


def test_huge_shard_count_is_refused_without_allocating_it():
    # Without the up-front check this would build a list of a billion empty lists.
    _assert_refused(_run_cli("0", "999999999"), 1)


def test_empty_shard_is_refused(tmp_path, capsys):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_big.py").write_text("#" * 50)
    (tests / "test_zero_a.py").write_text("")
    (tests / "test_zero_b.py").write_text("")
    # Three files, three shards, but both zero-byte files pile onto the lightest shard.
    assert ci_shard.partition(tmp_path, 3) == [
        ["tests/test_big.py"],
        ["tests/test_zero_a.py", "tests/test_zero_b.py"],
        [],
    ]
    assert ci_shard.main(["ci_shard.py", "2", "3"], root=tmp_path) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "shard 2/3 has no test files" in out.err


def test_more_shards_than_files_in_a_temp_tree_is_refused(tmp_path, capsys):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text("#")
    (tmp_path / "tests" / "test_two.py").write_text("#")
    assert ci_shard.main(["ci_shard.py", "0", "3"], root=tmp_path) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "only 2 test files" in out.err
    assert ci_shard.main(["ci_shard.py", "1", "2"], root=tmp_path) == 0


def test_tree_without_test_files_is_refused(tmp_path, capsys):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "helper.py").write_text("#")
    assert ci_shard.main(["ci_shard.py", "0", "1"], root=tmp_path) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "no test files" in out.err


def test_file_name_with_a_line_break_is_refused(tmp_path, capsys):
    (tmp_path / "tests").mkdir()
    try:
        (tmp_path / "tests" / "test_a\nb.py").write_text("#")
    except OSError:
        pytest.skip("this filesystem cannot create a file name containing a newline")
    assert ci_shard.main(["ci_shard.py", "0", "1"], root=tmp_path) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "line break" in out.err


def test_every_real_shard_succeeds_and_prints_a_newline_terminated_list():
    for i in range(SHARDS):
        names = ci_shard.shard_files(str(i), str(SHARDS), ROOT)
        assert names and names == sorted(names)
    result = _run_cli("0", str(SHARDS))
    assert result.returncode == 0 and result.stderr == ""
    assert result.stdout.endswith("\n") and not result.stdout.endswith("\n\n")
