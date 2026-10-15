"""Print the test files of shard I of N, one per line.

Usage: python scripts/ci_shard.py SHARD COUNT

SHARD and COUNT are plain decimal integers with 0 <= SHARD < COUNT. Each output line is a
test file path relative to the repository root, with "/" separators, so run pytest from
the repository root.

Work is balanced by weight: sorted by weight descending (ties by name) and each unit
assigned greedily to the currently lightest shard (ties to the lowest index).
Deterministic and stdlib only, so every CI shard computes the same partition.

A unit's weight is its measured seconds from scripts/ci_shard_weights.txt, else its
size in bytes times SECONDS_PER_BYTE. The weights file also names the few files that are
small but slow; those are SPLIT, and each of their test functions is a unit of its own,
printed as a pytest node id ("tests/test_x.py::test_fn"). The functions come from parsing
the file, not from the weights file, so a test added later still runs, exactly once.
CI hands every output line straight to pytest, which takes files and node ids alike.

This script fails closed, because CI runs its output as the list of tests to run and an
empty or missing list must never be mistaken for "nothing to do" or "run everything".
Anything it cannot answer with a non-empty list of files is an error: nothing is written
to stdout, one message goes to stderr, and the exit status is non-zero.

    exit 0  a non-empty list was printed
    exit 1  the arguments were valid but there is no usable list: the shard would be empty,
            COUNT is larger than the number of test files, the tree has no test files,
            or a test file name contains a line break
    exit 2  bad usage: wrong number of arguments, a non-integer or negative argument,
            COUNT < 1, or SHARD >= COUNT
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(__file__).resolve().parent / "ci_shard_weights.txt"

# Converts a file size to estimated CI seconds for a file with no measured weight. Fitted so
# that the unmeasured files and the measured sweeps sum to the shard-minutes seen on main
# (about 9,800 s over the whole suite on 2026-10-13).
SECONDS_PER_BYTE = 0.00057
# What a test function of a split file costs when the weights file does not name it.
DEFAULT_FUNCTION_SECONDS = 5.0

# ASCII digits only and a length cap: int() also accepts "1_0", " 3 ", "+3" and non-ASCII
# digits, and refuses very long digit strings, none of which is a shard number.
_NUMBER = re.compile(r"[0-9]{1,9}")


class ShardError(Exception):
    """A request the script refuses to answer; ``status`` is the process exit status."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def test_files(root: Path = ROOT) -> list[Path]:
    """Every pytest-collectable file under ``root/tests`` as an absolute path, sorted.

    A file is collectable when its name matches ``test_*.py`` or ``*_test.py``, which are
    pytest's default ``python_files`` patterns. Directories that merely have such a name
    are not files and are left out; the test files inside them are found on their own.
    """
    tests = root / "tests"
    found = {p for pattern in ("test_*.py", "*_test.py") for p in tests.rglob(pattern) if p.is_file()}
    return sorted(found)


def load_weights(path: Path = WEIGHTS) -> dict[str, float]:
    """Measured seconds per target, from lines of ``<seconds> <path>[::<test_function>]``.

    A missing file means no measurements (everything falls back to size). A malformed line is
    an error rather than being skipped, because a silently dropped weight skews the balance.
    """
    if not path.is_file():
        return {}
    weights: dict[str, float] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            seconds = float(parts[0])
        except ValueError:
            seconds = -1.0
        if len(parts) != 2 or not seconds >= 0 or parts[1] in weights:
            raise ShardError(f"{path.name}:{number}: expected '<seconds> <target>' once per target, got {raw!r}", 1)
        weights[parts[1]] = seconds
    return weights


def test_function_ids(path: Path) -> list[str]:
    """Every test a module defines, as the names pytest selects them by.

    ``test_fn`` for a module-level function (or a ``test_fn = factory(...)`` assignment) and
    ``Class::test_fn`` for a method of a ``Test*`` class, in source order, each once. A name defined twice is one test (the later definition
    replaces the earlier), so it is listed once.
    """
    names: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.append(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("test")
            and isinstance(node.value, ast.Call)
        ):
            # ``test_shape = family_test("shape", 4)``: pytest collects a module-level callable
            # by name, so a test built by a factory is a test function like any other.
            names.append(node.targets[0].id)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names += [
                f"{node.name}::{item.name}"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test")
            ]
    return list(dict.fromkeys(names))


def work_units(root: Path, weights: dict[str, float] | None = None) -> list[tuple[float, str]]:
    """Every unit of work as ``(seconds, target)``: a test file, or a node id of a split file."""
    if weights is None:
        weights = load_weights()
    units: list[tuple[float, str]] = []
    for path in test_files(root):
        name = path.relative_to(root).as_posix()
        prefix = name + "::"
        if any(key.startswith(prefix) for key in weights):
            functions = test_function_ids(path)
            if not functions:
                raise ShardError(f"{name} is split in the weights file but defines no test function", 1)
            units += [(weights.get(prefix + fn, DEFAULT_FUNCTION_SECONDS), prefix + fn) for fn in functions]
        else:
            units.append((weights.get(name, path.stat().st_size * SECONDS_PER_BYTE), name))
    return units


def partition(root: Path, count: int, weights: dict[str, float] | None = None) -> list[list[str]]:
    """Split the test files into ``count`` shards balanced by estimated seconds.

    Returns one sorted list per shard of targets relative to ``root`` with "/" separators:
    a test file path, or ``path::test_function`` for a file split by the weights file.
    A shard can come back empty when there are fewer units than shards.
    """
    units = work_units(root, weights)
    units.sort(key=lambda item: (-item[0], item[1]))
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for seconds, name in units:
        target = min(range(count), key=lambda i: (loads[i], i))
        shards[target].append(name)
        loads[target] += seconds
    return [sorted(s) for s in shards]


def _parse_number(text: str, what: str) -> int:
    if not _NUMBER.fullmatch(text):
        raise ShardError(f"{what} must be a non-negative decimal integer, got {text[:30]!r}", 2)
    return int(text)


def shard_files(shard_text: str, count_text: str, root: Path = ROOT) -> list[str]:
    """The non-empty list of test files for one shard, or a ``ShardError``."""
    shard = _parse_number(shard_text, "SHARD")
    count = _parse_number(count_text, "COUNT")
    if count < 1 or not 0 <= shard < count:
        raise ShardError(f"shard must satisfy 0 <= shard < count, got {shard}/{count}", 2)
    available = len(test_files(root))
    if available == 0:
        raise ShardError(f"no test files found under {root / 'tests'}", 1)
    if count > available:
        # Checked before partitioning so a huge COUNT cannot allocate a huge list.
        raise ShardError(
            f"{count} shards requested but only {available} test files exist; some shards would be empty", 1
        )
    names = partition(root, count)[shard]
    if not names:
        raise ShardError(f"shard {shard}/{count} has no test files", 1)
    if any(ch in name for name in names for ch in "\r\n"):
        raise ShardError("a test file name contains a line break and cannot be listed one per line", 1)
    return names


def main(argv: list[str], root: Path = ROOT) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        names = shard_files(argv[1], argv[2], root)
    except ShardError as error:
        print(f"ci_shard.py: error: {error}", file=sys.stderr)
        return error.status
    sys.stdout.write("\n".join(names) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
