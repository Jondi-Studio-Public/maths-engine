"""Print the test files of shard I of N, one per line.

Usage: python scripts/ci_shard.py SHARD COUNT

SHARD and COUNT are plain decimal integers with 0 <= SHARD < COUNT. Each output line is a
test file path relative to the repository root, with "/" separators, so run pytest from
the repository root.

Files are balanced by size: sorted by size descending (ties by path) and each
assigned greedily to the currently lightest shard (ties to the lowest index).
Deterministic and stdlib only, so every CI shard computes the same partition.

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

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def partition(root: Path, count: int) -> list[list[str]]:
    """Split the test files into ``count`` shards balanced by byte size.

    Returns one sorted list per shard of paths relative to ``root`` with "/" separators.
    A shard can come back empty when there are fewer files than shards.
    """
    sized = [(p.stat().st_size, p.relative_to(root).as_posix()) for p in test_files(root)]
    sized.sort(key=lambda item: (-item[0], item[1]))
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count
    for size, name in sized:
        target = min(range(count), key=lambda i: (loads[i], i))
        shards[target].append(name)
        loads[target] += size
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
