"""Print how long each test file (or split test function) took in one CI shard.

Usage: python scripts/ci_durations.py JUNIT_XML

CI runs pytest with ``--junitxml`` and calls this afterwards, so every shard's log ends with one
line per target:

    ci-seconds: <seconds> <target>

The targets are the same as in scripts/ci_shard_weights.txt (a test file, or ``file::test_fn`` for a
file that weights file splits), so the log is exactly what is needed to refresh that file. The
seconds are the summed per-test times, which pytest-xdist runs in parallel: they add up to roughly
four times the shard's wall-clock time on a 4-vCPU runner, so only the ratios between targets
carry over, and ``ci-total:`` at the end gives the sum to scale them by.

It only ever reports. A missing or unreadable report (a cancelled job) prints one line and exits 0,
so it can never turn a green shard red or hide the real failure of a red one.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_shard  # noqa: E402


def _target(classname: str, name: str, split_files: set[str], root: Path) -> str | None:
    """The weights-file target a test belongs to, or None when its file cannot be found."""
    parts = classname.split(".")
    for end in range(1, len(parts) + 1):
        rel = "/".join(parts[:end]) + ".py"
        if (root / rel).is_file():
            if rel not in split_files:
                return rel
            owner = parts[end:]
            return rel + "::" + "::".join([*owner, name.split("[", 1)[0]])
    return None


def durations(report: Path, root: Path = ROOT, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Summed seconds per target from a pytest junit report."""
    if weights is None:
        weights = ci_shard.load_weights()
    split_files = {key.split("::", 1)[0] for key in weights if "::" in key}
    totals: dict[str, float] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        target = _target(case.get("classname", ""), case.get("name", ""), split_files, root)
        if target is not None:
            totals[target] = totals.get(target, 0.0) + float(case.get("time", "0") or 0)
    return totals


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        totals = durations(Path(argv[1]))
    except (OSError, ET.ParseError, ci_shard.ShardError) as error:
        print(f"ci-seconds: no durations ({error})")
        return 0
    for target, seconds in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        print(f"ci-seconds: {seconds:.1f} {target}")
    print(f"ci-total: {sum(totals.values()):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
