"""Build the skill -> topic -> generator-module index, and answer "where is this generator?".

The index is the YAML file `skills/index.yaml`: a mapping from skill id to `{domain, status, sources}`, where
`sources` lists each distinct source topic with `module`, the file that registers the topic's generator (null
when no generator is registered). Everything is read from the loaded library and the generator registry, never grepped, because a
`@generator(...)` decorator is often split over several lines.

The committed file holds module paths only. A line number changes with every edit to a generator module, so
putting it in the file would make it stale on almost every generator pull request. `--where` finds the line
on demand instead.

Usage:
    python scripts/build_skill_index.py --write        # regenerate the file
    python scripts/build_skill_index.py --check        # exit 1 if the file is stale or missing
    python scripts/build_skill_index.py --where simplify-surds     # a skill id: each source's generator
    python scripts/build_skill_index.py --where surds_simplify     # a topic: its generator and its skills

The output is deterministic: rows are sorted by skill id, topics keep their order in the skill file, and
nothing depends on the machine, the clock or the working directory.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen import REGISTRY  # noqa: E402  (importing the package registers every generator)
from mathgen.skills import load_skills  # noqa: E402

REGENERATE = "python scripts/build_skill_index.py --write"
NOT_REGISTERED = "(no generator registered)"

HEADER = f"""# Generated, do not edit. Which generator module builds each skill's questions.
# Regenerate with `{REGENERATE}`
# after adding, removing or renaming a skill, a source topic, or the module a generator lives in;
# a test fails while this file is stale. Paths are relative to the repository root. Line
# numbers are left out because they change on every edit: for one, run
# `python scripts/build_skill_index.py --where <skill-or-topic>`.
"""


def index_path() -> Path:
    return REPO_ROOT / "skills" / "index.yaml"


@dataclass(frozen=True)
class Location:
    path: str  # relative to the repository root, forward slashes
    line: int  # the `def` line of the generator function


def _relative(file: str) -> str:
    resolved = Path(file).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()  # a generator registered from outside the checkout


def locate(topic: str) -> Location | None:
    """Where a topic's generator function is defined, or None when no generator is registered for it."""
    entry = REGISTRY.get(topic)
    if entry is None:
        return None
    fn = inspect.unwrap(entry[1])
    file = inspect.getsourcefile(fn)
    if file is None:
        return None
    lines, first = inspect.getsourcelines(fn)
    # `first` is the first decorator line when the function is decorated; the definition is the `def` after it.
    offset = next((i for i, text in enumerate(lines) if text.lstrip().startswith(("def ", "async def "))), 0)
    return Location(_relative(file), first + offset)


def _skills():
    """The library in this checkout only: not a MATHS_SKILLS_DIR override or any extra folder."""
    return sorted(load_skills(REPO_ROOT / "skills", extra_roots=[])[1], key=lambda s: s.id)


def _distinct_topics(skill) -> list[str]:
    return list(dict.fromkeys(src.topic for src in skill.sources))


def build() -> dict:
    """The index as plain data: {skill id: {domain, status, sources: [{topic, module}]}}, sorted by id."""
    index = {}
    for skill in _skills():
        sources = []
        for topic in _distinct_topics(skill):
            where = locate(topic)
            sources.append({"topic": topic, "module": where.path if where else None})
        index[skill.id] = {"domain": skill.domain, "status": skill.status, "sources": sources}
    return index


def render() -> str:
    return HEADER + yaml.safe_dump(build(), sort_keys=False, width=1000, allow_unicode=True)


def stale_reason(current: str | None = None) -> str | None:
    """None when the committed file matches what `render` gives now; otherwise how it differs and how to fix it."""
    path = index_path()
    wanted = render() if current is None else current
    if not path.is_file():
        return f"{path.relative_to(REPO_ROOT).as_posix()} is missing. Regenerate it with: {REGENERATE}"
    have = path.read_text(encoding="utf-8")
    if have == wanted:
        return None
    diff = list(
        difflib.unified_diff(have.splitlines(), wanted.splitlines(), "committed", "regenerated", lineterm="", n=0)
    )
    shown = "\n".join(diff[:12]) + (f"\n... and {len(diff) - 12} more diff lines" if len(diff) > 12 else "")
    return (
        f"{path.relative_to(REPO_ROOT).as_posix()} is out of date with the skills and generators. "
        f"Regenerate it with: {REGENERATE}\n{shown}"
    )


def where(query: str) -> list[str] | None:
    """Output lines for a skill id or a topic; None when the query matches neither."""
    skills = _skills()
    by_id = {s.id: s for s in skills}
    out: list[str] = []
    if query in by_id:
        skill = by_id[query]
        out.append(f"skill {skill.id} ({skill.domain}, {skill.status}): {_relative(str(skill.path))}")
        for number, src in enumerate(skill.sources):
            loc = locate(src.topic)
            at = f"{loc.path}:{loc.line}" if loc else NOT_REGISTERED
            detail = ", ".join(
                f"{k}={v!r}" for k, v in sorted({"difficulty": src.difficulty, **src.params}.items()) if v
            )
            out.append(f"  source {number}: {src.topic}{f' [{detail}]' if detail else ''} -> {at}")
        return out
    if query in REGISTRY:
        loc = locate(query)
        out.append(f"topic {query}: {loc.path}:{loc.line}" if loc else f"topic {query}: {NOT_REGISTERED}")
        users = [s.id for s in skills if any(src.topic == query for src in s.sources)]
        out.append("  used by: " + (", ".join(users) if users else "(no skill)"))
        return out
    topics_used = {src.topic for s in skills for src in s.sources}
    if query in topics_used:  # a skill names it but nothing registers it
        out.append(f"topic {query}: {NOT_REGISTERED}")
        out.append("  used by: " + ", ".join(s.id for s in skills if any(x.topic == query for x in s.sources)))
        return out
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the index file")
    group.add_argument("--check", action="store_true", help="exit 1 if the index file is stale")
    group.add_argument("--where", metavar="SKILL_OR_TOPIC", help="print where a skill's or topic's generator is")
    args = ap.parse_args(argv)

    if args.write:
        text = render()
        index_path().write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {index_path().relative_to(REPO_ROOT).as_posix()} ({text.count(chr(10))} lines)")
        return 0
    if args.check:
        reason = stale_reason()
        if reason:
            print(reason, file=sys.stderr)
            return 1
        print("skill index is up to date")
        return 0
    found = where(args.where)
    if found is None:
        names = sorted({s.id for s in _skills()} | set(REGISTRY))
        close = difflib.get_close_matches(args.where, names, n=5, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        print(f"no skill or topic named {args.where!r}.{hint}", file=sys.stderr)
        return 1
    print("\n".join(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
