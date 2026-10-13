"""Derive `year_level` and `naplan_domain` for each skill from the curriculum codes the file already carries.

    python scripts/backfill_year_level.py --check    # list files whose fields disagree with the derivation
    python scripts/backfill_year_level.py --write    # insert or correct the two lines in place

Nothing is guessed. A skill gets a value only from evidence in its own file:

  * Australian Curriculum v9 F-10 content codes, `ac9m<level><strand><nn>`, found in `tags`,
    `framework_areas` or `alignments[].code` (any case). Level is `f` or 1 to 10 (10A optional content,
    `ac9m10a..`, is Year 10). Strand is `n` number, `a` algebra, `m` measurement, `sp` space,
    `st` statistics or `p` probability.
  * Year tags `acara-v9-foundation` and `acara-v9-year-N` (also `-N-M` and `-year-N-year-M` for a span,
    read as the whole range).
  * University skills: the `university-core-v1` tag, an id starting `ucv1-`, or an alignment whose
    framework starts with `university`. They get `["university"]`.

`year_level` is every level named, in the schema's order. `naplan_domain` is the strands of the codes at
Foundation to Year 9 only (NAPLAN numeracy tests nothing above Year 9); year tags carry no strand, so they
add a year but no domain. Senior (`senior-*`, ACARA senior v8.4) files carry no unit or year, so they get
neither field. A skill with no evidence is left alone, and a field set by hand on such a skill is never
reported or changed. `exam_board` is not derived: no file names a board.

Only the two lines are edited, right after `domain:`; the rest of each file is left as written.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathgen.skills import NAPLAN_DOMAINS, YEAR_LEVELS  # noqa: E402

SKILLS = ROOT / "skills"
STRANDS = {"n": "number", "a": "algebra", "m": "measurement", "sp": "space", "st": "statistics", "p": "probability"}
CODE = re.compile(r"ac9m(f|10|[1-9])(sp|st|n|a|m|p)p?\d+", re.IGNORECASE)
YEAR_TAG = re.compile(r"acara-v9-(?:foundation|year-(\d+)(?:-(?:year-)?(\d+))?)")
FIELDS = ("year_level", "naplan_domain")


def evidence_strings(data: dict) -> list[str]:
    out = [str(t) for t in data.get("tags") or []]
    out += [str(a) for a in data.get("framework_areas") or []]
    out += [str(a["code"]) for a in data.get("alignments") or [] if a.get("code")]
    return out


def derive(data: dict) -> dict[str, list[str]]:
    """The fields this skill's own evidence supports; an empty dict when it has none."""
    years: set[str] = set()
    strands: set[str] = set()
    for text in evidence_strings(data):
        m = CODE.fullmatch(text)
        if m:
            level = m.group(1).upper() if m.group(1).lower() == "f" else m.group(1)
            years.add(level)
            if level == "F" or int(level) <= 9:
                strands.add(STRANDS[m.group(2).lower()])
            continue
        m = YEAR_TAG.fullmatch(text)
        if m:
            if text == "acara-v9-foundation":
                years.add("F")
            else:
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) else lo
                if lo <= hi <= 10:
                    years.update(str(n) for n in range(lo, hi + 1))
    alignments = data.get("alignments") or []
    if (
        "university-core-v1" in (data.get("tags") or [])
        or str(data.get("id", "")).startswith("ucv1-")
        or any(str(a.get("framework") or "").startswith("university") for a in alignments)
    ):
        years.add("university")
    out: dict[str, list[str]] = {}
    if years:
        out["year_level"] = [y for y in YEAR_LEVELS if y in years]
    if strands:
        out["naplan_domain"] = [d for d in NAPLAN_DOMAINS if d in strands]
    return out


def render(name: str, values: list[str]) -> str:
    # Digits are quoted so YAML keeps them strings; the labels F and university need no quotes.
    return f"{name}: [" + ", ".join(f"'{v}'" if v.isdigit() else v for v in values) + "]"


def current(data: dict) -> dict[str, list[str]]:
    return {n: [str(v) for v in data[n]] for n in FIELDS if data.get(n) is not None}


def apply(text: str, wanted: dict[str, list[str]]) -> str:
    """Set the wanted fields in the file text, touching only their own lines."""
    lines = text.split("\n")
    for name in reversed(FIELDS):
        if name not in wanted:
            continue
        new = render(name, wanted[name])
        for i, line in enumerate(lines):
            if line.startswith(f"{name}:"):
                lines[i] = new
                break
        else:
            at = next(i for i, line in enumerate(lines) if line.startswith("domain:"))
            lines.insert(at + 1, new)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report files that disagree with the derivation")
    mode.add_argument("--write", action="store_true", help="apply the derivation to the files")
    args = ap.parse_args(argv)

    disagree, none, changed = [], [], 0
    for path in sorted(SKILLS.glob("*/*.yaml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        want = derive(data)
        if not want:
            none.append(path.stem)
            continue
        have = current(data)
        if any(have.get(n) != want[n] for n in want):
            disagree.append(f"{path.parent.name}/{path.name}: has {have or 'nothing'}, derives {want}")
            if args.write:
                new = apply(text, want)
                if new != text:
                    path.write_text(new, encoding="utf-8")
                    changed += 1
    for line in disagree:
        print(line)
    print(f"{len(disagree)} files disagree; {len(none)} have no evidence and are left as they are.")
    if args.write:
        print(f"wrote {changed} files")
        return 0
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
