"""Tag skills and sources with `technology: required | none` for the technology flag.

Dry run by default: it renders every source over several seeds, classifies it, prints a summary and writes
a review list. `--write` inserts the tags into the skill YAML files, as plain text lines, so comments and
layout stay as they are. A tag that is already present is never changed.

Rules (see skills/README.md, `technology`):
  * the skills in REQUIRED_SKILLS name a tool in their own wording ("using technology", a financial
    calculator, a spreadsheet): the skill is tagged `required` and its sources inherit.
  * a source whose rendered text says calculator, technology or spreadsheet is `required`.
  * a source whose text asks for decimal places or significant figures is `required` when it is a hard or
    exam tier outside the early-years domains, and is otherwise left untagged for a hand sort (an untagged
    source is hidden when technology is off).
  * a source that never says any of that, over every seed, is `none`.
  * `either` is never written here: it needs a generator that declares both modes.

    python scripts/tag_technology.py                  # dry run, summary
    python scripts/tag_technology.py --write          # apply
    python scripts/tag_technology.py --report out.md  # also write the hand-sort list
"""

from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathgen.engine import make_for_skill  # noqa: E402
from mathgen.skills import load_skills  # noqa: E402

SEEDS = (1, 2, 3, 4, 5, 6)

REQUIRED_SKILLS = frozenset(
    {
        "senior-with-the-aid-of-a-financial-calculator-or",
        "senior-with-the-aid-of-a-financial-calculator-or-acmgm100",
        "senior-use-technology-and-a-recurrence-relation-to-model",
        "senior-use-technology-to-compare-numerically-and-graphically-the",
        "senior-use-technology-to-investigate-the-effect-of-the",
        "senior-use-technology-to-find-the-correlation-coefficient-an",
        "senior-use-numerical-integration-using-technology-f-left-t",
        "senior-estimate-the-limit-of-frac-a-h-1",
        "senior-use-a-spreadsheet-to-display-examples-of-the",
        "senior-deseasonalise-a-time-series-by-using-a-seasonal",
        "senior-find-distances-between-two-places-on-earth-using",
        "senior-calculate-probabilities-and-quantiles-associated-with-a-given",
    }
)

# Domains where "correct to 2 decimal places" is a rounding skill done by hand, not a calculator cue.
EARLY_DOMAINS = frozenset(
    {
        "early-foundations",
        "primary-number",
        "primary-geometry-time",
        "arithmetic",
        "measurement",
        "readiness",
        "secondary-number",
        "geometry",
        "linear-relations",
        "equations",
        "expressions-equivalence",
        "quadratics",
        "inequalities",
        "language-formulas",
    }
)

STRONG = re.compile(r"calculator|technology|\bCAS\b|(?:use|using|with) (?:a |the )?spreadsheet", re.I)
NO_TOOL = re.compile(r"(?:without|no|not use|not using|not allowed) (?:a |the |any )?calculator", re.I)
ROUNDING = re.compile(r"decimal place|\bd\.p\b|significant fig|significant digit", re.I)

_SKILLS: dict = {}


def _init() -> None:
    _, skills = load_skills()
    _SKILLS.update({s.id: s for s in skills})


def _blob(q) -> str:
    """What the learner reads and must type: the stem, any parts and the expected answer's wording."""
    return " ".join(str(getattr(q, k, "") or "") for k in ("text", "stem", "parts", "answer_text", "answer_form"))


def classify(job: tuple[str, int]) -> tuple[str, int, str, str]:
    """(skill id, source index) -> (skill id, index, label, reason). Label: required, none or review."""
    sid, i = job
    skill = _SKILLS[sid]
    src = skill.sources[i]
    saw_round = False
    for seed in SEEDS:
        try:
            q = make_for_skill(skill, seed, source=src)
        except Exception as e:  # a source that does not build is reported, never tagged
            return sid, i, "review", f"does not build: {type(e).__name__}"
        text = _blob(q)
        if STRONG.search(NO_TOOL.sub("", text)):
            return sid, i, "required", "text names a calculator or technology"
        if ROUNDING.search(text):
            saw_round = True
    if saw_round:
        if src.difficulty in ("hard", "exam") and skill.domain not in EARLY_DOMAINS:
            return sid, i, "required", "rounds to decimal places at a hard or exam tier"
        return sid, i, "review", "rounds to decimal places"
    return sid, i, "none", "no rounding or tool wording"


def insert_tags(text: str, skill_tag: str | None, src_tags: dict[int, str]) -> str:
    """Add `technology:` lines. Skill tag goes after `status:`; a source tag goes right after its `- topic:` line,
    or inside the braces of a one-line `- {topic: ...}` source."""
    lines = text.split("\n")
    out: list[str] = []
    in_sources = False
    idx = -1
    for line in lines:
        out.append(line)
        if re.match(r"^sources:\s*\[", line):  # every source inline on the one line

            def tag(m: re.Match) -> str:
                nonlocal idx
                idx += 1
                return f"{{technology: {src_tags[idx]}, topic:" if idx in src_tags else m.group(0)

            out[-1] = re.sub(r"\{topic:", tag, line)
            continue
        if re.match(r"^sources:\s*$", line):
            in_sources = True
            continue
        if in_sources and line and not line[0].isspace() and line[0] not in "-#":
            in_sources = False
        if skill_tag and re.match(r"^status:", line):
            out.append(f"technology: {skill_tag}")
        m = re.match(r"^(\s*)- topic:", line) if in_sources else None
        flow = re.match(r"^(\s*)- \{topic:", line) if in_sources else None
        if m:
            idx += 1
            if idx in src_tags:
                out.append(f"{m.group(1)}  technology: {src_tags[idx]}")
        elif flow:
            idx += 1
            if idx in src_tags:
                out[-1] = line.replace("{topic:", f"{{technology: {src_tags[idx]}, topic:", 1)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the tags to the skill files")
    ap.add_argument("--report", help="write the hand-sort list (markdown) here")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    missing = REQUIRED_SKILLS - by_id.keys()
    if missing:
        print("unknown REQUIRED_SKILLS:", sorted(missing))
        return 1
    jobs = [
        (s.id, i)
        for s in skills
        if s.id not in REQUIRED_SKILLS and s.technology is None
        for i, src in enumerate(s.sources)
        if src.technology is None
    ]
    with Pool(args.jobs, initializer=_init) as pool:
        results = pool.map(classify, jobs, chunksize=50)

    per_skill: dict[str, dict[int, str]] = {}
    review: list[tuple[str, int, str]] = []
    counts: dict[str, int] = {"required": 0, "none": 0, "review": 0}
    for sid, i, label, reason in results:
        counts[label] += 1
        if label == "review":
            review.append((sid, i, reason))
        else:
            per_skill.setdefault(sid, {})[i] = label
    print(f"skills forced required: {len(REQUIRED_SKILLS)}")
    print(f"sources: {counts}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("# Technology tags: sources left untagged for a hand sort\n\n")
            for sid, i, reason in sorted(review):
                s = by_id[sid]
                f.write(f"- `{sid}` source {i} ({s.sources[i].difficulty}, {s.domain}): {reason}\n")

    if args.write:
        paths = {p.stem: p for p in (ROOT / "skills").glob("*/*.yaml")}
        for sid in sorted(set(per_skill) | REQUIRED_SKILLS):
            p = paths[sid]
            text = p.read_text(encoding="utf-8")
            new = insert_tags(text, "required" if sid in REQUIRED_SKILLS else None, per_skill.get(sid, {}))
            if new != text:
                p.write_text(new, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
