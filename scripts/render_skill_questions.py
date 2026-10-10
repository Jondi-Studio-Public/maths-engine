"""Render sample questions for a set of skills to one Markdown file for review.

The regrade sweep proves that the grader accepts
the canonical answer and rejects the distractors. It never reads the prompt.
Questions such as ``Simplify: 5**3 × 5**3`` and ``1 more parts are used``
passed that sweep (0 false accepts, 0 false rejects) and still shipped. This
script is the prompt-side review: it renders every source and difficulty tier
of the selected skills across a few seeds, exactly as `mathgen.make` produces
them, so a human can read what a learner will read in ten minutes.

Each row also carries the mechanical flags from `mathgen.text_quality`, so the artefact and the test can never disagree.

Usage:
    python scripts/render_skill_questions.py --ids <id> [<id> ...] [--seeds N] [--out PATH]
    python scripts/render_skill_questions.py --domain <domain> [--seeds N] [--out PATH]
    python scripts/render_skill_questions.py --all-active [--seeds N] [--out PATH]
    python scripts/render_skill_questions.py --since <git-ref> [--seeds N] [--out PATH]

Without `--out` the Markdown goes to stdout. Exit 1 if any rendered question
carries a text flag or a generator raised; else 0.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen.skills import Skill, load_skills  # noqa: E402
from mathgen import make  # noqa: E402
from mathgen.text_quality import find_text_defects  # noqa: E402
from scripts.regrade_sweep import _difficulties_for, _tuplify, select_skills  # noqa: E402

DEFAULT_SEEDS = 5


@dataclass
class RenderedRow:
    seed: int
    text: str
    answer_text: str
    distractors: list[str]
    flags: list[str]


@dataclass
class RenderedSource:
    topic: str
    difficulty: str | None
    params: dict
    rows: list[RenderedRow] = field(default_factory=list)
    exceptions: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class RenderedSkill:
    skill: Skill
    sources: list[RenderedSource] = field(default_factory=list)

    @property
    def flagged(self) -> int:
        return sum(1 for src in self.sources for row in src.rows if row.flags)

    @property
    def exceptions(self) -> int:
        return sum(len(src.exceptions) for src in self.sources)

    @property
    def questions(self) -> int:
        return sum(len(src.rows) for src in self.sources)


def render_skill(skill: Skill, seeds: int = DEFAULT_SEEDS) -> RenderedSkill:
    """Generate `seeds` questions for every source and every difficulty tier
    the skill declares, in the order the skill YAML lists them."""
    rendered = RenderedSkill(skill=skill)
    for src in skill.sources:
        params = _tuplify(src.params)
        for difficulty in _difficulties_for(src.topic, src.difficulty):
            block = RenderedSource(topic=src.topic, difficulty=difficulty, params=dict(src.params or {}))
            for seed in range(seeds):
                try:
                    question = make(src.topic, seed=seed, difficulty=difficulty, **params)
                except Exception as exc:  # noqa: BLE001 - a generator exception is itself a finding
                    block.exceptions.append((seed, f"{type(exc).__name__}: {exc}"))
                    continue
                block.rows.append(
                    RenderedRow(
                        seed=seed,
                        text=question.text,
                        answer_text=question.answer_text,
                        distractors=list(question.distractors),
                        flags=find_text_defects(question),
                    )
                )
            rendered.sources.append(block)
    return rendered


def _cell(value: str) -> str:
    """One Markdown table cell: pipes escaped, newlines flattened."""
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", " ")


def to_markdown(rendered: list[RenderedSkill], seeds: int, title: str = "Rendered questions") -> str:
    total_questions = sum(r.questions for r in rendered)
    total_flagged = sum(r.flagged for r in rendered)
    total_exceptions = sum(r.exceptions for r in rendered)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out: list[str] = [
        f"# {title}",
        "",
        f"Rendered {stamp}: {len(rendered)} skills, {total_questions} questions, "
        f"{seeds} seeds per tier, {total_flagged} flagged, {total_exceptions} generator exceptions.",
        "",
        "Read every prompt as the learner will. The `flags` column is the mechanical check from "
        "`mathgen/text_quality.py`; an empty column is not a clean bill of health for wording.",
        "",
    ]

    flagged_skills = [r for r in rendered if r.flagged or r.exceptions]
    if flagged_skills:
        out.append("## Needs attention")
        out.append("")
        for r in flagged_skills:
            out.append(f"- `{r.skill.id}`: {r.flagged} flagged, {r.exceptions} exceptions")
        out.append("")

    for r in rendered:
        skill = r.skill
        out.append(f"## {skill.id}")
        out.append("")
        out.append(f"**{skill.name}** ({skill.domain}, answer: `{skill.answer}`)")
        out.append("")
        out.append(skill.description or "")
        out.append("")
        for src in r.sources:
            tier = src.difficulty or "default"
            out.append(f"### {tier}: `{src.topic}` {src.params if src.params else ''}".rstrip())
            out.append("")
            out.append("| seed | question | answer | distractors | flags |")
            out.append("|---|---|---|---|---|")
            for row in src.rows:
                out.append(
                    "| {seed} | {text} | {answer} | {distractors} | {flags} |".format(
                        seed=row.seed,
                        text=_cell(row.text),
                        answer=_cell(row.answer_text),
                        distractors=_cell("; ".join(row.distractors)),
                        flags=_cell("; ".join(row.flags)),
                    )
                )
            for seed, message in src.exceptions:
                out.append(f"| {seed} | **generator raised** | | | {_cell(message)} |")
            out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+", metavar="ID")
    group.add_argument("--domain", metavar="DOMAIN")
    group.add_argument("--all-active", action="store_true")
    group.add_argument("--since", metavar="GIT_REF")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", metavar="PATH", help="write the Markdown here instead of stdout")
    parser.add_argument("--title", default="Rendered questions")
    args = parser.parse_args(argv)

    _, skills = load_skills()
    selected = select_skills(
        skills,
        ids=args.ids,
        domain=args.domain,
        all_active=args.all_active,
        since=args.since,
    )
    if args.ids:
        missing = sorted(set(args.ids) - {s.id for s in selected})
        if missing:
            print(f"unknown skill ids: {', '.join(missing)}", file=sys.stderr)
            return 2
    if not selected:
        print("no skills selected", file=sys.stderr)
        return 2

    rendered = [render_skill(skill, seeds=args.seeds) for skill in selected]
    markdown = to_markdown(rendered, seeds=args.seeds, title=args.title)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        sys.stdout.write(markdown + "\n")

    flagged = sum(r.flagged for r in rendered)
    exceptions = sum(r.exceptions for r in rendered)
    questions = sum(r.questions for r in rendered)
    print(
        f"{len(rendered)} skills, {questions} questions, {flagged} flagged, {exceptions} generator exceptions",
        file=sys.stderr,
    )
    return 1 if (flagged or exceptions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
