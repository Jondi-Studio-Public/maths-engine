"""Build and check questions for every active skill, every source, seeds 0..N-1.

For each (skill, source, seed) this builds the question with `engine.make_for_skill` and runs the cheap
checks the engine already has:

* the question builds (an exception is a failure);
* it builds within the per-question time limit (a timeout is a failure);
* its own answer, `answer_text`, grades correct with `engine.grade`;
* none of its distractors grades correct (for a multiple-choice skill, none of the other options);
* with `--text-quality`, `mathgen.text_quality.find_text_defects` finds nothing in the prompt, answer or
  distractors. Off by default: the repository's tests apply that checker to chosen build topics only, and
  many other questions still trip it (a `*` operator or `**` in the text), so on by default it would fail
  every run for defects that predate it.

A failure never stops the sweep: every failing (skill, source, seed, reason) is collected and listed at the
end, and the exit status is 1 if there is any. This is wider than `scripts/regrade_sweep.py`, which also tries
synthetic near-miss answers but does not apply the text checks or a time limit, and it does not replace it.

Usage:
    python scripts/sweep_skills.py                          # everything, 40 seeds, one worker per CPU
    python scripts/sweep_skills.py --seeds 2                # a quick pass
    python scripts/sweep_skills.py --skills simplify-surds  # some skills only (ids; repeat or space-separate)
    python scripts/sweep_skills.py --workers 2 --timeout 10
    python scripts/sweep_skills.py --text-quality           # also fail on learner-facing text defects
    python scripts/sweep_skills.py --shard 0/4              # a quarter of the skills, split by a hash of the id

When `$GITHUB_STEP_SUMMARY` is set the report is also written there as Markdown.
The time limit uses `signal.setitimer`, so it is not enforced on Windows (a warning says so).
Exit status: 0 clean, 1 at least one failure or nothing to sweep, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen import engine  # noqa: E402
from mathgen.core import mcq_options  # noqa: E402
from mathgen.text_quality import find_text_defects  # noqa: E402

DEFAULT_SEEDS = 40
DEFAULT_TIMEOUT_S = 10.0
MAX_REASON_CHARS = 300
MAX_SUMMARY_ROWS = 200


class QuestionTimeout(BaseException):
    """Raised by the alarm. A BaseException so a generator's `except Exception` cannot swallow it."""


def _on_alarm(_signum, _frame):
    raise QuestionTimeout


@dataclass(frozen=True)
class Failure:
    skill: str
    source: str  # "<index>:<topic>[/<difficulty>]"
    seed: int
    reason: str


def _short(text: object) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= MAX_REASON_CHARS else one_line[: MAX_REASON_CHARS - 3] + "..."


def _source_label(index: int, src) -> str:
    return f"{index}:{src.topic}" + (f"/{src.difficulty}" if src.difficulty else "")


def check_question(skill, question, text_quality: bool = False) -> list[str]:
    """Every problem with one built question, as short reasons. Empty when it is clean."""
    problems: list[str] = []
    if question.parts:
        # A multi-part question is marked part by part: the labelled answer_text must earn full marks, and each
        # part's distractors, with every other part answered correctly, must not be accepted for that part.
        if not engine.grade(skill, question, question.answer_text).correct:
            problems.append(f"answer_text {question.answer_text!r} does not grade correct")
        canonical = {part.label: part.answer_text for part in question.parts}
        for part in question.parts:
            for given in part.distractors:
                marked = engine.grade(skill, question, {**canonical, part.label: given})
                if marked.parts[part.label].correct:
                    problems.append(f"distractor {given!r} of part ({part.label}) grades correct")
    elif skill.answer == "mcq":
        options = mcq_options(question)
        if question.answer_text not in options:
            problems.append(f"answer_text {question.answer_text!r} is not among the options")
        else:
            right = options.index(question.answer_text)
            if not engine.grade(skill, question, str(right)).correct:
                problems.append(f"the correct option {right} does not grade correct")
            for index, option in enumerate(options):
                if index != right and engine.grade(skill, question, str(index)).correct:
                    problems.append(f"wrong option {index} {option!r} grades correct")
    else:
        if not engine.grade(skill, question, question.answer_text).correct:
            problems.append(f"answer_text {question.answer_text!r} does not grade correct")
        for distractor in question.distractors:
            if engine.grade(skill, question, distractor).correct:
                problems.append(f"distractor {distractor!r} grades correct")
    if text_quality:
        problems.extend(f"text: {defect}" for defect in find_text_defects(question))
    return problems


def sweep_source(
    skill, index: int, seeds: int, timeout_s: float, text_quality: bool = False
) -> tuple[int, list[Failure]]:
    """Seeds 0..seeds-1 of one source of one skill: (questions built and checked, failures)."""
    src = skill.sources[index]
    label = _source_label(index, src)
    failures: list[Failure] = []
    checked = 0
    use_timer = hasattr(signal, "setitimer") and timeout_s > 0
    if use_timer:
        previous = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        for seed in range(seeds):
            reasons: list[str] = []
            try:
                if use_timer:
                    signal.setitimer(signal.ITIMER_REAL, timeout_s)
                try:
                    question = engine.make_for_skill(skill, seed, source=src)
                    reasons = check_question(skill, question, text_quality)
                finally:
                    if use_timer:
                        signal.setitimer(signal.ITIMER_REAL, 0)
            except QuestionTimeout:
                reasons = [f"timeout: over {timeout_s:g} s"]
            except Exception as exc:  # noqa: BLE001 - an exception from a generator or grader is the finding
                reasons = [f"exception: {type(exc).__name__}: {_short(exc)}"]
            checked += 1
            failures.extend(Failure(skill.id, label, seed, _short(reason)) for reason in reasons)
    finally:
        if use_timer:
            signal.signal(signal.SIGALRM, previous)
    return checked, failures


# --- the process pool -----------------------------------------------------------------------------------------

_SKILLS: dict = {}


def _init_worker() -> None:
    _SKILLS.update({s.id: s for s in engine.load_skills()[1]})


def _run_task(task: tuple[str, int, int, float, bool]) -> tuple[str, int, int, list[Failure]]:
    skill_id, index, seeds, timeout_s, text_quality = task
    checked, failures = sweep_source(_SKILLS[skill_id], index, seeds, timeout_s, text_quality)
    return skill_id, index, checked, failures


def in_shard(skill_id: str, shard: int, shards: int) -> bool:
    """Whether a skill belongs to shard `shard` of `shards`; a stable hash of the id, so it never depends on order."""
    return int(hashlib.sha256(skill_id.encode()).hexdigest(), 16) % shards == shard


def select(skills, ids: list[str] | None, shard: tuple[int, int] | None):
    chosen = [s for s in skills if s.status == "active" and s.sources]
    if ids:
        chosen = [s for s in chosen if s.id in set(ids)]
    if shard:
        chosen = [s for s in chosen if in_shard(s.id, *shard)]
    return sorted(chosen, key=lambda s: s.id)


def run(
    skills, seeds: int, workers: int, timeout_s: float, text_quality: bool = False, progress: bool = False
) -> tuple[int, int, list[Failure]]:
    """(sources swept, questions checked, failures sorted by skill, source, seed)."""
    tasks = [(s.id, i, seeds, timeout_s, text_quality) for s in skills for i in range(len(s.sources))]
    questions = 0
    failures: list[Failure] = []
    if workers <= 1:
        _init_worker()
        results = map(_run_task, tasks)
        pool = None
    else:
        pool = multiprocessing.Pool(workers, initializer=_init_worker)
        results = pool.imap_unordered(_run_task, tasks, chunksize=1)
    try:
        for done, (skill_id, _index, checked, found) in enumerate(results, 1):
            questions += checked
            failures.extend(found)
            if progress and (done % 250 == 0 or done == len(tasks)):
                print(f"  {done}/{len(tasks)} sources, {len(failures)} failures so far", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    failures.sort(key=lambda f: (f.skill, f.source, f.seed, f.reason))
    return len(tasks), questions, failures


def render_markdown(
    skills: int, sources: int, questions: int, seeds: int, seconds: float, failures: list[Failure]
) -> str:
    status = "clean" if not failures else f"{len(failures)} failure(s)"
    lines = [
        "## Skill sweep",
        "",
        f"{skills} skills, {sources} sources, {seeds} seeds each: {questions} questions in {seconds / 60:.1f} min. "
        f"Result: **{status}**.",
        "",
    ]
    if failures:
        lines += ["| skill | source | seed | reason |", "|---|---|---|---|"]
        for f in failures[:MAX_SUMMARY_ROWS]:
            reason = f.reason.replace("|", "\\|").replace("`", "'")
            lines.append(f"| `{f.skill}` | `{f.source}` | {f.seed} | {reason} |")
        if len(failures) > MAX_SUMMARY_ROWS:
            lines += ["", f"{len(failures) - MAX_SUMMARY_ROWS} more are in the job log."]
    return "\n".join(lines) + "\n"


def _shard(text: str) -> tuple[int, int]:
    try:
        left, right = text.split("/")
        shard, shards = int(left), int(right)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected I/N such as 0/4, got {text!r}") from None
    if shards < 1 or not 0 <= shard < shards:
        raise argparse.ArgumentTypeError(f"expected 0 <= I < N, got {text!r}")
    return shard, shards


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds 0..N-1 per source (default %(default)s)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes (default: CPU count)")
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per question (default %(default)s)"
    )
    ap.add_argument("--text-quality", action="store_true", help="also fail on text_quality defects (off by default)")
    ap.add_argument("--skills", nargs="+", metavar="ID", help="only these skill ids")
    ap.add_argument("--shard", type=_shard, metavar="I/N", help="only the I-th of N hash-split groups of skills")
    args = ap.parse_args(argv)
    if args.seeds < 1 or args.workers < 1 or args.timeout <= 0:
        ap.error("--seeds, --workers and --timeout must be positive")

    _, all_skills = engine.load_skills()
    if args.skills:
        known = {s.id for s in all_skills}
        unknown = [i for i in args.skills if i not in known]
        if unknown:
            ap.error(f"unknown skill id(s): {', '.join(unknown)}")
    skills = select(all_skills, args.skills, args.shard)
    if not skills:
        print("nothing to sweep: no active skill with sources matched", file=sys.stderr)
        return 1
    if not hasattr(signal, "setitimer"):
        print("warning: this platform has no signal.setitimer, so the per-question time limit is not enforced")

    print(
        f"sweeping {len(skills)} skills, seeds 0..{args.seeds - 1}, {args.workers} worker(s), {args.timeout:g} s limit"
    )
    start = time.perf_counter()
    sources, questions, failures = run(skills, args.seeds, args.workers, args.timeout, args.text_quality, progress=True)
    seconds = time.perf_counter() - start

    print(f"\nswept {len(skills)} skills, {sources} sources, {questions} questions in {seconds / 60:.1f} min")
    if failures:
        print(f"\nFAILURES: {len(failures)}")
        for f in failures:
            print(f"  {f.skill} [{f.source}] seed {f.seed}: {f.reason}")
    else:
        print("failures: []")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(render_markdown(len(skills), sources, questions, args.seeds, seconds, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
