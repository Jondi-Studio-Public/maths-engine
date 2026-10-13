"""CLI: mass-generate questions as JSONL, or make and grade one question for a skill.

Examples:
    python -m mathgen --list
    python -m mathgen --topic quadratic -n 5
    python -m mathgen --level 9 -n 100 --out quadratics.jsonl
    python -m mathgen --all -n 3 --pretty
    python -m mathgen --topic addition -n 1 --seed 42   # reproducible
    python -m mathgen --skills                          # list skill ids
    python -m mathgen --skill simplify-surds --seed 7 --answer "6√21"   # one skill question, graded
"""

import argparse
import contextlib
import difflib
import os
import sys

from . import REGISTRY, __version__, make, topics
from .core import PRESETS, mcq_options

# Always accepted as difficulty names. A topic with no presets runs on its defaults for any of them.
STANDARD_TIERS = ("easy", "medium", "hard")


def _fail(message, code=1):
    """Print `mathgen: error: ...` to stderr and return the exit status."""
    print(f"mathgen: error: {message}", file=sys.stderr)
    return code


def _squash(text):
    """`text` with runs of whitespace collapsed to one space."""
    return " ".join(text.split())


def _positive_int(text):
    """argparse `type=` for a count: a whole number of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _source_problem(skill, source):
    """Why `source` cannot generate (a topic or preset that does not exist, usually a typo), or None."""
    if source.topic not in REGISTRY:
        close = difflib.get_close_matches(source.topic, REGISTRY, n=3)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else " (see --list)"
        return f"skill {skill.id!r} names topic {source.topic!r}, which has no generator{hint}"
    presets = PRESETS.get(source.topic, {})
    if source.difficulty is not None and presets and source.difficulty not in presets:
        known = ", ".join(sorted(presets))
        return f"skill {skill.id!r}: topic {source.topic!r} has no preset {source.difficulty!r}; known: {known}"
    return None


def _labels(options):
    """The labels printed beside a multiple-choice question's options: A, B, C, ...

    A few questions ask which of the lettered statements A to D is true, so their options are the letters
    themselves; a letter label would then name a different option from the text. Those get 1, 2, 3, ... instead.
    """
    letters = [chr(ord("A") + i) for i in range(len(options))]
    texts = {_squash(o).casefold() for o in options}
    if len(options) > 26 or any(letter.casefold() in texts for letter in letters):
        return [str(i + 1) for i in range(len(options))]
    return letters


def _choose_option(options, given):
    """The index of the option `given` names, or None. Tried in this order:

    1. a printed label, exactly ("B", "2"), so a label wins over option text that looks the same;
    2. an option's text, exactly (whitespace collapsed, case kept);
    3. an option's text ignoring case, only if one option matches that way ("dS/ds = bS" and "dS/ds = bs"
       are different options, and a reply that fits both names neither);
    4. a label written loosely ("b", "(B)", "B)").
    """
    labels = _labels(options)
    reply = _squash(given)
    if reply in labels:
        return labels.index(reply)
    exact = [i for i, option in enumerate(options) if _squash(option) == reply]
    if exact:
        return exact[0]
    folded = [i for i, option in enumerate(options) if _squash(option).casefold() == reply.casefold()]
    if folded:
        return folded[0] if len(folded) == 1 else None
    loose = reply.strip("().:").casefold()
    for i, label in enumerate(labels):
        if label.casefold() == loose:
            return i
    return None


def _attach_answer(argv):
    """Let a reply that starts with a minus sign follow `--answer` as its own word: `--answer -3/4`.

    argparse would take "-3/4" for an option; joining it as `--answer=-3/4` makes it a value.
    """
    argv = list(argv)
    for i, word in enumerate(argv[:-1]):
        nxt = argv[i + 1]
        if word == "--answer" and nxt.startswith("-") and not nxt.startswith("--") and nxt not in ("-h", "-n"):
            argv[i : i + 2] = [f"--answer={nxt}"]
            break
    return argv


def _run_skills(ap, args):
    """`--skills` and `--skill`: list the library, or make (and optionally grade) one question."""
    from . import engine
    from .grading import GRADERS
    from .skills import SkillsNotFoundError, load_problems

    try:
        skill_list = engine.load_skills()[1]
    except SkillsNotFoundError as exc:
        return _fail(exc)

    if args.skills:
        for sk in skill_list:
            status = "invalid" if sk.deferred_reason == "invalid" else sk.status
            print(f"{sk.domain}/{sk.id}  {status}")
        problems = load_problems(skill_list)
        if problems:
            lines = "\n".join(f"  {p}" for p in problems)
            return _fail(f"{len(problems)} skill file(s) did not load:\n{lines}")
        return 0

    found = next((sk for sk in skill_list if sk.id == args.skill), None)
    if found is None:
        close = difflib.get_close_matches(args.skill, [sk.id for sk in skill_list], n=3)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else " (use --skills)"
        ap.error(f"unknown skill {args.skill!r}{hint}")
    problems = load_problems([found])
    if problems:
        return _fail(f"skill {found.id!r} did not load: {problems[0]}")
    if found.status == "deferred":
        print(
            f"mathgen: note: skill {found.id!r} is deferred (reason: {found.deferred_reason}), so it is not ready "
            f"for practice; its questions may be incomplete or have no grader",
            file=sys.stderr,
        )
    try:
        source = engine.pick_source(found, args.difficulty)
    except KeyError as exc:
        ap.error(exc.args[0])
    except ValueError as exc:
        return _fail(f"{exc}; there is nothing to generate yet")
    problem = _source_problem(found, source)
    if problem:
        return _fail(problem)
    try:
        q = engine.make_for_skill(found, seed=args.seed, source=source)
    except Exception as exc:  # noqa: BLE001 - a deferred skill may name a generator that does not exist yet
        if found.status != "deferred":
            raise
        return _fail(f"cannot make a question for deferred skill {found.id!r}: {exc}")

    options = None
    answer_text = q.answer_text
    print(f"[{found.id} #{q.seed}]\n  Q: {q.text}")
    if found.answer == "mcq" and not q.parts:
        options = mcq_options(q)
        labels = _labels(options)
        for label, option in zip(labels, options):
            print(f"     {label}) {option}")
        if q.answer_text in options:
            answer_text = f"{labels[options.index(q.answer_text)]}) {q.answer_text}"
    print(f"  A: {answer_text}")

    if args.answer is None:
        return 0
    if found.answer not in GRADERS:
        return _fail(f"skill {found.id!r} cannot be graded yet: there is no grader for answer kind {found.answer!r}")
    given = args.answer
    unmatched = ""
    if options is not None:
        # The mcq grader wants the 0-based index of the chosen option; the CLI takes a letter or the text.
        index = _choose_option(options, args.answer)
        given = "" if index is None else str(index)
        if index is None:
            labels = _labels(options)
            unmatched = f" (not an option label {labels[0]}-{labels[-1]} or option text)"
    verdict = engine.grade(found, q, given)
    print(f"  given {args.answer!r}: {'correct' if verdict.correct else 'wrong'}{unmatched}")
    if verdict.misconception is not None:
        print(f"  likely error ({verdict.misconception}): {verdict.note or 'no description'}")
    if verdict.parts:
        # A multi-part question: the marks, then each part's verdict ("follow-through" when its marks came from
        # the learner's own wrong earlier answer).
        print(f"  marks: {verdict.marks} out of {verdict.max_marks}")
        for label, part in verdict.parts.items():
            how = "follow-through" if part.follow_through else ("correct" if part.correct else "wrong")
            print(f"    ({label}) {part.marks} of {part.max_marks}: {how}")
    return 0


def _run_topics(ap, args):
    """`--topic`, `--level`, `--all`: generate questions for registered topics."""
    if args.difficulty is not None:
        known = sorted({name for presets in PRESETS.values() for name in presets} | set(STANDARD_TIERS))
        if args.difficulty not in known:
            ap.error(f"unknown difficulty {args.difficulty!r}; valid names are {', '.join(known)}")
    if args.topic is not None:
        if args.topic not in REGISTRY:
            close = difflib.get_close_matches(args.topic, REGISTRY, n=3)
            hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else " (use --list)"
            ap.error(f"unknown topic {args.topic!r}{hint}")
        chosen = [args.topic]
    elif args.level is not None:
        chosen = [t for lvl, t in topics() if lvl == args.level]
        if not chosen:
            ap.error(f"no topics at level {args.level} (use --list)")
    else:
        chosen = [t for _lvl, t in topics()]

    if args.difficulty is not None:
        # A topic with presets must have the one asked for; a topic with none uses its defaults.
        skipped = [t for t in chosen if t in PRESETS and args.difficulty not in PRESETS[t]]
        if args.topic and skipped:
            known = ", ".join(sorted(PRESETS[args.topic]))
            ap.error(f"topic {args.topic!r} has no preset {args.difficulty!r}; known: {known}")
        if skipped:
            chosen = [t for t in chosen if t not in skipped]
            print(
                f"mathgen: skipped {len(skipped)} topic(s) with no {args.difficulty!r} preset: {', '.join(skipped)}",
                file=sys.stderr,
            )
        if not chosen:
            ap.error(f"no topic has a {args.difficulty!r} preset")

    to_file = args.out is not None
    try:
        out = open(args.out, "w", encoding="utf-8") if to_file else sys.stdout
    except OSError as exc:
        return _fail(f"cannot write {args.out!r}: {exc.strerror or exc}")
    count = 10 if args.count is None else args.count
    n_written = 0
    failure = None
    try:
        for topic in chosen:
            for i in range(count):
                seed = None if args.seed is None else args.seed + i
                q = make(topic, seed=seed, difficulty=args.difficulty)
                if args.pretty:
                    tag = f" {q.difficulty}" if q.difficulty else ""
                    line = f"[L{q.level} {q.topic} #{q.seed}{tag}]\n  Q: {q.text}\n  A: {q.answer_text}\n"
                    if q.distractors:
                        line += f"  wrong: {' | '.join(q.distractors)}\n"
                    out.write(line)
                else:
                    out.write(q.to_json() + "\n")
                n_written += 1
        out.flush()  # a full disk or a closed pipe shows here, not at interpreter exit
        if to_file:
            out.close()
    except BrokenPipeError:
        raise  # `main` ends the run quietly
    except OSError as exc:
        failure = exc
    finally:
        if to_file:
            with contextlib.suppress(OSError):
                out.close()  # a file that failed to flush or close is closed again; the error is reported below
    if failure is not None:
        where = f"{args.out!r}" if to_file else "standard output"
        return _fail(f"cannot write {where}: {failure.strerror or failure}")
    if to_file:
        print(f"wrote {n_written} questions to {args.out}", file=sys.stderr)
    return 0


def _run(argv):
    # Windows consoles default to cp1252; our questions use 00d7, 00f7, 03b8, superscripts.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="mathgen", description="Generate maths questions with verified answers.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--topic", help="topic name (see --list)")
    g.add_argument("--level", type=int, help="generate for every topic at this level")
    g.add_argument("--all", action="store_true", help="generate for every registered topic")
    g.add_argument("--list", action="store_true", help="list topics and exit")
    g.add_argument(
        "--skills",
        action="store_true",
        help="list skill ids (domain/id, status) and exit; exits non-zero if a skill file does not load",
    )
    g.add_argument("--skill", help="skill id: make one question for it (see --skills); --answer grades a reply")
    ap.add_argument("-n", "--count", type=_positive_int, help="questions per topic (default 10)")
    ap.add_argument("--seed", type=int, help="base seed for reproducible output")
    ap.add_argument(
        "--difficulty",
        help="difficulty preset, usually easy, medium or hard (some topics also have exam); topics without "
        "presets use defaults; with --level or --all, topics that have presets but not this one are skipped; "
        "a name that no topic has is an error that lists the valid ones",
    )
    ap.add_argument(
        "--answer",
        help="with --skill: grade this answer against the question and print the verdict; "
        "for a multiple-choice question give the option's label (A, B, ... or 1, 2, ...) or its text",
    )
    ap.add_argument("--out", help="output file (JSONL); default stdout")
    ap.add_argument("--pretty", action="store_true", help="human-readable output instead of JSONL")
    ap.add_argument("--version", action="version", version=f"mathgen {__version__}")
    args = ap.parse_args(_attach_answer(sys.argv[1:] if argv is None else argv))

    if args.answer is not None and args.skill is None:
        ap.error("--answer needs --skill")
    if args.out == "":
        ap.error("--out needs a file name")
    mode = "--skill" if args.skill is not None else "--skills" if args.skills else "--list" if args.list else None
    if mode:  # these commands print to the terminal; the options below only shape bulk generation
        for flag, given in (
            ("--out", args.out is not None),
            ("-n/--count", args.count is not None),
            ("--pretty", args.pretty),
        ):
            if given:
                ap.error(f"{flag} does not apply to {mode}; it is for --topic, --level and --all")

    if args.list:
        for lvl, t in topics():
            print(f"{lvl:>2}  {t}")
        return 0

    if args.skills or args.skill is not None:
        return _run_skills(ap, args)
    return _run_topics(ap, args)


def main(argv=None):
    try:
        return _run(argv)
    except BrokenPipeError:
        # The reader went away (`mathgen --skills | head`). Point stdout at /dev/null so the interpreter's
        # final flush does not raise again, and stop quietly.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
