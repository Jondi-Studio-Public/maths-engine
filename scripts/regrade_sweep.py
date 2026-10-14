"""Read-only generate-and-regrade sweep with a negative control.

For any generated question that carries a `figure`, this is also the
`figure_stimulus` capability gate's per-candidate plausibility check: every
such question's figure spec must pass `figures.validate` with no problems,
must not leak its own answer text in a label or in the alt text
(`figures.assert_no_leak`), and, when its topic has registered a
plausibility reader (`figures.register_reader`), the answer that reader
re-derives from the spec alone must grade correct against the real key with
the live grader. This runs for every candidate of every seed, exactly like
the grading checks above, and is included whether the sweep was asked for
by `--ids`, `--domain`, `--all-active` or `--since` -- selection by explicit
`--ids` also reaches a `status: deferred` skill (see `select_skills`), which
is how a figure-bearing skill still awaiting the gate gets swept.

Usage:
    python scripts/regrade_sweep.py --ids <id> [<id> ...] [--seeds N]
    python scripts/regrade_sweep.py --domain <domain> [--seeds N]
    python scripts/regrade_sweep.py --all-active [--seeds N]
    python scripts/regrade_sweep.py --since <git-ref> [--seeds N]

Exit 1 if any false reject, false accept, figure-plausibility failure, or
generator exception occurred, or if there was nothing to sweep; else 0. An id
that is not a skill, or a domain with no skills, is a usage error (exit 2).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgen.grading.typed import GRADERS, mcq_options  # noqa: E402
from mathgen.skills import Skill, load_skills  # noqa: E402
from mathgen import figures, make  # noqa: E402
from mathgen.core import PRESETS  # noqa: E402

DEFAULT_SEEDS = 40
DIFFICULTY_TIERS = ("easy", "medium", "hard")

_NUMERIC_LITERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _tuplify(params: dict) -> dict:
    """Mirrors `mathgen.skills._tuplify` -- copied rather than imported so this read-only script does not
    depend on that private helper."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()}


def _difficulties_for(topic: str, explicit: str | None) -> list[str | None]:
    """The source's own `difficulty` when set; else each of easy/medium/hard
    the topic actually declares a preset for; else `[None]` (the topic's
    default, no preset at all)."""
    if explicit:
        return [explicit]
    supported = [d for d in DIFFICULTY_TIERS if d in PRESETS.get(topic, {})]
    return supported or [None]


def _perturb_last_decimal_place(text: str) -> str | None:
    """Perturb the *last* numeric literal in `text` by one unit in its own
    last stated decimal place -- exactly the near miss that the stated-precision
    tolerance must reject (see `mathgen/grading/typed.py::_typed_number_tolerance`). The last
    literal (not the first) is used because a labelled answer_text such as
    `f"f({n}) = {ans}"` states the function argument before the answer;
    perturbing the trailing literal is what a learner's wrong *answer*
    looks like, not a wrong function argument. Returns None if `text`
    contains no numeric literal at all (nothing to perturb)."""
    matches = list(_NUMERIC_LITERAL_RE.finditer(text or ""))
    if not matches:
        return None
    match = matches[-1]
    literal = match.group()
    sign = ""
    body = literal
    if body.startswith("-"):
        sign = "-"
        body = body[1:]
    if "." in body:
        int_part, frac_part = body.split(".", 1)
        places = len(frac_part)
        scaled = int(int_part + frac_part)
    else:
        places = 0
        scaled = int(body)
    scaled += 1
    if places:
        digits = str(scaled).zfill(places + 1)
        new_body = f"{digits[:-places]}.{digits[-places:]}"
    else:
        new_body = str(scaled)
    new_literal = sign + new_body
    return text[: match.start()] + new_literal + text[match.end() :]


def _tuple_components(text: str) -> list[str] | None:
    """The components of a parenthesised tuple such as `(a, f(b, c))`, or None when `text` is not one.
    `(a)(b)` and `(a) + (b, c)` are not tuples: the opening bracket must be the one the last bracket closes."""
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    parts, depth, start = [], 0, 1
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return None
        elif char == "," and depth == 1:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:-1].strip())
    return parts if len(parts) > 1 else None


def _bump(value):
    """Return a value that reliably differs from `value` for a synthetic
    wrong answer: `value + 1` for anything numeric-like (`int`, `float`,
    `fractions.Fraction`, a `sympy`/other numeric type supporting `+`), or a
    distinguishing string suffix otherwise. Deliberately tries the
    arithmetic path first rather than gating on `isinstance(value, (int,
    float))` -- a `Fraction` (e.g. `jacobi_iteration_2x2`'s `{"x": Fraction,
    "y": Fraction}` answer) is not an `int`/`float` instance but supports
    `+ 1` perfectly well; gating on that isinstance check produced a
    `"1_x"`-shaped garbled value whose embedded numeral `extract_values`
    would still parse identically to the *unperturbed* value, silently
    defeating the whole synthetic-wrong-answer check (found empirically
    while building this sweep, not anticipated)."""
    try:
        return value + 1
    except TypeError:
        return f"{value}_x"


def _synthetic_wrong_given(skill: Skill, question) -> str | None:
    """The synthetic deliberately-wrong given text for one question, keyed
    by the skill's grader kind (per-kind construction rules).
    `ordered_list` is handled separately (`_ordered_list_synthetics`) since
    it needs *two* synthetic variants, not one. Returns None when this
    kind's answer shape doesn't match what synthesis needs (skipped, not a
    failure -- e.g. a non-{p,q} `fraction` answer already falls through to
    `_check_answer_kind` in the real grader and isn't this script's concern)."""
    kind = skill.answer
    if kind == "number":
        return _perturb_last_decimal_place(question.answer_text)
    if kind == "fraction":
        answer = question.answer
        if isinstance(answer, dict) and {"p", "q"} <= set(answer):
            wrong = Fraction(int(answer["p"]) + 1, int(answer["q"]))
            return str(wrong.numerator) if wrong.denominator == 1 else f"{wrong.numerator}/{wrong.denominator}"
        return None
    if kind == "expression":
        # `(a, b)+1` adds a number to a tuple, which SymPy deprecates; change the first component instead.
        parts = _tuple_components(question.answer_text)
        if parts:
            return "(" + ", ".join([f"({parts[0]})+1", *parts[1:]]) + ")"
        return f"{question.answer_text}+1"
    if kind in ("pair", "list"):
        answer = question.answer
        if isinstance(answer, dict):
            values = list(answer.values())
        elif isinstance(answer, list):
            values = list(answer)
        else:
            return None
        if not values:
            return None
        values[0] = _bump(values[0])
        return ", ".join(str(v) for v in values)
    return None  # ordered_list, mcq: handled by their own dedicated paths


def _ordered_list_synthetics(question) -> list[tuple[str, str]]:
    """`[(label, given), ...]` for `ordered_list`: the first element changed
    (always), plus the fully-reversed order -- but only when reversing
    actually differs from the expected order (a length-<=1 or palindromic
    answer would otherwise generate a spurious "wrong" answer that is
    trivially identical to the right one)."""
    expected = question.answer
    if not isinstance(expected, list) or not expected:
        return []
    out: list[tuple[str, str]] = []

    changed = list(expected)
    changed[0] = _bump(changed[0])
    out.append(("first_element_changed", ", ".join(str(v) for v in changed)))

    reversed_items = list(reversed(expected))
    if reversed_items != expected:
        out.append(("reversed_order", ", ".join(str(v) for v in reversed_items)))
    return out


def _figure_answer_texts(question) -> list[str]:
    """Text forms of `question`'s own correct answer, in whatever shapes a
    learner would type or that appear as a labelled key -- passed to
    `figures.assert_no_leak`. Never includes distractors: those are wrong
    answers, so their appearing in a label is not a leak."""
    texts = [question.answer_text]
    if question.answer_alt is not None:
        texts.append(str(question.answer_alt))
    return texts


def _check_figure_plausibility(skill: Skill, question, grader) -> str | None:
    """One question's figure plausibility check: `figures.validate`, the
    no-leak rule, and (if the topic registered one) the reader's re-derived
    answer against the real grader. Returns a failure message, or None when
    the figure is clean. Never raises: `assert_no_leak` raises
    AssertionError by design, which this turns into a message like every
    other failure kind the sweep collects."""
    spec = question.figure
    problems = figures.validate(spec)
    if problems:
        return "invalid figure: " + "; ".join(problems)

    try:
        figures.assert_no_leak(spec, _figure_answer_texts(question))
    except AssertionError as exc:
        return str(exc)

    reader = figures.get_reader(question.topic)
    if reader is None:
        return None
    try:
        reread = reader(question)
    except Exception as exc:  # noqa: BLE001 - a reader failing IS a plausibility failure
        return f"figure reader raised: {exc}"
    if skill.answer == "mcq":
        # grade_mcq takes the option INDEX; a reader returns the option text.
        options = mcq_options(question)
        if reread not in options:
            return f"figure reader derived {reread!r}, which is not one of the mcq options"
        reread = str(options.index(reread))
    result = grader(question, reread)
    if not result.correct:
        return f"figure reader derived {reread!r}, which does not grade correct against the key"
    return None


def _sweep_parts(question, grader, result, where) -> None:
    """The sweep for a multi-part question (`question.parts`), which every grader marks part by part.

    The labelled `answer_text` must earn full marks. Then, one part at a time with every other part answered
    correctly, each of that part's distractors, and (for a number part with no follow-through) its answer moved
    by one unit in its last stated place, must not be accepted for that part. A follow-through part is not
    perturbed: one unit away can be exactly what the learner's correctly rounded earlier answer gives."""
    positive = grader(question, question.answer_text)
    if positive.correct:
        result.accepts += 1
    else:
        result.false_rejects.append((*where, question.answer_text, question.answer_text))
    canonical = {part.label: part.answer_text for part in question.parts}
    for part in question.parts:
        wrong = list(part.distractors)
        if part.answer_kind == "number" and not part.follow_through:
            nudged = _perturb_last_decimal_place(part.answer_text)
            if nudged is not None:
                wrong.append(nudged)
        for given in wrong:
            marked = grader(question, {**canonical, part.label: given})
            if marked.parts[part.label].correct:
                result.false_accepts.append((*where, f"({part.label}) {part.answer_text}", f"({part.label}) {given}"))


@dataclass
class SkillSweepResult:
    skill_id: str
    questions: int = 0
    accepts: int = 0
    false_rejects: list = field(default_factory=list)
    false_accepts: list = field(default_factory=list)
    exceptions: list = field(default_factory=list)
    figure_checks: int = 0
    figure_failures: list = field(default_factory=list)  # (kind, topic, difficulty, seed, message)

    @property
    def clean(self) -> bool:
        return not (self.false_rejects or self.false_accepts or self.exceptions or self.figure_failures)


def sweep_skill(skill: Skill, seeds: int = DEFAULT_SEEDS, graders: dict | None = None) -> SkillSweepResult:
    """Run the full positive+negative sweep for one skill's every source and
    every difficulty it declares, across `seeds` seeds each."""
    graders = graders if graders is not None else GRADERS
    result = SkillSweepResult(skill_id=skill.id)
    grader = graders.get(skill.answer)
    if grader is None:
        return result  # unregistered answer kind: nothing this tool can grade

    for src in skill.sources:
        params = _tuplify(src.params)
        for difficulty in _difficulties_for(src.topic, src.difficulty):
            for seed in range(seeds):
                try:
                    question = make(src.topic, seed=seed, difficulty=difficulty, **params)
                except Exception as exc:  # noqa: BLE001 - a generator exception is itself a failure
                    result.exceptions.append((src.topic, difficulty, seed, str(exc)))
                    continue

                result.questions += 1

                if question.figure is not None:
                    result.figure_checks += 1
                    failure = _check_figure_plausibility(skill, question, grader)
                    if failure is not None:
                        result.figure_failures.append(
                            (question.figure.get("kind"), src.topic, difficulty, seed, failure)
                        )

                if question.parts:
                    _sweep_parts(question, grader, result, (src.topic, difficulty, seed))
                    continue

                if skill.answer == "mcq":
                    options = mcq_options(question)
                    try:
                        correct_idx = options.index(question.answer_text)
                    except ValueError:
                        result.false_rejects.append(
                            (src.topic, difficulty, seed, question.answer_text, "<answer_text not in mcq_options>")
                        )
                        continue
                    positive = grader(question, correct_idx)
                    if positive.correct:
                        result.accepts += 1
                    else:
                        result.false_rejects.append(
                            (src.topic, difficulty, seed, question.answer_text, str(correct_idx))
                        )
                    for idx in range(len(options)):
                        if idx == correct_idx:
                            continue
                        negative = grader(question, idx)
                        if negative.correct:
                            result.false_accepts.append((src.topic, difficulty, seed, question.answer_text, str(idx)))
                    continue

                positive = grader(question, question.answer_text)
                if positive.correct:
                    result.accepts += 1
                else:
                    result.false_rejects.append(
                        (src.topic, difficulty, seed, question.answer_text, question.answer_text)
                    )

                for distractor in question.distractors:
                    negative = grader(question, distractor)
                    if negative.correct:
                        result.false_accepts.append((src.topic, difficulty, seed, question.answer_text, distractor))

                if skill.answer == "ordered_list":
                    for _label, given in _ordered_list_synthetics(question):
                        negative = grader(question, given)
                        if negative.correct:
                            result.false_accepts.append((src.topic, difficulty, seed, question.answer_text, given))
                else:
                    synthetic = _synthetic_wrong_given(skill, question)
                    if synthetic is not None:
                        negative = grader(question, synthetic)
                        if negative.correct:
                            result.false_accepts.append((src.topic, difficulty, seed, question.answer_text, synthetic))

    return result


def sweep(skills: list[Skill], seeds: int = DEFAULT_SEEDS, graders: dict | None = None) -> dict[str, SkillSweepResult]:
    return {skill.id: sweep_skill(skill, seeds=seeds, graders=graders) for skill in skills}


def _added_skill_ids_since(ref: str) -> set[str]:
    """Skill ids whose YAML file was added (git status `A`) under `skills/`
    in any commit reachable from `HEAD` but not from `ref`. A skill's id is
    taken from its filename stem, matching `mathgen.skills.validate`'s own
    id-equals-filename-stem rule -- never re-parsed from the diff content."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-status", f"{ref}..HEAD", "--", "skills/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise ValueError("--since needs git, which is not installed") from None
    except subprocess.CalledProcessError as exc:
        reason = (exc.stderr or "").strip().splitlines()
        raise ValueError(
            f"--since {ref!r} needs a git checkout and a ref it knows; git said: "
            f"{reason[-1] if reason else f'exit status {exc.returncode}'}"
        ) from None
    ids: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].startswith("A"):
            continue
        path = parts[-1]
        if path.endswith(".yaml"):
            ids.add(Path(path).stem)
    return ids


def select_skills(
    skills: list[Skill],
    ids: list[str] | None = None,
    domain: str | None = None,
    all_active: bool = False,
    since: str | None = None,
) -> list[Skill]:
    by_id = {s.id: s for s in skills}
    if ids:
        return [by_id[i] for i in ids if i in by_id]
    if domain:
        return [s for s in skills if s.domain == domain]
    if all_active:
        return [s for s in skills if s.status == "active"]
    if since:
        wanted = _added_skill_ids_since(since)
        return [s for s in skills if s.id in wanted]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+", metavar="ID")
    group.add_argument("--domain", metavar="DOMAIN")
    group.add_argument("--all-active", action="store_true")
    group.add_argument("--since", metavar="GIT_REF")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    _, skills = load_skills()
    if args.ids:
        unknown = [i for i in args.ids if i not in {s.id for s in skills}]
        if unknown:
            parser.error(f"unknown skill id(s): {', '.join(unknown)}")
    elif args.domain and not any(s.domain == args.domain for s in skills):
        parser.error(
            f"no skills in domain {args.domain!r}; domains with skills: {', '.join(sorted({s.domain for s in skills}))}"
        )
    try:
        selected = select_skills(
            skills,
            ids=args.ids,
            domain=args.domain,
            all_active=args.all_active,
            since=args.since,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for s in selected:
        if s.answer not in GRADERS:
            print(f"skipped {s.id}: no grader for answer kind {s.answer!r}", file=sys.stderr)
    selected = [s for s in selected if s.answer in GRADERS]

    if not selected:
        print("SWEEP: 0 skills, 0 questions, 0 false rejects, 0 false accepts")
        # No new skills since a ref is a normal answer; an explicit selection that sweeps nothing is a mistake.
        return 0 if args.since else 1

    results = sweep(selected, seeds=args.seeds)

    total_questions = 0
    total_false_rejects = 0
    total_false_accepts = 0
    total_figure_checks = 0
    total_figure_failures = 0
    figure_failures_by_kind: Counter = Counter()
    any_failure = False

    for skill_id in sorted(results):
        r = results[skill_id]
        total_questions += r.questions
        total_false_rejects += len(r.false_rejects)
        total_false_accepts += len(r.false_accepts)
        total_figure_checks += r.figure_checks
        total_figure_failures += len(r.figure_failures)
        for kind, *_rest in r.figure_failures:
            figure_failures_by_kind[kind] += 1
        status = "OK" if r.clean else "FAIL"
        print(
            f"{status} {skill_id}: questions={r.questions} accepts={r.accepts} "
            f"false_rejects={len(r.false_rejects)} false_accepts={len(r.false_accepts)} "
            f"exceptions={len(r.exceptions)} figure_checks={r.figure_checks} "
            f"figure_failures={len(r.figure_failures)}"
        )
        for kind_label, items in (
            ("false_reject", r.false_rejects),
            ("false_accept", r.false_accepts),
            ("exception", r.exceptions),
            ("figure_failure", r.figure_failures),
        ):
            for item in items[:3]:
                print(f"    {kind_label}: {item}")
        if not r.clean:
            any_failure = True

    if total_figure_checks:
        by_kind = ", ".join(f"{k}={v}" for k, v in sorted(figure_failures_by_kind.items())) or "none"
        print(
            f"SWEEP FIGURES: {total_figure_checks} figure candidates checked, "
            f"{total_figure_failures} failures ({by_kind})"
        )

    print(
        f"SWEEP: {len(results)} skills, {total_questions} questions, "
        f"{total_false_rejects} false rejects, {total_false_accepts} false accepts"
    )
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
