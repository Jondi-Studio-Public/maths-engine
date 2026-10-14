# Contributing

Thanks for helping. This package is small on purpose: question generators,
the graders that check answers to them, and the skill definitions that tie
the two together.

## Licence and sign-off (DCO)

There is no contributor licence agreement. Contributions are accepted under
the licences already in the repository: MIT for code (`LICENSE`) and CC BY 4.0
for skill YAML (`LICENSE-CONTENT`). You keep your copyright.

Every commit must carry a Developer Certificate of Origin sign-off, which says
you wrote the change or have the right to submit it under these licences
(https://developercertificate.org/). Add it with:

```
git commit -s -m "Add fraction-of-quantity generator"
```

That appends `Signed-off-by: Your Name <you@example.com>`. CI fails the pull
request if any commit in it is unsigned. A merge commit needs a sign-off too
(`git merge --signoff`); or rebase your branch instead of merging into it.

Do not submit text copied from textbooks, exam papers or other sources you
do not have the right to relicense. Questions should be original templates.
Text under a licence that requires attribution (CC BY, for example) is
allowed only with that attribution added in the same pull request; `NOTICE`
shows how it is done for Australian Curriculum text.

## Setup

Python 3.12 or newer.

```
git clone https://github.com/Jondi-Studio-Public/maths-engine.git   # or your fork's URL
cd maths-engine
python3 -m venv .venv
. .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Running the tests

Name the files you touched and use several workers:

```
python -m pytest tests/test_my_generator.py -q -n 8
```

`tests/public/` is a small, fast set (`python -m pytest tests/public -q`) that
covers the command line, skill loading, packaging and the public API. The whole
suite has about 90,000 tests, so expect tens of minutes on four cores. Run it once
before opening a pull request, or let CI do it:

```
python -m pytest tests -q -n auto
```

Then format and lint with the version CI uses (see `.github/workflows/ci.yml`):

```
pip install ruff==0.16.10
ruff format . && ruff check .
```

## Adding a generator

A generator is one decorated function in `mathgen/generators/`. Look at a
neighbouring file and at `mathgen/README.md` for the registration decorator
and the shape of what `make()` returns.

1. **Read `mathgen/core.py` first** and reuse its helpers. Do not hand-roll
   equivalents. The ones you will most often need are `check_equivalent`,
   `check_distinct`, `join_terms`, `paren_negative`, `pick_distractors`,
   `rational_text` and `fmt_signed`. If a helper you need is missing and
   would help more than one generator, say so in the pull request rather
   than writing a private copy.
2. **Be deterministic.** All randomness must come from the seeded random
   source passed in. The same seed must give the same question and answer.
   Never use the global `random` module, the clock or the environment.
3. **Verify the answer independently.** Compute it a second way (SymPy or a
   direct calculation) inside the generator, not just once.
4. **Make distractors clear the grader.** Every wrong option or
   deliberately-wrong test value must be rejected by the grader, checked on
   every candidate and not only on a sample. Typed numeric answers are graded
   to their own stated precision. Fraction answers are compared by value, so a
   "does not simplify" distractor cannot be built (unless the stem asks for lowest terms or simplest form, where
   an unreduced fraction or a decimal is itself wrong).
5. **Keep it fast.** `python scripts/check_generator_speed.py` times the median of 8
   `make()` calls for every topic's bare default and for every distinct (topic,
   difficulty, params) a skill source uses. It fails if a median is over 250 ms, if
   a call raises or runs for over 5 s, and flags anything over 15 ms.
6. **Add tests** that call the generator across many seeds, grade its own
   answer as correct, and grade each distractor as wrong.
7. **Never write an uncapped loop.** A `while True:` that redraws until a
   condition holds, or a `while a == b: b = rng.randint(...)`, can spin forever on an
   unlucky seed. Use `core.draw_until(rng, draw, accept, tries=1000, what="...")`, or
   `for _ in core.attempts(1000, "..."):` with a `break` when a loop draws several values.
   Both raise `VerificationError` naming what failed when the budget runs out, and both
   consume `rng` exactly as the `while` loop would, so seeds keep their questions.
   `tests/test_no_unbounded_loops.py` fails on a new uncapped loop in
   `mathgen/generators/`; the loops that predate it are in
   `tests/unbounded_loops_allowlist.yaml`, which only shrinks. When you convert one, run
   `python scripts/list_unbounded_loops.py --write` and commit the shorter list.

## Adding or editing a skill

One YAML file per skill under `skills/<domain>/<id>.yaml`. See
`skills/README.md` for the fields. Run `python -m mathgen --skills` to check
your file loads: a file that is not valid YAML, has no `id`, or has an `id` or
`domain` that does not match its file name or folder is reported by path and the
command exits non-zero. That check does not generate questions. Run
`python -m mathgen --skill <id> --seed 1` to see one, and
`python scripts/regrade_sweep.py --ids <id>` to check its graders against its
questions. `skills/README.md` shows how to run the full validator over the
library. Difficulty progression should be genuine reasoning escalation, not
just bigger numbers.

If a skill's name or description adapts Australian Curriculum Senior Secondary
text, its id starts with `senior-` and its file starts with the same two
comment lines as every other `senior-*` file (copy them from any of those);
`tests/public/test_skill_headers.py` checks this. For Australian Curriculum
text from anywhere else, ask in the pull request how to credit it.

## Seed stability

A question is named by a question id, `<skill id>/<source key>/<seed>/<hash>`
(`mathgen/engine.py`), and `tests/golden/questions.txt` records the id of
seeds 0, 1 and 2 of every active source. Changing a generator, a source's
`params`, `difficulty` or `key`, or a helper in `mathgen/core.py` that
generators use, changes the hashes of the questions it touches. `tests/test_golden_questions.py`
fails and lists them. When the change is intended:

```
python scripts/golden_questions.py --write
```

commit the regenerated file (never edit it by hand) and say in the pull
request which skills' existing questions changed and why. A change that should
not alter existing questions (a new generator, a new source, a fix that only
affects other seeds) leaves the old lines untouched, so the diff shows only
added lines. Reordering a skill's sources does not change ids; renaming a
source's `key` does. Two sources of one skill with the same `difficulty` need
distinct `key`s (`validate()` reports a repeat). The check takes about 30 s
(`python scripts/golden_questions.py --check`).

## Finding a generator, and the nightly sweep

`skills/index.yaml` lists, for every skill, the module that registers each of its source
topics. It is generated: after adding or renaming a skill, a source topic or a generator
module, run `python scripts/build_skill_index.py --write` (a test fails while it is stale).
`python scripts/build_skill_index.py --where <skill-or-topic>` prints the file and line.

A scheduled workflow (`.github/workflows/nightly.yml`) runs
`python scripts/sweep_skills.py`: seeds 0 to 39 of every source of every active skill, each
checked for an exception, a timeout of over 10 s, an answer that does not grade correct and
a distractor that does. Run it locally with `--skills <id> --seeds 5`; add `--text-quality`
to apply `mathgen/text_quality.py` too.

## Pull requests

Keep them small and focused. Describe what changed and how you checked it.
CI must pass (lint, tests, secret scan) before merge. If this is your first
pull request here, the maintainer has to approve the CI run before it starts,
so it may wait a while. Do not commit secrets, personal data or third-party
copyrighted material.
