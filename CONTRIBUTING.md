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

If an AI agent wrote part of a change, add `Co-Authored-By: Claude <noreply@anthropic.com>`
as a trailer and end the pull request description with
`Generated with [Claude Code](https://claude.com/claude-code)`. Never add a
session link or a `Claude-Session:` trailer; those are private URLs.

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
   "does not simplify" distractor cannot be built.
5. **Keep it fast.** `python scripts/check_generator_speed.py` fails if the
   median `make()` takes over 250 ms and flags anything over 15 ms.
6. **Add tests** that call the generator across many seeds, grade its own
   answer as correct, and grade each distractor as wrong.

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

## Pull requests

Keep them small and focused. Describe what changed and how you checked it.
CI must pass (lint, tests, secret scan) before merge. If this is your first
pull request here, the maintainer has to approve the CI run before it starts,
so it may wait a while. Do not commit secrets, personal data or third-party
copyrighted material.
