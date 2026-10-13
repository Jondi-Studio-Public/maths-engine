# mathgen package

The question generators, the shared helpers they are built from, and the
graders that check replies. For install and a first run, see the top-level
`README.md`. This file is for people writing or changing a generator.

## Layout

| Path | What it is |
|---|---|
| `core.py` | `Question`, the `@generator` decorator, `make()`, and the shared helpers |
| `generators/` | The generators, one decorated function each, in about 1,000 modules |
| `grading/typed.py` | The graders and the `GRADERS` registry (`number`, `fraction`, `pair`, `list`, `ordered_list`, `expression`, `mcq`, and a few structured kinds) |
| `grading/spoken.py` | Number and expression parsing the typed graders share, and `check_answer` for transcripts of spoken answers. For a single-number answer a transcript must state one value and not negate or hedge it ("not twelve", "twelve or thirteen" and "twelve and a half" are wrong for 12) |
| `answers.py` | The typed answer contract: `answer_spec()`, `Distractor`, the answer kinds and forms |
| `misconceptions.py` | The shared catalogue of misconception ids that distractors are tagged with |
| `engine.py` | Public API over skills: `load_skills()`, `make_for_skill()`, `grade()` |
| `skills.py` | Skill loader and validator |
| `figures.py` | Figure specs for questions that show a picture |
| `text_quality.py` | Learner-facing text checks (no `**`, no `*` as a times sign, no "1 parts") |
| `capabilities.py` | Capability-gate registry loader: blocks skills whose gates have not passed |
| `slots.py` | Shared word-problem vocabulary (names, objects, units) drawn through `core.phrase` |
| `figure_kinds/` | The figure kinds `figures.py` builds, by area |
| `scheduling/` | Spaced-review maths: FSRS, stages, prerequisites, calibration |

For how these fit together, see `docs/architecture.md`.

Generator file names carry historical batch prefixes (such as `p19h_`). They
are identifiers, not categories. Dropping a module into `generators/` is enough:
the package imports everything in that folder.

## Running generators

```
python -m mathgen --list                                  # every topic, with its level
python -m mathgen --topic quadratic -n 5 --pretty
python -m mathgen --topic addition -n 3 --seed 42         # reproducible
python -m mathgen --level 9 -n 100 --out questions.jsonl
python -m mathgen --all -n 50 --difficulty easy --out bank.jsonl
```

With `--level` or `--all`, a topic that has presets but not the one you named is skipped, and
the command says which on stderr (four topics have no `easy`, for instance). `--topic` with a
preset the topic lacks is an error that lists the presets it has, and a difficulty name that no
topic has (a typo) is an error that lists the valid names.

From Python:

```python
from mathgen import make

q = make("quadratic", seed=5)  # same (topic, seed) gives the same question
q = make("quadratic", difficulty="hard")  # a named preset
q = make("addition", lo=100, hi=999)  # raw params override a preset
q.distractors  # wrong answers, ready for multiple choice
```

`level` is a coarse band used by the `--level` filter. It does not decide where a
skill uses a topic.

## Writing a generator

A generator is one function that takes a seeded `random.Random` plus keyword
parameters and returns a dict. The decorator registers it:

```python
from ..core import check_distinct, check_equivalent, generator, pick_distractors


@generator("my_topic", level=21, presets={"easy": dict(hi=9), "hard": dict(hi=99)})
def my_topic(rng, hi=20):
    # 1. Pick the answer first, with rng.
    # 2. Build the question backwards from it.
    # 3. If the answer was derived (solved, integrated, inverted), check it a
    #    second way with check_equivalent(...).
    # 4. Build distractors from specific misconceptions.
    return dict(text=..., latex=..., answer=..., answer_text=..., distractors=...)
```

Required keys are `text`, `latex`, `answer` and `answer_text`. Optional keys include
`answer_latex`, `working` (worked steps), `distractors`, `answer_alt`, `option_notes`,
`answer_form`, `answer_kind` and `figure`; see the `Question` dataclass in `core.py` for what each means.
If you leave `latex` empty or copy `text` into it, `make()` fills it in with
`latex_prompt()`. If you write your own `latex`, you are responsible for its escaping.

**Determinism.** Use only the `rng` you are given. The same `(topic, difficulty, seed)`
must give the same question on any machine. Never use the global `random` module, the
clock or the environment.

**Build backwards.** Choose clean roots, values or solutions first, then construct the
question from them. The answer is clean by construction and nothing has to be solved at
generation time.

**Check it a second way.** Where the answer is derived rather than chosen, recompute it
by a different route (SymPy solve, integrate, substitute back) and fail loudly on a
mismatch with `check_equivalent(actual, expected)` or `check(cond, msg)`, which raise
`VerificationError`. Where the question is a single step from a chosen answer, a check
would only restate the construction, so skip it.

**Distractors the grader rejects.** Each distractor is what a learner making one specific
error would compute (a carry slip, a sign flip, an inverted ratio, a dropped constant).
Collisions with the answer are the main hazard, so:

- Run `pick_distractors(answer_text, candidates)`; it drops candidates equal in value to the
  answer, not only in spelling. Pass `unordered=True` when the grader accepts any order, and
  `tolerance=` when the grader compares numbers with a tolerance.
- For symbolic answers, call `check_distinct(answer, *candidates)` on every candidate, and
  `check_pairwise_distinct(...)` if two distractors might equal each other.
- A question may end up with fewer than three distractors after filtering. That is fine.
- If a misconception is algebraically identical to the answer for some parameter value,
  exclude that value when you design the generator.

**Tag each distractor with its error.** Return `mathgen.answers.Distractor(text, "<id>", value)` instead of a bare
string: the id comes from the shared catalogue in `misconceptions.py`, and `value` (shaped like `answer`) lets the
grader recognise that error however the learner spells it, so a wrong reply is diagnosed, not just marked. Set
`answer_kind` too. See `docs/design/typed-answer-contract.md`; plain strings still work.

Typed numeric answers are graded to their own stated precision: a rounded answer (`0.33` for 1/3)
accepts half a unit in its last place, and an answer shown exactly (`2.6` for 7.7 - 5.1) is matched
exactly. If the stored answer is itself rounded, say so in the question ("to the nearest cent").
`grade_fraction` compares
by value, so `2/4` is accepted for `1/2` and you cannot build a "does not simplify"
distractor. Expressions are compared by symbolic equivalence.

**Presets.** `presets` maps names such as `easy`, `medium` and `hard` to keyword bundles.
Explicit params passed to `make()` override a preset. A topic with no presets uses its
defaults for any difficulty name, and an unknown name for a topic that has presets raises
`KeyError`. Some topics have only an `exam` preset, or presets named for their own cases.

## Multi-part questions

An exam question in parts ("(a) find the radius; (b) hence find the circumference") returns `text` as the
stem and a list of `parts` instead of a single answer. Each part names its own grader (`answer_kind`), its
`marks`, and, when it uses an earlier result, a `follow_through` expression such as `"2*pi*part_a"`, so a
learner who gets (a) wrong still earns (b)'s marks for using their own (a) correctly:

```python
return dict(
    text="A circle has an area of 120.76 cm².",
    parts=[
        dict(
            prompt="Find the radius, correct to 1 decimal place.",
            answer_kind="number",
            answer=6.2,
            answer_text="6.2",
            marks=2,
            distractors=[...],
        ),
        dict(
            prompt="Hence find the circumference, correct to 1 decimal place.",
            answer_kind="number",
            answer=39.0,
            answer_text="39.0",
            follow_through="2*pi*part_a",
            distractors=[...],
        ),
    ],
)
```

`make()` fills the question-level `answer`, `answer_text` and `working` from the parts and checks every
follow-through expression against the canonical answers. `grade()` takes a dict of replies keyed by label (or
labelled text) and returns `marks`, `max_marks` and a result per part. Give a part that later parts follow
through from a `sensible_range` when its answer has one, such as `(0, None)` for a length or `(0, 1)` for a
probability, so an impossible earlier reply carries no marks forward. The design, including show-that parts,
is in `docs/design/multi-part-questions.md`.

Labelled text may mark each part as `(a)`, `a)`, `a:`, `a=` or `a.` (then a space), in any case, with whatever
labels the parts use (`a`/`b`/`c`, `i`/`ii`/`iii`, `1`/`2`/`3`); a decimal point or a variable inside an answer is
not a marker. Unlabelled text is split by `;` or line breaks when it has one piece per part. Text that cannot be
split scores 0 and says so: the result's `note` explains how to label parts and its `note_kind` is
`"unreadable_parts"`, so a front end can show it instead of a bare 0.

## Use the helpers in `core.py`

Read `core.py` before writing any utility. It already has, among others:

| Helper | Use |
|---|---|
| `check`, `check_equivalent`, `check_distinct`, `check_pairwise_distinct` | Verification; raise `VerificationError` |
| `pick_distractors` | Dedupe and filter misconception candidates |
| `fmt_signed`, `paren_negative`, `join_terms`, `rational_text` | Render signs, brackets, sums of terms and fractions |
| `latex_answer`, `latex_prompt` | LaTeX for `answer_latex` and for prompts |
| `numeric_answer`, `stated_precision_string`, `round_half_up` | Decimal answers at a stated precision |
| `phrase`, `compose` | Vary wording; chain multi-step questions |
| `mcq_options` | The shuffled option list the `mcq` grader uses |
| `draw_until`, `attempts` | Bounded rejection sampling; never write an uncapped `while` loop (see `CONTRIBUTING.md`) |

If a helper you need is missing and would serve more than one generator, say so in your
pull request instead of writing a private copy.

## Figures

A generator that draws a picture sets `Question.figure` to a spec built by `figures.py`
(`make()` validates it with `figures.check`). Such a generator should also register a
reader that recomputes the answer from the spec alone, never from `question.answer`:

```python
from ..figures import register_reader


def _read_my_figure(question):
    spec = question.figure
    ...  # recompute the answer from spec only
    return answer_text


register_reader("my_topic", _read_my_figure)
```

`scripts/regrade_sweep.py` runs the registered reader on every figure-bearing question,
and checks the figure spec and that its labels do not leak the answer.

## Tests and tools

```
python -m pytest tests/public -q                     # engine API tests
python -m pytest tests/test_my_generator.py -q -n 8  # name your own file and use workers
python -m pytest tests -q -n auto                    # the whole suite
```

A generator's test should call it across many seeds, grade its own answer as correct, and
grade each distractor as wrong. Naming files keeps runs fast; collecting the whole `tests/`
folder takes noticeably longer before anything runs.

```
python scripts/check_generator_speed.py     # median make() per topic default and per skill-source call: fails over 250 ms, on an error or a call over 5 s; flags over 15 ms
python scripts/sweep_skills.py --skills <skill-id> --seeds 5    # build, grade and check every source (the nightly job runs all, 40 seeds)
python scripts/build_skill_index.py --where <skill-or-topic>    # which module and line define a generator
python scripts/regrade_sweep.py --ids <skill-id> [--seeds N]    # false rejects and false accepts
python scripts/render_skill_questions.py --ids <skill-id> --out sample.md   # read the questions
```

The tests and sweeps call every generator many times, so one slow generator slows every run.
Keep `make()` well under the flag threshold unless the work is genuinely symbolic.

Topics in modules named `*_build.py` should pass the text checks in `text_quality.py` (no `**`,
no `*` as a times sign, no "1 parts"). `render_skill_questions.py` applies them to each rendered
question and exits non-zero if any is flagged, so run it for your skill. Older generators (modules
not named `*_build.py`) still show Python syntax such as `x**2` in `text`, and are not held to it.
