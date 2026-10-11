# Multi-part questions

Status: implemented in this change for the engine; the site's UI is not built (see "What the site needs").

## Why

Australian senior exams ask connected questions: part (a) finds a value, part (b) uses it, and a learner who
gets (a) wrong can still earn the marks for (b) by using their own (a) correctly ("follow-through", or
"consequential" marking). Before this change the engine had no such shape. An exam tier was one answer, graded
right or wrong, and the three tiers that asked "(a) ... and (b) ..." packed independent answers into one `pair`,
so a slip in (a) cost the whole question. None of the 435 exam tiers on main and in the open exam PRs chained
one part into the next.

## The model

A generator may return `parts` instead of a single answer. The question keeps a `text` (the stem, the
information every part shares) and gains an ordered list of parts:

```python
return dict(
    text="A circle has an area of 120.76 cm².",
    parts=[
        dict(
            prompt="Find the radius of the circle, correct to 1 decimal place.",
            answer_kind="number",
            answer=6.2,
            answer_text="6.2",
            marks=2,
            working=[...],
            distractors=[...],
        ),
        dict(
            prompt="Hence find the circumference, correct to 1 decimal place.",
            answer_kind="number",
            answer=39.0,
            answer_text="39.0",
            marks=1,
            follow_through="2*pi*part_a",
        ),
    ],
)
```

Each part is a `mathgen.core.Part`. It carries exactly the answer fields a single-answer question carries
(`answer`, `answer_text`, `answer_latex`, `answer_form`, `answer_alt`, `distractors`, `option_notes`), plus:

| Field | Meaning |
|---|---|
| `label` | `"a"`, `"b"`, ... in order unless the generator names them |
| `prompt`, `prompt_latex` | what this part asks; `prompt_latex` defaults to `latex_prompt(prompt)` |
| `answer_kind` | the grader for this part: any kind in `GRADERS`. Parts of one question may differ |
| `marks` | whole marks for the part, default 1 |
| `follow_through` | optional SymPy expression giving this part's answer from earlier parts' values, named `part_a`, `part_b`, ... |
| `working` | this part's worked steps |
| `sensible_range` | optional open interval `(low, high)`, `None` for no bound, that a sensible answer to this part lies in: `(0, None)` for a length, area or amount of money, `(0, 1)` for a probability. A reply outside it carries no follow-through into later parts. `make()` checks that the canonical answer lies inside it |

Because a part has the same attribute names as a question, every existing grader grades a part unchanged:
`grade(part.answer_kind, part, reply)`. The typed answer contract (separate change) adds `answer_kind`,
`distractor_values` and `distractor_credit` to graded objects; a part takes them by the same names, so the two
changes meet without a second answer representation.

`make()` fills the question-level fields from the parts, so code that knows nothing about parts still works:
`text` is the stem followed by "(a) prompt (b) prompt", `answer` is `{label: part.answer}`, `answer_text` is
`"(a) 6.2; (b) 39.0"`, and `working` is every part's working, labelled. A multi-part question has no
question-level distractors.

`make()` also checks each follow-through expression: it may only name earlier parts, and evaluated at the
canonical earlier answers it must reproduce the part's own canonical answer (to within one unit in the last
displayed place, which allows for rounding the earlier answer). A wrong expression fails generation with
`VerificationError`, like any other self-check.

## Marking

`grade()` on a multi-part question takes the reply as a dict `{label: reply}`, as a JSON object string, or as
text labelled "(a) ... (b) ..." (in any order, with any part left out; or split by ";" or new lines when it
has exactly one piece per part). Each
part is graded with its own grader, in order:

1. **Correct** against the canonical answer: full marks.
2. **Follow-through.** If the part has `follow_through`, the learner's own replies to the parts it names are
   read as values, the expression is evaluated at them, and the reply is graded against that value at the
   part's own precision and form. A match earns full marks. This covers both cases exams accept: an error
   carried forward from a wrong earlier part (marked `follow_through=True` on the part result), and a correct
   earlier answer rounded as asked and then used (which can move the last digit of a later part).
   Markers withhold follow-through when the error makes the later part trivial or impossible, and so does
   the engine: no follow-through from an earlier reply of zero where the canonical value is not zero, from a
   reply outside that earlier part's `sensible_range`, or for a reply that just repeats the earlier value it
   uses. Without these, one number typed into every part earned follow-through marks in every pilot chain.
3. **Partial credit** for a reply that matches a distractor carrying `distractor_credit` (a fraction of the
   part's marks), once the typed answer contract supplies `matched_distractor`. Until then this step is inert.
4. Otherwise no marks.

The result is a `GradeResult` with three new fields: `marks` (a `Fraction`, the total earned), `max_marks` and
`parts` (`{label: GradeResult}` with each part's own `marks`, `max_marks` and `follow_through`). `correct` is
true only when every part earns full marks, so existing callers that only read `correct` see "all of it
right".

Only parts graded as a single value (`number`, `fraction`, `expression`) can be named in a follow-through
expression, because only those replies read as one value. A part that names a `pair` or `list` part is
rejected when the question is made.

Every grader in `GRADERS`, handed a multi-part question, grades it part by part whatever kind it was called
as. That is what lets a skill whose other tiers are `pair` or `number` add a multi-part exam tier without the
library sweeps (`scripts/regrade_sweep.py`, the per-skill grading tests) knowing about parts: they grade
`answer_text`, which is the labelled reply that gets full marks.

### Writing a part

- A part's prompt states the precision of its own answer only: the number grader reads "correct to 1 decimal
  place" from the part's `text` (its prompt). A show-that prompt that quotes the target to 1 decimal place
  while asking for an exact checkpoint must keep the checkpoint exact (an integer), or word the target
  without "correct to".
- A follow-through expression is written per question, with that question's numbers in it
  (`"337*part_b"`, `"x0 - y0/part_b"` with x0 and y0 filled in).
- `follow_through` on a part's result is true when a part it names was answered other than canonically, so a
  chain carried from a wrong (a) through (b) into (c) says so at every step.

### Show-that parts and method marks

Real extended questions (VCE, HSC, QCE) often say "show that the height is 7.4 cm, correct to 1 decimal
place" so that everyone continues from the same value. The model handles this without anything new: the
show-that part states the target in its prompt and asks for the key intermediate value (the checkpoint, such
as h² = 54.75) as its answer, and later parts use the stated value, so they need no follow-through. A method
mark is the same idea: a part that asks for the checkpoint value a correct method passes through. A capstone
worth 3 or 4 marks is a part with `marks: 3` whose earlier checkpoints are their own parts.

### What this does not do

- **Marks for written working.** The engine never sees working; checkpoint parts and follow-through are the
  deterministic parts of method marking. Marking written working stays with the rubric route.
- **Sub-parts** ("(a)(i)"). Labels are flat. A generator can use `label="a(i)"`, but nothing nests.
- **Per-part figures.** A figure belongs to the question.

## Scheduler and stages

A multi-part question is one item in the `exam` stage. The site's scheduler never schedules exam sources; they
reach learners through tests and practice rounds, so the scheduler needs no change. An attempt records `correct`
(full marks) as today, plus the marks.

## Pilot

Five exam tiers are rebuilt as connected chains with follow-through, each a new generator so the existing
tiers' outputs and tests are untouched:

| Skill | Parts (marks) |
|---|---|
| `circle-area-to-diameter-and-circumference` | radius from the area (2); hence the circumference (1, FT); the arc of a sector using that radius (2, FT) |
| `square-pyramid-volume-and-surface-area` | show that the height is h, giving h² (1); volume from the stated h (2); surface area (2); volume of a scaled copy from (b) (1, FT) |
| `senior-use-implicit-differentiation-to-determine-the-gradient-of` | dy/dx (2); hence the gradient at P (1, FT); hence the tangent (2, FT); where it meets the x-axis (1, FT) |
| `senior-calculate-probabilities-and-quantiles-associated-with-a-given` | z-score (1); hence the probability (2, FT); expected count in a sample (1, FT); a quantile (2) |
| `senior-use-technology-to-investigate-the-effect-of-the` | balance (2); hence the interest (1, FT); balance with annual compounding or a lower rate (1); the difference (1, FT) |

Each was given an independent oracle test that re-derives every part from the question text, follow-through
tests from wrong earlier answers, and a clean `regrade_sweep.py` run over 60 seeds.

The old exam presets of the converted skills are left in their generators, unused by any skill, and their
tests still run against them. Five existing tests pinned the old exam topic for these skills; each got the
smallest edit that points it at the new topic or at the old preset directly. Retiring the old presets is a
later clean-up.

## Distractors

Each part's distractors come from that part's own errors only, and are graded against that part alone. Do not
build whole-question options in which every other part is right: in a third of the old `pair` tiers that made
the key the most common value in each position, which gives the answer away. An error that carries into later
parts is follow-through's job, not a distractor's.

## Migrating other exam tiers

Exam-tier building is paused until this model and the answer contract exist. A new exam tier should be a
chain when the exam topic is one: write the parts so later parts use earlier results and set
`follow_through` on each part that does. A single-step exam tier stays acceptable where real papers ask a
single step. Single 1 to 3 mark steps are a fair match for short-answer exam sections, so the single-step tiers stay;
what is missing is extended-response chains. Build a modest set of chain items per topic family rather than
converting tiers one by one.

## What the site needs

The engine change is backwards compatible; the private site keeps working without changes, showing the
composed `text` and grading one text box. To show parts properly it needs:

1. **Render parts.** When `question.parts` is non-empty: the stem, then one labelled block per part with its
   `prompt_latex`, its marks ("2 marks"), and an input matching the part's `answer_kind` (the same inputs
   single-answer questions use, including the table grid and mcq options from `mcq_options(part)`).
2. **Submit a dict** `{label: reply}` (or a JSON object string) to `grade()`.
3. **Show per-part feedback** from `result.parts`: right, wrong, or "follow-through: marks given for using your
   answer to (a)", the part's canonical `answer_text`, and its `working`.
4. **Hide the answers.** Wherever the site strips answers before sending a question to the browser, also
   strip each part's `answer`, `answer_text`, `answer_latex`, `answer_alt`, `working`, `follow_through`,
   `distractor_values` and `distractor_credit`.
5. **Store the score.** Add nullable `marks`, `max_marks` and the per-part results to an attempt, with
   `correct` meaning full marks. The `table` kind is the pattern to copy: a JSON reply, per-cell results in the
   grade result, and its own answer component.
6. **Mental mode and voice** skip multi-part questions: they are exam-stage only.
