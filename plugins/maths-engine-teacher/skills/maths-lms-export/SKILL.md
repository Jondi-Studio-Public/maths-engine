---
name: maths-lms-export
description: Exports a maths-engine paper as a Moodle XML question bank or an IMS QTI 2.1 package (as used by Canvas and other LMSs) so the questions can be auto-marked online. Explains which answer kinds the LMS can mark itself and which are text matches a teacher must read, and gives generic import steps. Use when a teacher wants questions in Moodle, Canvas or another LMS, wants an online quiz, or asks for QTI or Moodle XML.
license: MIT
---

# Exporting to an LMS

`build_paper` can write two LMS formats from the same paper spec used for printed worksheets:

- `moodle`: a Moodle XML question bank (`.xml`).
- `qti`: an IMS QTI 2.1 package (`.zip`), which Canvas and several other systems import.

Both need answers, so they only exist in this teacher plugin. The spec format is in the `maths-worksheets` skill (`references/paper-spec.md`). Use `maths-skill-finder` to choose skills.

## When to use it

Use an export when the class will do the work online and the LMS should mark it. For paper tests use `maths-worksheets` instead. If the teacher has not said, ask which they want.

## Workflow

1. Choose gradeable skills and sample them with `make_questions(..., answers=true)`, as for a worksheet. Pay particular attention to the answer kinds below, because they decide whether the LMS can mark the item.
2. Write the spec. A bank holds **one version** of a paper: pass `version` ("A", "B" ...; A if left out). To import several versions, call `build_paper` once per version, or use `count` to make a larger pool in one version.
3. Call `build_paper(spec, format="moodle")` or `format="qti"`. The tool returns the path of the file it wrote. Report that path.
4. Tell the teacher what the LMS will and will not mark (below), and the import steps.

## What the LMS marks itself

| Answer kind | Becomes | Auto-marked? |
|---|---|---|
| `mcq` | single-answer multiple choice | Yes |
| `number` | numerical answer within a tolerance (none if the answer is an exact integer or terminating decimal) | Yes |
| `table` | a grid with a box per blank | Yes |
| multi-part question | one item (Moodle: a cloze) with a box or menu per part | Yes, per part |
| `expression`, `pair`, `fraction`, `inequality`, `ordered_list`, `list`, `set`, `matrix` ... | a text answer compared with the single answer given, ignoring case | Only if the student writes it exactly that way |

For the last row, a student who writes an equivalent form (for example `2/4` for `1/2`, or `√2*3` for `3√2`) is marked **wrong** by the LMS. The engine's own `grade_answer` accepts equivalent forms, but an LMS cannot. These items suit a teacher who reads the responses. The full answer and working are placed in each question's feedback so it is easy to check.

Say this to the teacher plainly, and give counts: for example "6 of 10 items auto-mark; 4 are text answers you should review." Offer to swap text-answer skills for `mcq` or `number` skills (`find_skills(answer_kind="mcq")` or `"number"`) if they want full auto-marking.

Figures are embedded as SVG. Without a drawing, the figure's description is printed instead.

## Importing (general steps)

Menus differ between versions and institutions, so these are deliberately general. Check your own system's help if a step is not where you expect.

**Moodle**
1. Open your course and find the question bank.
2. Choose the import option and pick the format "Moodle XML".
3. Upload the `.xml` file and run the import. The questions land in a category named after the paper and version.
4. Create a quiz and add questions from that category.

**Canvas and other QTI systems**
1. In your course, find the import option for course content or quizzes.
2. Choose the QTI package type (QTI 2.1 where it is offered, or the generic QTI option) and upload the `.zip`.
3. Run the import and check that it completes. Open the imported quiz or question bank and look at a few items.

Always preview an imported quiz as a student before releasing it, and check one item of each answer kind. If an import fails, tell the user the exact message; do not guess a fix.

## Attribution

The LMS files do not carry the paper's attribution lines. If the quiz uses any senior secondary skill (ids beginning `senior-`, framework `acara-senior-v8-4`), tell the teacher to add this to the course page or quiz description, and print the exact text from the paper's `attribution` list (build a `json` copy to read it, or take it from `engine_info`). Do not paraphrase it.

## Cautions

- Do not give students the generated files: they contain the answers.
- An LMS stores the questions as they were; if the library is later updated, rebuild and re-import rather than editing.
- Rubric skills (proofs, explanations) cannot be exported as auto-marked items.
