# Worked example of a session loop

This exchange is **illustrative**. The question text and ids are invented to show the pattern; real ones come from the tools.

**Learner:** Can I practise surds? I'm in Year 9.

**Tutor:** (calls `find_skills(text="surd")`, then `get_skill("simplify-surds")`, sees `gradeable: true`, sources `easy`, `medium`, `hard`, `exam`.)
Let's start with "Simplify surds", which is about writing things like √125 in the form a√b. We'll begin on easy.

(calls `make_questions("simplify-surds", count=1, source="easy")`)

**Tutor:** Question 1 (1 mark): Simplify √72. Give your answer in the form a√b.

**Learner:** 36√2

**Tutor:** (calls `grade_answer(id, "36√2")`; result: `correct: false`, `misconception: "extracts-without-rooting"`.)
Not quite. You found the right perfect square, so that part worked. Check what happens to a perfect square when it comes out of the root sign. What number multiplies the outside, the square or its square root? Have another go.

**Learner:** 6√2

**Tutor:** (calls `grade_answer(id, "6√2")`; `correct: true`.)
That's right. Next one.

(Tutor records: Q1 wrong then right on second attempt, misconception: took the square instead of the root.)

... two more questions go well ...

**Tutor:** Three in a row. Shall we step up to medium? These have two surds to combine.

---

## A struggling run

Suppose the learner misses three questions in a row with the same misconception (`drops-the-radical`).

**Tutor:** (coaches once more, then calls `get_skill("simplify-surds")` and sees prerequisites `f10-square-roots-perfect-squares` and `prime-factorisation`. The errors point at factors, so it chooses `prime-factorisation`.)
I think the sticking point is breaking numbers into factors. Let's do a few on that first, then come back to surds.

(calls `make_questions("prime-factorisation", count=1, source="easy")`, one question at a time, marks each with `grade_answer`.)

After a short streak, the tutor returns to `simplify-surds` and says so.

---

## Multi-part question

When a question has `parts`, show the stem and each part with its label and marks:

> Stem: A rectangle has sides 12 cm and 5 cm.
> (a) [1 mark] Find its area.
> (b) [2 marks] Find the length of a diagonal.

Collect an answer for each part, then call `grade_answer(id, {"a": "60", "b": "13"})`. Report each part from `parts`, and coach only the parts that were wrong.

---

## Session summary

> Today: simplify-surds (easy to medium), 8 questions, 6 right. The main slip was forgetting to take the square root of the extracted factor. Next time: medium surds with two terms, or `f10-square-roots-perfect-squares` if that slip returns.
