"""Spaced-review scheduling for skills: no database, no clock, no app code.

- `stages`: which of a skill's sources a review at a given stage draws from.
- `fsrs`: the FSRS-5 memory model (stability, difficulty, predicted recall).
- `review`: a finished review's outcome turned into an FSRS rating, memory and due date.
- `prerequisites`: the prerequisite graph's closure, the skills a learner is ready to start, and the
  gaps before a skill. Advice only: prerequisites never gate a skill.
- `calibration`: item difficulty (Rasch) and FSRS calibration from attempt rows an app supplies.
"""
