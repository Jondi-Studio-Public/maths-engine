"""Answer grading for mathgen questions.

`spoken` holds the number and expression tokenisers shared by the typed-answer graders and the
spoken-answer parser; `typed` holds the graders, the `GRADERS` registry and `grade()`. `mathgen.grading` is a
shim that aliases `typed`.
"""

from . import spoken  # noqa: F401  (typed imports it; importing it first avoids a package cycle)
from .typed import GRADERS, GradeResult, grade, mcq_options  # noqa: F401
from .inequality import grade_inequality
from .matrix import grade_matrix

# Registered here rather than in `typed`, because both modules build on `typed`'s readers.
GRADERS["inequality"] = grade_inequality
GRADERS["matrix"] = grade_matrix
