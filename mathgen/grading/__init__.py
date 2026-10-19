"""Answer grading for mathgen questions.

`spoken` holds the number and expression tokenisers shared by the typed-answer graders and the
spoken-answer parser; `typed` holds the graders, the `GRADERS` registry and `grade()`. `mathgen.grading` is a
shim that aliases `typed`.
"""

from . import spoken  # noqa: F401  (typed imports it; importing it first avoids a package cycle)
from .typed import GRADERS, GradeResult, grade, mcq_options  # noqa: F401
from .grid import grade_grid_cells
from .inequality import grade_inequality
from .matrix import grade_matrix
from .open_response import grade_open_response
from .chain import grade_step_chain

# Registered here rather than in `typed`, because these modules build on `typed`'s readers.
GRADERS["grid_cells"] = grade_grid_cells
GRADERS["inequality"] = grade_inequality
GRADERS["matrix"] = grade_matrix
GRADERS["open_response"] = grade_open_response  # pending until a marker returns a result: open_response.py
GRADERS["step_chain"] = grade_step_chain  # a typed proof, one expression per line: chain.py
