"""Figure kinds beyond the coordinate plane and the number line.

Each module registers its kinds with :func:`mathgen.figures.register_kind` on
import, and :mod:`mathgen.figures` imports this package once its own helpers
exist, so every kind is known wherever ``figures`` is imported.

- :mod:`.data`: statistical displays (charts and plots).
- :mod:`.early`: money, the hundred chart, fraction models, counters and base-ten blocks.
- :mod:`.measure`: clocks, scales, money and early-years counting pictures.
- :mod:`.space`: angles, two-dimensional diagrams, trees and Venn diagrams.
- :mod:`.tally`: tally tables and scatters of objects to sort.
"""

from mathgen.figure_kinds import data, early, measure, space, tally  # noqa: F401
