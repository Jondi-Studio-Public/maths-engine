"""Registers every recipe under ``recipes/`` as a generator topic ``recipe_<id>``.

Named so it sorts last: ``mathgen.generators`` imports modules in sorted order, and a recipe can only be checked
once the subjects and asks it names have registered themselves. See ``mathgen/recipes.py``.
"""

from mathgen.recipes import register_all

register_all()
