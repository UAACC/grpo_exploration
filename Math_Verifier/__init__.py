"""Math_Verifier — the project's canonical math equivalence checker.

Faithful port of DeepSeek-Math's evaluation pipeline. See `math_eval.py`
inside this package for the full source attribution and `docs/eval_methodology.md`
for the upgrade story.

Public API (use these from anywhere in the project):

    from Math_Verifier import is_equiv_multi          # multi-candidate, full pipeline
    from Math_Verifier import is_equiv                # single-candidate
    from Math_Verifier import math_equal, strip_string
    from Math_Verifier import extract_math_answer, extract_boxed_answers

Critical dependency: `pip install antlr4-python3-runtime==4.11`. Without
it, sympy's parse_latex silently no-ops and accuracy is undercounted by
5-15pp on MATH-style inputs.
"""

from .math_eval import (
    # Comparators (the production callables)
    is_equiv,
    is_equiv_multi,
    # Lower-level building blocks
    math_equal,
    symbolic_equal,
    is_digit,
    parse_digits,
    strip_string,
    # Answer extractors
    extract_answer,
    extract_boxed_answers,
    extract_math_answer,
    extract_program_output,
)

__all__ = [
    "is_equiv",
    "is_equiv_multi",
    "math_equal",
    "symbolic_equal",
    "is_digit",
    "parse_digits",
    "strip_string",
    "extract_answer",
    "extract_boxed_answers",
    "extract_math_answer",
    "extract_program_output",
]
