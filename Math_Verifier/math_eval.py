"""Faithful port of DeepSeek-Math's MATH evaluation: `math_equal` + `strip_string`.

DeepSeek's full pipeline applies `strip_string` (LaTeX canonicalization) to both
prediction and reference before calling `math_equal` (sympy-based numeric +
symbolic equivalence). The wrapper `is_equiv` here composes both.

Sources:
- math_equal, symbolic_equal, is_digit, parse_digits:
    https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/eval/eval_utils.py
- strip_string, _fix_fracs, _fix_a_slash_b, _fix_sqrt, _fix_tan:
    https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/data_processing/answer_extraction.py

Both fetched 2026-05-01.

Requirements: `pip install antlr4-python3-runtime==4.11` (sympy.parse_latex
otherwise silently fails — the original cause of our 19.8pp accuracy
underestimate when running with `math_verify` and a missing antlr4).

Behavior is unchanged from upstream modulo:
- The `timeout=True` branch (uses `multiprocessing` + `call_with_timeout`) is
  omitted; we always run synchronously. Pathological inputs that hang sympy
  could in principle wedge a worker; we have not seen this on MATH-500.
"""

from __future__ import annotations

from math import isclose
from typing import Union

import re
import regex

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex


def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except:
        pass
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """Exact match of math iff:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction) == str(reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, abs_tol=1e-3):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    # symbolic equal with sympy
    if symbolic_equal(prediction, reference):
        return True

    return False


# ---------------------------------------------------------------------------
# strip_string and helpers (ported from data_processing/answer_extraction.py)
# ---------------------------------------------------------------------------

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("x\\in", "")

    string = string.replace("\\%", "%")
    string = string.replace(r"\%", "%")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    string = string.replace("\\cdot", "")

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace("\"", "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def is_equiv(prediction, reference) -> bool:
    """Strip-then-math_equal — DeepSeek's full equivalence check."""
    if prediction is None:
        return False
    try:
        return math_equal(strip_string(prediction), strip_string(reference))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Answer extractors (ported from data_processing/answer_extraction.py)
# ---------------------------------------------------------------------------

def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == "{":
                n += 1
            elif piece[i] == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(pred_str):
    if "```output" not in pred_str:
        return ""
    if "```output" in pred_str:
        pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    output = pred_str.strip()
    return output


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            pred.append(program_output)
        else:
            pattern = r"-?\d*\.?\d+"
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if len(ans) >= 1:
                ans = ans[-1]
            else:
                ans = ""
            if ans:
                pred.append(ans)

    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def extract_math_answer(question, reasoning, task="math"):
    """DeepSeek's question-aware multi-candidate math extractor (exhaust=True)."""
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if "separated by commas" in question and all(ch not in ans for ch in "()[]"):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend(
                [a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")]
            )
        else:
            answer.append(ans.strip())
    return answer


def is_equiv_multi(question, reasoning, gold) -> bool:
    """DeepSeek's full math eval: extract all candidate answers, mark correct
    if any is equivalent to the gold under strip_string + math_equal.
    """
    candidates = extract_math_answer(question, reasoning, task="math")
    if not candidates:
        return False
    gold_stripped = strip_string(gold)
    for c in candidates:
        try:
            if math_equal(c, gold_stripped):
                return True
        except Exception:
            continue
    return False
