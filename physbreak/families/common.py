"""Shared helpers for problem families.

Every problem is a dict with:
  id, family, level, level_name, prompt,
  answer        float, or the string "UNPREDICTABLE"
  unit          unit the answer must be given in
  tol           tolerance (relative unless tol_type == "abs")
  tol_type      "rel" or "abs"
  naive_answer  what the textbook formula predicts (None if no textbook formula)
  params        the generating parameters (for reproducibility)
"""

LEVEL_NAMES = {
    0: "textbook",
    1: "disguised",
    2: "regime_shift",
    3: "counterfactual",
    4: "chaos",
}

ANSWER_FORMAT = (
    "\n\nReply ONLY with a JSON object of the form "
    '{"answer": <number or the string "UNPREDICTABLE">, '
    '"confidence": <number from 0 to 1>, '
    '"method": "<one sentence on the formula or approach you used>"}. '
    "Give the answer as a plain number in the requested unit, with no units in the answer field."
)


def make_problem(family, level, idx, prompt, answer, unit, params,
                 naive_answer=None, tol=0.05, tol_type="rel"):
    return {
        "id": f"{family}-L{level}-{idx:03d}",
        "family": family,
        "level": level,
        "level_name": LEVEL_NAMES[level],
        "prompt": prompt.strip() + ANSWER_FORMAT,
        "answer": answer if isinstance(answer, str) else float(answer),
        "unit": unit,
        "tol": tol,
        "tol_type": tol_type,
        "naive_answer": None if naive_answer is None else float(naive_answer),
        "params": params,
    }


def r(x, n=3):
    """Round to n significant figures for display in prompts."""
    return float(f"{x:.{n}g}")
