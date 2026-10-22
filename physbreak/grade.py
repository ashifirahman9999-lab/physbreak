"""Score answers and classify every failure."""
import math

COMMON_FACTORS = {"x2": 2.0, "x1/2": 0.5, "x2pi": 2 * math.pi, "x1/2pi": 1 / (2 * math.pi),
                  "xpi": math.pi, "x1/pi": 1 / math.pi, "x4": 4.0, "x1/4": 0.25,
                  "xsqrt2": math.sqrt(2), "x1/sqrt2": 1 / math.sqrt(2)}


def _to_number(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.upper().startswith("UNPREDICT"):
            return "UNPREDICTABLE"
        try:
            return float(s.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def _close(a, b, tol, tol_type, unit):
    if tol_type == "abs":
        d = abs(a - b)
        if unit == "deg":
            d = abs((a - b + 180.0) % 360.0 - 180.0)
        return d <= tol
    return abs(a - b) <= tol * abs(b)


def grade(problem, parsed):
    """Return dict: correct (bool), category (str), given, rel_error, confidence."""
    out = {"correct": False, "category": None, "given": None, "rel_error": None,
           "confidence": None}
    if not isinstance(parsed, dict) or "answer" not in parsed:
        out["category"] = "parse_error"
        return out
    conf = parsed.get("confidence")
    out["confidence"] = conf if isinstance(conf, (int, float)) else None
    given = _to_number(parsed["answer"])
    out["given"] = given
    truth = problem["answer"]

    if given is None:
        out["category"] = "no_answer" if parsed["answer"] is None else "parse_error"
        return out
    if truth == "UNPREDICTABLE":
        if given == "UNPREDICTABLE":
            out.update(correct=True, category="correct")
        else:
            out["category"] = "false_certainty"  # confident number for a chaotic system
        return out
    if given == "UNPREDICTABLE":
        out["category"] = "false_abstain"
        return out
    if given == 0 or not math.isfinite(given):
        out["category"] = "other_wrong"
        return out

    out["rel_error"] = abs(given - truth) / abs(truth) if truth != 0 else None
    if _close(given, truth, problem["tol"], problem["tol_type"], problem["unit"]):
        out.update(correct=True, category="correct")
        return out

    naive = problem.get("naive_answer")
    if naive is not None and abs(naive - truth) > problem["tol"] * abs(truth) \
            and abs(given - naive) <= 0.05 * abs(naive):
        out["category"] = "textbook_trap"  # used the textbook formula outside its validity
        return out
    if problem["tol_type"] == "rel":
        ratio = given / truth
        if ratio > 0:
            k = round(math.log10(ratio))
            if k != 0 and abs(ratio / 10**k - 1) <= 0.05:
                out["category"] = "unit_or_magnitude"
                return out
            for f in COMMON_FACTORS.values():
                if abs(ratio / f - 1) <= 0.05:
                    out["category"] = "factor_error"
                    return out
    out["category"] = "other_wrong"
    return out
