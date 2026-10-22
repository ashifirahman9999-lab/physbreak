"""Run models on the benchmark. Resumable: re-running skips problems already answered.

python -m physbreak.run --models gemini-2.5-flash baseline-textbook --sleep 4
"""
import argparse
import json
import os
import time

from .grade import grade
from .models import get_model


def results_path(outdir, model_name):
    safe = model_name.replace("/", "_").replace("@", "_")
    return os.path.join(outdir, f"{safe}.jsonl")


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def run_model(model, problems, outdir, sleep, limit=None, retry_errors=False):
    path = results_path(outdir, model.name)
    done = load_done(path)
    todo = [p for p in problems
            if p["id"] not in done
            or (retry_errors and str(done[p["id"]].get("raw", "")).startswith("ERROR"))]
    if limit:
        todo = todo[:limit]
    print(f"{model.name}: {len(done)} done, {len(todo)} to go -> {path}")
    with open(path, "a") as f:
        for i, p in enumerate(todo, 1):
            t0 = time.time()
            raw, parsed = model.ask(p)
            g = grade(p, parsed)
            rec = {"id": p["id"], "model": model.name, "family": p["family"],
                   "level": p["level"], "level_name": p["level_name"],
                   "truth": p["answer"], "naive": p["naive_answer"], "raw": raw,
                   "method": (parsed or {}).get("method") if isinstance(parsed, dict) else None,
                   **g, "latency_s": round(time.time() - t0, 2)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            mark = "OK " if g["correct"] else "X  "
            print(f"  [{i}/{len(todo)}] {mark}{p['id']:22s} {g['category']}")
            if not model.name.startswith("baseline-"):
                time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="seconds between API calls (free-tier rate limits)")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="optional Gemini thinking budget (e.g. 0 to switch thinking off)")
    ap.add_argument("--limit", type=int, default=None, help="only run this many new problems")
    ap.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args()
    with open(a.problems) as f:
        problems = json.load(f)
    os.makedirs(a.outdir, exist_ok=True)
    for name in a.models:
        run_model(get_model(name, a.thinking_budget), problems, a.outdir, a.sleep,
                  a.limit, a.retry_errors)


if __name__ == "__main__":
    main()
