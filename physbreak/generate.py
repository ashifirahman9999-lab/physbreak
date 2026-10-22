"""Build the benchmark: python -m physbreak.generate --n 8 --out data/problems.json"""
import argparse
import json
import zlib
import os

import numpy as np

from .families import FAMILIES

LEVELS = {name: [0, 1, 2, 3] for name in FAMILIES}
LEVELS["pendulum"] = [0, 1, 2, 3, 4]  # chaos level


def build(n_per_cell=8, seed=0, families=None):
    problems = []
    for name, mod in FAMILIES.items():
        if families and name not in families:
            continue
        for level in LEVELS[name]:
            # independent, reproducible stream per (family, level)
            rng = np.random.default_rng([seed, zlib.crc32(name.encode()), level])
            problems += mod.generate(level, n_per_cell, rng)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="problems per family per level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/problems.json")
    ap.add_argument("--families", nargs="*")
    a = ap.parse_args()
    probs = build(a.n, a.seed, a.families)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(probs, f, indent=1)
    print(f"Wrote {len(probs)} problems to {a.out}")


if __name__ == "__main__":
    main()
