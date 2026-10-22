"""Turn results/*.jsonl into figures and a summary table.

python -m physbreak.analysis --results results --figdir figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEVEL_LABELS = {0: "0\ntextbook", 1: "1\ndisguised", 2: "2\nregime shift",
                3: "3\ncounterfactual", 4: "4\nchaos"}
CATEGORY_ORDER = ["correct", "textbook_trap", "factor_error", "unit_or_magnitude",
                  "false_certainty", "false_abstain", "other_wrong", "no_answer",
                  "parse_error"]
CATEGORY_COLORS = {"correct": "#2e7d32", "textbook_trap": "#c62828",
                   "factor_error": "#ef6c00", "unit_or_magnitude": "#f9a825",
                   "false_certainty": "#6a1b9a", "false_abstain": "#8e24aa",
                   "other_wrong": "#546e7a", "no_answer": "#b0bec5",
                   "parse_error": "#cfd8dc"}


def load_results(results_dir="results"):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep only the latest record per (model, id) in case of retries
    df = df.drop_duplicates(subset=["model", "id"], keep="last")
    df["is_baseline"] = df["model"].str.startswith("baseline-")
    return df


def _models(df, include_oracle=False):
    ms = sorted(df["model"].unique(), key=lambda m: (m.startswith("baseline-"), m))
    return [m for m in ms if include_oracle or m != "baseline-oracle"]


def accuracy_table(df):
    t = df.pivot_table(index="model", columns="level", values="correct", aggfunc="mean")
    return (100 * t).round(1)


def fig_cliff(df):
    """The headline figure: accuracy vs distance from the textbook."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in _models(df):
        sub = df[(df.model == m) & (df.level <= 3)]
        acc = sub.groupby("level")["correct"].mean() * 100
        n = sub.groupby("level")["correct"].count()
        err = 100 * np.sqrt((acc / 100) * (1 - acc / 100) / n)
        style = dict(ls="--", color="grey", alpha=0.8) if m.startswith("baseline-") else {}
        ax.errorbar(acc.index, acc.values, yerr=err.values, marker="o", capsize=3,
                    lw=2, label=m, **style)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LEVEL_LABELS[i] for i in range(4)])
    ax.set_ylim(-3, 103)
    ax.set_ylabel("Accuracy (%) vs. simulated ground truth")
    ax.set_xlabel("Distance from the textbook")
    ax.set_title("The memorisation cliff: LLM physics accuracy as problems leave the textbook")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_heatmap(df):
    models = [m for m in _models(df) if not m.startswith("baseline-")] or _models(df)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.8), squeeze=False)
    for ax, m in zip(axes[0], models):
        t = df[df.model == m].pivot_table(index="family", columns="level",
                                          values="correct", aggfunc="mean") * 100
        im = ax.imshow(t.values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(t.shape[1]))
        ax.set_xticklabels([str(c) for c in t.columns])
        ax.set_yticks(range(t.shape[0]))
        ax.set_yticklabels(t.index)
        for (i, j), v in np.ndenumerate(t.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("level")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="accuracy (%)")
    fig.suptitle("Where each model breaks: accuracy by physics family and level")
    return fig


def fig_failures(df):
    models = [m for m in _models(df) if not m.startswith("baseline-")] or _models(df)
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = df[df.model == m]
        t = pd.crosstab(sub.level, sub.category, normalize="index") * 100
        bottom = np.zeros(len(t))
        for c in CATEGORY_ORDER:
            if c in t:
                ax.bar(t.index, t[c], bottom=bottom, color=CATEGORY_COLORS[c], label=c)
                bottom += t[c].values
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("level")
        ax.set_ylabel("% of answers")
        ax.set_xticks(t.index)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    fig.suptitle("How models fail: failure taxonomy by level")
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    return fig


def fig_overconfidence(df):
    """Stated confidence vs. actual accuracy. A growing gap = overconfidence."""
    models = [m for m in _models(df) if not m.startswith("baseline-")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        sub = df[(df.model == m) & df.confidence.notna()]
        if sub.empty:
            continue
        g = sub.groupby("level").agg(acc=("correct", "mean"), conf=("confidence", "mean"))
        line, = ax.plot(g.index, g.acc * 100, marker="o", lw=2, label=f"{m}: accuracy")
        ax.plot(g.index, g.conf * 100, marker="s", ls="--", color=line.get_color(),
                label=f"{m}: stated confidence")
        ax.fill_between(g.index, g.acc * 100, g.conf * 100, color=line.get_color(), alpha=0.12)
    ax.set_xticks(sorted(df.level.unique()))
    ax.set_xticklabels([LEVEL_LABELS[i] for i in sorted(df.level.unique())])
    ax.set_ylim(-3, 103)
    ax.set_ylabel("%")
    ax.set_title("Overconfidence: stated confidence vs. actual accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chaos_table(df):
    sub = df[df.level == 4].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["case"] = np.where(sub.truth == "UNPREDICTABLE", "chaotic (truly unpredictable)",
                           "regular (predictable)")
    return (sub.pivot_table(index="model", columns="case", values="correct", aggfunc="mean")
            * 100).round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    df = load_results(a.results)
    if df.empty:
        raise SystemExit(f"No results found in {a.results}/")
    os.makedirs(a.figdir, exist_ok=True)
    for name, fn in [("1_memorisation_cliff", fig_cliff), ("2_heatmap", fig_heatmap),
                     ("3_failure_taxonomy", fig_failures),
                     ("4_overconfidence", fig_overconfidence)]:
        fig = fn(df)
        fig.savefig(os.path.join(a.figdir, f"{name}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    print("Accuracy (%) by level:\n", accuracy_table(df).to_string(), "\n")
    ct = chaos_table(df)
    if not ct.empty:
        print("Chaos level (%) correct:\n", ct.to_string(), "\n")
    print("Failure categories (% of all answers):")
    print((pd.crosstab(df.model, df.category, normalize="index") * 100).round(1).to_string())
    print(f"\nFigures saved to {a.figdir}/")


if __name__ == "__main__":
    main()
