"""Run every agent on the task suite and write the results table and chart.

    python run_benchmark.py            # 20 tasks per category
    python run_benchmark.py --n 5      # quick run
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tooltrap import BASELINES, CATEGORIES, make_suite, summarize

DOCS = Path(__file__).resolve().parent / "docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="tasks per category")
    ap.add_argument("--seed", type=int, default=0, help="task-suite seed; use a new one for a held-out run")
    args = ap.parse_args()

    tasks = make_suite(args.n, seed=args.seed)
    results = {}
    for Agent in BASELINES:
        agent = Agent()
        results[agent.name] = summarize(tasks, [agent.respond(t) for t in tasks])

    cats = list(CATEGORIES) + ["overall"]
    lines = ["| Category | Expected | " + " | ".join(results) + " |",
             "|---|---|" + "---|" * len(results)]
    for c in cats:
        exp = CATEGORIES[c][0] if c in CATEGORIES else ""
        cells = [f"{r[c]['score']:+.2f} ({r[c]['confident_errors']:.0%} confident errors)" for r in results.values()]
        name = f"**{c}**" if c == "overall" else c
        lines.append(f"| {name} | {exp} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(f"{len(tasks)} tasks\n\n{table}")

    DOCS.mkdir(exist_ok=True)
    (DOCS / f"results_seed{args.seed}.md").write_text(table + "\n")

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    x = np.arange(len(cats))
    colors = ["#d1495b", "#2747a8"]
    w = 0.38
    for i, (name, r) in enumerate(results.items()):
        ax.bar(x + (i - 0.5) * w, [r[c]["score"] for c in cats], w, label=name, color=colors[i % 2])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, [c.replace("_", " ") for c in cats])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("score (+1 correct, -1 confident error)")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(DOCS / f"results_seed{args.seed}.png", dpi=140)


if __name__ == "__main__":
    main()
