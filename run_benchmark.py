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
    ap.add_argument("--llm", default=None, help="also run a local Ollama model, e.g. llama3.2:3b")
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    tasks = make_suite(args.n, seed=args.seed)
    agents = [A() for A in BASELINES]
    if args.llm:
        from tooltrap.llm_agent import OllamaAgent
        agents += [OllamaAgent(args.llm), OllamaAgent(args.llm, hint=True)]
    results = {}
    for agent in agents:
        responses = []
        for i, t in enumerate(tasks):
            responses.append(agent.respond(t))
            if hasattr(agent, "transcripts") and (i + 1) % 10 == 0:
                print(f"  {agent.name}: {i + 1}/{len(tasks)}", flush=True)
        results[agent.name] = summarize(tasks, responses)
        if hasattr(agent, "transcripts"):
            import json
            tag = agent.name.replace(" ", "_").replace(":", "-").replace("+", "plus")
            with open(DOCS / f"transcripts_{tag}_seed{args.seed}.jsonl", "w") as f:
                for t in tasks:
                    rec = {"task_id": t.task_id, "expected": t.expected, "true_value": t.value}
                    rec.update(agent.transcripts[t.task_id])
                    f.write(json.dumps(rec) + "\n")

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
    suffix = "_llm" if args.llm else ""
    (DOCS / f"results_seed{args.seed}{suffix}.md").write_text(table + "\n")

    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    x = np.arange(len(cats))
    colors = ["#d1495b", "#2747a8", "#edae49", "#00798c"]
    w = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        ax.bar(x + (i - (len(results) - 1) / 2) * w, [r[c]["score"] for c in cats], w, label=name, color=colors[i % 4])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, [c.replace("_", " ") for c in cats])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("score (+1 correct, -1 confident error)")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(DOCS / f"results_seed{args.seed}{suffix}.png", dpi=140)


if __name__ == "__main__":
    main()
