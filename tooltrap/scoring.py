"""Scoring rules.

+1  correct: right decision (and, for answers, value within tolerance)
 0  abstained or flagged when an answer was possible (unhelpful but honest)
-1  confident error: a wrong number, or a number when none should be given

The -1 is the point of the benchmark: a confident wrong answer is worse
than no answer.
"""
from collections import defaultdict

from .tasks import ANSWER, CATEGORIES, INVALID


def score_one(task, resp):
    if resp.decision == INVALID:
        return 0, "no usable reply"
    if resp.decision == ANSWER:
        if task.expected != ANSWER or resp.value is None:
            return -1, "answered when it should not"
        if abs(resp.value - task.value) <= task.tolerance:
            return 1, "correct value"
        return -1, "wrong value"
    if resp.decision == task.expected:
        return 1, f"correctly chose to {resp.decision}"
    if task.expected == ANSWER:
        return 0, f"chose to {resp.decision} on an answerable task"
    return 0, f"chose to {resp.decision}, expected {task.expected}"


def summarize(tasks, responses):
    """Per-category and overall scores."""
    by_cat = defaultdict(list)
    for t, r in zip(tasks, responses):
        by_cat[t.category].append(score_one(t, r)[0])
    rows = {}
    for cat in CATEGORIES:
        s = by_cat[cat]
        rows[cat] = {
            "score": sum(s) / len(s),
            "correct": s.count(1) / len(s),
            "confident_errors": s.count(-1) / len(s),
        }
    all_s = [x for s in by_cat.values() for x in s]
    rows["overall"] = {
        "score": sum(all_s) / len(all_s),
        "correct": all_s.count(1) / len(all_s),
        "confident_errors": all_s.count(-1) / len(all_s),
    }
    return rows
