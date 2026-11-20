"""Baseline agents.

An agent gets a task and the tools, and returns a Response. Any agent,
including an LLM, only has to implement `respond`.
"""
from dataclasses import dataclass

from .tasks import ABSTAIN, ANSWER, FLAG
from .tools import check_reliability, measure_strain


@dataclass
class Response:
    decision: str            # "answer", "abstain" or "flag"
    value: float = None      # strain exx, when decision == "answer"
    explanation: str = ""


class NaiveAgent:
    """Calls the measurement tool and reports whatever it returns."""
    name = "Naive (trusts the tool)"

    def respond(self, task):
        return Response(ANSWER, measure_strain(task.image)["exx"], "Reported the tool output.")


class CheckingAgent:
    """Runs the reliability checks first, then decides whether to answer."""
    name = "Checking (verifies first)"

    def respond(self, task):
        rel = check_reliability(task.image)
        status = {c["name"]: c for c in rel["checks"]}
        # Give a specific diagnosis when possible: a defect in a clear lattice is a finding.
        if status["Dislocation cores"]["status"] != "PASS" and status["Lattice peak strength"]["status"] != "FAIL":
            return Response(FLAG, explanation=f"{status['Dislocation cores']['value']} dislocation core(s) found.")
        failed = [c["name"] for c in rel["checks"] if c["status"] == "FAIL"]
        if failed:
            return Response(ABSTAIN, explanation="Not trustworthy: " + ", ".join(failed))
        return Response(ANSWER, measure_strain(task.image)["exx"], f"Checks grade {rel['grade']}.")


BASELINES = [NaiveAgent, CheckingAgent]
