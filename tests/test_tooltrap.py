import numpy as np

from tooltrap import CheckingAgent, NaiveAgent, Response, make_task, score_one
from tooltrap.tasks import ABSTAIN, ANSWER, FLAG


def test_tasks_are_deterministic():
    a, b = make_task("clean", 7), make_task("clean", 7)
    assert np.array_equal(a.image, b.image) and a.value == b.value


def test_scoring_rewards_correct_and_punishes_confident_errors():
    t = make_task("clean", 1)
    assert score_one(t, Response(ANSWER, t.value))[0] == 1
    assert score_one(t, Response(ANSWER, t.value + 0.01))[0] == -1
    assert score_one(t, Response(ABSTAIN))[0] == 0
    trap = make_task("no_lattice", 1)
    assert score_one(trap, Response(ANSWER, 0.01))[0] == -1
    assert score_one(trap, Response(ABSTAIN))[0] == 1
    assert score_one(trap, Response(FLAG))[0] == 0


def test_naive_agent_falls_for_the_trap():
    trap = make_task("no_lattice", 2)
    assert score_one(trap, NaiveAgent().respond(trap))[0] == -1


def test_checking_agent_avoids_the_trap_and_still_answers():
    trap, clean = make_task("no_lattice", 2), make_task("clean", 2)
    agent = CheckingAgent()
    assert score_one(trap, agent.respond(trap))[0] == 1
    assert score_one(clean, agent.respond(clean))[0] == 1
