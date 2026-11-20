"""ToolTrap: does an AI agent notice when a scientific tool gives a confident wrong answer?"""
from .agents import BASELINES, CheckingAgent, NaiveAgent, Response
from .scoring import score_one, summarize
from .tasks import CATEGORIES, make_suite, make_task

__version__ = "0.1.0"
