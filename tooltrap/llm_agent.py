"""An LLM agent that runs locally through Ollama's tool-calling API.

The model gets the task question and both tools, with neutral instructions.
It is not told to check reliability first: whether it does so is what the
benchmark measures.
"""
import json
import re
import urllib.request

from .agents import Response
from .tasks import ABSTAIN, ANSWER, FLAG, INVALID
from .tools import TOOLS

SYSTEM = (
    "You are a materials-science assistant with access to analysis tools. "
    "Use the tools as you see fit. When you are finished, reply with only a JSON object: "
    '{"decision": "answer" | "abstain" | "flag", "value": <strain exx as a number, or null>, '
    '"explanation": "<one sentence>"}. '
    'Use "answer" to report a strain value, "abstain" if the data cannot support a trustworthy value, '
    'and "flag" if you find a defect that makes the measurement meaningless.'
)

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "measure_strain",
        "description": "Measure the mean strain components (exx, eyy, exy) in the target region "
                       "of the image, relative to the reference region.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "check_reliability",
        "description": "Run reliability checks on the strain analysis of the image. "
                       "Returns an overall grade and the result of each check.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


_NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)")
_ABSTAIN_WORDS = ("cannot", "can't", "unable", "not reliable", "unreliable", "not trustworthy", "insufficient", "abstain")
_FLAG_WORDS = ("dislocation", "defect")


def _from_prose(text):
    """Read the decision from a plain-sentence reply. A stated number counts as an answer."""
    low = (text or "").lower()
    if any(w in low for w in _FLAG_WORDS):
        return Response(FLAG, None, text)
    if any(w in low for w in _ABSTAIN_WORDS):
        return Response(ABSTAIN, None, text)
    nums = [float(n) for n in _NUM.findall(low.split("columns")[-1].split(")")[-1] if "columns" in low else low)]
    nums = [n for n in nums if abs(n) < 1]          # a strain, not a column number
    if nums:
        return Response(ANSWER, nums[0], text)
    return None


def _parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("decision") not in (ANSWER, ABSTAIN, FLAG):
        return None
    v = d.get("value")
    try:
        v = float(v) if v is not None else None
    except (TypeError, ValueError):
        v = None
    return Response(d["decision"], v, str(d.get("explanation", "")))


VERIFY_HINT = " Before reporting a value, verify that the measurement is reliable."


class OllamaAgent:
    def __init__(self, model="llama3.2:3b", host="http://localhost:11434", max_turns=6, hint=False):
        self.model, self.host, self.max_turns = model, host, max_turns
        self.system = SYSTEM + (VERIFY_HINT if hint else "")
        self.name = f"{model}" + (" + verify hint" if hint else "")
        self.transcripts = {}

    def _chat(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "tools": TOOL_SPECS,
                           "stream": False, "options": {"temperature": 0, "seed": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]

    def respond(self, task):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task.question}]
        calls = []
        for _ in range(self.max_turns):
            msg = self._chat(messages)
            messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break
            for tc in tool_calls:
                name = tc["function"]["name"]
                calls.append(name)
                out = TOOLS[name](task.image) if name in TOOLS else {"error": f"unknown tool {name}"}
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
        first_reply = messages[-1].get("content", "")
        resp = _parse(first_reply)
        format_retry = False
        if resp is None:
            # One reminder of the required format, as most agent harnesses do.
            format_retry = True
            messages.append({"role": "user", "content": "Reply with only the JSON object."})
            messages.append(self._chat(messages))
            resp = _parse(messages[-1].get("content", ""))
        prose = False
        if resp is None:
            prose = True
            resp = _from_prose(messages[-1].get("content", "")) or _from_prose(first_reply)
        invalid = resp is None
        if invalid:
            resp = Response(INVALID, None, "unusable final reply")
        self.transcripts[task.task_id] = {
            "tool_calls": calls, "first_reply": first_reply, "final": messages[-1].get("content", ""), "invalid": invalid,
            "format_retry": format_retry, "read_from_prose": prose,
            "decision": resp.decision, "value": resp.value,
        }
        return resp
