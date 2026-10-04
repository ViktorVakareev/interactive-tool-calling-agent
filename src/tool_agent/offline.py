"""``OfflineToolModel`` - a deterministic, rule-based stand-in for an LLM.

It is **not an LLM**. It speaks the real tool-calling protocol (returns ``AIMessage``
with ``tool_calls``, reads ``ToolMessage`` results), so the manual loop, the agent
classes and ``create_agent`` all run for real without an API key or any cost.

It understands simple phrasing only:
- arithmetic with symbols or words: "3 + 2", "one plus 2", "three times two", "10 / 4"
- chains with "then":               "add 3 and 2, then multiply by 4"  (sequential calls)
- independent parts with "and":      "what is 3+2 and 10*4?"            (parallel calls)
- tips:                              "tip on $60 at 20%, split between 3"
Anything else gets a direct answer with no tool call.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)
NUM = r"-?\d+(?:\.\d+)?"
_SYMBOL_RE = re.compile(rf"({NUM})\s*([+\-*/x×÷])\s*({NUM})")
_SYMBOLS = {"+": "add", "-": "subtract", "*": "multiply", "x": "multiply", "×": "multiply", "/": "divide", "÷": "divide"}
_KEYWORDS = [
    ("divide", r"divide|divided|over|quotient"),
    ("multiply", r"multipl|times|product"),
    ("subtract", r"subtract|minus|less|difference|take away"),
    ("add", r"\badd\b|plus|sum|total of|\band\b"),
]
_SYMBOL_TEXT = {"add": "+", "subtract": "-", "multiply": "×", "divide": "÷"}


def words_to_digits(text: str) -> str:
    return _WORD_RE.sub(lambda m: str(_WORDS[m.group(1).lower()]), text)


def _num(text: str) -> float | int:
    """"3" -> 3, "2.5" -> 2.5 (keeps tool-call args readable)."""
    value = float(text)
    return int(value) if value.is_integer() else value


def _numbers(text: str) -> list[float | int]:
    return [_num(n) for n in re.findall(rf"(?<![\d.]){NUM}", text)]


def parse_arithmetic(clause: str, previous: Optional[float] = None) -> Optional[tuple[str, dict]]:
    """One clause -> ("multiply", {"a": 3, "b": 2}) or None if not understood."""
    text = words_to_digits(clause).replace(",", " ")
    m = _SYMBOL_RE.search(text)
    if m and previous is None:
        return _SYMBOLS[m.group(2).lower()], {"a": _num(m.group(1)), "b": _num(m.group(3))}
    nums = _numbers(text)
    for name, pattern in _KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            if previous is not None and len(nums) >= 1:
                return name, {"a": previous, "b": nums[0]}
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if name == "subtract" and re.search(r"\bfrom\b", text, re.IGNORECASE):
                    a, b = b, a  # "subtract 5 from 20" -> 20 - 5
                return name, {"a": a, "b": b}
    return None


def parse_tip(query: str) -> Optional[dict]:
    """"tip on $60 at 20%, split between 3" -> {"total_bill": 60, "tip_percent": 20, "split_between": 3}."""
    text = words_to_digits(query)
    if not re.search(r"\btip", text, re.IGNORECASE):
        return None
    pct = re.search(rf"({NUM})\s*(%|percent)", text, re.IGNORECASE)
    bill = re.search(rf"\$\s*({NUM})", text) or re.search(rf"(?:on|bill(?: of)?)\s+({NUM})", text, re.IGNORECASE)
    if not (pct and bill):
        return None
    args: dict[str, Any] = {"total_bill": _num(bill.group(1)), "tip_percent": _num(pct.group(1))}
    split = re.search(r"(?:split|between|among|for)\s+(?:\w+\s+)?(\d+)\s*(?:people|persons|ways|friends)?", text, re.IGNORECASE)
    if split and int(split.group(1)) > 1 and float(split.group(1)) not in (args["total_bill"], args["tip_percent"]):
        args["split_between"] = int(split.group(1))
    return args


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else f"{x:.4g}" if abs(x) < 1e6 else str(x)
    return str(x)


def _load(content: Any) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (TypeError, ValueError, SyntaxError):
            continue
    return content


def _call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


class OfflineToolModel(BaseChatModel):
    """Rule-based chat model that drives the real tools through real tool calls."""

    @property
    def _llm_type(self) -> str:
        return "offline-tool-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineToolModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[start].content)
        rounds = [m for m in messages[start:] if isinstance(m, AIMessage) and m.tool_calls]
        results = [m for m in messages[start:] if isinstance(m, ToolMessage)]
        message = self._next(query, rounds, results)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------ planning
    def _next(self, query: str, rounds: list[AIMessage], results: list[ToolMessage]) -> AIMessage:
        errors = [r.content for r in results if str(r.content).startswith("Error") or '"error"' in str(r.content)
                  or "'error'" in str(r.content)]
        if errors:
            return AIMessage(content=f"I couldn't complete that: {errors[-1]}")

        tip = parse_tip(query)
        if tip:
            if not rounds:
                return AIMessage(content="", tool_calls=[_call("calculate_tip", tip)])
            r = _load(results[-1].content)
            share = f" - {_money(r['per_person'])} each for {tip['split_between']} people" if tip.get("split_between") else ""
            return AIMessage(content=f"A tip of {_fmt(tip['tip_percent'])}% on {_money(tip['total_bill'])} is "
                                     f"{_money(r['tip'])}, so the total is {_money(r['total'])}{share}.")

        clauses = [c for c in re.split(r"\s*,?\s*\b(?:and\s+)?then\b\s*", query.rstrip("?.! "), flags=re.IGNORECASE) if c]
        if len(clauses) > 1:  # sequential chain: each step uses the previous result
            if len(rounds) < len(clauses):
                previous = _num(str(_load(results[-1].content))) if results else None
                plan = parse_arithmetic(clauses[len(rounds)], previous)
                if plan:
                    return AIMessage(content="", tool_calls=[_call(*plan)])
            if rounds:
                steps = ", then ".join(f"{c['name']}({_fmt(c['args']['a'])}, {_fmt(c['args']['b'])})"
                                       for r in rounds for c in r.tool_calls)
                return AIMessage(content=f"{steps} → the result is {_fmt(_load(results[-1].content))}.")

        parts = [p for p in re.split(r"\s*(?:;|\band\b(?=[^+\-*/]*\d\s*[+\-*/x×÷]))\s*", query) if p.strip()]
        plans = [parse_arithmetic(p) for p in parts] if len(parts) > 1 else []
        if len(plans) > 1 and all(plans) and all(_SYMBOL_RE.search(words_to_digits(p)) for p in parts):
            if not rounds:  # independent sub-questions -> parallel tool calls in ONE response
                return AIMessage(content="", tool_calls=[_call(*p) for p in plans])
            return AIMessage(content="; ".join(self._sentence(c, r) for c, r in zip(rounds[0].tool_calls, results)) + ".")

        plan = parse_arithmetic(query)
        if plan:
            if not rounds:
                return AIMessage(content="", tool_calls=[_call(*plan)])
            return AIMessage(content=self._sentence(rounds[-1].tool_calls[0], results[-1]) + ".")

        return AIMessage(content="Offline demo model: no tool needed for that. I can add, subtract, multiply, "
                                 "divide (\"3 + 2\", \"three times two\", \"add 3 and 2, then multiply by 4\") "
                                 "and calculate tips (\"tip on $60 at 20%\").")

    @staticmethod
    def _sentence(call: dict, result: ToolMessage) -> str:
        a, b = _fmt(call["args"]["a"]), _fmt(call["args"]["b"])
        return f"{a} {_SYMBOL_TEXT[call['name']]} {b} = {_fmt(_load(result.content))}"


def _money(x: Any) -> str:
    return f"${float(x):,.2f}"
