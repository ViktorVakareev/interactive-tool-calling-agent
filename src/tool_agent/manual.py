"""The tool-calling loop written out by hand - the core of the lab.

1. bind the tools to the model           llm.bind_tools(tools)
2. send the chat history                 response = llm_with_tools.invoke(history)
3. parse the tool calls                  name, args, id = call["name"], call["args"], call["id"]
4. run each tool                         result = tool_map[name].invoke(args)
5. return the result with the call id    ToolMessage(content=str(result), tool_call_id=id)
6. repeat until no more tool calls       -> the last AIMessage is the answer

`ToolCallingAgent` wraps this loop in a class (the lab's agent, fixed and generalized);
`TipAgent` is the lab's Exercise 3. LangChain's `create_agent` (see agent.py) runs the
same loop for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .tools import ALL_TOOLS, TIP_TOOLS, tool_map

StepCallback = Callable[[str, Any], None]


@dataclass
class LoopResult:
    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    steps: int = 0
    stopped: bool = False  # True if max_steps was reached

    @property
    def tool_calls(self) -> list[dict]:
        results = {m.tool_call_id: m.content for m in self.messages if isinstance(m, ToolMessage)}
        return [{"name": c["name"], "args": c["args"], "result": results.get(c["id"])}
                for m in self.messages if isinstance(m, AIMessage) for c in m.tool_calls]


def run_tool_call(call: dict, tools_by_name: dict) -> ToolMessage:
    """Execute one tool call and wrap the result. Unknown tools and bad arguments
    become error messages for the model instead of exceptions."""
    tool = tools_by_name.get(call["name"])
    if tool is None:
        content = f"Error: unknown tool '{call['name']}'. Available: {', '.join(tools_by_name)}."
    else:
        try:
            content = str(tool.invoke(call["args"]))
        except Exception as exc:  # validation errors, bad types, ...
            content = f"Error: {type(exc).__name__}: {exc}"
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])


def run_tool_loop(llm: BaseChatModel, query: str, tools: Sequence = ALL_TOOLS, max_steps: int = 5,
                  history: Optional[list[BaseMessage]] = None, on_step: Optional[StepCallback] = None) -> LoopResult:
    """Run the manual loop until the model answers without asking for a tool."""
    llm_with_tools = llm.bind_tools(list(tools))
    tools_by_name = tool_map(tools)
    messages: list[BaseMessage] = [*(history or []), HumanMessage(content=query)]
    _emit(on_step, "human", messages[-1])

    for step in range(1, max_steps + 1):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            _emit(on_step, "answer", response)
            return LoopResult(answer=_text(response), messages=messages, steps=step)
        _emit(on_step, "tool_calls", response)
        for call in response.tool_calls:  # several calls in one response are allowed
            tool_message = run_tool_call(call, tools_by_name)
            messages.append(tool_message)
            _emit(on_step, "tool_result", tool_message)

    return LoopResult(answer="Stopped: too many tool-calling steps.", messages=messages, steps=max_steps, stopped=True)


class ToolCallingAgent:
    """The lab's agent class: owns the tools and runs the loop for each query.

    Fixes vs. the lab: `response.contet` typo, only the first tool call handled,
    a single round only (no chaining), globals instead of instance state.
    """

    def __init__(self, llm: BaseChatModel, tools: Sequence = ALL_TOOLS, max_steps: int = 5):
        self.llm = llm
        self.tools = list(tools)
        self.max_steps = max_steps

    def run(self, query: str, on_step: Optional[StepCallback] = None) -> str:
        return self.run_detailed(query, on_step).answer

    def run_detailed(self, query: str, on_step: Optional[StepCallback] = None) -> LoopResult:
        return run_tool_loop(self.llm, query, self.tools, self.max_steps, on_step=on_step)


class TipAgent(ToolCallingAgent):
    """Exercise 3: an agent that only knows `calculate_tip`."""

    def __init__(self, llm: BaseChatModel, max_steps: int = 3):
        super().__init__(llm, TIP_TOOLS, max_steps)


def _emit(callback: Optional[StepCallback], kind: str, message: Any) -> None:
    if callback:
        callback(kind, message)


def _text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)
