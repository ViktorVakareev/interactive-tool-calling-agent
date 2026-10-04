"""Command line: ask in plain English, watch the tool-calling loop.

    tool-agent "three times two" --provider offline --trace
    tool-agent "tip on $60 at 20%, split between 3" -p offline --engine manual -t
    tool-agent -p gemini                      # interactive chat ('exit' to quit)
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .agent import PROVIDERS, ask, build_agent, final_answer, get_llm
from .manual import ToolCallingAgent

_COLOR = sys.stdout.isatty() or bool(os.getenv("FORCE_COLOR"))


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def print_step(kind: str, msg) -> None:
    """Pretty-print one step of the loop (used live by the manual engine)."""
    if kind == "human":
        print(f"{_c('1;36', 'Human')}        {msg.content}")
    elif kind == "tool_calls":
        calls = msg.tool_calls
        label = "AI → tool" if len(calls) == 1 else f"AI → {len(calls)} tools"
        for call in calls:
            print(f"{_c('1;33', f'{label:<12}')} {_c('33', call['name'])}({call['args']})  {_c('2', 'id=' + call['id'])}")
    elif kind == "tool_result":
        print(f"{_c('1;35', 'ToolMessage')}  {_c('35', msg.name or '')} → {msg.content}  {_c('2', 'id=' + msg.tool_call_id)}")
    elif kind == "answer":
        print(f"{_c('1;32', 'AI answer')}    {final_answer({'messages': [msg]})}")


def print_trace(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            print_step("human", msg)
        elif isinstance(msg, AIMessage):
            print_step("tool_calls" if msg.tool_calls else "answer", msg)
        elif isinstance(msg, ToolMessage):
            print_step("tool_result", msg)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="tool-agent", description="Interactive tool-calling agent (LangChain)")
    parser.add_argument("question", nargs="*", help="question; omit for interactive chat")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "gemini"))
    parser.add_argument("-m", "--model", default=None, help="override the model name")
    parser.add_argument("-e", "--engine", choices=("agent", "manual"), default="agent",
                        help="agent = LangChain create_agent; manual = the hand-written loop from the lab")
    parser.add_argument("-t", "--trace", action="store_true", help="show every message in the loop")
    args = parser.parse_args(argv)

    llm = get_llm(args.provider, args.model)
    if args.engine == "manual":
        manual = ToolCallingAgent(llm)

        def run(question: str) -> None:
            if args.trace:
                manual.run(question, on_step=print_step)
            else:
                print(_c("1;32", "Agent: ") + manual.run(question))
    else:
        agent = build_agent(llm)
        history: list = []

        def run(question: str) -> None:
            nonlocal history
            response = ask(agent, question, history)
            if args.trace:
                print_trace(response["messages"][len(history):])
            else:
                print(_c("1;32", "Agent: ") + final_answer(response))
            history = response["messages"]

    if args.question:
        run(" ".join(args.question))
        return 0

    print(f"🧮 Tool-calling agent {_c('2', f'(provider: {args.provider}, engine: {args.engine})')} - type 'exit' to quit")
    while True:
        try:
            question = input(_c("1;36", "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        run(question)
    print("bye")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
