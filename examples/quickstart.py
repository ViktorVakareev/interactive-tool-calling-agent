"""Quickstart with LangChain's create_agent.

    python examples/quickstart.py offline    # or gemini / openai
"""

import sys

from dotenv import load_dotenv

from tool_agent import ask, build_agent, final_answer, get_llm, tool_calls

load_dotenv()
agent = build_agent(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))
for question in ["one plus 2", "What is 3+2 and 10*4?", "How much should I tip on $60 at 20%?"]:
    response = ask(agent, question)
    print(f"\nQ: {question}")
    for call in tool_calls(response):
        print(f"   🔧 {call['name']}({call['args']}) -> {call['result']}")
    print(f"A: {final_answer(response)}")
