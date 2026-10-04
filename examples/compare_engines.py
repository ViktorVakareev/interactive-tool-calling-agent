"""Same questions, two engines: the lab's hand-written loop vs LangChain's create_agent.

    python examples/compare_engines.py offline
"""

import sys

from dotenv import load_dotenv

from tool_agent import ToolCallingAgent, ask, build_agent, final_answer, get_llm, tool_calls

load_dotenv()
provider = sys.argv[1] if len(sys.argv) > 1 else None
manual = ToolCallingAgent(get_llm(provider))
prebuilt = build_agent(get_llm(provider))

for question in ["three times two", "Add 3 and 2, then multiply by 4", "Tip on $80 at 18%"]:
    m = manual.run_detailed(question)
    p = ask(prebuilt, question)
    print(f"\nQ: {question}")
    print(f"   manual loop   : {[c['name'] for c in m.tool_calls]} -> {m.answer}")
    print(f"   create_agent  : {[c['name'] for c in tool_calls(p)]} -> {final_answer(p)}")
