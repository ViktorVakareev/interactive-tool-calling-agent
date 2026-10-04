"""The lab's core lesson, one step at a time: how a tool call really works.

    python examples/manual_loop_step_by_step.py            # provider from .env (default gemini)
    python examples/manual_loop_step_by_step.py offline    # no key, no cost
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from tool_agent import ARITHMETIC_TOOLS, get_llm, tool_map

load_dotenv()
llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else None)

# 1) Bind the tools: the model now knows their names, descriptions and argument schemas
llm_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)
tools_by_name = tool_map(ARITHMETIC_TOOLS)

# 2) Start the chat history with the user's question
chat_history = [HumanMessage(content="What is 3 + 2?")]
print("1. Human:", chat_history[0].content)

# 3) The model doesn't answer - it asks for a tool
response = llm_with_tools.invoke(chat_history)
chat_history.append(response)
call = response.tool_calls[0]
print(f"2. AIMessage.tool_calls[0]: name={call['name']!r} args={call['args']} id={call['id']!r}")

# 4) YOU run the tool, looked up by name
result = tools_by_name[call["name"]].invoke(call["args"])
print(f"3. tool_map[{call['name']!r}].invoke({call['args']}) -> {result}")

# 5) Return the result, linked to the request by tool_call_id
chat_history.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
print(f"4. ToolMessage(content={str(result)!r}, tool_call_id={call['id']!r})")

# 6) Send the whole history back: now the model can answer
answer = llm_with_tools.invoke(chat_history)
print("5. Final answer:", answer.content)
print("\nchat_history:", " -> ".join(type(m).__name__ for m in [*chat_history, answer]))
