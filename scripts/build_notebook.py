"""Generate and execute notebooks/walkthrough.ipynb (offline provider by default).

    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Interactive tool-calling agent — walkthrough\n\n"
       "How an LLM calls your Python functions, step by step, then the same thing automated. "
       "Runs **offline by default** (rule-based stand-in model: no API key, no cost). "
       "Set `PROVIDER = \"gemini\"` (free key) or `\"openai\"` for a real LLM."),
    code("PROVIDER = \"offline\"  # \"gemini\" | \"openai\" | \"offline\"\n\n"
         "from dotenv import load_dotenv\nload_dotenv()\n"
         "from langchain_core.messages import HumanMessage, ToolMessage\n"
         "from tool_agent import (ARITHMETIC_TOOLS, ALL_TOOLS, calculate_tip, tool_map, get_llm,\n"
         "                        run_tool_loop, ToolCallingAgent, TipAgent, build_agent, ask, final_answer, tool_calls)\n"
         "from tool_agent.cli import print_step, print_trace\n"
         "llm = get_llm(PROVIDER)"),
    md("## 1. A tool = function + name + docstring + type hints\nThat's all the model sees when deciding what to call."),
    code("for t in ALL_TOOLS:\n    print(f\"{t.name:14} {t.args}\")"),
    md("## 2. One tool call, by hand\nBind → the model returns a **tool call** (name, args, id) → you run it → send back a `ToolMessage` with the same id → the model answers."),
    code("llm_with_tools = llm.bind_tools(ARITHMETIC_TOOLS)\n"
         "chat_history = [HumanMessage(content=\"What is 3 + 2?\")]\n"
         "response = llm_with_tools.invoke(chat_history)\n"
         "chat_history.append(response)\n"
         "call = response.tool_calls[0]\nprint(\"tool call:\", call[\"name\"], call[\"args\"], call[\"id\"])"),
    code("result = tool_map(ARITHMETIC_TOOLS)[call[\"name\"]].invoke(call[\"args\"])\n"
         "chat_history.append(ToolMessage(content=str(result), tool_call_id=call[\"id\"]))\n"
         "answer = llm_with_tools.invoke(chat_history)\n"
         "print(\"tool result:\", result, \"| final answer:\", answer.content)"),
    md("## 3. The loop: keep going until the model stops asking for tools\nHandles several calls in one response and chains across rounds."),
    code("run_tool_loop(llm, \"Add 3 and 2, then multiply by 4\", on_step=print_step);"),
    code("run_tool_loop(llm, \"What is 3+2 and 10*4?\", on_step=print_step);"),
    md("## 4. The agent class (the lab's `ToolCallingAgent`)"),
    code("agent = ToolCallingAgent(llm)\nfor q in [\"one plus 2\", \"one - 2\", \"three times two\"]:\n    print(f\"{q:16} -> {agent.run(q)}\")"),
    md("## 5. Exercises\n**1** — a `calculate_tip` tool. **2** — one tool-calling round trip with it. **3** — a `TipAgent` that automates the process."),
    code("# Exercise 1\nprint(calculate_tip.invoke({\"total_bill\": 120, \"tip_percent\": 15}))"),
    code("# Exercise 2\nresult = run_tool_loop(llm, \"How much should I tip on $60 at 20%?\", tools=[calculate_tip])\n"
         "print(result.tool_calls[0][\"name\"], result.tool_calls[0][\"args\"], \"->\", result.answer)"),
    code("# Exercise 3\ntip_agent = TipAgent(llm)\nfor q in [\"How much should I tip on $60 at 20%?\", \"Tip on $120 at 18%, split between 4 people\"]:\n"
         "    print(tip_agent.run(q))"),
    md("## 6. The same loop, prebuilt: LangChain `create_agent`"),
    code("prebuilt = build_agent(llm)\nprint_trace(ask(prebuilt, \"Tip on $80 at 18%\")[\"messages\"])"),
    md("## 7. Evaluate: check the tool calls, not the wording"),
    code("checks = [(\"one plus 2\", \"add\", 3.0), (\"one - 2\", \"subtract\", -1.0), (\"three times two\", \"multiply\", 6.0)]\n"
         "passed = 0\nfor q, tool, expected in checks:\n    c = agent.run_detailed(q).tool_calls[0]\n"
         "    ok = c[\"name\"] == tool and float(c[\"result\"]) == expected\n    passed += ok\n"
         "    print(\"✅\" if ok else \"❌\", q, \"->\", c[\"name\"], c[\"result\"])\nprint(f\"\\n{passed}/{len(checks)} checks passed\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, ROOT / "notebooks" / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
