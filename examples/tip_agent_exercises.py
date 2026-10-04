"""The lab's three exercises, solved.

    python examples/tip_agent_exercises.py offline
"""

import sys

from dotenv import load_dotenv

from tool_agent import TipAgent, calculate_tip, get_llm, run_tool_loop

load_dotenv()
llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else None)

print("Exercise 1 - call the tool directly")
print("   calculate_tip(total_bill=120, tip_percent=15) ->", calculate_tip.invoke({"total_bill": 120, "tip_percent": 15}))

print("\nExercise 2 - one tool-calling round trip with the LLM")
result = run_tool_loop(llm, "How much should I tip on $60 at 20%?", tools=[calculate_tip])
for call in result.tool_calls:
    print(f"   tool: {call['name']}({call['args']}) -> {call['result']}")
print("   answer:", result.answer)

print("\nExercise 3 - TipAgent automates it")
agent = TipAgent(llm)
for question in ["How much should I tip on $60 at 20%?",
                 "What's a 15% tip on a $45.50 dinner?",
                 "Tip on $120 at 18%, split between 4 people"]:
    print(f"   Q: {question}\n   A: {agent.run(question)}")
