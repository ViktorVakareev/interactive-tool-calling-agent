"""Interactive tool-calling agent - the manual LangChain tool loop and its prebuilt equivalent."""

from .agent import SYSTEM_PROMPT, ask, build_agent, final_answer, get_llm, tool_calls
from .manual import LoopResult, TipAgent, ToolCallingAgent, run_tool_call, run_tool_loop
from .tools import ALL_TOOLS, ARITHMETIC_TOOLS, TIP_TOOLS, add, calculate_tip, divide, multiply, subtract, tool_map

__version__ = "1.0.0"

__all__ = [
    "ALL_TOOLS", "ARITHMETIC_TOOLS", "LoopResult", "SYSTEM_PROMPT", "TIP_TOOLS", "TipAgent", "ToolCallingAgent",
    "add", "ask", "build_agent", "calculate_tip", "divide", "final_answer", "get_llm", "multiply",
    "run_tool_call", "run_tool_loop", "subtract", "tool_calls", "tool_map",
]
