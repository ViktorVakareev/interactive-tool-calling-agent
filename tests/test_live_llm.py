"""Live evaluation with a real LLM - skipped unless an API key is set.  pytest -m live -v

Checks WHICH tool was called with WHICH arguments (and the tool result), not the wording.
"""

import math
import os

import pytest

from tool_agent import TipAgent, ToolCallingAgent, get_llm

PROVIDERS = [
    pytest.param("gemini", marks=pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")),
    pytest.param("openai", marks=pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
]
pytestmark = pytest.mark.live
CASES = [("one plus 2", "add", 3), ("one - 2", "subtract", -1), ("three times two", "multiply", 6)]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("query, tool, expected", CASES)
def test_lab_examples(provider, query, tool, expected):
    result = ToolCallingAgent(get_llm(provider)).run_detailed(query)
    call = result.tool_calls[0]
    assert call["name"] == tool and math.isclose(float(call["result"]), expected)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_tip_agent(provider):
    result = TipAgent(get_llm(provider)).run_detailed("How much should I tip on $60 at 20%?")
    call = result.tool_calls[0]
    assert call["name"] == "calculate_tip"
    assert float(call["args"]["total_bill"]) == 60 and float(call["args"]["tip_percent"]) == 20
    assert "12" in result.answer
