"""Offline rule-based model + LangChain create_agent; provider wiring."""

import pytest
from langchain_core.messages import AIMessage

from conftest import calls
from tool_agent import TipAgent, ToolCallingAgent, ask, build_agent, final_answer, get_llm, tool_calls
from tool_agent.offline import OfflineToolModel, parse_arithmetic, parse_tip, words_to_digits


@pytest.mark.parametrize("text, expected", [
    ("What is 3 + 2?", ("add", {"a": 3, "b": 2})), ("one plus 2", ("add", {"a": 1, "b": 2})),
    ("one - 2", ("subtract", {"a": 1, "b": 2})), ("three times two", ("multiply", {"a": 3, "b": 2})),
    ("10 / 4", ("divide", {"a": 10, "b": 4})), ("Divide 10 by 4", ("divide", {"a": 10, "b": 4})),
    ("Subtract 5 from 20", ("subtract", {"a": 20, "b": 5})), ("Add 1.5 and 2", ("add", {"a": 1.5, "b": 2})),
    ("multiply -2 by 3", ("multiply", {"a": -2, "b": 3})), ("hello", None), ("add 5", None),
])
def test_parse_arithmetic(text, expected):
    assert parse_arithmetic(text) == expected


def test_parse_arithmetic_with_previous_result():
    assert parse_arithmetic("multiply by 4", previous=5) == ("multiply", {"a": 5, "b": 4})


@pytest.mark.parametrize("text, expected", [
    ("How much should I tip on $60 at 20%?", {"total_bill": 60, "tip_percent": 20}),
    ("What's a 15% tip on a $45.50 dinner?", {"total_bill": 45.5, "tip_percent": 15}),
    ("Tip on $120 at 18%, split between 4 people", {"total_bill": 120, "tip_percent": 18, "split_between": 4}),
    ("tip for a bill of 80 at twenty percent", {"total_bill": 80, "tip_percent": 20}),
    ("tip please", None), ("What is 3 + 2?", None),
])
def test_parse_tip(text, expected):
    assert parse_tip(text) == expected


def test_words_to_digits():
    assert words_to_digits("three times Two") == "3 times 2"


@pytest.mark.parametrize("query, expected_calls, answer_part", [
    ("What is 3 + 2?", [("add", "5.0")], "3 + 2 = 5"),
    ("Add 3 and 2, then multiply by 4", [("add", "5.0"), ("multiply", "20.0")], "result is 20"),
    ("What is 3+2 and 10*4?", [("add", "5.0"), ("multiply", "40.0")], "10 × 4 = 40"),
    ("Tip on $120 at 18%, split between 4 people", [("calculate_tip", None)], "$35.40 each for 4 people"),
    ("Divide 10 by 0", [("divide", "Error: cannot divide by zero.")], "couldn't complete"),
    ("Hi there", [], "no tool needed"),
])
@pytest.mark.parametrize("engine", ["manual", "create_agent"])
def test_offline_model_end_to_end_both_engines(engine, query, expected_calls, answer_part):
    if engine == "manual":
        result = ToolCallingAgent(OfflineToolModel()).run_detailed(query)
        made, answer = result.tool_calls, result.answer
    else:
        response = ask(build_agent(OfflineToolModel()), query)
        made, answer = tool_calls(response), final_answer(response)
    assert [c["name"] for c in made] == [n for n, _ in expected_calls]
    for call, (_, result) in zip(made, expected_calls):
        if result is not None:
            assert call["result"] == result
    assert answer_part in answer


def test_tip_agent_offline():
    assert TipAgent(OfflineToolModel()).run("How much should I tip on $60 at 20%?") == \
        "A tip of 20% on $60.00 is $12.00, so the total is $72.00."


def test_create_agent_with_scripted_model_and_history(scripted_model):
    agent = build_agent(scripted_model(calls(("add", {"a": 1, "b": 2})), AIMessage(content="3"), AIMessage(content="It was 3.")))
    first = ask(agent, "1+2")
    second = ask(agent, "What was it?", history=first["messages"])
    assert tool_calls(first)[0]["result"] == "3.0"
    assert tool_calls(second) == [] and final_answer(second) == "It was 3."


def test_final_answer_content_blocks():
    assert final_answer({"messages": [AIMessage(content=[{"type": "text", "text": "x"}, "y"])]}) == "xy"


def test_get_llm_providers(monkeypatch):
    assert isinstance(get_llm("offline"), OfflineToolModel)
    with pytest.raises(ValueError):
        get_llm("nope")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert type(get_llm("gemini")).__name__ == "ChatGoogleGenerativeAI"
    assert type(get_llm("openai")).__name__ == "ChatOpenAI"
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    assert isinstance(get_llm(), OfflineToolModel)


def test_tool_schemas_convert_for_gemini_and_openai():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    from tool_agent import ALL_TOOLS

    declared = [f.name for t in convert_to_genai_function_declarations(ALL_TOOLS) for f in (t.function_declarations or [])]
    assert declared == [t.name for t in ALL_TOOLS]
    assert convert_to_openai_tool(ALL_TOOLS[-1])["function"]["parameters"]["required"] == ["total_bill", "tip_percent"]
