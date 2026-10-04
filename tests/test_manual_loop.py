"""The hand-written loop from the lab, driven by a scripted fake model."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import calls
from tool_agent import TipAgent, ToolCallingAgent, run_tool_call, run_tool_loop
from tool_agent.tools import ALL_TOOLS, tool_map


def test_single_round_trip(scripted_model):
    llm = scripted_model(calls(("add", {"a": 3, "b": 2})), AIMessage(content="3 + 2 = 5"))
    result = run_tool_loop(llm, "What is 3 + 2?")
    assert [type(m) for m in result.messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    tool_msg = result.messages[2]
    assert tool_msg.content == "5.0" and tool_msg.tool_call_id == "call_0" and tool_msg.name == "add"
    assert result.answer == "3 + 2 = 5" and result.steps == 2 and not result.stopped
    # the model's 2nd call received the full history incl. the tool result
    assert [type(m) for m in llm.received[1]] == [HumanMessage, AIMessage, ToolMessage]


def test_parallel_tool_calls_in_one_response(scripted_model):
    llm = scripted_model(calls(("add", {"a": 3, "b": 2}), ("multiply", {"a": 10, "b": 4})), AIMessage(content="5 and 40"))
    result = run_tool_loop(llm, "3+2 and 10*4?")
    assert [(c["name"], c["result"]) for c in result.tool_calls] == [("add", "5.0"), ("multiply", "40.0")]
    ids = [m.tool_call_id for m in result.messages if isinstance(m, ToolMessage)]
    assert ids == ["call_0", "call_1"]  # each result linked to its request


def test_chained_calls_over_several_rounds(scripted_model):
    llm = scripted_model(calls(("add", {"a": 3, "b": 2})), calls(("multiply", {"a": 5, "b": 4})), AIMessage(content="20"))
    result = run_tool_loop(llm, "add 3 and 2 then multiply by 4")
    assert [c["name"] for c in result.tool_calls] == ["add", "multiply"] and result.steps == 3


def test_no_tool_needed_returns_direct_answer(scripted_model):
    result = run_tool_loop(scripted_model(AIMessage(content="Hello!")), "hi")
    assert result.answer == "Hello!" and result.tool_calls == [] and result.steps == 1


def test_max_steps_stops_endless_loops(scripted_model):
    llm = scripted_model(*[calls(("add", {"a": 1, "b": 1}))] * 5)
    result = run_tool_loop(llm, "loop forever", max_steps=3)
    assert result.stopped and result.steps == 3 and "too many" in result.answer


def test_unknown_tool_and_bad_args_become_error_messages():
    by_name = tool_map(ALL_TOOLS)
    unknown = run_tool_call({"name": "sqrt", "args": {"x": 4}, "id": "c1"}, by_name)
    assert unknown.content.startswith("Error: unknown tool 'sqrt'") and unknown.tool_call_id == "c1"
    bad = run_tool_call({"name": "add", "args": {"a": "three"}, "id": "c2"}, by_name)
    assert bad.content.startswith("Error: ValidationError")


def test_history_is_prepended(scripted_model):
    llm = scripted_model(AIMessage(content="ok"))
    result = run_tool_loop(llm, "and now?", history=[HumanMessage(content="earlier"), AIMessage(content="sure")])
    assert [m.content for m in result.messages[:3]] == ["earlier", "sure", "and now?"]


def test_on_step_callback_sees_every_step(scripted_model):
    seen = []
    llm = scripted_model(calls(("add", {"a": 1, "b": 2})), AIMessage(content="3"))
    run_tool_loop(llm, "1+2", on_step=lambda kind, msg: seen.append(kind))
    assert seen == ["human", "tool_calls", "tool_result", "answer"]


def test_tool_calling_agent_class(scripted_model):
    agent = ToolCallingAgent(scripted_model(calls(("multiply", {"a": 3, "b": 2})), AIMessage(content="6")))
    assert agent.run("three times two") == "6"


def test_tip_agent_only_has_the_tip_tool(scripted_model):
    llm = scripted_model(calls(("add", {"a": 1, "b": 2})), AIMessage(content="sorry"))
    result = TipAgent(llm).run_detailed("1+2")
    assert "unknown tool 'add'" in result.tool_calls[0]["result"]
    assert [t.name for t in TipAgent(llm).tools] == ["calculate_tip"]


def test_list_content_blocks_are_joined(scripted_model):
    llm = scripted_model(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
    assert run_tool_loop(llm, "x").answer == "ab"
