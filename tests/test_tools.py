import pytest

from tool_agent.tools import ALL_TOOLS, add, calculate_tip, divide, multiply, subtract, tool_map


@pytest.mark.parametrize("tool, a, b, expected", [
    (add, 3, 2, 5), (add, 1.5, 2.25, 3.75), (subtract, 1, 2, -1), (subtract, 20, 5, 15),
    (multiply, 3, 2, 6), (multiply, -2, 2.5, -5), (divide, 10, 4, 2.5), (divide, -9, 3, -3)])
def test_arithmetic(tool, a, b, expected):
    assert tool.invoke({"a": a, "b": b}) == expected


def test_divide_by_zero_is_returned_not_raised():
    assert divide.invoke({"a": 1, "b": 0}) == "Error: cannot divide by zero."


@pytest.mark.parametrize("args, expected", [
    ({"total_bill": 120, "tip_percent": 15}, {"tip": 18.0, "total": 138.0, "per_person": 138.0}),
    ({"total_bill": 60, "tip_percent": 20}, {"tip": 12.0, "total": 72.0, "per_person": 72.0}),
    ({"total_bill": 45.50, "tip_percent": 15}, {"tip": 6.83, "total": 52.33, "per_person": 52.33}),
    ({"total_bill": 120, "tip_percent": 18, "split_between": 4}, {"tip": 21.6, "total": 141.6, "per_person": 35.4}),
    ({"total_bill": 0, "tip_percent": 20}, {"tip": 0.0, "total": 0.0, "per_person": 0.0}),
])
def test_calculate_tip(args, expected):
    assert calculate_tip.invoke(args) == expected


@pytest.mark.parametrize("args, message", [
    ({"total_bill": -5, "tip_percent": 10}, "must not be negative"),
    ({"total_bill": 50, "tip_percent": -1}, "must not be negative"),
    ({"total_bill": 50, "tip_percent": 10, "split_between": 0}, "at least 1"),
])
def test_calculate_tip_rejects_bad_input(args, message):
    assert message in calculate_tip.invoke(args)["error"]


def test_bad_argument_types_raise_validation_errors():
    with pytest.raises(Exception):
        add.invoke({"a": "three", "b": 2})


def test_schemas_the_llm_sees():
    assert [t.name for t in ALL_TOOLS] == ["add", "subtract", "multiply", "divide", "calculate_tip"]
    assert set(add.args) == {"a", "b"} and add.args["a"]["type"] == "number"
    assert set(calculate_tip.args) == {"total_bill", "tip_percent", "split_between"}
    assert calculate_tip.args["split_between"]["default"] == 1
    assert "subtract 5 from 20" in subtract.description and "a=20" in subtract.description
    assert tool_map(ALL_TOOLS)["divide"] is divide
