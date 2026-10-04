"""The tools the LLM can call - typed, documented, and safe to call with bad input.

Each tool takes structured, typed arguments (the LLM fills them in from the user's
words, e.g. "three times two" -> multiply(a=3, b=2)). Name, docstring and type hints
become the JSON schema the model sees.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers: a + b."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a: a - b. For "subtract 5 from 20" use a=20, b=5."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b."""
    return a * b


@tool
def divide(a: float, b: float) -> Any:
    """Divide a by b: a / b. Returns an error message if b is 0."""
    if b == 0:
        return "Error: cannot divide by zero."
    return a / b


@tool
def calculate_tip(total_bill: float, tip_percent: float, split_between: int = 1) -> dict[str, Any]:
    """Calculate the tip for a bill, the total to pay, and each person's share.

    Args:
        total_bill: bill amount before tip, e.g. 60 or 45.50
        tip_percent: tip as a percentage, e.g. 20 for 20%
        split_between: number of people sharing the bill (default 1)

    Returns:
        {"tip": ..., "total": ..., "per_person": ...} rounded to cents,
        or {"error": "..."} for invalid input.
    """
    if total_bill < 0 or tip_percent < 0:
        return {"error": "Bill and tip percent must not be negative."}
    if split_between < 1:
        return {"error": "split_between must be at least 1."}
    tip = round(total_bill * tip_percent / 100, 2)
    total = round(total_bill + tip, 2)
    return {"tip": tip, "total": total, "per_person": round(total / split_between, 2)}


ARITHMETIC_TOOLS = [add, subtract, multiply, divide]
TIP_TOOLS = [calculate_tip]
ALL_TOOLS = [*ARITHMETIC_TOOLS, *TIP_TOOLS]


def tool_map(tools) -> dict:
    """{"add": add, ...} - how a tool call's *name* is turned back into a function."""
    return {t.name: t for t in tools}
