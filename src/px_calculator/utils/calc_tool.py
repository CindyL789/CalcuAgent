"""The single tool the agent is allowed to use: one binary arithmetic operation.

The agent never computes anything itself.  It decomposes an expression such as
``9 / 3 - 4 * 5`` into a sequence of binary steps and calls :func:`calculate`
for each one, feeding earlier results into later calls.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "OPERATORS",
    "CalcError",
    "calculate",
    "format_number",
    "run_tool",
    "TOOL_NAME",
    "TOOL_SPEC",
]

TOOL_NAME = "calculate"

#: Operator symbol -> (aliases accepted from the model, human-readable name).
OPERATORS: dict[str, tuple[tuple[str, ...], str]] = {
    "+": (("add", "plus", "sum"), "addition"),
    "-": (("sub", "subtract", "minus"), "subtraction"),
    "*": (("mul", "multiply", "times", "x", "×"), "multiplication"),
    "/": (("div", "divide", "over", "÷"), "division"),
}

_ALIASES: dict[str, str] = {
    alias: symbol
    for symbol, (aliases, _) in OPERATORS.items()
    for alias in (symbol, *aliases)
}


class CalcError(ValueError):
    """Raised when the tool is called with arguments it cannot honour."""


def _coerce_number(value: Any, field: str) -> float:
    """Accept int/float, or a numeric string like ``"3"`` / ``"-2.5"``."""
    if isinstance(value, bool):  # bool is an int subclass -- reject explicitly
        raise CalcError(f"{field!r} must be a number, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise CalcError(f"{field!r} is not a number: {value!r}") from None
    raise CalcError(f"{field!r} must be a number, got {type(value).__name__}")


def _coerce_op(op: Any) -> str:
    if not isinstance(op, str):
        raise CalcError(f"'op' must be a string, got {type(op).__name__}")
    symbol = _ALIASES.get(op.strip().lower())
    if symbol is None:
        raise CalcError(f"unknown operator {op!r}; use one of + - * /")
    return symbol


def calculate(a: Any, op: Any, b: Any) -> float:
    """Apply one binary arithmetic operation and return the result.

    >>> calculate(2, "*", 8)
    16.0
    >>> calculate(9, "/", 3)
    3.0
    """
    x = _coerce_number(a, "a")
    y = _coerce_number(b, "b")
    symbol = _coerce_op(op)

    if symbol == "+":
        return x + y
    if symbol == "-":
        return x - y
    if symbol == "*":
        return x * y
    # symbol == "/"
    if y == 0:
        raise CalcError("division by zero is undefined")
    return x / y


def format_number(value: float) -> str:
    """Render a result without a pointless trailing ``.0``."""
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(round(value, 12))


# --------------------------------------------------------------------------- #
# Provider-neutral tool declaration.  llm.py translates this into whatever
# shape each vendor's API expects.
# --------------------------------------------------------------------------- #

TOOL_SPEC: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Perform ONE binary arithmetic operation on two numbers and return the "
        "result. Call it repeatedly to evaluate a composite expression, feeding "
        "the result of one call into the next."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The left-hand operand."},
            "op": {
                "type": "string",
                "enum": ["+", "-", "*", "/"],
                "description": "The operator to apply.",
            },
            "b": {"type": "number", "description": "The right-hand operand."},
        },
        "required": ["a", "op", "b"],
        "additionalProperties": False,
    },
}


def run_tool(name: str, arguments: Any) -> str:
    """Dispatch a tool call from the model and return a string for the model.

    Errors come back as text rather than exceptions so the model can see what
    it did wrong and correct itself on the next turn.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc})."
    if not isinstance(arguments, dict):
        return "ERROR: arguments must be a JSON object with keys a, op, b."

    if name != TOOL_NAME:
        return f"ERROR: unknown tool {name!r}; the only tool is {TOOL_NAME!r}."

    missing = [k for k in ("a", "op", "b") if k not in arguments]
    if missing:
        return f"ERROR: missing argument(s): {', '.join(missing)}."

    try:
        result = calculate(arguments["a"], arguments["op"], arguments["b"])
    except CalcError as exc:
        return f"ERROR: {exc}"
    return format_number(result)
