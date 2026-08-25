"""The agent: an LLM that orchestrates repeated calls to one arithmetic tool.

No framework -- just a loop.  The model is given a single tool that does one
binary operation, so evaluating ``9 / 3 - 4 * 5`` forces it to plan:
``9/3 -> 3``, ``4*5 -> 20``, ``3-20 -> -17``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..utils.calc_tool import TOOL_SPEC, run_tool
from .llm import LLMBackend, ToolCall, get_backend

__all__ = ["Step", "AgentResult", "CalculatorAgent", "SYSTEM_PROMPT"]


SYSTEM_PROMPT = """\
You are a careful arithmetic agent. You cannot do mental arithmetic; the only \
way you are allowed to obtain a numeric result is by calling the `calculate` \
tool, which performs exactly ONE binary operation (a op b) per call.

Rules:
1. Never state a number you computed yourself. Every arithmetic result in your
   answer must come from a `calculate` call.
2. Break the expression into single binary operations and call the tool once
   per operation, feeding earlier results into later calls.
3. Respect the usual precedence: parentheses first, then * and /, then + and -,
   evaluating operators of equal precedence left to right.
   Example: for "9 / 3 - 4 * 5" call 9/3 -> 3, then 4*5 -> 20, then 3-20 -> -17.
4. Subtraction and division are not commutative: keep the operands in order.
5. A tool result beginning with "ERROR:" means your call was invalid. Read it,
   fix the call, and try again.
6. When every operation is done, reply in plain text with the final result.
   Keep it to one short line, e.g. "9 / 3 - 4 * 5 = -17".
7. If the request is not an arithmetic problem you can solve with +, -, * and /,
   say so plainly instead of guessing.
"""


@dataclass
class Step:
    """One executed tool call."""

    tool: str
    arguments: Any
    result: str

    def __str__(self) -> str:
        if isinstance(self.arguments, dict) and {"a", "op", "b"} <= set(self.arguments):
            expr = f"{self.arguments['a']} {self.arguments['op']} {self.arguments['b']}"
        else:
            expr = repr(self.arguments)
        return f"{expr} = {self.result}"


@dataclass
class AgentResult:
    """What the agent produced, plus the reasoning trail that produced it."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    elapsed: float = 0.0
    stopped_early: bool = False

    def __str__(self) -> str:
        return self.answer


class CalculatorAgent:
    """An LLM plus one arithmetic tool.

    >>> agent = CalculatorAgent(provider="openai")        # doctest: +SKIP
    >>> agent.run("2*8 + 1").answer                       # doctest: +SKIP
    '2 * 8 + 1 = 17'
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_turns: int = 12,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: int = 0,
        on_step: Callable[[int, "Step"], None] | None = None,
        **backend_options: Any,
    ) -> None:
        self.backend = backend or get_backend(provider, model, **backend_options)
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        #: 0 = silent, 1 = print each tool call, 2 = also print each API round-trip.
        self.verbose = int(verbose)
        #: Optional callback fired after every tool call: ``on_step(index, step)``.
        self.on_step = on_step
        self.tools = [TOOL_SPEC]

    # -- progress ---------------------------------------------------------- #

    @staticmethod
    def _say(message: str) -> None:
        """Print immediately -- unflushed output makes a live run look hung."""
        print(message, flush=True)

    # -- the loop ---------------------------------------------------------- #

    def run(self, question: str) -> AgentResult:
        """Answer one arithmetic question, calling the tool as many times as needed."""
        history: list[dict[str, Any]] = [{"role": "user", "content": question}]
        steps: list[Step] = []
        started = time.monotonic()

        for turn in range(1, self.max_turns + 1):
            if self.verbose >= 2:
                self._say(f"  ... turn {turn}: asking {self.backend.model}")
            reply = self.backend.complete(self.system_prompt, history, self.tools)
            history.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": reply.tool_calls,
                }
            )

            if not reply.tool_calls:
                answer = (reply.text or "").strip() or "(the model returned no answer)"
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    history=history,
                    turns=turn,
                    elapsed=time.monotonic() - started,
                )

            for call in reply.tool_calls:
                step = self._execute(call)
                steps.append(step)
                if self.verbose >= 1:
                    self._say(f"  [{len(steps)}] {step}")
                if self.on_step is not None:
                    self.on_step(len(steps), step)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": step.result,
                    }
                )

        return AgentResult(
            answer=(
                f"Gave up after {self.max_turns} turns without a final answer. "
                f"Last computed value: {steps[-1].result if steps else 'none'}."
            ),
            steps=steps,
            history=history,
            turns=self.max_turns,
            elapsed=time.monotonic() - started,
            stopped_early=True,
        )

    # -- tool dispatch ----------------------------------------------------- #

    @staticmethod
    def _execute(call: ToolCall) -> Step:
        return Step(tool=call.name, arguments=call.arguments, result=run_tool(call.name, call.arguments))

    def __repr__(self) -> str:
        return f"<CalculatorAgent backend={self.backend!r}>"
