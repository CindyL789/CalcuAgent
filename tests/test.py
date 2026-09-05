# -- Add your code below -- #
"""Offline tests: the arithmetic tool, the agent loop, and the message adapters.

No API key and no network needed -- the LLM is replaced by a scripted fake.

    PYTHONPATH=src python3 tests/test.py
"""

from __future__ import annotations

import unittest

from px_calculator.models.agent import CalculatorAgent
from px_calculator.models.llm import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    AnthropicBackend,
    LLMBackend,
    LLMReply,
    OpenAIBackend,
    ToolCall,
    get_backend,
)
from px_calculator.utils.calc_tool import (
    TOOL_NAME,
    CalcError,
    calculate,
    format_number,
    run_tool,
)


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #


class TestCalculate(unittest.TestCase):
    def test_four_operations(self):
        self.assertEqual(calculate(2, "+", 3), 5)
        self.assertEqual(calculate(10, "-", 4), 6)
        self.assertEqual(calculate(3, "*", 3), 9)
        self.assertEqual(calculate(4, "/", 10), 0.4)

    def test_order_matters(self):
        self.assertEqual(calculate(3, "-", 20), -17)
        self.assertEqual(calculate(1, "/", 4), 0.25)

    def test_accepts_aliases_and_numeric_strings(self):
        self.assertEqual(calculate("2", "times", "8"), 16)
        self.assertEqual(calculate(7, "MINUS", 2), 5)

    def test_rejects_bad_input(self):
        with self.assertRaises(CalcError):
            calculate(1, "^", 2)
        with self.assertRaises(CalcError):
            calculate(1, "/", 0)
        with self.assertRaises(CalcError):
            calculate("apple", "+", 1)
        with self.assertRaises(CalcError):
            calculate(True, "+", 1)

    def test_format_number(self):
        self.assertEqual(format_number(16.0), "16")
        self.assertEqual(format_number(0.4), "0.4")
        self.assertEqual(format_number(-17.0), "-17")


class TestRunTool(unittest.TestCase):
    def test_happy_path(self):
        self.assertEqual(run_tool(TOOL_NAME, {"a": 2, "op": "*", "b": 8}), "16")

    def test_json_string_arguments(self):
        self.assertEqual(run_tool(TOOL_NAME, '{"a": 9, "op": "/", "b": 3}'), "3")

    def test_errors_come_back_as_text(self):
        for bad in (
            {"a": 1, "op": "/", "b": 0},
            {"a": 1, "op": "%", "b": 2},
            {"a": 1, "op": "+"},
            "not json",
        ):
            self.assertTrue(run_tool(TOOL_NAME, bad).startswith("ERROR:"), bad)
        self.assertTrue(run_tool("sqrt", {"a": 4}).startswith("ERROR:"))


# --------------------------------------------------------------------------- #
# The agent loop, driven by a scripted fake LLM
# --------------------------------------------------------------------------- #


class FakeBackend(LLMBackend):
    """Replays a script of replies and records what it was sent."""

    name = "fake"

    def __init__(self, script):
        super().__init__(model="fake-1")
        self.script = list(script)
        self.seen_histories = []

    def complete(self, system, history, tools):
        self.seen_histories.append([dict(item) for item in history])
        return self.script.pop(0)


def _call(cid, a, op, b):
    return ToolCall(id=cid, name=TOOL_NAME, arguments={"a": a, "op": op, "b": b})


class TestAgentLoop(unittest.TestCase):
    def test_single_operation(self):
        backend = FakeBackend(
            [
                LLMReply(tool_calls=[_call("c1", 2, "+", 3)]),
                LLMReply(text="2 + 3 = 5"),
            ]
        )
        result = CalculatorAgent(backend).run("2+3")
        self.assertEqual(result.answer, "2 + 3 = 5")
        self.assertEqual([s.result for s in result.steps], ["5"])
        self.assertEqual(result.turns, 2)
        self.assertFalse(result.stopped_early)

    def test_composite_expression_chains_results(self):
        backend = FakeBackend(
            [
                LLMReply(tool_calls=[_call("c1", 9, "/", 3)]),
                LLMReply(tool_calls=[_call("c2", 4, "*", 5)]),
                LLMReply(tool_calls=[_call("c3", 3, "-", 20)]),
                LLMReply(text="9 / 3 - 4 * 5 = -17"),
            ]
        )
        result = CalculatorAgent(backend).run("9/3 - 4*5")
        self.assertEqual([s.result for s in result.steps], ["3", "20", "-17"])
        self.assertEqual(result.answer, "9 / 3 - 4 * 5 = -17")
        # every tool result was fed back to the model
        last = backend.seen_histories[-1]
        self.assertEqual([m["content"] for m in last if m["role"] == "tool"], ["3", "20", "-17"])

    def test_parallel_tool_calls_in_one_turn(self):
        backend = FakeBackend(
            [
                LLMReply(tool_calls=[_call("c1", 9, "/", 3), _call("c2", 4, "*", 5)]),
                LLMReply(tool_calls=[_call("c3", 3, "-", 20)]),
                LLMReply(text="-17"),
            ]
        )
        result = CalculatorAgent(backend).run("9/3 - 4*5")
        self.assertEqual([s.result for s in result.steps], ["3", "20", "-17"])

    def test_model_can_recover_from_a_tool_error(self):
        backend = FakeBackend(
            [
                LLMReply(tool_calls=[_call("c1", 1, "^", 2)]),
                LLMReply(tool_calls=[_call("c2", 1, "*", 2)]),
                LLMReply(text="1 * 2 = 2"),
            ]
        )
        result = CalculatorAgent(backend).run("1*2")
        self.assertTrue(result.steps[0].result.startswith("ERROR:"))
        self.assertEqual(result.steps[1].result, "2")
        self.assertEqual(result.answer, "1 * 2 = 2")

    def test_max_turns_guard(self):
        backend = FakeBackend([LLMReply(tool_calls=[_call("c", 1, "+", 1)])] * 5)
        result = CalculatorAgent(backend, max_turns=3).run("loop forever")
        self.assertTrue(result.stopped_early)
        self.assertEqual(len(result.steps), 3)


# --------------------------------------------------------------------------- #
# Provider adapters (pure translation -- no SDK, no network)
# --------------------------------------------------------------------------- #

HISTORY = [
    {"role": "user", "content": "2*8 + 1"},
    {"role": "assistant", "content": None, "tool_calls": [_call("c1", 2, "*", 8)]},
    {"role": "tool", "tool_call_id": "c1", "name": TOOL_NAME, "content": "16"},
    {"role": "assistant", "content": None, "tool_calls": [_call("c2", 16, "+", 1)]},
    {"role": "tool", "tool_call_id": "c2", "name": TOOL_NAME, "content": "17"},
]


class TestOpenAIAdapter(unittest.TestCase):
    def test_messages(self):
        messages = OpenAIBackend._to_messages("SYS", HISTORY)
        self.assertEqual(messages[0], {"role": "system", "content": "SYS"})
        self.assertEqual([m["role"] for m in messages[1:]],
                         ["user", "assistant", "tool", "assistant", "tool"])
        call = messages[2]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], TOOL_NAME)
        self.assertIn('"op": "*"', call["function"]["arguments"])
        self.assertEqual(messages[3]["tool_call_id"], "c1")

    def test_tools(self):
        tools = OpenAIBackend._to_tools([{"name": "x", "description": "d", "parameters": {}}])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "x")


class TestAnthropicAdapter(unittest.TestCase):
    def test_messages(self):
        messages = AnthropicBackend._to_messages(HISTORY)
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "c1")

    def test_consecutive_tool_results_merge_into_one_turn(self):
        history = [
            {"role": "user", "content": "9/3 - 4*5"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("c1", 9, "/", 3), _call("c2", 4, "*", 5)],
            },
            {"role": "tool", "tool_call_id": "c1", "name": TOOL_NAME, "content": "3"},
            {"role": "tool", "tool_call_id": "c2", "name": TOOL_NAME, "content": "20"},
        ]
        messages = AnthropicBackend._to_messages(history)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(len(messages[-1]["content"]), 2)

    def test_tools(self):
        tools = AnthropicBackend._to_tools(
            [{"name": "x", "description": "d", "parameters": {"type": "object"}}]
        )
        self.assertEqual(tools[0]["input_schema"], {"type": "object"})


class TestRegistry(unittest.TestCase):
    def test_known_providers(self):
        self.assertEqual(get_backend("ollama").name, "ollama")
        self.assertEqual(get_backend("anthropic").name, "anthropic")
        self.assertEqual(get_backend("openai", model="gpt-4o").model, "gpt-4o")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_backend("nope")

    def test_ollama_defaults_to_localhost(self):
        self.assertEqual(get_backend("ollama")._resolve_base_url(),
                         "http://localhost:11434/v1")


# --------------------------------------------------------------------------- #
# Timeouts, retries, and live progress
# --------------------------------------------------------------------------- #


class TestTimeoutsAndRetries(unittest.TestCase):
    def test_defaults_are_applied(self):
        backend = get_backend("openai")
        self.assertEqual(backend.timeout, DEFAULT_TIMEOUT)
        self.assertEqual(backend.max_retries, DEFAULT_MAX_RETRIES)

    def test_defaults_are_finite(self):
        # The whole point: a stalled connection must not hang the agent forever.
        self.assertIsNotNone(DEFAULT_TIMEOUT)
        self.assertGreater(DEFAULT_TIMEOUT, 0)

    def test_explicit_overrides(self):
        backend = get_backend("anthropic", timeout=5, max_retries=0)
        self.assertEqual(backend.timeout, 5)
        self.assertEqual(backend.max_retries, 0)

    def test_explicit_none_means_wait_forever(self):
        # Saying nothing gets the default; saying None is a deliberate opt-out.
        self.assertIsNone(get_backend("openai", timeout=None).timeout)
        self.assertEqual(get_backend("openai").timeout, DEFAULT_TIMEOUT)


class TestProgressReporting(unittest.TestCase):
    def _script(self):
        return FakeBackend(
            [
                LLMReply(tool_calls=[_call("c1", 9, "/", 3)]),
                LLMReply(tool_calls=[_call("c2", 3, "+", 6)]),
                LLMReply(tool_calls=[_call("c3", 9, "-", 20)]),
                LLMReply(text="9/3 + 6 - 20 = -11"),
            ]
        )

    def test_on_step_callback_fires_per_tool_call(self):
        seen = []
        agent = CalculatorAgent(self._script(), on_step=lambda i, s: seen.append((i, s.result)))
        agent.run("9/3 + 6 - 20")
        self.assertEqual(seen, [(1, "3"), (2, "9"), (3, "-11")])

    def test_verbose_levels(self):
        import contextlib
        import io

        for level, wants_steps, wants_turns in ((0, False, False), (1, True, False), (2, True, True)):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                CalculatorAgent(self._script(), verbose=level).run("9/3 + 6 - 20")
            out = buffer.getvalue()
            self.assertEqual("9 / 3 = 3" in out, wants_steps, f"verbose={level}")
            self.assertEqual("turn 1" in out, wants_turns, f"verbose={level}")

    def test_result_records_elapsed_time(self):
        result = CalculatorAgent(self._script()).run("9/3 + 6 - 20")
        self.assertGreaterEqual(result.elapsed, 0.0)
        self.assertEqual(result.answer, "9/3 + 6 - 20 = -11")


class TestCLIWiring(unittest.TestCase):
    def test_verbosity_mapping(self):
        from px_calculator.cli import _verbosity, build_parser

        parse = build_parser().parse_args
        self.assertEqual(_verbosity(parse(["2+3"])), 1)          # tool calls by default
        self.assertEqual(_verbosity(parse(["-t", "2+3"])), 2)
        self.assertEqual(_verbosity(parse(["-q", "2+3"])), 0)
        self.assertEqual(_verbosity(parse(["-q", "-t", "2+3"])), 0)  # quiet wins

    def test_timeout_and_retry_flags_parse(self):
        from px_calculator.cli import build_parser

        args = build_parser().parse_args(["--timeout", "5", "--max-retries", "0", "2+3"])
        self.assertEqual(args.timeout, 5.0)
        self.assertEqual(args.max_retries, 0)

    def test_timeout_zero_becomes_no_timeout(self):
        from px_calculator.cli import _make_agent, build_parser

        args = build_parser().parse_args(["--timeout", "0", "-p", "ollama", "2+3"])
        self.assertIsNone(_make_agent(args).backend.timeout)
        args = build_parser().parse_args(["-p", "ollama", "2+3"])
        self.assertEqual(_make_agent(args).backend.timeout, DEFAULT_TIMEOUT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
