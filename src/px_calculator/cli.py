"""Command-line front end for the calculator agent."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from .models.agent import CalculatorAgent
from .models.llm import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, PROVIDERS

__all__ = ["main", "build_parser"]

_BANNER = "px-calculator agent -- type an expression, or 'quit' to leave."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="px-calculator",
        description=(
            "An LLM agent that evaluates arithmetic by calling a one-operation "
            "Python tool, repeatedly."
        ),
    )
    parser.add_argument(
        "expression",
        nargs="*",
        help="The expression to evaluate, e.g. '9/3 - 4*5'. Omit for an interactive session.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default=None,
        choices=sorted(PROVIDERS),
        help="LLM provider (default: $LLM_PROVIDER, else 'openai').",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name (default: $LLM_MODEL, else the provider's default).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL (handy for Ollama or a self-hosted endpoint).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="Safety cap on LLM round-trips per question (default: 12).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            f"Seconds to wait for one API response (default: {DEFAULT_TIMEOUT:g}, "
            "or $LLM_TIMEOUT). Use 0 to wait forever."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            f"How many times to retry a failed request "
            f"(default: {DEFAULT_MAX_RETRIES}, or $LLM_MAX_RETRIES)."
        ),
    )
    parser.add_argument(
        "-t",
        "--trace",
        action="store_true",
        help="Also show each API round-trip and the elapsed time.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print only the final answer (tool calls are shown by default).",
    )
    return parser


def _verbosity(args: argparse.Namespace) -> int:
    """0 = answer only, 1 = show tool calls (default), 2 = show round-trips too."""
    if args.quiet:
        return 0
    return 2 if args.trace else 1


def _make_agent(args: argparse.Namespace) -> CalculatorAgent:
    # --timeout 0 is the documented escape hatch for "wait indefinitely".
    options = {
        "timeout": args.timeout if args.timeout > 0 else None,
        "max_retries": args.max_retries,
    }
    if args.base_url:
        options["base_url"] = args.base_url
    return CalculatorAgent(
        provider=args.provider,
        model=args.model,
        max_turns=args.max_turns,
        verbose=_verbosity(args),
        **options,
    )


def _ask(agent: CalculatorAgent, question: str) -> None:
    result = agent.run(question)
    if agent.verbose >= 2:
        print(
            f"  ({len(result.steps)} tool call(s), {result.turns} turn(s), "
            f"{result.elapsed:.1f}s)",
            flush=True,
        )
    print(result.answer, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        agent = _make_agent(args)
        # Build the vendor client up front so a missing SDK or API key is
        # reported cleanly instead of blowing up mid-conversation.
        if hasattr(agent.backend, "client"):
            agent.backend.client()
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.expression:
        try:
            _ask(agent, " ".join(args.expression))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    print(_BANNER)
    print(
        f"provider: {agent.backend.name}   model: {agent.backend.model}   "
        f"timeout: {agent.backend.timeout or 'none'}s\n"
    )
    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            return 0
        try:
            _ask(agent, question)
        except Exception as exc:  # keep the REPL alive on transient API errors
            print(f"error: {exc}", file=sys.stderr)
