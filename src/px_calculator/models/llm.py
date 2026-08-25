"""A thin, framework-free abstraction over several LLM providers.

Everything above this module speaks one neutral vocabulary:

* a *history* is a list of plain dicts --
  ``{"role": "user", "content": str}``,
  ``{"role": "assistant", "content": str | None, "tool_calls": [ToolCall]}``,
  ``{"role": "tool", "tool_call_id": str, "name": str, "content": str}``
* a *tool spec* is ``{"name", "description", "parameters"}`` (see ``calc_tool``)
* a reply is an :class:`LLMReply`

Each backend translates that into whatever its vendor's API expects.  Vendor
SDKs are imported lazily, so you only need the one you actually use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "ToolCall",
    "LLMReply",
    "LLMBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "AnthropicBackend",
    "PROVIDERS",
    "register_provider",
    "get_backend",
]


#: Seconds to wait for one API response before giving up.  Without this a
#: dropped connection hangs the agent forever.  Override with $LLM_TIMEOUT.
DEFAULT_TIMEOUT: float = float(os.environ.get("LLM_TIMEOUT", "30"))

#: How many times the vendor SDK retries a failed request on its own.
DEFAULT_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "2"))

#: Distinguishes "caller said nothing" from an explicit ``timeout=None``
#: (which legitimately means "wait forever").
_UNSET: Any = object()


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: Any  # dict, or the raw JSON string if the model emitted junk


@dataclass
class LLMReply:
    """One assistant turn: some text, some tool calls, or both."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class LLMBackend:
    """Interface every provider implements."""

    name: str = "base"
    default_model: str = ""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float | None = _UNSET,
        max_retries: int | None = None,
        **options: Any,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL") or self.default_model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = DEFAULT_TIMEOUT if timeout is _UNSET else timeout
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        self.options = options
        self._client: Any = None

    def complete(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMReply:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model!r}>"


def _missing_sdk(package: str, provider: str) -> ImportError:
    return ImportError(
        f"The {provider!r} provider needs the {package!r} package. "
        f"Install it with:  pip install {package}"
    )


# --------------------------------------------------------------------------- #
# OpenAI (and every OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #


class OpenAIBackend(LLMBackend):
    """Chat Completions with native function calling."""

    name = "openai"
    default_model = "gpt-4o-mini"
    env_api_key = "OPENAI_API_KEY"
    env_base_url = "OPENAI_BASE_URL"
    fallback_api_key: str | None = None  # for keyless local servers

    def _resolve_api_key(self) -> str:
        key = self.api_key or os.environ.get(self.env_api_key) or self.fallback_api_key
        if not key:
            raise RuntimeError(
                f"No API key for provider {self.name!r}. "
                f"Set the {self.env_api_key} environment variable."
            )
        return key

    def _resolve_base_url(self) -> str | None:
        return self.base_url or os.environ.get(self.env_base_url) or None

    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise _missing_sdk("openai", self.name) from None
            self._client = OpenAI(
                api_key=self._resolve_api_key(),
                base_url=self._resolve_base_url(),
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    # -- translation ------------------------------------------------------- #

    @staticmethod
    def _to_messages(system: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import json

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for item in history:
            role = item["role"]
            if role == "user":
                messages.append({"role": "user", "content": item["content"]})
            elif role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": item.get("content") or None,
                }
                calls = item.get("tool_calls") or []
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments
                                if isinstance(call.arguments, str)
                                else json.dumps(call.arguments),
                            },
                        }
                        for call in calls
                    ]
                messages.append(entry)
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["tool_call_id"],
                        "content": item["content"],
                    }
                )
            else:  # pragma: no cover - guarded by the agent
                raise ValueError(f"unknown history role {role!r}")
        return messages

    @staticmethod
    def _to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"type": "function", "function": spec} for spec in tools]

    # -- call -------------------------------------------------------------- #

    def complete(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMReply:
        import json

        response = self.client().chat.completions.create(
            model=self.model,
            messages=self._to_messages(system, history),
            tools=self._to_tools(tools),
            temperature=self.temperature,
            **self.options,
        )
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            raw = call.function.arguments or "{}"
            try:
                arguments: Any = json.loads(raw)
            except json.JSONDecodeError:
                arguments = raw  # run_tool() will report it back to the model
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return LLMReply(text=message.content, tool_calls=calls, raw=response)


class OllamaBackend(OpenAIBackend):
    """A local Ollama server, reached through its OpenAI-compatible endpoint.

    Needs a tool-capable model (llama3.1, qwen2.5, mistral-nemo, ...).
    """

    name = "ollama"
    default_model = "llama3.1"
    env_api_key = "OLLAMA_API_KEY"
    env_base_url = "OLLAMA_BASE_URL"
    fallback_api_key = "ollama"  # the server ignores it, the SDK requires it

    def _resolve_base_url(self) -> str:
        return self.base_url or os.environ.get(self.env_base_url) or "http://localhost:11434/v1"


class OpenAICompatibleBackend(OpenAIBackend):
    """Any other OpenAI-shaped API (DeepSeek, Groq, Together, vLLM, ...).

    Point it at the endpoint with ``LLM_BASE_URL`` and ``LLM_API_KEY``.
    """

    name = "openai-compatible"
    default_model = ""
    env_api_key = "LLM_API_KEY"
    env_base_url = "LLM_BASE_URL"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API with native tool use."""

    name = "anthropic"
    default_model = "claude-sonnet-4-5"
    env_api_key = "ANTHROPIC_API_KEY"
    env_base_url = "ANTHROPIC_BASE_URL"
    max_tokens = 1024

    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise _missing_sdk("anthropic", self.name) from None
            key = self.api_key or os.environ.get(self.env_api_key)
            if not key:
                raise RuntimeError(
                    f"No API key for provider {self.name!r}. "
                    f"Set the {self.env_api_key} environment variable."
                )
            base_url = self.base_url or os.environ.get(self.env_base_url) or None
            self._client = Anthropic(
                api_key=key,
                base_url=base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    # -- translation ------------------------------------------------------- #

    @staticmethod
    def _to_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in history:
            role = item["role"]
            if role == "user":
                messages.append({"role": "user", "content": item["content"]})
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if item.get("content"):
                    blocks.append({"type": "text", "text": item["content"]})
                for call in item.get("tool_calls") or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments
                            if isinstance(call.arguments, dict)
                            else {},
                        }
                    )
                messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": item["tool_call_id"],
                    "content": item["content"],
                }
                # Consecutive tool results belong in a single user turn.
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
            else:  # pragma: no cover - guarded by the agent
                raise ValueError(f"unknown history role {role!r}")
        return messages

    @staticmethod
    def _to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
            for spec in tools
        ]

    # -- call -------------------------------------------------------------- #

    def complete(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMReply:
        response = self.client().messages.create(
            model=self.model,
            system=system,
            messages=self._to_messages(history),
            tools=self._to_tools(tools),
            max_tokens=self.options.pop("max_tokens", self.max_tokens),
            temperature=self.temperature,
            **self.options,
        )
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
        return LLMReply(
            text="\n".join(texts) if texts else None, tool_calls=calls, raw=response
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

PROVIDERS: dict[str, type[LLMBackend]] = {
    OpenAIBackend.name: OpenAIBackend,
    OllamaBackend.name: OllamaBackend,
    AnthropicBackend.name: AnthropicBackend,
    OpenAICompatibleBackend.name: OpenAICompatibleBackend,
}


def register_provider(name: str, backend: type[LLMBackend]) -> None:
    """Add your own backend: ``register_provider("gemini", GeminiBackend)``."""
    PROVIDERS[name.lower()] = backend


def get_backend(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """Build a backend by name.

    Falls back to ``$LLM_PROVIDER``, then to ``"openai"``.
    """
    key = (provider or os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    try:
        backend = PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {key!r}; known providers: {known}") from None
    return backend(model=model, **kwargs)
