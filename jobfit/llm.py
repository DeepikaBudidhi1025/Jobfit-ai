"""Thin wrapper around any OpenAI-compatible chat API, with reliable JSON output."""

from __future__ import annotations

import json
import re
import time
from typing import Iterator, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


class ChatModel(Protocol):
    def complete(self, messages: list[dict], temperature: float | None = None) -> str: ...


class OpenAICompatibleLLM:
    """Works with Groq, OpenAI, Together, OpenRouter, Ollama (`http://localhost:11434/v1`)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        key = api_key or settings.api_key
        if not key:
            raise LLMError("No API key found. Set LLM_API_KEY in your .env file.")
        self.client = OpenAI(api_key=key, base_url=base_url or settings.base_url)
        self.model = model or settings.model
        self.temperature = settings.temperature if temperature is None else temperature
        self.max_retries = max_retries

    def complete(self, messages: list[dict], temperature: float | None = None) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # network / rate-limit errors -> exponential backoff
                last_err = err
                time.sleep(2**attempt)
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last_err}")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, stream=True
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and extra prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        candidate = text[start : end + 1]
    return json.loads(candidate)


def complete_json(llm: ChatModel, messages: list[dict], schema: Type[T], max_attempts: int = 3) -> T:
    """Ask for JSON matching a Pydantic schema; on bad output, feed the error back and retry."""
    schema_hint = json.dumps(schema.model_json_schema(), indent=None)
    convo = [
        *messages,
        {
            "role": "system",
            "content": "Respond with ONLY a single valid JSON object (no markdown, no commentary) "
            f"that conforms to this JSON schema:\n{schema_hint}",
        },
    ]
    last_err = ""
    for _ in range(max_attempts):
        raw = llm.complete(convo, temperature=0)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as err:
            last_err = str(err)[:800]
            convo = [
                *convo,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That was not valid. Error: {last_err}\nReturn corrected JSON only."},
            ]
    raise LLMError(f"Model did not return valid JSON for {schema.__name__}: {last_err}")
