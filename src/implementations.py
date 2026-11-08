import json
from typing import Any, Iterable, Optional, cast
from pydantic import BaseModel

import httpx2
from openai import OpenAI
from openai._types import Omit, omit, NotGiven, not_given
from openai.types.chat import ChatCompletionMessageToolCallUnion
from openai.types.chat.completion_create_params import ResponseFormat

from base_llm import BaseLLM
from messages import Message, StructuredT, ToolCall, Roles


class LLMOpenai(BaseLLM, provider="openai"):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "none"
    ):
        super().__init__(
            model=model,
            api_key=api_key
        )
        self.base_url: str = base_url
        self._set_client()

    def _set_client(self) -> None:
        self.client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] | Omit = omit,
        tool_choice: str | Omit = omit,
        tools: list[dict] | Omit = omit,
        response_format: type[StructuredT] | ResponseFormat | Omit = omit,
        timeout: float | httpx2.Timeout | None | NotGiven = not_given,
        **kwargs
    ) -> Message[StructuredT]:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Pydantic models must go through `parse()`, which converts the
            # model into a JSON schema and parses the response back into it.
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                timeout=timeout,
                extra_body=kwargs
            )
            message = completion.choices[0].message
            return Message[StructuredT](
                role=Roles.LLM,
                content=message.content,
                structured_content=message.parsed,
                tool_calls=self._parse_tool_calls(message.tool_calls)
            )

        # Raw response formats (e.g. {"type": "json_object"}) are valid for
        # `create()`, but not for `parse()`.
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=cast(ResponseFormat | Omit, response_format),
            timeout=timeout,
            extra_body=kwargs
        )
        message = completion.choices[0].message
        return Message[StructuredT](
            role=Roles.LLM,
            content=message.content,
            tool_calls=self._parse_tool_calls(message.tool_calls)
        )

    def _parse_tool_calls(
        self,
        tool_calls: Iterable[ChatCompletionMessageToolCallUnion] | None,
    ) -> list[ToolCall]:
        return [
            ToolCall(
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
            for call in (tool_calls or [])
            if call.type == "function"
        ]


class LLMManual(BaseLLM, provider="manual"):
    """Interactive implementation that plays the LLM role instead of calling one.

    It prints the exact messages that would be sent to the provider, then uses
    the text typed by the developer as the assistant response.
    """

    def __init__(
        self,
        model: str = "manual",
        api_key: str = "none",
    ):
        super().__init__(model=model, api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        *args,
        **kwargs,
    ) -> Message:
        self._print_messages(messages)

        response_format = kwargs.get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            content = input("Assistant response (JSON matching the schema): ")
            return Message(
                role=Roles.LLM,
                content=content,
                structured_content=response_format.model_validate_json(content),
            )

        content = input("Assistant response: ")
        return Message(role=Roles.LLM, content=content)

    @staticmethod
    def _print_messages(messages: list[dict]) -> None:
        print("\n" + "=" * 80)
        print("LLMManual: messages the LLM would see")
        print("=" * 80)
        for index, message in enumerate(messages):
            print(f"\n--- message {index} ---")
            print(json.dumps(message, indent=2, ensure_ascii=False, default=str))
        print("=" * 80)


def create_llm(provider: str, **kwargs: Any) -> BaseLLM:
    """Create a registered LLM implementation by provider name.

    Scripts can keep importing this single function while the concrete
    implementation (and its constructor arguments) changes.
    """
    return BaseLLM.create(provider, **kwargs)

