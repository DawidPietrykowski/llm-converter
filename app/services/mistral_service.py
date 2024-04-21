import json
import os
from typing import Iterator, AnyStr

from dotenv import load_dotenv

from app.models import TargetApiBackend, AvailableModel, OpenAICompletionRequest, OpenAICompletionResponse, \
    OpenAICompletionChunkResponse

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatCompletionResponse

load_dotenv()

class MistralCompletionRequest:
    def __init__(self, model: str, max_tokens: int | None, tools, messages,
                 temperature: float | None = None, top_p: float | None = None, tool_choice: str = "auto"):
        self.model = model
        self.max_tokens = max_tokens
        self.tools = tools
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.tool_choice = tool_choice

    @classmethod
    def from_openai_request(cls, completionRequest: OpenAICompletionRequest) -> 'MistralCompletionRequest':
        mistral_messages = _format_openai_messages_to_mistral_messages(completionRequest.messages)

        tool_choice = completionRequest.tool_choice
        if tool_choice == "required":
            tool_choice = "any"

        mistral_request = MistralCompletionRequest(
            model=completionRequest.model,
            max_tokens=completionRequest.max_tokens,
            tools=completionRequest.tools,  # No conversion needed
            messages=mistral_messages.messages,
            temperature=completionRequest.temperature,
            top_p=completionRequest.top_p,
            tool_choice=tool_choice
        )
        return mistral_request

    def make_api_request(self, base_url: str, api_key: str) -> ChatCompletionResponse:
        client = MistralClient(api_key=api_key, endpoint=base_url)

        additional_args = {}
        if self.temperature is not None:
            additional_args["temperature"] = self.temperature
        if self.top_p is not None:
            additional_args["top_p"] = self.top_p
        if self.max_tokens is not None:
            additional_args["max_tokens"] = self.max_tokens
        if self.tool_choice is not None:
            additional_args["tool_choice"] = self.tool_choice

        return client.chat(self.messages, self.model, self.tools, **additional_args)


class MistralChat:
    def __init__(self, messages):
        self.messages = messages


def _format_openai_messages_to_mistral_messages(openai_messages) -> MistralChat:
    mistral_messages = [x for x in openai_messages if x["role"] != "function"]

    return MistralChat(messages=mistral_messages)


def _format_mistral_response_to_openai_response(response: ChatCompletionResponse) -> OpenAICompletionResponse:
    choices_json = []
    for choice in response.choices:
        choices_json.append({
            "index": choice.index,
            "message": {
                "role": choice.message.role,
                "content": choice.message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                    "type": tool_call.type
                } for tool_call in choice.message.tool_calls] if choice.message.tool_calls else None
            },
            "finish_reason": choice.finish_reason
        })
    return OpenAICompletionResponse(
        completion_id=response.id,
        model=response.model,
        choices=choices_json,
        completionTokens=response.usage.completion_tokens,
        promptTokens=response.usage.prompt_tokens)


def _stream_request(completionRequest: OpenAICompletionRequest, client: MistralClient) -> Iterator[AnyStr]:
    mistral_messages = completionRequest.messages

    stream_args = {
        "max_tokens": completionRequest.max_tokens,
        "messages": mistral_messages,
        "model": completionRequest.model
    }

    for chunk in client.chat_stream(**stream_args):
        chunk_message = str(OpenAICompletionChunkResponse(
            completion_id=chunk.id,
            model=completionRequest.model,
            choices=[{"index": 0, "delta": {
                "content": chunk.choices[0].delta.content,
                "role": "assistant"
            }, "finish_reason": None}]
        ).to_json())

        yield "data:" + chunk_message + "\n\n"


class MistralApiBackend(TargetApiBackend):
    def __init__(self, base_url: str = os.environ.get("MISTRAL_API_URL", "https://api.mistral.ai"),
                 api_key: str = os.environ.get("MISTRAL_API_KEY")):
        """Load the available models from data/mistral_models.json."""

        models = []
        with open('data/mistral_models.json') as f:
            data = json.load(f)
            for model in data:
                models.append(AvailableModel(**model))
        super().__init__(base_url, api_key, models)

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        request = MistralCompletionRequest.from_openai_request(completionRequest)
        response = request.make_api_request(self.base_url, self.get_api_key(completionRequest, pass_api_key))

        return _format_mistral_response_to_openai_response(response)

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        client = MistralClient(api_key=self.get_api_key(completionRequest, pass_api_key), endpoint=self.base_url)

        return _stream_request(completionRequest, client)
