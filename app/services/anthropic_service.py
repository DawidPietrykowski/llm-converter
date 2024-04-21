import json
import os
from typing import Iterator, AnyStr

import anthropic
import requests
from dotenv import load_dotenv

from app.models import TargetApiBackend, AvailableModel, OpenAICompletionRequest, OpenAICompletionResponse, \
    OpenAICompletionChunkResponse

load_dotenv()

class AnthropicCompletionRequest:
    def __init__(self, model: str, max_tokens: int | None, tools, messages, system_prompt: str | None = None,
                 temperature: float | None = None, top_p: float | None = None):
        self.model = model
        self.max_tokens = max_tokens if max_tokens is not None else 4096
        self.tools = tools
        self.messages = messages
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

    @classmethod
    def from_openai_request(cls, completionRequest: OpenAICompletionRequest) -> 'AnthropicCompletionRequest':
        openai_tools = completionRequest.tools
        openai_messages = completionRequest.messages

        # convert tools from OpenAI to Anthropic format
        anthropic_tools = None
        if openai_tools is not None:
            anthropic_tools = [_format_openai_tool_to_anthropic_tool(tool) for tool in openai_tools]

        # convert messages from OpenAI to Anthropic format
        anthropic_chat = _format_openai_messages_to_anthropic_chat(openai_messages)

        anthropic_request = AnthropicCompletionRequest(
            model=completionRequest.model,
            max_tokens=completionRequest.max_tokens,
            tools=anthropic_tools,
            messages=anthropic_chat.messages,
            system_prompt=anthropic_chat.system_prompt,
            temperature=completionRequest.temperature,
            top_p=completionRequest.top_p
        )
        return anthropic_request

    def make_api_request(self, base_url: str, api_key: str):
        url = base_url + "/v1/messages"

        # Anthropic API version and beta (required for tools)
        ANTHROPIC_VERSION = "2023-06-01"
        ANTHROPIC_BETA = "tools-2024-04-04"
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_BETA,
        }

        # send the request
        data = self.to_dict()
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response.json()

    def to_dict(self) -> dict:
        args = {
            "model": self.model
        }
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens
        if self.tools is not None:
            args["tools"] = self.tools
        if self.messages is not None:
            args["messages"] = self.messages
        if self.system_prompt is not None:
            args["system"] = self.system_prompt
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        return args


class AnthropicChat:
    """Anthropic chat object."""

    def __init__(self, messages, functions, system_prompt: str | None = None):
        self.messages = messages
        self.tools = functions
        self.system_prompt = system_prompt


def _format_openai_messages_to_anthropic_chat(openai_messages) -> AnthropicChat:
    system_prompt: str | None = None

    anthropic_messages = []

    tool_result_messages = []

    functions = []

    index = 0
    for message in openai_messages:
        if message["role"] == "system":
            if system_prompt is None:
                system_prompt = ""
            system_prompt += message["content"] + "\n"
        elif message["role"] == "tool":
            tool_result_messages.append(
                {
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                    "append_id": index
                }
            )
        elif message["role"] == "function":
            functions.append(_format_openai_function_to_anthropic_tool(message["content"]))
        else:
            if len(tool_result_messages) != 0:
                content = []
                for tool_result_message in tool_result_messages:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_result_message["tool_call_id"],
                        "content": tool_result_message["content"]
                    })

                result_message = {
                    "role": "user",
                    "content": content
                }

                anthropic_messages.append(result_message)

                tool_result_messages = []

            if message.get("tool_calls"):
                new_content = []
                for tool_call in message["tool_calls"]:
                    new_content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"])
                    })
                new_message = {
                    "role": "assistant",
                    "content": new_content
                }
                anthropic_messages.append(new_message)

            else:
                anthropic_messages.append(message)
        index += 1

    if len(tool_result_messages) != 0:
        content = []
        for tool_result_message in tool_result_messages:
            content.append({
                "type": "tool_result",
                "tool_use_id": tool_result_message["tool_call_id"],
                "content": tool_result_message["content"]
            })

        result_message = {
            "role": "user",
            "content": content
        }

        anthropic_messages.append(result_message)

    if len(anthropic_messages) == 0:
        anthropic_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": system_prompt}]
            }
        )
        system_prompt = None

    # Anthropic requires a user message at the start
    if anthropic_messages[0]["role"] == "assistant":
        anthropic_messages.insert(0, {
            "role": "user",
            "content": [{"type": "text", "text": "<no input>"}]
        })

    return AnthropicChat(messages=anthropic_messages, functions=functions, system_prompt=system_prompt)


def _format_openai_tool_to_anthropic_tool(input_json: dict) -> dict:
    output_json = {
        "name": input_json["function"]["name"],
        "description": input_json["function"]["description"],
        "input_schema": input_json["function"]["parameters"],
    }

    return output_json


def _format_openai_function_to_anthropic_tool(input_json: dict) -> dict:
    output_json = {
        "name": input_json["name"],
        "description": input_json["description"],
        "input_schema": input_json["parameters"],
    }

    return output_json


def _format_anthropic_message_to_openai_response(message) -> OpenAICompletionResponse:
    tool_messages = []

    for content in message["content"]:
        if content["type"] == "tool_use":
            tool_messages.append(
                {
                    "id": content["id"],
                    "name": content["name"],
                    "input": content["input"]
                }
            )
    if len(tool_messages) != 0:
        tool_calls = []
        for tool_message in tool_messages:
            tool_calls.append(
                {
                    "id": tool_message["id"],
                    "type": "function",
                    "function": {
                        "name": tool_message["name"],
                        "arguments": json.dumps(tool_message["input"])
                    }
                }
            )

        choices = [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": tool_calls
                },
            }
        ]
    else:
        finish_reason = "length" if message["stop_reason"] == "max_tokens" else "stop"
        choices = [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {
                    "content": message["content"][0]["text"],
                    "role": "assistant"
                },
            }
        ]

    usage = message["usage"]
    openai_response = OpenAICompletionResponse(
        completion_id=message["id"],
        model=message["model"],
        choices=choices,
        completionTokens=usage["output_tokens"],
        promptTokens=usage["input_tokens"]
    )

    return openai_response


def _stream_request(completionRequest: OpenAICompletionRequest, client) -> Iterator[AnyStr]:
    sent_role = False

    last_id = None

    stream_args = AnthropicCompletionRequest.from_openai_request(completionRequest).to_dict()

    with client.messages.stream(**stream_args) as stream:
        for text in stream.text_stream:
            delta = {"content": text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            last_id = stream.current_message_snapshot.id
            chunk_message = str(OpenAICompletionChunkResponse(
                completion_id=last_id,
                model=completionRequest.model,
                choices=[{"index": 0, "delta": delta, "finish_reason": None}]
            ).to_json())

            yield "data:" + chunk_message + "\n\n"

    chunk_message = str(OpenAICompletionChunkResponse(
        completion_id=last_id,
        model=completionRequest.model,
        choices=[
            {"index": 0, "delta": {}, "finish_reason": "stop"}]
    ).to_json())

    yield "data:" + chunk_message + "\n\n"


class AnthropicApiBackend(TargetApiBackend):
    def __init__(self, base_url: str = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com"),
                 api_key: str = os.environ.get("ANTHROPIC_API_KEY")):
        """Load the available models from data/anthropic_models.json."""

        models = []
        with open('data/anthropic_models.json') as f:
            data = json.load(f)
            for model in data:
                models.append(AvailableModel(**model))
        super().__init__(base_url, api_key, models)

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        anthropic_request = AnthropicCompletionRequest.from_openai_request(completionRequest)

        anthropic_response = anthropic_request.make_api_request(
            self.base_url, self.get_api_key(completionRequest, pass_api_key))

        if anthropic_response["type"] == "error":
            raise Exception("Anthropic API returned an error: " + str(anthropic_response))

        return _format_anthropic_message_to_openai_response(anthropic_response)

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        client = anthropic.Anthropic(
            base_url=self.base_url,
            api_key=self.get_api_key(completionRequest, pass_api_key),
        )

        return _stream_request(completionRequest, client)
