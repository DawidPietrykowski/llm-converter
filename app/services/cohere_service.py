import json
import os
from typing import Iterator, AnyStr, Sequence

import cohere
from cohere.types import NonStreamedChatResponse, Tool, ChatRequestToolResultsItem
from dotenv import load_dotenv

from app.models import TargetApiBackend, AvailableModel, OpenAICompletionRequest, OpenAICompletionResponse, \
    OpenAICompletionChunkResponse

load_dotenv()

class CohereCompletionRequest:
    def __init__(self, model: str, max_tokens: int | None, tools: Sequence[Tool] | None,
                 tool_results: Sequence[ChatRequestToolResultsItem] | None, chat_history, message: str,
                 preamble: str | None, temperature: float | None = None, top_p: float | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.tools = tools
        self.tool_results = tool_results
        self.chat_history = chat_history
        self.message = message
        self.preamble = preamble
        self.temperature = temperature
        self.top_p = top_p

    @classmethod
    def from_openai_request(cls, completionRequest: OpenAICompletionRequest) -> 'CohereCompletionRequest':
        openai_tools = completionRequest.tools

        # convert openai tool format to cohere tool format
        cohere_tools: Sequence[Tool] | None = None
        if openai_tools is not None:
            cohere_tools = [_format_openai_tool_to_cohere_tool(tool) for tool in openai_tools]

        # convert openai messages to cohere chat format
        cohere_chat = _format_openai_messages_to_cohere_chat(completionRequest.messages)

        cohere_request = CohereCompletionRequest(
            model=completionRequest.model,
            max_tokens=min(completionRequest.max_tokens, 4000) if completionRequest.max_tokens is not None else None,
            tools=cohere_tools,
            tool_results=cohere_chat.tool_results,
            chat_history=cohere_chat.chat_history,
            message=cohere_chat.last_message,
            preamble=cohere_chat.preamble,
            temperature=completionRequest.temperature,
            top_p=completionRequest.top_p
        )
        return cohere_request

    def make_api_request(self, base_url: str, api_key: str) -> NonStreamedChatResponse:
        client = cohere.Client(api_key, base_url=base_url)

        response = client.chat(**self.to_dict())
        return response

    def to_dict(self) -> dict:
        args = {}
        if self.max_tokens is not None:
            args["max_tokens"] = self.max_tokens
        if self.tools is not None and len(self.tools) > 0:
            args["tools"] = self.tools
        if self.tool_results is not None and len(self.tool_results) > 0:
            args["tool_results"] = self.tool_results
        if self.preamble is not None:
            args["preamble"] = self.preamble
        if self.chat_history is not None and len(self.chat_history) > 0:
            args["chat_history"] = self.chat_history
        if self.message is not None:
            args["message"] = self.message
        if self.model is not None:
            args["model"] = self.model
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["p"] = self.top_p
        return args


class CohereChat:
    def __init__(self, chat_history, last_message: str, preamble: str | None = None, tool_results=None):
        self.chat_history = chat_history
        self.last_message = last_message
        self.preamble = preamble
        self.tool_results = tool_results


def _format_openai_messages_to_cohere_chat(openai_messages) -> CohereChat:
    chat_history = []

    preamble = None

    tool_results = []

    for message in openai_messages:
        if message["role"] == "user":
            chat_history.append(cohere.ChatMessage(role="USER", message=message["content"]))
        elif message["role"] == "assistant" and message.get("content") is not None:
            chat_history.append(cohere.ChatMessage(role="CHATBOT", message=message["content"]))
        elif message["role"] == "tool":
            corresponding_call_arguments = None
            corresponding_call_name = None
            for tool_message in openai_messages:
                if tool_message["role"] == "assistant" and tool_message.get("tool_calls") is not None:
                    for tool_call in tool_message["tool_calls"]:
                        if tool_call["id"] == message["tool_call_id"]:
                            corresponding_call_arguments = tool_call["function"]["arguments"]
                            corresponding_call_name = tool_call["function"]["name"]
                            break
            call = {
                "name": corresponding_call_name,
                "parameters": json.loads(corresponding_call_arguments),
                "generation_id": corresponding_call_name + message["tool_call_id"]  # TODO: reconsider this
            }

            # replace content ' with " to make it json compatible
            content = message["content"].replace("'", "\"")
            tool_results.append(
                {
                    "call": call,
                    "outputs": [json.loads(content)]
                }
            )
        elif message["role"] == "system":
            if preamble is None:
                preamble = ""
            preamble += message["content"] + "\n"

    # the last message is the user prompt so remove it from the chat history
    last_message = chat_history[-1]
    chat_history = chat_history[:-1]

    return CohereChat(chat_history=chat_history, last_message=last_message.message, preamble=preamble,
                      tool_results=tool_results)


def _format_openai_tool_to_cohere_tool(input_json: dict) -> Tool:
    # converts openai tool with JSON schema properties to cohere tool with simple list properties
    # only supports simple list of parameters due to those limitations
    parameter_definitions = {}
    for prop_name, prop_value in input_json["function"]['parameters']['properties'].items():
        if prop_name == "required":
            continue
        if prop_value['type'] == 'object':
            raise ValueError("Object type is not supported for property: " + prop_name)
        converted_prop = {
            'description': prop_value.get('description', ''),
            'type': prop_value['type'].__class__.__name__,
            'required': prop_name in input_json["function"]['parameters'].get('required', [])
        }
        parameter_definitions[prop_name] = converted_prop

    return Tool(
        name=input_json["function"]["name"],
        description=input_json["function"]["description"],
        parameter_definitions=parameter_definitions
    )


def _format_openai_function_to_anthropic_tool(input_json: dict) -> dict:
    output_json = {
        "name": input_json["name"],
        "description": input_json["description"],
        "input_schema": input_json["parameters"],
    }

    return output_json


def _format_cohere_response_to_openai_response(openai_request: OpenAICompletionRequest,
                                               cohere_response: NonStreamedChatResponse) -> OpenAICompletionResponse:
    tool_messages = []

    if cohere_response.tool_calls is not None:
        for tool_call in cohere_response.tool_calls:
            tool_messages.append(
                {
                    "id": "a",
                    "name": tool_call.name,
                    "input": tool_call.parameters
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
        finish_reason = "length" if cohere_response.finish_reason == "MAX_TOKENS" else "stop"
        choices = [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {
                    "content": cohere_response.text,
                    "role": "assistant"
                },
            }
        ]

    openai_response = OpenAICompletionResponse(
        completion_id=cohere_response.generation_id,
        model=openai_request.model,
        choices=choices,
        completionTokens=cohere_response.meta.tokens.output_tokens if cohere_response.meta.tokens.output_tokens else 0,
        promptTokens=cohere_response.meta.tokens.input_tokens if cohere_response.meta.tokens.input_tokens else 0
    )

    return openai_response


def _stream_request(completionRequest: OpenAICompletionRequest, client: cohere.client.Client) -> Iterator[AnyStr]:
    sent_role = False

    response_stream = client.chat_stream(**CohereCompletionRequest.from_openai_request(completionRequest).to_dict())

    completionId = "static_id"

    for response in response_stream:
        if response.event_type == "stream-start":
            continue

        if response.event_type == "stream-end":
            chunk_message = str(OpenAICompletionChunkResponse(
                completion_id=completionId,
                model=completionRequest.model,
                choices=[
                    {"index": 0, "delta": {}, "finish_reason": "stop"}]
            ).to_json())

            yield "data:" + chunk_message + "\n\n"

            break

        if response.event_type == "text-generation":
            delta = {"content": response.text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            chunk_message = str(OpenAICompletionChunkResponse(
                completion_id=completionId,
                model=completionRequest.model,
                choices=[{"index": 0, "delta": delta, "finish_reason": None}]
            ).to_json())

            yield "data:" + chunk_message + "\n\n"


class CohereApiBackend(TargetApiBackend):
    def __init__(self, base_url: str = os.environ.get("COHERE_API_URL", "https://api.cohere.ai"),
                 api_key: str = os.environ.get("COHERE_API_KEY")):
        """Load the available models from data/cohere_models.json."""

        models = []
        with open('data/cohere_models.json') as f:
            data = json.load(f)
            for model in data:
                models.append(AvailableModel(**model))
        super().__init__(base_url, api_key, models)

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        cohere_response = CohereCompletionRequest.from_openai_request(completionRequest).make_api_request(
            self.base_url, self.get_api_key(completionRequest, pass_api_key))

        return _format_cohere_response_to_openai_response(completionRequest, cohere_response)

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        co = cohere.Client(self.get_api_key(completionRequest, pass_api_key), base_url=self.base_url)
        return _stream_request(completionRequest, co)
