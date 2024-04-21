import json
import time
from typing import Iterator, AnyStr, Iterable
from openai.types.chat import ChatCompletionToolParam

from app.config import Config


class OpenAICompletionRequest:
    """An OpenAI compatible completion request."""

    def __init__(self, api_key: str, model: str, max_tokens: int | None, messages,
                 tools: Iterable[ChatCompletionToolParam] | None = None, stream: bool = False,
                 temperature: float | None = None, top_p: float | None = None, frequency_penalty: float | None = None,
                 presence_penalty: float | None = None, tool_choice: str | None = None,
                 stop: str | list[str] | None = None, **kwargs):
        self.api_key: str | None = api_key
        self.model: str = model
        self.messages = messages
        self.max_tokens: int | None = max_tokens
        self.tools: Iterable[ChatCompletionToolParam] | None = tools
        self.streamed: bool | None = stream
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.frequency_penalty: float | None = frequency_penalty
        self.presence_penalty: float | None = presence_penalty
        self.tool_choice: str = tool_choice
        self.stop: str | list[str] | None = stop

    @classmethod
    def from_request(cls, request, config: Config) -> 'OpenAICompletionRequest':
        """
        Create an OpenAI completion request from a request object.

        :param request: flask request object
        :param config: app config object
        :return: OpenAICompletionRequest object
        """

        request_json = request.json

        # extract api key from request headers
        header_api_key = request.headers.get("Authorization")
        api_key: str = header_api_key.split("Bearer ")[1]

        args = request_json

        # override max tokens if not specified
        if args.get("max_tokens") is None:
            args["max_tokens"] = 4096

        # override model if specified in config
        if config.get("MODEL_NAME") is not None:
            args["model"] = config.get("MODEL_NAME")

        return cls(api_key=api_key, **args)

    def to_dict(self) -> dict:
        request_args = {
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "model": self.model
        }
        if self.tool_choice:
            request_args["tool_choice"] = self.tool_choice
        if self.temperature:
            request_args["temperature"] = self.temperature
        if self.top_p:
            request_args["top_p"] = self.top_p
        if self.presence_penalty:
            request_args["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            request_args["frequency_penalty"] = self.frequency_penalty
        if self.tools:
            request_args["tools"] = self.tools
        if self.stop:
            request_args["stop"] = self.stop
        return request_args


class OpenAICompletionResponse:
    """A non streamed completion response."""

    def __init__(self, completion_id: str, model: str, choices, completionTokens: int, promptTokens: int,
                 system_fingerprint: str = 'static_fingerprint'):
        self.id = completion_id
        self.model = model
        self.choices = choices
        self.completionTokens = completionTokens
        self.promptTokens = promptTokens
        self.system_fingerprint = system_fingerprint

    def to_json(self) -> str:
        return json.dumps({
            'choices': self.choices,
            'created': int(time.time()),
            'id': self.id,
            'model': self.model,
            'object': "chat.completion",
            'system_fingerprint': self.system_fingerprint,
            'usage': {
                "completion_tokens": self.completionTokens,
                "prompt_tokens": self.promptTokens,
                "total_tokens": self.promptTokens + self.completionTokens,
            }})


class OpenAICompletionChunkResponse:
    """A streaming completion chunk response."""

    def __init__(self, completion_id: str, model: str, choices):
        self.id = completion_id
        self.model = model
        self.choices = choices

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "system_fingerprint": 'static_fingerprint',
            "choices": self.choices,
        })


class AvailableModel:
    """A model available for completion requests."""

    def __init__(self, id: str, object: str, created: int, owned_by: str):
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by
        }


class TargetApiBackend:
    """Base class for all API backends."""

    def __init__(self, base_url: str, api_key: str, models: list[AvailableModel]):
        self.base_url = base_url
        self.api_key = api_key
        self.models = models

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        """
        Handle a (non-streamed) completion request.

        :param completionRequest: The completion request to handle.
        :param pass_api_key: Whether to pass the API key from the request to the backend or use the server's API key.
        """
        pass

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        """
        Handle a streamed completion request.

        :param completionRequest: The completion request to handle.
        :param pass_api_key:  Whether to pass the API key from the request to the backend or use the server's API key.
        :return: Stream of completion chunks.
        """
        pass

    def get_api_key(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) -> str:
        """
        Get the API key to use for a completion request.

        :param completionRequest: The completion request.
        :param pass_api_key: Whether to pass the API key from the request to the backend or use the server's API key.
        :return: The API key to use.
        """
        return completionRequest.api_key if pass_api_key else self.api_key
