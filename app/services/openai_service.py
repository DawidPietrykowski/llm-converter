import json
import os
from typing import Iterator, AnyStr

import requests
from dotenv import load_dotenv
from openai import OpenAI

from app.models import TargetApiBackend, AvailableModel, OpenAICompletionRequest, OpenAICompletionResponse

load_dotenv()

def _stream_request(completionRequest: OpenAICompletionRequest, client: OpenAI) -> Iterator[AnyStr]:
    stream_args = completionRequest.to_dict()
    stream_args["stream"] = True

    url = str(client.base_url) + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + client.api_key,
        "Content-Type": "application/json"
    }

    response = requests.post(url=url, json=stream_args, headers=headers, stream=True)

    for chunk in response.iter_lines():
        yield chunk + b"\n\n"


class OpenAIApiBackend(TargetApiBackend):
    def __init__(self, base_url: str = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
                 api_key: str = os.environ.get("OPENAI_API_KEY")):
        """Load the available models from data/openai_models.json."""
        models = []
        with open('data/openai_models.json') as f:
            data = json.load(f)
            for model in data:
                models.append(AvailableModel(**model))
        super().__init__(base_url, api_key, models)

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        client: OpenAI = OpenAI(api_key=self.get_api_key(completionRequest, pass_api_key), base_url=self.base_url)

        response = client.chat.completions.create(**completionRequest.to_dict())

        return OpenAICompletionResponse(
            completion_id=response.id,
            model=response.model,
            choices=[choice.to_dict() for choice in response.choices],
            completionTokens=response.usage.completion_tokens,
            promptTokens=response.usage.prompt_tokens,
            system_fingerprint=response.system_fingerprint
        )

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        client = OpenAI(api_key=self.get_api_key(completionRequest, pass_api_key), base_url=self.base_url)

        return _stream_request(completionRequest, client)
