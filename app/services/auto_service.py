from typing import Iterator, AnyStr
from app.models import TargetApiBackend, OpenAICompletionRequest, OpenAICompletionResponse


class AutoApiBackend(TargetApiBackend):
    """A backend that automatically selects the correct backend based on the model ID."""

    def __init__(self, supported_backends: list[TargetApiBackend]):
        self.supported_backends = supported_backends

        models = [model for backend in supported_backends for model in backend.models]
        super().__init__('', '', models)

    def handle_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> OpenAICompletionResponse:
        for backend in self.supported_backends:
            if completionRequest.model in [model.id for model in backend.models]:
                return backend.handle_completion_request(completionRequest, pass_api_key)

        raise ValueError("Model not found in any supported backend")

    def handle_streamed_completion_request(self, completionRequest: OpenAICompletionRequest, pass_api_key: bool) \
            -> Iterator[AnyStr]:
        for backend in self.supported_backends:
            if completionRequest.model in [model.id for model in backend.models]:
                return backend.handle_streamed_completion_request(completionRequest, pass_api_key)

        raise ValueError("Model not found in any supported backend")
