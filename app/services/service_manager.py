from app.models import TargetApiBackend
from app.services.anthropic_service import AnthropicApiBackend
from app.services.auto_service import AutoApiBackend
from app.services.cohere_service import CohereApiBackend
from app.services.mistral_service import MistralApiBackend
from app.services.openai_service import OpenAIApiBackend

current_target_api: TargetApiBackend | None = None


def init_target_api_backend(target_api: str):
    """Initialize the target API backend based on the provided configuration."""

    global current_target_api
    if target_api == "anthropic":
        current_target_api = AnthropicApiBackend()
    elif target_api == "mistral":
        current_target_api = MistralApiBackend()
    elif target_api == "cohere":
        current_target_api = CohereApiBackend()
    elif target_api == "openai":
        current_target_api = OpenAIApiBackend()
    else:
        current_target_api = AutoApiBackend(supported_backends=[
            AnthropicApiBackend(),
            MistralApiBackend(),
            CohereApiBackend(),
            OpenAIApiBackend()
        ])


def get_current_target_api_backend() -> TargetApiBackend:
    """Get the current target API backend."""

    global current_target_api
    return current_target_api
