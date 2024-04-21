from flask import request, jsonify, Blueprint, current_app

from app.config import AuthMode
from app.models import OpenAICompletionRequest
from app.services.service_manager import get_current_target_api_backend

routes_blueprint = Blueprint('routes', __name__)


@routes_blueprint.route("/v1/models", methods=["GET"])
def models():
    """Returns a list of models available on the target API backend."""

    current_app.logger.info("Received model list request")

    # return the list of models
    return jsonify({
        "object": "list",
        "data": [x.to_dict() for x in get_current_target_api_backend().models]
    })


@routes_blueprint.route("/v1/chat/completions", methods=["POST"])
def completions():
    current_app.logger.info("Received completion request")

    # make sure the request has the correct API key
    header_api_key = request.headers.get("Authorization")
    if current_app.config.get("AUTH_MODE") == AuthMode.PASS_API_KEY and header_api_key is None:
        return jsonify({"error": "No API key provided"}), 401
    if current_app.config.get("AUTH_MODE") == AuthMode.CUSTOM_KEY and header_api_key.split("Bearer ")[1] != current_app.config.get("AUTH_KEY"):
        return jsonify({"error": "Invalid API key provided"}), 401

    # parse the request
    completionRequest = OpenAICompletionRequest.from_request(request, current_app.config)

    current_app.logger.debug("Address: " + request.remote_addr)
    current_app.logger.debug("Model: " + completionRequest.model)
    current_app.logger.debug("Max tokens: " + str(completionRequest.max_tokens))

    # log the messages and tools
    current_app.logger.debug("Messages:")
    for message in completionRequest.messages:
        current_app.logger.debug(str(message) + ",")

    if completionRequest.tools:
        current_app.logger.debug("Tools:")
        for tool in completionRequest.tools:
            current_app.logger.debug(str(tool) + ",")

    current_app.logger.debug("Received request with body: " + str(request.json))

    # handle the request
    target_api_backend = get_current_target_api_backend()
    current_app.logger.info("Handling completion request with backend: " + target_api_backend.__class__.__name__)

    if completionRequest.streamed:
        return target_api_backend.handle_streamed_completion_request(
            completionRequest, current_app.config.get("AUTH_MODE") == AuthMode.PASS_API_KEY)
    else:
        response = target_api_backend.handle_completion_request(
            completionRequest, current_app.config.get("AUTH_MODE") == AuthMode.PASS_API_KEY).to_json()
        current_app.logger.debug("Returning response: " + str(response))
        return response


def init_app(app):
    app.register_blueprint(routes_blueprint)
