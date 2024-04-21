from flask import Flask
from waitress import serve

from app.config import Config
from app.services.service_manager import init_target_api_backend


def create_app(config: Config | None = None):
    app = Flask(__name__)

    # load the configuration from the provided Config object or the default Config object (env vars)
    if config:
        app.config.from_object(config)
    else:
        app.config.from_object(Config())

    # initialize the target API backend
    init_target_api_backend(app.config.get("TARGET_API"))

    # set the log level
    app.logger.setLevel(app.config.get("LOG_LEVEL"))

    # import and initialize the routes
    from app import routes
    routes.init_app(app)

    return app


if __name__ == '__main__':
    app = create_app()
    print(f"Running server on port {app.config.get('SERVER_PORT')}")
    serve(app, listen=f'*:{app.config.get("SERVER_PORT")}')
