
## LLM Converter

This project is a simple server that converts OpenAI API requests to other LLM providers, such as Anthropic,
Mistral or Cohere. This enables usage of different models for apps or services that only support the OpenAI API spec. 

Features:
- Automatically selects the target API based on the model name.
- Supports multiple authentication modes.
- CLI for running tests and chatting with the models.
- Supports function calling, stop sequences, max tokens, temperature and top p parameters.

# Usage

---
## Server

#### Configuration
By default, the server runs on port 8000 and passes the API key received in the request to the target API.

To change the default configuration:

1. Copy the .env.example file to .env
2. Fill out the api keys or other configuration options in the .env file

#### Running the server

##### Using docker-compose (recommended):
```bash
# Downloads the image from DockerHub and runs the server
docker compose up -d 
```
##### Using poetry:
```bash
# Installs the dependencies and runs the server
poetry install
poetry run python -m app
```
##### Using pip:
```bash
# Installs the dependencies and runs the server
python -m pip install -r requirements.txt
python -m app
```

---
## CLI

The CLI supports two main features:
1. Evaluation tests - runs a set of tests using different options on the api and verifies the results.
2. Chat - starts a chat session with the selected model.


To run the CLI, use the following command:

```bash
# using poetry

poetry install
poetry run python -m cli

# or using pip

python -m pip install -r requirements.txt
python -m cli
```

---

# Configuration

### Server Configuration
`TARGET_API` - one of `anthropic`, `mistral` or `cohere`. Default picks automatically based on received model name.

`MODEL_NAME` - name of the model to use. Default is model received in the request.

`SERVER_PORT` - port to run the server on. Default is `8000`.

`LOG_LEVEL` - log level for the server. Default is `INFO`.

`AUTH_MODE` - authentication mode for the server. Default is `NO_AUTH`. Possible values are:
- `PASS_API_KEY` - forwards the API key to the target API.
- `CUSTOM_KEY` - verifies the key from request with `AUTH_KEY` and uses keys provided in environment variables to authenticate with the target API.
- `NO_AUTH` - no authentication is required, server uses keys provided in environment variables to authenticate with the target API.

`AUTH_KEY` - custom key required when `AUTH_MODE` is set to `CUSTOM_KEY`.

### API Configuration
`ANTHROPIC_API_KEY` - API key for the Anthropic API. You can get one by signing up at [https://anthropic.com](https://anthropic.com).

`ANTHROPIC_API_URL` - URL for the Anthropic API. Default is `https://api.anthropic.com`.

`MISTRAL_API_KEY` - API key for the Mistral API. You can get one by signing up at [https://mistral.ai](https://mistral.ai).

`MISTRAL_API_URL` - URL for the Mistral API. Default is `https://api.mistral.ai`.

`COHERE_API_KEY` - API key for the Cohere API. You can get one by signing up at [https://cohere.ai](https://cohere.ai).

`COHERE_API_URL` - URL for the Cohere API. Default is `https://api.cohere.ai`.

---

## Tests

To run unit tests, use the following command:

```bash
# using poetry
poetry run pytest

# without poetry
pytest
```