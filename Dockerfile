FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy poetry files
COPY poetry.lock pyproject.toml ./

# Install Poetry with dependencies
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

# Copy app files
COPY app app
COPY data data

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "python", "-m", "app"]