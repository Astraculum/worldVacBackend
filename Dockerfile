FROM python:3.13.0-slim

ENV USER=uv-example-user \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

# Install necessary packages, including git for submodule management
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash $USER

# Copy uv from the pre-built image
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/

# Configure environment
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy all files first
COPY . /app

# Initialize git repository and setup submodules
RUN git init && \
    git config --global --add safe.directory /app && \
    git submodule update --init --recursive

# Install dependencies
RUN uv sync

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
