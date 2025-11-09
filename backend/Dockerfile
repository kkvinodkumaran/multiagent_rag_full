# uv base image with Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync

# copy code
COPY . /app

# data dir for Chroma (mounted via compose)
ENV CHROMA_DIR=/data/chroma

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
