FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync

COPY . /app

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
