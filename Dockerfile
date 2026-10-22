FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
WORKDIR /app

# Pick providers at build time, e.g. --build-arg EXTRAS="--extra anthropic --extra local"
ARG EXTRAS="--extra anthropic --extra openai"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS}
COPY src ./src
RUN uv sync --frozen --no-dev ${EXTRAS}

ENV RAGGITY_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "raggity.main:app", "--host", "0.0.0.0", "--port", "8000"]
