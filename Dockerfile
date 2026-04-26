# syntax=docker/dockerfile:1.7
# Multi-stage build: compile deps in stage 1, copy minimal runtime to stage 2.

# ============================================================
# Stage 1: builder — install uv, sync deps into a venv
# ============================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

# uv official image gives us a static binary
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps first for layer caching (only re-runs when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now copy source and install the project itself
COPY src/ ./src/
COPY config.toml.example ./config.toml.example
RUN uv sync --frozen --no-dev


# ============================================================
# Stage 2: runtime — slim Python + the venv from builder
# ============================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user
RUN groupadd -r firefly-bot && useradd -r -g firefly-bot -m -d /home/firefly-bot firefly-bot

WORKDIR /app

# Copy the prebuilt venv and source from builder
COPY --from=builder --chown=firefly-bot:firefly-bot /app/.venv /app/.venv
COPY --from=builder --chown=firefly-bot:firefly-bot /app/src /app/src
COPY --chown=firefly-bot:firefly-bot config.toml.example /app/config.toml.example

# Data dir is mounted as a volume in compose; create as a fallback
RUN mkdir -p /app/data && chown -R firefly-bot:firefly-bot /app/data

USER firefly-bot

# Healthcheck: process running + can import. We don't expose any port,
# so this is the cheapest signal that the container isn't deadlocked.
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import firefly_agent" || exit 1

CMD ["python", "-m", "firefly_agent"]
