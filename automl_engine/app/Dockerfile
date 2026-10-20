FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Setup user and working directory
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app
# Ensure the nonroot user owns the workdir before we start
RUN chown nonroot:nonroot /app

# 2. Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PATH="/app/.venv/bin:$PATH"

# 3. Switch to nonroot user NOW
USER nonroot

# 4. Copy lockfiles with correct ownership
COPY --chown=nonroot:nonroot pyproject.toml uv.lock ./

# 5. Install dependencies
# (uv will now create the .venv as the nonroot user)
RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=999 \
    uv sync --locked --no-install-project --no-dev

# 6. Copy application code with correct ownership
COPY --chown=nonroot:nonroot app/ app/

# 7. Final sync
RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=999 \
    uv sync --locked --no-dev

# Healthcheck port: overridable at build time (ARG) and at runtime (env),
# so the container can target the port it actually listens on.
ARG HEALTHCHECK_PORT=8001
ENV HEALTHCHECK_PORT=${HEALTHCHECK_PORT}
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${HEALTHCHECK_PORT}/health || exit 1

ENTRYPOINT []
