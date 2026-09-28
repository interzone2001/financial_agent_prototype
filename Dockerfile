# Single-service image: React build served by FastAPI alongside /api.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
ENV PATH="/app/backend/.venv/bin:$PATH" \
    FRONTEND_DIST=/app/frontend/dist \
    DATA_DIR=/data
# Bake Chroma's ONNX embedding model into the image (~80MB) so cold starts don't download it.
RUN python -c "from app.vectorstore import warm_embedding_model; warm_embedding_model()"
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
