# ── SHL Assessment Advisor — Dockerfile ──────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder): installs all Python deps in an isolated layer.
#   Stage 2 (runtime): copies only the installed site-packages + app code.
#                      Result image is ~30% smaller.
#
# Build:  docker build -t shl-advisor .
# Run:    docker run -p 8000:8000 -e GEMINI_API_KEY=<key> shl-advisor
# ─────────────────────────────────────────────────────────────────────────────

# ---------- Stage 1: dependency builder ----------
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools needed for some native extensions (faiss, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# ---------- Stage 2: runtime image ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY data_prep.py   .
COPY retriever.py   .
COPY models.py      .
COPY agent.py       .
COPY main.py        .
COPY shl_product_catalog.json .

# The catalog is read at startup; the FAISS index is built in memory — no
# volume mounts are required for basic operation.

# Expose the API port
EXPOSE 8000

# Health check — evaluator expects /health to return {"status": "ok"}
# Allow 3 minutes for cold-start (embedding model download on first run)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# GEMINI_API_KEY must be passed at runtime via -e or a secrets manager.
# Never bake the key into the image.
ENV GEMINI_API_KEY=""

# Use a non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
