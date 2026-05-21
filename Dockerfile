# Stage 1: Build stage
FROM python:3.11-slim AS builder

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /app

# System dependency required by PyTorch (OpenMP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (force CPU-only torch)
RUN pip install --no-cache-dir --user \
    fastmcp \
    httpx \
    numpy

RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cu132

RUN pip install --no-cache-dir --user \
    sentence-transformers

# Pre-download models to avoid runtime cold start
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer(\"google/embeddinggemma-300m\", device=\"cpu\")"
RUN python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder(\"cross-encoder/ettin-reranker-400m-v1\", device=\"cpu\")"


# Stage 2: Runtime stage
FROM python:3.11-slim

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /app

# Runtime system dependency
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /root/.local /root/.local

# Copy HuggingFace cache (model weights)
COPY --from=builder /root/.cache /root/.cache

# Copy server code
COPY pse_server.py pse_server.py
COPY worker.py worker.py

# Environment setup
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Expose MCP port
EXPOSE 8000

# Start server
ENTRYPOINT ["python", "pse_server.py"]
