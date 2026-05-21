"""Persistent inference worker. Reads JSON commands from stdin, writes JSON results to stdout.
Keeps models in CPU RAM between requests for fast RAM→GPU transfers."""
import json
import os
import sys

# Disable tqdm progress bars — they write to stdout and corrupt the JSON protocol
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore[import-not-found]

HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "google/embeddinggemma-300m")
RERANKING_MODEL_NAME = os.environ.get("RERANKING_MODEL", "cross-encoder/ettin-reranker-400m-v1")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "128"))

# Pre-load models on CPU at startup (one-time disk read)
print("[worker] loading models on CPU...", file=sys.stderr, flush=True)
_embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN, device="cpu")
_rerank_model = CrossEncoder(RERANKING_MODEL_NAME, token=HF_TOKEN, device="cpu")
print("[worker] models loaded on CPU", file=sys.stderr, flush=True)

# Signal readiness — server waits for this before sending commands
print(json.dumps({"ready": True}), flush=True)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _to_gpu(model):
    """Move model to GPU if available, otherwise keep on CPU."""
    if not torch.cuda.is_available():
        print("[worker] GPU not available, staying on CPU", file=sys.stderr, flush=True)
        return model
    try:
        torch.cuda.empty_cache()
        model = model.to("cuda")
        print("[worker] model on GPU", file=sys.stderr, flush=True)
        return model
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[worker] GPU OOM, staying on CPU", file=sys.stderr, flush=True)
        return model  # Stay on CPU


def _to_cpu(model):
    """Move model back to CPU and free GPU cache."""
    if next(model.parameters()).device.type == "cuda":
        model = model.to("cpu")
        torch.cuda.empty_cache()
    return model


def run_embed(query: str, texts: list[str]) -> list[float]:
    global _embed_model
    model = _to_gpu(_embed_model)
    try:
        query_emb = model.encode([query])[0]
        embs = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embs.append(model.encode(batch))
        all_embs = np.concatenate(embs)
        return [cosine_sim(query_emb, e) for e in all_embs]
    finally:
        _embed_model = _to_cpu(model)


def run_rerank(query: str, texts: list[str]) -> list[float]:
    global _rerank_model
    model = _to_gpu(_rerank_model)
    try:
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs, batch_size=EMBED_BATCH_SIZE)
        return [float(s) for s in scores]
    finally:
        _rerank_model = _to_cpu(model)


def main():
    # Ensure line-buffered stdout for subprocess communication
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            kind = cmd["kind"]
            if kind == "embed":
                scores = run_embed(cmd["query"], cmd["texts"])
                print(json.dumps({"ok": True, "scores": scores}), flush=True)
            elif kind == "rerank":
                scores = run_rerank(cmd["query"], cmd["texts"])
                print(json.dumps({"ok": True, "scores": scores}), flush=True)
            else:
                print(json.dumps({"ok": False, "error": f"unknown kind: {kind}"}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
