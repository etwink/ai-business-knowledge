"""
Knowledge base: chunk source documents, embed with sentence-transformers,
store embeddings + chunk metadata on disk for fast retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBED_BATCH = 32

# Module-level singleton so the model is only loaded once per process.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


@dataclass
class Chunk:
    text: str
    source_file: str
    file_type: str
    chunk_index: int
    char_start: int


class KnowledgeBase:
    """
    Manages chunked document embeddings for a single session.

    Persists to:
      <persist_dir>/chunks.json     — serialised Chunk list
      <persist_dir>/embeddings.npy  — float32 numpy array (n_chunks × dim)
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_built(self) -> bool:
        return (
            (self.persist_dir / "chunks.json").exists()
            and (self.persist_dir / "embeddings.npy").exists()
        )

    def build(
        self,
        files: list[Path],
        progress_callback=None,   # (message: str, current: int, total: int)
    ) -> int:
        """Chunk + embed all files and persist to disk.  Returns total chunk count."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: chunk ─────────────────────────────────────────────
        all_chunks: list[Chunk] = []
        for i, path in enumerate(files):
            if progress_callback:
                progress_callback(f"Reading {path.name}", i + 1, len(files) * 2)
            all_chunks.extend(_chunk_file(path))

        if not all_chunks:
            return 0

        # ── Step 2: embed in batches ──────────────────────────────────
        model = _get_model()
        texts = [c.text for c in all_chunks]
        embedding_batches: list[np.ndarray] = []
        base = len(files)  # progress offset after chunking phase
        for batch_start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[batch_start: batch_start + _EMBED_BATCH]
            emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embedding_batches.append(emb)
            if progress_callback:
                done = min(batch_start + _EMBED_BATCH, len(texts))
                progress_callback(
                    f"Embedding {done}/{len(texts)} chunks",
                    base + done,
                    base + len(texts),
                )

        embeddings = np.vstack(embedding_batches).astype(np.float32)

        # ── Step 3: persist ───────────────────────────────────────────
        with open(self.persist_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False)
        np.save(str(self.persist_dir / "embeddings.npy"), embeddings)

        self._chunks = all_chunks
        self._embeddings = embeddings
        return len(all_chunks)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Return the top-k most relevant chunks for query (cosine similarity)."""
        self._ensure_loaded()
        if self._embeddings is None or not self._chunks:
            return []

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        # Embeddings are already L2-normalised → dot product = cosine similarity
        scores: np.ndarray = self._embeddings @ q_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**asdict(self._chunks[i]), "score": float(scores[i])}
            for i in top_indices
        ]

    def get_stats(self) -> dict:
        self._ensure_loaded()
        unique_files = {c.source_file for c in self._chunks}
        return {"chunks": len(self._chunks), "files": len(unique_files)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._chunks:
            return
        chunks_path = self.persist_dir / "chunks.json"
        emb_path = self.persist_dir / "embeddings.npy"
        if chunks_path.exists() and emb_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                self._chunks = [Chunk(**c) for c in json.load(f)]
            self._embeddings = np.load(str(emb_path))


# ------------------------------------------------------------------
# File chunking
# ------------------------------------------------------------------

def _chunk_file(path: Path) -> list[Chunk]:
    """Load a file with the document loader and split it into overlapping chunks."""
    try:
        from src.document_loaders import get_loader
        doc = get_loader(path).load()
        content = doc.content
        file_type = doc.file_type
    except Exception:
        return []

    # Remove excessive whitespace runs while preserving paragraph breaks
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if not content:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        # Try to break on a whitespace boundary so chunks don't cut mid-word
        if end < len(content):
            boundary = content.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        text = content[start:end].strip()
        if text:
            chunks.append(Chunk(
                text=text,
                source_file=path.name,
                file_type=file_type,
                chunk_index=idx,
                char_start=start,
            ))
            idx += 1

        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks
