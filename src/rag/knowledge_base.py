"""
Knowledge base: chunk source documents, embed with sentence-transformers,
store embeddings + chunk metadata on disk for fast retrieval.

Tier system (higher tier = boosted score at retrieval time):
  Tier 1 — Generated process document  (most authoritative summary)
  Tier 2 — Word / Doc files            (business documentation)
  Tier 3 — Everything else             (code, Excel, COBOL, etc.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBED_BATCH = 32

# Multiplicative boost applied to cosine scores before ranking.
# Tier 1 — Word/Doc files (primary source-of-truth business documents)
# Tier 2 — Generated process document (authoritative summary, but derivative)
# Tier 3 — Everything else (code, Excel, COBOL, etc.)
_TIER_BOOST: dict[int, float] = {1: 1.5, 2: 1.2, 3: 1.0}

_WORD_EXTENSIONS = {".doc", ".docx"}

# Module-level singleton so the model loads only once per process.
_model = None


def _get_model():
    global _model
    if _model is None:
        import os
        import warnings

        # After the model is downloaded once it lives in the local HF cache.
        # Prevent the library from phoning home on every load — this also
        # eliminates the "unauthenticated requests" warning.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # Suppress noisy FutureWarnings from the transformers internals
        # (e.g. "__path__ alias" deprecation notices).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message=r".*HF Hub.*")
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
    tier: int = 3   # 1=process_doc, 2=word, 3=other


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
        progress_callback=None,
    ) -> int:
        """Chunk + embed all files and persist to disk.  Returns total chunk count."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        all_chunks: list[Chunk] = []
        for i, path in enumerate(files):
            if progress_callback:
                progress_callback(f"Reading {path.name}", i + 1, len(files) * 2)
            tier = 1 if path.suffix.lower() in _WORD_EXTENSIONS else 3
            all_chunks.extend(_chunk_file(path, tier=tier))

        if not all_chunks:
            return 0

        embeddings = self._embed(all_chunks, progress_callback, base=len(files))
        self._persist(all_chunks, embeddings)
        return len(all_chunks)

    def index_process_document(self, process_doc) -> int:
        """
        Add the generated process document sections as tier-1 chunks.
        Can be called after build() or on an already-loaded KB.
        Merges into the existing index and re-persists.
        Returns the number of new chunks added.
        """
        self._ensure_loaded()

        section_map = {
            "Overview": process_doc.overview,
            "Integrated Processes": process_doc.integrated_processes,
            "Dependencies": process_doc.dependencies,
            "Data Flow": process_doc.data_flow,
            "Decision Points": process_doc.decision_points,
            "Systems and Components": process_doc.systems_and_components,
            "Appendix": process_doc.appendix,
        }

        new_chunks: list[Chunk] = []
        for section_name, content in section_map.items():
            if not content:
                continue
            source = f"[Process Document — {section_name}]"
            new_chunks.extend(_chunk_text(content, source_file=source, file_type="process_document", tier=2))

        if not new_chunks:
            return 0

        model = _get_model()
        texts = [c.text for c in new_chunks]
        new_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

        merged_chunks = self._chunks + new_chunks
        merged_emb = (
            np.vstack([self._embeddings, new_emb])
            if self._embeddings is not None and len(self._embeddings)
            else new_emb
        )
        self._persist(merged_chunks, merged_emb)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 25) -> list[dict]:
        """Return top-k chunks ranked by tier-boosted cosine similarity."""
        self._ensure_loaded()
        if self._embeddings is None or not self._chunks:
            return []

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        raw_scores: np.ndarray = self._embeddings @ q_emb

        # Apply tier boost
        boosts = np.array(
            [_TIER_BOOST.get(c.tier, 1.0) for c in self._chunks], dtype=np.float32
        )
        boosted = raw_scores * boosts

        top_indices = np.argsort(boosted)[::-1][:top_k]
        return [
            {**asdict(self._chunks[i]), "score": float(raw_scores[i]),
             "boosted_score": float(boosted[i])}
            for i in top_indices
        ]

    def search_tiered(
        self,
        query: str,
        process_doc_k: int = 3,
        word_k: int = 3,
        code_k: int = 2,
        exclude_ids: set[int] | None = None,
        include_code: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Pull top-k chunks separately from each tier, respecting per-query exclusions.

        Returns {"process_doc": [...], "word": [...], "code": [...]}.
        Each chunk dict includes "_array_idx" (position in the internal chunk array)
        so callers can track which chunks to exclude in subsequent calls.

        Tier mapping:
          tier 2 → process_doc   (generated process document sections)
          tier 1 → word          (.doc / .docx business documents)
          tier 3 → code          (everything else: COBOL, Python, Excel, etc.)
        """
        self._ensure_loaded()
        if self._embeddings is None or not self._chunks:
            return {"process_doc": [], "word": [], "code": []}

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        raw_scores: np.ndarray = self._embeddings @ q_emb

        excluded = exclude_ids or set()

        tier_indices: dict[str, list[int]] = {"process_doc": [], "word": [], "code": []}
        for i, chunk in enumerate(self._chunks):
            if i in excluded:
                continue
            if chunk.tier == 2:
                tier_indices["process_doc"].append(i)
            elif chunk.tier == 1:
                tier_indices["word"].append(i)
            else:
                tier_indices["code"].append(i)

        limits = {
            "process_doc": process_doc_k,
            "word": word_k,
            "code": code_k if include_code else 0,
        }

        results: dict[str, list[dict]] = {"process_doc": [], "word": [], "code": []}
        for group, indices in tier_indices.items():
            k = limits[group]
            if k == 0 or not indices:
                continue
            idx_arr = np.array(indices)
            group_scores = raw_scores[idx_arr]
            top_n = min(k, len(idx_arr))
            top_local = np.argsort(group_scores)[::-1][:top_n]
            for li in top_local:
                gi = int(idx_arr[li])
                results[group].append({
                    **asdict(self._chunks[gi]),
                    "score": float(raw_scores[gi]),
                    "_array_idx": gi,
                })

        return results

    def get_stats(self) -> dict:
        self._ensure_loaded()
        unique_files = {c.source_file for c in self._chunks}
        tier_counts = {1: 0, 2: 0, 3: 0}
        for c in self._chunks:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
        return {
            "chunks": len(self._chunks),
            "files": len(unique_files),
            "tier_counts": tier_counts,
        }

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

    def _embed(
        self,
        chunks: list[Chunk],
        progress_callback=None,
        base: int = 0,
    ) -> np.ndarray:
        model = _get_model()
        texts = [c.text for c in chunks]
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i: i + _EMBED_BATCH]
            batches.append(
                model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            )
            if progress_callback:
                done = min(i + _EMBED_BATCH, len(texts))
                progress_callback(
                    f"Embedding {done}/{len(texts)} chunks",
                    base + done,
                    base + len(texts),
                )
        return np.vstack(batches).astype(np.float32)

    def _persist(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self.persist_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, ensure_ascii=False)
        np.save(str(self.persist_dir / "embeddings.npy"), embeddings)
        self._chunks = chunks
        self._embeddings = embeddings


# ------------------------------------------------------------------
# Chunking helpers
# ------------------------------------------------------------------

def _chunk_text(
    text: str,
    source_file: str,
    file_type: str,
    tier: int = 3,
) -> list[Chunk]:
    """Split arbitrary text into overlapping Chunk objects."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                file_type=file_type,
                chunk_index=idx,
                char_start=start,
                tier=tier,
            ))
            idx += 1

        next_start = end - CHUNK_OVERLAP
        start = next_start if next_start > start else end

    return chunks


def _chunk_file(path: Path, tier: int = 3) -> list[Chunk]:
    """Load a file via the document loader and split into overlapping chunks."""
    try:
        from src.document_loaders import get_loader
        doc = get_loader(path).load()
        return _chunk_text(doc.content, source_file=path.name, file_type=doc.file_type, tier=tier)
    except Exception:
        return []
