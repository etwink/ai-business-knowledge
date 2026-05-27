"""
RAG agent: retrieve relevant chunks from the knowledge base and generate
a grounded answer using the Azure LLM.
"""

from __future__ import annotations

from .knowledge_base import KnowledgeBase

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant helping users understand business processes, "
    "systems, and procedures. Answer questions using ONLY the provided context excerpts. "
    "If the context does not contain enough information to answer fully, say so explicitly "
    "rather than guessing. Always state which source document(s) your answer draws from."
)

_MAX_HISTORY_TURNS = 4   # Number of prior exchanges to include for context


class RAGAgent:
    """
    Conversational RAG agent backed by a KnowledgeBase.

    Each call to chat() retrieves relevant chunks, builds a prompt with
    conversation history, and returns both the LLM answer and the
    source chunks so the UI can show citations.
    """

    def __init__(self, knowledge_base: KnowledgeBase, top_k: int = 6):
        self.kb = knowledge_base
        self.top_k = top_k
        self._history: list[dict] = []  # {"role": "user"|"assistant", "content": str}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> tuple[str, list[dict]]:
        """
        Retrieve relevant chunks and generate a grounded answer.

        Returns:
            answer (str) — LLM response
            sources (list[dict]) — retrieved chunks with score, source_file, text
        """
        from src.llm_integration import AzureLLMClient

        # Retrieve
        chunks = self.kb.search(user_message, top_k=self.top_k)

        # Build context block
        context_parts = []
        for i, c in enumerate(chunks, 1):
            score_pct = int(c["score"] * 100)
            context_parts.append(
                f"[{i}] Source: {c['source_file']} (relevance {score_pct}%)\n{c['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(no relevant context found)"

        # Build history snippet
        history_text = self._format_history()

        prompt = (
            f"Context excerpts from the knowledge base:\n\n"
            f"{context_block}\n\n"
            f"{'Conversation so far:' + chr(10) + history_text + chr(10) if history_text else ''}"
            f"User question: {user_message}\n\n"
            f"Answer based on the context above. "
            f"Cite source documents by number (e.g. [1], [2]) where relevant. "
            f"If the answer spans multiple sources, integrate them cohesively."
        )

        llm = AzureLLMClient()
        answer = llm.query(prompt, system_message=_SYSTEM_PROMPT, max_tokens=3000)

        # Update history
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})

        return answer, chunks

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_history(self) -> str:
        recent = self._history[-(2 * _MAX_HISTORY_TURNS):]
        if not recent:
            return ""
        lines = []
        for m in recent:
            role = m["role"].upper()
            # Truncate long history entries
            content = m["content"][:300] + ("…" if len(m["content"]) > 300 else "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
