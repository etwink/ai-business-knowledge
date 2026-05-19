"""
ProcessContextAgent — a short chat that establishes the process context before analysis.

The user identifies a foundation document (the authoritative source) and/or describes
the overall purpose of the process.  The resulting ProcessContext is injected into every
downstream synthesis prompt so the LLM builds around the right anchor.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm_integration import AzureLLMClient


@dataclass
class ProcessContext:
    foundation_document: Optional[str] = None   # Exact filename from available docs
    process_description: str = ""               # User-described purpose
    additional_notes: str = ""                  # Any extra context captured in conversation

    def is_set(self) -> bool:
        return bool(self.foundation_document or self.process_description)

    def to_prompt_block(self) -> str:
        """Return a formatted context block ready to inject into any LLM prompt."""
        if not self.is_set():
            return ""
        parts = ["[USER-PROVIDED PROCESS CONTEXT — treat this as the highest-priority guide]"]
        if self.foundation_document:
            parts.append(
                f"Foundation document: {self.foundation_document}\n"
                "  → This document is the authoritative source for the overall process. "
                "Its terminology, structure, and business intent MUST be treated as the "
                "primary reference when interpreting all other documents and code."
            )
        if self.process_description:
            parts.append(
                f"Process purpose (as described by the user):\n{self.process_description}"
            )
        if self.additional_notes:
            parts.append(f"Additional context from the user:\n{self.additional_notes}")
        parts.append("[END USER-PROVIDED PROCESS CONTEXT]")
        return "\n\n".join(parts)


_CONTEXT_SYSTEM_PROMPT = (
    "You are helping a user set context before their business process documents are analyzed. "
    "Your only job is to understand: (1) which document, if any, should be treated as the "
    "foundation or authoritative source, and (2) what the overall business process does and "
    "why it exists. Ask focused follow-up questions if the user's answer is vague. "
    "Be concise — this is a short scoping conversation, not a deep dive.\n\n"
    "When you have gathered enough information to understand the scope, output this marker "
    "on its own line (the user will NOT see it):\n"
    "<<CONTEXT_READY>>\n"
    "Then immediately give a brief 2-3 sentence confirmation of what you understood."
)


class ProcessContextAgent:
    """Short conversational agent that establishes the process context before analysis."""

    def __init__(self, available_documents: list[str]):
        self.llm = AzureLLMClient()
        self.available_documents = available_documents
        self.chat_history: list[dict] = []
        self.extracted_context: Optional[ProcessContext] = None
        self.context_ready: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_message(self) -> str:
        """Generate the opening question."""
        doc_list = "\n".join(f"  • {d}" for d in self.available_documents[:40])
        if len(self.available_documents) > 40:
            doc_list += f"\n  ... and {len(self.available_documents) - 40} more files"

        prompt = (
            "You are helping a user set context before their business process documents are analyzed.\n\n"
            f"Available documents/files:\n{doc_list}\n\n"
            "Write a concise, friendly opening message (3–5 sentences) that:\n"
            "1. Explains that before diving into the analysis you want to make sure the "
            "documentation is built around the right understanding of the process\n"
            "2. Asks two specific questions:\n"
            "   a) Is there one document from the list above that should be treated as the "
            "foundation or authoritative source? (e.g. the main requirements doc, the "
            "primary specification, or the document that best describes the overall process)\n"
            "   b) In their own words — what does this process do and why does it exist?\n"
            "3. Makes clear that either answer alone is helpful — they can answer one or both"
        )
        msg = self.llm.query(prompt)
        self.chat_history.append({"role": "assistant", "content": msg})
        return msg

    def chat(self, user_message: str) -> tuple[str, Optional[ProcessContext]]:
        """
        Process a user message.

        Returns
        -------
        (response_text, extracted_context)
            extracted_context is non-None only when the agent has flagged <<CONTEXT_READY>>.
        """
        self.chat_history.append({"role": "user", "content": user_message})

        input_messages = [
            {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
            *self.chat_history,
        ]
        raw = self.llm.query_raw(input_messages)

        ready = "<<CONTEXT_READY>>" in raw
        clean = raw.replace("<<CONTEXT_READY>>", "").strip()
        self.chat_history.append({"role": "assistant", "content": clean})

        extracted: Optional[ProcessContext] = None
        if ready:
            self.context_ready = True
            extracted = self._extract_context()
            self.extracted_context = extracted

        return clean, extracted

    def force_extract(self) -> ProcessContext:
        """
        Extract whatever context has been gathered so far, even without <<CONTEXT_READY>>.
        Called when the user clicks "Confirm" without the agent having flagged readiness.
        """
        ctx = self._extract_context()
        self.extracted_context = ctx
        self.context_ready = True
        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_context(self) -> ProcessContext:
        """Use a structured LLM call to pull out foundation_document and process_description."""
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.chat_history
        )
        available_json = json.dumps(self.available_documents[:40])

        prompt = (
            "Read the conversation below and extract the context the user provided.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Available document filenames: {available_json}\n\n"
            "Return ONLY valid JSON (no markdown fences, no explanation):\n"
            '{"foundation_document": "<exact filename from the list if the user named one, otherwise null>", '
            '"process_description": "<the user\'s description of what the process does and why it exists — '
            'preserve their wording as much as possible>", '
            '"additional_notes": "<any other useful context the user mentioned, or empty string>"}'
        )

        response = self.llm.query(prompt, max_tokens=800)
        try:
            clean = re.sub(r"```(?:json)?|```", "", response).strip()
            data = json.loads(clean)
            return ProcessContext(
                foundation_document=data.get("foundation_document") or None,
                process_description=data.get("process_description", ""),
                additional_notes=data.get("additional_notes", ""),
            )
        except Exception:
            # Fallback: treat the full user conversation as the description
            user_text = " ".join(
                m["content"] for m in self.chat_history if m["role"] == "user"
            )
            return ProcessContext(process_description=user_text)
