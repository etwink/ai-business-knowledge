"""
Two-pass process context extraction.

Pass 1 — Document detection: check whether the user mentioned a specific file
          from the available set; if so, read its contents.
Pass 2 — Context extraction: pull a structured ProcessContext out of the user's
          description and (optionally) the document content.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.llm_integration import AzureLLMClient

_MAX_DOC_CHARS = 8_000   # how much of the reference doc to feed into Pass 2

# Human-readable labels shown in the UI audience selector.
AUDIENCE_LABELS: dict[str, str] = {
    "new_employee": "New Employee / No Prior Knowledge",
    "business":     "Business / Executive",
    "developer":    "Technical / Developer",
    "expert":       "Subject Matter Expert",
}

# Instruction text injected into every LLM prompt so writing style matches
# the intended reader.  Keyed by the same keys as AUDIENCE_LABELS.
AUDIENCE_NOTES: dict[str, str] = {
    "new_employee": (
        "IMPORTANT — Audience: This document will be read by people who have NO prior "
        "knowledge of these systems, processes, or business domain. Write as if handing "
        "this to a new employee on their first day. Never assume the reader already knows "
        "what a component does or why it exists. Every time you mention a subprocess, "
        "program, file, system, team, or business concept, briefly explain: (1) what it is, "
        "(2) what it does, (3) who owns it, and (4) why it exists. Avoid acronyms without "
        "first spelling them out. Write in plain language."
    ),
    "business": (
        "IMPORTANT — Audience: This document will be read by business executives and "
        "stakeholders who understand company operations but not technical implementation "
        "details. Focus on business purpose, outcomes, risks, and decisions. Avoid "
        "low-level implementation specifics (file formats, code logic, protocols) unless "
        "directly relevant to a business decision. Use business terminology, not technical "
        "jargon. When you must mention a technical system, describe only what it does in "
        "business terms."
    ),
    "developer": (
        "IMPORTANT — Audience: This document will be read by experienced software "
        "developers and technical architects. Use precise technical terminology freely. "
        "Focus on implementation details: data structures, interfaces, APIs, file formats, "
        "processing logic, error handling, and system dependencies. Include specific "
        "program names, file names, and technical specifications. Skip high-level business "
        "context unless it directly influences a technical design decision."
    ),
    "expert": (
        "IMPORTANT — Audience: This document will be read by subject matter experts who "
        "are deeply familiar with the business domain and existing systems. Assume full "
        "knowledge of all terminology, acronyms, and standard processes. Focus on nuances, "
        "edge cases, cross-system interactions, and exceptions to standard rules. Skip "
        "basic definitions and standard process descriptions."
    ),
}


@dataclass
class ProcessContext:
    foundation_document: Optional[str] = None   # Exact filename, if one was identified
    process_description: str = ""               # Plain-language purpose of the process
    additional_notes: str = ""                  # Any other useful detail the user provided

    def is_set(self) -> bool:
        return bool(self.foundation_document or self.process_description)

    def to_prompt_block(self) -> str:
        """Return a formatted block ready to inject into any downstream LLM prompt."""
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


class ProcessContextAgent:
    """
    Extracts a ProcessContext from a single free-text user description.

    Usage
    -----
    agent = ProcessContextAgent(available_files)
    ctx, found_doc = agent.extract_from_input(user_text)
    # found_doc is the filename that was detected and read (or None)
    """

    def __init__(self, available_files: list[Path]):
        self.llm = AzureLLMClient()
        self.available_files = available_files
        self._file_names: list[str] = [f.name for f in available_files]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_input(
        self, user_text: str
    ) -> tuple["ProcessContext", Optional[str]]:
        """
        Two-pass extraction.

        Returns (ProcessContext, detected_doc_name | None).
        The caller can use detected_doc_name to show the user which document was read.
        """
        # Pass 1: did the user mention a specific document?
        detected_doc = self._identify_reference_doc(user_text)

        # Read the document if one was found
        doc_content = ""
        if detected_doc:
            doc_content = self._read_doc(detected_doc)

        # Pass 2: extract structured context
        ctx = self._extract_context(user_text, doc_content, detected_doc)
        return ctx, detected_doc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _identify_reference_doc(self, user_text: str) -> Optional[str]:
        """
        Pass 1 — ask the LLM whether the user's text refers to a specific file.
        Returns the exact filename or None.
        """
        if not self._file_names:
            return None

        file_list = json.dumps(self._file_names[:60])
        prompt = (
            f"The user wrote:\n\"{user_text}\"\n\n"
            f"Available files:\n{file_list}\n\n"
            "Does the user's text refer to a specific file from the list above as a "
            "reference or foundation document? Look for filename mentions, partial names, "
            "or clear references like 'see X', 'based on X', 'use X as the guide'.\n\n"
            "Return ONLY valid JSON (no markdown, no explanation):\n"
            '{"document_name": "<exact filename from the list, or null if none mentioned>"}'
        )
        try:
            raw = self.llm.query(prompt, max_tokens=200)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            name = data.get("document_name")
            # Validate the name is actually in our list (LLM can hallucinate)
            if name and any(f.lower() == name.lower() for f in self._file_names):
                # Return the canonically-cased name from the list
                return next(f for f in self._file_names if f.lower() == name.lower())
        except Exception:
            pass
        return None

    def _read_doc(self, doc_name: str) -> str:
        """Read the document content for use in Pass 2."""
        path = next((f for f in self.available_files if f.name == doc_name), None)
        if not path:
            return ""
        try:
            from src.document_loaders import get_loader
            doc = get_loader(path).load()
            return doc.content[:_MAX_DOC_CHARS]
        except Exception:
            try:
                return path.read_text(encoding="utf-8", errors="ignore")[:_MAX_DOC_CHARS]
            except Exception:
                return ""

    def _extract_context(
        self,
        user_text: str,
        doc_content: str,
        foundation_doc: Optional[str],
    ) -> "ProcessContext":
        """
        Pass 2 — extract structured ProcessContext from the user's text and
        (optionally) the content of the reference document.
        """
        doc_section = (
            f"\n\nContent of the reference document ({foundation_doc}):\n{doc_content}"
            if doc_content else ""
        )
        prompt = (
            f"The user provided this description of their business process:\n\"{user_text}\""
            f"{doc_section}\n\n"
            "Extract a structured process context. Return ONLY valid JSON:\n"
            '{\n'
            '  "process_description": "<2-4 sentence plain-English description of what '
            'this process does and why it exists — synthesise the user text and document '
            'content if both are present>",\n'
            '  "additional_notes": "<any other useful detail: teams, systems, constraints, '
            'environment — or empty string if nothing else was mentioned>"\n'
            '}'
        )
        try:
            raw = self.llm.query(prompt, max_tokens=600)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            return ProcessContext(
                foundation_document=foundation_doc,
                process_description=data.get("process_description", user_text),
                additional_notes=data.get("additional_notes", ""),
            )
        except Exception:
            return ProcessContext(
                foundation_document=foundation_doc,
                process_description=user_text,
            )
