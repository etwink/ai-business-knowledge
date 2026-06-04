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
    "custom":       "Other (describe below)",
}

# Instruction text injected into every LLM prompt so writing style matches
# the intended reader.  Keyed by the same keys as AUDIENCE_LABELS.
AUDIENCE_NOTES: dict[str, str] = {
    "new_employee": (
        "IMPORTANT — Audience: People on their very first day — zero prior knowledge of "
        "these systems, processes, or business domain.\n\n"
        "WRITING RULES:\n"
        "• Define EVERY term on first use — never assume the reader knows acronyms, system "
        "names, or internal jargon\n"
        "• Spell out every abbreviation in full before using the short form\n"
        "• After naming any system, team, or tool, add a brief parenthetical explaining "
        "what it does and who owns it\n"
        "• Use plain, conversational language — avoid formal or bureaucratic phrasing\n"
        "• If a step requires prior knowledge to execute, say so and point toward where "
        "that knowledge can be found\n"
        "• Organize content so it can be followed chronologically by someone doing it for "
        "the very first time"
    ),
    "business": (
        "IMPORTANT — Audience: Business executives and stakeholders who understand "
        "operations but NOT technical implementation.\n\n"
        "VOCABULARY ENFORCEMENT — apply these rules to every sentence:\n"
        "• NEVER use these terms: COBOL, JCL, copybook, subroutine, mainframe, stored "
        "procedure, PIC clause, WORKING-STORAGE, hex, binary, API, endpoint, schema, "
        "payload, parsing, runtime, compile, regex, SQL, dataset (technical sense), "
        "file extensions like .CBL .CPY .JCL, program names that are not business terms\n"
        "• TRANSLATE automatically: 'batch job' → 'automated process', 'COBOL program' → "
        "'automated system process', 'copybook' → 'shared data template', 'API call' → "
        "'connection to [system name]', 'stored procedure' → 'automated database step', "
        "'subroutine' → 'sub-process', 'schema' → 'data layout'\n"
        "• If a technical file or program name must appear, write what it DOES instead of "
        "its technical name\n"
        "• FOCUS on: business outcomes, who is affected, decisions required, risks, "
        "timelines, and costs\n"
        "• OMIT: technical specifications, code logic, file layouts, system internals"
    ),
    "developer": (
        "IMPORTANT — Audience: Experienced software developers and technical architects.\n\n"
        "WRITING RULES:\n"
        "• Technical terminology is expected and appropriate — COBOL, JCL, copybooks, "
        "APIs, schemas, file formats, etc.\n"
        "• Include specific program names, file names, and identifiers where known\n"
        "• Focus on implementation: data structures, interfaces, processing logic, error "
        "handling, and dependencies\n"
        "• Skip high-level business context unless it directly influences a technical "
        "design decision\n"
        "• Be terse and precise — prefer code references and exact values over narrative\n"
        "• When uncertain, say so explicitly rather than hedging"
    ),
    "expert": (
        "IMPORTANT — Audience: Subject matter experts deeply familiar with this business "
        "domain and its systems.\n\n"
        "WRITING RULES:\n"
        "• Assume full knowledge of all terminology, acronyms, and standard processes — "
        "skip basic definitions\n"
        "• Focus on nuances, edge cases, exceptions, cross-system interactions, and "
        "non-obvious dependencies\n"
        "• Be concise — bullet points and direct statements over narrative prose\n"
        "• Highlight anything that deviates from expected patterns or documented behavior\n"
        "• Include specific identifiers, thresholds, and values wherever they appear in "
        "the source material"
    ),
}

# Response mode labels shown in the UI mode selector.
RESPONSE_MODE_LABELS: dict[str, str] = {
    "summary": "Quick Summary",
    "standard": "Standard",
    "guide": "Step-by-Step Guide",
}

# Detail level labels shown in the UI detail-level selector.
DETAIL_LEVEL_LABELS: dict[str, str] = {
    "overview": "Overview",
    "standard": "Standard",
    "detailed": "In-Depth",
}

# Depth instructions appended to answer-generation prompts alongside response-mode instructions.
DETAIL_LEVEL_INSTRUCTIONS: dict[str, str] = {
    "overview": (
        "DETAIL LEVEL — Overview:\n"
        "• Describe major phases at a high level — what to do and where, not every keystroke\n"
        "• Name the system or screen but do not enumerate every individual field or value\n"
        "• 2–4 bullet points or sentences per major phase is the right granularity\n"
        "• Suitable for an executive who needs to understand the workflow without executing it"
    ),
    "standard": (
        "DETAIL LEVEL — Standard:\n"
        "• Name the exact screen, command, or menu path for every step\n"
        "• Call out the key fields the user must fill in and any important values\n"
        "• Where source materials include specific codes or values, include them\n"
        "• FORBIDDEN — never write these vague phrases:\n"
        "  ✗ 'enter the appropriate data'  →  name the exact field and value\n"
        "  ✗ 'navigate to the correct screen'  →  name the screen/command and path\n"
        "  ✗ 'using the system'  →  name the system and the specific action\n"
        "  ✗ 'fill in the required fields'  →  list the fields by name\n"
        "• If exact values are not in the source material, write '(value from your records)' "
        "rather than omitting the field entirely"
    ),
    "detailed": (
        "DETAIL LEVEL — In-Depth:\n"
        "• Every step must include: exact screen ID or command, exact field name/number, "
        "exact value to enter or select, and what the system displays after the action\n"
        "• Quote field labels and valid values verbatim from the source material\n"
        "• Use keypress-level imperative language: 'Type X', 'Press Enter', 'Tab to field Y', "
        "'Select option Z from the dropdown'\n"
        "• ABSOLUTELY FORBIDDEN — scan for and replace every instance before finalizing:\n"
        "  ✗ 'enter the appropriate data'  →  specify the exact field and value\n"
        "  ✗ 'navigate to the relevant screen'  →  give the screen name and how to reach it\n"
        "  ✗ 'complete the required fields'  →  list each field with its expected value\n"
        "  ✗ 'using the system'  →  name the system, screen, and exact action\n"
        "  ✗ 'as needed', 'as applicable', 'where appropriate'  →  always be specific\n"
        "  ✗ 'enter the correct information'  →  say what information goes in which field\n"
        "• SELF-CHECK RULE: Before finalizing, re-read every step for vague language. "
        "Replace each instance with the specific screen/field/value from the source context. "
        "If the detail is not in the source material, flag it explicitly inline: "
        "'⚠️ (Not documented — consult [system/team name])'\n"
        "• A reader who has never performed this task should be able to follow the steps "
        "without asking anyone for clarification"
    ),
}

# Format/length instructions appended to answer-generation prompts.
RESPONSE_MODE_INSTRUCTIONS: dict[str, str] = {
    "summary": (
        "RESPONSE FORMAT — Quick Summary:\n"
        "• Length: 150–250 words MAXIMUM — shorter is better\n"
        "• Use plain paragraphs or a single tight bullet list (5–7 points max)\n"
        "• Cover WHAT and WHY only — omit HOW it is technically implemented\n"
        "• No section headers, no sub-bullets, no tables\n"
        "• End with one clear takeaway sentence"
    ),
    "standard": (
        "RESPONSE FORMAT — Standard:\n"
        "• Length: Cover the topic fully, but stop when done — no padding or repetition\n"
        "• Use headers and bullet points only where they genuinely improve readability\n"
        "• Cite source documents where relevant\n"
        "• Aim for completeness, not brevity and not exhaustiveness"
    ),
    "guide": (
        "RESPONSE FORMAT — Step-by-Step Guide:\n"
        "• Number every user action — one discrete action per step, never combine steps\n"
        "• For EACH step include three things:\n"
        "  1. WHERE — the screen name, menu path, or system to navigate to\n"
        "  2. WHAT — exact field names, values to enter, or buttons to click\n"
        "  3. RESULT — what the user should see or receive after completing the step\n"
        "• If the source documents contain specific screen names, system commands, or "
        "field values — quote them EXACTLY as written\n"
        "• If exact operational details are NOT available in the source documents, flag "
        "it inline: '(Note: exact steps for [X] are not in the available documentation "
        "— consult the system runbook or IT support.)'\n"
        "• Use imperative verbs: Click, Enter, Select, Navigate to, Verify\n"
        "• No narrative prose — every sentence is a concrete user action\n"
        "• End with a 'Common Issues' section listing known failure points and resolutions "
        "if the source materials mention any"
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
