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
    "new_employee":       "New Employee Onboarding (Team Lead / Trainer as SME)",
    "business_user":      "Business User / Operations Staff",
    "expert":             "Subject Matter Expert",
    "business":           "Executive / Management",
    "developer":          "Technical / Developer",
    "compliance":         "Compliance / Audit Staff",
    "trainer":            "Trainer / Learning Designer",
    "care_coordinator":   "Care Coordinator / Clinical Staff",
    "claims_adjudicator": "Claims Adjudicator",
    "member_services":    "Member Services Representative",
    "custom":             "Other (describe below)",
}

# One-line plain-English description of each audience — shown in the UI so the
# user can confirm the LLM will write for the right reader.
AUDIENCE_DESCRIPTIONS: dict[str, str] = {
    "new_employee":       "interviews a team lead, process owner, or trainer who onboards new staff — they know the complete process end-to-end and fill in what newcomers need but the docs leave out",
    "business_user":      "a business operations user who works with screens and spreadsheets daily and knows the domain well but is not a technical professional",
    "expert":             "a subject matter expert with deep domain and technical knowledge",
    "business":           "an executive or manager who needs to understand outcomes, risks, and decisions — not technical implementation details",
    "developer":          "a software developer or engineer responsible for implementation or maintenance",
    "compliance":         "a compliance officer or auditor focused on regulatory requirements, controls, audit trails, and policy adherence",
    "trainer":            "a trainer or instructional designer building learning materials who needs content chunked into teachable steps with context for classroom or e-learning use",
    "care_coordinator":   "a care coordinator or clinical staff member familiar with health insurance workflows, medical terminology, and member care processes",
    "claims_adjudicator": "a claims adjudicator who processes insurance claims and needs exact adjudication rules, edit/reason codes, and coordination of benefits logic",
    "member_services":    "a member services representative answering member calls who needs fast-lookup summaries, escalation scripts, and member-facing explanations",
    "custom":             "",
}

# Groups used to visually separate audiences in the UI selector.
# Order within each group determines display order; group order determines column order.
AUDIENCE_GROUPS: dict[str, list[str]] = {
    "Experience Level":          ["new_employee", "business_user", "expert"],
    "Business & Technology":     ["business", "developer", "compliance", "trainer"],
    "Health Insurance Ops":      ["care_coordinator", "claims_adjudicator", "member_services"],
    "Other":                     ["custom"],
}

# Instruction text injected into every LLM prompt so writing style matches
# the intended reader.  Keyed by the same keys as AUDIENCE_LABELS.
AUDIENCE_NOTES: dict[str, str] = {
    "compliance": (
        "IMPORTANT — Audience: Compliance officers, internal auditors, and regulatory staff "
        "responsible for ensuring adherence to laws, standards, and internal policies.\n\n"
        "WRITING RULES:\n"
        "• Map every process step to the regulation, standard, or policy it satisfies "
        "(e.g. HIPAA §164.312, NCQA UM Standard, state insurance code)\n"
        "• Identify the control at each step: who approves, what is validated, what is logged\n"
        "• Describe what evidence is produced — logs, reports, signatures — and how long it "
        "is retained\n"
        "• Clearly assign ownership: who is responsible for compliance at each stage and who "
        "signs off on exceptions\n"
        "• Flag any gaps where a required control, documentation step, or audit trail is absent\n"
        "• INCLUDE: segregation-of-duties details, exception approval workflows, and any "
        "known audit findings or remediation history\n"
        "• Be precise about dates, deadlines, and regulatory thresholds — round numbers or "
        "approximations are insufficient for audit purposes"
    ),
    "trainer": (
        "IMPORTANT — Audience: Trainers and instructional designers who will turn this "
        "material into classroom sessions, job aids, or e-learning modules.\n\n"
        "WRITING RULES:\n"
        "• Begin each section with a clear learning objective: 'After this section, the "
        "learner will be able to…'\n"
        "• Explain the REASON behind each step — learners retain steps they understand, "
        "not just steps they memorize\n"
        "• Break every process into the smallest teachable units: one concept or action per "
        "step, no compound steps\n"
        "• Include at least one concrete example, sample case, or scenario for each major "
        "process stage\n"
        "• Identify knowledge-check points: where in the flow a learner's understanding "
        "should be tested before proceeding\n"
        "• Flag prerequisites: what background knowledge, system access, or prior training "
        "a learner needs before starting\n"
        "• Use active, imperative language ('Click', 'Enter', 'Verify') so steps can be "
        "used directly as job aids"
    ),
    "claims_adjudicator": (
        "IMPORTANT — Audience: Claims adjudicators who review, process, and adjudicate "
        "health insurance claims — deeply familiar with claims terminology and workflow.\n\n"
        "WRITING RULES:\n"
        "• Claims terminology is expected: EOB, COB, primary/secondary payer, timely filing, "
        "edit codes, reason codes, ANSI X12 835/837, NPI, procedure modifier, place of service\n"
        "• For every decision point, state the specific rule: which edit fires, what the "
        "threshold is, which benefit applies, and what the system action is\n"
        "• Document all edit codes and denial reason codes with their exact meaning and the "
        "correct adjudicator response for each\n"
        "• Coordination of benefits: specify the COB method (standard, non-duplication, "
        "maintenance of benefits), payer sequencing rules, and crossover claim handling\n"
        "• Include timely filing windows, retro-adjustment limits, and late submission "
        "override procedures\n"
        "• Describe override and manual review paths: when to escalate, what documentation "
        "is required, and who has approval authority\n"
        "• Be precise about dollar amounts, percentages, and day counts — ranges are "
        "not acceptable where exact values exist"
    ),
    "member_services": (
        "IMPORTANT — Audience: Member services representatives who answer member calls and "
        "need to quickly find answers, explain outcomes, and resolve issues.\n\n"
        "WRITING RULES:\n"
        "• Write in plain, conversational language that can be read aloud to a member\n"
        "• For every process outcome (approval, denial, pending), provide the member-facing "
        "explanation: what happened, why, and what the member can do next\n"
        "• Include turnaround times and timelines that staff can quote to members: "
        "'You can expect a letter within X business days'\n"
        "• Document escalation triggers: when to transfer, who to transfer to, and what "
        "information to gather and summarize before the handoff\n"
        "• Provide standard responses or talking points for the most common member "
        "questions, objections, and complaints\n"
        "• Explain coverage limits, exclusions, cost-share, and network rules in terms "
        "a member without insurance knowledge can understand\n"
        "• AVOID: internal system names, technical codes, and operational jargon that "
        "would confuse or alarm a member if accidentally repeated"
    ),
    "business_user": (
        "IMPORTANT — Audience: Business operations staff who use these systems day-to-day — "
        "entering data, processing transactions, running reports. They know the business domain "
        "well but are NOT IT professionals.\n\n"
        "WRITING RULES:\n"
        "• Use exact screen names, menu paths, button labels, and field names exactly as they "
        "appear in the system\n"
        "• One action per step — state what the user clicks, types, or selects\n"
        "• State what the screen shows in response to each action\n"
        "• Business and domain terminology is fine — these staff know the business context\n"
        "• AVOID: technical terms (API, database, schema, program name, batch job, server), "
        "infrastructure concepts, code references\n"
        "• For data entry fields: specify the exact field name and any acceptable values, "
        "formats, or validation rules the user needs to know\n"
        "• Explain what error messages mean in plain language and what the user should do next"
    ),
    "care_coordinator": (
        "IMPORTANT — Audience: Care coordinators and clinical support staff in health insurance "
        "operations — familiar with medical terminology, authorization workflows, and member "
        "services processes.\n\n"
        "WRITING RULES:\n"
        "• Clinical and medical terminology is appropriate: ICD/CPT codes, prior authorization, "
        "utilization management, level of care, care plan, concurrent review, etc.\n"
        "• Tie every process step to its member-facing or clinical context — explain why this "
        "step matters for the member's care or the authorization decision\n"
        "• Reference standard health insurance workflows by name: prior auth, concurrent review, "
        "discharge planning, case management, appeals and grievances\n"
        "• Include regulatory context where relevant: HIPAA privacy rules, required turnaround "
        "times, state mandates, NCQA/URAC standards\n"
        "• Clearly document escalation paths: who to contact for clinical exceptions, urgent "
        "cases, or peer-to-peer review requests\n"
        "• AVOID: deep technical system internals — focus on what staff need to do and why, "
        "not how the system processes the request behind the scenes"
    ),
    "new_employee": (
        "IMPORTANT — Audience: People on their very first day — zero prior knowledge of "
        "these systems, processes, or business domain. The document is written FOR new "
        "employees; gaps are filled by interviewing the team lead or trainer who onboards them.\n\n"
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
