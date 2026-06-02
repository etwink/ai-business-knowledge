"""Azure OpenAI LLM integration module."""

from typing import Optional
from openai import AzureOpenAI
import config
from .llm_logger import log_call


class AzureLLMClient:
    """Client for interacting with Azure OpenAI."""

    def __init__(self):
        """Initialize Azure OpenAI client."""
        config.validate_config()
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        self.deployment_name = config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.reasoning_effort = config.MODEL_REASONING_EFFORT
        self.max_tokens = config.MODEL_MAX_TOKENS

    def query(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send a query to the LLM and get a response.

        Args:
            prompt: The user message/prompt
            system_message: Optional system message to set context
            temperature: Ignored — reasoning models do not use temperature
            max_tokens: Optional override for max output tokens

        Returns:
            The LLM response text
        """
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        input_messages = []
        if system_message:
            input_messages.append({"role": "system", "content": system_message})
        input_messages.append({"role": "user", "content": prompt})

        response = self.client.responses.create(
            model=self.deployment_name,
            input=input_messages,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=tokens,
        )

        log_call("query", input_messages, response.output_text, self.deployment_name, tokens)
        return response.output_text

    def query_raw(
        self,
        input_messages: list[dict],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a pre-built message list directly to the API (used by conversational agent)."""
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        response = self.client.responses.create(
            model=self.deployment_name,
            input=input_messages,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=tokens,
        )
        log_call("query_raw", input_messages, response.output_text, self.deployment_name, tokens)
        return response.output_text

    def query_with_context(
        self,
        context: str,
        question: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Query the LLM with additional context.

        Args:
            context: Background information/document context
            question: The question to ask
            system_message: Optional system message

        Returns:
            The LLM response
        """
        prompt = f"""Context:
{context}

Question:
{question}"""
        return self.query(prompt, system_message)


class PromptBuilder:
    """Helper class to build structured prompts for analysis tasks."""

    @staticmethod
    def build_document_summary_prompt(document_content: str) -> str:
        """Build prompt for document summarization."""
        return f"""Analyze the following document and respond using EXACTLY these labeled sections:

SUMMARY:
Write 2-3 sentences describing the document's purpose and main objectives.

KEY PROCESSES:
List each distinct process, workflow, or business step described in this document. One dash-prefixed bullet per item.

SYSTEMS MENTIONED:
List each system, application, database, file, or external component named in this document. One dash-prefixed bullet per item.

TECHNICAL DETAILS:
List each important technical fact, data structure, format, or implementation detail. One dash-prefixed bullet per item.

Document:
{document_content[:10000]}"""

    @staticmethod
    def build_process_document_prompt(summaries: list[str]) -> str:
        """Build prompt for creating a comprehensive process document."""
        combined = "\n\n".join([f"Document {i+1}:\n{s}" for i, s in enumerate(summaries)])
        return f"""Based on the following document summaries, create a comprehensive process document.

Summaries:
{combined}

Write a detailed process document using EXACTLY these section headers, each on its own line followed by a colon:

OVERVIEW:
(High-level description of the overall business process and purpose)

INTEGRATED PROCESSES:
(Step-by-step description of all integrated processes and workflows)

DEPENDENCIES:
(All dependencies between processes, systems, and components)

DATA FLOW:
(How data moves through the system — inputs, transformations, and outputs)

DECISION POINTS:
(Key decision points, business rules, and conditional logic)

SYSTEMS AND COMPONENTS:
(All systems, applications, databases, and external components involved)

Use exactly these header names. Do not rename, reorder, or add sections.
The underlying process looks for the header names to structure the output, so it's critical to follow the format precisely. Each header name must be followed by a colon and then the content for that section. Do not include any additional text outside of these sections. Only the header name should be on the line by itself, followed by the relevant content for that section.

An example output format:
DEPENDENCIES:
### Inter-subsystem dependencies and common call chains
* (bullet points describing dependencies)"""

    @staticmethod
    def build_gap_analysis_prompt(process_document: str) -> str:
        """Build prompt for identifying gaps and missing information."""
        return f"""Analyze the following process document and respond using EXACTLY these labeled sections:

MISSING STEPS:
List each missing or incomplete process step. One dash-prefixed bullet per item.

UNDEFINED DEPENDENCIES:
List each unexplained technical dependency or reference. One dash-prefixed bullet per item.

INCOMPLETE TRANSFORMATIONS:
List each undefined or unclear data transformation. One dash-prefixed bullet per item.

MISSING INTEGRATIONS:
List each missing or undocumented system integration. One dash-prefixed bullet per item.

ERROR HANDLING GAPS:
List each area where error handling is absent or incomplete. One dash-prefixed bullet per item.

SECURITY GAPS:
List each missing security or compliance consideration. One dash-prefixed bullet per item.

RESOURCE GAPS:
List each undefined role, team, or resource. One dash-prefixed bullet per item.

Process Document:
{process_document}"""

    # ------------------------------------------------------------------
    # Per-section process document prompts
    # These are used when building the process document section-by-section
    # to avoid a single large call running out of output tokens.
    # ------------------------------------------------------------------

    # Writing style instruction injected into every section prompt.
    _AUDIENCE_NOTE = (
        "IMPORTANT — Audience: This document will be read by people who have NO prior knowledge "
        "of these systems, processes, or business domain. Write as if handing this to a new "
        "employee on their first day. Never assume the reader already knows what a component "
        "does or why it exists. Every time you mention a subprocess, program, file, system, "
        "team, or business concept, you MUST briefly explain: (1) what it is, (2) what it does, "
        "(3) who owns or is responsible for it, and (4) why it exists in the first place. "
        "Avoid acronyms without first spelling them out. Write in plain language."
    )

    _SECTION_INSTRUCTIONS: dict = {
        "overview": (
            "Write a high-level introduction to the complete system. "
            "Assume the reader knows nothing about this business or these systems. "
            "Cover: (1) the business problem this system solves and why it was built, "
            "(2) the primary business objectives it achieves, (3) who uses it and who is "
            "affected by it (roles, teams, external parties), (4) a plain-language summary "
            "of what the system does from start to finish, and (5) a brief map of how the "
            "major subsystems relate to each other. This section sets the stage for everything "
            "that follows — a reader should finish it understanding WHY this system exists."
        ),
        "integrated_processes": (
            "Write a detailed, step-by-step end-to-end workflow description. "
            "Number every step. For each step: state what happens, which program/module/person "
            "performs it, why that step is necessary, and what the output of the step is. "
            "When you first mention any program, module, or external system in a step, add a "
            "parenthetical explanation of what it is (e.g., 'PAYCALC (the COBOL program "
            "responsible for calculating gross pay from time-card inputs)'). "
            "Trace the complete flow from the initial trigger or business event through every "
            "processing stage to the final output or outcome."
        ),
        "dependencies": (
            "Describe all dependencies between subsystems and components. "
            "Organise by subsystem. For each dependency entry: name the component that relies "
            "on another, name the component being relied upon, explain in plain English WHY "
            "this dependency exists (what would break if it were removed), and describe any "
            "shared data structures, files, or interfaces involved. "
            "If a component is mentioned for the first time, include a one-sentence explanation "
            "of what it is before describing the dependency."
        ),
        "data_flow": (
            "Describe how data moves through the entire system. "
            "For a reader who has never seen this system: explain what each input is "
            "(source, format, business meaning), how it is transformed or processed at each "
            "stage (and why that transformation is needed), where it is stored along the way "
            "(file names, table names — with a plain-language description of each), and what "
            "the final outputs are (destination, format, who uses them and for what purpose). "
            "Use a logical flow — follow the data chronologically from entry to exit."
        ),
        "decision_points": (
            "List and explain every significant business rule, conditional logic, and decision "
            "point in the system. "
            "For each decision point: (1) describe in plain English the condition being "
            "evaluated, (2) explain the possible outcomes and what happens in each case, "
            "(3) explain the BUSINESS REASON this decision exists (the rule it enforces or "
            "the risk it manages), and (4) identify which program, module, or person applies "
            "it. Do not assume the reader knows the domain — spell out what each rule means "
            "in business terms."
        ),
        "systems_and_components": (
            "Provide a comprehensive, plain-language inventory of every system and component "
            "involved. "
            "Organise into clearly labelled categories such as: Core Processing Programs, "
            "Shared Modules and Copybooks, Input Data Sources, Output Files and Reports, "
            "Databases and Data Stores, External Interfaces, and Batch/Scheduled Jobs. "
            "For EVERY item listed: give its technical name, a plain-English description of "
            "what it does, who is responsible for it, and how it connects to the rest of "
            "the system. Write as if creating a reference guide for a new team member."
        ),
        "process_flow_ascii": (
            "Generate a plain-text process flow diagram suitable for PRINTING ON PAPER.\n"
            "Rules:\n"
            "- Use Unicode box-drawing characters for the diagram structure.\n"
            "- Rectangular process steps:  ┌─────────────┐\n"
            "                              │  Step Name  │\n"
            "                              └─────────────┘\n"
            "- Diamond decision points:       ╱           ╲\n"
            "                               ╱  Question?   ╲\n"
            "                               ╲              ╱\n"
            "                                ╲            ╱\n"
            "- Vertical flow: │ and ▼\n"
            "- Horizontal branches: ──── and → with Yes/No labels\n"
            "- Keep step labels short (max 35 characters).\n"
            "- Aim for 10–18 nodes so it fits on one printed page.\n"
            "- Output ONLY the diagram — no Mermaid syntax, no explanation, no prose."
        ),
        "process_flow_diagram": (
            "Generate a Mermaid flowchart diagram of the end-to-end process flow.\n"
            "Output ONLY valid Mermaid syntax — no markdown fences, no explanation, no prose.\n"
            "Rules:\n"
            "- First line must be: flowchart TD\n"
            "- Process steps: rectangle  A[Step Name]\n"
            "- Decision points: diamond   B{Condition?}\n"
            "- Data stores / files: cylinder  C[(File or DB)]\n"
            "- External systems: stadium  D([External System])\n"
            "- Arrows: A --> B  or  A -->|label| B\n"
            "- Keep it to 10–20 nodes for readability.\n"
            "- Node IDs must be short alphanumeric tokens (A, B, CALC, etc.).\n"
            "- Labels must not contain parentheses, quotes, or special characters except hyphens.\n"
            "Focus on the business process flow from trigger to final outcome."
        ),
        "appendix": (
            "Create a reference appendix containing the following sub-sections. "
            "Each sub-section should begin with a clear heading.\n\n"
            "A. GLOSSARY — Define every technical term, acronym, system name, and "
            "domain-specific phrase used in this document. List alphabetically. "
            "Each entry: term/acronym — full name and plain-English definition.\n\n"
            "B. COMPONENT INDEX — A quick-reference table (formatted as a list) of every "
            "program, file, database, and system mentioned. For each: name, type "
            "(program / file / database / external system), and one-sentence purpose.\n\n"
            "C. ROLES AND RESPONSIBILITIES — List every role, team, or person type "
            "referenced in the document. For each: role name, which part of the process "
            "they own, and what they are responsible for.\n\n"
            "D. KEY ASSUMPTIONS — List any assumptions made in producing this document "
            "due to incomplete information, and note what would need to be confirmed to "
            "validate them."
        ),
    }

    @classmethod
    def build_section_prompt_from_clusters(
        cls, section_key: str, cluster_summaries: list[str], context_block: str = ""
    ) -> str:
        """Build a prompt for a single process document section using cluster summaries."""
        combined = "\n\n".join(
            f"--- Cluster {i+1} ---\n{s}" for i, s in enumerate(cluster_summaries)
        )
        instruction = cls._SECTION_INSTRUCTIONS.get(section_key, "Write this section.")
        context_part = f"\n{context_block}\n" if context_block else ""

        if section_key == "process_flow_diagram":
            return (
                f"You are generating a process flow diagram for a business process.\n"
                f"{context_part}\n"
                f"Cluster Summaries:\n{combined}\n\n"
                f"{instruction}"
            )

        section_label = section_key.replace("_", " ").upper()
        # When a context_block is provided it already contains the audience note,
        # so suppress the hardcoded _AUDIENCE_NOTE to avoid duplication.
        audience_instruction = cls._AUDIENCE_NOTE if not context_block else ""
        return (
            f"You are writing one section of a business process document.\n"
            f"{audience_instruction}\n"
            f"{context_part}\n"
            f"Below are summaries of all subsystems (COBOL programs and business documents).\n\n"
            f"Cluster Summaries:\n{combined}\n\n"
            f"Write ONLY the {section_label} section of the process document.\n"
            f"{instruction}\n"
            f"Do not include any other section headers or content. "
            f"Output plain prose and/or bullet lists — no markdown fences."
        )

    @classmethod
    def build_section_prompt_from_analyses(
        cls, section_key: str, summaries: list[str], context_block: str = ""
    ) -> str:
        """Build a prompt for a single process document section using document summaries."""
        combined = "\n\n".join(f"Document {i+1}:\n{s}" for i, s in enumerate(summaries))
        instruction = cls._SECTION_INSTRUCTIONS.get(section_key, "Write this section.")
        context_part = f"\n{context_block}\n" if context_block else ""

        if section_key == "process_flow_diagram":
            return (
                f"You are generating a process flow diagram for a business process.\n"
                f"{context_part}\n"
                f"Document Summaries:\n{combined}\n\n"
                f"{instruction}"
            )

        section_label = section_key.replace("_", " ").upper()
        audience_instruction = cls._AUDIENCE_NOTE if not context_block else ""
        return (
            f"You are writing one section of a business process document.\n"
            f"{audience_instruction}\n"
            f"{context_part}\n"
            f"Below are summaries of the analyzed documents.\n\n"
            f"Document Summaries:\n{combined}\n\n"
            f"Write ONLY the {section_label} section of the process document.\n"
            f"{instruction}\n"
            f"Do not include any other section headers or content. "
            f"Output plain prose and/or bullet lists — no markdown fences."
        )

    @staticmethod
    def build_clarification_questions_prompt(
        process_document: str,
        gap_analysis: str
    ) -> str:
        """Build prompt for generating clarification questions."""
        return f"""Based on the process document and gap analysis, generate clarification questions to enhance documentation:

Process Document:
{process_document}

Gap Analysis:
{gap_analysis}

Generate 10-15 specific, technical questions that would help clarify:
- Ambiguous processes
- Undefined dependencies
- Missing details
- Edge cases and error scenarios

Format each question clearly and explain why it's important for the documentation."""
