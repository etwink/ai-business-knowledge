"""Conversational agent for interactive document gap-filling."""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.analyzers import ProcessDocument, GapAnalysis, AnalysisResult
from src.llm_integration import AzureLLMClient


@dataclass
class GapItem:
    category: str
    description: str
    resolved: bool = False
    resolution: str = ""
    related_sources: list[dict] = field(default_factory=list)  # {file, text, score}


@dataclass
class DocumentUpdate:
    document_name: str
    version: int
    content: str
    gap_addressed: str


@dataclass
class AgentResponse:
    message: str
    document_updates: list[DocumentUpdate] = field(default_factory=list)
    gaps_remaining: int = 0
    is_complete: bool = False


_SYSTEM_PROMPT_BASE = """You are a senior business analyst conducting a knowledge-capture interview. \
Your goal is to help the user document their business processes as completely as possible.

Each turn you will receive a [CONTEXT] block (not visible to the user) listing the open gaps \
in the current documentation. These gaps are your guide, not a rigid script.

Your behavior:
- Encourage the user to speak freely and share everything they know about the process. \
  Treat this as an open-ended interview, not a form to fill in.
- Listen for information that addresses ANY of the listed gaps, not just the first one. \
  A single user response may resolve multiple gaps at once.
- Ask follow-up questions that deepen understanding: who owns this? why does it work this way? \
  what happens when it fails? who else is involved?
- When information from the user is sufficient to document a gap — even partially — emit this \
  marker on its own line (the user will NOT see this):
  <<GAP_RESOLVED: {concise summary of what was learned} | {most_relevant_document_name}>>
  You may emit multiple <<GAP_RESOLVED>> markers in a single response if the user's answer \
  addressed multiple gaps.
- If the user cannot answer something, mark it resolved with "N/A — user confirmed unknown" \
  and note it as an assumption to validate later.
- When all listed gaps are addressed, output <<ALL_GAPS_RESOLVED>> and offer a brief summary \
  of what was learned, then ask if there is anything else the user wants to add.

Tone: curious, professional, and encouraging. Show genuine interest in understanding the process. \
Never make the user feel like they are just answering a checklist."""


class ConversationalAgent:
    """Chatbot agent that fills document gaps through conversation and produces per-document updates."""

    def __init__(
        self,
        analyses: list[AnalysisResult],
        process_document: ProcessDocument,
        gap_analysis: GapAnalysis,
        knowledge_base=None,
        audience_note: str = "",
    ):
        self.llm = AzureLLMClient()
        self.analyses = analyses
        self.process_document = process_document
        self.gap_analysis = gap_analysis
        self.knowledge_base = knowledge_base
        self._audience_note = audience_note
        self._system_prompt = (
            _SYSTEM_PROMPT_BASE + "\n\n" + audience_note
            if audience_note else _SYSTEM_PROMPT_BASE
        )
        self.chat_history: list[dict] = []
        self.gap_queue: list[GapItem] = self._build_gap_queue()
        # document_name -> list of content versions (v0 = original summary)
        self.document_versions: dict[str, list[str]] = {
            a.document_name: [a.summary] for a in analyses
        }
        if knowledge_base is not None:
            self._link_gaps_to_sources()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _link_gaps_to_sources(self) -> None:
        """Search the knowledge base for each gap to find the most relevant source passages."""
        for gap in self.gap_queue:
            try:
                results = self.knowledge_base.search(gap.description, top_k=5)
                seen: set[str] = set()
                for r in results:
                    if r['score'] >= 0.25 and r['source_file'] not in seen:
                        gap.related_sources.append({
                            'file': r['source_file'],
                            'text': r['text'][:250],
                            'score': round(r['score'], 3),
                        })
                        seen.add(r['source_file'])
            except Exception:
                pass

    def _build_gap_queue(self) -> list[GapItem]:
        gaps = []
        # Dynamic audience-specific categories
        for category, items in self.gap_analysis.gaps_by_category.items():
            for item in items:
                gaps.append(GapItem(category=category, description=item))
        # Always-present sections
        for item in self.gap_analysis.edge_cases:
            gaps.append(GapItem(category="edge_cases", description=item))
        for item in self.gap_analysis.resource_gaps:
            gaps.append(GapItem(category="resource_gaps", description=item))
        return gaps

    @property
    def remaining_gaps(self) -> list[GapItem]:
        return [g for g in self.gap_queue if not g.resolved]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_message(self) -> str:
        """Generate the agent's first message, setting up an open knowledge-capture conversation."""
        total = len(self.gap_queue)
        if total == 0:
            msg = (
                "Great news — no significant gaps were found in your documentation. "
                "Your process documents appear comprehensive! Feel free to share any "
                "additional context or details about the process that you think would "
                "be valuable for someone new to the system."
            )
            self.chat_history.append({"role": "assistant", "content": msg})
            return msg

        doc_names = ", ".join(a.document_name for a in self.analyses)
        gap_categories: dict[str, list[str]] = {}
        for g in self.gap_queue:
            gap_categories.setdefault(g.category, []).append(g.description)
        gap_summary = "\n".join(
            f"  {cat.replace('_', ' ').title()} ({len(items)} item{'s' if len(items) != 1 else ''}): "
            f"{items[0][:80]}{'...' if len(items[0]) > 80 else ''}"
            + (f" + {len(items)-1} more" if len(items) > 1 else "")
            for cat, items in gap_categories.items()
        )

        audience_line = (
            f"Target audience for the documentation: {self._audience_note}\n"
            if self._audience_note else
            "Target audience: a reader with NO prior knowledge of these systems.\n"
        )
        prompt = (
            f"You are a senior business analyst beginning a knowledge-capture interview.\n\n"
            f"Documents analyzed: {doc_names}\n"
            f"Total documentation gaps identified: {total}\n"
            f"Gap areas:\n{gap_summary}\n"
            f"{audience_line}\n"
            f"Write a warm, professional opening message that:\n"
            f"1. Briefly explains you've analyzed their documents and identified areas that need "
            f"more detail to make the documentation useful for the target audience\n"
            f"2. Presents the gaps as 'areas we want to understand better' — NOT as a rigid list "
            f"of questions to answer in sequence\n"
            f"3. Invites the user to start by describing the process in their own words — "
            f"who is involved, what triggers it, what it achieves, and any important details "
            f"a reader would need to know\n"
            f"4. Makes clear that the more context they share, the better — this is an open "
            f"conversation, not a form\n\n"
            f"Keep the opening to 4-6 sentences. End with an open-ended invitation to share."
        )
        msg = self.llm.query(prompt)
        self.chat_history.append({"role": "assistant", "content": msg})
        return msg

    def chat(self, user_message: str) -> AgentResponse:
        """Process a user message and return the agent response plus any document updates."""
        self.chat_history.append({"role": "user", "content": user_message})

        remaining = self.remaining_gaps
        context = self._build_turn_context(
            remaining[0] if remaining else None,
            remaining[1] if len(remaining) > 1 else None,
            len(remaining),
        )

        input_messages = [
            {"role": "system", "content": self._system_prompt},
            # Inject context as a user turn so the model sees it, then confirm receipt
            {"role": "user", "content": context},
            {"role": "assistant", "content": "Understood. I'll work through the gaps now."},
            *self.chat_history,
        ]

        raw = self.llm.query_raw(input_messages)

        document_updates: list[DocumentUpdate] = []

        # Build a lookup of unresolved gaps (snapshot before we start resolving)
        unresolved: list[GapItem] = list(self.remaining_gaps)

        # Parse ALL <<GAP_RESOLVED: summary | doc_name>> markers in the response.
        # Use re.DOTALL so summaries that span a line still match.
        for resolved_match in re.finditer(
            r'<<GAP_RESOLVED:\s*(.*?)\s*\|\s*(.*?)\s*>>', raw, re.DOTALL
        ):
            gap_summary_text = resolved_match.group(1).strip()
            doc_name_hint = resolved_match.group(2).strip()

            # Find the best-matching unresolved gap by comparing the summary the
            # LLM emitted against each gap description (case-insensitive word overlap).
            gap_to_resolve: Optional[GapItem] = self._match_gap(
                gap_summary_text, unresolved
            )

            if gap_to_resolve:
                gap_to_resolve.resolved = True
                gap_to_resolve.resolution = gap_summary_text  # what was actually learned
                unresolved.remove(gap_to_resolve)

                update = self._generate_document_update(
                    doc_name_hint, gap_to_resolve, user_message, gap_summary_text
                )
                if update:
                    document_updates.append(update)

        all_done = "<<ALL_GAPS_RESOLVED>>" in raw or len(self.remaining_gaps) == 0

        # Strip markers before showing to the user.
        # Use .*? with DOTALL to handle multi-line markers cleanly.
        clean = re.sub(r'<<GAP_RESOLVED:.*?>>', '', raw, flags=re.DOTALL)
        clean = re.sub(r'<<ALL_GAPS_RESOLVED>>', '', clean).strip()

        self.chat_history.append({"role": "assistant", "content": clean})

        return AgentResponse(
            message=clean,
            document_updates=document_updates,
            gaps_remaining=len(self.remaining_gaps),
            is_complete=all_done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_turn_context(
        self,
        current: Optional[GapItem],
        next_gap: Optional[GapItem],
        remaining_count: int,
    ) -> str:
        doc_names = ", ".join(a.document_name for a in self.analyses)
        doc_excerpts = "\n".join(
            f"  {a.document_name}: {a.summary[:300]}..." for a in self.analyses[:5]
        )
        remaining = self.remaining_gaps
        if remaining:
            gap_lines = []
            for i, g in enumerate(remaining[:20]):
                line = f"  [{i+1}] [{g.category}] {g.description}"
                if g.related_sources:
                    files = ", ".join(s['file'] for s in g.related_sources[:3])
                    line += f"\n      → Related sources: {files}"
                gap_lines.append(line)
            if len(remaining) > 20:
                gap_lines.append(f"  ... and {len(remaining) - 20} more")
            gaps_list = "\n".join(gap_lines)
        else:
            gaps_list = "  All gaps have been addressed."

        return (
            f"[CONTEXT — do not read this block to the user]\n"
            f"Documents analyzed: {doc_names}\n"
            f"Open gaps ({remaining_count} remaining — resolve any that the user's response addresses):\n"
            f"{gaps_list}\n"
            f"Relevant document excerpts:\n{doc_excerpts}\n"
            f"[END CONTEXT]"
        )

    def _generate_document_update(
        self,
        doc_name_hint: str,
        gap: GapItem,
        user_answer: str,
        gap_summary: str,
    ) -> Optional[DocumentUpdate]:
        target = self._find_document(doc_name_hint)
        if not target:
            return None

        existing = self.document_versions.get(target.document_name, [target.summary])
        current_content = existing[-1]
        new_version = len(existing) + 1

        source_section = ""
        if gap.related_sources:
            excerpts = "\n\n".join(
                f"From {s['file']} (relevance {int(s['score'] * 100)}%):\n{s['text']}"
                for s in gap.related_sources[:3]
            )
            source_section = f"\n\nRELEVANT SOURCE PASSAGES (from the original documents — use these to ensure accuracy):\n{excerpts}"

        prompt = (
            f"Update the following document to incorporate new information provided by a user.\n\n"
            f"DOCUMENT: {target.document_name}\n\n"
            f"CURRENT CONTENT:\n{current_content}\n\n"
            f"GAP THAT WAS ADDRESSED:\n[{gap.category}] {gap.description}\n\n"
            f"NEW INFORMATION FROM USER:\n{user_answer}"
            f"{source_section}\n\n"
            f"Instructions:\n"
            f"1. Keep all existing content intact\n"
            f"2. Incorporate the new information naturally in the appropriate section\n"
            f"3. Prefix each newly added sentence or paragraph with '(Added) '\n"
            f"4. Maintain the same professional tone and structure"
        )

        updated = self.llm.query(prompt, max_tokens=2000)

        self.document_versions.setdefault(target.document_name, [target.summary])
        self.document_versions[target.document_name].append(updated)

        return DocumentUpdate(
            document_name=target.document_name,
            version=new_version,
            content=updated,
            gap_addressed=gap_summary,
        )

    def _match_gap(self, summary: str, candidates: list[GapItem]) -> Optional[GapItem]:
        """Return the unresolved gap whose description best matches the LLM summary.

        Scores by word overlap (case-insensitive). Falls back to the first
        candidate so we always make progress rather than stalling.
        """
        if not candidates:
            return None
        summary_words = set(re.findall(r'\w+', summary.lower()))
        best, best_score = candidates[0], -1
        for gap in candidates:
            desc_words = set(re.findall(r'\w+', gap.description.lower()))
            overlap = len(summary_words & desc_words)
            if overlap > best_score:
                best, best_score = gap, overlap
        return best

    def _find_document(self, name_hint: str) -> Optional[AnalysisResult]:
        """Return the analysis whose name best matches name_hint."""
        if not self.analyses:
            return None
        if name_hint.upper() in ("PROCESS_DOCUMENT", "PROCESS DOCUMENT", ""):
            return self.analyses[0]
        # Exact match
        for a in self.analyses:
            if a.document_name.lower() == name_hint.lower():
                return a
        # Partial match
        for a in self.analyses:
            if (
                name_hint.lower() in a.document_name.lower()
                or a.document_name.lower() in name_hint.lower()
            ):
                return a
        return self.analyses[0]
