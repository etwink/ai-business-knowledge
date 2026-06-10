"""Conversational agent for interactive document gap-filling.

Pipeline (per user turn):
  1. Conversation turn  — LLM conducts the interview and emits <<GAP_RESOLVED>> markers
  2. Gap identification — focused LLM call maps each resolution summary to a GapItem
  3. Sufficiency check  — focused LLM call gates document updates (thin answers stay open)
  4. Snippet extraction — text-only: find the paragraph most related to the gap
  5. Snippet update     — focused LLM call rewrites only that passage (not the whole doc)
  6. Version commit     — patch the full document and create a new versioned DocumentUpdate

Clarification path: short messages matching "what do you mean by X" are intercepted before
the conversation turn; the agent searches the KB for source context and responds directly.
"""

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
    rephrased: bool = False
    original_description: str = ""  # set when description is replaced via rephrase


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


# ── System prompts ─────────────────────────────────────────────────────────────

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
  <<GAP_RESOLVED: {concise summary of what was learned}>>
  You may emit multiple <<GAP_RESOLVED>> markers in a single response if the user's answer \
  addressed multiple gaps.
- If the user cannot answer something, mark it resolved with "N/A — user confirmed unknown" \
  and note it as an assumption to validate later.
- When all listed gaps are addressed, output <<ALL_GAPS_RESOLVED>> and offer a brief summary \
  of what was learned, then ask if there is anything else the user wants to add.

HANDLING GAPS THE SME CANNOT ANSWER:
If the user indicates the SME being interviewed cannot answer a gap because it is outside their \
role, access, or knowledge (phrases like "they wouldn't know", "that's too technical for them", \
"they don't have access to that", "not their area", "wrong person"):
1. Acknowledge this is a valid constraint.
2. Try to rephrase the gap into something this SME CAN answer from their directly observable \
   experience (what they see on screen, what steps they take, what messages appear to them).
   If a useful rephrasing exists: emit <<GAP_REPHRASE: [new one-sentence description]>> on its \
   own line, then continue the conversation using the rephrased version.
3. If the gap truly cannot be answered by this SME in any useful form: \
   emit <<GAP_SKIP: [brief reason]>> on its own line to remove it from the list.
Never silently drop a gap — always emit one of the two markers and explain what you did.

Tone: curious, professional, and encouraging. Show genuine interest in understanding the process. \
Never make the user feel like they are just answering a checklist."""

_IDENTIFY_SYSTEM = (
    "You are a gap-matching assistant. Given a short resolution summary and a numbered list of "
    "open documentation gaps, identify which gap number (1-based) the summary most likely "
    "addresses. Reply with ONLY the integer (e.g. '3'), or '0' if none match."
)

_SUFFICIENCY_SYSTEM = (
    "You are a documentation quality reviewer. Given a gap description and a user answer, "
    "decide whether the answer provides enough information to write a useful documentation entry. "
    "Reply with exactly one word: SUFFICIENT or INSUFFICIENT."
)

# Short messages matching these patterns are treated as clarification requests
_CLARIFICATION_RE = re.compile(
    r"\b(what do you mean|what does .{0,40} mean|can you (explain|clarify)|"
    r"what is .{0,40}\?|i don'?t understand|unclear about|"
    r"which (gap|issue|item) (is|are) (you|this)|what('?s| is) a .{0,40}\?)\b",
    re.IGNORECASE,
)


class ConversationalAgent:
    """Chatbot agent that fills document gaps through conversation and produces per-document updates."""

    def __init__(
        self,
        analyses: list[AnalysisResult],
        process_document: ProcessDocument,
        gap_analysis: GapAnalysis,
        knowledge_base=None,
        audience_note: str = "",
        response_mode: str = "standard",
        detail_level: str = "standard",
    ):
        self.llm = AzureLLMClient()
        self.analyses = analyses
        self.process_document = process_document
        self.gap_analysis = gap_analysis
        self.knowledge_base = knowledge_base
        self._audience_note = audience_note
        self._response_mode = response_mode
        self._detail_level = detail_level
        self._system_prompt = (
            _SYSTEM_PROMPT_BASE + "\n\n" + audience_note
            if audience_note else _SYSTEM_PROMPT_BASE
        )
        self.chat_history: list[dict] = []
        self.gap_queue: list[GapItem] = self._build_gap_queue()
        # document_name → list of content versions; v0 = original summary
        self.document_versions: dict[str, list[str]] = {
            a.document_name: [a.summary] for a in analyses
        }
        if knowledge_base is not None:
            self._link_gaps_to_sources()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _link_gaps_to_sources(self) -> None:
        """Search the knowledge base for each gap and attach the top source passages."""
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
        for category, items in self.gap_analysis.gaps_by_category.items():
            for item in items:
                gaps.append(GapItem(category=category, description=item))
        for item in self.gap_analysis.edge_cases:
            gaps.append(GapItem(category="edge_cases", description=item))
        for item in self.gap_analysis.resource_gaps:
            gaps.append(GapItem(category="resource_gaps", description=item))
        return gaps

    @property
    def remaining_gaps(self) -> list[GapItem]:
        return [g for g in self.gap_queue if not g.resolved]

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_opening_message(self) -> str:
        """Generate the agent's first message."""
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
            f"Person being interviewed: {self._audience_note}\n"
            if self._audience_note else
            "Person being interviewed: a domain expert familiar with these processes.\n"
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

        # ── Clarification path: "what do you mean by X" ────────────────────────
        if self._is_clarification_request(user_message):
            reply = self._handle_clarification(user_message)
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return AgentResponse(message=reply, gaps_remaining=len(self.remaining_gaps))

        # ── Step 1: Conversation turn ───────────────────────────────────────────
        self.chat_history.append({"role": "user", "content": user_message})
        remaining = self.remaining_gaps
        context = self._build_turn_context(remaining, len(remaining))
        input_messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "Understood. I'll work through the gaps now."},
            *self.chat_history,
        ]
        raw = self.llm.query_raw(input_messages)

        # ── Steps 2-6: Pipeline for each GAP_RESOLVED signal ───────────────────
        resolution_summaries = re.findall(r'<<GAP_RESOLVED:\s*(.*?)\s*>>', raw, re.DOTALL)
        document_updates: list[DocumentUpdate] = []
        unresolved_snapshot = list(self.remaining_gaps)
        could_not_identify: list[str] = []

        for summary in resolution_summaries:
            summary = summary.strip()

            # Step 2: Identify which gap this summary addresses
            gap = self._identify_gap(summary, unresolved_snapshot)
            if gap is None:
                could_not_identify.append(summary)
                continue

            # Step 3: Sufficiency check — if the answer is too thin, leave gap open
            if not self._check_sufficiency(gap, user_message):
                continue

            # Mark resolved and remove from the snapshot so later summaries don't grab it
            gap.resolved = True
            gap.resolution = summary
            unresolved_snapshot = [g for g in unresolved_snapshot if g is not gap]

            # Steps 4-6: Find doc, extract snippet, update targeted passage, commit version
            update = self._pipeline_document_update(gap, user_message, summary)
            if update:
                document_updates.append(update)

        # Process GAP_REPHRASE markers — update the gap's description in-place
        for new_desc in re.findall(r'<<GAP_REPHRASE:\s*(.*?)\s*>>', raw, re.DOTALL):
            new_desc = new_desc.strip()
            target = self._match_gap_by_keyword(new_desc, unresolved_snapshot)
            if target:
                target.original_description = target.original_description or target.description
                target.description = new_desc
                target.rephrased = True

        # Process GAP_SKIP markers — resolve the gap with an N/A note
        for skip_reason in re.findall(r'<<GAP_SKIP:\s*(.*?)\s*>>', raw, re.DOTALL):
            # Match against the last-mentioned gap keywords
            target = self._match_gap_by_keyword(skip_reason.strip(), unresolved_snapshot)
            if target:
                target.resolved = True
                target.resolution = "N/A — not applicable to this SME's role"
                unresolved_snapshot = [g for g in unresolved_snapshot if g is not target]

        all_done = "<<ALL_GAPS_RESOLVED>>" in raw or len(self.remaining_gaps) == 0

        # Strip all internal markers before showing to the user
        clean = re.sub(r'<<GAP_RESOLVED:.*?>>', '', raw, flags=re.DOTALL)
        clean = re.sub(r'<<GAP_REPHRASE:.*?>>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<<GAP_SKIP:.*?>>', '', clean, flags=re.DOTALL)
        clean = re.sub(r'<<ALL_GAPS_RESOLVED>>', '', clean).strip()

        # Append a soft follow-up when we couldn't match a resolution to any gap
        if could_not_identify:
            clean += (
                "\n\n*I captured some information in that response — could you confirm "
                "which gap or topic area you were addressing so I can log it correctly?*"
            )

        self.chat_history.append({"role": "assistant", "content": clean})
        return AgentResponse(
            message=clean,
            document_updates=document_updates,
            gaps_remaining=len(self.remaining_gaps),
            is_complete=all_done,
        )

    # ── Clarification handler ──────────────────────────────────────────────────

    def _is_clarification_request(self, text: str) -> bool:
        """True for short messages that ask what a gap or term means."""
        return len(text.split()) < 35 and bool(_CLARIFICATION_RE.search(text))

    def _handle_clarification(self, user_message: str) -> str:
        """Explain what a gap means and surface the source passages it came from."""
        gap = self._match_gap_by_keyword(user_message, self.gap_queue)

        source_block = ""
        if gap and gap.related_sources:
            passages = "\n\n".join(
                f"From **{s['file']}**:\n> {s['text']}"
                for s in gap.related_sources[:3]
            )
            source_block = (
                "\n\nHere are the source passages that this gap was identified from:"
                f"\n\n{passages}"
            )
        elif self.knowledge_base:
            try:
                results = self.knowledge_base.search(user_message, top_k=3)
                good = [r for r in results if r['score'] >= 0.2]
                if good:
                    passages = "\n\n".join(
                        f"From **{r['source_file']}**:\n> {r['text'][:300]}"
                        for r in good
                    )
                    source_block = (
                        "\n\nHere are relevant passages from the source documents:"
                        f"\n\n{passages}"
                    )
            except Exception:
                pass

        gap_context = (
            f"Most relevant gap: [{gap.category}] {gap.description}"
            if gap else "No specific gap identified from the message."
        )
        gap_list = "\n".join(
            f"  [{i+1}] [{g.category}] {g.description}"
            for i, g in enumerate(self.remaining_gaps[:10])
        )
        prompt = (
            f"The user is asking for clarification: \"{user_message}\"\n\n"
            f"{gap_context}\n\n"
            f"Open gaps for reference:\n{gap_list}\n\n"
            f"Write a brief, helpful clarification (3-5 sentences) that:\n"
            f"1. Explains in plain language what this gap means or why it matters\n"
            f"2. Uses the source passages to provide concrete context if available\n"
            f"3. Ends with an invitation for the user to share what they know\n"
            f"Do NOT repeat the gap description verbatim — rephrase it naturally."
        )
        reply = self.llm.query(prompt)
        return reply + source_block

    # ── Step 2: Gap identification ─────────────────────────────────────────────

    def _identify_gap(self, summary: str, candidates: list[GapItem]) -> Optional[GapItem]:
        """Return the unresolved gap best matching the resolution summary.

        Uses a focused single-token LLM call; falls back to keyword overlap on any error.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        numbered = "\n".join(
            f"[{i+1}] [{g.category}] {g.description}"
            for i, g in enumerate(candidates[:30])
        )
        prompt = (
            f"Resolution summary: \"{summary}\"\n\n"
            f"Open gaps:\n{numbered}\n\n"
            f"Which gap number does the summary most address? "
            f"Reply with ONLY the integer (1–{min(30, len(candidates))}), or 0 if none match."
        )
        try:
            result = self.llm.query(prompt, system_message=_IDENTIFY_SYSTEM, max_tokens=10)
            m = re.search(r'\b(\d+)\b', result.strip())
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass

        return self._match_gap_by_keyword(summary, candidates)

    # ── Step 3: Sufficiency check ──────────────────────────────────────────────

    def _check_sufficiency(self, gap: GapItem, user_answer: str) -> bool:
        """Return True if the user's answer is sufficient to document this gap."""
        if len(user_answer.split()) < 8:
            return False
        prompt = (
            f"Gap to fill: \"{gap.description}\"\n\n"
            f"User's answer: \"{user_answer[:600]}\"\n\n"
            f"Is the answer sufficient to write a useful documentation entry for this gap? "
            f"Reply with exactly one word: SUFFICIENT or INSUFFICIENT."
        )
        try:
            result = self.llm.query(prompt, system_message=_SUFFICIENCY_SYSTEM, max_tokens=10)
            return "SUFFICIENT" in result.upper()
        except Exception:
            return True  # optimistic fallback — let the update proceed

    # ── Steps 4-6: Targeted document update pipeline ──────────────────────────

    def _pipeline_document_update(
        self,
        gap: GapItem,
        user_answer: str,
        gap_summary: str,
    ) -> Optional[DocumentUpdate]:
        """Find target doc → extract relevant snippet → update snippet only → commit version."""

        # Step 4a: Find target document (KB source match > keyword overlap)
        target = self._find_document_for_gap(gap)
        if not target:
            return None

        existing = self.document_versions.get(target.document_name, [target.summary])
        current_content = existing[-1]
        new_version = len(existing) + 1

        source_context = ""
        if gap.related_sources:
            source_context = "\n\nRELEVANT SOURCE PASSAGES:\n" + "\n\n".join(
                f"From {s['file']} (relevance {int(s['score'] * 100)}%):\n{s['text']}"
                for s in gap.related_sources[:3]
            )

        # Step 4b: Extract the specific paragraph/sentences related to the gap
        snippet = self._extract_relevant_snippet(current_content, gap)

        # Step 5: Generate a targeted update for the snippet only
        if snippet:
            prompt = (
                f"Update this passage from '{target.document_name}' to incorporate new information.\n\n"
                f"GAP ADDRESSED: [{gap.category}] {gap.description}\n\n"
                f"EXISTING PASSAGE:\n{snippet}\n\n"
                f"NEW INFORMATION FROM USER:\n{user_answer}"
                f"{source_context}\n\n"
                f"Instructions:\n"
                f"- Return ONLY the updated passage text (similar length to the original)\n"
                f"- Incorporate the new information naturally into the existing text\n"
                f"- Do NOT rewrite or expand beyond the scope of this passage\n"
                f"- Preserve any heading or bullet structure that is present\n"
                f"- Prefix each newly added sentence with '(Added) '"
            )
            updated_snippet = self.llm.query(prompt, max_tokens=1500)

            if snippet in current_content:
                updated_content = current_content.replace(snippet, updated_snippet, 1)
            else:
                # Snippet shifted (e.g., earlier update changed offsets) — append addendum
                updated_content = (
                    current_content
                    + f"\n\n---\n**Addendum — {gap.category}:** {updated_snippet}"
                )
        else:
            # No clear matching passage — write a short addendum at the end
            prompt = (
                f"Write a concise addendum (2-4 sentences) for '{target.document_name}' "
                f"addressing this gap:\n\n"
                f"GAP: [{gap.category}] {gap.description}\n\n"
                f"NEW INFORMATION:\n{user_answer}"
                f"{source_context}\n\n"
                f"Prefix each new sentence with '(Added) '."
            )
            addendum = self.llm.query(prompt, max_tokens=500)
            updated_content = (
                current_content
                + f"\n\n---\n**Addendum — {gap.category}:** {addendum}"
            )

        # Step 6: Commit new version
        self.document_versions.setdefault(target.document_name, [target.summary])
        self.document_versions[target.document_name].append(updated_content)

        return DocumentUpdate(
            document_name=target.document_name,
            version=new_version,
            content=updated_content,
            gap_addressed=gap_summary,
        )

    def _extract_relevant_snippet(self, doc_content: str, gap: GapItem) -> str:
        """Return the paragraph in doc_content most related to the gap (up to 8 sentences).

        Returns empty string if nothing scores above the relevance threshold.
        """
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "is", "are", "was", "were", "be", "been", "it", "this",
            "that", "as", "by", "from", "not", "no", "have", "has", "had", "will",
        }
        gap_words = {
            w.lower()
            for w in re.findall(r"\b\w+\b", gap.description)
            if w.lower() not in stop_words and len(w) > 3
        }
        if not gap_words:
            return ""

        paragraphs = [p.strip() for p in doc_content.split("\n\n") if len(p.strip()) > 30]
        best_score, best_para = 0.0, ""
        for para in paragraphs:
            words = {
                w.lower() for w in re.findall(r"\b\w+\b", para)
                if w.lower() not in stop_words
            }
            score = len(gap_words & words) / max(len(gap_words), 1)
            if score > best_score:
                best_score, best_para = score, para

        if best_score < 0.1 or not best_para:
            return ""

        sentences = re.split(r'(?<=[.!?])\s+', best_para)
        if len(sentences) <= 8:
            return best_para

        # Keep the 8 highest-scoring sentences in their original order
        scored = [
            (
                len(gap_words & {
                    w.lower() for w in re.findall(r"\b\w+\b", s)
                    if w.lower() not in stop_words
                }),
                i,
                s,
            )
            for i, s in enumerate(sentences)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = sorted(idx for _, idx, _ in scored[:8])
        return " ".join(sentences[i] for i in top_indices)

    # ── Document targeting ─────────────────────────────────────────────────────

    def _find_document_for_gap(self, gap: GapItem) -> Optional[AnalysisResult]:
        """Pick the best-matching document using KB source links, then keyword overlap."""
        if not self.analyses:
            return None

        for src in gap.related_sources:
            for a in self.analyses:
                if (
                    src['file'].lower() in a.document_name.lower()
                    or a.document_name.lower() in src['file'].lower()
                ):
                    return a

        gap_words = set(re.findall(r'\w+', gap.description.lower()))
        best, best_score = self.analyses[0], -1
        for a in self.analyses:
            summary_words = set(re.findall(r'\w+', a.summary.lower()))
            score = len(gap_words & summary_words)
            if score > best_score:
                best, best_score = a, score
        return best

    # ── Gap management (rephrase / skip) ──────────────────────────────────────

    def rephrase_gap(self, gap: GapItem) -> None:
        """Ask the LLM to rewrite a gap so it is answerable by the current SME.

        Updates gap.description in-place and sets gap.rephrased = True.
        """
        sme_context = self._audience_note or "a business operations user"
        prompt = (
            f"A documentation gap was written assuming a technical reader, but the SME "
            f"being interviewed is:\n{sme_context}\n\n"
            f"Original gap:\n{gap.description}\n\n"
            f"Rewrite this gap to ask about the same topic from what this SME directly "
            f"observes, does, or experiences — not internal system behavior or technical "
            f"details invisible to them.\n\n"
            f"Example:\n"
            f"  Before: 'Audit logging write failure when RECOVERY-OPT=1 — program behavior?'\n"
            f"  After:  'What message appears on your screen when the system cannot save "
            f"changes, and what steps do you take?'\n\n"
            f"Return ONLY the rephrased gap — one sentence, no quotes, no explanation."
        )
        new_desc = self.llm.query(prompt, max_tokens=200).strip()
        gap.original_description = gap.original_description or gap.description
        gap.description = new_desc
        gap.rephrased = True

    def skip_gap(self, gap: GapItem) -> None:
        """Mark a gap as not applicable to this SME and resolve it."""
        gap.resolved = True
        gap.resolution = "N/A — not applicable to this SME's role"

    # ── Turn context ───────────────────────────────────────────────────────────

    def _build_turn_context(self, remaining: list[GapItem], remaining_count: int) -> str:
        doc_names = ", ".join(a.document_name for a in self.analyses)
        doc_excerpts = "\n".join(
            f"  {a.document_name}: {a.summary[:300]}..." for a in self.analyses[:5]
        )
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

        from src.pipeline.context_agent import RESPONSE_MODE_INSTRUCTIONS, DETAIL_LEVEL_INSTRUCTIONS
        mode_block = RESPONSE_MODE_INSTRUCTIONS.get(self._response_mode, "")
        detail_block = DETAIL_LEVEL_INSTRUCTIONS.get(self._detail_level, "")
        format_block = "\n\n".join(part for part in [mode_block, detail_block] if part)

        return (
            f"[CONTEXT — do not read this block to the user]\n"
            f"Documents analyzed: {doc_names}\n"
            f"Open gaps ({remaining_count} remaining — resolve any that the user's response addresses):\n"
            f"{gaps_list}\n"
            f"Relevant document excerpts:\n{doc_excerpts}\n"
            + (f"Response format:\n{format_block}\n" if format_block else "")
            + "[END CONTEXT]"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _match_gap_by_keyword(self, text: str, candidates: list[GapItem]) -> Optional[GapItem]:
        """Return the candidate whose description best matches text by word overlap."""
        if not candidates:
            return None
        words = set(re.findall(r'\w+', text.lower()))
        best, best_score = candidates[0], -1
        for gap in candidates:
            desc_words = set(re.findall(r'\w+', gap.description.lower()))
            overlap = len(words & desc_words)
            if overlap > best_score:
                best, best_score = gap, overlap
        return best

    def _match_gap(self, summary: str, candidates: list[GapItem]) -> Optional[GapItem]:
        """Backward-compat alias."""
        return self._match_gap_by_keyword(summary, candidates)

    def _find_document(self, name_hint: str) -> Optional[AnalysisResult]:
        """Backward-compat: find document by name hint."""
        if not self.analyses:
            return None
        if name_hint.upper() in ("PROCESS_DOCUMENT", "PROCESS DOCUMENT", ""):
            return self.analyses[0]
        for a in self.analyses:
            if a.document_name.lower() == name_hint.lower():
                return a
        for a in self.analyses:
            if (
                name_hint.lower() in a.document_name.lower()
                or a.document_name.lower() in name_hint.lower()
            ):
                return a
        return self.analyses[0]
