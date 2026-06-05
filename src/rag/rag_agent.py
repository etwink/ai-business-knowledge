"""
RAG agent: adaptive tiered retrieval with relevance filtering and sufficiency gating.

Retrieval loop per query:
  1. Pull X chunks from process doc, Y from word docs, Z from code, and W from
     cluster summaries (plain-English descriptions of COBOL/code clusters).
  2. Two-stage cluster expansion: for each cluster summary retrieved, pull top-k
     raw code chunks from that cluster's files as a follow-up stage.
  3. Batch-check chunk relevance (one LLM call) — drop irrelevant chunks and
     add their indices to a per-query exclusion set so they are never re-pulled.
  4. Check sufficiency (one LLM call) — if enough context, stop.
  5. If not sufficient and iterations remain:
       a. Check "does this query need code?" (once per query, after first pass).
       b. Pull another round with exclusions applied and code omitted if unneeded.
       Repeat from step 3.
  6. Generate a grounded answer from all accepted chunks.
"""

from __future__ import annotations

import json
import re

import config as _rag_cfg
from .knowledge_base import KnowledgeBase

# ---------------------------------------------------------------------------
# Retrieval parameters (loaded from config so they can be tuned via .env)
# ---------------------------------------------------------------------------

_DEFAULT_PROCESS_DOC_K = _rag_cfg.RAG_PROCESS_DOC_K
_DEFAULT_WORD_K = _rag_cfg.RAG_WORD_K
_DEFAULT_CODE_K = _rag_cfg.RAG_CODE_K
_DEFAULT_CLUSTER_SUMMARY_K = _rag_cfg.RAG_CLUSTER_SUMMARY_K
_DEFAULT_CLUSTER_TECHNICAL_K = _rag_cfg.RAG_CLUSTER_TECHNICAL_K
_DEFAULT_CLUSTER_RAW_K = _rag_cfg.RAG_CLUSTER_RAW_K
_MAX_ITERATIONS = _rag_cfg.RAG_MAX_ITERATIONS
_CHUNK_PREVIEW_CHARS = 400       # how much of each chunk to show the relevance judge

# ---------------------------------------------------------------------------
# LLM system prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = (
    "You are a knowledgeable assistant helping users understand business processes, "
    "systems, and procedures. Answer questions using ONLY the provided context excerpts. "
    "If the context does not contain enough information to answer fully, say so explicitly "
    "rather than guessing. Always state which source document(s) your answer draws from. "
    "Format ALL responses using Markdown: use ## for section headers, ### for sub-headers, "
    "**bold** for key terms and field names, numbered lists (1. 2. 3.) for sequential steps, "
    "and - bullets for non-sequential items."
)

_CONVERSATIONAL_ROUTING_SYSTEM = (
    "You are a query routing classifier. "
    "Decide whether a new user message requires searching a document database for new information, "
    "or whether it can be answered entirely from the existing conversation history "
    "(e.g. reformatting, rephrasing, clarifying, summarizing what was already said). "
    "Respond with exactly one word: HISTORY or LOOKUP"
)

_ENRICH_SYSTEM = (
    "You are a search query optimizer. Your only job is to rewrite a user's question "
    "into a richer, more specific search query that will retrieve the most relevant "
    "chunks from a business process knowledge base. "
    "Output ONLY the enriched query — no explanation, no preamble, no bullet points."
)

_RELEVANCE_SYSTEM = (
    "You are a retrieval relevance judge. "
    "Given a query and a list of numbered text chunks, decide which chunks contain "
    "information that is useful for answering the query. "
    "Respond ONLY with a JSON array of booleans (one per chunk, in order). "
    "Example for 3 chunks: [true, false, true]"
)

_SUFFICIENCY_SYSTEM = (
    "You are a retrieval quality judge. "
    "Decide whether the provided context chunks contain enough information to give "
    "a complete and accurate answer to the query. "
    "Respond with exactly one word: YES or NO."
)

_NEEDS_CODE_SYSTEM = (
    "You are a retrieval routing expert. "
    "Decide whether answering the query requires looking at technical source code "
    "(COBOL programs, Python scripts, SQL, etc.) beyond the business documentation "
    "already provided. "
    "Respond with exactly one word: YES or NO."
)

_GAP_DETECTION_SYSTEM = (
    "You are a documentation gap analyst. "
    "Given a draft answer that flags missing information, produce targeted search queries "
    "that could find the missing details in a business document database. "
    "Return ONLY a JSON array of short query strings (up to 5). "
    'Example: ["BCHI screen field names plan load", "intake screen commands CICS step"]'
)

_MAX_HISTORY_TURNS = 4


class RAGAgent:
    """
    Conversational RAG agent with adaptive tiered retrieval.

    On each chat() call:
      - Enriches the query for better vector search.
      - Iteratively pulls and filters chunks until sufficient or max iterations reached.
      - Generates a grounded answer from the accepted chunk set.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        process_doc_k: int = _DEFAULT_PROCESS_DOC_K,
        word_k: int = _DEFAULT_WORD_K,
        code_k: int = _DEFAULT_CODE_K,
        cluster_summary_k: int = _DEFAULT_CLUSTER_SUMMARY_K,
        cluster_technical_k: int = _DEFAULT_CLUSTER_TECHNICAL_K,
        cluster_raw_k: int = _DEFAULT_CLUSTER_RAW_K,
        audience_note: str = "",
        response_mode: str = "standard",
        detail_level: str = "standard",
    ):
        from src.llm_integration import AzureLLMClient
        self.kb = knowledge_base
        self._process_doc_k = process_doc_k
        self._word_k = word_k
        self._code_k = code_k
        self._cluster_summary_k = cluster_summary_k
        self._cluster_technical_k = cluster_technical_k
        self._cluster_raw_k = cluster_raw_k
        self._llm = AzureLLMClient()
        self._history: list[dict] = []
        self._audience_note = audience_note
        self._response_mode = response_mode
        self._detail_level = detail_level
        self._system_prompt = (
            _SYSTEM_PROMPT_BASE + "\n\n" + audience_note
            if audience_note else _SYSTEM_PROMPT_BASE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> tuple[str, list[dict], str]:
        """
        Run adaptive retrieval and return (answer, accepted_chunks, enriched_query).

        accepted_chunks: the final filtered set used to generate the answer.
        enriched_query:  the rewritten query used for vector search.

        If the message is a conversational follow-up (reformatting, rephrasing, etc.)
        that can be answered from history alone, RAG retrieval is skipped entirely.
        """
        # Short-circuit: conversational follow-ups don't need new retrieval
        if self._history and self._is_conversational_followup(user_message):
            answer = self._answer_from_history(user_message)
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": answer})
            return answer, [], user_message

        enriched_query = self._enrich_query(user_message)

        excluded_ids: set[int] = set()   # per-query exclusion set (not persisted)
        accepted_chunks: list[dict] = []
        include_code = True
        needs_code_checked = False

        # Track which cluster IDs have already been expanded to avoid duplicate pulls
        expanded_cluster_ids: set[str] = set()

        for iteration in range(_MAX_ITERATIONS):
            # After the first pass: check once whether code is needed.
            # Guide mode always keeps code — screen/field details often live in code files.
            if iteration == 1 and not needs_code_checked:
                include_code = (
                    True if self._response_mode == "guide"
                    else self._check_needs_code(enriched_query, accepted_chunks)
                )
                needs_code_checked = True

            # Guide mode: heavily boost procedure docs and Word files, reduce high-level summaries
            _is_guide = self._response_mode == "guide"
            _word_k = self._word_k * 3 if _is_guide else self._word_k
            _process_doc_k = self._process_doc_k * 2 if _is_guide else self._process_doc_k
            _cluster_summary_k = (
                max(1, self._cluster_summary_k - 1) if _is_guide else self._cluster_summary_k
            )

            tiered = self.kb.search_tiered(
                enriched_query,
                process_doc_k=_process_doc_k,
                word_k=_word_k,
                code_k=self._code_k,
                cluster_summary_k=_cluster_summary_k,
                cluster_technical_k=self._cluster_technical_k,
                exclude_ids=excluded_ids,
                include_code=include_code,
            )

            # Stage 2: expand any cluster summary OR technical-map chunk into raw code
            cluster_index_chunks = (
                tiered.get("cluster_summary", []) + tiered.get("cluster_technical", [])
            )
            cluster_raw = self._expand_cluster_summaries(
                enriched_query,
                cluster_index_chunks,
                expanded_cluster_ids,
                excluded_ids,
            )

            new_chunks = (
                tiered["cluster_summary"]
                + tiered["cluster_technical"]
                + tiered["process_doc"]
                + tiered["word"]
                + tiered["code"]
                + cluster_raw
            )
            if not new_chunks:
                break

            # Filter for relevance — drop irrelevant chunks from future pulls
            relevant, irrelevant_ids = self._filter_relevance(enriched_query, new_chunks)
            excluded_ids.update(irrelevant_ids)
            accepted_chunks.extend(relevant)

            # Stop early if we already have enough context
            if accepted_chunks and self._check_sufficient(enriched_query, accepted_chunks):
                break

        # Generate initial answer then do a gap-fill pass for guide mode
        answer = self._generate_answer(user_message, accepted_chunks)

        if self._response_mode == "guide":
            gap_queries = self._extract_gap_queries(user_message, answer)
            if gap_queries:
                seen_ids = {c["_array_idx"] for c in accepted_chunks}
                extra_chunks: list[dict] = []
                for gq in gap_queries:
                    gap_tiered = self.kb.search_tiered(
                        gq,
                        process_doc_k=4,
                        word_k=8,
                        code_k=4,
                        cluster_summary_k=0,
                        cluster_technical_k=2,
                        exclude_ids=seen_ids,
                        include_code=True,
                    )
                    gap_candidates = (
                        gap_tiered.get("process_doc", [])
                        + gap_tiered.get("word", [])
                        + gap_tiered.get("code", [])
                        + gap_tiered.get("cluster_technical", [])
                    )
                    relevant, irr_ids = self._filter_relevance(gq, gap_candidates)
                    extra_chunks.extend(relevant)
                    seen_ids.update(c["_array_idx"] for c in relevant)
                    seen_ids.update(irr_ids)
                if extra_chunks:
                    accepted_chunks.extend(extra_chunks)
                    answer = self._generate_answer(user_message, accepted_chunks)

        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})

        return answer, accepted_chunks, enriched_query

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Conversational routing
    # ------------------------------------------------------------------

    def _is_conversational_followup(self, message: str) -> bool:
        """
        Check whether the message can be answered from history alone (no retrieval needed).
        Catches reformatting, rephrasing, clarification, and summary-of-prior-answer requests.
        Fails open (returns False → do retrieval) on any error so we never silently drop context.
        """
        history_text = self._format_history()
        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"New message: {message}\n\n"
            "Does this new message require searching a document database for NEW information, "
            "or can it be answered entirely from the conversation above?\n"
            "HISTORY examples: 'reformat that', 'show it as a table', 'make it shorter', "
            "'can you summarize what you just said', 'explain that differently', "
            "'put that in bullet points', 'what did you mean by X in your last answer'\n"
            "LOOKUP examples: 'what about X process', 'how does Y work', "
            "'what are the steps for Z', 'tell me about W'\n"
            "Respond with exactly one word: HISTORY or LOOKUP"
        )
        try:
            result = self._llm.query(
                prompt, system_message=_CONVERSATIONAL_ROUTING_SYSTEM, max_tokens=10
            )
            return result.strip().upper().startswith("H")
        except Exception:
            return False

    def _answer_from_history(self, user_message: str) -> str:
        """Generate an answer from conversation history alone, skipping retrieval."""
        from src.pipeline.context_agent import RESPONSE_MODE_INSTRUCTIONS, DETAIL_LEVEL_INSTRUCTIONS

        history_text = self._format_history()
        mode_instruction = RESPONSE_MODE_INSTRUCTIONS.get(self._response_mode, "")
        detail_instruction = DETAIL_LEVEL_INSTRUCTIONS.get(self._detail_level, "")
        combined_instructions = "\n\n".join(
            part for part in [mode_instruction, detail_instruction] if part
        )
        audience_instruction = f"\n\n{self._audience_note}" if self._audience_note else ""

        prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"User message: {user_message}"
            f"{audience_instruction}\n\n"
            f"{combined_instructions}\n\n"
            "Respond based on the conversation above. No need to cite sources — "
            "this is a follow-up to information already provided."
        )
        max_tokens = {"summary": 3000, "guide": 50000}.get(self._response_mode, 15000)
        return self._llm.query(
            prompt, system_message=self._system_prompt, max_tokens=max_tokens
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _filter_relevance(
        self, query: str, chunks: list[dict]
    ) -> tuple[list[dict], set[int]]:
        """
        Batch-check which chunks are relevant to the query.

        Returns (relevant_chunks, excluded_array_indices).
        Falls back to accepting all chunks on any parse failure.
        """
        if not chunks:
            return [], set()

        lines = []
        for i, c in enumerate(chunks, 1):
            preview = c["text"][:_CHUNK_PREVIEW_CHARS].replace("\n", " ")
            lines.append(f"[{i}] {c['source_file']}: {preview}")

        prompt = (
            f"Query: {query}\n\n"
            f"Chunks:\n" + "\n\n".join(lines) + "\n\n"
            f"Return a JSON boolean array (true=relevant, false=not relevant), "
            f"one entry per chunk in order. Example for 3 chunks: [true, false, true]"
        )

        try:
            raw = self._llm.query(prompt, system_message=_RELEVANCE_SYSTEM, max_tokens=200)
            match = re.search(r"\[[\s\S]*?\]", raw)
            if match:
                decisions = json.loads(match.group())
                if isinstance(decisions, list) and len(decisions) == len(chunks):
                    relevant = [c for c, d in zip(chunks, decisions) if d]
                    excluded = {
                        c["_array_idx"] for c, d in zip(chunks, decisions) if not d
                    }
                    return relevant, excluded
        except Exception:
            pass

        # Fallback: accept everything (no exclusions)
        return chunks, set()

    def _check_sufficient(self, query: str, chunks: list[dict]) -> bool:
        """
        Ask the LLM whether the current accepted chunks are sufficient to answer the query.
        Uses a stricter bar for guide mode (requires operational specifics).
        Fails open (returns True) so the loop does not spin on an LLM error.
        """
        if not chunks:
            return False

        # Show more chunks at greater depth for guide mode
        preview_count = 14 if self._response_mode == "guide" else 8
        preview_chars = 600 if self._response_mode == "guide" else 300

        context_preview = "\n\n".join(
            f"[{i+1}] {c['source_file']}: {c['text'][:preview_chars]}"
            for i, c in enumerate(chunks[:preview_count])
        )

        if self._response_mode == "guide" and self._detail_level == "detailed":
            sufficiency_q = (
                "Is there sufficient information here to write a DETAILED step-by-step guide? "
                "This requires all of: specific screen or menu names, exact field names and labels, "
                "exact commands or values to enter, and expected system responses per step. "
                "Only say YES if you can see that level of operational specificity. YES or NO."
            )
        elif self._response_mode == "guide":
            sufficiency_q = (
                "Is there sufficient information here to write a step-by-step guide? "
                "This requires: named screens or menus, key fields and commands, "
                "and the full sequence of actions a user must take. YES or NO."
            )
        else:
            sufficiency_q = (
                "Is there sufficient information here to answer the query completely? YES or NO."
            )

        prompt = (
            f"Query: {query}\n\n"
            f"Context chunks retrieved so far:\n{context_preview}\n\n"
            f"{sufficiency_q}"
        )
        try:
            result = self._llm.query(
                prompt, system_message=_SUFFICIENCY_SYSTEM, max_tokens=10
            )
            return result.strip().upper().startswith("Y")
        except Exception:
            return True

    def _check_needs_code(self, query: str, accepted_chunks: list[dict]) -> bool:
        """
        After the first retrieval pass, decide whether code/technical chunks are needed.
        Fails open (returns True = include code) on any error.
        """
        context_summary = "\n".join(
            f"- {c['source_file']}: {c['text'][:200].replace(chr(10), ' ')}"
            for c in accepted_chunks[:6]
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Context retrieved so far (process docs and business documents):\n"
            f"{context_summary}\n\n"
            f"To fully answer this query, do we also need to examine technical source code "
            f"(COBOL programs, Python scripts, SQL queries, etc.)? YES or NO."
        )
        try:
            result = self._llm.query(
                prompt, system_message=_NEEDS_CODE_SYSTEM, max_tokens=10
            )
            return result.strip().upper().startswith("Y")
        except Exception:
            return True

    def _extract_gap_queries(self, user_question: str, draft_answer: str) -> list[str]:
        """
        Scan a draft answer for 'not documented' markers and return targeted search
        queries to fill those specific gaps.  Returns [] quickly if no markers detected.
        """
        gap_markers = (
            "not in the available documentation",
            "not documented",
            "⚠️ (not documented",
            "not available in the",
            "no information",
            "consult the system",
            "consult [",
            "not in the source",
            "not found in",
        )
        if not any(m in draft_answer.lower() for m in gap_markers):
            return []

        prompt = (
            f"Original question: {user_question}\n\n"
            f"Draft answer (contains gaps flagged as 'not documented'):\n"
            f"{draft_answer[:3000]}\n\n"
            "For each place in the answer where information is flagged as missing or not "
            "documented, write a short, specific search query that could find that missing "
            "detail in a business process documentation database. "
            "Focus on: screen names, field names, commands, navigation paths, system IDs.\n"
            "Return ONLY a JSON array of up to 5 short query strings. "
            'Example: ["BCHI plan load screen field names", "BCOP intake screen commands"]'
        )
        try:
            raw = self._llm.query(
                prompt, system_message=_GAP_DETECTION_SYSTEM, max_tokens=300
            )
            match = re.search(r'\[[\s\S]*?\]', raw)
            if match:
                queries = json.loads(match.group())
                if isinstance(queries, list):
                    return [str(q).strip() for q in queries[:5] if str(q).strip()]
        except Exception:
            pass
        return []

    def _expand_cluster_summaries(
        self,
        query: str,
        summary_chunks: list[dict],
        expanded_ids: set[str],
        excluded_ids: set[int],
    ) -> list[dict]:
        """
        Stage-2 retrieval: for each cluster summary chunk, pull the top-k raw code/content
        chunks from that cluster that haven't been seen yet.

        Mutates *expanded_ids* so each cluster is only expanded once per query.
        """
        raw_chunks: list[dict] = []
        for sc in summary_chunks:
            cid = sc.get("cluster_id")
            if not cid or cid in expanded_ids:
                continue
            expanded_ids.add(cid)
            raw = self.kb.search_within_cluster(
                query,
                cluster_id=cid,
                top_k=self._cluster_raw_k,
                exclude_ids=excluded_ids,
            )
            raw_chunks.extend(raw)
        return raw_chunks

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(self, user_message: str, chunks: list[dict]) -> str:
        from src.pipeline.context_agent import RESPONSE_MODE_INSTRUCTIONS, DETAIL_LEVEL_INSTRUCTIONS

        context_parts = []
        for i, c in enumerate(chunks, 1):
            score_pct = int(c["score"] * 100)
            context_parts.append(
                f"[{i}] Source: {c['source_file']} (relevance {score_pct}%)\n{c['text']}"
            )
        context_block = (
            "\n\n---\n\n".join(context_parts)
            if context_parts else "(no relevant context found)"
        )

        history_text = self._format_history()
        mode_instruction = RESPONSE_MODE_INSTRUCTIONS.get(self._response_mode, "")
        detail_instruction = DETAIL_LEVEL_INSTRUCTIONS.get(self._detail_level, "")

        audience_instruction = (
            f"\n\n{self._audience_note}"
            if self._audience_note else ""
        )

        combined_instructions = "\n\n".join(
            part for part in [mode_instruction, detail_instruction] if part
        )

        prompt = (
            f"Context excerpts from the knowledge base:\n\n"
            f"{context_block}\n\n"
            f"{'Conversation so far:' + chr(10) + history_text + chr(10) if history_text else ''}"
            f"User question: {user_message}"
            f"{audience_instruction}\n\n"
            f"{combined_instructions}\n\n"
            f"Answer based on the context above. "
            f"Cite source documents by number (e.g. [1], [2]) where relevant. "
            f"If the answer spans multiple sources, integrate them cohesively."
        )

        max_tokens = {"summary": 3000, "guide": 50000}.get(self._response_mode, 15000)
        return self._llm.query(
            prompt, system_message=self._system_prompt, max_tokens=max_tokens
        )

    # ------------------------------------------------------------------
    # Query enrichment / history
    # ------------------------------------------------------------------

    def _enrich_query(self, user_message: str) -> str:
        """
        Rewrite the raw user question into a richer domain-specific search query.
        Uses recent conversation history and audience context.
        Falls back to the original message if the LLM call fails.
        """
        history_text = self._format_history()
        history_section = f"Recent conversation:\n{history_text}\n\n" if history_text else ""
        audience_section = (
            f"The person asking is: {self._audience_note}\n"
            f"The enriched query should retrieve content useful and relevant to this audience "
            f"(e.g. if they are a front-end user, prefer user-facing steps, screens, and "
            f"workflows over backend technical implementation details).\n\n"
            if self._audience_note else ""
        )
        guide_section = (
            "This is a step-by-step guide request. Enrich the query to find:\n"
            "- Specific screen names, system commands, and menu paths\n"
            "- Field names, valid values, and data entry instructions\n"
            "- Procedure documents, job aids, SOPs, and user manuals\n"
            "- Navigation instructions and operational how-to content\n\n"
            if self._response_mode == "guide" else ""
        )

        prompt = (
            f"{history_section}"
            f"{audience_section}"
            f"{guide_section}"
            f"User question: {user_message}\n\n"
            f"Rewrite this into a specific, detailed search query for a business process "
            f"knowledge base. Expand abbreviations, add relevant domain terms, and make "
            f"implicit intent explicit. If the question is already specific, return it unchanged."
        )
        try:
            enriched = self._llm.query(
                prompt, system_message=_ENRICH_SYSTEM, max_tokens=1000
            )
            return enriched.strip() or user_message
        except Exception:
            return user_message

    def _format_history(self) -> str:
        recent = self._history[-(2 * _MAX_HISTORY_TURNS):]
        if not recent:
            return ""
        lines = []
        for m in recent:
            role = m["role"].upper()
            content = m["content"][:1000] + ("…" if len(m["content"]) > 1000 else "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
