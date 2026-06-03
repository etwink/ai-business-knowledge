"""
Hierarchical summarizer: produces one ClusterSummary per DocumentCluster.

For COBOL clusters:  each file is summarized individually, then a cluster-level
  synthesis is produced that understands the dependency structure (copybooks
  used by which programs, call chains, entry points).

For mixed clusters:  COBOL files are summarized first; then Word/Excel docs
  are summarized with awareness of the COBOL programs they document.

For pure-doc clusters:  all files are summarized together if ≤ 5 files, or in
  batches of 4 that are then rolled up into one cluster summary.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .cluster_builder import DocumentCluster

# ---------------------------------------------------------------------------
# Concurrency / retry settings
# ---------------------------------------------------------------------------
_MAX_SUMMARY_WORKERS = 4   # parallel LLM calls for per-file summaries
_RATE_LIMIT_RETRY_WAIT = 60  # seconds to pause after an HTTP 429
_RATE_LIMIT_MAX_RETRIES = 3  # max retry attempts per call


def _llm_call_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), transparently retrying on Azure rate-limit (429) errors.
    Sleeps _RATE_LIMIT_RETRY_WAIT seconds between attempts so the per-minute quota
    has time to refresh before the next try.
    """
    from openai import RateLimitError
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt < _RATE_LIMIT_MAX_RETRIES:
                time.sleep(_RATE_LIMIT_RETRY_WAIT)
            else:
                raise


@dataclass
class ClusterSummary:
    cluster_id: str
    cluster_name: str
    cluster_type: str        # "cobol" | "mixed" | "docs"
    summary: str             # Full narrative cluster summary
    key_processes: list[str] = field(default_factory=list)
    systems_mentioned: list[str] = field(default_factory=list)
    file_count: int = 0
    entry_point: str | None = None
    edge_cases: list[str] = field(default_factory=list)   # boundary/exception scenarios


class HierarchicalSummarizer:
    """Summarize each cluster into a single ClusterSummary."""

    def __init__(self):
        from src.llm_integration import AzureLLMClient
        self.llm = AzureLLMClient()

    def summarize_all(
        self,
        clusters: list[DocumentCluster],
        progress_callback=None,
        context_block: str = "",  # from ProcessContext.to_prompt_block()
    ) -> list[ClusterSummary]:
        """
        Summarize all clusters. progress_callback(cluster_name, index, total)
        is called before each cluster if provided.
        """
        summaries: list[ClusterSummary] = []
        for i, cluster in enumerate(clusters):
            if progress_callback:
                progress_callback(cluster.cluster_name, i + 1, len(clusters))
            summaries.append(self._summarize_cluster(cluster, context_block))
        return summaries

    # ------------------------------------------------------------------
    # Cluster-type dispatch
    # ------------------------------------------------------------------

    def _summarize_cluster(self, cluster: DocumentCluster, context_block: str = "") -> ClusterSummary:
        if cluster.cluster_type == "cobol":
            return self._summarize_cobol_cluster(cluster, context_block)
        elif cluster.cluster_type == "mixed":
            return self._summarize_mixed_cluster(cluster, context_block)
        else:
            return self._summarize_doc_cluster(cluster, context_block)

    # ------------------------------------------------------------------
    # COBOL cluster
    # ------------------------------------------------------------------

    def _summarize_cobol_cluster(self, cluster: DocumentCluster, context_block: str = "") -> ClusterSummary:
        file_summaries = self._summarize_files_individually(cluster.cobol_files, context_block)
        synthesis = self._synthesize_cobol(cluster, file_summaries, context_block)
        edge_cases = self._extract_cluster_edge_cases(synthesis, cluster.cluster_name)
        return ClusterSummary(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.cluster_name,
            cluster_type="cobol",
            summary=synthesis,
            key_processes=_extract_bullets(synthesis, max_items=_scaled_max_bullets(cluster.file_count)),
            systems_mentioned=list(cluster.cobol_program_names)[:25],
            file_count=cluster.file_count,
            entry_point=cluster.entry_point,
            edge_cases=edge_cases,
        )

    def _synthesize_cobol(
        self,
        cluster: DocumentCluster,
        file_summaries: dict[str, str],
        context_block: str = "",
    ) -> str:
        dep_info = _format_cobol_dep_info(cluster)
        summaries_block = "\n\n".join(
            f"=== {name} ===\n{summary}"
            for name, summary in file_summaries.items()
        )
        context_part = f"\n{context_block}\n" if context_block else ""
        shared_block = self._load_shared_docs_block(cluster)
        lower, upper = _synthesis_word_range(cluster.file_count)
        prompt = f"""You are analyzing a COBOL subsystem called "{cluster.cluster_name}".
{f"Entry-point program: {cluster.entry_point}" if cluster.entry_point else "This cluster contains shared utilities and copybooks."}
{context_part}
Dependency structure:
{dep_info}

Individual file summaries:
{summaries_block}{shared_block}

Write a cohesive subsystem summary ({lower}–{upper} words) covering:
1. Business purpose of this subsystem
2. Overall processing flow (entry → processing → output)
3. Key data structures and copybooks used
4. External systems or files accessed
5. Important business rules or decision points
6. Error handling approach

Be specific to the actual code described above."""
        return _llm_call_with_retry(self.llm.query, prompt, max_tokens=4000)

    # ------------------------------------------------------------------
    # Mixed cluster (COBOL + matching docs)
    # ------------------------------------------------------------------

    def _summarize_mixed_cluster(self, cluster: DocumentCluster, context_block: str = "") -> ClusterSummary:
        cobol_summaries = self._summarize_files_individually(cluster.cobol_files, context_block)
        doc_summaries = self._summarize_doc_files_with_cobol_context(
            cluster.doc_files, cluster.cobol_program_names, context_block
        )
        synthesis = self._synthesize_mixed(cluster, cobol_summaries, doc_summaries, context_block)
        edge_cases = self._extract_cluster_edge_cases(synthesis, cluster.cluster_name)
        return ClusterSummary(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.cluster_name,
            cluster_type="mixed",
            summary=synthesis,
            key_processes=_extract_bullets(synthesis, max_items=_scaled_max_bullets(cluster.file_count)),
            systems_mentioned=list(cluster.cobol_program_names)[:10],
            file_count=cluster.file_count,
            entry_point=cluster.entry_point,
            edge_cases=edge_cases,
        )

    def _summarize_doc_files_with_cobol_context(
        self,
        doc_files: list[Path],
        cobol_program_names: set[str],
        context_block: str = "",
    ) -> dict[str, str]:
        """Summarize Word/Excel docs in parallel, informing the LLM of the COBOL programs they may reference."""
        context_note = (
            f"Note: this document is associated with the following COBOL programs: "
            f"{', '.join(sorted(cobol_program_names)[:15])}. "
            "Highlight any references to those programs or the business rules they implement."
        )
        context_part = f"\n{context_block}\n" if context_block else ""

        def _summarize_one(path: Path):
            doc = _load_doc(path)
            if doc is None:
                return path.name, None
            prompt = (
                f"Summarize the following business document in 150–250 words.\n"
                f"{context_note}{context_part}\n\n"
                f"Document ({path.name}):\n{doc.content[:5000]}"
            )
            return path.name, _llm_call_with_retry(self.llm.query, prompt, max_tokens=2000)

        summaries: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=_MAX_SUMMARY_WORKERS) as executor:
            futures = {executor.submit(_summarize_one, p): p for p in doc_files}
            for future in as_completed(futures):
                try:
                    name, summary = future.result()
                    if summary is not None:
                        summaries[name] = summary
                except Exception:
                    pass
        return summaries

    def _synthesize_mixed(
        self,
        cluster: DocumentCluster,
        cobol_summaries: dict[str, str],
        doc_summaries: dict[str, str],
        context_block: str = "",
    ) -> str:
        dep_info = _format_cobol_dep_info(cluster)
        cobol_block = "\n\n".join(
            f"=== COBOL: {n} ===\n{s}" for n, s in cobol_summaries.items()
        )
        doc_block = "\n\n".join(
            f"=== Document: {n} ===\n{s}" for n, s in doc_summaries.items()
        )
        context_part = f"\n{context_block}\n" if context_block else ""
        shared_block = self._load_shared_docs_block(cluster)
        lower, upper = _synthesis_word_range(cluster.file_count)
        prompt = f"""You are analyzing the "{cluster.cluster_name}" business subsystem.
This cluster combines COBOL source code with associated policy, procedure, and requirements documents.
{context_part}
COBOL dependency structure:
{dep_info}

COBOL Code Summaries:
{cobol_block}

Business Documentation Summaries:
{doc_block}{shared_block}

Write a unified subsystem summary ({lower}–{upper} words) that integrates both the technical implementation and the business intent:
1. Business purpose and stakeholders
2. How the COBOL code implements the documented business rules
3. End-to-end processing flow
4. Data inputs, transformations, and outputs
5. Key decision points and business rules
6. Gaps or discrepancies between the documentation and the code
7. Error handling and exception paths"""
        return _llm_call_with_retry(self.llm.query, prompt, max_tokens=5000)

    # ------------------------------------------------------------------
    # Pure-doc cluster
    # ------------------------------------------------------------------

    def _summarize_doc_cluster(self, cluster: DocumentCluster, context_block: str = "") -> ClusterSummary:
        if len(cluster.doc_files) <= 5:
            synthesis = self._summarize_small_doc_cluster(cluster, context_block)
        else:
            synthesis = self._summarize_large_doc_cluster(cluster, context_block)
        edge_cases = self._extract_cluster_edge_cases(synthesis, cluster.cluster_name)
        return ClusterSummary(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.cluster_name,
            cluster_type="docs",
            summary=synthesis,
            key_processes=_extract_bullets(synthesis, max_items=_scaled_max_bullets(cluster.file_count)),
            file_count=cluster.file_count,
            edge_cases=edge_cases,
        )

    def _summarize_small_doc_cluster(self, cluster: DocumentCluster, context_block: str = "") -> str:
        combined = ""
        for path in cluster.doc_files:
            doc = _load_doc(path)
            if doc:
                combined += f"\n\n=== {path.name} ===\n{doc.content[:4000]}"
        context_part = f"\n{context_block}\n" if context_block else ""
        shared_block = self._load_shared_docs_block(cluster)
        lower, upper = _synthesis_word_range(len(cluster.doc_files))
        prompt = (
            f'Summarize the following "{cluster.cluster_name}" documents ({lower}–{upper} words).\n'
            "Cover: business purpose, key processes, data involved, stakeholders, and important rules.\n"
            f"{context_part}{combined}{shared_block}"
        )
        return _llm_call_with_retry(self.llm.query, prompt, max_tokens=4000)

    def _summarize_large_doc_cluster(self, cluster: DocumentCluster, context_block: str = "") -> str:
        # Build batches of 4, then summarize each batch in parallel
        batch_size = 4
        batches = [
            cluster.doc_files[i : i + batch_size]
            for i in range(0, len(cluster.doc_files), batch_size)
        ]

        def _summarize_batch(batch: list[Path]) -> str:
            combined = ""
            for path in batch:
                doc = _load_doc(path)
                if doc:
                    combined += f"\n\n=== {path.name} ===\n{doc.content[:3000]}"
            prompt = (
                f'Summarize this batch of "{cluster.cluster_name}" documents in 250–350 words.\n'
                f"Cover: key processes, data, business rules.\n{combined}"
            )
            return _llm_call_with_retry(self.llm.query, prompt, max_tokens=2000)

        # Submit all batches in parallel; collect results in original order
        batch_summaries: list[str] = [""] * len(batches)
        with ThreadPoolExecutor(max_workers=_MAX_SUMMARY_WORKERS) as executor:
            future_to_idx = {executor.submit(_summarize_batch, b): i for i, b in enumerate(batches)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_summaries[idx] = future.result()
                except Exception:
                    batch_summaries[idx] = ""

        # Roll up
        batches_text = "\n\n".join(
            f"Batch {i+1}:\n{s}" for i, s in enumerate(batch_summaries) if s
        )
        shared_block = self._load_shared_docs_block(cluster)
        lower, upper = _synthesis_word_range(len(cluster.doc_files))
        rollup_prompt = (
            f'Combine the following batch summaries for "{cluster.cluster_name}" into one cohesive summary ({lower}–{upper} words).\n\n'
            f"{batches_text}{shared_block}"
        )
        return _llm_call_with_retry(self.llm.query, rollup_prompt, max_tokens=4000)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_shared_docs_block(self, cluster: "DocumentCluster") -> str:
        """
        Load any global/shared reference docs attached to this cluster and
        return a formatted block suitable for appending to synthesis prompts.
        Returns an empty string when there are no shared docs.
        """
        if not cluster.shared_doc_files:
            return ""
        parts: list[str] = []
        for path in cluster.shared_doc_files:
            doc = _load_doc(path)
            if doc:
                parts.append(f"=== Shared Reference: {path.name} ===\n{doc.content[:3000]}")
        if not parts:
            return ""
        return (
            "\n\nShared reference documents (cross-cluster overview docs — "
            "use these to understand how this subsystem fits into the broader process):\n"
            + "\n\n".join(parts)
        )

    def _extract_cluster_edge_cases(self, synthesis: str, cluster_name: str) -> list[str]:
        """
        Quick follow-up pass on a cluster synthesis to extract edge cases —
        boundary conditions, unusual inputs, and unaddressed exception paths.
        Returns a list of bullet strings (empty list on any failure).
        """
        prompt = (
            f'Based on the following summary of the "{cluster_name}" cluster, '
            "identify specific edge cases — unusual inputs, boundary conditions, "
            "failure scenarios, and exception paths — that are either mentioned but "
            "not fully addressed, or implied by the logic but entirely undocumented.\n\n"
            f"Cluster summary:\n{synthesis[:2500]}\n\n"
            "List every edge case you can identify. Be concrete and specific to this "
            "cluster's actual functionality. Use one dash-prefixed bullet per item.\n"
            "If no meaningful edge cases are apparent, output: - None identified"
        )
        try:
            result = _llm_call_with_retry(self.llm.query, prompt, max_tokens=1000)
            items = [
                line.strip().lstrip("-•* ").strip()
                for line in result.splitlines()
                if line.strip() and len(line.strip()) > 15
            ]
            # Drop the "None identified" placeholder
            return [i for i in items if "none identified" not in i.lower()]
        except Exception:
            return []

    def _summarize_files_individually(
        self, paths: list[Path], context_block: str = ""
    ) -> dict[str, str]:
        """Return {filename: summary} for each file, using parallel threads."""

        def _summarize_one(path: Path):
            doc = _load_doc(path)
            if doc is None:
                return path.name, None
            context_part = f"\n{context_block}\n" if context_block else ""

            if doc.file_type == "excel":
                focus = (
                    "This Excel file contains two sections: DATA VALUES and FORMULAS. "
                    "Analyse both angles:\n"
                    "1. DATA VALUES — What data is stored? What do the numbers/text represent "
                    "in business terms? What decisions or reports does this spreadsheet support?\n"
                    "2. FORMULAS — What calculations are performed? What business rules or logic "
                    "do the formulas encode? What inputs drive the outputs?\n"
                    "3. Overall purpose — What business question does this spreadsheet answer, "
                    "and how does it connect to the larger process?"
                )
            elif doc.file_type == "excel_macro":
                focus = (
                    "This macro-enabled Excel workbook contains three sections: DATA VALUES, "
                    "FORMULAS, and VBA MACROS. Analyse all three angles:\n"
                    "1. DATA VALUES — What data is stored? What do the numbers/text represent "
                    "in business terms? What decisions or reports does this workbook support?\n"
                    "2. FORMULAS — What calculations are performed? What business rules do "
                    "the formulas encode?\n"
                    "3. VBA MACROS — What automation or logic do the macros implement? "
                    "What triggers them (button click, workbook open, cell change, scheduled)? "
                    "What inputs do they read, what outputs or side effects do they produce "
                    "(writes to cells, sends emails, calls external systems, etc.)? "
                    "What business process do the macros automate, and what would break if "
                    "they stopped running?\n"
                    "4. Overall purpose — How do the data, formulas, and macros work together "
                    "to support the larger business process?"
                )
            elif doc.file_type == "excel_binary":
                focus = (
                    "This is a binary Excel workbook (.xlsb). It contains DATA VALUES and "
                    "optionally VBA MACROS (formula expressions are unavailable in binary format). "
                    "Analyse every available angle:\n"
                    "1. DATA VALUES — What data is stored? What do the numbers/text represent "
                    "in business terms? What decisions or reports does this workbook support?\n"
                    "2. VBA MACROS (if present) — What automation or logic do the macros "
                    "implement? What triggers them? What inputs do they read and what outputs "
                    "or side effects do they produce (cell writes, emails, external calls, etc.)? "
                    "What business process do the macros automate, and what would break if "
                    "they stopped running?\n"
                    "3. Overall purpose — How do the data and macros work together to support "
                    "the larger business process? Note any limitations due to the binary format."
                )
            elif doc.file_type in ("text", "cobol"):
                focus = (
                    "Focus on: business purpose, key logic/processes, data structures used, "
                    "external dependencies called or referenced."
                )
            else:
                focus = (
                    "Focus on: business purpose, key processes described, "
                    "systems or components referenced, important rules or decisions."
                )

            prompt = (
                f"Summarize this {doc.file_type.upper()} file in 150–250 words.\n"
                f"{focus}\n"
                f"{context_part}\n"
                f"File: {path.name}\n\n{doc.content[:6000]}"
            )
            return path.name, _llm_call_with_retry(self.llm.query, prompt, max_tokens=2000)

        summaries: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=_MAX_SUMMARY_WORKERS) as executor:
            futures = {executor.submit(_summarize_one, p): p for p in paths}
            for future in as_completed(futures):
                try:
                    name, summary = future.result()
                    if summary is not None:
                        summaries[name] = summary
                except Exception:
                    pass
        return summaries


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _load_doc(path: Path):
    try:
        from src.document_loaders import get_loader
        loader = get_loader(path)
        return loader.load()
    except Exception:
        return None


def _format_cobol_dep_info(cluster: DocumentCluster) -> str:
    if not cluster.entry_point:
        return "Shared copybooks and utilities (no single entry point)."
    lines = [f"Entry point: {cluster.entry_point}"]
    members = sorted(cluster.cobol_program_names - {cluster.entry_point})
    if members:
        lines.append(f"Dependencies: {', '.join(members)}")
    return "\n".join(lines)


def _extract_bullets(text: str, max_items: int = 6) -> list[str]:
    lines = [
        line.strip().lstrip("-•* ").strip()
        for line in text.splitlines()
        if line.strip() and len(line.strip()) > 15
    ]
    return lines[:max_items]


def _scaled_max_bullets(file_count: int) -> int:
    """Scale bullet extraction limit with cluster size: min 6, max 20."""
    return max(6, min(20, file_count // 2))


def _synthesis_word_range(file_count: int) -> tuple[int, int]:
    """Return (lower, upper) word count targets that grow with cluster size."""
    lower = max(400, min(1000, file_count * 25))
    upper = max(600, min(2500, file_count * 50))
    return lower, upper
